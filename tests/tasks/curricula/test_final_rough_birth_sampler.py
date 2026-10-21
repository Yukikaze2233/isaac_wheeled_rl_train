"""Final request sampling against frozen methods and the bound107 mechanism."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tests.tasks.curricula.test_final_rough_birth_source import source_owner, quat_from_euler_xyz
from wheeled_tasks.chassis.tasks.recovery.final_rough_birth import FinalRoughBirthSampler
from wheeled_tasks.chassis.simulation.runtime.flat_birth_placement import PhysicalFlatPlacement
from wheeled_tasks.chassis.control.leg_pose_mapping import LegPoseMapper


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def isolated_torch_rng_and_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(threads)


def permissions(ids, variant):
    if variant == "all":
        return dict(air_allowed=torch.ones(len(ids), dtype=torch.bool),
                    ground_allowed=torch.ones(len(ids), dtype=torch.bool),
                    axis_allowed=torch.zeros(len(ids), dtype=torch.bool))
    if variant == "mixed":
        return dict(air_allowed=ids % 4 == 0, ground_allowed=ids % 5 != 0,
                    axis_allowed=ids % 4 == 1)
    return dict(air_allowed=torch.zeros(len(ids), dtype=torch.bool),
                ground_allowed=torch.zeros(len(ids), dtype=torch.bool),
                axis_allowed=torch.ones(len(ids), dtype=torch.bool))


@pytest.mark.parametrize("variant", ["all", "mixed", "ordinary"])
def test_ordered_partial_requests_match_executed_source_methods_and_rng(variant):
    n = 1024
    ids = torch.randperm(n, generator=torch.Generator().manual_seed(900))[:768]
    masks = permissions(ids, variant)
    before_masks = deepcopy(masks)
    owner = source_owner(n)
    full = {key: torch.zeros(n, dtype=torch.bool) for key in masks}
    for key, value in masks.items():
        full[key][ids] = value
    owner._get_predefined_reset_air_disabled_mask = lambda selected: ~full["air_allowed"][selected]
    owner._get_predefined_reset_ground_disabled_mask = lambda selected: ~full["ground_allowed"][selected]
    owner._get_axis_aligned_reset_heading_mask = lambda selected: full["axis_allowed"][selected]
    write_joints = owner.robot.write_joint_state_to_sim

    def record_wheels(q, dq, joint_ids, env_ids):
        if torch.equal(joint_ids, owner._wheel_idx):
            owner.sampled_wheel = q.clone()
        write_joints(q, dq, joint_ids, env_ids)

    owner.robot.write_joint_state_to_sim = record_wheels
    torch.manual_seed(3027)
    owner.reset_root_event(ids, **owner.cfg.root_reset_event)
    owner._custom_reset_random(ids)
    owner._apply_axis_aligned_reset_heading(ids)
    source_rng = torch.get_rng_state().clone()
    sampler = FinalRoughBirthSampler(n, "cpu")
    generator = torch.Generator().manual_seed(3027)
    global_rng = torch.get_rng_state().clone()
    batch = sampler.sample(ids, generator, iteration=0, **masks)
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.equal(generator.get_state(), source_rng)
    request = batch.requests
    assert torch.equal(request.env_ids, ids)
    assert torch.equal(request.ground_birth, torch.isin(ids, owner.ground_ids))
    assert torch.equal(batch.birth_kind == 2, torch.isin(ids, owner.air_ids))
    assert torch.equal(batch.birth_kind == 1, request.ground_birth)
    assert torch.equal(batch.air_mode_index, torch.where(batch.birth_kind == 2, 0, -1))
    assert torch.equal(request.virtual_leg_length_m, owner.sampled_length)
    assert torch.equal(request.virtual_leg_angle_rad, owner.sampled_angle)
    assert torch.equal(request.wheel_angle_rad, owner.sampled_wheel)
    assert torch.equal(request.virtual_foot_height_m,
                       owner.sampled_length * owner.sampled_angle.cos())
    dq = owner.robot.joint_velocity[ids][:, [0, 1, 12, 6, 7, 13]]
    assert torch.equal(request.control_velocity_rad_s, dq)
    root = owner.robot.data.root_state_w[ids]
    torch.testing.assert_close(request.source_root_position_m,
        root[:, :3] - owner.terrain.env_origins[ids], atol=2e-7, rtol=0)
    quaternion = quat_from_euler_xyz(*request.source_root_rpy_rad.unbind(-1))
    torch.testing.assert_close(quaternion, root[:, 3:7], atol=2e-7, rtol=0)
    torch.testing.assert_close(request.source_root_velocity6, root[:, 7:], atol=1e-6, rtol=0)
    assert (request.leg_mode[~request.ground_birth] == -1).all()
    assert torch.equal(request.leg_mode[request.ground_birth],
                       (owner.sampled_angle[request.ground_birth] < 0).to(torch.int8))
    assert batch.definition_sha256 == sampler.definition_sha256
    assert all(torch.equal(masks[key], value) for key, value in before_masks.items())


@pytest.mark.parametrize("invalid", ["clock_missing", "clock_bool", "clock_negative", "mask_full_owner",
                                     "mask_wrong_dtype", "duplicate_ids"])
def test_invalid_reset_context_is_rejected_before_rng_or_configuration_changes(invalid):
    sampler = FinalRoughBirthSampler(8, "cpu")
    ids, iteration = torch.tensor([6, 1, 4]), 0
    masks = permissions(ids, "all")
    if invalid == "clock_missing":
        iteration = None
    elif invalid == "clock_bool":
        iteration = True
    elif invalid == "clock_negative":
        iteration = -1
    elif invalid == "mask_full_owner":
        masks["air_allowed"] = torch.ones(8, dtype=torch.bool)
    elif invalid == "mask_wrong_dtype":
        masks["ground_allowed"] = masks["ground_allowed"].long()
    else:
        ids = torch.tensor([6, 1, 6])
    generator = torch.Generator().manual_seed(9)
    before_rng, before_summary = generator.get_state().clone(), sampler.summary()
    with pytest.raises(ValueError):
        sampler.sample(ids, generator, iteration=iteration, **masks)
    assert torch.equal(generator.get_state(), before_rng)
    assert sampler.summary() == before_summary


def test_replay_empty_subset_and_summary_do_not_hide_mutable_sampling_state():
    sampler = FinalRoughBirthSampler(32, "cpu")
    ids = torch.tensor([30, 2, 17, 4])
    masks = permissions(ids, "all")
    first = sampler.sample(ids, torch.Generator().manual_seed(10000), iteration=500, **masks)
    other = sampler.sample(ids, torch.Generator().manual_seed(10000), iteration=500, **masks)
    for key, value in vars(first.requests).items():
        assert torch.equal(value, getattr(other.requests, key))
    assert torch.equal(first.birth_kind, other.birth_kind)
    assert torch.equal(first.air_mode_index, other.air_mode_index)
    summary = sampler.summary()
    summary["ground_modes"]["positive"]["prob"] = 1.
    assert sampler.summary()["ground_modes"]["positive"]["prob"] == .2
    generator = torch.Generator().manual_seed(8)
    before = generator.get_state().clone()
    empty = torch.empty(0, dtype=torch.long)
    result = sampler.sample(empty, generator, iteration=0, **permissions(empty, "all"))
    assert result.requests.virtual_leg_length_m.shape == (0, 2)
    assert result.birth_kind.shape == (0,) and result.air_mode_index.shape == (0,)
    assert torch.equal(generator.get_state(), before)


@pytest.fixture(scope="module")
def bound_mechanism():
    config = json.loads((ROOT / "configs/v7_formal_20261012/recovery_after10000_056.json").read_text())
    directory = ROOT / config["asset_directory"]
    manifest = json.loads((directory / "manifest.json").read_text())
    spec = json.loads((directory / "model_spec.json").read_text())
    settings = config["flat_episode_runtime"]
    table_bytes = (ROOT / settings["branch_table"]).read_bytes()
    assert hashlib.sha256(table_bytes).hexdigest() == settings["branch_table_sha256"]
    mapper = LegPoseMapper(spec, manifest, json.loads(table_bytes), "cpu")
    placement = PhysicalFlatPlacement(spec, manifest, directory, clearance_m=settings["minimum_clearance_m"])
    return mapper, placement


def test_final_requests_map_to_complete_own_tree_and_explicit_com_world_placement(bound_mechanism):
    mapper, placement = bound_mechanism
    n = 384
    sampler = FinalRoughBirthSampler(n, "cpu")
    batch = sampler.sample(torch.arange(n), torch.Generator().manual_seed(10000), iteration=0,
                           **permissions(torch.arange(n), "all"))
    r = batch.requests
    mapped = mapper.map_requests(r.virtual_leg_length_m.double(), -r.virtual_leg_angle_rad.double(),
                                r.wheel_angle_rad.double(), r.control_velocity_rad_s.double())
    for kind in (0, 1, 2):
        assert (mapped.valid & (batch.birth_kind == kind)).any()
    good = mapped.valid
    q, dq = mapped.q[good], mapped.dq[good]
    assert mapper.kinematics.residual(q).abs().max() < 1e-8
    assert (mapper.kinematics.jacobian(q) @ dq[:, :, None]).abs().max() < 1e-10
    metrics = mapper.metrics(q)
    torch.testing.assert_close(metrics["length_m"], r.virtual_leg_length_m[good].double(), atol=1e-8, rtol=0)
    error = metrics["angle_rad"] + r.virtual_leg_angle_rad[good].double()
    assert torch.atan2(error.sin(), error.cos()).abs().max() < 1e-8
    torch.testing.assert_close(dq[:, mapper.kinematics.control_indices], r.control_velocity_rad_s[good].double())
    count = int(good.sum())
    origins = torch.tensor([[10., 20., 1.]] * count, dtype=torch.float64)
    xy = origins[:, :2] + torch.tensor([2., 3.])
    placed = placement.propose(q, r.source_root_position_m[good].double(),
        r.source_root_rpy_rad[good].double(), r.source_root_velocity6[good].double(),
        origins, xy, origins[:, 2], root_velocity_frame="com_world")
    assert placed.valid.all()
    assert torch.equal(placed.root_velocity, r.source_root_velocity6[good].double())
    assert (placed.minimum_shape_clearance_m >= placement.clearance_m).all()
    assert torch.equal(placed.root_pose[:, :2], xy + r.source_root_position_m[good, :2].double())
    assert (placed.root_pose[:, 2] >= origins[:, 2] + r.source_root_position_m[good, 2].double()).all()
