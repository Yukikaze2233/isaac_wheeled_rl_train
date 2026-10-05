"""CPU execution of reset geometry, ownership masks and native velocity writes."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.nonstandard_start import NonstandardStart, nonstandard_start_settings
from wheeled_tasks.chassis.ordinary_posture_curriculum import (
    OrdinaryPostureCurriculum, ordinary_eligibility, posture_settings, rotate_vectors,
    runtime_expectation, validate_posture_settings)
from wheeled_tasks.chassis.training_schedule import stage_actor_updates
from test_nonstandard_start_integration import execute, method, mentions


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "model/纯底盘_v6_232mm_stop105/urdf"
MANIFEST = json.loads((ASSET / "manifest.json").read_text())
NAMES = MANIFEST["tree_joint_names"]


def module(count=256, *, eligible=None, enabled=True, evaluation=False, origin=100):
    eligible = torch.ones(count, dtype=torch.bool) if eligible is None else eligible
    return OrdinaryPostureCurriculum(count, "cpu", posture_settings(origin, enabled), eligible,
        evaluation=evaluation)


@pytest.mark.parametrize("patch", [
    {"origin_actor_update": -1}, {"enabled": 1}, {"clock": "critic_updates"},
    {"birth": {"probability": .5, "start_roll_pitch_rad": 0., "end_roll_pitch_rad": .15}},
    {"push": {"enabled": True, "start_speed_m_s": 0., "end_speed_m_s": .25,
        "interval_seconds": [5., 10.]}},
])
def test_schema_does_not_silently_disable_nonzero_curriculum(patch):
    with pytest.raises(ValueError):
        validate_posture_settings(posture_settings(0) | patch)


def test_actor_origin_ramp_and_critic_clock_are_explicit():
    lesson = module(origin=5552)
    assert lesson.amplitudes(0) == (0, .03, .05)
    assert lesson.amplitudes(5552 + 1000) == (1000, .03, .05)
    assert lesson.amplitudes(5552 + 2000) == pytest.approx((2000, .09, .15))
    assert lesson.amplitudes(5552 + 3000) == pytest.approx((3000, .15, .25))
    env = NS(cfg={"task_campaign": {"domain": "jump"}}, task_actor_update=5552,
        stage_actor_update=6000, global_actor_update=100000)
    assert stage_actor_updates(env) == 5552
    assert lesson.amplitudes(stage_actor_updates(env)) == (0, .03, .05)
    assert lesson.runtime_state() == runtime_expectation(posture_settings(5552))


def test_flat_terrain_birth_excludes_flat_looking_request_script_and_spin():
    groups = ["stand", "terrain", "request_flat", "raised", "script", "spin", "jump"]
    cfg = {"behavior_pool_membership": dict(zip(groups,
        ["stand_anchors", "untriggered_steps", "basic_translation", "raised_requests",
            "stand_anchors", "high_motion", "stand_anchors"])),
        "skill_specs": {group: {"kind": "stand", "mode": 0} for group in groups}}
    cfg["skill_specs"]["terrain"] = {"kind": "body_climb", "mode": 0}
    cfg["skill_specs"]["request_flat"] = {"kind": "forward", "mode": 4}
    cfg["skill_specs"]["script"]["recovery_pose"] = "upright"
    recovery = torch.tensor([False, False, False, False, True, False, False])
    spin = torch.tensor([False, False, False, False, False, True, False])
    assert ordinary_eligibility(cfg, groups, ["flat"] * 6 + ["jump"], "cpu",
        recovery=recovery, spin=spin).tolist() == [True, True, False, False, False, False, False]


def test_partial_reset_and_disabled_curriculum_preserve_other_lanes_and_rng():
    lesson = module(4, eligible=torch.tensor([True, False, True, True]))
    lesson.angles[0] = .012
    lesson.selected[0] = True
    lesson.reset(torch.tensor([1, 2, 3]), 100, generator=torch.Generator().manual_seed(9))
    assert lesson.selected[0] and torch.equal(lesson.angles[0], torch.tensor([.012, .012]))
    assert not lesson.selected[1] and not lesson.angles[1].any()
    for disabled in (module(4, enabled=False), module(4, evaluation=True)):
        generator = torch.Generator().manual_seed(4)
        before = generator.get_state().clone()
        disabled.reset(torch.arange(4), 100, generator=generator)
        assert torch.equal(generator.get_state(), before)
        assert not disabled.selected.any() and disabled.birth_population == 0


@pytest.fixture(scope="module")
def bound():
    lesson = module()
    bank = NonstandardStart(lesson.count, "cpu", .005, nonstandard_start_settings(100), ROOT, NAMES)
    lesson.bind_geometry(ASSET, MANIFEST, NAMES, bank)
    return lesson, bank


def reset_buffers(lesson):
    n = lesson.count
    ids = torch.arange(n)
    root = torch.zeros(n, 7); root[:, 2] = .32; root[:, 6] = 1.
    joints = torch.tensor([[MANIFEST["nominal_joint_pos"][name] for name in NAMES]]).repeat(n, 1)
    origins = torch.zeros(n, 3)
    surfaces = torch.tensor([[[-4., 4., 0., 0., 0.]]]).repeat(n, 1, 1)
    return ids, root, joints, origins, surfaces, torch.ones(n, 1, dtype=torch.bool)


def test_actual_sealed_asset_nominal_and_bank_rotations_keep_support_and_closure(bound):
    lesson, bank = bound
    ids, root, joints, origins, surfaces, valid = reset_buffers(lesson)
    generator = torch.Generator().manual_seed(18)
    lesson.reset(ids, 100, generator=generator)
    lesson.initial_orientation(ids, root)
    bank.reset(ids, 3100, eligible=lesson.eligible, generator=generator)
    velocity = torch.zeros(lesson.count, 6)
    bank.initial_state(ids, root, joints, velocity, torch.zeros(lesson.count), preserve_orientation=True)
    before_q = joints.clone()
    before_quat = root[:, 3:7].clone()
    before_selected = lesson.selected.clone()
    lesson.align_support(ids, root, joints, origins, surfaces, valid, profile_ids=bank.profile_ids)
    torch.testing.assert_close(joints, before_q)
    torch.testing.assert_close(root[~before_selected, 3:7], before_quat[~before_selected])
    assert (bank.profile_ids == -1).any() and (bank.profile_ids >= 0).any()
    # Independent source FK verifies the complete unchanged joint tree. A root
    # rotation preserves closure distances; no geometry-only teleport is used.
    import importlib.util
    path = ASSET / "tools/v6_mechanism.py"
    loader = importlib.util.spec_from_file_location("posture_test_fk", path)
    fk = importlib.util.module_from_spec(loader); loader.loader.exec_module(fk)
    spec = json.loads((ASSET / "model_spec.json").read_text())
    for profile in (0, 1, len(lesson.profile_positions) - 1):
        q = dict(zip(NAMES, lesson.profile_positions[profile].numpy()))
        frames = fk.fk(spec, q)
        errors = np.asarray([fk.point(frames[row["body0"]], row["local_pos0_m"])
            - fk.point(frames[row["body1"]], row["local_pos1_m"]) for row in spec["constraints"]])
        assert np.linalg.norm(errors, axis=-1).max() < 2e-7
    selected = lesson.selected.nonzero().flatten()
    profiles = bank.profile_ids[selected] + 1
    wheels = rotate_vectors(root[selected, 3:7], lesson.wheel_points[profiles])
    axes = rotate_vectors(root[selected, 3:7], lesson.wheel_axes[profiles])
    extent_z = lesson.wheel_radius * (1. - axes[..., 2].square()).clamp_min(0.).sqrt()
    extent_z += lesson.wheel_halfwidth * axes[..., 2].abs()
    bottoms = root[selected, 2:3] + wheels[..., 2] - extent_z
    assert bottoms.min() >= -1e-6
    torch.testing.assert_close(bottoms.amin(-1), torch.zeros(len(selected)), atol=1e-6, rtol=0.)
    assert lesson.birth_abs_angle_max_rad > 0.


def test_only_selected_profiles_are_bound_and_checked(bound):
    lesson, bank = bound
    ids, root, joints, origins, surfaces, valid = reset_buffers(lesson)
    lesson.selected.zero_(); lesson.selected[0] = True; lesson.angles.zero_(); lesson.angles[0] = .02
    lesson.initial_orientation(ids, root)
    bank.profile_ids.fill_(-1); bank.profile_ids[1] = 0
    # Nonselected bank/fixture values must not select the wrong nominal row.
    joints[1] = 9.
    lesson.align_support(ids, root, joints, origins, surfaces, valid, profile_ids=bank.profile_ids)
    assert joints[1].eq(9.).all() and root[1, 2] == pytest.approx(.32)
    joints[0, 0] += .1
    with pytest.raises(ValueError, match="unbound reset joint"):
        lesson.align_support(ids, root, joints, origins, surfaces, valid, profile_ids=bank.profile_ids)


def test_internal_terrain_strip_is_seen_by_full_collision_guard(bound):
    lesson, bank = bound
    ids, root, joints, origins, surfaces, valid = reset_buffers(lesson)
    lesson.selected.zero_(); lesson.selected[0] = True
    lesson.angles.zero_(); lesson.angles[0] = .02
    bank.profile_ids.fill_(-1)
    lesson.initial_orientation(ids, root)
    surfaces = torch.cat((surfaces, torch.tensor([[[.18, .20, .3, 0., 0.]]]).repeat(lesson.count, 1, 1)), 1)
    valid = torch.ones(lesson.count, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="no collision-free wheel support"):
        lesson.align_support(ids, root, joints, origins, surfaces, valid, profile_ids=bank.profile_ids)


def test_rejected_slope_tilt_keeps_clearance_without_rejecting_legal_nominal_pose(bound):
    from wheeled_tasks.chassis.release_stop import release_stop_surfaces

    lesson, bank = bound
    ids, root, joints, origins, _, _ = reset_buffers(lesson)
    worker = json.loads((ROOT / "contracts/v6_task_campaign_terrain_worker_v18.json").read_text())
    spec = worker["skill_specs"]["release_stop_translate"]
    strips = release_stop_surfaces("slope", spec["terrain_limits"], spec.get("terrain_level", 1.),
        3742, worker["release_stop"])
    surfaces = torch.tensor([[[s.x0, s.x1, s.z0, s.slope, s.cross_slope] for s in strips]]).repeat(lesson.count, 1, 1)
    valid = torch.ones(lesson.count, len(strips), dtype=torch.bool)
    lesson.selected.zero_(); lesson.selected[0] = True
    lesson.angles.zero_(); lesson.angles[0] = torch.tensor([.1181546375, .0265939608])
    bank.profile_ids.fill_(-1)
    lesson.initial_orientation(ids, root)
    original_joints = joints.clone()
    rejected_before = lesson.birth_rejected_count
    lesson.align_support(ids, root, joints, origins, surfaces, valid, profile_ids=bank.profile_ids)
    assert not lesson.selected[0] and not lesson.angles[0].any()
    assert lesson.birth_rejected_count == rejected_before + 1
    torch.testing.assert_close(root[0, 3:7], torch.tensor([0., 0., 0., 1.]), atol=0., rtol=0.)
    # Independent sealed-geometry audit: nominal wheel support is .189167246 m,
    # while the conservative body box with its 1 mm margin needs .189942315 m.
    assert root[0, 2] == pytest.approx(.189942315, abs=1e-6)
    assert 0. < float(root[0, 2]) - .189167246 < .001
    assert root[1:, 2].eq(.32).all() and torch.equal(joints, original_joints)


def test_geometry_hash_is_not_assumed_from_a_file_path():
    lesson = module(1)
    manifest = deepcopy(MANIFEST)
    manifest["files_sha256"]["tools/v6_mechanism.py"] = "0" * 64
    with pytest.raises(ValueError, match="sealed asset"):
        lesson.bind_geometry(ASSET, manifest, NAMES)


def test_per_lane_push_clocks_are_independent_and_exclude_non_actor_intervals():
    lesson = module(4096)
    clocks = lesson.sample_intervals(torch.arange(4096), generator=torch.Generator().manual_seed(71))
    assert clocks.min() >= 5. and clocks.max() < 10. and clocks.std() > 1.
    assert torch.unique(clocks).numel() > 4000
    masks = module(6)
    due = torch.ones(6, dtype=torch.bool)
    ground = torch.tensor([True, True, True, True, False, True])
    actor = torch.tensor([True, False, False, True, True, True]) # SCRIPT, BLEND excluded.
    hold = torch.tensor([False, False, False, True, False, False])
    done = torch.tensor([False, False, False, False, False, True])
    assert masks.push_mask(due, ground, actor, hold, done).tolist() == [True, False, False, False, False, False]


def native_push(env, ids):
    return execute(method("apply_push").body, env, ids=ids, wp=NS(to_torch=lambda x: x),
        math=math, stage_actor_updates=stage_actor_updates)


def test_native_apply_push_is_read_back_and_keeps_independent_intervals():
    lesson = module(32)
    velocities = torch.zeros(32, 6)
    def write(*, root_velocity, env_ids, full_data):
        assert full_data is False
        velocities[env_ids] = root_velocity
    generator = torch.Generator().manual_seed(17)
    env = NS(ordinary_posture=lesson, device="cpu", cfg={"task_campaign": {}}, task_actor_update=100,
        stage_actor_update=100, skills=NS(push_speed=torch.zeros(32)), push_clock=torch.zeros(32),
        generator=generator, random=lambda n: torch.rand(n, generator=generator),
        robot=NS(root_view=NS(get_root_velocities=lambda: velocities), write_root_com_velocity_to_sim_index=write))
    native_push(env, torch.arange(32))
    assert velocities[:, :2].norm(dim=-1).max() <= .05
    assert velocities[:, :2].norm(dim=-1).max() > .03
    assert lesson.push_events == 32 and lesson.push_velocity_delta_max_m_s > .03
    assert env.push_clock.min() >= 5. and env.push_clock.max() < 10.
    env.robot.write_root_com_velocity_to_sim_index = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="native readback"):
        native_push(env, torch.arange(32))


def test_absent_field_keeps_native_push_draws_and_velocity_exact():
    n, seed = 8, 17
    gen = torch.Generator().manual_seed(seed)
    expected_gen = torch.Generator().manual_seed(seed)
    angle, amplitude = torch.rand(n, generator=expected_gen) * math.tau, torch.rand(n, generator=expected_gen) * .1
    expected_velocity = torch.stack((angle.cos(), angle.sin()), -1) * amplitude[:, None]
    expected_clock = 3 + 2 * torch.rand(n, generator=expected_gen)
    velocities = torch.zeros(n, 6)
    env = NS(device="cpu", cfg={}, stage_actor_update=0, skills=None, stage_cfg={"push_max": .1},
        push_clock=torch.zeros(n), random=lambda count: torch.rand(count, generator=gen),
        robot=NS(root_view=NS(get_root_velocities=lambda: velocities),
            write_root_com_velocity_to_sim_index=lambda **kw: velocities.__setitem__(kw["env_ids"], kw["root_velocity"])))
    native_push(env, torch.arange(n))
    torch.testing.assert_close(velocities[:, :2], expected_velocity, atol=0., rtol=0.)
    torch.testing.assert_close(env.push_clock, expected_clock, atol=0., rtol=0.)
    assert torch.equal(gen.get_state(), expected_gen.get_state())


def test_native_reset_composes_root_and_bank_before_exactly_one_atomic_write(bound):
    lesson, bank = bound
    n = lesson.count
    writes = []
    robot = NS(write_root_link_pose_to_sim_index=lambda **kw: writes.append(("root", kw)),
        write_root_com_velocity_to_sim_index=lambda **kw: writes.append(("velocity", kw)),
        write_joint_position_to_sim_index=lambda **kw: writes.append(("q", kw)),
        write_joint_velocity_to_sim_index=lambda **kw: writes.append(("dq", kw)))
    ids, root, joints, origins, surfaces, valid = reset_buffers(lesson)
    env = NS(device="cpu", cfg={"task_campaign": {"domain": "speed"}}, task_actor_update=100,
        reset_positions=root[:, :3], nominal=joints, origins=origins, surface_data=surfaces,
        surface_valid=valid, generator=torch.Generator().manual_seed(32), robot=robot,
        skills=None, step_jump=None, recovery=None, nonstandard_start=bank,
        nonstandard_start_eligible=lesson.eligible, ordinary_posture=lesson, joint_count=len(NAMES))
    reset = method("reset")
    end = next(i for i, node in enumerate(reset.body) if mentions(node, "self.robot.write_joint_velocity_to_sim_index"))
    execute(reset.body[:end + 1], env, ids=ids, stage_actor_updates=stage_actor_updates)
    assert [name for name, _ in writes] == ["root", "velocity", "q", "dq"]
    assert writes[0][1]["root_pose"][:, 3:6].abs().max() > .01
    assert not writes[1][1]["root_velocity"].any() and not writes[3][1]["velocity"].any()
