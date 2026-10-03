"""Reset sampling and geometry checks, without claiming dynamic recovery."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from wheeled_tasks.chassis.recovery_curriculum import (
    RecoveryResetCurriculum, pose_probabilities, validate_pose_curriculum,
)


ROOT = Path(__file__).resolve().parents[1]
V1_SHA = "5b09a3bb27285ab7571bd133151d7118091997309ccc9d59c6b1259836dc1eef"


@pytest.fixture
def table():
    return json.loads((ROOT / "contracts/v6_recovery_profiles_v2.json").read_text())


@pytest.fixture
def settings():
    draft = json.loads((ROOT / "contracts/v6_recovery_pose_curriculum_draft_v1.json").read_text())
    return draft["recovery_training"]["pose_curriculum"]


def make_env(table, count=10, seed=519, only_profile=None):
    poses = [only_profile] if only_profile else ["front_down_90", "step200_lip_near_shell",
        "double_step_first_riser_split", "double_step_second_side_top", None]
    groups = [f"lane_{index % len(poses)}" for index in range(count)]
    specs = {f"lane_{index}": {"kind": "recovery_handoff", "recovery_pose": pose}
             for index, pose in enumerate(poses) if pose is not None}
    names = list(reversed(table["tree_joint_names"]))
    return SimpleNamespace(num_envs=count, device="cpu", cfg={
        "asset_manifest_sha256": table["asset_manifest_sha256"]},
        scene_groups=groups, kinds=[table["profiles"][poses[i % len(poses)]]["terrain"]
                                   if poses[i % len(poses)] else "flat" for i in range(count)],
        skills=SimpleNamespace(specs=specs), robot=SimpleNamespace(joint_names=names),
        nominal=torch.zeros(count, len(names)), generator=torch.Generator().manual_seed(seed))


def test_frozen_actor_origin_and_interpolation(settings):
    settings["origin_actor_update"] = 700
    validate_pose_curriculum(settings)
    assert pose_probabilities(settings, 0) == (.9, .1, 0.)
    assert pose_probabilities(settings, 700) == (.9, .1, 0.)
    assert pose_probabilities(settings, 1200) == pytest.approx((.7, .225, .075))
    assert pose_probabilities(settings, 1700) == (.5, .35, .15)
    assert pose_probabilities(settings, 3700) == (.25, .35, .4)
    assert pose_probabilities(settings, 6700) == (.2, .2, .6)
    assert pose_probabilities(settings, 99999) == (.2, .2, .6)


@pytest.mark.parametrize("clock", [-1, .5, True, float("nan")])
def test_invalid_actor_clock_rejected(settings, clock):
    with pytest.raises(ValueError, match="actor update"):
        pose_probabilities(settings, clock)


@pytest.mark.parametrize("change", [
    lambda s: s.update(origin_actor_update=-1),
    lambda s: s.update(origin_actor_update=True),
    lambda s: s.update(clock="global_actor_updates"),
    lambda s: s.update(sampling="every_tick"),
    lambda s: s.update(interpolation="guess"),
    lambda s: s.update(unreviewed_fallback=True),
    lambda s: s.update(keyframes=[]),
    lambda s: s["keyframes"][0].update(actor_update=1),
    lambda s: s["keyframes"][1].update(actor_update=0),
    lambda s: s["keyframes"][0]["probabilities"].update(easy=.8),
    lambda s: s["keyframes"][0]["probabilities"].update(easy=float("nan")),
    lambda s: s["keyframes"][0]["probabilities"].update(easy=1.1, medium=-.1),
    lambda s: s["keyframes"][0]["probabilities"].pop("hard"),
    lambda s: s["profile_sets"]["flat"].update(easy=[]),
    lambda s: s["profile_sets"]["flat"].update(easy=["upright", "upright"]),
    lambda s: s["profile_sets"]["flat"].update(required_platform_height_m=.2),
    lambda s: s["profile_sets"].update(duplicate=deepcopy(s["profile_sets"]["flat"])),
])
def test_bad_schedule_rejected(settings, change):
    change(settings)
    with pytest.raises(ValueError):
        validate_pose_curriculum(settings)


def test_reset_only_gather_preserves_other_lanes_and_scene(table, settings):
    env = make_env(table)
    before = deepcopy((env.scene_groups, env.kinds, env.skills.specs))
    sampler = RecoveryResetCurriculum(env, table, settings)
    # The simulator order intentionally differs from the table order.
    upright = sampler.profile_index["upright"]
    assert sampler.joint_pose_bank[upright].tolist() == pytest.approx([
        table["profiles"]["upright"]["full_joint_positions"][name] for name in env.robot.joint_names])
    selected = sampler.reset(torch.arange(10), 0)
    assert selected[4] == selected[9] == -1
    assert sampler.level_ids[sampler.enabled].max() <= 1
    old_profiles, old_levels = sampler.profile_ids.clone(), sampler.level_ids.clone()
    selected = sampler.reset(torch.tensor([0, 2, 4]), 6000)
    assert selected[-1] == -1
    unchanged = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])
    assert torch.equal(sampler.profile_ids[unchanged], old_profiles[unchanged])
    assert torch.equal(sampler.level_ids[unchanged], old_levels[unchanged])
    assert int(sampler.profile_reset_counts.sum()) == 10
    assert int(sampler.level_reset_counts.sum()) == 10
    assert (env.scene_groups, env.kinds, env.skills.specs) == before
    assert env.nominal.eq(0).all()
    assert sampler.root_pose_bank[selected[:2]].shape == (2, 7)
    assert sampler.initial_release_bank.shape == sampler.required_platform_height_bank.shape


def test_generator_reproducibility_and_resume_origin(table, settings):
    settings["origin_actor_update"] = 300
    env1, env2 = make_env(table, count=100), make_env(table, count=100)
    first, second = (RecoveryResetCurriculum(env, table, settings) for env in (env1, env2))
    ids = torch.arange(100)
    torch.manual_seed(1)
    one = first.reset(ids, 1300)
    torch.manual_seed(999)
    two = second.reset(ids, 1300)
    assert torch.equal(one, two)
    assert torch.equal(env1.generator.get_state(), env2.generator.get_state())
    # A reconstructed sampler uses the same frozen absolute origin and restored RNG.
    restored = make_env(table, count=100)
    restored.generator.set_state(env1.generator.get_state())
    resumed = RecoveryResetCurriculum(restored, table, settings)
    assert torch.equal(first.reset(ids, 3300), resumed.reset(ids, 3300))
    settings["origin_actor_update"] = 9999
    assert resumed.settings["origin_actor_update"] == 300


def test_hard_zero_initially_then_uses_original_lane_and_retains_easy(table, settings):
    env = make_env(table, count=8192, only_profile="front_down_90")
    sampler = RecoveryResetCurriculum(env, table, settings)
    ids = torch.arange(env.num_envs)
    hard = sampler.profile_index["front_down_90"]
    assert not (sampler.reset(ids, 0) == hard).any()
    chosen = sampler.reset(ids, 6000)
    assert torch.equal(chosen == hard, sampler.level_ids == 2)
    fractions = torch.bincount(sampler.level_ids, minlength=3).float() / env.num_envs
    assert fractions.tolist() == pytest.approx([.2, .2, .6], abs=.02)


@pytest.mark.parametrize("surface,hard,height", [
    ("first", "double_step_first_riser_split", .2),
    ("second", "double_step_second_side_top", .45),
])
def test_executable_stair_beginner_pool_uses_upright_without_removing_hard_pose(
        table, surface, hard, height):
    from wheeled_tasks.chassis.recovery_revision import pose_curriculum

    settings = pose_curriculum(400)
    upright = f"double_step_{surface}_upright_top"
    pool = settings["profile_sets"][f"stairs_{surface}"]
    assert pool["easy"] == pool["medium"] == [upright]
    env = make_env(table, count=8192, only_profile=hard)
    sampler = RecoveryResetCurriculum(env, table, settings)
    ids = torch.arange(env.num_envs)
    chosen = sampler.reset(ids, 400)
    assert chosen.eq(sampler.profile_index[upright]).all()
    assert sampler.required_platform_height_bank[chosen].eq(height).all()
    assert not sampler.level_ids.eq(2).any()
    # Narrowing the reset pool preserves every lane's original difficult pose.
    chosen = sampler.reset(ids, 6400)
    original = sampler.profile_index[hard]
    assert sampler.level_ids.eq(2).any()
    assert torch.equal(chosen.eq(original), sampler.level_ids.eq(2))
    nonhard = sampler.level_ids != 2
    assert chosen[nonhard].eq(sampler.profile_index[upright]).all()
    v1 = json.loads((ROOT / "contracts/v6_recovery_profiles_v1.json").read_text())
    assert table["profiles"][hard] == v1["profiles"][hard]


@pytest.mark.parametrize("pool,bad", [
    ("flat", "missing_profile"),
    ("flat", "step200_upright_top"),
    ("stairs_first", "double_step_second_upright_top"),
    ("step200", "double_step_first_upright_top"),
])
def test_cross_terrain_or_surface_candidates_rejected(table, settings, pool, bad):
    settings["profile_sets"][pool]["easy"] = [bad]
    with pytest.raises(ValueError, match="profile/terrain mismatch"):
        RecoveryResetCurriculum(make_env(table), table, settings)


def test_exact_evaluation_and_wrong_lane_terrain_rejected(table, settings):
    env = make_env(table)
    env.cfg["evaluation_exact_cases"] = True
    with pytest.raises(ValueError, match="Exact recovery evaluation"):
        RecoveryResetCurriculum(env, table, settings)
    env.cfg.pop("evaluation_exact_cases")
    env.kinds[0] = "stairs"
    with pytest.raises(ValueError, match="lane profile/terrain mismatch"):
        RecoveryResetCurriculum(env, table, settings)


@pytest.mark.parametrize("change", [
    lambda t: t.update(asset_manifest_sha256="0" * 64),
    lambda t: t.update(tree_joint_names=t["tree_joint_names"][:-1]),
    lambda t: t["profiles"]["upright"].update(enabled=False),
    lambda t: t["profiles"]["upright"].update(geometry_valid=False),
    lambda t: t["profiles"]["upright"].update(root_linear_velocity_m_s=[1., 0., 0.]),
    lambda t: t["profiles"]["upright"].update(root_quaternion_xyzw=[0., 0., 0., 0.]),
])
def test_invalid_bank_rejected(table, settings, change):
    env = make_env(table)
    change(table)
    with pytest.raises(ValueError):
        RecoveryResetCurriculum(env, table, settings)


def test_inactive_reset_does_not_consume_randomness(table, settings):
    env = make_env(table)
    sampler = RecoveryResetCurriculum(env, table, settings)
    rng = env.generator.get_state().clone()
    assert sampler.reset(torch.tensor([4, 9]), 1000).tolist() == [-1, -1]
    assert torch.equal(env.generator.get_state(), rng)
    assert sampler.reset(torch.tensor([], dtype=torch.long), 1000).numel() == 0
    assert sampler.profile_ids.eq(-1).all()


def test_actual_selection_metrics_and_diagnostics_keep_tensor_counters(table, settings):
    settings["origin_actor_update"] = 123
    sampler = RecoveryResetCurriculum(make_env(table), table, settings)
    sampler.reset(torch.arange(10), 123)
    prefix = "/recovery/curriculum/"
    metrics = sampler.metrics(1123)
    assert metrics[prefix + "relative_actor_updates"] == 1000.
    assert metrics[prefix + "easy_probability"] == .5
    assert metrics[prefix + "hard_probability"] == .15
    # Current episode selections still belong to their original reset time.
    assert metrics[prefix + "hard_actuallevel_fraction"] == 0.
    assert sum(metrics[prefix + level + "_actuallevel_fraction"]
               for level in ("easy", "medium", "hard")) == pytest.approx(1.)
    counters = [value for key, value in metrics.items() if "/profile_reset_count/" in key]
    assert len(counters) == 30 and all(isinstance(value, torch.Tensor) for value in counters)
    assert sum(counters) == 8
    diagnostics = sampler.diagnostics()
    before = diagnostics["recovery_actual_profile_id"].clone()
    sampler.reset(torch.arange(10), 6123)
    assert torch.equal(diagnostics["recovery_actual_profile_id"], before)
    assert diagnostics["recovery_actual_profile_id"][4] == -1


def test_v2_preserves_v1_and_has_explicit_frozen_provenance(table):
    v1_path = ROOT / "contracts/v6_recovery_profiles_v1.json"
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == V1_SHA
    v1 = json.loads(v1_path.read_text())
    assert table["source_profiles_sha256"] == V1_SHA
    assert table["source_profiles_file"] == "contracts/v6_recovery_profiles_v1.json"
    assert table["generator_sha256"] == hashlib.sha256(
        (ROOT / "scripts/build_v6_recovery_profiles.py").read_bytes()).hexdigest()
    assert len(table["profiles"]) == 30
    for name, profile in v1["profiles"].items():
        assert table["profiles"][name] == profile
    assert table["targets"] == v1["targets"]
    assert table["prepare_reference"] == v1["prepare_reference"]
    assert all(row["geometry_valid"] and not row["dynamic_recovery_validated"]
               for row in table["profiles"].values())


def test_new_stair_easy_poses_are_closed_and_on_the_correct_physical_surface(table):
    path = ROOT / "scripts/build_v6_recovery_profiles.py"
    spec = importlib.util.spec_from_file_location("curriculum_profile_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = module.RecoveryProfileBuilder()
    for surface, height, lower_x, upper_x in (("first", .2, 0., .8), ("second", .45, .8, 1.4)):
        for posture in ("upright", "crouched"):
            name = f"double_step_{surface}_{posture}_top"
            profile = table["profiles"][name]
            assert profile["terrain"] == "stairs"
            assert profile["required_platform_height_m"] == height
            assert profile["initial_release_s"] == 0.
            assert profile["full_joint_positions"] == table["profiles"][posture]["full_joint_positions"]
            assert builder.validate_q(profile["full_joint_positions"])["max_closure_gap_m"] < 1e-8
            centers = profile["geometry_validation"]["wheel_centers_world_m"]
            assert all(lower_x < center[0] < upper_x for center in centers)
            assert max(profile["geometry_validation"]["wheel_bottom_clearance_over_center_surface_m"]) < .006
            rotation = Rotation.from_quat(profile["root_quaternion_xyzw"]).as_matrix()
            for row in builder.collision_bounds(profile["full_joint_positions"], rotation):
                points = row["points"] + np.array(profile["root_position_m"])
                assert points[:, 2].min() > .0028
                for plane in profile["terrain_platforms"]:
                    lowest = builder.halfspace_min_z(points, plane["start_x_m"])
                    if lowest is not None:
                        assert lowest >= plane["height_m"] + .0029, name


def test_draft_profiles_hash_matches_new_table():
    draft = json.loads((ROOT / "contracts/v6_recovery_pose_curriculum_draft_v1.json").read_text())
    binding = draft["recovery_training"]
    assert binding["profiles_sha256"] == hashlib.sha256((ROOT / binding["profiles_file"]).read_bytes()).hexdigest()
    assert draft["status"].startswith("nonexecutable")
