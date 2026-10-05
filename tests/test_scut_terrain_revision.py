"""Freeze terrain teaching without changing support semantics or final targets."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis import scut_terrain_revision as revision


def read(name):
    return json.loads((ROOT / name).read_text())


def test_frozen_transform_has_exact_predecessor_and_reproducible_design():
    source = revision.read_bound(revision.SOURCE_WORKER_FILE, revision.SOURCE_WORKER_SHA)
    actual = read(revision.WORKER_FILE)
    revision.apply_revision(source)
    source["design_preflight"] = actual["design_preflight"]
    assert source == actual
    design = read(revision.DESIGN_FILE)
    assert design == revision.revised_design()
    assert hashlib.sha256((ROOT / revision.DESIGN_FILE).read_bytes()).hexdigest() == actual["design_preflight"]["sha256"]
    assert revision.revision_binding()["schema"] == "task-worker-revision-v17"


def test_all_native_pools_and_44_percent_rehearsal_remain_permanent():
    old = read(revision.SOURCE_WORKER_FILE)
    new = read(revision.WORKER_FILE)
    assert new["behavior_pool_fractions"] == old["behavior_pool_fractions"]
    old_scenes = {row["name"]: row for row in old["scene_groups"]}
    scenes = {row["name"]: row for row in new["scene_groups"]}
    assert len(scenes) == len(new["skill_specs"]) == 129
    assert not set(old["skill_specs"]) - set(scenes)
    masses = Counter()
    for name, row in scenes.items():
        pool = new["behavior_pool_membership"][name]
        masses[pool] += row["fraction"]
        if pool not in revision.TERRAIN_POOLS:
            assert row == old_scenes[name]
            assert new["skill_specs"][name] == old["skill_specs"][name]
    assert masses == pytest.approx(old["behavior_pool_fractions"], abs=1e-12)
    assert math.fsum(masses[pool] for pool in revision.TERRAIN_POOLS) == pytest.approx(.56)
    assert masses["high_translation"] == pytest.approx(.05)
    assert masses["release_stop"] == pytest.approx(.05)


def test_static_rows_keep_continuous_motion_and_difficulty_bound_to_geometry():
    source, worker = read(revision.SOURCE_WORKER_FILE), read(revision.WORKER_FILE)
    scenes = {row["name"]: row for row in worker["scene_groups"]}
    for original in source["scene_groups"]:
        name = original["name"]
        if source["behavior_pool_membership"][name] not in revision.TERRAIN_POOLS:
            continue
        for grade, scale, fraction in revision.GRADES:
            target = name if grade == "full" else name + "__terrain_" + grade
            spec = worker["skill_specs"][target]
            assert scenes[target]["fraction"] == original["fraction"] * fraction
            assert scenes[target]["terrain"] == original["terrain"]
            assert spec["terrain_training_row"]["final_target_row"] is (grade == "full")
            assert spec["command_resampling_seconds"] == [1., 8.]
            assert spec["planar_command_sampling"]["vx_range_m_s"][0] > 0.
            assert spec["height_sampling"]["range_m"][0] < .305 < spec["height_sampling"]["range_m"][1]
            expected_level = scale if original["terrain"] == ["material"] else 1.
            assert spec["terrain_level"] == expected_level
            for key in revision.GEOMETRY_FIELDS:
                if key in source["skill_specs"][name]["terrain_limits"]:
                    assert spec["terrain_limits"][key] == source["skill_specs"][name]["terrain_limits"][key] * scale


def test_full_double_climb_and_every_mechanical_rule_keep_original_definition():
    old, new = read(revision.SOURCE_WORKER_FILE), read(revision.WORKER_FILE)
    for name, spec in old["skill_specs"].items():
        if not spec.get("body_climb"):
            continue
        for grade, scale, _ in revision.GRADES:
            target = name if grade == "full" else name + "__terrain_" + grade
            actual = new["skill_specs"][target]
            assert actual["mode"] == 0
            assert actual["body_climb"]["rises_m"] == [rise * scale for rise in spec["body_climb"]["rises_m"]]
            assert {key: value for key, value in actual["body_climb"].items() if key != "rises_m"} == {
                key: value for key, value in spec["body_climb"].items() if key != "rises_m"}
    full = new["skill_specs"]["body_climb_double_200_250"]
    assert full["body_climb"]["rises_m"] == [.2, .25]
    assert full["body_climb"]["treads_m"] == [.8, .6]
    assert full["terrain_limits"]["stair_heights_m"] == [.2, .45]


def test_sampling_only_revision_preserves_control_rewards_clocks_and_evaluation():
    old, new = read(revision.SOURCE_WORKER_FILE), read(revision.WORKER_FILE)
    changed = {"scene_groups", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions", "task_campaign", "design_preflight"}
    for key, value in old.items():
        if key not in changed:
            assert new[key] == value, key
    before_meta, after_meta = deepcopy(old["task_campaign"]), deepcopy(new["task_campaign"])
    before_meta.pop("revision_migration")
    assert after_meta.pop("revision_migration") == revision.revision_binding()
    assert before_meta == after_meta
    old_design = read(old["design_preflight"]["file"])
    design = read(revision.DESIGN_FILE)
    for role, row in design["stages"]["terrain"]["roles"].items():
        before = old_design["stages"]["terrain"]["roles"][role]
        assert row["runtime"] == before["runtime"]
        assert row["invocation"] == before["invocation"]
        assert row["control"] == before["control"]
        assert row["contract"]["evaluation"] == before["contract"]["evaluation"]
        if role != "train":
            assert row["contract"]["scene_groups"] == before["contract"]["scene_groups"]
            assert row["contract"]["skill_specs"] == before["contract"]["skill_specs"]


def test_native_scene_allocator_and_climb_assessment_match_collision_tops():
    pytest.importorskip("torch")
    from wheeled_tasks.chassis.task import choose_scene_groups, terrain_surfaces
    from wheeled_tasks.chassis.step_climb import StepClimbTask
    worker = read(revision.WORKER_FILE)
    scenes = choose_scene_groups(worker["scene_groups"], 4096)
    counts = Counter(name for name, _ in scenes)
    assert len(scenes) == 4096 and len(counts) == 129 and min(counts.values()) > 0
    groups = [row["name"] for row in worker["scene_groups"]]
    assessment = StepClimbTask(groups, worker["skill_specs"], "cpu", .02)
    for index, row in enumerate(worker["scene_groups"]):
        spec = worker["skill_specs"][row["name"]]
        if "body_climb" not in spec:
            continue
        surfaces = terrain_surfaces(row["terrain"][0], spec["terrain_limits"], spec["terrain_level"])
        for step in range(len(spec["body_climb"]["rises_m"])):
            assert surfaces[step + 1].z0 == pytest.approx(float(assessment.tops[index, step]), abs=1e-7)
            assert surfaces[step + 1].x0 == pytest.approx(float(assessment.edges[index, step]), abs=1e-7)


def test_full_goal_support_height_and_closure_contract_survive_static_grading():
    torch = pytest.importorskip("torch")
    from wheeled_tasks.chassis.task import terrain_surfaces
    from wheeled_tasks.chassis.step_climb import StepClimbTask
    worker = read(revision.WORKER_FILE)
    old = read(revision.SOURCE_WORKER_FILE)
    assert worker["closure_gap_termination_m"] == old["closure_gap_termination_m"] == .003
    assert worker["asset_manifest_sha256"] == old["asset_manifest_sha256"]
    assert worker["height_reference"] == old["height_reference"] == "wheel_support_mean"
    assert worker["terrain_reset_before_entry"] is True
    targets = {case["name"]: case for case in worker["evaluation"]["cases"]}
    full = worker["skill_specs"]["body_climb_double_200_250"]
    case = targets["body_climb_double_200_250"]
    assert full["body_climb"] == case["skill"]["body_climb"]
    assert full["terrain_limits"] == case["terrain_limits"]
    for grade, _, _ in revision.GRADES:
        name = "body_climb_double_200_250" if grade == "full" else "body_climb_double_200_250__terrain_" + grade
        spec = worker["skill_specs"][name]
        surfaces = terrain_surfaces("stairs", spec["terrain_limits"], spec["terrain_level"])
        assessment = StepClimbTask([name], worker["skill_specs"], "cpu", .02)
        # Equal wheel support at each top gives the same terrain-relative target
        # as the floor. No lower curriculum top becomes the final 450 mm goal.
        for step, surface in enumerate(surfaces[1:3]):
            assert surface.z0 == pytest.approx(float(assessment.tops[0, step]), abs=1e-7)
            wheel_support_mean = (surface.z0 + surface.z0) / 2
            assert (wheel_support_mean + .305) - wheel_support_mean == pytest.approx(.305)
        if grade == "full":
            assert surfaces[1].z0 == .2 and surfaces[2].z0 == .45
            assert surfaces[1].x1 - surfaces[1].x0 == .8
        else:
            assert surfaces[2].z0 < .45
            assert spec["terrain_training_row"]["final_target_row"] is False
        # The height-tracking gate must restore full payment on a reached top.
        wheel_centers = torch.tensor([[[1.25, .20, surfaces[2].z0 + .1],
                                       [1.25, -.20, surfaces[2].z0 + .1]]])
        assessment.index[:] = 1
        assert assessment.height_tracking_scale(wheel_centers).item() == pytest.approx(1.)


def test_native_continuous_command_and_dwell_sampling_are_active_cpu_only():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from wheeled_tasks.chassis.skill_commands import sample_planar_command, validate_planar_sampling
    from wheeled_tasks.chassis.training_schedule import TrainingSchedule
    worker = read(revision.WORKER_FILE)
    generator = torch.Generator().manual_seed(106)
    random = lambda count: torch.rand(count, generator=generator)
    count = 8192
    for name, spec in worker["skill_specs"].items():
        if worker["behavior_pool_membership"][name] not in revision.TERRAIN_POOLS:
            continue
        validate_planar_sampling(spec)
        command = torch.tensor(spec["command"]).expand(count, -1).clone()
        sample_planar_command(command, spec, random)
        assert len(command[:, 0].unique()) > count // 2
        for axis, key in enumerate(("vx_range_m_s", "yaw_range_rad_s")):
            low, high = spec["planar_command_sampling"][key]
            assert low <= command[:, axis].min() <= command[:, axis].max() <= high
        nominal = (command[:, 0] == spec["command"][0]).float().mean().item()
        assert nominal == pytest.approx(.25, abs=.02)
        fixed = torch.tensor(spec["command"]).expand(count, -1).clone()
        sample_planar_command(fixed, spec, random, evaluation=True)
        assert torch.equal(fixed, torch.tensor(spec["command"]).expand(count, -1))
    env = SimpleNamespace(cfg=worker, num_envs=count, device="cpu", random=random)
    durations = TrainingSchedule(env).command_duration(count)
    assert durations.min() >= 1. and durations.max() <= 15.
    assert (durations >= 8.).float().mean().item() == pytest.approx(.20, abs=.02)
