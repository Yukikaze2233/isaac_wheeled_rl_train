"""CPU geometry, command semantics and immutable small-stair inheritance checks."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis import small_stair_revision as recipe
from wheeled_tasks.chassis import terrain_runout_revision as runout
from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis.skill_commands import SkillCommands, sample_planar_command
from wheeled_tasks.chassis.task import Phase, choose_scene_groups, terrain_surfaces
from wheeled_tasks.chassis.terrain_runout import append_route_runout, route_runout_selection
from wheeled_tasks.chassis.terrain_runout_runtime import runtime_expectation


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (recipe.DESIGN_FILE, recipe.WORKER_FILE, recipe.PLAN_FILE, recipe.MANIFEST_FILE)
PATCH_FIELDS = ("scene_groups", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions")


def read(name, root=ROOT):
    return json.loads((root / name).read_text())


def inputs():
    names = {recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE,
             recipe.SOURCE_DESIGN_FILE, recipe.OWN_SOURCE_WORKER_FILE}
    for row in read(recipe.SOURCE_PLAN_FILE)["domains"]:
        names.add(row["contract"])
        names.add(read(row["contract"])["design_preflight"]["file"])
    return names


def copy_inputs(root):
    (root / "contracts").mkdir()
    for name in inputs():
        shutil.copyfile(ROOT / name, root / name)


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("small_stair_contracts")
    copy_inputs(root)
    before = {name: (ROOT / name).read_bytes() if (ROOT / name).exists() else None
              for name in OUTPUTS}
    receipt = recipe.freeze_contracts(root)
    assert before == {name: (ROOT / name).read_bytes() if (ROOT / name).exists() else None
                      for name in OUTPUTS}
    return root, receipt


def test_pinned_inputs_direct_v21_freeze_is_exclusive_and_has_no_deployment_claim(candidate):
    root, receipt = candidate
    for name, sha in ((recipe.SOURCE_PLAN_FILE, recipe.SOURCE_PLAN_SHA),
            (recipe.SOURCE_WORKER_FILE, recipe.SOURCE_WORKER_SHA),
            (recipe.SOURCE_DESIGN_FILE, recipe.SOURCE_DESIGN_SHA),
            (recipe.OWN_SOURCE_WORKER_FILE, recipe.OWN_SOURCE_WORKER_SHA)):
        assert recipe.digest(ROOT / name) == sha
    assert all((root / name).read_bytes() == (ROOT / name).read_bytes() for name in inputs())
    assert not (root / runout.WORKER_FILE).exists()
    assert receipt["output_sha256"] == {name: recipe.digest(root / name) for name in OUTPUTS}
    assert receipt["critic_adaptation_updates"] == 10
    assert receipt["qualification"] is None
    assert receipt["deployment"] is receipt["native_runtime_verified"] is False
    with pytest.raises(ValueError, match="replace"):
        recipe.freeze_contracts(root)


def test_only_target_pool_and_version_bindings_change_other_training_semantics_exactly(candidate):
    root, _ = candidate
    old, target = read(recipe.SOURCE_WORKER_FILE), read(recipe.WORKER_FILE, root)
    assert len(target["scene_groups"]) == len(target["skill_specs"]) == 101
    restored = deepcopy(target)
    for field in PATCH_FIELDS:
        restored[field] = old[field]
    assert restored.pop("terrain_route_runout") == runout.SETTINGS
    assert restored["task_campaign"].pop("terrain_runout_revision_migration") == runout.revision_binding()
    assert restored["task_campaign"].pop("small_stair_revision_migration") == recipe.revision_binding()
    restored["task_campaign"]["worker_revision_id"] = old["task_campaign"]["worker_revision_id"]
    restored["design_preflight"] = old["design_preflight"]
    assert restored == old
    assert target["task_campaign"]["revision_migration"] == old["task_campaign"]["revision_migration"]
    old_groups = {row["name"]: row for row in old["scene_groups"]}
    new_groups = {row["name"]: row for row in target["scene_groups"]}
    for name, pool in old["behavior_pool_membership"].items():
        if pool == recipe.POOL:
            assert name not in new_groups and name not in target["skill_specs"]
        else:
            assert new_groups[name] == old_groups[name]
            assert target["skill_specs"][name] == old["skill_specs"][name]
            assert target["behavior_pool_membership"][name] == pool
    assert target["behavior_pool_fractions"] == old["behavior_pool_fractions"]
    assert not any("body_climb" in spec for spec in target["skill_specs"].values())
    assert target["evaluation"] == old["evaluation"]


def test_actual_native_surfaces_have_three_signed_risers_and_matching_terminal_support(candidate):
    root, _ = candidate
    target = read(recipe.WORKER_FILE, root)
    fractions = Counter()
    for row in target["scene_groups"]:
        if target["behavior_pool_membership"][row["name"]] != recipe.POOL:
            continue
        name, terrain = row["name"], row["terrain"][0]
        mm = int(name.removesuffix("mm").rsplit("_", 1)[1])
        spec, rise = target["skill_specs"][name], mm / 1000.
        surfaces = terrain_surfaces(terrain, spec["terrain_limits"], spec["terrain_level"])
        assert len(surfaces) == 5
        assert [surface.z0 for surface in surfaces] == pytest.approx(
            [0., rise, 2 * rise, 3 * rise, 3 * rise] if terrain == "stairs"
            else [3 * rise, 2 * rise, rise, 0., 0.])
        assert [surface.x1 - surface.x0 for surface in surfaces[1:4]] == pytest.approx([.6] * 3)
        assert surfaces[3].x1 == pytest.approx(1.8)
        assert surfaces[-1].x1 == 4.
        assert route_runout_selection(target, name, terrain)
        extended = append_route_runout(surfaces, target["terrain_route_runout"])
        assert extended[:-1] == surfaces and extended[-1].x1 == 6.2
        assert extended[-1].z0 == surfaces[-1].z0
        fractions[mm] += row["fraction"]
        assert spec["terrain_level"] == 1.
        assert spec["terrain_training_row"]["final_target_row"] is False
    assert fractions == pytest.approx({height: .42 * share for height, share in recipe.BINS})
    rows = choose_scene_groups(target["scene_groups"], 4096)
    assert len(rows) == 4096 and len(set(name for name, _ in rows)) == 101
    runtime = runtime_expectation(target)
    # The unchanged 30 surface rows also retain their original route runout.
    assert runtime["selected_group_count"] == 38
    assert runtime == read(recipe.DESIGN_FILE, root)["stages"]["terrain"]["roles"]["train"]["runtime"]["terrain_route_runout"]


def test_native_ordinary_commands_are_continuous_forward_and_do_not_trigger_step_requests(candidate):
    root, _ = candidate
    target = read(recipe.WORKER_FILE, root)
    selected = {name: spec for name, spec in target["skill_specs"].items()
                if target["behavior_pool_membership"][name] == recipe.POOL}
    generator = torch.Generator().manual_seed(23)
    random = lambda count: torch.rand(count, generator=generator)
    for name, spec in selected.items():
        assert spec["kind"] == "rolling_stairs"
        assert spec["mode"] == (2 if "_up_" in name else 3)
        assert spec["episode_seconds"] == 32.
        assert spec["command_resampling_seconds"] == [1., 8.]
        assert spec["height_sampling"]["range_m"] == [.29, .34]
        assert not set(spec) & {"manual_request_seconds", "manual_hold_seconds", "request_events",
                                "step_jump", "body_climb", "command_sequence"}
        command = torch.tensor(spec["command"]).expand(2048, -1).clone()
        sample_planar_command(command, spec, random)
        assert .3 <= command[:, 0].min() <= command[:, 0].max() <= .75
        assert -.02 <= command[:, 1].min() <= command[:, 1].max() <= .02
        assert len(command[:, 0].unique()) > 1024
    config = deepcopy(target)
    config["skill_specs"] = selected
    env = SimpleNamespace(cfg=config, num_envs=8, device="cpu", scene_groups=list(selected),
                          policy_dt=.02, random=random)
    commands = SkillCommands(env)
    assert not commands.manual_step_mode.any()
    assert commands.manual_step_start.isinf().all()
    # Keep the inherited global request protocol enabled. The actual env
    # callsite excludes these ordinary mode2/3 rows from eligible requests.
    assert commands.schedule.requests_enabled
    modes = torch.tensor([spec["mode"] for spec in selected.values()])
    eligible = (modes == 4) | (commands.manual_step_mode != 0)
    assert not eligible.any()
    schedule = commands.schedule
    schedule.request_stress.fill_(True)
    schedule.next_request.zero_()
    phases = torch.full((8,), Phase.GROUND, dtype=torch.long)
    for elapsed in (0., 2., 8., 16.):
        request = schedule.update_requests(torch.full((8,), elapsed), phases,
            eligible, torch.ones(8, dtype=torch.bool))
        assert not request.any() and not schedule.request_active.any()
        assert not schedule.accepted_request_count.any()
        assert schedule.request_start.isinf().all()


def test_all_five_roles_and_fixed_cases_retain_geometry_thresholds_and_other_runtime_fields(candidate):
    root, _ = candidate
    source, target = read(recipe.SOURCE_DESIGN_FILE), read(recipe.DESIGN_FILE, root)
    assert target == recipe.revised_design(root)
    for role, row in target["stages"]["terrain"]["roles"].items():
        old = source["stages"]["terrain"]["roles"][role]
        restored = deepcopy(row)
        restored["runtime"]["modules"].pop("terrain_route_runout")
        restored["runtime"].pop("terrain_route_runout")
        restored["contract"].pop("terrain_route_runout")
        metadata = restored["contract"]["task_campaign"]
        metadata.pop("terrain_runout_revision_migration")
        metadata.pop("small_stair_revision_migration")
        metadata["worker_revision_id"] = old["contract"]["task_campaign"]["worker_revision_id"]
        if role == "train":
            for field in PATCH_FIELDS:
                restored["contract"][field] = old["contract"][field]
        assert restored == old
    old = fixed_suite_contract(read(recipe.SOURCE_WORKER_FILE))
    exam = fixed_suite_contract(read(recipe.WORKER_FILE, root))
    assert len(exam["evaluation"]["cases"]) == 322
    assert any(case["name"] == "body_climb_double_200_250" for case in exam["evaluation"]["cases"])
    exam.pop("terrain_route_runout")
    exam["task_campaign"].pop("terrain_runout_revision_migration")
    exam["task_campaign"].pop("small_stair_revision_migration")
    exam["task_campaign"]["worker_revision_id"] = old["task_campaign"]["worker_revision_id"]
    exam["design_preflight"] = old["design_preflight"]
    assert exam == old


def test_campaign_keeps_other_five_workers_schedule_parents_budgets_and_fixed_case_lists(candidate):
    root, _ = candidate
    old, target = read(recipe.SOURCE_PLAN_FILE), read(recipe.PLAN_FILE, root)
    restored = deepcopy(target)
    restored["contract_id"] = old["contract_id"]
    restored["revision_migration"] = old["revision_migration"]
    old_row = next(row for row in old["domains"] if row["name"] == "terrain")
    row = next(row for row in restored["domains"] if row["name"] == "terrain")
    for field in ("contract", "sha256", "sampling_pools"):
        row[field] = old_row[field]
    assert restored == old
    actual = next(row for row in target["domains"] if row["name"] == "terrain")
    assert actual["sampling_pools"] == recipe.plan_sampling_pools(root)
    worker = read(recipe.WORKER_FILE, root)
    for pool in actual["sampling_pools"]:
        assert set(pool["skills"]) == {name for name, parent in worker["behavior_pool_membership"].items()
                                     if parent == pool["name"]}
    manifest = read(recipe.MANIFEST_FILE, root)
    assert manifest == recipe.revised_manifest(root)
    assert manifest["target_plan_sha256"] == recipe.digest(root / recipe.PLAN_FILE)
    assert manifest["source_plan_file"] == recipe.SOURCE_PLAN_FILE
    assert manifest["intermediate_checkpoint"] == "none_direct_v21_to_v23"
    assert manifest["task_clock"] == "restored" and manifest["consumption"] == "preserved_no_refund"
    assert len([row for row in manifest["workers"] if row["source_sha256"] != row["target_sha256"]]) == 1
    assert target["total_ceiling"] == 121000
    assert recipe.validate_frozen_workers(root)["domain"] == "terrain"


@pytest.mark.parametrize("name", [recipe.SOURCE_WORKER_FILE, recipe.OWN_SOURCE_WORKER_FILE])
def test_real_own_domain_learning_source_preserves_full_state_and_clock_with_ten_critic_updates(candidate, name):
    root, _ = candidate
    source, target = read(name), read(recipe.WORKER_FILE, root)
    binding = recipe.validate_worker_revision(source, target, root)
    assert binding["source_contract_file"] == name
    assert binding["inheritance_path"][-1] == recipe.WORKER_SCHEMA
    assert binding["actor_critic_optimizer"] == "preserved"
    assert binding["task_clock"] == "restored"
    assert binding["consumption"] == "preserved_no_refund"
    assert binding["critic_adaptation_updates"] == 10
    assert binding["intermediate_checkpoint"] == "none_direct_v21_to_v23"


@pytest.mark.parametrize("path,value", [
    (("physics_dt",), .001), (("learning_rate",), .0002),
    (("resume_critic_warmup_updates",), 0), (("success_hold_seconds",), .1),
    (("terrain_route_runout", "enabled"), 1),
    (("task_campaign", "clock"), "reset"),
    (("skill_specs", "small_stairs_up_005mm", "terrain_limits", "stair_heights_m"), [.005, .01, .2]),
    (("skill_specs", "small_stairs_down_040mm", "kind"), "stairs_down"),
])
def test_full_recipe_rejects_undeclared_geometry_control_gate_clock_or_trigger_changes(candidate, path, value):
    root, _ = candidate
    target = read(recipe.WORKER_FILE, root)
    leaf = target
    for field in path[:-1]:
        leaf = leaf[field]
    leaf[path[-1]] = value
    with pytest.raises(ValueError, match="outside"):
        recipe.validate_worker_revision(read(recipe.OWN_SOURCE_WORKER_FILE), target, root)


@pytest.mark.parametrize("name", OUTPUTS)
def test_existing_output_blocks_any_partial_or_replacement_freeze(tmp_path, name):
    copy_inputs(tmp_path)
    (tmp_path / name).write_bytes(b"occupied\n")
    with pytest.raises(ValueError, match="replace"):
        recipe.freeze_contracts(tmp_path)
    assert (tmp_path / name).read_bytes() == b"occupied\n"
    assert all(not (tmp_path / other).exists() for other in OUTPUTS if other != name)
