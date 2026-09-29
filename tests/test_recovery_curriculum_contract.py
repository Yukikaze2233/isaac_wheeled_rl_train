"""Independently verify recovery scope, preserved skills, budgets and all roles."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis.full_curriculum import stage_contract
from wheeled_tasks.chassis.preflight import design_preflight, validate_runtime_state
from wheeled_tasks.chassis.response_migration import validate_response_resume
from wheeled_tasks.chassis.skill_curriculum import skill_spec

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("S1", "S2", "S3", "S4", "S5", "S5T", "S5D", "S6")
ROLES = ("train", "baseline", "quick", "full", "confirmation")


def read(name):
    return json.loads((ROOT / name).read_text())


@lru_cache(maxsize=1)
def plans():
    return (read("contracts/v6_recovery_curriculum_training_v1.json"),
            read("contracts/v6_step_curriculum_training_v1.json"))


def config_for(index, role="train", *, previous=False):
    plan = plans()[int(previous)]
    cfg = stage_contract({}, plan, plan["stages"][index], 20000)
    seed = 290927 if role == "train" else 290930 if role == "confirmation" else 290929
    if role != "train":
        cfg["evaluation"]["seed"] = seed
        cfg = fixed_suite_contract(cfg)
    count = 20000 if role == "train" else len(cfg["evaluation"]["cases"]) * cfg["evaluation"]["episodes_per_case"]
    return cfg, seed, count


def test_frozen_files_payload_budget_and_claim_are_authenticated():
    plan, old = plans()
    assert plan["contract_id"] == old["contract_id"]
    assert plan["curriculum_revision"] == "v6_recovery_curriculum_v1"
    assert [s["name"] for s in plan["stages"]] == list(STAGES)
    assert [s["updates"] for s in plan["stages"]] == [1000, 2000, 1000, 1500, 1000, 600, 400, 1000]
    assert plan["total_update_ceiling"] == 8500 and plan["target_num_envs"] == 20000
    budget = read(plan["budget_spec_file"])
    assert budget["budget_id"] == "v6-recovery-curriculum-budget-v1"
    assert budget["formal"] == read(old["budget_spec_file"])["formal"]
    assert plan["release_scope"]["R0"] == "scripted_physical_recovery_then_RL_handoff_separate_acceptance"
    cfg = plan["runtime_base"]["recovery_training"]
    files = [(plan["budget_spec_file"], plan["budget_spec_sha256"]),
             (plan["design_preflight"]["file"], plan["design_preflight"]["sha256"])]
    files += [(cfg[prefix + "_file"], cfg[prefix + "_sha256"]) for prefix in ("profiles", "height_lookup")]
    for name, sha in files:
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    payload = deepcopy(plan)
    payload.pop("design_preflight")
    assert hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode()).hexdigest() == read(plan["design_preflight"]["file"])["reviewed_plan_payload_sha256"]


@pytest.mark.parametrize("index", range(8))
@pytest.mark.parametrize("role", ROLES)
def test_all_forty_frozen_roles_and_actual_module_expectations(index, role):
    cfg, seed, count = config_for(index, role)
    control = read(cfg["control_math_source"])
    result = design_preflight(cfg, control, ROOT, stage=cfg["enabled_stages"][0],
                              role=role, seed=seed, num_envs=count)
    assert result["status"] == "static_passed"
    expected = result["runtime_expected"]
    assert expected["modules"]["recovery_training"] is True
    observed = deepcopy(expected)
    assert validate_runtime_state(expected, observed)["status"] == "runtime_configuration_passed"
    observed["modules"]["recovery_training"] = False
    with pytest.raises(ValueError, match="recovery_training"):
        validate_runtime_state(expected, observed)


@pytest.mark.parametrize("index", range(8))
def test_old_case_order_and_definitions_unchanged_recovery_is_separate(index):
    cfg, _, _ = config_for(index)
    old, _, _ = config_for(index, previous=True)
    cases, before = cfg["evaluation"]["cases"], old["evaluation"]["cases"]
    assert len(before) == 268
    assert cases[:268] == before
    additions = cases[268:]
    assert len(additions) == 54
    assert all(case["recovery_case"] and case["skill"]["kind"] == "recovery_handoff" for case in additions)
    assert [(terrain, sum(case["terrain"] == terrain for case in additions))
            for terrain in ("flat", "step_up", "stairs")] == [("flat", 34), ("step_up", 10), ("stairs", 10)]
    assert len({case["skill"]["recovery_pose"] for case in additions}) == 26
    names = {case["name"] for case in additions}
    assert names <= set(cfg["evaluation"]["retention_case_names"])
    for case in additions:
        taught = STAGES.index(case["introduced_stage"]) <= index
        assert (case["name"] in cfg["evaluation"]["promotion_case_names"]) is taught
        assert case["command"] == case["skill"]["command"] == case["skill"]["recovery_followup_command"]
        expected_stage = {"flat": "S3", "step_up": "S4", "stairs": "S5D"}[case["terrain"]]
        assert case["introduced_stage"] == expected_stage


@pytest.mark.parametrize("index", range(8))
def test_explicit_deductions_preserve_standing_motion_height_and_previous_recovery(index):
    cfg, _, _ = config_for(index)
    old, _, _ = config_for(index, previous=True)
    before, after = old["behavior_pool_fractions"], cfg["behavior_pool_fractions"]
    if index < 2:
        for key in ("scene_groups", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions",
                    "new_asset_curriculum", "stages", "episode_seconds"):
            assert cfg[key] == old[key]
        return
    assert after["scripted_recovery"] == .05
    for name in ("stand", "motion", "height", "recovery"):
        assert after[name] == before[name]
    expected = {"S3": {"terrain": .30}, "S4": {"steps": .45}, "S5": {"jumps": .45},
                "S5T": {"step_jumps": .30}, "S5D": {"double_step_jumps": .125, "double_body_climb": .125},
                "S6": {"steps": .15}}[STAGES[index]]
    assert {name: after[name] for name in expected} == expected
    assert sum(after.values()) == pytest.approx(1.)
    assert sum(group["fraction"] for group in cfg["scene_groups"]) == pytest.approx(1.)
    assert cfg["new_asset_curriculum"]["edge_reset_fraction"] == .05
    assert old["new_asset_curriculum"]["edge_reset_fraction"] == .1


def test_recovery_extension_can_resume_s1_only_and_keeps_every_previous_gate():
    source, _, _ = config_for(0, previous=True)
    target, _, _ = config_for(0)
    receipt = validate_response_resume(source, target)
    assert len(receipt["added_evaluation_case_names"]) == 54
    assert source["evaluation"]["promotion_case_names"] == target["evaluation"]["promotion_case_names"]
    before, _, _ = config_for(1, previous=True)
    after, _, _ = config_for(1)
    with pytest.raises(ValueError, match="S1 source"):
        validate_response_resume(before, after)


@pytest.mark.parametrize("override", [{"recovery_pose": ""}, {"recovery_followup_command": [0., float("nan"), .305]},
                                      {"command": [.5, 0., .305]}])
def test_recovery_skill_rejects_missing_geometry_and_mismatched_followup(override):
    recipe = {"skill": "recovery_handoff", "recovery_pose": "upright", "command": [0., 0., .305],
              "recovery_followup_command": [0., 0., .305], **override}
    with pytest.raises(ValueError, match="Recovery handoff"):
        skill_spec(recipe)


def test_platform_recovery_preserves_sealed_terrain_and_required_top():
    plan, _ = plans()
    profiles = read(plan["runtime_base"]["recovery_training"]["profiles_file"])["profiles"]
    cfg, _, _ = config_for(7)
    for case in cfg["evaluation"]["cases"][268:]:
        profile = profiles[case["skill"]["recovery_pose"]]
        assert profile["enabled"] and profile["geometry_valid"]
        assert not profile["dynamic_recovery_validated"]
        if case["terrain"] == "flat":
            continue
        for key in ("required_platform_height_m", "min_ending_x"):
            assert case[key] == case["skill"][key] == profile[key]
        limits = case["terrain_limits"]
        if case["terrain"] == "step_up":
            assert limits["step_up_m"] == .2 and case["required_platform_height_m"] == .2
        else:
            assert limits["stair_heights_m"] == [.2, .45]
            assert limits["stair_tread_m"] == [.8, .6]
            assert case["required_platform_height_m"] in (.2, .45)
