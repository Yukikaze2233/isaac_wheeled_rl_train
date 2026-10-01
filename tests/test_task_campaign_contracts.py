"""Task separation preserves physical identity and exact inherited acceptance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
from wheeled_tasks.chassis.preflight import ROLES, validate_design_contract
from wheeled_tasks.chassis.task_campaign import validate_plan, validate_task_resume

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_task_campaign import DOMAIN_ORDER, HEIGHTS, PARENTS, SOURCE_PLAN, STAND


def read(path):
    return json.loads((ROOT / path).read_text())


@pytest.fixture(scope="module")
def plan():
    return read("contracts/v6_task_campaign_v1.json")


def effective_role(worker, domain, role):
    effective = deepcopy(worker)
    seed, count = 290927, 20000
    if role != "train":
        seed = 290930 if role == "confirmation" else 290929
        effective["evaluation"]["seed"] = seed
        effective["evaluation"] = select_evaluation_cases(effective["evaluation"], domain["required_case_names"])
        effective = fixed_suite_contract(effective)
        count = 4 * len(domain["required_case_names"])
    return effective, seed, count


def test_frozen_hashes_topology_and_nonrefundable_account(plan):
    validate_plan(plan)
    assert [d["name"] for d in plan["domains"]] == list(DOMAIN_ORDER)
    assert plan["historical_consumed_updates"] == 1000
    assert plan["historical_actor_consumed_updates"] == 980
    assert plan["total_ceiling"] == 1000 + 6 * 20000
    assert plan["domain_completion"]["continue_after_failed_assessment"]
    assert not plan["domain_completion"]["integrated_policy_qualified"]
    for domain in plan["domains"]:
        assert domain["parent_domain"] == PARENTS[domain["name"]]
        assert hashlib.sha256((ROOT / domain["contract"]).read_bytes()).hexdigest() == domain["sha256"]
        worker = read(domain["contract"])
        binding = worker["design_preflight"]
        assert hashlib.sha256((ROOT / binding["file"]).read_bytes()).hexdigest() == binding["sha256"]
        source = read(worker["task_campaign"]["source_contract_file"])
        assert validate_task_resume(source, worker)["actor_critic_optimizer"] == "preserved"
        assert validate_task_resume(worker, worker)["task_clock"] == "restored"
    changed = deepcopy(plan)
    changed["historical_consumed_updates"] = 658
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_every_original_case_has_domain_without_changed_thresholds(plan):
    original = read(SOURCE_PLAN)
    old = read(original["design_preflight"]["file"])["stages"]["S6"]["roles"]["train"]["contract"]["evaluation"]["cases"]
    assert len(old) == 322
    covered = set()
    for domain in plan["domains"]:
        worker = read(domain["contract"])
        assert worker["evaluation"]["cases"] == old
        required = domain["required_case_names"]
        assert len(required) < 322
        assert worker["evaluation"]["promotion_case_names"] == required
        covered.update(required)
    assert covered == {case["name"] for case in old}
    terrain = next(d for d in plan["domains"] if d["name"] == "terrain")
    stepjump = next(d for d in plan["domains"] if d["name"] == "stepjump")
    assert "body_climb_double_200_250" in terrain["required_case_names"]
    assert "double_step_jump_200_250_manual" not in terrain["required_case_names"]
    assert "double_step_jump_200_250_manual" in stepjump["required_case_names"]
    assert "body_climb_double_200_250" not in stepjump["required_case_names"]


def test_all_five_independent_roles_and_silent_field_failures(plan):
    for domain in plan["domains"]:
        worker = read(domain["contract"])
        design = read(worker["design_preflight"]["file"])
        control = load_chassis_control(ROOT / worker["control_math_source"])
        for role in ROLES:
            config, seed, count = effective_role(worker, domain, role)
            validate_design_contract(config, design, control, stage=domain["name"], role=role,
                                     seed=seed, num_envs=count, block_index=0)
        for key, value in (("physics_dt", .005), ("learning_rate_schedule", "adaptive"),
                           ("checkpoint_interval", 10), ("actor_dim", 36)):
            broken = deepcopy(worker)
            broken[key] = value
            with pytest.raises(ValueError):
                validate_design_contract(broken, design, control, stage=domain["name"], role="train",
                                         seed=290927, num_envs=20000, block_index=0)
        for field in ("command_transport", "actuator_response", "dense_tracking"):
            broken = deepcopy(worker)
            broken[field]["enabled"] = False
            with pytest.raises(ValueError):
                validate_design_contract(broken, design, control, stage=domain["name"], role="train",
                                         seed=290927, num_envs=20000, block_index=0)


def test_flat_has_actual_named_height_anchors_joint_sampling_and_long_stops(plan):
    flat = read(plan["domains"][0]["contract"])
    groups = {g["name"]: g for g in flat["scene_groups"]}
    assert all(groups[n]["fraction"] == pytest.approx(.02) for n in STAND)
    for height, name in zip(HEIGHTS, STAND):
        spec = flat["skill_specs"][name]
        assert spec["command"] == [0., 0., height]
        assert "height_sampling" not in spec
        assert spec["episode_seconds"] >= 20
    for name in ("forward_05", "backward_1", "rotate_1", "rotate_1__reverse_train", "start_stop_05"):
        cross = flat["skill_specs"][name + "__height_train"]
        assert cross["height_sampling"]["range_m"] == [.23, .43]
        assert cross["height_sampling"]["boundary_weight"] == 2.
        assert groups[name + "__height_train"]["fraction"] == groups[name]["fraction"]
    assert flat["new_asset_curriculum"]["start_stop_start_delay_seconds"] == [0., 8.]
    assert flat["new_asset_curriculum"]["long_command_resampling_seconds"] == [8., 15.]
    assert flat["skill_specs"]["start_stop_05"]["episode_seconds"] == 16.


def test_response_landing_pd_and_recovery_scope_survive_all_domains(plan):
    reference = read("contracts/v6_s1_persistent_recovery_worker_v1.json")
    for domain in plan["domains"]:
        worker = read(domain["contract"])
        assert sum(g["fraction"] for g in worker["scene_groups"]) == pytest.approx(1.)
        assert (worker["physics_dt"], worker["policy_dt"], worker["actor_dim"], worker["critic_dim"],
                worker["target_num_envs"], worker["num_steps_per_env"]) == (.001, .02, 35, 81, 20000, 24)
        for key in ("actuator_response", "actuator_response_evidence", "v5_control", "policy_architecture",
                    "landing_tracking", "recovery_training"):
            assert worker[key] == reference[key]
        assert worker["learning_rate"] == 1e-4
        assert worker["learning_rate_schedule"] == "fixed"
        assert worker["task_campaign"]["evaluation_interval"] == 1000
        assert worker["task_campaign"]["clock"] == "domain_actor_updates"
        assert worker["checkpoint_interval"] == 100
        assert worker["jump_assist"]["enabled"] == (domain["name"] in ("jump", "stepjump"))
        for spec in worker["skill_specs"].values():
            if spec["kind"] == "body_climb":
                assert not spec.get("trigger_source")
    recovery = next(d for d in plan["domains"] if d["name"] == "recovery")
    worker = read(recovery["contract"])
    assert sum(s["kind"] == "recovery_handoff" for s in worker["skill_specs"].values()) == 54
    assert recovery["scene_validation"]["prior_engineering_result"] == "65/108; not qualified"
    assert worker["recovery_training"]["learning_mask"] == "pure_rl_intervals_only"
