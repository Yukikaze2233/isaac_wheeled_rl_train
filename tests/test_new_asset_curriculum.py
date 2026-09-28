"""Executable new-asset curriculum: budgets, ABI, coverage and retained quotas."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.task import choose_scene_groups


ROOT = Path(__file__).resolve().parents[1]


def plan():
    return json.loads((ROOT / "contracts/v6_new_asset_training_v1.json").read_text())


def materialize(value=None):
    value = value or plan()
    return [stage_contract({}, value, stage, 20000) for stage in value["stages"]]


def test_independent_contract_has_exact_frozen_budget_and_no_legacy_dependencies():
    value = plan()
    calls = []
    assert resolve_plan(value, lambda path: calls.append(path)) == value
    assert not calls
    budget_file = ROOT / value["budget_spec_file"]
    assert hashlib.sha256(budget_file.read_bytes()).hexdigest() == value["budget_spec_sha256"]
    budget = json.loads(budget_file.read_text())["formal"]
    contracts = materialize(value)
    assert [c["curriculum_stage"] for c in contracts] == [s["id"] for s in budget["stages"]]
    for config, expected in zip(contracts, budget["stages"]):
        assert config["total_updates"] == expected["update_ceiling"]
        assert config["total_updates"] * 20000 * 24 == expected["transition_ceiling"]
        assert config["num_mini_batches"] * config["ppo_minibatch_samples"] == 480000
        assert config["num_learning_epochs"] == 5
        assert config["rollback_refunds_consumed_updates"] is False
    assert sum(c["total_updates"] for c in contracts) == 8500
    # The required `base` positional argument cannot leak old gains or stages.
    assert stage_contract({"v5_control": {"leg_kp": 1.}}, value, value["stages"][0], 20000) == contracts[0]


@pytest.mark.parametrize("envs", [32, 4096, 8192, 19999, 20001])
def test_formal_budget_never_silently_rescales(envs):
    value = plan()
    with pytest.raises(ValueError, match="fixed at"):
        stage_contract({}, value, value["stages"][0], envs)


@pytest.mark.parametrize("mutation", ["updates", "environment", "network", "inheritance"])
def test_changed_budget_or_abi_is_rejected(mutation):
    value = plan()
    if mutation == "updates":
        value["stages"][0]["updates"] += 1
    elif mutation == "environment":
        value["stages"][0]["num_envs"] = 4096
    elif mutation == "network":
        value["runtime_base"]["actor_dim"] = 36
    else:
        value["extends_plan"] = "contracts/v6_scut35_budget_retry_v1.json"
    with pytest.raises(ValueError):
        materialize(value)


def test_new_asset_control_and_observation_contract_are_pinned():
    value = plan()
    for config in materialize(value):
        assert config["requires_design_preflight"] is True
        assert (config["physics_dt"], config["policy_dt"]) == (.001, .02)
        assert (config["actor_dim"], config["critic_dim"], config["action_dim"]) == (35, 81, 6)
        assert config["v5_control"]["leg_kp"] == 160.
        assert config["v5_control"]["leg_kd"] == 2.5
        assert config["hardware_deployment_ready"] is False
        assert config["physics_prior_scope"]["plant_fit_applied"] is False
        assert config["physics_prior_scope"]["hip_axes"] == "continuous_no_physical_stops"
        assert config["working_margin_weight"] == 0.
        control_path = ROOT / config["control_math_source"]
        assert hashlib.sha256(control_path.read_bytes()).hexdigest() == config["control_math_sha256"]
        assert json.loads(control_path.read_text())["actuators"]["wheel"]["kd"] == .6
        asset_path = ROOT / config["asset_directory"] / "manifest.json"
        assert hashlib.sha256(asset_path.read_bytes()).hexdigest() == config["asset_manifest_sha256"]


def test_every_required_domain_is_an_actual_promotion_case():
    contracts = materialize()
    s1, s2, s3, s4, s5, s6 = [set(c["evaluation"]["promotion_case_names"]) for c in contracts]
    assert {"stand_230mm", "stand_305mm", "stand_320mm", "stand_430mm", "height_scan"} <= s1
    for velocity in (1, 2, 3, 4, 5):
        assert {f"forward_{velocity}", f"backward_{velocity}"} <= s2
    assert {"rotate_3rps", "rotate_3rps_reverse", "spin_translate_3rps", "spin_translate_3rps_reverse",
            "forward_5_height_230mm", "backward_5_height_430mm"} <= s2
    assert {"slope_up", "slope_down", "cross_slope", "rough", "material_transfer", "landing",
            "airborne", "push_recovery", "stand_305mm__payload_heavy", "forward_05__usbhs"} <= s3
    assert {f"step_up_{v:02d}" for v in (3, 6, 10, 15, 20, 25)} <= s4
    assert {f"step_down_{v:02d}" for v in (5, 10, 15)} <= s4
    assert {"stairs", "stairs_down", "stairs_200_250", "stairs_down_200_250"} <= s4
    assert {f"jump_{v:02d}" for v in (3, 6, 10, 15, 20, 25, 30)} <= s5
    assert {f"running_jump_{v:02d}" for v in (3, 6, 10)} <= s5
    assert {"request_repeat", "request_cancel_before_takeoff", "request_cancel_airtime",
            "request_repeat_after_landing", "move_stop_jump_resume"} <= s6 - s5
    assert {name for name in s6 if name.startswith("heldout_")} <= s6 - s5
    assert len([name for name in s6 if name.startswith("heldout_")]) == 8
    assert s1 < s2 < s3 < s4 < s5 < s6
    assert all(case["success_rate_min"] == .95 for case in contracts[-1]["evaluation"]["cases"])
    assert contracts[-1]["evaluation"]["episodes_per_case"] == 20


def test_all_stages_share_the_same_fixed_cases_and_sampling_covers_every_allocated_pool():
    contracts = materialize()
    canonical = contracts[0]["evaluation"]["cases"]
    for config in contracts:
        assert config["evaluation"]["cases"] == canonical
        assert config["evaluation"]["quick_case_names"] == [case["name"] for case in canonical]
        assert sum(g["fraction"] for g in config["scene_groups"]) == pytest.approx(1.)
        assert all(g["fraction"] > 0 for g in config["scene_groups"])
        sampled = choose_scene_groups(config["scene_groups"], 20000)
        actual_groups = {name for name, _ in sampled}
        assert actual_groups == set(config["skill_specs"])
        for pool, fraction in config["behavior_pool_fractions"].items():
            planned = sum(g["fraction"] for g in config["scene_groups"]
                          if config["behavior_pool_membership"][g["name"]] == pool)
            assert planned == pytest.approx(fraction)
        assert config["terrain_difficulty_tiers"] == [1.]


def test_frozen_master_plan_quotas_and_module_matrix_are_independent_expectations():
    expected_pools = [
        {"stand": .45, "height": .25, "translation": .20, "yaw": .10},
        {"motion": .50, "transitions": .25, "stand": .15, "height": .10},
        {"terrain": .35, "recovery": .15, "motion": .25, "stand": .15, "height": .10},
        {"steps": .50, "motion": .20, "stand": .15, "height": .05, "terrain": .05, "recovery": .05},
        {"jumps": .50, "motion": .20, "stand": .15, "height": .05, "steps": .05, "terrain": .025, "recovery": .025},
        {"jumps": .20, "steps": .20, "motion": .25, "stand": .15, "height": .10, "terrain": .05, "recovery": .05},
    ]
    for index, config in enumerate(materialize()):
        assert config["behavior_pool_fractions"] == expected_pools[index]
        for module in ("dynamics_randomization", "contact_domain", "command_transport", "signal_perturbations"):
            assert config[module]["enabled"] is (index >= 2)
            assert config[module]["enabled_fraction"] == .5
        assert config["usb_transport"]["enabled"] is False
        assert config["step_assist"]["enabled"] is False
        assert config["evaluation"]["frozen_module_enablement"] == {
            "dynamics_randomization": True, "contact_domain": True,
            "command_transport": True, "signal_perturbations": False}
        assert config["evaluation"]["block_updates"] == 40
        assert config["evaluation"]["seed"] == 290929
        assert config["evaluation"]["confirmation_seed"] == 290930
        assert config["evaluation"]["survival_rate_min"] == .95
        if index:
            cross_fraction = sum(g["fraction"] for g in config["scene_groups"]
                                 if g["name"].endswith("__height_train"))
            ordinary_motion = expected_pools[index]["motion"] * (.6 if index == 5 else 1.)
            assert cross_fraction == pytest.approx(ordinary_motion * .2)
        names = {c["name"] for c in config["evaluation"]["cases"]}
        assert names == set(config["evaluation"]["retention_case_names"])
        assert set(config["evaluation"]["promotion_case_names"]) <= names
        # Names are backed by explicit measured-unit model inputs, not a suffix.
        usb_cases = [c for c in config["evaluation"]["cases"] if c["name"].endswith("__usbhs")]
        assert len(usb_cases) == 4
        assert all(c["communication_profile"] == {"delay_ms": .16} for c in usb_cases)


def test_schedule_is_stage_local_and_eval_never_uses_training_assistance():
    contracts = materialize()
    assert [c["jump_assist"]["enabled"] for c in contracts] == [False] * 4 + [True, False]
    assert contracts[4]["jump_assist"]["actor_updates"] == 300
    assert contracts[4]["new_asset_curriculum"]["jump_curriculum"]["zero_airtime_actor_updates"] == 80
    assert contracts[0]["new_asset_curriculum"]["height_sampling_schedule"]["actor_update_boundary"] == 200
    assert contracts[-1]["new_asset_curriculum"]["request_protocol"]["stress_episode_fraction"] == .5
    assert contracts[-1]["new_asset_curriculum"]["long_command_fraction"] == .2
    assert contracts[-1]["new_asset_curriculum"]["command_resampling_seconds"] == [.5, 4.]
    for config in contracts:
        for case in config["evaluation"]["cases"]:
            if case["task"] == "jump":
                assert case["jump_assistance_enabled"] is False
        assert config["evaluation"]["zero_success_fuse_updates"] is None


def test_protection_is_gate_state_and_cannot_mutate_frozen_sampling():
    value = plan()
    protected = {"stand_230mm", "forward_5", "rotate_3rps_reverse", "jump_03"}
    config = stage_contract({}, value, value["stages"][4], 20000, protected_cases=protected)
    assert config == stage_contract({}, value, value["stages"][4], 20000)
    assert config["evaluation"]["protected_case_names"] == []
    assert config["transfer_critic"] is True
    assert config["behavior_pool_fractions"]["stand"] == .15
    assert config["behavior_pool_fractions"]["motion"] == .2


def test_s6_sequences_are_real_training_specs_with_a_fixed_tenth_of_sampling():
    config = materialize()[-1]
    names = {"request_repeat", "request_cancel_before_takeoff", "request_cancel_airtime",
             "request_repeat_after_landing", "move_stop_jump_resume"}
    sequence_groups = [group for group in config["scene_groups"] if group["name"] in names]
    assert len(sequence_groups) == 5
    assert sum(group["fraction"] for group in sequence_groups) == pytest.approx(.10)
    cases = {case["name"]: case for case in config["evaluation"]["cases"]}
    for name in names:
        assert config["skill_specs"][name]["request_events"] == cases[name]["skill"]["request_events"]
        assert cases[name]["skill"]["kind"] == "jump"
        assert cases[name]["skill"]["semantics"]["jump_forward_distance_m"] == 0.
        assert cases[name]["skill"]["semantics"]["jump_apex_delta_m"] == .1
    assert cases["request_cancel_airtime"]["skill"]["request_events"][1]["after_phase"] == "flight"
    sequence = cases["move_stop_jump_resume"]["skill"]["command_sequence"]
    assert [item["command"][0] for item in sequence] == [.4, 0., 0., .4]
    assert sequence[-1]["after_phase"] == "ground_after_recovery"


def test_all_traversal_tasks_can_reach_the_route_completion_branch():
    for config in materialize():
        for case in config["evaluation"]["cases"]:
            if case["task"] == "traverse":
                assert case["skill"]["mode"] in (2, 3), case["name"]
            elif case["task"] == "jump":
                assert case["skill"]["mode"] == 4, case["name"]
        if "material_transfer" in config["skill_specs"]:
            assert config["skill_specs"]["material_transfer"]["mode"] == 2
    value = plan()
    next(item for item in value["skill_catalog"] if item["name"] == "material_transfer")["mode"] = 1
    with pytest.raises(ValueError, match="route-completion skill mode"):
        materialize(value)
