"""Independent frozen design expectations must reject effective-config drift."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis.full_curriculum import stage_contract
from wheeled_tasks.chassis.preflight import (
    NEW_ASSET_CONTRACT_PATHS, REQUIRED_CONTRACT_PATHS, design_preflight,
    validate_design_contract, validate_runtime_state,
)


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("train", "baseline", "quick", "full", "confirmation")


@lru_cache(maxsize=1)
def inputs():
    plan = json.loads((ROOT / "contracts/v6_new_asset_training_v1.json").read_text())
    design = json.loads((ROOT / "contracts/v6_new_asset_expectations_v1.json").read_text())
    control = json.loads((ROOT / "contracts/v6_research_control_v1.json").read_text())
    return plan, design, control


def effective(index, role):
    plan, _, _ = inputs()
    config = stage_contract({}, plan, plan["stages"][index], 20000)
    seed = 290927 if role == "train" else 290930 if role == "confirmation" else 290929
    count = 20000 if role == "train" else 4780 if index == 5 else 956
    if role != "train":
        config["evaluation"]["seed"] = seed
        config = fixed_suite_contract(config)
    return config, seed, count


def test_expectations_are_a_hash_pinned_artifact_of_the_reviewed_plan():
    plan, design, _ = inputs()
    assert design["status"] == "frozen"
    assert design["scope"] == "v6_new_asset_formal_research"
    assert plan["design_preflight"]["sha256"] == hashlib.sha256(
        (ROOT / plan["design_preflight"]["file"]).read_bytes()).hexdigest()
    reviewed = deepcopy(plan)
    reviewed.pop("design_preflight")
    payload = json.dumps(reviewed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(payload).hexdigest() == design["reviewed_plan_payload_sha256"]
    assert set(design["stages"]) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    assert all(set(stage["roles"]) == set(ROLES) for stage in design["stages"].values())


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("role", ROLES)
def test_all_thirty_roles_match_static_and_observed_runtime_expectations(index, role):
    config, seed, count = effective(index, role)
    _, design, control = inputs()
    name = f"S{index + 1}"
    expected = design["stages"][name]["roles"][role]
    assert set(REQUIRED_CONTRACT_PATHS + NEW_ASSET_CONTRACT_PATHS) <= set(expected["contract"])
    assert expected["invocation"]["num_envs"] == count
    assert expected["contract"]["total_updates"] == [1000, 2000, 1000, 1500, 2000, 1000][index]
    assert expected["contract"]["target_num_envs"] == 20000
    assert expected["contract"]["physics_dt"] == .001
    assert expected["contract"]["policy_dt"] == .02
    assert expected["contract"]["dense_tracking"]["enabled"] is True
    assert expected["contract"]["jump_assist"]["enabled"] is (index == 4 and role == "train")
    assert expected["contract"]["evaluation_exact_cases"] is (role != "train")
    result = design_preflight(config, control, ROOT, stage=config["enabled_stages"][0],
                              role=role, seed=seed, num_envs=count)
    observed = {
        "physics_dt": .001, "policy_dt": .02, "pc_control_dt": .001, "decimation": 20,
        "seed": seed, "leg_kp": 160., "leg_kd": 2.5, "wheel_velocity_p": .6,
        "precision_tracking": False, "dense_tracking": True, "command_reference": True,
        "modules": {"usb_transport": False, "step_assist": False,
                    "command_transport": role != "train" or index >= 2,
                    "contact_domain": role != "train" or index >= 2,
                    "dynamics_randomization": role != "train" or index >= 2,
                    "signal_perturbations": role == "train" and index >= 2,
                    "jump_assist": role == "train" and index == 4,
                    "training_schedule": True},
    }
    assert validate_runtime_state(result["runtime_expected"], observed)["status"] == "runtime_configuration_passed"


@pytest.mark.parametrize("path", NEW_ASSET_CONTRACT_PATHS)
def test_new_required_fields_cannot_be_omitted_even_when_runtime_defaults_exist(path):
    config, seed, count = effective(0, "train")
    _, design, control = inputs()
    parent = config
    parts = path.split(".")
    for key in parts[:-1]:
        parent = parent[key]
    del parent[parts[-1]]
    with pytest.raises(ValueError, match="Preflight rejected"):
        validate_design_contract(config, design, control, stage="S1", role="train", seed=seed, num_envs=count)


@pytest.mark.parametrize("mutation", ["height_mixture", "reward", "sampling", "skill", "wrong_seed", "wrong_count", "injected_clock"])
def test_final_effective_overrides_are_rejected(mutation):
    config, seed, count = effective(0, "train")
    _, design, control = inputs()
    if mutation == "height_mixture":
        config["new_asset_curriculum"]["height_sampling_schedule"]["initial_fixed_fraction"] = .5
    elif mutation == "reward":
        config["dense_tracking"]["height"]["broad_weight"] = .01
    elif mutation == "sampling":
        config["scene_groups"][0]["fraction"] = .01
    elif mutation == "skill":
        config["skill_specs"]["stand_continuous"]["command"][2] = .40
    elif mutation == "wrong_seed":
        seed += 1
    elif mutation == "wrong_count":
        count = 4096
    else:
        config["new_asset_curriculum"]["global_actor_update"] = 400
    with pytest.raises(ValueError, match="Preflight rejected"):
        validate_design_contract(config, design, control, stage="S1", role="train", seed=seed, num_envs=count)


def test_training_block_seed_is_root_plus_index_and_confirmation_is_separate():
    config, _, count = effective(0, "train")
    _, design, control = inputs()
    assert validate_design_contract(config, design, control, stage="S1", role="train",
        seed=290927 + 7, block_index=7, num_envs=count)["seed"] == 290934
    config, _, count = effective(0, "confirmation")
    with pytest.raises(ValueError, match="invocation.seed"):
        validate_design_contract(config, design, control, stage="S1", role="confirmation", seed=290929, num_envs=count)


def test_runtime_cannot_claim_an_uninstantiated_reward_or_curriculum_module():
    _, design, _ = inputs()
    expected = design["stages"]["S5"]["roles"]["train"]["runtime"]
    observed = deepcopy(expected)
    observed["modules"]["training_schedule"] = False
    with pytest.raises(ValueError, match="modules.training_schedule"):
        validate_runtime_state(expected, observed)
    observed = deepcopy(expected)
    observed["dense_tracking"] = False
    with pytest.raises(ValueError, match="dense_tracking"):
        validate_runtime_state(expected, observed)


def load_full_entry():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("new_asset_full_entry", ROOT / "scripts/run_full_chassis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_actor_offsets_use_explicit_consumption_not_learning_lineage(tmp_path):
    module = load_full_entry()
    args = SimpleNamespace(contract=ROOT / "contracts/v6_new_asset_training_v1.json",
        run_dir=tmp_path / "run", max_runtime_seconds=100., transfer=None,
        global_actor_update_offset=100)
    runner = module.FullCurriculum(args)
    # 112 current learning updates after rollback does not refund 312 consumed.
    first = runner.account_stage_actor_updates({"actor_updates_consumed": 312,
                                               "successful_updates": 322, "learning_updates": 112})
    assert first == {"global_actor_update_offset": 100, "actor_updates_consumed": 312,
                     "global_budget_actor_updates": 412}
    second = runner.account_stage_actor_updates({"actor_updates_consumed": 80, "successful_updates": 85})
    assert second["global_actor_update_offset"] == 412
    assert second["global_budget_actor_updates"] == 492
    with pytest.raises(ValueError, match="explicit actor-update consumption"):
        runner.account_stage_actor_updates({"successful_updates": 1000})
