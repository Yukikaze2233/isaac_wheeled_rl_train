"""An enabled metadata row cannot conceal a dead native posture lesson."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import validate_design_contract, validate_runtime_state

ROOT = Path(__file__).resolve().parents[1]


def bound_role(domain, role="train"):
    worker = json.loads((ROOT / f"contracts/v6_task_campaign_{domain}_worker_v18.json").read_text())
    design = json.loads((ROOT / worker["design_preflight"]["file"]).read_text())
    row = design["stages"][domain]["roles"][role]
    config = deepcopy(row["contract"])
    control = load_chassis_control(ROOT / config["control_math_source"])
    invocation = row["invocation"]
    rule = invocation["seed"]
    seed = rule.get("root_seed", rule.get("value"))
    return config, design, control, seed, invocation["num_envs"]


@pytest.mark.parametrize("domain", ("recovery", "speed", "terrain", "jump", "stepjump"))
def test_independent_train_rows_require_full_settings_and_enabled_runtime(domain):
    config, design, control, seed, count = bound_role(domain)
    receipt = validate_design_contract(config, design, control, stage=domain, role="train",
        seed=seed, num_envs=count)
    assert receipt["runtime_expected"]["modules"]["ordinary_posture_curriculum"] is True
    config.pop("ordinary_posture_curriculum")
    with pytest.raises(ValueError, match="ordinary_posture"):
        validate_design_contract(config, design, control, stage=domain, role="train", seed=seed, num_envs=count)


def observed_runtime():
    config, design, control, seed, count = bound_role("speed")
    expected = validate_design_contract(config, design, control, stage="speed", role="train",
        seed=seed, num_envs=count)["runtime_expected"]
    actual = deepcopy(expected)
    actual["ordinary_posture_curriculum"]["runtime_observed"] = dict(
        eligible_count=4096, birth_eligible_population=4096, relative_actor_updates=0,
        birth_selected_count=2000, birth_rejected_count=0, push_events=0,
        birth_angle_cap_rad=.03, push_speed_cap_m_s=.05, birth_abs_angle_max_rad=.029,
        push_clock_min_s=5.001, push_clock_max_s=9.999, push_clock_std_s=1.4,
        push_velocity_delta_max_m_s=0., geometry_bound=True)
    return expected, actual


def test_startup_population_observed_without_inventing_future_push_events():
    expected, actual = observed_runtime()
    assert validate_runtime_state(expected, actual)["status"] == "runtime_configuration_passed"
    assert "runtime_observed" in actual["ordinary_posture_curriculum"]


@pytest.mark.parametrize("fault", ("dead_hook", "dead_birth", "zero_cap", "missing_population", "geometry", "nan", "hidden_settings"))
def test_native_startup_rejects_disabled_or_unbound_lesson(fault):
    expected, actual = observed_runtime()
    observed = actual["ordinary_posture_curriculum"]["runtime_observed"]
    if fault == "dead_hook":
        actual["modules"]["ordinary_posture_curriculum"] = False
    elif fault == "dead_birth":
        observed["birth_selected_count"] = 0
        observed["birth_abs_angle_max_rad"] = 0.
    elif fault == "zero_cap":
        observed["push_speed_cap_m_s"] = 0.
    elif fault == "missing_population":
        actual["ordinary_posture_curriculum"].pop("runtime_observed")
    elif fault == "geometry":
        observed["geometry_bound"] = False
    elif fault == "nan":
        observed["push_velocity_delta_max_m_s"] = float("nan")
    else:
        actual["ordinary_posture_curriculum"]["settings"]["push"]["end_speed_m_s"] = .5
    with pytest.raises(ValueError, match="ordinary_posture"):
        validate_runtime_state(expected, actual)
