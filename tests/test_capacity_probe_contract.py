"""The engineering capacity artifact cannot fall through into PPO training."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from wheeled_tasks.chassis.control_contract import load_chassis_control, validate_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/v6_asset_capacity_probe_v1.json"


def inputs():
    config = json.loads(CONTRACT.read_text())
    control = load_chassis_control(ROOT / config["control_math_source"])
    return config, control


def test_independent_capacity_design_matches_explicit_zero_budget_scene():
    config, control = inputs()
    receipt = design_preflight(config, control, ROOT, stage="foundation", role="train",
                                seed=918, num_envs=20000)
    assert receipt["decimation"] == 20
    assert config["engineering_probe_only"] and not config["training_release"]
    assert config["total_updates"] == sum(stage["updates"] for stage in config["stages"]) == 0
    assert config["capacity_probe"]["policy_steps"] == 100
    assert "policy_network" not in config and "policy_estimator" not in config
    assert [config[k] for k in ("actor_dim", "critic_dim", "action_dim", "history_length")] == [35, 81, 6, 1]
    assert config["precision_tracking"]["height_weight"] > 0


@pytest.mark.parametrize("key,value", [("gear_ratio", 11.), ("effort_limit", 4.5 * 15.8),
    ("curve_side", "motor"), ("torque_reference", "motor"), ("kd", .2)])
def test_control_prior_cannot_repeat_gear_conversion_or_restore_old_parameters(key, value):
    _, control = inputs()
    control["actuators"]["wheel"][key] = value
    with pytest.raises(ValueError, match="wheel prior"):
        validate_chassis_control(control)


def test_ambiguous_output_cap_plus_motor_curve_rejected():
    _, control = inputs()
    control["actuators"]["wheel"]["motor_torque_nm"] = [4.5 / 15.8, 0.]
    with pytest.raises(ValueError, match="ambiguous motor-side"):
        validate_chassis_control(control)


def test_wheel_output_torque_limit_is_4p5_without_second_ratio_multiplication():
    torch = pytest.importorskip("torch")
    from wheeled_tasks.v40.core import motor_torque_limit
    _, control = inputs()
    wheel = control["actuators"]["wheel"]
    speed = torch.tensor([[-100., -1., 0., 1., 100.]])
    assert torch.equal(motor_torque_limit(speed, wheel), torch.full_like(speed, 4.5))
    ambiguous = {**wheel, "motor_speed_rad_s": [0., 100.], "motor_torque_nm": [1., 0.]}
    with pytest.raises(ValueError, match="Ambiguous"):
        motor_torque_limit(speed, ambiguous)
    legacy = load_chassis_control(ROOT / "contracts/own_v40_v2.json")
    result = motor_torque_limit(torch.zeros(1), legacy["actuators"]["wheel"])
    assert result.item() == pytest.approx(legacy["actuators"]["wheel"]["effort_limit"])


@pytest.mark.parametrize("steps", [0, 99, 101])
def test_train_entry_refuses_ppo_or_wrong_scene_count_before_creating_run(steps, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("capacity_train_entry", ROOT / "scripts/train_chassis.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    config, _ = inputs()
    monkeypatch.setattr(entry, "preflight", lambda path: (deepcopy(config), {}))
    run = tmp_path / "must_not_exist"
    monkeypatch.setattr(sys, "argv", ["train_chassis.py", "--contract", str(CONTRACT),
        "--num-envs", "20000", "--seed", "918", "--stage", "foundation", "--research",
        "--run-dir", str(run), "--validate-scene", str(steps)])
    with pytest.raises(SystemExit) as caught:
        entry.main()
    assert caught.value.code == 2
    assert not run.exists()


def test_capacity_contract_cannot_be_mistaken_for_an_evaluation_role():
    config, control = inputs()
    with pytest.raises(ValueError, match="missing design row"):
        design_preflight(config, control, ROOT, stage="foundation", role="full", seed=918, num_envs=20000)
