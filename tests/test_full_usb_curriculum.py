"""Complete budgets, corrected workspace and opt-in substep communication priors."""
import hashlib
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.usb_transport import SubstepUsbTransport


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v5_full_usb_v59.json"), load)
    return plan, load(plan["base_contract"])


def test_complete_plan_advances_after_initial_adaptation_and_preserves_time_units():
    plan, base = configuration()
    assert len(plan["stages"]) == 8
    assert sum(s["updates"] for s in plan["stages"]) == 44500
    updates, samples = 0, 0
    for recipe in plan["stages"]:
        config = stage_contract(base, plan, recipe, 16384)
        assert config["policy_dt"] == .02 and config["physics_dt"] == .005
        assert config["evaluation"]["mode"] == "monitor"
        assert config["signal_perturbations"]["max_delay_steps"] == 0
        assert config["performance_curriculum"].get("height_course") is None
        assert config["height_workspace"]["height_m"][::len(config["height_workspace"]["height_m"]) - 1] == [.23, .43]
        assert config["height_workspace"]["model_spec_sha256"] == hashlib.sha256(
            (ROOT / "model/纯底盘_v5/urdf/model_spec.json").read_bytes()).hexdigest()
        assert config["target_num_envs"] * config["num_steps_per_env"] // config["num_mini_batches"] == 49152
        SubstepUsbTransport(4, "cpu", config["physics_dt"], config["usb_transport"], 617)
        cases = {c["name"]: c for c in config["evaluation"]["cases"]}
        for name, case in cases.items():
            if not name.endswith("_usb"):
                assert not case["transport_enabled"]
                assert cases[name + "_usb"]["transport_enabled"]
                assert cases[name + "_usb"]["reset_seed_key"] == case["reset_seed_key"]
        updates += config["total_updates"]
        samples += config["total_updates"] * config["target_num_envs"] * config["num_steps_per_env"]
    assert updates == 14625
    assert samples == 44500 * plan["target_num_envs"] * plan["num_steps_per_env"]


def test_height_extension_keeps_nominal_rehearsal_groups():
    plan, base = configuration()
    first = stage_contract(base, plan, plan["stages"][0], 16384)
    height = stage_contract(base, plan, plan["stages"][2], 16384)
    assert first["critic_warmup_updates"] == 25
    assert "height_sampling" not in height["skill_specs"]["forward_3"]
    assert height["skill_specs"]["forward_3__height_train"]["height_sampling"]["range_m"] == [.23, .43]
    assert height["skill_specs"]["height_full"]["height_range_m"] == [.23, .43]
