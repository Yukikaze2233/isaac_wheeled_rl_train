"""Full curriculum contracts, actuator envelope, jump semantics and delay resets."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.full_curriculum import compatible_control_transfer, stage_contract
from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.robustness import V5SignalPerturbations
from wheeled_tasks.chassis.task import Phase, choose_scene_groups
from wheeled_tasks.chassis.v5_control import V5Control

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    base = json.loads((ROOT / "contracts/v5_locomotion_v2.json").read_text())
    plan = json.loads((ROOT / "contracts/v5_full_curriculum_v2.json").read_text())
    return base, plan


def test_all_stages_are_executable_contracts_and_preserve_regression_cases():
    base, plan = inputs()
    assert {s["kind"] for s in plan["stages"]} == {"foundation", "speed", "terrain", "jump", "mixed"}
    for recipe in plan["stages"]:
        contract = stage_contract(base, plan, recipe, 512)
        assert sum(s["updates"] for s in contract["stages"]) == contract["total_updates"]
        assert contract["total_updates"] == recipe["updates"] // 2
        assert len(choose_scene_groups(contract["scene_groups"], 64)) == 64
        cases = contract["evaluation"]["cases"]
        assert len({c["name"] for c in cases}) == len(cases)
        assert all(c["anchor"] for c in cases[:7])
        assert contract["v5_control"]["wheel_action_clip"] * contract["v5_control"]["wheel_velocity_scale"] * .06 >= recipe["vx_max"]
    assert base["v5_control"].get("wheel_action_clip") is None


def test_wheel_bounds_expand_without_changing_leg_action_or_torque_limits():
    base, plan = inputs()
    config = stage_contract(base, plan, plan["stages"][-1], 512)
    directory = ROOT / base["asset_directory"]
    read = lambda name: json.loads((directory / name).read_text())
    prior = json.loads((ROOT / base["control_math_source"]).read_text())
    controller = V5Control(read("manifest.json"), read("model_spec.json"), read("fit_10mpa.json"), prior, config["v5_control"], "cpu")
    q = controller.nominal[None]
    legs, wheels, clipped = controller.decode(torch.full_like(q, 100.), q)
    assert clipped[0, [0, 1, 3, 4]].tolist() == [3.] * 4
    assert wheels.tolist() == [[75., 75.]]
    tau = controller.motor_efforts(q, torch.zeros_like(q), legs, wheels)
    assert tau[0, [2, 5]].abs().max() < 3.838
    assert compatible_control_transfer(base["v5_control"], config["v5_control"])
    changed = deepcopy(config["v5_control"])
    changed["wheel_velocity_scale"] = 20.
    assert not compatible_control_transfer(base["v5_control"], changed)
    assert not compatible_control_transfer(config["v5_control"], base["v5_control"])


def test_rolling_step_success_does_not_require_airborne_state():
    task = FullTaskSemantics(2, "cpu", .01, {})
    mode = torch.tensor([2, 3])
    position = torch.tensor([[.7, 0., .305], [.7, 0., .305]])
    result = task.completion(mode, position, torch.full((2, 2), .7), torch.full((2,), .65),
        torch.ones(2, 2, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
        torch.full((2,), Phase.GROUND), torch.full((2,), .305))
    assert result.all()


def test_in_place_jump_requires_real_clearance_upward_release_and_recovery():
    task = FullTaskSemantics(1, "cpu", .01, {"jump_apex_delta_m": .06})
    mode, height = torch.tensor([4]), torch.tensor([.37])
    for _ in range(8):
        task.observe(mode, height, torch.tensor([[.02, .02]]), torch.tensor([Phase.FLIGHT]), torch.tensor([.4]))
    arguments = (mode, torch.tensor([[0., 0., .305]]), torch.zeros(1, 2), torch.tensor([.65]),
                 torch.ones(1, 2, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.tensor([Phase.RECOVERY]), torch.tensor([.305]))
    assert task.completion(*arguments).item()
    task.release_velocity[:] = -.1
    assert not task.completion(*arguments).item()
    task.release_velocity[:] = .4
    task.clear_air_time_peak[:] = .02
    assert not task.completion(*arguments).item()


def test_jump_reference_is_continuous_at_preload_push_boundary():
    task = FullTaskSemantics(3, "cpu", .01, {})
    times = torch.tensor([.25 - 1e-6, .25, .25 + 1e-6])
    h, v = task.jump_reference(torch.full((3,), .305), torch.full((3,), Phase.TAKEOFF), times,
                               torch.zeros(3), torch.zeros(3))
    assert float(h.max() - h.min()) < 1e-6
    assert float(v.abs().max()) < 1e-4
    assert h[1] == pytest.approx(.285, abs=1e-6)


def test_delay_zero_lag_partial_reset_and_repeat_queries():
    generator = torch.Generator().manual_seed(17)
    perturb = V5SignalPerturbations(2, "cpu", {"max_delay_steps": 2, "noise_scale": 0.}, generator)
    perturb.reset(torch.arange(2))
    perturb.act_lag[:] = torch.tensor([0, 2])
    first = torch.ones(2, 6)
    torch.testing.assert_close(perturb.action(first, 0), first)
    second = first * 2
    result = perturb.action(second, 1)
    torch.testing.assert_close(result[0], second[0])
    torch.testing.assert_close(result[1], first[1])
    perturb.reset(torch.tensor([1]))
    torch.testing.assert_close(perturb.action(second * 2, 1)[1], second[1] * 2)
    frame = torch.randn(2, 46)
    torch.testing.assert_close(perturb.observation(frame, 0), perturb.observation(frame, 0))
