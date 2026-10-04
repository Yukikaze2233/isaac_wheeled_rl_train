"""Exercise physical landing accounting against command and full-task semantics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.jump_landing import JUMP_LANDING_V1, JumpLandingWindow
from wheeled_tasks.chassis.references import CommandReference
from wheeled_tasks.chassis.rewards import reference_jump_terms
from wheeled_tasks.chassis.task import Phase


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "contracts/v6_task_campaign_jump_worker_v6.json").read_text())["command_reference"]


def rig(count=1, *, vx=0.):
    window = JumpLandingWindow(count, "cpu", .02, deepcopy(JUMP_LANDING_V1))
    state = dict(request=torch.ones(count, dtype=torch.bool),
        phase=torch.full((count,), Phase.GROUND),
        contacts=torch.ones(count, 2, dtype=torch.bool),
        com_released=torch.zeros(count, dtype=torch.bool),
        clear_air_time_peak=torch.zeros(count), raw_height=torch.full((count,), .305),
        height=torch.full((count,), .305),
        commands=torch.tensor([[vx, 0., .305]]).repeat(count, 1),
        goal_apex=torch.full((count,), .06), com_rise=torch.full((count,), .01),
        com_vz=torch.zeros(count), accepted_event=torch.ones(count, dtype=torch.bool))
    return window, state


def observed_flight(window, state):
    window.update(**state)
    state.update(phase=torch.full_like(state["phase"], Phase.FLIGHT),
        contacts=torch.zeros_like(state["contacts"]),
        com_released=torch.ones_like(state["com_released"]),
        clear_air_time_peak=torch.full_like(state["clear_air_time_peak"], .06),
        accepted_event=torch.zeros_like(state["accepted_event"]),
        com_vz=torch.full_like(state["com_vz"], -.2))
    window.update(**state)


def observed_touchdown(window, state):
    observed_flight(window, state)
    state.update(phase=torch.full_like(state["phase"], Phase.LANDING),
                 contacts=torch.ones_like(state["contacts"]))
    window.update(**state)


def costs(window, **changes):
    count = len(window.elapsed)
    args = dict(height=torch.full((count,), .325), com_vz=torch.zeros(count),
        velocity=torch.zeros(count, 3), omega=torch.zeros(count, 3),
        gravity=torch.tensor([[0., 0., -1.]]).repeat(count, 1),
        wheel_velocity=torch.zeros(count, 2), wheel_torque=torch.zeros(count, 2),
        contact_peak=torch.zeros(count, 2), weight_force=torch.full((count,), 135.),
        position_xy=torch.zeros(count, 2), wheel_clearance=torch.full((count, 2), .05))
    args.update(changes)
    return window.reward_terms(**args)


def test_early_physical_touchdown_survives_nominal_phase_and_trigger_release():
    window, state = rig()
    observed_touchdown(window, state)
    ref = CommandReference(1, "cpu", REFERENCE)
    ref.update(torch.tensor([.305]), torch.zeros(1), torch.zeros(1), torch.zeros(1),
        torch.full((1,), 7.), torch.ones(1, dtype=torch.bool), torch.tensor([.20]), torch.tensor([.06]))
    assert ref.phase.item() < ref.LAND
    task = FullTaskSemantics(1, "cpu", .02, {})
    task.com_released[:] = True
    task.com_rise[:] = .01
    task.clear_air_time_peak[:] = .06
    old = reference_jump_terms(ref, task, state["phase"], state["contacts"],
        torch.tensor([.325]), torch.zeros(1), torch.zeros(1, 3),
        torch.tensor([[0., 0., -1.]]), torch.tensor([.22]), torch.zeros(1, 3), state["commands"])
    assert old["jump_landing_height"].item() == 0.
    state.update(request=torch.zeros(1, dtype=torch.bool), phase=torch.tensor([Phase.RECOVERY]))
    window.update(**state)
    assert window.active.item() and window.elapsed.item() == pytest.approx(.02)
    assert costs(window)["jump_landing_height_broad"].item() < 0.


def test_release_after_verified_flight_preserves_pending_actual_touchdown():
    window, state = rig()
    observed_flight(window, state)
    state["request"][:] = False
    window.update(**state)
    assert window.armed.item() and window.verified_flight.item()
    state.update(phase=torch.tensor([Phase.LANDING]), contacts=torch.ones(1, 2, dtype=torch.bool))
    window.update(**state)
    assert window.active.item() and window.touchdown.item()


def test_cancel_before_verified_flight_does_not_pay_an_unrequested_later_fall():
    window, state = rig()
    window.update(**state)
    state.update(request=torch.zeros(1, dtype=torch.bool), accepted_event=torch.zeros(1, dtype=torch.bool))
    window.update(**state)
    state.update(phase=torch.tensor([Phase.FLIGHT]), contacts=torch.zeros(1, 2, dtype=torch.bool),
        com_released=torch.ones(1, dtype=torch.bool), clear_air_time_peak=torch.tensor([.10]))
    window.update(**state)
    state.update(phase=torch.tensor([Phase.LANDING]), contacts=torch.ones(1, 2, dtype=torch.bool))
    window.update(**state)
    assert not window.active.item() and all(value.eq(0).all() for value in costs(window).values())


def test_ordinary_standing_has_no_jump_cost_even_with_large_observation_errors():
    window, state = rig(2)
    state["request"][:] = False
    state["accepted_event"][:] = False
    for _ in range(180):
        window.update(**state)
    values = costs(window, height=torch.tensor([.10, .65]),
                   velocity=torch.full((2, 3), 5.), contact_peak=torch.full((2, 2), 1000.))
    assert all(value.eq(0).all() for value in values.values())


def test_bounce_and_busy_repeat_cannot_renew_the_finite_touchdown_clock():
    window, state = rig()
    observed_touchdown(window, state)
    for tick in range(1, 151):
        state["phase"][:] = Phase.FLIGHT if tick % 3 == 0 else Phase.RECOVERY
        state["contacts"][:] = tick % 3 != 0
        state["accepted_event"][:] = tick % 2 == 0
        window.update(**state)
        if tick < 150:
            assert window.active.item() and window.elapsed.item() == pytest.approx(tick * .02)
            assert not window.touchdown.item()
    assert not window.active.item() and window.weight.item() == 0.
    assert all(value.eq(0).all() for value in costs(window).values())


def test_cost_expiration_keeps_the_independent_full_task_deadline():
    window, state = rig()
    observed_touchdown(window, state)
    state["request"][:] = False
    while window.request_elapsed.item() < 3.2:
        window.update(**state)
    assert not window.active.item() and window.outcome_active.item()
    assert not window.outcome_timeout.item()
    ticks_to_deadline = round((6. - window.request_elapsed.item()) / .02)
    for _ in range(ticks_to_deadline):
        window.update(**state)
    assert window.outcome_timeout.item() and not window.outcome_active.item()
    assert all(value.eq(0).all() for value in costs(window).values())


def test_raised_touchdown_uses_support_relative_height_and_preserves_latched_rise():
    window, state = rig()
    state["landing_delta"] = torch.tensor([.20])
    state["height"][:] = .330
    observed_touchdown(window, state)
    assert window.goal_height.item() == pytest.approx(.305)
    assert window.landing_delta.item() == pytest.approx(.20)
    assert window.reference()[0].item() == pytest.approx(.330)
    assert not window.entry_height_clipped.item()


def test_conditional_landing_accounting_cannot_relax_full_jump_completion():
    window, state = rig()
    observed_touchdown(window, state)
    task = FullTaskSemantics(1, "cpu", .02,
        {"jump_apex_frame": "com_release", "jump_apex_delta_m": .06})
    task.com_rise[:] = .01
    task.com_release_speed[:] = .5
    task.release_velocity[:] = .5
    task.clear_air_time_peak[:] = .06
    task.height_peak[:] = .32
    complete = task.completion(torch.tensor([4]), torch.zeros(1, 3), torch.zeros(1, 2),
        torch.zeros(1), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, dtype=torch.bool),
        torch.tensor([Phase.RECOVERY]), torch.tensor([.305]), torch.zeros(1))
    assert window.active.item() and not complete.item()
    assert costs(window)["jump_landing_height_broad"].item() < 0.


@pytest.mark.parametrize("vx", [-.6, .6])
def test_precontact_speed_uses_actual_opposite_wheel_axes_and_commanded_direction(vx):
    window, state = rig(2, vx=vx)
    observed_flight(window, state)
    raw = torch.tensor([[vx / .06, -vx / .06], [-vx / .06, vx / .06]])
    values = costs(window, wheel_velocity=raw, com_vz=torch.full((2,), -.3))
    assert values["jump_precontact_wheel_velocity_broad"][0].item() == pytest.approx(0.)
    assert values["jump_precontact_wheel_velocity_broad"][1].item() < 0.
    assert all(value.le(0).all() for value in values.values())


def test_impact_uses_peak_of_each_wheel_in_newtons_with_actual_bodyweight():
    window, state = rig(2)
    observed_touchdown(window, state)
    window.update(**state)
    values = costs(window, contact_peak=torch.tensor([[300., 0.], [150., 150.]]))
    assert values["jump_impact_broad"][0].item() < 0.
    assert values["jump_impact_broad"][1].item() == 0.


def test_selected_terminal_reset_does_not_clear_the_other_landing_lane():
    window, state = rig(2)
    observed_touchdown(window, state)
    window.update(**state)
    previous = window.elapsed[1].clone()
    window.reset(torch.tensor([0]))
    assert window.active.tolist() == [False, True]
    assert window.elapsed[0] == 0. and window.elapsed[1] == previous
    values = costs(window)
    assert all(value[0].item() == 0. for value in values.values())
    assert values["jump_landing_height_broad"][1].item() < 0.
