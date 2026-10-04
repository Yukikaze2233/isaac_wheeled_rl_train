"""Physical contact windows retain a released request without renewable credit."""
from copy import deepcopy

import pytest
import torch

from wheeled_tasks.chassis.jump_landing import (
    JUMP_LANDING_V1, JumpLandingWindow, validate_jump_landing,
)
from wheeled_tasks.chassis.task import Phase


def rig(count=1, dt=.02, **changes):
    config = deepcopy(JUMP_LANDING_V1) | changes
    window = JumpLandingWindow(count, "cpu", dt, config)
    commands = torch.zeros(count, 3)
    commands[:, 2] = .305
    data = dict(request=torch.ones(count, dtype=torch.bool),
        phase=torch.full((count,), Phase.TAKEOFF), contacts=torch.ones(count, 2, dtype=torch.bool),
        com_released=torch.zeros(count, dtype=torch.bool), clear_air_time_peak=torch.zeros(count),
        raw_height=commands[:, 2].clone(), commands=commands,
        goal_apex=torch.full((count,), .10), height=commands[:, 2].clone(), com_vz=torch.zeros(count))
    return window, data


def flight(window, data, clear_time=.08):
    window.update(**data)
    data["phase"].fill_(Phase.FLIGHT)
    data["contacts"].zero_()
    data["com_released"].fill_(True)
    data["clear_air_time_peak"].fill_(clear_time)
    data["com_vz"].fill_(-.5)
    window.update(**data)


def land(window, data, *, physical_phase=Phase.LANDING):
    flight(window, data)
    data["phase"].fill_(physical_phase)
    data["contacts"].fill_(True)
    window.update(**data)


def physical(window, *, height=None, com_vz=None):
    n = len(window.elapsed)
    h_ref, v_ref = window.reference()
    gravity = torch.zeros(n, 3)
    gravity[:, 2] = -1.
    return dict(height=h_ref if height is None else height,
        com_vz=v_ref if com_vz is None else com_vz, velocity=torch.zeros(n, 3),
        omega=torch.zeros(n, 3), gravity=gravity, wheel_velocity=torch.zeros(n, 2),
        wheel_torque=torch.zeros(n, 2), contact_peak=torch.zeros(n, 2),
        weight_force=torch.full((n,), 125.), wheel_clearance=torch.zeros(n, 2))


def axis(terms, name):
    return terms[f"{name}_broad"] + terms[f"{name}_precision"]


def test_ordinary_stationary_recovery_cannot_receive_any_jump_payment():
    window, data = rig(3)
    data["request"].zero_()
    data["phase"].fill_(Phase.RECOVERY)
    for _ in range(200):
        window.update(**data)
    assert not window.active.any() and not window.outcome_active.any()
    terms = window.reward_terms(**physical(window, height=torch.full((3,), .22),
                                          com_vz=torch.ones(3)))
    assert all(value.eq(0.).all() for value in terms.values())


@pytest.mark.parametrize("evidence", ["no_release", "short_air", "no_flight", "no_support_loss"])
def test_contact_requires_all_independent_flight_evidence(evidence):
    window, data = rig()
    window.update(**data)
    data["phase"].fill_(Phase.FLIGHT)
    data["contacts"].zero_()
    data["com_released"].fill_(True)
    data["clear_air_time_peak"].fill_(.08)
    if evidence == "no_release":
        data["com_released"].zero_()
    elif evidence == "short_air":
        data["clear_air_time_peak"].fill_(.059)
    elif evidence == "no_flight":
        data["phase"].fill_(Phase.TAKEOFF)
    else:
        data["contacts"].fill_(True)
    window.update(**data)
    data["phase"].fill_(Phase.LANDING)
    data["contacts"].fill_(True)
    window.update(**data)
    assert not window.active.item() and not window.verified_flight.item()


def test_first_physical_touchdown_starts_before_nominal_land_or_phase_debounce():
    window, data = rig()
    # A contact observer can still label this sample FLIGHT. No command-clock
    # LAND predicate is required by the landing observer.
    land(window, data, physical_phase=Phase.FLIGHT)
    assert window.active.item() and window.touchdown.item() and window.absorption.item()
    assert window.elapsed.item() == 0.
    assert window.weight.item() == 1.


def test_recovery_release_preserves_latched_goals_and_cost_window():
    window, data = rig()
    data["commands"][:, 0] = .4
    data["commands"][:, 1] = -.2
    data["landing_delta"] = torch.full((1,), .20)
    land(window, data)
    goal_commands = window.goal_commands.clone()
    goal_height = window.goal_height.clone()
    data["request"].zero_()
    data["phase"].fill_(Phase.RECOVERY)
    data["raw_height"].fill_(.43)
    data["commands"][:] = torch.tensor([[-1., 1., .43]])
    data["goal_apex"].fill_(.3)
    data["landing_delta"].zero_()
    for _ in range(3):
        window.update(**data)
    assert window.active.item() and window.outcome_active.item()
    assert window.elapsed.item() == pytest.approx(.06)
    torch.testing.assert_close(window.goal_commands, goal_commands)
    torch.testing.assert_close(window.goal_height, goal_height)
    assert window.goal_apex.item() == pytest.approx(.10)
    assert window.landing_delta.item() == pytest.approx(.20)
    terms = window.reward_terms(**physical(window, height=window.goal_height + .05))
    assert axis(terms, "jump_landing_height").item() < 0.


def test_bounces_and_busy_request_repeats_do_not_renew_clock():
    window, data = rig()
    land(window, data)
    for tick in range(1, 80):
        data["request"].fill_(tick % 2 == 0)
        data["phase"].fill_(Phase.FLIGHT if tick % 3 == 0 else Phase.RECOVERY)
        data["contacts"].fill_(tick % 3 != 0)
        # A busy protocol pulse also cannot overwrite the ongoing target.
        data["accepted_event"] = torch.ones(1, dtype=torch.bool)
        window.update(**data)
        assert not window.touchdown.item()
        assert not window.cycle_accepted.item()
        assert window.elapsed.item() == pytest.approx(tick * .02)
    assert window.active.item()


def test_cancel_before_verified_flight_clears_but_after_flight_keeps_outcome():
    window, data = rig(2)
    window.update(**data)
    data["phase"].fill_(Phase.FLIGHT)
    data["contacts"].zero_()
    data["com_released"][1] = True
    data["clear_air_time_peak"][1] = .08
    window.update(**data)
    data["request"].zero_()
    window.update(**data)
    assert window.armed.tolist() == [False, True]
    assert window.outcome_active.tolist() == [False, True]
    data["phase"].fill_(Phase.LANDING)
    data["contacts"].fill_(True)
    window.update(**data)
    assert window.active.tolist() == [False, True]


def test_cost_expiration_is_finite_while_independent_task_deadline_survives():
    window, data = rig()
    land(window, data)
    data["request"].zero_()
    data["phase"].fill_(Phase.RECOVERY)
    for _ in range(149):
        window.update(**data)
    assert window.active.item()
    window.update(**data)
    assert window.expired.item() and not window.active.item()
    assert window.elapsed.item() == 0. and window.weight.item() == 0.
    assert window.outcome_active.item() and not window.outcome_timeout.item()
    assert window.request_elapsed.item() == pytest.approx(3.04)
    for _ in range(147):
        window.update(**data)
    assert window.outcome_active.item()
    window.update(**data)
    assert window.outcome_timeout.item() and not window.outcome_active.item()
    # The old held request cannot rearm or restart either deadline.
    for _ in range(400):
        window.update(**data)
    assert not window.active.item() and not window.outcome_active.item()


def test_no_touchdown_still_has_a_finite_task_outcome_deadline():
    window, data = rig()
    flight(window, data)
    data["request"].zero_()
    for _ in range(149):
        window.update(**data)
    assert not window.armed.item() and window.outcome_active.item()
    for _ in range(150):
        window.update(**data)
    assert window.outcome_timeout.item() and not window.outcome_active.item()


def test_reset_and_terminal_only_clear_selected_lanes():
    window, data = rig(3)
    land(window, data)
    window.reset(torch.tensor([0]))
    data["terminal"] = torch.tensor([False, True, False])
    data["request"].zero_()
    window.update(**data)
    assert window.active.tolist() == [False, False, True]
    assert window.goal_height[:2].eq(0.).all()
    before = window.elapsed[2].clone()
    window.close(torch.tensor([False, False, True]))
    assert not window.active.any() and before > 0.


def test_genuine_later_request_allowed_only_after_closed_outcome_on_ground():
    window, data = rig()
    land(window, data)
    window.close(torch.ones(1, dtype=torch.bool))
    data["request"].zero_()
    data["phase"].fill_(Phase.RECOVERY)
    window.update(**data)
    data["request"].fill_(True)
    window.update(**data)
    assert not window.armed.item()
    data["request"].zero_()
    data["phase"].fill_(Phase.GROUND)
    window.update(**data)
    data["request"].fill_(True)
    window.update(**data)
    assert window.armed.item() and window.cycle_accepted.item()


def test_bounded_measured_entry_reference_reaches_quiet_goal_at_absorption_end():
    window, data = rig(3)
    data["height"] = torch.tensor([.10, .305, .60])
    land(window, data)
    assert window.entry_height_clipped.tolist() == [True, False, True]
    assert window.entry_speed_clipped.all()
    lower = window.goal_height - window.settings["absorption_stroke_m"]
    upper = window.goal_height + window.settings["absorption_stroke_m"]
    for step in range(16):
        h_ref, v_ref = window.reference()
        assert (h_ref >= lower - 1e-6).all() and (h_ref <= upper + 1e-6).all()
        if step < 15:
            window.update(**data)
    assert not window.absorption.any()
    torch.testing.assert_close(h_ref, window.goal_height)
    torch.testing.assert_close(v_ref, torch.zeros_like(v_ref))


def test_settling_costs_zero_at_goal_and_improve_monotonically_with_lower_error():
    window, data = rig(4)
    land(window, data)
    for _ in range(15):
        window.update(**data)
    physical_data = physical(window)
    errors = torch.tensor([0., .005, .02, .12])
    physical_data["height"] += errors
    physical_data["velocity"][:, 0] = errors
    physical_data["omega"][:, 2] = errors
    physical_data["gravity"][:, 0] = errors
    terms = window.reward_terms(**physical_data)
    assert all((value <= 0.).all() for value in terms.values())
    for name in ("height", "velocity", "yaw", "orientation"):
        score = axis(terms, f"jump_landing_{name}")
        assert score[0] == 0. and (score.diff() < 0.).all()


def test_precontact_wheel_alignment_covers_low_speed_and_both_motor_directions():
    window, data = rig(3)
    data["commands"][:, 0] = torch.tensor([.4, -.4, 0.])
    flight(window, data)
    sample = physical(window, com_vz=torch.full((3,), -.5))
    sample["wheel_velocity"] = (data["commands"][:, 0, None] / .06) * torch.tensor([[1., -1.]])
    terms = window.reward_terms(**sample)
    assert axis(terms, "jump_precontact_wheel_velocity").eq(0.).all()
    sample["wheel_velocity"] += torch.tensor([[1., -1.]])
    sample["wheel_torque"].fill_(4.)
    terms = window.reward_terms(**sample)
    assert axis(terms, "jump_precontact_wheel_velocity").lt(0.).all()
    assert axis(terms, "jump_precontact_wheel_torque").lt(0.).all()
    sample["wheel_clearance"].fill_(.2)
    assert axis(window.reward_terms(**sample), "jump_precontact_wheel_velocity").eq(0.).all()
    sample["wheel_clearance"].zero_()
    sample["com_vz"].fill_(.5)
    assert axis(window.reward_terms(**sample), "jump_precontact_wheel_velocity").eq(0.).all()


def test_impact_uses_per_wheel_force_bodyweight_and_actual_contact_duration():
    window, data = rig(3, dt=.01)
    land(window, data)
    sample = physical(window, com_vz=torch.full((3,), -.5))
    sample["contact_peak"][:] = torch.tensor([[190., 190.], [260., 0.], [260., 0.]])
    sample["weight_force"][:] = torch.tensor([125., 125., 150.])
    assert axis(window.reward_terms(**sample), "jump_impact").eq(0.).all()
    window.update(**data)
    terms = window.reward_terms(**sample)
    assert axis(terms, "jump_impact")[0] == 0.  # A summed-wheel value would exceed 2 BW.
    assert axis(terms, "jump_impact")[1] < 0.
    assert axis(terms, "jump_impact")[2] == 0.
    assert axis(terms, "jump_down_speed").lt(0.).all()


def test_static_position_cost_keeps_launch_accountability_without_step_zone_conflict():
    window, data = rig(3)
    data["origin_xy"] = torch.tensor([[.1, -.2], [.1, -.2], [.1, -.2]])
    data["commands"][1, 0] = .4
    data["landing_delta"] = torch.tensor([0., 0., .2])
    land(window, data)
    sample = physical(window)
    sample["position_xy"] = data["origin_xy"] + torch.tensor([[.3, 0.]])
    score = axis(window.reward_terms(**sample), "jump_landing_position")
    assert score[0] < 0. and score[1:].eq(0.).all()
    sample["position_xy"] = data["origin_xy"].clone()
    assert axis(window.reward_terms(**sample), "jump_landing_position").eq(0.).all()


def test_runtime_and_diagnostics_are_copies_and_never_modify_inputs_or_rng():
    window, data = rig()
    land(window, data)
    originals = {name: value.clone() for name, value in data.items()}
    rng = torch.random.get_rng_state().clone()
    for _ in range(10):
        window.reward_terms(**physical(window))
        diagnostics = window.diagnostics()
        diagnostics["jump_landing_goal_height_m"].zero_()
    for name, value in originals.items():
        assert torch.equal(data[name], value)
    assert torch.equal(torch.random.get_rng_state(), rng)
    runtime = window.runtime_state()
    assert runtime["clock_semantics"] == "first_verified_physical_contact_nonrenewable"
    assert runtime["settings"] == JUMP_LANDING_V1
    runtime["settings"]["height"]["fine_weight"] = 0.
    assert window.settings["height"]["fine_weight"] > 0.
    assert window.goal_height.item() == pytest.approx(.305)


@pytest.mark.parametrize("change", [
    {"window_seconds": float("inf")}, {"absorption_seconds": 3.},
    {"zero_tracking_baselines": False}, {"version": "unknown"},
])
def test_unbounded_or_nonzero_baseline_configurations_are_rejected(change):
    with pytest.raises(ValueError, match="Jump landing"):
        validate_jump_landing(deepcopy(JUMP_LANDING_V1) | change)
