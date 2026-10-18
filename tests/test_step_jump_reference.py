"""Raised-platform references, physical touchdown rewards and visible35 ABI."""
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.references import CommandReference
from wheeled_tasks.chassis.rewards import reference_jump_terms
from wheeled_tasks.chassis.scut_observation import build_manual35
from wheeled_tasks.chassis.task import Phase
from wheeled_tasks.chassis.step_jump import StepJumpTask


CONFIG = {
    "step_height_m": .4, "step_down_height_m": .28,
    "step_rise_seconds": .5, "step_lower_seconds": .5,
    "step_request_range_s": [.4, 1.4], "step_hold_range_s": [5., 9.],
    "preload_seconds": .35, "preload_depth_m": .06, "release_offset_m": .035,
    "landing_seconds": .3, "takeoff_margin_seconds": .3, "pitch_limit_rad": .15,
}


def inputs(count, *, dtype=torch.float32):
    zero = torch.zeros(count, dtype=dtype)
    return [torch.full((count,), .305, dtype=dtype), zero.clone(), zero.clone(), zero.clone(),
            torch.full((count,), 7., dtype=dtype), torch.ones(count, dtype=torch.bool),
            zero.clone(), torch.full((count,), .28, dtype=dtype)]


def test_flat_reference_matches_frozen_outputs_and_explicit_zero_bitwise():
    args = inputs(8)
    args[-1][:] = .06
    args[-2][:] = torch.tensor([0., .125, .25, .29, .4, .55, .7, 1.2])
    before = CommandReference(8, "cpu", CONFIG)
    after = CommandReference(8, "cpu", CONFIG)
    assert torch.equal(before.update(*args), after.update(*args, landing_delta=torch.zeros(8)))
    for name in ("height", "height_delta", "vertical_velocity", "phase", "release_speed", "push_end"):
        assert torch.equal(getattr(before, name), getattr(after, name))
    # Saved from the flat reference before adding the raised-platform branch.
    assert torch.equal(after.height, torch.tensor([
        .3050000071525574, .2875072956085205, .2568950653076172, .24968524277210236,
        .25274473428726196, .36396074295043945, .3797312378883362, .3050000071525574]))
    assert torch.equal(after.vertical_velocity, torch.tensor([
        -0., -.2361515909433365, -.2099125236272812, -.14609912037849426,
        .3097895681858063, .8408864736557007, -.6306133270263672, 0.]))
    assert after.phase.tolist() == [4, 4, 4, 4, 5, 6, 6, 8]


@pytest.mark.parametrize("delta", [.20, .25, -.15])
def test_step_ballistic_touchdown_and_hermite_are_continuous(delta):
    args = inputs(2, dtype=torch.float64)
    ref = CommandReference(2, "cpu", CONFIG)
    target = torch.full((2,), delta, dtype=torch.float64)
    ref.update(*args, landing_delta=target)
    release = ref.release_speed[0].item()
    flight_time = (ref.flight_end - ref.push_end)[0].item()
    assert CONFIG["release_offset_m"] + release * flight_time - 4.905 * flight_time**2 == pytest.approx(delta)
    assert ref.touchdown_velocity[0] == pytest.approx(release - 9.81 * flight_time)
    assert ref.touchdown_velocity[0] < 0.
    for boundary in (CONFIG["preload_seconds"], ref.push_end[0].item(),
                     ref.flight_end[0].item(), ref.flight_end[0].item() + CONFIG["landing_seconds"]):
        args[-2][:] = torch.tensor([boundary - 1e-8, boundary + 1e-8], dtype=torch.float64)
        ref.update(*args, landing_delta=target)
        assert abs(float(ref.height[1] - ref.height[0])) < 1e-6
        assert abs(float(ref.vertical_velocity[1] - ref.vertical_velocity[0])) < 1e-6
    assert ref.height_delta[1] == pytest.approx(delta)
    assert ref.vertical_velocity[1] == 0.
    assert ref.phase[1] == ref.RECOVER


@pytest.mark.parametrize("delta,apex", [(.20, .10), (float("nan"), .28),
                                         (float("inf"), .28), (.20, 0.), (.20, float("nan"))])
def test_unreachable_or_nonfinite_commands_are_rejected(delta, apex):
    args = inputs(1)
    args[-1][:] = apex
    with pytest.raises(ValueError, match="reachable"):
        CommandReference(1, "cpu", CONFIG).update(*args, landing_delta=torch.tensor([delta]))


def test_mixed_flat_and_step_rows_keep_legacy_values_and_reset_rise():
    args = inputs(3)
    args[-2][:] = .7
    flat = CommandReference(3, "cpu", CONFIG)
    mixed = CommandReference(3, "cpu", CONFIG)
    flat.update(*args)
    rise = torch.tensor([0., .20, .25])
    mixed.update(*args, landing_delta=rise)
    for name in ("height", "height_delta", "vertical_velocity", "phase", "release_speed", "push_end"):
        assert torch.equal(getattr(flat, name)[:1], getattr(mixed, name)[:1])
    mixed.reset(torch.tensor([1]))
    assert rise.tolist() == pytest.approx([0., .20, .25])
    assert mixed.landing_delta.tolist() == pytest.approx([0., 0., .25])
    assert mixed.flight_end[1] == mixed.touchdown_velocity[1] == 0.


def test_manual35_uses_only_reserved_slot_for_preset_platform_rise():
    args = inputs(3)
    args[5][0] = False
    args[-2][:] = .2
    ref = CommandReference(3, "cpu", CONFIG)
    ref.update(*args, landing_delta=torch.tensor([.20, 0., .25]))
    zero3, zero6 = torch.zeros(3, 3), torch.zeros(3, 6)
    observation_args = (zero3, zero3, torch.tensor([[.4, 0., .305]]).repeat(3, 1),
                        zero6, zero6, zero6, torch.zeros(6), args[5], args[-1])
    raised = build_manual35(*observation_args, ref)
    ref.landing_delta.zero_()
    flat = build_manual35(*observation_args, ref)
    assert raised.shape == flat.shape == (3, 35)
    assert torch.equal(raised[:, :30], flat[:, :30])
    assert torch.equal(raised[:, 31:], flat[:, 31:])
    torch.testing.assert_close(raised[:, 30], torch.tensor([0., 0., 1.25]))


def reward_inputs(ref, phase, contact_time):
    n = len(ref.height)
    h, speed = ref.step_landing_reference(contact_time)
    task = SimpleNamespace(com_released=torch.ones(n, dtype=torch.bool),
        com_displacement=h.clone(), com_rise=ref.release_speed.square() / (2 * 9.81),
        clear_air_time_peak=torch.full((n,), .1), com_release_speed=ref.release_speed.clone(),
        origin_xy=torch.zeros(n, 2))
    args = (task, phase, torch.ones(n, 2, dtype=torch.bool), torch.full((n,), .305),
            speed, torch.zeros(n, 3), torch.tensor([[0., 0., -1.]]).repeat(n, 1),
            torch.full((n,), .16), torch.zeros(n, 3), torch.tensor([[0., 0., .305]]).repeat(n, 1))
    return task, args


def test_physical_early_touchdown_earns_step_landing_reward_before_reference_land():
    ref = CommandReference(1, "cpu", CONFIG)
    args = inputs(1)
    ref.update(*args, landing_delta=torch.tensor([.20]))
    args[-2][:] = ref.flight_end - .05
    ref.update(*args, landing_delta=torch.tensor([.20]))
    assert ref.phase.item() == ref.FLIGHT
    contact_time = torch.tensor([.04])
    _, reward_args = reward_inputs(ref, torch.tensor([Phase.LANDING]), contact_time)
    terms = reference_jump_terms(ref, *reward_args, landing_contact_time=contact_time)
    assert terms["jump_landing_height"].item() == pytest.approx(2.)
    assert terms["jump_landing_vertical_velocity"].item() == pytest.approx(1.)
    assert terms["jump_recovery_height"].item() == 0.


def test_step_recovery_targets_platform_rise_with_broad_reward_outside_fine_kernel():
    ref = CommandReference(4, "cpu", CONFIG)
    args = inputs(4)
    args[-2][:] = 2.
    ref.update(*args, landing_delta=torch.full((4,), .20))
    contact_time = torch.full((4,), .5)
    task, reward_args = reward_inputs(ref, torch.full((4,), Phase.RECOVERY), contact_time)
    task.com_displacement -= torch.tensor([0., .15, .30, .60])
    terms = reference_jump_terms(ref, *reward_args, landing_contact_time=contact_time)
    reward = terms["jump_recovery_height"]
    assert reward[0] == pytest.approx(2.)
    assert (reward[:-1] > reward[1:]).all()
    assert terms["jump_landing_position"].eq(0.).all()


def test_flat_rewards_remain_bitwise_legacy_even_with_contact_time():
    ref = CommandReference(3, "cpu", CONFIG)
    args = inputs(3)
    args[-2][:] = torch.tensor([.3, .8, 2.])
    ref.update(*args, landing_delta=torch.zeros(3))
    contact_time = torch.tensor([0., .02, .5])
    _, reward_args = reward_inputs(ref, torch.tensor([Phase.TAKEOFF, Phase.LANDING, Phase.RECOVERY]), contact_time)
    legacy_reference = SimpleNamespace(**vars(ref), PRELOAD=ref.PRELOAD, PUSH=ref.PUSH,
        LAND=ref.LAND, RECOVER=ref.RECOVER)
    legacy = reference_jump_terms(legacy_reference, *reward_args)
    current = reference_jump_terms(ref, *reward_args, landing_contact_time=contact_time)
    assert legacy.keys() == current.keys()
    for key in legacy:
        assert torch.equal(legacy[key], current[key]), key


@pytest.mark.parametrize("rise", [.03, .20, .25])
def test_request_controller_and_reference_share_release_and_flight_semantics(rise):
    settings = {"rises_m": [rise], "treads_m": [.8], "trigger_source": "laser",
        "launch_distance_m": .25, "approach_speed_m_s": .4, "landing_margin_m": .1,
        "settle_seconds": .35, "landing_clearance_m": .08}
    controller = StepJumpTask(["jump"], {"jump": {"step_jump": settings}}, "cpu", .02)
    controller.control(torch.tensor([.02]), torch.zeros(1), torch.ones(1, dtype=torch.bool),
        torch.tensor([9.81]), torch.tensor([.2]), torch.ones(1, dtype=torch.bool), torch.zeros(1),
        preload_seconds=CONFIG["preload_seconds"], preload_depth=CONFIG["preload_depth_m"],
        release_offset=CONFIG["release_offset_m"])
    args = inputs(1)
    args[-1] = controller.apex
    ref = CommandReference(1, "cpu", CONFIG)
    ref.update(*args, landing_delta=controller.landing_delta)
    assert torch.equal(ref.flight_end, controller.flight_end)
    assert ref.release_speed.item() ** 2 / (2 * 9.81) == pytest.approx(rise + .08)
    assert CONFIG["release_offset_m"] + ref.release_speed.item() ** 2 / (2 * 9.81) > rise
