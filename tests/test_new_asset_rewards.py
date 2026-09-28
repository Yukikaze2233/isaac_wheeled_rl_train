"""Research reward freeze and training-only support impulse contracts."""
from copy import deepcopy
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.rewards import (DENSE_TRACKING_V1, apply_dense_tracking,
    dense_error_terms, validate_dense_tracking)
from wheeled_tasks.chassis.scut_observation import build_manual35
from wheeled_tasks.chassis.step_assist import JUMP_ASSIST_V1, JumpAssist

ROOT = Path(__file__).resolve().parents[1]


def dense_slice(command, measured, *, settings=DENSE_TRACKING_V1, reference=None, spin_error=None):
    n = len(command)
    velocity, omega = measured[:, :3], measured[:, 3:6]
    gravity = torch.zeros_like(velocity)
    gravity[:, 2] = -1.
    terms = {key: torch.full((n,), 999.) for key in ("height_square", "velocity_huber",
        "height_velocity_tracking", "stationary_speed_precision", "spin_translation")}
    apply_dense_tracking(terms, velocity, omega, gravity, measured[:, 6], command,
        torch.ones(n), torch.ones(n), reference, settings, spin_velocity_error=spin_error)
    return terms


def test_broad_cost_keeps_discriminating_far_outside_precision_band():
    for axis in ("height", "velocity", "lateral", "yaw", "stationary"):
        cfg = DENSE_TRACKING_V1[axis]
        error = torch.tensor([0., .01, 1., 5., 10.], dtype=torch.float64) * cfg["broad_scale"]
        error.requires_grad_()
        broad, fine = dense_error_terms(error, cfg)
        score = broad + fine
        assert score[0] == 0.
        assert (score.diff() < 0.).all()
        slope = torch.autograd.grad(score.sum(), error)[0]
        assert slope[0] == 0.
        assert (slope[1:] < 0.).all()
        assert slope[-1].abs() > .95 * cfg["broad_weight"] / cfg["broad_scale"]
        torch.testing.assert_close(score, sum(dense_error_terms(-error, cfg)))


def test_tracking_replaces_duplicate_terms_and_targets_are_visible():
    command = torch.tensor([[.5, 0., .305], [.5, 0., .305]])
    measured = torch.tensor([[.5, 0., 0., 0., 0., 0., .305]]).repeat(2, 1)
    reference = SimpleNamespace(jumping=torch.zeros(2, dtype=torch.bool),
        acceleration=torch.tensor([5., -5.]), vertical_velocity=torch.tensor([.2, -.2]))
    terms = dense_slice(command, measured, reference=reference)
    for name in ("height_square", "velocity_huber", "height_velocity_tracking",
                 "stationary_speed_precision", "spin_translation"):
        assert name not in terms
    for term in terms.values():
        assert term[0] == term[1]
    assert terms["vertical_velocity"].eq(0).all()
    assert sum(terms.values()).tolist() == [3., 3.]


def test_stationary_blend_is_continuous_and_does_not_bias_small_commands_to_zero():
    command = torch.tensor([[.00999, 0., .305], [.01001, 0., .305], [.05, 0., .305]])
    measured = torch.zeros(3, 7)
    measured[:, 0], measured[:, 6] = command[:, 0], .305
    assert sum(dense_slice(command, measured).values()).eq(3.).all()
    measured[:, 0] += .03
    score = sum(dense_slice(command, measured).values())
    assert abs(float(score[1] - score[0])) < 1e-3


def test_spin_error_is_measured_against_native_filtered_reference():
    n = 2
    terms = {}
    zeros = torch.zeros(n, 3)
    gravity = zeros.clone()
    gravity[:, 2] = -1.
    command = zeros.clone()
    command[:, 2] = .305
    apply_dense_tracking(terms, zeros, zeros, gravity, command[:, 2], command,
        torch.ones(n), torch.zeros(n), None, DENSE_TRACKING_V1,
        spin_velocity_error=torch.tensor([0., .3]))
    assert sum(terms.values())[0] > sum(terms.values())[1]
    assert terms["track_lin_vel"].eq(1.).all()


def test_jump_apex_remains_observable_in_frozen_35d():
    n = 7
    z3, z6 = torch.zeros(n, 3), torch.zeros(n, 6)
    reference = SimpleNamespace(terrain_mode=torch.zeros(n), phase=torch.zeros(n),
        STEP_LOWER=3, elapsed=torch.full((n,), .2))
    apex = torch.tensor([.03, .06, .10, .15, .20, .25, .30])
    obs = build_manual35(z3, z3, z3, z6, z6, z6, torch.zeros(6),
        torch.ones(n, dtype=torch.bool), apex, reference)
    assert obs.shape == (7, 35)
    torch.testing.assert_close(obs[:, 33], 5 * apex)
    assert obs[:, 34].eq(.2).all()


def test_precision_cannot_be_disabled_or_hidden_derivative_enabled():
    settings = deepcopy(DENSE_TRACKING_V1)
    settings["height"]["fine_weight"] = 0.
    with pytest.raises(ValueError, match="height.fine_weight"):
        validate_dense_tracking(settings)
    settings = deepcopy(DENSE_TRACKING_V1)
    settings["ordinary_vertical_velocity_weight"] = .5
    with pytest.raises(ValueError, match="derivative"):
        validate_dense_tracking(settings)


def test_full_reward_counterexamples_and_termination_bootstrap():
    suite = runpy.run_path(str(ROOT / "scripts/calibrate_v6_rewards.py"))["calibration_suite"]
    result, _ = suite()
    assert result["all_checks_passed"]
    assert len(result["comparisons"]) == 12
    unsafe, _ = suite(terminal_cost=80.)
    assert not unsafe["termination"]["failure_worse_than_survival_and_bootstrap"]


def test_density_integrates_once_using_policy_clock():
    value = sum(dense_error_terms(torch.tensor([.1]), DENSE_TRACKING_V1["velocity"]))
    torch.testing.assert_close((value.expand(50) * .02).sum(), (value.expand(100) * .01).sum())


def test_assistance_uses_actor_schedule_and_preserves_unassisted_samples():
    assist = JumpAssist(5, "cpu", JUMP_ASSIST_V1)
    assert assist.schedule(0) == (.8, 1.)
    assert assist.schedule(150) == pytest.approx((.5, .5))
    assert assist.schedule(299)[0] > .2
    assert assist.schedule(300) == (0., 0.)
    ids = torch.arange(5)
    assist.reset(ids, 0, torch.tensor([.0, .2, .5, .8, .99]))
    assert assist.selected.tolist() == [True, True, True, False, False]
    assist.reset(ids, 0, torch.zeros(5), evaluation=True)
    assert not assist.selected.any()


def test_assistance_cannot_apply_in_air_after_window_or_in_evaluation():
    assist = JumpAssist(4, "cpu", JUMP_ASSIST_V1)
    assist.reset(torch.arange(4), 0, torch.zeros(4))
    mass, apex = torch.full((4,), 13.886), torch.full((4,), .03)
    push = torch.tensor([True, True, True, False])
    supported = torch.tensor([True, False, True, True])
    airborne = torch.tensor([False, True, True, False])
    force = assist.force(mass, apex, push, supported, airborne, .001, 0).clone()
    assert force[0] == pytest.approx(.35 * 13.886 * 9.81)
    assert force[1:].eq(0).all()
    assert assist.assisted.tolist() == [True, False, False, False]
    assert assist.force(mass, apex, push, supported, airborne, .001, 300).eq(0).all()
    assert assist.force(mass, apex, push, supported, airborne, .001, 0, evaluation=True).eq(0).all()


def test_assistance_impulse_and_duration_caps_across_repeated_push_requests():
    assist = JumpAssist(2, "cpu", JUMP_ASSIST_V1)
    assist.reset(torch.arange(2), 0, torch.zeros(2))
    mass, apex = torch.full((2,), 13.886), torch.tensor([.03, .30])
    yes, no = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    for _ in range(1000):
        assist.force(mass, apex, yes, yes, no, .001, 0)
    cap = .25 * mass * (2 * 9.81 * apex).sqrt()
    assert (assist.impulse <= cap + 1e-5).all()
    assert (assist.application_seconds <= .12 + 1e-6).all()
    assert assist.last_force.eq(0).all()
    torch.testing.assert_close(assist.impulse[0], cap[0])
    assert assist.impulse[1] < cap[1]
    # Partial reset does not refund another ongoing episode's impulse budget.
    kept = assist.impulse[1].clone()
    assist.reset(torch.tensor([0]), 0, torch.zeros(1))
    assert assist.impulse[0] == 0 and assist.impulse[1] == kept
