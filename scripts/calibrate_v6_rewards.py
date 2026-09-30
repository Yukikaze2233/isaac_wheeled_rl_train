#!/usr/bin/env python3
"""CPU reward counterexamples; synthetic state traces are not physical rollouts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.rewards import (DENSE_TRACKING_V1, StationaryAnchor,
    apply_dense_tracking, dense_error_terms, reward_terms, validate_dense_tracking)


def discounted_return(density, policy_dt, gamma, *, terminal_cost=0., success_bonus=0., bootstrap=0.):
    """True failure has zero bootstrap; time-limit truncation retains it."""
    rewards = density * policy_dt
    powers = gamma ** torch.arange(len(rewards), dtype=rewards.dtype)
    return float((rewards * powers).sum() + powers[-1] * (success_bonus - terminal_cost)
                 + gamma ** len(rewards) * bootstrap)


def trajectory_terms(command, velocity, height, *, omega=None, settings=None, dt=.02,
                     motion_cost=False, stationary_anchor=True):
    """Recompute native supported-locomotion physical terms plus new tracking.

    Contact/geometry failure events are zero in this controlled state slice.
    Motor trajectories are illustrative bounded costs, not inverse dynamics.
    """
    settings = DENSE_TRACKING_V1 if settings is None else settings
    n = len(command)
    gravity = torch.zeros(n, 3, dtype=command.dtype)
    gravity[:, 2] = -1.
    omega = torch.zeros_like(gravity) if omega is None else omega
    motors = torch.zeros(n, 6, dtype=command.dtype)
    motors[:, [2, 5]] = velocity[:, :1] / .06
    acceleration = torch.diff(motors, dim=0, prepend=motors[:1]) / dt
    torque = torch.zeros_like(motors)
    actions = torch.zeros_like(motors)
    if motion_cost:
        torque[:, [0, 1, 3, 4]] = 5.
        torque[:, [2, 5]] = .15
        motors[:, [0, 1, 3, 4]] = .2
        actions[:, [0, 1, 3, 4]] = (.05 * torch.sin(torch.arange(n, dtype=command.dtype) * dt))[:, None]
    previous = torch.cat((actions[:1], actions[:-1]))
    before = torch.cat((previous[:1], previous[:-1]))
    support = torch.ones(n, dtype=command.dtype)
    ordinary = torch.ones(n, dtype=torch.bool)
    result = reward_terms(velocity, omega, gravity, height, command, motors, acceleration,
        torque, actions, previous, before, torch.zeros(n, 2, 3, dtype=command.dtype),
        support, ordinary, torch.zeros(n, dtype=torch.bool))
    reference = SimpleNamespace(jumping=torch.zeros(n, dtype=torch.bool))
    apply_dense_tracking(result, velocity, omega, gravity, height, command, support,
                         ordinary, reference, settings)
    if stationary_anchor:
        anchor = StationaryAnchor(1, "cpu", 1., .1)
        position = torch.zeros(1, 2, dtype=command.dtype)
        requested = (command[:, :2].abs() < .01).all(-1)
        values = []
        for i in range(n):
            position += velocity[i:i+1, :2] * dt
            values.append(anchor.reward(position, requested[i:i+1], torch.ones(1, dtype=torch.bool))[0])
        result["stationary_anchor"] = torch.stack(values)
    return result


def calibration_suite(settings=None, *, policy_dt=.02, gamma=.99, terminal_cost=200., stationary_anchor=True):
    settings = deepcopy(DENSE_TRACKING_V1 if settings is None else settings)
    validate_dense_tracking(settings)
    steps = round(10. / policy_dt)
    clock = torch.arange(steps, dtype=torch.float64) * policy_dt
    results, raw = {}, {}

    def add(name, cmd, vx, height, vy=0., motion_cost=False, yaw=None):
        command = torch.tensor(cmd, dtype=torch.float64).expand(steps, -1).clone() if isinstance(cmd, list) else cmd
        velocity = torch.zeros(steps, 3, dtype=torch.float64)
        velocity[:, 0], velocity[:, 1] = vx, vy
        actual_height = torch.full((steps,), float(height), dtype=torch.float64) if isinstance(height, (int, float)) else height
        omega = None
        if yaw is not None:
            omega = torch.zeros_like(velocity)
            omega[:, 2] = yaw
        terms = trajectory_terms(command, velocity, actual_height, settings=settings, dt=policy_dt,
                                 motion_cost=motion_cost, omega=omega, stationary_anchor=stationary_anchor)
        density = torch.stack(list(terms.values())).sum(0)
        raw[name] = (density, terms)
        results[name] = {"discounted_return": discounted_return(density, policy_dt, gamma),
            "mean_density": float(density.mean()),
            "term_discounted_returns": {key: discounted_return(value, policy_dt, gamma) for key, value in terms.items()}}

    add("stand_correct", [0., 0., .305], 0., .305, motion_cost=True)
    add("stand_drift", [0., 0., .305], .03, .305)
    add("stand_side_drift", [0., 0., .305], 0., .305, vy=.03)
    add("stand_wrong_height", [0., 0., .305], 0., .350)
    add("height_error_10mm", [0., 0., .305], 0., .315)
    add("height_error_5mm", [0., 0., .305], 0., .310)
    add("move_correct", [.5, 0., .305], .5, .305, motion_cost=True)
    add("move_half_speed", [.5, 0., .305], .25, .305)
    add("move_overspeed", [.5, 0., .305], .8, .305)
    add("move_high_and_fast", [.5, 0., .305], .8, .350)
    add("velocity_error_10cm_s", [.5, 0., .305], .6, .305)
    add("velocity_error_5cm_s", [.5, 0., .305], .55, .305)
    command = torch.zeros(steps, 3, dtype=torch.float64)
    command[:, 2] = .33 + .08 * torch.sin(clock * torch.pi / 5.)
    add("height_scan_correct", command, 0., command[:, 2], motion_cost=True)
    add("height_scan_middle", command, 0., .33)
    add("height_scan_drifting", command, .05, command[:, 2])
    # Visible smooth start/brake/reverse cycle and held zero-command final dwell.
    command = torch.zeros(steps, 3, dtype=torch.float64)
    command[:, 2] = .305
    command[:, 0] = torch.where(clock < 6., .5 * torch.sin(clock * torch.pi / 3.), 0.)
    add("start_stop_correct", command, command[:, 0], .305, motion_cost=True)
    add("start_stop_delayed", command, torch.roll(command[:, 0], round(.4 / policy_dt)), .305)
    add("start_stop_residual_drift", command, torch.where(clock < 6., command[:, 0], .04), .305)
    # Deliberately harsh simultaneous error envelope; this is an algebraic
    # finite-domain test, not evidence these states form a reachable trajectory.
    add("combined_error_envelope", [5., 6. * torch.pi, .43], 0., .23, yaw=0.)
    pairs = [("stand_correct", "stand_drift"), ("stand_correct", "stand_side_drift"),
        ("stand_correct", "stand_wrong_height"), ("height_error_5mm", "height_error_10mm"),
        ("move_correct", "move_half_speed"), ("move_correct", "move_overspeed"),
        ("move_correct", "move_high_and_fast"), ("velocity_error_5cm_s", "velocity_error_10cm_s"),
        ("height_scan_correct", "height_scan_middle"), ("height_scan_correct", "height_scan_drifting"),
        ("start_stop_correct", "start_stop_delayed"), ("start_stop_correct", "start_stop_residual_drift")]
    comparisons = [{"preferred": good, "counterexample": bad,
        "return_advantage": results[good]["discounted_return"] - results[bad]["discounted_return"]}
        for good, bad in pairs]
    failures = {str(t): discounted_return(raw["combined_error_envelope"][0][:max(1, round(t / policy_dt))],
        policy_dt, gamma, terminal_cost=terminal_cost) for t in (policy_dt, 1., 5., 10.)}
    early_failure = failures[str(policy_dt)]
    survived = results["combined_error_envelope"]["discounted_return"]
    constant = raw["combined_error_envelope"][0][-1]
    continuing_value = float(constant * policy_dt / (1. - gamma))
    truncated = discounted_return(raw["combined_error_envelope"][0], policy_dt, gamma, bootstrap=continuing_value)
    success = discounted_return(raw["stand_correct"][0], policy_dt, gamma, success_bonus=5.)
    finite_checks = {"failure_return": early_failure, "surviving_return": survived,
        "time_limit_bootstrapped_return": truncated, "success_return": success,
        "failure_returns_by_seconds": failures,
        "failure_worse_than_survival_and_bootstrap": all(value < min(survived, truncated) for value in failures.values())}
    return {"scope": "research_freeze_synthetic_counterexamples_not_physical_or_PPO_validation",
        "settings": settings, "policy_dt_s": policy_dt, "gamma": gamma, "stationary_anchor": stationary_anchor,
        "failure_event_cost": terminal_cost, "trajectories": results, "comparisons": comparisons,
        "termination": finite_checks,
        "all_checks_passed": all(x["return_advantage"] > 0 for x in comparisons)
            and finite_checks["failure_worse_than_survival_and_bootstrap"]}, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    settings = None
    if args.plan:
        plan = json.loads(args.plan.read_text())
        settings = plan.get("runtime_base", plan)["dense_tracking"]
    summary, raw = calibration_suite(settings)
    args.output.mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(args.output)) as writer:
        for name, (density, terms) in raw.items():
            for tick, value in enumerate(density):
                writer.add_scalar(f"RewardCalibration/{name}/total_density_per_s", float(value), tick)
            for term, values in terms.items():
                writer.add_scalar(f"RewardCalibration/{name}/mean_density_per_s/{term}", float(values.mean()), 0)
        for i, comparison in enumerate(summary["comparisons"]):
            writer.add_scalar("RewardCalibration/counterexample_return_advantage", comparison["return_advantage"], i)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": summary["all_checks_passed"], "comparisons": summary["comparisons"],
                      "termination": summary["termination"]}, indent=2))
    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
