"""Observe native policy-step payments without evaluating a second reward.

The caller authenticates this file with the other diagnostic sources. This
bounded helper owns one temporary native event observer, not a reward producer.
It neither changes control histories nor reads delay buffers or random streams.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import importlib
import json

import torch

from locomotion_diagnostic_suite import DEFAULT_SUITE, get_suite


CASES = get_suite(DEFAULT_SUITE).case_names
EVENTS = ("success_event", "failure_event")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor(value, shape, name, *, boolean=False, integer=False):
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
            f"Native {name} shape differs from {shape}")
    if boolean:
        require(value.dtype == torch.bool, f"Native {name} must be bool")
    elif integer:
        require(value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64),
                f"Native {name} must be integer")
    else:
        require(value.is_floating_point(), f"Native {name} must be floating point")
    require(bool(torch.isfinite(value).all()), f"Nonfinite native {name}")
    return value.detach().clone()


def values(value):
    return value.detach().cpu().tolist()


class NativeRewardPaymentCapture:
    """Record actual density/event payments before the first-cohort ledger closes.

    ``begin`` precedes exactly one native env.step; ``capture`` follows it and
    precedes ledger.observe. First native terminal frames are included. Later
    native payments remain visible but receive no first-cohort credit. A caller
    must close this observer in finally, including when native stepping fails.
    """

    def __init__(self, env, ledger, stream, *, fixture_role, suite_name=DEFAULT_SUITE):
        self.suite = get_suite(suite_name)
        self.suite_name = self.suite.name
        require(fixture_role in ("clean", "pipeline_age_only"), "Unsupported fixture role")
        require(env.num_envs == self.suite.num_envs and (env.dt, env.policy_dt, env.decimation) == (.005, .02, 4),
                "Named suite population and native 200/50Hz clocks required")
        names = tuple(env.scene_groups)
        require(Counter(names) == Counter({name: self.suite.replicas for name in self.suite.case_names})
                and ledger.suite_name == self.suite_name
                and tuple(row["case"] for row in ledger.rows) == names,
                "Native named suite cases x N4 and matching ledger required")
        require(env.max_episode_length == self.suite.policy_steps
                and torch.equal(tensor(env.episode_limits, (env.num_envs,), "episode_limits", integer=True),
                                torch.full_like(env.episode_limits, self.suite.policy_steps)),
                "Native episode limits differ from named suite horizon")
        cfg = env.cfg
        require(cfg.get("auto_reset") is False and cfg.get("record_diagnostics") is True,
                "Pre-reset native diagnostics and no auto-reset required")
        require(cfg.get("dense_tracking", {}).get("enabled") is True
                and "working_margin_weight" in cfg,
                "Fixture must expose all density payments (dense and working margin)")
        require(cfg.get("reward_profile") == "scut_v14_flat_v5", "Native SCUT fixture profile required")
        require(hasattr(stream, "write"), "Writable JSONL stream required")
        require(env.tick == 0 and all(ledger.live), "Observer must start at fresh first-cohort birth")
        self.env, self.ledger, self.stream = env, ledger, stream
        self.fixture_role, self.names, self.count = fixture_role, names, len(names)
        self.module = importlib.import_module("wheeled_tasks.chassis.rewards")
        self.original = self.module.episode_event_terms
        require(not getattr(self.original, "_native_payment_observer", False), "Native payment observer already installed")
        self.active, self.closed = None, False
        self.completed_policy_steps = self.rows_written = self.event_calls = 0
        self.max_residual = 0.
        self.totals = {scope: defaultdict(float) for scope in ("all_native", "first_cohort")}
        self.cohort_rows = 0

        def observe(success_event, terminated, settings):
            require(self.active is not None and not self.closed, "Native event occurred outside one policy interval")
            self.event_calls += 1
            self.active["event_calls"] += 1
            result = self.original(success_event, terminated, settings)  # Exactly one native producer call.
            # Preserve the native result object and every tensor handed to env.step.
            self.active["event"] = {"success_input": success_event.detach().clone(),
                "terminated_input": terminated.detach().clone(),
                "payments": {name: value.detach().clone() for name, value in result.items()}}
            return result

        observe._native_payment_observer = True
        self.observer = observe
        self.module.episode_event_terms = observe

    def begin(self, policy_tick, physics_tick_start, phases):
        require(not self.closed and self.active is None, "Unclosed native payment interval")
        require(type(policy_tick) is int and 1 <= policy_tick <= self.suite.policy_steps
                and policy_tick == self.completed_policy_steps + 1, "Sequential bounded native policy clock required")
        require(type(physics_tick_start) is int and physics_tick_start == 4 * (policy_tick - 1)
                and self.env.tick == policy_tick - 1, "Pre-step native clocks disagree")
        require(isinstance(phases, (tuple, list)) and len(phases) == self.count
                and all(isinstance(phase, str) and phase for phase in phases), "Trajectory phase per lane required")
        self.active = {"policy_tick": policy_tick, "physics_tick_start": physics_tick_start,
            "phases": tuple(phases), "live": tuple(self.ledger.live), "event_calls": 0, "event": None,
            "old_actions": tensor(self.env.actions, (self.count, 6), "pre-step actions"),
            "old_previous_actions": tensor(self.env.previous_actions, (self.count, 6), "pre-step previous_actions"),
            "old_motor_velocity": tensor(self.env.previous_motor_velocity, (self.count, 6), "previous_motor_velocity used")}

    def capture(self, reward, done, extras, physics_tick_end):
        """Consume final native extras; reconciliation is an audit, never an env input."""
        require(self.active is not None and not self.closed, "No active native payment interval")
        frame = self.active
        tick = frame["policy_tick"]
        require(type(physics_tick_end) is int and physics_tick_end <= self.suite.physics_steps
                and physics_tick_end == 4 * tick and self.env.tick == tick,
                "Post-step native clocks disagree")
        require(tuple(self.ledger.live) == frame["live"], "Capture must precede first-cohort ledger.observe")
        require(frame["event_calls"] == 1, "Expected exactly one native episode-event call per policy step")
        require(isinstance(extras, Mapping) and isinstance(extras.get("diagnostics"), Mapping), "Native extras diagnostics missing")
        diag = extras["diagnostics"]
        reward = tensor(reward, (self.count,), "returned reward")
        done = tensor(done, (self.count,), "returned done", boolean=True)
        require(torch.equal(tensor(diag.get("reward"), (self.count,), "diagnostics reward"), reward)
                and torch.equal(tensor(diag.get("done"), (self.count,), "diagnostics done", boolean=True), done),
                "Native returned totals/flags differ from cached diagnostics")
        flags = {name: tensor(diag.get(name), (self.count,), name, boolean=True)
                 for name in ("terminated", "success")}
        flags["timeout"] = tensor(extras.get("time_outs"), (self.count,), "time_outs", boolean=True)
        reasons = diag.get("reasons")
        require(isinstance(reasons, Mapping) and all(isinstance(name, str) for name in reasons), "Native reasons missing")
        reasons = {name: tensor(mask, (self.count,), f"reason {name}", boolean=True) for name, mask in reasons.items()}
        event = frame["event"]
        success_input = tensor(event["success_input"], (self.count,), "success_event input", boolean=True)
        require(torch.equal(tensor(event["terminated_input"], (self.count,), "terminated event input", boolean=True), flags["terminated"])
                and not bool((success_input & ~flags["success"]).any()), "Native event inputs and terminal diagnostics disagree")
        require(set(event["payments"]) == set(EVENTS), "Unexpected native discrete event payments")
        events = {name: tensor(event["payments"][name], (self.count,), name) for name in EVENTS}
        components = extras.get("native_reward_components")
        require(isinstance(components, Mapping) and bool(components)
                and all(isinstance(name, str) and name for name in components), "Final native density dictionary missing")
        densities = {name: tensor(value, (self.count,), f"density {name}") for name, value in components.items()}
        require(not set(EVENTS).intersection(densities), "Discrete event incorrectly exposed as density")
        # One consumer audit reduction; do not re-evaluate native reward functions.
        density_total = torch.stack(list(densities.values())).sum(0) * self.env.policy_dt
        event_total = events["success_event"] + events["failure_event"]
        reconstructed = density_total + events["success_event"] + events["failure_event"]
        residual = reward - reconstructed
        tolerance = 2e-5 + 1e-6 * reward.abs()
        require(bool((residual.abs() <= tolerance).all()), "Native density/event ledger does not reconcile with original reward")
        current = tensor(self.env.actions, (self.count, 6), "current control actions")
        previous = tensor(self.env.previous_actions, (self.count, 6), "current previous_actions")
        before = tensor(self.env.before_previous_actions, (self.count, 6), "current before_previous_actions")
        require(torch.equal(previous, frame["old_actions"]) and torch.equal(before, frame["old_previous_actions"]),
                "Native action histories shifted/reset outside bounded fixture semantics")
        if "actions" in diag:
            require(torch.equal(tensor(diag["actions"], (self.count, 6), "diagnostics actions"), current), "Cached control actions disagree")
        shapes = {"commands": (3,), "velocity": (3,), "omega": (3,), "gravity": (3,), "height": (),
                  "motor_position": (6,), "motor_velocity": (6,), "motor_effort": (6,), "physical_phase_time_s": ()}
        state = {name: tensor(diag.get(name), (self.count, *shape), name) for name, shape in shapes.items()}
        state["physical_phase"] = tensor(diag.get("physical_phase"), (self.count,), "physical_phase", integer=True)
        state.update(actions=current, previous_actions=previous, before_previous_actions=before,
                     previous_motor_velocity_used=frame["old_motor_velocity"])
        cpu_state = {name: values(value) for name, value in state.items()}
        cpu_flags = {name: values(value) for name, value in flags.items()} | {"done": values(done), "success_event_input": values(success_input)}
        cpu_reasons = {name: values(value) for name, value in reasons.items()}
        cpu_densities, cpu_events = ({name: values(value) for name, value in group.items()} for group in (densities, events))
        totals = {name: values(value) for name, value in {"density_total": density_total, "event_total": event_total,
            "reconstructed_total": reconstructed, "residual": residual, "absolute_tolerance": tolerance, "reward": reward}.items()}
        rows = []
        for lane, identity in enumerate(self.ledger.rows):
            rows.append({"schema": "native-reward-payment-v1", "fixture_role": self.fixture_role,
                "suite_name": self.suite_name,
                **{name: identity[name] for name in ("lane", "case", "replica", "birth_id")},
                "policy_tick": tick, "physics_tick_start": frame["physics_tick_start"], "physics_tick_end": physics_tick_end,
                "policy_dt_s": self.env.policy_dt, "physics_dt_s": self.env.dt, "phase": frame["phases"][lane],
                "first_cohort": frame["live"][lane], "after_first_terminal": not frame["live"][lane],
                "reward_total": totals["reward"][lane],
                "density_per_s": {name: value[lane] for name, value in cpu_densities.items()},
                "density_payment": {name: value[lane] * self.env.policy_dt for name, value in cpu_densities.items()},
                "density_payment_semantics": "derived_density_times_policy_dt_not_second_native_evaluation",
                "event_payment": {name: value[lane] for name, value in cpu_events.items()},
                "reconciliation": {name: value[lane] for name, value in totals.items() if name != "reward"} | {"passed": True},
                "state": {name: value[lane] for name, value in cpu_state.items()},
                "terminal": {name: value[lane] for name, value in cpu_flags.items()} | {
                    "reasons": {name: value[lane] for name, value in cpu_reasons.items()}}})
        self.stream.write("".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows))
        for row in rows:
            scopes = ("all_native", "first_cohort") if row["first_cohort"] else ("all_native",)
            for scope in scopes:
                for name in ("density_total", "event_total"):
                    self.totals[scope][name] += row["reconciliation"][name]
                self.totals[scope]["reward_total"] += row["reward_total"]
            self.cohort_rows += row["first_cohort"]
        self.max_residual = max(self.max_residual, float(residual.abs().max()))
        self.completed_policy_steps += 1
        self.rows_written += len(rows)
        self.active = None
        return rows

    def summary(self):
        return {"schema": "native-reward-payment-v1", "fixture_role": self.fixture_role,
            "suite_name": self.suite_name, "suite": self.suite.as_dict(),
            "native_policy_steps_captured": self.completed_policy_steps, "native_event_calls": self.event_calls,
            "rows_written": self.rows_written, "first_cohort_rows": self.cohort_rows,
            "incomplete_policy_interval": self.active is not None, "closed": self.closed,
            "max_abs_reconciliation_residual": self.max_residual,
            "payment_totals": {scope: dict(total) for scope, total in self.totals.items()},
            "density_units": "per_second_times_policy_dt", "event_units": "discrete_not_times_dt",
            "history_order": "native_clipped_control_order_6",
            "source": "final_native_reward_components_and_one_native_episode_event_call",
            "post_first_terminal_native_payments_recorded_but_not_credited": True, "qualification": None}

    def close(self):
        if self.closed:
            return
        require(self.module.episode_event_terms is self.observer, "Native event observer ownership changed")
        self.module.episode_event_terms = self.original
        self.closed = True
