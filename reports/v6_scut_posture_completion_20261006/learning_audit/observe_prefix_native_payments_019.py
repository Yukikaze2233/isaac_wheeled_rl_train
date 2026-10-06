"""Consume actual native density/event outputs; never evaluate a reward producer twice.

This artifact has no App, controller, network, optimizer or random sampling entry.
The driver binds its bytes before importing it. Densities are reward/s; discrete
events are reward/event. The whole elapsed interval mask owns actor eligibility.
"""
from __future__ import annotations

from collections.abc import Mapping
import importlib
import math

import torch

LESSONS = ("prefix_stand", "prefix_release", "original_chain", "ordinary_base")
GOALS = ("stand", "forward", "yaw_positive", "yaw_negative", "ordinary_base_not_recovery_goal")
PARTITIONS = ("ordinary_base_actor_valid", "ordinary_base_excluded",
              "recovery_excluded", "recovery_actor_valid_terminal",
              "recovery_actor_valid_release_interval", "recovery_actor_valid_still_holding",
              "recovery_actor_valid_postrelease")
SCOPES = ("all_native_intervals", "first_birth_intervals")
EVENTS = ("success_event", "failure_event")
GATES = ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed")
STATE_FIELDS = ("dq_L_joint1_rad_s", "dq_LL_joint1_rad_s", "dq_L_joint3_rad_s",
                "dq_R_joint1_rad_s", "dq_RR_joint1_rad_s", "dq_R_joint3_rad_s",
                "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s", "actual_vx_m_s",
                "actual_vy_m_s", "command_vx_m_s", "command_yaw_rad_s",
                "wheel_rim_max_m_s", "leg_max_rad_s", "gyro_all_norm_rad_s",
                "planar_speed_world_m_s", "sensor_all_eight_pass_at_last_reference",
                "actual_contact_hold_peak_s", "sensor_hold_peak_s", "followup_s",
                "height_m", "gravity_x", "gravity_y", "gravity_z", "height_command_m") + tuple(
                    "sensor_" + gate + "_pass_at_last_reference" for gate in GATES)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def tensor(value, shape, name, *, boolean=False):
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
            "Native tensor shape differs: " + name)
    require(value.dtype == torch.bool if boolean else value.is_floating_point(),
            "Native tensor dtype differs: " + name)
    return value.detach()


def original_goal_ids(env):
    """Immutable skill goals, before the L0 adapter has overwritten followup."""
    r = env.recovery
    specs = env.skills.specs
    ids = []
    for name, enabled in zip(env.scene_groups, r.enabled.detach().cpu().tolist()):
        if not enabled:
            ids.append(4)
            continue
        command = specs[name].get("recovery_followup_command")
        require(isinstance(command, (list, tuple)) and len(command) == 3
                and all(type(x) in (int, float) and math.isfinite(x) for x in command),
                "Original recovery goal absent or nonfinite: " + name)
        vx, yaw, _ = command
        require((vx == 0 and yaw in (0, .6, -.6)) or (vx == .3 and yaw == 0),
                "Original recovery goal differs: " + name)
        ids.append(1 if vx else 2 if yaw > 0 else 3 if yaw < 0 else 0)
    return torch.tensor(ids, dtype=torch.long, device=env.device)


class NativePaymentObserver:
    """A bounded device ledger with one observational event producer hook."""
    BINS = len(LESSONS) * len(GOALS) * len(PARTITIONS)

    def __init__(self, env, live_reader, hold_reader, goals, *, event_module=None, policy_steps=600):
        require((env.dt, env.policy_dt, env.decimation) == (.005, .02, 4), "Native payment clocks differ")
        require("recovery_takeover" not in env.cfg and env.recovery_takeover_settings is None,
                "Inactive takeover boundary differs")
        require(type(policy_steps) is int and policy_steps > 0, "Positive bounded policy steps required")
        self.env, self.n, self.device = env, env.num_envs, env.device
        self.live_reader, self.hold_reader = live_reader, hold_reader
        require(isinstance(goals, torch.Tensor) and goals.shape == (self.n,) and goals.dtype == torch.long
                and bool(((goals >= 0) & (goals < len(GOALS))).all()), "Original goal ids differ")
        self.goals = goals.detach().clone()
        self.steps, self.tick, self.event_calls = policy_steps, 0, 0
        self.active, self.closed = None, False
        self.first_closed = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.counts = torch.zeros(2, self.BINS, dtype=torch.long, device=self.device)
        self.term_names = None
        self.sums = self.squares = self.minimum = self.maximum = None
        self.max_residual = torch.zeros((), dtype=torch.float64, device=self.device)
        self.max_residual_tick = torch.zeros((), dtype=torch.long, device=self.device)
        self.max_residual_lane = torch.zeros((), dtype=torch.long, device=self.device)
        self.module = event_module or importlib.import_module("wheeled_tasks.chassis.rewards")
        self.original_event = self.module.episode_event_terms
        require(not getattr(self.original_event, "_prefix_native_payment_observer", False),
                "Native payment event observer already installed")

        def event(success_event, terminated, settings):
            require(not self.closed and self.active is not None, "Native event outside payment interval")
            require(self.active["event"] is None, "Duplicate native event producer call")
            require(settings is self.env.cfg, "Native event contract object differs")
            checked_success = tensor(success_event, (self.n,), "success_event input", boolean=True)
            checked_terminated = tensor(terminated, (self.n,), "terminated input", boolean=True)
            result = self.original_event(success_event, terminated, settings)
            require(isinstance(result, Mapping) and set(result) == set(EVENTS), "Native event output catalog differs")
            self.active["event"] = dict(success_input=checked_success.clone(), terminated_input=checked_terminated.clone(),
                payments={name: tensor(result[name], (self.n,), name).clone() for name in EVENTS})
            self.event_calls += 1
            return result  # Exact original mapping and tensors handed back to env.step.

        event._prefix_native_payment_observer = True
        self.event_observer = event
        self.module.episode_event_terms = event

    def begin(self, tick):
        require(not self.closed and self.active is None, "Unclosed native payment interval")
        require(type(tick) is int and tick == self.tick + 1 and tick <= self.steps
                and self.env.tick == tick - 1, "Native payment policy clock differs")
        r = self.env.recovery
        live = tensor(self.live_reader(), (self.n,), "first live", boolean=True)
        require(torch.equal(live, ~self.first_closed), "First-birth reader reopened or omitted a lane")
        self.active = dict(tick=tick, event=None, live=live.clone(),
            hold=tensor(self.hold_reader(), (self.n,), "original hold start", boolean=True).clone(),
            enabled=tensor(r.enabled, (self.n,), "old enabled", boolean=True).clone(),
            lesson=r.pose_curriculum.level_ids.detach().clone(),
            profile=r.pose_curriculum.profile_ids.detach().clone())

    def _allocate(self, names):
        self.term_names = names
        shape = (2, self.BINS, len(names))
        self.sums = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.squares = torch.zeros_like(self.sums)
        self.minimum = torch.full_like(self.sums, float("inf"))
        self.maximum = torch.full_like(self.sums, -float("inf"))

    def capture(self, reward, done, extras):
        require(not self.closed and self.active is not None, "Missing native payment interval")
        f, n = self.active, self.n
        require(self.env.tick == f["tick"], "Native payment physical end clock differs")
        require(f["event"] is not None, "Missing native event producer call")
        require(torch.equal(self.live_reader(), f["live"]), "Payment capture must precede first-cohort closure")
        require(isinstance(extras, Mapping) and isinstance(extras.get("diagnostics"), Mapping),
                "Pre-reset native diagnostics absent")
        d, event = extras["diagnostics"], f["event"]
        reward = tensor(reward, (n,), "returned reward")
        done = tensor(done, (n,), "returned done", boolean=True)
        terminated = tensor(d.get("terminated"), (n,), "diagnostic terminated", boolean=True)
        success = tensor(d.get("success"), (n,), "diagnostic success", boolean=True)
        valid = tensor(extras.get("policy_learning_mask"), (n,), "whole policy learning mask", boolean=True)
        enabled = tensor(d.get("recovery_enabled"), (n,), "diagnostic enabled", boolean=True)
        released = tensor(d.get("recovery_motion_released"), (n,), "diagnostic released", boolean=True)
        require(torch.equal(reward, tensor(d.get("reward"), (n,), "diagnostic reward"))
                and torch.equal(done, tensor(d.get("done"), (n,), "diagnostic done", boolean=True)),
                "Actual returned reward/done differs from pre-reset diagnostics")
        require(torch.equal(event["terminated_input"], terminated)
                and not bool((event["success_input"] & ~success).any()), "Native event input/diagnostic flags differ")
        require(torch.equal(enabled, f["enabled"])
                and torch.equal(d.get("recovery_prefix_lesson_id"), f["lesson"])
                and torch.equal(d.get("recovery_actual_profile_id"), f["profile"]), "Old lesson/profile attribution differs")
        require(not bool((~enabled & ~valid).any()), "Ordinary base unexpectedly excluded")
        require(not bool((enabled & valid & ~f["hold"] & ~released & ~terminated).any()),
                "Actor-valid nonholding recovery interval lacks native release")
        components = extras.get("native_reward_components")
        require(isinstance(components, Mapping) and components
                and all(isinstance(k, str) and k for k in components), "Native density dictionary absent")
        require(not any(k in EVENTS or k.startswith("recovery_takeover_") for k in components),
                "Discrete/inactive takeover term exposed as density")
        density_values = [tensor(v, (n,), "density " + k) for k, v in components.items()]
        density_total = torch.stack(density_values).sum(0) * self.env.policy_dt
        payments = torch.stack([event["payments"][k] for k in EVENTS], -1)
        residual = reward - (density_total + payments[:, 0] + payments[:, 1])
        density_names = tuple(sorted(components))
        densities = torch.stack([components[k].detach() for k in density_names], -1).double()
        require(bool(torch.isfinite(densities).all() & torch.isfinite(payments).all()
                     & torch.isfinite(reward).all()), "Nonfinite actual native payment")
        require(bool((residual.abs() <= 2e-5 + 1e-6 * reward.abs()).all()), "Native density/event ledger does not reconcile")
        commands = tensor(d.get("commands"), (n, 3), "commands")
        dq = tensor(d.get("motor_velocity"), (n, 6), "actual motor velocity")
        gyro = tensor(d.get("omega"), (n, 3), "actual body gyro")
        velocity = tensor(d.get("velocity"), (n, 3), "actual body velocity")
        gates = torch.stack([tensor(d.get("recovery_stable_" + k + "_pass"), (n,), k, boolean=True)
                             for k in GATES], -1)
        state = torch.cat((dq, gyro, velocity[:, :2], commands[:, :2],
            (.06 * dq[:, (2, 5)].abs().amax(-1))[:, None], dq[:, (0, 1, 3, 4)].abs().amax(-1)[:, None],
            gyro.norm(dim=-1)[:, None], tensor(d.get("planar_speed_world"), (n,), "planar speed")[:, None],
            gates.all(-1).to(dq.dtype)[:, None],
            torch.stack([tensor(d.get(k), (n,), k) for k in
                         ("recovery_rl_stable_s", "recovery_stable_peak_seconds", "recovery_followup_s")], -1),
            tensor(d.get("height"), (n,), "actual height")[:, None],
            tensor(d.get("gravity"), (n, 3), "actual gravity"), commands[:, 2:3], gates.to(dq.dtype)), -1).double()
        require(bool(torch.isfinite(state).all()), "Nonfinite actual native payment state")
        selected = torch.where(valid, reward, torch.zeros_like(reward))
        values = torch.cat((densities, payments.double(), reward.double()[:, None], selected.double()[:, None], state), -1)
        names = density_names + EVENTS + ("returned_immediate_reward", "actor_eligible_immediate_reward") + STATE_FIELDS
        require(len(set(names)) == len(names), "Native density collides with payment audit field")
        if self.term_names is None:
            self._allocate(names)
        require(names == self.term_names, "Native component catalog changed during bounded observation")
        lesson = torch.where(enabled, f["lesson"], 3)
        goal = torch.where(enabled, self.goals, 4)
        require(bool(((lesson >= 0) & (lesson < 4) & ((~enabled) | (goal < 4))).all()), "Native lesson/goal labels differ")
        partition = torch.where(~enabled, torch.where(valid, 0, 1),
            torch.where(~valid, 2, torch.where(terminated, 3,
                torch.where(f["hold"], torch.where(released, 4, 5), 6))))
        bins = (lesson * len(GOALS) + goal) * len(PARTITIONS) + partition
        for scope, members in enumerate((torch.ones_like(valid), f["live"])):
            ids, selected_values = bins[members], values[members]
            index = ids[:, None].expand_as(selected_values)
            self.counts[scope].scatter_add_(0, ids, torch.ones_like(ids))
            self.sums[scope].scatter_add_(0, index, selected_values)
            self.squares[scope].scatter_add_(0, index, selected_values.square())
            self.minimum[scope].scatter_reduce_(0, index, selected_values, reduce="amin", include_self=True)
            self.maximum[scope].scatter_reduce_(0, index, selected_values, reduce="amax", include_self=True)
        peak, lane = residual.abs().max(0)
        exceeds = peak.double() > self.max_residual
        self.max_residual_tick = torch.where(exceeds, self.max_residual_tick.new_tensor(f["tick"]), self.max_residual_tick)
        self.max_residual_lane = torch.where(exceeds, lane, self.max_residual_lane)
        self.max_residual = torch.maximum(self.max_residual, peak.double())
        self.first_closed |= f["live"] & done
        self.tick = f["tick"]
        self.active = None

    def close(self):
        if self.closed:
            return
        require(self.module.episode_event_terms is self.event_observer, "Native event observer ownership changed")
        self.module.episode_event_terms = self.original_event
        self.closed = True

    def summary(self):
        require(self.active is None and self.tick == self.steps and self.event_calls == self.steps,
                "Incomplete native payment window")
        require(int(self.counts[0].sum()) == self.n * self.steps, "All-native payment population differs")
        require(self.term_names is not None, "No native payment samples")
        arrays = [v.cpu().tolist() for v in (self.counts, self.sums, self.squares, self.minimum, self.maximum)]
        result = []
        density_count = len(self.term_names) - len(EVENTS) - 2 - len(STATE_FIELDS)
        for scope, scope_name in enumerate(SCOPES):
            for b, count in enumerate(arrays[0][scope]):
                if not count:
                    continue
                lesson_goal, part = divmod(b, len(PARTITIONS))
                lesson, goal = divmod(lesson_goal, len(GOALS))
                fields = {}
                for k, name in enumerate(self.term_names):
                    total = arrays[1][scope][b][k]
                    mean = total / count
                    fields[name] = dict(sum=total, mean=mean,
                        standard_deviation=max(0., arrays[2][scope][b][k] / count - mean * mean) ** .5,
                        minimum=arrays[3][scope][b][k], maximum=arrays[4][scope][b][k])
                    if k < density_count:
                        fields[name].update(unit="reward_per_second", sum_payment=total * .02,
                                            mean_payment=mean * .02)
                    elif k < density_count + 4:
                        fields[name]["unit"] = "reward_per_policy_interval"
                result.append(dict(scope=scope_name, lesson=LESSONS[lesson], original_goal=GOALS[goal],
                    payment_partition=PARTITIONS[part], policy_lane_intervals=count, lane_seconds=count * .02,
                    components=fields))
        return dict(schema="native-prefix-actual-payment-ledger-019-v1", status="native_payments_reconciled_preclose",
            complete=True, policy_steps=self.tick, event_producer_calls=self.event_calls,
            native_lane_intervals=self.n * self.tick,
            first_birth_lane_intervals=int(self.counts[1].sum()),
            first_birth_closed_lanes=int(self.first_closed.sum()),
            first_birth_censored_lanes=int((~self.first_closed).sum()),
            max_reconciliation_residual=float(self.max_residual),
            max_residual_policy_tick=int(self.max_residual_tick), max_residual_lane=int(self.max_residual_lane),
            density_components=list(self.term_names[:density_count]), rows=result,
            empty_bins="Absent rows mean zero samples, no performance estimate.",
            units="Original per-second densities multiply .02 once; discrete native events have no dt.",
            state_input_clock="Final actual plant50Hz; hold gates last200Hz feedback sample, not identical dq/gyro inputs.",
            mask_scope="Whole interval native actor mask including actor-caused terminal; final pure_rl is not eligibility.",
            first_birth_scope="Terminal inclusive then permanently closed; subsequent births separate in all-native.",
            actor_payment_limit="Immediate mask-eligible payment only; not advantage, entropy, optimizer credit or causal proof.",
            parent_report_admission_required=True, app_close_returned=None, PPO_updates=0, optimizer_steps=0)
