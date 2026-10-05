#!/usr/bin/env python3
"""Four fallen V6 first cohorts through the native physical FSM; zero PPO.

--preflight-only authenticates the V8 projection and full PT on CPU. It never
imports AppLauncher. A physical run is a separately requested diagnostic, not
qualification, a state-restoration experiment or a training continuation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from recovery_prefix_support import (
    BASELINE_PT_SHA, BASELINE_CONTRACT_SHA, CASES, CohortLedger, StateWriteGuard, core_sources, digest, materialize,
    read_json, require, validate_checkpoint_payload, write_json, effective_control_ownership,
)

WRITE_METHODS = ("write_root_link_pose_to_sim_index", "write_root_com_velocity_to_sim_index",
                 "write_joint_position_to_sim_index", "write_joint_velocity_to_sim_index")
GATES = ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed")
UNCOVERED = (
    "PhysX contact/solver/warm-start internal state; exact state restoration unsupported",
    "standalone script_requested and script_capped torque locals are not exported by native torque()",
    "raw legs/wheels before signal-delay are not re-decoded; actual motor_efforts inputs are captured",
    "training physical signal-delay condition is excluded by the frozen full role",
    "state-write interception covers the four native reset APIs, not arbitrary external PhysX view writes",
)


def sources():
    paths = [*sorted((ROOT / "src/wheeled_tasks/chassis").glob("*.py")),
             ROOT / "src/wheeled_tasks/v40/core.py", ROOT / "src/wheeled_tasks/agents/v40_ppo_cfg.py",
             ROOT / "scripts/train_chassis.py", ROOT / "scripts/probe_v6_recovery_runtime.py",
             ROOT / "scripts/probe_recovery_takeover.py", ROOT / "scripts/recovery_prefix_support.py",
             Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def checkpoint_authentication(path, config):
    import torch
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count
    source_path = checkpoint_contract_path(path)
    require(digest(path) == BASELINE_PT_SHA, "Frozen flat baseline PT SHA required")
    require(digest(source_path) == BASELINE_CONTRACT_SHA, "Frozen flat baseline contract SHA required")
    require(path.with_suffix(".checkpoint.json").exists(), "Immutable per-file checkpoint seal required")
    source = read_json(source_path)
    sha = digest(source_path)
    updates = checkpoint_update_count(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    clocks = validate_checkpoint_payload(saved, source, config, sha, updates)
    require(all(isinstance(v, torch.Tensor) and torch.isfinite(v).all()
                for v in saved["actor_state_dict"].values()), "Actor state contains invalid tensors")
    return {"checkpoint": str(path.resolve()), "checkpoint_sha256": digest(path),
            "checkpoint_contract": str(source_path.resolve()), "checkpoint_contract_sha256": sha,
            "checkpoint_learning_updates": updates, "checkpoint_clocks": clocks,
            "actor_only_deterministic_inference": True, "seal_and_infos_authenticated": True}


def json_value(value):
    """JSON previews explicitly use null for valid uninitialized history infinity."""
    if hasattr(value, "detach"):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class PrefixRecorder:
    """Read-only wrappers, including state capture before native auto-reset."""
    def __init__(self, env, output, seed):
        import torch
        from wheeled_tasks.chassis.recovery import RecoveryPhase
        self.torch, self.phase_names = torch, {int(p): p.name for p in RecoveryPhase}
        self.env, self.output = env, output
        self.guard = StateWriteGuard(env.num_envs)
        self.ledger = CohortLedger(env.scene_groups, seed)
        self.tick = self.physics_tick = 0
        self.completed_policy_intervals = 0
        self.in_policy = self.pending_physics = False
        self.outcome_tick = -1
        self.pending_terminals = set()
        self.snapshots, self.events, self.frames, self.policies = [], [], [], []
        self.missing = set()
        self.cache, self.observations, self.action = {}, {}, None
        self.blend_counts = Counter()
        self.last_mix = {}
        self.gate_counts = Counter()
        self.ownership_counts, self.release_evidence_counts = Counter(), Counter()
        self.originals = []
        self._install()

    def wrap(self, owner, name, replacement):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        setattr(owner, name, replacement(original))

    def ids(self, ids):
        if ids is None:
            return list(range(self.env.num_envs))
        return [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]

    def clone(self, value):
        if isinstance(value, self.torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, (bool, int, float, str, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [self.clone(v) for v in value]
        if isinstance(value, slice):
            return {"slice": [value.start, value.stop, value.step]}
        raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")

    def snapshot(self):
        env, r = self.env, self.env.recovery
        state = {}
        def collect(prefix, obj, fields):
            for field in fields.split():
                key = prefix + "." + field
                try:
                    value = getattr(obj, field)
                    if hasattr(value, "torch"):
                        value = value.torch
                    state[key] = self.clone(value)
                except (AttributeError, TypeError):
                    self.missing.add(key)
        collect("physics", env.robot.data,
                "joint_pos joint_vel root_link_pose_w root_link_lin_vel_w root_com_lin_vel_w "
                "root_com_ang_vel_w root_com_ang_vel_b projected_gravity_b applied_torque")
        for key in ("joint_pos", "joint_vel", "root_link_pose_w", "root_link_lin_vel_w", "root_com_ang_vel_b"):
            require("physics." + key in state and self.torch.isfinite(state["physics." + key]).all(),
                    f"Missing/nonfinite physical state: {key}")
        require(state["physics.joint_pos"].shape == (env.num_envs, 18), "Native full DOF shape changed")
        collect("env", env, "origins actions previous_actions before_previous_actions commands command_target "
                "command_acceleration torque motor_strength spring_strength contact_force contact_peak")
        collect("history", env.history, "buffer initialized last_tick")
        collect("v5", env.v5, "requested_motor_effort current_motor_bounds")
        collect("response", env.actuator_response, "enabled state alpha direct_axes has_lag dt")
        collect("recovery", r, "elapsed initial_release release_finished policy_motion_hold true_stable_seconds "
                "true_stable_peak followup_seconds motion_release_observed premature_release true_success "
                "learning_mask wheel_probe_torque observed_height observed_support reference_tick")
        collect("fsm", r.fsm, "phase route age_ticks phase_ticks ready_ticks stable_ticks stable_peak_ticks "
                "stable_break_count reroute_count failure_code script_targets motion_released stable_pass "
                "ready holding_rl unsupported_ticks orbit_angle continuous_pitch")
        collect("fsm", r.fsm, "plant_ticks continuous_q last_raw_q last_pitch pitch_initialized orbit_start "
                "thrust_anchor plant_pitch side_start side_direction")
        for name, value in r.fsm.stable_pass_masks.items():
            state["gate." + name] = self.clone(value)
        collect("encoders", r.encoders, "q last_raw winding initialized relative_limits")
        collect("imu", r.imu, "previous_velocity")
        collect("support_probe", r.probe, "previous_velocity previous_gyro previous_pulse evidence age "
                "bad_samples quiet_time lost_time")
        for module, fields in (
            ("command_transport", "enabled pending valid generated pending_tick held held_generated last_arrival "
             "age tick delay base_delay loss_probability burst_remaining sent lost can_phase"),
            ("usb_transport", "enabled current previous feedback previous_torque split_fraction rtt uplink downlink"),
            ("signal_delay", "enabled profile base_ms bounds_ms lags feedback strength physics_steps"),
        ):
            obj = getattr(env, module)
            state[module + ".present"] = obj is not None
            if obj is not None:
                collect(module, obj, fields)
                for generator in ("generator", "reset_generator"):
                    if hasattr(obj, generator):
                        state[module + "." + generator] = getattr(obj, generator).get_state().clone()
                if module == "signal_delay":
                    for buffer in ("sensor_buffer", "target_buffer"):
                        collect(module + "." + buffer, getattr(obj, buffer),
                                "_buffer _pointer _num_pushes _pending_resets")
        state["env.generator"] = env.generator.get_state().clone()
        state.update({"control." + key: self.clone(value) for key, value in self.cache.items()})
        state.update({"consumed_observation." + key: self.clone(value) for key, value in self.observations.items()})
        if self.action is not None:
            state["actor.requested_action"] = self.clone(self.action)
        state["closure_gap_m"] = self.clone(env.closure_gap())
        state["wheel_centers_world_m"] = self.clone(env.wheel_centers())
        return state

    def stamp(self):
        return {"policy_tick": self.tick, "physical_tick": self.physics_tick,
                "sim_time_s": self.physics_tick * self.env.dt,
                "reference_tick": self.env.recovery.reference_tick}

    def event(self, lane, name, *, snapshot=None, stamp=None, **metadata):
        stamp = dict(stamp or self.stamp())
        self.ledger.event(lane, name, stamp["sim_time_s"])
        index = len(self.snapshots)
        self.snapshots.append(snapshot if snapshot is not None else self.snapshot())
        self.events.append({"lane": lane, "case": self.env.scene_groups[lane], "birth_id": 1,
                            "event": name, **stamp, "snapshot_index": index, **json_value(metadata)})

    def _install(self):
        env, r = self.env, self.env.recovery
        for method in WRITE_METHODS:
            def build(original, method=method):
                def guarded(*args, **kwargs):
                    require("env_ids" in kwargs, f"Native write must identify lanes: {method}")
                    self.guard.write(method, self.ids(kwargs["env_ids"]))
                    return original(*args, **kwargs)
                return guarded
            self.wrap(env.robot, method, build)

        def reset_wrapper(original):
            def reset(ids):
                live = sorted(self.guard.live.intersection(self.ids(ids)))
                if not self.guard.birth_open and live:
                    require(self.in_policy and self.outcome_tick == self.tick,
                            "Unexpected reset of active first cohort outside native terminal path")
                    for lane in live:
                        self.event(lane, "terminal_pre_autoreset")
                    self.guard.close_terminal(live)
                    self.pending_terminals.update(live)
                return original(ids)
            return reset
        self.wrap(env, "reset", reset_wrapper)

        def reference_wrapper(original):
            def reference(q, dq, gravity, gyro):
                before_phase = r.fsm.phase.clone()
                before_release = r.release_finished.clone()
                result = original(q, dq, gravity, gyro)
                if self.in_policy:
                    for lane in sorted(self.guard.live):
                        old, new = int(before_phase[lane]), int(r.fsm.phase[lane])
                        if not bool(before_release[lane]) and bool(r.release_finished[lane]):
                            self.event(lane, "passive_release_finished")
                        if old != new:
                            self.event(lane, "phase_" + self.phase_names[new], phase_before=old, phase_after=new)
                            if old == 7 and new == 8:
                                self.event(lane, "blend_entry_pre_effort")
                        for name, value in r.fsm.stable_pass_masks.items():
                            self.gate_counts[(lane, new, name, "passed")] += bool(value[lane])
                            self.gate_counts[(lane, new, name, "ticks")] += 1
                        self.gate_counts[(lane, new, "all", "passed")] += bool(r.fsm.stable_pass[lane])
                        self.gate_counts[(lane, new, "all", "ticks")] += 1
                        if r.fsm.stable_peak_ticks[lane] * r.reference_dt >= 1. - 1e-5:
                            if "sensor_hold_dwell_complete" not in self.ledger.rows[lane]["events"]:
                                self.event(lane, "sensor_hold_dwell_complete")
                return result
            return reference
        self.wrap(r, "advance_reference", reference_wrapper)

        def effort_wrapper(original):
            def effort(q, dq, legs, wheels):
                result = original(q, dq, legs, wheels)
                if self.in_policy:
                    self.cache = {"pd_q": q.clone(), "pd_dq": dq.clone(), "pd_leg_targets": legs.clone(),
                                  "pd_wheel_targets": wheels.clone(), "rl_pd_torque": result.clone()}
                return result
            return effort
        self.wrap(env.v5, "motor_efforts", effort_wrapper)

        def mix_wrapper(original):
            def mix(q, dq, rl_torque):
                fraction, phase = r.fsm.blend.clone(), r.fsm.phase.clone()
                # torque() advances elapsed, so capture the drive enablement
                # actually used for this output before calling it.
                release = r.release_finished.clone()
                ownership = {"release_finished_used": release, "initial_release_s": r.initial_release.clone(),
                    "release_elapsed_s_before_torque": r.elapsed.clone(), "recovery_enabled_used": r.enabled.clone(),
                    "motor_enabled_used": (release & ~r.failed).clone(), "native_pure_rl_used": r.pure_rl.clone(),
                    "policy_motion_hold_used": r.policy_motion_hold.clone(), "fsm_motion_hold_used": r.fsm.motion_hold.clone()}
                result = original(q, dq, rl_torque)
                if self.in_policy:
                    self.cache.update(blend_fraction=fraction, phase_used=phase,
                                      post_native_mix_torque=result.clone(), **ownership,
                                      native_policy_learning_mask_after_torque=r.learning_mask.clone())
                return result
            return mix
        self.wrap(r, "torque", mix_wrapper)
        for name in ("command_transport", "usb_transport", "actuator_response"):
            obj = getattr(env, name)
            if obj is None:
                continue
            def transport_wrapper(original, name=name):
                def apply(command):
                    before = command.clone() if self.in_policy else None
                    result = original(command)
                    if self.in_policy:
                        self.cache[name + "_input"] = before
                        self.cache[name + "_output_before_reclip"] = result.clone()
                    return result
                return apply
            self.wrap(obj, "apply_torque", transport_wrapper)

        def step_wrapper(original):
            def step(*args, **kwargs):
                result = original(*args, **kwargs)
                if self.in_policy:
                    self.pending_physics = True
                return result
            return step
        self.wrap(env.sim, "step", step_wrapper)

        def update_wrapper(original):
            def update(*args, **kwargs):
                result = original(*args, **kwargs)
                if self.in_policy and self.pending_physics:
                    self.pending_physics = False
                    self.physics_tick += 1
                    self.after_physics()
                return result
            return update
        self.wrap(env.robot, "update", update_wrapper)

        def outcome_wrapper(original):
            def outcome(contact, undesired, gravity, omega, height, velocity, **kwargs):
                result = original(contact, undesired, gravity, omega, height, velocity, **kwargs)
                if self.in_policy:
                    self.outcome_tick = self.tick
                    self.cache.update(true_wheel_contacts=contact.clone(), true_undesired_contacts=undesired.clone(),
                                      outcome_gravity=gravity.clone(), outcome_omega=omega.clone(),
                                      outcome_height=height.clone(), outcome_velocity=velocity.clone())
                    for lane in sorted(self.guard.live):
                        for name, condition in (
                            ("truth_hold_dwell_complete", r.true_stable_peak[lane] >= 1. - 1e-5),
                            ("motion_unlock", r.motion_release_observed[lane]),
                            ("followup_success", r.true_success[lane]),
                        ):
                            if bool(condition) and name not in self.ledger.rows[lane]["events"]:
                                self.event(lane, name, premature_release=bool(r.premature_release[lane]))
                return result
            return outcome
        self.wrap(r, "outcome", outcome_wrapper)

    def close_birth(self):
        env = self.env
        table = read_json(ROOT / env.cfg["recovery_training"]["profiles_file"])
        require(env.recovery.pose_curriculum is None, "Fixed role must not sample train reset curriculum")
        require(set(env.scene_groups) == set(CASES), "Native lane layout lost a selected case")
        require(bool(env.recovery.enabled.all()), "Every selected lane must use recovery FSM")
        for lane, name in enumerate(env.scene_groups):
            case = env.cases[name]
            profile = table["profiles"][case["skill"]["recovery_pose"]]
            require(profile["geometry_valid"] and not profile["dynamic_recovery_validated"],
                    "Unexpected profile geometry/qualification status")
            expected = self.torch.tensor([profile["full_joint_positions"][n] for n in env.robot.joint_names],
                                         device=env.device)
            actual = env.robot.data.joint_pos.torch[lane]
            require(self.torch.allclose(actual, expected, atol=2e-5, rtol=0.), "Native birth q18 differs from bank")
            pose = env.robot.data.root_link_pose_w.torch[lane].clone()
            pose[:3] -= env.origins[lane]
            expected_root = self.torch.tensor(profile["root_position_m"] + profile["root_quaternion_xyzw"],
                                              device=env.device)
            require(self.torch.allclose(pose[[0, 1, 3, 4, 5, 6]], expected_root[[0, 1, 3, 4, 5, 6]],
                                       atol=2e-5, rtol=0.), "Native birth root differs from bank")
            jitter = float(pose[2] - expected_root[2])
            expected_jitter = float(env.fixed_reset_samples[lane, 2]) * .002
            require(abs(jitter - expected_jitter) <= 2e-5, "Native root-z seeded jitter changed")
            self.event(lane, "birth_complete", seeded_root_z_jitter_m=jitter,
                       geometry_valid=True, dynamic_recovery_validated=False)
        self.guard.close_birth()

    def before_policy(self, tick, observations, action):
        self.tick, self.in_policy = tick, True
        self.observations = {k: value.clone() for k, value in observations.items()}
        self.action = action.clone()

    def after_physics(self):
        env, stamp = self.env, self.stamp()
        if not self.guard.live:
            return
        require("phase_used" in self.cache, "Physical step lacked native control evidence")
        q, dq, pose = (env.robot.data.joint_pos.torch, env.robot.data.joint_vel.torch,
                       env.robot.data.root_link_pose_w.torch)
        require(all(self.torch.isfinite(v).all() for v in (q, dq, pose)), "Nonfinite physical trajectory")
        for lane in sorted(self.guard.live):
            phase, fraction = int(self.cache["phase_used"][lane]), float(self.cache["blend_fraction"][lane])
            flags = {name: bool(self.cache[name][lane]) for name in (
                "recovery_enabled_used", "release_finished_used", "motor_enabled_used",
                "native_pure_rl_used", "policy_motion_hold_used", "fsm_motion_hold_used",
                "native_policy_learning_mask_after_torque")}
            owner = effective_control_ownership(phase=phase, enabled=flags["recovery_enabled_used"],
                release_finished=flags["release_finished_used"], motor_enabled=flags["motor_enabled_used"],
                native_pure_rl=flags["native_pure_rl_used"], policy_motion_hold=flags["policy_motion_hold_used"])
            self.ownership_counts[(lane, owner)] += 1
            post_mix = self.cache["post_native_mix_torque"][lane]
            applied = env.robot.data.applied_torque.torch[lane, env.ids]
            self.frames.append({"lane": lane, "case": env.scene_groups[lane], **stamp,
                "phase_used": phase, "blend_fraction_used": fraction,
                "raw_fsm_phase_used": phase, "effective_control_ownership": owner, **flags,
                "initial_release_s": float(self.cache["initial_release_s"][lane]),
                "release_elapsed_s_before_torque": float(self.cache["release_elapsed_s_before_torque"][lane]),
                "q18": self.clone(q[lane]), "dq18": self.clone(dq[lane]), "root_pose_world_xyzw": self.clone(pose[lane]),
                "root_linear_velocity_world_m_s": self.clone(env.robot.data.root_link_lin_vel_w.torch[lane]),
                "root_angular_velocity_body_rad_s": self.clone(env.robot.data.root_com_ang_vel_b.torch[lane]),
                "post_mix_torque": self.clone(post_mix), "applied_torque6": self.clone(applied),
                "gates": {name: bool(value[lane]) for name, value in env.recovery.fsm.stable_pass_masks.items()},
                "ready_ticks": int(env.recovery.fsm.ready_ticks[lane]),
                "stable_ticks": int(env.recovery.fsm.stable_ticks[lane]),
                "script_targets6": self.clone(env.recovery.fsm.script_targets[lane])})
            if owner == "passive_release":
                require(post_mix.eq(0.).all() and not flags["native_policy_learning_mask_after_torque"],
                        "Native passive release lacks zero upstream effort or has actor learning credit")
                self.release_evidence_counts[(lane, "zero_upstream_effort_ticks")] += 1
                self.release_evidence_counts[(lane, "zero_applied_effort_ticks")] += bool(applied.eq(0.).all())
                # Applied effort is observed, not universally forced to zero:
                # downstream transport/response can retain earlier drive state.
                if "passive_zero_effort_post_physics" not in self.ledger.rows[lane]["events"]:
                    self.event(lane, "passive_zero_effort_post_physics", raw_fsm_phase=phase,
                               upstream_zero=True, applied_zero=bool(applied.eq(0.).all()))
            if owner == "script_actor_blend":
                require(0. <= fraction < 1., "Native BLEND fraction differs from physical control evidence")
                self.blend_counts[lane] += 1
                snapshot = self.snapshot()
                self.last_mix[lane] = (snapshot, dict(stamp))
                if self.blend_counts[lane] == 1:
                    self.event(lane, "blend_first_applied_post_physics", snapshot=snapshot)
            if owner in ("rl_hold", "rl_unlocked") and "first_pure_rl_post_physics" not in self.ledger.rows[lane]["events"]:
                require(fraction == 1. and self.blend_counts[lane] == 40 and lane in self.last_mix,
                        "Pure RL reached without full native 200ms physical BLEND")
                snapshot, old_stamp = self.last_mix.pop(lane)
                self.event(lane, "blend_last_mixed_post_physics", snapshot=snapshot, stamp=old_stamp)
                self.event(lane, "first_pure_rl_post_physics",
                           ownership="upstream native torque mix; transport/response retained history remains")

    def after_policy(self, reward, done, extras, interval):
        data = extras["diagnostics"]
        require(interval.torque_calls == 4, "Interval lacked four native torque calls")
        require(self.torch.equal(extras["policy_learning_mask"], interval.expected), "Native interval ownership mask differs")
        native_terminals = {int(i) for i in done.nonzero().flatten().tolist()}
        first_terminals = {i for i in native_terminals if self.ledger.rows[i]["partition"] is None}
        require(first_terminals == self.pending_terminals, "Terminal/reset first-cohort accounting mismatch")
        for lane in sorted(first_terminals):
            reasons = {k: bool(v[lane]) for k, v in data["reasons"].items()}
            success, terminated = bool(data["success"][lane]), bool(data["terminated"][lane])
            boundary = reasons.get("boundary", False)
            timeout = bool(extras["time_outs"][lane])
            if success:
                require(bool(data["recovery_true_success"][lane]), "Native success lacks recovery outcome")
                needed = {"blend_last_mixed_post_physics", "first_pure_rl_post_physics", "truth_hold_dwell_complete",
                          "motion_unlock", "followup_success"}
                require(needed.issubset(self.ledger.rows[lane]["events"]), "Success lacks physical chain witnesses")
            self.ledger.finish(lane, success=success, terminated=terminated, boundary=boundary,
                               timeout=timeout,
                               time_s=self.physics_tick * self.env.dt, reasons=reasons)
        self.pending_terminals.clear()
        counted = sorted(self.guard.live | first_terminals)
        for lane in counted:
            self.policies.append({"lane": lane, "case": self.env.scene_groups[lane], **self.stamp(),
                "reward": float(reward[lane]), "learning_mask": bool(extras["policy_learning_mask"][lane]),
                "native_recovery": {k: json_value(v[lane]) for k, v in data.items()
                                    if k.startswith("recovery_") and isinstance(v, self.torch.Tensor)
                                    and v.ndim and v.shape[0] == self.env.num_envs},
                "recovery_costs": {k: float(v[lane]) for k, v in extras.get("native_reward_components", {}).items()
                                   if k.startswith("recovery_takeover_")}})
        self.in_policy = False
        self.completed_policy_intervals = self.tick

    def summary(self):
        result = self.ledger.summary(censor_time_s=self.physics_tick * self.env.dt)
        result.update(actual_blend_physical_substeps={str(k): v for k, v in self.blend_counts.items()},
                      terminal_pending_lanes=sorted(self.pending_terminals),
                      watched_state_write_calls=[{"method": k[0], "scope": k[1], "calls": v}
                                                 for k, v in sorted(self.guard.calls.items())],
                      snapshot_field_missing=sorted(self.missing), uncovered_fields=list(UNCOVERED),
                      native_sensor_gate_ticks=[{"lane": k[0], "phase": self.phase_names[k[1]], "gate": k[2],
                                                "statistic": k[3], "value": v} for k, v in sorted(self.gate_counts.items())],
                      native_control_ownership_ticks=[{"lane": k[0], "ownership": k[1], "ticks": v}
                                                      for k, v in sorted(self.ownership_counts.items())],
                      initial_release_effort_evidence=[{"lane": k[0], "statistic": k[1], "ticks": v}
                                                       for k, v in sorted(self.release_evidence_counts.items())],
                      physical_frame_count=len(self.frames), policy_frame_count=len(self.policies),
                      event_count=len(self.events), all_first_cohorts_closed=not self.guard.live)
        return result

    def save(self):
        for name, rows in (("events.jsonl", self.events), ("physical_frames.jsonl", self.frames),
                           ("policy_frames.jsonl", self.policies)):
            with (self.output / name).open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(json_value(row), allow_nan=False) + "\n")
        self.torch.save({"schema": "v6-prefix-event-snapshots-v1", "qualification": False,
                         "physx_internal_state_restore_claim": False, "snapshots": self.snapshots},
                        self.output / "event_snapshots.pt")
        write_json(self.output / "cohort_summary.json", self.summary())

    def restore_wrappers(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True, help="Explicit frozen V8 worker")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--source-identity", type=Path, help="Frozen external H/C identity manifest; no overlay")
    parser.add_argument("--source-identity-sha256", help="Independent SHA of the external identity manifest")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "v6-native-recovery-physical-prefix-v1", "status": "starting",
              "scope": "engineering_diagnostic_only_no_PPO_no_qualification", "qualification": False,
              "formal_updates_consumed": 0, "optimizer_steps": 0, "execution_status": "not_run",
              "physx_internal_state_restore_claim": False, "requested_episode_seconds": 20.,
              "policy_steps": 0, "policy_intervals_started": 0, "physics_steps": 0, "elapsed_physics_s": 0.,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "case_names": list(CASES),
              "source_files_sha256": sources(), "randomization_or_delay_overrides": []}
    path, launcher, env, recorder = args.output / "report.json", None, None, None
    provenance = actor = before = None
    write_json(path, report)
    try:
        source_options = {"identity_path": args.source_identity, "identity_sha256": args.source_identity_sha256,
                          "actual_sources": report["source_files_sha256"]}
        report["frozen_physical_control_sources"] = core_sources(ROOT, **source_options)
        cfg, manifest, control, stage, seed, count, receipt = materialize(ROOT, args.contract, args.output, 1)
        if args.source_identity is not None:
            require(digest(args.output / "engineering_expectations.json") ==
                    report["frozen_physical_control_sources"]["engineering_expectations_sha256"],
                    "Actual independent engineering expectations differ from frozen A/B input")
        require(count == 4, "This minimal entry requires exactly four first cohorts")
        provenance = checkpoint_authentication(args.checkpoint, cfg)
        report.update(design_preflight=receipt, checkpoint=provenance, num_envs=count, seed=seed,
                      run_role="full", stage=stage, contract_sha256=digest(args.output / "effective_contract.json"),
                      source_worker_sha256=digest(args.contract), signal_delay_enabled=False,
                      command_transport_enabled=True, uncovered_fields=list(UNCOVERED))
        if args.preflight_only:
            report["status"] = "preflight_passed_scene_not_run"
        else:
            write_json(path, report)
            os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
            from isaaclab.app import AppLauncher
            launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": "cuda:0",
                "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
            import torch
            from wheeled_tasks.chassis.eval_env import FixedCaseEnv
            from probe_recovery_takeover import IntervalAudit, load_actor
            require(sources() == report["source_files_sha256"], "Source changed before native scene import")
            require(core_sources(ROOT, **source_options) == report["frozen_physical_control_sources"],
                    "Frozen external source identity changed before native scene import")
            torch.set_num_threads(4)
            torch.manual_seed(seed)
            env = FixedCaseEnv(cfg, manifest, control, ROOT, stage_name=stage, num_envs=count,
                               device="cuda:0", level=1., seed=seed, run_role="full")
            require(env.recovery is not None and env.recovery.has_tasks, "Native recovery FSM absent")
            require(env.startup_report.get("gpu_dynamics_enabled") is True, "Native GPU physics required")
            require(env.startup_report.get("runtime_preflight", {}).get("status") == "runtime_configuration_passed",
                    "Native runtime independent design comparison failed")
            require(env.signal_delay is None and env.command_transport is not None, "Actual delay modules differ from full role")
            interval = IntervalAudit(env.recovery)
            recorder = PrefixRecorder(env, args.output, seed)
            actor, before = load_actor(env, args.checkpoint, cfg, provenance, seed)
            observations = env.reset_suite()
            recorder.close_birth()
            joint_types = {j.attrib["name"]: j.attrib["type"] for j in
                           ET.parse(ROOT / cfg["asset_directory"] / manifest["urdf"]).getroot().findall("joint")}
            report.update(status="running", execution_status="running", startup=deepcopy(env.startup_report),
                native_joint_names=env.robot.joint_names,
                joint_position_units=["m" if joint_types[n] == "prismatic" else "rad" for n in env.robot.joint_names],
                actor_weights_loaded_strictly=True)
            write_json(path, report)
            started = time.monotonic()
            for tick in range(1, 1001):
                require(launcher.app.is_running(), "Simulator exited before bounded diagnostic completion")
                with torch.inference_mode():
                    action = actor(observations["policy"])
                    require(tuple(action.shape) == (count, 6) and torch.isfinite(action).all(), "Invalid actor output")
                    recorder.before_policy(tick, observations, action)
                    observations, reward, done, extras = env.step(action)
                    recorder.after_policy(reward, done, extras, interval)
                require(tuple(observations["policy"].shape) == (count, 35)
                        and tuple(observations["critic"].shape) == (count, 81), "Native observation ABI changed")
                require(all(torch.isfinite(v).all() for v in (*observations.values(), reward)), "Nonfinite native policy data")
                timing = env.control_audit.last
                require(timing["pd_updates"] == timing["feedback_refreshes"] == 4
                        and timing["target_decodes"] == 1 and timing["physics_steps"] == tick * 4
                        and recorder.physics_tick == tick * 4, "Native 200/50 Hz control clock changed")
                if tick == 1 or tick % 50 == 0:
                    report.update(policy_steps=tick, elapsed_physics_s=tick * .02,
                                  wall_seconds=time.monotonic() - started, summary=recorder.summary())
                    write_json(path, report)
                if not recorder.guard.live:
                    break
            require(all(torch.equal(v.detach().cpu(), before[n]) for n, v in actor.state_dict().items()), "Actor tensors changed")
            require(digest(args.checkpoint) == provenance["checkpoint_sha256"], "Source PT changed during inference")
            report.update(status="engineering_probe_completed", execution_status="completed", policy_steps=tick,
                elapsed_physics_s=tick * .02, wall_seconds=time.monotonic() - started,
                actor_tensors_unchanged=True, checkpoint_unchanged=True, control_timing=env.control_audit.last,
                summary=recorder.summary())
    except Exception:
        report.update(status="failed", execution_status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        after_sources = sources()
        changed = {k for k in set(report["source_files_sha256"]) | set(after_sources)
                   if after_sources.get(k) != report["source_files_sha256"].get(k)}
        report.update(source_files_changed=sorted(changed), finished_at_utc=datetime.now(timezone.utc).isoformat())
        if changed:
            report.update(status="failed_source_changed", execution_status="failed")
        if "frozen_physical_control_sources" in report:
            try:
                source_options["actual_sources"] = after_sources
                require(core_sources(ROOT, **source_options) == report["frozen_physical_control_sources"],
                        "Frozen source identity changed during physical diagnostic")
                report["source_identity_final_verified"] = True
            except Exception:
                report.update(status="failed_source_identity", execution_status="failed",
                              source_identity_final_verified=False, source_identity_error=traceback.format_exc())
        # A diagnostic failure must still authenticate unchanged input weights.
        if provenance is not None:
            try:
                report["checkpoint_unchanged"] = digest(args.checkpoint) == provenance["checkpoint_sha256"]
                require(report["checkpoint_unchanged"], "Source PT changed during diagnostic")
                if actor is not None and before is not None:
                    report["actor_tensors_unchanged"] = all(torch.equal(v.detach().cpu(), before[n])
                                                           for n, v in actor.state_dict().items())
                    require(report["actor_tensors_unchanged"], "Actor tensors changed during diagnostic")
            except Exception:
                report.update(status="failed_input_changed", execution_status="failed", input_integrity_error=traceback.format_exc())
        if recorder is not None:
            recorder.in_policy = False
            report.update(policy_steps=recorder.completed_policy_intervals, policy_intervals_started=recorder.tick,
                          physics_steps=recorder.physics_tick, elapsed_physics_s=recorder.physics_tick * env.dt)
            try:
                recorder.save()
                report["summary"] = recorder.summary()
            except Exception:
                report.update(status="failed_evidence_write", execution_status="failed", evidence_write_error=traceback.format_exc())
            finally:
                recorder.restore_wrappers()
        write_json(path, report)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    print(json.dumps({"status": report["status"], "report": str(path), "formal_updates_consumed": 0}))
    return 0 if report["status"] in ("engineering_probe_completed", "preflight_passed_scene_not_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
