"""Inference and paid-episode execution for an already constructed flat env.

This module neither launches Kit nor grants mechanical/host admission. Its
stream adapter is also tested with explicit CPU physical-data doubles. A public
native worker must pass the independent admission gate before construction.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from wheeled_tasks.chassis.application.evaluation.flat_evaluation_runtime import EvaluationActor, PreResetLedgerAdapter, actor_state_digest


VERSION = "flat-evaluation-stream-v1"
RESET_VERSION = "flat-evaluation-reset-snapshot-v1"
RESET_DIAGNOSTICS = (
    "ground", "holding", "birth_proposal_count", "birth_rejection_count",
    "placement_raise_m", "minimum_shape_clearance_m", "birth_source_angle_rad", "birth_own_angle_rad",
)


class _CaseTrackingMeasurements:
    """Stage pre-reset readings, then commit only ledger-accepted batches."""

    def __init__(self, lane_cases, policy_dt, *, required,
                 height_reference="root_link_z_minus_env_origin_z"):
        self.lane_cases, self.policy_dt, self.required = tuple(lane_cases), policy_dt, required
        self.height_reference = height_reference
        self.cases = {name: dict(accepted_frames=0, valid_frames=0, invalid_frames=0,
            unavailable_frames=0, missing_fields=set(), height_abs=0., height_sq=0.,
            vx_abs=0., yaw_abs=0., height_error=0., vx_error=0., yaw_error=0.,
            height_observed=0., vx_observed=0., yaw_observed=0., gap_max=None)
            for name in dict.fromkeys(lane_cases)}

    def prepare(self, diagnostics, active_lanes):
        fields = {"height": ((), "float"), "gap": ((), "float"),
            "measurement_valid_mask": ((), "bool"), "effective_command": ((3,), "float"),
            "root_COM_velocity_body": ((3,), "float"), "angular_velocity_body": ((3,), "float")}
        missing = sorted(set(fields) - set(diagnostics))
        if missing:
            if self.required:
                raise ValueError("Formal evaluation tracking diagnostics missing: " + ", ".join(missing))
            return [(self.lane_cases[lane], None, missing) for lane in active_lanes]
        n = len(self.lane_cases)
        values = {name: PreResetLedgerAdapter._values(diagnostics[name], n, tail, name, kind=kind)
                  for name, (tail, kind) in fields.items()}
        pending = []
        for lane in active_lanes:
            reading = None
            if values["measurement_valid_mask"][lane]:
                command = values["effective_command"][lane]
                reading = (values["height"][lane] - command[2],
                    values["root_COM_velocity_body"][lane][0] - command[0],
                    values["angular_velocity_body"][lane][2] - command[1], values["gap"][lane],
                    values["height"][lane], values["root_COM_velocity_body"][lane][0],
                    values["angular_velocity_body"][lane][2])
                if not all(math.isfinite(value) for value in reading) or reading[3] < 0.:
                    raise ValueError("Valid evaluation tracking diagnostics must be finite with nonnegative gap")
            pending.append((self.lane_cases[lane], reading, []))
        return pending

    def accept(self, pending):
        for name, reading, missing in pending:
            case = self.cases[name]
            case["accepted_frames"] += 1
            if missing:
                case["unavailable_frames"] += 1
                case["missing_fields"].update(missing)
            elif reading is None:
                case["invalid_frames"] += 1
            else:
                height, vx, yaw, gap, observed_height, observed_vx, observed_yaw = reading
                case["valid_frames"] += 1
                case["height_abs"] += abs(height)
                case["height_sq"] += height * height
                case["vx_abs"] += abs(vx)
                case["yaw_abs"] += abs(yaw)
                case["height_error"] += height
                case["vx_error"] += vx
                case["yaw_error"] += yaw
                case["height_observed"] += observed_height
                case["vx_observed"] += observed_vx
                case["yaw_observed"] += observed_yaw
                case["gap_max"] = gap if case["gap_max"] is None else max(case["gap_max"], gap)

    def report(self):
        cases = {}
        for name, case in self.cases.items():
            n = case["valid_frames"]
            cases[name] = {"status": "collected" if n else "no_valid_frames" if case["invalid_frames"] else "not_collected",
                "accepted_frames": case["accepted_frames"], "valid_frames": n,
                "invalid_measurement_frames": case["invalid_frames"],
                "unavailable_diagnostic_frames": case["unavailable_frames"],
                "missing_diagnostic_fields": sorted(case["missing_fields"]),
                "valid_lane_seconds": n * self.policy_dt,
                "root_link_height_mae_m": case["height_abs"] / n if n else None,
                "root_link_height_rmse_m": math.sqrt(case["height_sq"] / n) if n else None,
                "root_com_body_vx_tracking_mae_m_s": case["vx_abs"] / n if n else None,
                "body_omega_z_tracking_mae_rad_s": case["yaw_abs"] / n if n else None,
                "root_link_height_mean_m": case["height_observed"] / n if n else None,
                "root_link_height_signed_error_mean_m": case["height_error"] / n if n else None,
                "root_com_body_vx_mean_m_s": case["vx_observed"] / n if n else None,
                "root_com_body_vx_signed_error_mean_m_s": case["vx_error"] / n if n else None,
                "body_omega_z_mean_rad_s": case["yaw_observed"] / n if n else None,
                "body_omega_z_signed_error_mean_rad_s": case["yaw_error"] / n if n else None,
                "closure_gap_max_m": case["gap_max"]}
        return {"aggregation": "valid pre-reset policy frames pooled by case, including terminal frames; no learning-mask exclusion",
            "quota_scope": "only lanes with an open requested episode before ledger acceptance",
            "height_reference": self.height_reference + "; error against effective_command[2]",
            "velocity_reference": "root rigid-body COM velocity expressed in root-link body frame",
            "yaw_reference": "body angular_velocity_body[2]; error against effective_command[1]",
            "signed_error_convention": "observed minus effective command; means use the same valid frames as MAE",
            "closure_gap_scope": "maximum diagnostics.gap, already the four 5ms physics-substep peak",
            "by_case": cases}


class EpisodeRecordWriter:
    """Exclusive JSONL artifact; acknowledge a batch only after file fsync."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.stream = self.path.open("xb")
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.digest, self.count, self.byte_count = hashlib.sha256(), 0, 0

    def __call__(self, records):
        raw = b"".join((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode() for record in records)
        self.stream.write(raw)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.digest.update(raw)
        self.count += len(records)
        self.byte_count += len(raw)

    def close(self):
        self.stream.close()

    def receipt(self):
        raw = self.path.read_bytes()
        return {"path": str(self.path), "acknowledged_records": self.count,
                "acknowledged_bytes": self.byte_count, "acknowledged_sha256": self.digest.hexdigest(),
                "current_file_sha256": hashlib.sha256(raw).hexdigest(),
                "current_file_matches_acknowledged_bytes": len(raw) == self.byte_count and hashlib.sha256(raw).hexdigest() == self.digest.hexdigest()}


def reset_snapshot(env, reset_ids):
    """Copy returned native fields without resetting or advancing any clock.

    Only newly reset lanes get metadata rows. All-lane poses/identities remain
    tensors so the paid-state adapter can validate its returned batch. Lazy SDK
    fields must already represent the returned state; no fake dt refresh here.
    """
    if getattr(env, "flat_episode", None) is None:
        raise ValueError("Reset snapshot requires the source flat episode owner")
    n, flat = env.num_envs, env.flat_episode
    ids = reset_ids
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.int64
            or ids.device != flat.device or ids.unique().numel() != ids.numel()
            or bool(((ids < 0) | (ids >= n)).any())):
        raise ValueError("Reset IDs must be unique int64 lanes on the episode device")
    if bool((env.episode_length_buf[ids] != 0).any()):
        raise ValueError("Reset metadata can only be captured at episode tick zero")
    data = env.robot.data
    pose = data.root_link_pose_w.torch.detach().clone()
    yaw = env._flat_heading().detach().clone()
    q, dq = data.joint_pos.torch, data.joint_vel.torch
    velocity = torch.cat((data.root_com_lin_vel_w.torch, data.root_com_ang_vel_w.torch), -1)
    joint_names = list(env.robot.joint_names)
    if len(joint_names) != 18 or len(set(joint_names)) != 18:
        raise ValueError("Reset snapshot requires the full named18 native joint order")
    if pose.shape != (n, 7) or yaw.shape != (n,) or q.shape != (n, 18) or dq.shape != (n, 18) or velocity.shape != (n, 6):
        raise ValueError("Reset snapshot physical field shape mismatch")
    if any(not bool(torch.isfinite(value[ids]).all()) for value in (pose, yaw, q, dq, velocity)):
        raise ValueError("Newborn physical state must be finite")
    owned = flat.diagnostics()
    selected = {}
    for name in RESET_DIAGNOSTICS:
        value = owned[name]
        if not isinstance(value, torch.Tensor) or value.shape[0] != n:
            raise ValueError(f"Reset diagnostic shape mismatch: {name}")
        if value.is_floating_point() and not bool(torch.isfinite(value[ids]).all()):
            raise ValueError(f"Newborn reset diagnostic must be finite: {name}")
        selected[name] = value[ids].detach().cpu().tolist()
    rows = [None] * n
    physical = {name: value[ids].detach().cpu().tolist() for name, value in
                (("root_link_pose_world_xyzw", pose), ("root_COM_velocity_world6", velocity),
                 ("joint_position_rad_or_m", q), ("joint_velocity_rad_s_or_m_s", dq))}
    for row, lane in enumerate(ids.detach().cpu().tolist()):
        rows[lane] = {"schema": RESET_VERSION, "episode_id": int(flat.episode_id[lane]),
            "physical_tick": env.physical_tick, "episode_tick": 0, "joint_names_native_order": joint_names[:],
            **{name: value[row] for name, value in physical.items()},
            "source_birth": {name: value[row] for name, value in selected.items()},
            "scope": "fresh returned reset state, not a PhysX solver restart"}
    # Reject even accidental non-JSON scalar data before the ledger mutates.
    json.dumps(rows, allow_nan=False)
    return {"episode_id": flat.episode_id.detach().clone(),
            "episode_tick": env.episode_length_buf.detach().clone(),
            "root_position_world": pose[:, :3], "yaw_rad": yaw, "reset_metadata": rows}


class FlatEvaluationLoop:
    """Execute frozen actor inference and retain paid frames through resets.

    The caller owns environment/Kit cleanup and a real wall/GPU limit owner.
    This class performs no PPO, no policy update and no claim of native runtime
    admission. Every actual step is preceded by the external limit check.
    """
    def __init__(self, inputs, *, device, limit_check):
        inputs.authenticate()
        if not callable(limit_check):
            raise TypeError("Evaluation requires an external limit owner")
        # No supplied model, same-mode surrogate actor or caller RNG may replace
        # the model loaded directly from these captured checkpoint bytes.
        actor = EvaluationActor.from_prepared(inputs, device=device)
        self.inputs, self.actor, self.limit_check = inputs, actor, limit_check
        self.expected_state_sha256 = actor_state_digest(inputs.load_checkpoint()["actor_state_dict"])
        self.adapter = PreResetLedgerAdapter(inputs.protocol, inputs.overlay["lane_case_ids"],
            reward_profile=inputs.evaluation_config().get("reward_profile", "scut0719_flat_v1"))
        self.tracking_measurements = _CaseTrackingMeasurements(inputs.overlay["lane_case_ids"],
            inputs.protocol["policy_dt_s"], required=bool(inputs.evaluation_config().get("formal_v7_flat")),
            height_reference=("root_link_world_z_minus_base_ground_scan" if inputs.is_rough
                              else "root_link_z_minus_env_origin_z"))
        self.rough_measurements = None
        self.recovery_measurements = None
        self.steps, self.status, self.error = 0, "not_started", None
        self.attempted_steps, self.physical_tick_readback = 0, None
        self.accepted_steps, self.accepted_lane_transitions, self.writer_acknowledged_episodes = 0, 0, 0
        self.last_returned, self.failure_returned_tensor_bindings = None, None
        self.failure_diagnostic_error = None
        from wheeled_tasks.chassis.application.evaluation.scripted_recovery_evaluation import WINDOW_SCHEMA
        self.max_steps = ((1400 if inputs.protocol["schema"] == WINDOW_SCHEMA else 999)
            * inputs.protocol["episodes_per_lane"])
        self.snapshots, self.step_records = [], []
        self.environment_observations = 0
        self.numerical_capture_event = None
        self.capture_statistics = None

    def _stop_on_captured_numerical_event(self, extras, physical_tick, capture_summary):
        capturing = getattr(self.inputs, "numerical_capture", False)
        if not (capturing or getattr(self.inputs, "is_small_rough_regression", False)):
            return
        reasons = extras["diagnostics"]["reasons"]["numerical_safety"]
        if (not isinstance(reasons, torch.Tensor) or reasons.dtype != torch.bool
                or reasons.shape != (self.inputs.overlay["num_envs"],)):
            raise ValueError("Numerical capture requires the existing per-lane numerical safety reason")
        source_guard = bool(reasons.any())
        if source_guard or capturing and capture_summary.get("recorded") is True:
            self.numerical_capture_event = {"policy_interval_id": self.steps,
                "physical_tick_after_return": physical_tick,
                "reason": "source_policy_end_numerical_safety" if source_guard else "first_physical_numerical_observation",
                "source_policy_end_numerical_lanes": reasons.nonzero(as_tuple=False).flatten().cpu().tolist(),
                "capture": deepcopy(capture_summary),
                "scope": "diagnostic stops after its first event; original policy-end guard is reported separately"}
            raise FloatingPointError("Formal V7 native numerical safety termination" if source_guard else
                "Diagnostic stopped after first physical numerical observation; source policy-end guard is false")

    def _validate_env(self, env):
        n, c, auth = self.inputs.overlay["num_envs"], self.inputs.evaluation_config(), self.inputs.authentication
        if env.num_envs != n or torch.device(env.device) != self.actor.device:
            raise ValueError("Evaluation lane/device differs from the prepared actor")
        invocation_fields = {"stage", "num_envs", "terrain_level", "seed"}
        if set(env.cfg) != set(c) | invocation_fields or any(env.cfg.get(key) != value for key, value in c.items()):
            raise ValueError("Effective environment config differs from the frozen evaluation overlay")
        if (env.cfg.get("seed") != self.inputs.protocol["environment_seed"]
                or env.cfg.get("stage") != self.inputs.execution_stage or env.cfg.get("num_envs") != n
                or env.cfg.get("terrain_level") != 0. or env.physical_tick != 0
                or env.dt != .005 or env.policy_dt != .02 or env.decimation != 4):
            raise ValueError("Evaluation must start at the source200/50Hz first reset and declared seed/stage")
        clocks = auth["actor_clocks"]
        if env.stage_actor_update != clocks["stage_actor_update"] or env.global_actor_update != clocks["global_actor_update"]:
            raise ValueError("Evaluation must retain the authenticated actor clocks")
        receipt = env.startup_report.get("flat_evaluation_setup", {})
        expected = self.inputs.build_setup()
        if (receipt.get("protocol_sha256") != hashlib.sha256(expected._protocol_json.encode()).hexdigest()
                or receipt.get("lane_case_ids") != list(expected.lane_case_ids)
                or receipt.get("environment_rng_reseeded_after_startup") is not False):
            raise ValueError("Actual first-reset typed setup receipt is missing or differs")
        if self.inputs.is_scripted_recovery:
            from wheeled_tasks.chassis.application.evaluation.scripted_recovery_evaluation import actual_lane_cases, RecoveryMeasurements
            lanes = actual_lane_cases(expected, env)
            if receipt.get("actual_lane_case_ids") != list(lanes) or receipt.get("run_role") != self.inputs.run_role:
                raise ValueError("Actual recovery lane permutation/role receipt differs")
            self.adapter = PreResetLedgerAdapter(self.inputs.protocol, lanes,
                reward_profile=self.inputs.evaluation_config().get("reward_profile", "scut0719_flat_v1"))
            self.tracking_measurements = _CaseTrackingMeasurements(lanes, self.inputs.protocol["policy_dt_s"],
                required=True, height_reference="root_link_z_minus_env_origin_z")
            self.recovery_measurements = RecoveryMeasurements(self.inputs.protocol, lanes, env.recovery.source_identity())
        if self.inputs.is_plane_regression:
            terrain = env.startup_report.get("formal_terrain", {})
            if (receipt.get("evaluation_surface") != "plane_regression_v1"
                    or receipt.get("run_role") != "source_rough_plane_regression"
                    or terrain.get("type") != "plane" or terrain.get("xy_boundary_enabled") is not False):
                raise ValueError("Rough plane regression must read back its explicit infinite-plane overlay")
        if getattr(env, "flat_episode", None) is None or env.flat_episode.commands.physics_steps != 0:
            raise ValueError("Evaluation requires a fresh source episode owner")
        if bool((env.episode_length_buf != 0).any()) or bool((env.flat_episode.episode_id != 0).any()):
            raise ValueError("Initial ledger must precede the first paid episode step")

    def _verify_actor(self):
        if (self.actor.state_digest() != self.expected_state_sha256
                or self.actor.captured_checkpoint_sha256 != self.inputs.inputs["checkpoint"].digest
                or self.actor.mode != self.inputs.protocol["actor_mode"]
                or self.actor.seed != self.inputs.protocol["actor_seed"]):
            raise ValueError("Actual actor state differs from the captured checkpoint stream")

    def _run_capture(self, env):
        """Event diagnosis only: keep per-lane counters on device, no ledger copy.

        Actor sampling, env.step, resets and the episode-attempt quota are the
        same as the full evaluator. This path does not produce reward/skill
        assessments; pre-event physical evidence is captured inside env.step.
        """
        threshold = env.startup_report.get("numerical_failure_capture", {}).get("closure", {}).get("threshold_m")
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Compact capture requires the active closure capture threshold")
        n, device = env.num_envs, self.actor.device
        attempts = torch.zeros(n, dtype=torch.long, device=device)
        streak = torch.zeros_like(attempts)
        # Count, gap sum, maximum, above-threshold intervals, excursions,
        # longest run of consecutive above-threshold policy intervals.
        sums = torch.zeros(6, dtype=torch.float64, device=device)
        self.capture_statistics = dict(threshold_m=threshold, attempts=attempts, sums=sums)
        obs = env.get_observations()
        self.environment_observations += 1
        self.status = "running"
        clocks = (env.stage_actor_update, env.global_actor_update)
        quota = self.inputs.protocol["episodes_per_lane"]
        while bool((attempts < quota).any()):
            if self.steps >= self.max_steps:
                raise RuntimeError("Source999-tick episode bound exhausted before every lane quota")
            self.limit_check()
            before = env.physical_tick
            action = self.actor.step(obs["policy"])
            self.attempted_steps += 1
            obs, reward, done, extras = env.step(action)
            self.steps += 1
            self.last_returned = (reward, done, extras)
            if env.physical_tick != before + 4 or (env.stage_actor_update, env.global_actor_update) != clocks:
                raise RuntimeError("Evaluation changed physics decimation or actor update clocks")
            self._stop_on_captured_numerical_event(extras, env.physical_tick,
                env.startup_report.get("numerical_failure_capture", {}))
            gap = extras["diagnostics"]["gap"]
            if (not isinstance(gap, torch.Tensor) or gap.shape != (n,) or gap.device != device
                    or not gap.is_floating_point() or not isinstance(done, torch.Tensor)
                    or done.shape != (n,) or done.dtype != torch.bool or done.device != device):
                raise ValueError("Compact capture needs per-lane gap and done tensors")
            active = attempts < quota
            if bool((active & (~torch.isfinite(gap) | (gap < 0))).any()):
                raise FloatingPointError("Invalid closure gap in compact capture")
            selected = torch.where(active, gap.double(), 0.)
            above = active & (gap > threshold)
            sums[0] += active.sum()
            sums[1] += selected.sum()
            sums[2] = torch.maximum(sums[2], selected.max())
            sums[3] += above.sum()
            sums[4] += (above & (streak == 0)).sum()
            streak = torch.where(above, streak + 1, 0)
            sums[5] = torch.maximum(sums[5], streak.max())
            # Do not merge separate episodes into one excursion.
            streak = torch.where(done, 0, streak)
            attempts.add_((active & done).long())
        self.limit_check()
        self.inputs.assert_unchanged()
        self._verify_actor()
        self.status = "attempt_quota_completed"
        self.physical_tick_readback = env.physical_tick
        return self.report()

    def run(self, env, *, commit_records):
        """One run; durable row writer must acknowledge every completed record.

        A failure is retained as a partial/censored report and re-raised. Quota
        means attempted episodes, including failures and local boundary exits;
        it is never renamed a skill-success quota.
        """
        if self.status != "not_started" or not callable(commit_records):
            raise ValueError("An evaluation loop is single-use and needs a durable record writer")
        try:
            self.limit_check()
            self.inputs.assert_unchanged()
            self._verify_actor()
            self._validate_env(env)
            if getattr(self.inputs, "numerical_capture", False):
                return self._run_capture(env)
            if self.inputs.is_rough and not self.inputs.is_plane_regression:
                from wheeled_tasks.chassis.application.evaluation.rough_evaluation import RoughTerrainMeasurements
                self.rough_measurements = RoughTerrainMeasurements.from_env(env,
                    self.inputs.protocol["policy_dt_s"])
            initial = env.flat_evaluation_snapshot(torch.arange(env.num_envs, device=env.flat_episode.device))
            self.adapter.begin_initial(initial, root_position_world=initial["root_position_world"],
                yaw_rad=initial["yaw_rad"], reset_metadata=initial["reset_metadata"])
            self.snapshots.append(deepcopy(initial["reset_metadata"]))
            obs = env.get_observations()
            self.environment_observations += 1
            self.status = "running"
            clocks = (env.stage_actor_update, env.global_actor_update)
            quota = self.inputs.protocol["episodes_per_lane"]
            while not all(count >= quota for count in self.adapter.ledger.attempts):
                if self.steps >= self.max_steps:
                    raise RuntimeError("Source999-tick episode bound exhausted before every lane quota")
                self.limit_check()
                before = env.physical_tick
                action = self.actor.step(obs["policy"])
                self.attempted_steps += 1
                obs, reward, done, extras = env.step(action)
                self.steps += 1
                self.last_returned = (reward, done, extras)
                if env.physical_tick != before + 4 or (env.stage_actor_update, env.global_actor_update) != clocks:
                    raise RuntimeError("Evaluation changed physics decimation or actor update clocks")
                # env has already saved its first pre-reset failing substep.
                # Stop before reset snapshots/ledger parsing can obscure that
                # error, and never execute a second policy interval after it.
                self._stop_on_captured_numerical_event(extras, env.physical_tick,
                    env.startup_report.get("numerical_failure_capture", {}))
                returned = extras["returned_episode_state"]
                ids = (done & torch.tensor([active is not None for active in self.adapter.ledger.active],
                    dtype=torch.bool, device=done.device)).nonzero(as_tuple=False).flatten()
                snapshot = None
                if ids.numel():
                    snapshot = env.flat_evaluation_snapshot(ids)
                    if not torch.equal(snapshot["episode_id"], returned["episode_id"]) or not torch.equal(snapshot["episode_tick"], returned["episode_tick"]):
                        raise RuntimeError("Native returned snapshot identity differs from the paid-returned boundary")
                active_lanes = [lane for lane, active in enumerate(self.adapter.ledger.active) if active is not None]
                pending_measurements = self.tracking_measurements.prepare(extras["diagnostics"], active_lanes)
                pending_rough = (self.rough_measurements.prepare(extras["diagnostics"], active_lanes)
                                 if self.rough_measurements is not None else None)
                pending_recovery = (self.recovery_measurements.prepare(extras, active_lanes, policy_interval_id=self.steps)
                                    if self.recovery_measurements is not None else None)
                records = self.adapter.observe_paid(extras, reward, done,
                    returned_root_position_world=snapshot["root_position_world"] if snapshot else None,
                    returned_yaw_rad=snapshot["yaw_rad"] if snapshot else None,
                    reset_metadata=snapshot["reset_metadata"] if snapshot else None)
                self.tracking_measurements.accept(pending_measurements)
                if self.rough_measurements is not None:
                    self.rough_measurements.accept(pending_rough)
                if self.recovery_measurements is not None:
                    self.recovery_measurements.accept(pending_recovery)
                    self.recovery_measurements.close_records(records)
                self.accepted_steps += 1
                self.accepted_lane_transitions += len(active_lanes)
                if records:
                    commit_records(records)
                    self.writer_acknowledged_episodes += len(records)
                    self.step_records.extend(records)
                if ids.numel():
                    self.snapshots.append(deepcopy(snapshot["reset_metadata"]))
            self.limit_check()
            self.inputs.assert_unchanged()
            self._verify_actor()
            self.status = "attempt_quota_completed"
            self.physical_tick_readback = env.physical_tick
            return self.report()
        except BaseException as error:
            self.status, self.error = "failed", f"{type(error).__name__}: {error}"
            # A later substep/finish may raise after the first dump, before
            # env.step returns. Preserve that event without replacing the error.
            capture = getattr(env, "startup_report", {}).get("numerical_failure_capture", {})
            if (getattr(self.inputs, "numerical_capture", False)
                    and self.numerical_capture_event is None and capture.get("recorded") is True):
                self.numerical_capture_event = {"policy_interval_id": self.attempted_steps,
                    "reason": "first_physical_numerical_observation_before_step_error",
                    "capture": deepcopy(capture), "returned_policy_batch": False}
            if self.last_returned is not None:
                # Failure evidence is best effort: malformed extras or a failed
                # device readback must never replace the original step error.
                try:
                    reward, done, extras = self.last_returned
                    fields = {"returned_reward": reward, "returned_done": done}
                    if not isinstance(extras, Mapping):
                        raise TypeError("Returned extras is not a mapping")
                    for prefix, section in (("paid", "diagnostics"), ("returned", "returned_episode_state")):
                        values = extras.get(section, {})
                        if not isinstance(values, Mapping):
                            raise TypeError(f"Returned {section} is not a mapping")
                        for name, value in values.items():
                            if isinstance(value, torch.Tensor):
                                fields[prefix + "/" + name] = value
                    self.failure_returned_tensor_bindings = {
                        name: {"shape": list(value.shape), "dtype": str(value.dtype),
                               "sha256_named_tensor_bytes": actor_state_digest({name: value}),
                               "finite": bool(torch.isfinite(value).all())}
                        for name, value in fields.items() if isinstance(value, torch.Tensor)}
                except BaseException as diagnostic_error:
                    self.failure_diagnostic_error = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                    error.add_note(f"Failure tensor evidence also failed: {diagnostic_error!r}")
            censored = self.adapter.ledger.censor_open_episodes("execution_failed_or_external_limit")
            if censored:
                try:
                    if self.recovery_measurements is not None:
                        self.recovery_measurements.close_records(censored)
                    commit_records(censored)
                    self.writer_acknowledged_episodes += len(censored)
                    self.step_records.extend(censored)
                except BaseException as writer_error:
                    error.add_note(f"Censored record write also failed: {writer_error!r}")
            raise
        finally:
            tick = getattr(env, "physical_tick", None)
            self.physical_tick_readback = tick if type(tick) is int else None

    def report(self):
        if self.capture_statistics is not None:
            stats = self.capture_statistics
            count, total, maximum, above, excursions, longest = stats["sums"].detach().cpu().tolist()
            attempts = stats["attempts"].detach().cpu().tolist()
            return {"schema": "closure-event-capture-v1", "status": self.status, "error": self.error,
                "completed_returned_policy_steps": self.steps, "completed_returned_physics_intervals": self.steps * 4,
                "attempted_policy_steps": self.attempted_steps, "actual_physical_tick_readback": self.physical_tick_readback,
                "source_policy_step_bound": self.max_steps, "initial_observation_reads": self.environment_observations,
                "completed_attempts_per_lane": attempts,
                "attempted_episode_quota_met": all(x >= self.inputs.protocol["episodes_per_lane"] for x in attempts),
                "observed_policy_lane_intervals": int(count), "closure_gap_mean_m": total/count if count else None,
                "closure_gap_max_m": maximum if count else None,
                "capture_threshold_m": stats["threshold_m"], "above_threshold_policy_lane_intervals": int(above),
                "above_threshold_excursions": int(excursions), "longest_above_threshold_policy_intervals": int(longest),
                "measurement_scope": "Per-lane maximum across four5ms substeps, before reset. Only lanes below episode-attempt quota. Consecutive20ms flagged intervals do not measure exact5ms exceedance duration.",
                "actor": self.actor.receipt(), "ppo_updates": 0, "training_release": False,
                "native_qualification": False, "repair_verified": False,
                "assessment": "closure_event_diagnosis_only_no_skill_or_reward_assessment",
                "failure_last_returned_tensor_bindings": deepcopy(self.failure_returned_tensor_bindings),
                "failure_diagnostic_error": self.failure_diagnostic_error,
                "numerical_capture_diagnostic": {"event": deepcopy(self.numerical_capture_event),
                    "outcome": "numerical_event_detected" if self.numerical_capture_event else
                        "not_reproduced_within_observed_window" if self.steps else "no_native_rollout_observed",
                    "training_release": False, "repair_verified": False}}
        result = {"schema": VERSION, "status": self.status, "error": self.error,
            "completed_returned_policy_steps": self.steps, "completed_returned_physics_intervals": self.steps * 4,
            "ledger_accepted_policy_batches": self.accepted_steps,
            "ledger_accepted_lane_transitions": self.accepted_lane_transitions,
            "record_writer_acknowledged_episodes": self.writer_acknowledged_episodes,
            "attempted_policy_steps": self.attempted_steps,
            "actual_physical_tick_readback": self.physical_tick_readback,
            "physics_count_scope": "completed_returned_physics_intervals counts the four expected physics intervals in each returned policy batch; a failed or invalid native call can advance differently, see actual_physical_tick_readback",
            "failure_last_returned_tensor_bindings": deepcopy(self.failure_returned_tensor_bindings),
            "failure_diagnostic_error": self.failure_diagnostic_error,
            "count_scope": "Quota-complete lanes still participate in native actor/environment batches, but acquire no further evaluation attempts. Returned batches, accepted ledger transitions and record writer acknowledgements are distinct.",
            "source_policy_step_bound": self.max_steps, "initial_observation_reads": self.environment_observations,
            "actor": self.actor.receipt(), "episode_ledger": self.adapter.ledger.report(),
            "case_tracking_measurements": self.tracking_measurements.report(),
            "reset_snapshot_batches": deepcopy(self.snapshots),
            "assessment": "measurement_only", "ppo_updates": 0,
            "native_qualification": False, "training_release": False,
            "execution_tree_frozen": False, "deployment_1000Hz_verified": False}
        if self.rough_measurements is not None:
            result["rough_terrain_measurements"] = self.rough_measurements.report()
        if self.recovery_measurements is not None:
            result["scripted_recovery_measurements"] = self.recovery_measurements.report()
        if self.inputs.is_plane_regression:
            result["evaluation_surface"] = "plane_regression_v1"
            result["assessment_scope"] = "rough checkpoint on fixed flat cases; no rough route capability claim"
        if getattr(self.inputs, "numerical_capture", False):
            result["numerical_capture_diagnostic"] = {
                "event": deepcopy(self.numerical_capture_event),
                "outcome": "numerical_event_detected" if self.numerical_capture_event else "not_reproduced_within_observed_window",
                "training_release": False, "repair_verified": False,
                "scope": "fixed 4701 sampled actor with fresh resets; not exact replay of failed 4729 rollout"}
        return result
