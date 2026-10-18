"""CPU protocol and pre-reset episode accounting for the source flat evaluator.

This module does not construct a simulator, alter a command, grade capability,
or authorize an asset. A future entry point must authenticate the training
owner and mechanical admission before constructing ChassisEnv. Its protocol is
an evaluation document, not a guessed scut_flat_commands training schema.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math

from .scut_flat_reward import WEIGHTS
from .scut_reward_selection import SCUT_FLAT_PROFILE, SCUT_SOURCE_PROFILES, V7_FINAL_ROUGH_PROFILE
from .spin_center_measurement import SpinCenterMeasurements


SCHEMA = "flat-policy-evaluation-v1"
CENTER_REFERENCE = "root_link_origin_world"
HEADING_HOLD_VERSION = "initial_heading_gain5_limit2pi_v1"


def _canonical_payment_names(reward_profile):
    if reward_profile not in SCUT_SOURCE_PROFILES:
        raise ValueError("Unknown evaluation reward profile")
    if reward_profile == V7_FINAL_ROUGH_PROFILE:
        from .final_rough_reward import WEIGHTS as final_weights
        return tuple(final_weights)
    return tuple(WEIGHTS)


def runtime_status():
    """A code-status receipt, never an asset or native admission receipt."""
    return {"implementation_scope": "cpu_protocol_actor_preparation_and_pre_reset_episode_ledger",
            "runtime_integration_verified": False, "native_execution_allowed": False,
            "cpu_components": ["source_episode_reset_and_advance_owner", "typed_before_first_reset_setup",
                "checkpoint_byte_binding_and_actor_rng", "paid_pre_reset_ledger_adapter",
                "pure_spin_per_turn_center_and_periodic_offset_measurements",
                "bounded_inference_loop_and_returned_native_reset_snapshot_adapter"],
            "missing_integration": ["admitted_native_evaluation_execution_loop", "unified_mechanical_asset_admission",
                "independent_host_and_gpu_usage_admission", "deployment_1000Hz_consistency_qualification"]}


def require_native_execution():
    """Fail closed until a reviewed runtime entry replaces this explicit gate."""
    raise RuntimeError("Native flat evaluation is not integrated or mechanically admitted; CPU ledger only")


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _vector(value, width, label, *, finite=True):
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"{label} requires {width} explicit values")
    if finite:
        return [_number(v, label) for v in value]
    if any(type(v) not in (int, float) for v in value):
        raise ValueError(f"{label} requires numeric values")
    return list(value)


def command_kind(command):
    """Classify exact requested zeros; these are not performance tolerances."""
    vx, yaw, _ = _vector(command, 3, "command [vx_m_s,yaw_rad_s,height_m]")
    return ("standing" if vx == 0 and yaw == 0 else "pure_spin" if vx == 0
            else "pure_translation" if yaw == 0 else "combined")


def validate_protocol(protocol):
    from .scripted_recovery_evaluation import selected, validate_protocol as recovery_protocol
    if isinstance(protocol, Mapping) and selected(protocol):
        return recovery_protocol(protocol)
    return _validate_flat_protocol(protocol)


def _validate_flat_protocol(protocol):
    required = {"schema", "mode", "actor_mode", "episodes_per_lane", "environment_seed",
                "actor_seed", "policy_dt_s", "horizon_seconds", "center_reference", "assessment"}
    if not isinstance(protocol, Mapping) or set(protocol) - (required | {"cases", "heading_hold", "temporal_flat_requests"}) or required - set(protocol):
        raise ValueError("Incomplete or unknown flat evaluation protocol fields")
    if (protocol["schema"] != SCHEMA or protocol["mode"] not in ("source_distribution", "fixed_measurement")
            or protocol["actor_mode"] not in ("mean", "sampled")
            or protocol["center_reference"] != CENTER_REFERENCE
            or protocol["assessment"] != "measurement_only"):
        raise ValueError("Unsupported flat evaluation protocol/measurement scope")
    _integer(protocol["episodes_per_lane"], "episodes_per_lane", 1)
    for key in ("environment_seed", "actor_seed"):
        _integer(protocol[key], key)
    for key, expected in (("policy_dt_s", .02), ("horizon_seconds", 20.)):
        if _number(protocol[key], key) != expected:
            raise ValueError(f"Source flat evaluation requires {key}={expected}")
    if "heading_hold" in protocol and (protocol["heading_hold"] != HEADING_HOLD_VERSION
            or protocol["mode"] != "fixed_measurement"):
        raise ValueError("Heading hold requires the fixed initial-heading controller and fixed measurement mode")
    if protocol["mode"] == "source_distribution":
        if "cases" in protocol:
            raise ValueError("Source distribution cannot declare fixed measurement cases")
    else:
        cases = protocol.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("Fixed measurement requires explicit nonempty cases")
        names = []
        for case in cases:
            if not isinstance(case, Mapping) or set(case) != {"case_id", "base_request"}:
                raise ValueError("Fixed case requires case_id and base_request only")
            name = case["case_id"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Fixed case_id must be nonempty")
            request = _vector(case["base_request"], 3, "fixed base request")
            if request[2] <= 0:
                raise ValueError("Fixed base request requires positive height")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("Duplicate fixed case IDs")
        if "heading_hold" in protocol and not any(
                command_kind(case["base_request"]) == "pure_translation" for case in cases):
            raise ValueError("Heading hold requires at least one pure translation case")
    from .temporal_flat_evaluation import selected, validate
    if selected(protocol):
        validate(protocol)
    return deepcopy(dict(protocol))


def _hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} requires a lowercase SHA-256")
    return value


def _source_hashes(values, label):
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} requires nonempty source provenance")
    for name, digest in values.items():
        if not isinstance(name, str) or not name or name.startswith("/") or ".." in name.split("/"):
            raise ValueError(f"{label} requires relative source paths")
        _hash(digest, label)
    return dict(values)


def verify_checkpoint_identity(checkpoint_infos, *, expected_contract_sha256, expected_asset_sha256,
        expected_control_sha256, checkpoint_sha256, runtime_source_sha256, required_sources):
    """Check caller-hashed bytes; never infer identity from a checkpoint name.

    The caller also uses training_profile.verify_profile_evaluation for runner
    profile/schema identity. This check cannot replace mechanical admission.
    """
    if not isinstance(checkpoint_infos, Mapping):
        raise ValueError("Checkpoint infos are required")
    expected = {"contract_sha256": expected_contract_sha256,
                "asset_manifest_sha256": expected_asset_sha256,
                "control_math_sha256": expected_control_sha256}
    for name, digest in expected.items():
        if checkpoint_infos.get(name) != _hash(digest, name):
            raise ValueError(f"Checkpoint identity mismatch: {name}")
    _hash(checkpoint_sha256, "checkpoint_sha256")
    saved = _source_hashes(checkpoint_infos.get("profile_source_sha256"), "checkpoint sources")
    runtime = _source_hashes(runtime_source_sha256, "runtime sources")
    encoded = json.dumps(saved, sort_keys=True, separators=(",", ":")).encode()
    if checkpoint_infos.get("training_source_sha256") != hashlib.sha256(encoded).hexdigest():
        raise ValueError("Checkpoint source digest mismatch")
    if (not isinstance(required_sources, (list, tuple)) or not required_sources
            or any(not isinstance(n, str) for n in required_sources)
            or len(required_sources) != len(set(required_sources))):
        raise ValueError("Required source dependency list must be explicit and unique")
    if set(required_sources) - saved.keys() or set(required_sources) - runtime.keys():
        raise ValueError("Required source provenance is missing")
    changed = sorted(name for name in saved.keys() & runtime.keys() if saved[name] != runtime[name])
    if changed:
        raise ValueError("Shared training/runtime sources differ: " + ", ".join(changed))
    return {**expected, "checkpoint_sha256": checkpoint_sha256,
            "shared_sources_checked": sorted(saved.keys() & runtime.keys()),
            "runtime_only_sources": sorted(runtime.keys() - saved.keys()),
            "mechanical_admission_verified": False, "native_qualified": False}


class _Measurements:
    def __init__(self):
        self.count = self.segments = 0
        self.vx_error_sq = self.yaw_error_sq = self.speed_sq = 0.
        self.path = self.max_displacement = self.max_heading = self.max_lateral = 0.
        self.max_abs_yaw_rate = self.abs_yaw_rate_sum = 0.
        self.vy_sum = self.vy_sq = 0.
        self.attitude_count = self.roll_count = self.attitude_missing = self.attitude_invalid = 0
        self.roll_sq = self.pitch_sq = self.tilt_sq = 0.
        self.max_roll = self.max_pitch = self.max_tilt = 0.
        self.anchor = self.heading_anchor = self.last_position = None
        self.last_world_velocity = self.last_world_acceleration = None
        self.acceleration_count = self.jerk_count = self.world_velocity_missing = 0
        self.acceleration_sq = self.jerk_sq = self.max_acceleration = self.max_jerk = 0.

    def begin_segment(self, position, heading):
        self.segments += 1
        self.anchor = list(position)
        self.last_position = list(position)
        self.heading_anchor = heading
        self.last_world_velocity = self.last_world_acceleration = None

    def observe(self, position, heading, velocity, omega, command, projected_gravity=None,
                world_velocity=None):
        self.count += 1
        if world_velocity is None:
            self.world_velocity_missing += 1
            self.last_world_velocity = self.last_world_acceleration = None
        else:
            acceleration = None
            if self.last_world_velocity is not None:
                acceleration = [(a - b) / .02 for a, b in zip(world_velocity, self.last_world_velocity)]
                squared = sum(v * v for v in acceleration)
                self.acceleration_count += 1
                self.acceleration_sq += squared
                self.max_acceleration = max(self.max_acceleration, math.sqrt(squared))
                if self.last_world_acceleration is not None:
                    jerk = [(a - b) / .02 for a, b in zip(acceleration, self.last_world_acceleration)]
                    squared = sum(v * v for v in jerk)
                    self.jerk_count += 1
                    self.jerk_sq += squared
                    self.max_jerk = max(self.max_jerk, math.sqrt(squared))
            self.last_world_velocity = list(world_velocity)
            self.last_world_acceleration = acceleration
        self.vx_error_sq += (velocity[0] - command[0]) ** 2
        self.yaw_error_sq += (omega[2] - command[1]) ** 2
        self.speed_sq += velocity[0] ** 2 + velocity[1] ** 2
        self.max_abs_yaw_rate = max(self.max_abs_yaw_rate, abs(omega[2]))
        self.abs_yaw_rate_sum += abs(omega[2])
        self.vy_sum += velocity[1]
        self.vy_sq += velocity[1] ** 2
        if projected_gravity is None:
            self.attitude_missing += 1
        else:
            gx, gy, gz = projected_gravity
            yz = math.hypot(gy, gz)
            if math.hypot(gx, yz) == 0.:
                self.attitude_invalid += 1
            else:
                # Gravity is the clean pre-reset root-body reading, not actor noise.
                # These scale-invariant angles use the ZYX upright convention.
                pitch = math.atan2(gx, yz)
                tilt = math.atan2(math.hypot(gx, gy), -gz)
                self.attitude_count += 1
                self.pitch_sq += pitch * pitch
                self.tilt_sq += tilt * tilt
                self.max_pitch = max(self.max_pitch, abs(pitch))
                self.max_tilt = max(self.max_tilt, tilt)
                if yz != 0.:
                    roll = math.atan2(-gy, -gz)
                    self.roll_count += 1
                    self.roll_sq += roll * roll
                    self.max_roll = max(self.max_roll, abs(roll))
        dx, dy = position[0] - self.anchor[0], position[1] - self.anchor[1]
        self.path += math.hypot(position[0] - self.last_position[0], position[1] - self.last_position[1])
        self.max_displacement = max(self.max_displacement, math.hypot(dx, dy))
        self.max_lateral = max(self.max_lateral, abs(-math.sin(self.heading_anchor) * dx
                                                   + math.cos(self.heading_anchor) * dy))
        self.max_heading = max(self.max_heading, abs(heading - self.heading_anchor))
        self.last_position = list(position)

    def report(self, dt):
        rms = lambda value: math.sqrt(value / self.count) if self.count else None
        return {"valid_frames": self.count, "observed_seconds": self.count * dt, "segments": self.segments,
                "body_vx_tracking_rmse_m_s": rms(self.vx_error_sq),
                "body_omega_z_tracking_rmse_rad_s": rms(self.yaw_error_sq),
                "root_com_body_planar_speed_rms_m_s": rms(self.speed_sq),
                "root_link_path_length_m": self.path,
                "root_link_net_displacement_last_segment_m": math.hypot(
                    self.last_position[0] - self.anchor[0], self.last_position[1] - self.anchor[1]) if self.count else None,
                "root_link_max_displacement_from_segment_start_m": self.max_displacement if self.count else None,
                "max_abs_unwrapped_heading_change_rad": self.max_heading if self.count else None,
                "max_lateral_deviation_from_segment_heading_m": self.max_lateral if self.count else None,
                "max_abs_body_omega_z_rad_s": self.max_abs_yaw_rate if self.count else None,
                "body_omega_z_abs_mean_rad_s": self.abs_yaw_rate_sum / self.count if self.count else None,
                "root_com_body_vy_signed_mean_m_s": self.vy_sum / self.count if self.count else None,
                "root_com_body_vy_rms_m_s": rms(self.vy_sq),
                "body_roll_rms_rad": math.sqrt(self.roll_sq / self.roll_count) if self.roll_count else None,
                "max_abs_body_roll_rad": self.max_roll if self.roll_count else None,
                "body_pitch_rms_rad": math.sqrt(self.pitch_sq / self.attitude_count) if self.attitude_count else None,
                "max_abs_body_pitch_rad": self.max_pitch if self.attitude_count else None,
                "body_tilt_rms_rad": math.sqrt(self.tilt_sq / self.attitude_count) if self.attitude_count else None,
                "max_body_tilt_rad": self.max_tilt if self.attitude_count else None,
                "world_COM_motion": {
                    "status": "collected" if self.acceleration_count else
                              "not_recorded" if self.world_velocity_missing == self.count else "insufficient_contiguous_frames",
                    "source": "clean_pre_reset_root_COM_velocity_world",
                    "sample_dt_s": .02, "difference": "unfiltered backward difference; no reset or invalid gap crossing",
                    "acceleration_samples": self.acceleration_count, "jerk_samples": self.jerk_count,
                    "missing_world_velocity_frames": self.world_velocity_missing,
                    "acceleration_rms_m_s2": math.sqrt(self.acceleration_sq / self.acceleration_count) if self.acceleration_count else None,
                    "acceleration_max_m_s2": self.max_acceleration if self.acceleration_count else None,
                    "jerk_rms_m_s3": math.sqrt(self.jerk_sq / self.jerk_count) if self.jerk_count else None,
                    "jerk_max_m_s3": self.max_jerk if self.jerk_count else None,
                    "scope": "20ms 3D world root-rigid-body COM derivatives; not whole-robot COM,5ms peaks,stopping distance or capability thresholds; all retains valid command-change boundaries,by-kind splits at kind changes"},
                "attitude_measurement": {
                    "status": ("partial" if self.attitude_missing or self.attitude_invalid or self.roll_count != self.attitude_count
                               else "collected") if self.attitude_count else
                              "invalid_gravity" if self.attitude_invalid else "not_recorded",
                    "source": "clean_pre_reset_projected_gravity_body",
                    "valid_frames": self.attitude_count, "roll_defined_frames": self.roll_count,
                    "missing_frames": self.attitude_missing, "zero_gravity_frames": self.attitude_invalid,
                    "roll_undefined_frames": self.attitude_count - self.roll_count,
                    "convention": "ZYX roll/pitch from body gravity; tilt is body Z angle from world up; radians",
                    "limitation": "Roll is undefined at exact pitch +/-pi/2 and ill-conditioned nearby; no attitude threshold"}}


class FlatEpisodeLedger:
    """Consume CPU pre-reset rows; explicit begin_episode prevents reset leakage.

    One row represents one whole policy interval. Payments are already dt
    scaled. Excluded learning intervals remain in whole-episode measurements.
    Completed failures and boundary truncations consume attempted-episode quota;
    only horizon truncation is labelled a complete horizon, never a success.
    """
    FRAME_FIELDS = frozenset(("episode_id", "episode_tick", "reward", "payments", "done", "terminated",
        "timeout", "boundary", "termination_reasons", "learning_mask", "bootstrap_mask",
        "measurement_valid", "command_override_active", "effective_command", "base_request",
        "root_link_position_world", "root_com_velocity_body", "angular_velocity_body", "yaw_rad"))
    OPTIONAL_FRAME_FIELDS = frozenset(("projected_gravity_body", "root_com_velocity_world"))

    def __init__(self, protocol, lane_case_ids, payment_names, *, reward_profile=SCUT_FLAT_PROFILE):
        self.protocol = validate_protocol(protocol)
        canonical_names = _canonical_payment_names(reward_profile)
        if (not isinstance(payment_names, (list, tuple)) or len(payment_names) != len(canonical_names)
                or any(not isinstance(n, str) or not n for n in payment_names)
                or set(payment_names) != set(canonical_names)):
            raise ValueError(f"Ledger requires exactly the canonical {len(canonical_names)} source payment names")
        self.payment_names = tuple(payment_names)
        self.cases = {c["case_id"]: c["base_request"] for c in self.protocol.get("cases", [])}
        permitted = set(self.cases) or {"source_distribution"}
        if (not isinstance(lane_case_ids, (list, tuple)) or not lane_case_ids
                or any(name not in permitted for name in lane_case_ids)):
            raise ValueError("Every lane needs an explicit protocol case ID")
        if set(lane_case_ids) != permitted:
            raise ValueError("Every declared evaluation case needs at least one lane")
        self.lane_cases = tuple(lane_case_ids)
        self.active = [None] * len(lane_case_ids)
        self.attempts = [0] * len(lane_case_ids)
        self.last_ids = [-1] * len(lane_case_ids)
        self.records = []

    def begin_episode(self, lane, episode_id, *, root_link_position_world, yaw_rad, reset_metadata):
        _integer(lane, "lane")
        if lane >= len(self.active) or self.active[lane] is not None:
            raise ValueError("Invalid lane or previous episode not finalized")
        _integer(episode_id, "episode_id")
        if episode_id <= self.last_ids[lane]:
            raise ValueError("Episode IDs must increase per lane")
        if self.attempts[lane] >= self.protocol["episodes_per_lane"]:
            raise ValueError("Lane has exhausted its attempted-episode quota")
        position = _vector(root_link_position_world, 3, "reset root link position")
        yaw = _number(yaw_rad, "reset yaw")
        if not isinstance(reset_metadata, Mapping) or not reset_metadata:
            raise ValueError("Explicit source reset metadata is required")
        # This data must be serializable before it can become a durable ledger.
        json.dumps(reset_metadata, allow_nan=False)
        all_measurements = _Measurements()
        all_measurements.begin_segment(position, yaw)
        self.active[lane] = dict(episode_id=episode_id, tick=0, position=position, yaw=yaw,
            unwrapped_yaw=yaw, reset_yaw=yaw, reset_position=position[:], reset_metadata=deepcopy(dict(reset_metadata)),
            reward=0., payments={n: 0. for n in self.payment_names},
            rl_payments={n: 0. for n in self.payment_names}, rl_frames=0, invalid_frames=0,
            override_frames=0, metrics={"all": all_measurements}, last_kind=None,
            spin_centers=SpinCenterMeasurements(),
            previous_measurement_valid=True,
            first_command=None, last_command=None, first_request=None, last_request=None)
        self.last_ids[lane] = episode_id
        from .temporal_flat_evaluation import selected, TemporalEpisodeMeasurements
        if selected(self.protocol):
            self.active[lane]["temporal_measurements"] = TemporalEpisodeMeasurements(
                self.cases[self.lane_cases[lane]], position, yaw)

    def observe(self, lane, frame):
        _integer(lane, "lane")
        if lane >= len(self.active) or self.active[lane] is None:
            raise ValueError("A pre-reset frame requires an explicitly opened episode")
        if (not isinstance(frame, Mapping) or self.FRAME_FIELDS - set(frame)
                or set(frame) - (self.FRAME_FIELDS | self.OPTIONAL_FRAME_FIELDS)):
            raise ValueError("Missing or unknown pre-reset diagnostic fields")
        episode = self.active[lane]
        if (_integer(frame["episode_id"], "episode_id") != episode["episode_id"]
                or _integer(frame["episode_tick"], "episode_tick", 1) != episode["tick"] + 1):
            raise ValueError("Episode identity/ticks are discontinuous; possible post-reset data")
        for key in ("done", "terminated", "timeout", "boundary", "learning_mask", "bootstrap_mask",
                    "measurement_valid", "command_override_active"):
            if type(frame[key]) is not bool:
                raise ValueError(f"{key} must be explicit bool")
        if (frame["terminated"] and frame["timeout"] or frame["done"] != (frame["terminated"] or frame["timeout"])
                or frame["boundary"] and not frame["done"]):
            raise ValueError("Inconsistent pre-reset termination/truncation flags")
        reasons = frame["termination_reasons"]
        if not isinstance(reasons, list) or any(not isinstance(n, str) or not n for n in reasons):
            raise ValueError("termination_reasons must be an explicit string list")
        if frame["terminated"] and not reasons:
            raise ValueError("A terminal episode requires a named cause")
        if (frame["episode_tick"] > 999
                or (frame["episode_tick"] == 999 and not frame["done"])
                or (frame["timeout"] and not frame["boundary"] and frame["episode_tick"] != 999)):
            raise ValueError("Timeout/ticks differ from the source 20s max_episode_length-1 boundary")
        command = _vector(frame["effective_command"], 3, "effective command")
        request = _vector(frame["base_request"], 3, "base request")
        # Serialization may expose float32 command values as Python float64.
        # These numerical comparison tolerances are not capability thresholds;
        # exact requested zeros still determine command_kind.
        same = lambda a, b: all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-8) for x, y in zip(a, b))
        expected_request = self.cases.get(self.lane_cases[lane])
        if "temporal_measurements" in episode:
            from .temporal_flat_evaluation import request_at_tick
            expected_request = request_at_tick(expected_request, frame["episode_tick"])
            if frame["command_override_active"]:
                raise ValueError("Temporal measurement requires nominal requests without a capture override")
        if self.cases and not same(request, expected_request):
            raise ValueError("Fixed base request changed; fixed mode is not source-distribution sampling")
        heading_hold = (self.protocol.get("heading_hold") == HEADING_HOLD_VERSION
            and command_kind(request) == "pure_translation")
        if self.cases and not frame["command_override_active"]:
            if heading_hold:
                if not episode["previous_measurement_valid"]:
                    raise ValueError("Heading hold requires a valid preceding pre-reset yaw")
                delta = episode["reset_yaw"] - episode["yaw"]
                expected_yaw = max(-2. * math.pi, min(2. * math.pi,
                    5. * math.atan2(math.sin(delta), math.cos(delta))))
                if (not same([command[0], command[2]], [request[0], request[2]])
                        or not math.isclose(command[1], expected_yaw, rel_tol=1e-6, abs_tol=5e-6)):
                    raise ValueError("Heading hold effective command differs from the declared initial-heading controller")
            elif not same(command, request):
                raise ValueError("Fixed command differs outside the declared source reset override")
        reward = _number(frame["reward"], "reward")
        if not isinstance(frame["payments"], Mapping) or set(frame["payments"]) != set(self.payment_names):
            raise ValueError(f"All {len(self.payment_names)} payments must be present without legacy additions")
        payments = {n: _number(frame["payments"][n], n) for n in self.payment_names}
        if not math.isclose(sum(payments.values()), reward, abs_tol=1e-5, rel_tol=1e-5):
            raise ValueError("Already-scaled component payments do not sum to reward")
        valid = frame["measurement_valid"]
        position = _vector(frame["root_link_position_world"], 3, "root link position", finite=valid)
        velocity = _vector(frame["root_com_velocity_body"], 3, "root COM velocity", finite=valid)
        omega = _vector(frame["angular_velocity_body"], 3, "angular velocity", finite=valid)
        gravity = (_vector(frame["projected_gravity_body"], 3, "projected gravity", finite=valid)
                   if "projected_gravity_body" in frame else None)
        world_velocity = (_vector(frame["root_com_velocity_world"], 3, "root COM world velocity", finite=valid)
                          if "root_com_velocity_world" in frame else None)
        yaw = _number(frame["yaw_rad"], "yaw") if valid else frame["yaw_rad"]
        if type(yaw) not in (int, float):
            raise ValueError("yaw must be numeric even when measurement is invalid")
        # Mutate only after all input checks have succeeded.
        if "temporal_measurements" in episode:
            episode["temporal_measurements"].observe(frame)
        episode["tick"] += 1
        episode["reward"] += reward
        episode["rl_frames"] += frame["learning_mask"]
        episode["invalid_frames"] += not valid
        episode["override_frames"] += frame["command_override_active"]
        for name, value in payments.items():
            episode["payments"][name] += value
            if frame["learning_mask"]:
                episode["rl_payments"][name] += value
        if episode["first_command"] is None:
            episode["first_command"], episode["first_request"] = command, request
        episode["last_command"], episode["last_request"] = command, request
        if valid:
            if not episode["previous_measurement_valid"]:
                # Do not infer a path or unwrap heading across missing physics.
                episode.update(position=position, yaw=yaw, unwrapped_yaw=yaw)
                episode["metrics"]["all"].begin_segment(position, yaw)
            delta = math.atan2(math.sin(yaw - episode["yaw"]), math.cos(yaw - episode["yaw"]))
            heading = episode["unwrapped_yaw"] + delta
            kind = ("capture_override" if frame["command_override_active"] else
                    command_kind(request) if heading_hold else command_kind(command))
            spin = episode["spin_centers"]
            if kind == "pure_spin":
                if episode["last_kind"] != kind:
                    start_tick = episode["tick"] - int(episode["previous_measurement_valid"])
                    spin.begin_segment(episode["position"], episode["unwrapped_yaw"],
                        start_tick * self.protocol["policy_dt_s"])
                # A first valid frame after missing physics only establishes
                # an anchor; no path/yaw is inferred across the missing row.
                if episode["previous_measurement_valid"]:
                    spin.observe(position, heading, episode["tick"] * self.protocol["policy_dt_s"])
            elif episode["last_kind"] == "pure_spin":
                spin.end_segment("command_changed")
            metric = episode["metrics"].setdefault(kind, _Measurements())
            if kind != episode["last_kind"]:
                metric.begin_segment(episode["position"], episode["unwrapped_yaw"])
            for measure in (episode["metrics"]["all"], metric):
                measure.observe(position, heading, velocity, omega, command, gravity, world_velocity)
            episode.update(position=position, yaw=yaw, unwrapped_yaw=heading, last_kind=kind)
        else:
            if episode["last_kind"] == "pure_spin":
                episode["spin_centers"].end_segment("invalid_measurement")
            episode["last_kind"] = None
        episode["previous_measurement_valid"] = valid
        if frame["done"]:
            outcome = "terminated" if frame["terminated"] else "boundary_truncated" if frame["boundary"] else "horizon_truncated"
            return self._finish(lane, outcome, reasons, frame["bootstrap_mask"])
        return None

    def _finish(self, lane, outcome, reasons, bootstrap_mask=None):
        episode = self.active[lane]
        episode["spin_centers"].end_segment("episode_" + outcome)
        duration = episode["tick"] * self.protocol["policy_dt_s"]
        record = {"lane": lane, "case_id": self.lane_cases[lane], "episode_id": episode["episode_id"],
            "actor_mode": self.protocol["actor_mode"], "protocol_mode": self.protocol["mode"],
            "assessment": "measurement_only", "outcome": outcome, "termination_reasons": list(reasons),
            "complete_horizon": outcome == "horizon_truncated", "center_reference": CENTER_REFERENCE,
            "policy_ticks": episode["tick"], "observed_seconds": duration,
            "learning_frames": episode["rl_frames"], "excluded_learning_frames": episode["tick"] - episode["rl_frames"],
            "invalid_measurement_frames": episode["invalid_frames"], "command_override_frames": episode["override_frames"],
            "return": episode["reward"], "payments": episode["payments"], "learning_eligible_payments": episode["rl_payments"],
            "payment_per_observed_second": {n: v / duration if duration else None for n, v in episode["payments"].items()},
            "initial_reset": episode["reset_metadata"], "effective_command_first": episode["first_command"],
            "effective_command_last": episode["last_command"], "base_request_first": episode["first_request"],
            "base_request_last": episode["last_request"], "returned_state_bootstrap_mask": bootstrap_mask,
            "pure_spin_center_decomposition": episode["spin_centers"].report(),
            "measurements": {n: metric.report(self.protocol["policy_dt_s"]) for n, metric in episode["metrics"].items()}}
        if self.protocol.get("heading_hold") == HEADING_HOLD_VERSION:
            record["heading_hold"] = {
                "controller": HEADING_HOLD_VERSION,
                "applies_to": "pure_translation_base_request_without_source_reset_override",
                "reference_yaw_rad": episode["reset_yaw"],
                "reference_yaw_source": "begin_episode_returned_reset_yaw; fixed for the whole episode",
                "feedback_yaw_source": "preceding accepted pre-reset yaw; reset yaw before the first interval",
                "formula": "clamp(5 * atan2(sin(reference_yaw-feedback_yaw), cos(reference_yaw-feedback_yaw)), -2*pi, 2*pi)",
                "classification": "original base request; source reset overrides retain capture_override",
                "yaw_tracking_error_reference": "effective corrected yaw command; use body_omega_z_abs_mean_rad_s for raw angular motion"}
        if "temporal_measurements" in episode:
            record["temporal_measurements"] = episode["temporal_measurements"].report(outcome)
        self.records.append(deepcopy(record))
        self.active[lane] = None
        if outcome != "censored":
            self.attempts[lane] += 1
        return deepcopy(record)

    def censor_open_episodes(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Censor reason must be explicit")
        return [self._finish(i, "censored", [reason]) for i, episode in enumerate(self.active) if episode is not None]

    def report(self):
        return {"schema": SCHEMA, "protocol": deepcopy(self.protocol), "assessment": "measurement_only",
                "implementation_status": runtime_status(),
                "attempted_episode_quota_met": all(n >= self.protocol["episodes_per_lane"] for n in self.attempts),
                "completed_attempts_per_lane": self.attempts[:], "open_episodes": sum(e is not None for e in self.active),
                "records": deepcopy(self.records), "native_qualification": False,
                "declared_sampling_mode": self.protocol["mode"], "source_distribution_verified": False}
