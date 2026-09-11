"""Pure stdlib V40 trajectory metrics. No torch, Isaac, RSL or simulator import.

Inputs are owned, PRE-auto-reset snapshots, not reward logs. Invalid numeric or
clock evidence is rejected rather than repaired. Targets are research goals.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable

from wheeled_tasks.v40.contract import CONTRACT_ID, CONTRACT_V2_ID

MAX_RECORDS = 2001
LIMITATIONS = {
    "scope": "engineering research targets, not user-measured specifications or hardware acceptance",
    "reset": "deterministic reset and fixed commands; seeds are not independent robustness trials",
    "randomization": "not_tested", "perturbations": "not_tested",
    "high_spin_12_57_rad_s": "not_tested", "world_xy_position_holding": "not_tested",
    "motor_mapping": "not_verified; chain ratios/angle references/coaxial output assignment unresolved",
    "effort": "last physics sample per policy tick: explicit-actuator applied_torque into simulation; prior utilization, not current certification or full-rate saturation statistics",
    "clearance": "rotated base_visual_bbox conservative lower bound, not exact mesh distance",
    "contact": "policy-tick maximum of two sensor-history samples; not complete per-physics-step impact statistics or ground-pair proof",
    "velocity": "root COM velocity in root link frame, matching training observations/reward",
    "drift": "XY displacement of initial wheel LINK-axis-origin midpoint; NOT COM and never reanchored after spin-up",
    "video": "not_recorded",
    "runtime_validation": "IsaacLab 2.3.0 fields source-audited; server integration still required",
}


class InvalidTrajectory(ValueError):
    """Missing, nonfinite, out-of-domain, or inconsistent evidence cannot pass."""


def finite(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise InvalidTrajectory(f"{name}: finite numeric value required")
    return float(value)


def vector(value, size: int, name: str) -> list[float]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise InvalidTrajectory(f"{name}: expected {size} values")
    return [finite(v, name) for v in value]


def validate_command(contract: dict, stage: str, command) -> tuple[float, float, float]:
    values = vector(command, 3, "command")
    identity = contract.get("contract_id")
    if identity not in (CONTRACT_ID, CONTRACT_V2_ID):
        raise InvalidTrajectory("unsupported V40 contract identity")
    if stage not in ("stand", "height", "locomotion"):
        raise InvalidTrajectory("unsupported training stage")
    bounds = contract["commands"]["stages"][stage]
    # V2 velocities follow the actual trained contract, including reduced ranges.
    # Preserve V1's absolute caps and the shared approved height domain.
    absolute_bounds = ((-.5, .5), (-1., 1.), (.28, .32)) if identity == CONTRACT_ID else (None, None, (.28, .32))
    for value, key, absolute in zip(values, ("vx", "wz", "height"), absolute_bounds):
        low, high = vector(bounds[key], 2, key + " bounds")
        if (not low <= value <= high
                or (absolute is not None and not absolute[0] <= low <= high <= absolute[1])):
            raise InvalidTrajectory(f"command {key} outside trained-stage/approved V40 domain")
    return tuple(values)


def require_trained_stage(run_manifest: dict, stage: str) -> None:
    if run_manifest.get("stage") != stage:
        raise InvalidTrajectory("checkpoint trained stage missing/different; out-of-domain is not acceptance")


def default_cases(contract: dict, stage: str) -> list[dict]:
    commands = {
        "stand": [(0., 0., .32)],
        "height": [(0., 0., h) for h in (.28, .30, .32)],
        "locomotion": [(0., 0., .30), (.5, 0., .30), (-.5, 0., .30),
                       (0., 1., .30), (0., -1., .30)],
    }
    if stage not in commands:
        raise InvalidTrajectory("unsupported stage")
    # Reduced training ranges are allowed by train preflight, but do not invent coverage.
    result = []
    for i, command in enumerate(commands[stage]):
        try:
            checked = validate_command(contract, stage, command)
        except InvalidTrajectory:
            continue
        result.append({"case_id": f"{stage}_{i}", "command": list(checked)})
    if not result:
        raise InvalidTrajectory("no default cases within actual trained-stage domain; specify --command")
    return result


@dataclass(frozen=True)
class Thresholds:
    duration_s: float = 10.0
    steady_start_s: float = 2.0
    max_drift_m: float = .05
    final_drift_m: float = .05
    height_rmse_m: float = .03
    height_steady_bias_m: float = .01
    vx_rmse_m_s: float = .15
    vx_steady_bias_m_s: float = .05
    wz_rmse_rad_s: float = .30
    wz_steady_bias_rad_s: float = .10
    peak_tilt_deg: float = 20.0
    max_effort_prior_utilization: float = 1.00001

    def __post_init__(self):
        for name, value in asdict(self).items():
            if finite(value, name) < 0:
                raise InvalidTrajectory(f"{name} must be nonnegative")
        if not 0 < self.duration_s <= 10 or not 0 <= self.steady_start_s < self.duration_s:
            raise InvalidTrajectory("duration must be (0,10] seconds and steady_start < duration")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path | None) -> "Thresholds":
        if path is None:
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise InvalidTrajectory("thresholds must be an object with known fields only")
        return cls(**data)


BATCH_FIELDS = {
    "episode_step", "episode_time_s", "command", "root_link_pos_w_m", "root_link_quat_wxyz",
    "root_com_lin_vel_b_m_s", "root_com_ang_vel_b_rad_s", "projected_gravity_b", "height_m",
    "joint_pos_rad", "joint_vel_rad_s", "sim_joint_effort_nm", "wheel_axis_midpoint_w_m",
    "wheel_net_force_max_n", "non_wheel_net_force_max_n", "base_visual_clearance_lower_bound_m",
    "terminated", "timeout",
}
VECTOR_FIELDS = {
    "command": 3, "root_link_pos_w_m": 3, "root_link_quat_wxyz": 4,
    "root_com_lin_vel_b_m_s": 3, "root_com_ang_vel_b_rad_s": 3, "projected_gravity_b": 3,
    "joint_pos_rad": 6, "joint_vel_rad_s": 6, "sim_joint_effort_nm": 6,
    "wheel_axis_midpoint_w_m": 3, "wheel_net_force_max_n": 2,
}
SCALAR_FIELDS = {"episode_time_s", "time_s", "sim_time_s", "policy_dt_s", "physics_dt_s",
                 "height_m", "non_wheel_net_force_max_n", "base_visual_clearance_lower_bound_m"}
CLOCK_FIELDS = {"policy_tick", "physics_steps", "episode_step"}
REASONS = {"nonfinite", "non_wheel_contact", "knee_limit", "tilt", "low_height", "base_visual_bounds_ground"}


def snapshot_to_record(snapshot: dict, env_index: int = 0) -> dict:
    """Convert one environment's owned tensor snapshot; no zero fallback or live env read."""
    def plain(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        return deepcopy(value)
    result = {}
    for key, value in snapshot.items():
        if key == "termination_flags":
            result[key] = {name: plain(flag[env_index]) for name, flag in value.items()}
        else:
            result[key] = plain(value[env_index] if key in BATCH_FIELDS else value)
    return result


def validate_record(record: dict) -> None:
    required = BATCH_FIELDS | SCALAR_FIELDS | CLOCK_FIELDS | {"schema_version", "sample_kind", "contact_history_samples", "termination_flags"}
    if not isinstance(record, dict) or required - record.keys():
        raise InvalidTrajectory(f"missing record fields: {sorted(required - record.keys()) if isinstance(record, dict) else 'record'}")
    if record["schema_version"] != 1 or record["sample_kind"] not in ("initial", "pre_reset"):
        raise InvalidTrajectory("unknown snapshot schema/sample_kind (post-reset state forbidden)")
    for key, length in VECTOR_FIELDS.items():
        vector(record[key], length, key)
    for key in SCALAR_FIELDS:
        finite(record[key], key)
    for key in CLOCK_FIELDS:
        if type(record[key]) is not int or record[key] < 0:
            raise InvalidTrajectory(f"{key}: nonnegative integer ticks required")
    for key in ("terminated", "timeout"):
        if type(record[key]) is not bool:
            raise InvalidTrajectory(f"{key}: boolean required")
    flags = record["termination_flags"]
    if (not isinstance(flags, dict) or set(flags) - REASONS
            or any(type(value) is not bool for value in flags.values())):
        raise InvalidTrajectory("invalid termination_flags")
    if record["sample_kind"] == "pre_reset" and set(flags) != REASONS:
        raise InvalidTrajectory("pre-reset termination_flags incomplete")
    if any(flags.values()) and not record["terminated"]:
        raise InvalidTrajectory("termination flags contradict terminated")
    if record["contact_history_samples"] != 2:
        raise InvalidTrajectory("expected two contact history samples per policy tick")
    if (record["non_wheel_net_force_max_n"] < 0 or any(f < 0 for f in record["wheel_net_force_max_n"])):
        raise InvalidTrajectory("contact magnitudes must be nonnegative")
    for key in ("projected_gravity_b", "root_link_quat_wxyz"):
        if not math.isclose(sum(v*v for v in record[key]), 1., abs_tol=.02):
            raise InvalidTrajectory(f"{key}: expected unit vector/quaternion")


def _hard_failures(record: dict, contract: dict) -> list[str]:
    limits = contract["termination"]
    reasons = [name for name, flag in record["termination_flags"].items() if flag]
    if record["terminated"]:
        reasons.append("terminated")
    if record["timeout"]:
        reasons.append("timeout")
    if record["non_wheel_net_force_max_n"] > limits["contact_force_threshold"]:
        reasons.append("non_wheel_contact")
    if _tilt(record) > limits["max_tilt_deg"]:
        reasons.append("tilt")
    if record["height_m"] < limits["min_base_height"]:
        reasons.append("low_height")
    if record["base_visual_clearance_lower_bound_m"] <= 0:
        reasons.append("base_visual_bounds_ground")
    for joint, (low, high) in contract["joints"]["knee_hard_limits"].items():
        q = record["joint_pos_rad"][contract["joints"]["action_order"].index(joint)]
        if q < low - limits["knee_limit_tolerance"] or q > high + limits["knee_limit_tolerance"]:
            reasons.append("knee_limit")
    return sorted(set(reasons))


def _tilt(record: dict) -> float:
    gravity = record["projected_gravity_b"]
    # Finite/unit checks precede this clamp; this only handles rounding at acos endpoints.
    cosine = -gravity[2] / math.sqrt(sum(v*v for v in gravity))
    return math.degrees(math.acos(max(-1., min(1., cosine))))


def evaluate_trajectory(records: list[dict], contract: dict, stage: str, command,
                        thresholds: Thresholds | None = None) -> dict:
    target = thresholds or Thresholds()
    command = validate_command(contract, stage, command)
    if not 1 <= len(records) <= MAX_RECORDS:
        raise InvalidTrajectory(f"trajectory requires 1..{MAX_RECORDS} records")
    dt = contract["timing"]["policy_dt"]
    physics_dt = contract["timing"]["physics_dt"]
    if (not math.isclose(dt, .01, abs_tol=1e-12) or not math.isclose(physics_dt, .005, abs_tol=1e-12)
            or contract["timing"]["decimation"] != 2):
        raise InvalidTrajectory("only initial V40 .01s policy / .005s physics timing is supported")
    requested_steps = target.duration_s / dt
    if not math.isclose(requested_steps, round(requested_steps), abs_tol=1e-7):
        raise InvalidTrajectory("duration must be an exact policy-tick multiple")
    first = records[0]
    for i, record in enumerate(records):
        validate_record(record)
        if record["sample_kind"] != ("initial" if i == 0 else "pre_reset"):
            raise InvalidTrajectory("one initial anchor then exclusively pre-reset snapshots required")
        if record["episode_step"] != i:
            raise InvalidTrajectory("episode clock reset/gap: possible auto-reset leakage")
        for key, expected in (("policy_dt_s", dt), ("physics_dt_s", physics_dt), ("episode_time_s", i * dt),
                              ("time_s", record["policy_tick"] * dt),
                              ("sim_time_s", record["physics_steps"] * physics_dt)):
            if not math.isclose(record[key], expected, rel_tol=1e-6, abs_tol=1e-6):
                raise InvalidTrajectory(f"{key}: clock/time-unit mismatch")
        if (record["policy_tick"] != first["policy_tick"] + i
                or record["physics_steps"] != first["physics_steps"] + 2*i):
            raise InvalidTrajectory("nonmonotonic/gapped policy or physics clock")
        if any(not math.isclose(a, b, abs_tol=1e-6, rel_tol=0) for a, b in zip(record["command"], command)):
            raise InvalidTrajectory("command sampler overwrote fixed command or mixed cases")
        if i and (records[i-1]["terminated"] or records[i-1]["timeout"]):
            raise InvalidTrajectory("records after terminal/timeout: auto-reset leakage")
    observed = (len(records)-1)*dt
    if observed > target.duration_s + 1e-7:
        raise InvalidTrajectory("trajectory exceeds bounded requested duration")
    hard = [_hard_failures(record, contract) for record in records]
    failures = {reason for reasons in hard for reason in reasons}
    if observed + 1e-7 < target.duration_s:
        failures.add("short_episode")
    first_failure = next((i for i, reasons in enumerate(hard) if reasons), None)
    # Conservative uninterrupted duration excludes the entire first failing policy interval.
    valid_duration = observed if first_failure is None else max(0, first_failure-1)*dt
    samples = records[1:]  # reset anchor has no executed action/physics outcome
    steady = [r for r in samples if r["episode_time_s"] >= target.steady_start_s]
    if not steady:
        failures.add("no_steady_window")
    tracking = {}
    for name, key, component, column, unit in (
        ("vx", "root_com_lin_vel_b_m_s", 0, 0, "m_s"),
        ("wz", "root_com_ang_vel_b_rad_s", 2, 1, "rad_s"),
        ("height", "height_m", None, 2, "m"),
    ):
        def error(record):
            return (record[key] if component is None else record[key][component]) - command[column]
        rmse = math.sqrt(sum(error(r)**2 for r in samples)/len(samples)) if samples else None
        bias = sum(error(r) for r in steady)/len(steady) if steady else None
        tracking[name] = {"rmse": rmse, "steady_bias": bias, "unit": unit}
        if rmse is not None and rmse > getattr(target, f"{name}_rmse_{unit}"):
            failures.add(name + "_rmse")
        if bias is not None and abs(bias) > getattr(target, f"{name}_steady_bias_{unit}"):
            failures.add(name + "_steady_bias")
    anchor = first["wheel_axis_midpoint_w_m"][:2]
    drifts = [math.dist(anchor, r["wheel_axis_midpoint_w_m"][:2]) for r in records]
    drift_applies = command[0] == 0.0  # stand and low-rate rotation; not a position command claim
    drift = {"applicable": drift_applies, "anchor_xy_m": anchor,
             "max_m": max(drifts), "final_m": drifts[-1], "anchor": "initial wheel link-axis midpoint"}
    if drift_applies and (max(drifts) > target.max_drift_m or drifts[-1] > target.final_drift_m):
        failures.add("wheel_axis_drift")
    peak_tilt = max(_tilt(r) for r in records)
    if peak_tilt > target.peak_tilt_deg:
        failures.add("peak_tilt_target")
    joint_metrics = {}
    for index, name in enumerate(contract["joints"]["action_order"]):
        profile = contract["actuators"]["leg" if index in contract["joints"]["leg_indices"] else "wheel"]
        prior = finite(profile["effort_limit"], "effort prior")
        if prior <= 0:
            raise InvalidTrajectory("positive effort prior required")
        util = [abs(r["sim_joint_effort_nm"][index])/prior for r in samples]
        item = {"peak_abs_q_rad": max(abs(r["joint_pos_rad"][index]) for r in records),
                "peak_abs_dq_rad_s": max(abs(r["joint_vel_rad_s"][index]) for r in records),
                "effort_prior_nm": prior, "peak_effort_prior_utilization": max(util) if util else None,
                "effort_prior_ge_99pct_sample_fraction": sum(u >= .99 for u in util)/len(util) if util else None}
        bounds = contract["joints"]["knee_hard_limits"].get(name)
        if bounds is not None:
            low, high = bounds
            qs = [r["joint_pos_rad"][index] for r in records]
            item.update({"hard_limits_rad": bounds, "minimum_hard_limit_margin_rad": min(min(q-low, high-q) for q in qs),
                         "peak_hard_range_utilization": max(abs(q-(low+high)/2)/((high-low)/2) for q in qs)})
        else:
            item["hard_limit_status"] = "continuous_joint_no_finite_position_boundary"
        if index in contract["joints"]["hip_indices"]:
            nominal = contract["joints"]["nominal_positions"][index]
            item["peak_hip_soft_deviation_utilization"] = max(
                abs(math.atan2(math.sin(r["joint_pos_rad"][index]-nominal), math.cos(r["joint_pos_rad"][index]-nominal)))
                / contract["joints"]["hip_soft_deviation"] for r in records)
        if util and max(util) > target.max_effort_prior_utilization:
            failures.add("effort_prior_utilization")
        joint_metrics[name] = item
    last = records[-1]
    end_reason = ("terminated_and_timeout" if last["terminated"] and last["timeout"] else
                  "terminated" if last["terminated"] else "timeout" if last["timeout"] else
                  "horizon_reached" if observed + 1e-7 >= target.duration_s else "incomplete")
    return {
        "schema_version": 1, "passed": not failures, "stage": stage, "command": list(command),
        "failure_reasons": sorted(failures), "end_reason": end_reason,
        "observed_duration_s": observed, "sustained_valid_duration_s": valid_duration,
        "first_failure_episode_time_s": None if first_failure is None else first_failure*dt,
        "sample_count": len(samples), "tracking": tracking, "wheel_axis_drift": drift,
        "peak_tilt_deg": peak_tilt,
        "non_wheel_contact_policy_ticks": sum(r["non_wheel_net_force_max_n"] > contract["termination"]["contact_force_threshold"] for r in samples),
        "peak_non_wheel_net_force_n": max(r["non_wheel_net_force_max_n"] for r in records),
        "peak_wheel_net_force_n": [max(r["wheel_net_force_max_n"][i] for r in records) for i in range(2)],
        "minimum_base_visual_clearance_lower_bound_m": min(r["base_visual_clearance_lower_bound_m"] for r in records),
        "joints": joint_metrics, "thresholds": target.to_dict(), "limitations": deepcopy(LIMITATIONS),
    }


def aggregate_cases(cases: list[dict]) -> dict:
    """Any failed/invalid/missing case fails the suite. Never average away a fall."""
    passed = bool(cases) and all(case.get("passed") is True for case in cases)
    return {"passed": passed, "case_count": len(cases),
            "failed_case_ids": [case.get("case_id", f"case_{i}") for i, case in enumerate(cases) if case.get("passed") is not True],
            "cases": deepcopy(cases), "limitations": deepcopy(LIMITATIONS)}


def collect_case(raw_env, wrapped_env, policy: Callable, steps: int, is_running: Callable,
                 records: list[dict]) -> str:
    """Bounded adapter, also tested with fake snapshots/clock; never reads live post-reset state."""
    if type(steps) is not int or not 1 <= steps < MAX_RECORDS or records:
        raise InvalidTrajectory("collector requires an empty list and bounded positive step count")
    initial = snapshot_to_record(raw_env.capture_evaluation_initial_snapshot())
    records.append(initial)
    validate_record(initial)
    obs = wrapped_env.get_observations()
    for _ in range(steps):
        if not is_running():
            return "application_stopped"
        obs, _, _, _ = wrapped_env.step(policy(obs["policy"]))
        record = snapshot_to_record(raw_env.get_evaluation_snapshot())
        records.append(record)  # Retain invalid evidence for a diagnostic artifact before rejecting it.
        validate_record(record)
        if record["terminated"] or record["timeout"]:
            return "terminated" if record["terminated"] else "timeout"
    return "horizon_reached"


def diagnostic_json(value):
    """Losslessly label invalid numbers for failure artifacts; NEVER a metrics input repair."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_numeric": repr(value)}
    if isinstance(value, dict):
        return {key: diagnostic_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json(child) for child in value]
    return value


def write_json_exclusive(path: Path, data: dict) -> None:
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
