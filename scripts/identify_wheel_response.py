#!/usr/bin/env python3
"""Fit nominal wheel-output dynamics from archived, airborne 1 kHz recordings.

The P=0.6 run alone identifies J/C+/C-/b from non-overlapping integral windows.
All coast and predeclared validation segments, and both other controller gains,
are withheld. This script changes no simulator, controller, or training contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import lfilter

RUNS = {"p06": ("run_20260928T100509Z", .6),
        "p02": ("run_20260928T093805Z", .2),
        "p04": ("run_20260928T140705Z", .4)}
PARAMETERS = ("effective_inertia_kg_m2", "coulomb_positive_nm",
              "coulomb_negative_nm", "viscous_nm_per_rad_s")


def load_run(root: Path, name: str) -> tuple[dict, list, dict]:
    run = root / name
    old = json.loads((run / "fit/wheel_fit.json").read_text())
    plan_path = run / "trajectory-plan.json"
    if not plan_path.exists():
        plan_path = Path(old["source_run"]) / "trajectory-plan.json"
    segments = json.loads(plan_path.read_text())["segments"]
    path = run / "wheel_1khz.npz"
    with np.load(path) as source:
        active = source["phase"] == 2
        keys = ("control_steady_ns", "segment_id", "q_api", "dq_api", "tau_frame_api",
                "torque_fb_api", "wheel_velocity_target_api", "feedback_steady_ns")
        data = {k: source[k][active] for k in keys}
        raw = source["feedback_frame_bytes"][active, 32:48].reshape(-1, 2, 8).astype(np.int32)
    rpm = raw[:, :, 2] * 256 + raw[:, :, 3]
    rpm = np.where(rpm >= 32768, rpm - 65536, rpm)
    data["speed_consistent"] = abs(-rpm / 15.8 * 2 * np.pi / 60 - data["dq_api"][:, 4:6]) < 1e-9
    data["t"] = (data["control_steady_ns"].astype(np.int64)
                 - int(data["control_steady_ns"][0])) * 1e-9
    quality = json.loads((run / "wheel_quality.json").read_text())
    provenance = {"run": name, "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "quality_status": quality["status"], "quality_reasons": quality["reasons"],
                  "plan_path": str(plan_path)}
    return data, segments, provenance


def quiet_carrier(q: np.ndarray, dq: np.ndarray) -> bool:
    return bool(np.max(np.ptp(np.unwrap(q, axis=0), axis=0)) < .03
                and np.max(np.quantile(np.abs(dq), .95, axis=0)) < .15)


def integral_windows(data: dict, plan: list, side: int, size: int = 50):
    """Integrate held samples without differentiating quantized RPM feedback."""
    name, axis = ("left", "right")[side], side + 4
    x, y, groups = [], [], []
    for desc in plan:
        if desc["validation"] or desc["coast"] or not desc["label"].startswith(name + "_"):
            continue
        ids = np.flatnonzero(data["segment_id"] == desc["id"])
        for start in range(0, len(ids) - size, size):
            ii = ids[start:start + size + 1]
            t, w = data["t"][ii], data["dq_api"][ii, axis]
            dt = np.diff(t)
            if dt.min() <= 0 or dt.max() > .002 or abs(w).min() < 1 or w.min() * w.max() <= 0:
                continue
            age = (data["control_steady_ns"][ii].astype(np.int64)
                   - data["feedback_steady_ns"][ii, axis].astype(np.int64)) * 1e-9
            if age.min() < 0 or age.max() > .005 or not data["speed_consistent"][ii, side].all():
                continue
            if not quiet_carrier(data["q_api"][ii, :4], data["dq_api"][ii, :4]):
                continue
            x.append([w[-1] - w[0], np.sum(dt * (w[:-1] > 0)),
                      -np.sum(dt * (w[:-1] < 0)), np.sum(dt * w[:-1])])
            y.append(np.sum(dt * data["tau_frame_api"][ii[:-1], axis]))
            groups.append(desc["id"])
    return np.array(x), np.array(y), np.array(groups)


def fit_integrals(x, y):
    scale = np.sqrt(np.mean(x * x, axis=0))
    normalized = x / np.maximum(scale, 1e-12)
    result = lsq_linear(normalized, y, bounds=(0, np.inf), tol=1e-12)
    return result.x / scale, float(np.linalg.cond(normalized))


def replay(t, target, command, initial, params, kp, closed_loop, coast=False):
    """1 kHz explicit inertia/damping and dry-friction impulse, no extra Fs.

    Static resistance equals moving Coulomb resistance here. The archive shows
    extra presliding behavior; low-speed checks explicitly expose this limitation.
    """
    inertia, cp, cn, damping = (float(v) for v in params)
    prediction = np.empty(len(t))
    prediction[0] = initial
    for k in range(len(t) - 1):
        w = float(prediction[k])
        torque = float(command[k])
        if closed_loop:
            torque = 0. if coast else min(max(kp * (float(target[k]) - w), -4.5), 4.5)
        dt = float(t[k + 1] - t[k])
        free = w + dt * (torque - damping * w) / inertia
        prediction[k + 1] = max(free - dt * cp / inertia, 0.) if free >= 0 else min(free + dt * cn / inertia, 0.)
    return prediction


def wheel_cad_inertia(urdf: Path, link_name: str) -> float:
    """Project the inertial tensor plus parallel-axis term onto the wheel axis."""
    root = ET.parse(urdf).getroot()
    joint = next(j for j in root.findall("joint") if j.find("child").get("link") == link_name)
    node = root.find(f"link[@name='{link_name}']/inertial")
    origin = node.find("origin")
    if not np.allclose(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "), 0):
        raise ValueError("Nonzero inertial rotation needs explicit tensor transformation")
    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
    axis /= np.linalg.norm(axis)
    com = np.fromstring(origin.get("xyz"), sep=" ")
    a = node.find("inertia").attrib
    tensor = np.array([[float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                       [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                       [float(a["ixz"]), float(a["iyz"]), float(a["izz"])]])
    mass = float(node.find("mass").get("value"))
    return float(axis @ tensor @ axis + mass * (com @ com - (axis @ com) ** 2))


def current_channel_audit(data, plan, side):
    """A diagnostic of recorded current, NOT identification of shaft torque lag."""
    name, axis = ("left", "right")[side], side + 4
    ids = [s["id"] for s in plan if s["label"].startswith(name + "_")
           and not s["validation"] and not s["coast"]]
    mask = np.isin(data["segment_id"], ids) & (abs(data["dq_api"][:, axis]) > 1)
    u, measured = data["tau_frame_api"][:, axis], data["torque_fb_api"][:, axis]
    rows = []
    for delay in range(4):
        shifted = np.r_[np.full(delay, u[0]), u[:len(u) - delay]] if delay else u
        for tau_ms in (0., 1., 2., 2.5, 3., 4., 5., 6., 8., 10.):
            a = np.exp(-1. / tau_ms) if tau_ms else 0.
            filtered = lfilter([1. - a], [1., -a], shifted)
            gain = np.dot(filtered[mask], measured[mask]) / np.dot(filtered[mask], filtered[mask])
            error = float(np.sqrt(np.mean((gain * filtered[mask] - measured[mask]) ** 2)))
            rows.append({"delay_ticks": delay, "tau_ms": tau_ms, "dc_gain": float(gain), "rmse_nm": error})
    return {"best_recorded_current_channel": min(rows, key=lambda r: r["rmse_nm"]),
            "direct_command_rmse_nm": float(np.sqrt(np.mean((u[mask] - measured[mask]) ** 2))),
            "applied_to_motion_model": False,
            "reason": "Current reporting dynamics and shaft torque dynamics are not independently separated; gain is inconsistent with steady platforms."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {key: load_run(args.data_root, name) for key, (name, _) in RUNS.items()}
    result = {"schema": "wheel_response_calibration_v1", "status": "candidate_pending_isaac_same_trajectory_check",
              "coordinate": "wheel_output", "ratio_15_8_already_applied": True,
              "force_reference": "nominal_C620_current_to_torque_not_load_cell",
              "controller": {"update_hz": 1000, "target_hz": 50, "ki": 0., "kd": 0., "effort_limit_nm": 4.5},
              "split": {"fit": "p06, nonvalidation noncoast single-wheel, abs speed >=1 rad/s; 50ms integral windows",
                        "same_run_holdout": "all coast and predesignated validation segments",
                        "cross_gain_holdout": "entire p02 and p04 runs; old repeated trajectory family, not prospective blind data",
                        "low_speed": "diagnostic only, excluded from fit; stiction/presliding not fully modeled"},
              "provenance": {k: v[2] for k, v in runs.items()}, "wheels": [],
              "limitations": ["Airborne data do not identify tyre friction or loaded torque-speed/thermal envelopes.",
                              "J is total nominal effective inertia, not armature to add directly to CAD.",
                              "Current-channel lag is not transferred to the actuator without motion validation.",
                              "Constant Coulomb friction does not reproduce all low-speed sticking/history.",
                              "p04 incomplete seal/timing gaps remain in provenance; only continuous cases scored."]}
    series = {}
    for side, name in enumerate(("left", "right")):
        axis = side + 4
        d, plan, _ = runs["p06"]
        x, y, groups = integral_windows(d, plan, side)
        fitted, condition = fit_integrals(x, y)
        cad = wheel_cad_inertia(args.urdf, ("L_link3", "R_link3")[side])
        if fitted[0] < cad:
            raise ValueError("Negative inferred armature; review coordinate/inertia reference")
        sensitivity = {}
        for size in (20, 100):
            xx, yy, _ = integral_windows(d, plan, side, size)
            sensitivity[str(size) + "ms"] = dict(zip(PARAMETERS, fit_integrals(xx, yy)[0].tolist()))
        rng, samples = np.random.default_rng(290929 + side), []
        unique = np.unique(groups)
        for _ in range(args.bootstrap):
            selected = np.concatenate([np.flatnonzero(groups == group) for group in rng.choice(unique, len(unique), replace=True)])
            samples.append(fit_integrals(x[selected], y[selected])[0])
        spread = np.quantile(samples, [.05, .95], axis=0).T.tolist() if samples else []
        wheel = {"side": name, "joint_name": ("L_joint3", "R_joint3")[side],
                 "fit": dict(zip(PARAMETERS, fitted.tolist())), "cad_axial_inertia_kg_m2": cad,
                 "candidate_added_armature_kg_m2": float(fitted[0] - cad),
                 "runtime_inertia_must_be_confirmed": True, "integral_windows": len(y),
                 "training_segment_ids": unique.tolist(), "normalized_condition": condition,
                 "integral_rmse_nm_s": float(np.sqrt(np.mean((x @ fitted - y) ** 2))),
                 "segment_bootstrap_p05_p95": dict(zip(PARAMETERS, spread)),
                 "bootstrap_scope": "conditional segment-resampling sensitivity, not absolute physical uncertainty",
                 "window_sensitivity": sensitivity, "current_channel": current_channel_audit(d, plan, side),
                 "checks": []}
        for key, (d, plan, _) in runs.items():
            kp = RUNS[key][1]
            for desc in plan:
                own = desc["label"].startswith(name + "_")
                both = desc["label"].startswith("both_")
                low = (own and "low_" in desc["label"] and max(abs(desc["from"]), abs(desc["to"])) <= .5
                       and desc["waveform"] in (0, 1))
                coast = own and desc["label"].endswith("zero_current_release")
                validation = desc["validation"] and (own or both) and not desc["coast"]
                if not (low or coast or validation):
                    continue
                ii = np.flatnonzero(d["segment_id"] == desc["id"])
                if len(ii) < 30:
                    continue
                if coast:
                    ii = ii[(d["t"][ii] - d["t"][ii[0]] >= .02) & (abs(d["dq_api"][ii, axis]) > .2)]
                    split = np.flatnonzero((np.diff(ii) != 1) | (np.diff(d["t"][ii]) > .002))
                    if len(split):
                        ii = ii[:split[0] + 1]
                if len(ii) < 30:
                    continue
                t, measured = d["t"][ii] - d["t"][ii[0]], d["dq_api"][ii, axis]
                check = {"run": key, "segment_id": desc["id"], "label": desc["label"],
                         "kind": "coast" if coast else "low_speed" if low else "validation",
                         "duration_s": float(t[-1]), "speed_rms_rad_s": float(np.sqrt(np.mean(measured ** 2))),
                         "kp": kp, "models": {}}
                if np.diff(t).min() <= 0 or np.diff(t).max() > .002 or not quiet_carrier(d["q_api"][ii, :4], d["dq_api"][ii, :4]):
                    check["excluded"] = "timing gap or moving carrier"
                    wheel["checks"].append(check)
                    continue
                target, command = d["wheel_velocity_target_api"][ii, side], d["tau_frame_api"][ii, axis]
                for model, params in (("cad_no_resistance", [cad, 0., 0., 0.]), ("identified_nominal", fitted)):
                    metrics = {}
                    for closed in (False, True):
                        predicted = replay(t, target, command, measured[0], params, kp, closed, coast)
                        error = predicted - measured
                        mode = "closed_loop" if closed else "recorded_command"
                        metrics[mode + "_rmse_rad_s"] = float(np.sqrt(np.mean(error ** 2)))
                        metrics[mode + "_max_abs_rad_s"] = float(np.max(abs(error)))
                        if key == "p06" and desc["label"].endswith("heldout_multisine") and own:
                            prefix = name + "_" + str(desc["id"]) + "_"
                            series[prefix + "t"] = t
                            series[prefix + "target"] = target
                            series[prefix + "measured"] = measured
                            series[prefix + model + "_" + mode] = predicted
                    check["models"][model] = metrics
                wheel["checks"].append(check)
        wheel["summary"] = []
        for key in RUNS:
            for kind in ("validation", "coast", "low_speed"):
                checks = [c for c in wheel["checks"] if c["run"] == key and c["kind"] == kind and "excluded" not in c]
                if checks:
                    wheel["summary"].append({"run": key, "kind": kind, "case_count": len(checks),
                                            "segment_equal_weight_mean_rmse": {model: {mode: float(np.mean([c["models"][model][mode + "_rmse_rad_s"] for c in checks])) for mode in ("closed_loop", "recorded_command")} for model in ("cad_no_resistance", "identified_nominal")}})
        result["wheels"].append(wheel)
    fit_path = args.output / "wheel_response_fit.json"
    fit_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    candidate = {
        "schema": "wheel_response_candidate_v1", "status": "pending_isaac_validation",
        "fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        "coordinate": "wheel_output", "ratio_15_8_already_applied": True,
        "scope": "nominal_effective_inertia_and_moving_resistance",
        "measured_speed_range_rad_s": [-30., 30.],
        "fit_min_abs_speed_rad_s": 1., "loaded_torque_speed_envelope_identified": False,
        "static_breakaway_identified": False, "time_constant_identified": False,
        "axes": {w["joint_name"]: {
            "effective_total_inertia_kg_m2": w["fit"]["effective_inertia_kg_m2"],
            "cad_inertia_kg_m2": w["cad_axial_inertia_kg_m2"],
            "additional_armature_kg_m2": w["candidate_added_armature_kg_m2"],
            "coulomb_positive_nm": w["fit"]["coulomb_positive_nm"],
            "coulomb_negative_nm": w["fit"]["coulomb_negative_nm"],
            "symmetric_coulomb_approximation_nm": (w["fit"]["coulomb_positive_nm"] + w["fit"]["coulomb_negative_nm"]) / 2.,
            "viscous_nm_per_rad_s": w["fit"]["viscous_nm_per_rad_s"],
        } for w in result["wheels"]},
        "instruction": "Verify actual runtime axial inertia/armature and friction API units before applying; avoid duplicate inertia and friction. Symmetric average is an explicitly declared approximation for native friction APIs."
    }
    (args.output / "wheel_response_candidate.json").write_text(json.dumps(candidate, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "wheel_response_curves.npz", **series)
    print(json.dumps({"output": str(args.output), "wheels": [{"side": w["side"], "fit": w["fit"], "added_armature": w["candidate_added_armature_kg_m2"], "summary": w["summary"]} for w in result["wheels"]]}, indent=2))


if __name__ == "__main__":
    main()
