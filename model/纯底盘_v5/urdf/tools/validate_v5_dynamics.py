#!/usr/bin/env python3
"""Engine-driven V5 closure and nonlinear spring-force checks, without passive pose writes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def spring_force(compression, fit):
    # Mechanical stop constraints, rather than force clipping, limit the slider.
    # The reserve 90..100% uses explicitly labeled polynomial extrapolation.
    u = np.clip(compression / fit["stroke_m"], 0., 1.)
    return float(np.polynomial.polynomial.polyval(u, fit["monomial_coefficients_n"]))


def run(bundle, steps=8000):
    spec = json.loads((bundle / "model_spec.json").read_text())
    fit = json.loads((bundle / "fit_10mpa.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(bundle / "robot.xml"))
    data = mujoco.MjData(model)
    site_ids = [(model.site(c["name"] + "_0").id, model.site(c["name"] + "_1").id) for c in spec["constraints"]]
    spring_ids = {name: (model.joint(name).qposadr[0], model.actuator(name + "_gas_force").id)
                  for name in spec["spring_binding"]}
    results = {}
    for case in ("gravity_no_spring", "gravity_with_10mpa_spring", "small_motor_effort_with_spring", "constraints_disabled_control"):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        if case == "constraints_disabled_control":
            data.eq_active[:] = 0
        mujoco.mj_forward(model, data)
        count = min(steps, 600) if case == "constraints_disabled_control" else steps
        result = {"steps": count, "seconds": count * model.opt.timestep, "max_pin_gap_m": 0.,
                  "max_speed": 0., "max_abs_energy_j": 0., "reserve_extrapolation_steps": 0,
                  "spring_compression_min_m": [1., 1.], "spring_compression_max_m": [-1., -1.],
                  "samples": []}
        for step in range(count):
            data.ctrl[:] = 0
            if case == "small_motor_effort_with_spring":
                data.ctrl[:6] = .04 * np.sin(step * model.opt.timestep * 2 * np.pi)
            compression = []
            force = []
            for name, (q_id, actuator_id) in spring_ids.items():
                s = spec["spring_binding"][name]["compression_at_q_zero_m"] - data.qpos[q_id]
                f = spring_force(s, fit) if case != "gravity_no_spring" else 0.
                data.ctrl[actuator_id] = f
                compression.append(float(s))
                force.append(f)
            result["reserve_extrapolation_steps"] += int(max(compression) > fit["max_model_compression_m"])
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            gap = max(np.linalg.norm(data.site_xpos[a] - data.site_xpos[b]) for a, b in site_ids)
            if not np.isfinite(np.r_[data.qpos, data.qvel, data.energy]).all():
                raise RuntimeError(f"Nonfinite state in {case}")
            result["max_pin_gap_m"] = max(result["max_pin_gap_m"], float(gap))
            result["max_speed"] = max(result["max_speed"], float(np.abs(data.qvel).max()))
            result["max_abs_energy_j"] = max(result["max_abs_energy_j"], float(np.abs(data.energy).max()))
            result["spring_compression_min_m"] = np.minimum(result["spring_compression_min_m"], compression).tolist()
            result["spring_compression_max_m"] = np.maximum(result["spring_compression_max_m"], compression).tolist()
            if step % 200 == 0 or step == count - 1:
                result["samples"].append({"time": data.time, "pin_gap_m": float(gap),
                    "base_height_m": float(data.qpos[2]), "compression_m": compression, "force_n": force})
        result["warnings"] = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
        result["passed"] = (result["max_pin_gap_m"] > .01 if case == "constraints_disabled_control" else
                            result["max_pin_gap_m"] < .001 and result["max_abs_energy_j"] < 1000 and not result["warnings"])
        results[case] = result
    return {"model_sha256": hashlib.sha256((bundle / "robot.xml").read_bytes()).hexdigest(),
            "mujoco_version": mujoco.__version__, "passive_pose_writes_after_reset": 0,
            "per_step_kinematic_solver_calls": 0,
            "reserve_region_model": "explicit_cubic_extrapolation_90_to_100_percent_not_catalogue_validated",
            "mount_reference": "CAD_endpoint_reconstruction_not_hardware_calibration",
            "cases": results, "passed": all(r["passed"] for r in results.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    args = parser.parse_args()
    if not 100 <= args.steps <= 20000:
        parser.error("steps must be 100..20000")
    result = run(args.bundle, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "cases": {name: {k: v for k, v in case.items() if k != "samples"}
                      for name, case in result["cases"].items()}}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
