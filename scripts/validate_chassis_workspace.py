#!/usr/bin/env python3
"""Audit commanded heights at mass/COM corners using closed-chain virtual work."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from analyze_v5_spring_limits import balanced_standing_pose, side_geometry
from compare_v5_spring_load import prepare_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "model/纯底盘_v5_232mm/urdf")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    load = lambda name: json.loads((args.bundle / name).read_text())
    original, manifest, fit = load("model_spec.json"), load("manifest.json"), load("fit_10mpa.json")
    profiles = [("nominal", 1., 1., (0., 0., 0.))]
    for label, base, other in (("light", .9, .9), ("heavy", 1.3, 1.1)):
        profiles += [(f"{label}_corner_{i}", base, other, xyz)
                     for i, xyz in enumerate(itertools.product((-.04, .04), (-.02, .02), (-.02, .02)))]
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "symmetric_balanced_quasistatic_samples_not_dynamic_or_collision_certification",
        "sampled_heights_m": [.23, .305, .43], "cases": [],
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "scripts/validate_chassis_workspace.py", "scripts/compare_v5_spring_load.py",
            "tools/analyze_v5_spring_limits.py", "tools/build_v5_closedchain.py", "tools/v5_mechanism.py")},
        "asset_manifest_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, base_scale, other_scale, offset in profiles:
        spec = deepcopy(original)
        for body in spec["bodies"]:
            scale = base_scale if body["name"] == "base_link" else other_scale
            body["mass"] *= scale
            body["inertia"] = (np.asarray(body["inertia"]) * scale).tolist()
            if body["name"] == "base_link":
                body["com"] = (np.asarray(body["com"]) + offset).tolist()
        _, limits = side_geometry(spec, "L")
        low, high = limits["mechanical_minimum_knee_deg"], limits["mechanical_maximum_knee_deg"]
        entry = {"profile": name, "base_mass_scale": base_scale, "other_mass_scale": other_scale,
                 "base_com_offset_m": offset, "rows": [], "supported": True}
        for height in report["sampled_heights_m"]:
            try:
                angle = brentq(lambda a: balanced_standing_pose(spec, a)["base_frame_height_m"] - height, low, high)
                pose = balanced_standing_pose(spec, angle)
                trial = prepare_trial(spec, manifest, fit, angle, allow_reserve_extrapolation=True)
                torque = np.asarray(trial["feedforward_nm"][1])
                actions = [(pose["joint_positions"][joint] + torque[i] / 60. - manifest["nominal_joint_pos"][joint]) / .25
                           for i, joint in enumerate(manifest["control_joint_names"]) if i in (0, 1, 3, 4)]
                supported = (bool(np.max(np.abs(torque[[0, 1, 3, 4]])) <= 40.) and max(map(abs, actions)) <= 3.
                             and min(trial["compression_m"]) >= 0 and max(trial["compression_m"]) <= .08)
                entry["rows"].append({"height_m": height, "knee_inner_deg": angle,
                    "torque_nm": torque.tolist(), "leg_actions": actions,
                    "compression_m": trial["compression_m"], "supported": supported})
                entry["supported"] &= supported
            except (ValueError, RuntimeError) as error:
                entry["rows"].append({"height_m": height, "supported": False, "error": str(error)})
                entry["supported"] = False
        report["cases"].append(entry)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(name, "supported=" + str(entry["supported"]), flush=True)
    report.update(status="passed" if all(case["supported"] for case in report["cases"]) else "unsupported_samples",
                  finished_at=datetime.now(timezone.utc).isoformat())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
