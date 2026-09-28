#!/usr/bin/env python3
"""Check coaxial motor-output coupling against compiled closed-chain geometry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model/纯底盘_v5_232mm/urdf"


class MotorCouplingAudit:
    """Offline kinematics only; no pose solver is inserted into the control loop."""

    def __init__(self, bundle=BUNDLE):
        self.bundle = bundle
        self.spec = json.loads((bundle / "model_spec.json").read_text())
        self.manifest = json.loads((bundle / "manifest.json").read_text())
        for name, expected in self.manifest["files_sha256"].items():
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Asset dependency mismatch: {name}")
        if self.manifest["knee_inner_limits_deg"] != [40., 110.]:
            raise ValueError("This audit requires the current 40-110 degree V5.9 asset")
        self.model = mujoco.MjModel.from_xml_path(str(bundle / "robot.xml"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[:3] = 0.
        self.data.qpos[3:7] = [1., 0., 0., 0.]
        self.addresses = {name: self.model.joint(name).qposadr[0]
                          for name in self.manifest["tree_joint_names"]}
        self.sites = np.array([[self.model.site(c["name"] + f"_{end}").id for end in (0, 1)]
                              for c in self.spec["constraints"]])

    def solve(self, prescribed, initial=None):
        q = {**(initial or self.spec["nominal_joint_pos"]), **prescribed}
        unknown = [name for name in self.addresses if name not in prescribed]
        for name, value in q.items():
            self.data.qpos[self.addresses[name]] = value

        def residual(values):
            for name, value in zip(unknown, values):
                self.data.qpos[self.addresses[name]] = value
            mujoco.mj_kinematics(self.model, self.data)
            return (self.data.site_xpos[self.sites[:, 0]] - self.data.site_xpos[self.sites[:, 1]]).ravel()

        solved = least_squares(residual, [q[name] for name in unknown],
            xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=300)
        gap = np.linalg.norm(residual(solved.x).reshape(-1, 3), axis=1).max()
        if not solved.success or gap > 1e-6:
            raise RuntimeError(f"Compiled closure solve failed: gap={gap}, {solved.message}")
        q.update(zip(unknown, solved.x.tolist()))
        return q, float(gap)

    def inner_angle_deg(self, side):
        # Measure the two segments in the leg plane, independently of raw knee coordinates.
        hip = self.data.body(f"{side}_link1").xpos
        knee = self.data.body(f"{side}_link2").xpos
        wheel = self.data.body(f"{side}_link3").xpos
        axis = self.data.xaxis[self.model.joint(f"{side}_joint1").id]
        thigh, shank = hip - knee, wheel - knee
        thigh = thigh - np.dot(thigh, axis) * axis
        shank = shank - np.dot(shank, axis) * axis
        cosine = np.dot(thigh, shank) / (np.linalg.norm(thigh) * np.linalg.norm(shank))
        return float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))

    def run(self):
        nominal = self.spec["nominal_joint_pos"]
        pose_names = ("L_joint1", "L_joint2", "L_joint3", "R_joint1", "R_jonit2", "R_joint3")
        pose = {name: nominal[name] for name in pose_names}
        rows, examples = [], []
        for side, hip, auxiliary, knee, sign in (
                ("L", "L_joint1", "LL_joint1", "L_joint2", 1.),
                ("R", "R_joint1", "RR_joint1", "R_jonit2", -1.)):
            self.solve(pose)
            hip_id, aux_id = self.model.joint(hip).id, self.model.joint(auxiliary).id
            hip_axis, aux_axis = self.data.xaxis[[hip_id, aux_id]]
            separation = self.data.xanchor[aux_id] - self.data.xanchor[hip_id]
            coaxial_gap = float(np.linalg.norm(separation - hip_axis * np.dot(separation, hip_axis)))
            max_follow_error, max_angle_error, max_closure = 0., 0., 0.
            for inner in (40., 50., 67.85497076, 90., 110.):
                fixed = {**pose, knee: sign * (np.radians(inner) - (np.pi - 2.3573))}
                # Continue from nominal to retain the physical assembly branch.
                q = dict(nominal)
                for value in np.linspace(nominal[knee], fixed[knee], 21):
                    q, _ = self.solve({**fixed, knee: float(value)}, q)
                origin_aux = q[auxiliary]
                previous_shift = 0.
                for shift_deg in (-90., -30., -10., 0., 10., 30., 90.):
                    shift = float(np.radians(shift_deg))
                    increments = max(1, int(np.ceil(abs(shift - previous_shift) / np.radians(2.))))
                    for current_shift in np.linspace(previous_shift, shift, increments + 1)[1:]:
                        q, gap = self.solve({**fixed, hip: nominal[hip] + float(current_shift)}, q)
                    previous_shift = shift
                    max_follow_error = max(max_follow_error, abs(q[auxiliary] - origin_aux - shift))
                    max_angle_error = max(max_angle_error, abs(self.inner_angle_deg(side) - inner))
                    max_closure = max(max_closure, gap)
            rows.append({"side": side, "hip": hip, "auxiliary": auxiliary,
                "axis_dot": float(np.dot(hip_axis, aux_axis)), "axis_line_gap_m": coaxial_gap,
                "poses_tested": 35, "max_aux_minus_hip_increment_error_rad": max_follow_error,
                "max_measured_inner_error_deg": max_angle_error, "max_closure_gap_m": max_closure})
            active = {name: nominal[name] for name in self.manifest["control_joint_names"]}
            for label, hip_change, aux_change in (("nominal", 0., 0.), ("both_plus_10deg", 10., 10.),
                                                  ("hip_only_plus_10deg", 10., 0.),
                                                  ("auxiliary_only_plus_10deg", 0., 10.)):
                q, gap = self.solve({**active, hip: nominal[hip] + np.radians(hip_change),
                    auxiliary: nominal[auxiliary] + np.radians(aux_change)})
                examples.append({"side": side, "case": label, "hip_rad": q[hip],
                    "auxiliary_rad": q[auxiliary], "inner_angle_deg": self.inner_angle_deg(side),
                    "closure_gap_m": gap})
            q, gap = self.solve({**pose, knee: nominal[knee] + sign * np.radians(10.)})
            examples.append({"side": side, "case": "hip_held_inner_plus_10deg",
                "hip_rad": q[hip], "auxiliary_rad": q[auxiliary],
                "inner_angle_deg": self.inner_angle_deg(side), "closure_gap_m": gap,
                "active_increments_rad": {name: q[name] - nominal[name]
                                          for name in self.manifest["control_joint_names"]}})
        passed = all(row["axis_dot"] > 1 - 1e-10 and row["axis_line_gap_m"] < 1e-6
            # Exported root axes have sub-micrometre CAD rounding offsets.
            and row["max_aux_minus_hip_increment_error_rad"] < np.radians(.001)
            and row["max_measured_inner_error_deg"] < 1e-5 for row in rows)
        return {"status": "passed" if passed else "failed",
            "generated_at": datetime.now(timezone.utc).isoformat(), "mujoco_version": mujoco.__version__,
            "scope": "compiled_output_axis_kinematics_not_hardware_chain_calibration",
            "bundle": str(self.bundle.relative_to(ROOT)), "follow_tolerance_deg": .001,
            "source_sha256": {name: hashlib.sha256((self.bundle / name).read_bytes()).hexdigest()
                              for name in ("manifest.json", "model_spec.json", "robot.xml", "robot.usda")},
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "results": rows, "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = MotorCouplingAudit(args.bundle).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
