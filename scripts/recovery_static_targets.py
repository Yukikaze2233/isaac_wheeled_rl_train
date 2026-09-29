#!/usr/bin/env python3
"""Compute V6 recovery references on flat ground without changing frozen assets.

Geometry is a closed-chain, COM-balanced standing pose. The separate PD command
includes the static load divided by leg stiffness; it is not a reset pose. The
gas-spring curve is a catalogue prior, so these are candidates for dynamic A/B
validation, not identified hardware compensation or proven recovery targets.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

_PROFILE_PATH = Path(__file__).with_name("build_v6_recovery_profiles.py")
_SPEC = importlib.util.spec_from_file_location("v6_recovery_geometry", _PROFILE_PATH)
_PROFILES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROFILES)


class RecoveryStaticTargetBuilder(_PROFILES.RecoveryProfileBuilder):
    """Use the sealed V6 assembly branch and its existing geometry validation."""

    def __init__(self, bundle=_PROFILES.DEFAULT_ASSET,
                 manifest_sha256=_PROFILES.DEFAULT_MANIFEST_SHA256):
        super().__init__(bundle, manifest_sha256)
        self.fit = json.loads((self.bundle / "fit_10mpa.json").read_text())
        self.mass = sum(body["mass"] for body in self.spec["bodies"])
        self.names = self.manifest["tree_joint_names"]
        self.active_indices = [self.names.index(name) for name in self.controls]
        self.passive_indices = [i for i in range(len(self.names)) if i not in self.active_indices]

    def mass_and_wheel_centers(self, q, rotation):
        frames = self.fk(self.spec, q)
        center = sum(body["mass"] * (frames[body["name"]][:3, :3] @ body["com"]
                                     + frames[body["name"]][:3, 3])
                     for body in self.spec["bodies"]) / self.mass
        wheels = np.array([rotation @ self.wheel(side, q)[0] for side in ("L", "R")])
        return rotation @ center, wheels

    def balanced_pose(self, knee_deg):
        """Rotate each hip/auxiliary pair together; preserve the closed branch."""
        knee_pose = self.knee_pose(knee_deg)

        def pose(angle):
            return self.common_rotate(knee_pose, angle / self.axis_signs[0],
                                      angle / self.axis_signs[2])

        def balance_error(angle):
            center, wheels = self.mass_and_wheel_centers(pose(angle), np.eye(3))
            return center[0] - wheels[:, 0].mean()

        angle = brentq(balance_error, -.8, .8, xtol=1e-10)
        q = pose(angle)
        _, wheels = self.mass_and_wheel_centers(q, np.eye(3))
        delta = wheels[0] - wheels[1]
        # Compensate the sealed asset's slight left/right vertical asymmetry.
        roll = math.atan2(-delta[2], delta[1])
        rotation = Rotation.from_euler("x", roll).as_matrix()
        bounds = self.collision_bounds(q, rotation)
        wheel_bottoms = [min(row["lower"][2] for row in bounds if row["body"] == side + "_link3")
                         for side in ("L", "R")]
        height = -min(wheel_bottoms)
        if abs(wheel_bottoms[0] - wheel_bottoms[1]) > 1e-7:
            raise ValueError("Both wheel collision surfaces must reach the same flat ground")
        clearance = min(float(row["lower"][2]) + height for row in bounds)
        if clearance < -1e-8:
            raise ValueError("Standing reference penetrates the flat ground")
        center, wheels = self.mass_and_wheel_centers(q, rotation)
        if abs(center[0] - wheels[:, 0].mean()) > 1e-8:
            raise ValueError("COM does not balance over the wheel support line")
        return {"q": q, "rotation": rotation, "root_roll_rad": roll,
                "height_m": height, "com": center, "wheels": wheels,
                "minimum_ground_clearance_m": clearance}

    def static_load(self, pose):
        """Virtual work: gravity minus wheel reactions minus spring effort.

        The six active virtual displacements determine passive displacements
        through the differentiated V6 closure constraints. Root attitude is
        fixed for this local derivative; wheel axle directions are constant on
        this sagittal mechanism, so wheel-center z gives contact displacement.
        """
        q, rotation = pose["q"], pose["rotation"]
        self.validate_q(q)
        columns, gravity_gradient, wheel_gradient = [], [], []
        step = 1e-6
        for name in self.names:
            plus, minus = dict(q), dict(q)
            plus[name] += step
            minus[name] -= step
            columns.append((self.closure_error(self.spec, plus)
                            - self.closure_error(self.spec, minus)) / (2 * step))
            center_plus, wheels_plus = self.mass_and_wheel_centers(plus, rotation)
            center_minus, wheels_minus = self.mass_and_wheel_centers(minus, rotation)
            gravity_gradient.append(self.mass * 9.81 * (center_plus[2] - center_minus[2]) / (2 * step))
            wheel_gradient.append((wheels_plus[:, 2] - wheels_minus[:, 2]) / (2 * step))
        closure_jacobian = np.column_stack(columns)
        mapping = np.zeros((len(self.names), len(self.controls)))
        mapping[self.active_indices] = np.eye(len(self.controls))
        mapping[self.passive_indices] = np.linalg.lstsq(
            closure_jacobian[:, self.passive_indices],
            -closure_jacobian[:, self.active_indices], rcond=1e-8)[0]
        residual = float(np.abs(closure_jacobian @ mapping).max())
        if residual > 1e-7:
            raise ValueError("Invalid closed-chain virtual-work mapping")

        center, wheels = pose["com"], pose["wheels"]
        left_normal = self.mass * 9.81 * (center[1] - wheels[1, 1]) / (wheels[0, 1] - wheels[1, 1])
        normals = np.array([left_normal, self.mass * 9.81 - left_normal])
        if not np.all(normals > 0.):
            raise ValueError("COM lies outside the lateral wheel support span")
        spring_names = self.manifest["spring_joint_names"]
        compression = np.array([self.spec["spring_binding"][name]["compression_at_q_zero_m"] - q[name]
                                for name in spring_names])
        normalized = compression / self.fit["stroke_m"]
        lower, upper = self.fit["normalized_fit_domain"]
        if not np.all((normalized >= lower) & (normalized <= upper)):
            raise ValueError("Static reference requires gas-curve extrapolation")
        forces = np.polynomial.polynomial.polyval(normalized, self.fit["monomial_coefficients_n"])
        gravity = np.asarray(gravity_gradient) @ mapping
        support = normals @ np.asarray(wheel_gradient).T @ mapping
        spring = forces @ mapping[[self.names.index(name) for name in spring_names]]
        torque = gravity - support - spring
        if not np.all(np.isfinite(torque)):
            raise ValueError("Non-finite static load")
        return {"static_hold_torque_nm": torque.tolist(), "gravity_virtual_nm": gravity.tolist(),
                "wheel_support_virtual_nm": support.tolist(), "spring_virtual_nm": spring.tolist(),
                "wheel_normals_n": normals.tolist(), "spring_compression_m": compression.tolist(),
                "spring_force_n": forces.tolist(), "mapping_residual": residual}

    def target(self, height_m, leg_kp=160.):
        if not math.isfinite(height_m) or height_m <= 0.:
            raise ValueError("Height must be finite and positive")
        if not math.isfinite(leg_kp) or leg_kp <= 0.:
            raise ValueError("Leg stiffness must be finite and positive")
        # Keep the numerical continuation bracket just inside both stops;
        # geometry validation still enforces the unchanged 40..105 degree domain.
        knee = brentq(lambda beta: self.balanced_pose(beta)["height_m"] - height_m,
                      40.001, 104.999, xtol=1e-8)
        pose = self.balanced_pose(knee)
        validation = self.validate_q(pose["q"])
        load = self.static_load(pose)
        geometry = np.array([pose["q"][name] for name in self.controls])
        offset = np.asarray(load["static_hold_torque_nm"]) / leg_kp
        # Wheels use velocity control; only leg position references receive an offset.
        wheel_indices = [self.controls.index(name) for name in ("L_joint3", "R_joint3")]
        if np.max(np.abs(offset[wheel_indices])) > 1e-8:
            raise ValueError("A balanced standing pose must not require wheel torque")
        offset[wheel_indices] = 0.
        return {"requested_height_m": height_m, "height_m": float(pose["height_m"]),
                "root_position_m": [0., 0., float(pose["height_m"])],
                "root_quaternion_xyzw": Rotation.from_matrix(pose["rotation"]).as_quat().tolist(),
                "root_roll_rad": pose["root_roll_rad"],
                "full_joint_positions": {name: float(pose["q"][name]) for name in self.names},
                "prepare_reference": {"geometry_control6_rad": geometry.tolist(),
                    "pd_control6_rad": (geometry + offset).tolist(),
                    "hold_torque_nm": load["static_hold_torque_nm"], "kp": leg_kp,
                    "kp_units": "N*m/rad", "pd_offset_control6_rad": offset.tolist(),
                    "assumptions": ["static_two_wheel_support_on_flat_ground", "nominal_catalogue_gas_curve",
                                    "nominal_mass_and_COM", "zero_joint_velocity", "no_actuator_saturation"]},
                "static_load": load,
                "geometry_validation": {**validation,
                    "com_x_minus_wheel_support_m": float(pose["com"][0] - pose["wheels"][:, 0].mean()),
                    "minimum_ground_clearance_m": pose["minimum_ground_clearance_m"]},
                "pd_target_is_reset_pose": False, "dynamic_validation_passed": False}

    def build_targets(self, heights=(.305, .34, .35), leg_kp=160.):
        return {"schema": "v6_recovery_static_targets_v1",
                "asset_manifest_sha256": self.manifest_sha256,
                "model_spec_sha256": _PROFILES.digest(self.bundle / "model_spec.json"),
                "spring_fit_sha256": _PROFILES.digest(self.bundle / "fit_10mpa.json"),
                "solver_sha256": _PROFILES.digest(self.solver_path),
                "fk_sha256": _PROFILES.digest(self.bundle / "tools/v6_mechanism.py"),
                "profile_builder_sha256": _PROFILES.digest(_PROFILE_PATH),
                "generator_sha256": _PROFILES.digest(Path(__file__)),
                "control_joint_names": self.controls, "axis_signs": self.axis_signs,
                "mass_kg": self.mass, "passive_knee_domain_deg": [40., 105.],
                "scope": "static_flat_ground_candidates_with_catalogue_gas_curve; dynamic_AB_validation_required",
                "gas_force_model_identified": self.fit["physical_force_model_identified"],
                "targets": {f"{height:g}": self.target(height, leg_kp) for height in heights}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, nargs="+", default=[.305, .34, .35])
    parser.add_argument("--leg-kp", type=float, default=160.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    builder = RecoveryStaticTargetBuilder()
    if args.output.exists() or args.output.resolve().is_relative_to(builder.bundle.resolve()):
        parser.error("Output must be a new file outside the sealed asset")
    result = builder.build_targets(args.height, args.leg_kp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": _PROFILES.digest(args.output)}))


if __name__ == "__main__":
    main()
