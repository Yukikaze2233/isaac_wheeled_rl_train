#!/usr/bin/env python3
"""Generate encoder wheel-center lookup using a hash-verified V6 asset.

All solves use the asset's reviewed solve_pose/fk; only the two solver function
AST nodes are executed from its builder. No source asset is modified. Starting
from nominal, the passive knee is continued independently toward each endpoint
while its auxiliary motor is solved as part of the full closed chain.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = ROOT / "model/纯底盘_v6_232mm_stop105/urdf"
DEFAULT_MANIFEST_SHA256 = "875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LookupBuilder:
    def __init__(self, bundle, manifest_sha256):
        self.bundle = Path(bundle)
        manifest_path = self.bundle / "manifest.json"
        if digest(manifest_path) != manifest_sha256:
            raise ValueError("Asset manifest differs from requested frozen identity")
        self.manifest = json.loads(manifest_path.read_text())
        for name, expected in self.manifest["files_sha256"].items():
            if digest(self.bundle / name) != expected:
                raise ValueError(f"Sealed asset hash mismatch: {name}")
        self.manifest_sha256 = manifest_sha256
        self.spec = json.loads((self.bundle / "model_spec.json").read_text())
        module_path = self.bundle / "tools/v6_mechanism.py"
        module_spec = importlib.util.spec_from_file_location("v6_lookup_mechanism", module_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.fk = module.fk
        self.solver_path = self.bundle / "tools/build_v6_closedchain.py"
        nodes = [node for node in ast.parse(self.solver_path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ("closure_error", "solve_pose")]
        if len(nodes) != 2:
            raise ValueError("Reviewed closure solver functions missing")
        namespace = {"np": np, "fk": module.fk, "point": module.point, "least_squares": least_squares}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(self.solver_path), "exec"), namespace)
        self.solve_pose, self.closure_error = namespace["solve_pose"], namespace["closure_error"]
        self.nominal = self.spec["nominal_joint_pos"]
        self.active = {name: self.nominal[name] for name in self.manifest["control_joint_names"]}
        self.joints = {joint["name"]: joint for joint in self.spec["joints"]}
        self.bodies = {body["name"]: body for body in self.spec["bodies"]}
        self.max_gap_m = 0.

    @staticmethod
    def passive_name(side):
        return "L_joint2" if side == "L" else "R_jonit2"

    def solve_knee(self, side, beta_deg, initial):
        fixed = {key: value for key, value in self.active.items() if key != side * 2 + "_joint1"}
        sign = 1. if side == "L" else -1.
        fixed[self.passive_name(side)] = sign * (np.deg2rad(beta_deg) - (np.pi - 2.3573))
        return self.solve(fixed, initial)

    def solve(self, prescribed, initial):
        q = self.solve_pose(self.spec, prescribed, initial)
        gap = np.linalg.norm(self.closure_error(self.spec, q).reshape(-1, 3), axis=1).max()
        if gap > 1e-8:
            raise ValueError(f"Closure residual {gap} exceeds 10 nm")
        self.max_gap_m = max(self.max_gap_m, float(gap))
        return q

    def wheel(self, side, q):
        frame = self.fk(self.spec, q)[side + "_link3"]
        collision = self.bodies[side + "_link3"]["collisions"][0]
        if collision["type"] != "cylinder":
            raise ValueError("Expected sealed circular wheel collision geometry")
        collision_frame = frame @ np.asarray(collision["origin"])
        axis = collision_frame[:3, 2]
        return collision_frame[:3, 3], axis / np.linalg.norm(axis), collision["radius"]

    def side(self, side, knots):
        beta_grid = np.linspace(40., 105., knots)
        sign = 1. if side == "L" else -1.
        nominal_beta = np.rad2deg(sign * self.nominal[self.passive_name(side)] + np.pi - 2.3573)
        states = [None] * knots
        # Independent continuations retain the nominal assembly branch on both
        # sides; a sweep beginning from a remote endpoint is not branch proof.
        for indices in (np.flatnonzero(beta_grid < nominal_beta)[::-1],
                        np.flatnonzero(beta_grid >= nominal_beta)):
            q = dict(self.nominal)
            for index in indices:
                q = self.solve_knee(side, beta_grid[index], q)
                states[index] = q
        hip_name, auxiliary_name = side + "_joint1", side * 2 + "_joint1"
        hip_reference = self.nominal[hip_name]
        delta = np.asarray([q[auxiliary_name] - q[hip_name] for q in states])
        differences = np.diff(delta)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError(f"{side} relative coordinate is not a single-valued branch")
        order = np.argsort(delta)
        delta, beta_grid = delta[order], beta_grid[order]
        states = [states[index] for index in order]
        points = np.asarray([self.wheel(side, q)[0] for q in states])
        hip = self.joints[hip_name]
        hip_origin = np.asarray(hip["origin"])[:3, 3]
        hip_axis = np.asarray(hip["origin"])[:3, :3] @ np.asarray(hip["axis"])
        hip_axis /= np.linalg.norm(hip_axis)
        _, wheel_axis, radius = self.wheel(side, states[0])
        if abs(abs(wheel_axis @ hip_axis) - 1) > 1e-10:
            raise ValueError("Wheel and hip axes are not parallel; constant-axis observer invalid")
        checks = []
        for index in range(knots - 1):
            value = float((delta[index] + delta[index + 1]) / 2)
            fixed = {**self.active, auxiliary_name: hip_reference + value}
            exact = self.solve(fixed, states[index])
            exact_point = self.wheel(side, exact)[0]
            estimate = (points[index] + points[index + 1]) / 2
            checks.append({"kind": "midpoint", "delta_rad": value,
                           "common_rotation_rad": 0.,
                           "error_mm": float(np.linalg.norm(estimate - exact_point) * 1000)})
        for index in np.linspace(0, knots - 1, 9).astype(int):
            for common in (-np.pi, -np.pi / 2, np.pi / 2, np.pi):
                q = dict(states[index])
                # Follow the same assembly continuously for the independent
                # all-coordinate FK check, including unrestricted hip rotation.
                for angle in np.linspace(0., common, 25)[1:]:
                    fixed = {**self.active, hip_name: hip_reference + angle,
                             auxiliary_name: hip_reference + delta[index] + angle}
                    q = self.solve(fixed, q)
                vector = points[index] - hip_origin
                rotated = (np.cos(common) * vector + np.sin(common) * np.cross(hip_axis, vector)
                           + (1 - np.cos(common)) * hip_axis * (hip_axis @ vector))
                exact_point, exact_axis, _ = self.wheel(side, q)
                if np.linalg.norm(exact_axis - wheel_axis) > 1e-8:
                    raise ValueError("Wheel axle changed under common rotation")
                checks.append({"kind": "common_rotation", "delta_rad": float(delta[index]),
                               "common_rotation_rad": float(common),
                               "error_mm": float(np.linalg.norm(hip_origin + rotated - exact_point) * 1000)})
        maximum = max(row["error_mm"] for row in checks)
        if maximum > .1:
            raise ValueError(f"Lookup/direct FK error {maximum} mm exceeds 0.1 mm")
        return {
            "hip_reference_rad": hip_reference, "hip_origin_b_m": hip_origin.tolist(),
            "hip_axis_b": hip_axis.tolist(), "wheel_axis_b": wheel_axis.tolist(),
            "wheel_radius_m": radius, "delta_rad": delta.tolist(),
            "passive_inner_deg": beta_grid.tolist(), "wheel_center_b_m": points.tolist(),
            "validation": {"max_error_mm": maximum, "bound_mm": .1, "checks": checks},
        }

    def build(self, knots):
        sides = {word: self.side(side, knots) for word, side in (("left", "L"), ("right", "R"))}
        return {
            "schema": "v6_closedchain_height_lookup_v1",
            "asset_manifest_sha256": self.manifest_sha256,
            "model_spec_sha256": digest(self.bundle / "model_spec.json"),
            "solver_sha256": digest(self.solver_path),
            "fk_sha256": digest(self.bundle / "tools/v6_mechanism.py"),
            "generator_sha256": digest(Path(__file__)),
            "control_joint_names": self.manifest["control_joint_names"],
            "coordinate_definition": "V6 model q_aux - q_hip; aligned continuous encoder angles, not raw DM",
            "passive_knee_domain_deg": [40., 105.], "knots_per_side": knots,
            "hardware_calibrated": False,
            "status": "sealed_model_geometry_only_requires_real_encoder_alignment_and_support_estimate",
            "max_closure_gap_m": self.max_gap_m, "sides": sides,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--knots", type=int, default=257)
    parser.add_argument("--output", type=Path, default=ROOT / "contracts/v6_height_lookup_v1.json")
    args = parser.parse_args()
    if args.knots < 3:
        parser.error("At least three lookup knots required")
    if args.output.resolve().is_relative_to(args.asset.resolve()):
        parser.error("Output must be outside the sealed asset")
    table = LookupBuilder(args.asset, args.manifest_sha256).build(args.knots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "max_closure_gap_m": table["max_closure_gap_m"],
                      "max_error_mm": {side: data["validation"]["max_error_mm"]
                                       for side, data in table["sides"].items()}}))


if __name__ == "__main__":
    main()
