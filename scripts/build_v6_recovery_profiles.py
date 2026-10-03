#!/usr/bin/env python3
"""Freeze reset-only V6 recovery poses from the sealed closed-chain geometry.

No old V5 joint values are imported. Geometrically valid means closed linkage,
legal spring/knee coordinates and no terrain penetration; it does not mean that
a recovery controller has successfully recovered the pose.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("v6_lookup_builder", Path(__file__).with_name("build_v6_height_lookup.py"))
_LOOKUP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOOKUP)
LookupBuilder = _LOOKUP.LookupBuilder
digest = _LOOKUP.digest
DEFAULT_ASSET = _LOOKUP.DEFAULT_ASSET
DEFAULT_MANIFEST_SHA256 = _LOOKUP.DEFAULT_MANIFEST_SHA256
LEG_NAMES = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")
FLAT_POSES = {
    "upright": (0., 0., 0.), "crouched": (0., 0., 0.),
    "upright_release": (0., 0., .2), "crouched_release": (0., 0., .2),
    "left_side_90": (90., 0., 1.), "right_side_90": (-90., 0., 1.),
    "pitch_forward_45": (0., 45., .2), "pitch_backward_45": (0., -45., .2),
    "front_down_90": (0., 90., 1.), "back_down_90": (0., -90., 1.),
    "front_down_left_forward_right_back_90": (0., 90., 1.),
    "front_down_left_back_right_forward_90": (0., 90., 1.),
    "back_down_left_forward_right_back_90": (0., -90., 1.),
    "back_down_left_back_right_forward_90": (0., -90., 1.),
    "prone_belly_legs_above": (0., 0., 1.),
    "prone_back_legs_above": (180., 0., 1.),
}


class RecoveryProfileBuilder(LookupBuilder):
    def __init__(self, bundle=DEFAULT_ASSET, manifest_sha256=DEFAULT_MANIFEST_SHA256):
        super().__init__(bundle, manifest_sha256)
        self.controls = self.manifest["control_joint_names"]
        self.knee_zero_rad = math.radians(40.) - float(self.joints["L_joint2"]["limit"]["lower"])
        self.meshes = {}
        for body in self.spec["bodies"]:
            for collision in body["collisions"]:
                if collision["type"] == "mesh":
                    path = self.bundle / collision["file"]
                    self.meshes[collision["file"]] = np.asarray([
                        [float(x) for x in line.split()[1:4]]
                        for line in path.read_text().splitlines() if line.startswith("v ")])
        self.axes = {name: np.asarray(self.joints[name]["origin"])[:3, :3]
                     @ np.asarray(self.joints[name]["axis"]) for name in LEG_NAMES}
        if any(abs(axis[1]) < .99 for axis in self.axes.values()):
            raise ValueError("Expected sagittal V6 active axes")
        self.axis_signs = [float(np.sign(self.axes[name][1])) for name in LEG_NAMES]
        for i in (0, 2):
            if self.axis_signs[i] != self.axis_signs[i + 1]:
                raise ValueError("Hip and auxiliary must have the same coordinate direction")

    def knees(self, q):
        zero = self.knee_zero_rad
        return [math.degrees(q["L_joint2"] + zero), math.degrees(-q["R_jonit2"] + zero)]

    def validate_q(self, q):
        knees = self.knees(q)
        if any(not 40. - 1e-5 <= beta <= 105. + 1e-5 for beta in knees):
            raise ValueError(f"V6 passive knee outside 40..105 degrees: {knees}")
        compressions = {}
        for name, binding in self.spec["spring_binding"].items():
            compression = binding["compression_at_q_zero_m"] - q[name]
            limit = self.joints[name]["limit"]
            if not (float(limit["lower"]) - 1e-8 <= q[name] <= float(limit["upper"]) + 1e-8
                    and -1e-8 <= compression <= binding["stroke_m"] + 1e-8):
                raise ValueError(f"Spring outside physical travel: {name}")
            compressions[name] = compression
        gap = np.linalg.norm(self.closure_error(self.spec, q).reshape(-1, 3), axis=1).max()
        if gap > 1e-8:
            raise ValueError(f"Open closed chain: {gap}")
        return {"passive_knee_deg": knees, "spring_compression_m": compressions,
                "max_closure_gap_m": float(gap)}

    def knee_pose(self, left_beta, right_beta=None):
        right_beta = left_beta if right_beta is None else right_beta
        q = dict(self.nominal)
        start = self.knees(q)
        count = max(2, math.ceil(max(abs(left_beta - start[0]), abs(right_beta - start[1])) / 2.) + 1)
        for beta in np.linspace(start, [left_beta, right_beta], count)[1:]:
            fixed = {name: self.nominal[name] for name in self.controls if name not in ("LL_joint1", "RR_joint1")}
            fixed["L_joint2"] = math.radians(beta[0]) - self.knee_zero_rad
            fixed["R_jonit2"] = self.knee_zero_rad - math.radians(beta[1])
            q = self.solve(fixed, q)
            self.validate_q(q)
        return q

    def common_rotate(self, q, left_rad=0., right_rad=0.):
        """Move both active axes together along the same closure branch."""
        start = dict(q)
        count = max(2, math.ceil(max(abs(left_rad), abs(right_rad)) / .2) + 1)
        for fraction in np.linspace(0., 1., count)[1:]:
            fixed = {name: start[name] for name in self.controls}
            for names, shift in ((("L_joint1", "LL_joint1"), left_rad),
                                 (("R_joint1", "RR_joint1"), right_rad)):
                for name in names:
                    fixed[name] += fraction * shift
            q = self.solve(fixed, q)
            self.validate_q(q)
        return q

    def align_wheels(self, q, upward=False):
        """Aim each hip-to-wheel vector vertically, using this asset's axes."""
        shifts = []
        for side in ("L", "R"):
            center = self.wheel(side, q)[0]
            pivot = np.asarray(self.joints[side + "_joint1"]["origin"])[:3, 3]
            vector = center - pivot
            # Ry(theta) gives x'=cos(theta)*x+sin(theta)*z.
            physical = math.atan2(vector[0], -vector[2])
            if upward:
                physical = (physical + math.pi + math.pi) % (2 * math.pi) - math.pi
            shifts.append(physical / float(np.sign(self.axes[side + "_joint1"][1])))
        return self.common_rotate(q, *shifts)

    def collision_bounds(self, q, rotation):
        """Exact XYZ extrema of mesh hulls and analytic finite cylinders."""
        frames = self.fk(self.spec, q)
        rows = []
        for body in self.spec["bodies"]:
            for index, collision in enumerate(body["collisions"]):
                transform = frames[body["name"]] @ np.asarray(collision["origin"])
                rot = rotation @ transform[:3, :3]
                pos = rotation @ transform[:3, 3]
                if collision["type"] == "mesh":
                    points = self.meshes[collision["file"]] @ rot.T + pos
                    lower, upper = points.min(axis=0), points.max(axis=0)
                elif collision["type"] == "cylinder":
                    axis = rot[:, 2]
                    extent = collision["radius"] * np.sqrt(np.maximum(0., 1. - axis * axis))
                    extent += collision["length"] * .5 * np.abs(axis)
                    lower, upper = pos - extent, pos + extent
                    # Circumscribed polygon contains the analytic cylinder;
                    # half-space clipping therefore remains conservative.
                    angles = np.linspace(0., 2 * math.pi, 64, endpoint=False)
                    radius = collision["radius"] / math.cos(math.pi / 64)
                    rim = radius * (np.cos(angles)[:, None] * rot[:, 0] + np.sin(angles)[:, None] * rot[:, 1])
                    half_axis = collision["length"] * .5 * rot[:, 2]
                    points = np.concatenate((pos + rim - half_axis, pos + rim + half_axis))
                else:
                    raise ValueError(f"Unsupported collision primitive: {collision['type']}")
                rows.append({"body": body["name"], "index": index,
                             "lower": lower, "upper": upper, "points": points})
        return rows

    @staticmethod
    def halfspace_min_z(points, edge):
        """Minimum z of a convex vertex hull restricted to x >= edge.

        All crossing vertex pairs include every actual hull edge. Their extra
        interior chords cannot produce a lower point outside the hull, so this
        gives the exact restricted minimum without a topology dependency.
        """
        high = points[points[:, 0] >= edge]
        low = points[points[:, 0] < edge]
        if not len(high):
            return None
        minimum = float(high[:, 2].min())
        if len(low):
            fraction = (edge - low[:, None, 0]) / (high[None, :, 0] - low[:, None, 0])
            crossing_z = low[:, None, 2] + fraction * (high[None, :, 2] - low[:, None, 2])
            minimum = min(minimum, float(crossing_z.min()))
        return minimum

    def profile(self, name, q, roll=0., pitch=0., release=0., *, root_x=0., platforms=()):
        rotation = Rotation.from_euler("xyz", [roll, pitch, 0.], degrees=True)
        rows = self.collision_bounds(q, rotation.as_matrix())
        required = []
        for row in rows:
            needed = -float(row["lower"][2])
            for edge, height in platforms:
                minimum = self.halfspace_min_z(row["points"], edge - root_x)
                if minimum is not None:
                    needed = max(needed, height - minimum)
            required.append(needed)
        root_z = max(required) + .003
        support_index = int(np.argmax(required))
        world_centers = [rotation.apply(self.wheel(side, q)[0]) + [root_x, 0., root_z]
                         for side in ("L", "R")]
        lows = {body["name"]: min(row["lower"][2] for row in rows if row["body"] == body["name"])
                for body in self.spec["bodies"]}
        shell_top = max(row["upper"][2] for row in rows if row["body"] == "base_link")
        result = {
            "name": name, "enabled": True, "geometry_valid": True,
            "dynamic_recovery_validated": False,
            "terrain": "step_up" if platforms else "flat",
            "source": "sealed_V6_geometry_candidate_not_recorded_failure_trajectory",
            "root_position_m": [root_x, 0., root_z],
            "root_quaternion_xyzw": rotation.as_quat().tolist(),
            "root_linear_velocity_m_s": [0., 0., 0.], "root_angular_velocity_rad_s": [0., 0., 0.],
            "full_joint_positions": {key: float(q[key]) for key in self.manifest["tree_joint_names"]},
            "full_joint_velocities": {key: 0. for key in self.manifest["tree_joint_names"]},
            "control6_rad": [q[key] for key in self.controls],
            "initial_release_s": release, "state_writes": "once_at_episode_reset_only",
            "terrain_platforms": [{"start_x_m": x, "height_m": h} for x, h in platforms],
            "common_winding_policy": "preserve_continuous_feedback_and_paired_equivalent_reference_turns; never_wrap_script_PD_error",
            "geometry_validation": {**self.validate_q(q), "minimum_terrain_clearance_m": .003,
                "clearance_method": "exact_flat_support; convex_mesh_halfspace_and_circumscribed_cylinder_64",
                "lowest_support_body": rows[support_index]["body"],
                "wheel_centers_world_m": [v.tolist() for v in world_centers],
                "wheel_bottom_clearance_over_center_surface_m": [float(root_z + lows[side + "_link3"] - max([0.] + [height for edge, height in platforms if center[0] >= edge])) for side, center in zip(("L", "R"), world_centers)],
                "wheel_bottom_above_shell_top_m": [float(lows[s + "_link3"] - shell_top) for s in ("L", "R")],
                "collision_primitive_count": len(rows), "self_collision_free_proven": False},
        }
        return result

    def build(self, include_steps=True):
        nominal = dict(self.nominal)
        fold = self.align_wheels(self.knee_pose(50.))
        thrust = self.align_wheels(self.knee_pose(95.))
        belly = self.align_wheels(fold, upward=True)
        targets = {name: {"control6_rad": [q[key] for key in self.controls],
                          "full_joint_positions": q, "geometry_validation": self.validate_q(q)}
                   for name, q in (("nominal", nominal), ("fold", fold), ("thrust", thrust), ("support", nominal))}
        profiles = {}
        for name, (roll, pitch, release) in FLAT_POSES.items():
            q = dict(nominal)
            if name.startswith("crouched"):
                q = fold
            elif name == "prone_belly_legs_above":
                q = belly
            elif name == "prone_back_legs_above":
                q = thrust
            elif "left_forward" in name or "left_back" in name:
                direction = 1. if "left_forward" in name else -1.
                q = self.common_rotate(q, -.5 * direction / self.axis_signs[0],
                                       .5 * direction / self.axis_signs[2])
            profiles[name] = self.profile(name, q, roll, pitch, release)
            if name.startswith("prone_"):
                validation = profiles[name]["geometry_validation"]
                if validation["lowest_support_body"] != "base_link" or min(validation["wheel_bottom_above_shell_top_m"]) <= 0:
                    raise ValueError(f"{name} does not have shell down and wheels above: {validation}")
        if include_steps:
            self.add_step_profiles(profiles, nominal, fold, thrust, belly)
        frames = self.fk(self.spec, nominal)
        wheel_signs = []
        for name in ("L_joint3", "R_joint3"):
            joint = self.joints[name]
            axis = frames[joint["child"]][:3, :3] @ np.asarray(joint["axis"])
            wheel_signs.append(float(np.sign(axis[1])))
        # Import only after this geometry module is defined: the static helper
        # reuses RecoveryProfileBuilder and never changes reset configurations.
        static_path = Path(__file__).with_name("recovery_static_targets.py")
        static_spec = importlib.util.spec_from_file_location("recovery_static_targets", static_path)
        static_module = importlib.util.module_from_spec(static_spec)
        static_spec.loader.exec_module(static_module)
        standing = static_module.RecoveryStaticTargetBuilder(self.bundle, self.manifest_sha256).target(.305)
        prepare_reference = {**standing["prepare_reference"], "height_m": standing["height_m"],
            "full_joint_positions": standing["full_joint_positions"],
            "root_quaternion_xyzw": standing["root_quaternion_xyzw"],
            "geometry_validation": standing["geometry_validation"], "static_load": standing["static_load"],
            "pd_target_is_reset_pose": False, "dynamic_validation_passed": False,
            "source": "V6_closed_chain_static_virtual_work_candidate_requires_dynamic_AB_validation",
            "provenance": {"static_helper_sha256": digest(static_path),
                "spring_fit_sha256": digest(self.bundle / "fit_10mpa.json"),
                "model_spec_sha256": digest(self.bundle / "model_spec.json"),
                "asset_manifest_sha256": self.manifest_sha256,
                "solver_sha256": digest(self.solver_path),
                "fk_sha256": digest(self.bundle / "tools/v6_mechanism.py")}}
        return {"schema": "v6_recovery_profiles_v1", "asset_manifest_sha256": self.manifest_sha256,
                "model_spec_sha256": digest(self.bundle / "model_spec.json"),
                "solver_sha256": digest(self.solver_path), "generator_sha256": digest(Path(__file__)),
                "control_joint_names": self.controls, "tree_joint_names": self.manifest["tree_joint_names"],
                "axis_sign_joint_names": list(LEG_NAMES), "axis_signs": self.axis_signs,
                "wheel_axis_signs": wheel_signs, "targets": targets, "profiles": profiles,
                "prepare_reference": prepare_reference,
                "coordinate_frame": "Xforward_Yleft_Zup; root position relative to environment origin",
                "root_quaternion_order": "xyzw", "passive_knee_domain_deg": [40., 105.],
                "max_closure_gap_m": self.max_gap_m, "knee_zero_rad_from_sealed_limit": self.knee_zero_rad,
                "status": "geometry_only_reset_candidates; runtime_recovery_validation_required",
                "reset_injection_only": True, "hardware_deployment_ready": False}

    def add_step_profiles(self, profiles, nominal, fold, thrust, belly):
        # Geometry-safe initial conditions. Dynamic recovery and whether each
        # intended abnormal contact is reached must be measured in Isaac.
        candidates = (
            ("step200_wheels_split", self.common_rotate(self.align_wheels(self.knee_pose(50., 95.)), -.75, -.2), -.015, 0.),
            ("step200_shell_high_wheels_low", self.common_rotate(thrust, .5, -.5), .08, 0.),
            ("step200_lip_near_shell", thrust, -max(row["upper"][0] for row in self.collision_bounds(thrust, np.eye(3)) if row["body"] == "base_link") - .003, 0.),
            ("step200_upright_top", nominal, 1., 0.),
            ("step200_prone_belly_top", belly, 1., 0.),
        )
        for name, q, x, roll in candidates:
            profile = self.profile(name, q, roll=roll, release=.03, root_x=x, platforms=((0., .2),))
            profile.update(required_platform_height_m=.2, goal_support_height_m=.2,
                           min_ending_x=.22, intended_contact_requires_runtime_verification=True)
            profiles[name] = profile
        for name, q, x, roll, height, minimum in (
            ("double_step_first_riser_split", self.common_rotate(self.align_wheels(self.knee_pose(50., 95.)), -.75, -.2), -.015, 0., .2, .22),
            ("double_step_second_riser_split", self.common_rotate(self.align_wheels(self.knee_pose(50., 95.)), -1.1, -.2), .785, 0., .45, 1.02),
            ("double_step_first_prone_top", belly, .4, 0., .2, .22),
            ("double_step_second_prone_top", belly, 1.2, 0., .45, 1.02),
            ("double_step_second_side_top", nominal, 1.2, 90., .45, 1.02),
        ):
            profile = self.profile(name, q, roll=roll, release=.03, root_x=x, platforms=((0., .2), (.8, .45)))
            profile.update(required_platform_height_m=height, goal_support_height_m=height,
                           min_ending_x=minimum, terrain="stairs", stair_heights_m=[.2, .45],
                           stair_tread_lengths_m=[.8, .6], intended_contact_requires_runtime_verification=True)
            profiles[name] = profile

    def build_curriculum_profiles(self, source_path):
        """Extend the frozen v1 table without recomputing any historical pose."""
        source_path = Path(source_path)
        source = json.loads(source_path.read_text())
        if (source.get("schema") != "v6_recovery_profiles_v1"
                or source.get("asset_manifest_sha256") != self.manifest_sha256
                or source.get("model_spec_sha256") != digest(self.bundle / "model_spec.json")
                or source.get("tree_joint_names") != self.manifest["tree_joint_names"]):
            raise ValueError("Curriculum profiles require the frozen v1 table for this asset")
        result = deepcopy(source)
        profiles = result["profiles"]
        for surface, x, height, minimum in (("first", .4, .2, .22), ("second", 1.2, .45, 1.02)):
            for posture in ("upright", "crouched"):
                name = f"double_step_{surface}_{posture}_top"
                if name in profiles:
                    raise ValueError(f"Refusing to overwrite historical recovery profile: {name}")
                q = source["profiles"][posture]["full_joint_positions"]
                profile = self.profile(name, q, release=0., root_x=x, platforms=((0., .2), (.8, .45)))
                profile.update(required_platform_height_m=height, goal_support_height_m=height,
                    min_ending_x=minimum, terrain="stairs", stair_heights_m=[.2, .45],
                    stair_tread_lengths_m=[.8, .6], intended_contact_requires_runtime_verification=True,
                    curriculum_role="simple_takeover_on_same_support_surface")
                profiles[name] = profile
        result.update(schema="v6_recovery_profiles_v2", generator_sha256=digest(Path(__file__)),
            source_profiles_file=str(source_path.resolve().relative_to(ROOT)),
            source_profiles_sha256=digest(source_path),
            max_closure_gap_m=max(row["geometry_validation"]["max_closure_gap_m"] for row in profiles.values()))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--flat-only", action="store_true")
    parser.add_argument("--curriculum-simple-tops", action="store_true",
                        help="Build v2 by preserving v1 poses and adding simple staircase-top resets")
    parser.add_argument("--source-profiles", type=Path, default=ROOT / "contracts/v6_recovery_profiles_v1.json")
    args = parser.parse_args()
    if args.output is None:
        version = "v2" if args.curriculum_simple_tops else "v1"
        args.output = ROOT / f"contracts/v6_recovery_profiles_{version}.json"
    if args.curriculum_simple_tops and (args.flat_only or args.output.resolve() == args.source_profiles.resolve()):
        parser.error("Curriculum extension requires stairs and a new output; frozen v1 cannot be overwritten")
    if args.output.resolve().is_relative_to(args.asset.resolve()):
        parser.error("Do not write into the sealed asset")
    builder = RecoveryProfileBuilder(args.asset, args.manifest_sha256)
    result = (builder.build_curriculum_profiles(args.source_profiles) if args.curriculum_simple_tops
              else builder.build(not args.flat_only))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "profiles": len(result["profiles"]), "max_closure_gap_m": result["max_closure_gap_m"]}))


if __name__ == "__main__":
    main()
