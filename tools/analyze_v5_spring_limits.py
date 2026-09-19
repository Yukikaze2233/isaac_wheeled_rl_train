#!/usr/bin/env python3
"""Audit V5 knee-angle reachability against the installed gas-spring length domain.

This reads the delivered geometry only. It does not move attachment points,
change joint stops, or reinterpret a soft solver penetration as available travel.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def side_geometry(spec, side):
    joints = {j["name"]: j for j in spec["joints"]}
    knee_name = "L_joint2" if side == "L" else "R_jonit2"
    knee = joints[knee_name]
    lower = next(j for j in spec["joints"] if j["child"] == side * 3 + "_link2")
    upper = next(c for c in spec["constraints"] if c["name"] == side + "_spring_upper_mount")
    binding = spec["spring_binding"][side + "_spring_slide"]
    knee_origin = np.asarray(knee["origin"])
    p_upper = np.asarray(upper["local_pos0_m"])
    p_lower_in_shank = np.asarray(lower["origin"])[:3, 3]
    zero_inner = math.pi - 2.3573
    sign = 1. if side == "L" else -1.

    def values(angle_deg):
        q = sign * (math.radians(angle_deg) - zero_inner)
        rotation = Rotation.from_rotvec(np.asarray(knee["axis"]) * q).as_matrix()
        p_lower = knee_origin[:3, :3] @ rotation @ p_lower_in_shank + knee_origin[:3, 3]
        length = float(np.linalg.norm(p_lower - p_upper))
        compression = binding["full_extension_pin_distance_m"] - length
        return {"knee_inner_deg": angle_deg, "knee_raw_rad": q,
                "pin_distance_m": length, "compression_m": compression,
                "remaining_mechanical_compression_m": binding["stroke_m"] - compression,
                "remaining_recommended_compression_m": .9 * binding["stroke_m"] - compression,
                "slider_q_m": binding["compression_at_q_zero_m"] - compression}

    mechanical_min = brentq(lambda a: values(a)["compression_m"] - binding["stroke_m"], 30., 80.)
    recommended_min = brentq(lambda a: values(a)["compression_m"] - .9 * binding["stroke_m"], 30., 80.)
    samples = [values(float(a)) for a in [35, 36, 40, 45, recommended_min, 50, 55, 60, 65, 70, 75, 80]]
    return values, {"side": side, "mechanical_minimum_knee_deg": mechanical_min,
                    "recommended_minimum_knee_deg": recommended_min,
                    "stroke_m": binding["stroke_m"],
                    "full_extension_pin_distance_m": binding["full_extension_pin_distance_m"],
                    "source_reference": binding["reference_source"],
                    "hardware_reference_verified": binding["reference_hardware_verified"],
                    "samples": samples}


def analyze(bundle):
    spec = json.loads((bundle / "model_spec.json").read_text())
    report = {"model_spec_sha256": hashlib.sha256((bundle / "model_spec.json").read_bytes()).hexdigest(),
              "scope": "geometric knee/spring intersection, not collision/strength/hardware verification", "sides": {}}
    for side in ("L", "R"):
        values, result = side_geometry(spec, side)
        at35 = values(35.)
        result["at_35_deg"] = at35
        result["minimum_length_change_for_35_hard_stop_m"] = max(0., -at35["remaining_mechanical_compression_m"])
        result["minimum_length_change_for_35_with_10pct_reserve_m"] = max(0., -at35["remaining_recommended_compression_m"])
        grid = np.array([values(a)["compression_m"] for a in np.linspace(35., 80., 4501)])
        result["compression_monotonically_decreases_with_knee_angle"] = bool(np.all(np.diff(grid) < 0))
        report["sides"][side] = result
    return report


def cross_check_closed_chain(bundle, report):
    sys.path.insert(0, str(bundle / "tools"))
    from build_v5_closedchain import POSE_COORDINATES, closure_error, solve_pose
    spec = json.loads((bundle / "model_spec.json").read_text())
    joints = {j["name"]: j for j in spec["joints"]}
    rows = []
    q = spec["nominal_joint_pos"]
    for angle in (35., report["sides"]["L"]["mechanical_minimum_knee_deg"],
                  report["sides"]["L"]["recommended_minimum_knee_deg"], 80.):
        raw = math.radians(angle) - (math.pi - 2.3573)
        q = solve_pose(spec, dict(zip(POSE_COORDINATES, [.42, raw, 0., -.42, -raw, 0.])), q)
        values = {}
        for side in ("L", "R"):
            name = side + "_spring_slide"
            binding = spec["spring_binding"][name]
            position = q[name]
            values[side] = {"compression_m": binding["compression_at_q_zero_m"] - position,
                            "slider_lower_limit_violation_m": max(0., float(joints[name]["limit"]["lower"]) - position)}
        rows.append({"knee_inner_deg": angle, "max_loop_error_m": float(np.abs(closure_error(spec, q)).max()),
                     "springs": values})
    report["closed_chain_cross_check"] = rows
    report["cross_check_note"] = "IK is deliberately unconstrained to expose limit violations, not to claim a feasible 35-degree pose"


def balanced_standing_pose(spec, angle):
    """Geometric wheel-supported pose; no claim about controller or collision feasibility."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v5_closedchain import POSE_COORDINATES, solve_pose
    from v5_mechanism import fk, point
    bodies = {body["name"]: body for body in spec["bodies"]}
    knee = math.radians(angle) - (math.pi - 2.3573)

    def pose(hip):
        return solve_pose(spec, dict(zip(POSE_COORDINATES, [hip, knee, 0., -hip, -knee, 0.])), spec["nominal_joint_pos"])

    def geometry(q, root=None):
        frames = fk(spec, q, root)
        com = sum(b["mass"] * point(frames[b["name"]], b["com"]) for b in bodies.values()) / sum(b["mass"] for b in bodies.values())
        wheel_centers = np.array([point(frames[name], np.asarray(bodies[name]["collisions"][0]["origin"])[:3, 3])
                                  for name in ("L_link3", "R_link3")])
        return frames, com, wheel_centers

    def balance(hip):
        _, com, centers = geometry(pose(hip))
        return float(com[0] - centers[:, 0].mean())

    hip = brentq(balance, -.4, 1.4)
    q = pose(hip)
    _, _, centers = geometry(q)
    delta = centers[0] - centers[1]
    roll = math.atan2(-delta[2], delta[1])
    root = np.eye(4)
    root[:3, :3] = Rotation.from_euler("x", roll).as_matrix()
    frames, com, centers = geometry(q, root)
    bottoms = []
    for i, name in enumerate(("L_link3", "R_link3")):
        collision = bodies[name]["collisions"][0]
        rotation = frames[name][:3, :3] @ np.asarray(collision["origin"])[:3, :3]
        axial_z = float(rotation[2, 2])
        extent = collision["radius"] * math.sqrt(max(0., 1 - axial_z ** 2)) + collision["length"] * .5 * abs(axial_z)
        bottoms.append(float(centers[i, 2] - extent))
    height = -min(bottoms)
    leg_names = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")
    return {"knee_inner_deg": angle, "base_frame_height_m": height,
            "whole_robot_com_height_m": height + float(com[2]), "base_roll_rad": roll,
            "wheel_bottom_height_difference_m": max(bottoms) - min(bottoms),
            "motor_raw_positions_rad": {name: q[name] for name in leg_names},
            "motor_raw_positions_deg": {name: math.degrees(q[name]) for name in leg_names},
            "joint_positions": q}


def standing_ranges(bundle):
    original = json.loads((bundle / "model_spec.json").read_text())
    hypothetical = deepcopy(original)
    for binding in hypothetical["spring_binding"].values():
        delta = .237 - binding["full_extension_pin_distance_m"]
        binding["full_extension_pin_distance_m"] = .237
        binding["compression_at_q_zero_m"] += delta
        binding["stroke_m"] = .08
        binding["reference_source"] = "unverified_205_plus_32mm_hypothesis"
    result = {"scope": "symmetric knee angles, base pitch zero, CAD roll leveled, COM centered over wheel support line",
              "height_reference": "base_link frame, not the top or bottom of the chassis",
              "hardware_collision_and_torque_feasibility_verified": False,
              "wheel_joint_rotation": "continuous, not restricted by this static height calculation",
              "cases": {}}
    for label, spec in (("current_CAD_mesh_reference", original), ("nominal_32mm_hypothesis_only", hypothetical)):
        values, geometry = side_geometry(spec, "L")
        ranges = {}
        for name, lower in (("mechanical", geometry["mechanical_minimum_knee_deg"]),
                            ("with_10pct_compression_reserve", geometry["recommended_minimum_knee_deg"])):
            lower, upper = max(35., lower), 80.
            rows = [balanced_standing_pose(spec, float(a)) for a in np.linspace(lower, upper, 33)]
            heights = np.array([row["base_frame_height_m"] for row in rows])
            motor_ranges = {motor: [min(row["motor_raw_positions_deg"][motor] for row in rows),
                                    max(row["motor_raw_positions_deg"][motor] for row in rows)]
                            for motor in rows[0]["motor_raw_positions_deg"]}
            low_pin, high_pin = values(lower), values(upper)
            ranges[name] = {"knee_inner_deg": [lower, upper],
                "pin_distance_m": [low_pin["pin_distance_m"], high_pin["pin_distance_m"]],
                "compression_m": [high_pin["compression_m"], low_pin["compression_m"]],
                "used_spring_travel_m": high_pin["pin_distance_m"] - low_pin["pin_distance_m"],
                "base_frame_height_m": [float(heights.min()), float(heights.max())],
                "base_vertical_travel_m": float(np.ptp(heights)),
                "height_monotonic_on_33_point_scan": bool(np.all(np.diff(heights) > 0)),
                "motor_raw_ranges_deg": motor_ranges,
                "motor_angular_travel_deg": {motor: limits[1] - limits[0] for motor, limits in motor_ranges.items()},
                "endpoints": [rows[0], rows[-1]]}
        result["cases"][label] = ranges
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "model/纯底盘_v5/urdf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cross-check", action="store_true", help="Also solve the entire linkage and compare raw slider limits")
    parser.add_argument("--standing-ranges", action="store_true", help="Also calculate grounded height and active-leg angular travel")
    args = parser.parse_args()
    result = analyze(args.bundle)
    if args.cross_check:
        cross_check_closed_chain(args.bundle, result)
    if args.standing_ranges:
        result["standing_ranges"] = standing_ranges(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
