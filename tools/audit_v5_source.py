#!/usr/bin/env python3
"""Read-only V5 CAD-export audit; never repair masses/inertias implicitly."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model/纯底盘_v5/source"


def audit():
    urdf = SOURCE / "urdf/urdf_v5.0.urdf"
    robot = ET.parse(urdf).getroot()
    with (SOURCE / "urdf/urdf_v5.0.csv").open() as stream:
        components = {row["Link Name"]: row["SW Components"] for row in csv.DictReader(stream)}
    links = {}
    for link in robot.findall("link"):
        inertial = link.find("inertial")
        inertia = inertial.find("inertia").attrib
        matrix = np.array([[float(inertia["i" + "".join(sorted(a + b))]) for b in "xyz"] for a in "xyz"])
        eig = np.linalg.eigvalsh(matrix)
        name = link.get("name")
        meshes = []
        for node in link.findall(".//mesh"):
            filename = node.get("filename").removeprefix("package://urdf_v5.0/")
            path = SOURCE / filename
            meshes.append({"path": filename, "exists": path.is_file()})
        links[name] = {"mass_kg": float(inertial.find("mass").get("value")),
            "principal_inertias_kg_m2": eig.tolist(), "positive_definite": bool(eig[0] > 0),
            "triangle_inequality": bool(eig[0] + eig[1] >= eig[2] - 1e-12),
            "cad_components": components[name], "meshes": meshes}
    frames = {"base_link": np.eye(4)}
    joints = list(robot.findall("joint"))
    remaining = joints.copy()
    while remaining:
        before = len(remaining)
        for joint in remaining[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in frames:
                continue
            if child in frames:
                raise ValueError("Non-tree or duplicate child")
            origin = joint.find("origin")
            local = np.eye(4)
            local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
            local[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
            frames[child] = frames[parent] @ local
            remaining.remove(joint)
        if len(remaining) == before:
            raise ValueError("Disconnected joint tree")
    springs = {}
    for prefix in ("LLL", "RRR"):
        upper, lower = frames[prefix + "_link1"], frames[prefix + "_link2"]
        delta = lower[:3, 3] - upper[:3, 3]
        axis = upper[:3, 0]
        springs[prefix] = {
            "upper_link": prefix + "_link1", "lower_link": prefix + "_link2",
            "source_zero_mount_distance_m": float(np.linalg.norm(delta)),
            "mount_line_off_upper_x_axis_m": float(np.linalg.norm(np.cross(delta, axis))),
            "absolute_axis_alignment_dot": float(abs(axis @ lower[:3, 0])),
            "telescoping_joint_present": False,
            "note": "Zero-pose mount separation is not full-extension separation or installed preload",
        }
    hashes = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in SOURCE.rglob("*") if p.is_file()}
    return {"source_files_sha256": hashes, "link_count": len(links), "tree_joint_count": len(joints),
        "joint_types": {kind: sum(j.get("type") == kind for j in joints)
                        for kind in sorted({j.get("type") for j in joints})},
        "total_mass_kg": sum(v["mass_kg"] for v in links.values()), "links": links,
        "nonpositive_inertias": [n for n, v in links.items() if not v["positive_definite"]],
        "triangle_violations": [n for n, v in links.items() if not v["triangle_inequality"]],
        "mass_asymmetries": [{"left": n, "right": n.replace("L", "R"), "mass_left_kg": v["mass_kg"],
                              "mass_right_kg": links[n.replace("L", "R")]["mass_kg"]}
                             for n, v in links.items() if n.startswith("L") and n.replace("L", "R") in links
                             and v["mass_kg"] != links[n.replace("L", "R")]["mass_kg"]],
        "spring_geometry": springs,
        "closures_present_in_source": False, "training_ready": False,
        "blockers": ["10 inertia tensors are not positive definite; obtain higher-precision CAD values",
            "RR_link2 mass differs tenfold from LL_link2; confirmation required",
            "four-bar loop constraints and spring telescoping joints not encoded",
            "all source joints continuous, including knees; physical limits need explicit reconstruction",
            "full-extension mounting pin separation and gas-spring stroke reference pending",
            "six motor-driven coordinates require confirmation; do not reuse old hip/knee action mapping"]}


if __name__ == "__main__":
    report = audit()
    output = SOURCE.parent / "source_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("links", "source_files_sha256")},
                     indent=2, ensure_ascii=False))
