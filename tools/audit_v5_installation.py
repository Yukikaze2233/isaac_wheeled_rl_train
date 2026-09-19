#!/usr/bin/env python3
"""Measure pin-to-end-plane offsets without treating a mesh as a stop drawing."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from analyze_v5_spring_limits import side_geometry
from v5_mechanism import circular_sections, fk, source_spec


ROOT = Path(__file__).resolve().parents[1]


def measure_parts(mesh):
    parts = list(mesh.split(only_watertight=False))
    if len(parts) != 2:
        raise ValueError("Expected exactly the owned adapter and spring solid")
    candidates = []
    for index, part in enumerate(parts):
        rings = circular_sections(part)
        pin_rings = [r for r in rings if np.linalg.norm(r["center"][:2]) < 1e-6]
        if pin_rings:
            candidates.append((index, pin_rings))
    if len(candidates) != 1:
        raise ValueError("Cannot uniquely identify the adapter pin at the link origin")
    index, rings = candidates[0]
    adapter, spring = parts[index], parts[1 - index]
    pin_x = float(np.mean([r["center"][0] for r in rings]))
    end_x = float(spring.bounds[0, 0])
    # A matching adapter plane is stronger evidence than the spring AABB alone,
    # but does not establish the manufacturer's installed length datum.
    plane_faces = np.all(np.abs(adapter.triangles[:, :, 0] - end_x) < 1e-8, axis=1)
    result = {
        "pin_center_xy_m": np.mean([r["center"][:2] for r in rings], axis=0).tolist(),
        "pin_ring_max_fit_error_m": max(r["max_fit_error"] for r in rings),
        "adapter_axial_bounds_m": adapter.bounds[:, 0].tolist(),
        "adapter_axial_envelope_length_m": float(np.ptp(adapter.bounds[:, 0])),
        "spring_axial_bounds_m": spring.bounds[:, 0].tolist(),
        "spring_solid_axial_length_m": float(np.ptp(spring.bounds[:, 0])),
        "pin_to_spring_mesh_end_plane_m": end_x - pin_x,
        "matching_adapter_plane_triangle_count": int(plane_faces.sum()),
        "matching_adapter_plane_area_m2": float(adapter.area_faces[plane_faces].sum()),
        "adapter_spring_axial_projection_overlap_m": max(
            0., float(min(adapter.bounds[1, 0], spring.bounds[1, 0])
                      - max(adapter.bounds[0, 0], spring.bounds[0, 0]))),
        "adapter_faces": len(adapter.faces),
        "spring_faces": len(spring.faces),
    }
    rim = adapter.vertices[np.abs(adapter.vertices[:, 0] - end_x) < 1e-8]
    if not plane_faces.any() and len(rim) >= 12:
        yz = rim[:, 1:]
        offset = yz.mean(axis=0)
        centered = yz - offset
        fit = np.linalg.lstsq(np.c_[2 * centered, np.ones(len(rim))], (centered ** 2).sum(axis=1), rcond=None)[0]
        center = offset + fit[:2]
        radius = float(np.sqrt(fit[2] + fit[:2] @ fit[:2]))
        tip_candidates = adapter.vertices[(adapter.vertices[:, 0] < end_x - 1e-6)
            & (np.linalg.norm(adapter.vertices[:, 1:] - center, axis=1) < 1e-7)]
        if len(tip_candidates) == 1 and np.abs(np.linalg.norm(yz - center, axis=1) - radius).max() < 1e-7:
            tip_x = float(tip_candidates[0, 0])
            result["conical_bore_transition"] = {
                "tip_x_m": tip_x, "rim_x_m": end_x, "rim_radius_m": radius,
                "included_angle_deg": float(np.degrees(2 * np.arctan2(radius, end_x - tip_x))),
                "interpretation": "Mesh cone-to-bore transition, not a planar seating shoulder or verified thread datum",
            }
    return result, (adapter, spring)


def audit(bundle, source):
    manifest = json.loads((bundle / "manifest.json").read_text())
    ownership = json.loads((bundle / "mesh_ownership.json").read_text())
    names = [side * 3 + suffix for side in ("L", "R") for suffix in ("_link1", "_link2")]
    hashes = {}
    for relative in ["model_spec.json", "mesh_ownership.json", *[f"meshes/{n}.stl" for n in names]]:
        actual = hashlib.sha256((bundle / relative).read_bytes()).hexdigest()
        if actual != manifest["files_sha256"][relative]:
            raise ValueError(f"Delivered file identity mismatch: {relative}")
        hashes[relative] = actual
    with (source / "urdf/urdf_v5.0.csv").open(newline="") as stream:
        rows = {r["Link Name"]: r for r in csv.DictReader(stream)}
    _, source_tree = source_spec(source)
    frames = fk(source_tree, {})
    report = {
        "schema_version": 1,
        "status": "mesh_geometry_checked_catalogue_datums_and_hardware_stops_unverified",
        "units": "m unless field suffix says otherwise",
        "files_sha256": hashes,
        "parts": {},
        "sides": {},
        "limitations": [
            "The available V5 source is a triangulated URDF export, not a dimensioned manufacturing drawing.",
            "A matching mesh plane does not identify thread engagement or the catalogue measurement endpoints.",
            "Rod mesh length alone does not prove usable stroke or internal piston stops.",
            "Projected axial overlap is not a three-dimensional collision result.",
        ],
    }
    plot_parts = {}
    for name in names:
        source_hash = hashlib.sha256((source / "meshes" / f"{name}.STL").read_bytes()).hexdigest()
        if source_hash != ownership[name]["source_sha256"]:
            raise ValueError(f"Source mesh identity mismatch: {name}")
        mesh = trimesh.load(bundle / "meshes" / f"{name}.stl", force="mesh", process=True)
        measured, plot_parts[name] = measure_parts(mesh)
        measured.update(source_sha256=source_hash, source_sw_components=rows[name]["SW Components"])
        report["parts"][name] = measured
    spec = json.loads((bundle / "model_spec.json").read_text())
    for side in ("L", "R"):
        upper, lower = side * 3 + "_link1", side * 3 + "_link2"
        offset_sum = sum(report["parts"][n]["pin_to_spring_mesh_end_plane_m"] for n in (upper, lower))
        current_values, current_range = side_geometry(spec, side)
        hypothetical = copy.deepcopy(spec)
        binding = hypothetical["spring_binding"][side + "_spring_slide"]
        delta = .237 - binding["full_extension_pin_distance_m"]
        binding.update(full_extension_pin_distance_m=.237, stroke_m=.080,
                       compression_at_q_zero_m=binding["compression_at_q_zero_m"] + delta,
                       reference_source="unverified_205_plus_32mm_hypothesis")
        candidate_values, candidate_range = side_geometry(hypothetical, side)
        report["sides"][side] = {
            "source_zero_pin_distance_m": float(np.linalg.norm(frames[upper][:3, 3] - frames[lower][:3, 3])),
            "mesh_end_plane_offsets_sum_m": offset_sum,
            "difference_from_nominal_32mm_m": .032 - offset_sum,
            "current_research_reference": {
                "full_extension_pin_distance_m": current_range["full_extension_pin_distance_m"],
                "fully_compressed_pin_distance_m": current_range["full_extension_pin_distance_m"] - current_range["stroke_m"],
                "mechanical_minimum_knee_deg": current_range["mechanical_minimum_knee_deg"],
                "recommended_minimum_knee_deg": current_range["recommended_minimum_knee_deg"],
                "at_35_deg": current_values(35.),
            },
            "nominal_32mm_hypothesis_not_calibration": {
                "full_extension_pin_distance_m": .237,
                "fully_compressed_pin_distance_m": .157,
                "mechanical_minimum_knee_deg": candidate_range["mechanical_minimum_knee_deg"],
                "recommended_minimum_knee_deg": candidate_range["recommended_minimum_knee_deg"],
                "at_35_deg": candidate_values(35.),
            },
        }
    return report, plot_parts


def plot_geometry(parts, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
    for row, name in enumerate(("LLL_link1", "LLL_link2")):
        measured = report["parts"][name]
        offset = measured["pin_to_spring_mesh_end_plane_m"] * 1000
        for col, ax in enumerate(axes[row]):
            for part, color, label in zip(parts[name], ("#718fa6", "#d99435"), ("Adapter", "Spring solid")):
                polygons = part.triangles[:, :, [0, 2]] * 1000
                ax.add_collection(PolyCollection(polygons, facecolor=color, edgecolor=color,
                                                 alpha=.35, linewidth=.3, label=label))
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(offset, color="#b23b35", linestyle=":", linewidth=1)
            ax.annotate("", (offset, -17), (0, -17), arrowprops={"arrowstyle": "<->"})
            ax.text(offset / 2, -22, f"{offset:.4f} mm", ha="center", fontsize=10)
            ax.set(xlim=(-13, 151 if col == 0 else 30), ylim=(-26, 8),
                   xlabel="Local X (mm), pin axis at X=0", ylabel="Local Z (mm)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{name}: {'whole assembly' if col == 0 else 'end-plane detail'}")
            ax.grid(alpha=.15)
    axes[0, 0].legend(loc="upper center", ncols=2, fontsize=9)
    fig.suptitle("V5 original mesh geometry: pin-to-end-plane offsets, not measured hardware stops")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "model/纯底盘_v5/urdf")
    parser.add_argument("--source", type=Path, default=ROOT / "model/纯底盘_v5/source")
    parser.add_argument("--output", type=Path, required=True, help="New audit output directory")
    args = parser.parse_args()
    report, parts = audit(args.bundle, args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "dimensions.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_geometry(parts, report, args.output / "mesh_end_planes.png")
    print(json.dumps(report["sides"], indent=2))


if __name__ == "__main__":
    main()
