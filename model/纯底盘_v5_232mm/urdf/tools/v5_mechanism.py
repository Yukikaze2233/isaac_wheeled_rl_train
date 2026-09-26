"""V5 source-frame geometry and loop-coordinate utilities; no simulator imports."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation
import trimesh


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


def origin(node):
    return transform(np.fromstring(node.get("xyz", "0 0 0"), sep=" "),
                     np.fromstring(node.get("rpy", "0 0 0"), sep=" "))


def point(matrix, xyz):
    return matrix[:3, :3] @ xyz + matrix[:3, 3]


def source_spec(source):
    robot = ET.parse(source / "urdf/urdf_v5.0.urdf").getroot()
    joints = [{"name": j.get("name"), "parent": j.find("parent").get("link"),
               "child": j.find("child").get("link"), "type": j.get("type"),
               "origin": origin(j.find("origin")).tolist(),
               "axis": np.fromstring(j.find("axis").get("xyz"), sep=" ").tolist(),
               "limit": dict(j.find("limit").attrib)} for j in robot.findall("joint")]
    return robot, {"joints": joints}


def fk(spec, q, root=None):
    frames = {"base_link": np.eye(4) if root is None else root}
    remaining = list(spec["joints"])
    while remaining:
        before = len(remaining)
        for j in remaining[:]:
            if j["parent"] not in frames:
                continue
            motion = np.eye(4)
            if j["type"] == "prismatic":
                motion[:3, 3] = np.asarray(j["axis"]) * q.get(j["name"], 0.)
            else:
                motion[:3, :3] = Rotation.from_rotvec(np.asarray(j["axis"]) * q.get(j["name"], 0.)).as_matrix()
            frames[j["child"]] = frames[j["parent"]] @ np.asarray(j["origin"]) @ motion
            remaining.remove(j)
        if len(remaining) == before:
            raise ValueError("Disconnected or cyclic URDF tree")
    return frames


def circular_sections(mesh):
    """Fit circular boundaries of planar mesh faces normal to the local Z axis."""
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    triangles = vertices[faces]
    planar = np.ptp(triangles[:, :, 2], axis=1) < 1e-8
    levels = np.round(triangles[:, :, 2].mean(1), 7)
    rings = []
    for z in np.unique(levels[planar]):
        f = faces[planar & (levels == z)]
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = edges[counts == 1]
        if len(boundary) == 0:
            continue
        graph = coo_matrix((np.ones(len(boundary)), boundary.T), shape=(len(vertices), len(vertices))).tocsr()
        _, labels = connected_components(graph, directed=False)
        for label in np.unique(labels[boundary.ravel()]):
            e = boundary[labels[boundary[:, 0]] == label]
            v, degrees = np.unique(e, return_counts=True)
            if len(v) < 12 or not np.all(degrees == 2):
                continue
            xy = vertices[v, :2]
            offset = xy.mean(0)
            x = xy - offset
            fit = np.linalg.lstsq(np.c_[2 * x, np.ones(len(x))], (x * x).sum(1), rcond=None)[0]
            center = offset + fit[:2]
            radius = np.sqrt(max(0., fit[2] + fit[:2] @ fit[:2]))
            error = np.abs(np.linalg.norm(xy - center, axis=1) - radius).max()
            if error > 1e-6:
                continue
            angles = np.sort(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
            if np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max() > np.pi / 4:
                continue
            rings.append({"center": [*center.tolist(), float(vertices[v, 2].mean())],
                          "radius": float(radius), "max_fit_error": float(error)})
    return rings


def metrology(source):
    robot, spec = source_spec(source)
    frames = fk(spec, {})
    report = {}
    for link in robot.findall("link"):
        name = link.get("name")
        mesh = trimesh.load(source / "meshes" / f"{name}.STL", force="mesh", process=True)
        rings = circular_sections(mesh) if name != "base_link" else []
        groups = []
        for ring in rings:
            group = next((g for g in groups if np.linalg.norm(np.array(g["center_xy"]) - ring["center"][:2]) < 2e-6), None)
            if group is None:
                group = {"center_xy": ring["center"][:2], "radii": [], "sections": []}
                groups.append(group)
            group["radii"].append(ring["radius"])
            group["sections"].append(ring)
        report[name] = {"mass_volume_m3": float(mesh.volume), "watertight": bool(mesh.is_watertight),
                        "bounds": mesh.bounds.tolist(), "mesh_com": mesh.center_mass.tolist(),
                        "axes": groups, "source_zero_frame": frames[name].tolist(),
                        "components": [{"faces": len(part.faces), "volume": float(part.volume),
                                        "center": part.center_mass.tolist(),
                                        "bounds": part.bounds.tolist(),
                                        "inertia_eigen": np.linalg.eigvalsh(part.moment_inertia).tolist()}
                                       for part in mesh.split(only_watertight=False)]}
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    source = root / "model/纯底盘_v5/source"
    result = metrology(source)
    output = root / "model/纯底盘_v5/metrology.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    for name, data in result.items():
        if name.startswith(("LLL", "RRR")):
            print(name, "components", [(v["faces"], np.round(v["bounds"], 7).tolist()) for v in data["components"]])
