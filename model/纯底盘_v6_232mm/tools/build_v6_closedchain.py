#!/usr/bin/env python3
"""Build V6 16R+2P research geometry with six real loop constraints.

Reuse the previous mechanism exporters; retain the source snapshot and record
every mesh ownership, axis-rounding, and research-inertia adaptation separately.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from v6_mechanism import fk, origin, point, source_spec, transform


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "2c061fc49e1cf375b32c19002495aa03c2856941b326688f36cd81d7102e095d"
MODEL_KIND = "v6_gas_spring_closedchain_research"
INERTIA_PRIOR_LINKS = frozenset({
    "LL_link2", "RR_link2", "LL_link3", "RR_link3", "LL_link4", "RR_link4",
    "LLL_link1", "LLL_link2", "RRR_link1", "RRR_link2",
})
ACTIVE = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]
POSE_COORDINATES = ["L_joint1", "L_joint2", "L_joint3", "R_joint1", "R_jonit2", "R_joint3"]
SPRINGS = ["L_spring_slide", "R_spring_slide"]
# V6 root is already X-forward/Y-left/Z-up. Rz(+90) applies only when
# transporting measured V5 local pin coordinates into the new link frames.
S = transform()
V5_LOCAL_TO_V6 = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])


def load_exporter():
    bundled = Path(__file__).with_name("mechanism_export_legacy.py")
    path = bundled
    module_spec = importlib.util.spec_from_file_location("mechanism_export", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.CONTROL = ACTIVE
    module.tree_fk = fk
    return module, path


def closure_error(spec, q):
    frames = fk(spec, q)
    return np.concatenate([point(frames[c["body0"]], c["local_pos0_m"])
                           - point(frames[c["body1"]], c["local_pos1_m"]) for c in spec["constraints"]])


def solve_pose(spec, prescribed, initial=None):
    q = {j["name"]: 0. for j in spec["joints"]}
    q.update(initial or {})
    q.update(prescribed)
    unknown = [j["name"] for j in spec["joints"] if j["name"] not in prescribed]

    def residual(values):
        current = dict(q)
        current.update(zip(unknown, values))
        return closure_error(spec, current)

    result = least_squares(residual, [q[n] for n in unknown], xtol=1e-13, ftol=1e-13,
                           gtol=1e-13, max_nfev=200, diff_step=1e-5)
    q.update(zip(unknown, result.x.tolist()))
    if np.max(np.abs(residual(result.x))) > 2e-7:
        raise RuntimeError(f"V6 closure solve failed: {result.message}, {residual(result.x)}")
    return q


def clean_meshes(source, out, frames, exporter):
    reference = trimesh.load(source / "meshes/RR_link4.STL", force="mesh", process=True)
    reference_world = trimesh.transform_points(reference.vertices, frames["RR_link4"])
    reference_tree = cKDTree(reference_world)
    center = point(frames["RR_link4"], reference.center_mass)
    meshes, report = {}, {}
    for name in frames:
        path = source / "meshes" / f"{name}.STL"
        records = exporter.read_stl(path)
        mesh = trimesh.load(path, force="mesh", process=True)
        source_centers = records["vertices"].astype(float).mean(1)
        source_tree = cKDTree(source_centers)
        keep = np.ones(len(records), dtype=bool)
        removed = []
        components = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
        for indices in components:
            part = mesh.submesh([indices], append=True, repair=False)
            world = trimesh.transform_points(part.vertices, frames[name])
            if name == "RR_link4" or len(part.faces) != len(reference.faces):
                continue
            center_error = np.linalg.norm(point(frames[name], part.center_mass) - center)
            distances = reference_tree.query(world)[0]
            if center_error < 0.0002 and distances.max() < 0.0002:
                if abs(part.volume / reference.volume - 1) > 1e-4:
                    raise RuntimeError("Unexpected shared-component shape")
                face_error, source_indices = source_tree.query(part.triangles_center)
                if face_error.max() > 2e-8 or len(set(source_indices)) != len(part.faces):
                    raise RuntimeError(f"Cannot bind duplicate component to original STL records: {name}")
                keep[source_indices] = False
                removed.append({"source_face_indices": source_indices.tolist(), "faces": len(source_indices),
                    "reference_body": "RR_link4", "world_vertex_max_difference_m": float(distances.max()),
                    "world_center_difference_m": float(center_error)})
        selected = records[keep]
        target = out / "meshes" / f"{name}.stl"
        target.write_bytes(b"V6 owned source triangles; see mesh_ownership.json".ljust(80, b" ")
                           + struct.pack("<I", len(selected)) + selected.tobytes())
        meshes[name] = trimesh.load(target, force="mesh", process=True)
        report[name] = {"source_sha256": exporter.sha(path), "source_faces": len(records),
            "kept_faces": len(selected), "removed_duplicate_components": removed,
            "retained_vertices_modified": False}
    return meshes, report


def build_spec(source, out, exporter, installation, mechanical_limits):
    if exporter.sha(source / "urdf/urdf_v5.0.urdf") != SOURCE_SHA256:
        raise ValueError("V6 source URDF hash differs from the reviewed download")
    stroke = float(installation["stroke_m"])
    spring_length = float(installation["spring_extended_length_m"])
    adapter_length = float(installation["adapter_effective_length_m"])
    if not np.isfinite([stroke, spring_length, adapter_length]).all() or not 0 < stroke < spring_length or adapter_length < 0:
        raise ValueError("Invalid installed spring dimensions")
    free_length = spring_length + adapter_length
    knee_limits = mechanical_limits["knee_inner_angle_deg"]
    if len(knee_limits) != 2 or not 0 < knee_limits[0] < knee_limits[1] < 180:
        raise ValueError("Invalid thigh-shank inner-angle limits")
    robot, spec = source_spec(source)
    raw_frames = fk(spec, {})
    meshes, ownership = clean_meshes(source, out, raw_frames, exporter)
    # Source body and root frames are already the V6 convention; no root rotation.
    for j in spec["joints"]:
        if j["parent"] == "base_link":
            j["origin"] = (S @ np.asarray(j["origin"])).tolist()
        j["limit"] = {"effort": "100", "velocity": "100"}
        if j["name"] in ("L_joint2", "R_jonit2"):
            zero_inner = np.pi - 2.3573
            bounds = np.deg2rad(knee_limits) - zero_inner
            if j["name"] == "R_jonit2":
                bounds = -bounds[::-1]
            j["type"] = "revolute"
            j["limit"].update(lower=str(bounds[0]), upper=str(bounds[1]))
    frames = fk(spec, {})
    axis_changes = {}
    for j in spec["joints"]:
        side = j["child"][0]
        common = frames[f"{side}_link1"][:3, 2]
        old_axis = frames[j["child"]][:3, :3] @ j["axis"]
        axis = common * (1 if common @ old_axis > 0 else -1)
        axis_changes[j["name"]] = float(np.arccos(np.clip(old_axis @ axis, -1, 1)))
        j["axis"] = (frames[j["child"]][:3, :3].T @ axis).tolist()
    spec.update(bodies=[], constraints=[], control_frame="Xforward_Yleft_Zup", nominal_base_height_m=.32)
    inertias = {}
    for link in robot.findall("link"):
        name = link.get("name")
        mass, com, tensor = exporter.read_inertia(link)
        old_com, old_tensor = com.copy(), tensor.copy()
        eig = np.linalg.eigvalsh(tensor)
        invalid = bool(eig[0] <= 0 or eig[0] + eig[1] < eig[2] - 1e-12)
        artificial_floor = bool(name in INERTIA_PRIOR_LINKS and eig[0] <= 1.01e-8)
        estimated = invalid or artificial_floor
        mesh = meshes[name]
        if estimated:
            if mesh.volume <= 0 or not mesh.is_watertight or not mesh.is_winding_consistent:
                raise RuntimeError(f"No watertight positive oriented volume for research inertia: {name}")
            com = mesh.center_mass.copy()
            tensor = mesh.moment_inertia * (mass / mesh.volume)
        values = np.linalg.eigvalsh(tensor)
        if values[0] <= 0 or values[0] + values[1] < values[2] - 1e-12:
            raise RuntimeError(f"Invalid derived inertia: {name}")
        mesh_origin = S if name == "base_link" else np.eye(4)
        inertia_rotation = mesh_origin[:3, :3]
        inertia = inertia_rotation @ tensor @ inertia_rotation.T
        com = point(mesh_origin, com)
        collisions = []
        if name in ("L_link3", "R_link3"):
            # Measured tread radius and axial extent; the URDF local Z is the wheel axle.
            lower, upper = mesh.bounds
            center = [0., 0., float((lower[2] + upper[2]) / 2)]
            collisions.append({"type": "cylinder", "radius": .06, "length": float(upper[2] - lower[2]),
                               "origin": transform(center).tolist()})
        else:
            for k, part in enumerate(mesh.split(only_watertight=False)):
                if part.volume < 1e-10 or len(part.faces) < 20:
                    continue
                hull = part.convex_hull
                if len(hull.vertices) > 255:
                    # Deterministic bounding boxes avoid implicit PhysX hull simplification.
                    hull = part.bounding_box_oriented.to_mesh()
                name_collision = f"collisions/{name}_{k:03d}.obj"
                hull.export(out / name_collision)
                collisions.append({"type": "mesh", "file": name_collision, "origin": mesh_origin.tolist()})
        color = [.22, .45, .76] if "LLL" in name or "RRR" in name else [.65, .68, .72]
        spec["bodies"].append({"name": name, "component": name, "mass": mass, "com": com.tolist(),
            "inertia": inertia.tolist(), "mesh": f"meshes/{name}.stl", "mesh_origin": mesh_origin.tolist(),
            "color": color, "collisions": collisions})
        inertias[name] = {"source_mass_kg": mass, "source_com_m": old_com.tolist(), "source_tensor": old_tensor.tolist(),
                         "generated_com_m": com.tolist(), "generated_tensor": inertia.tolist(),
                         "method": "owned_mesh_uniform_density_research_prior" if estimated else "source_CAD_preserved",
                         "source_eigenvalues_kg_m2": eig.tolist(),
                         "generated_eigenvalues_kg_m2": values.tolist(),
                         "source_invalid": invalid, "source_artificial_floor": artificial_floor,
                         "repair_reason": "invalid_principal_moments" if invalid else
                             "reviewed_1e-8_floor_not_a_measured_inertia" if artificial_floor else None,
                         "owned_mesh_volume_m3": float(mesh.volume),
                         "owned_mesh_watertight": bool(mesh.is_watertight),
                         "hardware_identified": False,
                         "mass_changed": False}

    def loop(name, body0, body1, p0, p1):
        p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
        axis = frames[body0[0] + "_link1"][:3, 2]
        w0, w1 = point(frames[body0], p0), point(frames[body1], p1)
        # Coincident points along a revolute pin line may use different axial representatives.
        w1 = w1 + axis * ((w0 - w1) @ axis)
        p1 = point(np.linalg.inv(frames[body1]), w1)
        spec["constraints"].append({"name": name, "type": "spherical", "body0": body0, "body1": body1,
            "local_pos0_m": p0.tolist(), "local_pos1_m": p1.tolist(),
            "source_transverse_gap_m": float(np.linalg.norm(w0 - w1)), "physics:excludeFromArticulation": True})

    spring_binding = {}
    for side in ("L", "R"):
        # Independently measured circular sections, not old V4 mesh coordinates.
        loop(f"{side}_rocker_P", side + "_link1", side * 2 + "_link3",
             V5_LOCAL_TO_V6 @ [.1134, 0, 0], V5_LOCAL_TO_V6 @ [.1345073, -.0115232, 0])
        loop(f"{side}_upper_E", side + "_link2", side * 2 + "_link4",
             V5_LOCAL_TO_V6 @ [-.0647772, -.0171149, 0], V5_LOCAL_TO_V6 @ [.0966, 0, 0])
        upper, lower = side * 3 + "_link1", side * 3 + "_link2"
        removed = next(j for j in spec["joints"] if j["child"] == upper)
        original_parent, original_origin = removed["parent"], np.array(removed["origin"])
        distance = np.linalg.norm(frames[upper][:3, 3] - frames[lower][:3, 3])
        line = (frames[upper][:3, 3] - frames[lower][:3, 3]) / distance
        relative = np.linalg.inv(frames[lower]) @ frames[upper]
        axis = frames[upper][:3, :3].T @ line
        # Mesh endpoints are provenance only: they are not the installed length datums.
        rod = next(p for p in meshes[upper].split(only_watertight=False) if len(p.faces) == 78)
        cylinder = next(p for p in meshes[lower].split(only_watertight=False) if len(p.faces) == 912)
        compression0 = free_length - distance
        name = side + "_spring_slide"
        removed.update(name=name, parent=lower, type="prismatic", origin=relative.tolist(), axis=axis.tolist(),
            limit={"lower": str(compression0 - stroke), "upper": str(compression0), "effort": "1000", "velocity": "100"})
        loop(f"{side}_spring_upper_mount", original_parent, upper, original_origin[:3, 3], [0, 0, 0])
        spring_binding[name] = {"joint_coordinate": "extension_relative_to_CAD_zero_m",
            "source_mount_distance_m": float(distance), "stroke_m": stroke,
            "full_extension_pin_distance_m": free_length, "compression_at_q_zero_m": float(compression0),
            "compression_expression": "compression_at_q_zero_m - q_m",
            "effort_sign": 1, "pressure_mpa": 10.,
            "reference_source": installation["source"],
            "reference_hardware_verified": installation["hardware_stop_measurement_verified"],
            "spring_extended_length_m": spring_length, "adapter_effective_length_m": adapter_length,
            "mesh_long_axis": "local_Y",
            "body_axial_bounds_m": cylinder.bounds[:, 1].tolist(), "rod_axial_bounds_m": rod.bounds[:, 1].tolist()}
    # Prismatic children were reparented; make the tree topological for all exporters.
    ordered, names = [], {"base_link"}
    remaining = list(spec["joints"])
    while remaining:
        ready = [j for j in remaining if j["parent"] in names]
        if not ready:
            raise RuntimeError("Invalid generated tree")
        for j in ready:
            ordered.append(j)
            names.add(j["child"])
            remaining.remove(j)
    spec["joints"] = ordered
    q = solve_pose(spec, dict.fromkeys(POSE_COORDINATES, 0.))
    spec["source_closed_joint_pos"] = q
    for fraction in np.linspace(0, 1, 11)[1:]:
        prescribed = dict(zip(POSE_COORDINATES, np.array([-.42, .40, 0, .42, -.40, 0]) * fraction))
        q = solve_pose(spec, prescribed, q)
    spec["nominal_joint_pos"] = q
    spec["spring_binding"] = spring_binding
    return spec, {"mesh_ownership": ownership, "inertial_sources": inertias, "axis_rounding_corrections_rad": axis_changes}


def adapt_exports(out, spec, exporter):
    exporter.export_urdf(out, spec)
    exporter.export_mjcf(out, spec)
    exporter.export_usd(out, spec)
    # Extend the previous rotational exporter at its explicit joint boundary.
    from pxr import Gf, Usd, UsdPhysics
    stage = Usd.Stage.Open(str(out / "robot.usda"))
    for j in spec["joints"]:
        path = "/Robot/tree_joints/" + j["name"]
        stage.RemovePrim(path)
        prism = j["type"] == "prismatic"
        joint = (UsdPhysics.PrismaticJoint if prism else UsdPhysics.RevoluteJoint).Define(stage, path)
        joint.CreateBody0Rel().SetTargets(["/Robot/" + j["parent"]])
        joint.CreateBody1Rel().SetTargets(["/Robot/" + j["child"]])
        axis0 = np.array([1., 0, 0] if prism else [0., 0, 1.])
        alignment = np.eye(4)
        alignment[:3, :3] = Rotation.align_vectors([j["axis"]], [axis0])[0].as_matrix()
        t = np.array(j["origin"])

        def quat(matrix):
            xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            return Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*xyzw[:3]))

        joint.CreateAxisAttr("X" if prism else "Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*t[:3, 3]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(quat(t @ alignment))
        joint.CreateLocalRot1Attr(quat(alignment))
        joint.CreateExcludeFromArticulationAttr(False)
        if prism:
            joint.CreateLowerLimitAttr(float(j["limit"]["lower"]))
            joint.CreateUpperLimitAttr(float(j["limit"]["upper"]))
        elif j["type"] == "revolute":
            joint.CreateLowerLimitAttr(float(np.rad2deg(float(j["limit"]["lower"]))))
            joint.CreateUpperLimitAttr(float(np.rad2deg(float(j["limit"]["upper"]))))
    root = stage.GetDefaultPrim()
    root.SetCustomData({"model_kind": MODEL_KIND, "hardware_deployment_ready": False})
    stage.GetRootLayer().Save()
    mj = ET.parse(out / "robot.xml").getroot()
    mj.set("model", MODEL_KIND)
    for j in spec["joints"]:
        # PC torque mode owns active feedback; match the PhysX runtime's zero
        # active damping instead of inheriting the legacy passive joint prior.
        node = mj.find(f".//joint[@name='{j['name']}']")
        node.set("damping", "0" if j["name"] in ACTIVE else "0.002")
        if j["type"] == "prismatic":
            node.set("type", "slide")
            node.set("limited", "true")
            node.set("range", f"{j['limit']['lower']} {j['limit']['upper']}")
            ET.SubElement(mj.find("actuator"), "motor", name=j["name"] + "_gas_force", joint=j["name"], gear="1")
    exporter.write_xml(out / "robot.xml", mj)
    urdf = ET.parse(out / "robot.urdf").getroot()
    urdf.set("name", MODEL_KIND)
    for joint in urdf.findall("joint"):
        ET.SubElement(joint, "dynamics", damping="0" if joint.get("name") in ACTIVE else "0.002",
                      friction="0")
    exporter.write_xml(out / "robot.urdf", urdf)
    srdf = ET.parse(out / "collision_filters.srdf").getroot()
    srdf.set("name", MODEL_KIND)
    exporter.write_xml(out / "collision_filters.srdf", srdf)


def validate(out, spec):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(out / "robot.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    eq = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    jacobian = data.efc_J.reshape(data.nefc, model.nv)[eq]
    rank = np.linalg.matrix_rank(jacobian, tol=1e-8)
    result = {"bodies_excluding_world": model.nbody - 1, "tree_joints": model.njnt - 1,
        "hinges": int(sum(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)),
        "sliders": int(sum(model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE)),
        "loop_constraints": model.neq, "constraint_rank": int(rank),
        "independent_internal_velocities": int(model.nv - rank - 6),
        "nominal_max_closure_error_m": float(np.abs(closure_error(spec, spec["nominal_joint_pos"])).max()),
        "source_zero_max_closure_error_m": float(np.abs(closure_error(spec, spec["source_closed_joint_pos"])).max()),
        "nominal_spring_compression_m": {n: b["compression_at_q_zero_m"] - spec["nominal_joint_pos"][n]
                                          for n, b in spec["spring_binding"].items()},
        "total_mass_kg": float(model.body_mass.sum())}
    if (result["bodies_excluding_world"], result["hinges"], result["sliders"], model.neq, rank) != (19, 16, 2, 6, 12):
        raise RuntimeError(f"Unexpected compiled topology/rank: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "source")
    parser.add_argument("--output", type=Path, default=ROOT / "urdf")
    parser.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    parser.add_argument("--provenance", type=Path, default=ROOT / "source_provenance.json")
    parser.add_argument("--validate-only", type=Path, metavar="BUNDLE")
    args = parser.parse_args()
    if args.validate_only:
        bundle = args.validate_only.resolve()
        print(json.dumps(validate(bundle, json.loads((bundle / "model_spec.json").read_text())), indent=2))
        return
    if args.output.exists():
        parser.error("Refusing to overwrite an existing asset bundle")
    if args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error("Generated output must be outside the read-only source snapshot")
    exporter, exporter_path = load_exporter()
    provenance = json.loads(args.provenance.read_text())
    for name, digest in provenance["source_files_sha256"].items():
        if exporter.sha(args.source / name) != digest:
            parser.error(f"Source snapshot SHA mismatch: {name}")
    installation = json.loads((args.inputs / "installation.json").read_text())
    mechanical_limits = json.loads((args.inputs / "mechanical_limits.json").read_text())
    fit = json.loads((args.inputs / "fit_10mpa.json").read_text())
    if (installation["stroke_m"] != fit["stroke_m"] or
            not np.isclose(installation["spring_extended_length_m"] +
                           installation["adapter_effective_length_m"], .232) or
            not np.isclose(installation["stroke_m"], .080)):
        parser.error("V6 installation must use confirmed 232 mm pin distance / 80 mm stroke")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory(prefix=".v6-build-", dir=args.output.parent) as temporary:
        out = Path(temporary) / "bundle"
        out.mkdir()
        for folder in ("meshes", "collisions", "tools"):
            (out / folder).mkdir()
        spec, evidence = build_spec(args.source, out, exporter, installation, mechanical_limits)
        spec["model_kind"] = MODEL_KIND
        spec["coordinate_schema"] = "v6_download_link_frames_v1"
        exporter.save(out / "model_spec.json", spec)
        for name, value in evidence.items():
            exporter.save(out / f"{name}.json", value)
        adapt_exports(out, spec, exporter)
        result = validate(out, spec)
        exporter.save(out / "validation.json", result)
        exporter.save(out / "constraints.json", {
            "constraints": spec["constraints"], "urdf_is_tree_only": True,
            "required_for_dynamics": True, "point_equations": 18,
            "independent_constraint_rank": 12, "independent_internal_velocities": 6})
        exporter.save(out / "gas_spring_binding.json", spec["spring_binding"])
        for path in sorted(args.inputs.glob("*.json")):
            shutil.copyfile(path, out / path.name)
        for path in sorted(Path(__file__).parent.glob("*.py")):
            shutil.copyfile(path, out / "tools" / path.name)
        for name in ("urdf_v5.0.urdf", "urdf_v5.0.csv"):
            shutil.copyfile(args.source / "urdf" / name, out / ("source_" + name))
        if (ROOT / "README.md").is_file():
            shutil.copyfile(ROOT / "README.md", out / "README.md")
        exporter.save(out / "source_provenance.json", provenance)
        signs = {"L_joint1": -1, "LL_joint1": -1, "R_joint1": -1, "RR_joint1": -1,
                 "LLL_joint2": -1, "RRR_joint2": -1, "RR_joint2": -1,
                 "RR_joint3": -1, "RR_joint4": -1}
        exporter.save(out / "coordinate_mapping.json", {
            "schema_version": 1, "control_frame": "Xforward_Yleft_Zup",
            "source_root_to_control": S.tolist(),
            "v5_local_to_v6_local_rotation": V5_LOCAL_TO_V6.tolist(),
            "joint_sign_from_v5_232mm": {j["name"]: signs.get(j["name"], 1) for j in spec["joints"]},
            "coordinate_rule": "q_v6=sign*q_v5; dq_v6=sign*dq_v5; tau_v6=sign*tau_v5",
            "spring_slide_sign_from_v5": 1,
            "encoder_zero_or_hardware_binding": "not_inferred_from_asset",
            "nominal_method": "new_V6_loop_solve_continued_from_new_CAD_zero_not_copied_motor_values",
            "root_extra_Rz90_applied": False})
        inertia = evidence["inertial_sources"]
        manifest = {
            "schema_version": 1, "model_kind": MODEL_KIND, "coordinate_schema": spec["coordinate_schema"],
            "urdf": "robot.urdf", "usd": "robot.usda", "mjcf": "robot.xml",
            "usd_prim_path": "/Robot", "usd_articulation_root_path": "/Robot/base_link",
            "source_urdf_sha256": SOURCE_SHA256,
            "control_frame": "Xforward_Yleft_Zup", "control_joint_names": ACTIVE,
            "active_motor_mapping_verified": False,
            "rigid_body_names": [b["name"] for b in spec["bodies"]],
            "tree_joint_names": [j["name"] for j in spec["joints"]],
            "spring_joint_names": SPRINGS, "nominal_joint_pos": spec["nominal_joint_pos"],
            "nominal_base_height_m": .32, "total_mass_kg": result["total_mass_kg"],
            "mass_change_from_v5_232mm": {"link": "RR_link2", "old_mass_kg": .84,
                "new_mass_kg": .084, "source": "new_CAD_export_supported_by_symmetry_and_volume_not_weighed"},
            "wheel_gear_ratio": 15.8, "wheel_gear_ratio_source": "user_confirmed_20260928",
            "wheel_action_reference": "wheel_output_rad_per_s_no_extra_ratio_in_policy",
            "passive_damping": {"active_Nm_s_per_rad": 0., "passive_hinge_Nm_s_per_rad": .002,
                "slider_N_s_per_m": .002, "source": "inherited_unidentified_research_prior",
                "USD_application": "runtime_effort_actuators_only_no_implicit_USD_drive"},
            "self_collision_enabled": False,
            "source_inertia_invalid_count": sum(v["source_invalid"] for v in inertia.values()),
            "inertia_rebuilt_count": sum(v["method"] != "source_CAD_preserved" for v in inertia.values()),
            "inertia_adaptation": "explicit_invalid_or_reviewed_floor_link_owned_mesh_prior",
            "knee_inner_limits_deg": list(mechanical_limits["knee_inner_angle_deg"]),
            "knee_limits_source": mechanical_limits["source"],
            "knee_angle_definition": "pin_center_vectors_projected_normal_to_common_hinge_axis",
            "rmcs_105_deg_record": "requires_consistency_check_not_an_authorized_change_to_frozen_40_110",
            "static_validation_passed": True,
            "dynamics_validation": "pending", "physx_validation": "pending_GPU_not_run",
            "runtime_spring_force_required": True,
            "spring_force_runtime": "tools/validate_v6_dynamics.py applies the documented curve each step",
            "hardware_deployment_ready": False, "training_ready": False,
            "pending_gates": ["bounded_MuJoCo_dynamics", "CPU_domain_and_virtual_work",
                "PhysX_comparison", "collision_and_contact_review", "hardware_motor_and_spring_identification"],
            "files_sha256": {str(p.relative_to(out)): exporter.sha(p)
                             for p in sorted(out.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}}
        exporter.save(out / "manifest.json", manifest)
        shutil.move(str(out), str(args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
