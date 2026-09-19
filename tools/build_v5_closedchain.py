#!/usr/bin/env python3
"""Build V5 16R+2P research geometry with six real loop constraints.

Reuse the previous mechanism exporters; retain the source snapshot and record
every mesh ownership, axis-rounding, and research-inertia adaptation separately.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from v5_mechanism import fk, origin, point, source_spec, transform


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]
POSE_COORDINATES = ["L_joint1", "L_joint2", "L_joint3", "R_joint1", "R_jonit2", "R_joint3"]
SPRINGS = ["L_spring_slide", "R_spring_slide"]
S = transform()
S[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]


def load_exporter():
    bundled = Path(__file__).with_name("mechanism_export_legacy.py")
    path = bundled if bundled.exists() else ROOT / "model/纯底盘/urdf/tools/build_chassis_closedchain.py"
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
        raise RuntimeError(f"V5 closure solve failed: {result.message}, {residual(result.x)}")
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
        target.write_bytes(b"V5 owned source triangles; see mesh_ownership.json".ljust(80, b" ")
                           + struct.pack("<I", len(selected)) + selected.tobytes())
        meshes[name] = trimesh.load(target, force="mesh", process=True)
        report[name] = {"source_sha256": exporter.sha(path), "source_faces": len(records),
            "kept_faces": len(selected), "removed_duplicate_components": removed,
            "retained_vertices_modified": False}
    return meshes, report


def build_spec(source, out, exporter):
    robot, spec = source_spec(source)
    raw_frames = fk(spec, {})
    meshes, ownership = clean_meshes(source, out, raw_frames, exporter)
    # Body frames remain the CAD link frames; only the root uses the established control convention.
    for j in spec["joints"]:
        if j["parent"] == "base_link":
            j["origin"] = (S @ np.asarray(j["origin"])).tolist()
        j["limit"] = {"effort": "100", "velocity": "100"}
        if j["name"] in ("L_joint2", "R_jonit2"):
            zero_inner = np.pi - 2.3573
            bounds = np.deg2rad([35., 80.]) - zero_inner
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
        estimated = bool(eig[0] <= 0 or eig[0] + eig[1] < eig[2] - 1e-12)
        mesh = meshes[name]
        if estimated:
            if mesh.volume <= 0:
                raise RuntimeError(f"No positive source volume for research inertia: {name}")
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
        loop(f"{side}_rocker_P", side + "_link1", side * 2 + "_link3", [.1134, 0, 0], [.1345073, -.0115232, 0])
        loop(f"{side}_upper_E", side + "_link2", side * 2 + "_link4", [-.0647772, -.0171149, 0], [.0966, 0, 0])
        upper, lower = side * 3 + "_link1", side * 3 + "_link2"
        removed = next(j for j in spec["joints"] if j["child"] == upper)
        original_parent, original_origin = removed["parent"], np.array(removed["origin"])
        distance = np.linalg.norm(frames[upper][:3, 3] - frames[lower][:3, 3])
        line = (frames[upper][:3, 3] - frames[lower][:3, 3]) / distance
        relative = np.linalg.inv(frames[lower]) @ frames[upper]
        axis = frames[upper][:3, :3].T @ line
        # Mesh end planes define a research reference, not verified hardware stops.
        rod = next(p for p in meshes[upper].split(only_watertight=False) if len(p.faces) == 78)
        cylinder = next(p for p in meshes[lower].split(only_watertight=False) if len(p.faces) == 912)
        stroke = float(np.ptp(rod.bounds[:, 0]))
        free_length = float(cylinder.bounds[1, 0] + stroke + rod.bounds[0, 0])
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
            "reference_source": "owned CAD body/rod axial endpoints plus 80mm catalogue stroke",
            "reference_hardware_verified": False,
            "body_axial_bounds_m": cylinder.bounds[:, 0].tolist(), "rod_axial_bounds_m": rod.bounds[:, 0].tolist()}
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
        prescribed = dict(zip(POSE_COORDINATES, np.array([.42, .40, 0, -.42, -.40, 0]) * fraction))
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
    root.SetCustomData({"model_kind": "v5_gas_spring_closedchain_research", "hardware_deployment_ready": False})
    stage.GetRootLayer().Save()
    mj = ET.parse(out / "robot.xml").getroot()
    mj.set("model", "v5_gas_spring_closedchain_research")
    for j in spec["joints"]:
        if j["type"] == "prismatic":
            node = mj.find(f".//joint[@name='{j['name']}']")
            node.set("type", "slide")
            node.set("limited", "true")
            node.set("range", f"{j['limit']['lower']} {j['limit']['upper']}")
            ET.SubElement(mj.find("actuator"), "motor", name=j["name"] + "_gas_force", joint=j["name"], gear="1")
    exporter.write_xml(out / "robot.xml", mj)
    urdf = ET.parse(out / "robot.urdf").getroot()
    urdf.set("name", "v5_gas_spring_closedchain_research")
    exporter.write_xml(out / "robot.urdf", urdf)


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
    parser.add_argument("--source", type=Path, default=ROOT / "model/纯底盘_v5/source")
    parser.add_argument("--output", type=Path, default=ROOT / "model/纯底盘_v5/urdf")
    parser.add_argument("--spring-data-dir", type=Path, help="Directory containing fit_10mpa.json and catalogue_points.json")
    parser.add_argument("--validate-only", type=Path, metavar="BUNDLE", help="Read-only static/FK/rank validation of an existing bundle")
    args = parser.parse_args()
    if args.validate_only:
        bundle = args.validate_only.resolve()
        spec = json.loads((bundle / "model_spec.json").read_text())
        print(json.dumps(validate(bundle, spec), indent=2))
        return
    if args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error("Generated output must be outside the read-only source directory")
    spring_dir = args.spring_data_dir or args.source.parent / "gas_spring"
    for name in ("catalogue_points.json", "fit_10mpa.json"):
        if not (spring_dir / name).is_file():
            parser.error(f"Missing spring input: {spring_dir / name}")
    exporter, exporter_path = load_exporter()
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ("meshes", "collisions", "tools"):
        (args.output / folder).mkdir()
    spec, evidence = build_spec(args.source, args.output, exporter)
    exporter.save(args.output / "model_spec.json", spec)
    for name, value in evidence.items():
        exporter.save(args.output / f"{name}.json", value)
    adapt_exports(args.output, spec, exporter)
    result = validate(args.output, spec)
    exporter.save(args.output / "validation.json", result)
    exporter.save(args.output / "constraints.json", {"constraints": spec["constraints"], "urdf_is_tree_only": True})
    exporter.save(args.output / "gas_spring_binding.json", spec["spring_binding"])
    for name in ("catalogue_points.json", "fit_10mpa.json"):
        shutil.copyfile(spring_dir / name, args.output / name)
    for path in (Path(__file__), Path(__file__).with_name("v5_mechanism.py")):
        shutil.copyfile(path, args.output / "tools" / path.name)
    shutil.copyfile(exporter_path, args.output / "tools/mechanism_export_legacy.py")
    shutil.copyfile(Path(__file__).with_name("validate_v5_dynamics.py"), args.output / "tools/validate_v5_dynamics.py")
    for name in ("urdf_v5.0.urdf", "urdf_v5.0.csv"):
        shutil.copyfile(args.source / "urdf" / name, args.output / ("source_" + name))
    manifest = {"model_kind": "v5_gas_spring_closedchain_research", "urdf": "robot.urdf", "usd": "robot.usda", "mjcf": "robot.xml",
        "control_frame": "Xforward_Yleft_Zup", "control_joint_names": ACTIVE, "active_motor_mapping_verified": False,
        "rigid_body_names": [b["name"] for b in spec["bodies"]], "tree_joint_names": [j["name"] for j in spec["joints"]],
        "spring_joint_names": SPRINGS, "nominal_joint_pos": spec["nominal_joint_pos"], "nominal_base_height_m": .32,
        "total_mass_kg": result["total_mass_kg"], "mass_asymmetry_preserved": "RR_link2=0.84 kg vs LL_link2=0.084 kg",
        "self_collision_enabled": False, "source_inertia_invalid_count": 10,
        "knee_inner_limits_deg": [35., 80.], "knee_limits_source": "previous_user_mechanical_range_applied_to_V5_raw_zero",
        "inertia_adaptation": "owned_mesh_uniform_density_for_invalid_source_tensors_only",
        "static_validation_passed": True, "dynamics_validation": "pending", "hardware_deployment_ready": False,
        "files_sha256": {str(p.relative_to(args.output)): exporter.sha(p) for p in args.output.rglob("*") if p.is_file()}}
    exporter.save(args.output / "manifest.json", manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
