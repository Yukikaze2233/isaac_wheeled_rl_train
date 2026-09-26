#!/usr/bin/env python3
"""Build and validate a portable, source-frame-preserving research mechanism.

This tool deliberately has no task/environment imports. Build inputs are read-only;
publication refuses existing destinations. --validate-only needs only the bundle.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import struct
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


CONTROL = ["L_joint1", "L_joint2", "L_joint3", "R_joint1", "R_jonit2", "R_joint3"]
PARTS = {"C0": "crank", "C1": "lower_coupler", "C2": "link1",
         "C3": "rocker", "C4": "upper_coupler", "shank": "link2", "wheel": "link3"}
PIVOTS = {"C0": "C0_O", "C1": "C1_A", "C3": "C3_P", "C4": "C4_B"}
COLORS = {"C0": (.57, .23, .78), "C1": (.85, .60, .10), "C2": (.48, .54, .57),
          "C3": (.08, .66, .38), "C4": (.95, .30, .06), "shank": (.10, .40, .90),
          "wheel": (.08, .10, .14), "base": (.60, .65, .74)}
GEOMETRY_SHA = "4d502e88c5dcb32ad3f88e3c9bbcf2dc8aede3b615d27fbcdf69b5202d3099b9"
KINEMATICS_SHA = "c4e3ceead2e61e9bb75c9077e02c611cea4bcb49bdd7865446ee3c4af5e88bd0"
ORIGINAL_SHA = "be0fe413f1df85c9da4ac97027848f4526ad203a1ec36dab72c0faf5c6f7d6f9"
STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def numbers(value):
    return " ".join(format(float(x), ".17g") for x in np.asarray(value).ravel())


def vector(value):
    return np.fromstring(value, sep=" ")


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def origin(element):
    return transform(vector(element.get("xyz", "0 0 0")), vector(element.get("rpy", "0 0 0")))


def rotation(axis, q):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(np.asarray(axis) * q).as_matrix()
    return result


def point(matrix, xyz):
    return matrix[:3, :3] @ xyz + matrix[:3, 3]


def quaternion(matrix):
    xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.roll(xyzw, 1)


def add_origin(parent, matrix):
    ET.SubElement(parent, "origin", xyz=numbers(matrix[:3, 3]),
                  rpy=numbers(Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")))


def write_xml(path, root):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_stl(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    assert len(raw) == 84 + count * 50
    return np.frombuffer(raw, dtype=STL_DTYPE, offset=84, count=count)


class Geometry:
    """Offline fixed-branch circle reference; never called during dynamics steps."""

    def __init__(self, data):
        self.data = data

    @staticmethod
    def circle(a, r, b, s, branch):
        length = np.linalg.norm(b - a)
        assert abs(r - s) < length < r + s
        u = (b - a) / length
        along = (r * r - s * s + length * length) / (2 * length)
        return a + along * u + branch * np.sqrt(r * r - along * along) * np.array([-u[1], u[0]])

    @staticmethod
    def rod(a0, b0, a, b):
        u, v = b0 - a0, b - a
        angle = np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)
        matrix = rotation([0, 0, 1], angle)
        matrix[:2, 3] = a - matrix[:2, :2] @ a0
        return matrix

    def pose(self, q, root=None):
        root = np.eye(4) if root is None else root
        canonical = transform(rpy=[0, 0, np.pi / 2])
        canonical[:3, :3] = self.data["source_to_control_rotation"]
        frames = {"base_link": root @ canonical}
        for name, j in self.data["source_joint_frames"].items():
            frames[j["child"]["link"]] = (frames[j["parent"]["link"]] @ origin(j["origin"])
                                           @ rotation(vector(j["axis"]["xyz"]), q[name]))
        result = {"base": frames["base_link"]}
        for side, data in self.data["sides"].items():
            xy = {k: np.array(v) for k, v in data["axes_local_xy_m"].items()}
            knee = "L_joint2" if side == "L" else "R_jonit2"
            jf = self.data["source_joint_frames"][knee]
            shank = origin(jf["origin"]) @ rotation(vector(jf["axis"]["xyz"]), q[knee])
            e = point(shank, np.r_[xy["shank_E"], 0])[:2]
            p, o = xy["C3_P"], xy["C0_O"]
            b = self.circle(p, np.linalg.norm(xy["C3_B"] - p), e,
                            np.linalg.norm(xy["C4_E"] - xy["C4_B"]), data["upper_branch"])
            rocker = self.rod(p, xy["C3_B"], p, b)
            d = point(rocker, np.r_[xy["C3_D"], 0])[:2]
            a = self.circle(o, np.linalg.norm(xy["C0_A"] - o), d,
                            np.linalg.norm(xy["C1_D"] - xy["C1_A"]), data["lower_branch"])
            local = {"C0": self.rod(o, xy["C0_A"], o, a),
                     "C1": self.rod(xy["C1_A"], xy["C1_D"], a, d), "C2": np.eye(4),
                     "C3": rocker, "C4": self.rod(xy["C4_B"], xy["C4_E"], b, e)}
            for part, matrix in local.items():
                result[f"{side}_{part}"] = frames[f"{side}_link1"] @ matrix
            result[f"{side}_shank"] = frames[f"{side}_link2"]
            result[f"{side}_wheel"] = frames[f"{side}_link3"]
        return result


def tree_fk(spec, q, root=None):
    frames = {"base_link": np.eye(4) if root is None else root}
    for j in spec["joints"]:
        frames[j["child"]] = (frames[j["parent"]] @ np.array(j["origin"])
                               @ rotation(j["axis"], q[j["name"]]))
    return frames


def solve_coordinates(spec, geometry, active):
    poses = geometry.pose(active)
    body_frames = {b["name"]: poses[b["component"]] @ np.linalg.inv(b["mesh_origin"])
                   for b in spec["bodies"]}
    q = dict(active)
    for j in spec["joints"]:
        if j["name"] in CONTROL:
            continue
        relative = np.linalg.inv(j["origin"]) @ np.linalg.inv(body_frames[j["parent"]]) @ body_frames[j["child"]]
        q[j["name"]] = float(np.arctan2(relative[1, 0], relative[0, 0]))
    return q


def read_inertia(link):
    node = link.find("inertial")
    frame = origin(node.find("origin"))
    attrs = node.find("inertia").attrib
    tensor = np.array([[float(attrs["i" + "".join(sorted(x + y))]) for y in "xyz"] for x in "xyz"])
    return float(node.find("mass").get("value")), frame[:3, 3], frame[:3, :3] @ tensor @ frame[:3, :3].T


def generate_spec(out, canonical, geometry_dir, metrology_path, original):
    from pxr import Usd, UsdGeom

    assert sha(original / "urdf/urdf_V4.0.urdf") == ORIGINAL_SHA
    assert sha(geometry_dir / "chassis.usdc") == GEOMETRY_SHA
    assert sha(geometry_dir / "kinematics.json") == KINEMATICS_SHA
    urdf = ET.parse(canonical / "robot.urdf").getroot()
    source_manifest = load(canonical / "manifest.json")
    kin = load(geometry_dir / "kinematics.json")
    metrology = load(metrology_path)
    shutil.copyfile(geometry_dir / "kinematics.json", out / "kinematics.json")
    geometry = Geometry(kin)
    zero = geometry.pose(dict.fromkeys(CONTROL, 0.))
    spec = {"bodies": [], "joints": [], "constraints": [], "nominal_base_height_m": .32,
            "control_frame": "Xforward_Yleft_Zup"}
    inertial_sources = {"split_model": "uniform-density research prior, NOT CAD mass properties",
                        "source_total_mass_kg": 12.752, "rejected_zip_document_total_kg": 13.404,
                        "source_inertia_signs_automatically_flipped": False, "bodies": {},
                        "aggregate_discrepancy": {s: metrology["sides"][s]["mass"] for s in ("L", "R")}}
    mesh_audit, collision_sources = {}, {}
    confirmed = Usd.Stage.Open(str(geometry_dir / "chassis.usdc"))
    for component in kin["component_names"]:
        if component == "base":
            name, source_link, part = "base_link", "base_link", "base"
            mesh_origin = origin(urdf.find("link[@name='base_link']/visual/origin"))
        else:
            side, part = component.split("_")
            name = f"{side}_{PARTS[part]}"
            source_link = f"{side}_" + ("link1" if part.startswith("C") else PARTS[part])
            pivot = np.r_[kin["sides"][side]["axes_local_xy_m"][PIVOTS[part]], 0.] if part in PIVOTS else np.zeros(3)
            mesh_origin = transform(-pivot)
        source_mesh = original / f"meshes/{source_link}.STL"
        assert sha(source_mesh) == sha(canonical / f"meshes/{source_link}.STL")
        records = read_stl(source_mesh)
        file = f"meshes/{component}.stl"
        if part.startswith("C"):
            props = metrology["sides"][side]["components"][part]
            records = records[props["source_face_ids"]]
            (out / file).write_bytes(b"Source records selected without vertex modification".ljust(80, b" ")
                                    + struct.pack("<I", len(records)) + records.tobytes())
            rho = .326 / sum(c["volume_m3"] for c in metrology["sides"][side]["components"].values())
            mass = rho * props["volume_m3"]
            com = point(mesh_origin, props["com_m"])
            r = mesh_origin[:3, :3]
            inertia = r @ (rho * np.array(props["inertia_unit_density_kg_m2"])) @ r.T
            inertial_sources["bodies"][name] = {
                "source": "independent full-precision signed-volume integration",
                "density_kg_m3": rho, "volume_m3": props["volume_m3"],
                "com_in_source_mesh_m": props["com_m"],
                "inertia_unit_density_at_source_com": props["inertia_unit_density_kg_m2"],
                "source_face_ids": props["source_face_ids"],
                "watertight_source": props["watertight"], "seam_diagnostic": props["seam_diagnostic"],
                "mesh_to_link_transform": mesh_origin.tolist()}
        else:
            shutil.copyfile(source_mesh, out / file)
            source_node = urdf.find(f"link[@name='{source_link}']")
            mass, com, inertia = read_inertia(source_node)
            inertial_sources["bodies"][name] = {
                "source": "canonical URDF; reused, not independently measured",
                "original_inertial_xml": ET.tostring(source_node.find("inertial"), encoding="unicode")}
        triangles = records["vertices"]
        visual = UsdGeom.Mesh(confirmed.GetPrimAtPath("/Chassis/" + kin["component_names"][component] + "/Mesh"))
        assert np.array_equal(triangles.reshape(-1, 3), np.asarray(visual.GetPointsAttr().Get()))
        mesh_audit[component] = {"source_mesh_sha256": sha(source_mesh), "triangles": len(triangles),
                                 "confirmed_usdc_points_exact": True, "source_vertices_unmodified": True}
        collisions = []
        if part.startswith("C"):
            cloud = trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
                                    faces=np.arange(triangles.size // 3).reshape(-1, 3), process=False)
            hull = cloud.convex_hull
            method = "convex hull of this component only; holes/concavities filled"
            if len(hull.vertices) > 255:
                # A conservative PCA box avoids engine-specific hull simplification.
                axes = np.linalg.eigh(np.cov(cloud.vertices.T))[1]
                if np.linalg.det(axes) < 0:
                    axes[:, 0] *= -1
                local = cloud.vertices @ axes
                low, high = local.min(0), local.max(0)
                hull = trimesh.creation.box(high - low)
                hull.apply_transform(transform())
                hull.vertices = (hull.vertices + (high + low) / 2) @ axes.T
                method = "enclosing PCA box; original component hull exceeds 255-vertex screen"
            proxy = f"collisions/{component}.obj"
            hull.export(out / proxy)
            collisions.append({"type": "mesh", "file": proxy, "origin": mesh_origin.tolist()})
            collision_sources[name] = {"method": method, "vertices": len(hull.vertices),
                                       "research_proxy": True, "self_collision_approved": False}
        else:
            for node in urdf.findall(f"link[@name='{source_link}']/collision"):
                primitive = node.find("geometry")[0]
                collision = {"type": primitive.tag, "origin": origin(node.find("origin")).tolist()}
                if primitive.tag == "mesh":
                    relative = primitive.get("filename")
                    shutil.copyfile(canonical / relative, out / relative)
                    collision["file"] = relative
                elif primitive.tag == "cylinder":
                    collision.update(radius=float(primitive.get("radius")), length=float(primitive.get("length")))
                else:
                    raise ValueError(primitive.tag)
                collisions.append(collision)
            collision_sources[name] = {"method": "canonical local convex proxies / measured enclosing tire cylinder",
                                       "research_proxy": True, "self_collision_approved": False}
        spec["bodies"].append({"name": name, "component": component, "mesh": file,
                               "mesh_origin": mesh_origin.tolist(), "mass": mass, "com": np.asarray(com).tolist(),
                               "inertia": np.asarray(inertia).tolist(), "collisions": collisions,
                               "color": list(COLORS[part])})
    body_zero = {b["name"]: zero[b["component"]] @ np.linalg.inv(b["mesh_origin"]) for b in spec["bodies"]}
    for side in ("L", "R"):
        for joint in urdf.findall("joint"):
            if not joint.get("name").startswith(side):
                continue
            limit = dict(joint.find("limit").attrib)
            if joint.get("type") == "continuous":
                limit.pop("lower", None)
                limit.pop("upper", None)
            spec["joints"].append({"name": joint.get("name"), "parent": joint.find("parent").get("link"),
                                    "child": joint.find("child").get("link"), "type": joint.get("type"),
                                    "origin": origin(joint.find("origin")).tolist(),
                                    "axis": vector(joint.find("axis").get("xyz")).tolist(),
                                    "limit": limit, "canonical_xml": ET.tostring(joint, encoding="unicode")})
        for parent, child, suffix in (("link1", "crank", "crank_O"),
                                      ("crank", "lower_coupler", "lower_A"),
                                      ("link1", "rocker", "rocker_P"),
                                      ("rocker", "upper_coupler", "upper_B")):
            parent, child = f"{side}_{parent}", f"{side}_{child}"
            spec["joints"].append({"name": f"{side}_{suffix}", "parent": parent, "child": child,
                                    "type": "continuous", "origin": (np.linalg.inv(body_zero[parent]) @ body_zero[child]).tolist(),
                                    "axis": [0., 0., 1.], "limit": {"effort": "100", "velocity": "100"}})
        xy = kin["sides"][side]["axes_local_xy_m"]
        bodies = {b["name"]: b for b in spec["bodies"]}
        for tag, child0, feature0, child1, feature1 in (
                ("lower_D", "lower_coupler", "C1_D", "rocker", "C3_D"),
                ("upper_E", "upper_coupler", "C4_E", "link2", "shank_E")):
            b0, b1 = f"{side}_{child0}", f"{side}_{child1}"
            p0 = point(np.array(bodies[b0]["mesh_origin"]), np.r_[xy[feature0], 0])
            p1 = point(np.array(bodies[b1]["mesh_origin"]), np.r_[xy[feature1], 0])
            spec["constraints"].append({"name": f"{side}_{tag}", "type": "spherical", "body0": b0, "body1": b1,
                                        "local_pos0_m": p0.tolist(), "local_pos1_m": p1.tolist(),
                                        "local_axis0": [0, 0, 1], "local_axis1": [0, 0, 1],
                                        "physics:excludeFromArticulation": True,
                                        "axis_note": "Parallel hinge axes; spherical closure adds point coincidence only. z=0 is an axis-line representative, not a pin midplane."})
    spec["nominal_joint_pos"] = solve_coordinates(spec, geometry, source_manifest["nominal_joint_pos"])
    save(out / "model_spec.json", spec)
    save(out / "inertial_sources.json", inertial_sources)
    save(out / "collision_sources.json", collision_sources)
    save(out / "mesh_audit.json", mesh_audit)
    save(out / "constraints.json", {"constraints": spec["constraints"],
                                   "urdf_is_tree_only": True, "required_for_dynamics": True,
                                   "planar_independent_constraint_rank": 8,
                                   "point_equations": 12, "mimic_is_not_valid": True})
    save(out / "joint_mapping.json", {
        "control_joint_names": CONTROL, "control_coordinate_scope": "equivalent_output_joints_not_hardware_mapping",
        "hardware_deployment_ready": False, "real_C0_motor_mapping": "pending",
        "passive_joint_names": [j["name"] for j in spec["joints"] if j["name"] not in CONTROL],
        "passive_axis_convention": "+Z in each new child frame; q=0 is exact source-active-q=0 closure",
        "active_raw_zero_inner_deg": {s: float(np.rad2deg(np.pi + d["source_knee_origin_z_rad"])) for s, d in kin["sides"].items()},
        "active_effort_velocity_metadata": "copied verbatim from canonical (100 Nm / 1 rad/s); not identified hardware specifications",
        "passive_effort_velocity_metadata": "100 Nm / 100 rad/s parser placeholders, not motors, stops or hardware limits",
        "urdf_usd_drives": "none", "mjcf_actuators": "six unit-gear generalized torque research inputs, zero by default",
        "source_zero_closure_corrections": {s: metrology["sides"][s]["linkage"]["zero_closure_adjustment_m"] for s in ("L", "R")},
        "nominal_joint_pos": spec["nominal_joint_pos"],
        "component_to_body": {b["component"]: b["name"] for b in spec["bodies"]}})
    return spec


def export_urdf(out, spec):
    robot = ET.Element("robot", name="own_v40_closedchain_research")
    for b in spec["bodies"]:
        node = ET.SubElement(robot, "link", name=b["name"])
        inertial = ET.SubElement(node, "inertial")
        add_origin(inertial, transform(b["com"]))
        ET.SubElement(inertial, "mass", value=numbers([b["mass"]]))
        i = np.array(b["inertia"])
        ET.SubElement(inertial, "inertia", **{"i" + a + c: numbers([i[k, l]])
                                             for k, a in enumerate("xyz") for l, c in enumerate("xyz") if k <= l})
        visual = ET.SubElement(node, "visual")
        add_origin(visual, np.array(b["mesh_origin"]))
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=b["mesh"])
        material = ET.SubElement(visual, "material", name=b["component"])
        ET.SubElement(material, "color", rgba=numbers(b["color"] + [1]))
        for n, c in enumerate(b["collisions"]):
            col = ET.SubElement(node, "collision", name=f"{b['name']}_{n:03d}")
            add_origin(col, np.array(c["origin"]))
            geo = ET.SubElement(col, "geometry")
            if c["type"] == "mesh":
                ET.SubElement(geo, "mesh", filename=c["file"])
            else:
                ET.SubElement(geo, "cylinder", radius=numbers([c["radius"]]), length=numbers([c["length"]]))
    for j in spec["joints"]:
        if "canonical_xml" in j:
            joint = ET.fromstring(j["canonical_xml"])
            joint.find("limit").attrib = j["limit"]
            robot.append(joint)
        else:
            joint = ET.SubElement(robot, "joint", name=j["name"], type=j["type"])
            ET.SubElement(joint, "parent", link=j["parent"])
            ET.SubElement(joint, "child", link=j["child"])
            add_origin(joint, np.array(j["origin"]))
            ET.SubElement(joint, "axis", xyz=numbers(j["axis"]))
            ET.SubElement(joint, "limit", **j["limit"])
    write_xml(out / "robot.urdf", robot)
    # URDF has no standard collision filtering or loop closure representation.
    srdf = ET.Element("robot", name=robot.get("name"))
    for a, b in itertools.combinations([b["name"] for b in spec["bodies"]], 2):
        ET.SubElement(srdf, "disable_collisions", link1=a, link2=b, reason="ResearchSelfCollisionDisabled")
    write_xml(out / "collision_filters.srdf", srdf)


def export_mjcf(out, spec):
    root = ET.Element("mujoco", model="own_v40_closedchain_research")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false", fusestatic="false")
    option = ET.SubElement(root, "option", timestep="0.0005", gravity="0 0 -9.81", integrator="implicitfast",
                           solver="Newton", iterations="100", tolerance="1e-10", jacobian="dense")
    ET.SubElement(option, "flag", energy="enable")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.002", armature="0", solreflimit="0.002 1", solimplimit="0.99 0.999 0.001")
    ET.SubElement(default, "geom", friction="0.5 0.005 0.0001", solref="0.004 1", solimp="0.95 0.99 0.001")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="ground", type="plane", size="3 3 0.1", rgba="0.25 0.28 0.3 1",
                  contype="1", conaffinity="2")
    joints_by_child = {j["child"]: j for j in spec["joints"]}
    body_nodes, joint_order = {}, []

    def add_body(b):
        name = b["name"]
        j = joints_by_child.get(name)
        parent = body_nodes[j["parent"]] if j else world
        t = np.array(j["origin"]) if j else transform([0, 0, .32])
        body = ET.SubElement(parent, "body", name=name, pos=numbers(t[:3, 3]), quat=numbers(quaternion(t)))
        body_nodes[name] = body
        if j:
            attrs = dict(name=j["name"], type="hinge", axis=numbers(j["axis"]), limited=str(j["type"] == "revolute").lower())
            if j["type"] == "revolute":
                attrs["range"] = f"{j['limit']['lower']} {j['limit']['upper']}"
            ET.SubElement(body, "joint", **attrs)
            joint_order.append(j["name"])
        else:
            ET.SubElement(body, "freejoint", name="floating_base")
        i = np.array(b["inertia"])
        ET.SubElement(body, "inertial", pos=numbers(b["com"]), mass=numbers([b["mass"]]),
                      fullinertia=numbers([i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]))
        ET.SubElement(asset, "mesh", name=name + "_visual", file=b["mesh"])
        t = np.array(b["mesh_origin"])
        ET.SubElement(body, "geom", name=name + "_visual", type="mesh", mesh=name + "_visual",
                      pos=numbers(t[:3, 3]), quat=numbers(quaternion(t)), rgba=numbers(b["color"] + [1]),
                      contype="0", conaffinity="0", group="2", density="0")
        for n, c in enumerate(b["collisions"]):
            cname = f"{name}_collision_{n:03d}"
            t = np.array(c["origin"])
            attrs = dict(name=cname, type=c["type"], pos=numbers(t[:3, 3]), quat=numbers(quaternion(t)),
                         contype="2", conaffinity="1", group="3", rgba="0.5 0.5 0.5 0", density="0")
            if c["type"] == "mesh":
                ET.SubElement(asset, "mesh", name=cname, file=c["file"])
                attrs["mesh"] = cname
            else:
                attrs["size"] = numbers([c["radius"], c["length"] / 2])
            ET.SubElement(body, "geom", **attrs)
        for c in spec["constraints"]:
            for index in (0, 1):
                if c[f"body{index}"] == name:
                    ET.SubElement(body, "site", name=c["name"] + f"_{index}", pos=numbers(c[f"local_pos{index}_m"]),
                                  size="0.001", rgba="1 0 0 0")
        for child in spec["bodies"]:
            if child["name"] in joints_by_child and joints_by_child[child["name"]]["parent"] == name:
                add_body(child)

    add_body(spec["bodies"][0])
    eq = ET.SubElement(root, "equality")
    for c in spec["constraints"]:
        ET.SubElement(eq, "connect", name=c["name"], site1=c["name"] + "_0", site2=c["name"] + "_1",
                      solref="0.002 1", solimp="0.99 0.999 0.001")
    actuator = ET.SubElement(root, "actuator")
    for name in CONTROL:
        ET.SubElement(actuator, "motor", name=name + "_equivalent_torque", joint=name, gear="1")
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="nominal", qpos=numbers(
        [0, 0, .32, 1, 0, 0, 0] + [spec["nominal_joint_pos"][n] for n in joint_order]))
    write_xml(out / "robot.xml", root)


def export_usd(out, spec):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    visuals = Usd.Stage.CreateNew(str(out / "meshes/visuals.usdc"))
    visuals.SetDefaultPrim(UsdGeom.Xform.Define(visuals, "/Visuals").GetPrim())
    UsdGeom.SetStageMetersPerUnit(visuals, 1.)
    UsdGeom.SetStageUpAxis(visuals, "Z")

    def usd_mesh(stage, path, vertices, faces):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, dtype=np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr("none")
        return mesh

    def pose(node, t):
        xform = UsdGeom.Xformable(node)
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*t[:3, 3]))
        q = quaternion(t)
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:])))

    def quatf(t):
        q = quaternion(t)
        return Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:]))

    stage = Usd.Stage.CreateNew(str(out / "robot.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetCustomData({"model_kind": "coupled_fourbar_research", "hardware_deployment_ready": False,
                        "control_coordinate_scope": "equivalent_output_joints_not_hardware_mapping"})
    frames = tree_fk(spec, spec["nominal_joint_pos"], transform([0, 0, .32]))
    for b in spec["bodies"]:
        path = "/Robot/" + b["name"]
        body = UsdGeom.Xform.Define(stage, path).GetPrim()
        pose(body, frames[b["name"]])
        UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(b["mass"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*b["com"]))
        values, axes = np.linalg.eigh(b["inertia"])
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        principal = np.eye(4)
        principal[:3, :3] = axes
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
        mass.CreatePrincipalAxesAttr(quatf(principal))
        others = [Sdf.Path("/Robot/" + v["name"]) for v in spec["bodies"] if v["name"] != b["name"]]
        UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets(others)
        triangles = read_stl(out / b["mesh"])["vertices"]
        mesh = usd_mesh(visuals, "/Visuals/" + b["component"], triangles.reshape(-1, 3),
                        np.arange(len(triangles) * 3).reshape(-1, 3))
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*b["color"])])
        mesh.CreateDoubleSidedAttr(True)
        visual = stage.DefinePrim(path + "/Visual")
        visual.GetReferences().AddReference("meshes/visuals.usdc", "/Visuals/" + b["component"])
        pose(visual, np.array(b["mesh_origin"]))
        for n, c in enumerate(b["collisions"]):
            cpath = path + f"/Collision_{n:03d}"
            if c["type"] == "mesh":
                data = trimesh.load(out / c["file"], process=False, force="mesh")
                col = usd_mesh(stage, cpath, data.vertices, data.faces)
                UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")
            else:
                col = UsdGeom.Cylinder.Define(stage, cpath)
                col.CreateRadiusAttr(c["radius"])
                col.CreateHeightAttr(c["length"])
                col.CreateAxisAttr("Z")
            pose(col.GetPrim(), np.array(c["origin"]))
            UsdGeom.Imageable(col.GetPrim()).CreateVisibilityAttr("invisible")
            UsdPhysics.CollisionAPI.Apply(col.GetPrim()).CreateCollisionEnabledAttr(True)
    base = stage.GetPrimAtPath("/Robot/base_link")
    UsdPhysics.ArticulationRootAPI.Apply(base)
    # Author the genuine PhysX schema token without requiring an Isaac runtime.
    base.AddAppliedSchema("PhysxArticulationAPI")
    base.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    for j in spec["joints"]:
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/tree_joints/" + j["name"])
        joint.CreateBody0Rel().SetTargets(["/Robot/" + j["parent"]])
        joint.CreateBody1Rel().SetTargets(["/Robot/" + j["child"]])
        t = np.array(j["origin"])
        alignment = transform(rpy=[np.pi, 0, 0]) if j["axis"][2] < 0 else np.eye(4)
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*t[:3, 3]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(quatf(t @ alignment))
        joint.CreateLocalRot1Attr(quatf(alignment))
        joint.CreateExcludeFromArticulationAttr(False)
        if j["type"] == "revolute":
            joint.CreateLowerLimitAttr(float(np.rad2deg(float(j["limit"]["lower"]))))
            joint.CreateUpperLimitAttr(float(np.rad2deg(float(j["limit"]["upper"]))))
    for c in spec["constraints"]:
        joint = UsdPhysics.SphericalJoint.Define(stage, "/Robot/loop_joints/" + c["name"])
        joint.CreateBody0Rel().SetTargets(["/Robot/" + c["body0"]])
        joint.CreateBody1Rel().SetTargets(["/Robot/" + c["body1"]])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*c["local_pos0_m"]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*c["local_pos1_m"]))
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateExcludeFromArticulationAttr(True)
    visuals.GetRootLayer().Save()
    stage.GetRootLayer().Save()


def validate_static(out, external_reference=None):
    from pxr import Usd, UsdGeom, UsdPhysics

    spec, kin = load(out / "model_spec.json"), load(out / "kinematics.json")
    geometry = Geometry(kin)
    urdf = ET.parse(out / "robot.urdf").getroot()
    assert len(urdf.findall("link")) == 15 and len(urdf.findall("joint")) == 14
    assert urdf.find(".//mimic") is None
    parsed = copy.deepcopy(spec)
    parsed["joints"] = [{"name": j.get("name"), "parent": j.find("parent").get("link"),
                          "child": j.find("child").get("link"), "origin": origin(j.find("origin")).tolist(),
                          "axis": vector(j.find("axis").get("xyz")).tolist()} for j in urdf.findall("joint")]
    names = [b["name"] for b in spec["bodies"]]
    assert len(set(names)) == 15
    assert len({j["child"] for j in parsed["joints"]}) == 14
    assert set(names) - {j["child"] for j in parsed["joints"]} == {"base_link"}
    for mesh in urdf.findall(".//mesh"):
        path = Path(mesh.get("filename"))
        assert not path.is_absolute() and ".." not in path.parts and (out / path).is_file()
        assert mesh.get("scale") is None
    mass_errors, tensor_errors, min_eigenvalues = [], [], []
    for b in spec["bodies"]:
        mass, com, inertia = read_inertia(urdf.find(f"link[@name='{b['name']}']"))
        assert abs(mass - b["mass"]) < 1e-14
        assert np.linalg.norm(com - b["com"]) < 1e-14
        eig = np.linalg.eigvalsh(inertia)
        assert eig[0] > 0 and eig[0] + eig[1] >= eig[2] - 1e-14
        min_eigenvalues.append(float(eig[0]))
        assert np.max(np.abs(inertia - b["inertia"])) < 1e-14
        if b["component"].split("_")[-1].startswith("C"):
            source = load(out / "inertial_sources.json")["bodies"][b["name"]]
            records = read_stl(out / b["mesh"])
            triangles = records["vertices"].astype(float)
            mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3), faces=np.arange(len(triangles) * 3).reshape(-1, 3), process=False)
            rho = source["density_kg_m3"]
            mass_errors.append(abs(mesh.volume * rho - mass))
            # Rotate back to the mesh COM frame; translations do not change COM inertia.
            r = np.array(b["mesh_origin"])[:3, :3]
            tensor_errors.append(float(np.max(np.abs(r.T @ inertia @ r - mesh.moment_inertia * rho))))
            assert np.linalg.norm(point(np.linalg.inv(b["mesh_origin"]), com) - mesh.center_mass) < 1e-12
    assert max(mass_errors) < 1e-12 and max(tensor_errors) < 1e-12
    assert abs(sum(b["mass"] for b in spec["bodies"]) - 12.752) < 1e-12
    sample_q = [dict.fromkeys(CONTROL, 0.), {k: spec["nominal_joint_pos"][k] for k in CONTROL}]
    rng = np.random.default_rng(4072)
    for k in range(181):
        active = dict(zip(CONTROL, rng.uniform(-2., 2., 6)))
        for side, knee, angle in (("L", "L_joint2", 35 + k * .25), ("R", "R_jonit2", 80 - k * .25)):
            d = kin["sides"][side]
            active[knee] = (np.deg2rad(angle) - np.pi - d["source_knee_origin_z_rad"]) / d["source_knee_axis_sign"]
        active["L_joint3"], active["R_joint3"] = rng.uniform(-8 * np.pi, 8 * np.pi, 2)
        sample_q.append(active)
    max_fk, max_pin, max_axis, max_external = 0., 0., 0., 0.
    for active in sample_q:
        root = transform(rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3))
        q = solve_coordinates(spec, geometry, active)
        actual = tree_fk(parsed, q, root)
        reference = geometry.pose(active, root)
        for b in spec["bodies"]:
            visual = urdf.find(f"link[@name='{b['name']}']/visual/origin")
            matrix = actual[b["name"]] @ origin(visual)
            max_fk = max(max_fk, float(np.max(np.abs(matrix - reference[b["component"]]))))
        for c in spec["constraints"]:
            a, b = actual[c["body0"]], actual[c["body1"]]
            max_pin = max(max_pin, float(np.linalg.norm(point(a, c["local_pos0_m"]) - point(b, c["local_pos1_m"]))))
            max_axis = max(max_axis, float(np.linalg.norm(np.cross(a[:3, 2], b[:3, 2]))))
        if external_reference:
            ref, _ = external_reference.pose(root[:3, 3], Rotation.from_matrix(root[:3, :3]).as_quat(), active)
            max_external = max(max_external, *(float(np.max(np.abs(reference[n] - ref[n]))) for n in ref))
    assert max_fk < 1e-11 and max_pin < 1e-10 and max_axis < 1e-11 and max_external < 1e-11
    stage = Usd.Stage.Open(str(out / "robot.usda"))
    rigid = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    revolute = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    spherical = [p for p in stage.Traverse() if p.IsA(UsdPhysics.SphericalJoint)]
    assert len(rigid) == 15 and len(revolute) == 14 and len(spherical) == 4
    assert all(UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get() for p in spherical)
    assert all(not UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get() for p in revolute)
    assert not any("PhysicsDriveAPI" in api for p in stage.Traverse() for api in p.GetAppliedSchemas())
    assert not any(p.GetAttribute("physxJoint:excludeFromArticulation") for p in stage.Traverse())
    cache = UsdGeom.XformCache()
    nominal = tree_fk(spec, spec["nominal_joint_pos"], transform([0, 0, .32]))
    usd_pin, usd_axis, usd_tensor, usd_visual = 0., 0., 0., 0.
    for b in spec["bodies"]:
        prim = stage.GetPrimAtPath("/Robot/" + b["name"])
        t = np.array(cache.GetLocalToWorldTransform(prim)).T
        assert np.max(np.abs(t - nominal[b["name"]])) < 1e-12
        mass = UsdPhysics.MassAPI(prim)
        pq = mass.GetPrincipalAxesAttr().Get()
        r = Rotation.from_quat([*pq.GetImaginary(), pq.GetReal()]).as_matrix()
        usd_tensor = max(usd_tensor, float(np.max(np.abs(r @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ r.T - b["inertia"]))))
        assert abs(mass.GetMassAttr().Get() - b["mass"]) < 1e-6
        assert len(UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()) == 14
        visual = stage.GetPrimAtPath(str(prim.GetPath()) + "/Visual")
        usd_visual = max(usd_visual, float(np.max(np.abs(np.array(cache.GetLocalToWorldTransform(visual)).T
                                                   - t @ b["mesh_origin"]))))
        assert np.array_equal(np.asarray(UsdGeom.Mesh(visual).GetPointsAttr().Get()), read_stl(out / b["mesh"])["vertices"].reshape(-1, 3))
    for prim in revolute + spherical:
        joint = UsdPhysics.Joint(prim)
        poses = []
        axes = []
        for i in (0, 1):
            target = getattr(joint, f"GetBody{i}Rel")().GetTargets()[0]
            t = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(target))).T
            poses.append(point(t, getattr(joint, f"GetLocalPos{i}Attr")().Get()))
            q = getattr(joint, f"GetLocalRot{i}Attr")().Get()
            axes.append(t[:3, :3] @ Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()[:, 2])
        usd_pin = max(usd_pin, float(np.linalg.norm(poses[0] - poses[1])))
        usd_axis = max(usd_axis, float(np.linalg.norm(np.cross(*axes))))
    assert usd_pin < 1e-7 and usd_axis < 1e-6 and usd_tensor < 1e-7 and usd_visual < 1e-12
    collision_count = sum(len(b["collisions"]) for b in spec["bodies"])
    assert sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()) == collision_count
    for path in (out / "collisions").glob("*.obj"):
        mesh = trimesh.load(path, force="mesh", process=False)
        assert mesh.is_convex and mesh.is_watertight and len(mesh.vertices) <= 255
    return {"passed": True, "urdf_links": 15, "urdf_tree_joints": 14, "closed_chain_constraints": 4,
            "usd_rigid_bodies": 15, "usd_revolute_joints": 14, "usd_spherical_joints": 4,
            "collision_geoms": collision_count, "control_joints": 6, "passive_joints": 8,
            "pose_samples": len(sample_q), "random_root_and_hip_motion": True, "knee_inner_range_deg": [35, 80],
            "max_urdf_visual_matrix_error": max_fk, "max_closure_position_error_m": max_pin,
            "max_closure_axis_cross_norm": max_axis, "external_RepairedKinematics_checked": external_reference is not None,
            "max_external_reference_matrix_error": max_external if external_reference else None,
            "min_inertia_eigenvalue_kg_m2": min(min_eigenvalues), "max_split_mass_error_kg": max(mass_errors),
            "max_rebased_inertia_error_kg_m2": max(tensor_errors), "total_mass_kg": sum(b["mass"] for b in spec["bodies"]),
            "usd_float32_joint_position_error_m": usd_pin, "usd_float32_joint_axis_error": usd_axis,
            "usd_float32_inertia_error_kg_m2": usd_tensor, "usd_visual_matrix_error": usd_visual,
            "self_collision_validation": "not_passed_disabled_research_candidate", "isaac_simulation": "not_run"}


def validate_dynamics(out, seconds=4.):
    import mujoco

    spec = load(out / "model_spec.json")
    xml = ET.parse(out / "robot.xml").getroot()
    # Absolute references are used only in transient test strings, never exported.
    for mesh in xml.findall("asset/mesh"):
        mesh.set("file", str(out / mesh.get("file")))
    full = mujoco.MjModel.from_xml_path(str(out / "robot.xml"))
    assert full.nbody == 16 and full.njnt == 15 and full.neq == 4 and full.nu == 6
    assert abs(full.body_mass.sum() - 12.752) < 1e-12
    result = {"passed": True, "engine": "MuJoCo", "version": mujoco.__version__,
              "robot_rigid_bodies_excluding_world": full.nbody - 1, "tree_hinges_excluding_freejoint": full.njnt - 1,
              "equality_connect_constraints": full.neq, "actuators": full.nu,
              "isaac_simulation": "not_run", "ppo": "not_run", "cases": {},
              "timestep_s": full.opt.timestep, "joint_damping_Nm_s_per_rad": .002,
              "damping_source": "explicit numerical research prior, not measured",
              "passive_pose_writes_after_reset": 0, "per_step_kinematic_solver_calls": 0,
              "acceptance": {"max_pin_gap_m": 2e-4, "max_abs_qvel": 150., "max_abs_energy_J": 500.,
                             "max_abs_base_position_m": 5., "mujoco_warnings": 0}}

    def run(label, pinned, torque, constraints=True, duration=seconds):
        case_xml = copy.deepcopy(xml)
        if pinned:
            case_xml.find("worldbody/body").remove(case_xml.find("worldbody/body/freejoint"))
            case_xml.remove(case_xml.find("keyframe"))
        if not constraints:
            case_xml.remove(case_xml.find("equality"))
        model = mujoco.MjModel.from_xml_string(ET.tostring(case_xml, encoding="unicode"))
        data = mujoco.MjData(model)
        if not pinned:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            for name, q in spec["nominal_joint_pos"].items():
                data.qpos[model.joint(name).qposadr[0]] = q
        mujoco.mj_forward(model, data)
        pairs = [(model.site(c["name"] + "_0").id, model.site(c["name"] + "_1").id) for c in spec["constraints"]]
        passive = [model.joint(j["name"]).qposadr[0] for j in spec["joints"] if j["name"] not in CONTROL]
        initial_passive = data.qpos[passive].copy()
        initial_gap = max(np.linalg.norm(data.site_xpos[a] - data.site_xpos[b]) for a, b in pairs)
        assert initial_gap < 1e-9
        max_gap, max_speed, max_energy, max_pos, max_axis = 0., 0., 0., 0., 0.
        max_passive_motion = np.zeros(8)
        max_contacts, min_height, max_height = 0, float("inf"), -float("inf")
        samples = []
        steps = round(duration / model.opt.timestep)
        for step in range(steps):
            # Only generalized active torque is supplied. No qpos/xpos writes,
            # mocap, kinematic pose updates or hidden reset occur in this loop.
            if torque:
                data.ctrl[:] = .04 * np.sin(2 * np.pi * (step * model.opt.timestep) + np.arange(6) * .7)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.energy).all()
            gap = max(float(np.linalg.norm(data.site_xpos[a] - data.site_xpos[b])) for a, b in pairs)
            max_gap = max(max_gap, gap)
            max_speed = max(max_speed, float(np.max(np.abs(data.qvel))))
            max_energy = max(max_energy, float(np.max(np.abs(data.energy))))
            pos = data.xpos[model.body("base_link").id]
            max_pos = max(max_pos, float(np.max(np.abs(pos))))
            min_height, max_height = min(min_height, float(pos[2])), max(max_height, float(pos[2]))
            max_contacts = max(max_contacts, data.ncon)
            max_passive_motion = np.maximum(max_passive_motion, np.abs(data.qpos[passive] - initial_passive))
            for c in spec["constraints"]:
                axes = [data.xmat[model.body(c[k]).id].reshape(3, 3)[:, 2] for k in ("body0", "body1")]
                max_axis = max(max_axis, float(np.linalg.norm(np.cross(*axes))))
            if step % 200 == 0 or step == steps - 1:
                samples.append({"time_s": float(data.time), "max_pin_gap_m": gap,
                                "potential_kinetic_energy_J": data.energy.tolist(), "base_height_m": float(pos[2])})
        warnings = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
        case = {"steps": steps, "seconds": float(data.time), "fixed_base": pinned, "gravity_m_s2": [0, 0, -9.81],
                "active_torque_amplitude_Nm": .04 if torque else 0., "constraints_enabled": constraints,
                "initial_max_pin_gap_m": float(initial_gap), "max_pin_gap_m": max_gap,
                "max_axis_cross_norm": max_axis, "max_abs_qvel": max_speed, "max_abs_energy_J": max_energy,
                "final_potential_kinetic_energy_J": data.energy.tolist(), "max_abs_base_position_m": max_pos,
                "base_height_range_m": [min_height, max_height], "max_ground_contacts": max_contacts,
                "passive_motion_each_rad": max_passive_motion.tolist(), "warnings": warnings, "samples": samples}
        if constraints:
            assert max_gap < 2e-4, (label, "pin gap", max_gap)
            assert max_speed < 150 and max_energy < 500 and max_pos < 5 and not warnings, (label, case)
            assert np.min(max_passive_motion) > 1e-4, (label, "passive rods did not move")
            if not pinned:
                assert max_contacts > 0, "ground collision was not exercised"
        return case

    result["cases"]["free_gravity"] = run("free_gravity", False, False)
    result["cases"]["free_small_torque"] = run("free_small_torque", False, True)
    result["cases"]["fixed_base_gravity_small_torque"] = run("fixed_base_gravity_small_torque", True, True)
    ablation = run("constraints_disabled_ablation", True, True, constraints=False, duration=.3)
    assert ablation["max_pin_gap_m"] > .001
    result["cases"]["constraints_disabled_ablation"] = ablation
    result["passive_constraint_causality_demonstrated"] = True
    result["interpretation"] = "Uncontrolled gravity/fall and small torques are bounded; this is not balance control, hardware validation, or a self-collision approval."
    return result


README = """# 纯底盘：两级四杆闭链研究候选

这是独立完整的 `coupled_fourbar_research` 包。15 个真实刚体（含 base），14 个树转动关节，
4 个球形点闭合约束，共 18 个连接；6 个等效输出控制坐标、8 个被动坐标。
`robot.urdf` 是标准树；**仅导入 URDF 不会得到闭链**，必须同时应用 `constraints.json`。
可直接用 `robot.usda`（USD Physics）或 `robot.xml`（MuJoCo）导入闭链。

## 坐标与驱动

- 根 frame：X forward / Y left / Z up；原 canonical 六轴 origin、axis、raw q 保留。
- 控制顺序：`L_joint1,L_joint2,L_joint3,R_joint1,R_jonit2,R_joint3`，保留历史拼写。
- 每侧树：base→C2→shank→wheel；C2→C0@O→C1@A；C2→C3@P→C4@B。
  闭合为 C1–C3@D、C4–shank@E。新增 child 原点位于其父连接轴；visual/collision/COM 同步换帧。
- source active q=0 对应左膝内角约 42.261°、右约 44.937°，**不是 68°**。
  新增 passive q=0 采用该源零位精确闭合；微米级拟合调整见 `joint_mapping.json`。
- `nominal_joint_pos` 含全部 14 轴，六主动值不变；base 名义高度 0.32 m。
  USD 直接 author 该名义姿态；MJCF 使用 `mj_resetDataKeyframe(model, data, 0)`。
- hip/wheel/passive 为 continuous，无 ±π stop；仅两膝保留原 raw 坐标下 35–80° 限位。
- URDF/USD 无 drive。MJCF 有六个默认零输入的 unit-gear torque actuator，非硬件电机映射。
  C0 实际驱动、编码器/传动映射仍 pending；`hardware_deployment_ready=false`。

## 质量、惯量与碰撞

总质量 **12.752 kg** = 10.8 + 2×(0.326 + 0.45 + 0.2)，不是旧 ZIP 文档的 13.404 kg。
base/shank/wheel 复用 canonical 原惯量（未实测、未自动翻转非对角项符号）。
C0–C4 按独立完整精度体积积分，每侧共同密度归一至 0.326 kg；这是
**uniform-density research prior，非 CAD 真值**，没有重复计入旧 aggregate 惯量。
与原 aggregate COM 的差约 3.449/4.237 mm，惯量 Frobenius 差约 38.803%/40.121%。
完整来源、原 tensor、逐杆 COM、单位密度积分和 C2 极小浮点接缝诊断见 `inertial_sources.json`。

15 个 visual STL 顶点保持源值、不缩放；USD 原生 mesh 同值。动态 collision 使用本地 convex
研究代理和原轮 cylinder（R≈0.06 m，半宽≈0.0125 m，轴偏移±0.02035 m），没有动态 triangle collider。
base/shank 沿用组件 convex，新增杆用各自 convex hull，超过 255 点时用保守 PCA box。
这些代理填充孔洞/凹陷，不代表精确实体接触。来源见 `collision_sources.json`。
新候选默认关闭全部 self-collision，外界/地面接触保留：USD 使用真实 FilteredPairsAPI
及 `physxArticulation:enabledSelfCollisions=false`，MJCF 使用互斥机器人/地面 bitmask，
URDF 提供研究用途 `collision_filters.srdf`。**新 self-collision 审核未通过**，未继承旧 6-pair 批准。
USD 闭合为 `UsdPhysics.SphericalJoint` + **`physics:excludeFromArticulation=true`**。
平面 hinge 树使 4 个三维点约束的独立秩为 8，保留六个内部独立自由度。

## 文件与复验

- `manifest.json`：训练交接接口、全文件 SHA256（不含 manifest 自身）、状态。
- `robot.urdf`, `robot.usda`, `robot.xml`：三种自包含相对路径入口。
- `model_spec.json`, `kinematics.json`：完整换帧模型和确认的纯几何输入；后者的旧 physics_ready=false
  属于源视觉数据标签，不是候选验证结果。候选结果以 validation 文件为准。
- `static_validation.json` 与 `dynamics_validation.json`：分开的静态与 MuJoCo 实测证据。
- `tools/build_chassis_closedchain.py`：完整生成/复验源码。

依赖已有 Python + numpy/scipy/trimesh/pxr/mujoco。无需 reports、Downloads、训练仓库或环境代码即可复验：

```bash
python tools/build_chassis_closedchain.py --validate-only .
```

动力学验证从名义姿态只初始化一次，后续不写 passive qpos/pose、不调用几何闭合求解器。
包含自由基座重力跌落、小力矩、固定基座重力小力矩及移除约束的对照。
固定基座仅是临时测试 fixture；三个正式模型均为自由基座。MJCF 阻尼 0.002 Nm·s/rad
是显式数值研究先验，USD/URDF 不注入驱动；MuJoCo 通过不等同于 PhysX 稳定性通过。
**Isaac/PPO 尚未运行，后续由主 agent 集成独立 snapshot。**

旧 `model/纯底盘/urdf_V4.0_multilink.urdf` 是保留的他人草稿，其 hip 原点、mesh/COM rebase 和
wheel limits 不适合作为本候选源。有效的新候选入口是**本目录的 `robot.urdf`/`robot.usda`/`robot.xml`**。
"""


def build(args):
    out = args.output.resolve()
    destinations = [out] + ([args.materialize.resolve()] if args.materialize else [])
    for path in destinations + ([args.zip.resolve()] if args.zip else []):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing destination: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Destination parent must exist: {path.parent}")
    protected = {}
    for directory in (args.canonical, args.original, args.geometry):
        for path in directory.rglob("*"):
            if path.is_file():
                protected[path.resolve()] = sha(path)
    protected[args.metrology.resolve()] = sha(args.metrology)
    with tempfile.TemporaryDirectory(prefix=".chassis-build-", dir=out.parent) as temporary:
        staging = Path(temporary)
        for name in ("meshes", "collisions", "tools"):
            (staging / name).mkdir()
        spec = generate_spec(staging, args.canonical, args.geometry, args.metrology, args.original)
        export_urdf(staging, spec)
        export_mjcf(staging, spec)
        export_usd(staging, spec)
        external = None
        if args.reference_module:
            module_spec = importlib.util.spec_from_file_location("confirmed_repaired_visuals", args.reference_module)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            external = module.RepairedKinematics(args.geometry)
        static = validate_static(staging, external)
        save(staging / "static_validation.json", static)
        dynamics = validate_dynamics(staging)
        save(staging / "dynamics_validation.json", dynamics)
        (staging / "README.md").write_text(README)
        package = ET.Element("package", format="3")
        for name, text in (("name", "chassis_closedchain"), ("version", "0.1.0"),
                           ("description", "Own V40 coupled four-bar research assets; hardware mapping pending")):
            ET.SubElement(package, name).text = text
        ET.SubElement(package, "maintainer", email="yukikaze@localhost").text = "Yukikaze"
        ET.SubElement(package, "license").text = "Proprietary; source CAD rights retained"
        write_xml(staging / "package.xml", package)
        shutil.copyfile(Path(__file__), staging / "tools/build_chassis_closedchain.py")
        assert all(sha(p) == digest for p, digest in protected.items()), "Protected input changed during build"
        save(staging / "source_integrity.json", {
            "all_protected_inputs_unchanged": True, "protected_file_count": len(protected),
            "source_original_urdf_sha256": ORIGINAL_SHA, "source_geometry_usdc_sha256": GEOMETRY_SHA,
            "canonical_urdf_sha256": sha(args.canonical / "robot.urdf"), "metrology_sha256": sha(args.metrology),
            "protected_sha256": {str(p): d for p, d in protected.items()},
            "absolute_paths_are_provenance_only": True, "runtime_external_dependencies": []})
        manifest = {
            "schema_version": 1, "model_kind": "coupled_fourbar_research", "robot_id": "own_v40",
            "urdf": "robot.urdf", "usd": "robot.usda", "mjcf": "robot.xml", "usd_prim_path": "/Robot",
            "usd_articulation_root_path": "/Robot/base_link", "total_mass_kg": 12.752,
            "source_total_mass_kg": 12.752, "rigid_body_names": [b["name"] for b in spec["bodies"]],
            "control_joint_names": CONTROL, "passive_joint_names": [j["name"] for j in spec["joints"] if j["name"] not in CONTROL],
            "tree_joint_names": [j["name"] for j in spec["joints"]],
            "nominal_joint_pos": spec["nominal_joint_pos"], "nominal_base_height_m": .32,
            "closed_chain_constraints": spec["constraints"], "constraints_file": "constraints.json",
            "control_coordinate_scope": "equivalent_output_joints_not_hardware_mapping", "hardware_deployment_ready": False,
            "inertia_source": "uniform_density_research_prior_for_split_parts", "control_frame": spec["control_frame"],
            "source_to_control_rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "knee_inner_limits_deg": [35., 80.],
            "self_collision_enabled": False, "self_collision_validation": {"passed": False, "status": "not_validated_disabled_for_research"},
            "external_collision_enabled": True, "urdf_import_requires_constraints_and_collision_filters": True,
            "static_validation": {"passed": True, "report": "static_validation.json"},
            "mujoco_dynamics_validation": {"passed": True, "report": "dynamics_validation.json"},
            "isaac_validation": {"passed": False, "status": "not_run"}, "ppo_training": "not_run",
            "files_sha256_scope": "All bundle files except manifest.json itself",
            "files_sha256": {str(p.relative_to(staging)): sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}}
        save(staging / "manifest.json", manifest)
        # Publish only after all static and dynamic checks pass.
        shutil.copytree(staging, out)
    if args.materialize:
        shutil.copytree(out, args.materialize)
        assert {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()} == {
            str(p.relative_to(args.materialize)): sha(p) for p in args.materialize.rglob("*") if p.is_file()}
    if args.zip:
        with zipfile.ZipFile(args.zip, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    archive.write(path, Path("chassis_closedchain") / path.relative_to(out))
    print(json.dumps({"output": str(out), "materialized": str(args.materialize),
                      "static": static, "dynamic_case_summary": {k: {n: v[n] for n in ("steps", "max_pin_gap_m", "max_abs_energy_J", "warnings")}
                                                                  for k, v in dynamics["cases"].items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--canonical", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--metrology", type=Path)
    parser.add_argument("--reference-module", type=Path)
    parser.add_argument("--materialize", type=Path)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--validate-only", type=Path)
    args = parser.parse_args()
    if args.validate_only:
        out = args.validate_only.resolve()
        manifest = load(out / "manifest.json")
        assert all(sha(out / name) == digest for name, digest in manifest["files_sha256"].items())
        print(json.dumps({"static": validate_static(out), "dynamics": validate_dynamics(out)}, indent=2))
    else:
        if any(getattr(args, name) is None for name in ("output", "canonical", "original", "geometry", "metrology")):
            parser.error("Build requires --output --canonical --original --geometry --metrology")
        build(args)


if __name__ == "__main__":
    main()
