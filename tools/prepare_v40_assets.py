#!/usr/bin/env python3
"""Build independent, canonical own_v40 assets; never run dynamics or Isaac.

Only the confirmed knee35-80 source is accepted. Original visual STL bytes and
all masses/inertias are retained. Per-connected-component convex envelopes are
conservative CANDIDATES, not a certified concave decomposition. Wheel collisions
use enclosing STL-measured cylinders. Exactly six named adjacent pairs are
reviewed explicitly; unexplained source-material overlap keeps the static gate closed. Imports have no file,
process, simulator or argument-parser side effects. numpy/scipy are build-time
requirements; MuJoCo is optional and is imported only by static_inspection().
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import xml.etree.ElementTree as ET

import numpy as np

SOURCE_SHA256 = "835352595029eccc1dee951d4192f6dca4b00674d54e77290f79778cdb399431"
SOURCE_MESH_SHA256 = {
    "meshes/base_link.STL": "6c0450b99bef4e81af3126e8a90a27519551c1a1c5e958071e1ecc257bfb06cc",
    "meshes/L_link1.STL": "a2143af42a81d432d40276f6564825e4bb7bb2bf90f00e8ccacd549703d136d9",
    "meshes/L_link2.STL": "488f414ade9ed70c133a8aa35a88d6e355079b40dc687113588b0bccce7ea362",
    "meshes/L_link3.STL": "9ed93f2ce8ada4dd723ed6bd697113060dfab3ef359c428a44ea0a280218877f",
    "meshes/R_link1.STL": "72a8d0c3eae84305322b5e5152b4cb9fd83a7b8ee8b4bfe489db311034ee3af1",
    "meshes/R_link2.STL": "0c0440586143684a28d53cd4dd71a41659b8ff693d9461d146b847dc4a4b696d",
    "meshes/R_link3.STL": "c137d9c0f6dc9d0f42e624938d9bb0545246da669592e78fa735f5503680e9a5",
}
# Conservative offline portability screen, not a substitute for PhysX cooking.
CONVEX_VERTEX_SCREEN = 255
ACTION_ORDER = ("L_joint1", "L_joint2", "L_joint3", "R_joint1", "R_jonit2", "R_joint3")
KNEE_LIMITS = {
    "L_joint2": (-0.1267274153917777, 0.6586707480056704),
    "R_jonit2": (-0.6119707480056704, 0.1734274153917776),
}
NOMINAL_JOINT_POS = {
    "L_joint1": 0.41526541073209217,
    "L_joint2": 0.44890796255584864,
    "L_joint3": 0.0,
    "R_joint1": -0.42250891703703075,
    "R_jonit2": -0.4129956195280641,
    "R_joint3": 0.0,
}
NOMINAL_BASE_HEIGHT_M = 0.32
WHEEL_LINKS = ("L_link3", "R_link3")
WHEEL_PROXY_GROUND_TOLERANCE_M = 1e-4  # 0.1 mm, only measured enclosing wheel proxies.
CONTROL_ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


# Embedded source-hash-bound evidence keeps rebuilds conservative even without
# the upstream reports or an optional --evidence argument. This is not geometry.
HIP_MATERIAL_REVIEW_HOLD_PATH = "evidence/hip_material_review_hold.json"
HIP_MATERIAL_REVIEW_HOLD_JSON = r'''{
  "automatic_hip_filter_approval_permitted": false,
  "findings": [
    "Both base/thigh source pairs have material-interior witnesses, including the original right nominal pose.",
    "Axial steps and larger-hole direction differ between left/right thigh component 000; correct CAD design/assembly is not established.",
    "No direct identical-triangle or whole-component duplicate-assignment proof was found; component IDs are not CAD body identities.",
    "Five SW instances exported into a thigh link do not establish rigid-body ownership or transmission/closed-chain equivalence."
  ],
  "historical_export_evidence_sha256": {
    "export.log": "4ce48c1e804971f2a8b8eac26fc118f7b9d35f37ca41add39cbe54ebb8b3fa78",
    "urdf_V4.0.csv": "79dbc0291ef86b087a490c8dcfdc3b30d7b2fb3b2782f37652dbbf1decd016b7"
  },
  "limitations": [
    "Finite source-section/winding/ray/triangle-distance evidence is not CAD solid Boolean certification or a hardware-interference verdict.",
    "Near-boundary right samples exist; representative and original-nominal witnesses retain actual depths rather than a blanket safety claim.",
    "Open/non-manifold source topology remains disclosed. Neither hip filter is automatically approved."
  ],
  "old_sampling": {
    "historical_unfiltered_penetrating_contacts": 73,
    "interpretation": "Zero in the old finite right-hip sample is not absence of source overlap or permission to approve its filter",
    "left_hip_jointly_occupied_witnesses": 3,
    "right_hip_jointly_occupied_witnesses": 0,
    "total_witnesses": 880
  },
  "pending_joints": [
    "L_joint1",
    "R_joint1"
  ],
  "required_next_step": "Obtain CAD assembly/body identities, link ownership and kinematic/mate evidence; controlled CAD correction and independent re-export/review precede V4 training.",
  "review_id": "v40-bilateral-hip-source-material-cad-first-v1",
  "right_actual_nominal_overlap": {
    "base": {
      "classification": "inside_evidence",
      "nearest_component_id": 0,
      "nearest_source_triangle_index_0based": 23373,
      "surface_distance_mm": 0.06871886856810629,
      "three_ray_inside_parities": [
        1,
        1,
        1
      ],
      "winding": 0.9999999999889181
    },
    "coordinate_frame": "canonical base, metres; no spawn-height offset",
    "id": "Rm-10",
    "point_control_m": [
      -0.0144627212179471,
      -0.19341950642903252,
      0.018116959671021265
    ],
    "q_L": 0.41526541073209217,
    "q_R": -0.42250891703703075,
    "thigh": {
      "classification": "inside_evidence",
      "nearest_component_id": 0,
      "nearest_source_triangle_index_0based": 5456,
      "surface_distance_mm": 0.0735362748459427,
      "three_ray_inside_parities": [
        1,
        1,
        1
      ],
      "winding": 1.0000000000000002
    }
  },
  "right_representatives": [
    {
      "base": {
        "classification": "inside_evidence",
        "nearest_component_id": 0,
        "nearest_source_triangle_index_0based": 86520,
        "surface_distance_mm": 0.5695798862728763,
        "three_ray_inside_parities": [
          1,
          1,
          1
        ],
        "winding": 0.9999999999868485
      },
      "id": "R0-4",
      "point_control_m": [
        -0.016920504872355854,
        -0.18475763055213298,
        0.025006572969207417
      ],
      "q_L": 0,
      "q_R": 0,
      "thigh": {
        "classification": "inside_evidence",
        "nearest_component_id": 0,
        "nearest_source_triangle_index_0based": 4857,
        "surface_distance_mm": 0.8695870146416085,
        "three_ray_inside_parities": [
          1,
          1,
          1
        ],
        "winding": 0.9999999999999997
      }
    },
    {
      "base": {
        "classification": "inside_evidence",
        "nearest_component_id": 0,
        "nearest_source_triangle_index_0based": 84153,
        "surface_distance_mm": 1.0012409599085819,
        "three_ray_inside_parities": [
          1,
          1,
          1
        ],
        "winding": 0.9999999999918052
      },
      "id": "R0-6",
      "point_control_m": [
        0.028889646244660633,
        -0.18598652289684292,
        0.003374475909836189
      ],
      "q_L": 0,
      "q_R": 0,
      "thigh": {
        "classification": "inside_evidence",
        "nearest_component_id": 0,
        "nearest_source_triangle_index_0based": 5497,
        "surface_distance_mm": 0.5558252924174646,
        "three_ray_inside_parities": [
          1,
          1,
          1
        ],
        "winding": 0.9999999999999999
      }
    },
    {
      "base": {
        "classification": "inside_evidence",
        "nearest_component_id": 0,
        "nearest_source_triangle_index_0based": 23373,
        "surface_distance_mm": 0.06871886856810629,
        "three_ray_inside_parities": [
          1,
          1,
          1
        ],
        "winding": 0.9999999999889181
      },
      "id": "Rm-10",
      "point_control_m": [
        -0.0144627212179471,
        -0.19341950642903252,
        0.018116959671021265
      ],
      "q_L": 0.41526541073209217,
      "q_R": -0.41526541073209217,
      "thigh": {
        "classification": "inside_evidence",
        "nearest_component_id": 0,
        "nearest_source_triangle_index_0based": 5456,
        "surface_distance_mm": 0.10762301974218852,
        "three_ray_inside_parities": [
          1,
          1,
          1
        ],
        "winding": 1.0000000000000007
      }
    }
  ],
  "schema_version": 1,
  "scope": "Review-state synchronization only; no geometry, mass, inertia, limits, poses or collision-filter implementation changed",
  "source_hashes": {
    "meshes/L_link1.STL": "a2143af42a81d432d40276f6564825e4bb7bb2bf90f00e8ccacd549703d136d9",
    "meshes/R_link1.STL": "72a8d0c3eae84305322b5e5152b4cb9fd83a7b8ee8b4bfe489db311034ee3af1",
    "meshes/base_link.STL": "6c0450b99bef4e81af3126e8a90a27519551c1a1c5e958071e1ecc257bfb06cc",
    "robot.urdf": "f8928e61b213189bffc27e6c478ccefad84c40978f0200ab973f39e14648087a",
    "source.urdf": "835352595029eccc1dee951d4192f6dca4b00674d54e77290f79778cdb399431"
  },
  "status": "pending_material_assembly_review",
  "targeted_source_section_scan": {
    "counts_are_not_volume_or_probability": true,
    "points_per_pose_per_side": 14616,
    "rows": [
      {
        "q_L": 0,
        "q_R": 0,
        "side": "L",
        "slice_candidate_count": 2928,
        "verified_both_inside_count": 12,
        "verified_count": 12
      },
      {
        "q_L": 0,
        "q_R": 0,
        "side": "R",
        "slice_candidate_count": 540,
        "verified_both_inside_count": 12,
        "verified_count": 12
      },
      {
        "q_L": 0.41526541073209217,
        "q_R": -0.41526541073209217,
        "side": "L",
        "slice_candidate_count": 2928,
        "verified_both_inside_count": 12,
        "verified_count": 12
      },
      {
        "q_L": 0.41526541073209217,
        "q_R": -0.41526541073209217,
        "side": "R",
        "slice_candidate_count": 120,
        "verified_both_inside_count": 12,
        "verified_count": 12
      }
    ],
    "scope": "finite targeted source-section sample grid; not collision approval or exhaustive domain"
  },
  "training_permitted": false,
  "upstream_reports_sha256": {
    "docs/V40_HIP_COMPARISON.md": "cc59f590eb81fec291efac707daad5c891f91662e3cfe4197e67be8e589cc26f",
    "reports/hip_comparison/reliability.json": "4e5918333a4413b424933fa02e3b31f580435cec42a4d3c1593c34f5a08509fa",
    "reports/hip_comparison/sections_comparison.json": "4307fb1dea89b819865c6bbad2c40a71fc305282a69b2215c83227c5e7ea638e",
    "reports/hip_comparison/source_comparison.json": "bea38d0be21b94b1f7efbcf12b0006e5be7e7e8bac1d700123bb1ef79941e2ac",
    "reports/hip_comparison/verification.json": "0cc05e6b3015b746d259ae3942424981d4e5fcb77e84f22ea08646d57ec64b7f"
  },
  "user_decision": "repair_CAD_and_body_assignment_first_no_V4_training"
}
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vec(text):
    return np.array([float(x) for x in text.split()], dtype=float)


def fmt(values):
    return " ".join(format(float(x), ".17g") for x in values)


def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def origin(element):
    node = element.find("origin")
    attrs = {} if node is None else node.attrib
    return vec(attrs.get("xyz", "0 0 0")), rotation(vec(attrs.get("rpy", "0 0 0")))


def transform(element):
    pos, rot = origin(element)
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rot, pos
    return result


def axis_rotation(axis, angle):
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def topology(root):
    links = {e.get("name"): e for e in root.findall("link")}
    joints = {e.get("name"): e for e in root.findall("joint")}
    if len(links) != len(root.findall("link")) or len(joints) != len(root.findall("joint")):
        raise ValueError("Duplicate link/joint name")
    children = [j.find("child").get("link") for j in joints.values()]
    roots = set(links) - set(children)
    if len(set(children)) != len(children) or len(roots) != 1:
        raise ValueError("Expected one open tree")
    for joint in joints.values():
        if any(joint.find(role).get("link") not in links for role in ("parent", "child")):
            raise ValueError("Missing joint endpoint")
    return links, joints, roots.pop()


def forward(root, positions=None):
    links, joints, base = topology(root)
    frames, positions = {base: np.eye(4)}, positions or {}
    pending = list(joints.values())
    while pending:
        before = len(pending)
        for joint in pending[:]:
            parent, child = (joint.find(role).get("link") for role in ("parent", "child"))
            if parent not in frames:
                continue
            motion = np.eye(4)
            if joint.get("type") not in ("revolute", "continuous", "fixed"):
                raise ValueError("Unsupported joint type")
            if joint.get("type") != "fixed":
                axis = vec(joint.find("axis").get("xyz"))
                if not np.isclose(np.linalg.norm(axis), 1):
                    raise ValueError("Nonunit joint axis")
                motion[:3, :3] = axis_rotation(axis, positions.get(joint.get("name"), 0.))
            if child in frames:
                raise ValueError("Joint cycle")
            frames[child] = frames[parent] @ transform(joint) @ motion
            pending.remove(joint)
        if len(pending) == before:
            raise ValueError("Disconnected or cyclic tree")
    if set(frames) != set(links):
        raise ValueError("Disconnected links")
    return frames


def inertia(link):
    node = link.find("inertial")
    attrs = node.find("inertia").attrib
    tensor = np.array([[float(attrs["ixx"]), float(attrs["ixy"]), float(attrs["ixz"])],
                       [float(attrs["ixy"]), float(attrs["iyy"]), float(attrs["iyz"])],
                       [float(attrs["ixz"]), float(attrs["iyz"]), float(attrs["izz"])]])
    pos, rot = origin(node)
    mass = float(node.find("mass").get("value"))
    eig = np.linalg.eigvalsh(tensor)
    if not np.isfinite(tensor).all() or not np.isfinite(mass) or mass <= 0 or eig[0] <= 0 or eig[-1] > eig[:2].sum() + 1e-10:
        raise ValueError(f"Nonphysical mass/inertia for {link.get('name')}")
    return mass, pos, rot @ tensor @ rot.T


def validate_source(source, root):
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Source is not the byte-exact confirmed own-v40-knee35-80 URDF")
    links, joints, base = topology(root)
    if set(joints) != set(ACTION_ORDER) or len(links) != 7 or base != "base_link":
        raise ValueError("Expected own_v40 seven-link/six-joint topology; retain R_jonit2")
    for name, joint in joints.items():
        expected = "revolute" if name in KNEE_LIMITS else "continuous"
        if joint.get("type") != expected:
            raise ValueError(f"Wrong joint type: {name}")
        if name in KNEE_LIMITS:
            limits = [float(joint.find("limit").get(k)) for k in ("lower", "upper")]
            if not np.allclose(limits, KNEE_LIMITS[name], atol=1e-12, rtol=0):
                raise ValueError(f"Unconfirmed knee limits: {name}")
            if not limits[0] <= NOMINAL_JOINT_POS[name] <= limits[1]:
                raise ValueError("Nominal candidate outside hard knee limit")
    mass = sum(inertia(link)[0] for link in links.values())
    if not np.isclose(mass, 12.752, atol=1e-12, rtol=0):
        raise ValueError("Mass must remain exactly 12.752 kg")
    forward(root)
    mesh_uris = {m.get("filename") for m in root.iter("mesh")}
    if mesh_uris != set(SOURCE_MESH_SHA256):
        raise ValueError("Unexpected source mesh set")
    for uri in mesh_uris:
        if digest(portable_mesh_path(source, uri)) != SOURCE_MESH_SHA256[uri]:
            raise ValueError(f"Source STL differs from the confirmed snapshot: {uri}")


def canonicalize(root):
    """Premultiply ONLY root geometry/inertia origins and root joint origins.

    Local inertial coefficients stay byte-for-byte numeric strings: rotating the
    inertial frame rotates the effective root tensor, without double rotation.
    Child link frames, joint axes and every non-root origin remain untouched.
    """
    result = copy.deepcopy(root)
    links, joints, base = topology(result)
    affected = list(links[base].findall("visual")) + list(links[base].findall("collision")) + list(links[base].findall("inertial"))
    affected += [j for j in joints.values() if j.find("parent").get("link") == base]
    for element in affected:
        node = element.find("origin")
        if node is None:
            node = ET.SubElement(element, "origin")
        pos = vec(node.get("xyz", "0 0 0"))
        rpy = vec(node.get("rpy", "0 0 0"))
        rpy[2] += math.pi / 2
        node.set("xyz", fmt(CONTROL_ROTATION @ pos))
        node.set("rpy", fmt(rpy))
    result.set("name", "own_v40")
    return result


def sample_positions(count=32):
    rng = np.random.default_rng(403580)
    result = [dict(NOMINAL_JOINT_POS)]
    for _ in range(count - 1):
        q = {name: float(rng.uniform(-4 * math.pi, 4 * math.pi)) for name in ACTION_ORDER}
        for name, limits in KNEE_LIMITS.items():
            q[name] = float(rng.uniform(*limits))
        result.append(q)
    return result


def validate_canonical(source_root, canonical_root):
    source_links, source_joints, base = topology(source_root)
    links, joints, _ = topology(canonical_root)
    frame_rotation = np.eye(4)
    frame_rotation[:3, :3] = CONTROL_ROTATION
    max_frame = max_axis = max_com = max_tensor = 0.
    samples = sample_positions()
    for q in samples:
        old, new = forward(source_root, q), forward(canonical_root, q)
        old_com, new_com = np.zeros(3), np.zeros(3)
        for name in links:
            if name != base:
                max_frame = max(max_frame, float(np.abs(new[name] - frame_rotation @ old[name]).max()))
            m0, c0, i0 = inertia(source_links[name])
            m1, c1, i1 = inertia(links[name])
            if m0 != m1:
                raise ValueError("Mass changed")
            p0, p1 = old[name][:3, :3] @ c0 + old[name][:3, 3], new[name][:3, :3] @ c1 + new[name][:3, 3]
            old_com += m0 * p0
            new_com += m1 * p1
            max_com = max(max_com, float(np.abs(p1 - CONTROL_ROTATION @ p0).max()))
            t0 = old[name][:3, :3] @ i0 @ old[name][:3, :3].T
            t1 = new[name][:3, :3] @ i1 @ new[name][:3, :3].T
            max_tensor = max(max_tensor, float(np.abs(t1 - CONTROL_ROTATION @ t0 @ CONTROL_ROTATION.T).max()))
            for kind in ("visual", "collision"):
                for e0, e1 in zip(source_links[name].findall(kind), links[name].findall(kind), strict=True):
                    err = np.abs(new[name] @ transform(e1) - frame_rotation @ old[name] @ transform(e0)).max()
                    max_frame = max(max_frame, float(err))
        max_com = max(max_com, float(np.abs(new_com - CONTROL_ROTATION @ old_com).max() / 12.752))
        for name, joint in joints.items():
            child = joint.find("child").get("link")
            a0 = old[child][:3, :3] @ vec(source_joints[name].find("axis").get("xyz"))
            a1 = new[child][:3, :3] @ vec(joint.find("axis").get("xyz"))
            max_axis = max(max_axis, float(np.abs(a1 - CONTROL_ROTATION @ a0).max()))
    if max(max_frame, max_axis, max_com, max_tensor) > 1e-10:
        raise ValueError("Canonical FK/COM/inertia covariance failed")
    _, root_com, root_tensor = inertia(links[base])
    return {"passed": True, "samples": len(samples), "rotation_control_from_export": CONTROL_ROTATION.tolist(),
            "max_transform_error": max_frame, "max_axis_error": max_axis,
            "max_com_error_m": max_com, "max_inertia_error_kg_m2": max_tensor,
            "root_com_control_m": root_com.tolist(), "root_inertia_control_kg_m2": root_tensor.tolist(),
            "total_mass_kg": sum(inertia(link)[0] for link in links.values()),
            "joint_order": list(ACTION_ORDER), "joint_lookup": "by exact name, never imported index",
            "modified_origins": ["base_link visual/collision/inertial", "L_joint1", "R_joint1"],
            "other_local_frames_unchanged": True, "mass_or_inertia_reestimated": False}


def portable_mesh_path(source, uri):
    if not uri or ":" in uri or "\\" in uri or Path(uri).is_absolute():
        raise ValueError(f"Expected portable relative mesh URI: {uri}")
    result = (Path(source).parent / uri).resolve()
    if not result.is_relative_to(Path(source).resolve().parent) or not result.is_file():
        raise ValueError(f"Mesh missing or escapes source root: {uri}")
    return result


def stl_triangles(path):
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError("Truncated binary STL")
    count = struct.unpack_from("<I", raw, 80)[0]
    if count == 0 or len(raw) != 84 + 50 * count:
        raise ValueError("Only nonempty binary STL supported")
    triangles = np.frombuffer(raw, dtype=STL_DTYPE, count=count, offset=84)["vertices"].astype(float)
    if not np.isfinite(triangles).all():
        raise ValueError("Nonfinite STL vertices")
    return triangles


def connected_mesh(triangles):
    """Exact vertex connectivity, no tolerance weld, repair, deletion or rescale."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices))).tocsr()
    count, labels = connected_components(graph, directed=False)
    return vertices, faces, labels, count


def surface_stats(vertices, faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    tri = vertices[faces]
    center = vertices.mean(axis=0)
    shifted = tri - center
    signed_volume = float(np.einsum("ij,ij->i", shifted[:, 0], np.cross(shifted[:, 1], shifted[:, 2])).sum() / 6.)
    return {"boundary_edges": int((counts == 1).sum()), "nonmanifold_edges": int((counts > 2).sum()),
            "degenerate_triangles": int((np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) == 0).sum()),
            "signed_surface_volume_m3_diagnostic_only": signed_volume}


def convex_obj(vertices, path):
    """Write outward-oriented Qhull faces; never joggle/shrink input geometry."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    used = np.sort(hull.vertices)
    remap = {int(old): new for new, old in enumerate(used)}
    faces = []
    for face, equation in zip(hull.simplices, hull.equations, strict=True):
        tri = vertices[face]
        if np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), equation[:3]) < 0:
            face = face[[0, 2, 1]]
        indices = [remap[int(x)] for x in face]
        first = indices.index(min(indices))
        faces.append(indices[first:] + indices[:first])
    lines = ["# Exact connected-component conservative convex envelope; metres."]
    lines += ["v " + fmt(v) for v in vertices[used]]
    lines += ["f " + " ".join(str(x + 1) for x in f) for f in sorted(faces)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    max_outside = 0.
    for start in range(0, len(vertices), 2048):
        planes = vertices[start:start + 2048] @ hull.equations[:, :3].T + hull.equations[:, 3]
        max_outside = max(max_outside, float(planes.max()))
    if max_outside > 1e-9:
        raise ValueError("Collision envelope does not cover all source vertices")
    return {"hull_vertex_count": len(used), "hull_triangle_count": len(faces), "hull_volume_m3": float(hull.volume),
            "max_source_vertex_outside_hull_m": max_outside}


def split_collisions(root, source, out):
    """Keep exact component envelopes; wheel mesh envelopes are historical only."""
    links, _, _ = topology(root)
    report = {"method": "exact component convex envelopes plus measured enclosing wheel cylinders",
              "weld_tolerance_m": 0., "geometry_removed": False, "geometry_shrunk": False,
              "mass_or_inertia_reestimated": False, "certified_concave_decomposition": False,
              "links": {}, "blockers": [], "limitations": []}
    (out / "collisions").mkdir()
    (out / "evidence/legacy_wheel_hulls").mkdir(parents=True, exist_ok=True)
    for name, link in links.items():
        original = link.findall("collision")
        if len(original) != 1 or original[0].find("geometry/mesh") is None:
            raise ValueError("Expected one source mesh collision per link")
        geometry = original[0]
        mesh = geometry.find("geometry/mesh")
        triangles = stl_triangles(portable_mesh_path(source, mesh.get("filename")))
        vertices, faces, labels, count = connected_mesh(triangles)
        pieces = []
        for component in range(count):
            selected = np.flatnonzero(labels == component)
            component_faces = faces[labels[faces[:, 0]] == component]
            if not np.all(labels[component_faces] == component):
                raise ValueError("Cross-component triangle")
            points = vertices[selected]
            directory = "evidence/legacy_wheel_hulls" if name in WHEEL_LINKS else "collisions"
            uri = f"{directory}/{name}_part_{component:03d}.obj"
            stats = convex_obj(points, out / uri)
            stats.update(surface_stats(vertices, component_faces))
            stats.update(component_id=component, mesh=uri, shape="mesh", source_vertices=len(points), source_triangles=len(component_faces),
                         bounds_m=[points.min(0).tolist(), points.max(0).tolist()])
            pieces.append(stats)
            collision = copy.deepcopy(geometry)
            collision.set("name", f"{name}_part_{component:03d}")
            collision.find("geometry/mesh").set("filename", uri)
            link.append(collision)
        link.remove(geometry)
        if sum(p["source_triangles"] for p in pieces) != len(triangles) or sum(p["source_vertices"] for p in pieces) != len(vertices):
            raise ValueError("A source component was lost")
        report["links"][name] = {"source_mesh": mesh.get("filename"), "source_sha256": digest(portable_mesh_path(source, mesh.get("filename"))),
                                  "source_triangles": len(triangles), "source_vertices": len(vertices), "component_count": count,
                                  "all_source_triangles_covered": True, "pieces": pieces}
        if any(p["boundary_edges"] or p["nonmanifold_edges"] for p in pieces):
            report["limitations"].append(f"{name}: source has open/non-manifold surfaces; winding witnesses are diagnostic, not a watertight solid certificate")
    report["limitations"].append("Chassis/shank envelopes still fill cavities. Static named-adjacency review is not full-domain or physical collision certification.")
    return report


def adjacent_pairs(root):
    _, joints, _ = topology(root)
    return [{"body1": joints[name].find("parent").get("link"), "body2": joints[name].find("child").get("link"),
             "joint": name, "reason": "Adjacent: nominal proxy-fill review only"} for name in ACTION_ORDER]


def replace_wheels_with_cylinders(root, source, report):
    """Enclose every source vertex about the original wheel joint axis (+/-Z)."""
    links, joints, _ = topology(root)
    frames = forward(root, NOMINAL_JOINT_POS)
    wheels = {}
    for name in WHEEL_LINKS:
        link = links[name]
        visual = link.find("visual")
        mesh = visual.find("geometry/mesh")
        vertices = stl_triangles(portable_mesh_path(source, mesh.get("filename"))).reshape(-1, 3) * vec(mesh.get("scale", "1 1 1"))
        axis = vec(next(j for j in joints.values() if j.find("child").get("link") == name).find("axis").get("xyz"))
        if not np.allclose(np.abs(axis), [0, 0, 1], atol=1e-12):
            raise ValueError("Wheel cylinder requires the confirmed local +/-Z joint axis")
        radii = np.linalg.norm(vertices[:, :2], axis=1)
        radius = float(radii.max())
        lo, hi = float(vertices[:, 2].min()), float(vertices[:, 2].max())
        center = np.array([0., 0., (lo + hi) / 2])
        outer = vertices[radii > radius - 1e-7, :2]
        fit = np.linalg.lstsq(np.c_[2 * outer, np.ones(len(outer))], (outer * outer).sum(1), rcond=None)[0]
        if np.linalg.norm(fit[:2]) > 1e-7:
            raise ValueError("Source outer wheel circle is not centered on the named joint axis")
        pos, rot = origin(visual)
        local_center = rot @ center + pos
        collision = ET.Element("collision", name=f"{name}_cylinder")
        ET.SubElement(collision, "origin", xyz=fmt(local_center), rpy=visual.find("origin").get("rpy", "0 0 0"))
        ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder", radius=fmt([radius]), length=fmt([hi - lo]))
        for old in link.findall("collision"):
            link.remove(old)
        link.append(collision)
        frame = frames[name] @ transform(collision)
        world_center = frame[:3, 3] + [0, 0, NOMINAL_BASE_HEIGHT_M]
        world_axis = frame[:3, 2]
        min_z = float(world_center[2] - .5 * (hi - lo) * abs(world_axis[2]) - radius * math.sqrt(max(0., 1 - world_axis[2] ** 2)))
        entry = report["links"][name]
        entry["legacy_mesh_envelopes"] = entry["pieces"]
        piece = {"component_id": 0, "shape": "cylinder", "radius_m": radius, "length_m": hi - lo,
                 "center_link_m": local_center.tolist(), "axis_link": (rot @ np.array([0., 0., 1.])).tolist(),
                 "measured_axial_bounds_m": [lo, hi], "outer_ring_circle_fit_center_xy_m": fit[:2].tolist(),
                 "center_rule": "joint-axis transverse center, validated against STL outer-ring fit; midpoint of measured axial extrema",
                 "source_vertices": entry["source_vertices"], "source_triangles": entry["source_triangles"],
                 "max_source_radial_outside_m": max(0., float(radii.max() - radius)),
                 "max_source_axial_outside_m": max(0., float(np.abs(vertices[:, 2] - center[2]).max() - .5 * (hi - lo))),
                 "nominal_collision_min_z_m": min_z, "collision_geometry_shrunk": False,
                 "physx_mesh_cooking_required_for_collision": False}
        entry["pieces"] = [piece]
        wheels[name] = piece
    oversized = [{"link": name, "mesh": p["mesh"], "hull_vertices": p["hull_vertex_count"]}
                 for name, entry in report["links"].items() for p in entry["pieces"]
                 if p["shape"] == "mesh" and p["hull_vertex_count"] > CONVEX_VERTEX_SCREEN]
    report["physx_portability_screen"] = {"vertex_budget": CONVEX_VERTEX_SCREEN, "over_budget_parts": oversized,
        "isaac_physx_cooking_verified": False, "active_wheel_collision_type": "cylinder",
        "note": "Wheel mesh envelopes exist only as archived evidence. Active URDF/MJCF wheel collisions are primitives; remaining convex cooking must be checked on the server."}
    if oversized:
        report["blockers"].append("An active convex collision exceeds the offline vertex portability screen")
    report["wheel_cylinders"] = wheels
    return wheels


def generalized_winding(triangles, point):
    """Signed solid-angle diagnostic; open source surfaces are NOT silently repaired."""
    v = triangles - point
    lengths = np.linalg.norm(v, axis=2)
    numerator = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
    denominator = (lengths.prod(1) + np.einsum("ij,ij->i", v[:, 0], v[:, 1]) * lengths[:, 2]
                   + np.einsum("ij,ij->i", v[:, 1], v[:, 2]) * lengths[:, 0]
                   + np.einsum("ij,ij->i", v[:, 2], v[:, 0]) * lengths[:, 1])
    return float((2 * np.arctan2(numerator, denominator)).sum() / (4 * math.pi))


def source_surface_witness(triangles, point):
    """Nearest triangle distance and three ray parities for an ambiguous witness."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = b - a, c - a
    normal = np.cross(e1, e2)
    n2 = np.einsum("ij,ij->i", normal, normal)
    delta = point - a
    signed = np.einsum("ij,ij->i", delta, normal)
    projected = delta - normal * np.divide(signed, n2, out=np.zeros_like(signed), where=n2 > 0)[:, None]
    d00, d01, d11 = (np.einsum("ij,ij->i", u, v) for u, v in ((e1, e1), (e1, e2), (e2, e2)))
    d20, d21 = np.einsum("ij,ij->i", projected, e1), np.einsum("ij,ij->i", projected, e2)
    den = d00 * d11 - d01 * d01
    u = np.divide(d11 * d20 - d01 * d21, den, out=np.full_like(den, -1.), where=den > 0)
    v = np.divide(d00 * d21 - d01 * d20, den, out=np.full_like(den, -1.), where=den > 0)
    in_triangle = (u >= 0) & (v >= 0) & (u + v <= 1) & (n2 > 0)
    distance2 = np.full(len(triangles), np.inf)
    distance2[in_triangle] = signed[in_triangle] ** 2 / n2[in_triangle]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        length2 = np.einsum("ij,ij->i", edge, edge)
        t = np.clip(np.divide(np.einsum("ij,ij->i", point - start, edge), length2, out=np.zeros_like(length2), where=length2 > 0), 0, 1)
        off = point - (start + t[:, None] * edge)
        distance2 = np.minimum(distance2, np.einsum("ij,ij->i", off, off))
    parities = []
    for direction in ([.137, .731, 1.], [.897, -.221, .43], [-.563, .653, .751]):
        direction = np.array(direction) / np.linalg.norm(direction)
        h = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, h)
        inv = np.divide(1., det, out=np.zeros_like(det), where=np.abs(det) > 1e-12)
        s = point - a
        u = inv * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = inv * (q @ direction)
        t = inv * np.einsum("ij,ij->i", e2, q)
        hits = np.sort(t[(np.abs(det) > 1e-12) & (u > 1e-9) & (v > 1e-9) & (u + v < 1 - 1e-9) & (t > 1e-9)])
        count = int(1 + np.count_nonzero(np.diff(hits) > 1e-7)) if len(hits) else 0
        parities.append(count % 2)
    return {"nearest_source_triangle_distance_m": float(math.sqrt(distance2.min())),
            "three_ray_inside_parities": parities, "note": "Independent numerical witness only; source open edges remain disclosed"}


def review_adjacent_overlaps(root, directory):
    """Audit full convex intersections plus deterministic source-material witnesses.

    This is deliberately stronger than measuring the few contact manifold points.
    Cylinder intersection bounds use a circumscribed 128-sided polytope, never an
    inscribed/shrunken one. Source winding samples are evidence, not an exhaustive
    boolean mesh-intersection proof. An unexplained jointly occupied/ambiguous
    witness blocks approval of even this limited static adjacency policy.
    """
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    from scipy.optimize import linprog
    links, _, _ = topology(root)
    frames = forward(root, NOMINAL_JOINT_POS)
    convex, surfaces = {}, {}
    for name, link in links.items():
        visual = link.find("visual")
        mesh = visual.find("geometry/mesh")
        frame = frames[name] @ transform(visual)
        tri = stl_triangles(Path(directory) / mesh.get("filename")) * vec(mesh.get("scale", "1 1 1"))
        surfaces[name] = tri @ frame[:3, :3].T + frame[:3, 3]
        convex[name] = []
        for geometry in link.findall("collision"):
            frame = frames[name] @ transform(geometry)
            mesh, cylinder = geometry.find("geometry/mesh"), geometry.find("geometry/cylinder")
            if mesh is not None:
                points = np.array([[float(x) for x in line.split()[1:]] for line in (Path(directory) / mesh.get("filename")).read_text().splitlines() if line.startswith("v ")])
                points *= vec(mesh.get("scale", "1 1 1"))
                approximation = "exact convex OBJ halfspaces"
            else:
                radius, half = float(cylinder.get("radius")), .5 * float(cylinder.get("length"))
                angles = (np.arange(128) + .5) * 2 * math.pi / 128
                ring = radius / math.cos(math.pi / 128) * np.c_[np.cos(angles), np.sin(angles)]
                points = np.vstack([np.c_[ring, np.full(128, z)] for z in (-half, half)])
                approximation = "circumscribed 128-sided cylinder bound; <=19 micrometres radial overbound"
            world = points @ frame[:3, :3].T + frame[:3, 3]
            convex[name].append({"name": geometry.get("name"), "points": world,
                                 "equations": ConvexHull(world).equations, "approximation": approximation})
    result = {"scope": "nominal raw q at h=0.32; explicit direct-adjacency filter justification only",
              "physics_steps_executed": 0, "source_winding_is_watertight_certificate": False,
              "full_working_domain_checked": False, "pairs": [], "blockers": [],
              "witness_rule": "absolute winding <0.05 indicates empty source material; both nonempty/ambiguous blocks; deterministic interior/extreme witnesses, not exhaustive solid proof"}
    for pair in adjacent_pairs(root):
        parent, child = pair["body1"], pair["body2"]
        frame = frames[child]
        rows, all_local, unexplained = [], [], []
        for a in convex[parent]:
            for b in convex[child]:
                av, bv = a["points"], b["points"]
                if np.any(np.minimum(av.max(0), bv.max(0)) - np.maximum(av.min(0), bv.min(0)) < 1e-10):
                    continue
                eq = np.vstack([a["equations"], b["equations"]])
                fit = linprog([0, 0, 0, -1], A_ub=np.c_[eq[:, :3], np.ones(len(eq))], b_ub=-eq[:, 3],
                              bounds=[(None, None)] * 3 + [(0, None)], method="highs")
                if fit.status == 2:  # Provably infeasible halfspace intersection.
                    continue
                if not fit.success:
                    raise ValueError(f"Intersection LP failed for {a['name']}/{b['name']}: {fit.message}")
                if fit.x[3] < 1e-9:  # Sub-nanometre inradius cannot justify a macroscopic overlap.
                    continue
                iv = HalfspaceIntersection(eq, fit.x[:3]).intersections
                local = (iv - frame[:3, 3]) @ frame[:3, :3]
                all_local.extend(local)
                center = iv.mean(0)
                extremes = [int(np.linalg.norm(local[:, :2], axis=1).argmax())]
                extremes += [int(func(iv[:, axis])) for axis in range(3) for func in (np.argmin, np.argmax)]
                witnesses = [center, fit.x[:3]] + [.8 * iv[i] + .2 * center for i in sorted(set(extremes))]
                rng = np.random.default_rng(403580)
                candidates = rng.uniform(iv.min(0), iv.max(0), size=(4096, 3))
                inside = candidates[np.max(candidates @ eq[:, :3].T + eq[:, 3], axis=1) < -1e-8]
                witnesses.extend(inside[:max(0, 40 - len(witnesses))])
                samples = []
                for point in witnesses:
                    wp, wc = generalized_winding(surfaces[parent], point), generalized_winding(surfaces[child], point)
                    empty = min(abs(wp), abs(wc)) < .05
                    sample = {"position_base_control_m": point.tolist(), "parent_winding": wp, "child_winding": wc,
                              "at_least_one_source_material_empty": bool(empty)}
                    samples.append(sample)
                    if not empty:
                        local_point = (point - frame[:3, 3]) @ frame[:3, :3]
                        sample["position_child_joint_frame_m"] = local_point.tolist()
                        sample["radial_distance_about_joint_axis_m"] = float(np.linalg.norm(local_point[:2]))
                        sample["parent_surface_check"] = source_surface_witness(surfaces[parent], point)
                        sample["child_surface_check"] = source_surface_witness(surfaces[child], point)
                        unexplained.append(sample)
                rows.append({"parent_collision": a["name"], "child_collision": b["name"],
                             "intersection_vertex_count": len(iv), "radial_max_about_joint_axis_m": float(np.linalg.norm(local[:, :2], axis=1).max()),
                             "bounds_in_child_joint_frame_m": [local.min(0).tolist(), local.max(0).tolist()],
                             "intersection_bound_method": [a["approximation"], b["approximation"]],
                             "source_material_witnesses": samples})
        if not rows:
            result["blockers"].append(f"{pair['joint']}: no intersecting proxy volume was available for the proposed filter review")
        if unexplained:
            result["blockers"].append(f"{pair['joint']}: {len(unexplained)} source-material witnesses are jointly occupied or ambiguous; adjacency alone cannot justify ignoring them")
        points = np.array(all_local)
        radius = float(np.linalg.norm(points[:, :2], axis=1).max()) if len(points) else 0.
        span = .21 if pair["joint"] in ("L_joint1", "R_joint1") else (.25 if pair["joint"] in KNEE_LIMITS else .06)
        result["pairs"].append({**pair, "review_supported": bool(rows) and not unexplained,
            "interpretation": "Proxy-fill evidence supports this named direct-adjacency convention at the nominal pose only; NOT a claim that all overlap lies in a small axle hole",
            "full_intersection_radial_max_m": radius, "link_or_wheel_reference_span_m": span,
            "overlap_extends_beyond_small_joint_neighborhood": radius > .35 * span,
            "bounds_in_child_joint_frame_m": [points.min(0).tolist(), points.max(0).tolist()] if len(points) else None,
            "witness_count": sum(len(row["source_material_witnesses"]) for row in rows),
            "unexplained_witness_count": len(unexplained), "convex_pair_intersections": rows})
    result["review_supported"] = not result["blockers"]
    apply_hip_material_review_hold(result)
    return result


def hip_material_review_hold_metadata():
    """Source-bound, embedded CAD-first hold; no upstream file lookup or override."""
    evidence = json.loads(HIP_MATERIAL_REVIEW_HOLD_JSON)
    if evidence["source_hashes"]["source.urdf"] != SOURCE_SHA256:
        raise ValueError("Hip review hold belongs to another source URDF")
    for name in ("base_link", "L_link1", "R_link1"):
        uri = f"meshes/{name}.STL"
        if evidence["source_hashes"][uri] != SOURCE_MESH_SHA256[uri]:
            raise ValueError(f"Hip review hold belongs to another source mesh: {uri}")
    if evidence["pending_joints"] != ["L_joint1", "R_joint1"] or evidence["training_permitted"] is not False:
        raise ValueError("Both source hips must remain pending CAD/material/assembly review")
    return {"review_id": evidence["review_id"], "status": evidence["status"],
            "user_decision": evidence["user_decision"], "training_permitted": False,
            "automatic_hip_filter_approval_permitted": False, "pending_joints": evidence["pending_joints"],
            "evidence_file": HIP_MATERIAL_REVIEW_HOLD_PATH,
            "evidence_sha256": hashlib.sha256(HIP_MATERIAL_REVIEW_HOLD_JSON.encode("utf-8")).hexdigest(),
            "source_hashes": evidence["source_hashes"], "upstream_reports_sha256": evidence["upstream_reports_sha256"],
            "right_actual_nominal_overlap": evidence["right_actual_nominal_overlap"],
            "required_next_step": evidence["required_next_step"]}


def apply_hip_material_review_hold(report):
    """Overlay review state only; preserve every old sampled point/contact value.

    A clean finite sample (including the old right-hip sample) cannot erase the
    independent source-material findings or the user's CAD-first decision.
    Idempotent so a failed numerical review is also safely held by build().
    """
    hold = hip_material_review_hold_metadata()
    report.setdefault("limited_sampling_blockers", list(report.get("blockers", [])))
    for pair in report.get("pairs", []):
        pair.setdefault("limited_sample_no_unexplained_witnesses", bool(pair.get("witness_count")) and pair.get("unexplained_witness_count") == 0)
        if pair["joint"] in hold["pending_joints"]:
            pair.setdefault("limited_sample_interpretation", pair.get("interpretation"))
            pair["review_supported"] = False
            pair["material_assembly_review_status"] = "pending"
            pair["independent_review_evidence_file"] = HIP_MATERIAL_REVIEW_HOLD_PATH
            pair["interpretation"] = "Independent source comparison found material-interior hip witnesses; CAD/body assignment and assembly review is pending on BOTH sides. A zero count in the old finite sample is not filter approval."
    for joint in hold["pending_joints"]:
        reason = f"{joint}: source-material/assembly and CAD body-assignment review remains pending; user chose CAD correction first and no V4 training (see {HIP_MATERIAL_REVIEW_HOLD_PATH})"
        if reason not in report.setdefault("blockers", []):
            report["blockers"].append(reason)
    report["material_assembly_review_hold"] = hold
    report["review_supported"] = False
    return report


def pose_attributes(element):
    pos, rot = origin(element)
    return {"pos": fmt(pos), "xyaxes": fmt(np.concatenate([rot[:, 0], rot[:, 1]]))}


def make_mjcf(root, *, filter_adjacent=True):
    """Independent inspection model with explicit URDF inertias and named joints."""
    links, joints, base = topology(root)
    doc = ET.Element("mujoco", model="own_v40_canonical_static_inspection")
    ET.SubElement(doc, "compiler", angle="radian", meshdir=".", autolimits="true", inertiafromgeom="false")
    option = ET.SubElement(doc, "option", timestep="0.001", integrator="implicitfast", gravity="0 0 -9.81")
    if filter_adjacent:
        ET.SubElement(option, "flag", filterparent="disable")  # No implicit filtering; six explicit pairs below.
    assets = ET.SubElement(doc, "asset")
    world = ET.SubElement(doc, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.1", contype="1", conaffinity="1")
    mesh_names, emitted_joints = {}, []
    for mesh in root.iter("mesh"):
        key = (mesh.get("filename"), mesh.get("scale", "1 1 1"))
        if key not in mesh_names:
            name = f"mesh_{len(mesh_names):03d}"
            ET.SubElement(assets, "mesh", name=name, file=key[0], scale=key[1])
            mesh_names[key] = name

    def add_link(name, parent, joint=None):
        attrs = {"name": name}
        attrs.update({"pos": f"0 0 {NOMINAL_BASE_HEIGHT_M}"} if joint is None else pose_attributes(joint))
        body = ET.SubElement(parent, "body", attrs)
        if joint is None:
            ET.SubElement(body, "freejoint", name="floating_base")
        else:
            props = {"name": joint.get("name"), "type": "hinge", "axis": joint.find("axis").get("xyz"), "limited": "false"}
            if joint.get("type") == "revolute":
                lim = joint.find("limit")
                props.update(limited="true", range=f"{lim.get('lower')} {lim.get('upper')}")
            ET.SubElement(body, "joint", props)
            emitted_joints.append(joint.get("name"))
        link = links[name]
        mass, pos, tensor = inertia(link)
        ET.SubElement(body, "inertial", mass=str(mass), pos=fmt(pos), fullinertia=fmt([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]]))
        for kind in ("visual", "collision"):
            for i, geometry in enumerate(link.findall(kind)):
                mesh = geometry.find("geometry/mesh")
                attrs = pose_attributes(geometry)
                attrs.update(name=f"{name}_{kind}_{i:03d}")
                if mesh is not None:
                    attrs.update(type="mesh", mesh=mesh_names[(mesh.get("filename"), mesh.get("scale", "1 1 1"))])
                else:
                    cylinder = geometry.find("geometry/cylinder")
                    if cylinder is None:
                        raise ValueError("Only source meshes and measured wheel cylinders are supported")
                    attrs.update(type="cylinder", size=fmt([float(cylinder.get("radius")), .5 * float(cylinder.get("length"))]))
                if kind == "visual":
                    color = geometry.find("material/color")
                    attrs.update(group="2", contype="0", conaffinity="0", rgba=color.get("rgba") if color is not None else "0.8 0.8 0.8 1")
                else:
                    attrs.update(group="3", contype="1", conaffinity="1", margin="0", gap="0", rgba="0.8 0.3 0.1 0.25")
                ET.SubElement(body, "geom", attrs)
        for child_joint in joints.values():
            if child_joint.find("parent").get("link") == name:
                add_link(child_joint.find("child").get("link"), body, child_joint)
    add_link(base, world)
    if filter_adjacent:
        contact = ET.SubElement(doc, "contact")
        for pair in adjacent_pairs(root):
            ET.SubElement(contact, "exclude", name=pair["joint"] + "_adjacent", body1=pair["body1"], body2=pair["body2"])
    actuator = ET.SubElement(doc, "actuator")
    for name in ACTION_ORDER:
        ET.SubElement(actuator, "motor", name=name, joint=name, gear="1", ctrllimited="true", ctrlrange="-1 1")
    keyframe = ET.SubElement(doc, "keyframe")
    qpos = [0, 0, NOMINAL_BASE_HEIGHT_M, 1, 0, 0, 0] + [NOMINAL_JOINT_POS[name] for name in emitted_joints]
    ET.SubElement(keyframe, "key", name="nominal_candidate", qpos=fmt(qpos))
    return doc


def contact_records(model, data):
    result = []
    for contact in data.contact:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        result.append({"geom1": model.geom(g1).name, "geom2": model.geom(g2).name,
                       "body1": model.body(int(model.geom_bodyid[g1])).name,
                       "body2": model.body(int(model.geom_bodyid[g2])).name,
                       "distance_m": float(contact.dist), "position_m": contact.pos.tolist(),
                       "normal": contact.frame[:3].tolist()})
    return sorted(result, key=lambda x: (x["geom1"], x["geom2"], x["distance_m"]))


def static_inspection(root, mjcf_path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    links, joints, base = topology(root)
    if (model.nq, model.nv, model.nu) != (13, 12, 6):
        raise ValueError("Unexpected inspection DOFs")
    if not np.isclose(model.body_mass.sum(), 12.752, atol=1e-12, rtol=0):
        raise ValueError("Compiled mass changed")
    addresses = {}
    max_frame = max_axis = max_tensor = max_com = 0.
    for name, joint in joints.items():
        jid = model.joint(name).id
        addresses[name] = {"joint_id": int(jid), "qpos_address": int(model.jnt_qposadr[jid]), "dof_address": int(model.jnt_dofadr[jid]), "actuator_id": int(model.actuator(name).id)}
        if bool(model.jnt_limited[jid]) != (name in KNEE_LIMITS):
            raise ValueError(f"Compiled joint type changed: {name}")
        if name in KNEE_LIMITS and not np.allclose(model.jnt_range[jid], KNEE_LIMITS[name], atol=1e-12, rtol=0):
            raise ValueError(f"Compiled knee limit changed: {name}")
    for name, link in links.items():
        bid = model.body(name).id
        mass, com, tensor = inertia(link)
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, model.body_iquat[bid])
        compiled_tensor = rot.reshape(3, 3) @ np.diag(model.body_inertia[bid]) @ rot.reshape(3, 3).T
        max_tensor = max(max_tensor, float(np.abs(compiled_tensor - tensor).max()))
        max_com = max(max_com, float(np.abs(model.body_ipos[bid] - com).max()))
        if abs(float(model.body_mass[bid]) - mass) > 1e-12:
            raise ValueError(f"Compiled mass changed: {name}")
    # mj_forward only: no stepping, gravity changes, render/viewer or training.
    for q in sample_positions():
        mujoco.mj_resetData(model, data)
        for name, value in q.items():
            data.qpos[addresses[name]["qpos_address"]] = value
        mujoco.mj_forward(model, data)
        frames = forward(root, q)
        for name, frame in frames.items():
            bid = model.body(name).id
            max_frame = max(max_frame, float(np.abs(data.xpos[bid] - data.xpos[model.body(base).id] - frame[:3, 3]).max()),
                            float(np.abs(data.xmat[bid].reshape(3, 3) - frame[:3, :3]).max()))
        for name, joint in joints.items():
            axis = frames[joint.find("child").get("link")][:3, :3] @ vec(joint.find("axis").get("xyz"))
            max_axis = max(max_axis, float(np.abs(data.xaxis[model.joint(name).id] - axis).max()))
        if data.time != 0 or not np.isfinite(data.qpos).all() or np.any(data.warning.number):
            raise ValueError("Unexpected time advance/nonfinite state/MuJoCo warning")
    if max(max_frame, max_axis, max_com, max_tensor) > 1e-8:
        raise ValueError("MJCF FK/COM/inertia differs from canonical URDF")
    mujoco.mj_resetDataKeyframe(model, data, model.key("nominal_candidate").id)
    mujoco.mj_forward(model, data)
    nominal = contact_records(model, data)
    xml = ET.parse(mjcf_path).getroot()
    declared = [{"body1": e.get("body1"), "body2": e.get("body2")} for e in xml.findall("contact/exclude")]
    expected = [{"body1": p["body1"], "body2": p["body2"]} for p in adjacent_pairs(root)]
    if declared and declared != expected:
        raise ValueError("Unexpected collision exclusion beyond the six named direct-adjacency pairs")
    minimum_nonadjacent = {"distance_m": 1., "geom1": None, "geom2": None}
    adjacent = {frozenset((p["body1"], p["body2"])) for p in adjacent_pairs(root)}
    geoms = [i for i in range(model.ngeom) if model.geom_contype[i] and model.geom_bodyid[i] != 0]
    for i, ga in enumerate(geoms):
        for gb in geoms[i + 1:]:
            ba, bb = model.body(int(model.geom_bodyid[ga])).name, model.body(int(model.geom_bodyid[gb])).name
            if ba == bb or frozenset((ba, bb)) in adjacent:
                continue
            distance = float(mujoco.mj_geomDistance(model, data, ga, gb, 1., np.empty(6)))
            if distance < minimum_nonadjacent["distance_m"]:
                minimum_nonadjacent = {"distance_m": distance, "geom1": model.geom(ga).name, "geom2": model.geom(gb).name}
    # Recompile a diagnostic XML, removing exactly the declared excludes and also
    # disabling MuJoCo's implicit parent filter. No unsafe mutation of body masks.
    diagnostic_xml = copy.deepcopy(xml)
    contact = diagnostic_xml.find("contact")
    if contact is not None:
        for exclude in contact.findall("exclude"):
            contact.remove(exclude)
    option = diagnostic_xml.find("option")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("filterparent", "disable")
    mesh_bytes = {e.get("file"): portable_mesh_path(mjcf_path, e.get("file")).read_bytes() for e in diagnostic_xml.findall("asset/mesh")}
    diagnostic_model = mujoco.MjModel.from_xml_string(ET.tostring(diagnostic_xml, encoding="unicode"), assets=mesh_bytes)
    diagnostic_data = mujoco.MjData(diagnostic_model)
    mujoco.mj_resetDataKeyframe(diagnostic_model, diagnostic_data, diagnostic_model.key("nominal_candidate").id)
    mujoco.mj_forward(diagnostic_model, diagnostic_data)
    all_pairs = contact_records(diagnostic_model, diagnostic_data)
    report = {"available": True, "mujoco_version": mujoco.__version__, "passed_frame_mass_inertia": True,
            "physics_steps_executed": 0, "simulation_time_s": float(data.time), "nq": model.nq, "nv": model.nv, "nu": model.nu,
            "total_mass_kg": float(model.body_mass.sum()), "fk_samples": len(sample_positions()),
            "max_transform_error": max_frame, "max_axis_error": max_axis, "max_com_error_m": max_com, "max_inertia_error_kg_m2": max_tensor,
            "joint_addresses_resolved_by_name": addresses, "nominal_base_height_m": NOMINAL_BASE_HEIGHT_M,
            "nominal_joint_pos": dict(NOMINAL_JOINT_POS), "contacts_active_policy": nominal,
            "contacts_including_joint_adjacent_bodies": all_pairs, "warnings": data.warning.number.tolist(),
            "custom_contact_exclusions": declared, "minimum_nonadjacent_geom_distance": minimum_nonadjacent,
            "implicit_parent_filter_disabled": bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)),
            "explicit_six_pair_filter_verified_in_mujoco": declared == expected,
            "isaac_filter_verified": False, "collision_masks": "All robot collision geoms 1/1; only visuals 0/0",
            "scope": "Static nominal pose with declared filters; sampled FK only. No working-domain, dynamics, Isaac or hardware validation."}
    if not declared:
        report["contacts_default_parent_filter"] = nominal
    return report


def geometry_clearances(root, source):
    frames = forward(root, NOMINAL_JOINT_POS)
    result = {}
    for link in root.findall("link"):
        name = link.get("name")
        geometry = link.find("visual")
        mesh = geometry.find("geometry/mesh")
        vertices = stl_triangles(portable_mesh_path(source, mesh.get("filename"))).reshape(-1, 3) * vec(mesh.get("scale", "1 1 1"))
        frame = frames[name] @ transform(geometry)
        points = vertices @ frame[:3, :3].T + frame[:3, 3] + [0, 0, NOMINAL_BASE_HEIGHT_M]
        result[name] = {"visual_mesh_min_z_m": float(points[:, 2].min()), "visual_mesh_max_z_m": float(points[:, 2].max())}
    return result


def base_visual_bounds(root, source):
    links, _, base = topology(root)
    points = []
    for visual in links[base].findall("visual"):
        mesh = visual.find("geometry/mesh")
        vertices = stl_triangles(portable_mesh_path(source, mesh.get("filename"))).reshape(-1, 3) * vec(mesh.get("scale", "1 1 1"))
        pos, rot = origin(visual)
        points.append(vertices @ rot.T + pos)
    vertices = np.vstack(points)
    return [vertices.min(0).tolist(), vertices.max(0).tolist()]


def write_xml(path, root):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build(source, out, *, evidence_paths=(), run_mujoco=True):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Output already exists; use a fresh directory (no source/old assets are overwritten)")
    source_root = ET.parse(source).getroot()
    validate_source(source, source_root)
    root = canonicalize(source_root)
    frame_report = validate_canonical(source_root, root)
    clearances = geometry_clearances(root, source)
    meshes = {}
    for mesh in source_root.iter("mesh"):
        uri = mesh.get("filename")
        meshes[uri] = portable_mesh_path(source, uri)
    # Validate everything which can fail without output writes first.
    out.mkdir(parents=True)
    shutil.copyfile(source, out / "source.urdf")
    for uri, path in meshes.items():
        target = out / uri
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    evidence_hashes = {}
    for path in map(Path, evidence_paths):
        if path.name == Path(HIP_MATERIAL_REVIEW_HOLD_PATH).name and digest(path) != hashlib.sha256(HIP_MATERIAL_REVIEW_HOLD_JSON.encode("utf-8")).hexdigest():
            raise ValueError("Optional evidence cannot override the embedded bilateral CAD-first hold")
        target = out / "evidence" / path.name
        if target.exists():
            raise ValueError("Duplicate evidence basename")
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        evidence_hashes[target.relative_to(out).as_posix()] = digest(target)
    hold_metadata = hip_material_review_hold_metadata()
    (out / "evidence").mkdir(exist_ok=True)
    (out / HIP_MATERIAL_REVIEW_HOLD_PATH).write_text(HIP_MATERIAL_REVIEW_HOLD_JSON, encoding="utf-8")
    evidence_hashes[HIP_MATERIAL_REVIEW_HOLD_PATH] = hold_metadata["evidence_sha256"]
    # Preserve both baseline geometries; neither is the active training artifact.
    baseline_root = copy.deepcopy(root)
    write_xml(out / "inspection_whole_hulls.xml", make_mjcf(baseline_root, filter_adjacent=False))
    collision_report = split_collisions(root, source, out)
    legacy_root = copy.deepcopy(root)
    write_xml(out / "inspection_component_mesh_baseline.xml", make_mjcf(legacy_root, filter_adjacent=False))
    wheels = replace_wheels_with_cylinders(root, source, collision_report)
    write_xml(out / "robot.urdf", root)
    write_xml(out / "inspection.xml", make_mjcf(root))
    filters = adjacent_pairs(root)
    srdf = ET.Element("robot", name="own_v40")
    for pair in filters:
        ET.SubElement(srdf, "disable_collisions", link1=pair["body1"], link2=pair["body2"], reason="Adjacent")
    write_xml(out / "collision_filters.srdf", srdf)
    try:
        adjacency_review = review_adjacent_overlaps(root, out)
    except Exception as error:
        adjacency_review = {"review_supported": False, "pairs": [], "blockers": [f"Adjacency geometry review failed: {type(error).__name__}: {error}"]}
    apply_hip_material_review_hold(adjacency_review)  # Also covers numerical-review failure; never auto-upgrade either hip.
    reviewed_pairs = {p["joint"]: p for p in adjacency_review["pairs"]}
    for pair in filters:
        pair["geometry_review_supported"] = reviewed_pairs.get(pair["joint"], {}).get("review_supported", False)
        pair["policy_status"] = "nominal_proxy_filter_supported" if pair["geometry_review_supported"] else "proposed_pending_material_review"
    static_reports = {}
    for label, model_root, filename in (("components", root, "inspection.xml"),
                                       ("legacy_component_mesh", legacy_root, "inspection_component_mesh_baseline.xml"),
                                       ("whole_link_baseline", baseline_root, "inspection_whole_hulls.xml")):
        report = {"available": False, "physics_steps_executed": 0, "reason": "MuJoCo verification explicitly skipped"}
        if run_mujoco:
            try:
                report = static_inspection(model_root, out / filename)
            except Exception as error:
                report = {"available": False, "physics_steps_executed": 0, "reason": f"{type(error).__name__}: {error}"}
        static_reports[label] = report
    static_report = static_reports["components"]
    blockers = list(collision_report["blockers"]) + list(adjacency_review["blockers"])
    if not static_report["available"]:
        blockers.append("Static MuJoCo validation unavailable/failed: " + static_report["reason"])
    elif not static_report["explicit_six_pair_filter_verified_in_mujoco"] or not static_report["implicit_parent_filter_disabled"]:
        blockers.append("The exact six adjacent exclusions were not verified independently of implicit filtering")
    contacts = static_report.get("contacts_active_policy", [])
    penetrating_self = [c for c in contacts if c["body1"] != "world" and c["body2"] != "world" and c["distance_m"] < -1e-6]
    if penetrating_self:
        blockers.append(f"Nominal h=0.32 retains {len(penetrating_self)} non-adjacent penetrating self contacts; do not raise height or relax knees")
    if static_report.get("minimum_nonadjacent_geom_distance", {}).get("distance_m", 1.) < -1e-6:
        blockers.append("Direct unmasked convex distance query found a non-adjacent penetration")
    all_self = [c for c in static_report.get("contacts_including_joint_adjacent_bodies", [])
                if c["body1"] != "world" and c["body2"] != "world" and c["distance_m"] < -1e-6]
    pair_minimum = {}
    for contact in all_self:
        key = contact["body1"] + "/" + contact["body2"]
        pair_minimum[key] = min(pair_minimum.get(key, 0.), contact["distance_m"])
    ground_blockers = []
    for name, values in clearances.items():
        low = values["visual_mesh_min_z_m"]
        if name in WHEEL_LINKS:
            if abs(low) > 1e-5:
                ground_blockers.append(f"{name}: nominal source visual wheel not tangent to floor: {low} m")
            if not -WHEEL_PROXY_GROUND_TOLERANCE_M <= wheels[name]["nominal_collision_min_z_m"] <= 1e-5:
                ground_blockers.append(f"{name}: measured enclosing cylinder fails the explicit wheel-only 0.1mm floor proxy tolerance")
        elif low < -1e-6:
            ground_blockers.append(f"{name}: nonwheel source geometry below floor: {low} m")
    for contact in contacts:
        if "world" not in (contact["body1"], contact["body2"]):
            continue
        other = contact["body2"] if contact["body1"] == "world" else contact["body1"]
        tolerance = WHEEL_PROXY_GROUND_TOLERANCE_M if other in WHEEL_LINKS else 1e-6
        if contact["distance_m"] < -tolerance or other not in WHEEL_LINKS:
            ground_blockers.append(f"Unexpected/excessive floor contact: {other}, {contact['distance_m']} m")
    blockers.extend(ground_blockers)
    collision_report["nominal_visual_clearances"] = clearances
    collision_report["adjacency_review_report"] = "adjacency_review.json"
    collision_report["material_assembly_review_hold"] = hold_metadata
    collision_report["ground_validation"] = {"passed": not ground_blockers,
        "wheel_proxy_penetration_tolerance_m": WHEEL_PROXY_GROUND_TOLERANCE_M, "nonwheel_penetration_tolerance_m": 1e-6,
        "explanation": "Measured enclosing cylinders extend 27.8/60.3 micrometres below tangent source tire meshes at the unchanged 0.32m pose. This wheel-only tolerance is explicit, not a mesh shrink/height change.",
        "blockers": ground_blockers}
    collision_report["blockers"] = blockers
    write_json(out / "frame_validation.json", frame_report)
    write_json(out / "collision_report.json", collision_report)
    write_json(out / "adjacency_review.json", adjacency_review)
    write_json(out / "static_validation.json", static_reports)
    write_json(out / "provenance.json", {"source_urdf_snapshot": "source.urdf", "source_sha256": digest(source),
               "source_asset_label": "own-v40-knee35-80", "source_visual_meshes_sha256": {uri: digest(path) for uri, path in meshes.items()},
               "evidence_sha256": evidence_hashes, "material_assembly_review_hold": hold_metadata, "original_visual_bytes_preserved": True,
               "original_source_modified": False, "generator": "tools/prepare_v40_assets.py", "generator_sha256": digest(__file__),
               "dependencies": {"numpy": np.__version__, "scipy": __import__("scipy").__version__},
               "source_snapshot_rebuild": "python -B tools/prepare_v40_assets.py --source assets/urdf_v40/source.urdf --out NEW_DIRECTORY"})
    legacy_contacts = static_reports["legacy_component_mesh"].get("contacts_including_joint_adjacent_bodies", [])
    legacy_count = len([c for c in legacy_contacts if c["body1"] != "world" and c["body2"] != "world" and c["distance_m"] < -1e-6])
    passed = not blockers
    manifest = {
        "schema_version": 1, "robot_id": "own_v40", "control_frame": "Xforward_Yleft_Zup", "total_mass_kg": 12.752,
        "knee_inner_limits_deg": [35, 80], "urdf": "robot.urdf", "mjcf": "inspection.xml", "source_sha256": digest(source),
        "files_sha256": {path.relative_to(out).as_posix(): digest(path) for path in sorted(out.rglob("*")) if path.is_file()},
        "base_visual_bounds_m": base_visual_bounds(root, source), "isaac_cooking_tested": False,
        "server_adjacency_filter_verified": False, "adjacent_collision_filter_pairs": filters,
        "collision_validation": {"passed": passed, "scope": "offline_static_initial_pose_with_six_named_adjacent_filters; not full working domain, Isaac cooking, dynamics or hardware certification",
            "details": {"blockers": blockers, "candidate_status": "offline_static_initial_pose_checked" if passed else "blocked_candidate",
                "nominal_base_height_m": NOMINAL_BASE_HEIGHT_M, "nominal_joint_pos": dict(NOMINAL_JOINT_POS),
                "physics_steps_executed": 0, "mujoco_available": static_report["available"],
                "penetrating_nonadjacent_contacts": penetrating_self,
                "all_pair_penetrating_self_contact_count": len(all_self), "all_pair_minimum_distance_by_body_pair_m": pair_minimum,
                "contacts_active_policy": contacts, "legacy_component_mesh_penetrating_contact_count": legacy_count,
                "preserved_revision1_report": "evidence/revision1_static_validation.json" if (out / "evidence/revision1_static_validation.json").is_file() else None,
                "physx_portability_screen": collision_report["physx_portability_screen"],
                "ground_validation": collision_report["ground_validation"], "adjacency_review_supported": adjacency_review["review_supported"],
                "material_assembly_review_hold": hold_metadata,
                "adjacency_review_summary": [{k: p[k] for k in ("joint", "review_supported", "limited_sample_no_unexplained_witnesses", "full_intersection_radial_max_m", "witness_count", "unexplained_witness_count")} for p in adjacency_review["pairs"]],
                "unexplained_material_overlap_witnesses": [{"joint": p["joint"], "body1": p["body1"], "body2": p["body2"], **w} for p in adjacency_review["pairs"] for c in p["convex_pair_intersections"] for w in c["source_material_witnesses"] if not w["at_least_one_source_material_empty"]],
                "component_counts": {k: v["component_count"] for k, v in collision_report["links"].items()},
                "geometry_report": "collision_report.json", "static_report": "static_validation.json", "frame_report": "frame_validation.json",
                "adjacency_review_report": "adjacency_review.json", "adjacent_filter_srdf": "collision_filters.srdf",
                "no_global_self_collision_disable": True, "no_mass_inertia_reestimate": True,
                "requires_server_checks": ["First resolve both hip CAD/body-assignment and material/assembly reviews; no V4 training is authorized before a new controlled review", "Apply exactly the six named body-pair filters and verify they took effect; URDF alone does not carry SRDF exclusions", "Verify primitive cylinder and convex cooking", "Verify non-adjacent self collisions and floor contacts remain active"],
                "limitations": collision_report["limitations"],
                "action_order": list(ACTION_ORDER), "resolve_import_indices_by_name": True}},
        "hardware_deployment_ready": False, "nominal_joint_pos": dict(NOMINAL_JOINT_POS), "nominal_base_height_m": NOMINAL_BASE_HEIGHT_M,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


RESEARCH_SCOPE_ID = "own-v40-equivalent-serial-internal-contact-v1"
RESEARCH_COLLISION_SCOPE = "equivalent_serial_research_with_explicit_internal_joint_exclusions"
RESEARCH_APPROVAL_FILE = "research_model_approval.json"
RESEARCH_MANIFEST_FILE = "research_manifest.json"


def make_research_manifest(raw_assets, approval_path):
    """Validate explicit approval and immutable offline evidence; write nothing.

    This does not alter the raw material-review decision. Supplied mass/inertia
    and rigid-link grouping are serial-equivalent research priors, not hardware
    claims. Approval is a user-decision record bound by SHA, not a user signature.
    Only saved, hash-verified statistics are emitted, making cross-NumPy rebuilds
    deterministic; independent structural/analytic cross-checks are boolean gates.
    """
    if approval_path is None:
        raise ValueError("An explicit research approval input is required")
    directory, approval_path = Path(raw_assets).resolve(), Path(approval_path).resolve()
    raw_path = directory / "manifest.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    approval_bytes = approval_path.read_bytes()
    approval = json.loads(approval_bytes)
    raw_sha = digest(raw_path)
    if approval.get("schema_version") != 1 or approval.get("robot_id") != "own_v40" or approval.get("scope_id") != RESEARCH_SCOPE_ID:
        raise ValueError("Research approval identity/scope mismatch")
    decision = approval.get("approval_source", {})
    if (approval.get("approved") is not True or decision.get("kind") != "explicit_user_decision"
            or decision.get("question_id") != "v40-equivalent-research-scope"
            or decision.get("selected_option") != "同意，先用串联等效研究模型训练"):
        raise ValueError("Missing the explicit user decision for equivalent serial research")
    if approval.get("raw_manifest_file") != "manifest.json" or approval.get("raw_manifest_sha256") != raw_sha:
        raise ValueError("Approval is not bound to this immutable raw manifest")
    for key in ("nonadjacent_collisions_unchanged", "global_self_collision_enabled", "supersedes_prior_pause_for_this_research_scope_only", "raw_material_review_stays_unresolved"):
        if approval.get(key) is not True:
            raise ValueError(f"Research approval must explicitly preserve {key}")
    for key in ("source_geometry_modified", "hardware_deployment_approved"):
        if approval.get(key) is not False:
            raise ValueError(f"Research approval cannot authorize {key}")
    if approval.get("knee_inner_limits_deg") != [35, 80] or approval.get("dynamics_prior", {}).get("model") != "equivalent_serial_open_chain_research":
        raise ValueError("Research knee limits or dynamics-prior declaration mismatch")
    raw_pairs = raw.get("adjacent_collision_filter_pairs", [])
    expected_approved_pairs = [{**p, "raw_policy_status": p["policy_status"], "research_exclusion_approved": True,
                               "policy_status": "user_approved_joint_internal_contact_exclusion"} for p in raw_pairs]
    if len(raw_pairs) != 6 or approval.get("adjacent_collision_filter_pairs") != expected_approved_pairs:
        raise ValueError("Approval must contain exactly the six original named pairs and preserve geometry-review flags")
    if raw.get("robot_id") != "own_v40" or raw.get("control_frame") != "Xforward_Yleft_Zup" or raw.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("Unexpected raw model/frame/source")
    if raw.get("collision_validation", {}).get("passed") is not False or raw.get("hardware_deployment_ready") is not False:
        raise ValueError("This variant must preserve the original failed raw material review")
    inventory = raw.get("files_sha256", {})
    if not inventory or any(p in inventory for p in ("manifest.json", RESEARCH_APPROVAL_FILE, RESEARCH_MANIFEST_FILE)):
        raise ValueError("Raw hash inventory is empty or creates a manifest/approval cycle")
    for relative, expected in inventory.items():
        if digest(portable_mesh_path(raw_path, relative)) != expected:
            raise ValueError(f"Raw evidence/geometry hash mismatch: {relative}")
    for relative in (raw.get("urdf"), raw.get("mjcf"), "source.urdf", "static_validation.json", "collision_report.json", "frame_validation.json", "collision_filters.srdf"):
        if relative not in inventory:
            raise ValueError(f"Raw manifest does not bind required evidence: {relative}")
    source = ET.parse(directory / "source.urdf").getroot()
    validate_source(directory / "source.urdf", source)
    root = ET.parse(directory / raw["urdf"]).getroot()
    links, joints, _ = topology(root)
    named_pairs = adjacent_pairs(root)
    if [{k: p[k] for k in ("joint", "body1", "body2")} for p in raw_pairs] != [{k: p[k] for k in ("joint", "body1", "body2")} for p in named_pairs]:
        raise ValueError("Approved pairs differ from the actual direct-joint topology")
    xml = ET.parse(directory / raw["mjcf"]).getroot()
    static = json.loads((directory / "static_validation.json").read_text())["components"]
    collision = json.loads((directory / "collision_report.json").read_text())
    frame_report = json.loads((directory / "frame_validation.json").read_text())
    finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    checks = {}
    mass = sum(inertia(link)[0] for link in links.values())
    checks["mass_and_frame_preserved"] = abs(mass - 12.752) < 1e-12 and raw["total_mass_kg"] == 12.752 and frame_report.get("passed") is True
    checks["nominal_pose_unchanged"] = raw.get("nominal_joint_pos") == NOMINAL_JOINT_POS and raw.get("nominal_base_height_m") == NOMINAL_BASE_HEIGHT_M
    checks["hard_knee_limits_and_continuous_hips_wheels"] = set(joints) == set(ACTION_ORDER) and raw.get("knee_inner_limits_deg") == [35, 80]
    for name, joint in joints.items():
        if name in KNEE_LIMITS:
            limits = [float(joint.find("limit").get(k)) for k in ("lower", "upper")]
            valid = (joint.get("type") == "revolute" and np.allclose(limits, KNEE_LIMITS[name], atol=1e-12, rtol=0)
                     and limits[0] <= NOMINAL_JOINT_POS[name] <= limits[1])
        else:
            valid = joint.get("type") == "continuous"
        checks["hard_knee_limits_and_continuous_hips_wheels"] &= bool(valid)
    expected_excludes = [(p["body1"], p["body2"]) for p in named_pairs]
    xml_excludes = [(e.get("body1"), e.get("body2")) for e in xml.findall("contact/exclude")]
    srdf = ET.parse(directory / "collision_filters.srdf").getroot()
    srdf_excludes = [(e.get("link1"), e.get("link2")) for e in srdf.findall("disable_collisions")]
    flag = xml.find("option/flag")
    checks["exact_six_named_internal_pairs_only"] = (xml_excludes == expected_excludes == srdf_excludes
        and not xml.findall("contact/pair") and flag is not None and flag.get("filterparent") == "disable"
        and static.get("custom_contact_exclusions") == [{"body1": a, "body2": b} for a, b in expected_excludes]
        and static.get("explicit_six_pair_filter_verified_in_mujoco") is True and static.get("implicit_parent_filter_disabled") is True)
    checks["global_self_collision_and_floor_enabled"] = all(
        (g.get("contype"), g.get("conaffinity")) == (("0", "0") if "_visual_" in g.get("name", "") else ("1", "1"))
        for g in xml.iter("geom"))
    minimum = static.get("minimum_nonadjacent_geom_distance", {}).get("distance_m")
    checks["nonadjacent_positive_clearance"] = finite(minimum) and minimum > 0
    checks["static_frame_mass_evidence_valid"] = (static.get("available") is True and static.get("passed_frame_mass_inertia") is True
        and (static.get("nq"), static.get("nv"), static.get("nu")) == (13, 12, 6)
        and static.get("physics_steps_executed") == 0 and static.get("simulation_time_s") == 0
        and isinstance(static.get("warnings"), list) and not any(static["warnings"])
        and all(finite(static.get(k)) and 0 <= static[k] < 1e-8 for k in ("max_transform_error", "max_axis_error", "max_com_error_m", "max_inertia_error_kg_m2")))
    frames = forward(root, NOMINAL_JOINT_POS)
    recorded_wheel_min_z = {}
    checks["measured_wheel_cylinders_and_floor_tolerance"] = True
    for name in WHEEL_LINKS:
        geometries = links[name].findall("collision")
        cylinder = geometries[0].find("geometry/cylinder") if len(geometries) == 1 else None
        valid = cylinder is not None
        recorded = collision.get("wheel_cylinders", {}).get(name, {}).get("nominal_collision_min_z_m")
        if cylinder is not None:
            radius, half = float(cylinder.get("radius")), .5 * float(cylinder.get("length"))
            pose = frames[name] @ transform(geometries[0])
            axis_z = float(pose[2, 2])
            analytic = float(pose[2, 3] + NOMINAL_BASE_HEIGHT_M - half * abs(axis_z) - radius * math.sqrt(max(0., 1 - axis_z ** 2)))
            valid &= finite(radius) and finite(half) and radius > 0 and half > 0 and finite(recorded) and abs(analytic - recorded) < 1e-10
            valid &= finite(recorded) and -WHEEL_PROXY_GROUND_TOLERANCE_M <= recorded <= 1e-5
        checks["measured_wheel_cylinders_and_floor_tolerance"] &= bool(valid)
        recorded_wheel_min_z[name] = recorded
    clearances = geometry_clearances(root, directory / "source.urdf")
    checks["source_wheel_tangency_and_nonwheel_ground_clearance"] = all(
        abs(c["visual_mesh_min_z_m"]) < 1e-7 if name in WHEEL_LINKS else c["visual_mesh_min_z_m"] > 0
        for name, c in clearances.items())
    active = static.get("contacts_active_policy")
    valid_contacts = isinstance(active, list)
    for contact in active or []:
        if "world" in (contact["body1"], contact["body2"]):
            other = contact["body2"] if contact["body1"] == "world" else contact["body1"]
            valid_contacts &= other in WHEEL_LINKS and finite(contact["distance_m"]) and contact["distance_m"] >= -WHEEL_PROXY_GROUND_TOLERANCE_M
        else:
            valid_contacts &= finite(contact["distance_m"]) and contact["distance_m"] > 0
    checks["active_contacts_legal_for_research_scope"] = bool(valid_contacts)
    blockers = [f"Research static check failed: {name}" for name, passed in checks.items() if not passed]
    physical_keys = ("schema_version", "robot_id", "control_frame", "total_mass_kg", "knee_inner_limits_deg", "urdf", "mjcf",
                     "source_sha256", "nominal_joint_pos", "nominal_base_height_m", "base_visual_bounds_m", "hardware_deployment_ready",
                     "isaac_cooking_tested", "server_adjacency_filter_verified")
    result = {key: copy.deepcopy(raw[key]) for key in physical_keys}
    approval_sha = hashlib.sha256(approval_bytes).hexdigest()
    result.update({
        "files_sha256": {**copy.deepcopy(inventory), "manifest.json": raw_sha, RESEARCH_APPROVAL_FILE: approval_sha},
        "adjacent_collision_filter_pairs": copy.deepcopy(expected_approved_pairs),
        "nonadjacent_collisions_unchanged": True, "global_self_collision_enabled": True, "source_geometry_modified": False,
        "research_model": {"scope_id": RESEARCH_SCOPE_ID, "model": "equivalent_serial_open_chain_research",
            "approval_file": RESEARCH_APPROVAL_FILE, "approval_sha256": approval_sha,
            "raw_manifest_file": "manifest.json", "raw_manifest_sha256": raw_sha,
            "approval_source": copy.deepcopy(decision), "dynamics_prior": copy.deepcopy(approval["dynamics_prior"]),
            "research_exclusion_approved": True, "hardware_deployment_approved": False,
            "supersedes_prior_pause_for_this_research_scope_only": True, "raw_material_review_stays_unresolved": True,
            "generator": "tools/prepare_v40_assets.py", "generator_sha256": digest(__file__)},
        "raw_material_review": {"raw_collision_validation_passed": raw["collision_validation"]["passed"],
            "hip_geometry_review_supported": {p["joint"]: p["geometry_review_supported"] for p in raw_pairs if p["joint"] in ("L_joint1", "R_joint1")},
            "source_material_overlap_repaired": False, "raw_review_report": "adjacency_review.json",
            "raw_material_assembly_review_status": raw["collision_validation"]["details"]["material_assembly_review_hold"]["status"],
            "prior_training_hold_preserved_as_history": True},
        "collision_validation": {"passed": not blockers, "scope": RESEARCH_COLLISION_SCOPE,
            "details": {"checks": checks, "blockers": blockers, "approval_validated_for_scope": True,
                "raw_failure_preserved": True, "source_material_overlap_repaired": False,
                "nonadjacent_minimum_clearance_m": minimum, "wheel_collision_min_z_m": recorded_wheel_min_z,
                "wheel_proxy_penetration_tolerance_m": WHEEL_PROXY_GROUND_TOLERANCE_M,
                "nominal_base_height_m": NOMINAL_BASE_HEIGHT_M, "nominal_joint_pos": dict(NOMINAL_JOINT_POS),
                "static_report": "static_validation.json", "collision_report": "collision_report.json", "frame_report": "frame_validation.json",
                "physics_steps_executed": 0, "new_simulation_run_for_variant": False,
                "validation_basis": "Hash-bound existing CPU static evidence with source/URDF/MJCF/SRDF and analytic clearance cross-checks; emitted statistics are from immutable reports",
                "working_domain_validated": False, "dynamic_balance_validated": False, "isaac_cooking_tested": False,
                "server_adjacency_filter_verified": False, "hardware_deployment_approved": False,
                "limitations": copy.deepcopy(approval.get("limitations", [])),
                "requires_server_checks": ["Apply exactly the six named approved internal-joint exclusions; preserve all other self and terrain collisions", "Verify importer cylinder/convex cooking and the actual exclusion pairs before research execution"],
                "action_order": list(ACTION_ORDER), "resolve_import_indices_by_name": True}}})
    return result


def emit_research_variant(raw_assets, approval_path, *, out=None):
    """Add only two research records beside an immutable, hash-bound raw bundle.

    To reproduce in a new directory, first copy the immutable raw snapshot there.
    This emitter never rebuilds, retargets approval, or modifies original files.
    Existing byte-identical research records are an idempotent no-op; mismatches
    are refused rather than overwritten. No approval argument means no variant.
    """
    directory = Path(raw_assets).resolve()
    if out is not None and Path(out).resolve() != directory:
        raise ValueError("Research records must sit beside their raw snapshot; copy the immutable bundle before choosing a new output directory")
    result = make_research_manifest(directory, approval_path)
    approval_bytes = Path(approval_path).read_bytes()
    if hashlib.sha256(approval_bytes).hexdigest() != result["research_model"]["approval_sha256"]:
        raise ValueError("Approval changed during validation")
    manifest_bytes = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    payloads = [(directory / RESEARCH_APPROVAL_FILE, approval_bytes), (directory / RESEARCH_MANIFEST_FILE, manifest_bytes)]
    for path, payload in payloads:
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different research record: {path.name}")
    for path, payload in payloads:
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(payload)
    if digest(directory / "manifest.json") != result["research_model"]["raw_manifest_sha256"]:
        raise ValueError("Raw manifest changed during research emission")
    return result


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo / "assets/urdf_v40/source.urdf")
    parser.add_argument("--out", type=Path, required=True, help="Fresh directory for raw builds; the immutable --raw-assets directory for research records")
    parser.add_argument("--raw-assets", type=Path, help="Existing immutable raw bundle; used only with explicit --research-approval")
    parser.add_argument("--research-approval", type=Path, help="Explicit user-decision JSON bound to the exact raw manifest SHA; never inferred from sampling")
    parser.add_argument("--evidence", type=Path, action="append", default=[], help="Optional evidence file to snapshot; repeatable")
    parser.add_argument("--skip-mujoco", action="store_true", help="Generate without simulator import; forces blocked collision gate")
    args = parser.parse_args(argv)
    if args.raw_assets is not None or args.research_approval is not None:
        if args.raw_assets is None or args.research_approval is None:
            parser.error("--raw-assets and --research-approval must be supplied together")
        if args.evidence or args.skip_mujoco:
            parser.error("Research emission validates immutable saved evidence; do not mix raw-build options")
        manifest = emit_research_variant(args.raw_assets, args.research_approval, out=args.out)
    else:
        manifest = build(args.source, args.out, evidence_paths=args.evidence, run_mujoco=not args.skip_mujoco)
    print(json.dumps({"asset_written": str(args.out), "manifest_file": RESEARCH_MANIFEST_FILE if args.research_approval is not None else "manifest.json",
                      "collision_passed": manifest["collision_validation"]["passed"], "scope": manifest["collision_validation"]["scope"],
                      "hardware_deployment_ready": False, "blockers": manifest["collision_validation"]["details"]["blockers"]}, indent=2))
    return 0 if manifest["collision_validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
