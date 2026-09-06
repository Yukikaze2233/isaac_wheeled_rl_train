#!/usr/bin/env python3
"""Generate the training URDF for urdf_V3.1 (serial-leg wheeled biped).

Source: /home/yukikaze/Downloads/urdf_V3.1/urdf/urdf_V3.1.urdf
Output: assets/urdf_v31/urdf_V3.1_rl.urdf (+ meshes, defaults.json)

Frame conversion (audited, not assumed):
The SolidWorks export frame is right-handed but the robot faces -y there
(legs span +-x, wheel axles along +-x -> wheels roll along +-y). ROS/Isaac
REP-103 wants forward=+x, left=+y, up=+z. A global Rz(+90deg) maps
forward(-y)->+x, left(+x)->+y. This rotation is BAKED into every link
transform (origin xyz/rpy, joint axis, inertia tensor, COM offset) instead
of a wrapper fixed joint, because MuJoCo's URDF importer welds fixed joints
and silently drops their transforms. Baking keeps Isaac Lab and MuJoCo
bit-identical in frame semantics.

Other fixes:
- hips/knees (joint1/joint2) exported as 'continuous' but are revolute:
  set 'revolute' with limits around the default standing pose.
- wheels (joint3) stay 'continuous'.
- name typo R_jonit2 -> R_joint2 in the asset; firmware name R_jonit2 is
  preserved via defaults.json mapping (do not rename on deploy side).
- absolute mesh paths under this asset folder.
- placeholder joint damping/friction (TODO: real2sim identification).

Default standing pose: IK so both wheel axles sit directly under the hips,
base origin at 0.48 m (wheels touch ground: axle z = -(0.48 - 0.06)).
"""
import xml.etree.ElementTree as ET
import json, os, shutil, math
import numpy as np

SRC = "/home/yukikaze/Downloads/urdf_V3.1/urdf/urdf_V3.1.urdf"
MESH_SRC = "/home/yukikaze/Downloads/urdf_V3.1/meshes"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "urdf_v31")

BASE_HEIGHT = 0.48
WHEEL_RADIUS = 0.06

DEFAULT_POSE = {
    "L_joint1": +0.5796, "L_joint2": +0.8638, "L_joint3": 0.0,
    "R_joint1": -0.1828, "R_joint2": -0.6986, "R_joint3": 0.0,
}
STRAIGHT = {"L_hip": 1.2335, "L_knee": 2.0658, "R_hip": -0.8355, "R_knee": -1.9018}

# global frame rotation Rz(+90deg)
RZ = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

def parse(s):
    return np.array([float(x) for x in s.split()])

def rpy_to_R(r):
    rx, ry, rz = r
    cx, sx = np.cos(rx), np.sin(rx); cy, sy = np.cos(ry), np.sin(ry); cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]) @
            np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @
            np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]))

def R_to_rpy(R):
    # fixed-axis XYZ: R = Rz*Ry*Rx
    ry = math.atan2(-R[2,0], math.hypot(R[2,1], R[2,2]))
    if abs(abs(ry) - math.pi/2) < 1e-9:
        rz = 0.0
        rx = math.atan2(R[0,1], R[1,1]) if ry > 0 else math.atan2(-R[0,1], R[1,1])
    else:
        rx = math.atan2(R[2,1], R[2,2])
        rz = math.atan2(R[1,0], R[0,0])
    return np.array([rx, ry, rz])

ET.register_namespace("", "")
tree = ET.parse(SRC)
root = tree.getroot()
root.set("name", "urdf_V3.1_rl")

for link in root.findall("link"):
    for tag in ("inertial", "visual", "collision"):
        el = link.find(tag)
        if el is None: continue
        o = el.find("origin")
        if o is None: continue
        xyz = RZ @ parse(o.attrib.get("xyz", "0 0 0"))
        rpy = R_to_rpy(RZ @ rpy_to_R(parse(o.attrib.get("rpy", "0 0 0")) @ RZ.T))
        o.set("xyz", f"{xyz[0]:.8g} {xyz[1]:.8g} {xyz[2]:.8g}")
        o.set("rpy", f"{rpy[0]:.8g} {rpy[1]:.8g} {rpy[2]:.8g}")
    # inertia tensor: I' = Rz I Rz^T
    inn = link.find("inertial/inertia")
    if inn is not None:
        I = np.array([[parse(inn.attrib["ixx"])[0], parse(inn.attrib["ixy"])[0], parse(inn.attrib["ixz"])[0]],
                      [parse(inn.attrib["ixy"])[0], parse(inn.attrib["iyy"])[0], parse(inn.attrib["iyz"])[0]],
                      [parse(inn.attrib["ixz"])[0], parse(inn.attrib["iyz"])[0], parse(inn.attrib["izz"])[0]]])
        I2 = RZ @ I @ RZ.T
        for k, idx in (("ixx",(0,0)), ("iyy",(1,1)), ("izz",(2,2)),
                       ("ixy",(0,1)), ("ixz",(0,2)), ("iyz",(1,2))):
            inn.set(k, f"{I2[idx]:.6g}")

for j in root.findall("joint"):
    o = j.find("origin")
    if o is not None:
        xyz = RZ @ parse(o.attrib.get("xyz", "0 0 0"))
        rpy = R_to_rpy(RZ @ rpy_to_R(parse(o.attrib.get("rpy", "0 0 0")) @ RZ.T))
        o.set("xyz", f"{xyz[0]:.8g} {xyz[1]:.8g} {xyz[2]:.8g}")
        o.set("rpy", f"{rpy[0]:.8g} {rpy[1]:.8g} {rpy[2]:.8g}")
    ax = j.find("axis")
    if ax is not None:
        a = RZ @ parse(ax.attrib["xyz"])
        ax.set("xyz", f"{a[0]:.8g} {a[1]:.8g} {a[2]:.8g}")

def set_limit(j, lo, hi, effort="100", vel="15", damping="0.5", friction="0.1"):
    j.set("type", "revolute")
    lim = j.find("limit")
    if lim is None: lim = ET.SubElement(j, "limit")
    lim.set("lower", f"{lo:.4f}"); lim.set("upper", f"{hi:.4f}")
    lim.set("effort", effort); lim.set("velocity", vel)
    dyn = j.find("dynamics")
    if dyn is None: dyn = ET.SubElement(j, "dynamics")
    dyn.set("damping", damping); dyn.set("friction", friction)

def set_continuous(j, effort="100", vel="60", damping="0.02", friction="0.02"):
    j.set("type", "continuous")
    lim = j.find("limit")
    if lim is not None:
        lim.set("effort", effort); lim.set("velocity", vel)
    dyn = j.find("dynamics")
    if dyn is None: dyn = ET.SubElement(j, "dynamics")
    dyn.set("damping", damping); dyn.set("friction", friction)

joints = {j.attrib["name"]: j for j in root.findall("joint")}
set_limit(joints["L_joint1"], DEFAULT_POSE["L_joint1"]-0.8, DEFAULT_POSE["L_joint1"]+0.8)
set_limit(joints["L_joint2"], DEFAULT_POSE["L_joint2"]-1.6, STRAIGHT["L_knee"]+0.1)
set_limit(joints["R_joint1"], DEFAULT_POSE["R_joint1"]-0.8, DEFAULT_POSE["R_joint1"]+0.8)
set_limit(joints["R_jonit2"], STRAIGHT["R_knee"]-0.1, DEFAULT_POSE["R_joint2"]+1.3)
set_continuous(joints["L_joint3"])
set_continuous(joints["R_joint3"])
joints["R_jonit2"].set("name", "R_joint2")

os.makedirs(os.path.join(OUT_DIR, "meshes"), exist_ok=True)
for link in root.findall("link"):
    for geom in link.iter("mesh"):
        rel = os.path.basename(geom.attrib["filename"])
        geom.set("filename", os.path.join(OUT_DIR, "meshes", rel))
for name in ("base_link", "L_link1", "L_link2", "L_link3", "R_link1", "R_link2", "R_link3"):
    shutil.copy2(os.path.join(MESH_SRC, name + ".STL"), os.path.join(OUT_DIR, "meshes", name + ".STL"))

ET.indent(root, space="  ")
out_urdf = os.path.join(OUT_DIR, "urdf_V3.1_rl.urdf")
tree.write(out_urdf, xml_declaration=True, encoding="utf-8")

sidecar = {
    "source": SRC,
    "base_height_default": BASE_HEIGHT,
    "wheel_radius": WHEEL_RADIUS,
    "spawn_pos": [0.0, 0.0, BASE_HEIGHT],
    "default_joint_pos": DEFAULT_POSE,
    "joint_order_contract": {
        "legs": ["L_joint1", "L_joint2", "R_joint1", "R_joint2"],
        "wheels": ["L_joint3", "R_joint3"],
    },
    "firmware_name_map": {
        "R_joint2": "R_jonit2",
        "firmware_indices": {"L_joint1": 1, "L_joint2": 2, "L_joint3": 3,
                             "R_joint1": 4, "R_jonit2": 5, "R_joint3": 6},
    },
    "joint_limits": {
        "L_joint1": [DEFAULT_POSE["L_joint1"]-0.8, DEFAULT_POSE["L_joint1"]+0.8],
        "L_joint2": [DEFAULT_POSE["L_joint2"]-1.6, STRAIGHT["L_knee"]+0.1],
        "R_joint1": [DEFAULT_POSE["R_joint1"]-0.8, DEFAULT_POSE["R_joint1"]+0.8],
        "R_joint2": [STRAIGHT["R_knee"]-0.1, DEFAULT_POSE["R_joint2"]+1.3],
        "L_joint3": "continuous",
        "R_joint3": "continuous",
    },
    "frame_conversion": "global Rz(+90deg) baked into all transforms: export forward(-y)->+x, left(+x)->+y",
    "TODO_identification": [
        "real leg joint limits (CAD/实测) - currently default +-0.8 (hip), default-1.6..straight+0.1 (knee)",
        "joint damping/friction (placeholder 0.5/0.1 legs, 0.02/0.02 wheels)",
        "leg motor: torque limit / Kp-Kd / rotor inertia armature (placeholder 40 Nm, 60/2, armature 0)",
        "wheel motor: effort limit / velocity limit / damping (placeholder 5 Nm, 60 rad/s, 0.2)",
        "wheel-ground friction (currently simulator defaults; measure on field mats)",
        "inertias: MuJoCo needs balanceinertia for L_link2 (SolidWorks inertia products); verify CAD vs physical",
        "gimbal version: add yaw/pitch joints + links above base_link later",
    ],
}
with open(os.path.join(OUT_DIR, "defaults.json"), "w") as f:
    json.dump(sidecar, f, indent=2, ensure_ascii=False)
print("wrote", out_urdf)
