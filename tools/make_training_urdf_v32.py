#!/usr/bin/env python3
"""Generate the training URDF for urdf_V3.2 (华南虎-updated serial-leg wheeled biped).

Source: /home/yukikaze/Downloads/urdf_V3.2/urdf/urdf_V3.2.SLDASM.urdf
Output: assets/urdf_v32/urdf_V3.2_rl.urdf (+ cleaned meshes, defaults.json)

Audit findings (evidence in repo root logs urdf_audit_v32.log etc.):
1. L-side meshes contain stray export geometry (a 9204-vertex block at the
   R-side world position, duplicated into L_link1/L_link2/L_link3; L_link3
   also carries a second disc r=0.136 sitting exactly at the R wheel axle).
   Fixed: faces with all vertices at world x < -0.1 (source zero pose) are
   removed; large meshes are voxel-decimated to 2 mm for MuJoCo hull speed.
2. Hips/knees exported as 'continuous' -> 'revolute' with limits around the
   default standing pose; wheels stay 'continuous'.
3. Name typo R_jonit2 -> R_joint2 in the asset; firmware name kept in the
   defaults.json mapping (deploy side must keep R_jonit2).
4. Frame convention KEPT AS EXPORTED (no baking): the export frame is
   right-handed with forward = -y, left = +x (L leg at +x is the left leg;
   left = up x forward = z x (-y) = +x). The 35D policy obs contains no base
   linear velocity, so the contract is unaffected; the train env tracks the
   forward command against -v_y (see env_cfg.forward_direction).
5. Default standing pose solved by IK: wheel axles directly under the hips,
   base origin at 0.48 m (wheel radius 0.06 m).
"""
import xml.etree.ElementTree as ET
import json, os, shutil, math
import numpy as np, struct

SRC = "/home/yukikaze/Downloads/urdf_V3.2/urdf/urdf_V3.2.SLDASM.urdf"
MESH_SRC = "/home/yukikaze/Downloads/urdf_V3.2/meshes"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "urdf_v32")

BASE_HEIGHT = 0.48
WHEEL_RADIUS = 0.06

# ---------------------------------------------------------------- helpers
def parse(s): return np.array([float(x) for x in s.split()])
def rot(r):
    rx, ry, rz = r
    cx, sx = np.cos(rx), np.sin(rx); cy, sy = np.cos(ry), np.sin(ry); cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]) @
            np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @
            np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]))
def T(xyz, r):
    M = np.eye(4); M[:3,:3] = rot(r); M[:3,3] = xyz; return M
def axis_rot(axis, a):
    x, y, z = axis
    K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3) + np.sin(a)*K + (1-np.cos(a))*(K@K)

def read_stl(fn):
    with open(fn,"rb") as f:
        f.read(80); n = struct.unpack("<I", f.read(4))[0]
        faces = []
        for _ in range(n):
            f.read(12)
            v = [struct.unpack("<3f", f.read(12)) for _ in range(3)]
            f.read(2)
            faces.append(v)
    return faces

def write_stl(fn, faces):
    out = bytearray(80)
    out += struct.pack("<I", len(faces))
    for (a, b, c) in faces:
        a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
        n = np.cross(b - a, c - a)
        nrm = np.linalg.norm(n)
        n = n / nrm if nrm > 1e-12 else np.zeros(3)
        for v in (n, a, b, c):
            out += struct.pack("<3f", *v)
        out += struct.pack("<H", 0)
    with open(fn, "wb") as f:
        f.write(out)

# ---------------------------------------------------------------- FK (zero pose) on source
tree = ET.parse(SRC)
root = tree.getroot()
J = {}
for j in root.findall("joint"):
    o = j.find("origin")
    xyz = parse(o.attrib["xyz"]) if o is not None else np.zeros(3)
    r = parse(o.attrib["rpy"]) if o is not None and "rpy" in o.attrib else np.zeros(3)
    J[j.attrib["name"]] = dict(parent=j.find("parent").attrib["link"], child=j.find("child").attrib["link"],
                                xyz=xyz, rpy=r,
                                axis=parse(j.find("axis").attrib["xyz"]) if j.find("axis") is not None else np.array([1,0,0]))
F = {"base_link": np.eye(4)}
while len(F) < 7:
    for n, j in J.items():
        if j["child"] not in F and j["parent"] in F:
            F[j["child"]] = F[j["parent"]] @ T(j["xyz"], j["rpy"])

# ---------------------------------------------------------------- mesh cleaning
CLEAN = {"L_link1", "L_link2", "L_link3"}
os.makedirs(os.path.join(OUT_DIR, "meshes"), exist_ok=True)
for name in ("base_link","L_link1","L_link2","L_link3","R_link1","R_link2","R_link3"):
    faces = read_stl(os.path.join(MESH_SRC, name + ".STL"))
    M = F[name]
    if name in CLEAN:
        kept = []
        for (a, b, c) in faces:
            wa = (M[:3,:3] @ np.asarray(a) + M[:3,3])[0]
            wb = (M[:3,:3] @ np.asarray(b) + M[:3,3])[0]
            wc = (M[:3,:3] @ np.asarray(c) + M[:3,3])[0]
            if not (wa < -0.1 and wb < -0.1 and wc < -0.1):
                kept.append((a, b, c))
        n_removed = len(faces) - len(kept)
        faces = kept
        print(f"{name}: removed {n_removed}/{len(kept)+n_removed} stray faces")
    # voxel-decimate very large meshes to 2mm for MuJoCo hull speed
    verts = np.array([v for f in faces for v in f])
    if len(verts) > 50000:
        grid = np.floor(verts / 0.002).astype(np.int64)
        keys, idx = np.unique(grid, axis=0, return_index=True)
        sel = set(idx.tolist())
        faces = [f for f in faces if any(
            tuple(np.floor(np.asarray(v)/0.002).astype(np.int64)) in {} for v in f)]
        # simpler: keep faces where at least one vertex survives voxel downsampling
        keep_faces = []
        for f in faces:
            for v in f:
                if tuple(np.floor(np.asarray(v)/0.002).astype(np.int64)) in set(map(tuple, grid[idx])):
                    keep_faces.append(f); break
        print(f"{name}: decimated {len(verts)} verts -> {len(keys)} unique voxels")
        faces = keep_faces
    write_stl(os.path.join(OUT_DIR, "meshes", name + ".STL"), faces)
print("meshes written")

# ---------------------------------------------------------------- IK standing pose
def fk(hip, knee, wj, th1, th2):
    M = np.eye(4)
    for jname, a in ((hip, th1), (knee, th2)):
        j = J[jname]
        Mr = np.eye(4); Mr[:3,:3] = axis_rot(j["axis"], a)
        M = M @ T(j["xyz"], j["rpy"]) @ Mr
    wj_ = J[wj]
    return (M @ T(wj_["xyz"], wj_["rpy"]) @ np.array([0,0,0,1.0]))[:3]

def solve(hip, knee, wj, target_y, target_z, th0=(0.0,0.0)):
    def f(t):
        p = fk(hip, knee, wj, t[0], t[1])
        return np.array([p[1]-target_y, p[2]-target_z])
    t = np.array(th0, float)
    for it in range(100):
        e = f(t)
        if np.abs(e).max() < 1e-10: break
        Jc = np.zeros((2,2))
        for i in range(2):
            h = 1e-6; tp = t.copy(); tp[i]+=h
            Jc[:,i] = (f(tp)-e)/h
        try: d = np.linalg.solve(Jc, -e)
        except np.linalg.LinAlgError: d = -np.linalg.pinv(Jc) @ e
        t = t + d
    return t, e

tz = -(BASE_HEIGHT - WHEEL_RADIUS)
tL, eL = solve("L_joint1","L_joint2","L_joint3", 0.0, tz)
tR, eR = solve("R_joint1","R_jonit2","R_joint3", 0.0, tz)
print(f"IK standing (H={BASE_HEIGHT}): L(hip={tL[0]:+.4f},knee={tL[1]:+.4f}) err={np.abs(eL).max():.1e} "
      f"R(hip={tR[0]:+.4f},knee={tR[1]:+.4f}) err={np.abs(eR).max():.1e}")
# straight-leg reference (max extension, H=0.569) + crouch reference (H=0.36)
tLs, eLs = solve("L_joint1","L_joint2","L_joint3", 0.0, -(0.569-0.06))
tRs, eRs = solve("R_joint1","R_jonit2","R_joint3", 0.0, -(0.569-0.06))
tLc, eLc = solve("L_joint1","L_joint2","L_joint3", 0.0, -(0.36-0.06))
tRc, eRc = solve("R_joint1","R_jonit2","R_joint3", 0.0, -(0.36-0.06))
print(f"straight(H=0.569): L(hip={tLs[0]:+.4f},knee={tLs[1]:+.4f}) R(hip={tRs[0]:+.4f},knee={tRs[1]:+.4f})")
print(f"crouch (H=0.360): L(hip={tLc[0]:+.4f},knee={tLc[1]:+.4f}) R(hip={tRc[0]:+.4f},knee={tRc[1]:+.4f})")

DEFAULT_POSE = {
    "L_joint1": float(tL[0]), "L_joint2": float(tL[1]), "L_joint3": 0.0,
    "R_joint1": float(tR[0]), "R_joint2": float(tR[1]), "R_joint3": 0.0,
}
def krange(defv, crouch, straight, margin_hip=0.8, margin_knee=0.1):
    lo = min(crouch, defv) - margin_hip if abs(straight-defv) < 1e-9 else min(crouch, defv)
    return [lo, straight + margin_knee] if straight > defv else [straight - margin_knee, max(crouch, defv) + margin_hip]

LIMITS = {
    "L_joint1": [float(min(tLc[0], tL[0]) - 0.2), float(max(tLs[0], tL[0]) + 0.2)],
    "L_joint2": [float(min(tLc[1], tL[1]) - 0.2), float(max(tLs[1], tL[1]) + 0.1)],
    "R_joint1": [float(min(tRc[0], tR[0]) - 0.2), float(max(tRs[0], tR[0]) + 0.2)],
    "R_joint2": [float(min(tRc[1], tR[1]) - 0.2), float(max(tRs[1], tR[1]) + 0.1)],
    "L_joint3": "continuous", "R_joint3": "continuous",
}
print("LIMITS:", {k: (np.round(v,3) if isinstance(v,list) else v) for k,v in LIMITS.items()})

# ---------------------------------------------------------------- URDF edits
root.set("name", "urdf_V3.2_rl")
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
set_limit(joints["L_joint1"], LIMITS["L_joint1"][0], LIMITS["L_joint1"][1])
set_limit(joints["L_joint2"], LIMITS["L_joint2"][0], LIMITS["L_joint2"][1])
set_limit(joints["R_joint1"], LIMITS["R_joint1"][0], LIMITS["R_joint1"][1])
set_limit(joints["R_jonit2"], LIMITS["R_joint2"][0], LIMITS["R_joint2"][1])
set_continuous(joints["L_joint3"])
set_continuous(joints["R_joint3"])
joints["R_jonit2"].set("name", "R_joint2")

for link in root.findall("link"):
    for geom in link.iter("mesh"):
        rel = os.path.basename(geom.attrib["filename"])
        geom.set("filename", os.path.join(OUT_DIR, "meshes", rel))

ET.indent(root, space="  ")
out_urdf = os.path.join(OUT_DIR, "urdf_V3.2_rl.urdf")
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
    "joint_limits": LIMITS,
    "frame_convention": "export frame kept as-is: forward = -y, left = +x (L leg at +x). "
                        "Training env tracks forward command against -v_y (env_cfg.forward_direction). "
                        "Wheel rolling axis = +-x; positive wheel rate drives +y (backward) -> policy learns sign.",
    "mesh_cleanup": "removed stray faces (all verts world x < -0.1 at source zero pose) from L_link1/2/3; "
                    "large meshes voxel-decimated to 2 mm",
    "TODO_identification": [
        "real leg joint limits (CAD/实测) - currently crouch..straight + 0.1..0.2 margin",
        "joint damping/friction (placeholder 0.5/0.1 legs, 0.02/0.02 wheels)",
        "leg motor: torque limit / Kp-Kd / rotor inertia armature (placeholder 40 Nm, 60/2, armature 0)",
        "wheel motor: effort limit / velocity limit / damping (placeholder 5 Nm, 60 rad/s, 0.2)",
        "wheel-ground friction (simulator defaults; measure on field mats)",
        "confirm physical forward direction = -y (evidence: L leg at +x is left)",
        "gimbal version: add yaw/pitch joints + links above base_link later",
    ],
}
with open(os.path.join(OUT_DIR, "defaults.json"), "w") as f:
    json.dump(sidecar, f, indent=2, ensure_ascii=False)
print("wrote", out_urdf)
