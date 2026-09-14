"""Read-only portable geometry driven by one live serial-PhysX state snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


TITLE = "R3A ONNX | REPAIRED VISUALS | SERIAL PHYSX"
ASSET_SHA256 = "4d502e88c5dcb32ad3f88e3c9bbcf2dc8aede3b615d27fbcdf69b5202d3099b9"
KINEMATICS_SHA256 = "c4e3ceead2e61e9bb75c9077e02c611cea4bcb49bdd7865446ee3c4af5e88bd0"
POLICY_SHA256 = "3d8c682cfc94ac0ee1879bdc6fd90c208e123c05e6b48de3f970f42a474c1ad9"


class RepairedKinematics:
    """Source-URDF FK plus two fixed-branch circle closures on each side.

    Matrices use column vectors. The control-frame root is the actual root LINK,
    not its COM. Raw joint coordinates include the actual continuous wheel angle.
    No preview pose, height IK, angle clipping, or grounding is used here.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir).resolve()
        self.hashes = {}
        manifest = json.loads((self.model_dir / "manifest.json").read_bytes())
        for name, expected in (("chassis.usdc", ASSET_SHA256),
                               ("kinematics.json", KINEMATICS_SHA256)):
            raw = (self.model_dir / name).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != expected or digest != manifest["files"][name]["sha256"]:
                raise ValueError(f"repaired visual SHA256 mismatch: {name}")
            self.hashes[name] = digest
        self.data = json.loads((self.model_dir / "kinematics.json").read_bytes())
        self.names = self.data["component_names"]
        self.frames = self.data["source_joint_frames"]
        self.canonical = np.eye(4)
        self.canonical[:3, :3] = self.data["source_to_control_rotation"]
        self.origins = {name: self._origin(frame["origin"]) for name, frame in self.frames.items()}
        self.axes = {name: np.fromstring(frame["axis"]["xyz"], sep=" ")
                     for name, frame in self.frames.items()}

    @staticmethod
    def _rotation(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)

    @classmethod
    def _origin(cls, origin):
        roll, pitch, yaw = np.fromstring(origin["rpy"], sep=" ")
        result = np.eye(4)
        result[:3, 3] = np.fromstring(origin["xyz"], sep=" ")
        result[:3, :3] = (cls._rotation([0, 0, 1], yaw) @ cls._rotation([0, 1, 0], pitch)
                          @ cls._rotation([1, 0, 0], roll))
        return result

    @staticmethod
    def root_matrix(position, quaternion_xyzw):
        position = np.asarray(position, dtype=float)
        quat = np.asarray(quaternion_xyzw, dtype=float)
        if (position.shape != (3,) or quat.shape != (4,)
                or not np.isfinite(position).all() or not np.isfinite(quat).all()):
            raise ValueError("nonfinite or malformed root pose")
        if abs(np.linalg.norm(quat) - 1.) > 1e-4:
            raise ValueError("root quaternion is not unit xyzw")
        x, y, z, w = quat / np.linalg.norm(quat)
        matrix = np.eye(4)
        matrix[:3, :3] = [
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ]
        matrix[:3, 3] = position
        return matrix

    @staticmethod
    def _planar(pivot0, pivot1, angle):
        c, s = np.cos(angle), np.sin(angle)
        result = np.eye(4)
        result[:2, :2] = [[c, -s], [s, c]]
        result[:2, 3] = pivot1 - result[:2, :2] @ pivot0
        return result

    @classmethod
    def _rod(cls, a0, b0, a, b):
        u, v = b0 - a0, b - a
        angle = np.arctan2(u[0]*v[1] - u[1]*v[0], u @ v)
        return cls._planar(a0, a, angle)

    @staticmethod
    def _circle(c0, r0, c1, r1, branch):
        distance = float(np.linalg.norm(c1 - c0))
        margin = min(distance - abs(r0-r1), r0+r1-distance)
        if not np.isfinite([*c0, *c1, r0, r1]).all():
            raise ValueError("nonfinite circle geometry")
        if margin < -1e-12 or distance < 1e-12 or min(r0, r1) <= 0:
            raise ValueError(f"unreachable circle closure: margin={margin:.9g} m")
        direction = (c1-c0) / distance
        along = (r0*r0-r1*r1+distance*distance) / (2*distance)
        # Only roundoff at tangency is rounded; the physical q is never clipped.
        height = np.sqrt(max(0., r0*r0-along*along))
        point = c0 + along*direction + branch*height*np.array([-direction[1], direction[0]])
        return point, float(margin)

    def _closure(self, side, q):
        data = self.data["sides"][side]
        xy = {key: np.asarray(value) for key, value in data["axes_local_xy_m"].items()}
        knee = "L_joint2" if side == "L" else "R_jonit2"
        shank = self.origins[knee].copy()
        shank[:3, :3] = shank[:3, :3] @ self._rotation(self.axes[knee], q[knee])
        e = (shank @ np.r_[xy["shank_E"], 0., 1.])[:2]
        o, p = xy["C0_O"], xy["C3_P"]
        b, upper_margin = self._circle(p, np.linalg.norm(xy["C3_B"]-p), e,
                                      np.linalg.norm(xy["C4_E"]-xy["C4_B"]), data["upper_branch"])
        rocker = self._rod(p, xy["C3_B"], p, b)
        d = (rocker @ np.r_[xy["C3_D"], 0., 1.])[:2]
        a, lower_margin = self._circle(o, np.linalg.norm(xy["C0_A"]-o), d,
                                      np.linalg.norm(xy["C1_D"]-xy["C1_A"]), data["lower_branch"])
        local = {"C0": self._rod(o, xy["C0_A"], o, a),
                 "C1": self._rod(xy["C1_A"], xy["C1_D"], a, d), "C2": np.eye(4),
                 "C3": rocker, "C4": self._rod(xy["C4_B"], xy["C4_E"], b, e), "shank": shank}
        pairs = (("C0", "C0_A", "C1", "C1_A"), ("C1", "C1_D", "C3", "C3_D"),
                 ("C3", "C3_B", "C4", "C4_B"), ("C4", "C4_E", "shank", "shank_E"),
                 ("C2", "C2_P", "C3", "C3_P"), ("C0", "C0_O", "C2", "C2_O"),
                 ("C2", "C2_K", "shank", "shank_K"))
        def pin(part, feature):
            return (local[part] @ np.r_[xy[feature], 0., 1.])[:3]
        gap = max(float(np.linalg.norm(pin(a, b)-pin(c, d))) for a, b, c, d in pairs)
        if gap > 2e-8:
            raise ValueError(f"{side} closure residual too large: {gap} m")
        inner = float(np.rad2deg(np.pi + data["source_knee_origin_z_rad"]
                                 + data["source_knee_axis_sign"] * q[knee]))
        return local, dict(inner_deg=inner, outside_preview_range=not 35 <= inner <= 80,
                           max_pin_gap_m=gap, upper_margin_m=upper_margin, lower_margin_m=lower_margin)

    def pose(self, position, quaternion_xyzw, q):
        if set(q) != set(self.frames) or not np.isfinite(list(q.values())).all():
            raise ValueError("require six finite raw source joint coordinates")
        root = self.root_matrix(position, quaternion_xyzw)
        fk = {"base_link": root @ self.canonical}
        for name, frame in self.frames.items():
            motion = np.eye(4)
            motion[:3, :3] = self._rotation(self.axes[name], q[name])
            fk[frame["child"]["link"]] = fk[frame["parent"]["link"]] @ self.origins[name] @ motion
        transforms, diagnostics = {"base": fk["base_link"]}, {}
        for side in ("L", "R"):
            local, diagnostics[side] = self._closure(side, q)
            for name in ("C0", "C1", "C2", "C3", "C4"):
                transforms[f"{side}_{name}"] = fk[f"{side}_link1"] @ local[name]
            transforms[f"{side}_shank"] = fk[f"{side}_link2"]
            transforms[f"{side}_wheel"] = fk[f"{side}_link3"]
        if not all(np.isfinite(matrix).all() for matrix in transforms.values()):
            raise ValueError("nonfinite visual world matrices")
        return transforms, diagnostics


class RepairedVisuals:
    """USD render-only adapter. Solves all nodes before authoring any pose."""

    PATH = "/World/RepairedVisuals"
    SAMPLE_BOUNDARY = "post env.step auto-reset; one root-link xyzw + six-q snapshot; visible at next render"

    def __init__(self, stage, model, robot_prim_path, report_dir):
        from pxr import Usd, UsdGeom

        self.stage, self.model = stage, model
        if stage.GetPrimAtPath(self.PATH):
            raise ValueError("repaired visual path already occupied")
        root = UsdGeom.Xform.Define(stage, self.PATH)
        root.GetPrim().GetReferences().AddReference(str(model.model_dir / "chassis.usdc"))
        root.SetResetXformStack(True)
        prims = list(Usd.PrimRange(root.GetPrim()))
        if any("Physics" in api or "Physx" in api for prim in prims for api in prim.GetAppliedSchemas()):
            raise ValueError("render-only reference contains Physics APIs")
        if sum(prim.IsA(UsdGeom.Mesh) for prim in prims) != 15:
            raise ValueError("expected exactly 15 embedded visual meshes")
        self.ops = {}
        for source, name in model.names.items():
            node = UsdGeom.Xformable(stage.GetPrimAtPath(self.PATH + "/" + name))
            node.ClearXformOpOrder()
            self.ops[source] = node.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
        self.visibility = UsdGeom.Imageable(stage.GetPrimAtPath(robot_prim_path)).GetVisibilityAttr()
        if not self.visibility:
            raise ValueError(f"missing robot visibility: {robot_prim_path}")
        self.old_visibility = self.visibility.Get()
        self.old_authored = self.visibility.HasAuthoredValueOpinion()
        self.stream = (Path(report_dir) / "repaired_visual_samples.jsonl").open("x")
        self.visibility.Set(UsdGeom.Tokens.invisible)
        self.closed = False
        self.metadata = dict(title=TITLE, visual_bodies=15, actual_physics_bodies=7,
                             new_mass=False, new_dynamic_constraints=False, physics_apis_on_visuals=False,
                             model_dir=str(model.model_dir), sha256=model.hashes,
                             sample_boundary=self.SAMPLE_BOUNDARY, samples=0,
                             outside_preview_range_samples=0, max_pin_gap_m=0.)

    def update(self, env):
        from pxr import Gf

        # No app.update/render/physics step between these reads or the 15 writes.
        data = env.robot.data
        position = data.root_link_pos_w.torch[0].detach().cpu().tolist()
        quat = data.root_link_quat_w.torch[0].detach().cpu().tolist()
        values = env._joint_state()[0][0].detach().cpu().tolist()
        q = dict(zip(env.joint_names, values))
        sample = dict(policy_tick=int(env.common_step_counter), root_link_pos_w_m=position,
                      root_link_quat_xyzw=quat, raw_joint_q_rad=q, sample_boundary=self.SAMPLE_BOUNDARY)
        try:
            matrices, diagnostics = self.model.pose(position, quat, q)
            for name, matrix in matrices.items():
                self.ops[name].Set(Gf.Matrix4d(matrix.T.tolist()))
            sample["closure"] = diagnostics
        except Exception as exc:
            sample["error"] = str(exc)
            # Preserve even invalid raw input in the failure stream, never in summary JSON.
            self.stream.write(json.dumps(sample) + "\n")
            self.stream.flush()
            raise
        self.stream.write(json.dumps(sample, allow_nan=False) + "\n")
        self.stream.flush()
        self.metadata["samples"] += 1
        self.metadata["latest"] = sample
        self.metadata["outside_preview_range_samples"] += int(any(
            side["outside_preview_range"] for side in diagnostics.values()))
        self.metadata["max_pin_gap_m"] = max(self.metadata["max_pin_gap_m"],
                                              *(side["max_pin_gap_m"] for side in diagnostics.values()))

    def close(self):
        from pxr import UsdGeom

        if not self.closed:
            if self.old_authored:
                self.visibility.Set(self.old_visibility)
            else:
                self.visibility.Clear()
            UsdGeom.Imageable(self.stage.GetPrimAtPath(self.PATH)).MakeInvisible()
            self.stream.close()
            self.closed = True
