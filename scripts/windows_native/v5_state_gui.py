"""Kit --exec: actual V5 training poses, rendered without a second physics world."""
import builtins
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import omni.kit.app
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Usd, UsdGeom, UsdLux


class TrainingMirror:
    def __init__(self):
        self.bundle = Path(os.environ["V5_MIRROR_BUNDLE"])
        self.source = Path(os.environ["V5_MIRROR_STATE"])
        self.output = Path(os.environ["ISAAC_NATIVE_RUN_DIR"])
        manifest = json.loads((self.bundle / "manifest.json").read_text())
        self.asset_sha = hashlib.sha256((self.bundle / "manifest.json").read_bytes()).hexdigest()
        for name in ("model_spec.json", "meshes/visuals.usdc"):
            if hashlib.sha256((self.bundle / name).read_bytes()).hexdigest() != manifest["files_sha256"][name]:
                raise ValueError("Renderer asset identity mismatch")
        spec = json.loads((self.bundle / "model_spec.json").read_text())
        context = omni.usd.get_context()
        context.new_stage()
        self.stage = context.get_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.)
        world = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(world.GetPrim())
        self.operations = {}
        for body in spec["bodies"]:
            path = "/World/Robot/" + body["name"]
            self.operations[body["name"]] = UsdGeom.Xform.Define(self.stage, path).AddTransformOp()
            visual = self.stage.DefinePrim(path + "/Visual")
            visual.GetReferences().AddReference(str(self.bundle / "meshes/visuals.usdc"), "/Visuals/" + body["component"])
            UsdGeom.Xformable(visual).AddTransformOp().Set(Gf.Matrix4d(np.asarray(body["mesh_origin"]).T.tolist()))
        UsdLux.DomeLight.Define(self.stage, "/World/Light").CreateIntensityAttr(650.)
        camera = UsdGeom.Camera.Define(self.stage, "/World/Camera")
        camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 100.))
        camera.CreateFocalLengthAttr(35.)
        self.camera_op = camera.AddTransformOp()
        self.viewport = get_active_viewport()
        self.viewport.camera_path = "/World/Camera"
        self.viewport.resolution = (1280, 720)
        self.panel = ui.Window("V5 live training state - renderer mirror", width=440, height=280)
        self.groups = ()
        self.selection = None
        self.source_timestamp = None
        self.frames = 0
        self.received = 0
        self.last_write = 0.
        self.status = None
        self.rebuild_panel(("Waiting for source",))
        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update, name="v5-training-mirror")

    def rebuild_panel(self, groups):
        self.groups = tuple(groups)
        with self.panel.frame:
            with ui.VStack(spacing=8):
                ui.Label("REMOTE TRAINING STATE / NO VIEWER PHYSICS", word_wrap=True)
                self.selector = ui.ComboBox(0, *groups)
                self.status = ui.Label("Waiting for verified source data", word_wrap=True, height=140)
                with ui.HStack(height=24):
                    self.follow = ui.CheckBox(width=24)
                    self.follow.model.set_value(False)
                    ui.Label("Follow robot (disable to orbit freely)")

    def terrain(self, environment, floor_width):
        self.stage.RemovePrim("/World/Terrain")
        origin = np.asarray(environment["origin"])
        for index, s in enumerate(environment["surfaces"]):
            cross = s.get("cross_slope", 0.)
            width = floor_width if environment["terrain"] in ("flat", "jump") else 4.
            angle = math.atan(cross or s["slope"])
            if cross:
                size = (s["x1"] - s["x0"], width / math.cos(angle), .1)
                position = ((s["x0"] + s["x1"]) / 2, .05 * math.sin(angle), s["z0"] - .05 * math.cos(angle))
                quat = (math.cos(angle / 2), math.sin(angle / 2), 0., 0.)
            else:
                size = ((s["x1"] - s["x0"]) / math.cos(angle), width, .1)
                position = ((s["x0"] + s["x1"]) / 2 + .05 * math.sin(angle), 0.,
                            s["z0"] + (s["x1"] - s["x0"]) / 2 * s["slope"] - .05 * math.cos(angle))
                quat = (math.cos(angle / 2), 0., -math.sin(angle / 2), 0.)
            box = UsdGeom.Cube.Define(self.stage, f"/World/Terrain/piece_{index}")
            box.CreateSizeAttr(1.)
            box.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(position) + origin)))
            box.AddOrientOp().Set(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))
            box.AddScaleOp().Set(Gf.Vec3f(*size))
            box.CreateDisplayColorAttr([Gf.Vec3f(.23, .27, .30)])

    def update(self, _event):
        self.frames += 1
        try:
            state = json.loads(self.source.read_text())
            if state["asset_manifest_sha256"] != self.asset_sha or state["quaternion_order"] != "xyzw":
                raise ValueError("Source asset or quaternion convention mismatch")
            if set(state["body_names"]) != set(self.operations):
                raise ValueError("Source body identity mismatch")
            environments = state["environments"]
            groups = tuple(e["scene_group"] for e in environments)
            if groups != self.groups:
                self.rebuild_panel(groups)
            index = min(self.selector.model.get_item_value_model().as_int, len(environments) - 1)
            selected = environments[index]
            key = (state["contract_sha256"], selected["env_index"], selected["scene_group"])
            changed = key != self.selection
            if changed:
                self.terrain(selected, state["mirror"]["flat_floor_width_m"])
            if state["wall_time_unix"] != self.source_timestamp or changed:
                poses = np.asarray(selected["body_link_pose_w"], dtype=float)
                if poses.shape != (19, 7) or not np.isfinite(poses).all() or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-3):
                    raise ValueError("Invalid source pose frame")
                for name, p in zip(state["body_names"], poses):
                    matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(p[6], Gf.Vec3d(*p[3:6])))
                    self.operations[name].Set(matrix.SetTranslateOnly(Gf.Vec3d(*p[:3])))
                if changed or self.follow.model.as_bool:
                    p = poses[state["body_names"].index("base_link"), :3]
                    self.camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(p + [1., 1.8, .8])), Gf.Vec3d(*p), Gf.Vec3d(0, 0, 1)).GetInverse())
                self.selection, self.source_timestamp = key, state["wall_time_unix"]
                self.received += 1
            age = time.time() - state["wall_time_unix"]
            mirror = state["mirror"]
            flag = "FINISHED" if mirror["run_finished"] else "LIVE" if age < 2 and mirror["phase"] == "training" else "STALE / EVALUATING"
            self.status.text = (f"{flag} | source age {age:.2f}s | source <=4Hz\n"
                f"stage {mirror['active_stage']} | updates {mirror['updates']} | envs {mirror.get('num_envs')}\n"
                f"{selected['scene_group']} | sim {state['sim_time_s']:.2f}s\n"
                f"commands {selected['commands']}\nactor {mirror['actor_dim']}D | period {mirror['policy_dt']}s")
            if time.monotonic() - self.last_write > .5:
                camera = self.stage.GetPrimAtPath(str(self.viewport.camera_path))
                matrix = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                record = {"pid": os.getpid(), "render_updates": self.frames, "source_frames": self.received,
                    "source_age_s": age, "status": flag, "source_run": mirror["run_root"], "asset_manifest_sha256": self.asset_sha,
                    "selected_group": selected["scene_group"], "source_sim_time_s": state["sim_time_s"],
                    "selected_paths": omni.usd.get_context().get_selection().get_selected_prim_paths(),
                    "camera_matrix": [[float(x) for x in row] for row in matrix],
                    "local_physics": False, "scope": "real training state rendered by Windows Kit, not the training process GUI"}
                temporary = self.output / "gui-state.tmp"
                temporary.write_text(json.dumps(record, indent=2))
                temporary.replace(self.output / "gui-state.json")
                self.last_write = time.monotonic()
        except (OSError, ValueError, KeyError) as error:
            self.status.text = "WAIT / STALE: " + str(error)


builtins._v5_training_mirror = TrainingMirror()
print("V5_TRAINING_MIRROR_READY", flush=True)
