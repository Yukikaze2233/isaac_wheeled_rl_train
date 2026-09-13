"""Kit --exec diagnostic: native GUI/video motion and remote selection/camera evidence.

This first-stage scene has explicitly scripted motion, not a training policy.
It runs inside the real Windows Kit renderer; no browser geometry renderer exists.
"""
import builtins
import json
import math
import os
from pathlib import Path
import time

import omni.kit.app
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux
from omni.kit.viewport.utility import get_active_viewport
from isaacsim.core.utils.viewports import set_camera_view

output = Path(os.environ["ISAAC_NATIVE_RUN_DIR"])
context = omni.usd.get_context()
if context.get_stage() is None:
    context.new_stage()
stage = context.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
cube = UsdGeom.Cube.Define(stage, "/World/NativeGuiProbe")
cube.CreateSizeAttr(0.8)
cube.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.55, 0.85)])
translation = cube.AddTranslateOp()
rotation = cube.AddRotateZOp()
floor = UsdGeom.Cube.Define(stage, "/World/GroundVisual")
floor.CreateSizeAttr(1.0)
floor.AddScaleOp().Set(Gf.Vec3f(6, 6, 0.05))
floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.025))
floor.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.32, 0.35)])
light = UsdLux.DomeLight.Define(stage, "/World/Light")
light.CreateIntensityAttr(1500)
set_camera_view(eye=[3, 3, 2.5], target=[0, 0, 0.6])
started = time.monotonic()
frames = 0
last_write = 0.0


def update(_event):
    global frames, last_write
    frames += 1
    elapsed = time.monotonic() - started
    translation.Set(Gf.Vec3d(0, 0, 0.65 + 0.08 * math.sin(elapsed)))
    rotation.Set(elapsed * 30.0)
    if elapsed - last_write < 0.25:
        return
    last_write = elapsed
    viewport = get_active_viewport()
    camera_path = str(viewport.camera_path) if viewport else None
    camera = stage.GetPrimAtPath(camera_path) if camera_path else None
    matrix = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) if camera else None
    record = {"pid": os.getpid(), "frames": frames, "elapsed_seconds": elapsed,
              "selected_paths": context.get_selection().get_selected_prim_paths(),
              "camera_path": camera_path,
              "camera_matrix": [[float(x) for x in row] for row in matrix] if matrix else None,
              "viewport_resolution": list(viewport.resolution) if viewport else None,
              "scope": "native Windows Kit GUI / scripted render diagnostic, not training"}
    temporary = output / ".gui-state.json"
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(output / "gui-state.json")


builtins._isaac_native_gui_probe = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    update, name="native-gui-evidence")
print("NATIVE_GUI_PROBE_REGISTERED " + str(output), flush=True)
