#!/usr/bin/env python3
"""Render Kaiser's actual V5 env0 poses in local Kit; this is a labeled state mirror."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TITLE = "KAISER V5 LIVE TRAINING STATE | NATIVE KIT MIRROR"
REMOTE = r'''
import pathlib, sys, time
path = pathlib.Path(RUN_ROOT) / 'train/live_state.json'
last = None
started = time.monotonic()
while time.monotonic() - started < 259200:
    if path.exists():
        raw = path.read_bytes()
        if raw != last:
            sys.stdout.buffer.write(raw + b'\n')
            sys.stdout.buffer.flush()
            last = raw
    if (path.parent / 'completion.json').exists():
        break
    time.sleep(.1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_receipt", type=Path)
    parser.add_argument("--bundle", type=Path, default=ROOT / "model/纯底盘_v5/urdf")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.launch_receipt.read_text())
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    spec = json.loads((args.bundle / "model_spec.json").read_text())
    asset_sha = hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest()
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((args.bundle / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Viewer asset mismatch: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    ssh = ["ssh", "-S", plan["control_path"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-p", str(plan["ssh_port"]), plan["host"], "python3 -u -"]
    errors = (args.output / "ssh.log").open("w")
    process = subprocess.Popen(ssh, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
    process.stdin.write(("RUN_ROOT = " + repr(plan["remote_root"]) + "\n" + REMOTE).encode())
    process.stdin.close()
    latest, stats = {}, {"frames_received": 0, "validation_errors": 0, "source": plan["remote_root"],
                        "asset_manifest_sha256": asset_sha, "local_physics": False, "transport": "SSH_NDJSON_not_WebRTC"}

    def receive():
        import numpy as np
        for line in process.stdout:
            try:
                state = json.loads(line)
                pose = np.asarray(state["body_link_pose_w"], dtype=float)
                if (state["asset_manifest_sha256"] != asset_sha or state["quaternion_order"] != "xyzw"
                        or len(state["body_names"]) != 19
                        or set(state["body_names"]) != set(manifest["rigid_body_names"])
                        or pose.shape != (19, 7) or not np.isfinite(pose).all()
                        or not np.allclose(np.linalg.norm(pose[:, 3:], axis=1), 1., atol=1e-3)):
                    raise ValueError("Invalid source pose identity/layout")
                state["received_at_unix"] = time.time()
                latest["state"] = state
                stats["frames_received"] += 1
            except Exception as exc:
                stats["validation_errors"] += 1
                stats["last_error"] = str(exc)

    threading.Thread(target=receive, daemon=True).start()
    waiting_until = time.monotonic() + 10
    while "state" not in latest and process.poll() is None and time.monotonic() < waiting_until:
        time.sleep(.1)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": False, "width": 1600, "height": 1000,
                         "renderer": "RaytracedLighting", "multi_gpu": False})
    import carb.windowing
    import numpy as np
    import omni.appwindow
    import omni.ui as ui
    import omni.usd
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    from pxr import Gf, UsdGeom, UsdLux
    carb.windowing.acquire_windowing_interface().set_window_title(omni.appwindow.get_default_app_window().get_window(), TITLE)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    robot_root = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot_root.MakeInvisible()
    operations = {}
    for body in spec["bodies"]:
        path = "/World/Robot/" + body["name"]
        operations[body["name"]] = UsdGeom.Xform.Define(stage, path).AddTransformOp()
        visual = stage.DefinePrim(path + "/Visual")
        visual.GetReferences().AddReference(str((args.bundle / "meshes/visuals.usdc").resolve()), "/Visuals/" + body["component"])
        UsdGeom.Xformable(visual).AddTransformOp().Set(Gf.Matrix4d(np.asarray(body["mesh_origin"]).T.tolist()))
    floor = UsdGeom.Cube.Define(stage, "/World/FoundationFloor")
    floor.CreateSizeAttr(1.)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.05))
    floor.AddScaleOp().Set(Gf.Vec3f(8., 4., .1))
    floor.CreateDisplayColorAttr([Gf.Vec3f(.23, .27, .30)])
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(600.)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1200.)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-35., -20., 30.))
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 100.))
    camera.CreateFocalLengthAttr(35.)
    camera_op = camera.AddTransformOp()
    viewport = get_active_viewport()
    viewport.camera_path = str(camera.GetPath())
    viewport.resolution = (1600, 1000)
    panel = ui.Window("Kaiser training state / live source", width=470, height=300)
    initial = latest.get("state", {})
    choices = [e["scene_group"] for e in initial.get("environments", [])] or ["env0"]
    with panel.frame:
        with ui.VStack(spacing=8):
            ui.Label("LIVE ENV 0 / NATIVE KIT STATE MIRROR")
            ui.Label("Actual remote PhysX poses; no local physics or policy", word_wrap=True)
            ui.Label(Path(plan["remote_root"]).name, word_wrap=True)
            selector = ui.ComboBox(0, *choices)
            status = ui.Label("Waiting for a valid source frame...", word_wrap=True, height=110)
            with ui.HStack(height=24):
                follow = ui.CheckBox(width=24)
                follow.model.set_value(True)
                ui.Label("Follow chassis")
            ui.Label("Updates <=4 Hz. Falls and reset jumps belong to the untrained policy.", word_wrap=True)
    captures, frames, previous, selected_before = [], 0, None, None
    last_save = 0.
    try:
        while app.is_running():
            started = time.monotonic()
            state = latest.get("state")
            if state is not None:
                age = time.time() - state["wall_time_unix"]
                selected_index = selector.model.get_item_value_model().as_int
                environments = state.get("environments", [])
                selected = environments[min(selected_index, len(environments) - 1)] if environments else state
                if selected_index != selected_before and environments:
                    from wheeled_tasks.chassis.task import Surface
                    floor.MakeInvisible()
                    stage.RemovePrim("/World/SelectedTerrain")
                    for index, description in enumerate(selected["surfaces"]):
                        size, position, orientation = Surface(**description).box()
                        box = UsdGeom.Cube.Define(stage, f"/World/SelectedTerrain/piece_{index}")
                        box.CreateSizeAttr(1.)
                        position = np.asarray(position) + np.asarray(selected["origin"])
                        box.AddTranslateOp().Set(Gf.Vec3d(*position))
                        box.AddOrientOp().Set(Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))
                        box.AddScaleOp().Set(Gf.Vec3f(*size))
                        box.CreateDisplayColorAttr([Gf.Vec3f(.23, .27, .30)])
                if state["wall_time_unix"] != previous or selected_index != selected_before:
                    robot_root.MakeVisible()
                    for name, values in zip(state["body_names"], selected["body_link_pose_w"]):
                        matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(values[6], Gf.Vec3d(*values[3:6])))
                        matrix.SetTranslateOnly(Gf.Vec3d(*values[:3]))
                        operations[name].Set(matrix)
                    if follow.model.as_bool:
                        p = np.array(selected["body_link_pose_w"][state["body_names"].index("base_link")][:3])
                        eye, target = Gf.Vec3d(*(p + [1., 1.8, .8])), Gf.Vec3d(*p)
                        camera_op.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
                    previous = state["wall_time_unix"]
                    selected_before = selected_index
                flag = "LIVE" if age < 2 else "STALE"
                status.text = (f"{flag} | source age {age:.2f} s | received {stats['frames_received']} frames\n"
                    f"scene {choices[min(selected_index, len(choices)-1)]} | sim {state['sim_time_s']:.2f} s\n"
                    f"episode step {selected['episode_step']} | command {selected['commands']}\n"
                    f"asset {asset_sha[:12]} | quat xyzw")
                if time.monotonic() - last_save > 1:
                    (args.output / "runtime.json").write_text(json.dumps({**stats, "status": flag, "source_age_s": age,
                        "current": state, "selected_group": choices[min(selected_index, len(choices)-1)],
                        "pid": os.getpid()}, indent=2) + "\n")
                    last_save = time.monotonic()
                if stats["frames_received"] >= 10 and not captures:
                    captures.append(capture_viewport_to_file(viewport, str(args.output / "live_viewport.png"), is_hdr=False))
            app.update()
            frames += 1
            time.sleep(max(0, .05 - (time.monotonic() - started)))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        errors.close()
        (args.output / "finished.json").write_text(json.dumps(stats, indent=2) + "\n")
        app.close()


if __name__ == "__main__":
    main()
