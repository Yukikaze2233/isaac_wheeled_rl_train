#!/usr/bin/env python3
"""Independent Viser/WebGL renderer of canonical V40 meshes and live PhysX poses."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time

import numpy as np
import trimesh
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wheeled_tasks.v40.live_view import LatestState, normalize_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--status-port", type=int, default=8089)
    parser.add_argument("--seconds", type=float, default=3600)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 1 <= args.seconds <= 86400:
        parser.error("finite lifetime must be 1..86400 seconds")
    directory = args.manifest.parent
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    state = LatestState(manifest, digest)
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    socket_name = "v40-live-" + manifest["run_id"]
    receiver.bind("\0" + socket_name)
    receiver.setblocking(False)
    server = httpd = None
    thread = None
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    counts = {"clients": 0, "max_clients": 0, "mesh_count": 0, "vertices": 0, "faces": 0}
    reason = "startup_failed"
    try:
        server = viser.ViserServer(host="127.0.0.1", port=args.port, label="V40 | Training live state")
        if server.get_port() != args.port:
            raise RuntimeError("live-view port occupied; refusing silent port fallback")
        server.scene.set_up_direction("+z")
        origin = np.asarray(manifest["env_origin"])
        ground_z = -origin[2]
        server.scene.add_grid("/ground_grid", width=8, height=8, plane="xy", position=(0, 0, ground_z + 0.001))
        server.scene.add_box("/ground", dimensions=(8, 8, 0.02), position=(0, 0, ground_z - 0.01),
                             color=(155, 165, 175))
        prefix = "/run_" + manifest["run_id"]
        handles = {name: server.scene.add_frame(prefix + "/" + name, show_axes=False)
                   for name in manifest["names"]}
        for visual in manifest["visuals"]:
            path = (directory / visual["mesh"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("mesh outside run-local manifest directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != visual["sha256"]:
                raise ValueError("run-local visual mesh hash mismatch")
            mesh = trimesh.load(path, force="mesh", process=False)
            vertices = np.asarray(mesh.vertices) * np.asarray(visual["scale"])
            if not np.isfinite(vertices).all() or len(vertices) == 0:
                raise ValueError("invalid canonical mesh vertices")
            server.scene.add_mesh_simple(
                f"{prefix}/{visual['body']}/visual_{visual['index']}", vertices, mesh.faces,
                position=tuple(visual["position"]), wxyz=tuple(visual["wxyz"]),
                color=tuple(round(c * 255) for c in visual["rgba"][:3]),
                opacity=visual["rgba"][3], flat_shading=True)
            counts["mesh_count"] += 1
            counts["vertices"] += len(vertices)
            counts["faces"] += len(mesh.faces)
        status_text = server.gui.add_markdown("## WAITING FOR TRAINING")
        server.gui.add_markdown(
            f"Run `{manifest['run_id']}` · env **{manifest['env_id']} / {manifest['num_envs']}**\n\n"
            "Canonical V40 research URDF/STL · **7 PhysX bodies**.\n\n"
            "Latest sampled training state, target ≤20 Hz wall clock. Simulation may run faster or slower "
            "than wall time; this is **not real-speed video**.\n\n"
            "Old research mechanics, including short links. No geometry redesign, second physics, "
            "hardware acceptance, or Round3 claim.\n\n"
            "Poses are post-step/auto-reset; reward describes the preceding transition. "
            "Optimizer/export gaps may show STALE. Drag to orbit; scroll to zoom."
        )

        @server.on_client_connect
        def on_connect(client):
            client.camera.position = (1.05, 1.05, 0.85)
            client.camera.look_at = (0, 0, 0.25)
            ready = directory / "client-ready.json"
            temporary = directory / f".client-{client.client_id}.json"
            temporary.write_text(json.dumps({"run_id": manifest["run_id"], "client_id": client.client_id}))
            temporary.replace(ready)

        def snapshot():
            return {**state.status(), **counts, "names": manifest["names"],
                    "asset_manifest_sha256": manifest["asset_manifest_sha256"]}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = json.dumps(snapshot(), allow_nan=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_):
                pass

        httpd = ThreadingHTTPServer(("127.0.0.1", args.status_port), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        ready = {"run_id": manifest["run_id"], "manifest_sha256": digest,
                 "url": f"http://127.0.0.1:{args.port}", "status_url": f"http://127.0.0.1:{args.status_port}",
                 "transport": "unix-dgram-abstract", "pose_socket": socket_name,
                 "pid": os.getpid(), **counts}
        temporary = directory / ".ready.json"
        temporary.write_text(json.dumps(ready, indent=2))
        temporary.replace(directory / "ready.json")
        start, ended = time.monotonic(), None
        reason = "lifetime_expired"
        with (directory / "received.jsonl").open("w") as evidence:
            while not stop.is_set() and time.monotonic() - start < args.seconds:
                if os.getppid() != manifest["parent_pid"]:
                    reason = "training_parent_exited"
                    break
                newest = None
                while True:
                    try:
                        payload = receiver.recv(65535)
                    except BlockingIOError:
                        break
                    try:
                        message = json.loads(payload)
                    except (ValueError, UnicodeError):
                        state.rejected += 1
                        continue
                    if state.accept(message):
                        newest = message
                        evidence.write(json.dumps(message) + "\n")
                        evidence.flush()
                if newest:
                    with server.atomic():
                        for name, values in zip(newest["names"], newest["poses_xyzw"], strict=True):
                            pose = normalize_pose(values)
                            handles[name].position = tuple(np.asarray(pose[:3]) - origin)
                            handles[name].wxyz = (pose[6], *pose[3:6])
                counts["clients"] = len(server.get_clients())
                counts["max_clients"] = max(counts["clients"], counts["max_clients"])
                current = snapshot()
                last = current["last"]
                if last:
                    phase = last["phase"]
                    label = "TRAINING · LIVE" if current["live"] else "ENDED" if phase == "ended" else "READY" if phase == "ready" else "STALE · NO RECENT STEP"
                    ratio = current["sim_wall_ratio"]
                    rate = "n/a" if ratio is None else f"{ratio:.2f}× wall clock"
                    status_text.content = (
                        f"## {label}\n\nseq **{last['seq']}** · step **{last['step']}** · sim **{last['sim_time']:.3f}s**\n\n"
                        f"age {current['age_seconds']:.3f}s · {rate}\n\n"
                        f"command `{last['telemetry'].get('command')}`\n\nreward `{last['telemetry'].get('reward')}`"
                    )
                    if phase == "ended":
                        ended = ended or time.monotonic()
                        if time.monotonic() - ended >= 3:
                            reason = "training_ended"
                            break
                time.sleep(0.02)  # Sidecar only; never sleeps the trainer.
        if stop.is_set():
            reason = "signal"
    finally:
        (directory / "viewer.json").write_text(json.dumps({**state.status(), **counts, "exit_reason": reason}, indent=2))
        receiver.close()
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=2)
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
