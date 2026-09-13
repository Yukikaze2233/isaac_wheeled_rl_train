#!/usr/bin/env python3
"""Cross-check PPO/ONNX completion and browser-delivered poses against live publications."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import socket

import msgspec
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True, help="train directory containing completion.json")
parser.add_argument("--browser", type=Path, required=True, help="directory containing browser.json and websocket-frames.json")
args = parser.parse_args()
live = args.run / "live_view"
manifest = json.loads((live / "scene.json").read_text())
completion = json.loads((args.run / "completion.json").read_text())
publisher = json.loads((live / "publisher.json").read_text())
viewer = json.loads((live / "viewer.json").read_text())
browser = json.loads((args.browser / "browser.json").read_text())
onnx = json.loads((args.run / "policy.onnx.json").read_text())
published = [json.loads(line) for line in (live / "published.jsonl").read_text().splitlines()]
received = [json.loads(line) for line in (live / "received.jsonl").read_text().splitlines()]
assert completion["status"] == "completed" and completion["completed_updates"] in (2, 3)
assert completion["export_status"] == "verified" and onnx["validation"]["passed"]
assert publisher["child_reaped"] and publisher["viewer_exit"] == 0
assert not viewer["live"] and viewer["exit_reason"] == "training_ended"
assert viewer["mesh_count"] == len(manifest["names"]) == 7 and manifest["num_envs"] == 2
assert manifest["joint_limits"]["shape"] == [2, 6, 2]
assert browser["training_live_observed"] and browser["ended_observed"] and not browser["runtime_errors"]
assert viewer["run_id"] == publisher["run_id"] == manifest["run_id"]
by_seq = {message["seq"]: message for message in published}
assert all(message == by_seq[message["seq"]] for message in received)
training = [message for message in received if message["phase"] == "training"]
assert len(training) >= 3 and len({message["step"] for message in training}) >= 3
frames = json.loads((args.browser / "websocket-frames.json").read_text())
positions, orientations = {}, {}
mesh_messages = set()
for frame in frames:
    if frame["opcode"] != 2:
        continue
    bundle = msgspec.msgpack.decode(base64.b64decode(frame["payloadData"]))
    for message in bundle["messages"]:
        name = message.get("name", "")
        if message["type"] == "SetPositionMessage":
            positions.setdefault(name, []).append(message["position"])
        elif message["type"] == "SetOrientationMessage":
            orientations.setdefault(name, []).append(message["wxyz"])
        elif "Mesh" in message["type"] and "/visual_" in name:
            mesh_messages.add(name)
origin = np.asarray(manifest["env_origin"])
matches = []
for sample in training:
    for name, pose in zip(sample["names"], sample["poses_xyzw"], strict=True):
        path = f"/run_{manifest['run_id']}/{name}"
        p = np.asarray(pose[:3]) - origin
        q = [pose[6], *pose[3:6]]
        if (any(np.allclose(value, p, rtol=0, atol=1e-10) for value in positions.get(path, []))
                and any(np.allclose(value, q, rtol=0, atol=1e-10) for value in orientations.get(path, []))):
            matches.append({"seq": sample["seq"], "step": sample["step"], "body": name})
assert len(mesh_messages) == 7, mesh_messages
assert len({match["seq"] for match in matches}) >= 3, matches
assert {match["body"] for match in matches} == set(manifest["names"])
with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as probe:
    probe.bind("\0v40-live-" + manifest["run_id"])
    abstract_socket_released = True
ports = {}
ready = json.loads((live / "ready.json").read_text())
for url in (ready["url"], ready["status_url"]):
    port = int(url.rsplit(":", 1)[1])
    with socket.socket() as probe:
        ports[str(port)] = probe.connect_ex(("127.0.0.1", port)) == 0
assert not any(ports.values())
result = {"run_id": manifest["run_id"], "completed_updates": completion["completed_updates"],
          "learning_seconds": completion["learning_elapsed_seconds"], "onnx_validation": onnx["validation"],
          "training_messages": len(training), "published_frames": publisher["frames"],
          "received_frames": viewer["received"], "dropped": publisher["dropped"], "rejected": viewer["rejected"],
          "browser_frames": len(frames), "browser_meshes": sorted(mesh_messages),
          "matched_body_poses": len(matches), "matched_policy_steps": sorted({m["step"] for m in matches}),
          "matches": matches, "joint_limits": manifest["joint_limits"],
          "canonical_visual_sha256": {v["body"]: v["sha256"] for v in manifest["visuals"]},
          "child_reaped": publisher["child_reaped"], "tcp_listeners_present": ports,
          "abstract_socket_released": abstract_socket_released,
          "policy_sha256": hashlib.sha256((args.run / "policy.onnx").read_bytes()).hexdigest()}
(live / "integration-audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ("matches", "onnx_validation", "joint_limits")}, indent=2))
