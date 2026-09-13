"""Opt-in V40 live geometry transport; imports no renderer, Torch, or Isaac.

The training-side hook reads one selected environment after step/data refresh.
Only the sidecar interprets meshes; all body motion comes from current PhysX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

BODY_NAMES = frozenset(("base_link", "L_link1", "L_link2", "L_link3",
                        "R_link1", "R_link2", "R_link3"))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--live-view", action="store_true", help="Publish current V40 PhysX poses to an isolated browser renderer")
    parser.add_argument("--live-view-python", type=Path, default=os.environ.get("V40_VIEWER_PYTHON"),
                        help="Python in a separate viser==1.0.16 environment")
    parser.add_argument("--live-view-env", type=int, default=0)
    parser.add_argument("--live-view-port", type=int, default=8088)
    parser.add_argument("--live-view-status-port", type=int, default=8089)
    parser.add_argument("--live-view-seconds", type=float, default=3600,
                        help="Finite viewer lifetime; training continues after viewer expiry")
    parser.add_argument("--live-view-wait-seconds", type=float, default=0,
                        help="Optional bounded browser wait BEFORE training; never throttles rollout")


def clean_viewer_environment(environment: dict) -> dict:
    result = dict(environment)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "VIRTUAL_ENV"):
        result.pop(key, None)
    result["PYTHONNOUSERSITE"] = "1"
    return result


def validate_options(args) -> None:
    """CPU-only preflight; avoid starting a sidecar or touching the run directory."""
    if not args.live_view:
        return
    if not 0 <= args.live_view_env < args.num_envs:
        raise ValueError("live-view-env must select one existing environment")
    ports = (args.live_view_port, args.live_view_status_port)
    if any(not 1024 <= port <= 65535 for port in ports) or ports[0] == ports[1]:
        raise ValueError("live view requires two distinct ports in 1024..65535")
    for value, lower, upper in ((args.live_view_seconds, 1, 86400),
                                (args.live_view_wait_seconds, 0, 120)):
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError("live viewer lifetime must be 1..86400s; browser wait 0..120s")
    if args.live_view_python is None or not args.live_view_python.is_file():
        raise ValueError("--live-view-python must name an installed isolated viewer Python")
    code = (
        "import importlib.metadata as m,json; "
        "print(json.dumps({'viser':m.version('viser'),'websockets':m.version('websockets'),"
        "'sim_present':any(d.metadata['Name'].lower() in ('isaacsim','isaacsim-kernel') for d in m.distributions())}))"
    )
    result = subprocess.run([str(args.live_view_python.absolute()), "-c", code],
                            env=clean_viewer_environment(os.environ), capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise ValueError("isolated viewer metadata probe failed; install scripts/v40_live_view/requirements.txt separately")
    versions = json.loads(result.stdout)
    if (versions["viser"] != "1.0.16" or versions["sim_present"]
            or not 13 <= int(versions["websockets"].split(".")[0]) < 16):
        raise ValueError(f"viewer must use the independent, tested runtime: {versions}")


def normalize_pose(pose) -> list[float]:
    """Validate world pose [m, xyzw] and restore a unit quaternion."""
    if len(pose) != 7 or any(not math.isfinite(float(value)) for value in pose):
        raise ValueError("pose must contain seven finite xyz/xyzw values")
    norm = math.sqrt(sum(float(value) ** 2 for value in pose[3:]))
    if norm < 1e-8:
        raise ValueError("zero quaternion cannot represent a body pose")
    return [float(value) for value in pose[:3]] + [float(value) / norm for value in pose[3:]]


def urdf_visuals(asset: dict, body_names: list[str], directory: Path) -> list[dict]:
    """Copy the hash-bound canonical visual meshes and preserve link-local transforms."""
    if len(body_names) != 7 or set(body_names) != BODY_NAMES:
        raise ValueError("live V40 requires seven unique canonical body names")
    urdf = Path(asset["urdf_path"])
    expected_urdf = asset["manifest"]["files_sha256"][asset["manifest"]["urdf"]]
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != expected_urdf:
        raise ValueError("canonical URDF changed after validation")
    root = ET.parse(urdf).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    if len(root.findall("link")) != 7 or set(links) != BODY_NAMES:
        raise ValueError("URDF body identity differs from current PhysX")
    output = []
    for name in body_names:
        visuals = links[name].findall("visual")
        if not visuals:
            raise ValueError(f"missing canonical visual: {name}")
        for index, visual in enumerate(visuals):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError("canonical V40 visuals must reference audited mesh files")
            relative = mesh.attrib["filename"]
            path = (urdf.parent / relative).resolve()
            if not path.is_relative_to(Path(asset["directory"]).resolve()):
                raise ValueError("visual mesh escapes validated asset root")
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if asset["manifest"]["files_sha256"].get(relative) != digest:
                raise ValueError(f"visual mesh changed after asset validation: {relative}")
            local_name = f"{name}-{index}-{digest[:12]}{path.suffix}"
            (directory / local_name).write_bytes(data)

            def vector(element, key, default, size):
                values = [float(value) for value in (element.get(key, default) if element is not None else default).split()]
                if len(values) != size or not all(math.isfinite(value) for value in values):
                    raise ValueError(f"invalid URDF {key}")
                return values

            origin = visual.find("origin")
            xyz = vector(origin, "xyz", "0 0 0", 3)
            rpy = vector(origin, "rpy", "0 0 0", 3)
            r, p, y = [angle / 2 for angle in rpy]
            cr, cp, cy, sr, sp, sy = math.cos(r), math.cos(p), math.cos(y), math.sin(r), math.sin(p), math.sin(y)
            # URDF fixed-axis RPY means Rz(yaw) Ry(pitch) Rx(roll).
            q = normalize_pose([0, 0, 0, sr * cp * cy - cr * sp * sy,
                                cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
                                cr * cp * cy + sr * sp * sy])[3:]
            scale = vector(mesh, "scale", "1 1 1", 3)
            if min(scale) <= 0:
                raise ValueError("visual mesh scale must be positive")
            rgba = vector(visual.find("material/color"), "rgba", "0.8 0.8 0.8 1", 4)
            if not all(0 <= channel <= 1 for channel in rgba):
                raise ValueError("visual color must lie in [0,1]")
            output.append({"body": name, "index": index, "mesh": local_name, "sha256": digest,
                           "source": relative, "position": xyz, "wxyz": [q[3], *q[:3]],
                           "scale": scale, "rgba": rgba})
    return output


class PoseStream:
    """Single-run nonblocking Unix datagrams, avoiding WSL mirrored UDP fragmentation."""

    def __init__(self, run_id, manifest_sha256, names, address, samples, *, clock=time.monotonic):
        self.run_id, self.manifest_sha256 = run_id, manifest_sha256
        self.names, self.address, self.samples = list(names), address, samples
        self.clock, self.next_publish = clock, 0.0
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.seq, self.dropped, self.last = 0, 0, None

    def due(self) -> bool:
        now = self.clock()
        if now < self.next_publish:
            return False
        self.next_publish = now + 0.05
        return True

    def publish(self, poses, *, step, sim_time, phase="training", telemetry=None):
        if len(poses) != len(self.names):
            raise ValueError("pose count differs from manifest")
        self.seq += 1
        message = {"schema": 1, "run_id": self.run_id, "manifest_sha256": self.manifest_sha256,
                   "seq": self.seq, "sent_ns": time.time_ns(), "phase": phase,
                   "step": step, "sim_time": sim_time, "names": self.names,
                   "poses_xyzw": [normalize_pose(pose) for pose in poses], "telemetry": telemetry or {}}
        payload = json.dumps(message, allow_nan=False).encode()
        if len(payload) > 60000:
            raise ValueError("single-env live packet exceeds 60 KB")
        try:
            self.socket.sendto(payload, self.address)
        except (BlockingIOError, ConnectionRefusedError):
            self.dropped += 1
        self.last = message
        self.samples.write(json.dumps(message, allow_nan=False) + "\n")
        self.samples.flush()

    def close(self):
        self.socket.close()


class LatestState:
    """Reject wrong-run, reordered, expired and malformed packets before display."""

    def __init__(self, manifest, digest, *, clock=time.time_ns):
        self.manifest, self.digest, self.clock = manifest, digest, clock
        self.last = None
        self.received = self.rejected = 0
        self.first_training = None

    def accept(self, message) -> bool:
        try:
            if (message["schema"] != 1 or message["run_id"] != self.manifest["run_id"]
                    or message["manifest_sha256"] != self.digest
                    or message["names"] != self.manifest["names"]
                    or message["phase"] not in ("ready", "training", "ended")
                    or type(message["seq"]) is not int or message["seq"] <= 0
                    or type(message["step"]) is not int or message["step"] < 0
                    or not math.isfinite(message["sim_time"]) or message["sim_time"] < 0
                    or abs(self.clock() - message["sent_ns"]) > 2_000_000_000
                    or len(message["poses_xyzw"]) != len(self.manifest["names"])):
                raise ValueError("packet identity/shape/time mismatch")
            if self.last and (message["seq"] <= self.last["seq"] or self.last["phase"] == "ended"):
                raise ValueError("reordered or post-terminal packet")
            for pose in message["poses_xyzw"]:
                normalize_pose(pose)
            json.dumps(message, allow_nan=False)
        except (KeyError, TypeError, ValueError, OverflowError):
            self.rejected += 1
            return False
        self.last = message
        self.received += 1
        if self.first_training is None and message["phase"] == "training":
            self.first_training = message
        return True

    def status(self):
        age = None if self.last is None else (self.clock() - self.last["sent_ns"]) / 1e9
        live = self.last is not None and self.last["phase"] == "training" and 0 <= age < 1
        rate = None
        if self.first_training and self.last["sent_ns"] > self.first_training["sent_ns"]:
            rate = ((self.last["sim_time"] - self.first_training["sim_time"]) * 1e9
                    / (self.last["sent_ns"] - self.first_training["sent_ns"]))
        return {"run_id": self.manifest["run_id"], "received": self.received,
                "rejected": self.rejected, "last": self.last, "age_seconds": age,
                "live": bool(live), "sim_wall_ratio": rate}


class LiveViewSession:
    """Own one isolated renderer and a reversible post-step hook on the RL wrapper."""

    def __init__(self, args, raw_env, wrapped_env, asset, run_dir, clean_environment):
        self.args, self.raw_env, self.wrapped_env = args, raw_env, wrapped_env
        self.directory = Path(run_dir) / "live_view"
        self.directory.mkdir(exist_ok=False)
        self.run_id = uuid.uuid4().hex
        self.child = self.stream = self.samples = self.log = None
        self.original_step = wrapped_env.step
        self.hooked = False
        self.disabled_reason = None
        self.environment = clean_viewer_environment(clean_environment)
        self.asset = asset

    def start(self, *, budget=None):
        names = list(self.raw_env.robot.body_names)
        origin = self.raw_env.scene.env_origins[self.args.live_view_env].detach().cpu().tolist()
        manifest = {"schema": 1, "run_id": self.run_id, "parent_pid": os.getpid(),
                    "label": "V40 training · latest PhysX state", "names": names,
                    "env_id": self.args.live_view_env, "num_envs": self.raw_env.num_envs,
                    "env_origin": origin, "quaternion_order": "xyzw", "position_units": "m",
                    "asset_manifest_sha256": self.asset["asset_manifest_sha256"],
                    "contract_sha256": self.raw_env.contract_sha256,
                    "visuals": urdf_visuals(self.asset, names, self.directory),
                    "joint_names": list(self.raw_env.robot.joint_names),
                    "joint_limits": self.raw_env.joint_limit_physx_report,
                    "source": "Lab3 body_link_pose_w.torch; select env on GPU before CPU readback",
                    "scope": "old canonical V40 research asset; not hardware validation or Round3"}
        path = self.directory / "scene.json"
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        script = Path(__file__).resolve().parents[3] / "scripts/v40_live_view/viewer.py"
        self.log = (self.directory / "viewer.log").open("w")
        self.child = subprocess.Popen(
            [str(self.args.live_view_python.absolute()), str(script), "--manifest", str(path.absolute()),
             "--port", str(self.args.live_view_port), "--status-port", str(self.args.live_view_status_port),
             "--seconds", str(self.args.live_view_seconds)],
            env=self.environment, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        (self.directory / "viewer.pid").write_text(str(self.child.pid))
        deadline = time.monotonic() + 30
        ready_path = self.directory / "ready.json"
        while not ready_path.exists():
            if budget is not None:
                budget.check()
            if self.child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"live viewer failed startup; inspect {self.directory / 'viewer.log'}")
            time.sleep(0.05)  # Startup only, never in a training step.
        ready = json.loads(ready_path.read_text())
        if ready["run_id"] != self.run_id or ready["manifest_sha256"] != digest:
            raise RuntimeError("live viewer ready identity mismatch")
        self.samples = (self.directory / "published.jsonl").open("w")
        if ready["transport"] != "unix-dgram-abstract":
            raise RuntimeError("unsupported live-view transport")
        self.stream = PoseStream(self.run_id, digest, names, "\0" + ready["pose_socket"], self.samples)
        self.capture(phase="ready")
        print(f"V40_LIVE_VIEW_READY {json.dumps(ready)}", flush=True)
        deadline = time.monotonic() + self.args.live_view_wait_seconds
        while self.args.live_view_wait_seconds and not (self.directory / "client-ready.json").exists():
            if budget is not None:
                budget.check()
            if self.child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("bounded pre-training browser wait expired")
            time.sleep(0.05)

        def step(actions):
            result = self.original_step(actions)
            if self.disabled_reason is None and self.stream.due():
                if self.child.poll() is not None:
                    self.disabled_reason = f"viewer exited {self.child.returncode}; training continued"
                else:
                    self.capture(reward=result[1])
            return result

        self.wrapped_env.step = step
        self.hooked = True

    def capture(self, *, reward=None, phase="training"):
        index = self.args.live_view_env
        # ProxyArray.torch is a GPU alias. Index BEFORE detach/cpu to avoid N-env readback.
        poses = self.raw_env.robot.data.body_link_pose_w.torch[index].detach().cpu().tolist()
        telemetry = {
            "command": self.raw_env.commands[index].detach().cpu().tolist(),
            "action": self.raw_env.actions[index].detach().cpu().tolist(),
            "episode_step": int(self.raw_env.episode_length_buf[index].item()),
            "state_boundary": "post-step including any auto-reset; reward belongs to preceding transition",
        }
        if reward is not None:
            telemetry["reward"] = float(reward[index].item())
        step = int(self.raw_env.common_step_counter)
        self.stream.publish(poses, step=step, sim_time=step * self.raw_env.step_dt,
                            phase=phase, telemetry=telemetry)

    def close(self):
        if self.hooked:
            self.wrapped_env.step = self.original_step
            self.hooked = False
        try:
            if self.stream is not None and self.stream.last is not None:
                last = self.stream.last
                self.stream.publish(last["poses_xyzw"], step=last["step"], sim_time=last["sim_time"], phase="ended")
        finally:
            if self.stream is not None:
                self.stream.close()
            if self.samples is not None:
                self.samples.close()
            if self.child is not None:
                try:
                    self.child.wait(timeout=7)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGTERM)
                    try:
                        self.child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid, signal.SIGKILL)
                        self.child.wait(timeout=3)
            if self.log is not None:
                self.log.close()
            (self.directory / "publisher.json").write_text(json.dumps({
                "run_id": self.run_id, "frames": self.stream.seq if self.stream else 0,
                "dropped": self.stream.dropped if self.stream else 0,
                "viewer_exit": self.child.returncode if self.child else None,
                "child_reaped": self.child is None or self.child.poll() is not None,
                "disabled_reason": self.disabled_reason}, indent=2))
