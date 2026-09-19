#!/usr/bin/env python3
"""Read-only run snapshot and CRC-checked TensorBoard window summary."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c


ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import hashlib, io, json, os, pathlib, sys, tarfile
root = pathlib.Path(RUN_ROOT) / 'train'
paths = [root/n for n in ('progress.json','contract.json','torque_monitor.json','torque_history.jsonl','live_state.json')]
paths += sorted(root.glob('events.out.tfevents.*'))
manifest = {}
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
    for path in paths:
        if not path.exists():
            continue
        with path.open('rb') as stream:
            size = os.fstat(stream.fileno()).st_size
            data = stream.read(size)
        info = tarfile.TarInfo(path.name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        manifest[path.name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    data = json.dumps(manifest, indent=2).encode()
    info = tarfile.TarInfo('snapshot_manifest.json')
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))
'''


def events(path):
    data = path.read_bytes()
    offset, scalars = 0, {}
    while offset + 12 <= len(data):
        length, header_crc = struct.unpack_from("<QI", data, offset)
        if masked_crc32c(data[offset:offset + 8]) != header_crc:
            raise ValueError("TensorBoard record header CRC mismatch")
        end = offset + 12 + length + 4
        if end > len(data):
            break
        payload = data[offset + 12:end - 4]
        if masked_crc32c(payload) != struct.unpack_from("<I", data, end - 4)[0]:
            raise ValueError("TensorBoard record data CRC mismatch")
        event = Event.FromString(payload)
        for value in event.summary.value:
            if value.HasField("simple_value"):
                scalars.setdefault(value.tag, []).append((event.step, value.simple_value))
        offset = end
    return scalars, {"complete_record_bytes": offset, "incomplete_live_tail_bytes": len(data) - offset}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.receipt.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    command = ["ssh", "-S", plan["control_path"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-p", str(plan["ssh_port"]), plan["host"], "python3 -B -"]
    response = subprocess.run(command, input=("RUN_ROOT=" + repr(plan["remote_root"]) + "\n" + REMOTE).encode(),
                              capture_output=True, timeout=60, check=True)
    with tarfile.open(fileobj=io.BytesIO(response.stdout)) as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).name != member.name:
                raise ValueError("Unexpected snapshot member")
            (args.output / member.name).write_bytes(archive.extractfile(member).read())
    identity = json.loads((args.output / "snapshot_manifest.json").read_text())
    for name, metadata in identity.items():
        if hashlib.sha256((args.output / name).read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Snapshot transfer hash mismatch")
    scalar_data, record_audit = {}, {}
    for path in args.output.glob("events.out.tfevents.*"):
        values, info = events(path)
        record_audit[path.name] = info
        for tag, samples in values.items():
            scalar_data.setdefault(tag, []).extend(samples)
    selected = {tag: sorted(values) for tag, values in scalar_data.items()
                if any(s in tag.lower() for s in ("reward", "episode_length", "height", "success", "flight", "contact", "pin_gap"))
                and "/time" not in tag.lower()}
    windows = {}
    for tag, values in selected.items():
        last = max(s for s, _ in values)
        windows[tag] = {}
        for label, lo, hi in (("first100", 0, 99), ("updates500_599", 500, 599),
                              ("updates2000_2099", 2000, 2099), ("last100", last - 99, last)):
            array = np.array([v for s, v in values if lo <= s <= hi])
            if len(array):
                windows[tag][label] = {"step_range": [lo, hi], "samples": len(array),
                    "mean": float(array.mean()), "min": float(array.min()), "max": float(array.max())}
    contract = json.loads((args.output / "contract.json").read_text())
    monitor = json.loads((args.output / "torque_monitor.json").read_text())
    progress = json.loads((args.output / "progress.json").read_text())
    counts, allocated = {}, 0
    groups = contract.get("scene_groups", [{"name": "foundation", "fraction": 1.}])
    for index, group in enumerate(groups):
        count = progress["num_envs"] - allocated if index == len(groups) - 1 else int(progress["num_envs"] * group["fraction"])
        counts[group["name"]] = count
        allocated += count
    updates = monitor.get("successful_updates", progress["successful_updates"])
    sample_counts = {}
    for name, group in monitor["groups"].items():
        expected = updates * contract["num_steps_per_env"] * round(contract["policy_dt"] / contract["physics_dt"]) * counts[name]
        observed = group["physics_samples"]
        sample_counts[name] = {"expected": expected, "recorded": observed,
                               "relative_error": (observed - expected) / expected}
    spec = json.loads((ROOT / contract["asset_directory"] / "model_spec.json").read_text())
    state = json.loads((args.output / "live_state.json").read_text())
    representatives = []
    for env in state["environments"]:
        poses = dict(zip(state["body_names"], np.asarray(env["body_link_pose_w"])))
        row = {"scene_group": env["scene_group"], "episode_step": env["episode_step"], "commands": env["commands"], "sides": {}}
        for side in ("L", "R"):
            a, b = poses[side * 3 + "_link1"][:3], poses[side * 3 + "_link2"][:3]
            binding = spec["spring_binding"][side + "_spring_slide"]
            knee = next(j for j in spec["joints"] if j["name"] == ("L_joint2" if side == "L" else "R_jonit2"))
            parent = Rotation.from_quat(poses[side + "_link1"][3:]).as_matrix()
            child = Rotation.from_quat(poses[side + "_link2"][3:]).as_matrix()
            rotation = np.asarray(knee["origin"])[:3, :3].T @ parent.T @ child
            raw_q = Rotation.from_matrix(rotation).as_rotvec() @ np.asarray(knee["axis"])
            angle = np.degrees(np.pi - 2.3573 + (1 if side == "L" else -1) * raw_q)
            row["sides"][side] = {"knee_inner_deg": float(angle),
                "compression_mm": float(1000 * (binding["full_extension_pin_distance_m"] - np.linalg.norm(b-a)))}
        representatives.append(row)
    summary = {"progress": progress, "record_crc_audit": record_audit,
               "scalar_windows": windows, "torque_sample_counter_audit": sample_counts,
               "representative_instantaneous_states_not_population_statistics": representatives,
               "all_scalar_tags": sorted(scalar_data)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
