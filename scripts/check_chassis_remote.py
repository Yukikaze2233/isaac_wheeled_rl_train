#!/usr/bin/env python3
"""Read an owned remote run receipt, resource state and its actual pose publisher."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


REMOTE = r'''
import json, pathlib, subprocess, time, os
p = pathlib.Path(RUN_ROOT)
result = {'remote_root': str(p), 'wall_time_unix': time.time()}
for name in ('archiver_status', 'archiver_error'):
    path = p / (name + '.json')
    if path.exists():
        result[name] = json.loads(path.read_text())
index = p / 'batch_delivery/index.json'
if index.exists():
    batches = json.loads(index.read_text())['batches']
    result['sealed_batches'] = len(batches)
    if batches:
        latest = max(batches, key=lambda b: b['created_at'])
        result['latest_sealed_batch'] = {k: latest[k] for k in ('unit', 'kind', 'created_at')}
if TMUX_SOCKET:
    result['tmux_session_alive'] = subprocess.run(
        ['tmux', '-S', TMUX_SOCKET, 'has-session', '-t', TMUX_SESSION], capture_output=True).returncode == 0
for name in ('progress', 'completion'):
    path = p / 'train' / (name + '.json')
    if path.exists():
        value = json.loads(path.read_text())
        if name == 'completion':
            value = {k: v for k, v in value.items() if k not in ('startup', 'source_sha256')}
        result[name] = value
torque = p / 'train/torque_monitor.json'
if torque.exists():
    result['torque_monitor'] = json.loads(torque.read_text())
    result['torque_report_age_s'] = time.time() - torque.stat().st_mtime
path = p / 'train/startup.json'
if path.exists():
    startup = json.loads(path.read_text())
    result['startup'] = {k: v for k, v in startup.get('startup', {}).items()
                         if k not in ('contact_filters', 'terrain_collision_paths', 'mass_kg_each', 'solver_order')}
path = p / 'train/live_state.json'
if path.exists():
    raw = path.read_bytes()
    state = json.loads(raw)
    result['live_state'] = state
    result['state_age_s'] = time.time() - state['wall_time_unix']
    import hashlib
    result['state_sha256'] = hashlib.sha256(raw).hexdigest()
log = p / 'train.log'
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 5000))
        result['recent_log'] = stream.read().decode(errors='replace').splitlines()[-35:]
memory = dict((a[0], int(a[1])) for a in [line.split() for line in pathlib.Path('/proc/meminfo').read_text().splitlines()] if len(a)>1)
result['mem_available_kib'] = memory['MemAvailable:']
result['gpu'] = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
for label, key in (('supervisor_alive', 'pid'), ('worker_alive', 'worker_pid')):
    pid = result.get('progress', {}).get(key)
    if pid:
        cmd = pathlib.Path('/proc') / str(pid) / 'cmdline'
        result[label] = cmd.exists() and str(p) in cmd.read_bytes().decode(errors='replace')
old = pathlib.Path('/proc/10560/cmdline')
result['round4_alive'] = old.exists() and 'train_v40.py' in old.read_bytes().decode(errors='replace')
print(json.dumps(result, allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_receipt", type=Path)
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports/v5_remote_checks")
    args = parser.parse_args()
    plan = json.loads(args.launch_receipt.read_text())
    command = ["ssh", "-S", plan["control_path"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-p", str(plan["ssh_port"]), plan["host"], "python3 -B -"]
    if plan.get("identity_file"):
        command[1:1] = ["-o", "IdentitiesOnly=yes", "-i", plan["identity_file"]]
    script = ("RUN_ROOT = " + repr(plan["remote_root"]) + "\nTMUX_SOCKET = " + repr(plan.get("tmux_socket"))
              + "\nTMUX_SESSION = " + repr(plan.get("tmux")) + "\n" + REMOTE)
    response = subprocess.run(command, input=script, capture_output=True, text=True, timeout=45, check=True)
    result = json.loads(response.stdout)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / ("check-" + stamp + ".json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    summary = {k: v for k, v in result.items() if k != "live_state"}
    if args.brief:
        if "completion" in result:
            result["completion"] = {k: v for k, v in result["completion"].items()
                                    if k in ("status", "successful_updates", "blocked_stage", "stage_status", "error")}
        print(json.dumps({k: result[k] for k in ("remote_root", "progress", "supervisor_alive", "worker_alive", "tmux_session_alive",
            "completion", "archiver_status", "sealed_batches", "latest_sealed_batch", "state_age_s", "mem_available_kib", "gpu")
            if k in result}, indent=2, ensure_ascii=False))
        return
    if "completion" in summary:
        metrics = summary["completion"].get("metrics", {})
        for key in ("terrain_families", "final_relative_height_m", "final_filtered_wheel_force_n"):
            metrics.pop(key, None)
    if "startup" in summary:
        families = summary["startup"].pop("terrain_families", [])
        summary["startup"]["terrain_counts"] = {name: families.count(name) for name in set(families)}
    if "live_state" in result:
        state = result["live_state"]
        summary["live_state_summary"] = {k: v for k, v in state.items() if k != "body_link_pose_w"}
        summary["live_state_summary"]["body_count"] = len(state["body_link_pose_w"])
        if "environments" in summary["live_state_summary"]:
            summary["live_state_summary"]["environments"] = [
                {k: e[k] for k in ("env_index", "scene_group", "terrain", "episode_step")}
                for e in state["environments"]]
    if "torque_monitor" in summary:
        full = summary.pop("torque_monitor")
        summary["torque_summary"] = {name: {
            "rms_legs_nm": [g["rms_motor_torque_nm"][i] for i in (0, 1, 3, 4)],
            "peak_legs_nm": [g["peak_motor_torque_nm"][i] for i in (0, 1, 3, 4)],
            "peak_wheels_nm": [g["peak_motor_torque_nm"][i] for i in (2, 5)],
            "max_saturation_fraction": max(g["saturation_fraction_95pct"]),
            "peak_gas_force_n": g["peak_gas_force_n"],
            "max_compression_m": g["gas_compression_max_m"],
        } for name, g in full["groups"].items()}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
