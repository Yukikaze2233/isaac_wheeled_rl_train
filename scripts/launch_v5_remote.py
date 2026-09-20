#!/usr/bin/env python3
"""Freeze a committed V5 foundation run and start its own bounded Kaiser tmux job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.full_curriculum import resolve_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--contract", default="contracts/v5_foundation_v1.json")
    parser.add_argument("--stage", choices=("foundation", "mixed", "curriculum"), default="foundation")
    parser.add_argument("--transfer", help="Absolute remote V5 checkpoint path for weights-only scene transfer")
    parser.add_argument("--start-stage", help="Continue a full curriculum from a named stage")
    parser.add_argument("--num-envs", type=int, choices=(32, 64, 128, 256, 512, 1024, 2048, 4096), default=256)
    parser.add_argument("--updates", type=int, help="Override for a labeled bounded engineering probe")
    parser.add_argument("--max-runtime-seconds", type=int, default=172800)
    parser.add_argument("--host", default="kaiser@192.168.64.234")
    parser.add_argument("--ssh-port", type=int, default=2222)
    parser.add_argument("--control-path", default="/tmp/opencode/kaiser-model-budget-control")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", args.commit + "^{commit}"], cwd=ROOT, text=True).strip()
    loader = lambda name: json.loads(subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT))
    contract = resolve_plan(loader(args.contract), loader)
    full = contract["contract_id"].startswith("v5-complete-curriculum-plan-")
    recipes = contract["stages"]
    if args.start_stage:
        names = [s["name"] for s in recipes]
        if not full or args.start_stage not in names or not args.transfer:
            parser.error("--start-stage requires a full curriculum, known stage and transfer checkpoint")
        recipes = recipes[names.index(args.start_stage):]
    if full:
        args.stage = "curriculum"
        base_contract = json.loads(subprocess.check_output(["git", "show", commit + ":" + contract["base_contract"]], cwd=ROOT))
        steps = contract.get("num_steps_per_env", base_contract["num_steps_per_env"])
        budget = sum(s["updates"] for s in recipes)
        target_transitions = budget * contract["target_num_envs"] * steps
        updates = args.updates if args.updates is not None else sum(math.ceil(s["updates"] * contract["target_num_envs"] / args.num_envs) for s in recipes)
    else:
        steps = contract["num_steps_per_env"]
        budget = next(s["updates"] for s in contract["stages"] if s["name"] == args.stage)
        target_transitions = budget * contract["target_num_envs"] * steps
        updates = args.updates if args.updates is not None else target_transitions // (args.num_envs * steps)
    if updates < 1 or args.max_runtime_seconds < 1:
        parser.error("Positive updates and runtime required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "v5-scut35" if contract.get("actor_observation_source") else "v5-scut-v4" if contract["contract_id"].endswith("plan-v4") else "v5-scut-v3" if contract["contract_id"].endswith("plan-v3") else "v5-full-v2" if full else "v5-locomotion-v2" if contract["contract_id"] == "v5-gas-spring-locomotion-research-v2" else f"v5-{args.stage}"
    name = f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"
    base = "/home/kaiser/robot-rl-sim60"
    remote = base + "/experiments/" + name
    source = remote + "/isaac_wheeled_rl_train"
    session = name
    entry = "run_full_chassis.py" if full else "run_chassis_blocks.py" if contract["contract_id"] == "v5-gas-spring-locomotion-research-v2" else "train_chassis.py"
    command = [base + "/env/bin/python", "-B", source + "/scripts/" + entry,
        "--contract", source + "/" + args.contract, "--research", "--stage", args.stage,
        "--device", "cuda:0", "--num-envs", str(args.num_envs), "--updates", str(updates),
        "--seed", "617", "--publish-state", "--max-runtime-seconds", str(args.max_runtime_seconds),
        "--run-dir", remote + "/train"]
    if args.transfer:
        command += ["--transfer", args.transfer]
    if args.start_stage:
        command += ["--start-stage", args.start_stage]
    plan = {"commit": commit, "remote_root": remote, "source_directory": source, "tmux": session,
            "command": command, "num_envs": args.num_envs, "updates": updates,
            "training_transitions": args.num_envs * steps * updates,
            "full_foundation_target_transitions": target_transitions,
            "scope": "engineering_probe" if args.updates is not None else "formal_" + args.stage,
            "initialization": "weights_transfer" if args.transfer else "scratch", "state_publisher_hz_max": 4,
            "host": args.host, "ssh_port": args.ssh_port, "control_path": args.control_path,
             "execute": args.execute, "start_stage": args.start_stage}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    ssh = ["ssh", "-S", args.control_path, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", str(args.ssh_port), args.host]
    # Check current WSL headroom before staging a second independent process.
    resource_code = (
        "import json,pathlib; m=dict((p[0],int(p[1])) for p in "
        "[l.split() for l in pathlib.Path('/proc/meminfo').read_text().splitlines()] if len(p)>1); "
        "print(json.dumps({'mem_available_kib':m['MemAvailable:']}))"
    )
    resource = json.loads(subprocess.check_output(ssh + ["python3 -c " + shlex.quote(resource_code)], text=True))
    (args.output / "resource_before.json").write_text(json.dumps(resource, indent=2))
    if resource["mem_available_kib"] < 4 * 1024**2:
        raise RuntimeError("Insufficient WSL available RAM for an independent V5 launch")
    archive = args.output / "source.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(archive.resolve()), commit], cwd=ROOT, check=True)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(ssh + ["test -d " + shlex.quote(base + "/experiments") + " && test ! -e " + shlex.quote(remote)
                         + " && mkdir -p " + shlex.quote(source)], check=True)
    subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ControlPath=" + args.control_path, "-P", str(args.ssh_port),
                    str(archive), args.host + ":" + remote + "/source.tar.gz"], check=True)
    unpack = ("test \"$(sha256sum " + shlex.quote(remote + "/source.tar.gz") + " | cut -d ' ' -f 1)\" = "
              + shlex.quote(archive_sha) + " && tar -xzf " + shlex.quote(remote + "/source.tar.gz") + " -C " + shlex.quote(source))
    subprocess.run(ssh + [unpack], check=True)
    inner = ("source " + shlex.quote(base + "/bin/sim60-runtime.sh")
             + " && export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1"
             + " && timeout --signal=TERM --kill-after=120s " + str(args.max_runtime_seconds + 1200) + "s "
             + shlex.join([base + "/env/bin/python", "-B", source + "/scripts/chassis_remote_job.py",
                           "--run-root", remote, "--", *command])
             + " > " + shlex.quote(remote + "/job.log") + " 2>&1")
    launch = "tmux new-session -d -s " + shlex.quote(session) + " bash -lc " + shlex.quote(inner)
    subprocess.run(ssh + [launch], check=True)
    receipt = {**plan, "archive_sha256": archive_sha, "launched_at": datetime.now(timezone.utc).isoformat(),
               "status": "tmux_started_training_progress_must_be_checked"}
    (args.output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
