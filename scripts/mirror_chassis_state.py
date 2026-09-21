#!/usr/bin/env python3
"""One-way WSL training-state bridge to a Windows renderer's shared JSON file."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=7200.)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    default_source = args.run_root / "train/live_state.json"
    contracts = {}
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            progress = json.loads((args.run_root / "train/progress.json").read_text())
            source = default_source
            if progress.get("phase") == "training" and progress.get("worker_pid"):
                try:
                    command = (Path("/proc") / str(progress["worker_pid"]) / "cmdline").read_bytes().decode().split("\0")
                    if "--run-dir" in command:
                        directory = Path(command[command.index("--run-dir") + 1]).resolve()
                        if directory.is_relative_to((args.run_root / "train").resolve()):
                            candidate = directory / "live_state.json"
                            if candidate.exists():
                                source = candidate
                except (OSError, ValueError, IndexError):
                    pass
            if not source.exists():
                time.sleep(.25)
                continue
            state = json.loads(source.read_text())
            identity = state["contract_sha256"]
            if identity not in contracts:
                for path in (args.run_root / "train/stage_contracts").glob("*.json"):
                    raw = path.read_bytes()
                    contracts[hashlib.sha256(raw).hexdigest()] = json.loads(raw)
            contract = contracts.get(identity)
            if contract is None:
                time.sleep(.25)
                continue
            state["mirror"] = {"run_root": str(args.run_root), "phase": progress.get("phase"),
                "active_stage": progress.get("active_stage"), "updates": progress.get("successful_updates"),
                "num_envs": progress.get("num_envs"),
                "flat_floor_width_m": contract.get("flat_floor_width_m", 4.),
                "actor_dim": contract["actor_dim"], "policy_dt": contract["policy_dt"],
                "pose_source": str(source),
                "run_finished": (args.run_root / "train/completion.json").exists(),
                "bridge_time_unix": time.time(), "source_state_is_unmodified": True}
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(state) + "\n")
            temporary.replace(args.output)
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(.25)


if __name__ == "__main__":
    main()
