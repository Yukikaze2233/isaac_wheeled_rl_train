#!/usr/bin/env python3
"""Own sequential PPO/evaluation blocks and preserve behavior-accepted checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


class TrainingBlocks:
    def __init__(self, args):
        self.args = args
        self.contract = json.loads(args.contract.read_text())
        self.root = args.run_dir
        self.child = None
        self.stop_requested = False
        self.deadline = time.monotonic() + args.max_runtime_seconds
        self.report = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
            "successful_updates": 0, "blocks": [], "consecutive_evaluation_passes": 0,
            "contract_id": self.contract["contract_id"], "initial_checkpoint": str(args.transfer) if args.transfer else None}

    def request_stop(self, signum, _frame):
        self.stop_requested = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signum)

    def copy_atomic(self, source, name):
        temporary = self.root / (name + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(self.root / name)

    def publish(self, directory):
        path = directory / "progress.json"
        if path.exists():
            progress = json.loads(path.read_text())
            self.report["successful_updates"] = progress["successful_updates"] + progress["parent_updates"]
            progress.update(pid=os.getpid(), worker_pid=progress["pid"], phase="training",
                            successful_updates=self.report["successful_updates"], parent_updates=0,
                            orchestration="train_then_fixed_evaluate", active_block=directory.name)
            temporary = self.root / "progress.tmp"
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(self.root / "progress.json")
        for name in ("torque_monitor.json", "behavior_metrics.json", "live_state.json", "startup.json"):
            if (directory / name).exists():
                self.copy_atomic(directory / name, name)

    def execute(self, command, log_path, training_directory=None):
        with log_path.open("x") as stream:
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            progress_path = self.root / "progress.json"
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
                "successful_updates": 0, "parent_updates": 0, "training_transitions": 0,
                "num_envs": self.args.num_envs, "stage": self.args.stage}
            progress.update(pid=os.getpid(), worker_pid=self.child.pid,
                phase="training" if training_directory is not None else "fixed_evaluation",
                updated_at=datetime.now(timezone.utc).isoformat())
            progress_path.write_text(json.dumps(progress, indent=2) + "\n")
            while self.child.poll() is None:
                if training_directory is not None:
                    self.publish(training_directory)
                if time.monotonic() >= self.deadline and not self.stop_requested:
                    self.request_stop(signal.SIGTERM, None)
                time.sleep(1.)
            code = self.child.returncode
            self.child = None
        if training_directory is not None:
            self.publish(training_directory)
        return code

    def run(self):
        self.root.mkdir(parents=True, exist_ok=False)
        self.copy_atomic(self.args.contract, "contract.json")
        self.copy_atomic(Path(__file__), "orchestrator_source.py")
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        parent = None
        best_rank = None
        best_passing_rank = None
        regressions = 0
        settings = self.contract["evaluation"]
        try:
            self.report["status"] = "running"
            had_passing_baseline = False
            if self.args.transfer:
                baseline_dir = self.root / "baseline_evaluation"
                command = [sys.executable, "-B", str(ROOT / "scripts/evaluate_chassis.py"),
                    "--contract", str(self.args.contract.resolve()), "--checkpoint", str(self.args.transfer.resolve()),
                    "--device", self.args.device, "--output", str(baseline_dir)]
                code = self.execute(command, self.root / "baseline_evaluation.log")
                if self.stop_requested:
                    self.report["status"] = "stopped"
                    return 0
                if code != 0:
                    raise RuntimeError("Initial actor evaluation failed to execute")
                baseline = json.loads((baseline_dir / "evaluation.json").read_text())["candidates"][0]
                self.report["baseline_evaluation"] = baseline
                best_rank = baseline["rank_lower_is_better"]
                had_passing_baseline = baseline["passed"]
                self.copy_atomic(self.args.transfer, "baseline_actor.pt")
                self.copy_atomic(baseline_dir / "evaluation.json", "baseline_evaluation.json")
                print("V5_BASELINE_EVALUATED", json.dumps({"passed": baseline["passed"], "rank": best_rank}), flush=True)
            while self.report["successful_updates"] < self.args.updates and not self.stop_requested:
                index = len(self.report["blocks"])
                directory = self.root / f"block_{index:03d}"
                updates = min(settings["block_updates"], self.args.updates - self.report["successful_updates"])
                command = [sys.executable, "-B", str(ROOT / "scripts/train_chassis.py"),
                    "--contract", str(self.args.contract.resolve()), "--stage", self.args.stage,
                    "--research", "--num-envs", str(self.args.num_envs), "--updates", str(updates),
                    "--seed", str(self.args.seed + index), "--device", self.args.device,
                    "--max-runtime-seconds", str(max(1., self.deadline - time.monotonic())),
                    "--run-dir", str(directory)]
                if self.args.publish_state:
                    command.append("--publish-state")
                if parent:
                    command += ["--resume", str(parent)]
                elif self.args.transfer:
                    command += ["--transfer", str(self.args.transfer.resolve()), "--transfer-actor-only"]
                code = self.execute(command, self.root / f"block_{index:03d}.log", directory)
                completion_path = directory / "completion.json"
                if not completion_path.exists():
                    raise RuntimeError(f"Training block exited {code} without completion")
                completion = json.loads(completion_path.read_text())
                self.report["successful_updates"] = completion["parent_updates"] + completion["successful_updates"]
                if completion["status"] not in ("completed", "stopped"):
                    raise RuntimeError(f"Training block status: {completion['status']}")
                parent = directory / "model_final.pt"
                if not parent.exists():
                    self.report["status"] = "stopped"
                    break
                for name in ("model_final.pt", "policy.onnx", "policy.onnx.json", "agent_config.json"):
                    self.copy_atomic(directory / name, name)
                self.report["export"] = completion["export"]
                self.report["checkpoint_sha256"] = completion["checkpoint_sha256"]
                block = {"directory": directory.name, "successful_updates_total": self.report["successful_updates"],
                         "checkpoint_sha256": completion["checkpoint_sha256"], "training_status": completion["status"]}
                self.report["blocks"].append(block)
                if self.stop_requested or completion["status"] == "stopped":
                    self.report["status"] = "stopped"
                    break
                evaluation_dir = self.root / f"evaluation_{index:03d}"
                command = [sys.executable, "-B", str(ROOT / "scripts/evaluate_chassis.py"),
                    "--contract", str(self.args.contract.resolve()), "--checkpoint", str(parent),
                    "--device", self.args.device, "--output", str(evaluation_dir)]
                code = self.execute(command, self.root / f"evaluation_{index:03d}.log")
                if self.stop_requested:
                    self.report["status"] = "stopped"
                    break
                if code != 0:
                    raise RuntimeError(f"Fixed evaluation process failed with {code}")
                evaluation = json.loads((evaluation_dir / "evaluation.json").read_text())
                candidate = evaluation["candidates"][0]
                block["evaluation_passed"] = candidate["passed"]
                block["evaluation_rank"] = candidate["rank_lower_is_better"]
                self.copy_atomic(evaluation_dir / "evaluation.json", "latest_evaluation.json")
                if best_rank is None or candidate["rank_lower_is_better"] < best_rank:
                    best_rank = candidate["rank_lower_is_better"]
                    self.copy_atomic(parent, "best_candidate.pt")
                    self.copy_atomic(evaluation_dir / "evaluation.json", "best_candidate_evaluation.json")
                if candidate["passed"]:
                    if best_passing_rank is None or candidate["rank_lower_is_better"] < best_passing_rank:
                        best_passing_rank = candidate["rank_lower_is_better"]
                        self.copy_atomic(parent, "model_best.pt")
                        self.copy_atomic(directory / "policy.onnx", "policy_best.onnx")
                        self.copy_atomic(evaluation_dir / "evaluation.json", "best_passing_evaluation.json")
                    self.report["consecutive_evaluation_passes"] += 1
                    regressions = 0
                else:
                    self.report["consecutive_evaluation_passes"] = 0
                    if had_passing_baseline or (self.root / "model_best.pt").exists():
                        regressions += 1
                print("V5_BLOCK_EVALUATED", json.dumps(block), flush=True)
                (self.root / "curriculum.json").write_text(json.dumps(self.report, indent=2) + "\n")
                if self.report["consecutive_evaluation_passes"] >= settings["consecutive_passes_required"]:
                    self.report["status"] = "foundation_accepted"
                    break
                if regressions >= 3:
                    self.report["status"] = "regression_hold_best_preserved"
                    break
            if self.report["status"] == "running":
                self.report["status"] = "stopped" if self.stop_requested else "budget_exhausted_gate_pending"
        except Exception:
            self.report.update(status="failed", error=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
            (self.root / "completion.json").write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        return 1 if self.report["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage", choices=("foundation",), default="foundation")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--publish-state", action="store_true")
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--max-runtime-seconds", type=float, default=86400.)
    args = parser.parse_args()
    if not args.research or not (1 <= args.num_envs <= 4096 and 1 <= args.updates <= 100000 and args.max_runtime_seconds > 0):
        parser.error("Explicit research flag and bounded positive settings required")
    return TrainingBlocks(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
