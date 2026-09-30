#!/usr/bin/env python3
"""Observe sealed S1 snapshots without changing training, protection or promotion."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count
from wheeled_tasks.chassis.remediation import digest, validate_evaluation
from run_chassis_remediation import write_json

UPDATES = (860, 940)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def evaluation_lock(train_root, *, blocking=False):
    """Shared with the final evaluator; an observation never delays PPO."""
    with (Path(train_root) / ".s1-evaluation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def resource_snapshot():
    """Unknown capacity fails closed; these are headroom gates, not a capacity proof."""
    binary = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        result = subprocess.run([binary, "--id=0", "--query-gpu=memory.free",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                text=True, timeout=5)
        available = next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        return {"gpu_free_mib": float(result.stdout.strip()), "host_available_mib": int(available) / 1024}
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as error:
        return {"error": str(error)}


def sufficient_resources(resources, *, running=False):
    gpu, host = (1024, 1024) if running else (7 * 1024, 4 * 1024)
    return resources.get("gpu_free_mib", -1) >= gpu and resources.get("host_available_mib", -1) >= host


def worker_state(directory):
    if (directory / "completion.json").exists():
        return "finished"
    progress = directory / "progress.json"
    if not progress.exists():
        return "waiting"
    data = json.loads(progress.read_text())
    try:
        argv = Path(f"/proc/{int(data['pid'])}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return "finished"
    if not any(arg.endswith(b"/train_chassis.py") for arg in argv) or str(directory).encode() not in argv:
        return "finished"  # A stale or recycled PID is never treated as a training worker.
    return "running"


def authenticate_checkpoint(checkpoint, update, contract_sha):
    if not checkpoint.with_suffix(".checkpoint.json").is_file():
        raise ValueError("Observation requires a per-file sealed checkpoint")
    if checkpoint_update_count(checkpoint) != update or digest(checkpoint.with_suffix(".contract.json")) != contract_sha:
        raise ValueError("Snapshot update or frozen contract differs")
    return digest(checkpoint)


def authenticate_result(result, names, seed, contract_sha, checkpoint_sha, source_sha):
    candidate = validate_evaluation(result, names)
    if (result.get("contract_sha256") != contract_sha or result["settings"]["seed"] != seed
            or candidate.get("checkpoint_sha256") != checkpoint_sha or result["episodes_per_case"] != 4
            or result.get("script_sha256") != source_sha):
        raise ValueError("Observer evaluation provenance differs")
    sources = result.get("source_sha256", {})
    if not {"scripts/evaluate_chassis.py", "src/wheeled_tasks/chassis/env.py"} <= sources.keys():
        raise ValueError("Observer evaluation source manifest is incomplete")
    for name, sha in sources.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != sha:
            raise ValueError(f"Observer evaluation source differs: {name}")
    return candidate


def log_evaluation(directory, candidate, seed, update):
    # Same native TensorBoard writer and Evaluation tags as the final evaluator.
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(directory)) as writer:
        for name, case in candidate["cases"].items():
            for key, value in case.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    writer.add_scalar(f"Evaluation/seed_{seed}/{name}/{key}", value, update)


class Observer:
    def __init__(self, args):
        self.train_root = args.train_root.resolve()
        self.plan = json.loads(args.plan.read_text())
        recipe, = self.plan["stages"]
        if (self.plan["contract_id"] != "v6-s1-persistent-recovery-v1"
                or recipe["name"] != "A_persistent_recovery"
                or self.plan["source_consumed_updates"] != 778 or recipe["updates"] != 222):
            raise ValueError("Observer requires the frozen persistent S1 plan")
        self.contract = ROOT / recipe["contract"]
        self.contract_sha = recipe["sha256"]
        if digest(self.contract) != self.contract_sha:
            raise ValueError("Observer worker contract hash mismatch")
        config = json.loads(self.contract.read_text())
        self.names, self.seed = self.plan["evaluation_case_names"], self.plan["evaluation_seeds"][0]
        if (len(self.names) != 45 or len(set(self.names)) != 45
                or set(self.names) - {case["name"] for case in config["evaluation"]["cases"]}
                or self.seed != config["evaluation"]["seed"] or config["evaluation"]["episodes_per_case"] != 4):
            raise ValueError("Observer fixed suite/seed differs")
        self.worker = self.train_root / recipe["name"] / "train"
        self.root = self.train_root / recipe["name"] / "checkpoint_observations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_sha = digest(ROOT / "scripts/evaluate_chassis.py")
        self.deadline = time.monotonic() + args.max_runtime_seconds
        self.stop_requested, self.child = False, None
        self.report = {"status": "observing", "started_at": utc_now(), "plan_sha256": digest(args.plan),
                       "contract_sha256": self.contract_sha, "evaluation_seed": self.seed,
                       "scope": "single_seed_observation_only", "automatic_promotion": False,
                       "budget_or_protection_mutation": False, "observations": {}}

    def request_stop(self, *_):
        self.stop_requested = True

    def stop_child(self):
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.child.wait(timeout=5)
        self.child = None

    def stopping_reason(self):
        if self.stop_requested:
            return "observer_signal"
        if time.monotonic() >= self.deadline:
            return "observer_deadline"
        if worker_state(self.worker) == "finished":
            return "training_finished"
        return None

    def publish(self, update, receipt):
        self.report["observations"][str(update)] = receipt
        write_json(self.root / f"update_{update:08d}.json", receipt)
        write_json(self.root / "observer_summary.json", self.report)

    def evaluate(self, checkpoint, update, resources):
        checkpoint_sha = authenticate_checkpoint(checkpoint, update, self.contract_sha)
        output = self.root / f"evaluation_{update:08d}"
        receipt = {"status": "running", "started_at": utc_now(), "update": update,
                   "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                   "contract_sha256": self.contract_sha, "seed": self.seed, "resources_at_start": resources,
                   "coverage": self.names, "automatic_promotion": False}
        self.publish(update, receipt)
        command = [sys.executable, "-B", str(ROOT / "scripts/evaluate_chassis.py"),
                   "--contract", str(self.contract), "--checkpoint", str(checkpoint), "--output", str(output),
                   "--device", "cuda:0", "--seed", str(self.seed), "--role", "quick", "--no-traces",
                   "--cases", *self.names]
        try:
            with output.with_suffix(".log").open("x") as stream:
                self.child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                while self.child.poll() is None:
                    reason = self.stopping_reason()
                    current_resources = resource_snapshot()
                    if reason or not sufficient_resources(current_resources, running=True):
                        receipt.update(status="aborted", reason=reason or "low_or_unknown_runtime_headroom",
                                       resources_at_stop=current_resources)
                        return
                    time.sleep(2)
                if self.child.returncode:
                    raise RuntimeError(f"Observer evaluator exited {self.child.returncode}")
            result = json.loads((output / "evaluation.json").read_text())
            if authenticate_checkpoint(checkpoint, update, self.contract_sha) != checkpoint_sha:
                raise ValueError("Snapshot changed during observer evaluation")
            candidate = authenticate_result(result, self.names, self.seed, self.contract_sha,
                                            checkpoint_sha, self.source_sha)
            log_evaluation(self.root / "tensorboard", candidate, self.seed, update)
            receipt.update(status="evaluated", evaluation=str(output / "evaluation.json"),
                           evaluation_sha256=digest(output / "evaluation.json"),
                           passed_case_names=sorted(n for n, c in candidate["cases"].items() if c["passed"]),
                           source_sha256=result["source_sha256"])
        except Exception as error:
            receipt.update(status="failed", error=str(error))
        finally:
            self.stop_child()
            receipt["finished_at"] = utc_now()
            self.publish(update, receipt)

    def run(self):
        # Exclusive observer ownership survives restarts without duplicate evaluations.
        with (self.train_root / ".s1-observer.lock").open("a") as ownership:
            fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM, self.request_stop)
            signal.signal(signal.SIGINT, self.request_stop)
            try:
                for update in UPDATES:
                    previous = self.root / f"update_{update:08d}.json"
                    if previous.exists() and json.loads(previous.read_text())["status"] in ("evaluated", "failed", "aborted", "skipped"):
                        receipt = json.loads(previous.read_text())
                        if receipt["status"] == "evaluated":
                            result_path = Path(receipt["evaluation"])
                            if (digest(result_path) != receipt["evaluation_sha256"]
                                    or authenticate_checkpoint(Path(receipt["checkpoint"]), update,
                                                               self.contract_sha) != receipt["checkpoint_sha256"]):
                                raise ValueError("Previous observer evaluation changed")
                            authenticate_result(json.loads(result_path.read_text()), self.names, self.seed,
                                                self.contract_sha, receipt["checkpoint_sha256"], self.source_sha)
                        self.report["observations"][str(update)] = receipt
                        continue
                    if (self.root / f"evaluation_{update:08d}.log").exists():
                        self.publish(update, {"status": "skipped", "reason": "incomplete_previous_attempt"})
                        continue
                    checkpoint = self.worker / "checkpoints" / f"update_{update:08d}" / "model.pt"
                    while True:
                        reason = self.stopping_reason()
                        if reason:
                            self.publish(update, {"status": "skipped", "reason": reason})
                            break
                        if checkpoint.is_file() and worker_state(self.worker) == "running":
                            with evaluation_lock(self.train_root) as acquired:
                                resources = resource_snapshot()
                                if acquired and sufficient_resources(resources):
                                    self.evaluate(checkpoint, update, resources)
                                    break
                                self.publish(update, {"status": "deferred", "resources": resources,
                                                      "reason": "resource_headroom" if acquired else "evaluator_busy"})
                        time.sleep(5)
            except Exception as error:
                self.report.update(status="failed", error=str(error))
                raise
            finally:
                self.stop_child()
                self.report.update(status="failed" if self.report.get("status") == "failed" else "finished",
                                   finished_at=utc_now())
                write_json(self.root / "observer_summary.json", self.report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--max-runtime-seconds", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.max_runtime_seconds) or args.max_runtime_seconds <= 0:
        parser.error("max-runtime-seconds must be finite and positive")
    Observer(args).run()


if __name__ == "__main__":
    main()
