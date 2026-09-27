#!/usr/bin/env python3
"""Run paired, bounded P0 diagnostics without advancing the formal curriculum."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from run_chassis_blocks import ROOT, TrainingBlocks
from wheeled_tasks.chassis.diagnostic_curriculum import (
    diagnostic_contract, height_scan_contract, height_response_fit, validate_diagnostic_plan,
)
from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DiagnosticStudy(TrainingBlocks):
    def __init__(self, args):
        super().__init__(args)
        validate_diagnostic_plan(self.contract)
        self.plan = self.contract
        self.completed_updates = 0
        self.phase = "preflight"
        self.report.update(scope=self.plan["scope"], arms={}, evaluations={}, source_formal_updates=750,
                           automatic_formal_promotion=False, planned_new_updates=200)

    def publish(self, directory):
        source = directory / "progress.json"
        for name in ("torque_monitor.json", "behavior_metrics.json", "live_state.json", "startup.json"):
            if (directory / name).exists():
                self.copy_atomic(directory / name, name)
        if not source.exists():
            return False
        progress = json.loads(source.read_text())
        current = progress["successful_updates"] + progress["parent_updates"]
        self.report["successful_updates"] = self.completed_updates + current
        progress.update(successful_updates=self.report["successful_updates"], arm_updates=current,
                        pid=os.getpid(), worker_pid=progress.get("worker_pid", progress["pid"]), parent_updates=0,
                        source_formal_updates=750, diagnostic_phase=self.phase,
                        scope=self.plan["scope"], orchestration="paired_diagnostic")
        progress["training_transitions"] = self.report["successful_updates"] * self.args.num_envs * 24
        temporary = self.root / "progress.study.tmp"
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(self.root / "progress.json")
        return True

    def run_child(self, command, output, *, training=False):
        if self.stop_requested or time.monotonic() >= self.deadline:
            raise InterruptedError("Diagnostic runtime or stop request reached")
        output.parent.mkdir(parents=True, exist_ok=True)
        progress = {"successful_updates": self.report["successful_updates"], "parent_updates": 0,
                    "source_formal_updates": 750, "num_envs": self.args.num_envs, "stage": "diagnostic",
                    "training_transitions": self.report["successful_updates"] * self.args.num_envs * 24,
                    "diagnostic_phase": self.phase, "scope": self.plan["scope"]}
        temporary = self.root / "progress.phase.tmp"
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(self.root / "progress.json")
        code = self.execute(command, output.with_suffix(".log"), output if training else None)
        if self.stop_requested:
            raise InterruptedError("Diagnostic stopped by request")
        receipt = output / ("completion.json" if training else "evaluation.json")
        if not receipt.is_file():
            raise RuntimeError(f"Diagnostic child exited{code} without {receipt.name}")
        result = json.loads(receipt.read_text())
        expected = "completed" if training else "evaluated"
        if code != 0 or result["status"] != expected:
            raise RuntimeError(f"Diagnostic child failed: {result.get('error', result['status'])}")
        return result

    def evaluate(self, contract, checkpoints, directory, seed, labels, step, *, scan):
        self.phase = str(directory.relative_to(self.root))
        command = [sys.executable, "-B", str(ROOT / "scripts/evaluate_chassis.py"),
                   "--contract", str(contract), "--output", str(directory), "--device", self.args.device,
                   "--seed", str(seed)]
        for checkpoint in checkpoints:
            command += ["--checkpoint", str(checkpoint)]
        if not scan:
            command += ["--cases", *self.plan["regression_case_names"], "--no-traces"]
        result = self.run_child(command, directory)
        from torch.utils.tensorboard import SummaryWriter
        for label, candidate in zip(labels, result["candidates"]):
            event_dir = self.root / "tensorboard" / label / ("height_scan" if scan else "regression")
            with SummaryWriter(str(event_dir)) as writer:
                for name, case in candidate["cases"].items():
                    for key, value in case.items():
                        if isinstance(value, (float, int)):
                            writer.add_scalar(f"Evaluation/seed_{seed}/{name}/{key}", value, step)
                fits = {kind: height_response_fit(candidate, kind) for kind in ("stand", "forward")} if scan else {}
                for kind, fit in fits.items():
                    for key, value in fit.items():
                        if isinstance(value, (float, int)):
                            writer.add_scalar(f"HeightResponse/seed_{seed}/{kind}/{key}", value, step)
            self.report["evaluations"].setdefault(label, {}).setdefault("scan" if scan else "regression", {})[str(seed)] = {
                "file": str((directory / "evaluation.json").relative_to(self.root)),
                "checkpoint_sha256": candidate["checkpoint_sha256"], "height_response": fits,
                "passed_cases": sum(c["passed"] for c in candidate["cases"].values()),
                "case_count": len(candidate["cases"])}
        (self.root / "diagnostic_summary.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def run(self):
        self.root.mkdir(parents=True, exist_ok=False)
        self.copy_atomic(self.args.contract, "diagnostic_plan.json")
        self.copy_atomic(Path(__file__), "diagnostic_orchestrator.py")
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        try:
            self.report["status"] = "running"
            source_path = checkpoint_contract_path(self.args.resume)
            if (digest(self.args.resume) != self.plan["source_checkpoint_sha256"]
                    or digest(source_path) != self.plan["source_contract_sha256"]
                    or checkpoint_update_count(self.args.resume) != self.plan["source_updates"]
                    or digest(self.args.baseline_checkpoint) != self.plan["baseline_checkpoint_sha256"]):
                raise ValueError("Diagnostic source or retained baseline identity mismatch")
            source = json.loads(source_path.read_text())
            directory = self.root / "contracts"
            directory.mkdir()
            paths = {}
            for recipe in self.plan["stages"]:
                name = recipe["name"]
                config = diagnostic_contract(source, self.plan, name, self.args.num_envs)
                path = directory / (name + ".json")
                path.write_text(json.dumps(config, indent=2) + "\n")
                scan_path = directory / (name + "_scan.json")
                scan_path.write_text(json.dumps(height_scan_contract(config), indent=2) + "\n")
                paths[name] = (path, scan_path)
            self.report["source_checkpoint_sha256"] = digest(self.args.resume)
            self.report["source_contract_sha256"] = digest(source_path)
            self.report["materialized_contracts"] = {p.name: digest(p) for p in directory.glob("*.json")}
            for seed in self.plan["evaluation_seeds"]:
                self.evaluate(paths["A_control"][1], [self.args.baseline_checkpoint, self.args.resume],
                    self.root / "baseline_scan" / f"seed_{seed}", seed,
                    ["retained100", "source750"], 0, scan=True)
                self.evaluate(paths["A_control"][0], [self.args.resume],
                    self.root / "baseline_regression" / f"seed_{seed}", seed, ["source750"], 0, scan=False)
            for recipe in self.plan["stages"]:
                name, updates = recipe["name"], recipe["updates"]
                path, scan_path = paths[name]
                output = self.root / name / "train"
                self.phase = name + "/training"
                command = [sys.executable, "-B", str(ROOT / "scripts/train_chassis.py"),
                    "--contract", str(path), "--stage", "foundation", "--research", "--publish-state",
                    "--num-envs", str(self.args.num_envs), "--updates", str(updates),
                    "--seed", str(self.plan["seed"]), "--device", self.args.device,
                    "--run-dir", str(output), "--resume", str(self.args.resume), "--resume-diagnostic",
                    "--max-runtime-seconds", str(max(1., self.deadline - time.monotonic()))]
                result = self.run_child(command, output, training=True)
                if result["successful_updates"] != updates or result["parent_updates"] != 0:
                    raise ValueError("Diagnostic arm did not satisfy its independent update ledger")
                self.completed_updates += result["successful_updates"]
                self.report["successful_updates"] = self.completed_updates
                self.report["arms"][name] = {k: result[k] for k in (
                    "successful_updates", "actor_updates_in_block", "learning_rate", "checkpoint_sha256", "diagnostic_origin")}
                for seed in self.plan["evaluation_seeds"]:
                    self.evaluate(scan_path, [output / "model_final.pt"], self.root / name / f"scan_{seed}",
                                  seed, [name], updates, scan=True)
                    self.evaluate(path, [output / "model_final.pt"], self.root / name / f"regression_{seed}",
                                  seed, [name], updates, scan=False)
            self.report["status"] = "diagnostic_completed"
        except InterruptedError as error:
            self.report.update(status="stopped", reason=str(error))
        except Exception:
            self.report.update(status="failed", error=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
            (self.root / "completion.json").write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
            (self.root / "artifact_selection.json").write_text(json.dumps({
                "status": self.report["status"], "deployment_checkpoint": None,
                "scope": self.plan["scope"], "automatic_formal_promotion": False}, indent=2) + "\n")
            self.publish_terminal()
        return int(self.report["status"] == "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--max-runtime-seconds", type=float, default=14400.)
    parser.add_argument("--research", action="store_true")
    parser.set_defaults(transfer=None, stage="diagnostic", worker_source=None, publish_state=True, updates=200)
    args = parser.parse_args()
    if not args.research or not 32 <= args.num_envs <= 16384 or args.max_runtime_seconds <= 0:
        parser.error("Explicit research mode and bounded resources are required")
    return DiagnosticStudy(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
