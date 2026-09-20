#!/usr/bin/env python3
"""Deploy the complete single-policy curriculum, gated by each stage's real evaluation."""
from __future__ import annotations

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

from run_chassis_blocks import TrainingBlocks, ROOT
from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, stage_contract


class FullCurriculum(TrainingBlocks):
    def __init__(self, args):
        super().__init__(args)
        self.completed_updates = 0
        self.stage_name = None
        self.report.update(stages=[], training_plan_sha256=hashlib.sha256(args.contract.read_bytes()).hexdigest())

    def publish(self, directory):
        if not super().publish(directory):
            return
        path = self.root / "progress.json"
        if path.exists():
            progress = json.loads(path.read_text())
            current = progress["successful_updates"]
            self.report["successful_updates"] = self.completed_updates + current
            progress.update(successful_updates=self.report["successful_updates"], stage_successful_updates=current,
                            active_stage=self.stage_name, pid=os.getpid(), orchestration="complete_gated_curriculum")
            progress["stage_training_transitions"] = progress["training_transitions"]
            progress["training_transitions"] += self.completed_updates * self.args.num_envs * self.steps_per_env
            temporary = self.root / "progress.full.tmp"
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(path)

    def write_selection(self):
        accepted = self.root / "accepted_policy.pt"
        onnx = self.root / "accepted_policy.onnx"
        selection = {
            "deployment_checkpoint": accepted.name if accepted.exists() else None,
            "deployment_onnx": onnx.name if onnx.exists() else None,
            "checkpoint_sha256": hashlib.sha256(accepted.read_bytes()).hexdigest() if accepted.exists() else None,
            "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest() if onnx.exists() else None,
            "accepted_stage": next((s["name"] for s in reversed(self.report["stages"]) if s["status"] == "stage_accepted"), None),
            "latest_is_not_necessarily_accepted": True, "status": self.report["status"],
        }
        temporary = self.root / "artifact_selection.tmp"
        temporary.write_text(json.dumps(selection, indent=2) + "\n")
        temporary.replace(self.root / "artifact_selection.json")

    def run(self):
        self.root.mkdir(parents=True, exist_ok=False)
        self.copy_atomic(self.args.contract, "curriculum_plan.json")
        self.copy_atomic(Path(__file__), "full_orchestrator_source.py")
        base = json.loads((ROOT / self.contract["base_contract"]).read_text())
        self.steps_per_env = base["num_steps_per_env"]
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        checkpoint = self.args.transfer
        names = [recipe["name"] for recipe in self.contract["stages"]]
        start_stage = getattr(self.args, "start_stage", None)
        start_index = names.index(start_stage) if start_stage else 0
        self.report["start_stage"] = names[start_index]
        self.report["preceding_stages_not_retrained"] = names[:start_index]
        config_dir = self.root / "stage_contracts"
        config_dir.mkdir()
        try:
            self.report["status"] = "running"
            for index, recipe in enumerate(self.contract["stages"]):
                if index < start_index:
                    continue
                if self.stop_requested or time.monotonic() >= self.deadline:
                    self.report["status"] = "stopped"
                    break
                self.stage_name = recipe["name"]
                config = stage_contract(base, self.contract, recipe, self.args.num_envs)
                if self.args.updates is not None and self.completed_updates >= self.args.updates:
                    self.report.update(status="budget_exhausted_gate_pending", blocked_stage=recipe["name"])
                    break
                stage_updates = config["total_updates"]
                if self.args.updates is not None:
                    stage_updates = min(stage_updates, self.args.updates - self.completed_updates)
                contract_path = config_dir / (self.stage_name + ".json")
                contract_path.write_text(json.dumps(config, indent=2) + "\n")
                self.copy_atomic(contract_path, "contract.json")
                directory = self.root / f"stage_{index:02d}_{self.stage_name}"
                self.args.stage = recipe["kind"]
                command = [sys.executable, "-B", str(ROOT / "scripts/run_chassis_blocks.py"),
                    "--contract", str(contract_path.resolve()), "--stage", recipe["kind"], "--research",
                    "--num-envs", str(self.args.num_envs), "--updates", str(stage_updates),
                    "--seed", str(self.args.seed), "--device", self.args.device, "--publish-state",
                    "--max-runtime-seconds", str(max(1., self.deadline - time.monotonic())), "--run-dir", str(directory)]
                if checkpoint is not None:
                    command += ["--transfer", str(Path(checkpoint).resolve())]
                print("V5_FULL_STAGE_START", self.stage_name, flush=True)
                code = self.execute(command, self.root / f"stage_{index:02d}_{self.stage_name}.log", directory)
                result_path = directory / "completion.json"
                if not result_path.exists():
                    if self.stop_requested:
                        self.report["status"] = "stopped"
                        break
                    raise RuntimeError(f"Stage {self.stage_name} exited {code} without completion")
                result = json.loads(result_path.read_text())
                stage_result = {"name": self.stage_name, "status": result["status"],
                    "successful_updates": result["successful_updates"], "directory": directory.name,
                    "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                    "accepted_checkpoint": result.get("accepted_checkpoint")}
                self.completed_updates += result["successful_updates"]
                self.report["successful_updates"] = self.completed_updates
                self.report["stages"].append(stage_result)
                if code != 0:
                    raise RuntimeError(f"Stage {self.stage_name} failed with exit code {code}: {result['status']}")
                for name in ("model_final.pt", "model_final.contract.json", "policy.onnx", "policy.onnx.json", "policy.onnx.contract.json",
                             "latest_evaluation.json", "best_passing_evaluation.json", "baseline_actor.pt"):
                    if (directory / name).exists():
                        self.copy_atomic(directory / name, name)
                if result.get("export"):
                    self.report["export"] = result["export"]
                if result["status"] != "stage_accepted":
                    self.report.update(status="stopped" if self.stop_requested else "stage_gate_pending",
                                       blocked_stage=self.stage_name, stage_status=result["status"])
                    break
                checkpoint = Path(result["accepted_checkpoint"])
                self.copy_atomic(checkpoint, "accepted_policy.pt")
                self.copy_atomic(checkpoint_contract_path(checkpoint), "accepted_policy.contract.json")
                accepted_onnx = directory / ("policy_best.onnx" if (directory / "policy_best.onnx").exists() else "policy.onnx")
                if accepted_onnx.exists():
                    self.copy_atomic(accepted_onnx, "accepted_policy.onnx")
                    self.copy_atomic(Path(str(accepted_onnx) + ".json"), "accepted_policy.onnx.json")
                    self.copy_atomic(contract_path, "accepted_policy.onnx.contract.json")
                self.report["accepted_checkpoint"] = str((self.root / "accepted_policy.pt").resolve())
                self.report["accepted_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                self.write_selection()
                (self.root / "curriculum.json").write_text(json.dumps(self.report, indent=2) + "\n")
                print("V5_FULL_STAGE_ACCEPTED", json.dumps(stage_result), flush=True)
            else:
                self.report["status"] = "full_curriculum_accepted"
        except Exception:
            self.report.update(status="failed", error=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.write_selection()
            (self.root / "completion.json").write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        return 1 if self.report["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--max-runtime-seconds", type=float, default=259200.)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--publish-state", action="store_true")
    parser.add_argument("--stage", default="curriculum")
    parser.add_argument("--updates", type=int, help="Maximum new PPO updates across all stages")
    parser.add_argument("--start-stage", help="Continue at this stage; predecessor cases remain in regression evaluation")
    parser.add_argument("--prepare-only", action="store_true", help="Materialize all stage contracts without starting simulation")
    args = parser.parse_args()
    if args.start_stage:
        names = [s["name"] for s in json.loads(args.contract.read_text())["stages"]]
        if args.start_stage not in names or args.transfer is None:
            parser.error("--start-stage requires a known stage and an explicit transfer checkpoint")
    if not args.research or not (32 <= args.num_envs <= 4096 and args.max_runtime_seconds > 0):
        parser.error("Explicit research mode, at least 32 environments and a positive budget are required")
    if args.prepare_only:
        plan = json.loads(args.contract.read_text())
        base = json.loads((ROOT / plan["base_contract"]).read_text())
        args.run_dir.mkdir(parents=True, exist_ok=False)
        contracts = []
        for recipe in plan["stages"]:
            config = stage_contract(base, plan, recipe, args.num_envs)
            path = args.run_dir / (recipe["name"] + ".json")
            path.write_text(json.dumps(config, indent=2) + "\n")
            contracts.append({"stage": recipe["name"], "kind": recipe["kind"], "contract": str(path),
                              "updates": config["total_updates"], "evaluation_cases": len(config["evaluation"]["cases"])})
        print(json.dumps(contracts, indent=2))
        return 0
    return FullCurriculum(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
