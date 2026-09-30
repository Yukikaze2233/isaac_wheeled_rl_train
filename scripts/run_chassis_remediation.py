#!/usr/bin/env python3
"""Run one authenticated S1 paired study against a shared, non-refundable ledger."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from run_chassis_blocks import ROOT, TrainingBlocks
from wheeled_tasks.chassis.remediation import (
    ARMS, REVISION, SOURCE_SHA, CHECKPOINT_SHA, digest, charged_updates,
    compare_arms, validate_evaluation, validate_remediation_resume,
)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_plan(plan):
    if (plan["contract_id"] != REVISION or plan["source_checkpoint_sha256"] != CHECKPOINT_SHA
            or plan["source_contract_sha256"] != SOURCE_SHA
            or plan["target_num_envs"] != 20000 or plan["seed"] != 290927
            or plan["source_consumed_updates"] != 618 or plan["source_actor_consumed_updates"] != 618
            or plan["source_stage_actor_update"] != 338 or plan["study_update_ceiling"] != 80
            or plan["stage_ceiling"] != 1000 or plan["total_ceiling"] != 8500
            or plan["rollback_refunds_consumed_updates"] is not False
            or plan["automatic_promotion"] is not False or plan["automatic_formal_resume"] is not False
            or plan["evaluation_seeds"] != [290929, 290930]
            or [r["name"] for r in plan["stages"]] != list(ARMS)
            or any(r["updates"] != 40 for r in plan["stages"])):
        raise ValueError("Study changes the frozen origin, budget, seeds or decision boundary")


def validate_contracts(plan):
    """Check train and all evaluation entry roles before allocating an Isaac scene."""
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
    from wheeled_tasks.chassis.preflight import design_preflight, ROLES
    validate_plan(plan)
    source = json.loads((ROOT / "reports/v6_s1_remediation_20260930/source/contract.json").read_text())
    receipts = []
    for recipe in plan["stages"]:
        path = ROOT / recipe["contract"]
        if digest(path) != recipe["sha256"]:
            raise ValueError("Arm contract hash mismatch")
        config, _ = preflight(path)
        validate_remediation_resume(source, config)
        control = load_chassis_control(ROOT / config["control_math_source"])
        for role in ROLES:
            effective = deepcopy(config)
            seed, count = plan["seed"], 20000
            if role != "train":
                seed = plan["evaluation_seeds"][int(role == "confirmation")]
                effective["evaluation"]["seed"] = seed
                if role in ("quick", "confirmation"):
                    effective["evaluation"] = select_evaluation_cases(effective["evaluation"], plan["evaluation_case_names"])
                effective = fixed_suite_contract(effective)
                count = len(effective["evaluation"]["cases"]) * effective["evaluation"]["episodes_per_case"]
            receipts.append(design_preflight(effective, control, ROOT, stage="foundation",
                role=role, seed=seed, num_envs=count, block_index=0))
    return receipts


class RemediationStudy(TrainingBlocks):
    validate_plan = staticmethod(validate_plan)

    def __init__(self, args):
        super().__init__(args)
        self.plan = self.contract
        self.validate_plan(self.plan)
        self.records = {}
        self.claimed = False
        self.claim = Path(self.plan["source_stage"]) / (self.plan["contract_id"] + ".claim.json")
        self.report.update(successful_updates=self.plan["source_consumed_updates"],
            actor_updates_consumed=self.plan["source_actor_consumed_updates"],
            scope=self.plan["scope"], arms={}, evaluations={},
            inherited_recovery_attempts=self.plan["inherited_recovery_attempts"],
            protected_case_names=self.plan["protected_case_names"],
            automatic_promotion=False, automatic_formal_resume=False)

    def preflight(self):
        return validate_contracts(self.plan)

    def charge(self):
        return charged_updates(self.records.values())

    def validate_completion(self, recipe, result):
        expected_parent = 618 + 40 * ARMS.index(recipe["name"])
        if (result["successful_updates"] != 40 or result["actor_updates_in_block"] != 30
                or result["parent_updates"] != expected_parent or result["stage_actor_update"] != 368
                or result["initial_learning_rate"] != 1e-4 or result["learning_rate"] != 1e-4):
            raise ValueError("Arm did not preserve fixed LR, critic warmup or consumption")

    def finish(self, baselines, evaluations):
        self.report["comparison"] = compare_arms(baselines, evaluations, self.plan["evaluation_case_names"],
                                                 self.plan["protected_case_names"])
        self.report["status"] = "remediation_review_required"

    def artifact_selection(self):
        return {"status": self.report["status"], "deployment_checkpoint": None,
                "automatic_promotion": False}

    def account(self, name, record):
        old = self.records.get(name, {})
        if record.get("successful_updates", 0) < old.get("successful_updates", 0):
            raise ValueError("Consumption cannot move backwards")
        self.records[name] = {k: record[k] for k in ("successful_updates", "actor_updates_in_block") if k in record}
        self.report["successful_updates"] = self.charge()
        self.report["actor_updates_consumed"] = self.plan["source_actor_consumed_updates"] + sum(
            r.get("actor_updates_in_block", 0) for r in self.records.values())
        ledger = {"run_directory": str(self.root), "study": self.plan["contract_id"], "arms": self.records,
            "consumed_updates": self.report["successful_updates"],
            "actor_updates_consumed": self.report["actor_updates_consumed"],
            "rollback_refunds_consumed_updates": False}
        write_json(self.root / "consumption.json", ledger)
        if self.claimed:
            write_json(self.claim, ledger)

    def publish(self, directory):
        for name in ("torque_monitor.json", "behavior_metrics.json", "live_state.json", "startup.json"):
            if (directory / name).exists():
                self.copy_atomic(directory / name, name)
        path = directory / "progress.json"
        if not path.exists():
            return False
        progress = json.loads(path.read_text())
        self.account(directory.parent.name, progress)
        progress.update(successful_updates=self.report["successful_updates"], parent_updates=0,
            actor_updates_consumed=self.report["actor_updates_consumed"],
            pid=os.getpid(), worker_pid=progress["pid"], active_arm=directory.parent.name,
            scope=self.plan["scope"], orchestration=self.plan.get("orchestration", "paired_S1_shared_budget"))
        write_json(self.root / "progress.json", progress)
        return True

    def authenticate_origin(self):
        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count
        checkpoint = Path(self.plan["source_checkpoint"])
        if (digest(checkpoint) != CHECKPOINT_SHA or digest(checkpoint_contract_path(checkpoint)) != SOURCE_SHA
                or checkpoint_update_count(checkpoint) != 618):
            raise ValueError("Source checkpoint identity or seal differs")
        stage = Path(self.plan["source_stage"])
        for filename, expected in self.plan["source_files_sha256"].items():
            if digest(stage / filename) != expected:
                raise ValueError("Source protection/recovery ledger changed")
        completion = json.loads((stage / "completion.json").read_text())
        retention = json.loads((stage / "retention_state.json").read_text())
        if (completion["recovery_attempts"] != self.plan["inherited_recovery_attempts"]
                or completion["recovery_exhausted"] != "retry_limit"
                or set(retention["protected_case_names"]) != set(self.plan["protected_case_names"])):
            raise ValueError("Exhausted recovery or protection history was reset")
        source = json.loads(checkpoint_contract_path(checkpoint).read_text())
        names = set(source["evaluation"]["promotion_case_names"]) | set(retention["protected_case_names"])
        if names != set(self.plan["evaluation_case_names"]):
            raise ValueError("Study drops required evaluation coverage")
        baselines = []
        provenance = []
        for seed, suffix in zip(self.plan["evaluation_seeds"], ("", "_confirmation")):
            path = stage / ("evaluation_002" + suffix) / "evaluation.json"
            result = json.loads(path.read_text())
            candidate = validate_evaluation(result, [c["name"] for c in source["evaluation"]["cases"]])
            if (result["contract_sha256"] != SOURCE_SHA or result["settings"]["seed"] != seed
                    or candidate["checkpoint_sha256"] != CHECKPOINT_SHA
                    or result["episodes_per_case"] != source["evaluation"]["episodes_per_case"]):
                raise ValueError("Source baseline provenance mismatch")
            baselines.append(candidate)
            provenance.append({"file": str(path), "sha256": digest(path), "seed": seed,
                "full_cases": len(candidate["cases"]), "required_cases": sorted(names)})
        write_json(self.root / "baseline_reuse.json", provenance)
        return baselines

    def train_command(self, recipe):
        from wheeled_tasks.chassis.remediation import arm_changes
        settings = arm_changes(recipe["name"])["remediation"]
        return [sys.executable, "-B", str(ROOT / "scripts/train_chassis.py"),
            "--contract", str(ROOT / recipe["contract"]), "--stage", "foundation", "--research", "--publish-state",
            "--num-envs", "20000", "--updates", "40", "--seed", "290927", "--block-index", "0",
            "--device", self.args.device, "--run-dir", str(self.root / recipe["name"] / "train"),
            "--resume", self.plan["source_checkpoint"], "--resume-reward-change",
            "--consumed-updates", str(settings["arm_consumed_start"]),
            "--actor-updates-consumed", str(settings["arm_actor_consumed_start"]),
            "--stage-actor-update", "338", "--global-actor-update-offset", "0", "--learning-rate-scale", "1.0",
            "--max-runtime-seconds", str(max(1., self.deadline - time.monotonic()))]

    def run_child(self, command, output, *, training=False):
        if self.stop_requested or time.monotonic() >= self.deadline:
            raise InterruptedError("Study stopped or time budget reached")
        output.parent.mkdir(parents=True, exist_ok=True)
        code = self.execute(command, output.with_suffix(".log"), output if training else None)
        path = output / ("completion.json" if training else "evaluation.json")
        result = json.loads(path.read_text()) if path.exists() else {}
        if training and "successful_updates" in result:
            self.account(output.parent.name, result)  # Includes a failed or stopped worker.
        if self.stop_requested:
            raise InterruptedError("Study stopped; observed work remains charged")
        if code or result.get("status") != ("completed" if training else "evaluated"):
            raise RuntimeError(f"Study child failed: {output}: {result.get('error', result.get('status', code))}")
        return result

    def evaluate(self, recipe, seed, *, full=False):
        name = recipe["name"]
        output = self.root / name / f"evaluation_{seed}{'_full' if full else ''}"
        checkpoint = self.root / name / "train/model_final.pt"
        names = ([c["name"] for c in json.loads((ROOT / recipe["contract"]).read_text())["evaluation"]["cases"]]
                 if full else self.plan["evaluation_case_names"])
        command = [sys.executable, "-B", str(ROOT / "scripts/evaluate_chassis.py"),
            "--contract", str(ROOT / recipe["contract"]), "--checkpoint", str(checkpoint),
            "--output", str(output), "--device", self.args.device, "--seed", str(seed),
            "--role", ("full" if full else "quick") if seed == self.plan["evaluation_seeds"][0] else "confirmation"]
        if not full:
            command += ["--cases", *names]
        result = self.run_child(command, output)
        candidate = validate_evaluation(result, names)
        if (candidate["checkpoint_sha256"] != digest(checkpoint) or result["contract_sha256"] != recipe["sha256"]
                or result["settings"]["seed"] != seed or result["episodes_per_case"] != 4):
            raise ValueError("Evaluation does not match its frozen arm")
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(str(self.root / name / "tensorboard_evaluation")) as writer:
            for case_name, case in candidate["cases"].items():
                for key, value in case.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"Evaluation/seed_{seed}/{case_name}/{key}", value,
                                          self.report["successful_updates"])
        self.report["evaluations"].setdefault(name, {})[str(seed) + ("_full" if full else "")] = str(output / "evaluation.json")
        write_json(self.root / "study_summary.json", self.report)
        return candidate

    def run(self):
        self.root.mkdir(parents=True, exist_ok=False)
        self.copy_atomic(self.args.contract, "study_plan.json")
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        try:
            write_json(self.root / "preflight.json", self.preflight())
            baselines = self.authenticate_origin()
            # Persistent exclusive claim prevents repeating forks with fresh credit.
            with self.claim.open("x") as stream:
                json.dump({"run_directory": str(self.root), "consumed_updates": self.plan["source_consumed_updates"]}, stream)
            self.claimed = True
            self.report["status"] = "running"
            evaluations = {}
            for recipe in self.plan["stages"]:
                name = recipe["name"]
                result = self.run_child(self.train_command(recipe), self.root / name / "train", training=True)
                self.validate_completion(recipe, result)
                self.report["arms"][name] = {k: result[k] for k in ("successful_updates", "parent_updates",
                    "actor_updates_in_block", "stage_actor_update", "learning_rate", "checkpoint_sha256")}
                evaluations[name] = [self.evaluate(recipe, seed) for seed in self.plan["evaluation_seeds"]]
            self.finish(baselines, evaluations)
        except InterruptedError as error:
            self.report.update(status="stopped", reason=str(error))
        except Exception:
            self.report.update(status="failed", error=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(self.root / "completion.json", self.report)
            write_json(self.root / "study_summary.json", self.report)
            write_json(self.root / "artifact_selection.json", self.artifact_selection())
            self.publish_terminal()
        return int(self.report["status"] == "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=20000)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--max-runtime-seconds", type=float, default=14400.)
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.set_defaults(transfer=None, stage="S1_remediation", worker_source=None)
    args = parser.parse_args()
    if not args.research or args.num_envs != 20000 or args.max_runtime_seconds <= 0:
        parser.error("Frozen 20000-env research study and bounded runtime required")
    if args.preflight_only:
        rows = validate_contracts(json.loads(args.contract.read_text()))
        print(json.dumps({"status": "passed", "stage_roles": len(rows)}))
        return 0
    return RemediationStudy(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
