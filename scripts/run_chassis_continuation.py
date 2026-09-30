#!/usr/bin/env python3
"""Continue A's PPO learning state for one charged S1 block, then reassess."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from run_chassis_remediation import RemediationStudy, ROOT, write_json
from wheeled_tasks.chassis.continuation import (
    REVISION, SOURCE_SHA, CHECKPOINT_SHA, ORIGIN, charged_updates, compare, validate_resume,
)
from wheeled_tasks.chassis.remediation import digest, validate_evaluation


def validate_plan(plan):
    origin = json.loads((ROOT / ORIGIN).read_text())
    previous = json.loads((ROOT / "contracts/v6_s1_remediation_v1.json").read_text())
    expected = dict(contract_id=REVISION, source_checkpoint_sha256=CHECKPOINT_SHA,
        source_contract_sha256=SOURCE_SHA, source_consumed_updates=698,
        source_actor_consumed_updates=678, source_stage_actor_update=368,
        source_learning_lineage_updates=378, study_update_ceiling=40,
        origin_file=ORIGIN, origin_sha256=digest(ROOT / ORIGIN),
        scope="formal_S1_continuation_charged_to_original_budget", orchestration="formal_S1_continuation",
        protected_case_names=sorted(set(origin["old_protected"]) | set(origin["A_passed"])))
    expected.update({k: origin[k] for k in ("source_stage", "source_checkpoint", "source_files_sha256")})
    expected.update({k: previous[k] for k in ("target_num_envs", "num_steps_per_env", "seed",
        "evaluation_case_names", "evaluation_seeds", "inherited_recovery_attempts", "stage_ceiling",
        "total_ceiling", "rollback_refunds_consumed_updates", "automatic_promotion", "automatic_formal_resume")})
    if (any(plan.get(k) != v for k, v in expected.items()) or len(plan["stages"]) != 1
            or plan["stages"][0]["name"] != "A_continue" or plan["stages"][0]["updates"] != 40):
        raise ValueError("Continuation changes frozen source, budget, protection or review boundary")


def validate_contracts(plan):
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
    from wheeled_tasks.chassis.preflight import design_preflight, ROLES
    validate_plan(plan)
    source_path = ROOT / "contracts/v6_s1_A_fixed_v1.json"
    if digest(source_path) != SOURCE_SHA:
        raise ValueError("A source contract identity differs")
    recipe = plan["stages"][0]
    path = ROOT / recipe["contract"]
    if digest(path) != recipe["sha256"]:
        raise ValueError("Continuation contract hash mismatch")
    config, _ = preflight(path)
    validate_resume(json.loads(source_path.read_text()), config)
    control = load_chassis_control(ROOT / config["control_math_source"])
    receipts = []
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


class Continuation(RemediationStudy):
    validate_plan = staticmethod(validate_plan)

    def preflight(self):
        return validate_contracts(self.plan)

    def charge(self):
        return charged_updates(self.records.values())

    def authenticate_origin(self):
        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count
        checkpoint, stage = Path(self.plan["source_checkpoint"]), Path(self.plan["source_stage"])
        origin = json.loads((ROOT / ORIGIN).read_text())
        if (digest(checkpoint) != CHECKPOINT_SHA or digest(checkpoint_contract_path(checkpoint)) != SOURCE_SHA
                or checkpoint_update_count(checkpoint) != 658):
            raise ValueError("A learning checkpoint identity or seal differs")
        for filename, expected in self.plan["source_files_sha256"].items():
            if digest(stage / filename) != expected:
                raise ValueError(f"Source evidence changed: {filename}")
        completion = json.loads((stage / "completion.json").read_text())
        ledger = json.loads((stage / "consumption.json").read_text())
        if (completion["status"] != "remediation_review_required"
                or ledger != origin["study_consumption"]
                or completion["inherited_recovery_attempts"] != self.plan["inherited_recovery_attempts"]
                or set(completion["protected_case_names"]) != set(origin["old_protected"])):
            raise ValueError("Continuation would reset consumed work or recovery/protection history")
        baselines, provenance = [], []
        for seed in self.plan["evaluation_seeds"]:
            path = stage / "A_fixed" / f"evaluation_{seed}/evaluation.json"
            result = json.loads(path.read_text())
            candidate = validate_evaluation(result, self.plan["evaluation_case_names"])
            if (result["contract_sha256"] != SOURCE_SHA or result["settings"]["seed"] != seed
                    or candidate["checkpoint_sha256"] != CHECKPOINT_SHA or result["episodes_per_case"] != 4):
                raise ValueError("A baseline provenance mismatch")
            baselines.append(candidate)
            provenance.append(dict(file=str(path), sha256=digest(path), seed=seed))
        passed = {n for n in self.plan["evaluation_case_names"] if all(b["cases"][n]["passed"] for b in baselines)}
        if passed != set(origin["A_passed"]):
            raise ValueError("A's confirmed behavior protection differs")
        write_json(self.root / "baseline_reuse.json", provenance)
        write_json(self.root / "inherited_ledger.json", origin)
        return baselines

    def train_command(self, recipe):
        return [sys.executable, "-B", str(ROOT / "scripts/train_chassis.py"),
            "--contract", str(ROOT / recipe["contract"]), "--stage", "foundation", "--research", "--publish-state",
            "--num-envs", "20000", "--updates", "40", "--seed", "290927", "--block-index", "0",
            "--device", self.args.device, "--run-dir", str(self.root / recipe["name"] / "train"),
            "--resume", self.plan["source_checkpoint"], "--resume-reward-change",
            "--consumed-updates", "698", "--actor-updates-consumed", "678", "--stage-actor-update", "368",
            "--global-actor-update-offset", "0", "--learning-rate-scale", "1.0",
            "--max-runtime-seconds", str(max(1., self.deadline - time.monotonic()))]

    def validate_completion(self, recipe, result):
        if (result["successful_updates"] != 40 or result["actor_updates_in_block"] != 40
                or result["parent_updates"] != 698 or result["stage_actor_update"] != 408
                or result["initial_learning_rate"] != 1e-4 or result["learning_rate"] != 1e-4):
            raise ValueError("Continuation changed fixed LR, actor updates or consumed work")
        seal = json.loads((self.root / recipe["name"] / "train/model_final.checkpoint.json").read_text())
        if (seal["learning_lineage_updates"] != 418 or seal["actor_updates_consumed"] != 718
                or seal["learning_updates"] != 738 or not seal["sealed"]):
            raise ValueError("Continuation learning lineage or seal differs")

    def finish(self, baselines, evaluations):
        recipe = self.plan["stages"][0]
        config = json.loads((ROOT / recipe["contract"]).read_text())
        comparison = compare(baselines, evaluations[recipe["name"]], self.plan["protected_case_names"],
                             config["evaluation"]["promotion_case_names"])
        self.report["comparison"] = comparison
        # A short-suite pass is never treated as an exit gate without the full audit.
        if comparison["ready_for_full_audit"]:
            results = [self.evaluate(recipe, seed, full=True) for seed in self.plan["evaluation_seeds"]]
            required = set(self.plan["protected_case_names"]) | set(config["evaluation"]["promotion_case_names"])
            self.report["full_audit_required_failures"] = sorted(n for n in required
                if not all(r["cases"][n]["passed"] for r in results))
        self.report["status"] = "continuation_review_required"
        self.report["remaining_S1_updates"] = 1000 - self.report["successful_updates"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=20000)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--max-runtime-seconds", type=float, default=14400.)
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.set_defaults(transfer=None, stage="S1", worker_source=None)
    args = parser.parse_args()
    if not args.research or args.num_envs != 20000 or args.max_runtime_seconds <= 0:
        parser.error("Frozen 20000-env continuation and positive runtime required")
    if args.preflight_only:
        print(json.dumps(dict(status="passed", stage_roles=len(validate_contracts(json.loads(args.contract.read_text()))))))
        return 0
    return Continuation(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
