#!/usr/bin/env python3
"""Assess one sealed actor on six separately frozen paired small-stair cases.

--preflight authenticates CPU inputs only. Native evaluation rejects active PPO,
uses the existing campaign lock, and never writes training or budget records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from evaluate_integrated_checkpoint import IntegratedCheckpointEvaluation
from evaluate_chassis import evaluation_source_files
from run_task_campaign import ROOT
from wheeled_tasks.chassis import small_stair_supplement as supplement
from wheeled_tasks.chassis.task_campaign import digest


class SmallStairCheckpointEvaluation(IntegratedCheckpointEvaluation):
    report_schema = "single-checkpoint-six-small-stair-assessment-v1"

    def __init__(self, args):
        super().__init__(args)
        self.report_path = self.root / "small_stairs_assessment.json"
        for key in ("supplement_manifest_sha256", "supplement_design_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(args, key)) is None:
                raise ValueError("Explicit lowercase supplement SHA-256 bindings are required")
        self.bindings = dict(
            manifest_file=str(args.supplement_manifest.resolve(strict=True).relative_to(ROOT.resolve())),
            manifest_sha256=args.supplement_manifest_sha256,
            design_file=str(args.supplement_design.resolve(strict=True).relative_to(ROOT.resolve())),
            design_sha256=args.supplement_design_sha256)
        self.suites = {}
        self.proofs = {}

    def extend_inputs(self):
        for seed, role in supplement.ROLES.items():
            self.suites[seed], proof = supplement.prepare_suite(ROOT,
                manifest_file=self.bindings["manifest_file"], manifest_sha=self.bindings["manifest_sha256"],
                design_file=self.bindings["design_file"], design_sha=self.bindings["design_sha256"],
                role=role, seed=seed)
            self.proofs[seed] = proof
        parent = json.loads((ROOT / proof["parent_contract_file"]).read_text())
        native_files = evaluation_source_files(parent, root=ROOT, supplement=self.bindings)
        self.inputs["native_sources_sha256"][supplement.DOMAIN] = {
            name: digest(ROOT / name) for name in native_files}
        self.inputs["files_sha256"].update({str((ROOT / name).resolve()): digest(ROOT / name)
                                           for name in native_files})
        self.inputs["files_sha256"][str(Path(__file__).resolve())] = digest(Path(__file__))
        self.inputs["supplement"] = dict(**self.bindings, case_count=6,
            base_contract=proof["parent_contract_file"], base_contract_sha256=proof["parent_contract_sha256"],
            old418_changed=False, original_reset_seed_keys_preserved=True,
            down_case_scope=proof["down_case_scope"])
        self.supplement_row = dict(name=supplement.DOMAIN, contract=proof["parent_contract_file"],
            sha256=proof["parent_contract_sha256"],
            required_case_names=[case["name"] for case in self.suites[290929]["evaluation"]["cases"]],
            monitor_case_names=[])

    def prepare(self):
        report = super().prepare()
        report["scope"] = "six_paired_small_stair_cases_only_not_old418_or_deployment_qualification"
        report["native_case_definitions"] = 6
        report["old418_changed"] = False
        return report

    def assessment_rows(self):
        return [self.supplement_row]

    def evaluation_settings(self, row, seed):
        return deepcopy(self.suites[seed]["evaluation"])

    def evaluation_command(self, row, checkpoint, output, seed, names):
        command = super().evaluation_command(row, checkpoint, output, seed, names)
        index = command.index("--no-traces")
        command[index:index] = ["--supplement-manifest", str(ROOT / self.bindings["manifest_file"]),
            "--supplement-manifest-sha256", self.bindings["manifest_sha256"],
            "--supplement-design", str(ROOT / self.bindings["design_file"]),
            "--supplement-design-sha256", self.bindings["design_sha256"]]
        return command

    def read_evaluation(self, path, row, checkpoint, seed):
        candidate = super().read_evaluation(path, row, checkpoint, seed)
        record = json.loads(path.read_text())
        proof = record.get("supplement", {})
        if (proof != self.proofs[seed]
                or record.get("qualification") is not None or record.get("PPO_updates") != 0
                or record.get("old418_changed") is not False):
            raise ValueError("Supplement report lacks its exact independent bindings or research scope")
        return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True, help="Unchanged frozen v25 plan")
    parser.add_argument("--campaign-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supplement-manifest", type=Path, required=True)
    parser.add_argument("--supplement-manifest-sha256", required=True)
    parser.add_argument("--supplement-design", type=Path, required=True)
    parser.add_argument("--supplement-design-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--max-runtime-seconds", type=float, default=86400.)
    parser.add_argument("--preflight", action="store_true", help="CPU authentication only; never launch native")
    args = parser.parse_args(argv)
    if args.max_runtime_seconds <= 0:
        parser.error("A positive runtime is required")
    try:
        evaluator = SmallStairCheckpointEvaluation(args)
        if args.preflight:
            evaluator.prepare()
            evaluator.report.update(status="cpu_preflight_only", native_scene_started=False)
            evaluator.persist()
        else:
            evaluator.run_assessment()
        print(json.dumps(dict(status=evaluator.report["status"], report=str(evaluator.report_path),
                              checkpoint_sha256=evaluator.expected_sha, PPO_updates=0, qualification=None)))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps(dict(status="rejected_or_failed", error=f"{type(error).__name__}: {error}",
                              PPO_updates=0, qualification=None)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
