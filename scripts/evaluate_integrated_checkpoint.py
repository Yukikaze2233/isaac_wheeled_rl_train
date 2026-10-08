#!/usr/bin/env python3
"""Assess one explicitly sealed actor across all six frozen task domains.

Run in the existing Isaac runtime. --preflight only authenticates CPU inputs;
neither mode changes campaign records, consumption, training or qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import signal
import subprocess
import time
from types import SimpleNamespace

from run_task_campaign import ROOT, TaskCampaign, write_json
from evaluate_chassis import evaluation_source_files
from train_chassis import preflight
from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis.evaluation import select_evaluation_cases
from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count
from wheeled_tasks.chassis.task_campaign import DOMAINS, INTEGRATED_REVISIONS, assess_domain, digest


SCHEMA = "single-checkpoint-six-domain-assessment-v1"
ABI_KEYS = (*lineage.PHYSICAL_KEYS, "asset_directory", "actor_frame_dim", "control_math_source",
            "task_modes", "phases", "command_transport", "response_migration")
STOP_GRACE_SECONDS = 30.


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def authenticated_checkpoint(path, expected_sha, source, contract_sha):
    """Authenticate a portable sealed copy without adopting it as a learning parent."""
    import torch

    seal_path = path.with_suffix(".checkpoint.json")
    if not seal_path.is_file():
        raise ValueError("An explicit per-file sealed checkpoint is required")
    if digest(path) != expected_sha or digest(checkpoint_contract_path(path)) != contract_sha:
        raise ValueError("Bound checkpoint or source contract SHA differs")
    count = checkpoint_update_count(path)
    seal = json.loads(seal_path.read_text())
    saved = torch.load(path, map_location="cpu", weights_only=True)
    infos = saved["infos"]
    if (infos.get("contract_sha256") != contract_sha
            or infos.get("task_campaign") != source["task_campaign"]
            or seal.get("task_campaign") != source["task_campaign"]
            or any(infos.get(k) != source.get(k) for k in ("asset_manifest_sha256", "control_math_sha256"))):
        raise ValueError("Checkpoint seal, model infos or source identity differs")
    for key, info_key in (("learning_updates", "successful_updates_total"),
                          ("learning_lineage_updates", "learning_lineage_updates"),
                          ("stage_actor_update", "stage_actor_update"),
                          ("task_actor_update", "task_actor_update"),
                          ("actor_updates_consumed", "actor_updates_consumed")):
        value = seal.get(key)
        if type(value) is not int or value < 0 or infos.get(info_key) != value:
            raise ValueError("Checkpoint seal and full-state clocks differ")
    if count != infos["successful_updates_total"] or any(
            seal[k] > count for k in ("learning_lineage_updates", "actor_updates_consumed")):
        raise ValueError("Checkpoint learning clocks exceed charged work")
    for key in ("rng_state", "env_rng_state"):
        state = infos.get(key)
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1 or not state.numel():
            raise ValueError("Checkpoint lacks saved RNG state")
    lineage._validate_native_learning_state(source, saved)
    return dict(infos=infos, seal=seal)


class IntegratedCheckpointEvaluation(TaskCampaign):
    """Reuse fixed evaluation and child ownership, with an isolated persistence sink."""

    def __init__(self, args):
        self.campaign_root = args.campaign_run_dir.resolve()
        args.contract = args.contract.resolve(strict=True)
        args.run_dir = args.output.resolve()
        super().__init__(args)
        if self.plan["contract_id"] not in INTEGRATED_REVISIONS:
            raise ValueError("Assessment requires an authenticated frozen integrated plan")
        if str(self.campaign_root) != self.plan["revision_migration"]["run_directory"]:
            raise ValueError("Assessment must use the frozen campaign run directory and lock")
        if (self.root.is_relative_to(self.campaign_root) or self.campaign_root.is_relative_to(self.root)
                or self.root.is_relative_to(self.claim.parent) or self.claim.parent.is_relative_to(self.root)
                or self.root.is_relative_to(ROOT) or ROOT.is_relative_to(self.root)):
            raise ValueError("Independent assessment output must be outside campaign, budget and source roots")
        if re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256) is None:
            raise ValueError("An explicit lowercase checkpoint SHA-256 is required")
        self.checkpoint = args.checkpoint.resolve(strict=True)
        self.expected_sha = args.checkpoint_sha256
        self.report_path = self.root / "assessment.json"
        self.inputs = None
        self.previous_evaluations = {}
        self._stop_started = None

    def prepare(self):
        production.validate_plan(self.plan, ROOT)
        source_path = checkpoint_contract_path(self.checkpoint).resolve(strict=True)
        seal_path = self.checkpoint.with_suffix(".checkpoint.json").resolve(strict=True)
        source = json.loads(source_path.read_text())
        domain = source.get("task_campaign", {}).get("domain")
        row = next((r for r in self.plan["domains"] if r["name"] == domain), None)
        if row is None:
            raise ValueError("Checkpoint does not belong to one of the frozen six domains")
        # This validates the per-file seal, real CPU model/Adam/RNG and bound ABI.
        checked = authenticated_checkpoint(self.checkpoint, self.expected_sha, source, row["sha256"])
        meta = checked["infos"].get("integrated_lineage")
        if domain in dict(lineage.CHAIN) and (
                not isinstance(meta, dict) or meta != checked["seal"].get("integrated_lineage")
                or meta.get("lineage_id") != self.plan["integrated_continuation"]["lineage_id"]
                or meta.get("plan_sha256") != self.plan["integrated_continuation"]["lineage_manifest"]["sha256"]
                or meta.get("domain") != domain):
            raise ValueError("Checkpoint is not part of the selected joint learning lineage")
        files = [self.args.contract, self.checkpoint, source_path, seal_path, Path(__file__).resolve(),
                 ROOT / "scripts/evaluate_chassis.py", ROOT / "scripts/run_task_campaign.py",
                 ROOT / "scripts/train_chassis.py"]
        domains, native_sources = [], {}
        for target in self.plan["domains"]:
            path = ROOT / target["contract"]
            if digest(path) != target["sha256"]:
                raise ValueError("Frozen evaluation worker contract SHA differs")
            config = json.loads(path.read_text())
            if any(source.get(k) != config.get(k) for k in ABI_KEYS):
                raise ValueError("Selected actor and fixed examination have different physical/control ABI")
            names = sorted(set(target["required_case_names"]) | set(target.get("monitor_case_names", [])))
            select_evaluation_cases(config["evaluation"], names)
            source_files = evaluation_source_files(config, root=ROOT)
            native_sources[target["name"]] = {name: digest(ROOT / name) for name in source_files}
            files.extend(ROOT / name for name in source_files)
            domains.append(dict(name=target["name"], contract=target["contract"],
                contract_sha256=target["sha256"], required_cases=target["required_case_names"],
                evaluated_cases=names, seeds=self.plan["evaluation_seeds"],
                roles=["quick", "confirmation"], episodes_per_case=config["evaluation"]["episodes_per_case"]))
            files.append(path)
        if tuple(row["name"] for row in domains) != DOMAINS:
            raise ValueError("Final assessment must cover all six domains")
        # The six contracts share this ABI. Reuse native preflight once, then
        # retain its authenticated manifest dependencies rather than re-read
        # the same asset bundle six times or invent another asset inventory.
        bundle = (ROOT / source["asset_directory"]).resolve(strict=True)
        _, manifest = preflight(source_path, asset_directory=bundle)
        physical_files = {str(bundle / "manifest.json"): source["asset_manifest_sha256"]}
        physical_files.update({str((bundle / name).resolve()): sha
                               for name, sha in manifest["files_sha256"].items()})
        control = (ROOT / source["control_math_source"]).resolve(strict=True)
        if digest(control) != source["control_math_sha256"]:
            raise ValueError("Bound control source SHA differs")
        physical_files[str(control)] = source["control_math_sha256"]
        self.inputs = dict(checkpoint=str(self.checkpoint), checkpoint_sha256=self.expected_sha,
            source_domain=domain, source_contract_sha256=digest(source_path),
            checkpoint_seal_sha256=digest(seal_path), learning_updates=checked["seal"]["learning_updates"],
            lineage_id=self.plan["integrated_continuation"]["lineage_id"],
            selected_checkpoint_scope="joint_candidate" if domain in dict(lineage.CHAIN) else "legacy_research_candidate",
            evaluation_device=self.args.device,
            plan_sha256=digest(self.args.contract), campaign_run_directory=str(self.campaign_root),
            source_root=str(ROOT), domains=domains, files_sha256={str(p): digest(p) for p in files})
        self.inputs["files_sha256"].update(physical_files)
        self.inputs["native_sources_sha256"] = native_sources
        history = []
        if self.report_path.exists():
            previous = json.loads(self.report_path.read_text())
            if previous.get("schema") != SCHEMA or previous.get("inputs") != self.inputs:
                raise ValueError("Assessment output already belongs to different immutable inputs")
            self.previous_evaluations = deepcopy(previous.get("evaluations", {}))
            history = previous.get("attempt_history", []) + [{k: previous.get(k) for k in
                ("status", "started_at", "finished_at", "error", "evaluations")}]
        elif self.root.exists() and any(self.root.iterdir()):
            raise ValueError("Assessment output contains unowned evidence")
        self.report = dict(schema=SCHEMA, status="prepared", inputs=self.inputs, domains={},
            evaluations=deepcopy(self.previous_evaluations), attempt_history=history,
            started_at=utc_now(), assessment_seconds=0., consumed_updates=self.inputs["learning_updates"],
            tensorboard_step_scope="selected_checkpoint_sealed_learning_updates",
            PPO_updates=0, qualification=None, accepted_stage=None, deployment_checkpoint=None,
            automatic_promotion=False, training_account_written=False,
            scope="same_actor_SHA_across_six_fixed_simulation_domains_not_deployment_qualification")
        return self.report

    def unchanged(self):
        if any(digest(Path(path)) != expected for path, expected in self.inputs["files_sha256"].items()):
            raise ValueError("Immutable assessment input changed")

    def persist(self):
        # Deliberately never call TaskCampaign.persist/account/open_account/run.
        write_json(self.report_path, self.report)

    def read_evaluation(self, path, row, checkpoint, seed):
        self.unchanged()
        for records in self.previous_evaluations.values():
            for evidence in records.values():
                if evidence["file"] == str(path) and evidence["sha256"] != digest(path):
                    raise ValueError("Previously recorded evaluation SHA changed")
        candidate = super().read_evaluation(path, row, checkpoint, seed)
        record = json.loads(path.read_text())
        expected_sources = self.inputs["native_sources_sha256"][row["name"]]
        if record.get("source_sha256") != expected_sources:
            raise ValueError("Native evaluation source map differs from authenticated inputs")
        if any(digest(path.parent / "source" / name) != sha for name, sha in expected_sources.items()):
            raise ValueError("Retained native evaluation source bytes differ")
        config = json.loads((ROOT / row["contract"]).read_text())
        names = sorted(set(row["required_case_names"]) | set(row.get("monitor_case_names", [])))
        settings = select_evaluation_cases(config["evaluation"], names)
        settings["seed"] = seed
        if (record.get("settings") != settings
                or record.get("evaluated_case_names") != [case["name"] for case in settings["cases"]]
                or record.get("episodes_per_case") != settings["episodes_per_case"]
                or record.get("device") != self.args.device
                or record.get("script_sha256") != digest(ROOT / "scripts/evaluate_chassis.py")
                or record.get("deterministic_actor") is not True
                or record.get("random_pushes") is not False or record.get("case_resampling") is not False
                or candidate.get("checkpoint_sha256") != self.expected_sha
                or candidate.get("cross_asset_evaluation") is not False
                or candidate.get("source_asset_manifest_sha256") != config["asset_manifest_sha256"]
                or candidate.get("evaluated_asset_manifest_sha256") != config["asset_manifest_sha256"]
                or candidate.get("actor_input_dim") != config["actor_dim"]
                or candidate.get("actor_output_dim") != config["action_dim"]
                or any(type(case.get("passed")) is not bool for case in candidate["cases"].values())):
            raise ValueError("Final assessment fixed settings, actor identity or provenance mismatch")
        return candidate

    def request_stop(self, signum, frame):
        super().request_stop(signum, frame)
        now = time.monotonic()
        if self._stop_started is None:
            self._stop_started = now
        elif (now - self._stop_started >= STOP_GRACE_SECONDS
              and self.eval_child is not None and self.eval_child.poll() is None):
            self.eval_child.kill()

    def run_assessment(self):
        # Identical durable campaign lock; fail immediately, never stop an owner.
        with self.lock_path.open("a") as ownership:
            fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
            TaskCampaign.assert_no_live_workers(SimpleNamespace(root=self.campaign_root))
            TaskCampaign.assert_no_live_workers(SimpleNamespace(root=self.root))
            self.prepare()
            self.report["status"] = "running_fixed_assessment"
            self.persist()
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
            for sig in handlers:
                signal.signal(sig, self.request_stop)
            try:
                label = "same_actor_" + self.expected_sha
                for row in self.plan["domains"]:
                    if self.stop_requested or time.monotonic() >= self.deadline:
                        raise RuntimeError("Fixed assessment stopped or reached its runtime limit")
                    self.unchanged()
                    results = self.evaluate_pair(row, self.checkpoint, label)
                    self.unchanged()
                    assessment = assess_domain(results, row["required_case_names"])
                    self.report["domains"][row["name"]] = dict(
                        checkpoint_sha256=self.expected_sha, passed_required_both_seeds=assessment["domain_qualified"],
                        required_failed=assessment["required_failed"], qualification=None,
                        source_evaluations=deepcopy(self.report["evaluations"][row["name"]]))
                    self.persist()
                self.report.update(status="assessed", all_required_passed_both_seeds=all(
                    row["passed_required_both_seeds"] for row in self.report["domains"].values()))
            except Exception as error:
                self.report.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                if self.eval_child is not None and self.eval_child.poll() is None:
                    self.eval_child.terminate()
                    try:
                        self.eval_child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.eval_child.kill()
                        self.eval_child.wait()
                    self.eval_child = None
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
                self.report["finished_at"] = utc_now()
                self.persist()
        return self.report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--campaign-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--max-runtime-seconds", type=float, default=86400.)
    parser.add_argument("--preflight", action="store_true", help="Authenticate CPU inputs and write plan; do not launch native evaluation")
    args = parser.parse_args(argv)
    if args.max_runtime_seconds <= 0:
        parser.error("A positive runtime is required")
    try:
        evaluator = IntegratedCheckpointEvaluation(args)
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
