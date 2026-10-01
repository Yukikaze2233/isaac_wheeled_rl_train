#!/usr/bin/env python3
"""Run independent, persistent task workers against one durable campaign account."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from run_chassis_blocks import ROOT, TrainingBlocks
from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count
from wheeled_tasks.chassis.remediation import validate_evaluation
from wheeled_tasks.chassis.task_campaign import (
    CAMPAIGN_ID, account_updates, assess_domain, digest, update_attempt,
    validate_plan, validate_task_resume, validate_plan_migration, REVISION_V2,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    # The shared claim must survive a power loss between rename and local state.
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_contracts(plan):
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
    from wheeled_tasks.chassis.preflight import design_preflight, ROLES
    validate_plan(plan)
    receipts = []
    parent_hashes = {"A": plan["source_contract_sha256"]}
    for row in plan["domains"]:
        path = ROOT / row["contract"]
        if digest(path) != row["sha256"]:
            raise ValueError("Task worker contract SHA differs")
        config, _ = preflight(path)
        metadata = config["task_campaign"]
        if (metadata["domain"] != row["name"] or metadata["source_contract_sha256"] != parent_hashes[row["parent_domain"]]
                or set(config["evaluation"]["promotion_case_names"]) != set(row["required_case_names"])):
            raise ValueError("Domain parent, identity or required gates differ from the frozen worker")
        parent_hashes[row["name"]] = row["sha256"]
        source = ROOT / config["task_campaign"]["source_contract_file"]
        validate_task_resume(json.loads(source.read_text()), config)
        control = load_chassis_control(ROOT / config["control_math_source"])
        names = sorted(set(row["required_case_names"]) | set(row.get("monitor_case_names", [])))
        for role in ROLES:
            effective = deepcopy(config)
            seed, count = plan["seed"], plan["target_num_envs"]
            if role != "train":
                seed = plan["evaluation_seeds"][int(role == "confirmation")]
                effective["evaluation"]["seed"] = seed
                effective["evaluation"] = select_evaluation_cases(effective["evaluation"], names)
                effective = fixed_suite_contract(effective)
                count = len(names) * effective["evaluation"]["episodes_per_case"]
            receipts.append(design_preflight(effective, control, ROOT, stage=row["kind"],
                role=role, seed=seed, num_envs=count, block_index=0))
    return receipts


class TaskCampaign(TrainingBlocks):
    def __init__(self, args):
        # Reuse only process ownership/publishing machinery, not stage gates.
        self.args, self.root = args, args.run_dir.resolve()
        self.plan = self.contract = json.loads(args.contract.read_text())
        validate_plan(self.plan)
        self.child = self.eval_child = None
        self.stop_requested = False
        self.deadline = time.monotonic() + args.max_runtime_seconds
        self.claim = Path(self.plan["budget_source_stage"]) / (CAMPAIGN_ID + ".account.json")
        self.lock_path = self.claim.with_suffix(".lock")
        self.records = {}
        self.report = dict(status="starting", campaign_id=CAMPAIGN_ID,
            contract_sha256=digest(args.contract), domains={}, evaluations={},
            started_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=0.,
            assessment_seconds=0., training_process_seconds=0., accepted_stage=None,
            deployment_checkpoint=None, automatic_promotion=False)
        self.active = None
        self._session_started = None
        self._elapsed_base = 0.

    def request_stop(self, signum, frame):
        super().request_stop(signum, frame)
        if self.eval_child is not None and self.eval_child.poll() is None:
            self.eval_child.send_signal(signum)

    def persist(self):
        if self._session_started is not None:
            self.report["elapsed_seconds"] = self._elapsed_base + time.monotonic() - self._session_started
        self.report.update(account_updates(self.plan, self.records))
        self.report["successful_updates"] = self.report["consumed_updates"]
        ledger = dict(campaign_id=CAMPAIGN_ID, run_directory=str(self.root),
            contract_sha256=digest(self.args.contract), attempts=self.records,
            rollback_refunds_consumed_updates=False, **account_updates(self.plan, self.records))
        write_json(self.claim, ledger)
        write_json(self.root / "consumption.json", ledger)
        write_json(self.root / "campaign_state.json", self.report)

    def authenticate(self):
        checkpoint = Path(self.plan["source_checkpoint"])
        if (digest(checkpoint) != self.plan["source_checkpoint_sha256"]
                or digest(checkpoint_contract_path(checkpoint)) != self.plan["source_contract_sha256"]):
            raise ValueError("Campaign A learning source changed")
        checkpoint_update_count(checkpoint)
        stage = Path(self.plan["budget_source_stage"])
        for name, sha in self.plan["budget_source_files_sha256"].items():
            if digest(stage / name) != sha:
                raise ValueError("Historical budget evidence changed")
        ledger = json.loads((stage / "consumption.json").read_text())
        if (ledger["consumed_updates"] != self.plan["historical_consumed_updates"]
                or ledger["actor_updates_consumed"] != self.plan["historical_actor_consumed_updates"]):
            raise ValueError("Campaign would refund historical optimization")

    def open_account(self):
        migration = None
        if self.claim.exists():
            ledger = json.loads(self.claim.read_text())
            if ledger["campaign_id"] != CAMPAIGN_ID or ledger["run_directory"] != str(self.root):
                raise ValueError("Campaign budget already belongs to a different run/revision")
            if ledger["contract_sha256"] != digest(self.args.contract):
                if self.plan["contract_id"] != REVISION_V2:
                    raise ValueError("Campaign budget already belongs to a different run/revision")
                binding = self.plan["revision_migration"]
                manifest = json.loads((ROOT / binding["manifest_file"]).read_text())
                source = json.loads((ROOT / manifest["source_plan_file"]).read_text())
                migration = validate_plan_migration(source, self.plan, self.root)
                if ledger["contract_sha256"] != migration["source_plan_sha256"]:
                    raise ValueError("Campaign account is not the named revision source")
                write_json(self.root / "revisions/v1_account_before_migration.json", ledger)
            self.records = ledger["attempts"]
            state = self.root / "campaign_state.json"
            if state.exists():
                self.report.update(json.loads(state.read_text()))
            # A crash can occur after the authoritative claim was persisted and
            # before its local report. Reconstruct counters from the claim.
            # Progress is durable per update and may be newer than supervisor polling.
            for path in sorted(self.root.glob("*/attempt_*/train/progress.json")):
                record = json.loads(path.read_text())
                key = str(path.parent.relative_to(self.root))
                domain = path.parents[2].name
                self.records = update_attempt(self.plan, self.records, key, domain, record)
                pid = record.get("pid")
                command = Path(f"/proc/{pid}/cmdline")
                if command.exists() and str(path.parent).encode() in command.read_bytes():
                    raise ValueError("Existing task worker is still alive; do not start a duplicate")
        elif (self.root / "campaign_state.json").exists():
            raise ValueError("Campaign state exists without its durable shared budget claim")
        elif self.plan["contract_id"] == REVISION_V2:
            raise ValueError("Task v2 requires the existing named campaign budget account")
        self.assert_no_live_workers()
        self.report["contract_sha256"] = digest(self.args.contract)
        if migration is not None:
            migration.update(**account_updates(self.plan, self.records), attempts=deepcopy(self.records))
            self.report["revision_migration"] = migration
            write_json(self.root / "revisions/v1_to_v2_account_migration.json", migration)
        self.persist()

    def assert_no_live_workers(self):
        # Includes the tiny child-start/first-progress window and orphaned evals.
        for process in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                arguments = process.read_bytes().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if (any(arg.endswith((b"/train_chassis.py", b"/evaluate_chassis.py")) for arg in arguments)
                    and any(arg.startswith(os.fsencode(str(self.root) + "/")) for arg in arguments)):
                raise ValueError("Existing task worker or assessment is still alive; do not start a duplicate")

    def account(self, directory, record):
        key = str(directory.relative_to(self.root))
        domain = directory.parents[1].name
        self.records = update_attempt(self.plan, self.records, key, domain, record)
        self.persist()

    def publish(self, directory):
        path = directory / "progress.json"
        if not path.exists():
            return False
        progress = json.loads(path.read_text())
        self.account(directory, progress)
        progress.update(successful_updates=self.report["consumed_updates"], parent_updates=0,
            actor_updates_consumed=self.report["actor_updates_consumed"],
            worker_pid=progress.get("pid"), pid=os.getpid(), active_domain=self.active["name"],
            phase="training", orchestration="independent_task_domain_campaign")
        write_json(self.root / "progress.json", progress)
        for name in ("startup.json", "behavior_metrics.json", "torque_monitor.json", "live_state.json"):
            if (directory / name).exists():
                self.copy_atomic(directory / name, name)
        if self.plan["contract_id"] != REVISION_V2:
            self.assess_request(directory)
        return True

    def assess_request(self, directory):
        if self.plan["contract_id"] == REVISION_V2:
            return
        request_path, decision_path = directory / "assessment_request.json", directory / "assessment_decision.json"
        if not request_path.exists():
            return
        request = json.loads(request_path.read_text())
        if decision_path.exists() and json.loads(decision_path.read_text()).get("token") == request["token"]:
            return
        checkpoint = Path(request["checkpoint"])
        if not checkpoint.resolve().is_relative_to(directory.resolve()) or digest(checkpoint) != request["checkpoint_sha256"]:
            raise ValueError("Assessment request does not identify this worker's sealed snapshot")
        consumed = checkpoint_update_count(checkpoint)
        seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
        task_clock = request["task_actor_update"]
        if (task_clock != seal.get("task_actor_update") or consumed != request["consumed_updates"]
                or consumed > self.report["consumed_updates"]):
            raise ValueError("Assessment request clock or consumption differs from its seal/account")
        if task_clock <= 0 or task_clock % 1000:
            raise ValueError("Assessment request changed the frozen interval")
        progress_path = self.root / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            progress.update(phase="cooperative_fixed_evaluation", assessment_task_actor_update=task_clock)
            write_json(progress_path, progress)
        assessment_started = time.monotonic()
        results = self.evaluate_pair(self.active, checkpoint, f"task_{task_clock:08d}_{digest(checkpoint)[:12]}")
        self.report["cooperative_assessment_seconds"] = self.report.get("cooperative_assessment_seconds", 0.) + time.monotonic() - assessment_started
        assessment = assess_domain(results, self.active["required_case_names"])
        confirmed_bins = self.confirmed_jump_bins(self.active, results,
            previous=seal.get("confirmed_jump_bins_m", []))
        decision = dict(token=request["token"], checkpoint_sha256=digest(checkpoint),
            confirmed_jump_bins_m=confirmed_bins,
            task_actor_update=task_clock, **assessment,
            decision="domain_qualified_stop" if assessment["domain_qualified"] else "continue",
            evaluations=[{**self.report["evaluations"][self.active["name"]][f"task_{task_clock:08d}_{digest(checkpoint)[:12]}/{seed}"],
                          "seed": seed} for seed in self.plan["evaluation_seeds"]])
        self.report["domains"].setdefault(self.active["name"], {})["latest_assessment"] = decision
        self.persist()
        write_json(decision_path, decision)

    def confirmed_jump_bins(self, row, results, *, previous=()):
        config = json.loads((ROOT / row["contract"]).read_text())
        allowed = config.get("new_asset_curriculum", {}).get("jump_curriculum", {}).get("bins_m", [])
        confirmed = set(previous) | set(self.report["domains"].get(row["name"], {}).get("confirmed_jump_bins_m", []))
        if confirmed - set(allowed):
            raise ValueError("Confirmed jump bins differ from the frozen curriculum")
        for case in config["evaluation"]["cases"]:
            if (case.get("task") != "jump" or case.get("skill", {}).get("kind") != "jump"
                    or case.get("jump_assistance_enabled") is not False):
                continue
            apex = case.get("skill", {}).get("semantics", {}).get("jump_apex_delta_m")
            if apex in allowed and all(result["cases"].get(case["name"], {}).get("passed") for result in results):
                confirmed.add(apex)
        bins = sorted(confirmed)
        self.report["domains"].setdefault(row["name"], {})["confirmed_jump_bins_m"] = bins
        return bins

    def read_evaluation(self, path, row, checkpoint, seed):
        result = json.loads(path.read_text())
        names = set(row["required_case_names"]) | set(row.get("monitor_case_names", []))
        candidate = validate_evaluation(result, names)
        if (candidate["checkpoint_sha256"] != digest(checkpoint)
                or result["contract_sha256"] != row["sha256"] or result["settings"]["seed"] != seed):
            raise ValueError("Task evaluation provenance mismatch")
        return candidate

    def evaluation_output(self, row, checkpoint, label, seed):
        output = self.root / row["name"] / "evaluations" / label / str(seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        attempts = [output, *sorted(output.parent.glob(output.name + "_retry_*"))]
        for attempt in attempts:
            path = attempt / "evaluation.json"
            if not path.exists():
                continue
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if record.get("status") == "evaluated":
                # A completed report with wrong provenance is an integrity error,
                # never a reason to silently replace the evidence.
                return attempt, self.read_evaluation(path, row, checkpoint, seed)
        if output.exists():
            retry = 1
            while output.with_name(output.name + f"_retry_{retry}").exists():
                retry += 1
            output = output.with_name(output.name + f"_retry_{retry}")
        return output, None

    def evaluate_pair(self, row, checkpoint, label):
        names = sorted(set(row["required_case_names"]) | set(row.get("monitor_case_names", [])))
        results = []
        started = time.monotonic()
        for seed in self.plan["evaluation_seeds"]:
            output, candidate = self.evaluation_output(row, checkpoint, label, seed)
            result_path = output / "evaluation.json"
            if candidate is None:
                if self.args.device == "cuda:0":
                    from watch_s1_checkpoints import resource_snapshot, sufficient_resources
                    resources = resource_snapshot()
                    write_json(output.with_suffix(".resources.json"), resources)
                    if not sufficient_resources(resources):
                        raise RuntimeError("Assessment needs 7 GiB free GPU and 4 GiB available host memory; training scene remains resident")
                command = [sys.executable, "-B", str(ROOT / "scripts/evaluate_chassis.py"),
                    "--contract", str(ROOT / row["contract"]), "--checkpoint", str(checkpoint),
                    "--output", str(output), "--device", self.args.device, "--seed", str(seed),
                    "--role", "quick" if seed == self.plan["evaluation_seeds"][0] else "confirmation",
                    "--no-traces", "--cases", *names]
                log = output.with_suffix(".log")
                with log.open("a") as stream:
                    self.eval_child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    while self.eval_child.poll() is None:
                        if self.stop_requested or time.monotonic() >= self.deadline:
                            self.request_stop(signal.SIGTERM, None)
                        time.sleep(.5)
                    code = self.eval_child.returncode
                    self.eval_child = None
                if code or self.stop_requested:
                    raise RuntimeError("Task assessment process failed or stopped")
                candidate = self.read_evaluation(result_path, row, checkpoint, seed)
            results.append(candidate)
            self.report["evaluations"].setdefault(row["name"], {})[f"{label}/{seed}"] = dict(
                file=str(result_path), sha256=digest(result_path))
            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter(str(self.root / row["name"] / "tensorboard_evaluation")) as writer:
                for name, case in candidate["cases"].items():
                    for key, value in case.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"Evaluation/seed_{seed}/{name}/{key}", value,
                                              self.report["consumed_updates"])
        self.report["assessment_seconds"] += time.monotonic() - started
        self.persist()
        return results

    def latest_checkpoint(self, row):
        candidates = []
        for path in (self.root / row["name"]).glob("attempt_*/train/**/*.checkpoint.json"):
            if any(part.endswith(".pending") for part in path.parts):
                continue
            checkpoint = path.with_suffix("").with_suffix(".pt")
            seal = json.loads(path.read_text())
            candidates.append((seal["learning_updates"], checkpoint))
        if not candidates:
            return None
        checkpoint = max(candidates, key=lambda item: (item[0], item[1].name == "model_final.pt"))[1]
        checkpoint_update_count(checkpoint)
        return checkpoint

    def learning_source(self, row):
        previous = self.latest_checkpoint(row)
        if previous is not None:
            return previous
        if row["parent_domain"] == "A":
            return Path(self.plan["source_checkpoint"])
        parent = self.report["domains"].get(row["parent_domain"], {})
        if not parent.get("candidate_checkpoint"):
            raise ValueError("Task parent lacks a sealed research learning candidate")
        return Path(parent["candidate_checkpoint"])

    def train_command(self, row, directory, source, updates):
        source_contract = json.loads(checkpoint_contract_path(source).read_text())
        seal = json.loads(source.with_suffix(".checkpoint.json").read_text())
        source_domain = source_contract.get("task_campaign", {}).get("domain")
        same = source_domain == row["name"]
        bins = sorted(set(seal.get("confirmed_jump_bins_m", [])) | set(
            self.report["domains"].get(source_domain, {}).get("confirmed_jump_bins_m", [])))
        return [sys.executable, "-B", str(ROOT / "scripts/train_chassis.py"),
            "--contract", str(ROOT / row["contract"]), "--stage", row["kind"], "--research", "--publish-state",
            "--num-envs", "20000", "--updates", str(updates), "--seed", str(self.plan["seed"]),
            "--block-index", "0", "--device", self.args.device, "--run-dir", str(directory),
            "--resume", str(source), "--resume-task-change", "--learning-rate-scale", "1.0",
            "--consumed-updates", str(self.report["consumed_updates"]),
            "--actor-updates-consumed", str(self.report["actor_updates_consumed"]),
            "--stage-actor-update", str(seal["stage_actor_update"]),
            "--task-actor-update", str(seal["task_actor_update"] if same else 0),
            "--confirmed-jump-bins-json", json.dumps(bins),
            "--global-actor-update-offset", "0", "--max-runtime-seconds", str(max(1., self.deadline-time.monotonic()))]

    def assessed_candidate(self, row, final):
        """Reuse an immutable assessment only if no learning occurred afterward."""
        source_contract = json.loads(checkpoint_contract_path(final).read_text())
        if source_contract.get("task_campaign", {}).get("domain") != row["name"]:
            return None
        directory = next((parent for parent in final.parents if parent.name == "train"), final.parent)
        request_path, decision_path = directory / "assessment_request.json", directory / "assessment_decision.json"
        if not request_path.exists() or not decision_path.exists():
            return None
        request, decision = json.loads(request_path.read_text()), json.loads(decision_path.read_text())
        checkpoint = Path(request["checkpoint"])
        if not checkpoint.resolve().is_relative_to((directory / "checkpoints").resolve()):
            raise ValueError("Assessed candidate is not this worker's immutable snapshot")
        final_count, snapshot_count = checkpoint_update_count(final), checkpoint_update_count(checkpoint)
        final_seal = json.loads(final.with_suffix(".checkpoint.json").read_text())
        snapshot_seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
        if (decision.get("token") != request["token"]
                or decision.get("checkpoint_sha256") != digest(checkpoint)
                or request["checkpoint_sha256"] != digest(checkpoint)
                or request["consumed_updates"] != snapshot_count
                or request["task_actor_update"] != snapshot_seal.get("task_actor_update")
                or decision.get("task_actor_update") != request["task_actor_update"]):
            raise ValueError("Assessed candidate request/decision authentication failed")
        fields = ("task_actor_update", "stage_actor_update", "learning_lineage_updates", "actor_updates_consumed", "contract_sha256")
        if final_count != snapshot_count or any(final_seal.get(k) != snapshot_seal.get(k) for k in fields):
            return None
        if snapshot_count > self.report["consumed_updates"]:
            raise ValueError("Assessed candidate consumption exceeds campaign account")
        evidence = decision.get("evaluations", [])
        if len(evidence) != 2 or {e["seed"] for e in evidence} != set(self.plan["evaluation_seeds"]):
            raise ValueError("Assessed candidate needs two authenticated evaluations")
        results = []
        for entry in evidence:
            path = Path(entry["file"])
            if digest(path) != entry["sha256"]:
                raise ValueError("Assessed candidate evaluation SHA differs")
            results.append(self.read_evaluation(path, row, checkpoint, entry["seed"]))
        assessment = assess_domain(results, row["required_case_names"])
        if (assessment["domain_qualified"] != decision.get("domain_qualified")
                or decision.get("decision") != ("domain_qualified_stop" if assessment["domain_qualified"] else "continue")):
            raise ValueError("Assessed candidate decision differs from fixed evidence")
        return checkpoint, results

    def run_domain(self, row):
        if self.plan["contract_id"] == REVISION_V2:
            return self.run_continuous_domain(row)
        self.active = row
        existing = self.report["domains"].get(row["name"], {})
        if existing.get("status") in ("domain_qualified", "domain_budget_exhausted", "domain_unqualified"):
            checkpoint_update_count(existing["candidate_checkpoint"])
            return
        source = self.learning_source(row)
        spent = self.report["domain_consumed_updates"][row["name"]]
        remaining = row["updates"] - spent
        recovered_assessment = self.assessed_candidate(row, source)
        already_qualified = recovered_assessment is not None and assess_domain(
            recovered_assessment[1], row["required_case_names"])["domain_qualified"]
        if remaining and not already_qualified:
            index = len(list((self.root / row["name"]).glob("attempt_*")))
            directory = self.root / row["name"] / f"attempt_{index:03d}" / "train"
            directory.parent.mkdir(parents=True, exist_ok=False)
            command = self.train_command(row, directory, source, remaining)
            write_json(directory.parent / "invocation.json", dict(command=command,
                parent_role="research_learning_candidate_not_accepted", source_checkpoint=str(source),
                source_checkpoint_sha256=digest(source), requested_updates=remaining))
            started = time.monotonic()
            code = self.execute(command, directory.parent / "train.log", directory)
            self.report["training_process_seconds"] += time.monotonic() - started
            self.persist()
            result_path = directory / "completion.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if "successful_updates" in result:
                self.account(directory, result)
            if code or result.get("status") not in ("completed", "stopped"):
                raise RuntimeError(f"Task worker failed; consumed work retained: {row['name']}")
            if self.stop_requested:
                raise InterruptedError("Campaign stopped; all observed work remains charged")
            if result["successful_updates"] < remaining:
                decision_path = directory / "assessment_decision.json"
                decision = json.loads(decision_path.read_text()) if decision_path.exists() else {}
                if decision.get("decision") != "domain_qualified_stop" or not decision.get("domain_qualified"):
                    raise RuntimeError("Task stopped early without a qualifying assessment")
            source = directory / "model_final.pt"
            checkpoint_update_count(source)
        final = source
        assessed = self.assessed_candidate(row, final)
        if assessed is not None:
            source, results = assessed
        else:
            results = self.evaluate_pair(row, source, "final_" + digest(source)[:12])
        assessment = assess_domain(results, row["required_case_names"])
        bins = self.confirmed_jump_bins(row, results,
            previous=json.loads(final.with_suffix(".checkpoint.json").read_text()).get("confirmed_jump_bins_m", []))
        self.report["domains"][row["name"]] = dict(**assessment,
            confirmed_jump_bins_m=bins, final_checkpoint=str(final), assessment_reused=assessed is not None,
            status=("domain_qualified" if assessment["domain_qualified"] else
                    "domain_budget_exhausted" if self.report["domain_consumed_updates"][row["name"]] >= row["updates"] else "domain_unqualified"),
            candidate_checkpoint=str(source), candidate_checkpoint_sha256=digest(source),
            parent_domain=row["parent_domain"], consumed_updates=self.report["domain_consumed_updates"][row["name"]])
        self.persist()

    def run_continuous_domain(self, row):
        """No score-controlled branch, assessment process or rollback in training."""
        self.active = row
        spent = self.report["domain_consumed_updates"][row["name"]]
        existing = self.report["domains"].get(row["name"], {})
        if spent >= row["updates"] and existing.get("candidate_checkpoint"):
            checkpoint_update_count(existing["candidate_checkpoint"])
            return
        source = self.learning_source(row)
        remaining = row["updates"] - spent
        if remaining:
            index = len(list((self.root / row["name"]).glob("attempt_*")))
            directory = self.root / row["name"] / f"attempt_{index:03d}" / "train"
            directory.parent.mkdir(parents=True, exist_ok=False)
            command = self.train_command(row, directory, source, remaining)
            write_json(directory.parent / "invocation.json", dict(command=command,
                parent_role="unassessed_research_candidate_not_accepted", source_checkpoint=str(source),
                source_checkpoint_sha256=digest(source), requested_updates=remaining))
            started = time.monotonic()
            code = self.execute(command, directory.parent / "train.log", directory)
            self.report["training_process_seconds"] += time.monotonic() - started
            result_path = directory / "completion.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if "successful_updates" in result:
                self.account(directory, result)
            self.persist()
            if self.stop_requested or time.monotonic() >= self.deadline:
                raise InterruptedError("Campaign stopped; all observed work remains charged")
            if code or result.get("status") not in ("completed", "stopped") or result.get("successful_updates", 0) < remaining:
                raise RuntimeError(f"Continuous task worker stopped before cap; consumed work retained: {row['name']}")
            source = directory / "model_final.pt"
            checkpoint_update_count(source)
        self.report["domains"][row["name"]] = dict(status="domain_budget_exhausted_unassessed",
            domain_qualified=None, assessment_status="not_run", accepted_stage=None,
            deployment_checkpoint=None, final_checkpoint=str(source), candidate_checkpoint=str(source),
            candidate_checkpoint_sha256=digest(source), parent_domain=row["parent_domain"],
            consumed_updates=self.report["domain_consumed_updates"][row["name"]],
            scope="unassessed_individual_task_research_candidate")
        self.persist()

    def evaluate_finished_domains(self):
        """Explicit separate invocation, protected by the same exclusive run lock."""
        for row in self.plan["domains"]:
            record = self.report["domains"].get(row["name"], {})
            if not record.get("candidate_checkpoint"):
                continue
            checkpoint = Path(record["candidate_checkpoint"])
            checkpoint_update_count(checkpoint)
            results = self.evaluate_pair(row, checkpoint, "offline_" + digest(checkpoint)[:12])
            record.update(assess_domain(results, row["required_case_names"]), assessment_status="completed",
                          offline_assessed_checkpoint_sha256=digest(checkpoint))
            self.persist()

    def run(self):
        started = time.monotonic()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a") as ownership:
            fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.authenticate()
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / "study_plan.json"
            if path.exists() and digest(path) != digest(self.args.contract):
                validate_plan_migration(json.loads(path.read_text()), self.plan, self.root)
                write_json(self.root / "revisions/v1_study_plan.json", json.loads(path.read_text()))
                self.copy_atomic(self.args.contract, "study_plan.json")
            if not path.exists():
                self.copy_atomic(self.args.contract, "study_plan.json")
            write_json(self.root / "preflight.json", validate_contracts(self.plan))
            self.open_account()
            elapsed_before = self.report["elapsed_seconds"]
            self._elapsed_base = elapsed_before
            self._session_started = started
            signal.signal(signal.SIGTERM, self.request_stop)
            signal.signal(signal.SIGINT, self.request_stop)
            try:
                self.report["status"] = "running"
                if getattr(self.args, "evaluate_only", False):
                    self.evaluate_finished_domains()
                    self.report["status"] = "offline_assessments_finished"
                else:
                    for row in self.plan["domains"]:
                        if self.stop_requested or time.monotonic() >= self.deadline:
                            raise InterruptedError("Campaign deadline/signal; budget remains charged")
                        self.run_domain(row)
                    self.report["status"] = ("task_domains_finished_unassessed" if self.plan["contract_id"] == REVISION_V2
                                             else "task_domains_finished")
            except InterruptedError as error:
                self.report.update(status="stopped", reason=str(error))
            except Exception:
                self.report.update(status="failed", error=traceback.format_exc())
                traceback.print_exc()
            finally:
                for process in (self.eval_child, self.child):
                    if process is not None and process.poll() is None:
                        process.send_signal(signal.SIGTERM)
                        try:
                            process.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                if self.active is not None:
                    for progress_path in (self.root / self.active["name"]).glob("attempt_*/train/progress.json"):
                        self.account(progress_path.parent, json.loads(progress_path.read_text()))
                self.report.update(finished_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=elapsed_before + time.monotonic()-started)
                self.report["ppo_process_active_seconds"] = max(0., self.report["training_process_seconds"] -
                    self.report.get("cooperative_assessment_seconds", 0.))
                self.report["non_training_seconds"] = max(0., self.report["elapsed_seconds"] -
                    self.report["ppo_process_active_seconds"])
                self.persist()
                write_json(self.root / "completion.json", self.report)
                write_json(self.root / "artifact_selection.json", dict(status=self.report["status"],
                    domains=self.report["domains"], accepted_stage=None, deployment_checkpoint=None,
                    automatic_promotion=False, independent_task_candidates=True))
                self.publish_terminal()
        return int(self.report["status"] == "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=20000)
    parser.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"))
    parser.add_argument("--max-runtime-seconds", type=float, default=180*86400.)
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="Assess sealed candidates without starting PPO")
    parser.add_argument("--stage", default="task_campaign")
    args = parser.parse_args()
    if not args.research or args.num_envs != 20000 or args.max_runtime_seconds <= 0:
        parser.error("Frozen task research requires 20000 environments and a positive runtime")
    if args.preflight_only:
        print(json.dumps(dict(status="passed", roles=len(validate_contracts(json.loads(args.contract.read_text()))))))
        return 0
    return TaskCampaign(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
