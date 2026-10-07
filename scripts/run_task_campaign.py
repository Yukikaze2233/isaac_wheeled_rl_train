#!/usr/bin/env python3
"""Run persistent task workers against one durable campaign account."""
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
    validate_plan, validate_task_resume, validate_plan_migration, validate_domain_review,
    REVISION_V2, REVISION_V9, REVISION_V10, REVISION_V11, REVISION_V12, REVISION_V13, REVISION_V14, REVISION_V15, REVISION_V16, REVISION_V17, REVISION_V18, REVISION_V19, REVISION_V20, REVISION_V21, REVISION_V22, REVISION_V23, REVISION_V24, SCHEDULE_REVISIONS, REVIEW_REVISIONS, CONTINUOUS_REVISIONS,
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
        if config.get("integrated_production"):
            from wheeled_tasks.chassis.integrated_production import production_source_files
            production_source_files(config, root=ROOT)
        elif ("small_stair_revision_migration" in metadata
                or "terrain_runout_revision_migration" in metadata or "terrain_route_runout" in config):
            from train_chassis import scut_curriculum_source_files
            scut_curriculum_source_files(config, root=ROOT)
        parent_matches = metadata["source_contract_sha256"] == parent_hashes[row["parent_domain"]]
        if not parent_matches and plan["contract_id"] in (REVISION_V12, REVISION_V13, REVISION_V14, REVISION_V15, REVISION_V16, REVISION_V17) and row["name"] == "stepjump":
            from wheeled_tasks.chassis.jump_landing_revision import validate_landing_parent_alias
            parent = next(parent for parent in plan["domains"] if parent["name"] == row["parent_domain"])
            parent_matches = validate_landing_parent_alias(json.loads((ROOT / parent["contract"]).read_text()), config, ROOT)
        if (metadata["domain"] != row["name"] or not parent_matches
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
            if role == "train" and config.get("integrated_production"):
                from wheeled_tasks.chassis.integrated_production import static_training_config
                effective = static_training_config(config, ROOT)
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
        if self.plan["contract_id"] == REVISION_V24:
            ledger["integrated_continuation"] = deepcopy(self.report["integrated_continuation"])
            ledger["joint_learning_sources"] = deepcopy(self.report.get("joint_learning_sources", {}))
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
        if self.plan["contract_id"] == REVISION_V24:
            return self.open_integrated_account()
        migration = None
        if self.claim.exists():
            ledger = json.loads(self.claim.read_text())
            if ledger["campaign_id"] != CAMPAIGN_ID or ledger["run_directory"] != str(self.root):
                raise ValueError("Campaign budget already belongs to a different run/revision")
            if ledger["contract_sha256"] != digest(self.args.contract):
                if self.plan["contract_id"] not in CONTINUOUS_REVISIONS:
                    raise ValueError("Campaign budget already belongs to a different run/revision")
                binding = self.plan["revision_migration"]
                manifest = json.loads((ROOT / binding["manifest_file"]).read_text())
                source = json.loads((ROOT / manifest["source_plan_file"]).read_text())
                migration = validate_plan_migration(source, self.plan, self.root)
                if ledger["contract_sha256"] != migration["source_plan_sha256"]:
                    raise ValueError("Campaign account is not the named revision source")
                previous_revision = source["contract_id"].rsplit("-", 1)[1]
                write_json(self.root / f"revisions/{previous_revision}_account_before_migration.json", ledger)
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
        elif self.plan["contract_id"] in CONTINUOUS_REVISIONS:
            raise ValueError("Task v2 requires the existing named campaign budget account")
        self.assert_no_live_workers()
        # The claim is committed before campaign_state. Replay the authenticated
        # migration receipt if a crash left the report on its predecessor.
        replay = None
        if self.plan["contract_id"] in (REVISION_V20, REVISION_V21, REVISION_V22, REVISION_V23) and migration is None:
            if self.plan["contract_id"] == REVISION_V23:
                from wheeled_tasks.chassis import small_stair_revision as recipe
            elif self.plan["contract_id"] == REVISION_V22:
                from wheeled_tasks.chassis import terrain_runout_revision as recipe
            elif self.plan["contract_id"] == REVISION_V21:
                from wheeled_tasks.chassis import recovery_entropy_revision as recipe
            else:
                from wheeled_tasks.chassis import recovery_script_revision as recipe
            source = recipe.read_bound(recipe.SOURCE_PLAN_FILE, recipe.SOURCE_PLAN_SHA, ROOT)
            expected = validate_plan_migration(source, self.plan, self.root)
            previous = source["contract_id"].rsplit("-", 1)[1]
            current = self.plan["contract_id"].rsplit("-", 1)[1]
            path = self.root / f"revisions/{previous}_to_{current}_account_migration.json"
            receipt = json.loads(path.read_text())
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError("Task restart source receipt differs from the frozen plan pair")
            report_sha = self.report.get("contract_sha256")
            if report_sha == recipe.SOURCE_PLAN_SHA:
                # Claim publication precedes the local report. Replay this one
                # named migration with the latest charged records; never reopen.
                migration, replay = expected, receipt
            elif report_sha != digest(self.args.contract):
                raise ValueError("Task account/report revision differs")
        if self.plan["contract_id"] == REVISION_V19 and migration is None:
            from wheeled_tasks.chassis import recovery_quality_revision as recipe
            source = recipe.read_bound(recipe.SOURCE_PLAN_FILE, recipe.SOURCE_PLAN_SHA, ROOT)
            expected = validate_plan_migration(source, self.plan, self.root)
            path = self.root / "revisions/v18_to_v19_account_migration.json"
            receipt = json.loads(path.read_text())
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError("Recovery quality restart source receipt differs from the frozen plan pair")
            report_sha = self.report.get("contract_sha256")
            if report_sha == recipe.SOURCE_PLAN_SHA:
                # Claim publication precedes the local report. Replay this one
                # named migration with the latest charged records; never reopen.
                migration, replay = expected, receipt
            elif report_sha != digest(self.args.contract):
                raise ValueError("Recovery quality account/report revision differs")
        if self.plan["contract_id"] == REVISION_V16 and migration is None:
            from wheeled_tasks.chassis.terrain_schedule_revision import SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, read_bound
            report_sha = self.report.get("contract_sha256")
            if report_sha == SOURCE_PLAN_SHA:
                source = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, ROOT)
                migration = validate_plan_migration(source, self.plan, self.root)
                path = self.root / "revisions/v15_to_v16_account_migration.json"
                replay = json.loads(path.read_text())
                if any(replay.get(key) != value for key, value in migration.items()):
                    raise ValueError("Terrain migration recovery receipt differs from the frozen plan pair")
            elif (report_sha != digest(self.args.contract)
                    or self.report.get("domains", {}).get("terrain", {}).get("reopening", {}).get("schema")
                    != "task-domain-reopen-v16"):
                raise ValueError("Terrain extension account lost its durable reopening state")
        if self.plan["contract_id"] in (REVISION_V17, REVISION_V18) and migration is None:
            if self.plan["contract_id"] == REVISION_V18:
                from wheeled_tasks.chassis import ordinary_posture_revision as recipe
            else:
                from wheeled_tasks.chassis import scut_curriculum_revision as recipe
            revision = self.plan["contract_id"].rsplit("-", 1)[1]
            path = self.root / f"revisions/v15_to_{revision}_account_migration.json"
            report_sha = self.report.get("contract_sha256")
            if report_sha == recipe.SOURCE_PLAN_SHA:
                source = recipe.read_bound(recipe.SOURCE_PLAN_FILE, recipe.SOURCE_PLAN_SHA, ROOT)
                migration = validate_plan_migration(source, self.plan, self.root)
                replay = json.loads(path.read_text())
                if any(replay.get(key) != value for key, value in migration.items()):
                    raise ValueError("SCUT migration recovery receipt differs from the frozen plan pair")
            elif (report_sha != digest(self.args.contract)
                    or any(self.report.get("domains", {}).get(domain, {}).get("reopening", {}).get("schema")
                           != "task-domain-reopen-" + revision for domain in recipe.ORIGINS)):
                raise ValueError("SCUT curriculum account lost its durable reopening state")
            else:
                # A same-revision restart must authenticate the original sealed
                # sources too; schema presence alone cannot authorize a rewrite.
                source = recipe.read_bound(recipe.SOURCE_PLAN_FILE, recipe.SOURCE_PLAN_SHA, ROOT)
                expected = validate_plan_migration(source, self.plan, self.root)
                receipt = json.loads(path.read_text())
                if any(receipt.get(key) != value for key, value in expected.items()):
                    raise ValueError("SCUT restart source receipt differs from the frozen plan pair")
                authenticated = recipe.reopen_completed_domains(self.report, self.records, source, self.plan, self.root)
                if receipt.get("domain_reopening") != authenticated:
                    raise ValueError("SCUT restart cannot replace any sealed learning source")
        self.report["contract_sha256"] = digest(self.args.contract)
        if migration is not None:
            migration.update(**account_updates(self.plan, self.records), attempts=deepcopy(self.records))
            if self.plan["contract_id"] in (REVISION_V13, REVISION_V14):
                from wheeled_tasks.chassis.nonstandard_start_revision import reopen_completed_domains
                migration["domain_reopening"] = reopen_completed_domains(self.report, self.records, source, self.plan, self.root)
            if self.plan["contract_id"] == REVISION_V16:
                from wheeled_tasks.chassis.terrain_schedule_revision import reopen_completed_terrain
                migration["domain_reopening"] = reopen_completed_terrain(self.report, self.records, source, self.plan, self.root)
                if replay is not None and replay.get("domain_reopening") != migration["domain_reopening"]:
                    raise ValueError("Terrain migration recovery cannot replace its sealed learning source")
            if self.plan["contract_id"] in (REVISION_V17, REVISION_V18):
                if self.plan["contract_id"] == REVISION_V18:
                    from wheeled_tasks.chassis import ordinary_posture_revision as recipe
                else:
                    from wheeled_tasks.chassis import scut_curriculum_revision as recipe
                migration["domain_reopening"] = recipe.reopen_completed_domains(self.report, self.records, source, self.plan, self.root)
                if replay is not None and replay.get("domain_reopening") != migration["domain_reopening"]:
                    raise ValueError("SCUT migration recovery cannot replace any sealed learning source")
            # V15 retains the v14 reopening state and clocks; it creates no new window.
            self.report["revision_migration"] = migration
            revision = self.plan["contract_id"].rsplit("-", 1)[1]
            previous_revision = source["contract_id"].rsplit("-", 1)[1]
            write_json(self.root / f"revisions/{previous_revision}_to_{revision}_account_migration.json", migration)
        self.persist()

    def open_integrated_account(self):
        """Rebase under the existing exclusive account lock, exactly once."""
        from wheeled_tasks.chassis import integrated_production as joint
        from wheeled_tasks.chassis import integrated_lineage as lineage
        if not self.claim.exists():
            raise ValueError("Joint continuation requires the original durable campaign account")
        ledger = json.loads(self.claim.read_text())
        plan_sha = digest(self.args.contract)
        if (ledger.get("campaign_id") != CAMPAIGN_ID or ledger.get("run_directory") != str(self.root)
                or ledger.get("contract_sha256") not in (joint.SOURCE_PLAN_SHA, plan_sha)):
            raise ValueError("Joint account belongs to another run or unnamed predecessor")
        receipt_path = self.root / "revisions/v21_to_v24_account_migration.json"
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        state_path = self.root / "campaign_state.json"
        if state_path.exists():
            self.report.update(json.loads(state_path.read_text()))
        elif previous is not None:
            self.report.update(deepcopy(previous["report_before_migration"]))
        else:
            raise ValueError("Joint boundary lacks the normal REC completion state")
        source = json.loads((ROOT / joint.SOURCE_PLAN_FILE).read_text())
        migration = validate_plan_migration(source, self.plan, self.root)
        if previous is not None and any(previous.get(k) != v for k, v in migration.items()):
            raise ValueError("Joint restart migration receipt differs from the frozen pair")
        self.records = joint.reconcile_records(self.plan, self.root, ledger["attempts"])
        self.assert_no_live_workers()
        binding = joint.binding_for(self.plan, ROOT, self.root)
        rec = self.report["domains"]["recovery"]
        entry = lineage.admit_entry(binding, dict(path=rec["candidate_checkpoint"], sha256=rec["candidate_checkpoint_sha256"]), rec)
        if previous is not None and previous["entry_admission"] != entry:
            raise ValueError("Joint restart would replace the normal REC origin")
        if ledger["contract_sha256"] == joint.SOURCE_PLAN_SHA:
            # Actual admission happens only here, after the old owner is gone.
            initial = joint.prepare_admission(self.plan, ROOT, self.root, account_path=self.claim, state_path=state_path)
            archive_path = self.root / "revisions/v21_account_before_migration.json"
            if not archive_path.exists():
                write_json(archive_path, ledger)
            before = deepcopy(self.report)
            if previous is None:
                previous = dict(**migration, entry_admission=entry,
                    initial_admission=initial, report_before_migration=before,
                    attempts_before_migration=deepcopy(self.records), **account_updates(self.plan, self.records))
                write_json(receipt_path, previous)
        elif previous is None:
            raise ValueError("Joint claim lacks its durable migration/REC admission receipt")
        self.report = joint.archive_legacy_domains(self.report, binding, entry)
        if ledger.get("integrated_continuation") not in (None, self.report["integrated_continuation"]):
            raise ValueError("Joint account and normal REC origin differ")
        # The authoritative claim can be newer than a local campaign report.
        for domain, selection in ledger.get("joint_learning_sources", {}).items():
            existing = self.report.setdefault("joint_learning_sources", {}).get(domain)
            if existing is not None and existing["checkpoint_consumed_updates"] > selection["checkpoint_consumed_updates"]:
                raise ValueError("Joint claim learning-source floor moved backwards")
            self.report["joint_learning_sources"][domain] = deepcopy(selection)
        self.report.update(contract_sha256=plan_sha, revision_migration=deepcopy(previous))
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
        # Worker receipts are atomically replaced. Polling an unchanged receipt
        # must not fsync the account or recopy a large startup report every second.
        published = getattr(self, "_published_files", {})
        self._published_files = published
        def stamp(source):
            stat = source.stat()
            return (str(source), stat.st_ino, stat.st_mtime_ns, stat.st_size)
        current = stamp(path)
        if published.get("progress.json") != current:
            progress = json.loads(path.read_text())
            self.account(directory, progress)
            progress.update(successful_updates=self.report["consumed_updates"], parent_updates=0,
                actor_updates_consumed=self.report["actor_updates_consumed"],
                worker_pid=progress.get("pid"), pid=os.getpid(), active_domain=self.active["name"],
                phase="training", orchestration=("single_policy_joint_campaign" if self.plan["contract_id"] == REVISION_V24
                                                    else "independent_task_domain_campaign"))
            write_json(self.root / "progress.json", progress)
            published["progress.json"] = current
        for name in ("startup.json", "behavior_metrics.json", "torque_monitor.json", "live_state.json"):
            source = directory / name
            if source.exists():
                current = stamp(source)
                if published.get(name) != current:
                    self.copy_atomic(source, name)
                    published[name] = current
        if self.plan["contract_id"] not in CONTINUOUS_REVISIONS:
            self.assess_request(directory)
        elif self.plan["contract_id"] in REVIEW_REVISIONS:
            self.review_request(directory)
        return True

    def review_request(self, directory=None):
        """Publish review availability and poll a small receipt without pausing PPO."""
        if self.plan["contract_id"] == REVISION_V24:
            return
        if self.plan["contract_id"] not in REVIEW_REVISIONS or getattr(self, "_active_review", None) is not None:
            return
        progress = json.loads((directory / "progress.json").read_text()) if directory is not None else {}
        bucket = progress.get("task_actor_update", 0) // self.plan["domain_completion"]["review_interval_actor_updates"]
        snapshots = directory / "checkpoints" if directory is not None else None
        identity = (self.active["name"], snapshots.stat().st_mtime_ns if snapshots is not None and snapshots.exists() else None)
        # Scan only this worker, once per new review window. Never repeatedly
        # hash a 33 MB checkpoint at the publisher's one-second polling rate.
        if bucket and identity != getattr(self, "_review_available", None):
            seals = [p for p in snapshots.glob("**/*.checkpoint.json")
                     if not any(part.endswith(".pending") for part in p.parts)]
            if seals:
                path = max(seals, key=lambda p: p.stat().st_mtime_ns)
                seal = json.loads(path.read_text())
                if seal.get("contract_sha256") == self.active["sha256"]:
                    checkpoint = path.with_suffix("").with_suffix(".pt")
                    checkpoint_update_count(checkpoint)
                    self._review_available = identity
                    write_json(self.root / "review_available.json", dict(domain=self.active["name"],
                        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                        task_actor_update=seal["task_actor_update"], consumed_updates=self.report["consumed_updates"],
                        status="review_available_training_continues", domain_qualified=None))
        path = self.root / "domain_review_request.json"
        if not path.exists():
            return
        token = digest(path)
        if token == getattr(self, "_review_request_token", None) or token in self.report.get("consumed_review_tokens", []):
            return
        self._review_request_token = token
        try:
            request = validate_domain_review(self.plan, self.root, self.active,
                json.loads(path.read_text()), self.report["consumed_updates"])
        except (ValueError, KeyError, TypeError, OSError) as error:
            write_json(self.root / "domain_review_response.json", dict(token=token,
                status="invalid_review_training_continues", reason=str(error)))
            return
        # Durable before the signal: a crash during sealing still retains the
        # reviewed finish on same-domain recovery, with no refund or rollback.
        request["token"] = token
        self._active_review = request
        self.report["domains"].setdefault(self.active["name"], {})["pending_review"] = request
        self.persist()
        write_json(self.root / "domain_review_response.json", dict(token=token, status="sealing_latest_learning_state"))
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)

    def assess_request(self, directory):
        if self.plan["contract_id"] in CONTINUOUS_REVISIONS:
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
        if self.plan["contract_id"] == REVISION_V24 and row["name"] in ("speed", "terrain", "jump", "stepjump"):
            from wheeled_tasks.chassis import integrated_production as joint
            self.report["active_joint_domain"] = row["name"]
            floor = self.report.get("joint_learning_sources", {}).get(row["name"])
            self._joint_binding, self._joint_selection = joint.select_source(
                self.plan, ROOT, self.root, self.report, previous_selection=floor)
            self.report.setdefault("joint_learning_sources", {})[row["name"]] = deepcopy(self._joint_selection)
            self.persist()
            return Path(self._joint_selection["checkpoint"]["path"])
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
        command = [sys.executable, "-B", str(ROOT / "scripts/train_chassis.py"),
            "--contract", str(ROOT / row["contract"]), "--stage", row["kind"], "--research", "--publish-state",
            "--num-envs", str(self.plan["target_num_envs"]), "--updates", str(updates), "--seed", str(self.plan["seed"]),
            "--block-index", "0", "--device", self.args.device, "--run-dir", str(directory),
            "--resume", str(source), "--resume-task-change", "--learning-rate-scale", "1.0",
            "--consumed-updates", str(self.report["consumed_updates"]),
            "--actor-updates-consumed", str(self.report["actor_updates_consumed"]),
            "--stage-actor-update", str(seal["stage_actor_update"]),
            "--task-actor-update", str(seal["task_actor_update"] if same else 0),
            "--confirmed-jump-bins-json", json.dumps(bins),
            "--global-actor-update-offset", "0", "--max-runtime-seconds", str(max(1., self.deadline-time.monotonic()))]
        if self.plan["contract_id"] == REVISION_V24 and row["name"] in ("speed", "terrain", "jump", "stepjump"):
            from wheeled_tasks.chassis import integrated_production as joint
            receipt = joint.make_admission(self.plan, ROOT, self.root, self._joint_selection, self.records,
                                           account_path=self.claim, state_path=self.root / "campaign_state.json",
                                           snapshot_directory=directory.parent)
            if (receipt["selection"]["checkpoint"]["path"] != str(source)
                    or receipt["allowance"]["requested_total_updates"] != updates
                    or not receipt["allowance"]["admitted"]):
                raise ValueError("Joint worker command differs from its actual parent and allowance")
            path = directory.parent / "integrated_source_selection.json"
            write_json(path, receipt)
            command += ["--integrated-source-selection-file", str(path),
                        "--integrated-source-selection-sha256", digest(path)]
        return command

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
        if self.plan["contract_id"] in CONTINUOUS_REVISIONS:
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
        if self.plan["contract_id"] == REVISION_V24 and row["name"] in ("speed", "terrain", "jump", "stepjump"):
            return self.run_integrated_domain(row)
        self.active = row
        spent = self.report["domain_consumed_updates"][row["name"]]
        existing = self.report["domains"].get(row["name"], {})
        if (spent >= row["updates"] or existing.get("status") in (
                "domain_review_finished_unassessed", "domain_schedule_finished_unassessed")) and existing.get("candidate_checkpoint"):
            checkpoint_update_count(existing["candidate_checkpoint"])
            return
        source = self.learning_source(row)
        remaining = row["updates"] - spent
        scheduled = self.plan["contract_id"] in SCHEDULE_REVISIONS
        endpoint = None
        if scheduled:
            import torch
            from train_chassis import resume_critic_warmup
            # Read once at domain entry, outside the rollout/optimizer hot path.
            # Use the same adaptation rule as the native worker, so critic-only
            # work is charged but never advances the curriculum's actor clock.
            infos = torch.load(source, map_location="cpu", weights_only=False)["infos"]
            config = json.loads((ROOT / row["contract"]).read_text())
            task_clock = (infos["task_actor_update"] if
                          infos.get("task_campaign", {}).get("domain") == row["name"] else 0)
            endpoint = self.plan["domain_completion"]["schedule"]["end_actor_updates"][row["name"]]
            warmup = resume_critic_warmup(config, infos, contract_sha256=row["sha256"], task_change=True)
            if self.plan["contract_id"] == REVISION_V16 and row["name"] == "terrain" and (
                    infos.get("contract_sha256") != row["sha256"] or warmup != 0):
                raise ValueError("Plan-only terrain extension cannot repeat critic adaptation or change its worker")
            remaining = min(remaining, max(0, endpoint - task_clock) + (warmup if task_clock < endpoint else 0))
        self._active_review = existing.get("pending_review")
        # A sealed predecessor can be reviewed during a named migration without
        # rebuilding its scene merely to stop it again. The same evidence and
        # checkpoint authentication used by a running worker still applies.
        if self.plan["contract_id"] in (REVISION_V9, REVISION_V10, REVISION_V11, REVISION_V12, REVISION_V13, REVISION_V14, REVISION_V15, REVISION_V16, REVISION_V17, REVISION_V18, REVISION_V19, REVISION_V20, REVISION_V21, REVISION_V22, REVISION_V23) and self._active_review is None:
            self.review_request()
        if self._active_review is not None:
            validate_domain_review(self.plan, self.root, row, self._active_review, self.report["consumed_updates"])
            if checkpoint_update_count(source) < checkpoint_update_count(self._active_review["checkpoint"]):
                raise ValueError("Pending reviewed finish cannot roll back the learning source")
            remaining = 0
        if remaining:
            index = len(list((self.root / row["name"]).glob("attempt_*")))
            directory = self.root / row["name"] / f"attempt_{index:03d}" / "train"
            directory.parent.mkdir(parents=True, exist_ok=False)
            command = self.train_command(row, directory, source, remaining)
            write_json(directory.parent / "invocation.json", dict(command=command,
                parent_role="unassessed_research_candidate_not_accepted", source_checkpoint=str(source),
                source_checkpoint_sha256=digest(source), requested_updates=remaining,
                scheduled_end_actor_update=endpoint))
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
            if (code or result.get("status") not in ("completed", "stopped")
                    or result.get("successful_updates", 0) < remaining and self._active_review is None):
                raise RuntimeError(f"Continuous task worker stopped before cap; consumed work retained: {row['name']}")
            source = directory / "model_final.pt"
            checkpoint_update_count(source)
        seal = json.loads(source.with_suffix(".checkpoint.json").read_text()) if scheduled else {}
        schedule_finished = scheduled and seal.get("task_actor_update", 0) >= endpoint
        if scheduled and not schedule_finished and self._active_review is None and self.report["domain_consumed_updates"][row["name"]] < row["updates"]:
            raise RuntimeError("Task worker completed before its frozen actor endpoint or charged cap")
        self.report["domains"][row["name"]] = dict(status=("domain_review_finished_unassessed" if self._active_review else
            "domain_schedule_finished_unassessed" if schedule_finished else "domain_budget_exhausted_unassessed"),
            domain_qualified=None, assessment_status="not_run", accepted_stage=None,
            deployment_checkpoint=None, final_checkpoint=str(source), candidate_checkpoint=str(source),
            candidate_checkpoint_sha256=digest(source), parent_domain=row["parent_domain"],
            consumed_updates=self.report["domain_consumed_updates"][row["name"]],
            scope="unassessed_individual_task_research_candidate")
        for key in ("completion_history", "reopening"):
            if key in existing:
                self.report["domains"][row["name"]][key] = deepcopy(existing[key])
        if self._active_review is not None:
            self.report["domains"][row["name"]]["finish_review"] = self._active_review
        if scheduled:
            self.report["domains"][row["name"]]["schedule_completion"] = dict(
                clock="domain_actor_updates", end_actor_update=endpoint,
                actual_actor_update=seal.get("task_actor_update", 0),
                reason="iteration_endpoint" if schedule_finished else "review" if self._active_review else "charged_cap",
                automatic_advance=True, domain_qualified=None)
        self.persist()

    def run_integrated_domain(self, row):
        """Reuse the original worker/process/account path with fresh joint clocks."""
        from wheeled_tasks.chassis import integrated_production as joint
        from wheeled_tasks.chassis import integrated_lineage as lineage
        self.active = row
        self._active_review = None
        existing = deepcopy(self.report["domains"].get(row["name"], {}))
        source = self.learning_source(row)
        selection = self._joint_selection
        binding = self._joint_binding
        endpoint = self.plan["domain_completion"]["schedule"]["end_actor_updates"][row["name"]]
        if not selection["domain_complete"]:
            allowance = lineage.remaining_budget(binding, selection, self.records)
            actor_requested = min(max(0, endpoint-selection["fresh_domain_actor_updates"]), allowance["maximum_actor_updates"])
            allowance = lineage.remaining_budget(binding, selection, self.records, requested_actor_updates=actor_requested)
            if not allowance["admitted"]:
                raise RuntimeError("Charged old domain/global cap leaves no new joint segment; legacy completion cannot substitute")
            remaining = allowance["requested_total_updates"]
            index = len(list((self.root / row["name"]).glob("attempt_*")))
            directory = self.root / row["name"] / f"attempt_{index:03d}" / "train"
            directory.parent.mkdir(parents=True, exist_ok=False)
            command = self.train_command(row, directory, source, remaining)
            write_json(directory.parent / "invocation.json", dict(command=command,
                parent_role="single_policy_joint_full_learning_state_unassessed", source_checkpoint=str(source),
                source_checkpoint_sha256=digest(source), requested_updates=remaining,
                scheduled_end_actor_update=endpoint, fresh_actor_clock=True))
            started = time.monotonic()
            code = self.execute(command, directory.parent / "train.log", directory)
            self.report["training_process_seconds"] += time.monotonic() - started
            path = directory / "completion.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            if "successful_updates" in result:
                self.account(directory, result)
            self.persist()
            if self.stop_requested or time.monotonic() >= self.deadline:
                raise InterruptedError("Joint campaign stopped; all observed work remains charged")
            if code or result.get("status") != "completed" or result.get("successful_updates", 0) != remaining:
                raise RuntimeError("Joint worker has not normally completed its admitted actor work and charged critic adaptation")
            source = self.learning_source(row)
            selection = self._joint_selection
            if not selection["domain_complete"]:
                raise ValueError("Joint worker final lacks a sealed normal domain completion")
        seal = json.loads(source.with_suffix(".checkpoint.json").read_text())
        meta = seal["integrated_lineage"]
        schedule_finished = meta["domain_actor_updates"] >= endpoint
        if not schedule_finished and self.report["domain_consumed_updates"][row["name"]] < row["updates"] and self.report["consumed_updates"] < self.plan["total_ceiling"]:
            raise ValueError("Joint final has reached neither its fresh actor endpoint nor charged cap")
        record = dict(status="joint_domain_finished_unassessed", domain_qualified=None,
            assessment_status="not_run", accepted_stage=None, deployment_checkpoint=None,
            final_checkpoint=str(source), candidate_checkpoint=str(source), candidate_checkpoint_sha256=digest(source),
            parent_domain=row["parent_domain"], consumed_updates=self.report["domain_consumed_updates"][row["name"]],
            integrated_lineage=deepcopy(meta), scope="single_policy_joint_learning_candidate_unassessed",
            legacy_completion_history=deepcopy(existing.get("legacy_completion_history", [])),
            schedule_completion=dict(clock="fresh_joint_domain_actor_updates", end_actor_update=endpoint,
                actual_actor_update=meta["domain_actor_updates"], reason="iteration_endpoint" if schedule_finished else "charged_cap",
                automatic_advance=True, domain_qualified=None))
        checkpoint = lineage._checkpoint(binding, selection["checkpoint"], row["name"], row["sha256"])
        lineage._finished(binding, checkpoint, record)
        self.report["domains"][row["name"]] = record
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
                previous = json.loads(path.read_text())
                revision = previous["contract_id"].rsplit("-", 1)[1]
                write_json(self.root / f"revisions/{revision}_study_plan.json", previous)
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
                self.report.pop("reason", None)
                self.report.pop("error", None)
                if getattr(self.args, "evaluate_only", False):
                    self.evaluate_finished_domains()
                    self.report["status"] = "offline_assessments_finished"
                else:
                    for row in self.plan["domains"]:
                        if self.stop_requested or time.monotonic() >= self.deadline:
                            raise InterruptedError("Campaign deadline/signal; budget remains charged")
                        self.run_domain(row)
                    self.report["status"] = ("task_domains_finished_unassessed" if self.plan["contract_id"] in CONTINUOUS_REVISIONS
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
                    automatic_promotion=False, independent_task_candidates=self.plan["contract_id"] != REVISION_V24))
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
    plan = json.loads(args.contract.read_text())
    validate_plan(plan)
    if not args.research or args.num_envs != plan["target_num_envs"] or args.max_runtime_seconds <= 0:
        parser.error("Frozen task research requires the declared environment count and a positive runtime")
    if args.preflight_only:
        print(json.dumps(dict(status="passed", roles=len(validate_contracts(json.loads(args.contract.read_text()))))))
        return 0
    return TaskCampaign(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
