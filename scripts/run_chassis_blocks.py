#!/usr/bin/env python3
"""Own PPO/evaluation blocks with either monitoring or explicit capability gates."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class TrainingBlocks:
    def __init__(self, args):
        self.args = args
        self.contract = json.loads(args.contract.read_text())
        # Gate/protection bookkeeping is runtime state, never a mutation of the
        # hash-pinned training or evaluation contract.
        self.evaluation_settings = deepcopy(self.contract.get("evaluation", {}))
        self.new_asset = bool(self.contract.get("new_asset_curriculum", {}).get("enabled"))
        self.root = args.run_dir
        self.worker_root = Path(getattr(args, "worker_source", None) or ROOT).resolve()
        self.child = None
        self.stop_requested = False
        self.deadline = time.monotonic() + args.max_runtime_seconds
        initial_checkpoint = getattr(args, "resume", None) or args.transfer
        self.report = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
            "successful_updates": 0, "blocks": [], "consecutive_evaluation_passes": 0,
            "contract_id": self.contract["contract_id"],
            "worker_source": str(self.worker_root),
            "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None}
        if self.new_asset:
            protected = json.loads(getattr(args, "protected_cases_json", "[]"))
            confirmed = json.loads(getattr(args, "confirmed_jump_bins_json", "[]"))
            self._validate_runtime_protection(protected, confirmed)
            self.evaluation_settings["protected_case_names"] = sorted(set(
                self.evaluation_settings.get("protected_case_names", [])) | set(protected))
            self.report.update(protected_case_names=self.evaluation_settings["protected_case_names"],
                confirmed_jump_bins_m=sorted(set(confirmed)), actor_updates_consumed=0,
                global_actor_update_offset=getattr(args, "global_actor_update_offset", 0))

    def _validate_runtime_protection(self, protected, confirmed):
        names = {case["name"] for case in self.evaluation_settings.get("cases", [])}
        bins = self.contract["new_asset_curriculum"].get("jump_curriculum", {}).get("bins_m", [])
        if (not isinstance(protected, list) or any(not isinstance(name, str) for name in protected)
                or set(protected) - names):
            raise ValueError("Runtime protection must name canonical evaluation cases")
        if (not isinstance(confirmed, list) or any(type(value) not in (int, float)
                or not math.isfinite(value) or value not in bins for value in confirmed)):
            raise ValueError("Confirmed jump bins must belong to the frozen jump curriculum")

    def record_confirmed_jump_bins(self, primary, confirmation):
        """Advance a bin only from two real unassisted canonical jump passes."""
        if not self.new_asset:
            return
        settings = self.evaluation_settings
        if settings.get("seed") == settings.get("confirmation_seed"):
            raise ValueError("Jump-bin confirmation requires two distinct evaluation seeds")
        bins = self.contract["new_asset_curriculum"].get("jump_curriculum", {}).get("bins_m", [])
        confirmed = set(self.report.get("confirmed_jump_bins_m", []))
        for case in settings["cases"]:
            # A running or airborne-reset variant cannot certify the ordinary
            # cold standing request for the bin's expansion schedule.
            if (case.get("task") != "jump" or case.get("skill", {}).get("kind") != "jump"
                    or case.get("jump_assistance_enabled") is not False):
                continue
            apex = case.get("skill", {}).get("semantics", {}).get("jump_apex_delta_m")
            if apex in bins and all(result.get("cases", {}).get(case["name"], {}).get("passed")
                                    for result in (primary, confirmation)):
                confirmed.add(apex)
        self.report["confirmed_jump_bins_m"] = sorted(confirmed)

    def establish_scratch_baseline(self, checkpoint):
        """Evaluate the saved real update-zero state, even after the first block."""
        from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count
        checkpoint = Path(checkpoint)
        if checkpoint_update_count(checkpoint) != 0:
            raise ValueError("Scratch baseline must be the authentic update-zero checkpoint")
        settings = self.evaluation_settings
        baseline_dir = self.root / "baseline_evaluation"
        baseline = self.evaluate_actor(checkpoint, baseline_dir, full=True)
        if baseline is None:
            return False
        confirmation_dir = self.root / "baseline_confirmation"
        confirmation = self.evaluate_actor(checkpoint, confirmation_dir,
            seed=settings["confirmation_seed"], full=True)
        if confirmation is None:
            return False
        expected = {case["name"] for case in settings["cases"]}
        if any(set(result.get("cases", {})) != expected for result in (baseline, confirmation)):
            raise ValueError("Scratch baseline must evaluate the full canonical manifest on both seeds")
        settings["initial_passed_case_names"] = sorted(name for name in expected
            if baseline["cases"][name]["passed"] and confirmation["cases"][name]["passed"])
        settings["protected_case_names"] = sorted(set(settings.get("protected_case_names", []))
            | set(settings["initial_passed_case_names"]))
        self.report.update(baseline_evaluation=baseline, baseline_confirmation=confirmation,
            baseline_checkpoint=str(checkpoint.resolve()), baseline_learning_updates=0,
            protected_case_names=settings["protected_case_names"],
            retention_checkpoint=str(checkpoint.resolve()), retention_learning_checkpoint=str(checkpoint.resolve()))
        self.record_confirmed_jump_bins(baseline, confirmation)
        self.copy_atomic(checkpoint, "baseline_actor.pt")
        self.copy_atomic(baseline_dir / "evaluation.json", "baseline_evaluation.json")
        self.save_retention_state(settings)
        return True

    def checkpoint_stage_actor_update(self, checkpoint):
        from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count
        checkpoint_update_count(checkpoint)  # Authenticate the checkpoint and contract seal first.
        seal = json.loads(Path(checkpoint).with_suffix(".checkpoint.json").read_text())
        clock = seal.get("stage_actor_update")
        if type(clock) is not int or clock < 0:
            raise ValueError("New campaign checkpoint lacks its nonnegative learning course clock")
        offset = self.report["global_actor_update_offset"]
        if seal.get("global_actor_update_offset") != offset:
            raise ValueError("Resume changes the global actor offset; pass the original stage offset")
        if clock > self.report["actor_updates_consumed"]:
            raise ValueError("Learning course clock exceeds the charged actor budget")
        return clock

    def inspect_zero_airtime(self, candidate, checkpoint, evaluation_directory):
        """One no-assistance evaluation diagnostic; never a stop or budget grant."""
        if not self.new_asset or self.contract.get("curriculum_stage") != "S5":
            return None
        threshold = self.contract["new_asset_curriculum"].get("jump_curriculum", {}).get("zero_airtime_actor_updates", 80)
        if (self.report.get("zero_airtime_checkpoint") is not None
                or self.report["actor_updates_consumed"] < threshold):
            return None
        cases, missing, attempted, airborne, sustained_airborne = {}, [], [], [], []
        def finite_json(value):
            if isinstance(value, dict):
                return {key: finite_json(item) for key, item in value.items()}
            if isinstance(value, list):
                return [finite_json(item) for item in value]
            return None if isinstance(value, float) and not math.isfinite(value) else value
        for spec in self.evaluation_settings["cases"]:
            if spec.get("task") != "jump" or spec.get("skill", {}).get("kind") != "jump":
                continue
            name = spec["name"]
            data = candidate.get("cases", {}).get(name, {})
            requested, flew = data.get("jump_episodes_requested"), data.get("jump_episodes_airborne_60ms")
            peak = data.get("jump_air_time_peak")
            valid = (spec.get("jump_assistance_enabled") is False
                and type(requested) is int and requested >= 0 and type(flew) is int and 0 <= flew <= requested
                and type(peak) in (int, float) and math.isfinite(peak) and peak >= 0
                # Metric counters compare float32 airtime against float32 .06.
                and ((flew > 0 and peak >= .06 - 1e-7) or (flew == 0 and peak < .06 + 1e-7))
                and type(data.get("observed_frames")) is int and data["observed_frames"] > 0)
            cases[name] = {key: data.get(key) for key in ("jump_episodes_requested", "jump_episodes_takeoff_entered",
                "jump_episodes_airborne_60ms", "jump_air_time_peak", "jump_clearance_peak",
                "jump_episodes_positive_com_release", "jump_com_release_speed", "jump_com_rise",
                "episodes", "observed_frames", "peak_motor_torque_nm", "rms_motor_torque_nm", "termination_reasons")}
            cases[name]["reference_phase_frames"] = {str(phase): data.get(f"reference_phase_{phase}_frames")
                                                        for phase in range(9)}
            cases[name] = finite_json(cases[name])
            if not valid:
                missing.append(name)
            elif requested:
                attempted.append(name)
                if peak > 0:
                    airborne.append(name)
                if flew:
                    sustained_airborne.append(name)
        if not cases:
            missing.append("canonical_cold_jump_cases")
        status = ("diagnostic_data_missing" if missing else "unassisted_airtime_observed" if airborne
                  else "zero_unassisted_airtime" if attempted else "no_valid_jump_requests")
        receipt = {"schema": "v6-zero-airtime-evaluation-diagnostic-v1", "status": status,
            "scope": "unassisted_fixed_evaluation_not_training_assistance_effectiveness",
            "trigger_actor_updates_consumed": threshold,
            "actor_updates_consumed": self.report["actor_updates_consumed"],
            "stage_actor_update": self.checkpoint_stage_actor_update(checkpoint),
            "global_actor_update_offset": self.report["global_actor_update_offset"],
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "evaluation_directory": str(Path(evaluation_directory).resolve()),
            "evaluation_seed": self.evaluation_settings["seed"], "assistance_enabled": False,
            "missing_data_cases": missing, "attempted_cases": attempted, "airborne_cases": airborne,
            "airborne_at_least_60ms_cases": sustained_airborne,
            "case_measurements": cases,
            "failed_case_info": {name: data for name, data in cases.items() if name not in airborne},
            "suggested_trace_fields": ["commands", "jump_requested", "reference_phase", "physical_phase",
                "previous_policy_action", "motor_effort", "requested_motor_effort", "motor_torque_bounds",
                "whole_com_vz", "jump_clearance_peak", "jump_air_time_peak", "spring_compression"],
            "trace_followup": "Inspect available recorded command/phase/action/torque/COM/spring traces; missing fields require an explicitly bounded diagnostic, not invented values.",
            "changes_training": False, "automatic_stop": False, "additional_updates_granted": 0}
        path = self.root / "zero_airtime_diagnostic.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        self.report["zero_airtime_checkpoint"] = receipt
        self.save_retention_state(self.evaluation_settings)
        print("V6_ZERO_AIRTIME_DIAGNOSTIC", json.dumps({"status": status,
            "actor_updates_consumed": receipt["actor_updates_consumed"],
            "stage_actor_update": receipt["stage_actor_update"], "receipt": str(path)}), flush=True)
        return receipt

    def request_stop(self, signum, _frame):
        self.stop_requested = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signum)

    def copy_atomic(self, source, name):
        temporary = self.root / (name + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(self.root / name)
        if name.endswith(".pt"):
            for suffix in (".checkpoint.json", ".contract.json"):
                sidecar = Path(source).with_suffix(suffix)
                if sidecar.exists():
                    destination = (self.root / name).with_suffix(suffix)
                    shutil.copyfile(sidecar, destination)

    def publish(self, directory):
        path = directory / "progress.json"
        progress_found = path.exists()
        if progress_found:
            try:
                progress = json.loads(path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                return False
            self.report["successful_updates"] = progress["successful_updates"] + progress["parent_updates"]
            progress.update(pid=os.getpid(), worker_pid=progress.get("worker_pid", progress["pid"]), phase=progress.get("phase", "training"),
                            successful_updates=self.report["successful_updates"], parent_updates=0,
                            orchestration="train_then_fixed_evaluate", active_block=progress.get("active_block", directory.name))
            temporary = self.root / "progress.tmp"
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(self.root / "progress.json")
        for name in ("torque_monitor.json", "behavior_metrics.json", "live_state.json", "startup.json"):
            if (directory / name).exists():
                self.copy_atomic(directory / name, name)
        return progress_found

    def execute(self, command, log_path, training_directory=None):
        with log_path.open("x") as stream:
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            progress_path = self.root / "progress.json"
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
                "successful_updates": self.report["successful_updates"], "parent_updates": 0,
                "training_transitions": self.report["successful_updates"] * self.args.num_envs * self.contract.get("num_steps_per_env", 0),
                "num_envs": self.args.num_envs, "stage": self.args.stage}
            progress.update(pid=os.getpid(), worker_pid=self.child.pid,
                phase="training" if training_directory is not None else "fixed_evaluation",
                updated_at=datetime.now(timezone.utc).isoformat())
            temporary = self.root / "progress.phase.tmp"
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(progress_path)
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

    def evaluate_actor(self, checkpoint, directory, *, seed=None, export_policy=False, full=False, contract=None):
        command = [sys.executable, "-B", str(self.worker_root / "scripts/evaluate_chassis.py"),
            "--contract", str((contract or self.args.contract).resolve()), "--checkpoint", str(Path(checkpoint).resolve()),
            "--device", self.args.device, "--output", str(directory)]
        if seed is not None:
            command += ["--seed", str(seed)]
        if export_policy:
            command.append("--export-policy")
        settings = self.evaluation_settings if contract is None else json.loads(contract.read_text())["evaluation"]
        role = ("confirmation" if seed == settings.get("confirmation_seed") else
                "baseline" if "baseline" in directory.name else "full" if full else "quick")
        if self.new_asset:
            # All roles retain the same canonical suite. This avoids a hidden
            # baseline/confirmation coverage difference; role and seed remain explicit.
            full = True
            command += ["--role", role]
        if settings.get("quick_checks") and not full:
            from wheeled_tasks.chassis.evaluation import quick_evaluation_cases
            names = quick_evaluation_cases(settings)
            settings["retention_case_names"] = names
            command += ["--cases", *names, "--no-traces"]
        code = self.execute(command, directory.with_suffix(".log"))
        if self.stop_requested:
            return None
        if code != 0:
            raise RuntimeError(f"Fixed evaluation process failed with {code}")
        return json.loads((directory / "evaluation.json").read_text())["candidates"][0]

    def save_retention_state(self, settings):
        state = {key: settings.get(key, []) for key in (
            "protected_case_names", "initial_passed_case_names", "promotion_case_names")}
        state["baseline_confirmation"] = self.report["baseline_confirmation"]
        state["clock_recovery_case_names"] = self.report.get("clock_recovery_case_names", [])
        state["response_recovery_case_names"] = self.report.get("response_recovery_case_names", [])
        for key in ("response_retention_comparison", "response_review_start_consumed_updates"):
            if key in self.report:
                state[key] = self.report[key]
        state["retention_learning_checkpoint"] = self.report.get("retention_learning_checkpoint")
        if self.new_asset:
            state.update(schema="chassis-runtime-retention-v1",
                contract_sha256=hashlib.sha256(self.args.contract.read_bytes()).hexdigest(),
                confirmed_jump_bins_m=self.report.get("confirmed_jump_bins_m", []),
                actor_updates_consumed=self.report["actor_updates_consumed"],
                global_actor_update_offset=self.report["global_actor_update_offset"],
                zero_airtime_checkpoint=self.report.get("zero_airtime_checkpoint"))
        temporary = self.root / "retention_state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(self.root / "retention_state.json")

    def record_response_review(self, primary, confirmation):
        """Track adaptation against the new plant without crediting old passes."""
        if "response_review_start_consumed_updates" not in self.report:
            return None
        targets = set(self.report.get("response_recovery_case_names", []))
        targets.update(self.report.get("response_retention_comparison", {}).get("response_recovery_case_names", []))
        restored = sorted(name for name in targets if all(result is not None
            and result.get("cases", {}).get(name, {}).get("passed") for result in (primary, confirmation)))
        result = {"updates_consumed_since_model_change": self.report["successful_updates"]
                    - self.report["response_review_start_consumed_updates"],
                  "confirmed_recovered_case_names": restored,
                  "remaining_recovery_case_names": sorted(targets - set(restored)),
                  "confirmation_available": confirmation is not None,
                  "additional_budget_granted": 0, "criterion": "unchanged_new_plant_two_seed_checks"}
        self.report["latest_response_review"] = result
        (self.root / "response_review.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def publish_terminal(self):
        path = self.root / "progress.json"
        progress = json.loads(path.read_text()) if path.exists() else {}
        progress.update(updated_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), worker_pid=None,
                        phase="finished", status=self.report["status"], parent_updates=0,
                        successful_updates=self.report["successful_updates"])
        progress.setdefault("training_transitions", self.report.get("training_transitions",
            self.report["successful_updates"] * self.args.num_envs * self.contract.get("num_steps_per_env", 0)))
        temporary = self.root / "progress.final.tmp"
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(path)

    def recover_regression(self, settings, gate):
        """Return a verified full-state parent; retries never replenish budgets."""
        recovery = settings.get("regression_recovery")
        if not recovery:
            return None
        attempts = self.report.setdefault("recovery_attempts", [])
        remaining = self.args.updates - self.report["successful_updates"]
        if len(attempts) >= recovery["max_retries"] or remaining <= 0:
            self.report["recovery_exhausted"] = "retry_limit" if remaining > 0 else "stage_budget"
            return None
        parent = Path(self.report["retention_learning_checkpoint"])
        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
        if not parent.is_file():
            raise ValueError("Regression recovery requires a full-state checkpoint in this exact stage contract")
        source_contract = checkpoint_contract_path(parent)
        if source_contract.read_bytes() != self.args.contract.read_bytes():
            if not getattr(self.args, "resume_budget_change", False):
                raise ValueError("Regression recovery checkpoint contract differs from the current stage")
            from wheeled_tasks.chassis.policy_transfer import validate_budget_resume
            # Quick checks and cumulative protection mutate self.contract in
            # memory. Identity validation must compare the frozen contracts.
            validate_budget_resume(json.loads(source_contract.read_text()), json.loads(self.args.contract.read_text()))
        previous_scale = self.report.get("learning_rate_scale", 1.)
        scale = previous_scale * recovery["learning_rate_factor"]
        rate = self.contract["learning_rate"] * scale
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Regression recovery requires a finite positive learning rate")
        self.report["learning_rate_scale"] = scale
        attempts.append({"checkpoint": str(parent), "consumed_updates": self.report["successful_updates"],
                         "remaining_updates": remaining, "learning_rate_scale": scale,
                         "previous_learning_rate_scale": previous_scale,
                         "previous_learning_rate": self.contract["learning_rate"] * previous_scale,
                         "learning_rate_factor": recovery["learning_rate_factor"], "learning_rate": rate,
                         "lost_cases": gate["lost_parent_passes"], "time": datetime.now(timezone.utc).isoformat()})
        print("V6_REGRESSION_RECOVERY", json.dumps(attempts[-1]), flush=True)
        (self.root / "recovery_history.json").write_text(json.dumps(attempts, indent=2) + "\n")
        return parent

    def pending_regression_recovery(self, settings, previous):
        """Finish a failed rollback transaction before taking another PPO step."""
        blocks = previous.get("blocks", [])
        patience = settings.get("regression_patience")
        if (previous.get("status") != "failed" or not patience or len(blocks) < patience
                or blocks[-1].get("successful_updates_total") != self.report.get("resumed_updates")):
            return None
        if not all(b.get("rollback_trigger_cases") or b.get("capability_gate", {}).get("mechanical_failures")
                   for b in blocks[-patience:]):
            return None
        parent = self.recover_regression(settings, blocks[-1]["capability_gate"])
        if parent is not None:
            self.report["completed_interrupted_recovery"] = True
        return parent

    def run(self):
        self.root.mkdir(parents=True, exist_ok=False)
        self.copy_atomic(self.args.contract, "contract.json")
        self.copy_atomic(Path(__file__), "orchestrator_source.py")
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        parent = getattr(self.args, "resume", None)
        best_rank = None
        best_passing_rank = None
        regressions = 0
        settings = self.evaluation_settings
        physics_resume = bool(getattr(self.args, "resume_physics_change", False))
        reward_resume = bool(getattr(self.args, "resume_reward_change", False))
        budget_resume = bool(getattr(self.args, "resume_budget_change", False))
        response_resume = bool(getattr(self.args, "resume_response_change", False))
        migrating = physics_resume or reward_resume or response_resume
        monitor_only = settings.get("mode", "gate") == "monitor"
        select_retained = settings.get("continuation_selection") == "retain_parent_passes_then_rank"
        continuation_rank = None
        self.report["evaluation_mode"] = "monitor" if monitor_only else "gate"
        try:
            self.report["status"] = "running"
            if parent is not None:
                from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count, checkpoint_contract_path
                self.report["successful_updates"] = checkpoint_update_count(parent)
                self.report["resumed_updates"] = self.report["successful_updates"]
                self.report["resume_checkpoint"] = str(parent)
                consumed = getattr(self.args, "consumed_updates", None)
                if response_resume and consumed is None:
                    raise ValueError("Response migration requires explicit consumed-update accounting")
                if consumed is not None:
                    if consumed < self.report["successful_updates"] or consumed >= self.args.updates:
                        raise ValueError("Consumed updates must include the checkpoint and leave a positive stage budget")
                    self.report["successful_updates"] = consumed
                    self.report["consumed_updates_before_run"] = consumed
                source_path = checkpoint_contract_path(parent)
                if budget_resume:
                    from wheeled_tasks.chassis.policy_transfer import validate_budget_resume
                    self.report["budget_resume"] = validate_budget_resume(json.loads(source_path.read_text()), self.contract)
                elif not migrating and source_path.read_bytes() != self.args.contract.read_bytes():
                    raise ValueError("Resume contract differs; use an explicit validated migration, not a different plan")
            if self.contract.get("new_asset_curriculum", {}).get("enabled"):
                # This campaign has no critic-only warmup. Consumed updates are
                # monotone even when the restored learning state is older.
                self.report["actor_updates_consumed"] = self.report["successful_updates"]
            recovery = settings.get("regression_recovery")
            if recovery and (type(recovery["max_retries"]) is not int or not 0 <= recovery["max_retries"] <= 5
                             or not 0 < recovery["learning_rate_factor"] < 1):
                raise ValueError("Recovery requires bounded retries and a decreasing learning rate")
            if select_retained and self.args.transfer is None and parent is None:
                raise ValueError("This adaptation contract requires the current parent actor, not scratch initialization")
            had_passing_baseline = False
            had_passing_anchor = False
            if migrating:
                from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                from wheeled_tasks.chassis.policy_transfer import validate_physics_resume, validate_reward_resume
                source_contract = json.loads(checkpoint_contract_path(parent).read_text())
                if response_resume:
                    from wheeled_tasks.chassis.response_migration import response_contract_provenance
                    _, self.report["response_resume"] = response_contract_provenance(source_contract, self.contract, ROOT)
                    self.report["confirmed_jump_bins_m"] = []
                else:
                    validate = validate_reward_resume if reward_resume else validate_physics_resume
                    self.report["reward_resume" if reward_resume else "physics_resume"] = validate(source_contract, self.contract)
                clock_changed = source_contract["physics_dt"] != self.contract["physics_dt"]
                for directory in parent.parents[:4]:
                    path = directory / "retention_state.json"
                    if path.is_file():
                        old_state = json.loads(path.read_text())
                        if response_resume:
                            if old_state.get("contract_sha256") != hashlib.sha256(source_path.read_bytes()).hexdigest():
                                raise ValueError("Response source retention ledger has a different contract")
                            if old_state.get("actor_updates_consumed", 0) > self.report["successful_updates"]:
                                raise ValueError("Response consumed updates precede the authenticated retention ledger")
                        settings["protected_case_names"] = sorted(set(settings.get("protected_case_names", []))
                            | set(old_state["protected_case_names"]))
                        self.report["previous_response_protected_cases" if response_resume else "previous_clock_protected_cases"] = settings["protected_case_names"][:]
                        break
                if response_resume:
                    for suffix, seed in (("primary", settings["seed"]), ("confirmation", settings["confirmation_seed"])):
                        result = self.evaluate_actor(parent, self.root / ("response_source_" + suffix),
                            seed=seed, full=True, contract=source_path)
                        if result is None:
                            self.report["status"] = "stopped"
                            return 0
            if parent is not None and not migrating and (settings.get("tiers") or select_retained or self.new_asset):
                for directory in parent.parents[:4]:
                    baseline_path = directory / "baseline_evaluation.json"
                    if baseline_path.is_file():
                        self.report["baseline_evaluation"] = json.loads(baseline_path.read_text())["candidates"][0]
                        self.report["retention_baseline_source"] = str(baseline_path)
                        self.copy_atomic(baseline_path, "baseline_evaluation.json")
                        if settings.get("cumulative_retention"):
                            state = json.loads((directory / "retention_state.json").read_text())
                            if self.new_asset:
                                if state.get("contract_sha256") != hashlib.sha256(self.args.contract.read_bytes()).hexdigest():
                                    raise ValueError("Resume runtime retention ledger has a different frozen contract")
                                self._validate_runtime_protection(state["protected_case_names"], state.get("confirmed_jump_bins_m", []))
                                settings["protected_case_names"] = sorted(set(settings["protected_case_names"])
                                    | set(state["protected_case_names"]))
                                settings["initial_passed_case_names"] = state["initial_passed_case_names"]
                                self.report["confirmed_jump_bins_m"] = sorted(set(self.report["confirmed_jump_bins_m"])
                                    | set(state.get("confirmed_jump_bins_m", [])))
                                self.report["actor_updates_consumed"] = max(self.report["actor_updates_consumed"],
                                    state.get("actor_updates_consumed", 0))
                                self.report["successful_updates"] = max(self.report["successful_updates"],
                                    self.report["actor_updates_consumed"])
                                if state.get("zero_airtime_checkpoint") is not None:
                                    self.report["zero_airtime_checkpoint"] = state["zero_airtime_checkpoint"]
                            else:
                                settings.update({k: state[k] for k in ("protected_case_names", "initial_passed_case_names")})
                            if state.get("promotion_case_names"):
                                settings["promotion_case_names"] = state["promotion_case_names"]
                            self.report["clock_recovery_case_names"] = state.get("clock_recovery_case_names", [])
                            self.report["response_recovery_case_names"] = state.get("response_recovery_case_names", [])
                            for key in ("response_retention_comparison", "response_review_start_consumed_updates"):
                                if key in state:
                                    self.report[key] = state[key]
                            self.report["baseline_confirmation"] = state["baseline_confirmation"]
                            self.report["protected_case_names"] = settings["protected_case_names"]
                            self.copy_atomic(directory / "retention_state.json", "retention_state.json")
                            self.report["retention_checkpoint"] = str(parent.resolve())
                            retained = state.get("retention_learning_checkpoint")
                            if retained is None and recovery:
                                retained = directory / "block_000/initial_state/model_final.pt"
                            if retained is not None:
                                if not Path(retained).is_file():
                                    raise ValueError("Saved full-state retention checkpoint is missing")
                                self.report["retention_learning_checkpoint"] = str(Path(retained).resolve())
                            completed = directory / "completion.json"
                            if recovery and completed.is_file():
                                prior_report = json.loads(completed.read_text())
                                self.report["recovery_attempts"] = prior_report.get("recovery_attempts", [])
                                self.report["learning_rate_scale"] = prior_report.get("learning_rate_scale", 1.)
                                restored_parent = self.pending_regression_recovery(settings, prior_report)
                                if restored_parent is not None:
                                    parent = restored_parent
                        if select_retained:
                            from wheeled_tasks.chassis.evaluation import continuation_assessment
                            baseline = self.report["baseline_evaluation"]
                            continuation_rank = continuation_assessment(baseline, baseline, settings)["rank"]
                            self.report["continuation_checkpoint"] = str((directory / "baseline_actor.pt").resolve())
                        break
                if select_retained and continuation_rank is None:
                    raise ValueError("Retained-candidate resume requires its original baseline evaluation")
                if self.new_asset and "baseline_evaluation" not in self.report:
                    initial = next((directory / "block_000/initial_state/model_final.pt"
                        for directory in parent.parents[:4]
                        if (directory / "block_000/initial_state/model_final.pt").is_file()), None)
                    if initial is None:
                        raise ValueError("New campaign resume requires its retained baseline ledger or authentic initial checkpoint")
                    if not self.establish_scratch_baseline(initial):
                        self.report["status"] = "stopped"
                        return 0
            baseline_source = self.args.transfer or (parent if migrating else None)
            if baseline_source:
                baseline_dir = self.root / "baseline_evaluation"
                baseline = self.evaluate_actor(baseline_source, baseline_dir)
                if baseline is None:
                    self.report["status"] = "stopped"
                    return 0
                self.report["baseline_evaluation"] = baseline
                if select_retained:
                    from wheeled_tasks.chassis.evaluation import continuation_assessment
                    continuation_rank = continuation_assessment(baseline, baseline, settings)["rank"]
                    self.report.update(continuation_checkpoint=str((self.root / "baseline_actor.pt").resolve()),
                                       continuation_source="stage_input_no_qualifying_improvement")
                best_rank = baseline["rank_lower_is_better"]
                had_passing_baseline = baseline["passed"]
                had_passing_anchor = baseline.get("anchor_passed", False)
                if not monitor_only and settings.get("require_passing_anchors") and not had_passing_anchor:
                    raise RuntimeError("Initial actor does not pass the required protected cases")
                self.copy_atomic(baseline_source, "baseline_actor.pt")
                from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                self.copy_atomic(checkpoint_contract_path(baseline_source), "baseline_actor.contract.json")
                source_manifest = baseline_source.parent / "asset_manifest.json"
                if source_manifest.is_file():
                    self.copy_atomic(source_manifest, "baseline_actor.asset_manifest.json")
                self.copy_atomic(baseline_dir / "evaluation.json", "baseline_evaluation.json")
                if settings.get("cumulative_retention"):
                    confirmation = self.evaluate_actor(baseline_source, self.root / "baseline_confirmation",
                                                       seed=settings["confirmation_seed"])
                    if confirmation is None:
                        self.report["status"] = "stopped"
                        return 0
                    self.report["baseline_confirmation"] = confirmation
                    self.record_confirmed_jump_bins(baseline, confirmation)
                    settings["initial_passed_case_names"] = sorted(name for name, case in baseline["cases"].items()
                        if case["passed"] and confirmation["cases"][name]["passed"])
                    if migrating and (clock_changed or response_resume):
                        # A changed simulator clock is a new baseline. Earlier
                        # passes that need recovery remain mandatory targets;
                        # they are not misreported as newly induced forgetting.
                        missing = set(settings.get("protected_case_names", [])) - set(settings["initial_passed_case_names"])
                        self.report["response_recovery_case_names" if response_resume else "clock_recovery_case_names"] = sorted(missing)
                        settings["promotion_case_names"] = sorted(set(settings["promotion_case_names"]) | missing)
                        settings["protected_case_names"] = sorted(set(settings.get("protected_case_names", [])) - missing)
                    if response_resume:
                        from wheeled_tasks.chassis.response_migration import assess_response_retention
                        comparison = assess_response_retention(*(
                            json.loads((self.root / name / "evaluation.json").read_text()) for name in (
                                "response_source_primary", "response_source_confirmation", "baseline_evaluation", "baseline_confirmation")),
                            required_cases=sorted(set(settings["promotion_case_names"])
                                | set(self.report.get("previous_response_protected_cases", []))))
                        self.report["response_retention_comparison"] = comparison
                        (self.root / "response_retention_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
                        self.report["response_review_start_consumed_updates"] = self.report["successful_updates"]
                    protected = set(settings.get("protected_case_names", [])) | set(settings["initial_passed_case_names"])
                    settings["protected_case_names"] = sorted(protected)
                    self.report["protected_case_names"] = sorted(protected)
                    self.report["retention_checkpoint"] = str((self.root / "baseline_actor.pt").resolve())
                    self.save_retention_state(settings)
                print("V5_BASELINE_EVALUATED", json.dumps({"passed": baseline["passed"], "rank": best_rank}), flush=True)
                if not response_resume and not monitor_only and baseline["passed"] and settings.get("skip_training_if_initially_accepted"):
                    confirmation_dir = self.root / "baseline_confirmation"
                    confirmation = self.evaluate_actor(self.args.transfer, confirmation_dir,
                        seed=settings["confirmation_seed"], export_policy=True)
                    if confirmation is None:
                        self.report["status"] = "stopped"
                        return 0
                    self.report["baseline_confirmation"] = confirmation
                    best_rank = max(best_rank, confirmation["rank_lower_is_better"])
                    had_passing_baseline &= confirmation["passed"]
                    had_passing_anchor &= confirmation.get("anchor_passed", False)
                    if had_passing_baseline:
                        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                        self.copy_atomic(self.args.transfer, "model_final.pt")
                        self.copy_atomic(checkpoint_contract_path(self.args.transfer), "model_final.contract.json")
                        for name in ("policy.onnx", "policy.onnx.json", "policy.onnx.contract.json"):
                            self.copy_atomic(Path(confirmation["export_directory"]) / name, name)
                        self.report.update(status="stage_accepted", skipped_training="initial_actor_passed_two_seeds",
                            accepted_checkpoint=str(self.args.transfer.resolve()), export=confirmation["export"],
                            checkpoint_sha256=baseline["checkpoint_sha256"], consecutive_evaluation_passes=2)
                        return 0
            while self.report["successful_updates"] < self.args.updates and not self.stop_requested:
                index = len(self.report["blocks"])
                directory = self.root / f"block_{index:03d}"
                updates = min(settings["block_updates"], self.args.updates - self.report["successful_updates"])
                consumed_before = self.report["successful_updates"]
                command = [sys.executable, "-B", str(self.worker_root / "scripts/train_chassis.py"),
                    "--contract", str(self.args.contract.resolve()), "--stage", self.args.stage,
                    "--research", "--num-envs", str(self.args.num_envs), "--updates", str(updates),
                    "--seed", str(self.args.seed + index), "--device", self.args.device,
                    "--max-runtime-seconds", str(max(1., self.deadline - time.monotonic())),
                    "--run-dir", str(directory)]
                if self.args.publish_state:
                    command.append("--publish-state")
                if parent:
                    command += ["--resume", str(parent)]
                    if physics_resume and parent == self.args.resume:
                        command.append("--resume-physics-change")
                    if reward_resume and parent == self.args.resume:
                        command.append("--resume-reward-change")
                    if response_resume and parent == self.args.resume:
                        command.append("--resume-response-change")
                    if budget_resume:
                        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
                        if checkpoint_contract_path(parent).read_bytes() != self.args.contract.read_bytes():
                            command.append("--resume-budget-change")
                    if recovery or self.new_asset or getattr(self.args, "consumed_updates", None) is not None:
                        command += ["--consumed-updates", str(self.report["successful_updates"])]
                    if self.report.get("recovery_attempts"):
                        command += ["--learning-rate-scale", str(self.report["learning_rate_scale"])]
                elif self.args.transfer:
                    command += ["--transfer", str(self.args.transfer.resolve())]
                    if not self.contract.get("transfer_critic", False):
                        command.append("--transfer-actor-only")
                if self.new_asset:
                    # Consumption never rewinds; the course clock follows the
                    # restored learning checkpoint, including on rollback.
                    stage_clock = self.checkpoint_stage_actor_update(parent) if parent else 0
                    command += ["--block-index", str(index), "--actor-updates-consumed",
                                str(self.report.get("actor_updates_consumed", 0)),
                                "--stage-actor-update", str(stage_clock),
                                "--global-actor-update-offset", str(self.report["global_actor_update_offset"]),
                                "--confirmed-jump-bins-json", json.dumps(self.report["confirmed_jump_bins_m"])]
                code = self.execute(command, self.root / f"block_{index:03d}.log", directory)
                completion_path = directory / "completion.json"
                if not completion_path.exists():
                    if self.stop_requested:
                        self.report["status"] = "stopped"
                        break
                    raise RuntimeError(f"Training block exited {code} without completion")
                completion = json.loads(completion_path.read_text())
                if self.new_asset:
                    completed = completion.get("successful_updates", 0)
                    if type(completed) is not int or not 0 <= completed <= updates:
                        raise ValueError("Worker reported an invalid completed-update count")
                    actor_completed = completion.get("actor_updates_in_block", 0)
                    if type(actor_completed) is not int or actor_completed != completed:
                        raise ValueError("The no-warmup campaign must charge every completed actor update")
                    if completed and completion.get("parent_updates") != consumed_before:
                        raise ValueError("Worker failed to preserve the charged stage budget")
                    self.report["successful_updates"] = consumed_before + completed
                self.report["actor_updates_consumed"] = self.report.get("actor_updates_consumed", 0) + completion.get("actor_updates_in_block", 0)
                if self.new_asset:
                    self.report["global_budget_actor_updates"] = (self.report["global_actor_update_offset"]
                        + self.report["actor_updates_consumed"])
                if completion["status"] not in ("completed", "stopped"):
                    raise RuntimeError(f"Training block status: {completion['status']}: {completion.get('error', '')}")
                if not self.new_asset:
                    self.report["successful_updates"] = completion["parent_updates"] + completion["successful_updates"]
                if self.new_asset and completion["successful_updates"] > 0 and "baseline_evaluation" not in self.report:
                    if not self.establish_scratch_baseline(directory / "initial_state/model_final.pt"):
                        self.report["status"] = "stopped"
                        break
                    best_rank = self.report["baseline_evaluation"]["rank_lower_is_better"]
                if recovery and completion["successful_updates"] > 0:
                    self.report["learning_rate_scale"] = completion["learning_rate"] / self.contract["learning_rate"]
                    initial = directory / "initial_state/model_final.pt"
                    if "retention_learning_checkpoint" not in self.report:
                        if not initial.is_file():
                            raise ValueError("Worker omitted its initial full-state recovery checkpoint")
                        self.report["retention_learning_checkpoint"] = str(initial.resolve())
                        if settings.get("cumulative_retention"):
                            self.save_retention_state(settings)
                parent = directory / "model_final.pt"
                if not parent.exists():
                    self.report["status"] = "stopped"
                    break
                for name in ("model_final.pt", "policy.onnx", "policy.onnx.json", "agent_config.json"):
                    self.copy_atomic(directory / name, name)
                self.copy_atomic(self.args.contract, "model_final.contract.json")
                self.copy_atomic(self.args.contract, "policy.onnx.contract.json")
                self.report["export"] = completion["export"]
                self.report["checkpoint_sha256"] = completion["checkpoint_sha256"]
                block = {"directory": directory.name, "successful_updates_total": self.report["successful_updates"],
                         "checkpoint_sha256": completion["checkpoint_sha256"], "training_status": completion["status"]}
                self.report["blocks"].append(block)
                if self.stop_requested or completion["status"] == "stopped":
                    self.report["status"] = "stopped"
                    break
                evaluation_dir = self.root / f"evaluation_{index:03d}"
                candidate = self.evaluate_actor(parent, evaluation_dir)
                if candidate is None:
                    self.report["status"] = "stopped"
                    break
                evaluation = json.loads((evaluation_dir / "evaluation.json").read_text())
                from wheeled_tasks.chassis.evaluation import capability_gate
                gate = capability_gate(candidate, self.report.get("baseline_evaluation"), settings)
                audit = (settings.get("full_audit_at_stage_exit") and (
                    (gate["passed"] and self.report["successful_updates"] >= settings.get("minimum_updates", 0))
                    or self.report["successful_updates"] >= self.args.updates))
                if audit:
                    full_dir = self.root / f"evaluation_{index:03d}_full"
                    candidate = self.evaluate_actor(parent, full_dir, full=True)
                    if candidate is None:
                        self.report["status"] = "stopped"
                        break
                    evaluation = json.loads((full_dir / "evaluation.json").read_text())
                    gate = capability_gate(candidate, self.report.get("baseline_evaluation"), settings)
                    block["full_evaluation"] = full_dir.name
                diagnostic = self.inspect_zero_airtime(candidate, parent, full_dir if audit else evaluation_dir)
                if diagnostic is not None:
                    block["zero_airtime_diagnostic_status"] = diagnostic["status"]
                needs_confirmation = gate["passed"] or settings.get("cumulative_retention")
                if settings.get("confirmation_on_candidate_or_regression"):
                    new_passes = {name for name, case in candidate["cases"].items() if case["passed"]}
                    new_passes -= set(settings.get("protected_case_names", []))
                    needs_confirmation = (gate["passed"] or bool(new_passes)
                        or bool(gate["lost_parent_passes"]) or bool(gate["mechanical_failures"]))
                    block["new_primary_passes"] = sorted(new_passes)
                if "response_review_start_consumed_updates" in self.report:
                    needs_confirmation = True
                confirmation = None
                if needs_confirmation and "confirmation_seed" in settings:
                    confirmation_dir = self.root / f"evaluation_{index:03d}_confirmation"
                    confirmation = self.evaluate_actor(parent, confirmation_dir, seed=settings["confirmation_seed"], full=bool(audit))
                    if confirmation is None:
                        self.report["status"] = "stopped"
                        break
                    block["confirmation_evaluation"] = confirmation_dir.name
                    confirmation_gate = capability_gate(confirmation,
                        self.report.get("baseline_confirmation", self.report.get("baseline_evaluation")), settings)
                    gate["confirmation"] = confirmation_gate
                    gate["passed"] &= confirmation_gate["passed"]
                    gate["lost_parent_passes"] = sorted(set(gate["lost_parent_passes"]) | set(confirmation_gate["lost_parent_passes"]))
                    self.record_confirmed_jump_bins(candidate, confirmation)
                    candidate = {**candidate, "primary_passed": candidate["passed"],
                        "confirmation_passed": confirmation["passed"],
                        "passed": candidate["passed"] and confirmation["passed"],
                        "anchor_passed": candidate.get("anchor_passed", False) and confirmation.get("anchor_passed", False),
                        "rank_lower_is_better": max(candidate["rank_lower_is_better"], confirmation["rank_lower_is_better"])}
                    evaluation["candidates"][0] = candidate
                    evaluation["confirmation_evaluation"] = str(confirmation_dir)
                    if settings.get("cumulative_retention") and not (
                            gate["lost_parent_passes"] or gate["mechanical_failures"] or confirmation_gate["mechanical_failures"]):
                        gained = {name for name, case in candidate["cases"].items()
                                  if case["passed"] and confirmation["cases"][name]["passed"]}
                        protected = set(settings["protected_case_names"]) | gained
                        settings["protected_case_names"] = sorted(protected)
                        self.report["protected_case_names"] = sorted(protected)
                        self.report["retention_checkpoint"] = str(parent.resolve())
                        self.report["retention_learning_checkpoint"] = str(parent.resolve())
                        self.save_retention_state(settings)
                    elif self.new_asset:
                        # Bin evidence and consumption are monotone runtime
                        # observations even when another skill regresses.
                        self.save_retention_state(settings)
                block["evaluation_passed"] = candidate["passed"]
                review = self.record_response_review(candidate, confirmation)
                if review is not None:
                    block["response_review"] = review
                block["capability_gate"] = gate
                from wheeled_tasks.chassis.evaluation import severe_retention_regressions
                severe = set(severe_retention_regressions(candidate, settings, gate["lost_parent_passes"]))
                if confirmation is not None:
                    severe.update(severe_retention_regressions(confirmation, settings, gate["lost_parent_passes"]))
                block["rollback_trigger_cases"] = sorted(severe)
                block["anchor_passed"] = candidate.get("anchor_passed", False)
                evaluated = [candidate]
                if "confirmation_evaluation" in block:
                    evaluated.append(confirmation)
                block["failed_anchor_cases"] = sorted({name for result in evaluated
                    for name, case in result.get("cases", {}).items() if case.get("anchor") and not case["passed"]})
                block["evaluation_rank"] = candidate["rank_lower_is_better"]
                if select_retained:
                    from wheeled_tasks.chassis.evaluation import continuation_assessment
                    assessment = continuation_assessment(candidate, self.report["baseline_evaluation"], settings)
                    if settings.get("cumulative_retention"):
                        assessment["eligible"] &= not (gate["lost_parent_passes"]
                            or gate["mechanical_failures"] or gate.get("confirmation", {}).get("mechanical_failures"))
                    block["continuation_assessment"] = assessment
                    if assessment["eligible"] and assessment["rank"] < continuation_rank:
                        continuation_rank = assessment["rank"]
                        self.copy_atomic(parent, "continuation_candidate.pt")
                        self.copy_atomic(self.args.contract, "continuation_candidate.contract.json")
                        self.report.update(continuation_checkpoint=str((self.root / "continuation_candidate.pt").resolve()),
                            continuation_source="nominal_retained_candidate", continuation_updates=self.report["successful_updates"])
                from wheeled_tasks.chassis.evaluation import summarize_evaluation_tiers
                tiers = summarize_evaluation_tiers(candidate, settings, self.report.get("baseline_evaluation"))
                if tiers is not None:
                    block["tier_summary"] = tiers
                    evaluation["tier_summary"] = tiers
                latest = self.root / "latest_evaluation.tmp"
                latest.write_text(json.dumps(evaluation, indent=2) + "\n")
                latest.replace(self.root / "latest_evaluation.json")
                if best_rank is None or candidate["rank_lower_is_better"] < best_rank:
                    best_rank = candidate["rank_lower_is_better"]
                    self.copy_atomic(parent, "best_candidate.pt")
                    self.copy_atomic(self.args.contract, "best_candidate.contract.json")
                    self.copy_atomic(self.root / "latest_evaluation.json", "best_candidate_evaluation.json")
                if gate["passed"]:
                    if best_passing_rank is None or candidate["rank_lower_is_better"] < best_passing_rank:
                        best_passing_rank = candidate["rank_lower_is_better"]
                        self.copy_atomic(parent, "model_best.pt")
                        self.copy_atomic(self.args.contract, "model_best.contract.json")
                        self.copy_atomic(directory / "policy.onnx", "policy_best.onnx")
                        self.copy_atomic(directory / "policy.onnx.json", "policy_best.onnx.json")
                        self.copy_atomic(self.args.contract, "policy_best.onnx.contract.json")
                        self.copy_atomic(self.root / "latest_evaluation.json", "best_passing_evaluation.json")
                    self.report["consecutive_evaluation_passes"] += 1
                    regressions = 0
                else:
                    self.report["consecutive_evaluation_passes"] = 0
                    if settings.get("promotion_case_names"):
                        regressions = regressions + 1 if severe or gate.get("mechanical_failures") else 0
                    elif (had_passing_baseline or (self.root / "model_best.pt").exists()
                            or (settings.get("protect_anchor_cases") and had_passing_anchor and not candidate.get("anchor_passed", False))
                            or (settings.get("require_passing_anchors") and not candidate.get("anchor_passed", False))):
                        regressions += 1
                    else:
                        regressions = 0
                print("V5_BLOCK_EVALUATED", json.dumps(block), flush=True)
                (self.root / "curriculum.json").write_text(json.dumps(self.report, indent=2) + "\n")
                if (not monitor_only and self.report["consecutive_evaluation_passes"] >= settings["consecutive_passes_required"]
                        and self.report["successful_updates"] >= settings.get("minimum_updates", 0)):
                    self.report["status"] = "stage_accepted" if self.contract.get("curriculum_stage") else "foundation_accepted"
                    self.report["accepted_checkpoint"] = str((self.root / "model_best.pt").resolve())
                    self.report["promotion_case_names"] = settings.get("promotion_case_names")
                    break
                if not monitor_only and settings.get("regression_patience", 3) is not None and regressions >= settings.get("regression_patience", 3):
                    restored_parent = self.recover_regression(settings, gate)
                    if restored_parent is not None:
                        parent = restored_parent
                        regressions = 0
                        continue
                    self.report["status"] = "regression_hold_best_preserved"
                    self.report["rollback_checkpoint"] = self.report.get("retention_learning_checkpoint",
                        self.report.get("retention_checkpoint", self.report.get("continuation_checkpoint")))
                    break
                fuse = settings.get("zero_success_fuse_updates")
                if fuse:
                    self.report["jump_success_seen"] = (self.report.get("jump_success_seen", False)
                        or candidate["cases"]["jump_cold_03"]["successes"] > 0)
                if fuse and self.report["successful_updates"] >= fuse:
                    if not self.report["jump_success_seen"]:
                        self.report["status"] = "zero_jump_success_fuse"
                        self.report["rollback_checkpoint"] = self.report.get("retention_checkpoint")
                        break
            if self.report["status"] == "running":
                self.report["status"] = ("stopped" if self.stop_requested else
                                         "training_budget_completed" if monitor_only else "budget_exhausted_gate_pending")
        except Exception:
            self.report.update(status="failed", error=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
            if self.new_asset and "baseline_confirmation" in self.report:
                self.report["protected_case_names"] = settings.get("protected_case_names", [])
                self.save_retention_state(settings)
            (self.root / "artifact_selection.json").write_text(json.dumps({
                "accepted_checkpoint": self.report.get("accepted_checkpoint"),
                "latest_checkpoint": str(self.root / "model_final.pt"),
                "latest_is_accepted": self.report["status"] in ("stage_accepted", "foundation_accepted"),
                "baseline_preserved": (self.root / "baseline_actor.pt").exists(),
                "continuation_checkpoint": self.report.get("continuation_checkpoint"),
                "continuation_source": self.report.get("continuation_source"),
                "promotion_case_names": settings.get("promotion_case_names"),
                "status": self.report["status"]}, indent=2) + "\n")
            (self.root / "completion.json").write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
            self.publish_terminal()
        return 1 if self.report["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage", choices=("foundation", "speed", "terrain", "jump", "mixed"), default="foundation")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker-source", type=Path, help="Frozen training/evaluation source used for controlled rollback")
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--publish-state", action="store_true")
    parent = parser.add_mutually_exclusive_group()
    parent.add_argument("--transfer", type=Path)
    parent.add_argument("--resume", type=Path, help="Resume the same contract and optimizer from a sealed checkpoint")
    parser.add_argument("--resume-physics-change", action="store_true")
    parser.add_argument("--resume-reward-change", action="store_true")
    parser.add_argument("--resume-budget-change", action="store_true")
    parser.add_argument("--resume-response-change", action="store_true")
    parser.add_argument("--consumed-updates", type=int)
    parser.add_argument("--global-actor-update-offset", type=int, default=0)
    parser.add_argument("--protected-cases-json", default="[]")
    parser.add_argument("--confirmed-jump-bins-json", default="[]")
    parser.add_argument("--max-runtime-seconds", type=float, default=86400.)
    args = parser.parse_args()
    if args.resume_physics_change and not args.resume:
        parser.error("--resume-physics-change requires --resume")
    if (args.resume_reward_change or args.resume_budget_change or args.resume_response_change or args.consumed_updates is not None) and not args.resume:
        parser.error("Reward migration and consumed updates require --resume")
    if sum((args.resume_physics_change, args.resume_reward_change, args.resume_budget_change, args.resume_response_change)) > 1:
        parser.error("Choose one resume migration")
    if args.resume_response_change and args.consumed_updates is None:
        parser.error("Response migration requires explicit consumed-update accounting")
    if not args.research or not (1 <= args.num_envs <= 20000 and 1 <= args.updates <= 100000 and args.max_runtime_seconds > 0):
        parser.error("Explicit research flag and bounded positive settings required")
    if args.global_actor_update_offset < 0:
        parser.error("Global actor offset must be nonnegative")
    return TrainingBlocks(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
