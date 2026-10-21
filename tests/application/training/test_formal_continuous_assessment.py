"""Execute production paid-update closures with CPU accounting, without PPO."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_algo.training_control import PlannedStop, TrainingBudget
from wheeled_tasks.chassis.application.runtime.training_budget import BudgetOwner, PAID_CAP, REVIEW_SCHEMA, digest_file


ENTRY = Path(__file__).resolve().parents[3] / "src/wheeled_tasks/chassis/application/training/train.py"
NOW = datetime(2026, 10, 14, tzinfo=timezone.utc)


def source_function(name):
    return deepcopy(next(node for node in ast.walk(ast.parse(ENTRY.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name))


@pytest.fixture
def entry():
    spec = importlib.util.spec_from_file_location("continuous_formal_entry_cpu", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(mode="record_and_continue"):
    value = {"formal_v7_flat": True, "formal_stage_policy": {
        "first_assessment_update": 500, "assessment_interval_updates": 500}}
    if mode is not None:
        value["formal_stage_policy"]["assessment_mode"] = mode
    return value


class CPUMeter:
    owner_id = "continuous-assessment-cpu-fixture"
    def read_total_gpu_seconds(self):
        return 0.


def open_owner(tmp_path, allocation):
    identity = {"contract_sha256": "a" * 64, "training_source_sha256": "b" * 64,
        "seed": 1011, "stage": "terrain", "requested_updates": allocation, "parent_checkpoint_sha256": None}
    review = {"schema": REVIEW_SCHEMA, "campaign_id": "cpu-fixture", "ledger_id": "cpu-fixture",
        "ledger_path": str(tmp_path / "paid.json"), "paid_update_cap": PAID_CAP,
        "pilot_paid_update_cap": 0, "absolute_deadline_utc": "2026-10-14T01:00:00+00:00",
        "max_attempt_wall_seconds": 300., "gpu_occupied_seconds_cap": 100.,
        "gpu_meter_owner": {"kind": "test_only_cpu_meter_v1", "id": CPUMeter.owner_id},
        "review_reference": "CPU data only; no native run or production authorization",
        "attempts": {"main": {**identity, "role": "main"}}}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    return BudgetOwner.authenticate(path, digest_file(path), review["ledger_path"], "main", identity,
        gpu_meter=CPUMeter(), allow_test_meter=True, now=lambda: NOW)


def fixture(tmp_path, entry, monkeypatch, *, mode="record_and_continue", allocation=501, failure=None):
    owner = open_owner(tmp_path, allocation)
    events = []
    c = {**config(mode), "checkpoint_snapshots": False}
    boundary = entry.next_formal_assessment_update(c, 6499)
    report = {"parent_updates": 6499, "successful_updates": 0, "actor_updates_in_block": 0,
        "parent_training_transitions": 0, "entropy_coef_source": "CPU fixture",
        "formal_assessment_schedule": {"next_assessment_update": boundary}}
    env = NS(stage_actor_update=0, global_actor_update=6499, recovery=None,
        performance_curriculum=None, torque_monitor=None)
    runner = NS(alg=NS(actor=NS(requires_grad_=lambda _: None), storage=NS(actions=torch.zeros(24, 1, 6))),
        logger=NS(writer=None), save=lambda *_: None)
    budget = TrainingBudget()
    def rate(*_, **kwargs):
        if "before_rate" in kwargs and failure == "postcheck":
            raise RuntimeError("postcheck fixture failure")
        return 1e-4
    def optimizer(*_, **__):
        events.append("optimizer")
        return {"cpu_optimizer_fixture": True}
    def physical_check(_):
        if failure == "physical":
            raise RuntimeError("physical fixture failure")
    def seal(run_dir, completed, save, contract, progress, *, budget_owner):
        assert budget_owner is owner
        assert owner.summary()["optimizer_completed_attempt"] == report["successful_updates"]
        assert budget.soft_stop is None
        if failure == "seal":
            raise RuntimeError("seal fixture failure")
        path = run_dir / "snapshots" / str(completed)
        path.mkdir(parents=True)
        (path / "model.pt").write_bytes(f"CPU sealed fixture {completed}".encode())
        events.append(("sealed", completed))
        return path
    monkeypatch.setitem(sys.modules, "wheeled_tasks.chassis.application.runtime.checkpoints", NS(seal_checkpoint=seal))
    context = {"torch": torch, "json": json, "os": os, "datetime": datetime, "timezone": timezone,
        "c": c, "report": report, "env": env, "runner": runner, "budget": budget, "budget_owner": owner,
        "args": NS(run_dir=tmp_path, contract=tmp_path / "contract.json", stage="terrain", num_envs=1,
                   global_actor_update_offset=6499, actor_updates_consumed=7569),
        "warmup_updates": 0, "stage_actor_start": 0, "task_actor_start": 0,
        "physx_errors": NS(raise_if_failed=physical_check), "PlannedStop": PlannedStop,
        "source_birth_exposure": None, "mixed_exposure": None,
        "recovery_evidence": None, "curriculum_runtime": None, "timing": None,
        "original_update": optimizer, "effective_algorithm": {"entropy_coef": .005},
        "verify_entropy_coef": lambda *_: .005, "verify_learning_rate": rate,
        "learning_schedule": "adaptive", "policy_probe": None, "learning_audit": None,
        "rehearsal_indices": None, "recovery_rollout": NS(learning=None, loss_population_mask=lambda: None,
            metrics={"valid_transitions": 24, "script_or_blend_transitions": 0}),
        "incoming_action_clipping_metrics": lambda *_: {}, "log_scalar_batch": lambda *_: None,
        "log_recovery_outcomes": lambda *_: None, "rollout_steps": 24, "metrics": None,
        "write_progress": entry.write_progress, "task_assessment_due": entry.task_assessment_due,
        "timed_call": entry.timed_call, "digest": entry.digest,
        "next_formal_assessment_update": entry.next_formal_assessment_update,
        "formal_original_log": lambda *_: events.append("logger")}
    # Keep the actual nonlocal declarations, binding the same names in a real
    # enclosing function. A missing nonlocal for the advancing boundary fails.
    factory = ast.parse("def make_closure():\n    pending_checkpoint = None\n    pending_assessment = None\n    formal_assessment_update = 6500\n").body[0]
    factory.body.extend(source_function(name) for name in ("update", "paid_update", "formal_log"))
    factory.body.extend(ast.parse("def read_boundary():\n    return formal_assessment_update\n").body)
    factory.body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Load())
        for name in ("paid_update", "formal_log", "read_boundary")], ctx=ast.Load())))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), str(ENTRY), "exec"), context)
    paid, log, read_boundary = context["make_closure"]()
    return owner, report, budget, events, paid, log, read_boundary


def test_two_future_boundaries_seal_request_and_continue_real_paid_closure(entry, tmp_path, monkeypatch):
    owner, report, budget, events, paid, log, boundary = fixture(tmp_path, entry, monkeypatch)
    try:
        for _ in range(501):
            assert paid() == {"cpu_optimizer_fixture": True}
            log()
        assert report["parent_updates"] + report["successful_updates"] == 7000
        assert [(kind, update) for value in events if isinstance(value, tuple) for kind, update in [value]] == [
            ("sealed", 6500), ("sealed", 7000)]
        assert boundary() == report["formal_assessment_schedule"]["next_assessment_update"] == 7500
        assert budget.soft_stop is None and "assessment_pending" not in report and "stop_reason" not in report
        requests = sorted((tmp_path / "assessment_requests").glob("*.json"))
        assert [path.name for path in requests] == ["update_6500.json", "update_7000.json"]
        for path, paid_count in zip(requests, (1, 501)):
            request = json.loads(path.read_text())
            assert request["learning_updates"] == request["boundary_update"] == int(path.stem.split("_")[1])
            assert request["gate_result"] == "not_assessed" and request["automatic_promotion"] is False
            assert request["training_continues"] is True and request["assessment_mode"] == "record_and_continue"
            assert request["training_budget"]["paid_updates_global"] == paid_count
            assert digest_file(Path(request["checkpoint"])) == request["checkpoint_sha256"]
        latest = json.loads((tmp_path / "assessment_request.json").read_text())
        assert latest == report["latest_assessment_request"] == json.loads(requests[-1].read_text())
        assert not (tmp_path / "assessment_pending.json").exists()
        assert owner.summary()["paid_updates_global"] == owner.summary()["optimizer_completed_attempt"] == 501
        assert events[-1] == "logger"
    finally:
        owner.close()


@pytest.mark.parametrize("mode", [None, "pause_for_assessment"])
def test_default_and_explicit_legacy_pause_keep_prior_stop_path(entry, tmp_path, monkeypatch, mode):
    owner, report, budget, events, paid, log, boundary = fixture(tmp_path, entry, monkeypatch, mode=mode, allocation=1)
    try:
        paid()
        assert boundary() == 6500 and budget.soft_stop == "soft_stop"
        assert report["stop_reason"] == "assessment_pending" and report["assessment_pending"]["gate_result"] == "not_assessed"
        assert json.loads((tmp_path / "assessment_pending.json").read_text()) == report["assessment_pending"]
        assert not (tmp_path / "assessment_requests").exists()
        with pytest.raises(PlannedStop, match="soft_stop"):
            log()
        assert events[-1] == "logger"
    finally:
        owner.close()


@pytest.mark.parametrize("mode", ["unknown", None, True, 1, [], {}])
def test_bad_mode_is_rejected_by_the_prekit_cadence_call(entry, mode):
    c = config()
    c["formal_stage_policy"]["assessment_mode"] = mode
    with pytest.raises(ValueError, match="assessment_mode"):
        entry.next_formal_assessment_update(c, 6000)
    main = source_function("main")
    validation = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "next_formal_assessment_update"
        and isinstance(node.args[-1], ast.Constant) and node.args[-1].value == 0)
    kit = next(node for node in ast.walk(main) if isinstance(node, ast.ImportFrom) and node.module == "isaaclab.app")
    assert validation.lineno < kit.lineno


@pytest.mark.parametrize("restored,expected", [(6000, 6500), (6499, 6500), (6500, 7000), (7000, 7500)])
def test_continuous_next_boundary_is_strictly_future_learning_clock(entry, restored, expected):
    assert entry.next_formal_assessment_update(config(), restored) == expected


def test_continuous_mode_keeps_budget_limit_and_external_exit_behavior(entry, tmp_path, monkeypatch):
    owner, report, budget, events, paid, log, boundary = fixture(tmp_path, entry, monkeypatch, allocation=1)
    try:
        paid()
        before = list(events)
        with pytest.raises(PlannedStop, match="request exhausted"):
            paid()
        assert events == before and owner.summary()["paid_updates_global"] == 1
        budget.request_stop("sigterm")
        runner = NS(env=NS(step=lambda: None), alg=NS(update=paid))
        with budget.bind(runner):
            with pytest.raises(PlannedStop, match="sigterm"):
                runner.env.step()
    finally:
        owner.close()


@pytest.mark.parametrize("failure", ["physical", "postcheck", "seal"])
def test_continuous_mode_does_not_suppress_failure_checks_or_publish_failed_requests(entry, tmp_path, monkeypatch, failure):
    owner, report, budget, events, paid, _, _ = fixture(tmp_path, entry, monkeypatch, failure=failure, allocation=1)
    try:
        with pytest.raises(RuntimeError, match=failure):
            paid()
        assert not (tmp_path / "assessment_requests").exists()
        assert not (tmp_path / "assessment_request.json").exists()
        assert budget.soft_stop is None and "assessment_pending" not in report
        assert owner.summary()["paid_updates_global"] == (0 if failure == "physical" else 1)
    finally:
        owner.close()
