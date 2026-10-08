"""Synthetic CPU evidence: authenticate one completed run, never execute physics."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_task_campaign_boundary_continuation import FakeAdapter, boundary, f, write
from test_integrated_early_boundary import EarlyAdapter, early, stopped
import adopt_known_native_cli_completion as adoption
import repair_known_native_cli_startup as repair


def known_case(q, monkeypatch, admission=None):
    """Fixture hashes replace hard pins only for these labelled synthetic bytes."""
    monkeypatch.setattr(repair, "MANIFEST_SHA", q.config_sha256)
    admission = admission or dict(schema="v6-integrated-production-admission-v1", qualification=None,
        selection=dict(mode="rebase_from_authorized_REC"), target_contract_sha256="e" * 64)
    old = q.run_root / "boundary_integrated_028.native"
    for name, data in {
        "receipt.json": dict(status="failed", exit_code=1, PPO_updates=0, report_sha256=None,
            manifest_sha256=q.config_sha256, command=["python", "--validate-scene", "1"]),
        "source_selection.json": admission, "invocation.intent.json": {},
        "stdout.log": "", "stderr.log": "known CLI refusal before Simulator",
    }.items():
        if isinstance(data, dict):
            write(old / name, data)
        else:
            (old / name).write_text(data)
    monkeypatch.setattr(repair, "FAILED_PINS", {p.name: boundary.sha(p) for p in old.iterdir()})
    directory = q.run_root / adoption.NATIVE
    write(directory / "source_selection.json", admission)
    selection_sha = boundary.sha(directory / "source_selection.json")
    command = ["python", "--learning-rate-scale", "1.0", "--global-actor-update-offset", "0",
        "--block-index", "0", "--validate-scene", "1", "--num-envs", "4096"]
    invocation = dict(command=command, PPO_updates=0, qualification=None,
        manifest_sha256=q.config_sha256, selection_sha256=selection_sha)
    write(directory / "invocation.intent.json", invocation)
    write(directory / "receipt.json", dict(invocation, status="failed", exit_code=0,
        report_sha256=None, wall_seconds=137.2645))
    report = dict(status="scene_validated", successful_updates=0, attempted_updates=0,
        contract_sha256=admission["target_contract_sha256"], integrated_admission_sha256=selection_sha,
        startup=dict(num_envs=4096, runtime_preflight=dict(status="runtime_configuration_passed",
            scoped_validation="v6-integrated-preflight-candidate-v1")),
        scene_capacity=dict(policy_steps=1, physics_steps_per_env=4))
    write(directory / "scene/completion.json", report)
    write(directory / "scene/startup.json", dict(report, status="running"))
    write(q.run_root / "boundary_integrated_028_cli_repair001.intent.json", dict(
        schema="diagnosed-native-CLI-repair-v1", manifest_sha256=q.config_sha256,
        native_check_PPO_updates=0, qualification=None))
    write(q.run_root / "boundary_integrated_028_cli_repair001.receipt.json", dict(result=dict(
        status="attention", manifest_sha256=q.config_sha256,
        detail="Actual joint native startup failed; retained receipt, no PPO launch")))
    monkeypatch.setattr(adoption, "REPAIR_PINS", {name: boundary.sha(q.run_root / name)
        for name in adoption.REPAIR_PINS})
    return dict(integrated_admission=admission)


@pytest.fixture
def q(tmp_path, monkeypatch):
    q = SimpleNamespace(run_root=tmp_path, target_revision=26, config_sha256=repair.MANIFEST_SHA,
        intent=tmp_path / "launch.intent.json", queue_receipt=tmp_path / "launch.receipt.json",
        launch_log=tmp_path / "launch.log", owners=lambda: [])
    known_case(q, monkeypatch)
    return q


def repin(q, relative, data):
    write(q.run_root / relative, data)
    adoption.REPAIR_PINS[relative] = boundary.sha(q.run_root / relative)


def test_real_completion_is_adopted_without_native_subprocess_or_overwrite(q, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no Simulator/subprocess validation is allowed")
    monkeypatch.setattr(boundary.NativeAdapter, "validate_runtime", forbidden)
    monkeypatch.setattr(boundary.subprocess, "run", forbidden)
    before = {str(p): p.read_bytes() for p in q.run_root.rglob("*") if p.is_file()}
    evidence = dict(integrated_admission=boundary.read_json(q.run_root / adoption.NATIVE / "source_selection.json"))
    result = adoption.CompletionAdoptionAdapter(q).validate_runtime(evidence)
    assert result["status"] == "passed" and result["PPO_updates"] == result["additional_native_runs"] == 0
    assert result["report_file"].endswith("scene/completion.json")
    assert result["report_sha256"] == boundary.sha(Path(result["report_file"]))
    assert all(Path(path).read_bytes() == value for path, value in before.items())
    assert boundary.read_json(q.run_root / adoption.NATIVE / "receipt.json")["status"] == "failed"
    with pytest.raises(FileExistsError):
        adoption.CompletionAdoptionAdapter(q).validate_runtime(evidence)


@pytest.mark.parametrize("relative", list(adoption.REPAIR_PINS))
def test_any_known_native_evidence_byte_change_rejects(q, relative):
    path = q.run_root / relative
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="evidence changed"):
        adoption.authenticate_existing_native(q)


@pytest.mark.parametrize("fault", ["status", "success", "attempt", "envs", "runtime", "scope",
    "policy", "physics", "contract", "admission", "startup", "startup_envs", "exit", "PPO",
    "legacy_report", "fresh_account", "owner", "launch", "launch_log"])
def test_even_re_pinned_synthetic_evidence_must_prove_all_passed_guards(q, fault):
    relative = adoption.NATIVE + "/scene/completion.json"
    report = boundary.read_json(q.run_root / relative)
    evidence = None
    if fault == "status": report["status"] = "completed"
    elif fault == "success": report["successful_updates"] = 1
    elif fault == "attempt": report["attempted_updates"] = 1
    elif fault == "envs": report["startup"]["num_envs"] = 1
    elif fault == "runtime": report["startup"]["runtime_preflight"]["status"] = "failed"
    elif fault == "scope": report["startup"]["runtime_preflight"]["scoped_validation"] = "other"
    elif fault == "policy": report["scene_capacity"]["policy_steps"] = 2
    elif fault == "physics": report["scene_capacity"]["physics_steps_per_env"] = 5
    elif fault == "contract": report["contract_sha256"] = "0" * 64
    elif fault == "admission": report["integrated_admission_sha256"] = "0" * 64
    elif fault in ("startup", "startup_envs"):
        relative = adoption.NATIVE + "/scene/startup.json"
        report = boundary.read_json(q.run_root / relative)
        if fault == "startup": report["status"] = "scene_validated"
        else: report["startup"]["num_envs"] = 1
    elif fault in ("exit", "PPO"):
        relative = adoption.NATIVE + "/receipt.json"
        report = boundary.read_json(q.run_root / relative)
        report["exit_code" if fault == "exit" else "PPO_updates"] = 1
    elif fault == "legacy_report": write(q.run_root / adoption.NATIVE / "scene/report.json", {})
    elif fault == "fresh_account":
        evidence = dict(integrated_admission=deepcopy(boundary.read_json(q.run_root / adoption.NATIVE / "source_selection.json")))
        evidence["integrated_admission"]["unexpected_account_change"] = 1
    elif fault == "owner": q.owners = lambda: [dict(role="worker")]
    elif fault == "launch": write(q.intent, {})
    elif fault == "launch_log": q.launch_log.write_text("")
    repin(q, relative, report)
    with pytest.raises(ValueError):
        adoption.authenticate_existing_native(q, evidence)
    assert not (q.run_root / (adoption.ADOPTION + ".native.receipt.json")).exists()


def test_exact_four_file_code_freeze_required(tmp_path):
    for name in adoption.CODE_FILES:
        (tmp_path / name).write_text("synthetic frozen CPU file")
    manifest = dict(schema="frozen-known-native-completion-adoption-code-v1",
        files={name: boundary.sha(tmp_path / name) for name in adoption.CODE_FILES})
    write(tmp_path / "code_manifest.json", manifest)
    assert adoption.authenticate_code(tmp_path) == manifest
    (tmp_path / "adopt_known_native_cli_completion.py").write_text("changed")
    with pytest.raises(ValueError, match="code differs"):
        adoption.authenticate_code(tmp_path)
    manifest["files"].pop("adopt_known_native_cli_completion.py")
    write(tmp_path / "code_manifest.json", manifest)
    with pytest.raises(ValueError, match="exact four"):
        adoption.authenticate_code(tmp_path)


def test_once_only_full_native_queue_launch_preserves_account_and_old_failures(f, monkeypatch):
    q = early(f)
    assert q.step()["reason"] == "one_authorized_REC_supervisor_sigterm_submitted"
    stopped(f)
    q.freeze_early_origin()
    evidence = q.adapter.prepare(q.previous_job())
    known_case(q, monkeypatch, evidence["integrated_admission"])
    class Adapter(adoption.CompletionAdoptionAdapter):
        def prepare(self, job):
            return EarlyAdapter(self.q, f).prepare(job)
        def launch(self, wrapper):
            return FakeAdapter(self.q, f).launch(wrapper)
        def preflight(self, job):
            return FakeAdapter(self.q, f).preflight(job)
    q.adapter = Adapter(q)
    def forbidden(*args, **kwargs):
        raise AssertionError("adoption must never rerun native or signal")
    monkeypatch.setattr(boundary.subprocess, "run", forbidden)
    monkeypatch.setattr(q.adapter, "signal_once", forbidden)
    account = Path(f.plan["budget_source_stage"]) / (f.plan["campaign_id"] + ".account.json")
    before = account.read_bytes()
    failed_before = {relative: (q.run_root / relative).read_bytes() for relative in adoption.REPAIR_PINS}
    result = adoption.run_adoption(q, dict(schema="synthetic-test-code-manifest"))
    assert result["status"] == "launch_submitted" and len(f.launches) == 1 and f.native_calls == 0
    assert account.read_bytes() == before
    assert all((q.run_root / relative).read_bytes() == value for relative, value in failed_before.items())
    assert boundary.read_json(q.queue_receipt)["native_startup"]["additional_native_runs"] == 0
    assert all((q.run_root / (adoption.ADOPTION + suffix)).exists()
        for suffix in (".intent.json", ".receipt.json", ".native.receipt.json"))
    with pytest.raises(ValueError, match="already has an artifact"):
        adoption.run_adoption(q, {})
    assert len(f.launches) == 1
