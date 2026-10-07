"""v24: normal REC boundary, actual scene validation, and once-only submission."""
from copy import deepcopy
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from test_task_campaign_boundary_continuation import (
    FakeAdapter, admitted, archiver_finished, boundary, checkpoint, f,
    proc, remove_owners, write,
)


def retarget(f):
    target = f.new / "contracts/v6_task_campaign_v24.json"
    plan = dict(f.plan, contract_id="v6-task-campaign-v24", seed=290929)
    for row in plan["domains"]:
        row["kind"] = "S3"
    write(target, plan)
    f.manifest.update(new_contract_relative_path=str(target.relative_to(f.new)),
        new_contract_sha256=boundary.sha(target),
        source_revision_receipt=str(f.root / "source_revision_026.json"),
        queue_receipt=str(f.root / "boundary_continuation_026.receipt.json"),
        status_path=str(f.root / "boundary_continuation_026.status.json"),
        launch_log=str(f.root / "boundary_continuation_026.launch.log"))
    f.manifest["command"][f.manifest["command"].index("--contract") + 1] = str(target)
    with tarfile.open(f.manifest["source_archive"], "w:gz", format=tarfile.PAX_FORMAT,
                      pax_headers={"comment": f.manifest["source_commit"]}) as output:
        for path in f.new.rglob("*"):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(f.new)), recursive=False)
    f.manifest["source_archive_sha256"] = boundary.sha(Path(f.manifest["source_archive"]))
    f.native_calls, f.native_error = 0, None
    f.q = lambda: boundary.BoundaryContinuationQueue(f.manifest, proc_root=f.proc,
        adapter_factory=lambda q: IntegratedFakeAdapter(q, f))


class IntegratedFakeAdapter(FakeAdapter):
    """Subprocess is mocked; files, owner identities, locks and accounts are real."""
    def validate_runtime(self, evidence):
        self.f.native_calls += 1
        if self.f.native_error:
            raise ValueError(self.f.native_error)
        return dict(status="passed", PPO_updates=0, qualification=None)


def test_v24_waits_for_normal_rec_and_does_not_signal_or_launch(f):
    retarget(f)
    q = f.q()
    assert q.target_revision == 24 and q.source_revision_index == 26
    assert q.step()["reason"] == "rec_normal_endpoint_not_complete"
    assert not f.signals and not f.launches and f.native_calls == 0


@pytest.mark.parametrize("artifact", [".task_campaign_boundary_v23.signal.intent.json",
    ".task_campaign_boundary_v23.launch.intent.json", "source_revision_025.json",
    "boundary_continuation_025.receipt.json"])
def test_old_source025_action_prevents_v24_supersession(f, artifact):
    retarget(f)
    write(f.root / artifact, {})
    assert f.q().step()["status"] == "attention"
    assert not f.signals and not f.launches and f.native_calls == 0


def test_old_waiting_observer_must_be_retired_before_new_queue(f):
    retarget(f)
    proc(f, 999, [f.manifest["command"][0], "-B",
        str(f.old / "scripts/queue_task_campaign_boundary_continuation.py"),
        "--manifest", str(f.root / "boundary025.json")], starttime=9001)
    assert f.q().step(check_only=True)["status"] == "attention"
    assert not f.signals and not f.launches and f.native_calls == 0


def test_v24_native_startup_precedes_once_only_launch_without_refund(f):
    retarget(f)
    q, speed = admitted(f)
    write(speed / "progress.json", dict(successful_updates=102, attempted_updates=103,
                                       actor_updates_in_block=92))
    f.records = {"speed/attempt_010/train": dict(domain="speed", consumed_updates=102, actor_updates=92)}
    q.step(); remove_owners(f); archiver_finished(f)
    account = Path(f.plan["budget_source_stage"]) / (f.plan["campaign_id"] + ".account.json")
    before = account.read_bytes()
    assert q.step()["status"] == "launch_submitted"
    assert f.q().step()["status"] == "launch_submitted"
    assert len(f.signals) == len(f.launches) == f.native_calls == 1
    assert account.read_bytes() == before
    receipt = boundary.read_json(q.queue_receipt)
    assert receipt["predecessor"]["totals"]["consumed_updates"] == 1103
    assert receipt["native_startup"]["PPO_updates"] == 0


def test_failed_native_startup_does_not_submit_training(f):
    retarget(f)
    q, _ = admitted(f)
    q.step(); remove_owners(f); archiver_finished(f)
    f.native_error = "actual scene startup rejected"
    assert q.step()["status"] == "attention"
    assert f.native_calls == 1 and not f.launches and not q.intent.exists()


def native_evidence(f):
    retarget(f)
    q = f.q()
    path = checkpoint(f, "recovery", f.rec_dir, count=100, task_clock=15000, final=True)
    seal = boundary.read_json(path.with_suffix(".checkpoint.json"))
    seal["stage_actor_update"] = 28473
    write(path.with_suffix(".checkpoint.json"), seal)
    admission = dict(schema="v6-integrated-production-admission-v1", qualification=None,
        selection=dict(mode="rebase_from_normal_REC", checkpoint=dict(path=str(path)),
                       fresh_domain_actor_updates=0), target_contract_sha256=f.rows["speed"]["sha256"])
    return q, dict(integrated_admission=admission,
        totals=dict(consumed_updates=42100, actor_updates_consumed=41920))


def scene_process(q, evidence, calls, *, override=None):
    def run(argv, **kwargs):
        calls.append(argv)
        path = Path(boundary.flag(argv, "--run-dir")) / "report.json"
        selection_sha = boundary.flag(argv, "--integrated-source-selection-sha256")
        report = dict(status="scene_validated", successful_updates=0, attempted_updates=0,
            contract_sha256=evidence["integrated_admission"]["target_contract_sha256"],
            integrated_admission_sha256=selection_sha,
            startup=dict(num_envs=4096, runtime_preflight=dict(status="runtime_configuration_passed",
                scoped_validation="v6-integrated-preflight-candidate-v1")),
            scene_capacity=dict(policy_steps=1, physics_steps_per_env=4))
        if override:
            override(report)
        write(path, report)
        return SimpleNamespace(returncode=0)
    return run


def test_native_scene_uses_actual_rec_clocks_and_zero_fresh_clock_once(f, monkeypatch):
    q, evidence = native_evidence(f)
    calls = []
    monkeypatch.setattr(boundary.subprocess, "run", scene_process(q, evidence, calls))
    adapter = boundary.NativeAdapter(q)
    result = adapter.validate_runtime(evidence)
    assert result["status"] == "passed" and result["PPO_updates"] == 0
    assert adapter.validate_runtime(evidence) == result and len(calls) == 1
    argv = calls[0]
    for flag, value in [("--num-envs", "4096"), ("--updates", "1"), ("--validate-scene", "1"),
        ("--seed", "290929"), ("--stage-actor-update", "28473"), ("--task-actor-update", "0"),
        ("--consumed-updates", "42100"), ("--actor-updates-consumed", "41920")]:
        assert boundary.flag(argv, flag) == value
    assert boundary.read_json(Path(boundary.flag(argv, "--integrated-source-selection-file"))) == evidence["integrated_admission"]


@pytest.mark.parametrize("fault", ["mode", "qualification", "schema"])
def test_native_rejects_non_admitted_source_before_invocation(f, fault):
    q, evidence = native_evidence(f)
    admission = evidence["integrated_admission"]
    if fault == "mode":
        admission["selection"]["mode"] = "old_own_speed"
    elif fault == "qualification":
        admission["qualification"] = "passed"
    else:
        admission["schema"] = "candidate-only"
    with pytest.raises(ValueError):
        boundary.NativeAdapter(q).validate_runtime(evidence)
    assert not (q.run_root / "boundary_integrated_026.native").exists()


@pytest.mark.parametrize("fault", ["updates", "num_envs", "admission", "contract", "physics"])
def test_native_bad_actual_report_is_retained_and_never_retried(f, monkeypatch, fault):
    q, evidence = native_evidence(f)
    calls = []
    def override(report):
        if fault == "updates": report["successful_updates"] = 1
        elif fault == "num_envs": report["startup"]["num_envs"] = 20
        elif fault == "admission": report["integrated_admission_sha256"] = "0" * 64
        elif fault == "contract": report["contract_sha256"] = "0" * 64
        else: report["scene_capacity"]["physics_steps_per_env"] = 20
    monkeypatch.setattr(boundary.subprocess, "run", scene_process(q, evidence, calls, override=override))
    adapter = boundary.NativeAdapter(q)
    with pytest.raises(ValueError, match="failed"):
        adapter.validate_runtime(evidence)
    with pytest.raises(ValueError, match="receipt"):
        adapter.validate_runtime(evidence)
    assert len(calls) == 1


def test_partial_native_intent_cannot_resubmit_after_timeout(f, monkeypatch):
    q, evidence = native_evidence(f)
    calls = []
    def timeout(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 720)
    monkeypatch.setattr(boundary.subprocess, "run", timeout)
    adapter = boundary.NativeAdapter(q)
    with pytest.raises(subprocess.TimeoutExpired):
        adapter.validate_runtime(evidence)
    with pytest.raises(ValueError, match="Partial"):
        adapter.validate_runtime(evidence)
    assert len(calls) == 1
