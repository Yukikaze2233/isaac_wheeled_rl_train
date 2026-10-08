"""CPU lifecycle checks for the explicitly authorized source028 REC boundary.

The old queue fixtures provide real temporary files, proc identities and native
account arithmetic. Their model bytes are labelled synthetic; adapters mock
signals, native scene construction and launch. No App, GPU or PPO is executed.
"""
from copy import deepcopy
import fcntl
from pathlib import Path
import tarfile

import pytest

from test_task_campaign_boundary_continuation import (
    FakeAdapter, archiver_finished, boundary, checkpoint, f, proc, remove_owners, worker, write,
)
from test_integrated_boundary_continuation import scene_process
from wheeled_tasks.chassis.task_campaign import account_updates


def archive_source(f):
    archive = Path(f.manifest["source_archive"])
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT,
                      pax_headers={"comment": f.manifest["source_commit"]}) as stream:
        for path in f.new.rglob("*"):
            if path.is_file():
                stream.add(path, arcname=str(path.relative_to(f.new)), recursive=False)
    f.manifest["source_archive_sha256"] = boundary.sha(archive)


class EarlyAdapter(FakeAdapter):
    def prepare(self, job):
        result = super().prepare(job)
        rec = result["latest_own"]["recovery"]
        origin = boundary.read_json(self.f.train / "integrated_early_origin_v26/receipt.json")
        result["integrated_admission"] = dict(schema="v6-integrated-production-admission-v1", qualification=None,
            selection=dict(mode="rebase_from_authorized_REC",
                checkpoint=dict(path=rec["path"], sha256=rec["sha256"]),
                origin_recovery_actor_update=rec["task_actor_update"], fresh_domain_actor_updates=0),
            target_contract_sha256=self.f.rows["speed"]["sha256"],
            entry_admission=dict(record={key: origin[key] for key in
                ("early_stop_campaign_state", "early_stop_campaign_completion")}))
        if self.f.admission_error:
            self.f.admission_error(result)
        return result

    def validate_runtime(self, evidence):
        self.f.native_calls += 1
        if self.f.native_error:
            raise ValueError(self.f.native_error)
        return dict(status="passed", PPO_updates=0, qualification=None)


def early(f):
    target = f.new / "contracts/v6_task_campaign_v26.json"
    authorization = f.new / "contracts/v6_integrated_early_entry_authorization_v26.json"
    auth = dict(schema="v6-authorized-REC-early-stop-v1", lineage_id="source028-joint-serial-v1",
        decision="seal_current_REC_then_same_actor_serial", run_root=str(f.train),
        source_commit=boundary.OLD_COMMIT, source_plan_sha256=f.manifest["old_contract_sha256"],
        recovery_contract_sha256=f.rows["recovery"]["sha256"], planned_actor_endpoint=15000,
        minimum_actor_update=13868, qualification=None)
    write(authorization, auth)
    lineage_file = f.new / "contracts/v6_integrated_lineage_v26.json"
    write(lineage_file, dict(lineage_id=auth["lineage_id"], recovery_entry=dict(
        contract_file=f.rows["recovery"]["contract"], contract_sha256=f.rows["recovery"]["sha256"],
        early_stop=dict(authorization_file=str(authorization.relative_to(f.new)),
                        authorization_sha256=boundary.sha(authorization)))))
    plan = deepcopy(f.plan)
    plan.update(contract_id="v6-task-campaign-v26", seed=290927, integrated_continuation=dict(
        lineage_manifest=dict(file=str(lineage_file.relative_to(f.new)), sha256=boundary.sha(lineage_file))))
    for row in plan["domains"]:
        row["kind"] = "S3"
    write(target, plan)
    f.manifest.update(new_contract_relative_path=str(target.relative_to(f.new)),
        new_contract_sha256=boundary.sha(target), source_revision_receipt=str(f.root / "source_revision_028.json"),
        queue_receipt=str(f.root / "boundary_continuation_028.receipt.json"),
        status_path=str(f.root / "boundary_continuation_028.status.json"),
        launch_log=str(f.root / "boundary_continuation_028.launch.log"),
        authorization_file=str(authorization), authorization_sha256=boundary.sha(authorization))
    f.manifest["command"][f.manifest["command"].index("--contract") + 1] = str(target)
    archive_source(f)
    f.records = {f"{domain}/attempt_old/train": dict(domain=domain, consumed_updates=consumed, actor_updates=actors)
                 for domain, consumed, actors in (("flat", 10000, 10000), ("recovery", 16800, 16690),
                     ("speed", 5000, 4900), ("terrain", 2000, 2000), ("jump", 2000, 2000), ("stepjump", 1000, 1000))}
    totals = account_updates(f.plan, f.records)
    f.state.update(totals)
    write(f.train / "campaign_state.json", f.state)
    write(f.rec_dir / "progress.json", dict(pid=202, num_envs=4096,
        task_campaign=dict(domain="recovery"), parent_updates=30000, successful_updates=100,
        attempted_updates=101, actor_updates_in_block=90, task_actor_update=13868,
        global_budget_actor_updates=totals["actor_updates_consumed"] + 90))
    f.native_calls, f.native_error, f.admission_error = 0, None, None
    f.q = lambda: boundary.BoundaryContinuationQueue(f.manifest, proc_root=f.proc,
        adapter_factory=lambda q: EarlyAdapter(q, f))
    return f.q()


def stopped(f, *, task_clock=13873, count=30105):
    path = checkpoint(f, "recovery", f.rec_dir, count=count, task_clock=task_clock, final=True)
    write(f.rec_dir / "completion.json", dict(status="stopped", parent_updates=30000,
        successful_updates=count - 30000, task_actor_update=task_clock, checkpoint_sha256=boundary.sha(path)))
    write(f.rec_dir / "progress.json", dict(successful_updates=count-30000, attempted_updates=count-30000,
        actor_updates_in_block=task_clock-13778))
    remove_owners(f)
    archiver_finished(f)
    return path


def test_authorized_check_only_never_signals_or_launches(f):
    q = early(f)
    result = q.step(check_only=True)
    assert result["reason"] == "check_only_authorized_REC_admitted_no_signal_or_launch"
    assert result["rec_floor"]["task_actor_update"] == 13868
    assert not f.signals and not f.launches and not f.native_calls
    assert not q.signal_intent.exists() and not q.state_path.exists()
    assert not (f.train / "integrated_early_origin_v26").exists()
    assert q.lock == f.root / ".task_campaign_boundary_v22.lock"


def test_early_rec_is_sealed_then_same_actor_native_parent_without_speed_bridge(f):
    q = early(f)
    account = Path(f.plan["budget_source_stage"]) / (f.plan["campaign_id"] + ".account.json")
    before = account.read_bytes()
    assert q.step()["reason"] == "one_authorized_REC_supervisor_sigterm_submitted"
    intent = boundary.read_json(q.signal_intent)
    assert intent["worker"]["pid"] == 202 and intent["rec_floor"]["learning_updates"] == 30100
    assert "speed" not in intent and "rec" not in intent
    assert q.step()["reason"] == "old_owners_or_wrapper_still_active"
    final = stopped(f)
    assert q.step()["status"] == "launch_submitted"
    assert f.q().step()["status"] == "launch_submitted"
    assert len(f.signals) == len(f.launches) == f.native_calls == 1
    assert account.read_bytes() == before
    receipt = boundary.read_json(q.queue_receipt)
    admission = receipt["predecessor"]["integrated_admission"]
    assert admission["selection"]["checkpoint"]["path"] == str(final)
    assert admission["selection"]["origin_recovery_actor_update"] == 13873
    assert admission["qualification"] is None
    assert receipt["predecessor"]["totals"]["consumed_updates"] >= intent["rec_floor"]["consumed_updates"]
    for key in ("early_stop_campaign_state", "early_stop_campaign_completion"):
        pin = admission["entry_admission"]["record"][key]
        assert boundary.sha(pin["file"]) == pin["sha256"]
        assert Path(pin["file"]).read_bytes() == (f.train / Path(pin["file"]).name).read_bytes()
    assert not f.state["domains"].get("recovery", {}).get("schedule_completion")


@pytest.mark.parametrize("fault", ["authorization_sha", "lineage_sha", "decision", "plan", "floor", "outside", "qualified"])
def test_wrong_frozen_authorization_rejects_before_any_signal(f, fault):
    q = early(f)
    auth_path = Path(f.manifest["authorization_file"])
    if fault == "authorization_sha":
        f.manifest["authorization_sha256"] = "0" * 64
    elif fault == "outside":
        f.manifest["authorization_file"] = str(f.root / "outside.json")
        with pytest.raises(ValueError):
            f.q()
        return
    elif fault == "lineage_sha":
        target = Path(boundary.flag(f.manifest["command"], "--contract"))
        plan = boundary.read_json(target)
        plan["integrated_continuation"]["lineage_manifest"]["sha256"] = "0" * 64
        write(target, plan)
        f.manifest["new_contract_sha256"] = boundary.sha(target)
    else:
        auth = boundary.read_json(auth_path)
        key, value = {"decision": ("decision", "complete_REC_now"), "plan": ("source_plan_sha256", "0" * 64),
            "floor": ("minimum_actor_update", 0), "qualified": ("qualification", "passed")}[fault]
        auth[key] = value
        write(auth_path, auth)
        f.manifest["authorization_sha256"] = boundary.sha(auth_path)
        lineage_path = f.new / "contracts/v6_integrated_lineage_v26.json"
        lineage = boundary.read_json(lineage_path)
        lineage["recovery_entry"]["early_stop"]["authorization_sha256"] = boundary.sha(auth_path)
        write(lineage_path, lineage)
        target = Path(boundary.flag(f.manifest["command"], "--contract"))
        plan = boundary.read_json(target)
        plan["integrated_continuation"]["lineage_manifest"]["sha256"] = boundary.sha(lineage_path)
        write(target, plan)
        f.manifest["new_contract_sha256"] = boundary.sha(target)
    archive_source(f)
    assert f.q().step()["status"] == "attention"
    assert not f.signals and not f.launches and not q.signal_intent.exists()


@pytest.mark.parametrize("fault", ["speed_worker", "no_worker", "old_observer", "stale_pid", "below_floor", "at_endpoint", "missing_progress", "preflight"])
def test_live_ownership_and_floor_are_required_before_early_signal(f, fault):
    q = early(f)
    if fault == "speed_worker":
        worker(f, "speed", "attempt_010")
    elif fault == "no_worker":
        (f.proc / "202/cmdline").write_bytes(b"")
    elif fault == "old_observer":
        proc(f, 999, [f.manifest["command"][0], "-B", str(f.old / Path(boundary.__file__).name),
            "--manifest", str(f.root / "boundary027.json")], starttime=9001)
    elif fault == "missing_progress":
        (f.rec_dir / "progress.json").unlink()
    elif fault == "preflight":
        f.preflight_error = "original frozen native design rejected"
    else:
        progress = boundary.read_json(f.rec_dir / "progress.json")
        key, value = {"stale_pid": ("pid", 999), "below_floor": ("task_actor_update", 13867),
                      "at_endpoint": ("task_actor_update", 15000)}[fault]
        progress[key] = value
        write(f.rec_dir / "progress.json", progress)
    assert q.step()["status"] == "attention"
    assert not f.signals and not q.signal_intent.exists()


def test_pre_signal_pid_reuse_or_progress_regression_does_not_create_intent(f):
    q = early(f)
    def race():
        progress = boundary.read_json(f.rec_dir / "progress.json")
        progress["successful_updates"] -= 1
        write(f.rec_dir / "progress.json", progress)
    f.signal_race = race
    assert q.step()["status"] == "attention"
    assert not f.signals and not q.signal_intent.exists()


def test_ambiguous_signal_intent_is_never_resubmitted(f):
    q = early(f)
    original = q.adapter.signal_once
    def lost_ack(owner, before_signal):
        original(owner, before_signal)
        raise OSError("signal acknowledgment lost")
    q.adapter.signal_once = lost_ack
    assert q.step()["status"] == "attention"
    assert f.q().step()["reason"] == "ambiguous_signal_intent_never_resignal"
    assert len(f.signals) == 1 and not f.launches


@pytest.mark.parametrize("fault", ["below_task", "below_learning", "completed", "bad_completion_sha", "wrong_parent", "wrong_mode", "account_floor", "missing_archive"])
def test_stopped_final_and_account_floor_are_authenticated_before_native_or_launch(f, fault):
    q = early(f)
    q.step()
    final = stopped(f, task_clock=13867 if fault == "below_task" else 13873,
                    count=30099 if fault == "below_learning" else 30105)
    if fault in ("completed", "bad_completion_sha"):
        completion = boundary.read_json(final.parent / "completion.json")
        completion["status" if fault == "completed" else "checkpoint_sha256"] = "completed" if fault == "completed" else "0" * 64
        write(final.parent / "completion.json", completion)
        archiver_finished(f)
    elif fault == "missing_archive":
        delivery = boundary.read_json(f.root / "delivery.json")
        del delivery["files"][str((final.parent / "completion.json").relative_to(f.root))]
        write(f.root / "delivery.json", delivery)
    elif fault in ("wrong_parent", "wrong_mode", "account_floor"):
        def corrupt(result):
            selection = result["integrated_admission"]["selection"]
            if fault == "wrong_parent": selection["checkpoint"] = result["latest_own"]["speed"]
            elif fault == "wrong_mode": selection["mode"] = "rebase_from_normal_REC"
            else: result["totals"]["actor_updates_consumed"] = 0
        f.admission_error = corrupt
    assert q.step()["status"] == "attention"
    assert f.native_calls == 0 and not f.launches and not q.intent.exists()


def test_immutable_stopped_snapshots_reject_partial_or_changed_origin(f):
    q = early(f)
    q.step(); stopped(f)
    origin = f.train / "integrated_early_origin_v26"
    origin.mkdir()
    (origin / "campaign_state.json").write_bytes(b"partial preserved failure")
    assert q.step()["status"] == "attention"
    assert (origin / "campaign_state.json").read_bytes() == b"partial preserved failure"
    assert not f.native_calls and not f.launches


def test_native_failure_retains_once_signal_and_never_submits_ppo(f):
    q = early(f)
    q.step(); stopped(f)
    f.native_error = "actual original native scene rejected"
    assert q.step()["status"] == "attention"
    assert len(f.signals) == 1 and f.native_calls == 1 and not f.launches and not q.intent.exists()


def test_original_campaign_claim_lock_is_not_stolen(f):
    q = early(f)
    q.step(); stopped(f)
    claim = Path(f.plan["budget_source_stage"]) / (f.plan["campaign_id"] + ".account.json")
    with claim.with_suffix(".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert q.step()["reason"] == "native_account_lock_still_owned"
    assert not f.native_calls and not f.launches and not (f.train / "integrated_early_origin_v26").exists()


def test_native_adapter_uses_actual_early_rec_source_and_original_scene_checks(f, monkeypatch):
    q = early(f)
    q.step(); final = stopped(f)
    q.freeze_early_origin()
    evidence = q.adapter.prepare(boundary.read_json(f.root / "job.json"))
    calls = []
    monkeypatch.setattr(boundary.subprocess, "run", scene_process(q, evidence, calls))
    adapter = boundary.NativeAdapter(q)
    assert adapter.validate_runtime(evidence)["status"] == "passed"
    assert adapter.validate_runtime(evidence)["PPO_updates"] == 0 and len(calls) == 1
    argv = calls[0]
    assert boundary.flag(argv, "--resume") == str(final)
    for name, value in (("--task-actor-update", "0"), ("--stage-actor-update", "13873"),
                        ("--updates", "1"), ("--validate-scene", "1"), ("--num-envs", "4096")):
        assert boundary.flag(argv, name) == value
    assert Path(boundary.flag(argv, "--run-dir")).parent.name == "boundary_integrated_028.native"
