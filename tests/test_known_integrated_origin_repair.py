"""Native CPU full-state fixtures for the once-only misdirected REC correction.

Synthetic SHA/clock pins replace the hard production pins; neither environment,
optimizer update, process signal nor launch is executed.
"""
from copy import deepcopy
from pathlib import Path

import pytest

import repair_known_integrated_origin as repair
from test_integrated_early_production import early_frozen
from test_integrated_heading_revision import heading_frozen
from test_integrated_lineage import checkpoint, write
from test_integrated_production import frozen, job_for, read, static_bundle
from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis.task_campaign import account_updates


@pytest.fixture
def stopped(early_frozen, monkeypatch, tmp_path):
    f = early_frozen
    f.rec = checkpoint(f, "recovery", count=34000, task=14052, lineage=24052, final=True,
                       actor_consumed=33930)
    completion = read(Path(f.rec["path"]).parent / "completion.json")
    completion.update(status="stopped", checkpoint_sha256=f.rec["sha256"])
    write(Path(f.rec["path"]).parent / "completion.json", completion)
    f.report["domains"]["recovery"].update(status="domain_reopened_unassessed", domain_qualified=None)
    f.report["domains"]["recovery"].pop("schedule_completion", None)
    for name in ("campaign_state.json", "completion.json"):
        write(f.run / name, f.report)
        (f.run / production.EARLY_ORIGIN_DIRECTORY / name).write_bytes((f.run / name).read_bytes())
    job = job_for(f, monkeypatch)
    job.args.contract = f.source / production.artifact_files(26)["plan"]
    job.open_integrated_account()
    f.latest = checkpoint(f, "recovery", count=34019, task=14071, lineage=24071, final=True,
                          suffix="010", actor_consumed=33949)
    path = Path(f.latest["path"])
    write(path.parent / "completion.json", dict(status="stopped", parent_updates=34000,
        successful_updates=19, task_actor_update=14071, checkpoint_sha256=f.latest["sha256"]))
    progress = dict(successful_updates=19, attempted_updates=19, actor_updates_in_block=19)
    write(path.parent / "progress.json", progress)
    job.records["recovery/attempt_010/train"] = dict(domain="recovery", consumed_updates=19, actor_updates=19)
    job.persist()
    (f.run / "completion.json").write_bytes((f.run / "campaign_state.json").read_bytes())
    f.job = job
    f.proc = tmp_path / "proc"
    f.proc.mkdir()
    monkeypatch.setattr(repair, "EXPECTED", dict(previous_checkpoint_sha256=f.rec["sha256"],
        previous_consumed_updates=34000, previous_REC_actor=14052, checkpoint_sha256=f.latest["sha256"],
        consumed_updates=34019, actor_updates_consumed=33949, REC_actor=14071, learning_lineage_updates=24071,
        checkpoint_relative_path="recovery/attempt_010/train/model_final.pt"))
    return f


def invoke(f, *, apply=False):
    return repair.correct_origin(f.plan, f.source, f.run, apply=apply, proc_root=f.proc)


def test_check_only_preserves_all_bytes_and_creates_no_correction(stopped):
    f = stopped
    before = {p: p.read_bytes() for p in f.run.rglob("*.json")}
    before[f.claim] = f.claim.read_bytes()
    result = invoke(f)
    assert result["status"] == "validated" and result["retained_new_actor_updates"] == 19
    assert result["PPO_updates_by_repair"] == 0 and result["qualification"] is None
    assert before == {p: p.read_bytes() for p in before}
    assert not (f.run / repair.DIRECTORY).exists()


def test_corrects_latest_full_state_then_original_entry_restarts_idempotently(stopped, monkeypatch):
    f = stopped
    origin = f.run / production.EARLY_ORIGIN_DIRECTORY
    original_bytes = {p: p.read_bytes() for p in origin.glob("*.json")}
    before_migration = (f.run / "revisions/v21_to_v26_account_migration.json").read_bytes()
    before_records = read(f.claim)["attempts"]
    result = invoke(f, apply=True)
    assert result["status"] == "corrected" and result["checkpoint"] == f.latest
    directory = f.run / repair.DIRECTORY
    assert (directory / "before_migration.json").read_bytes() == before_migration
    assert original_bytes == {p: p.read_bytes() for p in original_bytes}
    assert read(f.claim)["attempts"] == before_records
    assert read(f.claim)["consumed_updates"] == 34019
    assert read(f.claim)["actor_updates_consumed"] == 33949
    report = read(f.state)
    assert report["domains"]["recovery"]["status"] == "domain_reopened_unassessed"
    assert report["domains"]["recovery"]["domain_qualified"] is None
    assert report["authorized_REC_early_stop"]["candidate_checkpoint"] == f.latest["path"]
    admission = production.prepare_admission(f.plan, f.source, f.run, account_path=f.claim, state_path=f.state)
    assert admission["selection"]["checkpoint"] == f.latest
    assert admission["allowance"]["charged_critic_updates"] == 10
    assert admission["selection"]["origin_recovery_actor_update"] == 14071
    assert admission["selection"]["fresh_joint_actor_updates"] == 0
    job = job_for(f, monkeypatch)
    job.args.contract = f.source / production.artifact_files(26)["plan"]
    job.open_integrated_account()
    once = deepcopy(job.report), deepcopy(job.records)
    job.open_integrated_account()
    assert (job.report, job.records) == once
    assert job.learning_source(next(r for r in f.plan["domains"] if r["name"] == "speed")) == Path(f.latest["path"])
    with pytest.raises(ValueError, match="already has artifacts"):
        invoke(f, apply=True)


@pytest.mark.parametrize("fault", ["wrong_parent", "wrong_sha", "joint_seal", "selected_joint", "refund", "running", "owner"])
def test_bad_parent_or_joint_state_or_refund_rejects_without_mutation(stopped, fault):
    f = stopped
    if fault == "wrong_parent":
        path = Path(f.latest["path"]).parent / "completion.json"
        value = read(path)
        value["parent_updates"] -= 1
        write(path, value)
    elif fault == "wrong_sha":
        repair.EXPECTED["checkpoint_sha256"] = "0" * 64
    elif fault == "joint_seal":
        write(f.run / "speed/attempt_bad/train/checkpoints/model.checkpoint.json", dict(integrated_lineage={"domain": "speed"}))
    elif fault == "selected_joint":
        value = read(f.claim)
        value["joint_learning_sources"] = {"speed": {"origin": "forbidden"}}
        write(f.claim, value)
    elif fault == "refund":
        value = read(f.claim)
        value["consumed_updates"] -= 19
        write(f.claim, value)
    elif fault == "running":
        value = read(f.state)
        value["status"] = "running"
        write(f.state, value)
    else:
        path = f.proc / "77/cmdline"
        path.parent.mkdir()
        path.write_bytes(b"python\0/source/scripts/run_task_campaign.py\0--run-dir\0" + str(f.run).encode() + b"\0")
    before = {p: p.read_bytes() for p in f.run.rglob("*.json")}
    before[f.claim] = f.claim.read_bytes()
    with pytest.raises(ValueError):
        invoke(f, apply=True)
    assert before == {p: p.read_bytes() for p in before}
    assert not (f.run / repair.DIRECTORY).exists()


@pytest.mark.parametrize("fault", ["intent_sha", "intent_schema", "snapshot_path"])
def test_correction_evidence_is_bound_and_cannot_redirect_an_origin(stopped, fault):
    f = stopped
    invoke(f, apply=True)
    report = read(f.state)
    record = report["authorized_REC_early_stop"]
    if fault == "intent_sha":
        record["early_stop_origin_correction"]["sha256"] = "0" * 64
    elif fault == "intent_schema":
        path = f.run / repair.DIRECTORY / "intent.json"
        value = read(path)
        value["schema"] = "automatic-generic-retry"
        write(path, value)
        record["early_stop_origin_correction"]["sha256"] = lineage.digest(path)
    else:
        record["early_stop_campaign_state"] = dict(file=str(f.run / production.EARLY_ORIGIN_DIRECTORY / "campaign_state.json"),
            sha256=lineage.digest(f.run / production.EARLY_ORIGIN_DIRECTORY / "campaign_state.json"))
    with pytest.raises(ValueError):
        lineage.admit_entry(f.binding, f.latest, record)
