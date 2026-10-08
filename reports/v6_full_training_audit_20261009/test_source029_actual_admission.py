"""Exercise deployment admission against real native CPU full-state APIs.

Uses the isolated repair worktree's canonical v26 fixture. Only model SHA pins
are synthetic; consumption41393/actor41223/REC14188/allowance14907 are real
fixture arithmetic. No AppLauncher, environment, learning, process or SSH.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import repair_known_integrated_origin as repair
from test_integrated_early_production import early_frozen
from test_integrated_heading_revision import heading_frozen
from test_integrated_lineage import checkpoint, write
from test_integrated_production import frozen, job_for, read, static_bundle
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis.task_campaign import account_updates, CAMPAIGN_ID

HELPER = Path(__file__).with_name("deploy_source029.py")
SPEC = importlib.util.spec_from_file_location("source029_actual_fixture_helper", HELPER)
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


class FixtureWrapperRoot:
    """Adapt only the helper's RUN/train location to the native test run_root."""
    def __init__(self, train):
        self.train = train

    def __truediv__(self, name):
        assert name == "train"
        return self.train


@pytest.fixture
def exact_clock_stopped(early_frozen, monkeypatch, tmp_path):
    f = early_frozen
    f.records = {domain: dict(domain=domain, consumed_updates=consumed, actor_updates=actor)
                 for domain, consumed, actor in [
                     ("flat", 13020, 12990), ("recovery", 14483, 14423), ("speed", 5093, 5063),
                     ("terrain", 2206, 2186), ("jump", 5572, 5562)]}
    totals = account_updates(f.plan, f.records)
    assert totals["consumed_updates"] == 41374 and totals["actor_updates_consumed"] == 41204
    f.rec = checkpoint(f, "recovery", count=41374, task=14169, lineage=27627, final=True,
                       actor_consumed=41204)
    old_complete = read(Path(f.rec["path"]).parent / "completion.json")
    old_complete.update(status="stopped", checkpoint_sha256=f.rec["sha256"])
    write(Path(f.rec["path"]).parent / "completion.json", old_complete)
    f.report.update(totals)
    f.report["domains"]["recovery"].update(status="domain_reopened_unassessed", domain_qualified=None)
    f.report["domains"]["recovery"].pop("schedule_completion", None)
    for name in ("campaign_state.json", "completion.json"):
        write(f.run / name, f.report)
        (f.run / production.EARLY_ORIGIN_DIRECTORY / name).write_bytes((f.run / name).read_bytes())
    write(f.claim, dict(campaign_id=CAMPAIGN_ID, run_directory=str(f.run),
        contract_sha256=production.SOURCE_PLAN_SHA, attempts=f.records,
        rollback_refunds_consumed_updates=False, **totals))
    job = job_for(f, monkeypatch)
    job.args.contract = f.source / production.artifact_files(26)["plan"]
    job.open_integrated_account()
    f.latest = checkpoint(f, "recovery", count=41393, task=14188, lineage=27646, final=True,
                          suffix="010", actor_consumed=41223)
    latest = Path(f.latest["path"])
    write(latest.parent / "completion.json", dict(status="stopped", parent_updates=41374,
        successful_updates=19, task_actor_update=14188, checkpoint_sha256=f.latest["sha256"]))
    write(latest.parent / "progress.json", dict(successful_updates=19, attempted_updates=19, actor_updates_in_block=19))
    job.records["recovery/attempt_010/train"] = dict(domain="recovery", consumed_updates=19, actor_updates=19)
    job.persist()
    (f.run / "completion.json").write_bytes((f.run / "campaign_state.json").read_bytes())
    f.proc = tmp_path / "empty_proc"
    f.proc.mkdir()
    expected = deepcopy(repair.EXPECTED)
    expected.update(previous_checkpoint_sha256=f.rec["sha256"], checkpoint_sha256=f.latest["sha256"])
    monkeypatch.setattr(repair, "EXPECTED", expected)
    monkeypatch.setattr(deploy, "EXPECTED_PARENT", f.latest["sha256"])
    return f


def test_real_source029_correction_and_helper_admission_exact_clocks(exact_clock_stopped):
    f = exact_clock_stopped
    raw_before = {p: p.read_bytes() for p in (f.run / production.EARLY_ORIGIN_DIRECTORY).glob("*.json")}
    correction = repair.correct_origin(f.plan, f.source, f.run, apply=True, proc_root=f.proc)
    assert correction["status"] == "corrected"
    args = SimpleNamespace(run_root=FixtureWrapperRoot(f.run), new_source_root=f.source)
    before = {p: p.read_bytes() for p in [f.claim, f.state, f.run / "completion.json"]}
    admission = deploy.launch_evidence(args, f.plan)
    assert admission["selection"]["checkpoint"] == f.latest
    assert admission["selection"]["learning_lineage_updates"] == 27646
    assert admission["selection"]["origin_recovery_actor_update"] == 14188
    assert admission["allowance"]["requested_total_updates"] == 14907
    assert admission["allowance"]["requested_actor_updates"] == 14897
    assert before == {p: p.read_bytes() for p in before}
    assert raw_before == {p: p.read_bytes() for p in raw_before}


def test_real_actual_dispatch_first_speed_only(early_frozen):
    assert deploy.cpu_dispatch(early_frozen.plan) == ["speed", "terrain", "jump", "stepjump"]


def test_helper_refuses_uncorrected_newer_rec_origin(exact_clock_stopped):
    f = exact_clock_stopped
    before = {p: p.read_bytes() for p in [f.claim, f.state, f.run / "completion.json"]}
    args = SimpleNamespace(run_root=FixtureWrapperRoot(f.run), new_source_root=f.source)
    with pytest.raises(ValueError):
        deploy.launch_evidence(args, f.plan)
    assert before == {p: p.read_bytes() for p in before}
