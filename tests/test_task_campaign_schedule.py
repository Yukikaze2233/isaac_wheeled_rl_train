"""Actor-clock completion, full-state recovery and nonblocking domain progression."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner
from test_task_campaign_v2 import continuous_job


def read(name):
    return json.loads((ROOT / name).read_text())


PLAN = read("contracts/v6_task_campaign_v7.json")


def test_scheduler_only_migration_preserves_all_worker_bytes_and_30_role_bindings():
    old = read("contracts/v6_task_campaign_v6.json")
    receipt = campaign.validate_plan_migration(old, PLAN, PLAN["revision_migration"]["run_directory"])
    assert receipt["consumption"] == "preserved_no_refund"
    assert receipt["actor_critic_optimizer"] == "preserved"
    assert old["domains"] == PLAN["domains"]
    assert len(runner.validate_contracts(PLAN)) == 30
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(old, PLAN, "/other/run")


@pytest.mark.parametrize("field,value", [
    ("clock", "consumed_updates"), ("advance_requires_evaluation", True),
    ("result", "domain_qualified"), ("end_actor_updates", {"flat": 13500}),
    ("end_actor_updates", dict.fromkeys(campaign.DOMAINS, 20000)),
    ("end_actor_updates", dict.fromkeys(campaign.DOMAINS, 15000.0)),
])
def test_rejects_missing_or_unbounded_schedule_and_fake_qualification(field, value):
    bad = deepcopy(PLAN)
    bad["domain_completion"]["schedule"][field] = value
    with pytest.raises(ValueError):
        campaign.validate_plan(bad)


@pytest.mark.parametrize("domain,clock,spent,remaining_warmup,expected_updates", [
    ("flat", 12997, 14000, 0, 3),   # consumption and learning clocks differ
    ("flat", 12997, 14000, 2, 5),   # resume only unfinished critic adaptation
    ("flat", 13000, 14000, 0, 0),   # crash after final seal must not retrain
    ("recovery", 0, 0, 0, 15010),  # new task clock zero + one native adaptation
    ("flat", 10000, 19998, 0, 2),  # charged cap wins over missing lineage
])
def test_exact_actor_endpoint_resumes_once_without_evaluation_or_refund(
        tmp_path, monkeypatch, domain, clock, spent, remaining_warmup, expected_updates):
    job = continuous_job(tmp_path, PLAN)
    row = next(r for r in PLAN["domains"] if r["name"] == domain)
    parent_domain = domain if domain == "flat" else "flat"
    source_row = next(r for r in PLAN["domains"] if r["name"] == parent_domain)
    source = tmp_path / parent_domain / "attempt_000/train/model_final.pt"
    source.parent.mkdir(parents=True)
    infos = dict(task_campaign=dict(domain=parent_domain), task_actor_update=clock if domain == "flat" else 13500,
        successful_updates_total=1000 + spent, stage_actor_update=50000,
        contract_sha256=source_row["sha256"], critic_warmup_remaining=remaining_warmup)
    torch.save(dict(infos=infos), source)
    runner.write_json(source.with_suffix(".contract.json"), read(source_row["contract"]))
    runner.write_json(source.with_suffix(".checkpoint.json"), infos)
    job.records = {f"{domain}/previous": dict(domain=domain, consumed_updates=spent, actor_updates=min(spent, clock))}
    job.report.update(campaign.account_updates(PLAN, job.records))
    job.persist = lambda: job.report.update(campaign.account_updates(PLAN, job.records))
    monkeypatch.setattr(job, "learning_source", lambda _: source)
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda _: 1000 + spent)
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: pytest.fail("schedule must not block on evaluation"))
    calls = []

    def execute(command, log, directory):
        calls.append(command)
        assert int(command[command.index("--updates") + 1]) == expected_updates
        assert int(command[command.index("--task-actor-update") + 1]) == clock
        assert int(command[command.index("--stage-actor-update") + 1]) == 50000
        warmup = remaining_warmup if domain == "flat" else 10
        runner.write_json(directory / "completion.json", dict(status="completed",
            successful_updates=expected_updates, attempted_updates=expected_updates,
            actor_updates_in_block=max(0, expected_updates - warmup)))
        final = directory / "model_final.pt"
        torch.save(dict(infos=dict(infos, task_campaign=dict(domain=domain))), final)
        runner.write_json(final.with_suffix(".checkpoint.json"), dict(
            task_campaign=dict(domain=domain), task_actor_update=clock + max(0, expected_updates - warmup)))
        return 0

    monkeypatch.setattr(job, "execute", execute)
    job.run_domain(row)
    record = job.report["domains"][domain]
    assert record["domain_qualified"] is None
    assert record["assessment_status"] == "not_run"
    assert record["consumed_updates"] == spent + expected_updates
    assert len(calls) == int(expected_updates > 0)
    assert record["status"] == ("domain_budget_exhausted_unassessed" if spent == 19998 else
                                 "domain_schedule_finished_unassessed")
    assert record["schedule_completion"]["clock"] == "domain_actor_updates"
    assert job.report["consumed_updates"] == 1000 + spent + expected_updates
    job.run_domain(row)
    assert len(calls) == int(expected_updates > 0), "completed domain must not reopen"


def test_premature_native_completion_cannot_claim_scheduled_finish(tmp_path, monkeypatch):
    job = continuous_job(tmp_path, PLAN)
    row = PLAN["domains"][0]
    source = tmp_path / "flat/attempt_000/train/model_final.pt"
    source.parent.mkdir(parents=True)
    torch.save(dict(infos=dict(task_campaign=dict(domain="flat"), task_actor_update=13500,
        successful_updates_total=15000, contract_sha256=row["sha256"], critic_warmup_remaining=0)), source)
    runner.write_json(source.with_suffix(".checkpoint.json"), dict(task_actor_update=12000))
    monkeypatch.setattr(job, "learning_source", lambda _: source)
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda _: 15000)
    with pytest.raises(RuntimeError, match="endpoint"):
        job.run_domain(row)
