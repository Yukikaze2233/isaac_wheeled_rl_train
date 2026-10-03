"""A sealed task review must not rebuild a worker or discard newer learning."""
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


@pytest.mark.parametrize("revision", ["v9", "v10"])
def test_sealed_recovery_review_skips_scene_and_preserves_latest_state(tmp_path, monkeypatch, revision):
    plan_file = ROOT / f"contracts/v6_task_campaign_{revision}.json"
    plan = json.loads(plan_file.read_text())
    row = next(r for r in plan["domains"] if r["name"] == "recovery")
    reviewed = tmp_path / "recovery/attempt_001/train/checkpoints/update_00016000/model.pt"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_bytes(b"immutable reviewed checkpoint")
    seal = dict(contract_sha256=row["sha256"], task_campaign=dict(domain="recovery"),
                task_actor_update=1960, successful_updates_total=16000)
    runner.write_json(reviewed.with_suffix(".checkpoint.json"), seal)
    latest = reviewed.parents[2] / "model_final.pt"
    latest_infos = dict(seal, task_actor_update=2294, successful_updates_total=16334,
                        critic_warmup_remaining=0, stage_actor_update=15662)
    torch.save(dict(infos=latest_infos), latest)
    runner.write_json(latest.with_suffix(".checkpoint.json"), latest_infos)
    counts = {str(reviewed): 16000, str(latest): 16334}
    monkeypatch.setattr("wheeled_tasks.chassis.full_curriculum.checkpoint_update_count",
                        lambda p: counts[str(p)])
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda p: counts[str(p)])
    evidence = []
    for kind in ("curve_review", "behavior_review"):
        path = tmp_path / "reviews" / f"{kind}.json"
        runner.write_json(path, dict(checkpoint_sha256=campaign.digest(reviewed),
            observations={"comparison": "mean policy regressed at matched fixed conditions"}))
        evidence.append(dict(kind=kind, file=str(path), sha256=campaign.digest(path)))
    request = dict(schema="task-domain-review-v1", campaign_id=campaign.CAMPAIGN_ID,
        plan_sha256=campaign.digest(plan_file), domain="recovery", decision="finish_domain",
        reason="preserve this research branch and continue speed from the unchanged flat parent",
        checkpoint=str(reviewed), checkpoint_sha256=campaign.digest(reviewed), evidence=evidence)
    runner.write_json(tmp_path / "domain_review_request.json", request)
    job = continuous_job(tmp_path, plan)
    job.child = None
    job.records = {
        "flat/attempt_007/train": dict(domain="flat", consumed_updates=13020, actor_updates=13000),
        "recovery/attempt_001/train": dict(domain="recovery", consumed_updates=2314, actor_updates=2294),
    }
    job.report.update(campaign.account_updates(plan, job.records))
    job.persist = lambda: job.report.update(campaign.account_updates(plan, job.records))
    monkeypatch.setattr(job, "learning_source", lambda _: latest)
    monkeypatch.setattr(job, "execute", lambda *a: pytest.fail("sealed review must not rebuild a scene"))
    monkeypatch.setattr(job, "evaluate_pair", lambda *a: pytest.fail("review cannot block on evaluation"))
    before = campaign.digest(latest)
    job.run_domain(row)
    result = job.report["domains"]["recovery"]
    assert result["status"] == "domain_review_finished_unassessed"
    assert result["domain_qualified"] is None
    assert result["candidate_checkpoint"] == str(latest)
    assert result["candidate_checkpoint_sha256"] == before
    assert result["finish_review"]["checkpoint"] == str(reviewed)
    assert result["schedule_completion"]["actual_actor_update"] == 2294
    assert result["consumed_updates"] == 2314
    assert job.report["consumed_updates"] == 16334
    job.run_domain(row)
    assert campaign.digest(latest) == before
    assert job.report["consumed_updates"] == 16334


@pytest.mark.parametrize("revision", ["v9", "v10"])
def test_entry_review_still_rejects_wrong_identity_without_finishing_task(tmp_path, revision):
    plan = json.loads((ROOT / f"contracts/v6_task_campaign_{revision}.json").read_text())
    job = continuous_job(tmp_path, plan)
    job.child = None
    job.active = next(r for r in plan["domains"] if r["name"] == "recovery")
    runner.write_json(tmp_path / "domain_review_request.json", dict(schema="task-domain-review-v1"))
    job.review_request()
    assert getattr(job, "_active_review", None) is None
    response = json.loads((tmp_path / "domain_review_response.json").read_text())
    assert response["status"] == "invalid_review_training_continues"
    assert not job.stop_requested
