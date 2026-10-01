"""4096-env migration, real full-state continuation and reviewed early domain finish."""
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner
from train_chassis import resume_critic_warmup
from test_task_campaign_v2 import continuous_job


def read(name):
    return json.loads((ROOT / name).read_text())


PLAN = read("contracts/v6_task_campaign_v5.json")


def test_strict_resize_preserves_reward_control_cases_caps_and_curriculum_clock():
    previous = read("contracts/v6_task_campaign_v4.json")
    campaign.validate_plan_migration(previous, PLAN, PLAN["revision_migration"]["run_directory"])
    assert len(runner.validate_contracts(PLAN)) == 30
    for old, new in zip(previous["domains"], PLAN["domains"], strict=True):
        source, target = read(old["contract"]), read(new["contract"])
        campaign.validate_task_resume(source, target)
        campaign.validate_task_resume(target, target)
        campaign.validate_task_budget(target, ROOT)
        assert target["target_num_envs"] == 4096
        assert target["num_mini_batches"] * target["ppo_minibatch_samples"] == 4096 * 24
        for key in ("physics_dt", "policy_dt", "v5_control", "actuator_response", "dense_tracking",
                    "landing_tracking", "skill_specs", "scene_groups", "evaluation", "new_asset_curriculum",
                    "curriculum_reference_batch", "learning_rate", "learning_rate_schedule"):
            assert target[key] == source[key]
        for key, value in (("physics_dt", .001), ("learning_rate", .0002), ("target_num_envs", 8192),
                           ("success_event_bonus", 5.), ("num_mini_batches", 10), ("curriculum_reference_batch", 98304)):
            broken = deepcopy(target); broken[key] = value
            with pytest.raises(ValueError):
                campaign.validate_task_resume(source, broken)
    for count in (4096, 8192):
        broken = deepcopy(previous); broken["target_num_envs"] = count
        with pytest.raises(ValueError):
            campaign.validate_plan(broken)


def test_resize_does_not_repeat_critic_adaptation_and_new_task_still_adapts():
    config = read(PLAN["domains"][0]["contract"])
    old_sha = config["task_campaign"]["revision_migration"]["source_contract_sha256"]
    infos = dict(contract_sha256=old_sha, task_campaign=dict(domain="flat"),
                 successful_updates_total=1800, critic_warmup_remaining=0)
    assert resume_critic_warmup(config, infos, contract_sha256="v5", task_change=True) == 0
    infos["critic_warmup_remaining"] = 3
    assert resume_critic_warmup(config, infos, contract_sha256="v5", task_change=True) == 3
    infos["contract_sha256"] = "v5"
    assert resume_critic_warmup(config, infos, contract_sha256="v5", task_change=True) == 3
    infos["task_campaign"]["domain"] = "recovery"
    assert resume_critic_warmup(config, infos, contract_sha256="next", task_change=True) == 10


def review_fixture(tmp_path, monkeypatch):
    row = PLAN["domains"][0]
    checkpoint = tmp_path / "flat/attempt_005/train/checkpoints/update_00001801/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"immutable actor critic Adam learning state")
    runner.write_json(checkpoint.with_suffix(".checkpoint.json"), dict(contract_sha256=row["sha256"],
        task_campaign=dict(domain="flat"), task_actor_update=790, learning_updates=1801))
    monkeypatch.setattr("wheeled_tasks.chassis.full_curriculum.checkpoint_update_count", lambda _: 1801)
    evidence = []
    for kind in ("curve_review", "behavior_review"):
        path = tmp_path / "reviews" / (kind + ".json")
        runner.write_json(path, dict(checkpoint_sha256=campaign.digest(checkpoint),
            observations={"checked": "stable task tracking / offline observed behavior"}))
        evidence.append(dict(kind=kind, file=str(path), sha256=campaign.digest(path)))
    request = dict(schema="task-domain-review-v1", campaign_id=campaign.CAMPAIGN_ID,
        plan_sha256=campaign.digest(ROOT / "contracts/v6_task_campaign_v5.json"), domain="flat",
        decision="finish_domain", reason="reviewed domain candidate; cap is optional",
        checkpoint=str(checkpoint), checkpoint_sha256=campaign.digest(checkpoint), evidence=evidence)
    return checkpoint, request


def test_review_requires_matching_immutable_snapshot_and_both_evidence_types(tmp_path, monkeypatch):
    checkpoint, request = review_fixture(tmp_path, monkeypatch)
    row = PLAN["domains"][0]
    assert campaign.validate_domain_review(PLAN, tmp_path, row, request, 1801)["domain_qualified"] is None
    for field, value in (("domain", "speed"), ("plan_sha256", "0"*64),
                         ("checkpoint_sha256", "0"*64), ("reason", ""), ("evidence", request["evidence"][:1])):
        broken = deepcopy(request); broken[field] = value
        with pytest.raises(ValueError):
            campaign.validate_domain_review(PLAN, tmp_path, row, broken, 1801)
    with pytest.raises(ValueError):
        campaign.validate_domain_review(PLAN, tmp_path, row, request, 1800)
    request["evidence"][0]["sha256"] = "0"*64
    with pytest.raises(ValueError):
        campaign.validate_domain_review(PLAN, tmp_path, row, request, 1801)


def test_invalid_review_does_not_stop_ppo_and_valid_review_seals_only_child(tmp_path, monkeypatch):
    checkpoint, request = review_fixture(tmp_path, monkeypatch)
    job = continuous_job(tmp_path, PLAN)
    job.active = PLAN["domains"][0]
    job.report["consumed_updates"] = 1801
    signals = []
    job.child = SimpleNamespace(poll=lambda: None, send_signal=signals.append)
    directory = checkpoint.parents[2]
    runner.write_json(directory / "progress.json", dict(task_actor_update=790))
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda _: 1801)
    runner.write_json(tmp_path / "domain_review_request.json", dict(bad=True))
    job.review_request(directory)
    assert not signals and not job.stop_requested
    assert read_response(tmp_path)["status"] == "invalid_review_training_continues"
    runner.write_json(tmp_path / "domain_review_request.json", request)
    job.review_request(directory)
    job.review_request(directory)
    assert signals == [signal.SIGTERM] and not job.stop_requested
    assert job.report["domains"]["flat"]["pending_review"]["domain_qualified"] is None


def read_response(root):
    return json.loads((root / "domain_review_response.json").read_text())


def test_early_review_uses_latest_full_state_charges_work_and_does_not_reopen_domain(tmp_path, monkeypatch):
    checkpoint, request = review_fixture(tmp_path, monkeypatch)
    job = continuous_job(tmp_path, PLAN)
    row = PLAN["domains"][0]
    job.records = {"flat/attempt_000/train": dict(domain="flat", consumed_updates=801, actor_updates=791)}
    job.report.update(campaign.account_updates(PLAN, job.records))
    job.persist = lambda: job.report.update(campaign.account_updates(PLAN, job.records))
    monkeypatch.setattr(job, "learning_source", lambda _: checkpoint)
    monkeypatch.setattr(job, "train_command", lambda *a: ["native full-state PPO"])
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda _: 1801)
    monkeypatch.setattr(job, "evaluate_pair", lambda *a: pytest.fail("review cannot start blocking assessment"))

    def execute(command, log, directory):
        job._active_review = campaign.validate_domain_review(PLAN, tmp_path, row, request, 1801)
        runner.write_json(directory / "completion.json", dict(status="stopped", successful_updates=2,
            attempted_updates=2, actor_updates_in_block=2))
        (directory / "model_final.pt").write_bytes(b"latest state after review; two more charged updates")
        return 0

    monkeypatch.setattr(job, "execute", execute)
    job.run_domain(row)
    result = job.report["domains"]["flat"]
    assert result["status"] == "domain_review_finished_unassessed"
    assert result["consumed_updates"] == 803
    assert result["domain_qualified"] is None
    assert Path(result["candidate_checkpoint"]).name == "model_final.pt"
    assert result["candidate_checkpoint"] != str(checkpoint)
    monkeypatch.setattr(job, "execute", lambda *a: pytest.fail("reviewed domain must not restart on resume"))
    job.run_domain(row)
    assert campaign.account_updates(PLAN, job.records)["consumed_updates"] == 1803


def test_pending_review_crash_recovers_latest_snapshot_without_more_ppo(tmp_path, monkeypatch):
    checkpoint, request = review_fixture(tmp_path, monkeypatch)
    job = continuous_job(tmp_path, PLAN)
    job.records = {"flat/attempt_000/train": dict(domain="flat", consumed_updates=801, actor_updates=791)}
    job.report.update(campaign.account_updates(PLAN, job.records))
    job.report["domains"]["flat"] = dict(pending_review=request)
    monkeypatch.setattr(job, "learning_source", lambda _: checkpoint)
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda _: 1801)
    monkeypatch.setattr(job, "execute", lambda *a: pytest.fail("durable review must not lose early finish on recovery"))
    job.run_domain(PLAN["domains"][0])
    assert job.report["domains"]["flat"]["status"] == "domain_review_finished_unassessed"
    assert campaign.account_updates(PLAN, job.records)["consumed_updates"] == 1801
