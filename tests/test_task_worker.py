"""Live-scene assessment cannot consume stale decisions or falsify task clocks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_chassis import digest, wait_for_task_assessment, task_confirmed_bins, write_progress, task_assessment_due
from wheeled_tasks.chassis.training_schedule import stage_actor_updates


def assessment(tmp_path, *, action="continue", qualified=False, corrupt=False):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"sealed policy")
    evaluations = []
    for seed in (290929, 290930):
        path = tmp_path / f"evaluation_{seed}.json"
        path.write_text(json.dumps({"seed": seed}))
        evaluations.append(dict(file=str(path), sha256=digest(path), seed=seed))
    calls = []

    def respond(_):
        request = json.loads((tmp_path / "assessment_request.json").read_text())
        calls.append(request)
        decision = dict(token=request["token"], decision=action, domain_qualified=qualified,
                        checkpoint_sha256=request["checkpoint_sha256"], evaluations=evaluations)
        if corrupt:
            decision["evaluations"][0]["sha256"] = "changed"
        (tmp_path / "assessment_decision.json").write_text(json.dumps(decision))

    return checkpoint, respond, calls


def test_wait_yields_scene_until_matching_two_seed_assessment(tmp_path):
    checkpoint, respond, calls = assessment(tmp_path)
    (tmp_path / "assessment_decision.json").write_text(json.dumps({"token": "previous checkpoint"}))
    decision = wait_for_task_assessment(tmp_path, checkpoint, 1000, 2000, lambda: None, sleep=respond)
    assert decision["decision"] == "continue"
    assert len(calls) == 1
    assert calls[0]["task_actor_update"] == 1000
    assert calls[0]["consumed_updates"] == 2000
    assert calls[0]["checkpoint_sha256"] == digest(checkpoint)


@pytest.mark.parametrize("qualified,corrupt", [(False, False), (True, True)])
def test_stop_rejects_unqualified_or_unauthenticated_results(tmp_path, qualified, corrupt):
    checkpoint, respond, _ = assessment(tmp_path, action="domain_qualified_stop", qualified=qualified, corrupt=corrupt)
    with pytest.raises(ValueError):
        wait_for_task_assessment(tmp_path, checkpoint, 1000, 2000, lambda: None, sleep=respond)


def test_stop_retains_authenticated_decision(tmp_path):
    checkpoint, respond, _ = assessment(tmp_path, action="domain_qualified_stop", qualified=True)
    decision = wait_for_task_assessment(tmp_path, checkpoint, 1000, 2000, lambda: None, sleep=respond)
    assert decision["domain_qualified"]


def test_task_schedule_is_separate_from_source_actor_lineage():
    env = SimpleNamespace(cfg={"task_campaign": {"domain": "jump"}}, stage_actor_update=10368)
    assert stage_actor_updates(env) == 0
    env.task_actor_update = 80
    assert stage_actor_updates(env) == 80
    assert env.stage_actor_update == 10368
    env.cfg = {}
    assert stage_actor_updates(env) == 10368


def test_budget_stop_interrupts_wait(tmp_path):
    checkpoint, _, _ = assessment(tmp_path)

    def stopped():
        raise InterruptedError("operator stopped")

    with pytest.raises(InterruptedError):
        wait_for_task_assessment(tmp_path, checkpoint, 1000, 2000, stopped)


def test_confirmed_jump_bins_cannot_regress_or_escape_frozen_domain():
    assert task_confirmed_bins([.03], {"confirmed_jump_bins_m": [.03, .06]}, [.03, .06]) == [.03, .06]
    assert task_confirmed_bins([.03], {}, [.03, .06]) == [.03]
    for bins in ([], [.03, .8], [float("nan")]):
        with pytest.raises(ValueError):
            task_confirmed_bins([.03], {"confirmed_jump_bins_m": bins}, [.03, .06])


def test_reserved_update_persists_and_is_timed(tmp_path):
    from wheeled_tasks.chassis.training_timing import TrainingTiming
    timer = TrainingTiming()
    write_progress(tmp_path, {"successful_updates": 0, "attempted_updates": 1}, timer)
    assert json.loads((tmp_path / "progress.json").read_text())["attempted_updates"] == 1
    assert timer.finish_iteration()["Timing/progress_io_calls"] == 1


def test_continuous_worker_never_requests_a_blocking_assessment():
    continuous = {"task_campaign": {"evaluation_interval": 0}}
    legacy = {"task_campaign": {"evaluation_interval": 1000}}
    for update in (0, 1, 999, 1000, 2000, 19999, 20000):
        assert not task_assessment_due(continuous, update)
        assert not task_assessment_due({}, update)
        assert task_assessment_due(legacy, update) == (update > 0 and update % 1000 == 0)
