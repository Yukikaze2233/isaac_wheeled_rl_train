"""A resumes learning while all discarded-fork work and old protection survive."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis import continuation
from wheeled_tasks.chassis.policy_transfer import validate_reward_resume

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_chassis_continuation import Continuation, validate_contracts, validate_plan


def read(name):
    return json.loads((ROOT / name).read_text())


@pytest.fixture
def plan():
    return read("contracts/v6_s1_continuation_v1.json")


def test_exact_A_migration_and_all_design_roles(plan):
    source = read("contracts/v6_s1_A_fixed_v1.json")
    target = read(plan["stages"][0]["contract"])
    assert validate_reward_resume(source, target)["critic_warmup_updates"] == 0
    assert len(validate_contracts(plan)) == 5
    for key, value in (("physics_dt", .005), ("actor_dim", 36), ("learning_rate_schedule", "adaptive"),
                       ("resume_critic_warmup_updates", 10), ("checkpoint_first_update", 659)):
        changed = deepcopy(target)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_reward_resume(source, changed)
    for key in ("dense_tracking", "new_asset_curriculum", "evaluation", "actuator_response", "usb_transport"):
        changed = deepcopy(target)
        changed[key]["unreviewed"] = True
        with pytest.raises(ValueError):
            validate_reward_resume(source, changed)


def test_worker_must_keep_A_learning_clock_and_AB_consumption(plan, monkeypatch):
    config = read(plan["stages"][0]["contract"])
    args = SimpleNamespace(resume_reward_change=True, resume="A.pt", consumed_updates=698,
        actor_updates_consumed=678, stage_actor_update=368, updates=40, learning_rate_scale=1.,
        num_envs=20000, seed=290927, block_index=0, global_actor_update_offset=0)
    monkeypatch.setattr(continuation, "digest", lambda _: continuation.CHECKPOINT_SHA)
    continuation.validate_worker(config, args)
    for key, value in (("consumed_updates", 658), ("actor_updates_consumed", 648),
                       ("stage_actor_update", 698), ("updates", 302), ("learning_rate_scale", .5)):
        changed = deepcopy(args)
        setattr(changed, key, value)
        with pytest.raises(ValueError):
            continuation.validate_worker(config, changed)
    monkeypatch.setattr(continuation, "digest", lambda _: "B")
    with pytest.raises(ValueError):
        continuation.validate_worker(config, args)


def job(tmp_path):
    task = Continuation(SimpleNamespace(contract=ROOT / "contracts/v6_s1_continuation_v1.json",
        run_dir=tmp_path / "run", max_runtime_seconds=100, transfer=None, stage="S1",
        num_envs=20000, device="cpu"))
    task.root.mkdir()
    task.claim = tmp_path / "claim.json"
    task.claimed = True
    return task


def test_partial_failure_consumes_work_and_cannot_restart_with_fresh_credit(tmp_path, plan, monkeypatch):
    task = job(tmp_path)
    output = task.root / "A_continue/train"
    def fail(*args, **kwargs):
        output.mkdir()
        (output / "completion.json").write_text(json.dumps(dict(status="failed", successful_updates=13,
            actor_updates_in_block=13)))
        return 1
    monkeypatch.setattr(task, "execute", fail)
    with pytest.raises(RuntimeError):
        task.run_child([], output, training=True)
    assert task.report["successful_updates"] == 711
    assert task.report["actor_updates_consumed"] == 691
    with pytest.raises(ValueError):
        task.account("A_continue", dict(successful_updates=0, actor_updates_in_block=0))
    with pytest.raises(FileExistsError):
        task.claim.open("x")
    task.account("A_continue", dict(successful_updates=40, actor_updates_in_block=40))
    assert task.report["successful_updates"] == 738
    assert task.report["actor_updates_consumed"] == 718
    assert json.loads(task.claim.read_text())["consumed_updates"] == 738
    with pytest.raises(ValueError):
        continuation.charged_updates([dict(successful_updates=10, actor_updates_in_block=0)])


def test_plan_rejects_dropped_old_failures_or_new_A_successes(plan):
    validate_plan(plan)
    assert len(plan["protected_case_names"]) == 39
    for key, value in (("protected_case_names", plan["protected_case_names"][1:]),
                       ("inherited_recovery_attempts", []), ("source_consumed_updates", 658),
                       ("study_update_ceiling", 302), ("automatic_promotion", True)):
        changed = deepcopy(plan)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_plan(changed)


def test_better_total_score_cannot_replace_protection_or_coverage():
    baseline = {"cases": {"old_failure": {"passed": False}, "A_pass": {"passed": True},
                          "target": {"passed": False}}}
    better = deepcopy(baseline)
    better["cases"]["target"]["passed"] = True
    result = continuation.compare([baseline] * 2, [better] * 2, ["old_failure", "A_pass"], ["target"])
    assert result["protected_still_pending"] == ["old_failure"]
    assert not result["ready_for_full_audit"]
    assert not result["automatic_promotion"]
    del better["cases"]["A_pass"]
    with pytest.raises(ValueError, match="invalid_evaluation_coverage"):
        continuation.compare([baseline] * 2, [better] * 2, ["old_failure", "A_pass"], ["target"])


def test_stop_phase_uses_exact_evaluation_boundary():
    from wheeled_tasks.chassis.skill_commands import SkillCommands
    times = torch.tensor([2., 3., 4., 4.02, 5.98, 6.])
    count = len(times)
    fake = SimpleNamespace(env=SimpleNamespace(episode_length_buf=times / .02, policy_dt=.02),
        spin=torch.zeros(count, dtype=torch.bool), reference_target=torch.zeros(count, 2),
        start_stop_delay=torch.zeros(count),
        yaw=lambda: torch.zeros(count), sequence_cursor=torch.zeros(count, dtype=torch.long),
        batches=[(torch.arange(count), {"kind": "start_stop", "segment_seconds": 3.})],
        schedule=SimpleNamespace(diagnostics=lambda: {}))
    velocity = torch.tensor([[.1, .2, 0.]]).repeat(count, 1)
    data = SkillCommands.diagnostics(fake, velocity)
    assert data["braking_window"].tolist() == [False, True, True, False, False, False]
    assert data["settled_stop_window"].tolist() == [False, False, False, True, True, False]
    torch.testing.assert_close(data["settled_stop_speed"], velocity[:, :2].norm(dim=-1) * data["settled_stop_window"])


def test_stop_conditional_metrics_do_not_average_in_nonstop_samples(monkeypatch):
    from wheeled_tasks.chassis import learning_audit
    class Probe:
        def __init__(self, *args): pass
        def metrics(self): return {}
    monkeypatch.setattr(learning_audit, "PolicyUpdateProbe", Probe)
    commands = torch.tensor([[0., 0., .305], [0., 0., .305]])
    env = SimpleNamespace(num_envs=2, device="cpu", policy_dt=.02, commands=commands,
        skills=SimpleNamespace(batches=[(torch.arange(2), {"group_name": "stop", "kind": "start_stop"})]))
    audit = learning_audit.LearningAudit(env, SimpleNamespace(actor=None, storage=None))
    data = dict(commands=commands, reward=torch.zeros(2), planar_speed_world=torch.zeros(2),
        height=commands[:, 2], reasons={"fall": torch.zeros(2)}, done=torch.zeros(2, dtype=torch.bool),
        velocity=torch.tensor([[.3, .4, 0.], [10., 0., 0.]]),
        braking_window=torch.tensor([False, False]), settled_stop_window=torch.tensor([True, False]))
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        audit.observe(data, {"tracking": torch.zeros(2)})
    metrics = audit.drain()
    assert metrics["LearningAudit/stop/settled_stop/mean_speed_m_s"] == .5
    assert metrics["LearningAudit/stop/settled_stop/peak_speed_m_s"] == .5
    assert metrics["LearningAudit/stop/settled_stop/samples"] == 1
    assert metrics["LearningAudit/stop/braking/mean_speed_m_s"] == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert not audit.stop_sums.any() and not audit.stop_peak.any()
