"""Native lesson events and final-mask exposure, without a second outcome model."""
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis import learning_audit
from wheeled_tasks.chassis.recovery_prefix_curriculum import LESSONS, RecoveryPrefixCurriculum


def snapshot(levels):
    count = len(levels)
    return dict(recovery_enabled=torch.ones(count, dtype=torch.bool),
        recovery_prefix_lesson_id=torch.tensor(levels),
        recovery_ever_blended=torch.zeros(count, dtype=torch.bool),
        recovery_handover_count=torch.zeros(count, dtype=torch.long),
        recovery_stable_peak_seconds=torch.zeros(count),
        recovery_motion_released=torch.zeros(count, dtype=torch.bool),
        done=torch.zeros(count, dtype=torch.bool))


def test_events_survive_rollout_boundary_and_terminal_mask_is_authoritative():
    audit = learning_audit.RecoveryPhaseAudit(1, "cpu", 1.)
    data = snapshot([2])
    data["recovery_ever_blended"].fill_(True)
    data["recovery_handover_count"].fill_(1)
    # Pure RL was reached within this interval, but a blend tick excludes it.
    audit.observe(data, torch.tensor([False]))
    prefix = "RecoveryPhaseAudit/original_chain/"
    first = {key: value.item() for key, value in audit.metrics().items()}
    assert first[prefix + "first_pure_rl_started"] == 1
    assert first[prefix + "policy_owned_transitions"] == 0
    audit.clear_report()
    assert audit.seen.tolist() == [[True, True, False, False]]
    # An actor-caused terminal disarm can retain ownership of this interval.
    data["done"].fill_(True)
    audit.observe(data, torch.tensor([True]))
    ended = {key: value.item() for key, value in audit.metrics().items()}
    assert ended[prefix + "first_pure_rl_started"] == 0
    assert ended[prefix + "ended_with_pure_rl_started"] == 1
    assert ended[prefix + "policy_owned_transitions"] == 1
    audit.clear_report()
    # The next pre-reset snapshot belongs to a new, easier actual birth lesson.
    audit.observe(snapshot([0]), torch.tensor([False]))
    assert not audit.seen.any()
    assert audit.metrics()["RecoveryPhaseAudit/prefix_stand/observed_transitions"] == 1


def test_ended_funnel_joins_exact_producer_denominator_by_actual_lesson():
    data = snapshot([0, 0, 1, 1, 2, 2, -1])
    data["recovery_enabled"][-1] = False
    data["recovery_ever_blended"][:] = True
    data["recovery_handover_count"][:] = 1
    data["recovery_stable_peak_seconds"][:] = torch.tensor([1., .5, 1., .5, .5, 1., 1.])
    data["recovery_motion_released"][:] = torch.tensor([True, False, True, False, False, True, True])
    audit = learning_audit.RecoveryPhaseAudit(7, "cpu", 1.)
    audit.observe(data, torch.tensor([True, True, True, True, True, True, True]))
    data["done"][:] = torch.tensor([True, True, True, False, True, False, True])
    success = torch.tensor([True, False, True, False, False, False, True])
    terminated = torch.tensor([False, True, False, False, False, False, False])
    timeouts = torch.tensor([False, False, False, False, True, False, False])
    # Exercise the existing producer's real once-per-episode outcome methods.
    producer = RecoveryPrefixCurriculum.__new__(RecoveryPrefixCurriculum)
    producer._sampler = NS(enabled=data["recovery_enabled"],
        level_ids=data["recovery_prefix_lesson_id"])
    producer.outcome_recorded = torch.zeros(7, dtype=torch.bool)
    producer.outcome_counts = torch.zeros(3, 4, dtype=torch.long)
    producer.record_outcomes(data["done"], success, terminated, timeouts)
    producer.record_outcomes(data["done"], success, terminated, timeouts)
    audit.observe(data, torch.tensor([True, True, True, True, True, True, True]))
    outcomes, funnel = producer.rollout_outcome_metrics(), audit.metrics()
    for lesson, ended, stable in zip(LESSONS, (2, 1, 1), (1, 1, 0)):
        op, fp = "RecoveryOutcomes/" + lesson + "/", "RecoveryPhaseAudit/" + lesson + "/"
        assert outcomes[op + "ended"] == ended
        assert funnel[fp + "ended_with_pure_rl_started"] == ended
        assert funnel[fp + "ended_with_stable_hold"] == stable
        assert funnel[fp + "first_pure_rl_started"] == 2
        assert funnel[fp + "observed_transitions"] == 4
    assert outcomes["RecoveryOutcomes/original_chain/collection_timeouts"] == 1
    assert outcomes["RecoveryOutcomes/original_chain/cumulative_success_fraction"] == 0
    assert len(funnel) == 33  # Independent of behavior-group/profile cardinality.
    assert data["recovery_handover_count"].eq(1).all()


def test_learning_audit_is_opt_in_and_preserves_inputs_and_rng(monkeypatch):
    monkeypatch.setattr(learning_audit, "PolicyUpdateProbe", lambda *args: NS(metrics=lambda: {}))
    env = NS(num_envs=1, device="cpu", policy_dt=.02, commands=torch.tensor([[0., 0., .305]]),
        cfg={}, skills=NS(batches=[(torch.tensor([0]), dict(group_name="ordinary", kind="constant"))]))
    native = snapshot([1])
    native.update(commands=env.commands.clone(), done=torch.zeros(1, dtype=torch.bool),
        reward=torch.zeros(1), height=torch.tensor([.305]), planar_speed_world=torch.zeros(1),
        reasons={"fall": torch.zeros(1)})
    audit = learning_audit.LearningAudit(env, NS(actor=None, storage=None))
    assert audit.recovery_phases is None
    audit.observe(native, {})
    assert not any(key.startswith("RecoveryPhaseAudit/") for key in audit.drain())
    env.cfg["recovery_phase_audit"] = True
    env.recovery = NS(pose_curriculum=NS(rollout_outcome_metrics=lambda: {}),
        fsm=NS(config={"stable_seconds": 1.}))
    audit = learning_audit.LearningAudit(env, NS(actor=None, storage=None))
    rng = torch.get_rng_state().clone()
    original = {key: value.clone() for key, value in native.items() if isinstance(value, torch.Tensor)}
    with torch.inference_mode():
        audit.observe(native, {}, {"policy_learning_mask": torch.tensor([True])})
    result = audit.drain()
    assert len([key for key in result if key.startswith("RecoveryPhaseAudit/")]) == 33
    assert result["RecoveryPhaseAudit/prefix_release/policy_owned_transitions"] == 1
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(native[key], value) for key, value in original.items())
    assert audit.recovery_phases.counts.eq(0).all()
    env.cfg["auto_reset"] = False
    with pytest.raises(ValueError, match="auto-reset"):
        learning_audit.LearningAudit(env, NS(actor=None, storage=None))


@pytest.mark.parametrize("mask", [None, torch.tensor([1.]), torch.tensor([True, False])])
def test_missing_or_substituted_interval_mask_is_rejected(mask):
    audit = learning_audit.RecoveryPhaseAudit(1, "cpu", 1.)
    with pytest.raises(ValueError, match="final native"):
        audit.observe(snapshot([2]), mask)
