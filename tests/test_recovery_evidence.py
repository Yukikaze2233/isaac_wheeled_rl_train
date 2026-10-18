"""Actual episode identity, physical funnel and successful direct-loss exposure."""
from copy import deepcopy

import pytest
import torch

from wheeled_tasks.chassis.recovery_evidence import (
    EVENTS, FAILURES, FIELDS, RecoveryEvidenceJournal, TRANSITION_SCHEMA)


def journal(*, steps=4, count=3):
    return RecoveryEvidenceJournal(num_envs=count, steps_per_rollout=steps,
        source_identity={"schema": "v6-recovery-physical-event-source-v1",
                         "profile_names": ["upright", "fallen"]})


def physical_row(book, step, *, episode=0, interval=None, learning=None):
    n = book.num_envs
    enabled = torch.arange(n) < min(n, 2)
    lessons = torch.tensor([0, 2] + [-1] * (n - 2)) if n >= 2 else torch.tensor([0])
    profiles = torch.tensor([0, 1] + [-1] * (n - 2)) if n >= 2 else torch.tensor([0])
    mask = (~enabled if step < 2 else torch.ones(n, dtype=torch.bool)) if learning is None else learning
    done = enabled.clone() if step == book.steps_per_rollout - 1 else torch.zeros(n, dtype=torch.bool)
    event = {name: enabled.clone() if i == step else torch.zeros(n, dtype=torch.bool)
             for i, name in enumerate(EVENTS)}
    ids = torch.where(enabled, episode, 0).long()
    return {"schema": TRANSITION_SCHEMA, "source_identity_sha256": book.source_identity_sha256,
        "enabled": enabled, "episode_ids": ids, "episode_lesson_ids": lessons,
        "episode_profile_ids": profiles, "policy_interval_id": step + 1 if interval is None else interval,
        "events": event, "occurrence_ids": {name: torch.where(value, step + 1, -1).long()
                                             for name, value in event.items()},
        "policy_learning_mask": mask, "phase": torch.full((n,), 9, dtype=torch.long),
        "done": done, "terminated": torch.zeros(n, dtype=torch.bool),
        "timeouts": torch.zeros(n, dtype=torch.bool),
        "failure_reasons": {name: torch.zeros(n, dtype=torch.bool) for name in FAILURES},
        "returned_episode_ids": ids + done.long(), "reset_mask": done}


def collect(book, *, actor=True, modify=None, episode=0, start_interval=1):
    book.begin_rollout(actor_enabled=actor)
    rows = []
    for step in range(book.steps_per_rollout):
        row = physical_row(book, step, episode=episode, interval=start_interval + step)
        if modify is not None: modify(step, row)
        book.observe_transition(step_index=step, payload=row)
        rows.append(row)
    return torch.stack([row["policy_learning_mask"] for row in rows])


def total(book, name):
    return int(book.counts[..., FIELDS.index(name)].sum())


def test_script_blend_measurements_never_become_actor_samples_and_epochs_do_not_multiply():
    book = journal()
    learning = collect(book)
    book.bind_loss_population(final_learning_mask=learning, optimized_population_mask=learning)
    assert total(book, "policy_intervals") == 8
    assert total(book, "observed_episodes") == total(book, "closed_episodes") == 2
    assert total(book, "actor_direct_loss_intervals") == 0
    book.finish_update(actor_optimized=True)
    assert total(book, "actor_direct_loss_intervals") == 4
    assert total(book, "script_ready") == total(book, "rl_entered") == 2
    assert total(book, "actor_direct_loss_script_ready") == total(book, "actor_direct_loss_rl_entered") == 0
    assert total(book, "actor_direct_loss_legacy_followup_complete") == 2
    state = book.checkpoint_state()
    assert state["counts"][1][0][FIELDS.index("policy_intervals")] == 4
    assert state["counts"][3][1][FIELDS.index("policy_intervals")] == 4
    assert state["skill_qualification"] == "not_assessed"
    assert state["counting_scope"] == "current_worker_only_not_resumed"
    assert not state["native_exact_resume_qualified"] and not state["parent_evidence_restored"]
    before = book.counts.clone()
    metrics = book.metrics()
    assert metrics["/RecoveryEvidence/actor_direct_loss_intervals"] == 4
    for axis, names in (("lesson", ("prefix_stand", "original_chain")),
                        ("profile", ("upright", "fallen"))):
        for name in names:
            prefix = f"/RecoveryEvidence/{axis}/{name}/"
            assert metrics[prefix + "actor_direct_loss_intervals"] == 2
            assert metrics[prefix + "learning_eligible_intervals"] == 2
            assert metrics[prefix + "masked_intervals"] == 2
            assert metrics[prefix + "closed_episodes"] == 1
            assert metrics[prefix + "rl_entered"] == 1
    assert metrics["/RecoveryEvidence/lesson/prefix_release/actor_direct_loss_intervals"] == 0
    assert metrics == book.metrics() and torch.equal(book.counts, before)
    with pytest.raises(ValueError): book.finish_update(actor_optimized=True)


def test_real_postreset_ids_allow_new_episode_and_labels_on_next_physical_interval_only():
    book = journal()
    mask = collect(book)
    book.bind_loss_population(final_learning_mask=mask, optimized_population_mask=mask)
    book.finish_update(actor_optimized=True)
    mask = collect(book, episode=1, start_interval=5)
    book.bind_loss_population(final_learning_mask=mask, optimized_population_mask=mask)
    book.finish_update(actor_optimized=True)
    assert total(book, "observed_episodes") == total(book, "closed_episodes") == 4
    assert total(book, "legacy_followup_complete") == 4


@pytest.mark.parametrize("actor", [False, True])
def test_optimizer_failure_keeps_physical_events_and_zero_actor_credit(actor):
    book = journal()
    mask = collect(book, actor=actor)
    book.bind_loss_population(final_learning_mask=mask, optimized_population_mask=mask)
    book.fail_update()
    assert total(book, "legacy_followup_complete") == 2
    assert total(book, "actor_direct_loss_intervals") == 0
    assert book.metrics()["/RecoveryEvidence/profile/fallen/actor_direct_loss_intervals"] == 0
    assert book.metrics()["/RecoveryEvidence/profile/fallen/learning_eligible_intervals"] == 2
    assert book.failed_updates == 1
    book.fail_update()
    assert book.failed_updates == 1


def test_critic_only_update_does_not_credit_recovery_actor_exposure():
    book = journal()
    mask = collect(book, actor=False)
    book.bind_loss_population(final_learning_mask=mask, optimized_population_mask=mask)
    book.finish_update(actor_optimized=False)
    assert book.critic_only_updates == 1 and book.completed_actor_updates == 0
    assert total(book, "actor_direct_loss_intervals") == 0


def test_partial_unpaid_rollout_preserves_measurements_and_cannot_count_actor_loss():
    book = journal()
    book.begin_rollout(actor_enabled=True)
    book.observe_transition(step_index=0, payload=physical_row(book, 0))
    with pytest.raises(ValueError, match="pending"): book.checkpoint_state()
    book.discard_rollout()
    assert total(book, "policy_intervals") == total(book, "script_ready") == 2
    assert total(book, "actor_direct_loss_intervals") == 0 and book.discarded_rollouts == 1
    book.checkpoint_state()


@pytest.mark.parametrize("bad", ["duplicate_event", "relabel", "reset_without_done", "jump_id",
                                  "changed_cohort", "negative_id", "bad_occurrence", "failure_overlap"])
def test_bad_physical_identity_is_rejected_before_any_visible_counts(bad):
    book = journal()
    def modify(step, row):
        if step != 1: return
        if bad == "duplicate_event":
            row["events"][EVENTS[0]][0] = True
            row["occurrence_ids"][EVENTS[0]][0] = 2
        if bad == "relabel": row["episode_lesson_ids"][0] = 2
        if bad == "reset_without_done":
            row["reset_mask"][0] = True
            row["returned_episode_ids"][0] += 1
        if bad == "jump_id": row["episode_ids"][0] = 2
        if bad == "changed_cohort": row["enabled"][0] = False
        if bad == "negative_id": row["episode_ids"][0] = -1
        if bad == "bad_occurrence": row["occurrence_ids"][EVENTS[0]][0] = 1
        if bad == "failure_overlap":
            row["failure_reasons"][FAILURES[0]][0] = True
            row["failure_reasons"][FAILURES[1]][0] = True
    mask = collect(book, modify=modify)
    with pytest.raises(ValueError, match="identity"): book.bind_loss_population(
        final_learning_mask=mask, optimized_population_mask=mask)
    assert not book.counts.any()
    with pytest.raises(ValueError, match="faulted"): book.checkpoint_state()


def test_duplicate_physical_row_with_no_events_is_rejected():
    book = journal()
    book.begin_rollout(actor_enabled=True)
    row = physical_row(book, 0)
    book.observe_transition(step_index=0, payload=row)
    row["events"] = {name: torch.zeros(3, dtype=torch.bool) for name in EVENTS}
    row["occurrence_ids"] = {name: torch.full((3,), -1, dtype=torch.long) for name in EVENTS}
    with pytest.raises(ValueError, match="physical policy interval"):
        book.observe_transition(step_index=1, payload=row)
    assert not book.counts.any()


@pytest.mark.parametrize("bad", ["final_mask", "population_contains_script", "shape", "phase"])
def test_masks_and_phase_must_match_actual_optimizer_storage(bad):
    book = journal()
    mask = collect(book)
    final, population = mask.clone(), mask.clone()
    if bad == "final_mask": final[0, 0] = True
    if bad == "population_contains_script": population[0, 0] = True
    if bad == "shape": final = final[:, :2]
    if bad == "phase":
        book.bind_loss_population(final_learning_mask=final, optimized_population_mask=population)
        with pytest.raises(ValueError, match="phase"): book.finish_update(actor_optimized=False)
        return
    with pytest.raises(ValueError): book.bind_loss_population(
        final_learning_mask=final, optimized_population_mask=population)
    assert total(book, "actor_direct_loss_intervals") == 0


def test_allocation_failure_before_physical_commit_is_atomic_and_cleanup_cannot_double_count(monkeypatch):
    book = journal()
    mask = collect(book)
    original_clone = torch.Tensor.clone
    def broken_clone(value, *args, **kwargs):
        if value is book.pending_ids: raise MemoryError("injected live-state allocation failure")
        return original_clone(value, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", broken_clone)
        with pytest.raises(MemoryError): book.bind_loss_population(
            final_learning_mask=mask, optimized_population_mask=mask)
    assert not book.counts.any() and bool((book.live_ids == -1).all())
    assert not book.physical_committed
    book.fail_update()
    assert total(book, "script_ready") == 2 and total(book, "policy_intervals") == 8
    assert total(book, "actor_direct_loss_intervals") == 0


def test_native_generator_tail_is_excluded_from_direct_loss_evidence():
    from test_recovery_rollout_mask import algorithm, collect as rollout, observation
    from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
    alg = algorithm(steps=4, count=3)
    native = attach_recovery_rollout_mask(alg, num_envs=3)
    book = journal()
    def modify(step, row):
        row["policy_learning_mask"][:] = True
        if step == 3: row["policy_learning_mask"][2] = False
    mask = collect(book, modify=modify)
    next_obs = rollout(alg, mask)
    alg.compute_returns(next_obs)
    population = native.loss_population_mask()
    assert mask.sum() == 11 and population.sum() == 10
    book.bind_loss_population(final_learning_mask=native.learning, optimized_population_mask=population)
    before = deepcopy(alg.actor.state_dict())
    alg.update()
    book.finish_update(actor_optimized=True)
    assert any(not torch.equal(value, before[key]) for key, value in alg.actor.state_dict().items())
    # The last time-major valid row is recovery lane1 at its followup event;
    # the native minibatch tail must not gain either transition or event credit.
    assert total(book, "actor_direct_loss_intervals") == 7
    assert total(book, "legacy_followup_complete") == 2
    assert total(book, "actor_direct_loss_legacy_followup_complete") == 1
    metrics = book.metrics()
    assert metrics["/RecoveryEvidence/profile/upright/actor_direct_loss_intervals"] == 4
    assert metrics["/RecoveryEvidence/profile/fallen/actor_direct_loss_intervals"] == 3
    assert metrics["/RecoveryEvidence/profile/fallen/learning_eligible_intervals"] == 4
    assert metrics["/RecoveryEvidence/lesson/original_chain/actor_direct_loss_intervals"] == 3
    assert book.completed_actor_updates == 1
