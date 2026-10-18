"""CPU-only candidate measurements; no simulator or optimizer is executed."""
from copy import deepcopy

import pytest
import torch

from wheeled_tasks.chassis.flat_command_sampling import ScutFlatCommands
from wheeled_tasks.chassis.source_birth_exposure import FIELDS, GROUPS, SourceBirthExposure


def observer(*, count=2, steps=2, source=None, stream="native-worker-a", device="cpu", defer=False):
    if source is None:
        source = torch.ones(count, dtype=torch.bool)
    return SourceBirthExposure(num_envs=count, steps_per_rollout=steps,
                               source_dynamic_mask=source, stream_id=stream, device=device,
                               defer_tensor_validation=defer)


def info(*, ticks, ground, ids=None, done=None, learning=None, terminated=None, timeout=None):
    ticks = torch.tensor(ticks, dtype=torch.long)
    ground = torch.tensor(ground, dtype=torch.bool)
    n = ticks.numel()
    ids = torch.zeros(n, dtype=torch.long) if ids is None else torch.tensor(ids, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool) if done is None else torch.tensor(done, dtype=torch.bool)
    owner_learning = ~(ground & (ticks <= 40))
    learning = owner_learning.clone() if learning is None else torch.tensor(learning, dtype=torch.bool)
    owner = {"episode_id": ids.clone(), "episode_physics_ticks": ticks.clone(),
             "ground": ground.clone(), "holding": ground & (ticks < 40),
             "learning_mask": owner_learning.clone()}
    returned_owner = {"episode_id": ids + done.long(),
                      "episode_physics_ticks": torch.where(done, 0, ticks)}
    return {"policy_learning_mask": learning.clone(),
            "diagnostics": {"source_flat_episode": owner, "episode_id": ids.clone(),
                            "policy_learning_mask": learning.clone(), "done": done.clone(),
                            "terminated": (torch.zeros(n, dtype=torch.bool)
                                           if terminated is None else torch.tensor(terminated)),
                            "timeout": (torch.zeros(n, dtype=torch.bool)
                                        if timeout is None else torch.tensor(timeout))},
            "returned_episode_state": {"episode_id": returned_owner["episode_id"].clone(),
                                       "reset_mask": done.clone(), "source_flat_episode": returned_owner}}


def observe(book, payload, *, step=0, interval=None):
    if interval is None:
        interval = (book.last_policy_interval_id or 0) + 1
    book.observe_transition(step_index=step, policy_interval_id=interval,
                            physical_tick=interval * 4, info=payload)


def bind(book, population=None):
    learning = torch.stack([row["learning"] for row in book.rows])
    book.bind_loss_population(final_learning_mask=learning,
                             optimized_population_mask=learning if population is None else population)
    return learning


def count(book, field, group=None):
    column = FIELDS.index(field)
    return int(book.counts[:, column].sum() if group is None else book.counts[GROUPS.index(group), column])


def test_actual_source_owner_tick40_stays_masked_then44_post_motor_off_and_fixed_lanes_excluded():
    source = torch.tensor([True, True, False])
    book = observer(count=3, steps=12, source=source)
    runtime = ScutFlatCommands(3, "cpu")
    generator = torch.Generator().manual_seed(91)
    runtime.reset(torch.arange(3), torch.zeros(3), torch.tensor([True, False, False]), generator)
    source.zero_()  # Ownership is copied from the actual binding, not aliased.
    book.begin_rollout(actor_enabled=True)
    for step in range(12):
        runtime.begin_policy()
        for _ in range(4):
            runtime.advance_physics(.005)
        owned = runtime.diagnostics()
        payload = info(ticks=owned["episode_physics_ticks"].tolist(), ground=owned["ground"].tolist(),
                       ids=[1, 1, 1], done=[step == 11, step == 11, False])
        payload["diagnostics"]["source_flat_episode"].update(owned)
        payload["diagnostics"]["source_flat_episode"]["fixed_training_rehearsal"] = torch.tensor(
            [False, False, True])
        if step == 9:
            assert not bool(owned["holding"][0]) and not bool(owned["learning_mask"][0])
        if step == 10:
            assert bool(owned["learning_mask"][0])
        observe(book, payload, step=step)
    bind(book)
    assert count(book, "actor_direct_loss_intervals") == 0
    book.finish_update(actor_optimized=True)
    assert count(book, "policy_intervals") == 24
    assert count(book, "physics_ticks") == 96
    assert count(book, "motor_off_owner_intervals") == 10
    assert count(book, "actor_direct_loss_intervals", "ground_request_true") == 2
    assert count(book, "actor_direct_loss_intervals", "ground_request_false") == 12
    assert count(book, "post_motor_off_actor_direct_loss_intervals") == 2
    assert count(book, "birth_observed_episodes") == count(book, "complete_observed_episodes") == 2
    assert count(book, "complete_motor_off_window_episodes") == 1
    assert count(book, "post_motor_off_learning_eligible_episodes") == 1
    assert count(book, "complete_motor_off_episodes_with_post_motor_off_actor_loss") == 1


def test_unique_native_loss_population_tail_is_omitted_and_epochs_cannot_multiply_credit():
    book = observer(count=3)
    book.begin_rollout(actor_enabled=True)
    for step in range(2):
        observe(book, info(ticks=[44 + step * 4] * 3, ground=[True, False, True]), step=step)
    # A 6-entry population with M=4 uses the first floor(6/4)*4 entries.
    population = torch.tensor([[True, True, True], [True, False, False]])
    bind(book, population)
    population.fill_(True)
    book.finish_update(actor_optimized=True)
    assert count(book, "learning_eligible_intervals") == 6
    assert count(book, "actor_direct_loss_intervals") == 4
    assert count(book, "post_motor_off_actor_direct_loss_intervals") == 3
    assert count(book, "birth_observed_episodes") == 0
    with pytest.raises(ValueError):
        book.finish_update(actor_optimized=True)
    assert count(book, "actor_direct_loss_intervals") == 4


@pytest.mark.parametrize("phase", ["critic", "failed_actor", "failed_critic"])
def test_critic_only_and_failed_updates_preserve_physical_counts_but_no_actor_credit(phase):
    book = observer()
    book.begin_rollout(actor_enabled=phase != "critic" and phase != "failed_critic")
    for step in range(2):
        observe(book, info(ticks=[4 + step * 4] * 2, ground=[False, False], done=[step == 1] * 2), step=step)
    bind(book)
    if phase == "critic":
        book.finish_update(actor_optimized=False)
        assert book.critic_only_updates == 1
    else:
        book.fail_update()
        book.fail_update()
        assert book.failed_updates == 1
    assert count(book, "policy_intervals") == 4
    assert count(book, "complete_observed_episodes") == 2
    assert count(book, "actor_direct_loss_intervals") == 0
    assert count(book, "complete_episodes_with_actor_loss") == 0


def test_partial_lane_reset_keeps_terminal_birth_label_and_changes_only_next_paid_interval():
    book = observer()
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[4, 4], ground=[True, False], ids=[10, 20], done=[True, False]))
    observe(book, info(ticks=[4, 8], ground=[False, False], ids=[11, 20]), step=1)
    bind(book)
    book.finish_update(actor_optimized=True)
    assert count(book, "observed_episodes", "ground_request_true") == 1
    assert count(book, "observed_episodes", "ground_request_false") == 2
    assert count(book, "closed_episodes", "ground_request_true") == 1
    assert count(book, "actor_direct_loss_intervals", "ground_request_true") == 0
    assert count(book, "actor_direct_loss_intervals", "ground_request_false") == 3
    assert count(book, "complete_motor_off_window_episodes") == 0


def test_ground_false_is_not_relabelled_as_nominal_or_as_post_motor_off():
    book = observer(count=1, steps=1)
    payload = info(ticks=[4], ground=[False], done=[True])
    payload["diagnostics"]["source_flat_episode"].update(
        birth_source_angle_rad=torch.tensor([[1.1, -.9]]),
        birth_own_angle_rad=torch.tensor([[1.2, -.8]]))
    book.begin_rollout(actor_enabled=True)
    observe(book, payload)
    bind(book)
    book.finish_update(actor_optimized=True)
    assert GROUPS == ("ground_request_false", "ground_request_true")
    assert count(book, "actor_direct_loss_intervals", "ground_request_false") == 1
    assert count(book, "post_motor_off_actor_direct_loss_intervals") == 0
    assert book.checkpoint_state()["skill_qualification"] == "not_assessed"


def test_complete_motor_off_funnel_spans_rollouts_and_partial_attach_cannot_claim_it():
    book = observer(count=1, steps=4)
    for block in range(3):
        book.begin_rollout(actor_enabled=block != 0)
        for step in range(4):
            endpoint = (block * 4 + step + 1) * 4
            observe(book, info(ticks=[endpoint], ground=[True], done=[endpoint == 48]), step=step)
        bind(book)
        book.finish_update(actor_optimized=block != 0)
    assert count(book, "complete_motor_off_window_episodes") == 1
    assert count(book, "complete_episodes_with_post_motor_off_learning_eligible") == 1
    assert count(book, "complete_motor_off_episodes_with_post_motor_off_actor_loss") == 1
    partial = observer(count=1, steps=1)
    partial.begin_rollout(actor_enabled=True)
    observe(partial, info(ticks=[44], ground=[True], done=[True]))
    bind(partial)
    partial.finish_update(actor_optimized=True)
    assert count(partial, "actor_direct_loss_intervals") == 1
    assert count(partial, "post_motor_off_actor_direct_loss_intervals") == 1
    assert count(partial, "complete_observed_episodes") == 0
    assert count(partial, "complete_motor_off_episodes_with_post_motor_off_actor_loss") == 0


def test_past_successful_actor_exposure_survives_later_critic_or_failed_close_without_new_credit():
    book = observer(count=1, steps=1)
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[4], ground=[False]))
    bind(book)
    book.finish_update(actor_optimized=True)
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[8], ground=[False], done=[True], terminated=[True]))
    bind(book)
    book.fail_update()
    assert count(book, "actor_direct_loss_intervals") == 1
    assert count(book, "complete_episodes_with_actor_loss") == 1
    assert count(book, "terminated_episodes") == 1


def test_partial_unpaid_discard_advances_only_actual_physical_clock_and_deduplicates_cleanup():
    book = observer(steps=4)
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[4, 4], ground=[True, False]))
    with pytest.raises(ValueError):
        bind(book)
    book.discard_rollout()
    book.discard_rollout()
    assert count(book, "physics_ticks") == 8
    assert count(book, "actor_direct_loss_intervals") == 0
    assert book.discarded_rollouts == 1 and book.last_physical_tick == 4
    before = book.checkpoint_state()
    assert book.metrics() == book.metrics()
    assert before == book.checkpoint_state()


def test_masks_are_actual_storage_masks_and_snapshot_cannot_seal_pending_actor_attribution():
    book = observer(count=1, steps=1)
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[4], ground=[True]))
    with pytest.raises(ValueError):
        book.checkpoint_state()
    with pytest.raises(ValueError):
        book.finish_update(actor_optimized=True)
    for final, population in ((torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool)),
                              (torch.zeros(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool))):
        with pytest.raises(ValueError):
            book.bind_loss_population(final_learning_mask=final, optimized_population_mask=population)
    assert count(book, "actor_direct_loss_intervals") == 0
    bind(book)
    with pytest.raises(ValueError):
        book.finish_update(actor_optimized=False)
    book.finish_update(actor_optimized=True)


@pytest.mark.parametrize("bad", ["owner_gate", "holding", "final_mask", "reset", "returned_id",
                                 "returned_tick", "owner_id", "ticks", "timeout", "dtype"])
def test_bad_paid_or_returned_payload_is_rejected_before_counts_or_cursors_change(bad):
    book = observer(count=1, steps=1)
    payload = info(ticks=[4], ground=[True])
    owner = payload["diagnostics"]["source_flat_episode"]
    returned = payload["returned_episode_state"]
    if bad == "owner_gate": owner["learning_mask"].fill_(True)
    if bad == "holding": owner["holding"].fill_(False)
    if bad == "final_mask": payload["policy_learning_mask"].fill_(True)
    if bad == "reset": returned["reset_mask"].fill_(True)
    if bad == "returned_id": returned["episode_id"] += 1
    if bad == "returned_tick": returned["source_flat_episode"]["episode_physics_ticks"] += 4
    if bad == "owner_id": owner["episode_id"] += 1
    if bad == "ticks": owner["episode_physics_ticks"] += 1
    if bad == "timeout": payload["diagnostics"]["timeout"].fill_(True)
    if bad == "dtype": owner["ground"] = owner["ground"].long()
    before = book.checkpoint_state()
    book.begin_rollout(actor_enabled=True)
    with pytest.raises(ValueError):
        observe(book, payload)
    assert count(book, "policy_intervals") == 0 and book.last_policy_interval_id is None
    assert torch.equal(book.live["expected_ids"], torch.tensor([-1]))
    book.discard_rollout()
    assert before["counts"] == book.checkpoint_state()["counts"]


@pytest.mark.parametrize("field", ["fixed_training_rehearsal", "fixed_evaluation_request",
                                  "scripted_recovery_birth"])
def test_source_binding_cannot_silently_credit_fixed_nominal_or_scripted_recovery(field):
    book = observer(count=1, steps=1)
    payload = info(ticks=[4], ground=[False])
    payload["diagnostics"]["source_flat_episode"][field] = torch.ones(1, dtype=torch.bool)
    book.begin_rollout(actor_enabled=True)
    with pytest.raises(ValueError):
        observe(book, payload)
    assert count(book, "policy_intervals") == 0


@pytest.mark.parametrize("clock", [(1, 8), (5000, 20000), (True, 4), (2, 8)])
def test_duplicate_skipped_or_learning_global_clock_cannot_create_physical_exposure(clock):
    book = observer(count=1, steps=2)
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[4], ground=[False]))
    before = book.counts.clone()
    with pytest.raises(ValueError):
        book.observe_transition(step_index=1, policy_interval_id=clock[0], physical_tick=clock[1],
                                info=info(ticks=[4], ground=[False]))
    assert torch.equal(before, book.counts) and book.last_physical_tick == 4


@pytest.mark.parametrize("bad", ["new_id_without_reset", "same_id_after_reset", "changed_ground",
                                "skipped_episode_tick"])
def test_partial_reset_and_episode_clock_cannot_replay_or_relabel_samples(bad):
    book = observer(count=1, steps=2)
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[4], ground=[False], done=[bad == "same_id_after_reset"]))
    payload = info(ticks=[4 if bad == "same_id_after_reset" else 8], ground=[bad == "changed_ground"],
                   ids=[1 if bad == "new_id_without_reset" else 0])
    if bad == "skipped_episode_tick":
        payload = info(ticks=[12], ground=[False])
    before = book.counts.clone()
    with pytest.raises(ValueError):
        observe(book, payload, step=1)
    assert torch.equal(before, book.counts)


def finished_state():
    book = observer(count=1, steps=1)
    book.begin_rollout(actor_enabled=True)
    payload = info(ticks=[4], ground=[False])
    observe(book, payload)
    payload["policy_learning_mask"].fill_(False)
    payload["diagnostics"]["source_flat_episode"]["ground"].fill_(True)
    bind(book)
    book.finish_update(actor_optimized=True)
    return book.checkpoint_state()


def test_same_stream_restart_restores_cursors_and_counters_then_rejects_duplicate_tick():
    state = finished_state()
    book = observer(count=1, steps=1, device="cpu:0")
    book.restore_state(state)
    state["counts"][0][0] = 999  # Restore does not retain external references.
    book.begin_rollout(actor_enabled=True)
    with pytest.raises(ValueError):
        observe(book, info(ticks=[4], ground=[False]), interval=1)
    assert count(book, "policy_intervals") == 1
    observe(book, info(ticks=[8], ground=[False], done=[True]), interval=2)
    bind(book)
    book.finish_update(actor_optimized=True)
    assert count(book, "observed_episodes") == 1
    assert count(book, "actor_direct_loss_intervals") == 2
    assert count(book, "complete_episodes_with_actor_loss") == 1
    assert not book.checkpoint_state()["native_exact_resume_qualified"]


def test_new_native_stream_keeps_counts_but_cannot_inherit_complete_episode_or_reuse_old_stream():
    state = finished_state()
    book = observer(count=1, steps=1, stream="native-worker-b")
    book.restore_state(state)
    assert book.last_physical_tick is None and not bool(book.live["seen"].any())
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[44], ground=[True], done=[True]), interval=1)
    bind(book)
    book.finish_update(actor_optimized=True)
    assert count(book, "actor_direct_loss_intervals") == 2
    assert count(book, "observed_episodes") == 2
    assert count(book, "complete_observed_episodes") == 0
    assert count(book, "complete_episodes_with_actor_loss") == 0
    with pytest.raises(ValueError):
        observer(count=1, steps=1).restore_state(book.checkpoint_state())


@pytest.mark.parametrize("bad", ["counts", "counter", "clock", "live_ticks", "dimensions", "streams"])
def test_invalid_restore_is_atomic(bad):
    state = finished_state()
    book = observer(count=1, steps=1)
    before = book.checkpoint_state()
    if bad == "counts": state["counts"][0][0] = -1
    if bad == "counter": state["completed_actor_updates"] = True
    if bad == "clock": state["last_physical_tick"] = 8
    if bad == "live_ticks": state["live"]["ticks"] = [0]
    if bad == "dimensions": state["steps_per_rollout"] = 2
    if bad == "streams": state["seen_stream_ids"] *= 2
    with pytest.raises(ValueError):
        book.restore_state(state)
    assert before == book.checkpoint_state()


def test_missing_pre_reset_snapshot_does_not_fall_back_to_newborn_or_claim_measurement():
    book = observer(count=1, steps=1)
    payload = info(ticks=[4], ground=[True], done=[True])
    del payload["diagnostics"]["source_flat_episode"]
    book.begin_rollout(actor_enabled=False)
    with pytest.raises(ValueError):
        observe(book, payload)
    assert count(book, "physics_ticks") == 0


def test_snapshot_and_metric_reads_are_immutable():
    book = observer(count=1, steps=1)
    book.restore_state(finished_state())
    original = deepcopy(book.checkpoint_state())
    result = book.metrics()
    result["/SourceBirthExposure/actor_direct_loss_intervals"] = -1
    snapshot = book.checkpoint_state()
    snapshot["live"]["ticks"][0] = 999
    assert original == book.checkpoint_state()
    assert book.metrics()["/SourceBirthExposure/actor_direct_loss_intervals"] == 1


def test_deferred_sampling_uses_no_tensor_host_boolean_and_bind_checks_one_device_latch(monkeypatch):
    book = observer(count=1, steps=2, defer=True)
    host_reads = []
    original = torch.Tensor.__bool__

    def tensor_boolean(value):
        host_reads.append(value)
        return original(value)

    monkeypatch.setattr(torch.Tensor, "__bool__", tensor_boolean)
    book.begin_rollout(actor_enabled=True)
    for step in range(2):
        observe(book, info(ticks=[4 + 4 * step], ground=[False]), step=step)
    assert len(host_reads) == 0
    assert book.metrics()["/SourceBirthExposure/policy_intervals"] == 0
    bind(book)
    assert len(host_reads) == 1
    assert book.metrics()["/SourceBirthExposure/policy_intervals"] == 2
    book.finish_update(actor_optimized=True)
    assert len(host_reads) == 1


@pytest.mark.parametrize("bad", ["ground_relabel", "owner_gate", "final_mask", "source_fixed"])
def test_deferred_bad_payload_blocks_bind_and_restores_previously_valid_credit_before_optimizer(bad):
    book = observer(count=1, steps=1, defer=True)
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[4], ground=[False]))
    bind(book)
    book.finish_update(actor_optimized=True)
    previous = book.checkpoint_state()
    book.begin_rollout(actor_enabled=True)
    payload = info(ticks=[8], ground=[bad == "ground_relabel"])
    owner = payload["diagnostics"]["source_flat_episode"]
    if bad == "owner_gate": owner["learning_mask"].fill_(False)
    if bad == "final_mask": payload["diagnostics"]["policy_learning_mask"].fill_(False)
    if bad == "source_fixed": owner["fixed_training_rehearsal"] = torch.ones(1, dtype=torch.bool)
    observe(book, payload)
    assert book.metrics()["/SourceBirthExposure/policy_intervals"] == 1
    with pytest.raises(ValueError):
        bind(book)
    book.fail_update()
    book.discard_rollout()
    assert book.counts.cpu().tolist() == previous["counts"]
    assert book.last_physical_tick == previous["last_physical_tick"]
    assert count(book, "actor_direct_loss_intervals") == 1
    assert book.completed_actor_updates == 1
    with pytest.raises(ValueError):
        book.checkpoint_state()


@pytest.mark.parametrize("cleanup", ["fail_update", "discard_rollout"])
def test_deferred_partial_cleanup_validates_physical_measurements_once_without_actor_credit(cleanup):
    book = observer(count=1, steps=2, defer=True)
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[4], ground=[True]))
    assert book.metrics()["/SourceBirthExposure/physics_ticks"] == 0
    getattr(book, cleanup)()
    first = book.checkpoint_state()
    getattr(book, cleanup)()
    assert first == book.checkpoint_state()
    assert count(book, "physics_ticks") == 4
    assert count(book, "actor_direct_loss_intervals") == 0


def test_deferred_bad_loss_mask_cannot_publish_batch_or_snapshot():
    book = observer(count=1, steps=1, defer=True)
    book.begin_rollout(actor_enabled=True)
    observe(book, info(ticks=[4], ground=[True]))
    with pytest.raises(ValueError):
        book.bind_loss_population(final_learning_mask=torch.zeros(1, 1, dtype=torch.bool),
                                 optimized_population_mask=torch.ones(1, 1, dtype=torch.bool))
    assert book.metrics()["/SourceBirthExposure/policy_intervals"] == 0
    with pytest.raises(ValueError):
        book.checkpoint_state()
