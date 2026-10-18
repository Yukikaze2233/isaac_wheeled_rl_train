"""CPU-only population migration; no native state or admission is created."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.source_birth_exposure import FIELDS, SourceBirthExposure


OLD_MASK = torch.tensor([False, True, True, True])
NEW_MASK = torch.tensor([False, True, False, True])
COUNTERS = ("completed_actor_updates", "critic_only_updates", "failed_updates", "discarded_rollouts")


def _observer(mask=OLD_MASK, stream="native-parent"):
    return SourceBirthExposure(num_envs=4, steps_per_rollout=2,
        source_dynamic_mask=mask, stream_id=stream, defer_tensor_validation=True)


def _row(tick, *, done=False, mixed=False):
    ground = torch.tensor([False, False, True, False]) if not mixed else torch.ones(4, dtype=torch.bool)
    ticks = torch.full((4,), tick, dtype=torch.long)
    ids = torch.zeros(4, dtype=torch.long)
    done_mask = torch.full((4,), done, dtype=torch.bool)
    owner_learning = ~(ground & (ticks <= 40))
    learning = owner_learning.clone()
    owner = {"episode_id": ids, "episode_physics_ticks": ticks, "ground": ground,
             "holding": ground & (ticks < 40), "learning_mask": owner_learning}
    if mixed:
        owner["fixed_training_rehearsal"] = torch.tensor([True, False, False, False])
        owner["scripted_recovery_birth"] = torch.tensor([False, False, True, False])
        learning[2] = False
    returned_owner = {"episode_id": ids + done_mask.long(),
        "episode_physics_ticks": torch.where(done_mask, 0, ticks)}
    return {"policy_learning_mask": learning,
        "diagnostics": {"source_flat_episode": owner, "done": done_mask,
            "terminated": done_mask, "timeout": torch.zeros(4, dtype=torch.bool)},
        "returned_episode_state": {"episode_id": returned_owner["episode_id"],
            "reset_mask": done_mask, "source_flat_episode": returned_owner}}


def _observe(book, tick, step, **kwargs):
    interval = (book.last_policy_interval_id or 0) + 1
    book.observe_transition(step_index=step, policy_interval_id=interval,
        physical_tick=interval * 4, info=_row(tick, **kwargs))


def _bind(book):
    learning = torch.stack([row["learning"] for row in book.rows])
    book.bind_loss_population(final_learning_mask=learning, optimized_population_mask=learning)


def _history():
    book = _observer()
    for block, phase in enumerate(("actor", "critic", "failed")):
        book.begin_rollout(actor_enabled=phase != "critic")
        for step in range(2):
            _observe(book, (block * 2 + step + 1) * 4, step)
        _bind(book)
        if phase == "failed":
            book.fail_update()
        else:
            book.finish_update(actor_optimized=phase == "actor")
    book.begin_rollout(actor_enabled=True)
    _observe(book, 28, 0)
    book.discard_rollout()
    return book.checkpoint_state()


def _assert_atomic_rejection(state, *, mask=NEW_MASK, stream="native-child", **kwargs):
    book = _observer(mask, stream)
    before = book.checkpoint_state()
    with pytest.raises(ValueError):
        book.restore_state(state, **kwargs)
    assert book.checkpoint_state() == before


@pytest.mark.parametrize("stream", ["native-parent", "native-child"])
@pytest.mark.parametrize("explicit", [False, True])
def test_unchanged_population_retains_original_same_and_new_stream_behavior(stream, explicit):
    state = _history()
    book = _observer(stream=stream)
    book.restore_state(state, **({"allow_population_change": False} if explicit else {}))
    restored = book.checkpoint_state()
    assert restored["counts"] == state["counts"]
    assert {k: restored[k] for k in COUNTERS} == {k: state[k] for k in COUNTERS}
    if stream == state["stream_id"]:
        assert restored == state
    else:
        assert restored["last_policy_interval_id"] is None
        assert restored["last_physical_tick"] is None
        assert not any(restored["live"]["seen"])


def test_explicit_new_population_preserves_history_and_counts_only_new_lanes_after_two_updates():
    state = _history()
    original = deepcopy(state)
    book = _observer(NEW_MASK, "native-child")
    book.restore_state(state, allow_population_change=True)
    migrated = book.checkpoint_state()
    assert migrated["counts"] == state["counts"]
    assert [migrated[k] for k in COUNTERS] == [1, 1, 1, 1]
    assert migrated["source_dynamic_mask"] == NEW_MASK.tolist()
    assert migrated["seen_stream_ids"] == ["native-parent", "native-child"]
    assert migrated["live"] == _observer(NEW_MASK, "unused").checkpoint_state()["live"]
    assert migrated["last_policy_interval_id"] is migrated["last_physical_tick"] is None
    assert state == original

    # A partial post-off episode on a new worker cannot inherit old completeness.
    before = torch.tensor(state["counts"])
    with torch.inference_mode():
        for block in range(2):
            book.begin_rollout(actor_enabled=True)
            for step in range(2):
                _observe(book, 44 + (block * 2 + step) * 4, step,
                    mixed=True, done=block == 1 and step == 1)
            # The actual callback boundary leaves inference mode before binding.
            with torch.inference_mode(False):
                _bind(book)
                book.finish_update(actor_optimized=True)
    delta = book.counts - before
    assert int(delta[:, FIELDS.index("policy_intervals")].sum()) == 8
    assert int(delta[:, FIELDS.index("actor_direct_loss_intervals")].sum()) == 8
    assert int(delta[:, FIELDS.index("physics_ticks")].sum()) == 32
    for field in ("birth_observed_episodes", "complete_observed_episodes",
                  "complete_episodes_with_actor_loss",
                  "complete_motor_off_episodes_with_post_motor_off_actor_loss"):
        assert int(delta[:, FIELDS.index(field)].sum()) == 0
    assert [getattr(book, k) for k in COUNTERS] == [3, 1, 1, 1]
    assert book.last_policy_interval_id == 4 and book.last_physical_tick == 16
    assert book.checkpoint_state()["native_exact_resume_qualified"] is False


@pytest.mark.parametrize("kwargs", [{}, {"allow_population_change": False}])
def test_population_change_remains_strict_by_default(kwargs):
    _assert_atomic_rejection(_history(), **kwargs)


def test_equal_size_changed_lane_identity_is_not_accepted_by_population_count_alone():
    same_size_different_lanes = torch.tensor([True, True, False, True])
    assert int(same_size_different_lanes.sum()) == int(OLD_MASK.sum())
    _assert_atomic_rejection(_history(), mask=same_size_different_lanes)
    book = _observer(same_size_different_lanes, "native-child")
    book.restore_state(_history(), allow_population_change=True)
    assert torch.equal(book.source_dynamic_mask, same_size_different_lanes)


def test_same_stream_cannot_change_population_even_with_explicit_flag():
    _assert_atomic_rejection(_history(), stream="native-parent", allow_population_change=True)


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], torch.tensor(True)])
def test_population_flag_requires_actual_boolean(flag):
    _assert_atomic_rejection(_history(), allow_population_change=flag)


@pytest.mark.parametrize("mask", [[True], [[False, True, True, True]],
    [0, 1, 1, 1], [0., 1., 1., 1.], [False, True, 1, True]])
def test_migration_validates_old_mask_shape_and_boolean_dtype(mask):
    state = _history()
    state["source_dynamic_mask"] = mask
    _assert_atomic_rejection(state, allow_population_change=True)


@pytest.mark.parametrize("bad", ["counts_negative", "counts_float", "counts_shape", "counter",
    "clock", "live_ticks", "live_dtype", "live_shape", "stream_reuse", "fields", "dimensions"])
def test_migration_does_not_skip_original_journal_validation(bad):
    state = _history()
    if bad == "counts_negative": state["counts"][0][0] = -1
    if bad == "counts_float": state["counts"][0][0] = 1.5
    if bad == "counts_shape": state["counts"] = state["counts"][:1]
    if bad == "counter": state["completed_actor_updates"] = True
    if bad == "clock": state["last_physical_tick"] += 4
    if bad == "live_ticks": state["live"]["ticks"][1] = 0
    if bad == "live_dtype": state["live"]["seen"] = [1] * 4
    if bad == "live_shape": state["live"]["ground"] = [True]
    if bad == "stream_reuse": state["seen_stream_ids"].insert(0, "native-child")
    if bad == "fields": state["fields"] = state["fields"][:-1]
    if bad == "dimensions": state["num_envs"] = 8
    _assert_atomic_rejection(state, allow_population_change=True)


def test_local_real_9147_journal_migration_when_receipt_is_available():
    path = Path(__file__).resolve().parents[1] / (
        "reports/v7_formal_training_20261012/branch107_entry_preparation/"
        "user_paused043/completion.json")
    if not path.is_file():
        pytest.skip("Local sealed 9147 receipt is not part of the portable CPU fixture")
    parent = json.loads(path.read_text())
    assert parent["global_actor_update"] == 9147
    state = parent["source_birth_exposure_state"]
    assert (state["num_envs"], state["steps_per_rollout"], state["completed_actor_updates"],
            state["last_policy_interval_id"], state["discarded_rollouts"]) == (4096, 24, 864, 20758, 1)
    old_mask = torch.tensor(state["source_dynamic_mask"], dtype=torch.bool)
    assert int(old_mask.sum()) == 3072
    # Remove one fixture lane, without choosing a future training allocation.
    new_mask = old_mask.clone()
    new_mask[old_mask.nonzero()[0, 0]] = False
    book = SourceBirthExposure(num_envs=4096, steps_per_rollout=24,
        source_dynamic_mask=new_mask, stream_id="cpu-9147-population-migration-fixture")
    book.restore_state(state, allow_population_change=True)
    restored = book.checkpoint_state()
    assert restored["counts"] == state["counts"]
    assert {k: restored[k] for k in COUNTERS} == {k: state[k] for k in COUNTERS}
    assert restored["source_dynamic_mask"] == new_mask.tolist()
    assert restored["last_policy_interval_id"] is restored["last_physical_tick"] is None
    assert restored["seen_stream_ids"] == state["seen_stream_ids"] + [book.stream_id]
    assert not any(restored["live"]["seen"])
    assert not any(restored["live"]["actor_seen"])
    assert set(restored["live"]["expected_ids"]) == {-1}
