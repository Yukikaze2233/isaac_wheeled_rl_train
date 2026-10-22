"""Terrain credit follows the actual optimized population and terminal owner."""
from copy import deepcopy

import pytest
import torch

from wheeled_tasks.chassis.tasks.curricula.rough_course_exposure import (
    CELL_FIELDS, MODES, TOTAL_FIELDS, RELEASE_FIELDS, RoughCourseExposure,
)
from wheeled_tasks.chassis.tasks.commands.flat_command_sampling import ScutFlatCommands
from wheeled_tasks.chassis.application.configuration.small_rough_course import MixedCourseExposure, VERSION


def fixture(steps=2):
    # Deliberately non-contiguous native ownership and repeated family columns.
    mask = torch.tensor([False, True, True, False, True])
    grid = dict(definition=dict(rows=2, columns=3, keys=("ramps", "bars"),
                               tile_size_m=(9., 9.)),
                column_key_ids=torch.tensor([0, 0, 1]), start_xy=torch.tensor([-9., -13.5]))
    return RoughCourseExposure(mask, grid, steps), grid


def sample(tick, *, episode_id=0, done=False):
    origins = torch.tensor([[0., 300., 0.], [-4.5, -9., 0.], [-4.5, 0., 0.],
                            [0., 300., 0.], [4.5, 9., 0.]])
    commands = torch.zeros(5, 3)
    commands[:, 2] = .305
    commands[2, 0] = .05
    commands[4, 1] = 1.
    return dict(root_link_position_world=origins.clone(), rough_birth_origins_world=origins,
        commands=commands, measurement_valid_mask=torch.ones(5, dtype=torch.bool),
        episode_id=torch.full((5,), episode_id, dtype=torch.long),
        episode_ticks=torch.full((5,), tick, dtype=torch.long),
        done=torch.full((5,), done), timeout=torch.full((5,), done),
        boundary=torch.zeros(5, dtype=torch.bool), terminated=torch.zeros(5, dtype=torch.bool))


def table(book, field):
    return book.counts[CELL_FIELDS.index(field)].view(2, 3)


def test_real_mask_pre_reset_cells_and_modes_commit_once_after_success():
    book, _ = fixture()
    first, last = sample(1), sample(2, done=True)
    last["root_link_position_world"][1, 0] = 4.5
    last["root_link_position_world"][2] = torch.tensor([4.5, 9., 0.])
    last["root_link_position_world"][4, 0] = 100.
    last["commands"][2, 1] = 1.
    book.observe(0, first)
    book.observe(1, last)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, 2] = False
    before, rng = deepcopy(book.state_dict()), torch.get_rng_state()
    staged = book.prepare(mask)
    book.prepare(torch.zeros_like(mask))
    assert book.state_dict() == before and torch.equal(torch.get_rng_state(), rng)
    metrics = book.commit(staged)
    assert book.totals.tolist() == [5, 0, 1, 0]
    assert table(book, "current_actor_transitions").tolist() == [[1, 0, 0], [1, 0, 2]]
    assert table(book, "birth_actor_transitions").tolist() == [[2, 1, 0], [0, 0, 2]]
    assert table(book, "visited_distinct_episodes").tolist() == [[1, 0, 0], [1, 0, 2]]
    assert table(book, "birth_distinct_episodes").sum() == 3
    assert table(book, "complete_actor_episodes").tolist() == [[1, 0, 0], [0, 0, 0]]
    assert dict(zip(MODES, book.mode_counts.sum(-1).tolist())) == dict(
        stand=2, pure_translate=0, pure_spin=1, combined=1)
    assert metrics["rough/current_actor_transitions/row1/ramps"] == 1
    assert metrics["rough/current_actor_transitions/row1/bars"] == 2
    with pytest.raises(ValueError, match="only once"):
        book.commit(staged)


def test_invalid_measurement_and_outside_grid_are_not_fabricated_as_row_zero():
    book, _ = fixture(steps=1)
    data = sample(1)
    data["root_link_position_world"][1, 0] = float("nan")
    data["root_link_position_world"][2, 1] = 1000.
    data["measurement_valid_mask"][4] = False
    book.observe(0, data)
    book.commit(book.prepare(torch.ones(1, 5, dtype=torch.bool)))
    assert book.totals.tolist() == [3, 2, 1, 0]
    assert not book.counts[0].any()
    # The outside-grid interval still carries its real birth-cell attribution.
    assert table(book, "birth_actor_transitions")[0, 1] == 1


def test_micro_speed_stays_translation_and_spin_does_not_become_standing():
    book, _ = fixture(steps=1)
    book.observe(0, sample(1))
    book.commit(book.prepare(torch.ones(1, 5, dtype=torch.bool)))
    assert book.mode_counts.sum(-1).tolist() == [1, 1, 1, 0]


def test_cross_rollout_episodes_are_unique_but_critic_only_gaps_are_not_complete():
    book, _ = fixture(steps=1)
    mask = torch.ones(1, 5, dtype=torch.bool)
    book.observe(0, sample(1))
    book.commit(book.prepare(mask))
    book.observe(0, sample(2))  # Critic-only, or an update which failed.
    stale = book.prepare(mask)
    book.observe(0, sample(3, done=True))
    with pytest.raises(ValueError, match="only once"):
        book.commit(stale)
    book.commit(book.prepare(mask))
    assert table(book, "birth_distinct_episodes").sum() == 3
    assert table(book, "complete_actor_episodes").sum() == 0
    assert book.totals[0] == 6


def test_full_episode_can_cross_update_boundary_without_double_episode_credit():
    book, _ = fixture(steps=1)
    mask = torch.ones(1, 5, dtype=torch.bool)
    for tick in range(1, 5):
        book.observe(0, sample(tick, done=tick == 4))
        book.commit(book.prepare(mask))
    assert table(book, "birth_distinct_episodes").sum() == 3
    assert table(book, "complete_actor_episodes").sum() == 3
    assert table(book, "current_actor_transitions").sum() == 12


def test_newborn_labels_cannot_relabel_the_paid_terminal_interval():
    book, _ = fixture()
    old = sample(1, done=True)
    book.observe(0, old)
    # Mutating the returned newborn must not mutate the stored pre-reset row.
    old["rough_birth_origins_world"][:, :2] = 900.
    old["episode_id"] += 1
    newborn = sample(1, episode_id=1)
    newborn["rough_birth_origins_world"][1, 0] = 4.5
    newborn["root_link_position_world"][1, 0] = 4.5
    book.observe(1, newborn)
    book.commit(book.prepare(torch.ones(2, 5, dtype=torch.bool)))
    assert table(book, "birth_distinct_episodes").sum() == 6
    assert table(book, "birth_actor_transitions")[0, 0] == 1
    assert table(book, "birth_actor_transitions")[1, 0] == 1


def test_same_episode_cannot_teleport_its_birth_origin_for_terrain_credit():
    book, _ = fixture()
    book.observe(0, sample(1))
    second = sample(2)
    second["rough_birth_origins_world"][1, 0] = 4.5
    book.observe(1, second)
    with pytest.raises(ValueError, match="changed inside"):
        book.prepare(torch.ones(2, 5, dtype=torch.bool))


def test_checkpoint_keeps_counters_but_cold_episode_cannot_inherit_completeness():
    book, _ = fixture(steps=1)
    book.observe(0, sample(1))
    book.commit(book.prepare(torch.ones(1, 5, dtype=torch.bool)))
    state = book.state_dict()
    restored, _ = fixture(steps=1)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    restored.observe(0, sample(2, done=True))
    restored.commit(restored.prepare(torch.ones(1, 5, dtype=torch.bool)))
    assert table(restored, "complete_actor_episodes").sum() == 0
    assert table(restored, "birth_distinct_episodes").sum() == 6
    bad = deepcopy(state)
    bad["totals"][TOTAL_FIELDS.index("actor_transitions")] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        fixture(steps=1)[0].load_state_dict(bad)
    bad = deepcopy(state)
    bad["cell_counts"][0][0][0] = 1.5
    with pytest.raises(ValueError, match="integer"):
        fixture(steps=1)[0].load_state_dict(bad)


def test_grid_identity_and_native_lane_mask_are_not_interchangeable():
    book, grid = fixture(steps=1)
    changed = deepcopy(grid)
    changed["column_key_ids"][1] = 1
    other = RoughCourseExposure(book.source_mask, changed, 1)
    with pytest.raises(ValueError, match="grid/population"):
        other.load_state_dict(book.state_dict())


def test_existing_role_history_stays_unclassified_and_new_credit_is_persisted():
    book, grid = fixture(steps=1)
    spec = dict(population_index=[0, 4, 4, 2, 4], small_rough_mix=dict(
        version=VERSION, temporal=dict(phase_policy_ticks=[1] * 4)))
    old = MixedCourseExposure(spec, 1, "cpu")
    old.counts["source_rough"] = 100
    mixed = MixedCourseExposure(spec, 1, "cpu", rough_grid=grid)
    mixed.load_state_dict(old.state_dict())
    assert mixed.rough.historical_unclassified_actor_transitions == 100
    data = sample(1)
    phase = torch.tensor([-1, -1, -1, 0, -1])
    data["source_flat_episode"] = dict(temporal_published_phase=phase,
                                      episode_id=data["episode_id"])
    mixed.observe(0, dict(diagnostics=data))
    mixed.commit(mixed.prepare(torch.ones(1, 5, dtype=torch.bool)))
    state = mixed.state_dict()
    assert state["direct_loss_transitions"]["source_rough"] == 103
    assert sum(state["rough_exposure"]["totals"][:1]) == 3
    restored = MixedCourseExposure(spec, 1, "cpu", rough_grid=grid)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    bad = deepcopy(state)
    bad["direct_loss_transitions"]["source_rough"] += 1
    with pytest.raises(ValueError, match="role loss totals"):
        MixedCourseExposure(spec, 1, "cpu", rough_grid=grid).load_state_dict(bad)


def ground_sample(tick, *, done=False):
    data = sample(tick, done=done)
    ground = torch.tensor([False, True, False, False, True])
    data['source_flat_episode'] = dict(ground=ground,
        episode_physics_ticks=torch.full((5,), tick * 4, dtype=torch.long))
    return data


def test_actual_initial_release_mask_credits_a_complete_tail_without_changing_whole_episode_counts():
    book, _ = fixture(steps=14)
    owner = ScutFlatCommands(5, 'cpu')
    rng = torch.Generator().manual_seed(23)
    ground = torch.tensor([False, True, False, False, True])
    owner.reset(torch.arange(5), torch.zeros(5), ground, rng, iteration=6558)
    masks = []
    for tick in range(1, 15):
        owner.begin_policy()
        for _ in range(4):
            owner.advance_physics()
        data = sample(tick, done=tick == 14)
        data['source_flat_episode'] = owner.diagnostics()
        book.observe(tick - 1, data)
        masks.append(owner.learning_mask.clone())
        owner.advance(.02, torch.zeros(5), 6558, rng)
    mask = torch.stack(masks)
    assert not mask[:10, ground].any() and mask[10:, ground].all()
    before = deepcopy(book.state_dict())
    prepared = book.prepare(mask)
    assert book.state_dict() == before
    metrics = book.commit(prepared)
    assert table(book, 'complete_actor_episodes').sum() == 1
    assert book.release_counts[0].sum() == 8
    assert book.release_counts[1].sum() == 2
    assert book.release_counts[2].sum() == 8
    assert metrics['rough/release/complete_actor_windows_after_release/row0/ramps'] == 1
    assert metrics['rough/release/complete_actor_windows_after_release/row1/bars'] == 1
    assert book.release_unclassified == 0


@pytest.mark.parametrize('drop_interval', [None, 11, 12])
def test_release_tail_survives_update_boundaries_but_not_missing_actor_intervals(drop_interval):
    book, _ = fixture(steps=1)
    for tick in range(1, 15):
        book.observe(0, ground_sample(tick, done=tick == 14))
        mask = torch.ones(1, 5, dtype=torch.bool)
        mask[:, [1, 4]] = tick > 10 and tick != drop_interval
        book.commit(book.prepare(mask))
    assert book.release_counts[1].sum() == (2 if drop_interval is None else 0)
    assert table(book, 'complete_actor_episodes').sum() == 1


def test_release_tail_does_not_bridge_uncommitted_windows_or_cold_worker_state():
    book, _ = fixture(steps=1)
    mask = torch.ones(1, 5, dtype=torch.bool)
    book.observe(0, ground_sample(11))
    book.commit(book.prepare(mask))
    saved = book.state_dict()
    book.observe(0, ground_sample(12))
    book.prepare(mask)  # Failed/critic-only update: no actor commit.
    book.observe(0, ground_sample(13, done=True))
    book.commit(book.prepare(mask))
    assert book.release_counts[1].sum() == 0
    restored, _ = fixture(steps=1)
    restored.load_state_dict(saved)
    restored.observe(0, ground_sample(12, done=True))
    restored.commit(restored.prepare(mask))
    assert restored.release_counts[1].sum() == 0
    assert restored.release_counts[0].sum() == 4


def test_release_window_requires_exact_start_clock_and_clean_horizon():
    for start, terminate, bad_clock in ((11, False, False), (12, False, False),
                                       (11, True, False), (11, False, True)):
        book, _ = fixture(steps=1)
        for tick in range(start, 15):
            data = ground_sample(tick, done=tick == 14)
            if terminate and tick == 14:
                data['terminated'][[1, 4]] = True
            if bad_clock:
                data['source_flat_episode']['episode_physics_ticks'] += 1
            book.observe(0, data)
            book.commit(book.prepare(torch.ones(1, 5, dtype=torch.bool)))
        assert book.release_counts[1].sum() == (2 if start == 11 and not terminate and not bad_clock else 0)
        if bad_clock:
            assert book.release_unclassified == 3 * (15 - start)


def test_release_checkpoint_extension_preserves_old_counts_without_fabricating_labels():
    book, _ = fixture(steps=1)
    book.observe(0, ground_sample(11, done=True))
    book.commit(book.prepare(torch.ones(1, 5, dtype=torch.bool)))
    saved = book.state_dict()
    restored, _ = fixture(steps=1)
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    legacy = deepcopy(saved)
    legacy.pop('initial_release_exposure')
    restored.load_state_dict(legacy)
    assert torch.equal(restored.counts, book.counts) and not restored.release_counts.any()
    assert restored.release_unclassified == int(book.totals[0])
    invalid = deepcopy(saved)
    invalid['initial_release_exposure']['cell_counts'][0][0][0] = .5
    before = deepcopy(restored.state_dict())
    with pytest.raises(ValueError, match='invalid identity or counters'):
        restored.load_state_dict(invalid)
    assert restored.state_dict() == before
    assert RELEASE_FIELDS == ('actor_transitions_after_release', 'complete_actor_windows_after_release',
                              'complete_actor_window_policy_intervals_after_release')
