"""Actual command ownership and lane order, with explicit test-only fractions."""
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import (
    VERSION, CASE_FILE, CASE_SHA, load_fixed_flat_rehearsal,
    reorder_fixed_flat_rehearsal, install_fixed_flat_rehearsal,
)
from wheeled_tasks.chassis.tasks.commands.rough_commands import ScutRoughCommands


ROOT = Path(__file__).resolve().parents[3]


def config(fraction):
    # Fixture, not a selected formal training allocation.
    return {"formal_flat_rehearsal": {"version": VERSION, "fraction": fraction,
        "cases_file": CASE_FILE, "cases_sha256": CASE_SHA, "allocation": "equal_case_lanes"}}


@pytest.mark.parametrize("fraction,expected", [(.25, 1024), (.5, 2048)])
def test_allocation_preserves_rng_spreads_lanes_and_survives_native_permutation(fraction, expected):
    rng = torch.get_rng_state().clone()
    spec = load_fixed_flat_rehearsal(config(fraction), ROOT, 4096)
    assert torch.equal(rng, torch.get_rng_state())
    assert spec["rehearsal_lanes"] == expected
    assert sum(spec["lane_counts"].values()) == expected
    assert max(spec["lane_counts"].values()) - min(spec["lane_counts"].values()) <= 1
    selected = list(spec["origins_by_env"])
    assert selected[0] < 4 and selected[-1] >= 4092
    order = sorted(range(4096), key=lambda i: f"env_{i}")
    reordered = reorder_fixed_flat_rehearsal(spec, order)
    for actual, original in enumerate(order):
        assert reordered["case_index"][actual] == spec["case_index"][original]
    origins = torch.tensor(list(spec["origins_by_env"].values()))
    assert len(set(spec["origins_by_env"].values())) == expected
    assert bool((origins[:, 0].abs() + 54. < 200.).all())
    assert bool(((origins[:, 1] - 300.).abs() + 54. < 200.).all())
    assert bool((origins[:, 2] == 0.).all())
    assert torch.cdist(origins[:2], origins[:2])[0, 1] == 4.


def test_fixed_training_commands_survive_resampling_and_rough_profile_changes():
    spec = load_fixed_flat_rehearsal(config(.5), ROOT, 60)
    grid = torch.zeros(10, 14, 3)
    grid[:, :, 0] = ((torch.arange(10) + .5) * 9. - 45.)[:, None]
    grid[:, :, 1] = ((torch.arange(14) + .5) * 9. - 63.)[None, :]
    origins = grid[4, torch.arange(60) % 14].clone()
    for lane, origin in spec["origins_by_env"].items():
        origins[lane] = torch.tensor(origin)
    owner = ScutRoughCommands(60, "cpu", terrain_origins=grid, env_origins=origins)
    install_fixed_flat_rehearsal(spec, owner)
    ids = torch.arange(60)
    rng = torch.Generator().manual_seed(721)
    owner.reset(ids, torch.zeros(60), torch.zeros(60, dtype=torch.bool), rng, iteration=5000)
    selected = owner.fixed_request_mask
    assert int(selected.sum()) == 30
    for _ in range(3):
        owner.episode_physics_ticks[:] = 2000
        owner.time_left.zero_()
        owner.begin_policy()
        for _ in range(4):
            owner.advance_physics()
        owner.advance(.02, torch.ones(60), 5000, rng, world_positions=origins)
        assert torch.equal(owner.command3[selected], owner.fixed_request_command3[selected])
        assert torch.equal(owner.height[selected], owner.fixed_request_height[selected])
        assert bool((owner.special_mode_id[selected] == -1).all())
        assert not bool(owner.is_heading[selected].any())
    # The original evaluation-only API still rejects unlabelled use.
    with pytest.raises(ValueError, match="evaluation-only"):
        owner.set_fixed_requests(ids, torch.zeros(60, 3), torch.full((60,), .305))


def test_no_implicit_fraction_or_silent_missing_working_points():
    assert load_fixed_flat_rehearsal({}, ROOT, 4096) is None
    with pytest.raises(ValueError):
        load_fixed_flat_rehearsal(config(True), ROOT, 4096)
    with pytest.raises(ValueError):
        load_fixed_flat_rehearsal(config(.25), ROOT, 32)
