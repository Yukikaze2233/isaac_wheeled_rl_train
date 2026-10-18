"""CPU source-flat integration; fixture populations do not select a live course."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.final_rough_commands import FinalRoughCommands
from wheeled_tasks.chassis.fixed_flat_rehearsal import (
    load_fixed_flat_rehearsal, install_fixed_flat_rehearsal,
)
from wheeled_tasks.chassis.flat_command_sampling import ScutFlatCommands, _COMMAND_TENSORS


ROOT = Path(__file__).resolve().parents[1]


def owner(count=64, ids=None, offset=4500, column=0):
    tiles = torch.zeros(10, 13, 3)
    tiles[..., 0] = torch.arange(10)[:, None] * 9. + 4.5
    tiles[..., 1] = torch.arange(13)[None, :] * 9. + 4.5
    origins = tiles[0, torch.full((count,), column)].clone()
    commands = FinalRoughCommands(count, "cpu", terrain_origins=tiles, env_origins=origins)
    if ids is not None:
        commands.set_source_flat_rehearsal(ids, source_iteration_offset=offset)
    return commands, origins


def tick(commands, rng, iteration, positions=None):
    commands.begin_policy()
    for _ in range(4):
        commands.advance_physics()
    kwargs = {} if positions is None else {"world_positions": positions}
    commands.advance(.02, torch.linspace(-1., 1., commands.num_envs), iteration, rng, **kwargs)


def assert_commands_equal(left, right):
    for name in _COMMAND_TENSORS:
        torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0,
                                   equal_nan=True, msg=name)




@pytest.mark.parametrize("offset,local", [(0, 1999), (0, 2000), (0, 4000), (4500, 1000)])
def test_all_flat_lanes_match_original_sampler_clock_cache_and_rng(offset, local):
    count = 32
    commands, positions = owner(count, torch.arange(count), offset)
    flat = ScutFlatCommands(count, "cpu")
    rng_a, rng_b = torch.Generator().manual_seed(812), torch.Generator().manual_seed(812)
    ids, ground = torch.arange(count), torch.arange(count) % 3 == 0
    commands.reset(ids, torch.zeros(count), ground, rng_a, local)
    flat.reset(ids, torch.zeros(count), ground, rng_b, offset + local)
    assert_commands_equal(commands, flat)
    for _ in range(76):
        tick(commands, rng_a, local, positions)
        tick(flat, rng_b, offset + local)
    assert_commands_equal(commands, flat)
    for value in (commands, flat):
        value.episode_physics_ticks.fill_(1000)
        value.time_left.fill_(.001)
    tick(commands, rng_a, local, positions)
    tick(flat, rng_b, offset + local)
    assert_commands_equal(commands, flat)
    assert torch.equal(rng_a.get_state(), rng_b.get_state())


def test_mixed_groups_keep_flat_clock_and_final_terrain_hooks_isolated():
    count, selected = 1024, torch.arange(256)
    before_global_rng = torch.get_rng_state()
    commands, positions = owner(count, selected, offset=1999, column=10)
    rng = torch.Generator().manual_seed(917)
    commands.reset(torch.arange(count), torch.zeros(count), torch.zeros(count, dtype=torch.bool), rng)
    commands.episode_physics_ticks.fill_(996)
    commands.time_left.zero_()
    tick(commands, rng, 0, positions)
    assert commands.special_mode_id[selected].eq(-1).all()
    assert all(commands.special_mode_id[256:].eq(mode).any() for mode in range(4))
    commands.time_left.zero_()
    tick(commands, rng, 1, positions)
    assert commands.special_mode_id[selected].eq(2).any()
    assert not commands.special_mode_id[selected].eq(0).any()
    assert not commands.special_mode_id[selected].eq(1).any()
    positions[:, 1] = 4.5  # Move rough lanes to the restricted stair profile.
    tick(commands, rng, 1, positions)
    plane = commands._final["keys"].index("plane_for_rm")
    assert commands._final["terrain_key_ids"][selected].eq(plane).all()
    assert commands._final["profile_switch_count"][selected].eq(0).all()
    assert not commands._final["axis_mask"][selected].any()
    assert not commands._final["override_masks"][selected].any()
    assert commands.special_mode_id[256:].eq(-1).all()
    assert not commands.fixed_request_mask.any()
    assert "fixed_training_rehearsal" not in commands.diagnostics()
    assert commands.metrics()["/source_command/source_flat_training_rehearsal_fraction"] == .25
    assert torch.equal(before_global_rng, torch.get_rng_state())

    # Extra final-terrain reset hooks must not advance flat postprocessing/cache.
    commands.is_standing[selected] = True
    commands.generator_command3[selected] = .33
    commands.generator_after_postprocess3[selected] = .44
    commands.generator_cache_postprocessed[selected] = False
    commands._final["axis_mask"][256:] = True
    commands._sync_heading_target(torch.arange(count), torch.zeros(count))
    commands.special_mode_id[256:] = 2
    commands._disable_special_modes(torch.arange(count), torch.zeros(count), rng)
    assert commands.generator_command3[selected].eq(.33).all()
    assert commands.generator_after_postprocess3[selected].eq(.44).all()
    assert not commands.generator_cache_postprocessed[selected].any()


def test_mixed_restart_binds_group_offset_and_replays_without_partial_mutation():
    selected = torch.arange(0, 64, 4)
    commands, positions = owner(ids=selected)
    rng = torch.Generator().manual_seed(812)
    commands.reset(torch.arange(64), torch.zeros(64), torch.arange(64) % 3 == 0, rng, 1000)
    tick(commands, rng, 1000, positions)
    state = commands.state_dict(rng)
    restored, _ = owner(ids=selected)
    other_rng = torch.Generator().manual_seed(123)
    restored.load_state_dict(state, other_rng)
    for value in (commands, restored):
        value.time_left.zero_()
    positions[1, 1] = 94.5
    tick(commands, rng, 1001, positions)
    tick(restored, other_rng, 1001, positions)
    assert_commands_equal(commands, restored)
    assert torch.equal(rng.get_state(), other_rng.get_state())
    for name, value in commands.diagnostics().items():
        torch.testing.assert_close(value, restored.diagnostics()[name], rtol=0, atol=0, equal_nan=True)
    for changed_ids, offset in ((selected + 1, 4500), (selected, 4501), (None, 4500)):
        incompatible, _ = owner(ids=changed_ids, offset=offset)
        before_rng = other_rng.get_state().clone()
        before = incompatible.state_dict(other_rng)
        with pytest.raises(ValueError, match="identity mismatch"):
            incompatible.load_state_dict(deepcopy(state), other_rng)
        assert torch.equal(before_rng, other_rng.get_state())
        after = incompatible.state_dict(other_rng)
        for name in before["source"]["tensors"]:
            assert torch.equal(before["source"]["tensors"][name], after["source"]["tensors"][name])


def test_source_rehearsal_rejects_live_switch_and_fixed_overlap():
    commands, _ = owner(ids=[0])
    with pytest.raises(ValueError, match="once before"):
        commands.set_source_flat_rehearsal([1])
    with pytest.raises(ValueError, match="cannot become fixed"):
        commands.set_fixed_requests([0], torch.zeros(1, 3), torch.tensor([.305]), evaluation_only=True)
    with pytest.raises(ValueError, match="cannot become fixed"):
        commands.set_training_rehearsal_requests([0], torch.zeros(1, 3), torch.tensor([.305]))
    fresh, _ = owner()
    with pytest.raises(ValueError, match="nonnegative integer"):
        fresh.set_source_flat_rehearsal([0], source_iteration_offset=True)
    fresh.reset([0], torch.zeros(64), torch.zeros(64, dtype=torch.bool), torch.Generator())
    with pytest.raises(ValueError, match="before the first reset"):
        fresh.set_source_flat_rehearsal([0])


@pytest.mark.parametrize("fixed_first", [False, True])
def test_sampled_flat_and_fixed_training_roles_coexist_without_terrain_overrides(fixed_first):
    commands, positions = owner(count=32, column=10)
    sampled, fixed = torch.tensor([0, 2, 4, 6]), torch.tensor([1, 3, 5, 7])
    requested = torch.tensor([[2.7, 0., 0.], [-2.7, 0., 0.],
                              [0., 0., 4.5 * torch.pi], [0., 0., 0.]])
    heights = torch.tensor([.2, .305, .42, .305])
    if fixed_first:
        commands.set_training_rehearsal_requests(fixed, requested, heights)
    commands.set_source_flat_rehearsal(sampled, source_iteration_offset=4500)
    if not fixed_first:
        commands.set_training_rehearsal_requests(fixed, requested, heights)

    diagnostics, summary, metrics = commands.diagnostics(), commands.summary(), commands.metrics()
    sampled_mask, fixed_mask = torch.zeros(32, dtype=torch.bool), torch.zeros(32, dtype=torch.bool)
    sampled_mask[sampled], fixed_mask[fixed] = True, True
    assert torch.equal(diagnostics["source_flat_training_rehearsal"], sampled_mask)
    assert torch.equal(diagnostics["fixed_training_rehearsal"], fixed_mask)
    assert torch.equal(commands.fixed_request_mask, fixed_mask)
    assert not diagnostics["fixed_evaluation_request"].any()
    assert summary["source_flat_training_rehearsal"]["mask_sha256"] != (
        summary["fixed_training_rehearsal"]["mask_sha256"])
    assert metrics["/source_command/source_flat_training_rehearsal_fraction"] == .125
    assert metrics["/source_command/fixed_training_rehearsal_fraction"] == .125
    assert metrics["/source_command/fixed_evaluation_fraction"] == 0.

    rng = torch.Generator().manual_seed(127)
    commands.reset(torch.arange(32), torch.zeros(32), torch.zeros(32, dtype=torch.bool), rng, 1000)
    commands.episode_physics_ticks.fill_(1000)
    commands.time_left.zero_()
    tick(commands, rng, 1000, positions)
    positions[:, 1] = 4.5  # Cross into a profile that restricts ordinary rough lanes.
    tick(commands, rng, 1001, positions)
    plane_ids = torch.cat((sampled, fixed))
    plane_key = commands._final["keys"].index("plane_for_rm")
    assert commands._final["terrain_key_ids"][plane_ids].eq(plane_key).all()
    assert commands._final["profile_switch_count"][plane_ids].eq(0).all()
    assert not commands._final["override_masks"][plane_ids].any()
    assert not commands._final["axis_mask"][plane_ids].any()
    assert commands._final["profile_switch_count"][8:].eq(1).all()
    torch.testing.assert_close(commands.command3[fixed], requested, rtol=0, atol=0)
    torch.testing.assert_close(commands.height[fixed], heights, rtol=0, atol=0)
    assert commands.height[sampled].ge(.2).all() and commands.height[sampled].le(.42).all()

    with pytest.raises(ValueError, match="before the first reset"):
        commands.set_training_rehearsal_requests([9], torch.zeros(1, 3), torch.tensor([.305]))


def test_combined_roles_restart_exactly_and_bind_both_static_allocations():
    sampled, fixed = torch.arange(0, 32, 4), torch.tensor([1, 5, 9, 13])
    requests, heights = torch.zeros(4, 3), torch.full((4,), .305)
    requests[:, 0] = torch.tensor([2.7, -2.7, 1.35, -1.35])
    commands, positions = owner(32, sampled, column=10)
    commands.set_training_rehearsal_requests(fixed, requests, heights)
    rng = torch.Generator().manual_seed(127)
    commands.reset(torch.arange(32), torch.zeros(32), torch.zeros(32, dtype=torch.bool), rng, 1000)
    tick(commands, rng, 1000, positions)
    state = commands.state_dict(rng)

    restored, _ = owner(32, column=10)
    restored.set_training_rehearsal_requests(fixed, requests, heights)
    restored.set_source_flat_rehearsal(sampled)
    other_rng = torch.Generator().manual_seed(991)
    restored.load_state_dict(deepcopy(state), other_rng)
    for value in (commands, restored):
        value.episode_physics_ticks.fill_(1000)
        value.time_left.zero_()
    positions[:, 1] = 4.5
    tick(commands, rng, 1001, positions)
    tick(restored, other_rng, 1001, positions)
    assert_commands_equal(commands, restored)
    assert torch.equal(rng.get_state(), other_rng.get_state())
    for name, value in commands.diagnostics().items():
        torch.testing.assert_close(value, restored.diagnostics()[name], rtol=0, atol=0, equal_nan=True)

    for changed_sampled, changed_fixed in ((sampled[1:], fixed), (sampled, fixed + 2)):
        incompatible, _ = owner(32, changed_sampled, column=10)
        incompatible.set_training_rehearsal_requests(changed_fixed, requests, heights)
        before = incompatible.state_dict(other_rng)
        with pytest.raises(ValueError, match="identity mismatch"):
            incompatible.load_state_dict(deepcopy(state), other_rng)
        after = incompatible.state_dict(other_rng)
        assert torch.equal(before["source"]["generator_state"], after["source"]["generator_state"])
        assert before["source"]["clocks"] == after["source"]["clocks"]
        for name in before["final"]:
            assert torch.equal(before["final"][name], after["final"][name])
        for name in before["source"]["tensors"]:
            assert torch.equal(before["source"]["tensors"][name], after["source"]["tensors"][name])


def test_fixed_first_overlap_rejection_keeps_allocation_and_observer_unchanged():
    commands, _ = owner()
    commands.set_training_rehearsal_requests([1], torch.zeros(1, 3), torch.tensor([.305]))
    before = commands.state_dict(torch.Generator().manual_seed(17))
    with pytest.raises(ValueError, match="without fixed requests"):
        commands.set_source_flat_rehearsal([0, 1])
    after = commands.state_dict(torch.Generator().manual_seed(17))
    assert before["source"]["identity"] == after["source"]["identity"]
    for name in before["source"]["tensors"]:
        assert torch.equal(before["source"]["tensors"][name], after["source"]["tensors"][name])
    assert "source_flat_training_rehearsal" not in commands.diagnostics()


@pytest.mark.parametrize("physical_tick", [296, 300, 304])
@pytest.mark.parametrize("reset_before_advance", [False, True])
def test_partial_reset_leaves_other_flat_ground_expiry_for_command_advance(physical_tick, reset_before_advance):
    commands, positions = owner(3, [0, 1, 2], offset=0)
    flat = ScutFlatCommands(3, "cpu")
    rng_a, rng_b = torch.Generator().manual_seed(123), torch.Generator().manual_seed(123)
    ids, ground = torch.arange(3), torch.tensor([True, False, False])
    commands.reset(ids, torch.zeros(3), ground, rng_a)
    flat.reset(ids, torch.zeros(3), ground, rng_b)
    complete_intervals = physical_tick // 4 - int(reset_before_advance)
    for _ in range(complete_intervals):
        tick(commands, rng_a, 0, positions)
        tick(flat, rng_b, 0)
    if reset_before_advance:
        for value in (commands, flat):
            value.begin_policy()
            for _ in range(4):
                value.advance_physics()
    assert_commands_equal(commands, flat)
    commands.reset(torch.tensor([1]), torch.tensor([.3]), torch.tensor([False]), rng_a)
    flat.reset(torch.tensor([1]), torch.tensor([.3]), torch.tensor([False]), rng_b)
    assert_commands_equal(commands, flat)
    if reset_before_advance:
        commands.advance(.02, torch.linspace(-1., 1., 3), 0, rng_a, world_positions=positions)
        flat.advance(.02, torch.linspace(-1., 1., 3), 0, rng_b)
        assert_commands_equal(commands, flat)
    assert torch.equal(rng_a.get_state(), rng_b.get_state())
