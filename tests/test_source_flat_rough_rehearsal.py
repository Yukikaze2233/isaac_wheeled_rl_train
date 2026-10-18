"""Source-distribution rehearsal keeps flat semantics and restart identity."""
from copy import deepcopy

import pytest
import torch

from wheeled_tasks.chassis.flat_command_sampling import ScutFlatCommands, _COMMAND_TENSORS
from wheeled_tasks.chassis.rough_commands import ScutRoughCommands


def owner(count=128, ids=None, offset=4500):
    tiles = torch.zeros(10, 14, 3)
    tiles[:, :, 0] = ((torch.arange(10) + .5) * 9. - 45.)[:, None]
    tiles[:, :, 1] = ((torch.arange(14) + .5) * 9. - 63.)[None, :]
    origins = tiles[4, torch.arange(count) % 14].clone()
    commands = ScutRoughCommands(count, "cpu", terrain_origins=tiles, env_origins=origins)
    if ids is not None:
        commands.set_source_flat_rehearsal(ids, source_iteration_offset=offset)
    return commands, origins


def advance(commands, rng, iteration, positions=None):
    commands.begin_policy()
    for _ in range(4):
        commands.advance_physics()
    kwargs = {} if positions is None else {"world_positions": positions}
    commands.advance(.02, torch.linspace(-1., 1., commands.num_envs), iteration, rng, **kwargs)


def assert_flat_equal(a, b):
    for name in _COMMAND_TENSORS:
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0,
                                   equal_nan=True, msg=name)


@pytest.mark.parametrize("offset,local", [(0, 0), (0, 1999), (0, 2000), (0, 3000),
                                         (0, 4000), (4500, 1000)])
def test_all_rehearsal_lanes_replay_original_flat_sampler_exactly(offset, local):
    n = 128
    rough, positions = owner(n, torch.arange(n), offset)
    flat = ScutFlatCommands(n, "cpu")
    rng_a, rng_b = torch.Generator().manual_seed(812), torch.Generator().manual_seed(812)
    ground = torch.arange(n) % 3 == 0
    rough.reset(torch.arange(n), torch.zeros(n), ground, rng_a, local)
    flat.reset(torch.arange(n), torch.zeros(n), ground, rng_b, offset + local)
    assert_flat_equal(rough, flat)
    # Exercise ground override/cache restoration, where the rough V13 publish
    # hook must not replace the original flat hook on rehearsal lanes.
    for _ in range(76):
        advance(rough, rng_a, local, positions)
        advance(flat, rng_b, offset + local)
    assert_flat_equal(rough, flat)
    assert torch.equal(rng_a.get_state(), rng_b.get_state())
    # Make surviving lanes eligible for the source's timed special-mode draw.
    for commands in (rough, flat):
        commands.episode_physics_ticks.fill_(1000)
        commands.time_left.fill_(.001)
    advance(rough, rng_a, local, positions)
    advance(flat, rng_b, offset + local)
    assert_flat_equal(rough, flat)
    assert torch.equal(rng_a.get_state(), rng_b.get_state())
    assert not rough.fixed_request_mask.any()
    assert torch.unique(rough.height).numel() > 100


def test_mixed_schedule_keeps_rough_active_and_flat_inherited_clock():
    n = 4096
    selected = torch.arange(1024)
    commands, positions = owner(n, selected, 1999)
    rng = torch.Generator().manual_seed(917)
    commands.reset(torch.arange(n), torch.zeros(n), torch.zeros(n, dtype=torch.bool), rng)
    commands.episode_physics_ticks.fill_(1000)
    commands.time_left.fill_(.001)
    advance(commands, rng, 0, positions)
    assert (commands.special_mode_id[selected] == -1).all()
    assert all((commands.special_mode_id[1024:] == mode).sum() > 100 for mode in range(3))
    commands.time_left.fill_(.001)
    advance(commands, rng, 1, positions)  # 1999 + 1: only flat dash has opened.
    assert (commands.special_mode_id[selected] == 2).sum() > 200
    assert not ((commands.special_mode_id[selected] == 0) | (commands.special_mode_id[selected] == 1)).any()
    plane = commands._rough["definition"]["keys"].index("plane_for_rm_rot")
    assert (commands._rough["terrain_key_ids"][selected] == plane).all()
    assert commands.height[selected].min() < .205 and commands.height[selected].max() > .415
    assert torch.unique(commands.height[selected]).numel() > 1000
    assert not commands.fixed_request_mask.any()
    assert "fixed_training_rehearsal" not in commands.diagnostics()
    assert commands.metrics()["/source_command/source_flat_training_rehearsal_fraction"] == .25
    assert commands.metrics()["/source_command/fixed_evaluation_fraction"] == 0
    metrics = commands.metrics()
    prefix = "/source_command/source_flat_rehearsal_height_"
    assert metrics[prefix + "mean_m"] == commands.height[selected].mean()
    assert sum(metrics[prefix + band + "_fraction"] for band in ("low", "middle", "high")) == 1
    assert .12 < metrics[prefix + "low_fraction"] < .24
    assert .12 < metrics[prefix + "high_fraction"] < .24


def test_mixed_restart_preserves_rng_profiles_clock_and_rejects_config_change():
    ids = torch.arange(0, 128, 4)
    commands, positions = owner(ids=ids)
    rng = torch.Generator().manual_seed(812)
    commands.reset(torch.arange(128), torch.zeros(128), torch.zeros(128, dtype=torch.bool), rng, 1000)
    commands.episode_physics_ticks.fill_(1000)
    commands.time_left.fill_(.001)
    advance(commands, rng, 1000, positions)
    state = commands.state_dict(rng)
    restored, _ = owner(ids=ids)
    rng_restored = torch.Generator().manual_seed(123)
    restored.load_state_dict(state, rng_restored)
    positions[1, 1] = positions[5, 1]
    commands.time_left.fill_(.001)
    restored.time_left.fill_(.001)
    advance(commands, rng, 1001, positions)
    advance(restored, rng_restored, 1001, positions)
    for key, value in commands.diagnostics().items():
        assert torch.equal(value, restored.diagnostics()[key]), key
    assert torch.equal(rng.get_state(), rng_restored.get_state())
    for changed_ids, offset in ((ids, 4501), (ids + 1, 4500)):
        incompatible, _ = owner(ids=changed_ids, offset=offset)
        rng_before = rng_restored.get_state().clone()
        with pytest.raises(ValueError, match="identity mismatch"):
            incompatible.load_state_dict(deepcopy(state), rng_restored)
        assert torch.equal(rng_before, rng_restored.get_state())


def test_source_rehearsal_configuration_rejects_live_switch_and_fixed_overlap():
    commands, _ = owner(ids=[0])
    with pytest.raises(ValueError, match="once before"):
        commands.set_source_flat_rehearsal([1])
    with pytest.raises(ValueError, match="cannot become fixed"):
        commands.set_fixed_requests([0], torch.zeros(1, 3), torch.tensor([.305]), evaluation_only=True)
    with pytest.raises(ValueError, match="cannot become fixed"):
        commands.set_training_rehearsal_requests([0], torch.zeros(1, 3), torch.tensor([.305]))
    commands.set_training_rehearsal_requests([1], torch.zeros(1, 3), torch.tensor([.305]))
    fresh, _ = owner()
    with pytest.raises(ValueError, match="nonnegative integer"):
        fresh.set_source_flat_rehearsal([0], source_iteration_offset=True)
    fresh.reset([0], torch.zeros(128), torch.zeros(128, dtype=torch.bool), torch.Generator())
    with pytest.raises(ValueError, match="before the first reset"):
        fresh.set_source_flat_rehearsal([0])


def mixed_owner(order=("source", "fixed", "temporal")):
    commands, positions = owner(count=14)
    for group in order:
        if group == "source":
            commands.set_source_flat_rehearsal([0, 1], source_iteration_offset=14841)
        elif group == "fixed":
            commands.set_training_rehearsal_requests([2], torch.tensor([[.5, 0., 0.]]), torch.tensor([.42]))
        else:
            commands.set_temporal_training_requests([3, 4],
                torch.tensor([[3., 0., .305], [-2.7, 0., .2]]),
                phase_policy_ticks=[2, 3, 2, 2], episode_policy_ticks=10)
    return commands, positions


@pytest.mark.parametrize("order", [("source", "fixed", "temporal"), ("fixed", "temporal", "source"),
                                  ("temporal", "source", "fixed")])
def test_disjoint_rehearsal_owners_survive_terrain_switches(order):
    commands, positions = mixed_owner(order)
    rng = torch.Generator().manual_seed(702)
    commands.reset(torch.arange(14), torch.zeros(14), torch.zeros(14, dtype=torch.bool), rng)
    plane = commands._rough["definition"]["keys"].index("plane_for_rm_rot")
    positions[:, 1] = -62.
    for phase in [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]:
        assert commands.command3[2].tolist() == [.5, 0., 0.]
        assert commands.height[2] == pytest.approx(.42)
        assert commands.command3[3, 0] == [3., 0., -3., 0.][phase]
        assert commands.command3[4, 0] == pytest.approx([-2.7, 0., 2.7, 0.][phase])
        assert commands.command3[3:5, 1:].count_nonzero() == 0
        assert commands.height[3:5].tolist() == pytest.approx([.305, .2])
        assert (commands._rough["terrain_key_ids"][:5] == plane).all()
        advance(commands, rng, 0, positions)
    assert commands._rough["profile_switch_count"][:5].count_nonzero() == 0
    diagnostics = commands.diagnostics()
    assert diagnostics["fixed_training_rehearsal"].nonzero().flatten().tolist() == [2]
    assert diagnostics["source_flat_training_rehearsal"].nonzero().flatten().tolist() == [0, 1]
    assert diagnostics["temporal_training_rehearsal"].nonzero().flatten().tolist() == [3, 4]
    assert not diagnostics["fixed_evaluation_request"].any()
    metrics = commands.metrics()
    assert metrics["/source_command/source_flat_training_rehearsal_fraction"] == pytest.approx(2 / 14)
    assert metrics["/source_command/fixed_training_rehearsal_fraction"] == pytest.approx(1 / 14)


def test_mixed_rehearsal_resume_keeps_all_ownership_and_phase():
    commands, positions = mixed_owner()
    rng = torch.Generator().manual_seed(57)
    commands.reset(torch.arange(14), torch.zeros(14), torch.zeros(14, dtype=torch.bool), rng)
    for _ in range(4):
        advance(commands, rng, 17, positions)
    restored, _ = mixed_owner(("fixed", "source", "temporal"))
    other_rng = torch.Generator().manual_seed(1)
    restored.load_state_dict(commands.state_dict(rng), other_rng)
    for _ in range(5):
        advance(commands, rng, 18, positions)
        advance(restored, other_rng, 18, positions)
    assert_flat_equal(commands, restored)
    assert commands.summary() == restored.summary()
    for key, value in commands.diagnostics().items():
        assert torch.equal(value, restored.diagnostics()[key]), key
    assert torch.equal(rng.get_state(), other_rng.get_state())


@pytest.mark.parametrize("temporal_first", [False, True])
def test_source_temporal_overlap_rejected_before_ownership_changes(temporal_first):
    commands, _ = owner(count=14)
    request = torch.tensor([[3., 0., .305]])
    if temporal_first:
        commands.set_temporal_training_requests([0], request,
            phase_policy_ticks=[1, 1, 1, 1], episode_policy_ticks=4)
        with pytest.raises(ValueError, match="disjoint"):
            commands.set_source_flat_rehearsal([0])
        assert commands._rough["source_flat_rehearsal"] is None
    else:
        commands.set_source_flat_rehearsal([0])
        with pytest.raises(ValueError, match="disjoint"):
            commands.set_temporal_training_requests([0], request,
                phase_policy_ticks=[1, 1, 1, 1], episode_policy_ticks=4)
        assert commands._temporal_rehearsal is None
