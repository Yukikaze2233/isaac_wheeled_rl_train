"""Training plane ownership labels and static command restart identity, CPU only."""
import pytest
import torch

from wheeled_tasks.chassis.rough_commands import ScutRoughCommands


def owner(training_lane=None):
    grid = torch.zeros(10, 14, 3)
    grid[:, :, 0] = ((torch.arange(10) + .5) * 9. - 45.)[:, None]
    grid[:, :, 1] = ((torch.arange(14) + .5) * 9. - 63.)[None, :]
    origins = torch.tensor([[0., 200., 0.], [0., 200., 0.]])
    commands = ScutRoughCommands(2, "cpu", terrain_origins=grid, env_origins=origins)
    # Identical fixed command buffers let the restart test distinguish only
    # their static training/evaluation ownership, not their requested targets.
    requests = torch.tensor([[2.7, 0., 0.], [2.7, 0., 0.]])
    heights = torch.full((2,), .305)
    if training_lane is not None:
        commands.set_training_rehearsal_requests([training_lane], requests, heights)
    ids = [i for i in range(2) if i != training_lane]
    commands.set_fixed_requests(ids, requests, heights, evaluation_only=True)
    return commands, origins


def reset(commands, ground=False):
    generator = torch.Generator().manual_seed(910)
    commands.reset([0, 1], torch.zeros(2), torch.tensor([ground, False]), generator)
    return generator


def advance(commands, generator, origins):
    commands.begin_policy()
    for _ in range(4):
        commands.advance_physics()
    commands.advance(.02, torch.zeros(2), 5000, generator, world_positions=origins)


def test_plane_training_label_is_separate_from_unchanged_fixed_evaluation():
    commands, origins = owner(training_lane=0)
    generator = reset(commands)
    plane = commands._rough["definition"]["keys"].index("plane_for_rm_rot")
    random = commands._rough["definition"]["keys"].index("random_uniform_for_rm")
    for _ in range(2):
        diagnostics = commands.diagnostics()
        assert diagnostics["terrain_key_ids"].tolist() == [plane, random]
        assert diagnostics["fixed_training_rehearsal"].tolist() == [True, False]
        assert diagnostics["fixed_evaluation_request"].tolist() == [False, True]
        metrics = commands.metrics()
        assert metrics["/source_command/fixed_training_rehearsal_fraction"].item() == .5
        assert metrics["/source_command/fixed_evaluation_fraction"].item() == .5
        assert metrics["/source_command/terrain_fraction/plane_for_rm_rot"].item() == .5
        commands.time_left.zero_()
        advance(commands, generator, origins)
    ordinary, _ = owner()
    reset(ordinary)
    assert ordinary._training_rehearsal_mask is None
    assert "fixed_training_rehearsal" not in ordinary.diagnostics()
    assert "/source_command/fixed_training_rehearsal_fraction" not in ordinary.metrics()
    assert ordinary.metrics()["/source_command/fixed_evaluation_fraction"].item() == 1.


def test_training_plane_retains_source_ground_hold_and_temporary_command():
    commands, origins = owner(training_lane=0)
    generator = reset(commands, ground=True)
    temporary = commands.command3[0].clone()
    assert not torch.equal(temporary, commands.fixed_request_command3[0])
    for tick in range(75):
        assert torch.equal(commands.command3[0], temporary)
        assert commands.holding[0].item() == (tick < 10)
        advance(commands, generator, origins)
    assert commands.episode_physics_ticks[0].item() == 300
    assert not commands.ground_override_pending[0]
    assert torch.equal(commands.command3[0], commands.fixed_request_command3[0])
    assert commands.diagnostics()["fixed_training_rehearsal"][0]


def test_command_restart_binds_static_training_allocation_without_restoring_it():
    commands, origins = owner(training_lane=0)
    generator = reset(commands)
    advance(commands, generator, origins)
    snapshot = commands.state_dict(generator)
    restored, _ = owner(training_lane=0)
    other_rng = reset(restored)
    mask = restored._training_rehearsal_mask
    restored.load_state_dict(snapshot, other_rng)
    assert restored._training_rehearsal_mask is mask
    assert torch.equal(restored.command3, commands.command3)
    assert torch.equal(other_rng.get_state(), generator.get_state())
    for lane in (None, 1):
        different, _ = owner(training_lane=lane)
        rejected_rng = reset(different)
        before_rng, before_commands = rejected_rng.get_state().clone(), different.command3.clone()
        assert torch.equal(different.fixed_request_mask, commands.fixed_request_mask)
        with pytest.raises(ValueError, match="identity mismatch"):
            different.load_state_dict(snapshot, rejected_rng)
        assert torch.equal(rejected_rng.get_state(), before_rng)
        assert torch.equal(different.command3, before_commands)
