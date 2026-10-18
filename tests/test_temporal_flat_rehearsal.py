"""CPU command ownership checks; durations below are fixtures, not a course."""
import pytest
import torch

from wheeled_tasks.chassis.flat_command_sampling import ScutFlatCommands


def owner():
    commands = ScutFlatCommands(3, "cpu")
    commands.set_temporal_training_requests(torch.tensor([0, 2]),
        torch.tensor([[3., 0., .305], [-2.7, 0., .20]]),
        phase_policy_ticks=[2, 3, 4, 2], episode_policy_ticks=12)
    return commands


def reset(commands, generator, ids=None):
    ids = torch.arange(3) if ids is None else ids
    commands.reset(ids, torch.zeros(len(ids)), torch.zeros(len(ids), dtype=torch.bool), generator, 14841)


def step(commands, generator):
    commands.begin_policy()
    for _ in range(4):
        commands.advance_physics()
    paid_command = commands.command3.clone()
    paid_phase = commands.diagnostics()["temporal_published_phase"].clone()
    metrics = commands.metrics()
    commands.advance(.02, torch.zeros(3), 14841, generator)
    return paid_command, paid_phase, metrics


def test_exact_policy_boundary_signs_no_slew_and_partial_reset():
    commands, rng = owner(), torch.Generator().manual_seed(23)
    reset(commands, rng)
    assert commands.summary()["temporal_training_rehearsal"]["phase_policy_ticks"] == [2, 3, 4, 2]
    expected_phase = [0] * 2 + [1] * 3 + [2] * 4 + [3] * 3
    for phase in expected_phase:
        before = commands.command3.clone()
        paid, published_phase, metrics = step(commands, rng)
        torch.testing.assert_close(paid, before)
        assert published_phase.tolist() == [phase, -1, phase]
        assert paid[0, 0].item() == [3., 0., -3., 0.][phase]
        assert paid[2, 0].item() == pytest.approx([-2.7, 0., 2.7, 0.][phase])
        assert commands.height[[0, 2]].tolist() == pytest.approx([.305, .20])
        assert sum(v.item() for k, v in metrics.items() if k.startswith("/temporal_request/")) == 2
    ordinary = commands.command3[1].clone()
    reset(commands, rng, torch.tensor([2]))
    assert commands.command3[2, 0].item() == pytest.approx(-2.7)
    assert commands.diagnostics()["temporal_published_phase"].tolist() == [3, -1, 0]
    torch.testing.assert_close(commands.command3[1], ordinary)


def test_unselected_sampling_rng_and_masks_match_existing_runtime():
    temporal, baseline = owner(), ScutFlatCommands(3, "cpu")
    a, b = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    for commands, rng in ((temporal, a), (baseline, b)):
        reset(commands, rng)
    for _ in range(800):
        for commands, rng in ((temporal, a), (baseline, b)):
            commands.begin_policy()
            for _ in range(4):
                commands.advance_physics()
            commands.advance(.02, torch.zeros(3), 14841, rng)
        torch.testing.assert_close(temporal.command3[1], baseline.command3[1], rtol=0., atol=0.)
        torch.testing.assert_close(temporal.height[1], baseline.height[1], rtol=0., atol=0.)
        assert torch.equal(temporal.learning_mask, baseline.learning_mask)
        assert torch.equal(a.get_state(), b.get_state())


def test_restart_mid_interval_preserves_phase_reward_command_and_next_rng():
    commands, a = owner(), torch.Generator().manual_seed(72)
    reset(commands, a)
    for _ in range(4):
        step(commands, a)
    commands.begin_policy()
    commands.advance_physics()
    commands.advance_physics()
    state = commands.state_dict(a)
    restored, b = owner(), torch.Generator().manual_seed(901)
    restored.load_state_dict(state, b)
    for instance, rng in ((commands, a), (restored, b)):
        for _ in range(2):
            instance.advance_physics()
        assert instance.diagnostics()["temporal_published_phase"].tolist() == [1, -1, 1]
        instance.advance(.02, torch.zeros(3), 14841, rng)
    assert torch.equal(commands.command3, restored.command3)
    assert torch.equal(a.get_state(), b.get_state())
    wrong = ScutFlatCommands(3, "cpu")
    wrong.set_temporal_training_requests(torch.tensor([0, 2]),
        torch.tensor([[3., 0., .305], [-2.7, 0., .20]]),
        phase_policy_ticks=[3, 2, 4, 2], episode_policy_ticks=12)
    before = b.get_state().clone()
    with pytest.raises(ValueError, match="identity mismatch"):
        wrong.load_state_dict(state, b)
    assert torch.equal(before, b.get_state())


def test_conflicting_owners_and_ground_override_rejected_before_mutation():
    commands, rng = owner(), torch.Generator().manual_seed(44)
    before = rng.get_state().clone()
    with pytest.raises(ValueError, match="ground override"):
        commands.reset(torch.arange(3), torch.zeros(3), torch.tensor([True, False, False]), rng)
    assert torch.equal(before, rng.get_state()) and not commands.command_counter.any()
    with pytest.raises(ValueError, match="disjoint"):
        commands.set_training_rehearsal_requests(torch.tensor([0]), torch.zeros(1, 3), torch.tensor([.3]))
    assert not commands.fixed_request_mask.any()
    # Disjoint fixed anchors remain supported in either installation order.
    commands.set_training_rehearsal_requests(torch.tensor([1]), torch.tensor([[.5, 0., 0.]]), torch.tensor([.3]))
    other = ScutFlatCommands(3, "cpu")
    other.set_training_rehearsal_requests(torch.tensor([1]), torch.tensor([[.5, 0., 0.]]), torch.tensor([.3]))
    other.set_temporal_training_requests(torch.tensor([0]), torch.tensor([[3., 0., .305]]),
        phase_policy_ticks=[1, 1, 1, 1], episode_policy_ticks=4)
    reset(commands, rng)
    torch.testing.assert_close(commands.command3[1], torch.tensor([.5, 0., 0.]))


@pytest.mark.parametrize("ticks,horizon", [([1, 1, 1, 0], 4), ([1, 1, 1, 1], 3),
    ([1., 1, 1, 1], 4), ([True, 1, 1, 1], 4), ([1, 1, 1], 4)])
def test_incomplete_or_unbounded_sequence_rejected(ticks, horizon):
    commands = ScutFlatCommands(3, "cpu")
    with pytest.raises(ValueError, match="actual episode horizon"):
        commands.set_temporal_training_requests(torch.tensor([0]), torch.tensor([[3., 0., .305]]),
            phase_policy_ticks=ticks, episode_policy_ticks=horizon)
    assert commands._temporal_rehearsal is None
