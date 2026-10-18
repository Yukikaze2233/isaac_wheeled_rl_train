"""One mixed birth transaction with CPU doubles; no physics or PPO qualification."""
from dataclasses import replace

import pytest
import torch

from test_v7_birth_coverage_metrics import (
    NominalMapper, PlacementDouble, coverage, equal,
)
from wheeled_tasks.chassis.flat_episode_runtime import (
    FlatEpisodeRuntime, FlatResetState, MECHANICAL_RESTART_VERSION,
)
from wheeled_tasks.chassis.nonsteady_initial_state import CONTROL_JOINT_NAMES


class MixedMapper(NominalMapper):
    def map_requests(self, length, angle, wheel, velocity):
        result = super().map_requests(length, angle, wheel, velocity)
        result.q[:, 2], result.q[:, 5] = wheel[:, 0], wheel[:, 1]
        return result


def owner(*, refusals=(), redraw=False):
    identity = {"schema": MECHANICAL_RESTART_VERSION, **{name: char * 64 for name, char in (
        ("asset_manifest_sha256", "a"), ("model_spec_sha256", "b"),
        ("branch_table_sha256", "c"), ("adapter_configuration_sha256", "d"))}}
    names = (*CONTROL_JOINT_NAMES, *(f"passive{i}" for i in range(12)))
    return FlatEpisodeRuntime(5, "cpu", MixedMapper(refusals), PlacementDouble(), names,
        birth_rejection={"mode": "redraw", "max_attempts": 2} if redraw else {"mode": "abort"},
        restart_identity=identity)


def inputs():
    ids = torch.tensor([4, 1, 3, 0])
    q = torch.arange(72, dtype=torch.float64).reshape(4, 18) / 1000.
    return ids, q, dict(origins=torch.arange(12).reshape(4, 3).float(),
        reset_xy_world=torch.tensor([[40., 41.], [10., 11.], [30., 31.], [0., 1.]]),
        ground_z_world=torch.tensor([.4, .1, .3, 0.]))


def compare_rows(actual, rows, expected):
    for field in FlatResetState.__dataclass_fields__:
        if field == "requests":
            for name, value in vars(actual.requests).items():
                equal(value[rows], getattr(expected.requests, name))
        else:
            equal(getattr(actual, field)[rows], getattr(expected, field))


def test_mixed_unordered_ids_reuse_both_paths_and_only_dynamic_subset_consumes_rng():
    ids, q, values = inputs()
    mask = torch.tensor([True, False, False, True])
    mixed = owner(refusals=[[1], []], redraw=True)
    generator = torch.Generator().manual_seed(19)
    state = mixed.prepare_reset(ids, generator, **values, nominal_q=q, nominal_mask=mask)
    assert mixed.pending_reset is state
    assert torch.equal(state.env_ids, ids) and torch.equal(state.requests.env_ids, ids)
    equal(state.joint_position[mask], q[mask])
    assert not state.joint_velocity[mask].any()
    assert not state.requests.ground_birth[mask].any()
    assert not state.requests.control_velocity_rad_s[mask].any()
    assert state.requests.leg_mode[mask].eq(-1).all()
    assert state.requests.control_velocity_rad_s[~mask].ne(0.).all()
    assert not torch.equal(state.joint_position[~mask], q[~mask])
    assert state.proposal_counts.tolist() == [1, 1, 2, 1]
    assert mixed.episode_id.eq(-1).all()
    assert coverage(mixed)["accepted_reset_count"] == 0

    dynamic, dynamic_generator = owner(refusals=[[1], []], redraw=True), torch.Generator().manual_seed(19)
    expected_dynamic = dynamic.prepare_reset(ids[~mask], dynamic_generator,
        **{name: value[~mask] for name, value in values.items()})
    compare_rows(state, ~mask, expected_dynamic)
    assert torch.equal(generator.get_state(), dynamic_generator.get_state())
    nominal, nominal_generator = owner(), torch.Generator().manual_seed(19)
    before = nominal_generator.get_state().clone()
    expected_nominal = nominal.prepare_reset(ids[mask], nominal_generator,
        **{name: value[mask] for name, value in values.items()}, nominal_q=q[mask])
    compare_rows(state, mask, expected_nominal)
    assert torch.equal(nominal_generator.get_state(), before)


@pytest.mark.parametrize("fixed", [False, True])
def test_uniform_masks_are_exact_old_paths_including_owned_state_and_rng(fixed):
    ids, q, values = inputs()
    original, explicit = owner(), owner()
    first, second = torch.Generator().manual_seed(31), torch.Generator().manual_seed(31)
    expected = original.prepare_reset(ids, first, **values, nominal_q=q if fixed else None)
    actual = explicit.prepare_reset(ids, second, **values, nominal_q=q,
        nominal_mask=torch.full((4,), fixed, dtype=torch.bool))
    compare_rows(actual, torch.arange(4), expected)
    assert torch.equal(first.get_state(), second.get_state())
    equal(original.state_dict(first), explicit.state_dict(second))
    equal(coverage(original), coverage(explicit))


def test_dynamic_failure_clears_partial_pending_without_commit_or_refunding_proposals():
    ids, q, values = inputs()
    runtime = owner(refusals=[[0]])
    generator = torch.Generator().manual_seed(29)
    rng = generator.get_state().clone()
    commands = runtime.commands.state_dict(generator)
    with pytest.raises(RuntimeError, match="rejected lanes"):
        runtime.prepare_reset(ids, generator, **values, nominal_q=q,
            nominal_mask=torch.tensor([True, False, False, True]))
    assert runtime.pending_reset is None
    assert runtime.episode_id.eq(-1).all()
    assert runtime.birth_proposal_count[ids].tolist() == [1, 1, 1, 1]
    assert runtime.birth_rejection_count[ids].tolist() == [0, 1, 0, 0]
    assert coverage(runtime)["proposal_count"] == 4
    assert coverage(runtime)["accepted_reset_count"] == 0
    assert not torch.equal(generator.get_state(), rng)
    # The only intentional change is consumed generator state, not commands.
    after = runtime.commands.state_dict(generator)
    after["generator_state"] = commands["generator_state"]
    equal(after, commands)


def test_merged_pending_uses_existing_schema_and_commits_commands_exactly_once(monkeypatch):
    ids, q, values = inputs()
    runtime = owner()
    generator = torch.Generator().manual_seed(19)
    state = runtime.prepare_reset(ids, generator, **values, nominal_q=q,
        nominal_mask=torch.tensor([True, False, False, True]))
    saved = runtime.state_dict(generator)
    assert set(saved) == {"schema", "identity", "commands", "tensors", "pending_reset"}
    resumed, new_generator = owner(), torch.Generator().manual_seed(99)
    resumed.load_state_dict(saved, new_generator)
    equal(resumed.state_dict(new_generator), saved)
    compare_rows(resumed.pending_reset, torch.arange(4), state)
    with pytest.raises(RuntimeError, match="exact pending"):
        runtime.commit_reset(replace(state), torch.zeros(4), generator, 5000)
    assert runtime.pending_reset is state
    calls = []
    reset = type(runtime.commands).reset

    def counted_reset(commands, *args, **kwargs):
        calls.append(args[0].clone())
        return reset(commands, *args, **kwargs)

    monkeypatch.setattr(type(runtime.commands), "reset", counted_reset)
    runtime.commit_reset(state, torch.zeros(4), generator, 5000)
    assert len(calls) == 1 and torch.equal(calls[0], ids)
    assert runtime.pending_reset is None
    assert runtime.episode_id[ids].eq(0).all() and runtime.episode_id[2] == -1
    assert coverage(runtime)["accepted_reset_count"] == 4
    with pytest.raises(RuntimeError, match="exact pending"):
        runtime.commit_reset(state, torch.zeros(4), generator, 5000)


def test_ground_dynamic_rows_keep_full_200ms_zero_effort_and_interval_learning_mask():
    ids, q, values = inputs()
    mask = torch.tensor([True, False, False, False])
    runtime, generator = owner(), torch.Generator().manual_seed(19)
    state = runtime.prepare_reset(ids, generator, **values, nominal_q=q, nominal_mask=mask)
    assert state.requests.ground_birth[~mask].any()
    assert not state.requests.ground_birth[mask].any()
    runtime.commit_reset(state, torch.zeros(4), generator, 5000)
    ground = torch.zeros(5, dtype=torch.bool)
    ground[ids] = state.requests.ground_birth
    effort = torch.ones(5, 6)
    for policy in range(10):
        assert torch.equal(runtime.begin_policy(), ~ground)
        for _ in range(4):
            paid = runtime.gate_active_effort(effort)
            assert not paid[ground].any()
            assert torch.equal(paid[~ground], effort[~ground])
            runtime.advance_physics(.005)
        assert torch.equal(runtime.learning_mask, ~ground)
        runtime.advance(.02, torch.zeros(5), 5000, generator)
        assert torch.equal(runtime.holding, ground if policy < 9 else torch.zeros(5, dtype=torch.bool))
    assert runtime.begin_policy().all()
    assert torch.equal(runtime.gate_active_effort(effort), effort)


@pytest.mark.parametrize("invalid_mask,invalid_q", [
    (True, None), (torch.ones(4), None),
    (torch.ones(3, dtype=torch.bool), None),
    (torch.ones(4, dtype=torch.bool), torch.zeros(18)),
])
def test_invalid_mixed_inputs_fail_before_sampler_or_pending_mutation(invalid_mask, invalid_q):
    ids, q, values = inputs()
    runtime, generator = owner(), torch.Generator().manual_seed(7)
    before = runtime.state_dict(generator)
    with pytest.raises(ValueError, match="nominal_mask|nominal_q"):
        runtime.prepare_reset(ids, generator, **values, nominal_q=q if invalid_q is None else invalid_q,
            nominal_mask=invalid_mask)
    equal(runtime.state_dict(generator), before)
    assert runtime.pending_reset is None
