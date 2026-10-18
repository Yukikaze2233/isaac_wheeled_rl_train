"""Reset-request summaries are observational, current-worker telemetry only."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.flat_episode_runtime import FlatEpisodeRuntime, MECHANICAL_RESTART_VERSION
from wheeled_tasks.chassis.nonsteady_initial_state import CONTROL_JOINT_NAMES


PREFIX = "/source_birth/current_worker/"


class MappingDouble:
    """CPU data fixture; no mechanical or native qualification."""
    dtype = torch.float64

    def __init__(self, refusals=()):
        self.refusals = list(refusals)

    def map_requests(self, length, angle, wheel, velocity):
        valid = torch.ones(len(length), dtype=torch.bool)
        if self.refusals:
            valid[self.refusals.pop(0)] = False
        q = torch.zeros(len(length), 18, dtype=self.dtype)
        q[:, :2], q[:, 2:4] = length, wheel
        dq = torch.zeros_like(q)
        dq[:, :6] = velocity
        return SimpleNamespace(q=q, dq=dq, valid=valid)


class PlacementDouble:
    dtype = torch.float64

    def propose(self, q, position, rpy, velocity, origins, xy, ground):
        pose = torch.zeros(len(q), 7, dtype=self.dtype)
        pose[:, :2], pose[:, 2], pose[:, 6] = xy, ground + position[:, 2], 1.
        return SimpleNamespace(root_pose=pose, root_velocity=velocity.clone(),
            valid=torch.isfinite(q).all(-1), placement_raise_m=torch.zeros(len(q), dtype=self.dtype),
            minimum_shape_clearance_m=torch.full((len(q),), .003, dtype=self.dtype))


def owner(*, refusals=(), count=3):
    identity = {"schema": MECHANICAL_RESTART_VERSION, **{name: char * 64 for name, char in (
        ("asset_manifest_sha256", "a"), ("model_spec_sha256", "b"),
        ("branch_table_sha256", "c"), ("adapter_configuration_sha256", "d"))}}
    return FlatEpisodeRuntime(count, "cpu", MappingDouble(refusals), PlacementDouble(),
        tuple(f"axis{i}" for i in range(18)), birth_rejection={"mode": "abort"}, restart_identity=identity)


def prepare(runtime, ids, generator):
    count = len(ids)
    return runtime.prepare_reset(ids, generator, origins=torch.zeros(count, 3),
        reset_xy_world=torch.arange(count * 2).reshape(count, 2).float(), ground_z_world=torch.zeros(count))


def equal(before, after):
    if isinstance(before, torch.Tensor):
        torch.testing.assert_close(before, after, rtol=0., atol=0., equal_nan=True)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for name in before:
            equal(before[name], after[name])
    elif isinstance(before, (list, tuple)):
        assert type(before) is type(after) and len(before) == len(after)
        for left, right in zip(before, after):
            equal(left, right)
    else:
        assert before == after


def coverage(runtime):
    return {name[len(PREFIX):]: value for name, value in runtime.metrics().items()
            if name.startswith(PREFIX)}


class ReasonedMapper(MappingDouble):
    def map_requests(self, *args):
        result = super().map_requests(*args)
        rejected = ~result.valid
        result.reasons = {"unreachable_length": rejected.clone(), "position_domain_failed": rejected.clone()}
        result.q[rejected] = torch.nan
        return result


class ReasonedPlacement(PlacementDouble):
    def propose(self, *args):
        result = super().propose(*args)
        result.reasons = {"nonfinite_geometry": ~result.valid}
        return result


def test_redraw_proposals_and_overlapping_reasons_are_separate_from_committed_acceptance():
    runtime = FlatEpisodeRuntime(3, "cpu", ReasonedMapper([[1], []]), ReasonedPlacement(),
        tuple(f"axis{i}" for i in range(18)), birth_rejection={"mode": "redraw", "max_attempts": 2})
    generator = torch.Generator().manual_seed(5)
    state = prepare(runtime, [0, 2], generator)
    pending = coverage(runtime)
    assert pending["proposal_count"] == 3
    assert pending["rejected_proposal_count"] == 1
    assert pending["accepted_reset_count"] == 0
    assert not bool(pending["accepted_requested_summary_available"])
    assert pending["accepted_requested_length_left_min_m"] == 0.
    assert pending["rejected_reason_nonexclusive/mapping/unreachable_length"] == 1
    assert pending["rejected_reason_nonexclusive/mapping/position_domain_failed"] == 1
    assert pending["rejected_reason_nonexclusive/placement/nonfinite_geometry"] == 1
    assert pending["unclassified_rejected_proposal_count"] == 0
    runtime.commit_reset(state, torch.zeros(2), generator, 0)
    committed = coverage(runtime)
    assert committed["accepted_reset_count"] == 2
    assert committed["accepted_ground_true_reset_count"] == state.requests.ground_birth.long().sum()
    assert committed["accepted_ground_false_reset_count"] == (~state.requests.ground_birth).long().sum()
    for index, side in enumerate(("left", "right")):
        lengths = state.requests.virtual_leg_length_m[:, index]
        assert committed[f"accepted_requested_length_{side}_min_m"] == lengths.min()
        assert committed[f"accepted_requested_length_{side}_max_m"] == lengths.max()
    for index, name in enumerate(CONTROL_JOINT_NAMES):
        assert committed[f"accepted_requested_velocity_abs_max_rad_s/{name}"] == (
            state.requests.control_velocity_rad_s[:, index].abs().max())
    assert all(isinstance(value, torch.Tensor) and value.ndim == 0 for value in committed.values())


def test_aborted_partial_acceptance_is_not_committed_and_missing_reasons_are_counted():
    runtime = owner(refusals=[[1]])
    with pytest.raises(RuntimeError, match="rejected lanes"):
        prepare(runtime, [0, 2], torch.Generator().manual_seed(2))
    metrics = coverage(runtime)
    assert runtime.pending_reset is None
    assert metrics["proposal_count"] == 2
    assert metrics["rejected_proposal_count"] == 1
    assert metrics["unclassified_rejected_proposal_count"] == 1
    assert metrics["accepted_reset_count"] == 0
    assert not bool(metrics["accepted_requested_summary_available"])
    for side in ("left", "right"):
        assert metrics[f"accepted_requested_length_{side}_min_m"] == 0.
        assert metrics[f"accepted_requested_length_{side}_max_m"] == 0.


def test_metrics_are_detached_and_do_not_change_pending_state_control_masks_or_rng():
    runtime, generator = owner(count=4), torch.Generator().manual_seed(19)
    state = prepare(runtime, [0, 2], generator)
    runtime.commit_reset(state, torch.zeros(2), generator, 0)
    before = runtime.state_dict(generator)
    diagnostic = runtime.diagnostics()
    global_rng = torch.get_rng_state().clone()
    observed = coverage(runtime)
    for value in observed.values():
        value.zero_()
    assert coverage(runtime)["accepted_reset_count"] == 2
    equal(runtime.state_dict(generator), before)
    equal(runtime.diagnostics(), diagnostic)
    assert torch.equal(torch.get_rng_state(), global_rng)


class NominalMapper(MappingDouble):
    kinematics = SimpleNamespace(residual=lambda q: torch.zeros(len(q), 18, dtype=q.dtype))

    @staticmethod
    def metrics(q):
        return {"length_m": torch.full((len(q), 2), .25, dtype=q.dtype),
                "angle_rad": torch.zeros(len(q), 2, dtype=q.dtype)}


def test_nominal_birth_is_ground_false_with_zero_requested_velocity_not_a_source_distribution_claim():
    names = (*CONTROL_JOINT_NAMES, *(f"axis{i}" for i in range(6, 18)))
    runtime = FlatEpisodeRuntime(3, "cpu", NominalMapper(), PlacementDouble(), names,
                                birth_rejection={"mode": "abort"})
    generator = torch.Generator().manual_seed(4)
    state = runtime.prepare_reset([0, 2], generator, origins=torch.zeros(2, 3),
        reset_xy_world=torch.zeros(2, 2), ground_z_world=torch.zeros(2),
        nominal_q=torch.zeros(2, 18, dtype=torch.float64))
    runtime.commit_reset(state, torch.zeros(2), generator, 0)
    metrics = coverage(runtime)
    assert metrics["accepted_reset_count"] == metrics["accepted_ground_false_reset_count"] == 2
    assert metrics["accepted_ground_true_reset_count"] == 0
    assert metrics["proposal_count"] == 2 and metrics["rejected_proposal_count"] == 0
    assert metrics["accepted_requested_length_left_min_m"] == .25
    assert all(metrics[f"accepted_requested_velocity_abs_max_rad_s/{name}"] == 0.
               for name in CONTROL_JOINT_NAMES)


@pytest.mark.parametrize("pending", [False, True])
def test_old_owned_state_schema_remains_readable_without_restoring_worker_only_metrics(pending):
    runtime, generator = owner(count=4), torch.Generator().manual_seed(20)
    state = prepare(runtime, [0, 1], generator)
    if not pending:
        runtime.commit_reset(state, torch.zeros(2), generator, 0)
    legacy = deepcopy(runtime.state_dict(generator))
    assert set(legacy) == {"schema", "identity", "commands", "tensors", "pending_reset"}
    assert set(legacy["tensors"]) == {"episode_id", "birth_proposal_count", "birth_rejection_count",
        "placement_raise_m", "minimum_shape_clearance_m", "birth_source_angle_rad", "birth_own_angle_rad"}
    assert not any("coverage" in name or "current_worker" in name for name in legacy["identity"])
    resumed, new_generator = owner(count=4), torch.Generator().manual_seed(21)
    resumed.load_state_dict(legacy, new_generator)
    equal(resumed.state_dict(new_generator), legacy)
    metrics = coverage(resumed)
    assert metrics["proposal_count"] == metrics["accepted_reset_count"] == 0
    assert resumed.birth_proposal_count.sum() == 2
    assert not bool(metrics["accepted_requested_summary_available"])
    if pending:
        resumed.commit_reset(resumed.pending_reset, torch.zeros(2), new_generator, 0)
        metrics = coverage(resumed)
        assert metrics["accepted_reset_count"] == 2
        assert metrics["proposal_count"] == 0
