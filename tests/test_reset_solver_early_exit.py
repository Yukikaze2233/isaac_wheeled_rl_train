"""Compare early reset convergence with the frozen fixed-iteration methods."""
from copy import deepcopy
from dataclasses import is_dataclass
import importlib.util
import json
from pathlib import Path
from types import MethodType

import pytest
import torch

from wheeled_tasks.chassis.final_rough_birth import FinalRoughBirthSampler
from wheeled_tasks.chassis.leg_pose_mapping import LegPoseMapper
from wheeled_tasks.chassis.nonsteady_initial_state import FlatBirthSampler


ROOT = Path(__file__).resolve().parents[1]
ORACLE_FILE = ROOT / "tests/fixtures/v7_reset_fixed_iterations.py"
spec = importlib.util.spec_from_file_location("reset_fixed_iteration_oracle", ORACLE_FILE)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


@pytest.fixture(autouse=True)
def isolated_threads_and_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def bound_mapper():
    config = json.loads((ROOT / "configs/v7_formal_20261012/recovery_after10000_056.json").read_text())
    asset = ROOT / config["asset_directory"]
    return LegPoseMapper(json.loads((asset / "model_spec.json").read_text()),
        json.loads((asset / "manifest.json").read_text()),
        json.loads((ROOT / config["flat_episode_runtime"]["branch_table"]).read_text()), "cpu")


def fixed_mapper(mapper):
    value = deepcopy(mapper)
    value.map_requests = MethodType(oracle.fixed_iteration_map_requests, value)
    value.kinematics.solve_passive = MethodType(
        oracle.fixed_iteration_solve_passive, value.kinematics)
    return value


def compare_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_exact(left[key], right[key])
    elif is_dataclass(left):
        assert type(left) is type(right)
        compare_exact(vars(left), vars(right))
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            compare_exact(a, b)
    else:
        assert left == right


def sampled_requests(kind, count=96):
    ids, rng = torch.arange(count), torch.Generator().manual_seed(789)
    if kind == "retained_flat":
        return FlatBirthSampler(count, "cpu").sample(ids, rng)
    yes = torch.ones(count, dtype=torch.bool)
    return FinalRoughBirthSampler(count, "cpu").sample(ids, rng, iteration=0,
        air_allowed=yes, ground_allowed=yes, axis_allowed=~yes).requests


def mapped_arguments(mapper, kind):
    request = sampled_requests("retained_flat" if kind == "retained_flat" else "final")
    args = [request.virtual_leg_length_m.double(), -request.virtual_leg_angle_rad.double(),
            request.wheel_angle_rad.double(), request.control_velocity_rad_s.double()]
    if kind == "endpoints":
        args[0][:48] = mapper.length_bounds_m[:, 0]
        args[0][48:] = mapper.length_bounds_m[:, 1]
        args[3].zero_()
    elif kind == "invalid":
        args[0][:24] = mapper.length_bounds_m[:, 0] - .01
        args[0][24:48] = mapper.length_bounds_m[:, 1] + .01
        args[1][48:60] = torch.nan
        args[2][60:72] = torch.inf
        args[3][72:84] = torch.nan
    return args


@pytest.mark.parametrize("kind", ["retained_flat", "final", "endpoints", "invalid"])
@pytest.mark.parametrize("iterations", [0, 1, 8])
def test_complete_mapping_outputs_and_failure_reasons_match_old_methods(bound_mapper, kind, iterations):
    args = mapped_arguments(bound_mapper, kind)
    original = [value.clone() for value in args]
    rng = torch.get_rng_state().clone()
    old = fixed_mapper(bound_mapper).map_requests(*args, max_iterations=iterations)
    new = bound_mapper.map_requests(*args, max_iterations=iterations)
    compare_exact(old, new)
    for before, after in zip(original, args):
        compare_exact(before, after)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("iterations", [0, 1, 12])
@pytest.mark.parametrize("kind", ["nominal", "perturbed", "nonfinite", "rank_rejected"])
def test_passive_position_result_matches_old_method(bound_mapper, kind, iterations):
    current = deepcopy(bound_mapper.kinematics)
    baseline = deepcopy(current)
    q = bound_mapper._nominal.repeat(24, 1)
    if kind == "perturbed":
        q[:, current.passive_indices] += torch.linspace(-.0003, .0003, 24)[:, None]
    elif kind == "nonfinite":
        q[0, 0], q[1, 3] = torch.nan, torch.inf
    elif kind == "rank_rejected":
        current.rank_rtol = baseline.rank_rtol = 1.1
    controls = q[:, current.control_indices].clone()
    before = q.clone()
    old = oracle.fixed_iteration_solve_passive(baseline, q, controls, max_iterations=iterations)
    new = current.solve_passive(q, controls, max_iterations=iterations)
    compare_exact(old, new)
    compare_exact(q, before)
    compare_exact(new.q[:, current.control_indices], controls)


def count_qr(monkeypatch, function):
    count = 0
    original = torch.linalg.qr

    def qr(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    with monkeypatch.context() as scope:
        scope.setattr(torch.linalg, "qr", qr)
        result = function()
    return count, result


def test_closed_positions_skip_only_repeated_noop_passive_solves(bound_mapper, monkeypatch):
    q = bound_mapper._nominal.repeat(8, 1)
    model = bound_mapper.kinematics
    controls = q[:, model.control_indices]
    old_count, old = count_qr(monkeypatch,
        lambda: oracle.fixed_iteration_solve_passive(model, q, controls))
    new_count, new = count_qr(monkeypatch, lambda: model.solve_passive(q, controls))
    compare_exact(old, new)
    assert old.valid.all() and old_count == 13 and new_count == 2


@pytest.mark.parametrize("rejected", [False, True])
def test_mapping_reduces_repeated_factorizations_without_changing_acceptance(
        bound_mapper, monkeypatch, rejected):
    q = bound_mapper._nominal.repeat(8, 1)
    for side in bound_mapper._side_data:
        q[:, side["indices"]] = side["q"][0, side["indices"]]
    metrics = bound_mapper.metrics(q)
    length = metrics["length_m"].clone()
    if rejected:
        length[:] = bound_mapper.length_bounds_m[:, 0] - .01
    args = (length, metrics["angle_rad"], q[:, bound_mapper.wheel_indices], torch.zeros(8, 6, dtype=q.dtype))
    baseline = fixed_mapper(bound_mapper)
    old_count, old = count_qr(monkeypatch, lambda: baseline.map_requests(*args))
    new_count, new = count_qr(monkeypatch, lambda: bound_mapper.map_requests(*args))
    compare_exact(old, new)
    assert old_count == 22 and new_count < old_count
    assert new.valid.eq(not rejected).all()
    if rejected:
        assert new.reasons["unreachable_length"].all()
        assert torch.isnan(new.q[:, bound_mapper._free]).all()
