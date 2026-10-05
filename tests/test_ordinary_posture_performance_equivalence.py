"""Pair the performance candidate with verbatim 4ba helper/native-push hooks.

The fixed source fixtures work in a clean archive without a Git checkout. They
come from 4ba339299c203cf2aba77183ec81f2f017dc9bf9; only these two functions
change in this candidate. The remaining real helper handles ownership, ramps,
birth sampling and push evidence in both executions. Native buffers below are
transport doubles, so these tests do not claim Isaac/PhysX qualification.
Empty-call cache equivalence uses the online invariants: eligibility is static
and sampled timers only change through sample_intervals. Both the initial zero
cache and a previously populated cache are exercised without external mutation.

CPU runs by default. Set WHEELED_TEST_EQUIVALENCE_CUDA=1 to also run the tiny
Torch CUDA pairs when the GPU is available; this never starts Isaac.
"""
import ast
from copy import deepcopy
from hashlib import sha256
import math
import os
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from wheeled_tasks.chassis.ordinary_posture_curriculum import (
    OrdinaryPostureCurriculum, posture_settings)
from wheeled_tasks.chassis.training_schedule import stage_actor_updates


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "4ba339299c203cf2aba77183ec81f2f017dc9bf9"
BASELINE_SAMPLE = '''def sample_intervals(self, ids, *, generator):
    ids = self._ids(ids)
    lo, hi = self.settings["push"]["interval_seconds"]
    intervals = lo + (hi - lo) * torch.rand(len(ids), device=self.device, generator=generator)
    self.timer_samples[ids] = intervals
    timers = self.timer_samples[self.eligible]
    if timers.numel():
        self.timer_metrics = {"push_clock_min_s": float(timers.amin()),
            "push_clock_max_s": float(timers.amax()),
            "push_clock_std_s": float(timers.std(unbiased=False))}
    return intervals
'''
BASELINE_PUSH = '''def apply_push(self, ids):
    velocity_w = wp.to_torch(self.robot.root_view.get_root_velocities())[ids].clone()
    posture = getattr(self, "ordinary_posture", None)
    before = velocity_w.clone() if posture is not None and posture.enabled else None
    angle = self.random(len(ids)) * math.tau
    cap = self.skills.push_speed[ids] if self.skills is not None else self.stage_cfg["push_max"]
    if posture is not None and posture.enabled:
        _, _, ordinary_cap = posture.amplitudes(stage_actor_updates(self))
        cap = torch.as_tensor(cap, device=self.device).expand(len(ids)).clone()
        cap[posture.eligible[ids]] = ordinary_cap
    amplitude = self.random(len(ids)) * cap
    velocity_w[:, :2] += torch.stack((angle.cos(), angle.sin()), -1) * amplitude[:, None]
    self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity_w, env_ids=ids, full_data=False)
    self.push_clock[ids] = 3 + 2 * self.random(len(ids))
    if posture is not None and posture.enabled:
        ordinary = ids[posture.eligible[ids]]
        self.push_clock[ordinary] = posture.sample_intervals(ordinary, generator=self.generator)
        after = wp.to_torch(self.robot.root_view.get_root_velocities())[ids]
        if not torch.allclose(after[posture.eligible[ids]], velocity_w[posture.eligible[ids]],
                atol=1e-5, rtol=1e-5):
            raise RuntimeError("Ordinary posture velocity push did not match native readback")
        posture.record_push(ids, before, after)
'''


def compile_hook(source, name):
    namespace = {"torch": torch, "math": math, "wp": NS(to_torch=lambda value: value),
        "stage_actor_updates": stage_actor_updates}
    exec(compile(source, "<" + BASELINE_COMMIT + ">", "exec"), namespace)
    return namespace[name]


baseline_sample = compile_hook(BASELINE_SAMPLE, "sample_intervals")
baseline_push = compile_hook(BASELINE_PUSH, "apply_push")
env_source = ROOT / "src/wheeled_tasks/chassis/env.py"
env_class = next(node for node in ast.parse(env_source.read_text()).body
    if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
push_node = next(node for node in env_class.body
    if isinstance(node, ast.FunctionDef) and node.name == "apply_push")
candidate_push = compile_hook(ast.fix_missing_locations(ast.Module(
    body=[deepcopy(push_node)], type_ignores=[])), "apply_push")


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("WHEELED_TEST_EQUIVALENCE_CUDA") != "1",
    reason="CUDA micro-pairs require explicit idle-GPU opt-in"))])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device("cuda", torch.cuda.current_device()) if request.param == "cuda" else torch.device("cpu")


def generator(device, seed=971):
    return torch.Generator(device=device).manual_seed(seed)


def helper(device, eligible, *, baseline=False, enabled=True):
    result = OrdinaryPostureCurriculum(len(eligible), device,
        posture_settings(100, enabled=enabled), torch.tensor(eligible, device=device, dtype=torch.bool))
    if baseline:
        result.sample_intervals = MethodType(baseline_sample, result)
    return result


def trace_draws(function, *args, **kwargs):
    calls = []
    rand = torch.rand
    def observed_rand(*dimensions, **options):
        shape = dimensions[0] if len(dimensions) == 1 else dimensions
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        calls.append(shape)
        return rand(*dimensions, **options)
    with patch.object(torch, "rand", observed_rand):
        value = function(*args, **kwargs)
    return value, calls


def assert_generators_equal(left, right, device):
    assert torch.equal(left.get_state(), right.get_state())
    next_left, next_right = generator(device), generator(device)
    next_left.set_state(left.get_state()); next_right.set_state(right.get_state())
    assert torch.equal(torch.rand(13, device=device, generator=next_left),
        torch.rand(13, device=device, generator=next_right))


def assert_helpers_equal(left, right):
    # The baseline has one instance-bound method; all actual state is compared.
    state_left = {key: value for key, value in vars(left).items() if key != "sample_intervals"}
    assert state_left.keys() == vars(right).keys()
    for key, value in state_left.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, getattr(right, key)), key
        else:
            assert value == getattr(right, key), key


def test_baseline_fixtures_are_the_original_4ba_functions():
    assert sha256(BASELINE_SAMPLE.encode()).hexdigest() == (
        "be4e057f70caf968d95e722e50265329ced4320b022b4a2c57a6735fb2eea4ba")
    assert sha256(BASELINE_PUSH.encode()).hexdigest() == (
        "37a83b0c87172ff3ff33a852c403e66543858492f5723d30cf30b4dabc8107c5")


@pytest.mark.parametrize("eligible", [
    [False] * 7, [False, False, False, True, False, False, False],
    [False, True, False, True, False, True, False]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_interval_empty_partial_and_full_updates_preserve_rng_and_all_lane_statistics(device, eligible, dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        left, right = helper(device, eligible, baseline=True), helper(device, eligible)
        left_gen, right_gen = generator(device), generator(device)
        global_before = (torch.get_rng_state() if device.type == "cpu"
            else torch.cuda.get_rng_state(device)).clone()
        # Empty first call, nontrivial cached timers, non-contiguous partial
        # updates and noneligible-only updates all occur in this sequence.
        for values in ([], list(range(7)), [], [5, 1, 3], [0, 6], []):
            ids = torch.tensor(values, device=device, dtype=torch.long)
            actual_left, draws_left = trace_draws(left.sample_intervals, ids, generator=left_gen)
            actual_right, draws_right = trace_draws(right.sample_intervals, ids, generator=right_gen)
            assert actual_left.shape == actual_right.shape == (len(values),)
            assert actual_left.dtype == actual_right.dtype == dtype
            assert actual_left.device == actual_right.device == device
            assert torch.equal(actual_left, actual_right)
            assert draws_left == draws_right == [(len(values),)]  # Keeps rand(0).
            assert_helpers_equal(left, right)
            assert_generators_equal(left_gen, right_gen, device)
        global_after = torch.get_rng_state() if device.type == "cpu" else torch.cuda.get_rng_state(device)
        assert torch.equal(global_before, global_after)
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize("eligible_count", [577, 3928, 4096])
def test_population_size_statistics_keep_the_complete_eligible_sample_bank(device, eligible_count):
    eligible = [False] * 4096
    for index in range(eligible_count):
        eligible[index * 7 % 4096] = True
    left, right = helper(device, eligible, baseline=True), helper(device, eligible)
    left_gen, right_gen = generator(device), generator(device)
    for ids in (torch.arange(4096, device=device),
            torch.tensor([4079, 45, 912, 6], device=device), torch.empty(0, device=device, dtype=torch.long)):
        actual_left = left.sample_intervals(ids, generator=left_gen)
        actual_right = right.sample_intervals(ids, generator=right_gen)
        assert torch.equal(actual_left, actual_right)
        assert_helpers_equal(left, right)
        assert_generators_equal(left_gen, right_gen, device)
    assert right.timer_metrics["push_clock_min_s"] >= 5.
    assert right.timer_metrics["push_clock_max_s"] < 10.
    assert right.timer_metrics["push_clock_std_s"] > 1.


class NativeBuffers:
    def __init__(self, device, *, drop_lane=None):
        self.velocities = torch.arange(48, device=device, dtype=torch.float32).reshape(8, 6) / 100.
        self.reads, self.writes = 0, []
        self.drop_lane = drop_lane
        self.root_view = NS(get_root_velocities=self.get_root_velocities)

    def get_root_velocities(self):
        self.reads += 1
        return self.velocities

    def write_root_com_velocity_to_sim_index(self, *, root_velocity, env_ids, full_data):
        self.writes.append((root_velocity.clone(), env_ids.clone(), full_data))
        keep = torch.ones(len(env_ids), device=env_ids.device, dtype=torch.bool)
        if self.drop_lane is not None:
            keep &= env_ids != self.drop_lane
        self.velocities[env_ids[keep]] = root_velocity[keep]


def push_env(device, *, baseline=False, case="mixed", drop_lane=None):
    enabled = case != "disabled"
    eligible = ([False] * 8 if case == "zero_eligible" else
        [False, True, False, False, True, False, True, False])
    lesson = helper(device, eligible, baseline=baseline, enabled=enabled)
    gen = generator(device)
    lesson.reset(torch.arange(8, device=device), 2100, generator=gen)
    lesson.sample_intervals(torch.arange(8, device=device), generator=gen)
    robot = NativeBuffers(device, drop_lane=drop_lane)
    env = NS(ordinary_posture=lesson, device=device, cfg={"task_campaign": {"domain": "recovery"}},
        task_actor_update=2100, stage_actor_update=2100, stage_cfg={"push_max": .13},
        skills=NS(push_speed=torch.tensor([0., .12, .2, 0., .5, .07, 0., .13], device=device)),
        push_clock=torch.linspace(.2, 4., 8, device=device), generator=gen,
        random=lambda count: torch.rand(count, device=device, generator=gen), robot=robot)
    if case == "absent":
        del env.ordinary_posture
    if case == "scalar_cap":
        env.skills = None
    return env, lesson


def assert_push_state_equal(left, right, left_helper, right_helper, device):
    assert torch.equal(left.robot.velocities, right.robot.velocities)
    assert torch.equal(left.push_clock, right.push_clock)
    assert len(left.robot.writes) == len(right.robot.writes) == 1
    for expected, actual in zip(left.robot.writes, right.robot.writes):
        assert torch.equal(expected[0], actual[0])
        assert torch.equal(expected[1], actual[1])
        assert expected[2] is actual[2] is False
    assert_helpers_equal(left_helper, right_helper)
    assert_generators_equal(left.generator, right.generator, device)


@pytest.mark.parametrize("case, values", [
    ("empty", []), ("legacy_only", [7, 0, 3]), ("legacy_zero_cap", [0, 3]),
    ("ordinary_only", [6, 1, 4]), ("mixed", [7, 4, 0, 1, 6]),
    ("zero_eligible", [7, 4, 0, 1, 6]), ("disabled", [7, 4, 0, 1, 6]),
    ("absent", [7, 4, 0, 1, 6]), ("scalar_cap", [7, 4, 0, 1, 6])])
def test_push_preserves_writes_rng_clocks_and_evidence_with_fewer_empty_diagnostics(device, case, values):
    left, left_helper = push_env(device, baseline=True, case=case)
    right, right_helper = push_env(device, case=case)
    ids = torch.tensor(values, device=device, dtype=torch.long)
    _, draws_left = trace_draws(baseline_push, left, ids)
    _, draws_right = trace_draws(candidate_push, right, ids)
    active = case not in ("disabled", "absent")
    ordinary_count = int(left_helper.eligible[ids].sum()) if active else 0
    expected_draws = [(len(values),)] * 3 + ([(ordinary_count,)] if active else [])
    assert draws_left == draws_right == expected_draws
    assert_push_state_equal(left, right, left_helper, right_helper, device)
    assert left.robot.reads == (2 if active else 1)
    assert right.robot.reads == (2 if ordinary_count else 1)
    assert left_helper.push_events == right_helper.push_events == ordinary_count
    if ordinary_count:
        assert right_helper.push_velocity_delta_max_m_s > 0.
    if case == "legacy_zero_cap":
        # Zero-cap legacy work still draws, writes and updates its original clock.
        assert torch.equal(right.robot.writes[0][0], torch.arange(
            48, device=device, dtype=torch.float32).reshape(8, 6)[ids] / 100.)


@pytest.mark.parametrize("values", [[6, 1, 4], [7, 4, 0, 1, 6]])
def test_ordinary_and_mixed_push_still_reject_native_write_omission(device, values):
    # A nonfirst local row is omitted, so mixed/global-ID mapping is exercised.
    left, left_helper = push_env(device, baseline=True, drop_lane=1)
    right, right_helper = push_env(device, drop_lane=1)
    ids = torch.tensor(values, device=device, dtype=torch.long)
    for hook, env in ((baseline_push, left), (candidate_push, right)):
        with pytest.raises(RuntimeError, match="native readback"):
            hook(env, ids)
    assert_push_state_equal(left, right, left_helper, right_helper, device)
    assert left.robot.reads == right.robot.reads == 2
    assert left_helper.push_events == right_helper.push_events == 0
