"""CPU equivalence of the two host-reporting changes, without loading Isaac.

The reference is the exact ordinary-stair candidate without the host batching,
not a working-tree copy or a
reimplementation.  Only the actual finite/reporting statements are executed;
an AST comparison separately pins every other statement in ChassisEnv.step.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.v5_control import V5Control

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "src/wheeled_tasks/chassis/env.py"
PARENT = "741e13038a1f31dc312ffafe07763747e511b80b"
PARENT_ENV_SHA256 = "a6dd2838e944d743b99184f6b04c5af588f415b3deb512a0cc006f24960b9d63"


def _step(source):
    module = ast.parse(source)
    chassis = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ChassisEnv")
    return next(n for n in chassis.body if isinstance(n, ast.FunctionDef) and n.name == "step")


def _blocks(step):
    finite_end = next(i for i, node in enumerate(step.body) if isinstance(node, ast.If)
                      and any(isinstance(n, ast.Constant) and n.value == "Nonfinite physical transition"
                              for n in ast.walk(node)))
    finite_start = finite_end
    if finite_end and isinstance(step.body[finite_end - 1], ast.Assign):
        if any(isinstance(n, ast.Name) and n.id == "finite"
               for n in step.body[finite_end - 1].targets):
            finite_start -= 1
    count_end = next(i for i, node in enumerate(step.body) if isinstance(node, ast.Assign)
                     and any(isinstance(n, ast.Name) and n.id == "extras" for n in node.targets))
    count_start = next(i for i, node in enumerate(step.body[:count_end])
                       if (isinstance(node, ast.For)
                           and any(isinstance(n, ast.Attribute) and n.attr == "termination_counts"
                                   for n in ast.walk(node)))
                       or (isinstance(node, ast.Assign)
                           and any(isinstance(n, ast.Name) and n.id == "counts" for n in node.targets)))
    return (finite_start, finite_end + 1), (count_start, count_end)


def _compile(nodes, parameters):
    definition = ast.FunctionDef(name="execute", args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg=p) for p in parameters], vararg=None,
        kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=copy.deepcopy(nodes), decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(ENV), "exec"), namespace)
    return namespace["execute"]


@pytest.fixture(scope="module")
def versions():
    reference = subprocess.run(["git", "show", f"{PARENT}:src/wheeled_tasks/chassis/env.py"],
                               cwd=ROOT, check=True, capture_output=True).stdout
    assert hashlib.sha256(reference).hexdigest() == PARENT_ENV_SHA256
    return [_step(reference), _step(ENV.read_bytes())]


@pytest.fixture(scope="module")
def finite_checks(versions):
    return [_compile(s.body[slice(*_blocks(s)[0])],
                     ("self", "velocity", "omega", "height", "q", "dq")) for s in versions]


@pytest.fixture(scope="module")
def report_counts(versions):
    return [_compile(s.body[slice(*_blocks(s)[1])],
                     ("self", "reasons", "success_event", "timeouts")) for s in versions]


def test_control_reset_rng_and_physics_statements_are_unchanged(versions):
    normalized = []
    for step in versions:
        clone = copy.deepcopy(step)
        for start, end in reversed(_blocks(step)):
            clone.body[start:end] = [ast.Pass()]
        normalized.append(ast.dump(clone, include_attributes=False))
    # This also protects closure checks, spring accounting and their positions.
    assert normalized[0] == normalized[1]


def _physical_inputs(dtype, rows):
    backing = torch.arange(rows * 12, dtype=dtype).reshape(rows, 12)
    return (backing[:, ::2], backing[:, 1::2], backing[:, 0],
            backing[:, ::2], backing[:, 1::2], backing[:, 0::3])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [0, 1, 73])
def test_finite_strided_and_empty_transitions(finite_checks, dtype, rows):
    values = _physical_inputs(dtype, rows)
    before = [value.clone() for value in values]
    for check in finite_checks:
        assert check(SimpleNamespace(contact_force=values[-1]), *values[:-1]) is None
    for actual, expected in zip(values, before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("axis", range(6))
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_each_nonfinite_transition_preserves_runtime_error(finite_checks, axis, invalid, dtype):
    values = list(_physical_inputs(dtype, 3))
    # Give each input its own storage so all six physical fields are covered.
    values = [value.clone() for value in values]
    values[axis].reshape(-1)[-1] = invalid
    for check in finite_checks:
        with pytest.raises(RuntimeError) as error:
            check(SimpleNamespace(contact_force=values[-1]), *values[:-1])
        assert error.value.args == ("Nonfinite physical transition",)


def test_finite_check_retrieves_one_boolean(finite_checks, monkeypatch):
    original = torch.Tensor.__bool__
    calls = []

    def counted(tensor):
        calls.append(tensor.shape)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted)
    values = _physical_inputs(torch.float32, 73)
    for check, expected in zip(finite_checks, [6, 1]):
        calls.clear()
        check(SimpleNamespace(contact_force=values[-1]), *values[:-1])
        assert len(calls) == expected


def _report_state(names):
    return SimpleNamespace(termination_counts={name: 2 ** 63 + i for i, name in enumerate(names)},
                           success_count=2 ** 63 + 31, timeout_count=2 ** 63 + 41)


@pytest.mark.parametrize("rows", [0, 1, 73, 4096, 20000])
@pytest.mark.parametrize("pattern", ["none", "all", "overlap"])
def test_overlapping_reasons_success_and_timeout_count_exactly(report_counts, rows, pattern):
    indices = torch.arange(rows)
    mask = torch.zeros(rows, dtype=torch.bool) if pattern == "none" else torch.ones(rows, dtype=torch.bool)
    if pattern == "overlap":
        mask = indices.remainder(3) == 0
    # Include the dynamically added recovery reasons and preserve insertion order.
    reasons = {"fall": mask, "nonwheel_contact": ~mask, "knee": mask,
               "boundary": indices.remainder(2) == 0, "takeoff_timeout": mask,
               "recovery_timeout": ~mask, "recovery_failure": mask}
    reasons["fall"] = mask  # Overwriting an existing key must not move it.
    success = indices.remainder(5) == 0
    timeouts = indices.remainder(7) == 0
    inputs = [value.clone() for value in (*reasons.values(), success, timeouts)]
    states = [_report_state(reasons), _report_state(reasons)]
    for execute, state in zip(report_counts, states):
        for _ in range(3):
            execute(state, reasons, success, timeouts)
    assert vars(states[0]) == vars(states[1])
    assert list(states[1].termination_counts) == list(reasons)
    assert all(type(count) is int for count in states[1].termination_counts.values())
    assert type(states[1].success_count) is type(states[1].timeout_count) is int
    for actual, expected in zip((*reasons.values(), success, timeouts), inputs):
        assert torch.equal(actual, expected)


def test_empty_reason_registry_keeps_success_timeout(report_counts):
    states = [_report_state(()), _report_state(())]
    for execute, state in zip(report_counts, states):
        execute(state, {}, torch.tensor([True, False]), torch.tensor([True, True]))
    assert vars(states[0]) == vars(states[1])
    assert states[1].success_count == 2 ** 63 + 32
    assert states[1].timeout_count == 2 ** 63 + 43


def test_integer_reductions_do_not_round_above_float_exact_range(report_counts):
    # The physical inputs are boolean masks.  Int64 sums here additionally detect
    # an accidental float intermediate, which ordinary environment sizes cannot.
    reasons = {"fall": torch.tensor([2 ** 53 + 1, 2], dtype=torch.int64),
               "knee": torch.tensor([2 ** 53 + 3, 4], dtype=torch.int64)}
    states = [_report_state(reasons), _report_state(reasons)]
    success = torch.tensor([2 ** 53 + 5], dtype=torch.int64)
    timeout = torch.tensor([2 ** 53 + 7], dtype=torch.int64)
    for execute, state in zip(report_counts, states):
        execute(state, reasons, success, timeout)
    assert vars(states[0]) == vars(states[1])
    assert states[1].termination_counts["fall"] == 2 ** 63 + 2 ** 53 + 3
    assert states[1].termination_counts["knee"] == 2 ** 63 + 2 ** 53 + 8
    assert states[1].success_count == 2 ** 63 + 31 + 2 ** 53 + 5


def test_missing_counter_preserves_assignment_order_and_exception(report_counts):
    reasons = {"fall": torch.tensor([True]), "missing": torch.tensor([False]),
               "knee": torch.tensor([True])}
    states = [_report_state(("fall", "knee")), _report_state(("fall", "knee"))]
    for execute, state in zip(report_counts, states):
        with pytest.raises(KeyError) as error:
            execute(state, reasons, torch.tensor([True]), torch.tensor([True]))
        assert error.value.args == ("missing",)
    assert vars(states[0]) == vars(states[1])
    assert states[1].termination_counts["fall"] == 2 ** 63 + 1
    assert states[1].termination_counts["knee"] == 2 ** 63 + 1
    assert states[1].success_count == 2 ** 63 + 31


def _controller():
    read = lambda name: json.loads((ROOT / name).read_text())
    bundle = "model/纯底盘_v5/urdf/"
    settings = read("contracts/v5_foundation_v1.json")["v5_control"]
    settings.update(leg_kp=160., leg_kd=2.5)
    return V5Control(read(bundle + "manifest.json"), read(bundle + "model_spec.json"),
                     read(bundle + "fit_10mpa.json"), read("contracts/own_v40_v2.json"), settings, "cpu")


def test_reporting_preserves_real_pd_outputs_inputs_and_rng(finite_checks, report_counts):
    results = []
    for finite, report in zip(finite_checks, report_counts):
        control = _controller()
        generator = torch.Generator().manual_seed(6239)
        state = _report_state(("fall", "boundary"))
        outputs = []
        for tick in range(12):
            q = torch.randn(17, 6, generator=generator)
            dq = torch.randn(17, 6, generator=generator) * (1 + tick)
            legs = torch.randn(17, 4, generator=generator)
            wheels = torch.randn(17, 2, generator=generator) * 20
            torque = control.motor_efforts(q, dq, legs, wheels)
            tensors = (torque, control.requested_motor_effort, control.current_motor_bounds)
            before = [value.clone() for value in (*tensors, q, dq)]
            rng_before = generator.get_state().clone()
            finite(SimpleNamespace(contact_force=torch.zeros(17, 3)), dq[:, :3], dq[:, 3:],
                   q[:, 0], q, dq)
            report(state, {"fall": q[:, 0] > 0, "boundary": dq[:, 0] > 0},
                   q[:, 1] > 0, dq[:, 1] > 0)
            assert torch.equal(generator.get_state(), rng_before)
            for actual, expected in zip((*tensors, q, dq), before):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            outputs.append(tuple(value.clone() for value in tensors))
        results.append((outputs, vars(state), generator.get_state()))
    for left, right in zip(results[0][0], results[1][0]):
        for actual, expected in zip(left, right):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    assert results[0][1] == results[1][1]
    assert torch.equal(results[0][2], results[1][2])
