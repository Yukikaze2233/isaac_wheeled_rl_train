"""CPU schema and inertia checks for the fixed-carrier Isaac wheel bench."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("isaac_wheel_replay", ROOT / "scripts/replay_wheel_response_isaac.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def run(tmp_path):
    count = 40
    torque = np.zeros((count, 6))
    torque[:, 4:6] = [.6, -.6]
    arrays = {"phase": np.full(count, 2), "segment_id": np.full(count, 131),
        "control_steady_ns": np.arange(count) * 1_000_000,
        "q_api": np.zeros((count, 6)), "dq_api": np.zeros((count, 6)), "tau_frame_api": torque,
        "wheel_velocity_target_api": np.tile([1., -1.], (count, 1))}
    np.savez(tmp_path / "wheel_1khz.npz", **arrays)
    (tmp_path / "trajectory-plan.json").write_text(json.dumps(
        {"segments": [{"id": 131, "label": "left_heldout_multisine", "coast": False}]}))
    return tmp_path


def test_recorded_torque_checks_p_without_reapplying_ratio(run):
    trace = MODULE.load_inputs(run, [131], .6)[131]
    np.testing.assert_allclose(trace["wheel_velocity_target_api"][0], [1., -1.])
    assert trace["frame_pd_error_p99_nm"] == 0.
    with pytest.raises(ValueError, match="does not match"):
        MODULE.load_inputs(run, [131], .2)


def test_nominal_grid_rejects_omitted_time(run):
    with np.load(run / "wheel_1khz.npz") as original:
        data = dict(original)
    data["control_steady_ns"][20:] += 3_000_000
    np.savez(run / "wheel_1khz.npz", **data)
    with pytest.raises(ValueError, match="continuous nominal"):
        MODULE.load_inputs(run, [131], .6)


def test_candidate_armature_is_added_inertia_not_total(tmp_path):
    path = tmp_path / "parameters.json"
    axis = {"effective_total_inertia_kg_m2": .0045, "cad_inertia_kg_m2": .0005,
            "additional_armature_kg_m2": .004, "coulomb_positive_nm": .05,
            "coulomb_negative_nm": .07, "viscous_nm_per_rad_s": .001}
    payload = {"coordinate": "wheel_output", "ratio_15_8_already_applied": True,
               "axes": {"L_joint3": axis, "R_joint3": axis}}
    path.write_text(json.dumps(payload))
    params, _ = MODULE.wheel_parameters(path)
    np.testing.assert_allclose(params["armature"], [.004, .004])
    np.testing.assert_allclose(params["coulomb"], [.06, .06])
    axis["additional_armature_kg_m2"] = .0045
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="existing CAD plus added"):
        MODULE.wheel_parameters(path)


def test_native_inertia_mass_rotation_and_parallel_axis():
    half = np.sqrt(.5)
    value = MODULE.project_native_inertia(np.diag([1., 2., 3.]),
        [3., 0., 0., 0., half, 0., half], 2., [0., 0., 1.])
    assert value == pytest.approx(19.)
    wheel = MODULE.project_native_inertia(np.diag([.000296, .000296, .000575]),
        [0., 0., .020472, 0., 0., 0., 1.], .2, [0., 0., -1.])
    assert wheel == pytest.approx(.000575)


@pytest.mark.parametrize("mode,coast", [("closed_loop", False), ("closed_loop", True), ("torque_input", False)])
def test_device_trace_loop_preserves_control_and_initializes_once(mode, coast):
    torch = pytest.importorskip("torch")
    bench = MODULE.WheelBench.__new__(MODULE.WheelBench)
    bench.torch, bench.args = torch, SimpleNamespace(device="cpu", kp=.6)
    bench.ids, bench.carrier_ids = [0, 1], [2, 3]
    bench.nominal, bench.efforts = torch.zeros((1, 4)), torch.zeros((1, 4))
    q, dq = torch.zeros((1, 4)), torch.zeros((1, 4))
    bench.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=q),
                                                       joint_vel=SimpleNamespace(torch=dq)))
    initializations = []
    def initialize(position, velocity):
        initializations.append(True)
        q[0, :2] = torch.as_tensor(position, dtype=q.dtype)
        dq[0, :2] = torch.as_tensor(velocity, dtype=dq.dtype)
    def step(torque):
        dq[0, :2] += .001 * torque / .0045
        q[0, :2] += .001 * dq[0, :2]
    bench.initialize, bench.step = initialize, step
    target = np.tile([1., -2.], (20, 1))
    recorded = np.tile([.1, -.2], (20, 1))
    _, speed, torque, drift = bench.run_trace(target, recorded, np.zeros(2), np.zeros(2), coast, mode)
    expected_speed, expected_torque, velocity = [], [], np.zeros(2)
    for command, input_torque in zip(target, recorded):
        value = input_torque if mode == "torque_input" else (
            np.zeros(2) if coast else np.clip(.6 * (command - velocity), -4.5, 4.5))
        expected_speed.append(velocity.copy())
        expected_torque.append(value.copy())
        velocity += .001 * value / .0045
    np.testing.assert_allclose(speed, expected_speed, atol=3e-7)
    np.testing.assert_allclose(torque, expected_torque, atol=3e-7)
    np.testing.assert_array_equal(drift, np.zeros(20))
    assert len(initializations) == 1


def test_batch_padding_does_not_replace_recorded_targets():
    cases = [{"target": np.tile([1., 2.], (4, 1)), "recorded": np.tile([.1, .2], (4, 1))},
             {"target": np.array([[3., 4.], [5., 6.]]), "recorded": np.array([[.3, .4], [.5, .6]])}]
    reference, torque, sizes = MODULE.pad_wheel_cases(cases)
    assert reference.shape == (4, 2, 2)
    assert sizes == [4, 2]
    np.testing.assert_array_equal(reference[:2, 1], cases[1]["target"])
    np.testing.assert_allclose(torque[:2, 1], cases[1]["recorded"])
    np.testing.assert_array_equal(reference[2:, 1], [[5., 6.], [5., 6.]])


def test_wheel_batch_keeps_coast_and_tracking_cases_independent():
    torch = pytest.importorskip("torch")
    bench = MODULE.WheelBench.__new__(MODULE.WheelBench)
    bench.torch, bench.args = torch, SimpleNamespace(device="cpu", kp=.6, mode="closed_loop")
    bench.ids, bench.carrier_ids = [0, 1], [2, 3]
    bench.nominal, bench.efforts = torch.zeros((2, 4)), torch.zeros((2, 4))
    q, dq = torch.zeros((2, 4)), torch.zeros((2, 4))
    bench.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=q),
                                                       joint_vel=SimpleNamespace(torch=dq)))
    bench.configure_batch_response = lambda cases: setattr(bench, "cases", cases)
    initializations = []
    def initialize(position, velocity):
        initializations.append(True)
        q[:, :2] = torch.as_tensor(position, dtype=q.dtype)
        dq[:, :2] = torch.as_tensor(velocity, dtype=dq.dtype)
    def step(torque):
        dq[:, :2] += .001 * torque / .0045
        q[:, :2] += .001 * dq[:, :2]
    bench.initialize, bench.step = initialize, step
    cases = [{"coast": coast, "target": np.tile([1., -2.], (length, 1)),
        "recorded": np.zeros((length, 2)), "q_real": np.zeros((length, 2)),
        "dq_real": np.zeros((length, 2))} for coast, length in [(False, 20), (True, 10)]]
    result = bench.run_batch_trace(cases)
    assert len(initializations) == 1
    assert result[0][1].shape == (20, 7)
    assert result[1][1].shape == (10, 7)
    assert result[0][1][-1, 2] > 0.9
    np.testing.assert_array_equal(result[1][1], np.zeros((10, 7)))
