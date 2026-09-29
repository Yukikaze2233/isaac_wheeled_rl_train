"""CPU checks for the Isaac replay boundary; importing this never starts Kit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("isaac_pair_replay", ROOT / "scripts/replay_identification_isaac.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_nominal_and_candidate_parameters_keep_output_axis_units():
    nominal = MODULE.parameters({})
    np.testing.assert_array_equal(nominal["armature"], [0., 0.])
    candidate = MODULE.parameters({"armature": .004, "damping_hip": .1,
        "damping_knee": .2, "friction_hip": 1.2, "friction_knee": 1.3, "lag": .003})
    np.testing.assert_array_equal(candidate["armature"], [.004, .004])
    np.testing.assert_array_equal(candidate["damping"], [.1, .2])
    np.testing.assert_array_equal(candidate["frictionloss"], [1.2, 1.3])
    np.testing.assert_array_equal(candidate["time_constant_s"], [.003, .003])


@pytest.mark.parametrize("raw", [{"armature": -1.}, {"damping": [1., float("nan")]},
    {"frictionloss": [1., 2., 3.]}, {"gain": 0.}, {"lag": .1, "time_constant_s": .1},
    {"friction_hip": 1.}, {"damping": 1., "damping_hip": 1., "damping_knee": 1.},
    {"motor_lag_typo": .003}])
def test_invalid_or_ambiguous_response_parameters_rejected(raw):
    with pytest.raises(ValueError):
        MODULE.parameters(raw)


def test_lag_exact_step_and_passive_loss_sign():
    params = MODULE.parameters({"time_constant_s": .004, "damping": [.2, .3],
                                "frictionloss": [1., 2.]})
    motor, loss = MODULE.response_effort(np.zeros(2), np.array([10., -10.]),
                                        np.array([1., -1.]), params, .01)
    np.testing.assert_allclose(motor, (1 - np.exp(-.001 / .004)) * np.array([10., -10.]))
    np.testing.assert_allclose(loss, [1.2, -2.3])
    assert np.all(loss * np.array([1., -1.]) >= 0)
    direct, _ = MODULE.response_effort(np.ones(2) * 99, np.array([2., -3.]),
                                       np.zeros(2), MODULE.parameters({}), .01)
    np.testing.assert_allclose(direct, [2., -3.])


def test_initial_joint_mapping_uses_names_not_indices():
    joints = [SimpleNamespace(name="hip", qposadr=[1], dofadr=[0]),
              SimpleNamespace(name="knee", qposadr=[0], dofadr=[1])]
    replay = SimpleNamespace(spec={"nominal_joint_pos": {"hip": 0., "knee": 0., "other": .9}},
                             model=SimpleNamespace(njnt=2, joint=lambda i: joints[i]))
    q, dq = MODULE.map_initial_state(replay, [20., 10.], [1., 2.], ["knee", "other", "hip"])
    np.testing.assert_array_equal(q, [20., .9, 10.])
    np.testing.assert_array_equal(dq, [2., 0., 1.])


def test_pin_transform_uses_sim60_xyzw_quaternion_order():
    half = np.sqrt(.5)
    point = MODULE.point_world([1., 2., 3., 0., 0., half, half], [1., 0., 0.])
    np.testing.assert_allclose(point, [1., 3., 3.], atol=1e-12)


def test_numpy_double_effort_is_written_to_native_float_tensor():
    torch = pytest.importorskip("torch")
    bench = MODULE.IsaacReplay.__new__(MODULE.IsaacReplay)
    bench.torch, bench.args = torch, SimpleNamespace(device="cpu")
    bench.ids, bench.spring_ids, bench.s0 = [0, 1], [2, 3], np.array([.03, .03])
    bench.efforts = torch.zeros((1, 4), dtype=torch.float32)
    bench.replay = SimpleNamespace(curve={"stroke_m": .08, "monomial_coefficients_n": [100., 20., 0., 0.]})
    bench.robot = SimpleNamespace(
        set_joint_effort_target_index=lambda **kw: None, write_data_to_sim=lambda: None,
        update=lambda dt: None, data=SimpleNamespace(applied_torque=SimpleNamespace(torch=bench.efforts)))
    bench.sim = SimpleNamespace(step=lambda **kw: None)
    applied, _, spring = bench.step(np.array([2., -3.]), np.array([.1, -.2]), np.zeros(4))
    np.testing.assert_allclose(applied, [1.9, -2.8], atol=1e-6)
    np.testing.assert_allclose(bench.efforts[0, 2:].numpy(), spring, atol=1e-6)


@pytest.mark.parametrize("friction_model", ["native", "smooth"])
def test_segment_only_initializes_once_and_does_not_double_apply_native_loss(friction_model):
    count = 6
    reference = np.array([[.01, -.02]] * 3 + [[.03, -.04]] * 3)
    replay = SimpleNamespace(
        side="left", inner_limits_deg=[40., 105.], kp=np.array([160., 160.]),
        kd=np.array([2.5, 2.5]), cap=np.array([40., 40.]),
        reference=reference, real_q=np.zeros((count, 2)), real_dq=np.zeros((count, 2)),
        real_torque=np.zeros((count, 2)), api_to_model=np.ones(2),
        a={"tau_cmd_api": np.zeros((count, 2))},
        initial_state=lambda segment: (np.arange(count), np.zeros(2), np.zeros(2), 0.))
    args = SimpleNamespace(max_samples=None, mode="closed_loop", friction_velocity=.01,
                           friction_model=friction_model, log_every=10)
    initializations, requests = [], []
    bench = SimpleNamespace(replay=replay, args=args, ids=[0, 1],
        params=MODULE.parameters({"frictionloss": [1., 2.]}),
        initialize=lambda *state: initializations.append(state),
        snapshot=lambda: (np.zeros(2), np.array([1., -1.]), 104., 0.))

    def step(motor, loss, q):
        requests.append((motor.copy(), loss.copy()))
        return motor - loss, np.zeros(2), np.zeros(2)

    bench.step = step
    score, trace = MODULE.run_segment(bench, 37, SimpleNamespace(add_scalar=lambda *a: None))
    assert len(initializations) == 1
    assert score["samples"] == count
    assert score["knee_stop"]["upper_proximity_fraction"] == 0.
    np.testing.assert_allclose(trace["tau_request"], reference * 160 - [2.5, -2.5])
    assert len(requests) == count
    if friction_model == "native":
        np.testing.assert_array_equal(trace["tau_passive_loss"], np.zeros((count, 2)))
    else:
        np.testing.assert_allclose(trace["tau_passive_loss"], np.tile([1., -2.], (count, 1)))
