"""Known physical cases for offline wheel identification; no hardware/Kit IO."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wheel_response", ROOT / "scripts/identify_wheel_response.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_known_integral_model_recovers_inertia_and_directional_resistance():
    rng = np.random.default_rng(2810)
    speeds = np.r_[rng.uniform(1, 30, 100), -rng.uniform(1, 30, 100)]
    x = np.column_stack([rng.uniform(-2, 2, 200), .05 * (speeds > 0),
                         -.05 * (speeds < 0), .05 * speeds])
    truth = np.array([.0045, .055, .070, .0004])
    parameters, condition = MODULE.fit_integrals(x, x @ truth)
    np.testing.assert_allclose(parameters, truth, rtol=1e-10, atol=1e-12)
    assert condition < 20


def test_zero_current_dissipates_energy_without_reversing():
    time = np.arange(2001) * .001
    for sign in (-1, 1):
        prediction = MODULE.replay(time, np.zeros_like(time), np.zeros_like(time),
                                   sign * 10., [.0045, .05, .07, .0004], .6, False)
        assert np.all(sign * prediction >= 0)
        assert np.all(np.diff(prediction ** 2) <= 1e-12)
        assert prediction[-1] == 0


def test_known_p_remains_separate_from_viscous_resistance():
    time = np.arange(2001) * .001
    prediction = MODULE.replay(time, np.full_like(time, 10.), np.zeros_like(time),
                               0., [.0045, .05, .07, .001], .6, True)
    equilibrium = (.6 * 10. - .05) / (.6 + .001)
    np.testing.assert_allclose(prediction[-1], equilibrium, atol=1e-10)


def test_mechanical_projection_is_cad_inertia_not_already_added_effective_inertia():
    urdf = ROOT / "model/纯底盘_v6_232mm_stop105/urdf/robot.urdf"
    for link in ("L_link3", "R_link3"):
        np.testing.assert_allclose(MODULE.wheel_cad_inertia(urdf, link), .000575, atol=1e-14)


def test_predeclared_validation_and_coast_never_enter_integral_fit():
    n = 303
    segments = np.repeat([0, 1, 2], 101)
    time = np.arange(n, dtype=np.int64) * 1_000_000
    data = {"segment_id": segments, "t": time * 1e-9,
            "control_steady_ns": time, "feedback_steady_ns": np.tile(time[:, None], (1, 6)),
            "dq_api": np.zeros((n, 6)), "q_api": np.zeros((n, 6)),
            "tau_frame_api": np.zeros((n, 6)), "speed_consistent": np.ones((n, 2), dtype=bool)}
    data["dq_api"][:, 4] = 2.
    plan = [{"id": i, "label": "left_trajectory", "validation": i == 1, "coast": i == 2} for i in range(3)]
    _, _, groups = MODULE.integral_windows(data, plan, 0)
    np.testing.assert_array_equal(groups, [0, 0])
