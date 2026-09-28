"""Offline replay input/coordinate checks; no controller or GPU is started."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pair_replay_under_test", ROOT / "scripts/identify_pair_dynamics.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Replay = MODULE.Replay


@pytest.fixture
def run(tmp_path):
    count = 6
    arrays = {"phase": np.full(count, 2), "repetition_id": np.zeros(count),
        "q_ref_model": np.tile([.12, -.2], (count, 1)),
        "position_error_model": np.tile([.02, -.01], (count, 1)),
        "dq_api": np.tile([.3, -.4], (count, 1)),
        "tau_frame_api": np.array([[np.nan, np.nan], [1., -2.], [np.nan, 3.],
                                    [4., np.nan], [np.nan, np.nan], [5., 6.]]),
        "tau_cmd_api": np.zeros((count, 2)), "segment_id": np.full(count, 2),
        "tick": np.arange(count), "control_steady_ns": np.arange(count) * 1_000_000}
    np.savez(tmp_path / "response_1khz.npz", **arrays)
    controller = {"side": "left", "control_law": "rl_pd", "control_frequency_hz": 1000,
        "reference_frequency_hz": 50, "pd_kp": [160.] * 4, "pd_kd": [2.5] * 4,
        "max_torque": [40.] * 4, "model_sign": [1., -1., -1., 1.]}
    (tmp_path / "response_1khz.json").write_text(json.dumps({"controller": controller}))
    return tmp_path


@pytest.mark.parametrize("dt", [-.0005, 0., float("nan"), float("inf"), .002, .0003])
def test_invalid_physics_timesteps_rejected(run, dt):
    with pytest.raises(ValueError, match="[Pp]hysics timestep"):
        Replay(run, physics_dt=dt)


@pytest.mark.parametrize("key,value", [("side", "right"), ("control_law", "cascade"),
    ("control_frequency_hz", 200), ("reference_frequency_hz", 1000)])
def test_wrong_recorded_control_protocol_rejected(run, key, value):
    path = run / "response_1khz.json"
    metadata = json.loads(path.read_text())
    metadata["controller"][key] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="1000 Hz with 50 Hz"):
        Replay(run)


def test_invalid_coordinate_sign_rejected(run):
    with pytest.raises(ValueError, match="Coordinate sign"):
        Replay(run, coordinate_sign=0.)


def test_multiple_running_repetitions_rejected(run):
    arrays = dict(np.load(run / "response_1khz.npz"))
    arrays["repetition_id"][-1] = 1
    np.savez(run / "response_1khz.npz", **arrays)
    with pytest.raises(ValueError, match="exactly one running repetition"):
        Replay(run)


def test_system_frames_hold_last_finite_host_command_per_axis(run):
    replay = Replay(run)
    expected = np.array([[0., 0.], [1., 2.], [1., -3.], [4., -3.], [4., -3.], [5., -6.]])
    np.testing.assert_array_equal(replay.real_torque, expected)
    assert replay.missing_frame_samples == [3, 3]
    # No prior host torque is known at the initial system frame: explicit zero
    # initialization is a replay prior, not proof of a device acknowledgement.


def test_v6_axis_reversal_preserves_virtual_work_and_power(run):
    old = Replay(run, coordinate_sign=1.)
    new = Replay(run, coordinate_sign=-1.)
    np.testing.assert_array_equal(new.real_q, -old.real_q)
    np.testing.assert_array_equal(new.reference, -old.reference)
    np.testing.assert_array_equal(new.real_dq, -old.real_dq)
    np.testing.assert_array_equal(new.real_torque, -old.real_torque)
    delta = np.array([.004, -.007])
    np.testing.assert_allclose(new.real_torque @ (-delta), old.real_torque @ delta)
    np.testing.assert_allclose((new.real_torque * new.real_dq).sum(1),
                               (old.real_torque * old.real_dq).sum(1))
    old_pd = np.clip(old.kp * (old.reference - old.real_q) - old.kd * old.real_dq,
                     -old.cap, old.cap)
    new_pd = np.clip(new.kp * (new.reference - new.real_q) - new.kd * new.real_dq,
                     -new.cap, new.cap)
    np.testing.assert_allclose(new_pd, -old_pd)
    # This proves the coordinate transform algebra, not measured zero calibration
    # or equivalence of the old/new CAD inertia and gas-spring geometry.


def clock_replay(times, ticks=None, segments=None):
    replay = Replay.__new__(Replay)
    count = len(times)
    replay.a = {"segment_id": np.full(count, 2) if segments is None else np.array(segments),
                "tick": np.arange(count) if ticks is None else np.array(ticks),
                "control_steady_ns": np.array(times, dtype=np.int64)}
    replay.initial, replay.time_audits = {}, {}
    replay.state_at = lambda sample: (np.zeros(2), np.zeros(2), 0., np.eye(2))
    return replay


@pytest.mark.parametrize("times,ticks", [
    ([0, 0, 1_000_000], None), ([0, -1, 1_000_000], None),
    ([0, 2_100_000, 3_100_000], None),
    ([0, 1_600_000, 3_200_000], None),
    ([0, 1_000_000, 2_000_000], [0, 2, 3]),
])
def test_omitted_ticks_nonmonotonic_and_accumulated_clock_drift_rejected(times, ticks):
    with pytest.raises(ValueError, match="nominal 1 ms replay rejected"):
        clock_replay(times, ticks).initial_state(2)


@pytest.mark.parametrize("times,segments", [([], []), ([0], [2]),
    ([0, 1_000_000, 2_000_000], [2, 3, 2])])
def test_absent_singleton_or_split_segment_rejected(times, segments):
    with pytest.raises(ValueError, match="at least two contiguous samples"):
        clock_replay(times, segments=segments).initial_state(2)


def test_bounded_jitter_reports_grid_approximation_and_reuses_initial_state():
    replay = clock_replay([0, 1_100_000, 2_000_000, 3_050_000])
    first = replay.initial_state(2)
    assert replay.initial_state(2) is first
    assert replay.time_audits[2]["max_nominal_grid_error_s"] == pytest.approx(.0001)
    assert replay.time_audits[2]["min_dt_s"] == pytest.approx(.0009)
