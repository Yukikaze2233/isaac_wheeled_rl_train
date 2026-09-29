"""Independent case/metric checks for parallel same-trajectory validation."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("pair_batch", ROOT / "scripts/replay_pair_response_batch_isaac.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def replay_fixture():
    arrays = {37: np.arange(5), 39: np.arange(5, 8)}
    return SimpleNamespace(reference=np.arange(16).reshape(8, 2) * .01,
        real_torque=np.arange(16).reshape(8, 2) * .1,
        initial_state=lambda segment: (arrays[segment], np.ones(2) * segment,
                                       np.zeros(2), 1e-12))


def test_every_selected_segment_has_both_variants_without_filtering():
    rows = MODULE.case_plan(replay_fixture(), [37, 39], MODULE.parameters({"armature": .01}))
    assert [(r["variant"], r["segment"]) for r in rows] == [
        ("nominal", 37), ("nominal", 39), ("candidate", 37), ("candidate", 39)]
    np.testing.assert_array_equal(rows[0]["params"]["armature"], [0., 0.])
    np.testing.assert_array_equal(rows[2]["params"]["armature"], [.01, .01])
    assert [len(row["indices"]) for row in rows] == [5, 3, 5, 3]


def test_padding_only_holds_completed_cases_and_preserves_recorded_samples():
    replay = replay_fixture()
    rows = MODULE.case_plan(replay, [37, 39], MODULE.parameters({}))
    reference, torque = MODULE.padded_inputs(replay, rows)
    assert reference.shape == (5, 4, 2)
    for env, row in enumerate(rows):
        ix = row["indices"]
        np.testing.assert_allclose(reference[:len(ix), env], replay.reference[ix])
        np.testing.assert_allclose(torque[:len(ix), env], replay.real_torque[ix])
    np.testing.assert_allclose(reference[3:, 1], np.tile(replay.reference[7], (2, 1)))


def test_padded_values_never_enter_saved_trace_or_rmse(tmp_path):
    replay = replay_fixture()
    replay.side, replay.inner_limits_deg = "left", [40., 105.]
    replay.real_q, replay.real_dq = np.zeros((8, 2)), np.zeros((8, 2))
    row = MODULE.case_plan(replay, [39], MODULE.parameters({}))[0]
    trace = np.zeros((5, 16))
    trace[:3, 10] = 104.
    trace[3:] = 9999.
    writer = SimpleNamespace(add_scalar=lambda *a: None)
    score = MODULE.save_case(tmp_path, writer, replay, row, trace)
    assert score["samples"] == 3
    assert score["axis_rmse_rad"] == [0., 0.]
    assert score["simulated_inner_range_deg"] == [104., 104.]
    with np.load(tmp_path / "nominal/segment_39.npz") as result:
        assert len(result["q_sim"]) == 3
        np.testing.assert_array_equal(result["indices"], [5, 6, 7])


def test_vector_lag_matches_independent_exact_scalar_updates():
    torch = pytest.importorskip("torch")
    params = MODULE.parameters({"time_constant_s": [.003, .004], "gain": [1., 1.1]})
    tau = torch.tensor(np.stack([np.zeros(2), params["time_constant_s"]]), dtype=torch.float32)
    gain = torch.tensor([[1., 1.], [1., 1.1]], dtype=torch.float32)
    alpha = torch.exp(-.001 / tau.clamp_min(1e-12))
    previous, request = torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[5., 6.], [7., 8.]])
    result = alpha * previous + (1 - alpha) * gain * request
    expected_alpha = np.exp(-.001 / np.maximum(tau.numpy(), 1e-12))
    np.testing.assert_allclose(result.numpy(), expected_alpha * previous.numpy()
                              + (1 - expected_alpha) * gain.numpy() * request.numpy(), rtol=1e-6)
