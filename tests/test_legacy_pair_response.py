"""Tests for model-blind legacy selection and explicit input semantics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from validate_legacy_pair_response import LegacyReplay, choose_windows


def telemetry():
    # Each bin/direction has 1000 samples, enough for four disjoint 201-point windows.
    beta = np.repeat([80., 80., 90., 90., 100., 100.], 1000)
    relative = np.repeat([.2, -.2, .2, -.2, .2, -.2], 1000)
    n = len(beta)
    time = 1_000_000_000 + np.arange(n, dtype=np.int64) * 1_000_000
    return {
        "control_steady_ns": time,
        "feedback_steady_ns": np.repeat((time - 100_000)[:, None], 2, axis=1),
        "actuation_scope": np.ones(n),
        "tau_cmd_api": np.ones((n, 2)),
        "dq_api": np.column_stack([np.zeros(n), -relative]),
        "segment_id": np.repeat(np.arange(6), 1000),
        "tick": np.arange(n),
    }, beta


def test_full_horizon_both_directions_and_nonoverlap():
    arrays, beta = telemetry()
    windows, quotas = choose_windows(arrays, beta)
    assert len(windows) == 24
    assert all(q["selected"] == 4 for q in quotas)
    for i, w in enumerate(windows):
        assert w["stop"] - w["start"] == 201
        assert w["elapsed_s"] == pytest.approx(.200)
        assert np.all(beta[w["start"]:w["stop"]] >= w["bin_deg"][0])
        assert np.all(beta[w["start"]:w["stop"]] < w["bin_deg"][1])
        for v in windows[i + 1:]:
            assert w["stop"] <= v["start"] or v["stop"] <= w["start"]


@pytest.mark.parametrize("kind", ["scope", "missing_torque", "stale", "gaps"])
def test_unconfirmed_missing_stale_or_broken_windows_are_excluded(kind):
    arrays, beta = telemetry()
    if kind == "scope":
        arrays["actuation_scope"][:] = 0
    elif kind == "missing_torque":
        arrays["tau_cmd_api"][:] = np.nan
    elif kind == "stale":
        arrays["feedback_steady_ns"][:] -= 10_000_000
    else:
        arrays["control_steady_ns"] += (np.arange(len(beta)) // 100) * 3_000_000
        arrays["feedback_steady_ns"] = np.repeat(
            (arrays["control_steady_ns"] - 100_000)[:, None], 2, axis=1
        )
    windows, quotas = choose_windows(arrays, beta)
    assert windows == []
    assert all(q["selected"] == 0 for q in quotas)


def test_no_hidden_pd_or_claimed_frame_input():
    replay = object.__new__(LegacyReplay)
    for mode in ["closed_loop", "torque_input", "mit_frame"]:
        with pytest.raises(ValueError, match="cannot be replayed"):
            replay.replay(0, {}, mode=mode)


def test_shortage_is_reported_without_relaxing_selection():
    arrays, beta = telemetry()
    arrays["actuation_scope"][5000:] = 0
    windows, quotas = choose_windows(arrays, beta)
    assert len(windows) == 20
    assert quotas[-1] == {"bin_deg": [95., 104.], "sign": -1, "requested": 4, "selected": 0}
