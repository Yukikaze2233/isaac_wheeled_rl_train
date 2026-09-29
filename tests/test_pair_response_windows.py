"""A horizon sweep must simulate distinct windows at a shared start."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_pair_response import WindowReplay
from validate_pair_response import torque_input_scores


def test_window_cache_identity_includes_horizon():
    replay = object.__new__(WindowReplay)
    replay.initial = {37: (np.arange(1000), None, None, 0.)}
    replay.state_at = lambda sample: (np.array([sample]), np.zeros(1), 0., None)
    keys = [replay.window(37, 250, horizon) for horizon in (50, 100, 200)]
    assert len(set(keys)) == 3
    assert [len(replay.initial[key][0]) for key in keys] == [50, 100, 200]
    assert [replay.initial[key][0][-1] for key in keys] == [299, 349, 449]
    assert replay.window(37, 250, 100) == keys[1]


def test_validation_rejects_shortened_horizon_instead_of_mislabeling_it():
    class ShortenedReplay:
        def window(self, segment, start, count):
            return segment, start, count

        def replay(self, key, parameters, mode):
            assert mode == "torque_input"
            return {"samples": 50}, None

    results = torque_input_scores(ShortenedReplay(), 37, {}, 1000, "candidate")
    first = next(results)
    assert first["horizon_ms"] == first["samples"] == 50
    with pytest.raises(ValueError, match="horizon 100 ms replayed 50 samples"):
        next(results)


def test_validation_preserves_all_three_starts_and_three_horizons():
    class Replay:
        def window(self, segment, start, count):
            return segment, start, count

        def replay(self, key, parameters, mode):
            return {"samples": key[2]}, None

    results = list(torque_input_scores(Replay(), 37, {}, 24000, "candidate"))
    assert [(row["source_offset"], row["samples"]) for row in results] == [
        (start, horizon)
        for start in (6000, 12000, 18000)
        for horizon in (50, 100, 200)
    ]
