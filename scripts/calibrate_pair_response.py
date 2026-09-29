"""Calibrate paired suspended-leg dynamics across recorded orientations.

Uses complete, predeclared time windows for fitting and disjoint whole segments
for validation. Gains, gas curve, reference mapping and inertials stay fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from torch.utils.tensorboard import SummaryWriter

from identify_pair_dynamics import Replay


FIT_SEGMENTS = [9, 13, 15, 49, 53, 55, 61, 65, 67]
HOLDOUT_SEGMENTS = [37, 39, 41, 43, 79, 89]


class WindowReplay(Replay):
    """Cache each prescribed start and full horizon as a distinct window."""

    def window(self, segment, start, count):
        source, *_ = self.initial_state(segment)
        indices = source[start:start + count]
        if len(indices) != count:
            raise ValueError("Incomplete prescribed fit window")
        # Horizon is part of the identity: 50/100/200 ms checks share a start,
        # but must not silently reuse the first (shortest) cached trajectory.
        key = (segment, start, count)
        if key not in self.initial:
            position, velocity, gap, _ = self.state_at(indices[0])
            self.initial[key] = (indices, position, velocity, gap)
        return key


def fit_windows(replay):
    """Select the two frozen 2 s windows from each training-role segment."""
    windows = []
    for segment in FIT_SEGMENTS:
        indices, *_ = replay.initial_state(segment)
        if np.any(replay.a["segment_role"][indices] != 0):
            raise ValueError("Validation role leaked into fit")
        for start in (int(len(indices) * 0.25), len(indices) - 2100):
            windows.append(replay.window(segment, start, 2000))
    return windows


def decode_parameters(values):
    """Convert the optimizer's eight scaled coordinates to physical units."""
    return {
        "armature": (values[:2] * 0.02).tolist(),
        "damping": (values[2:4] * 0.2).tolist(),
        "frictionloss": (values[4:6] * 2).tolist(),
        "time_constant_s": (values[6:8] * 0.005).tolist(),
        "gain": 1.0,
    }


def initial_coordinates(path):
    parameters = json.loads(path.read_text()) if path else {
        "armature": [0.015, 0.015],
        "damping": [0.2, 0.3],
        "frictionloss": [1.5, 1.0],
        "time_constant_s": 0.004,
    }
    return np.r_[
        np.broadcast_to(parameters["armature"], 2) / 0.02,
        np.array(parameters["damping"]) / 0.2,
        np.array(parameters["frictionloss"]) / 2,
        np.broadcast_to(parameters["time_constant_s"], 2) / 0.005,
    ]


def residual_vector(replay, windows, parameters):
    """Use the original position/velocity scales and 10 ms residual spacing."""
    chunks = []
    for window in windows:
        _, trace = replay.replay(window, parameters)
        indices = trace["indices"]
        selected = np.arange(200, len(indices), 10)
        chunks.extend([
            ((trace["q_sim"][selected]
              - replay.real_q[indices[selected]]) / 0.02).ravel(),
            ((trace["dq_sim"][selected]
              - replay.real_dq[indices[selected]]) / 0.7).ravel(),
        ])
    return np.concatenate(chunks)


def calibrate(args):
    replay = WindowReplay(
        args.run, side=args.side, bundle=args.bundle,
        coordinate_sign=-1.0, physics_dt=0.001,
    )
    windows = fit_windows(replay)
    coordinates = initial_coordinates(args.initial)
    args.output.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(str(args.output / "events"))
    count, best = 0, np.inf

    def residual(values):
        nonlocal count, best
        parameters = decode_parameters(values)
        vector = residual_vector(replay, windows, parameters)
        loss = float(np.mean(2 * (np.sqrt(1 + vector * vector) - 1)))
        writer.add_scalar("pair_fit/robust_objective", loss, count)
        if loss < best:
            best = loss
            (args.output / "candidate_parameters.json").write_text(
                json.dumps(parameters, indent=2) + "\n"
            )
            print(json.dumps({
                "side": args.side, "evaluation": count,
                "loss": loss, "parameters": parameters,
            }), flush=True)
        count += 1
        return vector

    try:
        result = least_squares(
            residual,
            coordinates,
            bounds=(np.zeros(8), np.array([7.5, 7.5, 15, 15, 5, 5, 4, 4])),
            loss="soft_l1",
            diff_step=0.05,
            max_nfev=args.max_evaluations,
            ftol=0.001,
            xtol=0.001,
            gtol=0.001,
        )
        parameters = decode_parameters(result.x)
        (args.output / "parameters.json").write_text(
            json.dumps(parameters, indent=2) + "\n"
        )
        summary = {
            "scope": (
                "conditional orientation-stratified paired response; "
                "not a certified motor model"
            ),
            "side": args.side,
            "parameters": parameters,
            "fit_segments": FIT_SEGMENTS,
            "holdout_segments": HOLDOUT_SEGMENTS,
            "fit_windows": [{
                "key": window,
                "samples": len(replay.initial[window][0]),
                "source_start": int(replay.initial[window][0][0]),
            } for window in windows],
            "success": bool(result.success),
            "message": result.message,
            "nfev": result.nfev,
            "residual_evaluations": count,
            "active_bounds": result.active_mask.tolist(),
            "jacobian_singular_values": np.linalg.svd(
                result.jac, compute_uv=False
            ).tolist(),
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "asset_manifest_sha256": hashlib.sha256(
                (args.bundle / "manifest.json").read_bytes()
            ).hexdigest(),
            "npz_sha256": hashlib.sha256(
                (args.run / "response_1khz.npz").read_bytes()
            ).hexdigest(),
            "training_release": False,
        }
        (args.output / "fit.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--side", required=True, choices=["left", "right"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--max-evaluations", type=int, default=20)
    calibrate(parser.parse_args())


if __name__ == "__main__":
    main()
