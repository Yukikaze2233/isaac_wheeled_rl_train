"""Fixed held-out closed-loop and torque-input checks for paired response."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from calibrate_pair_response import HOLDOUT_SEGMENTS, WindowReplay


def closed_loop_score(replay, segment, parameters, measured_beta):
    score, trace = replay.replay(segment, parameters)
    indices = trace["indices"]
    free = measured_beta[indices] < 104.0
    score["free_measured_angle_axis_rmse_rad"] = (
        np.sqrt(np.mean(
            (trace["q_sim"][free] - replay.real_q[indices][free]) ** 2, axis=0
        )).tolist() if free.any() else None
    )
    score["free_measured_angle_samples"] = int(free.sum())
    score["free_measured_angle_false_sim_stop_fraction"] = (
        float(np.mean(trace["knee_limit_constraint_nm"][free] > 1e-6))
        if free.any() else None
    )
    return score, trace


def save_trace(replay, trace, output, writer, side, label, segment):
    indices = trace["indices"]
    np.savez_compressed(
        output / f"{label}-segment-{segment}.npz",
        q_real=replay.real_q[indices],
        dq_real=replay.real_dq[indices],
        q_reference=replay.reference[indices],
        **trace,
    )
    for sample in range(0, len(indices), 20):
        for axis in range(2):
            prefix = (
                f"pair_validation/{side}/{label}/segment{segment}/axis{axis}"
            )
            writer.add_scalar(
                prefix + "/q_real_rad",
                replay.real_q[indices[sample], axis], sample,
            )
            writer.add_scalar(
                prefix + "/q_sim_rad", trace["q_sim"][sample, axis], sample
            )
            writer.add_scalar(
                prefix + "/dq_real_rad_s",
                replay.real_dq[indices[sample], axis], sample,
            )
            writer.add_scalar(
                prefix + "/dq_sim_rad_s", trace["dq_sim"][sample, axis], sample
            )


def torque_input_scores(replay, segment, parameters, sample_count, label):
    """Replay each start/horizon with no within-window state injection."""
    # Multiple starts limit initial-condition drift; each full window is free.
    for fraction in (0.25, 0.5, 0.75):
        start = int(sample_count * fraction)
        for horizon in (50, 100, 200):
            key = replay.window(segment, start, horizon)
            score, _ = replay.replay(key, parameters, mode="torque_input")
            if score["samples"] != horizon:
                raise ValueError(
                    f"Torque-input horizon {horizon} ms replayed "
                    f"{score['samples']} samples instead of {horizon}"
                )
            yield {
                "model": label,
                "experiment": "torque_input",
                "source_segment": segment,
                "source_offset": start,
                "horizon_ms": horizon,
                **score,
            }


def validate(args):
    replay = WindowReplay(
        args.run, side=args.side, bundle=args.bundle,
        coordinate_sign=-1.0, physics_dt=0.001,
    )
    parameters = json.loads(args.params.read_text())
    nominal = {
        "armature": [0.0, 0.0],
        "damping": [0.0, 0.0],
        "frictionloss": [0.0, 0.0],
        "time_constant_s": 0.0,
        "gain": 1.0,
    }
    coverage_path = (
        Path(__file__).parents[1]
        / f"reports/v6_stop_audit_20260929/coverage/{args.side}.json"
    )
    coverage = json.loads(coverage_path.read_text())["table"]
    measured_beta = np.interp(
        replay.real_q[:, 1] - replay.real_q[:, 0],
        coverage["delta_grid_rad"],
        coverage["knee_inner_deg"],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(str(args.output / "events"))
    report = {
        "side": args.side,
        "parameters": parameters,
        "baseline": nominal,
        "scope": (
            "same-bag orientation/frequency holdout, "
            "no physical parameter uniqueness claim"
        ),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "results": [],
    }
    try:
        for label, model in (("nominal", nominal), ("candidate", parameters)):
            for segment in HOLDOUT_SEGMENTS:
                score, trace = closed_loop_score(
                    replay, segment, model, measured_beta
                )
                report["results"].append({
                    "model": label, "experiment": "closed_loop", **score,
                })
                save_trace(
                    replay, trace, args.output, writer,
                    args.side, label, segment,
                )
                report["results"].extend(torque_input_scores(
                    replay, segment, model, len(trace["indices"]), label
                ))
                (args.output / "report.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
                print(json.dumps({
                    "model": label,
                    "segment": segment,
                    "q_rmse": score["axis_rmse_rad"],
                    "v_rmse": score["velocity_rmse_rad_s"],
                    "stop_fraction": score["knee_stop"][
                        "limit_active_fraction"
                    ],
                }), flush=True)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--side", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
