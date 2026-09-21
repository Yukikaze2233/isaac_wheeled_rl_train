#!/usr/bin/env python3
"""Compare two source snapshots sequentially from the same optimizer checkpoint."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=6144)
    parser.add_argument("--updates", type=int, default=24)
    args = parser.parse_args()
    if args.updates < 8 or not 32 <= args.num_envs <= 8192:
        parser.error("Require 8+ PPO updates and 32-8192 environments")
    contract = json.loads(args.contract.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for label, source in (("reference", args.reference_source), ("candidate", args.candidate_source)):
        directory = args.output / label
        command = [sys.executable, "-B", str(source / "scripts/train_chassis.py"),
            "--contract", str(args.contract), "--resume", str(args.checkpoint), "--research",
            "--stage", contract["enabled_stages"][0], "--num-envs", str(args.num_envs),
            "--updates", str(args.updates), "--run-dir", str(directory), "--max-runtime-seconds", "900"]
        with (args.output / (label + ".log")).open("x") as log:
            subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        completion = json.loads((directory / "completion.json").read_text())
        if completion["status"] != "completed" or not completion.get("export", {}).get("verified"):
            raise RuntimeError(f"{label} did not finish and verify its export")
        events = EventAccumulator(str(directory)).Reload()
        results[label] = {
            "source": str(source), "updates": completion["successful_updates"],
            "parent_updates": completion["parent_updates"],
            "parent_checkpoint_sha256": completion["parent_checkpoint_sha256"],
            "max_closure_gap_m": completion["metrics"]["max_closure_gap_m"],
            "termination_counts": completion["metrics"]["termination_counts"],
            "onnx_verified": completion["export"]["verified"],
            "median": {tag: statistics.median(v.value for v in events.Scalars(tag)[5:])
                       for tag in ("Perf/collection_time", "Perf/learning_time", "Perf/total_fps")}}
        (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
        print(label, json.dumps(results[label]), flush=True)
    results["throughput_ratio"] = (results["candidate"]["median"]["Perf/total_fps"]
                                   / results["reference"]["median"]["Perf/total_fps"])
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print("THROUGHPUT_RATIO", results["throughput_ratio"], flush=True)


if __name__ == "__main__":
    main()
