"""CRC-read existing source019 event hold gates/commands; no simulator or model.

Upload beside the SHA-bound read_recovery_window_019.py. Same attempt003 source,
startup, invocation and PID authority. Explicit endpoint fixes the same window.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

READER_SHA = "ffefbaae2e1cbf30b520224fc286a8adbc194708570e4ce0fcb00fa7f86cdb17"
GATES = ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed")
GENERAL = ("hold_population", "stable_peak_seconds_mean", "stable_break_count_mean",
           "script_fraction", "eligible_policy_fraction", "full_chain_population", "full_chain_success")
COMMANDS = ("command_vx_m_s", "command_yaw_rad_s", "held_low_vx_fraction", "held_low_yaw_fraction", "rollout_samples")


def load_reader():
    path = Path(__file__).with_name("read_recovery_window_019.py")
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != READER_SHA:
        raise ValueError("Source019 reader SHA differs")
    spec = importlib.util.spec_from_file_location("source019_bound_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wanted_tags(authority):
    tags = {"/recovery/hold_gate/" + gate + "_pass_fraction" for gate in GATES}
    tags.update("/recovery/" + field for field in GENERAL)
    tags.update("/recovery/lesson/" + lesson + "/" + field
                for lesson in ("prefix_stand", "prefix_release", "original_chain")
                for field in ("population", "successes", "success_fraction"))
    tags.update("LearningAudit/" + group + "/" + field for group in authority["groups"] for field in COMMANDS)
    return tags


def summarize(reader, series, authority, width, endpoint):
    window = reader.window_spec(width, endpoint)
    steps = window["dense_steps"]
    missing = {tag: [step for step in steps if step not in series.get(tag, {})]
               for tag in sorted(wanted_tags(authority))}
    missing = {tag: values for tag, values in missing.items() if values}
    result = dict(complete=not missing, window=window,
        missing_required_tag_count=len(missing), missing=missing, qualification=None)
    if missing:
        return result
    result["general"] = {field: reader.stats(series, "/recovery/" + field, steps) for field in GENERAL}
    populations = [series["/recovery/hold_population"][step][0] for step in steps]
    reader.require(all(0. <= value <= 2647. for value in populations), "Invalid recovery hold population")
    result["rollouts_with_positive_mean_hold_population"] = sum(value > 0. for value in populations)
    result["hold_gates"] = {gate: reader.stats(series, "/recovery/hold_gate/" + gate + "_pass_fraction", steps)
                            for gate in GATES}
    for field in (*result["hold_gates"].values(), *(result["general"][key]
                   for key in ("script_fraction", "eligible_policy_fraction", "full_chain_success"))):
        reader.require(0. <= field["min"] <= field["max"] <= 1., "Invalid fraction")
    result["commands"] = {}
    for group in authority["groups"]:
        samples = [series["LearningAudit/" + group + "/rollout_samples"][step][0] for step in steps]
        expected = 24 * authority["population"][group]
        reader.require(all(value == expected for value in samples), "Per-group rollout sample denominator differs: " + group)
        row = {"physical_observation_samples": int(sum(samples))}
        for field in COMMANDS[:-1]:
            values = [series["LearningAudit/" + group + "/" + field][step][0] for step in steps]
            if field.endswith("fraction"):
                reader.require(all(0. <= value <= 1. for value in values), "Invalid command fraction")
            row[field] = sum(value * n for value, n in zip(values, samples)) / sum(samples)
        result["commands"][group] = row
    result["lessons"] = {lesson: {field: reader.stats(series, "/recovery/lesson/" + lesson + "/" + field, steps)
        for field in ("population", "successes", "success_fraction")}
        for lesson in ("prefix_stand", "prefix_release", "original_chain")}
    result["limitations"] = [
        "Eight gate fractions are marginal holding-lane fractions averaged across24policy frames and then updates, not per-lesson or joint continuous1s success rates.",
        "Gate fraction and population are separately averaged in each rollout. Multiplying their averages cannot recover exact passed/holding sample counts.",
        "When hold_population is zero, all gate fractions are zero by clamp_min(1), not evidence that every lane failed every gate.",
        "FSM stable peak/break means include all4096 lanes; not actual50Hz hold evidence or per-episode ending counts.",
        "Command means/low fractions use physical observation rollout_samples, including prefix frames; no per-group actor-learning mask or dynamic-lesson joint command exposure is logged.",
        "Low vx/yaw fractions cover only absolute commands.1–.3 with the other axis<.01; not all nonzero commands or actual tracking accuracy.",
        "Actual signed yaw angular velocity is not produced by these command tags.",
        "Lesson counters are24-frame means of cumulative ints, never exact ended rates.",
    ]
    return result


def main():
    reader = load_reader()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=reader.DEFAULT_RUN)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--updates", type=int, choices=(50, 100), default=100)
    parser.add_argument("--end-update", type=int, default=110)
    parser.add_argument("--max-read-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reader.require(not args.output.exists(), "Output exists; choose unique name")
    reader.require(1 <= args.max_read_mib <= 1024, "Invalid read bound")
    source = args.source_root or args.run_dir.resolve().parents[3] / "source_712d701473f3"
    try:
        authority = reader.load_authority(args.run_dir, source)
        window = reader.window_spec(args.updates, args.end_update)
        series, audit = reader.parse_events(Path(authority["event"]), wanted_tags(authority),
            window["end_step"], min_step=window["start_step"], max_read_bytes=args.max_read_mib * 1024**2)
        report = summarize(reader, series, authority, args.updates, args.end_update)
        reader.require({name: reader.digest(source / name) for name in reader.SOURCE_HASHES}
                       == authority["source_files_sha256"], "Frozen source changed during read")
        report.update(schema="source019-existing-native-hold-gate-command-window-v1",
            source_commit=reader.COMMIT, source_files_unchanged=True, event=audit,
            identity={key: authority[key] for key in ("source_root", "worker_pid", "event", "source_files_sha256", "metadata_sha256")},
            status="complete" if report["complete"] else "pending_missing_native_tags")
        exitcode = 0
    except Exception as error:
        report = dict(schema="source019-existing-native-hold-gate-command-window-v1", complete=False,
            status="read_failed", error=type(error).__name__ + ": " + str(error), qualification=None)
        exitcode = 2
    report.update(reader_sha256=READER_SHA, wrapper_sha256=reader.digest(Path(__file__)),
        queried_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="read_only_existing_attempt003_event_no_model_GPU_SSH_or_new_curve_source")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "complete")}, ensure_ascii=False))
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
