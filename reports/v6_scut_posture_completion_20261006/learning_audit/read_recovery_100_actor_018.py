#!/usr/bin/env python3
"""Read the existing v18 recovery native event: first 100 post-critic actors.

Runs directly in the existing simulator Python environment, without importing
torch, a simulator, a model, SSH, or starting a TensorBoard server. The event
remains the curve source; this script only writes a bounded evidence summary.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c

COMMIT = "4ba339299c203cf2aba77183ec81f2f017dc9bf9"
EXPECTED_PID = 3471715
BASELINE_STEP, START_STEP, END_STEP = 29214, 29215, 29314
BEHAVIOR_STEPS = list(range(BASELINE_STEP, END_STEP + 1, 10))
WINDOW_STEPS = list(range(START_STEP, END_STEP + 1))
DEFAULT_RUN = Path("/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_002/train")
COUNTS = ("frames", "observed_frames", "episodes", "successes", "failures", "timeouts",
          "boundary_truncations", "blocked_truncations")
MEANS = ("vx_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "planar_speed_mean_m_s",
         "vx_actual_mean_m_s", "vx_command_mean_m_s", "vx_bias_m_s",
         "height_actual_mean_m", "height_command_mean_m")
STATIONARY_FIELDS = ("samples", "endpoint_mean_m", "peak_mean_m", "peak_max_m", "mean_speed_m_s")
STOP_FIELDS = ("samples", "mean_speed_m_s", "above_3cm_s_fraction", "peak_speed_m_s")
TIMING_FIELDS = ("iteration_wall_seconds", "env_step_seconds", "optimizer_seconds",
                 "diagnostics_seconds", "logger_seconds")
LESSONS = ("prefix_stand", "prefix_release", "original_chain")
LESSON_FIELDS = ("probability", "population", "resets", "ended", "successes", "failures",
                 "collection_timeouts", "success_fraction")
POSTURE_REWARDS = ("pitch_exp", "roll_exp", "pitch_velocity", "roll_velocity",
                   "horizontal_angular_velocity", "locomotion_pitch_floor", "locomotion_roll_floor")
SOURCE_HASHES = {
    "contracts/v6_task_campaign_v18.json": "027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022",
    "contracts/v6_task_campaign_recovery_worker_v18.json": "7b5ecc7acc6871e072c2ca131d2ad2fa169dfd321b8a6c875cf2df643c5fd013",
    "scripts/train_chassis.py": "4326490e93e56f2240afa50d7ed99b82a6c4fdc3362b821c74af186956bdb73b",
    "src/wheeled_tasks/chassis/env.py": "0e5b0ae5ea6d443d567bb8b45055d967f56c1be5fde1f2a21e8b69d285e10c4b",
    "src/wheeled_tasks/chassis/episode_metrics.py": "26438717a12b3ab37b7279cbcae09dda91c549438c7a6090e7d6876c7a046511",
    "src/wheeled_tasks/chassis/learning_audit.py": "0ac7a3502199c0912bebf6a1689cf3d388ffe03e23fcb3217628a9a70d324715",
    "src/wheeled_tasks/chassis/recovery_prefix_curriculum.py": "c11cfe901744d943e27ac940c48e668be628c95ce57d6ea3f1461bca055b8bd4",
    "src/wheeled_tasks/chassis/recovery_training.py": "e6d0452ef90be408929cbfd56bba682171ea6e17ce54244cc395e6061626fba4",
    "src/wheeled_tasks/chassis/recovery_curriculum.py": "6617b7667dc703447b10c0ccd701424f833df51cf828f54e67b63e875b8e2a78",
    "src/wheeled_tasks/chassis/ordinary_posture_curriculum.py": "49add13707926965fdfdcc0dcada12860c6f84525fa5e6683cdc3d8fecea6a50",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path, limit=64 * 1024**2):
    require(path.stat().st_size <= limit, "Metadata exceeds byte limit: " + str(path))
    return json.loads(path.read_text())


def load_authority(run_dir, source):
    """Bind the catalog, native population, clocks and event to this attempt."""
    run_dir, source = run_dir.resolve(), source.resolve()
    require(run_dir.parts[-4:] == ("train", "recovery", "attempt_002", "train"), "Wrong recovery attempt")
    root = run_dir.parents[3]
    revision = read_json(root / "source_revision_018.json")
    require(revision["source_commit"] == COMMIT and revision["new_source_root"] == str(source), "Wrong frozen revision/source")
    actual = {name: digest(source / name) for name in SOURCE_HASHES}
    require(actual == SOURCE_HASHES, "Frozen metric/contract implementation differs")
    config = read_json(source / "contracts/v6_task_campaign_recovery_worker_v18.json")
    groups = [row["name"] for row in config["scene_groups"]]
    pools = config["behavior_pool_membership"]
    bases = [name for name in groups if pools[name] != "scripted_recovery"]
    require(len(groups) == len(set(groups)) == 95 and len(bases) == 41, "Wrong 95/41 group catalog")
    require(set(groups) == set(pools) == set(config["skill_specs"]), "Group/spec/pool catalog differs")
    require(Counter(pools.values()) == Counter(scripted_recovery=54, stand_anchors=9,
        stand_continuous=1, height=1, translation=16, transitions=6, yaw=8), "Base pool catalog differs")
    invocation = read_json(run_dir.parent / "invocation.json")
    argv = invocation["command"]
    def flag(name):
        require(name in argv and argv.count(name) == 1, "Missing/duplicate invocation flag " + name)
        return argv[argv.index(name) + 1]
    entry = next((arg for arg in argv[1:] if arg.endswith(".py")), None)
    require(entry == str(source / "scripts/train_chassis.py"), "Wrong native entry script")
    expected = {"--contract": str(source / "contracts/v6_task_campaign_recovery_worker_v18.json"),
        "--run-dir": str(run_dir), "--num-envs": "4096", "--seed": "290927", "--block-index": "0",
        "--task-actor-update": "2294", "--consumed-updates": "29205", "--stage": "mixed"}
    require(all(flag(key) == value for key, value in expected.items()), "Invocation differs from admitted v18 recovery")
    progress = read_json(run_dir / "progress.json")
    require(progress["parent_updates"] == 29205 and progress["num_envs"] == 4096,
        "Native origin/population differs")
    require(progress["task_actor_update"] == 2294 + progress["actor_updates_in_block"], "Native actor clocks differ")
    require(progress["successful_updates"] - progress["actor_updates_in_block"] == min(10, progress["successful_updates"]),
        "Unexpected critic warmup/replay; this reader cannot merge attempts")
    startup = read_json(run_dir / "startup.json")
    native = startup["startup"]
    require(native["runtime_preflight"]["status"] == "runtime_configuration_passed", "Native runtime not admitted")
    observed = native["runtime_preflight"]["observed"]
    require((observed["physics_dt"], observed["policy_dt"], observed["decimation"], observed["seed"])
        == (.005, .02, 4, 290927), "Native runtime timing/seed differs")
    population = native["scene_group_counts"]
    require(set(population) == set(groups) and all(type(n) is int and n > 0 for n in population.values())
        and sum(population.values()) == 4096, "Native positive 95-group population differs")
    ordinary = native["ordinary_posture_curriculum"]["runtime_observed"]
    base_population = sum(population[group] for group in bases)
    excluded_groups = [group for group in bases if pools[group] in {"translation", "transitions", "yaw"}]
    excluded_population = sum(population[group] for group in excluded_groups)
    require(base_population == 1449 and excluded_population == 872
        and ordinary["eligible_count"] == ordinary["birth_eligible_population"] == 577,
        "Actual admitted ordinary-posture eligibility differs")
    coverage = dict(available=True, native_eligible_population=577, base_group_population=1449,
        base_population_not_eligible=872, alias_excluded_base_groups=excluded_groups,
        scope="Frozen4ba startup population: translation/transitions/yaw legacy aliases are excluded from ordinary posture; base catalog coverage is not perturbation coverage.")
    pid = progress["pid"]
    require(pid == EXPECTED_PID, "Another worker/PID cannot supply this admitted first100 window")
    if "pid" in startup:
        require(startup["pid"] == pid, "Startup belongs to another worker")
    events = list(run_dir.glob("events.out.tfevents.*." + str(pid) + ".*"))
    require(len(events) == 1, "Expected one event belonging to the native worker PID")
    return dict(identity_verified=True, source_commit=COMMIT, source_root=str(source),
        worker_pid=pid, event=str(events[0]), groups=groups, base_groups=bases, group_pools=pools,
        population=population, progress=progress, invocation=invocation,
        ordinary_posture_coverage=coverage,
        clocks=dict(event_step_origin=29205, task_actor_origin=2294, critic_warmup_updates=10),
        source_files_sha256=actual, metadata_sha256={name: digest(path) for name, path in {
            "revision": root / "source_revision_018.json", "startup": run_dir / "startup.json",
            "invocation": run_dir.parent / "invocation.json"}.items()})


def wanted_tags(authority):
    tags = set()
    for group in authority["groups"]:
        tags.update(f"Behavior/{group}/{field}" for field in (*COUNTS, *MEANS, "tilt_max_deg"))
        tags.update(f"LearningAudit/{group}/stationary_10s/{field}" for field in STATIONARY_FIELDS)
        tags.update(f"LearningAudit/{group}/reward_density/{field}" for field in POSTURE_REWARDS)
    tags.update("Timing/" + field for field in TIMING_FIELDS)
    tags.add("Policy/mean_std")
    tags.update(f"/recovery/lesson/{lesson}/{field}" for lesson in LESSONS for field in LESSON_FIELDS)
    tags.update("/recovery/" + name for name in ("lesson/relative_actor_updates", "lesson/release_seconds",
        "full_chain_population", "full_chain_success"))
    tags.update("RecoveryRollout/" + name for name in ("valid_transitions", "script_or_blend_transitions", "valid_fraction"))
    for group in authority["base_groups"]:
        if authority["group_pools"][group] == "transitions":
            tags.update(f"LearningAudit/{group}/{phase}/{field}"
                for phase in ("braking", "settled_stop") for field in STOP_FIELDS)
    return tags


def canonical_tag(tag):
    if tag.startswith("Episode/") and tag[len("Episode/"):].lstrip("/").startswith("recovery/"):
        return "/" + tag[len("Episode/"):].lstrip("/")
    return tag


def parse_events(path, wanted, end_step, min_step=BASELINE_STEP, max_read_bytes=512 * 1024**2):
    """Stream native TFRecords from byte zero, checking both CRCs per record.

    This reuses review_chassis_progress.events framing/CRC semantics without
    holding a whole event in RAM. One incomplete live tail is reported, never
    accepted. No reservoir, simulator or model is involved.
    """
    series, seen_records, duplicates, unsupported = {}, 0, 0, set()
    hasher = hashlib.sha256()
    offset = last_step = 0
    incomplete = 0
    stopped = "snapshot_eof"
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        while offset < size:
            if offset + 12 > max_read_bytes:
                stopped = "byte_limit"; break
            header = stream.read(min(12, size - offset))
            if len(header) < 12:
                incomplete = len(header); stopped = "incomplete_live_tail"; break
            length, header_crc = struct.unpack("<QI", header)
            require(masked_crc32c(header[:8]) == header_crc, "TFRecord header CRC mismatch at " + str(offset))
            require(0 < length <= 16 * 1024**2, "Unexpected TFRecord payload length")
            if offset + length + 16 > max_read_bytes:
                stopped = "byte_limit"; break
            body = stream.read(min(length + 4, size - offset - 12))
            if len(body) != length + 4:
                incomplete = len(header) + len(body); stopped = "incomplete_live_tail"; break
            payload, data_crc = body[:-4], struct.unpack("<I", body[-4:])[0]
            require(masked_crc32c(payload) == data_crc, "TFRecord payload CRC mismatch at " + str(offset))
            event = Event.FromString(payload)
            hasher.update(header); hasher.update(body)
            offset += length + 16
            seen_records += 1
            if event.summary.value:
                last_step = max(last_step, int(event.step))
                # Train/*/time uses elapsed wall seconds as its step. Only an
                # admitted native update tag can establish the window boundary.
                if event.step > end_step and any(canonical_tag(value.tag) in wanted
                        for value in event.summary.value):
                    stopped = "first_record_after_target"; break
            if not min_step <= event.step <= end_step:
                continue
            for value in event.summary.value:
                tag = canonical_tag(value.tag)
                if tag not in wanted:
                    continue
                if not value.HasField("simple_value"):
                    unsupported.add(tag); continue
                number = float(value.simple_value)
                require(math.isfinite(number), "Nonfinite native scalar " + tag)
                item = [number, float(event.wall_time)]
                old = series.setdefault(tag, {}).get(int(event.step))
                require(old is None or old == item, "Conflicting duplicate native tag/step: " + tag)
                duplicates += int(old is not None)
                series[tag][int(event.step)] = item
    return series, dict(file=str(path.resolve()), file_bytes_at_open=size, read_complete_record_bytes=offset,
        read_limit_bytes=max_read_bytes, crc_valid_records=seen_records, last_summary_step_observed=last_step,
        incomplete_live_tail_bytes=incomplete, stop_reason=stopped, duplicate_identical_records=duplicates,
        unsupported_selected_scalar_encodings=sorted(unsupported), complete_record_prefix_sha256=hasher.hexdigest(),
        coverage="streamed_CRC_valid_native_prefix_not_reservoir")


def scalar(series, tag, step):
    row = series.get(tag, {}).get(step)
    return row[0] if row is not None else None


def stats(series, tag, steps):
    values = [scalar(series, tag, step) for step in steps if scalar(series, tag, step) is not None]
    complete = len(values) == len(steps)
    return dict(complete=complete, records=len(values), expected_records=len(steps),
        mean=statistics.mean(values) if complete and values else None,
        median=statistics.median(values) if complete and values else None,
        min=min(values) if complete and values else None, max=max(values) if complete and values else None)


def validate_counts(row, label):
    require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
        and row[k] == int(row[k]) for k in COUNTS), "Invalid integer counters: " + label)
    require(row["frames"] <= row["observed_frames"] and row["episodes"] == row["successes"] + row["failures"] + row["timeouts"]
        and row["boundary_truncations"] + row["blocked_truncations"] <= row["timeouts"], "Episode partition differs: " + label)


def summarize(series, authority, record_audit):
    require(authority.get("identity_verified") is True and authority["source_commit"] == COMMIT, "Unverified frozen identity")
    require(authority["clocks"] == dict(event_step_origin=29205, task_actor_origin=2294, critic_warmup_updates=10), "Wrong window clocks")
    groups, bases = authority["groups"], authority["base_groups"]
    population, pools = authority["population"], authority["group_pools"]
    require(len(groups) == len(set(groups)) == 95 and len(bases) == len(set(bases)) == 41
        and set(bases) == {g for g in groups if pools[g] != "scripted_recovery"}, "Wrong 95/41 catalog")
    require(set(population) == set(groups) and sum(population.values()) == 4096
        and all(type(n) is int and n > 0 for n in population.values()), "Wrong native population")
    missing = []
    def covered(section, tag, steps):
        absent = [step for step in steps if scalar(series, tag, step) is None]
        if absent:
            missing.append(dict(section=section, tag=tag, missing_steps=absent))
        return not absent
    behavior = dict(complete=True, groups={}, pools={}, base_groups=bases,
        static_recovery_groups=[g for g in groups if g not in bases], expected_cadence_steps=BEHAVIOR_STEPS,
        means_scope="Cumulative native frame-weighted means; resets, stochastic actions and prefix physical frames remain.",
        recovery_scope="54 static group labels mix reset-selected lessons; not original_chain-only success.")
    stationary = dict(complete=True, groups={}, scope="Windows completed in these rollouts; an anchor may precede the interval.")
    for group in groups:
        prefix = "Behavior/" + group + "/"
        checks = [covered("behavior", prefix + field, BEHAVIOR_STEPS) for field in (*COUNTS, *MEANS, "tilt_max_deg")]
        ready = all(checks)
        behavior["complete"] &= ready
        if ready:
            previous = None
            for step in BEHAVIOR_STEPS:
                row = {field: scalar(series, prefix + field, step) for field in COUNTS}
                validate_counts(row, group + "/" + str(step))
                worker_updates = step - 29205 + 1
                require(row["observed_frames"] == population[group] * worker_updates * 24, "Native group/frame denominator differs: " + group)
                require(previous is None or all(row[k] >= previous[k] for k in COUNTS), "Cumulative counter rollback: " + group)
                previous = row
            old = {k: scalar(series, prefix + k, BASELINE_STEP) for k in COUNTS}
            new = {k: scalar(series, prefix + k, END_STEP) for k in COUNTS}
            row = {k: int(new[k] - old[k]) for k in COUNTS}
            validate_counts(row, group + "/interval")
            require(row["observed_frames"] == population[group] * 100 * 24, "Interval population differs")
            for field in MEANS:
                row[field] = ((scalar(series, prefix + field, END_STEP) * new["frames"]
                    - scalar(series, prefix + field, BASELINE_STEP) * old["frames"]) / row["frames"] if row["frames"] else None)
            row.update(complete=True, population=population[group],
                native_success_per_ended_episode=row["successes"] / row["episodes"] if row["episodes"] else None,
                failures_per_ended_episode=row["failures"] / row["episodes"] if row["episodes"] else None,
                posture=dict(tilt_max_deg_since_observer_birth_before=scalar(series, prefix + "tilt_max_deg", BASELINE_STEP),
                    tilt_max_deg_since_observer_birth_after=scalar(series, prefix + "tilt_max_deg", END_STEP),
                    interval_max_available=False, separated_pitch_roll_available=False))
            behavior["groups"][group] = row
        else:
            behavior["groups"][group] = dict(complete=False, population=population[group])
        park_prefix = f"LearningAudit/{group}/stationary_10s/"
        park_ready = all([covered("stationary_10s", park_prefix + field, WINDOW_STEPS) for field in STATIONARY_FIELDS])
        park = dict(complete=park_ready, samples=None, endpoint_mean_m=None, peak_mean_m=None, peak_max_m=None, mean_speed_m_s=None)
        if park_ready:
            weights = [scalar(series, park_prefix + "samples", step) for step in WINDOW_STEPS]
            require(all(w >= 0 and w == int(w) for w in weights), "Invalid completed-window count: " + group)
            count = int(sum(weights)); park["samples"] = count
            if count:
                for field in ("endpoint_mean_m", "peak_mean_m", "mean_speed_m_s"):
                    park[field] = sum(scalar(series, park_prefix + field, step) * w for step, w in zip(WINDOW_STEPS, weights)) / count
                park["peak_max_m"] = max(scalar(series, park_prefix + "peak_max_m", step) for step, w in zip(WINDOW_STEPS, weights) if w)
        stationary["complete"] &= park_ready
        stationary["groups"][group] = park
    if behavior["complete"]:
        for pool in sorted(set(pools.values())):
            rows = [behavior["groups"][g] for g in groups if pools[g] == pool]
            totals = {key: sum(row[key] for row in rows) for key in COUNTS}
            for key in MEANS:
                totals[key] = sum((row[key] or 0.) * row["frames"] for row in rows) / totals["frames"] if totals["frames"] else None
            totals["native_success_per_ended_episode"] = totals["successes"] / totals["episodes"] if totals["episodes"] else None
            behavior["pools"][pool] = totals
        require(sum(row["observed_frames"] for row in behavior["groups"].values()) == 9830400, "Incomplete full native population")
    timing = {}
    for field in TIMING_FIELDS:
        tag = "Timing/" + field
        covered("timing", tag, WINDOW_STEPS)
        timing[field] = stats(series, tag, WINDOW_STEPS)
    covered("exploration", "Policy/mean_std", WINDOW_STEPS)
    exploration = stats(series, "Policy/mean_std", WINDOW_STEPS)
    lessons = dict(complete=True, levels={}, counter_scope="24_policy_frame_mean_of_cumulative_counter_not_exact_episode_count",
        fraction_scope="Time-averaged instantaneous lesson success/population; not ended-episode success rate.")
    for lesson in LESSONS:
        prefix = "/recovery/lesson/" + lesson + "/"
        fields = {}
        for field in LESSON_FIELDS:
            lessons["complete"] &= covered("dynamic_lessons", prefix + field, WINDOW_STEPS)
            fields[field] = stats(series, prefix + field, WINDOW_STEPS)
        # These are explicitly logged rollout averages, not exact integer counts.
        deltas = {}
        for field in ("resets", "ended", "successes", "failures", "collection_timeouts"):
            lessons["complete"] &= covered("dynamic_lesson_baseline", prefix + field, [BASELINE_STEP])
            a, b = scalar(series, prefix + field, BASELINE_STEP), scalar(series, prefix + field, END_STEP)
            deltas[field] = dict(before=a, after=b, delta=b - a if a is not None and b is not None else None)
        lessons["levels"][lesson] = dict(fields=fields, endpoint_rollout_average_deltas=deltas,
            counter_scope=lessons["counter_scope"])
    for name in ("lesson/relative_actor_updates", "lesson/release_seconds", "full_chain_population", "full_chain_success"):
        tag = "/recovery/" + name
        lessons["complete"] &= covered("dynamic_lessons", tag, WINDOW_STEPS)
        lessons[name] = stats(series, tag, WINDOW_STEPS)
    ownership = {}
    for name in ("valid_transitions", "script_or_blend_transitions", "valid_fraction"):
        tag = "RecoveryRollout/" + name
        covered("learning_ownership", tag, WINDOW_STEPS)
        ownership[name] = stats(series, tag, WINDOW_STEPS)
    ownership["scope"] = "Only policy_learning_mask-valid transitions can learn; other physical frames include SCRIPT/BLEND/passive birth."
    ownership["complete"] = all(ownership[name]["complete"] for name in ("valid_transitions", "script_or_blend_transitions", "valid_fraction"))
    if ownership["complete"]:
        valid_total = excluded_total = 0
        for step in WINDOW_STEPS:
            valid = scalar(series, "RecoveryRollout/valid_transitions", step)
            excluded = scalar(series, "RecoveryRollout/script_or_blend_transitions", step)
            fraction = scalar(series, "RecoveryRollout/valid_fraction", step)
            require(valid >= 0 and excluded >= 0 and valid == int(valid) and excluded == int(excluded)
                and valid + excluded == 98304, "Invalid learning-ownership transition partition")
            require(abs(fraction - valid / 98304) <= 2e-6, "Learning-ownership fraction denominator differs")
            valid_total += int(valid); excluded_total += int(excluded)
        ownership.update(valid_transitions_total=valid_total, excluded_transitions_total=excluded_total,
            observed_transitions_total=9830400)
    optional_posture = {g: {f: stats(series, f"LearningAudit/{g}/reward_density/{f}", WINDOW_STEPS)
        for f in POSTURE_REWARDS} for g in groups}
    # Conditional stop metrics are separate from start-stop whole-rollout speed.
    # They are an optional evidence section: missing fields stay explicitly null.
    stop_phases = dict(complete=True, groups={}, missing=[],
        scope="Native braking/settled_stop-mask physical samples, weighted by samples; whole start-stop motion is not parking.")
    for group in bases:
        if pools[group] != "transitions":
            continue
        stop_phases["groups"][group] = {}
        for phase in ("braking", "settled_stop"):
            prefix = f"LearningAudit/{group}/{phase}/"
            absent = {field: [step for step in WINDOW_STEPS if scalar(series, prefix + field, step) is None]
                for field in STOP_FIELDS}
            absent = {field: steps for field, steps in absent.items() if steps}
            ready = not absent
            row = dict(complete=ready, samples=None, mean_speed_m_s=None,
                above_3cm_s_fraction=None, peak_speed_m_s=None)
            stop_phases["complete"] &= ready
            if absent:
                stop_phases["missing"].append(dict(group=group, phase=phase, fields=absent))
            else:
                weights = [scalar(series, prefix + "samples", step) for step in WINDOW_STEPS]
                require(all(w >= 0 and w == int(w) for w in weights), "Invalid native stop-phase count: " + group)
                count = int(sum(weights)); row["samples"] = count
                if count:
                    for field in ("mean_speed_m_s", "above_3cm_s_fraction"):
                        row[field] = sum(scalar(series, prefix + field, step) * w for step, w in zip(WINDOW_STEPS, weights)) / count
                    row["peak_speed_m_s"] = max(scalar(series, prefix + "peak_speed_m_s", step)
                        for step, w in zip(WINDOW_STEPS, weights) if w)
            stop_phases["groups"][group][phase] = row
    progress = authority.get("progress")
    if progress is not None and progress["actor_updates_in_block"] < 100:
        missing.append(dict(section="progress", tag="progress/actor_updates_in_block",
            missing_steps=[100], observed=progress["actor_updates_in_block"]))
    complete = not missing
    return dict(schema="v18-recovery-first100-native-event-summary-v1",
        status="complete100_actor_window" if complete else "pending_complete100_actor_window", complete=complete,
        source_commit=COMMIT, qualification=None, curve_source="existing_native_TensorBoard_event_only",
        window=dict(task_actor_updates_inclusive=[2295, 2394], worker_cumulative_endpoints=[10, 110],
            cumulative_baseline_step=BASELINE_STEP, event_steps_inclusive=[START_STEP, END_STEP],
            actor_updates=100, critic_warmup_excluded=True, policy_seconds_per_environment=48.,
            expected_observed_frames=9830400, system_timing_cold_start_not_separately_removed=True),
        behavior=behavior, stationary_10s=stationary, timing=timing, exploration=exploration,
        dynamic_lessons=lessons, learning_ownership=ownership,
        ordinary_posture_coverage=authority.get("ordinary_posture_coverage", dict(available=False)),
        optional_conditional_stop_phases=stop_phases,
        optional_posture_reward_density=dict(scope="Reward densities/costs, not measured pitch/roll angles.", groups=optional_posture),
        missing=missing, missing_required_tag_count=len(missing), event=record_audit,
        limitations=["Stochastic training with resets/DR/physical prefix; not a deterministic mean-policy rollout.",
            "Behavior float32 event means are differenced with native integer frame counts; lifetime maxima cannot yield interval extrema.",
            "Recovery static group names do not identify dynamically sampled lesson levels.",
            "This frozen recovery run perturbs577 ordinary lanes; its1449 base-group lanes do not all have posture eligibility because872 legacy-alias lanes are excluded.",
            "Lesson cumulative counters are 24-frame rollout averages; their endpoint differences are not exact episode integers.",
            "Completed stationary windows need2s quiet then10s with height error<=10mm; anchors can precede this actor window; target need not stay fixed.",
            "Missing observations are unavailable, never imputed zero; zero completed stationary samples provide no stationary success evidence."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New summary evidence file; no event copying or rewriting")
    parser.add_argument("--max-read-mib", type=int, default=512)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists; use a unique evidence filename")
    require(1 <= args.max_read_mib <= 1024, "Read byte limit must be1–1024MiB")
    source = args.source_root or args.run_dir.resolve().parents[3] / "source_4ba339299c20"
    try:
        authority = load_authority(args.run_dir, source)
        series, audit = parse_events(Path(authority["event"]), wanted_tags(authority), END_STEP,
            max_read_bytes=args.max_read_mib * 1024**2)
        report = summarize(series, authority, audit)
        after = {name: digest(source / name) for name in SOURCE_HASHES}
        require(after == authority["source_files_sha256"], "Metric source changed during read")
        report["identity"] = {key: authority[key] for key in ("source_root", "worker_pid", "event", "base_groups",
            "population", "group_pools", "clocks", "source_files_sha256", "metadata_sha256", "invocation", "progress")}
        report["source_files_unchanged"] = True
        exitcode = 0
    except Exception as error:
        report = dict(schema="v18-recovery-first100-native-event-summary-v1", status="read_failed", complete=False,
            source_commit=COMMIT, qualification=None, error=type(error).__name__ + ": " + str(error))
        exitcode = 2
    report.update(queried_at_utc=datetime.now(timezone.utc).isoformat(), reader_sha256=digest(Path(__file__)),
        scope="Read_only_existing_native_event_no_GPU_no_SSH_no_model_no_TensorBoard_server")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], complete=report["complete"], output=str(args.output),
        missing_required_tag_count=report.get("missing_required_tag_count"), error=report.get("error"))))
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
