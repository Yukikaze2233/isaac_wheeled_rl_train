"""Independent, standards-based TFRecord fixtures for the v18 recovery reader."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys

import pytest
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.types_pb2 import DT_FLOAT
from tensorboard.summary.writer.record_writer import RecordWriter


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
CONTRACT = REPOSITORY / "contracts/v6_task_campaign_recovery_worker_v18.json"
FROZEN = Path("/tmp/scut-v18-frozen-sy04n9ck")
BASELINE_STEP, FIRST_STEP, LAST_STEP = 29214, 29215, 29314
LESSONS = ("prefix_stand", "prefix_release", "original_chain")
COUNTS = ("frames", "observed_frames", "episodes", "successes", "failures",
          "timeouts", "boundary_truncations", "blocked_truncations")
MEANS = ("vx_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "planar_speed_mean_m_s",
         "vx_actual_mean_m_s", "vx_command_mean_m_s", "vx_bias_m_s",
         "height_actual_mean_m", "height_command_mean_m")
STATIONARY_FIELDS = ("samples", "endpoint_mean_m", "peak_mean_m", "peak_max_m",
                     "mean_speed_m_s")
TIMING_FIELDS = ("iteration_wall_seconds", "env_step_seconds", "optimizer_seconds",
                 "diagnostics_seconds", "logger_seconds")
LESSON_FIELDS = ("probability", "population", "resets", "ended", "successes",
                 "failures", "collection_timeouts", "success_fraction")


@pytest.fixture(scope="module")
def reader():
    path = HERE / "read_recovery_100_actor_018.py"
    spec = importlib.util.spec_from_file_location("recovery_window_reader_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contract():
    config = json.loads(CONTRACT.read_text())
    groups = [row["name"] for row in config["scene_groups"]]
    base = [name for name in groups
            if config["behavior_pool_membership"][name] != "scripted_recovery"]
    assert len(groups) == len(set(groups)) == 95
    assert len(base) == len(set(base)) == 41
    assert set(groups) == set(config["skill_specs"])
    return config, groups, base


def scalar_event(step, values, *, tensor_tags=()):
    """Use real Event/Summary protobufs, including both scalar encodings."""
    summary = Summary()
    for tag, value in values.items():
        item = summary.value.add(tag=tag)
        if tag in tensor_tags:
            item.tensor.CopyFrom(TensorProto(dtype=DT_FLOAT,
                tensor_content=struct.pack("<f", float(value))))
            item.metadata.plugin_data.plugin_name = "scalars"
        else:
            item.simple_value = float(value)
    return Event(wall_time=1700000000.0 + step, step=step, summary=summary)


def record_bytes(event):
    stream = io.BytesIO()
    RecordWriter(stream).write(event.SerializeToString(deterministic=True))
    return stream.getvalue()


def event_file(tmp_path, events, *, tail=b""):
    path = tmp_path / "events.out.tfevents.synthetic"
    header = Event(wall_time=1700000000.0, file_version="brain.Event:2")
    path.write_bytes(record_bytes(header)
                     + b"".join(record_bytes(event) for event in events) + tail)
    return path


@pytest.fixture(scope="module")
def authority(contract):
    config, groups, base = contract
    return dict(groups=groups, base_groups=base,
        population={group: 44 if index < 11 else 43 for index, group in enumerate(groups)},
        group_pools={group: config["behavior_pool_membership"][group] for group in groups},
        clocks=dict(event_step_origin=29205, task_actor_origin=2294, critic_warmup_updates=10),
        identity_verified=True, source_commit="4ba339299c203cf2aba77183ec81f2f017dc9bf9",
        worker_pid=123456)


def interval_means(index):
    return dict(vx_mae_m_s=.02 if index % 2 == 0 else .08, yaw_mae_rad_s=.10,
        height_mae_m=.005, planar_speed_mean_m_s=.40, vx_actual_mean_m_s=.40,
        vx_command_mean_m_s=.50, vx_bias_m_s=-.10, height_actual_mean_m=.300,
        height_command_mean_m=.305)


def behavior_values(authority, step):
    """Cumulative history at its real ten-update cadence, including endpoints."""
    elapsed = step - BASELINE_STEP
    cadence = elapsed // 10
    result = {}
    for index, group in enumerate(authority["groups"]):
        frames = (100 if index % 2 == 0 else 300) * elapsed // 100
        # Six timeouts, three successes, then one failure: each cumulative
        # category remains monotone at all eleven real logging checkpoints.
        successes, failures = max(0, min(cadence - 6, 3)), int(cadence == 10)
        deltas = dict(frames=frames, observed_frames=authority["population"][group] * elapsed * 24,
            episodes=cadence, successes=successes, failures=failures,
            timeouts=min(cadence, 6),
            boundary_truncations=min(cadence // 2, 2), blocked_truncations=min(cadence // 5, 1))
        before = dict(frames=1000, observed_frames=authority["population"][group] * 10 * 24, episodes=10,
            successes=4, failures=2, timeouts=4, boundary_truncations=0, blocked_truncations=0)
        prefix = f"Behavior/{group}/"
        result.update({prefix + field: before[field] + deltas[field] for field in COUNTS})
        for field, desired in interval_means(index).items():
            # Historical .01 is deliberately different from this window's mean.
            result[prefix + field] = (1000 * .01 + frames * desired) / (1000 + frames)
        result[prefix + "tilt_max_deg"] = 5. + index * .1 + elapsed / 100.
    return result


def rollout_values(authority, step):
    index = step - FIRST_STEP
    result = {"Timing/" + field: .10 + index / 1000. for field in TIMING_FIELDS}
    result["Policy/mean_std"] = .30 - index / 1000.
    valid = 80000 + index
    result.update({"RecoveryRollout/valid_transitions": valid,
        "RecoveryRollout/script_or_blend_transitions": 4096 * 24 - valid,
        "RecoveryRollout/valid_fraction": valid / (4096 * 24)})
    for group in authority["groups"]:
        prefix = f"LearningAudit/{group}/stationary_10s/"
        values = dict.fromkeys(STATIONARY_FIELDS, 0.)
        if group == authority["base_groups"][0] and index in (0, 1):
            values.update(samples=2 if index == 0 else 6,
                endpoint_mean_m=.25 if index == 0 else .50,
                peak_mean_m=.50 if index == 0 else .75,
                peak_max_m=.75 if index == 0 else .90,
                mean_speed_m_s=.125 if index == 0 else .25)
        result.update({prefix + field: values[field] for field in STATIONARY_FIELDS})
    original_population = 0 if index % 2 == 0 else 8
    for lesson, probability, population in zip(LESSONS, (.90, .095, .005), (100, 20, original_population)):
        prefix = f"/recovery/lesson/{lesson}/"
        # These are rollout-averaged cumulative counters, deliberately fractional.
        values = dict(probability=probability, population=population,
            resets=20.125 + index / 2., ended=10.125 + index / 4.,
            successes=4.25 + index / 10., failures=2.125 + index / 20.,
            collection_timeouts=3.75 + index / 10.,
            success_fraction=0. if population == 0 else .25)
        result.update({prefix + field: values[field] for field in LESSON_FIELDS})
    result.update({"/recovery/full_chain_population": original_population,
        "/recovery/full_chain_success": 0. if original_population == 0 else .25,
        "/recovery/lesson/relative_actor_updates": index,
        "/recovery/lesson/release_seconds": .03})
    return result


@pytest.fixture(scope="module")
def full_events(authority):
    baseline = behavior_values(authority, BASELINE_STEP)
    for lesson in LESSONS:
        prefix = f"/recovery/lesson/{lesson}/"
        for field, value in dict(resets=19.7, ended=9.8, successes=4.2,
                                 failures=2.0, collection_timeouts=3.6).items():
            baseline[prefix + field] = value
    events = [scalar_event(BASELINE_STEP, baseline)]
    for step in range(FIRST_STEP, LAST_STEP + 1):
        values = rollout_values(authority, step)
        if (step - BASELINE_STEP) % 10 == 0:
            values.update(behavior_values(authority, step))
        events.append(scalar_event(step, values))
    return events


def read_summary(reader, authority, path):
    series, audit = reader.parse_events(path, reader.wanted_tags(authority), LAST_STEP)
    return reader.summarize(series, authority, audit), series, audit


def test_required_tags_cover_all_95_groups_and_real_behavior_cadence(reader, authority):
    wanted = reader.wanted_tags(authority)
    behavior = {f"Behavior/{group}/{field}" for group in authority["groups"]
                for field in (*COUNTS, *MEANS, "tilt_max_deg")}
    stationary = {f"LearningAudit/{group}/stationary_10s/{field}"
                  for group in authority["groups"] for field in STATIONARY_FIELDS}
    lessons = {f"/recovery/lesson/{lesson}/{field}"
               for lesson in LESSONS for field in LESSON_FIELDS}
    assert {tag for tag in wanted if tag.startswith("Behavior/")} == behavior
    assert behavior | stationary | lessons <= wanted
    assert {"Timing/" + field for field in TIMING_FIELDS} <= wanted
    assert "Policy/mean_std" in wanted
    assert sum(authority["population"].values()) == 4096


def test_real_crc_complete_100_actor_window_uses_interval_counts_and_frame_weights(
        reader, authority, full_events, tmp_path):
    report, series, audit = read_summary(reader, authority, event_file(tmp_path, full_events))
    assert report["status"] == "complete100_actor_window" and report["complete"] is True
    assert report["ordinary_posture_coverage"]["available"] is False
    assert report["optional_conditional_stop_phases"]["complete"] is False
    assert report["optional_conditional_stop_phases"]["missing"]
    assert audit["crc_valid_records"] == 102
    assert audit["incomplete_live_tail_bytes"] == 0
    assert len(series["Policy/mean_std"]) == 100
    behavior = report["behavior"]
    assert behavior["complete"] is True and len(behavior["groups"]) == 95
    assert behavior["base_groups"] == authority["base_groups"]
    assert len(behavior["static_recovery_groups"]) == 54
    for index, group in enumerate(authority["groups"]):
        actual = behavior["groups"][group]
        assert actual["frames"] == (100 if index % 2 == 0 else 300)
        assert actual["observed_frames"] == authority["population"][group] * 100 * 24
        assert actual["episodes"] == 10 and actual["successes"] == 3
        assert actual["failures"] == 1 and actual["timeouts"] == 6
        assert actual["native_success_per_ended_episode"] == pytest.approx(.3)
        for field, value in interval_means(index).items():
            assert actual[field] == pytest.approx(value, abs=2e-6)
        posture = actual["posture"]
        assert posture["interval_max_available"] is False
        assert posture["separated_pitch_roll_available"] is False
        assert posture["tilt_max_deg_since_observer_birth_before"] == pytest.approx(5. + index * .1)
        assert posture["tilt_max_deg_since_observer_birth_after"] == pytest.approx(6. + index * .1)
    for pool, actual in behavior["pools"].items():
        indices = [i for i, group in enumerate(authority["groups"])
                   if authority["group_pools"][group] == pool]
        weights = [100 if index % 2 == 0 else 300 for index in indices]
        expected = sum(interval_means(i)["vx_mae_m_s"] * weight
                       for i, weight in zip(indices, weights)) / sum(weights)
        assert actual["vx_mae_m_s"] == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize("tag,step", [
    ("Timing/optimizer_seconds", FIRST_STEP + 33),
    ("Behavior/stand_230mm/vx_mae_m_s", BASELINE_STEP + 10),
    ("/recovery/lesson/original_chain/ended", BASELINE_STEP),
    ("RecoveryRollout/valid_fraction", FIRST_STEP + 12),
])
def test_missing_required_tag_is_pending_with_exact_missing_step(
        reader, authority, full_events, tmp_path, tag, step):
    events = deepcopy(full_events)
    target = next(event for event in events if event.step == step)
    kept = [value for value in target.summary.value if value.tag != tag]
    del target.summary.value[:]
    target.summary.value.extend(kept)
    report, _, _ = read_summary(reader, authority, event_file(tmp_path, events))
    assert report["complete"] is False
    assert report["status"] == "pending_complete100_actor_window"
    missing = [entry for entry in report["missing"] if entry["tag"] == tag]
    assert len(missing) == 1 and missing[0]["missing_steps"] == [step]


@pytest.mark.parametrize("crc_offset", [8, -1])
def test_corrupt_length_or_payload_crc_is_rejected(reader, tmp_path, crc_offset):
    record = bytearray(record_bytes(scalar_event(FIRST_STEP, {"Policy/mean_std": .2})))
    record[crc_offset] ^= 1
    path = tmp_path / "events.corrupt"
    path.write_bytes(record)
    with pytest.raises(ValueError, match="(?i)crc"):
        reader.parse_events(path, {"Policy/mean_std"}, LAST_STEP)


def test_conflicting_duplicate_after_window_end_is_rejected(reader, tmp_path):
    tag = "Policy/mean_std"
    events = [scalar_event(FIRST_STEP, {tag: .2}), scalar_event(LAST_STEP, {tag: .1}),
              scalar_event(FIRST_STEP, {tag: .8})]
    with pytest.raises(ValueError, match="(?i)(duplicate|conflict)"):
        reader.parse_events(event_file(tmp_path, events), {tag}, LAST_STEP)


def test_identical_duplicate_is_idempotent(reader, authority, full_events, tmp_path):
    events = [*full_events, deepcopy(full_events[17])]
    report, series, _ = read_summary(reader, authority, event_file(tmp_path, events))
    assert report["complete"] is True and len(series["Policy/mean_std"]) == 100


@pytest.mark.parametrize("complete_window", [True, False])
def test_truncated_live_tail_preserves_only_complete_records(
        reader, authority, full_events, tmp_path, complete_window):
    events = full_events if complete_window else full_events[:-1]
    pending = scalar_event(LAST_STEP + 1, {"Policy/mean_std": .1}) if complete_window else full_events[-1]
    partial = record_bytes(pending)[:-5]
    report, _, audit = read_summary(reader, authority, event_file(tmp_path, events, tail=partial))
    assert audit["incomplete_live_tail_bytes"] == len(partial)
    assert report["complete"] is complete_window
    assert report["status"] == ("complete100_actor_window" if complete_window
                                  else "pending_complete100_actor_window")


def test_tensor_scalar_is_explicitly_missing_not_silently_read_as_zero(
        reader, authority, full_events, tmp_path):
    events = deepcopy(full_events)
    target = next(event for event in events if event.step == FIRST_STEP + 5)
    tag = "Policy/mean_std"
    value = next(value for value in target.summary.value if value.tag == tag)
    value.tensor.CopyFrom(TensorProto(dtype=DT_FLOAT, tensor_content=struct.pack("<f", .25)))
    value.metadata.plugin_data.plugin_name = "scalars"
    report, series, _ = read_summary(reader, authority, event_file(tmp_path, events))
    assert report["complete"] is False and FIRST_STEP + 5 not in series[tag]
    assert any(entry["tag"] == tag and entry["missing_steps"] == [FIRST_STEP + 5]
               for entry in report["missing"])


def test_dynamic_lesson_rollout_counter_means_are_never_exact_episode_rates(
        reader, authority, full_events, tmp_path):
    report, _, _ = read_summary(reader, authority, event_file(tmp_path, full_events))
    for lesson, level in report["dynamic_lessons"]["levels"].items():
        assert lesson in LESSONS
        assert level["counter_scope"] == "24_policy_frame_mean_of_cumulative_counter_not_exact_episode_count"
        assert "ended_rate" not in level and "native_success_per_ended_episode" not in level
        ended = level["endpoint_rollout_average_deltas"]["ended"]
        assert ended["before"] == pytest.approx(9.8)
        assert ended["delta"] == pytest.approx(25.075)
        assert level["fields"]["population"]["records"] == 100
    original = report["dynamic_lessons"]["levels"]["original_chain"]
    assert original["fields"]["population"]["min"] == 0
    assert original["fields"]["population"]["mean"] == pytest.approx(4.)
    assert original["fields"]["success_fraction"]["mean"] == pytest.approx(.125)


def test_stationary_averages_use_completed_window_counts_and_exclude_zero_samples(
        reader, authority, full_events, tmp_path):
    report, _, _ = read_summary(reader, authority, event_file(tmp_path, full_events))
    stationary = report["stationary_10s"]
    assert stationary["complete"] is True
    positive = stationary["groups"][authority["base_groups"][0]]
    assert positive["samples"] == 8
    assert positive["endpoint_mean_m"] == pytest.approx(.4375)
    assert positive["peak_mean_m"] == pytest.approx(.6875)
    assert positive["mean_speed_m_s"] == pytest.approx(.21875)
    assert positive["peak_max_m"] == pytest.approx(.9)
    empty = stationary["groups"][authority["base_groups"][1]]
    assert empty["samples"] == 0
    for field in STATIONARY_FIELDS[1:]:
        assert empty[field] is None


def test_nonfinite_valid_crc_event_is_rejected(reader, tmp_path):
    path = event_file(tmp_path, [scalar_event(FIRST_STEP, {"Policy/mean_std": float("nan")})])
    with pytest.raises(ValueError, match="(?i)(finite|nan)"):
        reader.parse_events(path, {"Policy/mean_std"}, LAST_STEP)


def test_unverified_authority_never_certifies_complete_window(reader, authority, full_events, tmp_path):
    series, audit = reader.parse_events(event_file(tmp_path, full_events),
                                       reader.wanted_tags(authority), LAST_STEP)
    unverified = deepcopy(authority)
    unverified["identity_verified"] = False
    with pytest.raises(ValueError, match="(?i)(identity|verif|authorit)"):
        reader.summarize(series, unverified, audit)


def test_episode_prefix_alias_preserves_dynamic_lesson_catalog(reader, authority, full_events, tmp_path):
    events = deepcopy(full_events)
    for event in events:
        for value in event.summary.value:
            if value.tag.startswith("/recovery/"):
                value.tag = "Episode/" + value.tag
    report, series, _ = read_summary(reader, authority, event_file(tmp_path, events))
    assert report["complete"] is True
    assert "/recovery/lesson/original_chain/population" in series
    assert report["dynamic_lessons"]["levels"]["original_chain"]["fields"]["population"]["mean"] == pytest.approx(4.)


@pytest.mark.parametrize("field,mutation", [
    ("frames", lambda old: -1.),
    ("episodes", lambda old: old + 1.),
    ("observed_frames", lambda old: old - 1.),
])
def test_valid_crc_invalid_native_counter_or_population_is_rejected(
        reader, authority, full_events, tmp_path, field, mutation):
    events = deepcopy(full_events)
    group = authority["groups"][0]
    value = next(value for value in events[-1].summary.value
                 if value.tag == f"Behavior/{group}/{field}")
    value.simple_value = mutation(value.simple_value)
    with pytest.raises(ValueError):
        read_summary(reader, authority, event_file(tmp_path, events))


@pytest.mark.parametrize("field", ["valid_transitions", "valid_fraction"])
def test_valid_crc_policy_ownership_partition_or_fraction_mismatch_is_rejected(
        reader, authority, full_events, tmp_path, field):
    events = deepcopy(full_events)
    event = next(event for event in events if event.step == FIRST_STEP + 50)
    value = next(value for value in event.summary.value if value.tag == "RecoveryRollout/" + field)
    value.simple_value += 1.
    with pytest.raises(ValueError):
        read_summary(reader, authority, event_file(tmp_path, events))


def test_wrong_frozen_source_rejected_even_with_true_verified_flag(
        reader, authority, full_events, tmp_path):
    series, audit = reader.parse_events(event_file(tmp_path, full_events),
                                       reader.wanted_tags(authority), LAST_STEP)
    wrong = deepcopy(authority)
    wrong["source_commit"] = "f" * 40
    with pytest.raises(ValueError, match="(?i)(identity|verif|authorit)"):
        reader.summarize(series, wrong, audit)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def admitted_attempt(tmp_path, authority):
    """Real admission metadata and actual frozen source, entirely local."""
    root = tmp_path / "campaign"
    run = root / "train/recovery/attempt_002/train"
    run.mkdir(parents=True)
    source = FROZEN.resolve(strict=True)
    revision = dict(source_commit="4ba339299c203cf2aba77183ec81f2f017dc9bf9",
                    new_source_root=str(source))
    command = ["/synthetic/cpu/python", "-B", str(source / "scripts/train_chassis.py"),
        "--contract", str(source / "contracts/v6_task_campaign_recovery_worker_v18.json"),
        "--run-dir", str(run), "--num-envs", "4096", "--seed", "290927", "--block-index", "0",
        "--task-actor-update", "2294", "--consumed-updates", "29205", "--stage", "mixed"]
    invocation = dict(command=command)
    progress = dict(parent_updates=29205, num_envs=4096, task_actor_update=2317,
                    actor_updates_in_block=23, successful_updates=33, pid=3471715)
    population = {}
    # Preserve the admitted population denominators without pretending this
    # synthetic fixture is the worker's exact per-group allocation.
    categories = [("permitted", 577), ("aliases", 872), ("recovery", 2647)]
    for category, total in categories:
        groups = [group for group in authority["groups"]
            if ("recovery" if authority["group_pools"][group] == "scripted_recovery"
                else "aliases" if authority["group_pools"][group] in {"translation", "transitions", "yaw"}
                else "permitted") == category]
        quotient, remainder = divmod(total, len(groups))
        population.update({group: quotient + int(index < remainder)
                           for index, group in enumerate(groups)})
    assert len(population) == 95 and sum(population.values()) == 4096
    assert sum(population[group] for group in authority["base_groups"]) == 1449
    startup = dict(pid=3471715, startup=dict(
        runtime_preflight=dict(status="runtime_configuration_passed",
            observed=dict(physics_dt=.005, policy_dt=.02, decimation=4, seed=290927)),
        scene_group_counts=population,
        ordinary_posture_curriculum=dict(runtime_observed=dict(
            eligible_count=577, birth_eligible_population=577))))
    paths = dict(revision=root / "source_revision_018.json",
                 invocation=run.parent / "invocation.json",
                 progress=run / "progress.json", startup=run / "startup.json")
    for name, value in dict(revision=revision, invocation=invocation,
                            progress=progress, startup=startup).items():
        write_json(paths[name], value)
    event = run / "events.out.tfevents.1700000000.synthetic-host.3471715.0"
    event.write_bytes(record_bytes(Event(wall_time=1700000000., file_version="brain.Event:2")))
    return dict(root=root, run=run, source=source, paths=paths, event=event,
                population=population)


def test_load_authority_authenticates_local_attempt_metadata_and_ten_frozen_hashes(
        reader, authority, admitted_attempt):
    attempt = admitted_attempt
    result = reader.load_authority(attempt["run"], attempt["source"])
    assert result["identity_verified"] is True
    assert result["worker_pid"] == 3471715
    assert result["source_root"] == str(FROZEN.resolve())
    assert Path(result["event"]) == attempt["event"]
    assert result["groups"] == authority["groups"]
    assert result["base_groups"] == authority["base_groups"]
    assert result["population"] == attempt["population"]
    assert result["clocks"] == authority["clocks"]
    assert len(result["source_files_sha256"]) == 10
    assert result["source_files_sha256"] == {
        name: hashlib.sha256((FROZEN / name).read_bytes()).hexdigest()
        for name in reader.SOURCE_HASHES}
    assert result["metadata_sha256"] == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in attempt["paths"].items() if name != "progress"}
    coverage = result["ordinary_posture_coverage"]
    assert coverage["available"] is True
    assert coverage["native_eligible_population"] == 577
    assert coverage["base_group_population"] == 1449
    assert coverage["base_population_not_eligible"] == 872
    assert len(coverage["alias_excluded_base_groups"]) == 30
    assert set(coverage["alias_excluded_base_groups"]) == {
        group for group in authority["base_groups"]
        if authority["group_pools"][group] in {"translation", "transitions", "yaw"}}


@pytest.mark.parametrize("fault,error", [
    ("pid", "(?i)(pid|worker)"),
    ("clock", "(?i)clock"),
    ("warmup", "(?i)(warmup|replay)"),
    ("invocation", "(?i)invocation"),
    ("source_sha", "(?i)(frozen|implementation|source)"),
    ("ordinary_count", "(?i)(ordinary|eligibility)"),
    ("birth_count", "(?i)(ordinary|eligibility)"),
    ("base_population", "(?i)(ordinary|eligibility)"),
    ("alias_population", "(?i)(ordinary|eligibility)"),
])
def test_load_authority_rejects_real_metadata_or_source_identity_mismatches(
        reader, admitted_attempt, tmp_path, fault, error):
    attempt = admitted_attempt
    source = attempt["source"]
    if fault in ("pid", "clock", "warmup"):
        path = attempt["paths"]["progress"]
        progress = json.loads(path.read_text())
        if fault == "pid":
            progress["pid"] = 3471716
        elif fault == "clock":
            progress["task_actor_update"] += 1
        else:
            progress["successful_updates"] += 1
        write_json(path, progress)
    elif fault == "invocation":
        path = attempt["paths"]["invocation"]
        invocation = json.loads(path.read_text())
        command = invocation["command"]
        command[command.index("--num-envs") + 1] = "20000"
        write_json(path, invocation)
    elif fault in {"ordinary_count", "birth_count", "base_population", "alias_population"}:
        path = attempt["paths"]["startup"]
        startup = json.loads(path.read_text())
        native = startup["startup"]
        if fault in {"ordinary_count", "birth_count"}:
            field = "eligible_count" if fault == "ordinary_count" else "birth_eligible_population"
            native["ordinary_posture_curriculum"]["runtime_observed"][field] += 1
        else:
            config = json.loads((FROZEN / "contracts/v6_task_campaign_recovery_worker_v18.json").read_text())
            pools = config["behavior_pool_membership"]
            counts = native["scene_group_counts"]
            permitted = next(group for group in counts if pools[group] == "stand_anchors")
            donor_pool = "scripted_recovery" if fault == "base_population" else "translation"
            donor = next(group for group in counts if pools[group] == donor_pool)
            counts[permitted] += 1
            counts[donor] -= 1
            assert sum(counts.values()) == 4096 and all(count > 0 for count in counts.values())
        write_json(path, startup)
    else:
        # Never mutate the actual frozen root: copy the ten authenticated
        # files to the owned fixture root, then change one byte-level identity.
        source = tmp_path / "altered-source"
        for name in reader.SOURCE_HASHES:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((FROZEN / name).read_bytes())
        altered = source / "contracts/v6_task_campaign_v18.json"
        altered.write_bytes(altered.read_bytes() + b"\n")
        revision_path = attempt["paths"]["revision"]
        revision = json.loads(revision_path.read_text())
        revision["new_source_root"] = str(source.resolve())
        write_json(revision_path, revision)
        invocation_path = attempt["paths"]["invocation"]
        invocation = json.loads(invocation_path.read_text())
        command = invocation["command"]
        command[2] = str(source / "scripts/train_chassis.py")
        command[command.index("--contract") + 1] = str(source / "contracts/v6_task_campaign_recovery_worker_v18.json")
        write_json(invocation_path, invocation)
    with pytest.raises(ValueError, match=error):
        reader.load_authority(attempt["run"], source)


def test_unselected_walltime_axis_does_not_end_native_iteration_prefix(
        reader, authority, full_events, tmp_path):
    unrelated = scalar_event(99999, {"Train/mean_reward/time": .5})
    report, series, audit = read_summary(reader, authority,
                                       event_file(tmp_path, [unrelated, *full_events]))
    assert report["complete"] is True
    assert "Train/mean_reward/time" not in series
    assert audit["crc_valid_records"] == 103
    assert audit["last_summary_step_observed"] == 99999
    assert audit["stop_reason"] == "snapshot_eof"


def test_future_selected_tag_stops_at_crc_checked_record_boundary(
        reader, authority, full_events, tmp_path):
    future = scalar_event(LAST_STEP + 1, {"Policy/mean_std": .2})
    # A later record lies outside the admitted selected-tag prefix. The audit
    # must delimit that prefix rather than claim it checked the whole file.
    outside = scalar_event(FIRST_STEP, {"Policy/mean_std": .99})
    path = event_file(tmp_path, [*full_events, future, outside])
    report, series, audit = read_summary(reader, authority, path)
    assert report["complete"] is True
    assert LAST_STEP + 1 not in series["Policy/mean_std"]
    assert audit["stop_reason"] == "first_record_after_target"
    assert audit["crc_valid_records"] == 103
    assert audit["read_complete_record_bytes"] == path.stat().st_size - len(record_bytes(outside))


def test_optional_conditional_stop_phases_weight_physical_samples_and_exclude_zero(
        reader, authority, full_events, tmp_path):
    transitions = [group for group in authority["base_groups"]
                   if authority["group_pools"][group] == "transitions"]
    assert len(transitions) == 6
    events = deepcopy(full_events)
    fields = ("samples", "mean_speed_m_s", "above_3cm_s_fraction", "peak_speed_m_s")
    for event in events[1:]:
        values = {}
        for group in transitions:
            for phase in ("braking", "settled_stop"):
                # Deliberately high zero-sample fields must never contribute
                # speed, exceedance or a peak to the admitted aggregate.
                row = dict(samples=0., mean_speed_m_s=99.,
                           above_3cm_s_fraction=1., peak_speed_m_s=99.)
                if group == transitions[0] and phase == "braking":
                    if event.step == FIRST_STEP:
                        row.update(samples=2, mean_speed_m_s=.025,
                                   above_3cm_s_fraction=.5, peak_speed_m_s=.1)
                    elif event.step == FIRST_STEP + 1:
                        row.update(samples=6, mean_speed_m_s=.05,
                                   above_3cm_s_fraction=.25, peak_speed_m_s=.2)
                values.update({f"LearningAudit/{group}/{phase}/{field}": row[field]
                               for field in fields})
        event.summary.value.extend(scalar_event(event.step, values).summary.value)
    report, _, _ = read_summary(reader, authority, event_file(tmp_path, events))
    assert report["complete"] is True
    phases = report["optional_conditional_stop_phases"]
    assert phases["complete"] is True and phases["missing"] == []
    assert set(phases["groups"]) == set(transitions)
    row = phases["groups"][transitions[0]]["braking"]
    assert row["complete"] is True and row["samples"] == 8
    assert row["mean_speed_m_s"] == pytest.approx((2 * .025 + 6 * .05) / 8)
    assert row["above_3cm_s_fraction"] == pytest.approx((2 * .5 + 6 * .25) / 8)
    assert row["peak_speed_m_s"] == pytest.approx(.2)
    for group in transitions:
        for phase in ("braking", "settled_stop"):
            if group == transitions[0] and phase == "braking":
                continue
            empty = phases["groups"][group][phase]
            assert empty["complete"] is True and empty["samples"] == 0
            assert all(empty[field] is None for field in fields[1:])
