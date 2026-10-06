"""Independent real-CRC fixtures for source019's zero-warmup recovery windows."""
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
FROZEN = Path("/tmp/scut-ordinary-alias-019-oit43rqd")
COMMIT = "712d701473f38ef6797725f8efdda9e6beb85e87"
EVENT_ORIGIN, ACTOR_ORIGIN, WORKER_PID = 29379, 2458, 3480218
COUNTS = ("frames", "observed_frames", "episodes", "successes", "failures",
          "timeouts", "boundary_truncations", "blocked_truncations")
MEANS = ("vx_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "planar_speed_mean_m_s",
         "vx_actual_mean_m_s", "vx_command_mean_m_s", "vx_bias_m_s",
         "height_actual_mean_m", "height_command_mean_m")
STATIONARY_FIELDS = ("samples", "endpoint_mean_m", "peak_mean_m", "peak_max_m",
                     "mean_speed_m_s")
TIMING_FIELDS = ("iteration_wall_seconds", "env_step_seconds", "optimizer_seconds",
                 "diagnostics_seconds", "logger_seconds")
LESSONS = ("prefix_stand", "prefix_release", "original_chain")
LESSON_FIELDS = ("probability", "population", "resets", "ended", "successes",
                 "failures", "collection_timeouts", "success_fraction")
STOP_FIELDS = ("samples", "mean_speed_m_s", "above_3cm_s_fraction", "peak_speed_m_s")


@pytest.fixture(scope="module")
def reader():
    path = HERE / "read_recovery_window_019.py"
    spec = importlib.util.spec_from_file_location("recovery_window019_reader_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def catalog():
    config = json.loads((FROZEN / "contracts/v6_task_campaign_recovery_worker_v18.json").read_text())
    groups = [row["name"] for row in config["scene_groups"]]
    pools = config["behavior_pool_membership"]
    bases = [name for name in groups if pools[name] != "scripted_recovery"]
    assert len(groups) == len(set(groups)) == 95 and len(bases) == 41
    population = {}
    aliases = [g for g in bases if pools[g] in {"translation", "transitions", "yaw"}]
    permitted = [g for g in bases if g not in aliases]
    assert len(permitted) == 11 and len(aliases) == 30
    for members, total in ((permitted, 577), (aliases, 872), ([g for g in groups if g not in bases], 2647)):
        quotient, remainder = divmod(total, len(members))
        population.update({g: quotient + int(index < remainder) for index, g in enumerate(members)})
    assert sum(population.values()) == 4096
    return dict(groups=groups, base_groups=bases, group_pools=pools, population=population)


@pytest.fixture(scope="module")
def authority(catalog):
    return dict(**deepcopy(catalog), identity_verified=True, source_commit=COMMIT,
        worker_pid=WORKER_PID,
        clocks=dict(event_step_origin=EVENT_ORIGIN, task_actor_origin=ACTOR_ORIGIN,
                    critic_warmup_updates=0),
        progress=dict(parent_updates=EVENT_ORIGIN, num_envs=4096, pid=WORKER_PID,
                      task_actor_update=ACTOR_ORIGIN + 130, actor_updates_in_block=130,
                      successful_updates=130))


def scalar_event(step, values, *, tensor_tags=()):
    summary = Summary()
    for tag, value in values.items():
        item = summary.value.add(tag=tag)
        if tag in tensor_tags:
            item.tensor.CopyFrom(TensorProto(dtype=DT_FLOAT,
                tensor_content=struct.pack("<f", float(value))))
            item.metadata.plugin_data.plugin_name = "scalars"
        else:
            item.simple_value = float(value)
    return Event(wall_time=1700000000. + step, step=step, summary=summary)


def record_bytes(event):
    stream = io.BytesIO()
    RecordWriter(stream).write(event.SerializeToString(deterministic=True))
    return stream.getvalue()


def event_file(tmp_path, events, *, tail=b""):
    path = tmp_path / "events.out.tfevents.synthetic019"
    header = Event(wall_time=1700000000., file_version="brain.Event:2")
    path.write_bytes(record_bytes(header) + b"".join(record_bytes(event) for event in events) + tail)
    return path


def step_at(update):
    assert update >= 1  # Source019 emits no native zero-update baseline.
    return EVENT_ORIGIN + update - 1


def behavior_values(authority, update):
    assert update >= 10 and update % 10 == 0
    cadence = update // 10
    result = {}
    for index, group in enumerate(authority["groups"]):
        prefix = f"Behavior/{group}/"
        successes, failures = (cadence + 1) // 2, cadence // 4
        timeouts = cadence - successes - failures
        counts = dict(frames=update * (2 if index % 2 == 0 else 6),
            observed_frames=authority["population"][group] * update * 24,
            episodes=cadence, successes=successes, failures=failures,
            timeouts=timeouts, boundary_truncations=timeouts // 2, blocked_truncations=0)
        result.update({prefix + field: counts[field] for field in COUNTS})
        desired = dict(vx_mae_m_s=.02 if index % 2 == 0 else .08, yaw_mae_rad_s=.1,
            height_mae_m=.005, planar_speed_mean_m_s=.4, vx_actual_mean_m_s=.4,
            vx_command_mean_m_s=.5, vx_bias_m_s=-.1, height_actual_mean_m=.3,
            height_command_mean_m=.305)
        for field in MEANS:
            result[prefix + field] = (10 * .01 + (update - 10) * desired[field]) / update
        result[prefix + "tilt_max_deg"] = 5. + index * .1 + update / 100.
    return result


def rollout_values(authority, update):
    result = {"Timing/" + field: .1 + update / 1000. for field in TIMING_FIELDS}
    result["Policy/mean_std"] = .3 - update / 1000.
    valid = 80000 + update
    result.update({"RecoveryRollout/valid_transitions": valid,
        "RecoveryRollout/script_or_blend_transitions": 98304 - valid,
        "RecoveryRollout/valid_fraction": valid / 98304})
    for group in authority["groups"]:
        result.update({f"LearningAudit/{group}/stationary_10s/{field}": 0.
                       for field in STATIONARY_FIELDS})
        if authority["group_pools"][group] == "transitions":
            result.update({f"LearningAudit/{group}/{phase}/{field}": 0.
                for phase in ("braking", "settled_stop") for field in STOP_FIELDS})
    for lesson, probability, population in zip(LESSONS, (.90, .095, .005), (100, 20, 8)):
        values = dict(probability=probability, population=population,
            resets=20.125 + update / 2., ended=10.125 + update / 4.,
            successes=4.25 + update / 10., failures=2.125 + update / 20.,
            collection_timeouts=3.75 + update / 10., success_fraction=.25)
        result.update({f"/recovery/lesson/{lesson}/{field}": values[field]
                       for field in LESSON_FIELDS})
    result.update({"/recovery/full_chain_population": 8., "/recovery/full_chain_success": .25,
        "/recovery/lesson/relative_actor_updates": update,
        "/recovery/lesson/release_seconds": .03})
    return result


def native_events(authority, updates):
    events = []
    for update in range(1, updates + 1):
        values = rollout_values(authority, update)
        if update % 10 == 0:
            values.update(behavior_values(authority, update))
        events.append(scalar_event(step_at(update), values))
    return events


@pytest.fixture(scope="module")
def full_events(authority):
    return native_events(authority, 130)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def admitted_attempt(tmp_path, authority):
    root = tmp_path / "campaign"
    run = root / "train/recovery/attempt_003/train"
    run.mkdir(parents=True)
    source = FROZEN.resolve(strict=True)
    revision = dict(source_commit=COMMIT, new_source_root=str(source))
    invocation = dict(command=["/synthetic/cpu/python", "-B", str(source / "scripts/train_chassis.py"),
        "--contract", str(source / "contracts/v6_task_campaign_recovery_worker_v18.json"),
        "--run-dir", str(run), "--num-envs", "4096", "--seed", "290927", "--block-index", "0",
        "--task-actor-update", str(ACTOR_ORIGIN), "--consumed-updates", str(EVENT_ORIGIN), "--stage", "mixed"])
    progress = dict(parent_updates=EVENT_ORIGIN, num_envs=4096, task_actor_update=ACTOR_ORIGIN + 22,
                    actor_updates_in_block=22, successful_updates=22, pid=WORKER_PID)
    startup = dict(pid=WORKER_PID, startup=dict(
        runtime_preflight=dict(status="runtime_configuration_passed",
            observed=dict(physics_dt=.005, policy_dt=.02, decimation=4, seed=290927)),
        scene_group_counts=authority["population"],
        ordinary_posture_curriculum=dict(origin_actor_update=2294, runtime_observed=dict(
            eligible_count=1449, birth_eligible_population=1449))))
    paths = dict(revision=root / "source_revision_019.json", invocation=run.parent / "invocation.json",
                 progress=run / "progress.json", startup=run / "startup.json")
    for name, value in dict(revision=revision, invocation=invocation,
                            progress=progress, startup=startup).items():
        write_json(paths[name], value)
    event = run / f"events.out.tfevents.1700000000.synthetic-host.{WORKER_PID}.0"
    event.write_bytes(record_bytes(Event(wall_time=1700000000., file_version="brain.Event:2")))
    return dict(root=root, run=run, source=source, paths=paths, event=event)


def with_progress(authority, updates):
    result = deepcopy(authority)
    result["progress"].update(actor_updates_in_block=updates, successful_updates=updates,
                              task_actor_update=ACTOR_ORIGIN + updates)
    return result


def readback(reader, authority, path, end_update=130):
    return reader.parse_events(path, reader.wanted_tags(authority), step_at(end_update))


@pytest.fixture(scope="module")
def full_readback(reader, authority, full_events, tmp_path_factory):
    return readback(reader, authority, event_file(tmp_path_factory.mktemp("native019"), full_events))


@pytest.mark.parametrize("width,end_update", [(50, 60), (100, 110)])
def test_first_admissible_window_has_native_tenth_update_baseline(reader, width, end_update):
    window = reader.window_spec(width, end_update)
    assert window["actor_updates"] == width
    assert window["baseline_update"] == 10 and window["end_update"] == end_update
    assert window["baseline_step"] == step_at(10)
    assert window["start_step"] == step_at(11)
    assert window["end_step"] == step_at(end_update)
    assert window["behavior_steps"] == [step_at(u) for u in range(10, end_update + 1, 10)]
    assert window["dense_steps"] == [step_at(u) for u in range(11, end_update + 1)]
    assert len(window["dense_steps"]) == width


@pytest.mark.parametrize("width,end_update", [(50, 50), (100, 100), (50, 61), (20, 120)])
def test_zero_baseline_misaligned_or_wrong_width_is_rejected(reader, width, end_update):
    with pytest.raises(ValueError):
        reader.window_spec(width, end_update)


@pytest.mark.parametrize("width,updates", [(50, 22), (50, 50), (50, 59),
                                           (100, 22), (100, 100), (100, 109)])
def test_short_native_attempt_never_manufactures_fifty_or_hundred_update_evidence(
        reader, authority, tmp_path, width, updates):
    current = with_progress(authority, updates)
    series, audit = readback(reader, current, event_file(tmp_path, native_events(current, updates)), updates)
    report = reader.select_window(series, current, audit, width)
    assert report["complete"] is False and report["status"] == "pending_complete_actor_window"
    assert report["window"]["actor_updates"] == width
    assert report["window"]["baseline_update"] == 10
    assert report["window"]["end_update"] == width + 10
    assert report["missing"]
    assert step_at(10) == report["window"]["baseline_step"]
    assert EVENT_ORIGIN - 1 not in series.get("Policy/mean_std", {})


@pytest.mark.parametrize("width,updates", [(50, 60), (100, 110)])
def test_real_crc_first_complete_window_starts_after_real_baseline(
        reader, authority, tmp_path, width, updates):
    current = with_progress(authority, updates)
    series, audit = readback(reader, current, event_file(tmp_path, native_events(current, updates)), updates)
    report = reader.select_window(series, current, audit, width)
    assert report["complete"] is True
    assert report["window"]["baseline_update"] == 10
    assert report["window"]["task_actor_updates_inclusive"] == [ACTOR_ORIGIN + 11, ACTOR_ORIGIN + updates]
    assert report["window"]["new_critic_warmup_updates"] == 0
    assert report["window"]["expected_observed_frames"] == width * 98304
    assert audit["crc_valid_records"] == updates + 1


@pytest.mark.parametrize("width", [50, 100])
def test_latest_full_window_has_exact_native_denominators_and_no_actual_yaw_inference(
        reader, authority, full_readback, width):
    series, audit = full_readback
    report = reader.select_window(series, authority, audit, width)
    assert report["complete"] is True and report["status"] == "complete_actor_window"
    assert report["source_commit"] == COMMIT and report["qualification"] is None
    window = report["window"]
    assert window["end_update"] == 130 and window["baseline_update"] == 130 - width
    assert window["event_steps_inclusive"] == [step_at(131 - width), step_at(130)]
    assert len(report["behavior"]["expected_cadence_steps"]) == width // 10 + 1
    assert len(report["behavior"]["groups"]) == 95
    assert len(report["behavior"]["base_groups"]) == 41
    assert len(report["behavior"]["static_recovery_groups"]) == 54
    for index, group in enumerate(authority["groups"]):
        row = report["behavior"]["groups"][group]
        assert row["observed_frames"] == authority["population"][group] * width * 24
        assert row["frames"] == width * (2 if index % 2 == 0 else 6)
        assert row["vx_mae_m_s"] == pytest.approx(.02 if index % 2 == 0 else .08, abs=1e-6)
        assert row["planar_speed_mean_m_s"] == pytest.approx(.4, abs=1e-6)
    assert sum(row["observed_frames"] for row in report["behavior"]["groups"].values()) == width * 98304
    assert report["learning_ownership"]["observed_transitions_total"] == width * 98304
    assert report["yaw_observability"]["actual_yaw_rate_available"] is False
    assert report["yaw_observability"]["actual_yaw_rate_mean_rad_s"] is None
    assert report["ordinary_posture_coverage"]["available"] is False
    assert report["optional_conditional_stop_phases"]["complete"] is True


@pytest.mark.parametrize("width", [50, 100])
def test_partial_last_nine_updates_selects_latest_complete_aligned_end(
        reader, authority, tmp_path, width):
    current = with_progress(authority, 119)
    series, audit = readback(reader, current, event_file(tmp_path, native_events(current, 119)), 119)
    report = reader.select_window(series, current, audit, width)
    assert report["complete"] is True and report["window"]["end_update"] == 110


@pytest.mark.parametrize("width", [50, 100])
def test_latest_missing_dense_tag_falls_back_to_preceding_complete_window(
        reader, authority, full_events, tmp_path, width):
    events = deepcopy(full_events)
    event = events[124]  # update125 is outside a window ending120.
    kept = [value for value in event.summary.value if value.tag != "Timing/env_step_seconds"]
    del event.summary.value[:]
    event.summary.value.extend(kept)
    series, audit = readback(reader, authority, event_file(tmp_path, events))
    report = reader.select_window(series, authority, audit, width)
    assert report["complete"] is True and report["window"]["end_update"] == 120
    explicit = reader.select_window(series, authority, audit, width, end_update=130)
    assert explicit["complete"] is False
    assert any(row["tag"] == "Timing/env_step_seconds" and row["missing_steps"] == [step_at(125)]
               for row in explicit["missing"])


def test_missing_required_behavior_cadence_cannot_be_replaced_by_endpoint_only_check(
        reader, authority, full_events, tmp_path):
    events = deepcopy(full_events)
    event = events[119]
    tag = f"Behavior/{authority['groups'][0]}/height_mae_m"
    kept = [value for value in event.summary.value if value.tag != tag]
    del event.summary.value[:]
    event.summary.value.extend(kept)
    series, audit = readback(reader, authority, event_file(tmp_path, events))
    report = reader.select_window(series, authority, audit, 100)
    assert report["complete"] is True and report["window"]["end_update"] == 110


def test_missing_native_parking_phase_is_required_for_complete_behavior_window(
        reader, authority, full_events, tmp_path):
    events = deepcopy(full_events)
    group = next(g for g in authority["base_groups"] if authority["group_pools"][g] == "transitions")
    tag = f"LearningAudit/{group}/settled_stop/mean_speed_m_s"
    event = events[124]
    kept = [value for value in event.summary.value if value.tag != tag]
    del event.summary.value[:]
    event.summary.value.extend(kept)
    series, audit = readback(reader, authority, event_file(tmp_path, events))
    report = reader.select_window(series, authority, audit, 50)
    assert report["complete"] is True and report["window"]["end_update"] == 120
    explicit = reader.select_window(series, authority, audit, 50, end_update=130)
    assert explicit["complete"] is False
    assert any(row["tag"] == tag and row["missing_steps"] == [step_at(125)] for row in explicit["missing"])


@pytest.mark.parametrize("fault", ["length_crc", "payload_crc"])
def test_genuine_record_corrupt_crc_is_rejected(reader, authority, full_events, tmp_path, fault):
    path = event_file(tmp_path, full_events)
    data = bytearray(path.read_bytes())
    data[8 if fault == "length_crc" else -1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="(?i)crc"):
        readback(reader, authority, path)


def test_conflicting_duplicate_with_valid_crc_is_rejected(reader, authority, full_events, tmp_path):
    duplicate = scalar_event(step_at(70), {"Policy/mean_std": .99})
    with pytest.raises(ValueError, match="(?i)(duplicate|conflict)"):
        readback(reader, authority, event_file(tmp_path, [*full_events, duplicate]))


@pytest.mark.parametrize("full_last_record", [False, True])
def test_truncated_live_record_never_certifies_its_unwritten_update(
        reader, authority, full_events, tmp_path, full_last_record):
    events = full_events if full_last_record else full_events[:-1]
    next_event = scalar_event(step_at(131), {"Policy/mean_std": .15}) if full_last_record else full_events[-1]
    series, audit = readback(reader, authority, event_file(tmp_path, events, tail=record_bytes(next_event)[:7]))
    report = reader.select_window(series, authority, audit, 50)
    assert report["complete"] is True
    assert report["window"]["end_update"] == (130 if full_last_record else 120)
    assert audit["incomplete_live_tail_bytes"] == 7
    if not full_last_record:
        assert reader.select_window(series, authority, audit, 50, end_update=130)["complete"] is False


def test_unselected_wallclock_step_does_not_bound_native_actor_window(
        reader, authority, full_events, tmp_path):
    unrelated = scalar_event(99999, {"Train/mean_reward/time": .5})
    series, audit = readback(reader, authority, event_file(tmp_path, [unrelated, *full_events]))
    report = reader.select_window(series, authority, audit, 100)
    assert report["complete"] is True and report["window"]["end_update"] == 130
    assert "Train/mean_reward/time" not in series
    assert audit["crc_valid_records"] == 132


def test_tensor_scalar_is_missing_instead_of_imputed(reader, authority, full_events, tmp_path):
    events = deepcopy(full_events)
    event = events[124]
    value = next(value for value in event.summary.value if value.tag == "Policy/mean_std")
    old = value.simple_value
    value.ClearField("simple_value")
    value.tensor.CopyFrom(TensorProto(dtype=DT_FLOAT, tensor_content=struct.pack("<f", old)))
    value.metadata.plugin_data.plugin_name = "scalars"
    series, audit = readback(reader, authority, event_file(tmp_path, events))
    assert step_at(125) not in series["Policy/mean_std"]
    report = reader.select_window(series, authority, audit, 100, end_update=130)
    assert report["complete"] is False
    assert any(row["tag"] == "Policy/mean_std" and row["missing_steps"] == [step_at(125)]
               for row in report["missing"])


def test_dynamic_fractional_counters_do_not_become_exact_episode_rates(reader, authority, full_readback):
    series, audit = full_readback
    report = reader.select_window(series, authority, audit, 50)
    lesson = report["dynamic_lessons"]["levels"]["original_chain"]
    assert lesson["counter_scope"] == "24_policy_frame_mean_of_cumulative_counter_not_exact_episode_count"
    assert lesson["endpoint_rollout_average_deltas"]["ended"]["delta"] == pytest.approx(12.5)
    assert "ended_rate" not in lesson and "native_success_per_ended_episode" not in lesson


def test_actual_frozen_source_metadata_binds_zero_warmup_and_all_1449_base_lanes(
        reader, authority, admitted_attempt):
    attempt = admitted_attempt
    result = reader.load_authority(attempt["run"], attempt["source"])
    assert result["identity_verified"] is True and result["source_commit"] == COMMIT
    assert result["worker_pid"] == WORKER_PID and Path(result["event"]) == attempt["event"]
    assert result["population"] == authority["population"]
    assert result["clocks"] == authority["clocks"]
    assert result["progress"]["successful_updates"] == result["progress"]["actor_updates_in_block"] == 22
    assert len(result["source_files_sha256"]) == 10
    assert result["source_files_sha256"] == {
        name: hashlib.sha256((FROZEN / name).read_bytes()).hexdigest()
        for name in reader.SOURCE_HASHES}
    assert result["metadata_sha256"] == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in attempt["paths"].items() if name != "progress"}
    coverage = result["ordinary_posture_coverage"]
    assert coverage["available"] is True
    assert coverage["native_eligible_population"] == coverage["base_group_population"] == 1449
    assert coverage["base_population_not_eligible"] == 0
    assert coverage["legacy_alias_population_now_eligible"] == 872
    assert len(coverage["alias_enabled_base_groups"]) == 30
    assert coverage["nonstandard_start_leg_bank_present"] is False


@pytest.mark.parametrize("fault,error", [
    ("pid", "(?i)(pid|worker)"), ("actor_clock", "(?i)clock"),
    ("consumed_origin", "(?i)(origin|population)"), ("warmup", "(?i)warmup"),
    ("source_commit", "(?i)(revision|source)"), ("source_sha", "(?i)(frozen|source|implementation)"),
    ("ordinary_old577", "(?i)(ordinary|eligibility)"),
    ("curriculum_origin", "(?i)(curriculum|origin)"), ("invocation", "(?i)invocation"),
])
def test_metadata_admission_rejects_old_source_clock_warmup_or_partial_eligibility(
        reader, admitted_attempt, tmp_path, fault, error):
    attempt, source = admitted_attempt, admitted_attempt["source"]
    if fault in {"pid", "actor_clock", "consumed_origin", "warmup"}:
        path = attempt["paths"]["progress"]
        value = json.loads(path.read_text())
        key = {"pid": "pid", "actor_clock": "task_actor_update",
               "consumed_origin": "parent_updates", "warmup": "successful_updates"}[fault]
        value[key] += 1
        write_json(path, value)
    elif fault == "source_commit":
        path = attempt["paths"]["revision"]
        value = json.loads(path.read_text())
        value["source_commit"] = "4ba339299c203cf2aba77183ec81f2f017dc9bf9"
        write_json(path, value)
    elif fault in {"ordinary_old577", "curriculum_origin"}:
        path = attempt["paths"]["startup"]
        value = json.loads(path.read_text())
        ordinary = value["startup"]["ordinary_posture_curriculum"]
        if fault == "ordinary_old577":
            ordinary["runtime_observed"].update(eligible_count=577, birth_eligible_population=577)
        else:
            ordinary["origin_actor_update"] = ACTOR_ORIGIN
        write_json(path, value)
    elif fault == "invocation":
        path = attempt["paths"]["invocation"]
        value = json.loads(path.read_text())
        command = value["command"]
        command[command.index("--task-actor-update") + 1] = "2294"
        write_json(path, value)
    else:
        source = tmp_path / "altered019"
        for name in reader.SOURCE_HASHES:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((FROZEN / name).read_bytes())
        altered = source / "src/wheeled_tasks/chassis/ordinary_posture_curriculum.py"
        altered.write_bytes(altered.read_bytes() + b"\n")
        path = attempt["paths"]["revision"]
        value = json.loads(path.read_text())
        value["new_source_root"] = str(source.resolve())
        write_json(path, value)
        path = attempt["paths"]["invocation"]
        value = json.loads(path.read_text())
        command = value["command"]
        command[2] = str(source / "scripts/train_chassis.py")
        command[command.index("--contract") + 1] = str(source / "contracts/v6_task_campaign_recovery_worker_v18.json")
        write_json(path, value)
    with pytest.raises(ValueError, match=error):
        reader.load_authority(attempt["run"], source)


def test_bounded_suffix_has_real_crc_alignment_and_does_not_claim_earlier_crc(
        reader, authority, full_events, tmp_path):
    path = event_file(tmp_path, full_events)
    header = record_bytes(Event(wall_time=1700000000., file_version="brain.Event:2"))
    prefix_length = len(header) + sum(len(record_bytes(event)) for event in full_events[:20])
    budget = path.stat().st_size - prefix_length
    series, audit = reader.parse_events(path, reader.wanted_tags(authority), step_at(130),
                                         max_read_bytes=budget)
    assert audit["source_region_start_offset"] == prefix_length
    assert audit["source_region_end_offset"] == path.stat().st_size
    assert audit["earlier_records_crc_verified"] is False
    assert audit["crc_valid_records"] == 110
    assert audit["read_complete_record_bytes"] == budget
    assert reader.select_window(series, authority, audit, 100)["complete"] is True


def test_crc_covered_hundred_dense_updates_without_behavior_baseline_remain_pending(
        reader, authority, full_events, tmp_path):
    path = event_file(tmp_path, full_events)
    header = record_bytes(Event(wall_time=1700000000., file_version="brain.Event:2"))
    prefix_length = len(header) + sum(len(record_bytes(event)) for event in full_events[:30])
    series, audit = reader.parse_events(path, reader.wanted_tags(authority), step_at(130),
        max_read_bytes=path.stat().st_size - prefix_length)
    assert audit["crc_valid_records"] == 100
    assert len(series["Policy/mean_std"]) == 100
    report = reader.select_window(series, authority, audit, 100)
    assert report["complete"] is False
    assert any(row["tag"].startswith("Behavior/") and step_at(30) in row["missing_steps"]
               for row in report["missing"])
    assert reader.select_window(series, authority, audit, 50)["complete"] is True


def test_stationary_and_conditional_stop_weight_two_six_samples_and_exclude_zero(
        reader, authority, full_events, tmp_path):
    events = deepcopy(full_events)
    stationary_group = authority["base_groups"][0]
    transitions = [g for g in authority["base_groups"] if authority["group_pools"][g] == "transitions"]
    for update, event in enumerate(events, 1):
        for value in event.summary.value:
            if value.tag.startswith(f"LearningAudit/{stationary_group}/stationary_10s/"):
                field = value.tag.rsplit("/", 1)[1]
                value.simple_value = {"samples": 2 if update == 81 else 6,
                    "endpoint_mean_m": .25 if update == 81 else .5,
                    "peak_mean_m": .5 if update == 81 else .75,
                    "peak_max_m": .75 if update == 81 else .9,
                    "mean_speed_m_s": .125 if update == 81 else .25}[field] if update in (81, 82) else 0.
        values = {}
        for group in transitions:
            for phase in ("braking", "settled_stop"):
                row = dict(samples=0., mean_speed_m_s=99., above_3cm_s_fraction=1., peak_speed_m_s=99.)
                if group == transitions[0] and phase == "braking" and update in (81, 82):
                    row = dict(samples=2 if update == 81 else 6,
                        mean_speed_m_s=.025 if update == 81 else .05,
                        above_3cm_s_fraction=.5 if update == 81 else .25,
                        peak_speed_m_s=.1 if update == 81 else .2)
                values.update({f"LearningAudit/{group}/{phase}/{field}": row[field] for field in STOP_FIELDS})
        for value in event.summary.value:
            if value.tag in values:
                value.simple_value = values[value.tag]
    series, audit = readback(reader, authority, event_file(tmp_path, events))
    report = reader.select_window(series, authority, audit, 50)
    stationary = report["stationary_10s"]["groups"][stationary_group]
    assert stationary["samples"] == 8
    assert stationary["endpoint_mean_m"] == pytest.approx(.4375)
    assert stationary["peak_mean_m"] == pytest.approx(.6875)
    assert stationary["peak_max_m"] == pytest.approx(.9)
    assert stationary["mean_speed_m_s"] == pytest.approx(.21875)
    empty = report["stationary_10s"]["groups"][authority["base_groups"][1]]
    assert empty["samples"] == 0 and all(empty[field] is None for field in STATIONARY_FIELDS[1:])
    phases = report["optional_conditional_stop_phases"]
    assert phases["complete"] is True and phases["missing"] == []
    braking = phases["groups"][transitions[0]]["braking"]
    assert braking["samples"] == 8
    assert braking["mean_speed_m_s"] == pytest.approx(.04375)
    assert braking["above_3cm_s_fraction"] == pytest.approx(.3125)
    assert braking["peak_speed_m_s"] == pytest.approx(.2)
    for group in transitions:
        for phase in ("braking", "settled_stop"):
            if group == transitions[0] and phase == "braking":
                continue
            zero = phases["groups"][group][phase]
            assert zero["samples"] == 0 and all(zero[field] is None for field in STOP_FIELDS[1:])


def test_nonfinite_valid_crc_scalar_is_rejected(reader, authority, full_events, tmp_path):
    events = deepcopy(full_events)
    value = next(value for value in events[124].summary.value if value.tag == "Policy/mean_std")
    value.simple_value = float("nan")
    with pytest.raises(ValueError, match="(?i)nonfinite"):
        readback(reader, authority, event_file(tmp_path, events))


@pytest.mark.parametrize("field", ["valid_transitions", "valid_fraction"])
def test_valid_crc_learning_ownership_is_not_actor_count_denominated(
        reader, authority, full_events, tmp_path, field):
    events = deepcopy(full_events)
    value = next(value for value in events[124].summary.value if value.tag == "RecoveryRollout/" + field)
    value.simple_value += 1.
    series, audit = readback(reader, authority, event_file(tmp_path, events))
    with pytest.raises(ValueError, match="(?i)(ownership|transition|fraction)"):
        reader.select_window(series, authority, audit, 100, end_update=130)
