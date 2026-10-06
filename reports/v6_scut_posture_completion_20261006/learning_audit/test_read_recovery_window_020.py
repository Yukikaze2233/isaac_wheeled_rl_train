"""CPU fixtures:020 identity adaptation, native CRC and strict50/100 cadence."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def adapter():
    return load_file("source020_adapter_under_test", HERE / "read_recovery_window_020.py")


@pytest.fixture(scope="module")
def original_fixture():
    # Reuse019's real TFRecord writer and scalar producers; never duplicate CRC.
    return load_file("source019_crc_fixture_for_020", HERE / "test_read_recovery_window_019.py")


@pytest.fixture(scope="module")
def reader(adapter):
    return adapter.make_reader()


@pytest.fixture(scope="module")
def source():
    receipt = json.loads((HERE.parent / "source_repair_020/source_candidate_020.json").read_text())
    return Path(receipt["clean_root"]).resolve(strict=True)


@pytest.fixture(scope="module")
def authority(adapter, source):
    config = json.loads((source / "contracts/v6_task_campaign_recovery_worker_v18.json").read_text())
    groups = [row["name"] for row in config["scene_groups"]]
    pools = config["behavior_pool_membership"]
    bases = [g for g in groups if pools[g] != "scripted_recovery"]
    aliases = [g for g in bases if pools[g] in {"translation", "transitions", "yaw"}]
    permitted = [g for g in bases if g not in aliases]
    population = {}
    for members, total in ((permitted, 577), (aliases, 872), ([g for g in groups if g not in bases], 2647)):
        quotient, remainder = divmod(total, len(members))
        population.update({g: quotient + int(i < remainder) for i, g in enumerate(members)})
    return dict(groups=groups, base_groups=bases, group_pools=pools, population=population,
        identity_verified=True, source_commit=adapter.COMMIT, worker_pid=adapter.EXPECTED_PID,
        clocks=dict(event_step_origin=adapter.CONSUMPTION_ORIGIN, task_actor_origin=adapter.ACTOR_ORIGIN,
                    critic_warmup_updates=0),
        progress=dict(parent_updates=adapter.CONSUMPTION_ORIGIN, num_envs=4096, pid=adapter.EXPECTED_PID,
            task_actor_update=adapter.ACTOR_ORIGIN + 120, actor_updates_in_block=120, successful_updates=120))


def set_updates(authority, adapter, updates):
    result = deepcopy(authority)
    result["progress"].update(actor_updates_in_block=updates, successful_updates=updates,
        task_actor_update=adapter.ACTOR_ORIGIN + updates)
    return result


def native_events(common, adapter, authority, updates):
    events = []
    for update in range(1, updates + 1):
        values = common.rollout_values(authority, update)
        if update % 10 == 0:
            values.update(common.behavior_values(authority, update))
        events.append(common.scalar_event(adapter.CONSUMPTION_ORIGIN + update - 1, values))
    return events


@pytest.fixture(scope="module")
def full_series(tmp_path_factory, original_fixture, adapter, reader, authority):
    path = original_fixture.event_file(tmp_path_factory.mktemp("native020crc"),
        native_events(original_fixture, adapter, authority, 120))
    return reader.parse_events(path, reader.wanted_tags(authority), adapter.CONSUMPTION_ORIGIN + 119,
        min_step=adapter.BASELINE_STEP)


@pytest.fixture
def admitted_attempt(tmp_path, original_fixture, adapter, source, authority):
    root = tmp_path / "campaign"
    run = root / "train/recovery/attempt_004/train"
    run.mkdir(parents=True)
    revision = dict(schema="task-source-revision-v1", source_commit=adapter.COMMIT,
        new_source_root=str(source), old_source_root=str(root / "source_712d701473f3"),
        source_archive_sha256=adapter.ARCHIVE_SHA256,
        old_contract_sha256=adapter.PLAN_SHA256, new_contract_sha256=adapter.PLAN_SHA256)
    invocation = dict(command=["/synthetic/cpu/python", "-B", str(source / "scripts/train_chassis.py"),
        "--contract", str(source / "contracts/v6_task_campaign_recovery_worker_v18.json"),
        "--run-dir", str(run), "--num-envs", "4096", "--seed", "290927", "--block-index", "0",
        "--task-actor-update", str(adapter.ACTOR_ORIGIN), "--consumed-updates", str(adapter.CONSUMPTION_ORIGIN),
        "--stage", "mixed"])
    progress = set_updates(authority, adapter, 65)["progress"]
    startup = dict(pid=adapter.EXPECTED_PID, startup=dict(runtime_preflight=dict(
        status="runtime_configuration_passed", observed=dict(physics_dt=.005, policy_dt=.02, decimation=4, seed=290927)),
        scene_group_counts=authority["population"], ordinary_posture_curriculum=dict(origin_actor_update=2294,
            runtime_observed=dict(eligible_count=1449, birth_eligible_population=1449))))
    paths = dict(revision=root / "source_revision_020.json", invocation=run.parent / "invocation.json",
        progress=run / "progress.json", startup=run / "startup.json")
    for name, value in dict(revision=revision, invocation=invocation, progress=progress, startup=startup).items():
        original_fixture.write_json(paths[name], value)
    event = run / f"events.out.tfevents.1700000000.synthetic-host.{adapter.EXPECTED_PID}.0"
    event.write_bytes(original_fixture.record_bytes(original_fixture.Event(file_version="brain.Event:2")))
    return dict(root=root, run=run, source=source, paths=paths, event=event)


def test_exact_parser_metric_code_is_reused_and_old_globals_stay_isolated(adapter):
    old = load_file("unchanged_source019_reader", HERE / "read_recovery_window_019.py")
    original_hash = adapter.digest(Path(old.__file__))
    reader = adapter.make_reader()
    for name in adapter.REUSED_FUNCTIONS:
        assert getattr(reader, name).__code__ == getattr(old, name).__code__
        assert getattr(reader, name).__code__.co_filename == old.__file__
    assert old.COMMIT.startswith("712d701") and old.EXPECTED_PID == 3480218
    assert old.window_spec(50, 60)["start_step"] == 29389
    assert reader.window_spec(50, 60)["start_step"] == 29689
    assert "root_kinematics.py" not in " ".join(old.SOURCE_HASHES)
    assert adapter.digest(Path(old.__file__)) == original_hash == adapter.BASE_READER_SHA256


def test_substituted_base_reader_bytes_are_rejected_before_exec(adapter, tmp_path):
    path = tmp_path / "reader.py"
    path.write_text("raise RuntimeError('must never execute')\n")
    with pytest.raises(ValueError, match="reader SHA"):
        adapter.make_reader(path)


def test_authenticates_current_attempt_header_population_origins_and_sources(adapter, reader, admitted_attempt):
    a = admitted_attempt
    result = adapter.load_authority(reader, a["run"], a["source"])
    assert result["source_commit"] == adapter.COMMIT and result["worker_pid"] == 3494096
    assert result["clocks"] == dict(event_step_origin=29679, task_actor_origin=2758, critic_warmup_updates=0)
    assert result["ordinary_posture_coverage"]["native_eligible_population"] == 1449
    assert result["ordinary_posture_coverage"]["legacy_alias_population_now_eligible"] == 872
    assert result["ordinary_posture_coverage"]["nonstandard_start_leg_bank_present"] is False
    assert len(result["source_files_sha256"]) == 13
    assert result["source_revision"]["ordinary_origin_actor_update"] == 2294
    assert result["event"] == str(a["event"])


@pytest.mark.parametrize("name,key,value,match", [
    ("revision", "source_commit", "712d701473f38ef6797725f8efdda9e6beb85e87", "frozen revision"),
    ("revision", "new_source_root", "/another/source", "frozen revision"),
    ("revision", "source_archive_sha256", "0" * 64, "archive/unchanged plan"),
    ("revision", "new_contract_sha256", "0" * 64, "archive/unchanged plan"),
    ("revision", "old_source_root", "/another/predecessor", "predecessor"),
    ("progress", "pid", 3480218, "worker/PID"),
    ("progress", "parent_updates", 29379, "origin/population"),
    ("progress", "task_actor_update", 2523, "actor clocks"),
    ("progress", "successful_updates", 66, "zero new critic warmup"),
    ("startup", "pid", 3480218, "another worker"),
])
def test_wrong_identity_resume_clock_or_contract_is_rejected(
        adapter, reader, admitted_attempt, original_fixture, name, key, value, match):
    a = admitted_attempt
    data = json.loads(a["paths"][name].read_text()); data[key] = value
    original_fixture.write_json(a["paths"][name], data)
    with pytest.raises(ValueError, match=match):
        adapter.load_authority(reader, a["run"], a["source"])


def test_old_attempt_path_and_duplicate_invocation_flag_rejected(adapter, reader, admitted_attempt, original_fixture):
    a = admitted_attempt
    with pytest.raises(ValueError, match="attempt"):
        adapter.load_authority(reader, a["root"] / "train/recovery/attempt_003/train", a["source"])
    data = json.loads(a["paths"]["invocation"].read_text())
    data["command"].extend(["--task-actor-update", "2758"])
    original_fixture.write_json(a["paths"]["invocation"], data)
    with pytest.raises(ValueError, match="duplicate invocation flag"):
        adapter.load_authority(reader, a["run"], a["source"])


def test_changed_frozen_com_implementation_is_rejected_without_modifying_source(adapter, admitted_attempt):
    reader = adapter.make_reader()
    reader.SOURCE_HASHES["src/wheeled_tasks/chassis/root_kinematics.py"] = "0" * 64
    with pytest.raises(ValueError, match="implementation differs"):
        adapter.load_authority(reader, admitted_attempt["run"], admitted_attempt["source"])


@pytest.mark.parametrize("width,updates,interval,laststep", [
    (50, 60, [2769, 2818], 29738), (100, 110, [2769, 2868], 29788),
])
def test_complete_windows_reuse_native_crc_cadence_and_exact_actor_clock(
        adapter, reader, authority, full_series, width, updates, interval, laststep):
    series, audit = full_series
    report = reader.select_window(series, set_updates(authority, adapter, updates), audit, width)
    assert report["complete"] is True and report["missing_required_tag_count"] == 0
    assert report["window"]["task_actor_updates_inclusive"] == interval
    assert report["window"]["baseline_step"] == 29688 and report["window"]["end_step"] == laststep
    assert len(report["window"]["behavior_steps"]) == width // 10 + 1
    assert len(report["window"]["dense_steps"]) == width
    assert report["window"]["expected_observed_frames"] == width * 98304
    assert audit["crc_valid_records"] == 121
    assert report["yaw_observability"]["actual_yaw_rate_available"] is False
    assert report["dynamic_lessons"]["counter_scope"].startswith("24_policy_frame_mean")


@pytest.mark.parametrize("available", [22, 50, 59, 65, 100, 109])
def test_hundred_window_never_certified_from_short_or_unbaselined_attempt(
        adapter, reader, authority, full_series, available):
    series, audit = full_series
    report = reader.select_window(series, set_updates(authority, adapter, available), audit, 100)
    assert report["complete"] is False
    assert report["selection"]["additional_actor_updates_needed"] == 110 - available
    assert report["window"]["baseline_update"] == 10
    assert report["selection"]["native_zero_baseline_fabricated"] is False


@pytest.mark.parametrize("tag,update", [("Policy/mean_std", 31), ("Behavior/PLACEHOLDER/frames", 30)])
def test_dense_or_behavior_cadence_hole_blocks_certification(adapter, reader, authority, full_series, tag, update):
    series, audit = full_series
    series = deepcopy(series)
    tag = tag.replace("PLACEHOLDER", authority["groups"][0])
    del series[tag][adapter.CONSUMPTION_ORIGIN + update - 1]
    report = reader.select_window(series, set_updates(authority, adapter, 60), audit, 50)
    assert report["complete"] is False and any(row["tag"] == tag for row in report["missing"])


def test_native_crc_corruption_is_rejected_by_original_parser(adapter, reader, original_fixture, authority, tmp_path):
    path = original_fixture.event_file(tmp_path, native_events(original_fixture, adapter, authority, 1))
    payload = bytearray(path.read_bytes()); payload[-5] ^= 1; path.write_bytes(payload)
    with pytest.raises(ValueError, match="payload CRC"):
        reader.parse_events(path, reader.wanted_tags(authority), 29679, min_step=adapter.BASELINE_STEP)


def test_real_completed_stationary_samples_remain_weighted_and_unselected_zero_unavailable(
        adapter, reader, authority, full_series):
    series, audit = deepcopy(full_series)
    group = authority["base_groups"][0]
    prefix = f"LearningAudit/{group}/stationary_10s/"
    for update, count, value in ((11, 2, .01), (12, 1, .07)):
        step = adapter.CONSUMPTION_ORIGIN + update - 1
        for field in reader.STATIONARY_FIELDS:
            series[prefix + field][step][0] = count if field == "samples" else value
    report = reader.select_window(series, set_updates(authority, adapter, 60), audit, 50)
    row = report["stationary_10s"]["groups"][group]
    assert row["samples"] == 3 and row["mean_speed_m_s"] == pytest.approx(.03)
    assert report["stationary_10s"]["groups"][authority["base_groups"][1]]["samples"] == 0
    assert report["stationary_10s"]["groups"][authority["base_groups"][1]]["mean_speed_m_s"] is None


def test_cli_latest65_preserves_full50_pending100_and_does_not_create_an_event(
        adapter, original_fixture, authority, admitted_attempt, tmp_path):
    a = admitted_attempt
    a["event"].write_bytes(original_fixture.record_bytes(original_fixture.Event(file_version="brain.Event:2"))
        + b"".join(original_fixture.record_bytes(event) for event in native_events(original_fixture, adapter, authority, 65)))
    before = adapter.digest(a["event"])
    output = tmp_path / "summary.json"
    code = adapter.main(["--run-dir", str(a["run"]), "--source-root", str(a["source"]), "--output", str(output)])
    report = json.loads(output.read_text())
    assert code == 0 and report["complete"] is False and report["status"] == "pending_requested_actor_windows"
    assert report["windows"]["50"]["complete"] is True and report["windows"]["100"]["complete"] is False
    assert report["windows"]["50"]["selection"]["chosen_end_update"] == 60
    assert report["windows"]["100"]["selection"]["additional_actor_updates_needed"] == 45
    assert report["source_files_unchanged"] is True and report["qualification"] is None
    assert adapter.digest(a["event"]) == before and len(list(a["run"].glob("events.out.tfevents.*"))) == 1
    assert report["base_reader_expected_sha256"] == adapter.BASE_READER_SHA256
    assert "COM" in " ".join(report["source020_context"])
    assert report["windows"]["50"]["curve_source"] == "existing_native_TensorBoard_event_only"
    with pytest.raises(ValueError, match="already exists"):
        adapter.main(["--output", str(output)])


def test_cli_wrong_source_emits_explicit_read_failed_receipt(adapter, admitted_attempt, tmp_path):
    output = tmp_path / "rejected.json"
    code = adapter.main(["--run-dir", str(admitted_attempt["run"]), "--source-root", str(tmp_path), "--output", str(output)])
    report = json.loads(output.read_text())
    assert code == 2 and report["status"] == "read_failed" and report["complete"] is False
    assert report["error"] and report["source_commit"] == adapter.COMMIT and "windows" not in report
