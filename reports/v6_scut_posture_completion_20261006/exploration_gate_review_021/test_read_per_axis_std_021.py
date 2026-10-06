"""Focused CPU fixtures: existing-tag selection, exact clocks, units, omissions and CRC."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = Path("/tmp/scut-recovery-frozen-021-qv1ukmbr")
spec = importlib.util.spec_from_file_location("per_axis_std_021_under_test", HERE / "read_per_axis_std_021.py")
sidecar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sidecar)


@pytest.fixture
def helper():
    return sidecar.load_reader(sidecar.DEFAULT_READER)


@pytest.fixture
def authority():
    return dict(groups=[f"g{i}" for i in range(95)], population={f"g{i}": 40 for i in range(95)},
        clocks=dict(consumption_origin=29996, task_actor_origin=3075, critic_warmup_updates=10,
            first_actor_event_step=30006, actor_completed=113))


@pytest.fixture
def layout(helper):
    cfg = json.loads((SOURCE / helper.WORKER).read_text())
    first = json.loads((HERE.parent / "source_repair_021/first_PPO_verification_021.json").read_text())
    binding = first["checks"]["actual_native_full_restore_new_modules_origins_and_clocks"]["detail"]["native_hold_cost_gate_binding"]
    return sidecar.action_layout(SOURCE, cfg, binding)


@pytest.fixture
def reader(helper, authority):
    return helper.make_reader(authority["clocks"])


@pytest.fixture
def series(authority):
    rows = {}
    for tag in sidecar.wanted_tags(authority["groups"]):
        value = .4 if tag in sidecar.STD_TAGS or tag == sidecar.AGGREGATE_STD else .01
        rows[tag] = {step: [value, float(step)] for step in range(30016, 30116)}
    return rows


def test_existing_reader_bytes_unchanged_and_pinned(helper):
    assert sidecar.digest(sidecar.DEFAULT_READER) == sidecar.READER021_SHA
    assert sidecar.digest(helper.OLD020) == helper.READER020_SHA
    assert sidecar.digest(helper.OLD019) == helper.load020().BASE_READER_SHA256


def test_exact_actor_window_excludes_critics_and_first_ten_actors(authority):
    window = sidecar.actor_window(authority["clocks"])
    assert window["dense_steps"] == list(range(30016, 30116))
    assert window["task_actor_updates_inclusive"] == [3086, 3185]
    assert window["attempt_actor_updates_inclusive"] == [11, 110]
    assert window["authoritative_parent_consumption"] == 29996


@pytest.mark.parametrize("name,value", [("consumption_origin", 30006), ("task_actor_origin", 3065),
    ("critic_warmup_updates", 0), ("first_actor_event_step", 29996)])
def test_clock_fabrication_rejected(authority, name, value):
    authority["clocks"][name] = value
    with pytest.raises(ValueError):
        sidecar.actor_window(authority["clocks"])


def test_std_axis_is_policy_order_and_units(layout):
    assert [row["joint_name"] for row in layout["axes"]] == [
        "L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"]
    assert [row["control_axis"] for row in layout["axes"]] == [0, 1, 3, 4, 2, 5]
    assert [row["clipped_target_absolute_bound"] for row in layout["axes"]] == [.75] * 4 + [90.] * 2
    assert sidecar.scale_std(.4, layout["axes"][0])["unclipped_linear_target_std"] == .1
    wheel = sidecar.scale_std(.4, layout["axes"][4])
    assert wheel["unclipped_linear_target_std"] == 4.
    assert wheel["unclipped_wheel_rim_target_std_m_s"] == .24
    assert sidecar.scale_std(.4, layout["axes"][0])["unclipped_wheel_rim_target_std_m_s"] is None


@pytest.mark.parametrize("field,value", [("leg_position_scale", .5), ("wheel_velocity_scale", 50.),
    ("action_clip", 9.), ("wheel_action_clip", 3.)])
def test_native_scale_or_clip_change_rejected(helper, field, value):
    cfg = json.loads((SOURCE / helper.WORKER).read_text())
    cfg["v5_control"][field] = value
    with pytest.raises(ValueError):
        sidecar.action_layout(SOURCE, cfg, dict(status="authenticated_native_fsm_and_geometry", wheel_radius_m=.06))


def test_native_radius_change_rejected(helper):
    cfg = json.loads((SOURCE / helper.WORKER).read_text())
    with pytest.raises(ValueError):
        sidecar.action_layout(SOURCE, cfg, dict(status="authenticated_native_fsm_and_geometry", wheel_radius_m=.08))


def test_complete100_points_and_two_halves(reader, authority, layout, series):
    report = sidecar.summarize(reader, series, authority, layout, {})
    assert report["complete"] and report["qualification"] is None
    assert len(report["metrics"]) == 103
    for metric in report["metrics"].values():
        assert metric["full100"]["records"] == 100
        assert metric["first50"]["records"] == metric["last50"]["records"] == 50
        assert metric["points"][0]["task_actor_update"] == 3086
        assert metric["points"][-1]["task_actor_update"] == 3185
    assert report["aggregate_std_reconciliation"]["all_comparable_passed"]
    assert report["global_group_KL_reconciliation"]["all_comparable_passed"]


@pytest.mark.parametrize("tag", sidecar.STD_TAGS + (sidecar.GLOBAL_KL, "DiagnosticUpdate/g0/kl"))
def test_missing_tag_or_point_never_imputed_zero(reader, authority, layout, series, tag):
    del series[tag][30050]
    report = sidecar.summarize(reader, series, authority, layout, {})
    assert not report["complete"]
    metric = report["metrics"][tag]
    assert metric["full100"]["mean"] is None
    assert metric["first50"]["mean"] is None
    assert metric["last50"]["mean"] is not None
    assert metric["points"][34]["value"] is None
    assert dict((row["tag"], row["missing_steps"]) for row in report["missing_required_tags"])[tag] == [30050]


def test_no_current_clock_completion_fabrication(reader, authority, layout, series):
    authority["clocks"]["actor_completed"] = 109
    report = sidecar.summarize(reader, series, authority, layout, {})
    assert not report["complete"] and report["window"]["actor_updates_needed"] == 1


def test_aggregate_cannot_replace_missing_six_axes(reader, authority, layout, series):
    for tag in sidecar.STD_TAGS:
        series.pop(tag)
    report = sidecar.summarize(reader, series, authority, layout, {})
    assert not report["complete"]
    assert report["aggregate_std_reconciliation"]["comparable_steps"] == 0
    assert all(row["full100"]["raw_action_std"] is None for row in report["physical_target_scales"])


def test_std_comparison_mismatch_is_evidence_not_rewriting(reader, authority, layout, series):
    series[sidecar.AGGREGATE_STD][30016][0] = 1.
    report = sidecar.summarize(reader, series, authority, layout, {})
    assert report["complete"]
    assert not report["aggregate_std_reconciliation"]["all_comparable_passed"]
    assert report["metrics"][sidecar.AGGREGATE_STD]["points"][0]["value"] == 1.


def test_KL_group_weight_matches_actual_native_probe_lane_selection(reader, authority, layout, series):
    authority["population"]["g0"] = 1
    for point in series["DiagnosticUpdate/g0/kl"].values():
        point[0] = .2
    expected = (.2 + .01 * 94 * 16) / (1 + 94 * 16)
    for point in series[sidecar.GLOBAL_KL].values():
        point[0] = expected
    report = sidecar.summarize(reader, series, authority, layout, {})
    assert report["global_group_KL_reconciliation"]["all_comparable_passed"]
    assert report["diagnostic_probe"]["selected_lanes_per_group"]["g0"] == 1


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_std_rejected(value, layout):
    with pytest.raises(ValueError):
        sidecar.scale_std(value, layout["axes"][0])


def framed_record(reader, step, tag, value):
    event = reader.Event(step=step, wall_time=float(step))
    event.summary.value.add(tag=tag, simple_value=value)
    payload = event.SerializeToString()
    length = struct.pack("<Q", len(payload))
    return length + struct.pack("<I", reader.masked_crc32c(length)) + payload + struct.pack("<I", reader.masked_crc32c(payload))


def test_reused_real_CRC_parser_diagnostic_tag_missing_and_complete_points(reader, authority, tmp_path):
    path = tmp_path / "native.synthetic.event"
    tag = sidecar.STD_TAGS[4]
    records = [framed_record(reader, step, tag, .4) for step in range(30016, 30116)]
    path.write_bytes(b"".join(records))
    series, audit = reader.parse_events(path, {tag}, 30115, min_step=30016)
    assert len(series[tag]) == 100 and audit["crc_valid_records"] == 100
    assert audit["complete_record_region_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reader.scalar(series, sidecar.STD_TAGS[5], 30016) is None


def test_reused_CRC_parser_rejects_corrupt_payload(reader, tmp_path):
    data = bytearray(framed_record(reader, 30016, sidecar.GLOBAL_KL, .01))
    data[15] ^= 1
    path = tmp_path / "corrupt.synthetic.event"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="payload CRC"):
        reader.parse_events(path, {sidecar.GLOBAL_KL}, 30115, min_step=30016)


def test_existing_tag_selection_does_not_canonicalize_unrelated_namespaces(helper):
    assert helper.canonical_extra_tag("DiagnosticUpdate/action_std_0", lambda tag: tag) == "DiagnosticUpdate/action_std_0"

