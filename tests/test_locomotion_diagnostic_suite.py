"""Protocol coverage: true speed/yaw endpoints and full final stop intervals."""
from collections import Counter
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p) for p in ("scripts", "src")]
from locomotion_diagnostic_suite import DEFAULT_SUITE, HIGH_SPEED_SUITE, get_suite, suite_schedule
from probe_locomotion_exploration import schedule
from wheeled_tasks.chassis.playback_command import advance_playback_command


@pytest.mark.parametrize("name,speed", [("stand_305mm", 0.), ("forward_0p5", .5), ("backward_0p5", -.5)])
def test_original_protocol_requests_and_clocks_are_unchanged(name, speed):
    plan = suite_schedule(name)
    assert plan == [(phase, (vx, 0., .305)) for phase, vx in schedule(speed)]
    assert get_suite().num_envs == 12 and get_suite().policy_steps == 1000


@pytest.mark.parametrize("name", ["anything", "high_speed_braking_v2", "low_speed_430mm"])
def test_unknown_suite_cannot_expand_the_diagnostic(name):
    with pytest.raises(ValueError):
        get_suite(name)


def test_high_protocol_crosses_every_named_domain_endpoint():
    suite = get_suite(HIGH_SPEED_SUITE)
    assert (suite.num_envs, suite.policy_steps, suite.physics_steps, suite.replicas) == (64, 1600, 6400, 4)
    assert len(set(suite.case_names)) == 16
    straight = { (c.vx_m_s, c.height_m) for c in suite.cases if c.kind == "ordinary" and c.vx_m_s }
    assert straight == {(speed, height) for speed in (-5., -3., 3., 5.) for height in (.305, .43)}
    assert {(c.yaw_rad_s, c.height_m) for c in suite.cases if c.yaw_rad_s} == {
        (sign * rev * 2 * math.pi, height) for sign in (-1, 1) for rev, height in ((1, .305), (3, .43))}


@pytest.mark.parametrize("case", get_suite(HIGH_SPEED_SUITE).case_names)
def test_full_native_slew_stop_window_and_contiguous_phases(case):
    suite = get_suite(HIGH_SPEED_SUITE); spec = suite.case(case)
    plan = suite_schedule(case, suite.name)
    assert len(plan) == 1600
    assert plan[:100] == [("settle", (0., 0., .305))] * 100
    if spec.height_m == .43:
        assert plan[100:200] == [("height_settle", (0., 0., .43))] * 100
    phases = [p for p, _ in plan]
    groups = []
    for phase in phases:
        if not groups or groups[-1] != phase:
            groups.append(phase)
    assert len(groups) == len(set(groups))
    previous = torch.tensor([[0., 0., .305]], dtype=torch.float64)
    visible = []
    for _, command in plan:
        requested = previous.new_tensor([command])
        current = advance_playback_command(previous, requested, .02,
            {"vx_m_s2": 1.5, "yaw_rad_s2": 4.}, None)
        assert abs(float(current[0, 0] - previous[0, 0])) <= .03 + 1e-12
        assert abs(float(current[0, 1] - previous[0, 1])) <= .08 + 1e-12
        previous = current; visible.append(current[0].tolist())
    if spec.vx_m_s or spec.yaw_rad_s:
        stop = [i for i, phase in enumerate(phases) if phase == "stop"]
        assert len(stop) == 500 and stop == list(range(stop[0], stop[0] + 500))
        assert all(abs(visible[i][0]) < 1e-10 and abs(visible[i][1]) < 1e-10 for i in stop)
        steady = [i for i, phase in enumerate(phases) if phase == "steady"]
        assert all(visible[i][:2] == pytest.approx([spec.vx_m_s, spec.yaw_rad_s]) for i in steady)
        brake = phases.index("braking")
        assert plan[brake][1][:2] == (0., 0.)
        assert abs(visible[brake][0]) + abs(visible[brake][1]) > 0.


def test_interrupted_brake_really_reaccelerates_before_final_release():
    plan = suite_schedule("cancel_forward_5_305mm", HIGH_SPEED_SUITE)
    counts = Counter(phase for phase, _ in plan)
    assert counts["cancelled_acceleration"] == counts["cancelled_braking"] == 50
    assert counts["interrupted_braking"] == counts["renewed_drive"] == 25
    assert counts["braking"] == 167 and counts["stop"] == 500
    assert len(plan) == 1600


def test_catalog_metadata_is_fresh_and_case_identity_strict():
    suite = get_suite(HIGH_SPEED_SUITE)
    receipt = suite.as_dict(); receipt["cases"][0]["height_m"] = 9
    assert suite.as_dict()["cases"][0]["height_m"] == .305
    with pytest.raises(ValueError):
        suite_schedule("forward_5_430mm", DEFAULT_SUITE)
