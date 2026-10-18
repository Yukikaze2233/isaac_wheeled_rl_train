from copy import deepcopy

import torch

from wheeled_tasks.chassis.evaluation import RequestProtocolMetrics, grade_fixed_suite, request_protocol_checks


def episode(**updates):
    return {"request_events_delivered": 2, "accepted_request_count": 1, "repeat_request_count": 1,
        "cancel_request_count": 0, "cancel_phase": -1, "repeat_clock_reset_errors": 0,
        "completed_jump_count": 1, "successful_jump_count": 1, "command_sequence_steps": 0, "command_sequence_length": 0,
        "final_phase": 0, "seen_flight": True, **updates}


def case(kind="request_repeat"):
    return {"name": kind, "command": [0., 0., .305], "task": "jump", "request_protocol_case": kind,
            "skill": {"request_events": [{"time_s": .8, "type": "new"}, {"time_s": .9, "type": "repeat"}]}}


def test_ordinary_success_cannot_pass_without_protocol_episode_evidence():
    c = case()
    settings = {"protocol_id": "test", "cases": [c], "survival_rate_min": 1., "height_mae_m_max": .02,
                "velocity_mae_m_s_max": .1, "yaw_mae_rad_s_max": .1, "tilt_max_deg": 20., "stand_drift_m_max": .1}
    group = {"episodes": 2, "timeouts": 0, "successes": 2, "failures": 0, "boundary_truncations": 0,
             "frames": 100, "height_mae_m": 0., "vx_mae_m_s": 0., "yaw_mae_rad_s": 0., "tilt_max_deg": 0.,
             "stand_drift_max_m": 0., "termination_reasons": {}}
    result = grade_fixed_suite({"groups": {c["name"]: group}}, settings, 2)
    assert not result["passed"] and not result["cases"][c["name"]]["checks"]["request_protocol_evidence"]
    group["request_protocol_episodes"] = [episode(), episode()]
    assert grade_fixed_suite({"groups": {c["name"]: group}}, settings, 2)["passed"]


def test_one_good_replica_cannot_mask_clock_restart_or_missing_repeat():
    c = case()
    for bad in (episode(repeat_request_count=0), episode(repeat_clock_reset_errors=1), episode(final_phase=4)):
        checks = request_protocol_checks(c, {"request_protocol_episodes": [episode(), bad]}, 2)
        assert not all(checks.values())


def test_cancel_cases_check_actual_cancel_phase_and_post_cancel_landing():
    before = episode(repeat_request_count=0, cancel_request_count=1, cancel_phase=1,
                     completed_jump_count=0, successful_jump_count=0, seen_flight=False)
    c = case("request_cancel_before_takeoff")
    assert all(request_protocol_checks(c, {"request_protocol_episodes": [before]}, 1).values())
    assert not all(request_protocol_checks(c, {"request_protocol_episodes": [{**before, "seen_flight": True}]}, 1).values())
    airborne = episode(repeat_request_count=0, cancel_request_count=1, cancel_phase=2)
    c = case("request_cancel_airtime")
    assert all(request_protocol_checks(c, {"request_protocol_episodes": [airborne]}, 1).values())
    assert not all(request_protocol_checks(c, {"request_protocol_episodes": [{**airborne, "cancel_phase": 1}]}, 1).values())


def test_two_jumps_and_resume_motion_require_distinct_completed_behaviors():
    c = case("request_repeat_after_landing")
    assert not all(request_protocol_checks(c, {"request_protocol_episodes": [episode()]}, 1).values())
    two = episode(accepted_request_count=2, completed_jump_count=2, successful_jump_count=2)
    assert all(request_protocol_checks(c, {"request_protocol_episodes": [two]}, 1).values())
    tiny_hops = {**two, "successful_jump_count": 0}
    assert not all(request_protocol_checks(c, {"request_protocol_episodes": [tiny_hops]}, 1).values())
    c = case("move_stop_jump_resume")
    c["skill"]["command_sequence"] = [{}, {}, {}, {}]
    sequence = episode(command_sequence_steps=4, command_sequence_length=4,
                       initial_motion_tracking_s=.6, stop_tracking_s=.6, resumed_motion_tracking_s=1.2)
    assert all(request_protocol_checks(c, {"request_protocol_episodes": [sequence]}, 1).values())
    sequence["resumed_motion_tracking_s"] = .1
    assert not all(request_protocol_checks(c, {"request_protocol_episodes": [sequence]}, 1).values())


def test_protocol_metrics_capture_final_counter_once_and_ignore_legacy_diagnostics():
    metric = RequestProtocolMetrics(["a"], "cpu", .02)
    metric.observe({}, torch.tensor([True]))
    raw = {"groups": {"a": {}}}
    assert metric.report_into(deepcopy(raw))["groups"]["a"]["request_protocol_episodes"] == []
    data = {k: torch.tensor([v]) for k, v in episode().items() if k in RequestProtocolMetrics.COUNTERS}
    data.update(episode_ticks=torch.tensor([1]), physical_phase=torch.tensor([2]),
                velocity=torch.zeros(1, 3), commands=torch.tensor([[0., 0., .305]]), done=torch.tensor([False]))
    metric.observe(data, torch.tensor([True]))
    data.update(episode_ticks=torch.tensor([2]), physical_phase=torch.tensor([0]), done=torch.tensor([True]))
    metric.observe(data, torch.tensor([True]))
    metric.observe(data, torch.tensor([False]))
    rows = metric.report_into(raw)["groups"]["a"]["request_protocol_episodes"]
    assert len(rows) == 1 and rows[0]["seen_flight"] and rows[0]["completed_jump_count"] == 1
