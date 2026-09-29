"""Sensor-only requests and independent single/two-platform physical evidence."""
import inspect

import pytest
import torch

from wheeled_tasks.chassis.step_jump import StepJumpTask


DT = .02


def make_task(*, source="manual", double=False, count=1, disabled=False):
    settings = {"rises_m": [.20, .25] if double else [.20],
        "treads_m": [.8, .6] if double else [.8], "launch_distance_m": .25,
        "approach_speed_m_s": .4, "landing_margin_m": .1, "settle_seconds": .35,
        "landing_clearance_m": .08, "trigger_source": source}
    groups = ["step"] * count + (["flat"] if disabled else [])
    return StepJumpTask(groups, {"step": {"step_jump": settings}}, "cpu", DT)


def vector(task, value, *, dtype=torch.float32):
    return torch.as_tensor(value, dtype=dtype).expand(task.count).clone()


def control(task, time, *, speed=.4, support=True, force=9.81, beam=float("inf"), valid=False, age=0., forward=None):
    return task.control(vector(task, time), vector(task, speed), vector(task, support, dtype=torch.bool),
        vector(task, force), vector(task, beam), vector(task, valid, dtype=torch.bool), vector(task, age),
        beam_forward_distance=None if forward is None else vector(task, forward))


def launch(task, time=0.):
    previous = task.request_count.clone()
    for _ in range(300):
        time += DT
        accepted = control(task, time, beam=.2, valid=True)
        if accepted.any():
            assert torch.equal(task.request_count, previous + task.enabled.long())
            return time
    raise AssertionError("No request accepted on preset approach")


def finish_command(task, time):
    before = task.command_completed.clone()
    time += DT
    control(task, time, speed=0., support=False, force=0., beam=.2, valid=True)
    for _ in range(100):
        time += DT
        control(task, time, speed=0., beam=.2, valid=True)
        if (task.command_completed > before).any():
            return time
    raise AssertionError("IMU flight plus sustained supported settling did not finish")


def observe(task, x, bottom, *, contacts=True, vz=0., stable=True, speed=0., armed=True, y=0.):
    centers = torch.zeros(task.count, 2, 3)
    centers[:, :, 0] = torch.as_tensor(x)
    centers[:, :, 1] = torch.as_tensor(y)
    centers[:, :, 2] = torch.as_tensor(bottom) + .06
    contact = torch.as_tensor(contacts, dtype=torch.bool).expand(task.count, 2).clone()
    result = task.observe(centers, contact, vector(task, vz), vector(task, speed),
        torch.tensor([[0., 0., -1.]]).expand(task.count, 3),
        vector(task, stable, dtype=torch.bool), vector(task, armed, dtype=torch.bool))
    return result, centers


def fly_and_land(task, *, index=0, valid=True):
    edge = 0. if index == 0 else .8
    top = .2 if index == 0 else .45
    observe(task, edge - .25, top - (.2 if index == 0 else .25))
    for x in (edge - .18, edge - .1, edge - .03, edge + .04):
        observe(task, x, top + .06, contacts=False, vz=.3)
    events = 0
    for _ in range(20):
        event, _ = observe(task, edge + .3, top if valid else top - .10)
        events += int(event.sum())
    return events


def test_manual_request_uses_encoder_approach_and_same_tick_is_idempotent():
    task = make_task()
    control(task, DT, speed=0.)
    assert not task.active.any()
    time = launch(task, DT)
    assert task.active.item() and task.accepted.item()
    assert task.odometry_x.item() >= -.25
    start, odometry = task.request_start.clone(), task.odometry_x.clone()
    control(task, time, speed=100.)
    assert task.request_count.item() == 1
    assert torch.equal(task.request_start, start)
    assert torch.equal(task.odometry_x, odometry)


@pytest.mark.parametrize("beam,valid,age", [(float("inf"), True, 0.), (float("nan"), True, 0.),
    (.20, False, 0.), (.20, True, .051), (.20, True, -.01), (.01, True, 0.), (.26, True, 0.)])
def test_laser_requires_fresh_valid_near_positive_return(beam, valid, age):
    task = make_task(source="laser")
    control(task, DT, beam=beam, valid=valid, age=age)
    assert not task.active.any()
    assert task.request_count.item() == task.index.item() == 0


def test_laser_uses_projected_forward_distance_for_trigger_and_landing_target():
    task = make_task(source="laser")
    # Oblique beam has a long slant range but its endpoint is 23 cm ahead of
    # the chassis; adding a second 10 cm sensor offset would be incorrect.
    event = control(task, DT, beam=.32, valid=True, forward=.23)
    assert event.item() and task.request_count.item() == 1
    expected = task.odometry_x.item() + .23 + .22
    assert task.target_odometry_x.item() == pytest.approx(expected)


@pytest.mark.parametrize("forward", [.251, 0., -.1, float("nan"), float("inf")])
def test_near_slant_range_does_not_override_invalid_or_far_projection(forward):
    task = make_task(source="laser")
    assert not control(task, DT, beam=.15, valid=True, forward=forward).item()
    assert task.request_count.item() == 0


def test_no_return_does_not_end_jump_and_request_held_through_imu_settling():
    task = make_task(source="laser")
    time = launch(task)
    # No flight evidence: elapsed time and a missing beam cannot complete it.
    for _ in range(80):
        time += DT
        control(task, time, speed=0.)
    assert task.active.item() and task.index.item() == 0
    time += DT
    control(task, time, speed=0., support=False, force=0.)
    for _ in range(10):
        time += DT
        control(task, time, speed=0.)
        assert task.active.item()
        assert task.landing_delta.item() == pytest.approx(.20)
    for _ in range(10):
        time += DT
        control(task, time, speed=0.)
    assert task.index.item() == task.command_completed.item() == 1
    assert not task.active.any() and task.command_vx.item() == 0.
    assert task.request_count.item() == 1


def test_single_jump_records_top_landing_once_and_two_jumps_require_both_events():
    single = make_task()
    launch(single)
    assert fly_and_land(single) == 1
    assert single.success.item()
    assert not observe(single, .3, .2)[0].item()
    double = make_task(double=True)
    time = launch(double)
    assert fly_and_land(double) == 1
    assert not double.success.item()
    time = finish_command(double, time)
    launch(double, time)
    assert double.assessment_index.item() == 1
    assert double.landing_delta.item() == pytest.approx(.25)
    assert fly_and_land(double, index=1) == 1
    assert double.success.item()
    assert double.completed.tolist() == [[True, True]]
    assert double.request_count.item() == 2


def test_second_valid_landing_cannot_hide_failed_first_platform():
    task = make_task(double=True)
    time = launch(task)
    assert fly_and_land(task, valid=False) == 0
    time = finish_command(task, time)
    launch(task, time)
    assert fly_and_land(task, index=1) == 1
    assert task.completed.tolist() == [[False, True]]
    assert not task.success.item()


@pytest.mark.parametrize("case", ["rolling", "wrong_top", "one_wheel", "outside", "one_front"])
def test_rolling_or_invalid_platform_contact_does_not_count(case):
    task = make_task()
    launch(task)
    observe(task, -.25, 0.)
    for x in (-.18, -.1, -.03, .04):
        bottom = [.26, .05] if case == "one_front" else .26
        observe(task, x, bottom, contacts=case == "rolling", vz=.3)
    for _ in range(25):
        x = .79 if case == "outside" else .3
        bottom = .30 if case == "wrong_top" else .2
        contacts = [True, False] if case == "one_wheel" else True
        observe(task, x, bottom, contacts=contacts)
    assert not task.completed.any()
    assert not task.success.any()


def test_early_low_ground_touchdown_cannot_be_followed_by_rolling_success():
    task = make_task()
    launch(task)
    observe(task, -.3, 0.)
    for x in (-.28, -.26, -.24, -.22):
        observe(task, x, .07, contacts=False, vz=.3)
    observe(task, -.2, 0.)  # This jump failed before reaching the step front.
    observe(task, .04, .2)  # Supported rolling motion must not become front clearance.
    for _ in range(25):
        observe(task, .3, .2)
    assert not task.success.item()


def test_control_is_independent_of_privileged_assessment_and_has_no_world_inputs():
    names = set(inspect.signature(StepJumpTask.control).parameters)
    assert not names & {"wheel_centers", "contacts", "world_position", "root_pose", "terrain_height"}
    passed, failed = make_task(double=True), make_task(double=True)
    time = launch(passed)
    assert launch(failed) == time
    fly_and_land(passed)
    fly_and_land(failed, valid=False)
    assert passed.completed[0, 0] and not failed.completed[0, 0]
    for step in range(100):
        time += DT
        for task in (passed, failed):
            control(task, time, speed=0., support=step != 0, force=0. if step == 0 else 9.81)
        for name in ("index", "active", "request_start", "command_completed", "command_vx", "landing_delta", "apex"):
            assert torch.equal(getattr(passed, name), getattr(failed, name)), name


def test_local_reset_preserves_other_lane_and_clears_event_evidence():
    task = make_task(count=2, disabled=True)
    launch(task)
    fly_and_land(task)
    before = {name: value[1].clone() for name, value in vars(task).items()
              if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == task.count}
    task.reset(torch.tensor([0]))
    for name, value in before.items():
        assert torch.equal(getattr(task, name)[1], value), name
    assert not task.active[0] and not task.completed[0].any() and not task.front_clear[0].any()
    assert task.request_count[0] == task.index[0] == 0
    assert torch.isinf(task.request_start[0])
    assert task.odometry_x[0] == pytest.approx(-.8)
    assert not task.enabled[2] and not task.active[2] and not task.success[2]


def test_reward_terms_are_finite_float_densities_with_disabled_lane_zero():
    task = make_task(disabled=True)
    launch(task)
    _, centers = observe(task, -.15, .1, contacts=False, vz=.2)
    terms = task.reward_terms(centers, torch.tensor([[0., 0., -1.]]).repeat(2, 1))
    assert terms.keys() == {"step_jump_landing_region", "step_jump_front_clearance",
                           "step_jump_lateral", "step_jump_platform_event"}
    for value in terms.values():
        assert value.dtype.is_floating_point and value.shape == (2,)
        assert torch.isfinite(value).all()
        assert value[1] == 0.


def test_laser_same_obstacle_without_encoder_progress_does_not_trigger_second_jump():
    task = make_task(source="laser", double=True)
    time = launch(task)
    time = finish_command(task, time)
    assert task.index.item() == 1
    for _ in range(5):
        time += DT
        control(task, time, speed=0., beam=.2, valid=True)
    assert task.request_count.item() == 1


def test_landing_region_without_completed_jump_cannot_pay_residence_reward():
    task = make_task()
    launch(task)
    _, centers = observe(task, .3, .2)
    terms = task.reward_terms(centers, torch.tensor([[0., 0., -1.]]))
    assert terms["step_jump_landing_region"].item() <= 0.
    assert terms["step_jump_platform_event"].item() == 0.
