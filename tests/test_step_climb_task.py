"""Ordinary forward climbing: top support allowed, route evidence remains ordered."""
import pytest
import torch

from wheeled_tasks.chassis.step_climb import StepClimbTask
from wheeled_tasks.chassis.rewards import apply_climb_height_tracking


DT = .02


def make_task(*, double=False, count=1, disabled=False):
    cfg = {"rises_m": [.20, .25] if double else [.20],
        "treads_m": [.8, .6] if double else [.8], "approach_speed_m_s": .3,
        "body_contact_min_force_n": 20., "support_hold_seconds": .10,
        "settle_seconds": .35, "landing_margin_m": .06}
    return StepClimbTask(["climb"] * count + (["flat"] if disabled else []),
        {"climb": {"body_climb": cfg}}, "cpu", DT)


def wheels(task, x, bottom, y=0.):
    result = torch.zeros(task.count, 2, 3)
    result[:, :, 0], result[:, :, 1], result[:, :, 2] = torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(bottom) + .06
    return result


def observe(task, x, bottom, *, contacts=True, stable=True, y=0., points=None, forces=None):
    return task.observe(wheels(task, x, bottom, y),
        torch.as_tensor(contacts, dtype=torch.bool).expand(task.count, 2),
        torch.as_tensor(stable, dtype=torch.bool).expand(task.count),
        body_bottom_points=points, body_contact_forces=forces)


def ascend(task, index=0):
    edge, top = (0., .20) if index == 0 else (.8, .45)
    observe(task, edge - .15, top - (.2 if index == 0 else .25))
    observe(task, edge + .07, top)
    events = 0
    # Keep rolling at 0.4 m/s: completion does not require stopping.
    for tick in range(25):
        events += int(observe(task, edge + .08 + .4 * DT * tick, top).sum())
    return events


def test_forward_only_climb_has_no_control_or_request_interface():
    task = make_task()
    assert not hasattr(task, "control")
    assert not hasattr(task, "command_vx")
    assert not hasattr(task, "request_count")
    assert ascend(task) == 1
    assert task.success.item()
    assert not observe(task, .5, .2).item()


def test_double_climb_requires_ordered_support_of_both_actual_tops():
    task = make_task(double=True)
    assert task.tops.tolist()[0] == pytest.approx([.2, .45])
    assert task.edges.tolist()[0] == pytest.approx([0., .8])
    assert ascend(task) == 1
    assert task.index.item() == 1 and not task.success.item()
    assert ascend(task, 1) == 1
    assert task.success.item()
    assert task.completed.tolist() == [[True, True]]


def test_direct_skip_to_second_top_gets_no_first_or_second_credit():
    task = make_task(double=True)
    observe(task, -.2, 0.)
    for _ in range(40):
        observe(task, 1.05, .45)
    assert task.index.item() == 0
    assert not task.completed.any() and not task.success.item()


@pytest.mark.parametrize("case", ["wrong_height", "one_wheel", "outside", "unstable", "wide"])
def test_invalid_wheel_support_never_completes_platform(case):
    task = make_task()
    observe(task, -.2, 0.)
    observe(task, .1, .2)
    for _ in range(30):
        observe(task, .78 if case == "outside" else .3, .3 if case == "wrong_height" else .2,
            contacts=[True, False] if case == "one_wheel" else True,
            stable=case != "unstable", y=1.6 if case == "wide" else 0.)
    assert not task.success.item()


def test_top_chassis_support_is_optional_for_completion_and_cannot_replace_wheels():
    task = make_task()
    points = torch.tensor([[[.15, 0., .2]]])
    forces = torch.tensor([[[0., 0., 35.]]])
    for _ in range(30):
        observe(task, -.1, 0., points=points, forces=forces)
    assert task.body_contact_allowed.item()
    assert task.diagnostics()["body_climb_body_supported"].item()
    assert not task.completed.any()
    assert ascend(task) == 1
    assert task.success.item()


def test_legal_chassis_support_allows_wheels_to_hang_without_becoming_a_jump():
    task = make_task(double=True)
    points, forces = torch.tensor([[[.15, 0., .2]]]), torch.tensor([[[0., 0., 35.]]])
    for _ in range(20):
        observe(task, -.1, .1, contacts=False, points=points, forces=forces)
    assert not task.flight_violation.item() and task.air_peak.item() == 0.
    assert ascend(task) == 1
    # The preceding top may still support the chassis during the next climb.
    for _ in range(20):
        observe(task, .7, .3, contacts=False, points=points, forces=forces)
    assert not task.flight_violation.item()
    assert ascend(task, 1) == 1 and task.success.item()


def test_full_airborne_jump_latches_and_cannot_count_as_ordinary_climb():
    task = make_task()
    observe(task, -.2, 0.)
    for tick in range(4):
        observe(task, -.1 + .08 * tick, .3, contacts=False)
    assert task.flight_violation.item()
    assert task.diagnostics()["body_climb_air_peak_s"].item() == pytest.approx(.08)
    assert ascend(task) == 0 and not task.success.item()
    assert not task.completed.any()
    assert task.diagnostics()["body_climb_flight_violation"].item()
    assert task.height_tracking_scale(wheels(task, -.1, .2)).item() == 1.
    terms = task.reward_terms(wheels(task, -.1, .2), torch.tensor([[0., 0., -1.]]))
    assert all(value.eq(0.).all() for value in terms.values())


def test_short_reset_drop_and_contact_chatter_do_not_accumulate_into_flight():
    task = make_task()
    for _ in range(3):
        observe(task, -.2, .004, contacts=False)
    observe(task, -.2, 0.)
    for _ in range(8):
        for _ in range(2):
            observe(task, -.2, .004, contacts=False)
        observe(task, -.2, 0., contacts=[True, False])
    assert not task.flight_violation.item()
    assert task.air_peak.item() == pytest.approx(.06)
    assert ascend(task) == 1 and task.success.item()


@pytest.mark.parametrize("point,force", [([-.01, 0., .2], [35., 0., 0.]),
                                       ([.15, 0., 0.], [0., 0., 35.])])
def test_riser_or_ground_contact_does_not_hide_a_climbing_flight(point, force):
    task = make_task()
    for _ in range(4):
        observe(task, -.1, .3, contacts=False,
            points=torch.tensor([[point]]), forces=torch.tensor([[force]]))
    assert task.flight_violation.item()


def test_second_platform_cannot_hide_flight_after_first_and_reset_is_local():
    task = make_task(double=True, count=2)
    assert ascend(task) == 2
    for _ in range(4):
        observe(task, .7, .3, contacts=[[False, False], [True, False]])
    assert task.flight_violation.tolist() == [True, False]
    assert ascend(task, 1) == 1
    assert task.completed.tolist() == [[True, False], [True, True]]
    assert task.success.tolist() == [False, True]
    task.reset(torch.tensor([0]))
    assert not task.flight_violation[0] and task.air_peak[0] == 0. and task.free_air_seconds[0] == 0.
    assert task.success.tolist() == [False, True]


def test_flight_after_completed_route_revokes_episode_success():
    task = make_task()
    ascend(task)
    for _ in range(4):
        observe(task, .4, .4, contacts=False)
    assert task.completed[0, 0] and not task.success.item()


def test_real_contact_patches_allow_only_upward_load_inside_declared_top_faces():
    task = make_task(double=True, disabled=True)
    points, forces = torch.zeros(2, 64, 3), torch.zeros(2, 64, 3)
    examples = [([.1, 0., .2], [0., 0., 25.]),       # First top.
                ([.9, 0., .45], [0., 0., 25.]),     # Second top.
                ([-.01, 0., .2], [30., 0., 0.]),    # Riser face.
                ([.1, 0., 0.], [0., 0., 30.]),      # Ground under chassis.
                ([.1, 0., .2], [0., 0., 10.]),      # Legal small share of top load.
                ([.1, 0., .2], [100., 0., 20.]),    # Primarily lateral impact.
                ([.1, 1.6, .2], [0., 0., 30.]),     # Outside platform width.
                ([.9, 0., .2], [0., 0., 30.])]      # Wrong height on second top.
    for index, (point, force) in enumerate(examples):
        points[:, index], forces[:, index] = torch.tensor(point), torch.tensor(force)
    allowed = task.legal_body_contacts(points, forces)
    assert allowed.shape == (2, 64)
    assert allowed[0, :8].tolist() == [True, True, False, False, True, False, False, False]
    assert not allowed[0, 8:].any() and not allowed[1].any()


@pytest.mark.parametrize("loads", [[10., 10., 10., 10.], [30., 10.], [10., 10.]])
def test_split_legal_top_load_supports_without_false_collision(loads):
    task = make_task()
    points = torch.tensor([[[.15, 0., .2]]]).expand(1, len(loads), 3).clone()
    forces = torch.zeros_like(points)
    forces[0, :, 2] = torch.tensor(loads)
    for _ in range(10):
        observe(task, -.1, .1, contacts=False, points=points, forces=forces)
    assert task.body_contact_allowed.all()
    # Match the environment's per-patch illegal collision reduction.
    assert not ((forces.norm(dim=-1) > 5.) & ~task.body_contact_allowed).any()
    assert task.diagnostics()["body_climb_body_supported"].item()
    assert not task.flight_violation.item() and task.air_peak.item() == 0.


def test_legal_weak_patch_is_not_collision_or_sufficient_chassis_support():
    task = make_task()
    points, forces = torch.tensor([[[.15, 0., .2]]]), torch.tensor([[[0., 0., 10.]]])
    for _ in range(4):
        observe(task, -.1, .1, contacts=False, points=points, forces=forces)
    assert task.body_contact_allowed.item()
    assert not ((forces.norm(dim=-1) > 5.) & ~task.body_contact_allowed).any()
    assert not task.diagnostics()["body_climb_body_supported"].item()
    assert task.body_support_credit.item() == 0. and task.flight_violation.item()


def test_illegal_or_downward_patch_load_is_not_aggregated_into_support():
    task = make_task()
    points = torch.tensor([[[.15, 0., .2], [-.01, 0., .2], [.15, 0., .2], [.15, 0., .2]]])
    forces = torch.tensor([[[0., 0., 10.], [30., 0., 30.], [0., 0., -30.], [0., 0., 0.]]])
    for _ in range(4):
        observe(task, -.1, .1, contacts=False, points=points, forces=forces)
    assert task.body_contact_allowed.tolist() == [[True, False, False, False]]
    assert ((forces.norm(dim=-1) > 5.) & ~task.body_contact_allowed).any()
    assert task.flight_violation.item() and task.body_support_credit.item() == 0.


def test_previous_top_load_prevents_flight_but_does_not_credit_current_top_support():
    task = make_task(double=True)
    ascend(task)
    points = torch.tensor([[[.15, 0., .2], [.2, 0., .2], [.9, 0., .45]]])
    forces = torch.tensor([[[0., 0., 10.], [0., 0., 10.], [0., 0., 10.]]])
    for _ in range(10):
        observe(task, .7, .3, contacts=False, points=points, forces=forces)
    assert task.body_contact_allowed.all() and not task.flight_violation.item()
    assert not task.diagnostics()["body_climb_body_supported"].item()
    assert task.body_support_credit.item() == 0.


def test_body_support_reward_has_finite_credit_even_after_repeated_contact():
    task = make_task()
    points, forces = torch.tensor([[[.1, 0., .2]]]), torch.tensor([[[0., 0., 30.]]])
    total = 0.
    for tick in range(100):
        supported = tick % 20 < 15
        observe(task, -.1, 0., points=points, forces=forces if supported else forces * 0.)
        total += float(task.reward_terms(wheels(task, -.1, 0.), torch.tensor([[0., 0., -1.]]))[
            "step_climb_body_support"]) * DT
    assert total == pytest.approx(.3 * .10, abs=1e-7)
    assert not task.success.item()


def test_dense_wheel_lift_distinguishes_large_errors_and_does_not_command_a_pose():
    task = make_task(count=3)
    centers = wheels(task, -.1, torch.tensor([[.18, .18], [.1, .1], [-.1, -.1]]))
    terms = task.reward_terms(centers, torch.tensor([[0., 0., -1.]]).repeat(3, 1))
    assert (terms["step_climb_wheel_lift"][:-1] > terms["step_climb_wheel_lift"][1:]).all()
    assert not hasattr(task, "target_position")


def test_stationary_correct_wheel_clearance_has_no_positive_residence_reward():
    task = make_task()
    centers = wheels(task, -.1, .2)
    for _ in range(3):
        observe(task, -.1, .2, contacts=False)
    terms = task.reward_terms(centers, torch.tensor([[0., 0., -1.]]))
    assert terms["step_climb_progress"].item() == 0.
    assert terms["step_climb_wheel_lift"].item() <= 0.
    assert sum(terms.values()).item() == pytest.approx(0., abs=1e-10)


def test_height_reward_relaxes_smoothly_at_riser_and_restores_after_both_wheels_cross():
    task = make_task(count=9, disabled=True)
    x = torch.tensor([-.4, -.3, -.2, -.1, 0., .08, .19, .3, .4, 0.])[:, None].expand(-1, 2)
    scale = task.height_tracking_scale(wheels(task, x, .2))
    torch.testing.assert_close(scale, torch.tensor([1., 1., .5, 0., 0., 0., .5, 1., 1., 1.]), atol=1e-6, rtol=0.)
    task = make_task()
    assert task.height_tracking_scale(wheels(task, [-.1, .4], .2)).item() == 0.
    ascend(task)
    assert task.height_tracking_scale(wheels(task, 0., .2)).item() == 1.


def test_height_relaxation_does_not_change_commands_or_unrelated_costs():
    scale = torch.tensor([1., .5, 0.])
    names = ("track_height", "dense_height_broad", "dense_height_precision", "landing_height_tracking",
             "landing_velocity_tracking", "vertical_velocity")
    terms = {name: torch.tensor([1., -2., -3.]) for name in names}
    untouched = {"dense_velocity_broad": torch.tensor([-1., -2., -3.]),
                 "motor_torque": torch.tensor([-3., -4., -5.]),
                 "undesired_contact": torch.tensor([-2., 0., -2.])}
    terms.update(untouched)
    apply_climb_height_tracking(terms, scale)
    for name in names:
        expected = [1., -2., -3.] if name == "track_height" else [1., -1., -0.]
        assert torch.equal(terms[name], torch.tensor(expected))
    for name, value in untouched.items():
        assert torch.equal(terms[name], value)


def test_legacy_height_kernel_is_scaled_but_dense_constant_peak_is_preserved():
    scale = torch.tensor([0., .5, 1.])
    legacy = {"track_height": torch.ones(3), "height_square": -torch.ones(3)}
    apply_climb_height_tracking(legacy, scale)
    assert torch.equal(legacy["track_height"], scale)
    assert torch.equal(legacy["height_square"], -scale)
    dense = {"track_height": torch.ones(3), "dense_height_broad": -torch.ones(3)}
    apply_climb_height_tracking(dense, scale)
    assert dense["track_height"].eq(1.).all()
    assert torch.equal(dense["dense_height_broad"], -scale)


def test_reset_is_local_and_clears_all_physical_evidence():
    task = make_task(count=2)
    ascend(task)
    before = {name: value[1].clone() for name, value in vars(task).items()
              if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == task.count}
    task.reset(torch.tensor([0]))
    for name, value in before.items():
        assert torch.equal(getattr(task, name)[1], value), name
    assert task.index[0] == 0 and not task.completed[0].any() and not task.front_passed[0].any()
    assert task.previous_wheel_x[0].tolist() == pytest.approx([-.8, -.8])
    assert not task.success[0] and task.success[1]


def test_rewards_are_finite_floats_and_disabled_lane_is_zero():
    task = make_task(disabled=True)
    observe(task, -.1, .1)
    terms = task.reward_terms(wheels(task, -.1, .1), torch.tensor([[.1, 0., -.995]]).repeat(2, 1))
    for value in terms.values():
        assert value.dtype.is_floating_point and torch.isfinite(value).all()
        assert value.shape == (2,) and value[1] == 0.


def test_body_patch_inputs_must_be_paired_and_shape_aligned():
    task = make_task()
    with pytest.raises(ValueError, match="Both body"):
        observe(task, -.1, 0., points=torch.zeros(1, 2, 3))
    with pytest.raises(ValueError, match="shape"):
        task.legal_body_contacts(torch.zeros(1, 2, 3), torch.zeros(1, 1, 3))


@pytest.mark.parametrize("dt", [0., -.02, float("nan")])
def test_invalid_timestep_is_rejected(dt):
    with pytest.raises(ValueError, match="timestep"):
        StepClimbTask(["flat"], {}, "cpu", dt)
