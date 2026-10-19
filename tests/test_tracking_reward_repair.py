"""Check payment units, micro commands and mixed-lane repair selection."""
from copy import deepcopy

import pytest
import torch

from test_small_rough_course import candidate
from wheeled_tasks.chassis.scut_flat_reward import (
    ScutFlatRewardFrame, scut_flat_reward, scut_rough_reward)
from wheeled_tasks.chassis.scut_flat_reward_runtime import ScutFlatRewardRuntime
from wheeled_tasks.chassis.scut_reward_selection import validate_reward_selection
from wheeled_tasks.chassis.small_rough_course import validate_resume
from wheeled_tasks.chassis.tracking_reward_repair import (
    CONTINUOUS_COST_VERSION, KEY, VERSION, pay)


def frame(count=1):
    def zeros(*shape):
        return torch.zeros(count, *shape, dtype=torch.float64)

    return ScutFlatRewardFrame(
        root_linear_velocity_body_m_s=zeros(3),
        root_angular_velocity_body_rad_s=zeros(3),
        projected_gravity=torch.tensor([[0., 0., -1.]], dtype=torch.float64).repeat(count, 1),
        pitch_rad=zeros(), root_height_m=zeros() + .3, target_height_m=zeros() + .3,
        command_vx_m_s=zeros(), command_yaw_rate_rad_s=zeros(),
        joint_velocity_rad_s=zeros(6), joint_acceleration_rad_s2=zeros(6),
        applied_torque_nm=zeros(6), raw_action=zeros(6), previous_raw_action=zeros(6),
        before_previous_raw_action=zeros(6), wheel_positions_body_m=zeros(2, 3),
        undesired_contact_forces_world_n=zeros(3, 2, 3),
        terminated=torch.zeros(count, dtype=torch.bool),
        numerical_safety_reset=torch.zeros(count, dtype=torch.bool))


@pytest.mark.parametrize("rough", [False, True])
def test_exact_small_commands_and_zero_speed_cost_have_no_measurement_deadzone(rough):
    f = frame(3)
    f.command_vx_m_s[:] = torch.tensor([.05, -.05, 0.])
    f.command_yaw_rate_rad_s[:] = torch.tensor([.03, -.03, .03])
    f.root_linear_velocity_body_m_s[:, 0] = f.command_vx_m_s
    f.root_angular_velocity_body_rad_s[:, 2] = f.command_yaw_rate_rad_s
    f.root_linear_velocity_body_m_s[2, 0] = .01
    result = pay(f, rough=rough)
    assert result.payments["track_lin_vel_xy_square"][:2].tolist() == [0., 0.]
    assert result.payments["track_ang_vel_z_square"].tolist() == [0., 0., 0.]
    assert result.payments["stand_still_lin_vel"][:2].tolist() == [0., 0.]
    assert result.payments["stand_still_lin_vel"][2] == pytest.approx(
        -.01 * .02 * (1. if rough else 2.))
    assert result.payments["track_lin_vel_xy_square"][2] == pytest.approx(-.125 * .01**2 * .02)


@pytest.mark.parametrize("rough", [False, True])
def test_repair_preserves_other_terms_terminal_payment_and_numeric_reset(rough):
    f = frame()
    f.root_linear_velocity_body_m_s.fill_(.2)
    f.root_angular_velocity_body_rad_s.fill_(.3)
    f.root_height_m.fill_(.375)
    f.joint_velocity_rad_s.fill_(2.)
    f.joint_acceleration_rad_s2.fill_(3.)
    f.applied_torque_nm.fill_(4.)
    f.raw_action.fill_(.5)
    f.undesired_contact_forces_world_n.fill_(10.)
    f.terminated.fill_(True)
    f.command_vx_m_s.fill_(3.)
    original = (scut_rough_reward if rough else scut_flat_reward)(f)
    repaired = pay(f, rough=rough)
    assert len(repaired.payments) == 26
    for name in original.payments:
        factor = 20. if name == "track_lin_vel_xy_square" else 1.
        torch.testing.assert_close(repaired.payments[name], factor * original.payments[name])
    assert repaired.payments["termination"].item() == -4.
    f.numerical_safety_reset.fill_(True)
    assert all(value.item() == 0. for value in pay(f, rough=rough).payments.values())


@pytest.mark.parametrize("version", [VERSION, CONTINUOUS_COST_VERSION])
def test_runtime_dispatch_and_readback_cover_rehearsal_and_rough_lanes(version):
    config = candidate()
    config[KEY] = version
    selected = validate_reward_selection(config)
    assert selected[KEY] == version and not selected["source_reward_terms_unchanged"]
    mapping = deepcopy(config["scut_reward_mapping"])
    mapping.update(undesired_link_names=["base"], terminal_link_names=["base"])
    runtime = ScutFlatRewardRuntime(
        2, "cpu", ["base", "left", "right"], mapping,
        reward_profile=config["reward_profile"],
        flat_rehearsal_mask=torch.tensor([True, False]),
        tracking_reward_repair=version)
    root = torch.tensor([[0., 0., .375, 0., 0., 0., 1.]]).repeat(2, 1)
    kwargs = dict(root_velocity_body=torch.zeros(2, 3),
        angular_velocity_body=torch.zeros(2, 3),
        projected_gravity=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        root_pose_world=root, wheel_link_positions_world=torch.zeros(2, 2, 3),
        origins=torch.zeros(2, 3), commands=torch.tensor([[3., 0., .3]]).repeat(2, 1),
        terminated=torch.zeros(2, dtype=torch.bool),
        numerical_safety_reset=torch.zeros(2, dtype=torch.bool),
        reward_ground_height_world=torch.zeros(2),
        flat_reward_height_m=root[:, 2])
    result = runtime.pay(**kwargs)
    torch.testing.assert_close(result.payments["track_lin_vel_xy_square"],
        torch.tensor([-.01125 if version == VERSION else -.0225, -.0225]))
    summary = runtime.summary()
    assert summary[KEY] == version
    assert summary["nonzero_weights"]["track_lin_vel_xy_square"] == -2.
    assert summary["flat_rehearsal_reward"]["nonzero_weights"]["track_lin_vel_xy_square"] == -2.
    assert not summary["squared_tracking_height_gate_enabled"]
    flat = summary["flat_rehearsal_reward"]
    assert flat["positive_tracking_height_gate_enabled"]
    assert flat["squared_tracking_height_gate_enabled"] == (version == VERSION)


def test_same_course_repair_preserves_optimizer_and_rejects_implicit_downgrade():
    old = candidate()
    old.pop(KEY)
    new = dict(old, **{KEY: VERSION})
    migration = validate_resume(old, new)
    assert migration["optimizer"] == "inherited"
    assert migration["changed_contract_fields"] == [KEY]
    with pytest.raises(ValueError, match="forward migration"):
        validate_resume(new, old)
    with pytest.raises(ValueError, match="entropy_coef"):
        validate_resume(old, dict(new, entropy_coef=0.))
    for value in (None, False, "unknown"):
        with pytest.raises(ValueError, match="Tracking repair"):
            validate_reward_selection(dict(old, **{KEY: value}))


@pytest.mark.parametrize("rough", [False, True])
def test_continuous_costs_preserve_other_payments_and_numerical_boundaries(rough):
    f = frame(5)
    f.root_height_m[:] += torch.tensor([0., .05, .075, .1, .127])
    f.command_vx_m_s.fill_(3.)
    f.command_yaw_rate_rad_s.fill_(4.)
    f.root_linear_velocity_body_m_s[:, 0] = 1.
    f.joint_velocity_rad_s.fill_(2.)
    f.joint_acceleration_rad_s2.fill_(3.)
    f.applied_torque_nm.fill_(4.)
    f.raw_action.fill_(.5)
    f.undesired_contact_forces_world_n.fill_(10.)
    f.terminated.fill_(True)
    historical = pay(f, rough=rough)
    continuous = pay(f, rough=rough, version=CONTINUOUS_COST_VERSION)
    assert len(continuous.payments) == 26
    torch.testing.assert_close(continuous.payments["track_lin_vel_xy_square"],
                               torch.full((5,), -.01, dtype=torch.float64))
    torch.testing.assert_close(continuous.payments["track_ang_vel_z_square"],
                               torch.full((5,), -.008, dtype=torch.float64))
    for name in continuous.payments:
        if name not in ("track_lin_vel_xy_square", "track_ang_vel_z_square"):
            torch.testing.assert_close(continuous.payments[name], historical.payments[name],
                                       rtol=0., atol=0.)
    if rough:
        torch.testing.assert_close(continuous.total, historical.total, rtol=0., atol=0.)
    assert continuous.payments["termination"].tolist() == [-4.] * 5
    f.numerical_safety_reset.fill_(True)
    assert all(torch.equal(value, torch.zeros_like(value))
               for value in pay(f, rough=rough, version=CONTINUOUS_COST_VERSION).payments.values())


def test_wrong_height_cannot_improve_failed_tracking_by_hiding_negative_costs():
    # This is a payment-function comparison, not an equilibrium simulation.
    commands = torch.cartesian_prod(
        torch.tensor([-3., -1.35, -.05, 0., .05, 1.35, 3.], dtype=torch.float64),
        torch.tensor([-4.5, -3.25, -2., 0., 2., 3.25, 4.5], dtype=torch.float64) * torch.pi)
    f = frame(len(commands) * 3)
    f.command_vx_m_s[:] = commands[:, 0].repeat_interleave(3)
    f.command_yaw_rate_rad_s[:] = commands[:, 1].repeat_interleave(3)
    f.root_height_m[:] += torch.tensor([0., .1, .127]).repeat(len(commands))
    historical = pay(f, rough=False).total.reshape(-1, 3)
    assert (historical[:, 1:] > historical[:, :1]).any()
    continuous = pay(f, rough=False, version=CONTINUOUS_COST_VERSION).total.reshape(-1, 3)
    assert (continuous[:, 1:] <= continuous[:, :1]).all()


@pytest.mark.parametrize("before", [None, VERSION])
def test_continuous_cost_migration_is_explicit_and_cannot_downgrade(before):
    old = candidate()
    if before is None:
        old.pop(KEY)
    else:
        old[KEY] = before
    new = dict(old, **{KEY: CONTINUOUS_COST_VERSION})
    migration = validate_resume(old, new)
    assert migration["changed_contract_fields"] == [KEY]
    assert migration["optimizer"] == "inherited"
    with pytest.raises(ValueError, match="forward migration"):
        validate_resume(new, old)
