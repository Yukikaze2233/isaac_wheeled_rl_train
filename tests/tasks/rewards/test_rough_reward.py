"""0730 rough payment differences and explicit height datum; no simulator/PPO."""
import pytest
import torch

from tests.support.support_flat_reward import frame, nontrivial_frame, SOURCE_EXPECTATIONS
from wheeled_tasks.chassis.tasks.rewards.flat_reward import ROUGH_WEIGHTS, scut_flat_reward, scut_rough_reward
from wheeled_tasks.chassis.simulation.runtime.flat_reward_runtime import ScutFlatRewardRuntime


def test_all_26_rough_payments_against_independent_archive_fixture():
    # 0730 retains every nonzero 0719 formula and scale except these three
    # coefficients and the disabled velocity-height gate. 0719 fixture's
    # physical height error is .06 m, hence its explicit .8 gate factor.
    result = scut_rough_reward(nontrivial_frame())
    changed = {"joint_torque": -1e-5, "wheel_power": -1e-5, "stand_still_lin_vel": -1.}
    ungated = {"track_lin_vel_xy", "track_lin_vel_xy_square", "track_ang_vel_z", "track_ang_vel_z_square"}
    expected_total = 0.
    assert list(result.payments) == list(SOURCE_EXPECTATIONS)
    for name, (flat_weight, flat_raw) in SOURCE_EXPECTATIONS.items():
        weight = changed.get(name, flat_weight)
        raw = flat_raw / .8 if name in ungated else flat_raw
        assert ROUGH_WEIGHTS[name] == weight
        assert result.raw_terms[name].item() == pytest.approx(raw, rel=2e-7)
        assert result.payments[name].item() == pytest.approx(raw * .02 * weight, rel=2e-7)
        expected_total += raw * .02 * weight
    assert result.total.item() == pytest.approx(expected_total, rel=2e-7)


def test_rough_height_error_never_silences_velocity_tracking_or_height_cost():
    f = frame(2)
    f.root_height_m[:] = torch.tensor([.3, .5])
    f.command_vx_m_s[:] = .4
    f.command_yaw_rate_rad_s[:] = .5
    rough, flat = scut_rough_reward(f), scut_flat_reward(f)
    for name in ("track_lin_vel_xy", "track_lin_vel_xy_square", "track_ang_vel_z", "track_ang_vel_z_square"):
        torch.testing.assert_close(rough.payments[name][0], rough.payments[name][1])
        assert flat.payments[name][1] == 0.
    assert rough.payments["track_height_square"][1].item() == pytest.approx(-.08, rel=1e-6)
    f.terminated[:] = True
    f.numerical_safety_reset[1] = True
    result = scut_rough_reward(f)
    assert result.payments["termination"].tolist() == [-4., 0.]
    assert all(value[1] == 0. for value in result.payments.values())


def runtime(rough=True):
    return ScutFlatRewardRuntime(2, "cpu", ["base", "wheel_l", "wheel_r"], {
        "root_velocity_reference": "root_rigid_body_com_in_root_link_frame",
        "wheel_position_reference": "wheel_link_origins_in_root_link_frame",
        "root_height_reference": "root_link_z_minus_base_ground_scan" if rough else "root_link_z_minus_env_origin",
        "undesired_link_names": ["base"], "terminal_link_names": ["base"],
    }, reward_profile="scut0730_rough_v1" if rough else "scut0719_flat_v1")


def test_runtime_ground_scan_datum_and_original_world_pose_are_separate():
    r = runtime()
    root = torch.tensor([[3., 7., .705, 0., 0., 0., 1.], [4., 8., 1.205, 0., 0., 0., 1.]])
    origins = torch.tensor([[1., 2., .1], [2., 3., .2]])
    ground = torch.tensor([.4, .9])
    original = root.clone()
    values = dict(root_velocity_body=torch.zeros(2, 3), angular_velocity_body=torch.zeros(2, 3),
                  projected_gravity=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
                  root_pose_world=root, wheel_link_positions_world=torch.zeros(2, 2, 3),
                  origins=origins, commands=torch.tensor([[0., 0., .305]]).repeat(2, 1),
                  terminated=torch.zeros(2, dtype=torch.bool), numerical_safety_reset=torch.zeros(2, dtype=torch.bool))
    result = r.pay(**values, reward_ground_height_world=ground)
    torch.testing.assert_close(result.payments["track_height_exp_tight"], torch.full((2,), .02))
    torch.testing.assert_close(result.payments["track_height_square"], torch.zeros(2), atol=1e-12, rtol=0)
    assert torch.equal(root, original)
    assert r.summary()["root_height_reference"] == "root_link_z_minus_base_ground_scan"
    assert r.summary()["velocity_height_gate_enabled"] is False
    assert "existing_mechanical_joint_and_3mm_guards" not in r.summary()["local_adaptations"]
    with pytest.raises(ValueError, match="must not silently"):
        runtime(False).pay(**values, reward_ground_height_world=ground)
    for invalid in (None, torch.tensor([float("nan"), .9]), torch.zeros(2, 1)):
        with pytest.raises(ValueError, match="finite base-scan"):
            r.pay(**values, reward_ground_height_world=invalid)


def test_pure_chassis_contact_termination_remains_an_explicit_adaptation():
    r = runtime()
    r.net_contact[:, 0, 2] = 2.
    q = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
    for _ in range(19):
        terminated, _, _ = r.termination(torch.zeros(2, 3), torch.zeros(2, 3), q)
        assert not terminated.any()
    assert r.termination(torch.zeros(2, 3), torch.zeros(2, 3), q)[0].all()
