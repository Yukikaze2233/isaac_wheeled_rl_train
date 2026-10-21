"""Mixed source rewards preserve each archive and its own height datum; CPU only."""
import pytest
import torch

from wheeled_tasks.chassis.tasks.rewards.flat_reward import WEIGHTS, ROUGH_WEIGHTS
from wheeled_tasks.chassis.simulation.runtime.flat_reward_runtime import ScutFlatRewardRuntime


def runtime(*, rough=True, mask=None, rehearsal_profile="scut0719_flat_v1"):
    return ScutFlatRewardRuntime(4, "cpu", ["base", "left", "right"], {
        "root_velocity_reference": "root_rigid_body_com_in_root_link_frame",
        "wheel_position_reference": "wheel_link_origins_in_root_link_frame",
        "root_height_reference": ("root_link_z_minus_base_ground_scan" if rough
                                  else "root_link_z_minus_env_origin"),
        "undesired_link_names": ["base"], "terminal_link_names": ["base"],
    }, reward_profile="scut0730_rough_v1" if rough else "scut0719_flat_v1",
        flat_rehearsal_mask=mask, flat_rehearsal_reward_profile=rehearsal_profile)


def physical_values():
    root = torch.zeros(4, 7)
    root[:, 6] = 1.
    root[:, 2] = torch.tensor([.475, .9, .65, 1.2])
    origins = torch.zeros(4, 3)
    origins[:, 2] = torch.tensor([.1, .2, .3, .4])
    return dict(root_velocity_body=torch.tensor([[.15, -.04, .03]]).repeat(4, 1),
        angular_velocity_body=torch.tensor([[.02, .03, .4]]).repeat(4, 1),
        projected_gravity=torch.tensor([[.03, -.02, -.999]]).repeat(4, 1),
        root_pose_world=root,
        wheel_link_positions_world=torch.tensor([[[.1, .2, .1], [.18, -.2, .1]]]).repeat(4, 1, 1),
        origins=origins, commands=torch.tensor([[0., .6, .3]]).repeat(4, 1),
        terminated=torch.tensor([True, False, True, False]),
        numerical_safety_reset=torch.tensor([False, False, True, False]))


def fill_history(r):
    for offset in (0., .2, .5):
        r.begin_policy(torch.arange(24.).reshape(4, 6) / 10. + offset)
    qd = torch.arange(24.).reshape(4, 6) + 1.
    r.observe_physics(qd, qd * 3., qd * .4)
    for force in (0., 4., 0.):
        r.net_contact[:, 0, 2] = force
        r.finish_contacts()


def assert_identical(actual, expected):
    assert list(actual.raw_terms) == list(expected.raw_terms) == list(WEIGHTS)
    assert list(actual.payments) == list(expected.payments) == list(WEIGHTS)
    for field in ("raw_terms", "payments"):
        for name in WEIGHTS:
            torch.testing.assert_close(getattr(actual, field)[name], getattr(expected, field)[name],
                                       rtol=0., atol=0., equal_nan=True)
    torch.testing.assert_close(actual.total, expected.total, rtol=0., atol=0.)


def test_each_lane_matches_its_source_archive_all_26_terms_and_payments():
    mask = torch.tensor([True, False, True, False])
    mixed, flat, rough = runtime(mask=mask), runtime(rough=False), runtime()
    for r in (mixed, flat, rough):
        fill_history(r)
    v = physical_values()
    # Rough terrain is neither z=0 nor the tile origin. In particular the
    # flat lane's plane is .1 while the supplied scan is .175: mixing these
    # datums would incorrectly remove its .075 height error and height gate.
    ground = torch.tensor([.175, .3, .2, .9])
    height = v["root_pose_world"][:, 2] - v["origins"][:, 2]
    result = mixed.pay(**v, reward_ground_height_world=ground, flat_reward_height_m=height)
    expected_flat = flat.pay(**v)
    expected_rough = rough.pay(**v, reward_ground_height_world=ground)
    assert len(result.payments) == len(result.raw_terms) == 26
    for field in ("raw_terms", "payments"):
        for name in WEIGHTS:
            expected = torch.where(mask, getattr(expected_flat, field)[name],
                                   getattr(expected_rough, field)[name])
            assert torch.equal(getattr(result, field)[name], expected), (field, name)
    assert torch.equal(result.total, torch.where(mask, expected_flat.total, expected_rough.total))
    assert torch.equal(result.total, torch.stack(list(result.payments.values()), -1).sum(-1))
    assert result.payments["termination"].tolist() == [-4., 0., 0., 0.]
    assert all(value[2] == 0. for value in result.payments.values())
    for name in ("joint_torque", "wheel_power", "stand_still_lin_vel"):
        assert result.payments[name][0] == expected_flat.payments[name][0]
        assert result.payments[name][0] != expected_rough.payments[name][0]
    # The .075 m error halves 0719 tracking, but the .3 m rough error does
    # not gate its velocity/yaw terms. Both keep the full height penalty.
    for name in ("track_lin_vel_xy", "track_lin_vel_xy_square", "track_ang_vel_z", "track_ang_vel_z_square"):
        assert result.raw_terms[name][0].item() == pytest.approx(
            expected_rough.raw_terms[name][0].item() * .5, rel=1e-6)
        assert result.raw_terms[name][1] == expected_rough.raw_terms[name][1]
    assert result.payments["track_height_square"][0].item() == pytest.approx(-.01125, rel=1e-6)
    assert result.payments["track_height_square"][1].item() == pytest.approx(-.18, rel=1e-6)


def test_no_selected_lanes_preserves_default_rough_exactly_and_ignores_other_height_rows():
    default, selected = runtime(), runtime(mask=torch.zeros(4, dtype=torch.bool))
    for r in (default, selected):
        fill_history(r)
    v, ground = physical_values(), torch.tensor([.175, .3, .2, .9])
    original = default.pay(**v, reward_ground_height_world=ground)
    mixed = selected.pay(**v, reward_ground_height_world=ground,
                         flat_reward_height_m=torch.full((4,), float("nan")))
    assert_identical(mixed, original)
    assert "flat_rehearsal_reward" not in default.summary()


def test_all_selected_lanes_match_default_flat_with_numeric_nan_sanitization():
    flat, selected = runtime(rough=False), runtime(mask=torch.ones(4, dtype=torch.bool))
    for r in (flat, selected):
        fill_history(r)
    v = physical_values()
    v["root_pose_world"][2, 2] = float("nan")
    original = flat.pay(**v)
    mixed = selected.pay(**v, reward_ground_height_world=torch.tensor([.2, .3, .4, .5]),
                         flat_reward_height_m=v["root_pose_world"][:, 2] - v["origins"][:, 2])
    assert_identical(mixed, original)
    assert mixed.total[2] == 0. and torch.isfinite(mixed.total).all()


def test_mask_is_owned_reset_preserves_assignment_and_summary_describes_both_rewards():
    mask = torch.tensor([True, False, True, False])
    selected = runtime(mask=mask)
    mask.zero_()
    selected.reset(torch.arange(4))
    assert selected.flat_rehearsal_mask.tolist() == [True, False, True, False]
    summary = selected.summary()
    assert summary["nonzero_weights"] == dict(ROUGH_WEIGHTS)
    rehearsal = summary["flat_rehearsal_reward"]
    assert rehearsal["lanes"] == 2 and rehearsal["velocity_height_gate_enabled"]
    assert rehearsal["nonzero_weights"] == dict(WEIGHTS)
    assert rehearsal["ordinary_rough_reward_profile"] == "scut0730_rough_v1"
    assert "plane_origin_z" in rehearsal["root_height_reference"]


@pytest.mark.parametrize("mask", [True, [True] * 4, torch.ones(4), torch.ones(3, dtype=torch.bool),
    torch.ones(4, 1, dtype=torch.bool), torch.ones(4, dtype=torch.bool, device="meta")])
def test_invalid_mask_is_rejected(mask):
    with pytest.raises(ValueError, match=r"bool\[N\]"):
        runtime(mask=mask)


def test_flat_profile_cannot_install_rough_rehearsal_selection():
    with pytest.raises(ValueError, match="requires the rough runtime"):
        runtime(rough=False, mask=torch.zeros(4, dtype=torch.bool))


@pytest.mark.parametrize("height", [None, .3, torch.zeros(4, 1), torch.zeros(3),
    torch.zeros(4, dtype=torch.float64), torch.zeros(4, device="meta")])
def test_selected_lane_payment_requires_explicit_height_with_physical_shape_and_dtype(height):
    selected = runtime(mask=torch.tensor([True, False, True, False]))
    with pytest.raises(ValueError, match=r"explicit height \[N\]"):
        selected.pay(**physical_values(), reward_ground_height_world=torch.zeros(4),
                     flat_reward_height_m=height)


@pytest.mark.parametrize("rough", [False, True])
def test_height_override_without_mask_is_rejected(rough):
    r = runtime(rough=rough)
    with pytest.raises(ValueError, match="requires an explicit rehearsal mask"):
        r.pay(**physical_values(), flat_reward_height_m=torch.zeros(4))


@pytest.mark.parametrize("mask", [torch.tensor([True, False, True, False]),
                                 torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)])
def test_unified_rehearsal_matches_original_rough_on_every_lane_all_26_terms(mask):
    selected = runtime(mask=mask, rehearsal_profile="scut0730_rough_v1")
    original = runtime()
    for r in (selected, original):
        fill_history(r)
    values, ground = physical_values(), torch.tensor([.175, .3, .2, .9])
    actual = selected.pay(**values, reward_ground_height_world=ground)
    expected = original.pay(**values, reward_ground_height_world=ground)
    assert_identical(actual, expected)
    assert actual.payments["termination"].tolist() == [-4., 0., 0., 0.]
    assert all(value[2] == 0. for value in actual.payments.values())
    summary = selected.summary()["flat_rehearsal_reward"]
    assert summary["reward_profile"] == "scut0730_rough_v1"
    assert summary["nonzero_weights"] == dict(ROUGH_WEIGHTS)
    assert summary["root_height_reference"] == "root_link_z_minus_base_ground_scan"
    assert not summary["velocity_height_gate_enabled"]
    assert "same_0730_payment" in summary["selection"]


def test_unified_rehearsal_rejects_legacy_height_override():
    selected = runtime(mask=torch.tensor([True, False, True, False]),
                       rehearsal_profile="scut0730_rough_v1")
    with pytest.raises(ValueError, match="shared scan height"):
        selected.pay(**physical_values(), reward_ground_height_world=torch.zeros(4),
                     flat_reward_height_m=torch.zeros(4))


def test_rehearsal_profile_cannot_be_unknown_or_selected_without_population():
    with pytest.raises(ValueError, match="0719 or 0730"):
        runtime(mask=torch.ones(4, dtype=torch.bool), rehearsal_profile="custom_reward")
    with pytest.raises(ValueError, match="explicit rehearsal mask"):
        runtime(rehearsal_profile="scut0730_rough_v1")
