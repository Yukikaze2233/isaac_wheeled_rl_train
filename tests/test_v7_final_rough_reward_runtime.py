"""Execute the native-input composition on authentic geometry, without simulation."""
import json
from pathlib import Path

import pytest
from scipy.spatial.transform import Rotation
import torch

from wheeled_tasks.chassis.closed_chain_kinematics import ClosedChainKinematics
from wheeled_tasks.chassis.final_rough_reward_runtime import V7FinalRoughRewardRuntime
from wheeled_tasks.chassis.scut_reward_selection import V7_FINAL_UNDESIRED_BODY_NAMES


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "reports/v7_training_review_20261011/mechanical_stop_candidates_20261011/stop107_reference_v2/asset"


def composition(n=1, *, ground=0., body_height=.55, rehearsal=False, exact_commands=False):
    manifest = json.loads((ASSET / "manifest.json").read_bytes())
    spec = json.loads((ASSET / "model_spec.json").read_bytes())
    names = manifest["tree_joint_names"]
    mapping = {"root_velocity_reference": "root_rigid_body_com_in_root_link_frame",
        "root_height_reference": "root_link_z_minus_base_ground_scan",
        "wheel_position_reference": "wheel_link_origins_in_root_link_frame",
        "undesired_link_names": list(V7_FINAL_UNDESIRED_BODY_NAMES),
        "terminal_link_names": ["base_link"]}
    runtime = V7FinalRoughRewardRuntime(n, "cpu", manifest["rigid_body_names"], mapping,
        model_spec=spec, manifest=manifest, native_joint_names=names,
        exact_commands=exact_commands,
        flat_rehearsal_mask=torch.arange(n) % 2 == 0 if rehearsal else None)
    q = torch.tensor([[manifest["nominal_joint_pos"][name] for name in names]], dtype=torch.float32).repeat(n, 1)
    fk = ClosedChainKinematics(spec, names, manifest["control_joint_names"], "cpu", dtype=torch.float64)
    frames = fk.frames(q.double())
    root = torch.zeros(n, 7)
    root[:, 2], root[:, 6] = body_height + ground, 1.
    wheels = torch.empty(n, 2, 7)
    for index, name in enumerate(("L_link3", "R_link3")):
        pose = frames[name]
        wheels[:, index, :3] = pose[:, :3, 3].float() + root[:, :3]
        wheels[:, index, 3:] = torch.tensor(Rotation.from_matrix(pose[:, :3, :3].numpy()).as_quat(), dtype=torch.float32)
    calls = []

    def raycast(starts, directions):
        calls.append(starts.clone())
        hits = starts.clone()
        hits[..., 2] = ground
        return hits

    runtime.begin_policy(torch.zeros(n, 6))
    for _ in range(4):
        runtime.observe_physics(torch.zeros(n, 6), torch.zeros(n, 6), torch.zeros(n, 6))
        runtime.finish_contacts()
    arguments = dict(root_velocity_body=torch.tensor([[1.6, 0., 0.]]).repeat(n, 1),
        angular_velocity_body=torch.zeros(n, 3), projected_gravity=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
        root_pose_world=root, wheel_link_positions_world=wheels[..., :3], origins=torch.zeros(n, 3),
        commands=torch.tensor([[1.5, .2, .32]]).repeat(n, 1),
        terminated=torch.zeros(n, dtype=torch.bool), numerical_safety_reset=torch.zeros(n, dtype=torch.bool),
        reward_ground_height_world=torch.full((n,), ground), native_joint_positions=q,
        wheel_link_pose_world=wheels, policy_tick=1, raycast=raycast, stair_active=torch.zeros(n, dtype=torch.bool))
    return runtime, arguments, calls


def test_exact_micro_targets_and_both_tracking_costs_survive_large_height_error():
    runtime, args, _ = composition(4, body_height=.45, rehearsal=True, exact_commands=True)
    args["commands"][:] = torch.tensor([
        [.05, .05, .20], [-.05, -.05, .20], [0., 0., .20], [0., 0., .20]])
    args["root_velocity_body"][:] = torch.tensor([.03, .02, 0.])
    result = runtime.pay(**args)
    expected_vx_error = torch.tensor([.02, -.08, -.03, -.03])
    expected_yaw_error = torch.tensor([.05, -.05, 0., 0.])
    torch.testing.assert_close(result.raw_terms["track_lin_vel_xy_square"],
                               (.25 * expected_vx_error).square())
    torch.testing.assert_close(result.raw_terms["track_ang_vel_z_square"],
                               (.5 * expected_yaw_error).square())
    torch.testing.assert_close(result.raw_terms["stand_still_lin_vel"], torch.tensor([0., 0., .05, .05]))
    assert result.payments["track_lin_vel_xy"].gt(0.).all()
    assert result.payments["track_lin_vel_xy_square"].lt(0.).all()
    assert runtime.summary()["standing_command_gate"] == "vx_exactly_zero"
    assert runtime.summary()["squared_tracking_height_gate_enabled"] is False
    baseline, original_args, _ = composition(4, body_height=.45, rehearsal=True)
    original_args.update(args)
    original = baseline.pay(**original_args)
    torch.testing.assert_close(original.raw_terms["stand_still_lin_vel"], torch.full((4,), .05))
    torch.testing.assert_close(original.raw_terms["track_lin_vel_xy_square"],
                               torch.full((4,), (.25 * .03) ** 2))


@pytest.mark.parametrize("rehearsal", [False, True])
def test_full_composition_pays_31_from_current_contact_pose_and_old_effective_commands(rehearsal):
    runtime, args, calls = composition(2, rehearsal=rehearsal)
    before = {name: value.clone() for name, value in args.items() if isinstance(value, torch.Tensor)}
    random_state = torch.random.get_rng_state().clone()
    result = runtime.pay(**args)
    assert len(result.payments) == 31 and result.total.shape == (2,)
    assert len(calls) == 1 and calls[0].shape == (2, 8, 3)
    assert runtime.last_airborne_transition.entered.all()
    assert torch.equal(runtime.airborne.entry_command3, torch.tensor([[1.5, 0., .2]]).repeat(2, 1))
    assert not runtime.airborne.wheel_contact_elapsed_s.any()
    torch.testing.assert_close(result.payments["airborne_air_wheel_zero_torque_exp"],
                               torch.full((2,), .4), rtol=1e-6, atol=0.)
    assert torch.equal(torch.random.get_rng_state(), random_state)
    for name, value in before.items():
        assert torch.equal(args[name], value)
    summary = runtime.summary()
    assert summary["nonzero_term_count"] == 31
    assert set(summary["undesired_link_names"]) == set(V7_FINAL_UNDESIRED_BODY_NAMES)
    assert summary["terminal_link_names"] == ["base_link"]
    assert summary["confirmation_policy_ticks"] == 10
    assert runtime.policy_metrics()["/airborne/physical_state_fraction"] == 1.


def test_duplicate_counter_and_partial_reset_preserve_other_lane_flight_clock():
    runtime, args, _ = composition(2)
    runtime.pay(**args)
    previous = runtime.airborne.snapshot()
    runtime.pay(**args)
    for name, value in previous.items():
        assert torch.equal(runtime.airborne.snapshot()[name], value)
    runtime.reset(torch.tensor([0]))
    runtime.pay(**args)
    assert torch.equal(runtime.last_airborne_transition.updated, torch.tensor([True, False]))
    assert torch.equal(runtime.last_airborne_transition.entered, torch.tensor([True, False]))
    assert torch.equal(runtime.airborne.current_duration_s, previous["current_duration_s"])


def test_newly_mapped_end_link_contact_exits_air_without_becoming_base_termination():
    runtime, args, _ = composition()
    runtime.pay(**args)
    link = runtime.body_names.index("LL_link4")
    runtime.net_contact[:, link, 2] = 6.
    for tick in range(2, 17):
        runtime.finish_contacts()
        args["policy_tick"] = tick
        result = runtime.pay(**args)
        terminated, numeric, confirmed = runtime.termination(args["root_velocity_body"],
            args["angular_velocity_body"], args["root_pose_world"][:, 3:7])
        assert not terminated.any() and not numeric.any() and not confirmed.any()
        assert result.payments["undesired_contact"].item() < 0.
        if runtime.last_airborne_transition.exited.any():
            break
    else:
        pytest.fail("Mapped end-link contact did not reach the source exit window")
    assert runtime.airborne.base_contact_elapsed_s.item() >= .25


def test_numerical_lane_is_not_scanned_or_advanced_and_remains_a_zero_paid_row():
    runtime, args, calls = composition(2)
    args["numerical_safety_reset"][1] = True
    args["terminated"][1] = True
    args["root_pose_world"][1] = torch.nan
    args["native_joint_positions"][1] = torch.nan
    runtime.airborne.request_force_enter(torch.tensor([False, True]))
    result = runtime.pay(**args)
    assert calls[0].shape == (1, 8, 3)
    assert torch.equal(runtime.wheel_scan_hit_count, torch.tensor([[4, 4], [-1, -1]]))
    assert torch.equal(runtime.airborne.last_update_step, torch.tensor([1, -1]))
    assert runtime.airborne.force_enter_request[1]
    assert result.total.shape == (2,) and result.total[1] == 0.
    assert all(value[1] == 0. for value in result.payments.values())
    runtime.reset(torch.tensor([1]))
    assert not runtime.airborne.force_enter_request[1]


def test_all_numerical_lanes_do_not_call_raycast():
    runtime, args, calls = composition(2)
    args["numerical_safety_reset"].fill_(True)
    args["terminated"].fill_(True)
    result = runtime.pay(**args)
    assert not calls and not result.total.any()
    assert not runtime.last_airborne_transition.updated.any()
    assert (runtime.wheel_scan_hit_count == -1).all()


@pytest.mark.parametrize("field", ["root_velocity_body", "angular_velocity_body", "projected_gravity", "terminated"])
def test_wrong_frame_shape_fails_before_scan_or_airborne_mutation(field):
    runtime, args, calls = composition()
    previous = runtime.airborne.snapshot()
    args[field] = args[field][:, :2] if args[field].ndim == 2 else torch.zeros(1)
    with pytest.raises(ValueError, match="V7 final rough"):
        runtime.pay(**args)
    assert not calls
    for name, value in previous.items():
        assert torch.equal(runtime.airborne.snapshot()[name], value)


def test_nonfinite_valid_pose_fails_and_missing_rays_remain_distinct_from_numeric_exclusion():
    runtime, args, calls = composition()
    args["root_pose_world"][0, 0] = torch.nan
    with pytest.raises(RuntimeError, match="outside numerical"):
        runtime.pay(**args)
    assert not calls
    args["root_pose_world"][0, 0] = 0.
    args["raycast"] = lambda starts, directions: torch.full_like(starts, torch.inf)
    result = runtime.pay(**args)
    assert (runtime.wheel_scan_hit_count == 0).all()
    assert runtime.last_airborne_transition.entered.item()
    assert torch.isfinite(result.total).all()
    assert runtime.policy_metrics()["/terrain/wheel_scan_valid_fraction"] == 0.


def test_base_terminal_confirms_at_ten_ticks_and_airborne_payment_is_minus24():
    runtime, args, _ = composition()
    runtime.net_contact[:, runtime.terminal_ids, 2] = 2.
    for tick in range(1, 11):
        args["policy_tick"] = tick
        terminated, numeric, confirmed = runtime.termination(args["root_velocity_body"],
            args["angular_velocity_body"], args["root_pose_world"][:, 3:7])
        assert confirmed.item() is (tick == 10)
        args["terminated"] = terminated
        result = runtime.pay(**args)
    assert result.payments["termination"].item() == -24.


def test_scripted_interval_clears_only_its_airborne_phase_and_keeps_real_scans_contacts():
    runtime, args, calls = composition(2)
    control, control_args, _ = composition(2)
    for owner, frame in ((runtime, args), (control, control_args)):
        owner.pay(**frame)
        owner.airborne.request_force_enter(torch.tensor([True, False]))
        owner.net_contact[:, owner.body_names.index("LL_link4"), 2] = 4.
        for _ in range(4):
            owner.finish_contacts()
        frame["policy_tick"] = 2
    raw_contacts = runtime.contact_history.clone()
    actions = runtime.raw_action.clone()
    result = runtime.pay(**args, scripted_interval=torch.tensor([True, False]))
    expected = control.pay(**control_args)
    transition = runtime.last_airborne_transition
    assert transition.updated.tolist() == [False, True]
    assert not transition.entered.any() and not transition.exited.any()
    for name, values in runtime.airborne.snapshot().items():
        assert torch.equal(values[1], control.airborne.snapshot()[name][1]), name
        if name == "last_update_step":
            assert values[0] == -1
        else:
            assert not values[0].any(), name
    assert len(calls) == 2 and calls[-1].shape == (2, 8, 3)
    assert runtime.wheel_scan_hit_count.eq(4).all()
    assert torch.equal(runtime.contact_history, raw_contacts)
    assert torch.equal(runtime.raw_action, actions)
    assert result.payments["airborne_air_wheel_zero_torque_exp"][0] == 0.
    for name, payment in result.payments.items():
        assert payment[1] == expected.payments[name][1], name
    assert runtime.policy_metrics()["/airborne/scripted_observation_excluded_fraction"] == .5
    assert runtime.policy_metrics()["/terrain/wheel_scan_valid_fraction"] == 1.


@pytest.mark.parametrize("handoff_height", [.305, .55])
def test_first_fully_autonomous_interval_matches_fresh_grounded_or_flying_phase(handoff_height):
    runtime, args, _ = composition(2)
    for tick in (1, 2):
        args["policy_tick"] = tick
        runtime.pay(**args, scripted_interval=torch.tensor([True, False]))
    fresh, fresh_args, _ = composition(2, body_height=handoff_height)
    for name in ("root_pose_world", "wheel_link_pose_world", "wheel_link_positions_world"):
        args[name] = fresh_args[name]
    args["policy_tick"] = fresh_args["policy_tick"] = 3
    result = runtime.pay(**args, scripted_interval=torch.zeros(2, dtype=torch.bool))
    expected = fresh.pay(**fresh_args)
    for name, values in runtime.airborne.snapshot().items():
        assert torch.equal(values[0], fresh.airborne.snapshot()[name][0]), name
    assert runtime.airborne.active[0].item() is (handoff_height == .55)
    for name, payment in result.payments.items():
        assert payment[0] == expected.payments[name][0], name
    before = runtime.airborne.snapshot()
    runtime.pay(**args, scripted_interval=torch.zeros(2, dtype=torch.bool))
    assert not runtime.last_airborne_transition.updated.any()
    for name, values in before.items():
        assert torch.equal(values, runtime.airborne.snapshot()[name]), name


def test_script_mask_preserves_numerical_row_exclusion_and_cannot_hide_invalid_physics():
    runtime, args, calls = composition(2)
    runtime.pay(**args)
    runtime.airborne.request_force_enter(torch.tensor([True, False]))
    before = runtime.airborne.snapshot()
    args["policy_tick"] = 2
    args["root_pose_world"][0, 0] = torch.nan
    with pytest.raises(RuntimeError, match="outside numerical"):
        runtime.pay(**args, scripted_interval=torch.tensor([True, False]))
    assert len(calls) == 1
    for name, values in before.items():
        assert torch.equal(values, runtime.airborne.snapshot()[name]), name
    args["numerical_safety_reset"][0] = True
    args["terminated"][0] = True
    result = runtime.pay(**args, scripted_interval=torch.tensor([True, False]))
    for name, values in before.items():
        assert torch.equal(values[0], runtime.airborne.snapshot()[name][0]), name
    assert calls[-1].shape == (1, 8, 3) and runtime.wheel_scan_hit_count[0].eq(-1).all()
    assert result.total[0] == 0. and all(value[0] == 0. for value in result.payments.values())
    assert runtime.policy_metrics()["/airborne/numerical_observation_excluded_fraction"] == .5
    assert runtime.policy_metrics()["/airborne/scripted_observation_excluded_fraction"] == 0.


@pytest.mark.parametrize("mask", [[True], torch.tensor([1]), torch.tensor([[True]]), torch.tensor([])])
def test_invalid_script_mask_fails_before_scans_or_phase_changes(mask):
    runtime, args, calls = composition()
    before = runtime.airborne.snapshot()
    with pytest.raises(ValueError, match="scripted interval"):
        runtime.pay(**args, scripted_interval=mask)
    assert not calls
    for name, values in before.items():
        assert torch.equal(values, runtime.airborne.snapshot()[name]), name
