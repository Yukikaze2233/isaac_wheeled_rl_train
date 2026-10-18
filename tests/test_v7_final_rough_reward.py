"""Frozen-source CPU reward checks; no chassis mapping or FSM timing claim."""
import ast
from collections import OrderedDict
import copy
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.final_rough_reward import (
    AIRBORNE_SCALES, VERSION, WEIGHTS, FinalRoughPhysicalFrame, final_rough_reward,
)
from wheeled_tasks.chassis.scut_flat_reward import (
    ROUGH_WEIGHTS, ScutFlatRewardFrame, scut_rough_reward,
)
from wheeled_tasks.chassis.scut_reward_selection import validate_reward_selection


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "references/v7_final_rough_b8ff79f3"
PIN = "b8ff79f3df855faf9dc92f4a282bd80c42649466"
ADDITIONS = {
    "airborne_joint_pos_limits", "airborne_wheel_heading_x_centering",
    "airborne_air_wheel_zero_torque_exp", "airborne_precontact_wheel_directional_speed",
    "airborne_precontact_wheel_directional_speed_shortfall",
}


def execute(nodes, scope, name):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, name, "exec"), scope)


def class_node(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def class_assignment(cls, name):
    return next(node for node in cls.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


@lru_cache
def frozen_configuration():
    """Run only reward/config AST; importing the frozen Kit module is unnecessary."""
    tree = ast.parse((FROZEN / "env_cfg.py").read_text())
    base = class_node(tree, "WheelbipeV14FlatEnvCfg")
    flat = class_node(tree, "WheelbipeV14FlatEnvCfg_v1")
    rough = class_node(tree, "WheelbipeV14RoughEnvCfg_v1")
    scope = {"OrderedDict": OrderedDict, "copy": copy, "torch": torch}
    cfg = NS(rewards=eval(compile(ast.Expression(class_assignment(base, "rewards").value),
                                 "frozen_base_reward_weights", "eval"), scope),
             airborne_state_machine_cfg=ast.literal_eval(
                 class_assignment(base, "airborne_state_machine_cfg").value))
    for cls in (base, flat):
        for node in cls.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.startswith("rear2_rear1_joint_limit_") or name in (
                        "undesired_contact_force_threshold", "height_obs_clip_enabled", "height_obs_clip_range"):
                    setattr(cfg, name, eval(compile(ast.Expression(node.value),
                                                   "frozen_reward_configuration", "eval"), scope))
    stages = [dict(cfg.rewards)]
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_apply_v14_airborne_landing_precontact_cfg")
    execute([helper], scope, "frozen_precontact_configuration_helper")
    scope["self"] = cfg
    post = next(node for node in flat.body if isinstance(node, ast.FunctionDef)
                and node.name == "__post_init__")
    start = next(i for i, node in enumerate(post.body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "reward_scales")
    stop = next(i for i, node in enumerate(post.body) if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_apply_v14_airborne_landing_precontact_cfg")
    execute(post.body[start:stop], scope, "frozen_flat_v1_reward_overrides")
    stages.append(dict(cfg.rewards))
    execute([post.body[stop]], scope, "frozen_shared_precontact_reward_overrides")
    stages.append(dict(cfg.rewards))
    rough_post = next(node for node in rough.body if isinstance(node, ast.FunctionDef)
                      and node.name == "__post_init__")
    reward_overrides = [node for node in rough_post.body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and ast.unparse(node.targets[0].value) == "self.rewards"]
    assert len(reward_overrides) == 3
    execute(reward_overrides, scope, "frozen_rough_v1_reward_overrides")
    stages.append(dict(cfg.rewards))
    return cfg, stages


@lru_cache
def frozen_airborne_oracle():
    tree = ast.parse((FROZEN / "airborne.py").read_text())
    cls = class_node(tree, "AirborneStateMachine")
    names = {"apply_reward_term_scales", "_get_reward_window_mask",
             "_get_reward_command_lin_vel_x", "_get_directional_wheel_speed_score",
             "_get_contact_timer_below_mask", "_get_rear2_rear1_joint_limit_terms"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    extracted = ast.ClassDef(name="FrozenAirborneReward", bases=[], keywords=[],
                             body=copy.deepcopy(methods), decorator_list=[])
    # Place the oracle at the post-wrap angle boundary. This checks the limit
    # consumer, not the SDK wrapping function, encoders or chassis mapping.
    scope = {"torch": torch, "wrap_to_pi": lambda value: value}
    execute([extracted], scope, "frozen_airborne_reward_hook_and_five_helpers")
    oracle = scope["FrozenAirborneReward"]()
    oracle._cfg = lambda env: env.cfg.airborne_state_machine_cfg
    return oracle


def frame(n=1, dtype=torch.float32):
    def scalar(value):
        return torch.full((n,), value, dtype=dtype)

    def rows(values):
        return torch.tensor(values, dtype=dtype).expand(n, *torch.tensor(values).shape).clone()

    base = ScutFlatRewardFrame(
        root_linear_velocity_body_m_s=rows([1.6, .05, -.1]),
        root_angular_velocity_body_rad_s=rows([.1, -.2, .3]),
        projected_gravity=rows([-.03, .04, -.998]), pitch_rad=scalar(.07),
        root_height_m=scalar(.3), target_height_m=scalar(.3), command_vx_m_s=scalar(1.5),
        command_yaw_rate_rad_s=scalar(.2), joint_velocity_rad_s=rows([.2, -.4, .1, .3, 4., 8.]),
        joint_acceleration_rad_s2=rows([1., 2., 3., 4., 5., 6.]),
        applied_torque_nm=rows([1., 2., -1., -2., .3, -.4]),
        raw_action=rows([.1, .2, .3, .4, .5, .6]),
        previous_raw_action=rows([-.1, .1, .1, .2, .3, .4]),
        before_previous_raw_action=rows([.2, .3, .3, .4, .5, .6]),
        wheel_positions_body_m=rows([[.04, -.22, -.28], [-.02, .22, -.25]]),
        undesired_contact_forces_world_n=torch.zeros(n, 3, 1, 3, dtype=dtype),
        terminated=torch.zeros(n, dtype=torch.bool), numerical_safety_reset=torch.zeros(n, dtype=torch.bool))
    base.undesired_contact_forces_world_n[:, 0, 0, 0] = 4.
    return FinalRoughPhysicalFrame(base=base, airborne=torch.ones(n, dtype=torch.bool),
        root_height_world_m=scalar(.32), ground_height_world_m=scalar(.02),
        wheel_contact_elapsed_s=torch.zeros(n, 2, dtype=dtype), base_contact_elapsed_s=scalar(0.),
        rear2_wrapped_angles_rad=rows([.3, .7]),
        wheel_positions_heading_m=rows([[.04, -.22, -.28], [-.02, .22, -.25]]))


def oracle_result(value):
    cfg, _ = frozen_configuration()
    clip = cfg.height_obs_clip_range
    height = value.root_height_world_m.clamp(*clip) - value.ground_height_world_m
    baseline = scut_rough_reward(replace(value.base, root_height_m=height))
    n = len(height)
    joint_pos = torch.zeros(n, 6, dtype=height.dtype)
    joint_pos[:, :2] = value.rear2_wrapped_angles_rad
    assert bool((value.rear2_wrapped_angles_rad.abs() <= math.pi).all())
    env = NS(cfg=cfg, num_envs=n, device="cpu", height_reward_airborne_state=value.airborne,
        airborne_wheel_contact_exit_time=value.wheel_contact_elapsed_s,
        airborne_base_contact_exit_time=value.base_contact_elapsed_s,
        _rear2_joint_idx=[0, 1], _wheel_idx=[4, 5], joint_pos=joint_pos,
        joint_vel=value.base.joint_velocity_rad_s,
        command=torch.stack((value.base.command_vx_m_s, torch.zeros_like(height),
                             value.base.command_yaw_rate_rad_s), -1),
        robot=NS(data=NS(root_lin_vel_b=value.base.root_linear_velocity_body_m_s,
                        applied_torque=value.base.applied_torque_nm)),
        _get_root_quat_inv_and_wheel_pos_b=lambda: (
            None, None, value.wheel_positions_heading_m, None, None))
    raw = frozen_airborne_oracle().apply_reward_term_scales(env, copy.deepcopy(baseline.raw_terms))
    weights = {name: weight for name, weight in cfg.rewards.items() if weight != 0.}
    paid = {name: torch.where(value.base.numerical_safety_reset, 0., torch.nan_to_num(
        raw[name] * .02 * weight, nan=0., posinf=0., neginf=0.)) for name, weight in weights.items()}
    total = torch.nan_to_num(torch.stack(list(paid.values()), -1).sum(-1), nan=0., posinf=0., neginf=0.)
    return raw, paid, total


def assert_source_oracle(value):
    actual = final_rough_reward(value)
    raw, paid, total = oracle_result(value)
    assert set(actual.raw_terms) == set(raw) == set(WEIGHTS)
    assert set(actual.payments) == set(paid) == set(WEIGHTS)
    for name in WEIGHTS:
        torch.testing.assert_close(actual.raw_terms[name], raw[name], rtol=2e-6, atol=1e-8, equal_nan=True)
        torch.testing.assert_close(actual.payments[name], paid[name], rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(actual.total, total, rtol=2e-6, atol=1e-8)
    return actual


def test_frozen_source_files_are_authenticated_without_importing_native_modules():
    manifest = json.loads((FROZEN / "source_manifest.json").read_text())
    assert manifest["commit"] == PIN
    assert set(manifest["files"]) == {"airborne.py", "env_cfg.py", "LICENSE"}
    for name, metadata in manifest["files"].items():
        assert hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() == metadata["sha256"]
    frozen_configuration()
    frozen_airborne_oracle()


def test_effective_weights_follow_the_frozen_inheritance_and_helper_order():
    cfg, stages = frozen_configuration()
    nonzero = [{name: weight for name, weight in stage.items() if weight != 0.} for stage in stages]
    assert list(map(len, nonzero)) == [26, 28, 31, 31]
    assert nonzero[-1] == dict(WEIGHTS)
    assert set(nonzero[-1]) - set(ROUGH_WEIGHTS) == ADDITIONS
    additions = cfg.airborne_state_machine_cfg["reward_additions"]
    assert set(additions) & set(nonzero[-1]) == ADDITIONS
    assert len(additions) > len(ADDITIONS)
    assert dict(AIRBORNE_SCALES) == cfg.airborne_state_machine_cfg["reward_scales"]
    assert cfg.airborne_state_machine_cfg["reward_full"] == {}
    assert cfg.airborne_state_machine_cfg["landing_trajectory"]["enabled"] is False
    assert nonzero[1]["wheel_power"] == -1e-4
    assert nonzero[-1]["wheel_power"] == nonzero[-1]["joint_torque"] == -1e-5
    assert nonzero[-1]["track_lin_vel_xy"] == 1.25


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normal_air_and_first_contact_windows_match_the_actual_source_hook(dtype):
    value = frame(11, dtype)
    value.airborne[0] = False
    value.wheel_contact_elapsed_s[:] = torch.tensor([
        [0., 0.], [0., 0.], [.019999, 0.], [.02, 0.], [.049999, 0.], [.05, 0.],
        [0., .005], [0., 0.], [0., 0.], [0., 0.], [0., 0.]], dtype=dtype)
    value.base_contact_elapsed_s[7:9] = torch.tensor([.019999, .02], dtype=dtype)
    value.wheel_positions_heading_m[9, 1, 2] = -.1
    value.wheel_positions_heading_m[10, :, 2] = -.100001
    actual = assert_source_oracle(value)
    centering = actual.raw_terms["airborne_wheel_heading_x_centering"]
    assert (centering > 0).tolist() == [False, True, True, False, False, False, True, True, False, False, True]
    torque = actual.raw_terms["airborne_air_wheel_zero_torque_exp"]
    speed = actual.raw_terms["airborne_precontact_wheel_directional_speed"]
    shortfall = actual.raw_terms["airborne_precontact_wheel_directional_speed_shortfall"]
    assert (torque > 0).tolist() == [False, True, True, False, False, False, True, True, True, True, True]
    assert (speed > 0).tolist() == (torque > 0).tolist()
    assert (shortfall > 0).tolist() == [False, False, True, True, True, False, True, False, False, False, False]
    assert actual.raw_terms["undesired_contact"].tolist() == [1.] + [25.] * 10
    assert actual.raw_terms["flat_orientation_y_v"][0] > 0
    assert not actual.raw_terms["flat_orientation_y_v"][1:].any()


@pytest.mark.parametrize("command,vx,wheels,score", [
    (1.2, 1.3, [10., 5.], .5), (-1.2, -1.3, [-10., -5.], .5),
    (1.2, -1.3, [10., 5.], 0.), (-1.2, 1.3, [-10., -5.], 0.),
    (1., 1.3, [10., 5.], 0.), (-1., -1.3, [-10., -5.], 0.),
    (1.2, 1., [10., 5.], 0.), (-1.2, -1., [-10., -5.], 0.),
    (1.2, 1.3, [-1., 8.], 0.), (-1.2, -1.3, [1., -8.], 0.),
    (1.2, 1.3, [20., 30.], 1.), (1.2, 1.3, [2., 9.], .2),
])
def test_direction_speed_gates_are_strict_signed_and_use_the_slower_wheel(command, vx, wheels, score):
    value = frame()
    value.base.command_vx_m_s.fill_(command)
    value.base.root_linear_velocity_body_m_s[:, 0] = vx
    value.base.joint_velocity_rad_s[:, 4:] = torch.tensor(wheels)
    value.wheel_contact_elapsed_s[:, 0] = .01
    actual = assert_source_oracle(value)
    assert actual.raw_terms["airborne_precontact_wheel_directional_speed"].item() == pytest.approx(score)
    direction_gate = abs(command) > 1. and abs(vx) > 1. and command * vx > 0
    assert actual.raw_terms["airborne_precontact_wheel_directional_speed_shortfall"].item() == pytest.approx(
        1. - score if direction_gate else 0.)


@pytest.mark.parametrize("torques,expected", [([0., 0.], 1.), ([1.5, 0.], math.exp(-1.)),
                                              ([-1.5, -3.], math.exp(-4.))])
def test_precontact_zero_torque_score_uses_maximum_absolute_applied_wheel_torque(torques, expected):
    value = frame()
    value.base.applied_torque_nm[:, 4:] = torch.tensor(torques)
    actual = assert_source_oracle(value)
    assert actual.raw_terms["airborne_air_wheel_zero_torque_exp"].item() == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_source_soft_limits_and_nextafter_neighbors_preserve_operation_order(dtype):
    cfg, _ = frozen_configuration()
    lower, upper = cfg.rear2_rear1_joint_limit_lower, cfg.rear2_rear1_joint_limit_upper
    span = upper - lower
    soft_lower = lower + span * cfg.rear2_rear1_joint_limit_lower_boundary_ratio
    soft_upper = upper - span * cfg.rear2_rear1_joint_limit_upper_boundary_ratio
    endpoints = torch.tensor([soft_lower, soft_upper], dtype=dtype)
    value = frame(7, dtype)
    value.rear2_wrapped_angles_rad[:] = endpoints
    value.rear2_wrapped_angles_rad[1, 0] = torch.nextafter(endpoints[0], torch.tensor(-torch.inf, dtype=dtype))
    value.rear2_wrapped_angles_rad[2, 0] = torch.nextafter(endpoints[0], torch.tensor(torch.inf, dtype=dtype))
    value.rear2_wrapped_angles_rad[3, 1] = torch.nextafter(endpoints[1], torch.tensor(-torch.inf, dtype=dtype))
    value.rear2_wrapped_angles_rad[4, 1] = torch.nextafter(endpoints[1], torch.tensor(torch.inf, dtype=dtype))
    value.rear2_wrapped_angles_rad[5] = torch.tensor([-math.pi, math.pi], dtype=dtype)
    value.airborne[6] = False
    value.rear2_wrapped_angles_rad[6] = value.rear2_wrapped_angles_rad[5]
    actual = assert_source_oracle(value)
    source_raw, _, _ = oracle_result(value)
    penalty = actual.raw_terms["airborne_joint_pos_limits"]
    assert torch.equal(penalty, source_raw["airborne_joint_pos_limits"])
    assert not penalty[[0, 2, 3, 6]].any()
    assert bool((penalty[[1, 4, 5]] > 0).all())
    assert penalty[1] == endpoints[0] - value.rear2_wrapped_angles_rad[1, 0]
    assert penalty[4] == value.rear2_wrapped_angles_rad[4, 1] - endpoints[1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_world_z_is_clipped_before_subtracting_the_scan_ground_height(dtype):
    value = frame(4, dtype)
    value.airborne.zero_()
    value.root_height_world_m[:] = torch.tensor([.8, .1, -.2, .3], dtype=dtype)
    value.ground_height_world_m[:] = torch.tensor([.4, .4, -.3, .4], dtype=dtype)
    value.base.target_height_m.fill_(.2)
    value.base.root_height_m.fill_(99.)
    actual = assert_source_oracle(value)
    height = value.root_height_world_m.clamp(.05, .45) - value.ground_height_world_m
    assert height.tolist() == pytest.approx([.05, -.3, .35, -.1], abs=5e-8)
    torch.testing.assert_close(actual.raw_terms["track_height_square"], (10. * (height - .2)).square())
    relative_clip = (value.root_height_world_m - value.ground_height_world_m).clamp(.05, .45)
    assert not torch.allclose(height, relative_clip)


def test_terminal_payment_is_minus_four_normal_minus_twenty_four_air_and_numeric_rows_are_zero():
    value = frame(4)
    value.airborne[:] = torch.tensor([False, True, True, False])
    value.base.terminated[:] = torch.tensor([True, True, True, False])
    value.base.numerical_safety_reset[2] = True
    actual = assert_source_oracle(value)
    assert actual.payments["termination"].tolist() == [-4., -24., 0., 0.]
    assert actual.total[2] == 0.
    assert all(payment[2] == 0. for payment in actual.payments.values())
    torch.testing.assert_close(actual.total, torch.stack(list(actual.payments.values()), -1).sum(-1))


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_one_nonfinite_physical_term_is_sanitized_without_erasing_other_payments(nonfinite):
    value = frame(2)
    baseline = final_rough_reward(value)
    value.base.joint_acceleration_rad_s2[0, 0] = nonfinite
    actual = assert_source_oracle(value)
    assert not torch.isfinite(actual.raw_terms["leg_joint_acc"][0])
    assert actual.payments["leg_joint_acc"][0] == 0.
    for name in WEIGHTS:
        if name != "leg_joint_acc":
            assert torch.equal(actual.payments[name], baseline.payments[name])
    assert all(bool(torch.isfinite(payment).all()) for payment in actual.payments.values())


def test_nan_in_one_addition_is_sanitized_independently_of_other_source_terms():
    value = frame(2)
    baseline = final_rough_reward(value)
    value.wheel_positions_heading_m[0, 0, 0] = torch.nan
    actual = assert_source_oracle(value)
    assert torch.isnan(actual.raw_terms["airborne_wheel_heading_x_centering"][0])
    assert actual.payments["airborne_wheel_heading_x_centering"][0] == 0.
    for name in WEIGHTS:
        if name != "airborne_wheel_heading_x_centering":
            assert torch.equal(actual.payments[name], baseline.payments[name])


@pytest.mark.parametrize("timer", ["wheel_contact_elapsed_s", "base_contact_elapsed_s"])
@pytest.mark.parametrize("bad", [-.01, float("nan"), float("inf")])
def test_contact_state_requires_supplied_finite_nonnegative_accumulated_timers(timer, bad):
    value = frame()
    getattr(value, timer).fill_(bad)
    with pytest.raises(ValueError, match="contact timers"):
        final_rough_reward(value)


@pytest.mark.parametrize("dt", [True, .005, .04, "0.02"])
def test_final_payments_require_one_twenty_millisecond_interval(dt):
    with pytest.raises(ValueError, match="20 ms"):
        final_rough_reward(frame(), policy_dt_s=dt)


def test_candidate_is_a_physical_consumer_without_formal_profile_registration():
    with pytest.raises(ValueError, match="Unknown reward_profile"):
        validate_reward_selection({"reward_profile": VERSION})
    value = frame()
    snapshots = {name: copy.deepcopy(getattr(value, name)) for name in value.__dataclass_fields__}
    assert_source_oracle(value)
    for name, before in snapshots.items():
        after = getattr(value, name)
        if name == "base":
            assert all(torch.equal(getattr(before, field), getattr(after, field)) for field in before.__dataclass_fields__)
        else:
            assert torch.equal(before, after)
