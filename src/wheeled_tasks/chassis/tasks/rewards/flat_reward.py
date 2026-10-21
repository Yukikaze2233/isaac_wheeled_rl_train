"""flat baseline flat and rough baseline rough payments, independent of the simulator.

Source: vendored reference implementation (MIT), public commit
b8ff79f3df855faf9dc92f4a282bd80c42649466. Parameters are from
pretrained/26_infantry/flat_and_rotation/2026-07-19_09-14-50/params/env.yaml
(rewards:1674-1711). Formulas are in wheelbipe25_v3/env.py:3502-3937;
dispatch/sanitization:3998-4018,4058-4060. This maps archived parameters to
available public source, not a replay of the original July executable.

The archive disables FSM reward overrides, jump and auxiliary launch forces;
it does not enable V14's gimbal_spin_translate override. Only the nonzero
configured terms are implemented. No legacy reward merge, success bonus,
task/rollout clock, failure-tail settlement or jump objective exists here.

ChassisEnv selects these cores through the explicit scut0719_flat_v1 or
scut0730_rough_v1 ScutFlatRewardRuntime adapter. The caller maps its root reference, height datum,
joint/action order, raw action history and contact body set. Termination and
numerical-safety observers remain separate responsibilities of that adapter.
CPU wiring is not evidence of native qualification or full source sampling.
Source reward scales and gates are not physical acceptance thresholds.

The separate rough baseline rough entry uses the 2026-07-30_09-16-27
rough_rotation_stair archive. All 37 configured keys were compared: the same
26 nonzero terms are paid, with three changed weights. Rough also disables
the velocity-height gate and supplies root z minus the base raycast ground
estimate. Its inherited V14 reward hook has no active gimbal translation or
FSM override. Unconfigured soft velocity tracking is not paid even though
its archived sigma changes from 1.0 to 1.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import torch


SOURCE_COMMIT = "b8ff79f3df855faf9dc92f4a282bd80c42649466"
REWARD_VERSION = "scut0719-flat-source-v1"
ROUGH_REWARD_VERSION = "scut0730-rough-source-v1"
POLICY_DT_S = 0.02
CONTACT_HISTORY_SAMPLES = 3
CONTACT_SAMPLE_DT_S = 0.005

# Exactly the nonzero archived weights, in archive order. Zero-weight and
# merely computed/unconfigured upstream terms must not silently become paid.
WEIGHTS = MappingProxyType({
    "termination": -200.0,
    "leg_joint_acc": -5.0e-7,
    "leg_joint_vel": -0.005,
    "joint_torque": -0.0001,
    "wheel_acc": -1.0e-8,
    "wheel_vel": -1.0e-5,
    "wheel_power": -0.0001,
    "lin_vel_z": -0.5,
    "ang_vel_xy": -0.05,
    "action_smoothness_leg": -0.05,
    "action_rate": -0.01,
    "action_smoothness_wheel": -0.01,
    "flat_orientation_y_v": -2.0,
    "flat_orientation_y_exp": 1.0,
    "flat_orientation_x_v": -2.0,
    "flat_orientation_x_exp": 1.0,
    "track_lin_vel_xy": 1.0,
    "track_lin_vel_xy_square": -0.1,
    "track_ang_vel_z": 1.0,
    "track_ang_vel_z_square": -0.1,
    "stand_still_lin_vel": -2.0,
    "track_height_exp_tight": 1.0,
    "track_height_square": -1.0,
    "no_fork": -1.0,
    "no_fork_square": -1.0,
    "undesired_contact": -2.0,
})

ROUGH_WEIGHTS = MappingProxyType({**WEIGHTS,
    "joint_torque": -1.0e-5,
    "wheel_power": -1.0e-5,
    "stand_still_lin_vel": -1.0,
})


@dataclass(frozen=True)
class ScutFlatRewardFrame:
    """Required physical values sampled after physics and before auto-reset.

    All floating tensors share dtype/device and leading dimension N. Joint and
    raw policy action vectors have order [four active legs, left/right wheels];
    passive joints, springs and gimbals are excluded. Velocities/accelerations
    are physical joint rad/s and rad/s^2, not observation-scaled values or a
    policy-interval acceleration approximation. Torque is physically applied
    N*m after actuation limits, not requested PD output.

    root_linear_velocity_body_m_s and angular velocity correspond to upstream
    robot.data.root_lin_vel_b/root_ang_vel_b. Mapping COM versus link-origin
    velocity on another SDK must be explicit. pitch_rad is upstream's wrapped
    Euler pitch from root orientation. Projected gravity is dimensionless.

    root_height_m is the explicitly selected reward datum: absolute root z
    for the flat entry, root z minus base-ground scan for the rough entry.
    target_height_m is the effective height command in that same datum.
    Do not substitute wheel support mean, net jump rise or observation scaling.
    Commands are raw external body-forward/yaw targets before the reward's
    deadzone. World translation is outside this baseline's input contract.

    raw_action/previous_raw_action/before_previous_raw_action are unscaled,
    unclipped incoming policy outputs (0719 action LPF is disabled). The caller
    owns history/reset; do not substitute held, scripted or decoded actions.
    wheel_positions_body_m contains both wheel LINK ORIGINS in the root frame.

    undesired_contact_forces_world_n is [N,3,K,3]: the three latest 5ms native
    net-force vectors for an explicitly mapped nonempty set of K undesired
    links. Its body set differs from the terminal-contact set. Never substitute
    wheel-only contact flags, force magnitudes summed across contacts, or zeros
    for missing sensors. The core does not create physics/contact history.

    terminated excludes timeout/collection truncation. numerical_safety_reset
    is the caller's source-equivalent physical-explosion mask. Nonfinite term
    values are sanitized independently even outside that mask, as upstream.
    This frame does not certify the caller's masks or mechanical mapping.
    """

    root_linear_velocity_body_m_s: torch.Tensor  # [N,3]
    root_angular_velocity_body_rad_s: torch.Tensor  # [N,3]
    projected_gravity: torch.Tensor  # [N,3]
    pitch_rad: torch.Tensor  # [N]
    root_height_m: torch.Tensor  # [N]
    target_height_m: torch.Tensor  # [N]
    command_vx_m_s: torch.Tensor  # [N]
    command_yaw_rate_rad_s: torch.Tensor  # [N]
    joint_velocity_rad_s: torch.Tensor  # [N,6]
    joint_acceleration_rad_s2: torch.Tensor  # [N,6]
    applied_torque_nm: torch.Tensor  # [N,6]
    raw_action: torch.Tensor  # [N,6]
    previous_raw_action: torch.Tensor  # [N,6]
    before_previous_raw_action: torch.Tensor  # [N,6]
    wheel_positions_body_m: torch.Tensor  # [N,2,3]
    undesired_contact_forces_world_n: torch.Tensor  # [N,3,K,3]
    terminated: torch.Tensor  # [N], bool
    numerical_safety_reset: torch.Tensor  # [N], bool


@dataclass(frozen=True)
class ScutFlatRewardResult:
    """Source raw terms and final, already dt-scaled per-step payments.

    raw_terms can contain NaN/Inf; they are diagnostic intermediates, not paid
    rewards. payments and total follow source sanitization. Do not multiply
    payments/total by dt again or add legacy event rewards to them.
    """

    raw_terms: dict[str, torch.Tensor]
    payments: dict[str, torch.Tensor]
    total: torch.Tensor


def _validate_frame(frame: ScutFlatRewardFrame) -> None:
    if not isinstance(frame, ScutFlatRewardFrame):
        raise TypeError("A complete ScutFlatRewardFrame is required")
    base = frame.root_height_m
    if not isinstance(base, torch.Tensor) or base.ndim != 1 or base.numel() == 0:
        raise ValueError("root_height_m requires a nonempty [N] tensor")
    if base.dtype not in (torch.float32, torch.float64):
        raise TypeError("Reward physical tensors require float32 or float64")
    n = base.shape[0]
    shapes = {
        "root_linear_velocity_body_m_s": (n, 3),
        "root_angular_velocity_body_rad_s": (n, 3),
        "projected_gravity": (n, 3),
        "pitch_rad": (n,), "root_height_m": (n,), "target_height_m": (n,),
        "command_vx_m_s": (n,), "command_yaw_rate_rad_s": (n,),
        "joint_velocity_rad_s": (n, 6), "joint_acceleration_rad_s2": (n, 6),
        "applied_torque_nm": (n, 6), "raw_action": (n, 6),
        "previous_raw_action": (n, 6), "before_previous_raw_action": (n, 6),
        "wheel_positions_body_m": (n, 2, 3),
    }
    for name, shape in shapes.items():
        value = getattr(frame, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} requires tensor shape {shape}")
        if value.dtype != base.dtype or value.device != base.device:
            raise ValueError(f"{name} must share physical dtype and device")
    contact = frame.undesired_contact_forces_world_n
    if (not isinstance(contact, torch.Tensor) or contact.ndim != 4
            or contact.shape[:2] != (n, CONTACT_HISTORY_SAMPLES)
            or contact.shape[2] < 1 or contact.shape[3] != 3):
        raise ValueError("Contact history requires [N,3,K,3] with K >= 1")
    if contact.dtype != base.dtype or contact.device != base.device:
        raise ValueError("Contact history must share physical dtype and device")
    for name in ("terminated", "numerical_safety_reset"):
        value = getattr(frame, name)
        if (not isinstance(value, torch.Tensor) or value.shape != (n,)
                or value.dtype != torch.bool or value.device != base.device):
            raise ValueError(f"{name} requires a bool [N] tensor on the physical device")
    # Intentionally do not reject/sanitize physical NaN/Inf here: source
    # sanitizes each weighted payment, then zeros numerical-safety rows.


def _source_reward_terms(frame: ScutFlatRewardFrame, *, velocity_height_gate,
                         exact_commands=False, squared_tracking_height_gate=True):
    """Shared physical terms, before task-specific scales and final payment."""
    _validate_frame(frame)
    v, omega = frame.root_linear_velocity_body_m_s, frame.root_angular_velocity_body_rad_s
    qd, qdd, torque = frame.joint_velocity_rad_s, frame.joint_acceleration_rad_s2, frame.applied_torque_nm
    gx, gy = frame.projected_gravity[:, 0], frame.projected_gravity[:, 1]
    vx_cmd, yaw_cmd = frame.command_vx_m_s, frame.command_yaw_rate_rad_s
    # env.py:3569-3582; strict '<' separately for forward and yaw commands.
    still_vx = vx_cmd.abs() < 0.1
    tracking_vx = torch.where(still_vx, 0., vx_cmd)
    tracking_yaw = torch.where(yaw_cmd.abs() < 0.1, 0., yaw_cmd)
    if exact_commands:
        still_vx = vx_cmd == 0.
        tracking_vx, tracking_yaw = vx_cmd, yaw_cmd
    ev = tracking_vx - v[:, 0] * torch.cos(frame.pitch_rad)
    ew = tracking_yaw - omega[:, 2]
    eh = frame.root_height_m - frame.target_height_m
    # 0719 enables ONLY the linear-band height gate; 0730 disables it.
    height_gate = ((0.1 - eh.abs()) / 0.05).clamp(0., 1.) if velocity_height_gate else 1.
    square_gate = height_gate if squared_tracking_height_gate else 1.
    orientation_scale = 2. * torch.exp(-vx_cmd.square() / 3.) + 2.
    delta = frame.raw_action - frame.previous_raw_action
    second = frame.raw_action - 2. * frame.previous_raw_action + frame.before_previous_raw_action
    fork = frame.wheel_positions_body_m[:, 0, 0] - frame.wheel_positions_body_m[:, 1, 0]
    # Archive contact sensor:3 samples at5ms; force threshold3N (env.yaml:928).
    harmful_contact = (frame.undesired_contact_forces_world_n.norm(dim=-1) > 3.).any(dim=2).any(dim=1)
    return {
        "termination": frame.terminated.to(v.dtype),
        "leg_joint_acc": qdd[:, :4].square().sum(-1),
        "leg_joint_vel": qd[:, :4].square().sum(-1),
        "joint_torque": torque.square().sum(-1),
        "wheel_acc": qdd[:, 4:].square().sum(-1),
        "wheel_vel": qd[:, 4:].square().sum(-1),
        "wheel_power": (torque[:, 4:] * qd[:, 4:]).clamp_min(0.).sum(-1),
        "lin_vel_z": v[:, 2].square(),
        "ang_vel_xy": omega[:, :2].square().sum(-1),
        "action_smoothness_leg": second[:, :4].square().sum(-1),
        "action_rate": delta.square().sum(-1),
        "action_smoothness_wheel": second[:, 4:].square().sum(-1),
        "flat_orientation_y_v": (orientation_scale * gx).square(),
        "flat_orientation_y_exp": torch.exp(-gx.square() / 0.02),
        "flat_orientation_x_v": (orientation_scale * gy).square(),
        "flat_orientation_x_exp": torch.exp(-gy.square() / 0.01),
        "track_lin_vel_xy": torch.exp(-ev.clamp(-1., 1.).square() / 0.5) * height_gate,
        "track_lin_vel_xy_square": (0.25 * ev).square() * square_gate,
        "track_ang_vel_z": torch.exp(-ew.clamp(-0.8, 0.8).square() / 0.25) * height_gate,
        "track_ang_vel_z_square": (0.5 * ew).square() * square_gate,
        "stand_still_lin_vel": v[:, :2].abs().sum(-1) * still_vx,
        "track_height_exp_tight": torch.exp(-eh.clamp(-0.15, 0.15).square() / 0.001),
        "track_height_square": (10. * eh).square(),
        "no_fork": (fork.abs() > 0.05).to(v.dtype),
        "no_fork_square": (5. * fork).square(),
        "undesired_contact": harmful_contact.to(v.dtype),
    }


def _source_reward(
    frame: ScutFlatRewardFrame, *, policy_dt_s: float = POLICY_DT_S,
    weights=WEIGHTS, velocity_height_gate=True, exact_commands=False,
    squared_tracking_height_gate=True,
) -> ScutFlatRewardResult:
    """Shared source formulas with explicit archive weights and gate selection.

    Termination is paid on each caller-supplied terminal transition, exactly
    like the source before auto-reset. This stateless core has no extra event
    latch or horizon. A terminal frame must not be submitted twice as one step.
    """
    if type(policy_dt_s) not in (float, int) or policy_dt_s != POLICY_DT_S:
        raise ValueError("flat baseline reward dispatch requires policy_dt_s=0.02")
    raw = _source_reward_terms(frame, velocity_height_gate=velocity_height_gate,
                               exact_commands=exact_commands,
                               squared_tracking_height_gate=squared_tracking_height_gate)
    payments = {}
    for name, weight in weights.items():
        # Match source ordering: multiply raw by dt, then weight, then sanitize.
        value = torch.nan_to_num(raw[name] * policy_dt_s * weight, nan=0., posinf=0., neginf=0.)
        payments[name] = torch.where(frame.numerical_safety_reset, 0., value)
    total = torch.stack(list(payments.values()), dim=-1).sum(-1)
    total = torch.nan_to_num(total, nan=0., posinf=0., neginf=0.)
    return ScutFlatRewardResult(raw_terms=raw, payments=payments, total=total)


def scut_flat_reward(
    frame: ScutFlatRewardFrame, *, policy_dt_s: float = POLICY_DT_S,
) -> ScutFlatRewardResult:
    """Unchanged flat baseline flat payments; height uses its fixed world datum."""
    return _source_reward(frame, policy_dt_s=policy_dt_s)


def scut_rough_reward(
    frame: ScutFlatRewardFrame, *, policy_dt_s: float = POLICY_DT_S,
) -> ScutFlatRewardResult:
    """rough baseline rough payments; caller supplies base-scan-relative height."""
    return _source_reward(frame, policy_dt_s=policy_dt_s,
                          weights=ROUGH_WEIGHTS, velocity_height_gate=False)
