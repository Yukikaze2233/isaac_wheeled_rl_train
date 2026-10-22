"""Pinned final Rough-v1 physical terms and final payments.

These consumers are not registered as formal reward profiles. The source
consumer requires source-coordinate rear2 angles. The separate V7 adaptation
uses the passive thigh-shank inner angle and authenticated chassis limits,
preserving the source's radian overflow, soft-margin ratios and final weight.
It also clips height after subtracting measured ground, matching the relative
command datum independently of a terrain's world elevation.
Neither consumer fills missing physical inputs or authenticates native wiring.
"""
from dataclasses import dataclass, replace
import math
from types import MappingProxyType

import torch

from wheeled_tasks.chassis.tasks.rewards.flat_reward import (
    POLICY_DT_S, ROUGH_WEIGHTS, ScutFlatRewardFrame, ScutFlatRewardResult,
    _source_reward_terms,
)


VERSION = "v7-final-rough-source-payments-v1"
CHASSIS_VERSION = "v7-final-rough-chassis-payments-v1"
WEIGHTS = MappingProxyType({**ROUGH_WEIGHTS,
    "leg_joint_acc": -1e-7, "leg_joint_vel": -1e-3,
    "wheel_acc": -2e-9, "wheel_vel": -2e-6,
    "action_smoothness_leg": -.005, "action_rate": -.002,
    "action_smoothness_wheel": -.001,
    "track_lin_vel_xy": 1.25,
    "track_lin_vel_xy_square": -1., "track_ang_vel_z_square": -1.,
    "airborne_joint_pos_limits": -10.,
    "airborne_wheel_heading_x_centering": 10.,
    "airborne_air_wheel_zero_torque_exp": 20.,
    "airborne_precontact_wheel_directional_speed": 10.,
    "airborne_precontact_wheel_directional_speed_shortfall": -10.,
})
AIRBORNE_SCALES = MappingProxyType({
    "undesired_contact": 25., "flat_orientation_y_v": 0.,
    "termination": 6., "track_height_square": 0., "foot_bound_square": 0.,
})


@dataclass(frozen=True)
class FinalRoughPhysicalFrame:
    base: ScutFlatRewardFrame
    airborne: torch.Tensor                         # [N], bool
    root_height_world_m: torch.Tensor              # [N]
    ground_height_world_m: torch.Tensor            # [N], base scan
    wheel_contact_elapsed_s: torch.Tensor          # [N,2], accumulated/frozen source timers
    base_contact_elapsed_s: torch.Tensor           # [N], accumulated/frozen source timer
    rear2_wrapped_angles_rad: torch.Tensor         # [N,2], source coordinates, not chassis knees
    wheel_positions_heading_m: torch.Tensor       # [N,2,3], yaw-level root frame


@dataclass(frozen=True)
class V7FinalRoughPhysicalFrame:
    base: ScutFlatRewardFrame
    airborne: torch.Tensor
    root_height_world_m: torch.Tensor
    ground_height_world_m: torch.Tensor
    wheel_contact_elapsed_s: torch.Tensor
    base_contact_elapsed_s: torch.Tensor
    knee_inner_angles_rad: torch.Tensor            # [N,2], passive thigh-shank angles
    wheel_positions_heading_m: torch.Tensor




def _validate_extra_frame(frame):
    if (not isinstance(frame, (FinalRoughPhysicalFrame, V7FinalRoughPhysicalFrame))
            or not isinstance(frame.base, ScutFlatRewardFrame)):
        raise TypeError("Final Rough requires its complete physical frame")
    base = frame.base.root_height_m
    if not isinstance(base, torch.Tensor) or base.ndim != 1:
        raise ValueError("Final Rough requires one physical row per lane")
    n = base.numel()
    angle_name = ("rear2_wrapped_angles_rad" if isinstance(frame, FinalRoughPhysicalFrame)
                  else "knee_inner_angles_rad")
    for name, shape in {"root_height_world_m": (n,), "ground_height_world_m": (n,),
            "wheel_contact_elapsed_s": (n, 2), "base_contact_elapsed_s": (n,),
            angle_name: (n, 2), "wheel_positions_heading_m": (n, 2, 3)}.items():
        value = getattr(frame, name)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != base.dtype or value.device != base.device):
            raise ValueError(f"Final Rough {name} must share the physical shape, dtype and device")
    if (not isinstance(frame.airborne, torch.Tensor) or frame.airborne.shape != (n,)
            or frame.airborne.dtype != torch.bool or frame.airborne.device != base.device):
        raise ValueError("Final Rough airborne state requires an explicit boolean per lane")
    # A broken clock is not a physical input that the source's reward
    # sanitization can repair. Missing/invalid state is not a fresh zero timer.
    for value in (frame.wheel_contact_elapsed_s, frame.base_contact_elapsed_s):
        if not bool(torch.isfinite(value).all()) or bool((value < 0.).any()):
            raise ValueError("Final Rough contact timers must be finite and nonnegative")


def final_rough_reward(frame: FinalRoughPhysicalFrame, *, policy_dt_s=POLICY_DT_S):
    """Pinned source-coordinate payment; reject the separate chassis frame."""
    if not isinstance(frame, FinalRoughPhysicalFrame):
        raise TypeError("Source final Rough requires source-coordinate rear2 angles")
    return _final_rough_reward(frame, angles=frame.rear2_wrapped_angles_rad,
        lower=-1. / 180. * math.pi, upper=68.5 / 180. * math.pi,
        policy_dt_s=policy_dt_s)


def v7_final_rough_reward(frame: V7FinalRoughPhysicalFrame, *, knee_inner_limits_rad,
                          policy_dt_s=POLICY_DT_S, exact_commands=False):
    """Use actual coupled-knee geometry, with no affine rescaling to source angles.

    Limits must come from the authenticated mechanical asset. The 10% lower
    and 5% upper margins retain their source proportions. Overflow remains in
    radians and keeps weight -10; four continuous motor angles are not inputs.
    Height clipping follows ground subtraction. The remaining 28 terms and
    numeric payment rules are shared with the pinned source.
    """
    if not isinstance(frame, V7FinalRoughPhysicalFrame):
        raise TypeError("V7 final Rough requires its passive inner-angle frame")
    if (not isinstance(knee_inner_limits_rad, (tuple, list)) or len(knee_inner_limits_rad) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in knee_inner_limits_rad)):
        raise ValueError("V7 knee limits require an explicit finite radian pair")
    lower, upper = knee_inner_limits_rad
    if not 0. < lower < upper < math.pi:
        raise ValueError("V7 knee inner limits must lie strictly between zero and pi")
    if type(exact_commands) is not bool:
        raise ValueError("Exact tracking commands require an explicit boolean")
    return _final_rough_reward(frame, angles=frame.knee_inner_angles_rad,
        lower=lower, upper=upper, policy_dt_s=policy_dt_s, relative_height_clip=True,
        exact_commands=exact_commands)


def _final_rough_reward(frame, *, angles, lower, upper, policy_dt_s, relative_height_clip=False,
                        exact_commands=False):
    """Apply source state/window masks before multiplying dt and final weights.

    Termination is supplied by the caller; this consumer does not implement
    the final task's 10-tick confirmation, airborne detection or wall reset.
    Numerical failures zero every final payment after per-term sanitization.
    """
    if type(policy_dt_s) not in (int, float) or policy_dt_s != POLICY_DT_S:
        raise ValueError("Final Rough payment requires a 20 ms policy interval")
    _validate_extra_frame(frame)
    # Keep the exact source datum available. V7 commands use root-to-ground
    # height, so its separate adapter clips the relative value instead. Neither
    # branch substitutes wheel height, COM height or a terrain-kind label.
    height = ((frame.root_height_world_m - frame.ground_height_world_m).clamp(.05, .45)
              if relative_height_clip else
              frame.root_height_world_m.clamp(.05, .45) - frame.ground_height_world_m)
    base = replace(frame.base, root_height_m=height)
    raw = _source_reward_terms(base, velocity_height_gate=False, exact_commands=exact_commands)
    active = frame.airborne
    for name, scale in AIRBORNE_SCALES.items():
        if name in raw:
            raw[name] = raw[name] * (1. + active.to(height.dtype) * (scale - 1.))

    span = upper - lower
    soft_lower, soft_upper = lower + span * .10, upper - span * .05
    joint_over = ((soft_lower - angles).clamp_min(0.)
                  + (angles - soft_upper).clamp_min(0.)).sum(-1)
    raw["airborne_joint_pos_limits"] = active.to(height.dtype) * joint_over
    timers = frame.wheel_contact_elapsed_s
    heading = frame.wheel_positions_heading_m
    centering = (active & (timers < .02).all(-1) & (frame.base_contact_elapsed_s < .02)
                 & (heading[:, :, 2] < -.1).all(-1))
    raw["airborne_wheel_heading_x_centering"] = (
        centering.to(height.dtype) * torch.exp(-heading[:, :, 0].square() / .02).mean(-1))
    precontact = active & ~(timers >= .02).any(-1)
    torque = base.applied_torque_nm[:, 4:].abs().amax(-1)
    raw["airborne_air_wheel_zero_torque_exp"] = (
        precontact.to(height.dtype) * torch.exp(-(torque / 1.5).square()))
    vx, cmd = base.root_linear_velocity_body_m_s[:, 0], base.command_vx_m_s
    positive, negative = (cmd > 1.) & (vx > 1.), (cmd < -1.) & (vx < -1.)
    direction = positive.to(height.dtype) - negative.to(height.dtype)
    score = (base.joint_velocity_rad_s[:, 4:] * direction[:, None] / 10.).clamp(0., 1.).amin(-1)
    speed_gate = (precontact & (positive | negative)).to(height.dtype)
    raw["airborne_precontact_wheel_directional_speed"] = speed_gate * score
    short_window = active & (timers > 0.).any(-1) & ~(timers >= .05).any(-1)
    short_gate = (short_window & (positive | negative)).to(height.dtype)
    # Preserve the source operation order, including its gated score.
    raw["airborne_precontact_wheel_directional_speed_shortfall"] = short_gate * (1. - short_gate * score)
    payments = {}
    for name, weight in WEIGHTS.items():
        paid = torch.nan_to_num(raw[name] * policy_dt_s * weight, nan=0., posinf=0., neginf=0.)
        payments[name] = torch.where(base.numerical_safety_reset, 0., paid)
    total = torch.nan_to_num(torch.stack(list(payments.values()), -1).sum(-1),
                             nan=0., posinf=0., neginf=0.)
    return ScutFlatRewardResult(raw_terms=raw, payments=payments, total=total)
