"""Versioned, simulator-independent 78D privileged observation contract.

Reference: scutrobotlab/wheeled-legged_RL at
b8ff79f3df855faf9dc92f4a282bd80c42649466, wheelbipe25_v3/env.py
(_get_observations, _get_dynamic_priv_obs, _get_static_priv_obs), and
pretrained/26_infantry/flat_and_rotation/2026-07-19_09-14-50/params/env.yaml.

Explicit V6 adaptations:
* Common28 and context7 are ALREADY encoded by the existing V6 35D builder,
  using current clean sensor inputs. Its wrapped leg angles, clipped action
  history, scale-then-clip and command context are preserved, not re-encoded.
* All six-axis privileged fields use the actor's named motor order below.
  PC gains replace native zero-stiffness actuators; Kp clips at 250, not 100.
* Root velocity refers to the link origin. The environment contract owns the
  height datum: flat uses root world z minus environment origin z; the explicit
  rough baseline rough profile uses root world z. Neither uses wheel support height.
  The SDK adapter owns these reference-point conversions.
* Delays are fractional ages / 5ms. Action age starts at the ORIGINAL PD
  calculation, not at retransmission or the 50Hz policy evaluation.
* Missing fields, padding and nan_to_num are forbidden. Value validation is
  explicit at startup/audit boundaries so the hot path never reads GPU scalars.

This module neither selects a critic contract nor modifies the legacy 81D path.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import torch


SCHEMA_VERSION = "scut78-v6-pc-adaptation-v1"
CRITIC_DIM = 78
DTYPE = torch.float32
AGE_TICK_SECONDS = 0.005
JOINT_ORDER = ("Lhip", "Lknee", "Rhip", "Rknee", "Lwheel", "Rwheel")
JOINT_NAMES = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3")
WHEEL_ORDER = ("left", "right")
OBSERVATION_AGE_ORDER = ("gyro", "gravity", "q", "dq")
ACTION_AGE_ORDER = ("leg", "wheel")
MATERIAL_ORDER = ("static_friction", "dynamic_friction", "restitution")


@dataclass(frozen=True)
class CriticField:
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]
    clip: tuple[float, float] | None
    scale: float = 1.0

    @property
    def indices(self) -> slice:
        return slice(self.start, self.stop)


# None means already encoded, not an unbounded raw privileged quantity.
FIELDS = (
    CriticField("clean_common28", 0, 28, (28,), None),
    CriticField("root_linear_velocity_b", 28, 31, (3,), (-100., 100.)),
    CriticField("root_height", 31, 32, (1,), (-10., 10.), 5.),
    CriticField("command_context7", 32, 39, (7,), None),
    CriticField("pc_kp", 39, 45, (6,), (-250., 250.)),
    CriticField("pc_kd", 45, 51, (6,), (-100., 100.)),
    CriticField("applied_torque", 51, 57, (6,), (-100., 100.), .05),
    CriticField("observation_age_ticks", 57, 61, (4,), (-100., 100.)),
    CriticField("action_age_ticks", 61, 63, (2,), (-100., 100.)),
    CriticField("wheel_relative_velocity_b", 63, 69, (2, 3), (-100., 100.)),
    CriticField("wheel_contact", 69, 71, (2,), (-1., 1.)),
    CriticField("base_mass_ratio", 71, 72, (1,), (-10., 10.)),
    CriticField("wheel_material", 72, 78, (2, 3), (-100., 100.)),
)
FIELD_SLICES = MappingProxyType({field.name: field.indices for field in FIELDS})


def critic_joint_indices(source_joint_names: Sequence[str]) -> tuple[int, ...]:
    """Resolve native/control order once; missing/ambiguous names are errors."""
    names = tuple(source_joint_names)
    for name in JOINT_NAMES:
        if names.count(name) != 1:
            raise ValueError(f"critic joint {name!r} must occur exactly once")
    return tuple(names.index(name) for name in JOINT_NAMES)


def _check_layout(inputs: Mapping[str, torch.Tensor]) -> None:
    expected = {field.name for field in FIELDS}
    if set(inputs) != expected:
        raise ValueError(f"critic fields mismatch: missing={expected - set(inputs)}, "
                         f"extra={set(inputs) - expected}")
    common = inputs["clean_common28"]
    if not isinstance(common, torch.Tensor) or common.ndim != 2:
        raise ValueError("clean_common28 must be a (batch, 28) tensor")
    for field in FIELDS:
        value = inputs[field.name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{field.name} must be a tensor")
        if value.shape != (common.shape[0], *field.shape):
            raise ValueError(f"{field.name} expected {(common.shape[0], *field.shape)}, "
                             f"got {tuple(value.shape)}")
        if value.device != common.device:
            raise ValueError(f"{field.name} must share the common28 device")
        if not value.is_floating_point() and not (
                field.name == "wheel_contact" and value.dtype == torch.bool):
            raise TypeError(f"{field.name} must be floating point (contact may be bool)")


def validate_critic78_inputs(**inputs: torch.Tensor) -> None:
    """Strict startup/audit check; intentionally synchronizes when tensors use CUDA.

    Run on raw inputs before clipping. Runtime finite monitoring belongs in the
    environment's existing batched diagnostics, not thirteen per-step .item()s.
    """
    _check_layout(inputs)
    for name, value in inputs.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
    if bool((inputs["clean_common28"][:, 14:16] != 0).any()):
        raise ValueError("common28 wheel position slots 14:16 must be zero")
    for name in ("clean_common28", "command_context7"):
        if bool((inputs[name].abs() > 100).any()):
            raise ValueError(f"{name} must already use the V6 encoded clip")
    for name in ("observation_age_ticks", "action_age_ticks"):
        if bool((inputs[name] < 0).any()):
            raise ValueError(f"{name} must be nonnegative")
    contact = inputs["wheel_contact"]
    if bool(((contact != 0) & (contact != 1)).any()):
        raise ValueError("wheel_contact must contain only 0/1")
    if bool((inputs["base_mass_ratio"] <= 0).any()):
        raise ValueError("base_mass_ratio must be positive")


def build_scut78(*, clean_common28: torch.Tensor,
                 root_linear_velocity_b: torch.Tensor, root_height: torch.Tensor,
                 command_context7: torch.Tensor, pc_kp: torch.Tensor,
                 pc_kd: torch.Tensor, applied_torque: torch.Tensor,
                 observation_age_ticks: torch.Tensor, action_age_ticks: torch.Tensor,
                 wheel_relative_velocity_b: torch.Tensor, wheel_contact: torch.Tensor,
                 base_mass_ratio: torch.Tensor, wheel_material: torch.Tensor,
                 validate: bool = False) -> torch.Tensor:
    """Build a float32 (N,78) frame without importing Isaac or reading GPU scalars.

    Privileged inputs are raw, unscaled physical values except the explicitly
    named ages in 5ms ticks. All motor tensors must follow JOINT_NAMES. Wheels
    and materials are (N, left/right, xyz or static/dynamic/restitution).
    wheel_relative_velocity_b is R^-1(v_wheel - v_root), NOT minus omega x r;
    its y coordinates are muted here. Contacts are booleans from the maximum
    force norm over three 5ms samples > 5N, not force magnitudes.

    clean_common28/context7 must come from the clean V6 frame, not the delayed,
    noisy actor frame. They are copied without another scale or clip.
    """
    inputs = {
        "clean_common28": clean_common28,
        "root_linear_velocity_b": root_linear_velocity_b,
        "root_height": root_height,
        "command_context7": command_context7,
        "pc_kp": pc_kp,
        "pc_kd": pc_kd,
        "applied_torque": applied_torque,
        "observation_age_ticks": observation_age_ticks,
        "action_age_ticks": action_age_ticks,
        "wheel_relative_velocity_b": wheel_relative_velocity_b,
        "wheel_contact": wheel_contact,
        "base_mass_ratio": base_mass_ratio,
        "wheel_material": wheel_material,
    }
    if validate:
        validate_critic78_inputs(**inputs)
    else:
        _check_layout(inputs)
    blocks = []
    for field in FIELDS:
        value = inputs[field.name].to(dtype=DTYPE)
        if field.name == "wheel_relative_velocity_b":
            # Build new values: never zero an aliased native/cache tensor in place.
            value = torch.stack((value[..., 0], torch.zeros_like(value[..., 1]),
                                 value[..., 2]), dim=-1)
        if field.clip is not None:
            value = value.clamp(*field.clip) * field.scale
        blocks.append(value.reshape(value.shape[0], field.stop - field.start))
    return torch.cat(blocks, dim=-1)


def pd_action_age_ticks(applied_at_s: torch.Tensor,
                        original_pd_computed_at_s: torch.Tensor, *,
                        validate: bool = False) -> torch.Tensor:
    """Return (N, leg/wheel) ages from original PD timestamps, in 5ms ticks.

    applied_at_s is scalar, (N,), (N,1), or (N,2); original timestamps are
    (N,2). Use float64 timestamps for long-running clocks: subtract BEFORE
    converting the resulting ages to float32. Packet transmission timestamps
    have no place in this API. A transport that integrates multiple commands
    must first retain the corresponding original-PD provenance/effective time.
    Reset both timestamps at episode reset; do not clamp negative ages to zero.
    """
    origin = original_pd_computed_at_s
    if origin.ndim != 2 or origin.shape[1] != 2:
        raise ValueError("original_pd_computed_at_s must be (batch, 2)")
    if not origin.is_floating_point() or not applied_at_s.is_floating_point():
        raise TypeError("PD timestamps must be floating-point tensors")
    if origin.device != applied_at_s.device:
        raise ValueError("PD timestamps must share a device")
    now = applied_at_s
    if now.ndim == 1:
        now = now.unsqueeze(-1)
    if now.shape not in (torch.Size([]), torch.Size([origin.shape[0], 1]), origin.shape):
        raise ValueError("applied_at_s must be scalar, (batch,), (batch,1) or (batch,2)")
    age = (now - origin) / AGE_TICK_SECONDS
    if validate and (not bool(torch.isfinite(age).all()) or bool((age < 0).any())):
        raise ValueError("PD age must be finite and nonnegative")
    return age.to(dtype=DTYPE)
