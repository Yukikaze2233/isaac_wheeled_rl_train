"""Optional posture floors preserve legacy targets and allow soft transients."""
from copy import deepcopy
from types import SimpleNamespace

import math
import pytest
import torch

from wheeled_tasks.chassis.rewards import (
    DENSE_TRACKING_V1, apply_dense_tracking, reference_motion_terms,
    reward_terms, validate_dense_tracking,
)


POSTURE = {"version": "bounded-posture-floor-v1", "pitch_scale_floor": 3.,
           "roll_scale_floor": 4.}
REFERENCE = {"velocity_huber_weight": .25, "velocity_huber_delta_m_s": .5,
             "velocity_wide_weight": .5, "velocity_wide_sigma_m_s": 1.5,
             "spin_translation_weight": 1.5, "visible_reference_only": True,
             "step_pitch_rad": .08}


def scene(count=1):
    zeros = torch.zeros(count, 3, dtype=torch.float64)
    gravity = zeros.clone()
    gravity[:, 2] = -1.
    commands = zeros.clone()
    commands[:, 2] = .305
    return dict(velocity=zeros.clone(), gravity=gravity, commands=commands,
                support=torch.ones(count, dtype=torch.float64),
                ordinary=torch.ones(count, dtype=torch.bool),
                reference=SimpleNamespace(pitch=torch.zeros(count, dtype=torch.float64),
                    acceleration=torch.zeros(count, dtype=torch.float64),
                    terrain_mode=torch.zeros(count, dtype=torch.long),
                    elapsed=torch.zeros(count, dtype=torch.float64),
                    jumping=torch.zeros(count, dtype=torch.bool)))


def pitch(data, radians):
    angles = torch.as_tensor(radians, dtype=torch.float64).expand(len(data["commands"]))
    data["gravity"][:, 0] = angles.sin()
    data["gravity"][:, 2] = -angles.cos()


def evaluate(data, *, settings=REFERENCE, posture=None):
    return reference_motion_terms(**data, settings=settings, locomotion_posture=posture)


def test_optional_config_does_not_change_frozen_default_or_legacy_components():
    assert "locomotion_posture" not in DENSE_TRACKING_V1
    configured = deepcopy(DENSE_TRACKING_V1) | {"locomotion_posture": deepcopy(POSTURE)}
    assert validate_dense_tracking(configured) is configured
    data = scene(4)
    data["commands"][:, 0] = torch.tensor([0., -1., 3., -5.])
    pitch(data, [.1, -.1, .2, -.2])
    legacy = reference_motion_terms(**data, settings=REFERENCE)
    explicit_none = evaluate(data)
    for key, value in legacy.items():
        assert torch.equal(explicit_none[key], value), key
    assert "locomotion_pitch_floor" not in legacy
    assert "locomotion_roll_floor" not in legacy


@pytest.mark.parametrize("posture", [None, False, {}, [],
    {"version": "bounded-posture-floor-v1", "pitch_scale_floor": 3.},
    POSTURE | {"phase_gate": True},
])
def test_incomplete_or_extra_posture_config_is_rejected(posture):
    config = deepcopy(DENSE_TRACKING_V1) | {"locomotion_posture": posture}
    with pytest.raises(ValueError, match="locomotion_posture"):
        validate_dense_tracking(config)


@pytest.mark.parametrize("key,value", [
    ("version", "unknown"), ("version", None),
    ("pitch_scale_floor", 1.9), ("pitch_scale_floor", 4.1),
    ("pitch_scale_floor", True), ("pitch_scale_floor", float("nan")),
    ("pitch_scale_floor", float("inf")), ("pitch_scale_floor", "3"),
    ("roll_scale_floor", -4.), ("roll_scale_floor", float("-inf")),
    ("roll_scale_floor", False), ("roll_scale_floor", "4"),
])
def test_invalid_floor_is_rejected_at_startup(key, value):
    posture = POSTURE | {key: value}
    with pytest.raises(ValueError, match="locomotion_posture"):
        validate_dense_tracking(deepcopy(DENSE_TRACKING_V1) | {"locomotion_posture": posture})


def test_only_new_deficit_terms_are_added_to_full_ordinary_reward_pipeline():
    data = scene(3)
    data["commands"][:, 0] = torch.tensor([0., 3., -5.])
    data["velocity"][:, 0] = torch.tensor([.03, 2.5, -4.5])
    pitch(data, [.1, .2, -.2])
    data["gravity"][:, 1] = torch.tensor([.01, -.1, .15])
    n = len(data["commands"])
    zero6 = torch.zeros(n, 6, dtype=torch.float64)
    actions = zero6 + .1
    base = reward_terms(data["velocity"], data["velocity"] * .1, data["gravity"],
        data["commands"][:, 2], data["commands"], zero6 + .2, zero6 + .3,
        zero6 + .4, actions, zero6, zero6 - .1, torch.zeros(n, 2, 3),
        data["support"], data["ordinary"], torch.zeros(n, dtype=torch.bool))
    before = {**base, **evaluate(data)}
    after = {**base, **evaluate(data, posture=POSTURE)}
    for terms in (before, after):
        apply_dense_tracking(terms, data["velocity"], data["velocity"] * .1,
            data["gravity"], data["commands"][:, 2], data["commands"],
            data["support"], data["ordinary"], data["reference"], DENSE_TRACKING_V1)
    assert set(after) == set(before) | {"locomotion_pitch_floor", "locomotion_roll_floor"}
    for key, value in before.items():
        assert torch.equal(after[key], value), key
    for key in ("pitch_exp", "roll_exp", "horizontal_angular_velocity", "action_rate",
                "leg_action_smoothness", "wheel_action_smoothness", "motor_torque"):
        assert key in before


def test_high_speed_coefficients_are_bounded_by_existing_static_strength():
    data = scene(5)
    data["commands"][:, 0] = torch.tensor([0., 1., 3., 5., -5.])
    pitch(data, .2)
    data["gravity"][:, 1] = .1
    terms = evaluate(data, posture=POSTURE)
    scale = 2. * torch.exp(-data["commands"][:, 0].square() / 3.) + 2.
    torch.testing.assert_close(terms["pitch_velocity"] + terms["locomotion_pitch_floor"],
        -2. * scale.square().clamp_min(9.) * math.sin(.2)**2)
    original_roll = -2. * scale.square() * .1**2
    torch.testing.assert_close(original_roll + terms["locomotion_roll_floor"],
        torch.full_like(scale, -32. * .1**2))
    assert terms["locomotion_pitch_floor"][0] == 0.
    assert terms["locomotion_roll_floor"][0] == 0.
    assert terms["locomotion_pitch_floor"][3] == terms["locomotion_pitch_floor"][4]


def test_support_loss_cannot_exempt_the_new_pitch_floor():
    data = scene(3)
    data["commands"][:, 0] = 5.
    data["support"][:] = torch.tensor([1., .5, 0.])
    pitch(data, .2)
    terms = evaluate(data, posture=POSTURE)
    total = terms["pitch_velocity"] + terms["locomotion_pitch_floor"]
    torch.testing.assert_close(total, torch.full_like(total, -18. * math.sin(.2)**2))
    assert terms["pitch_velocity"][2] == 0.
    assert terms["locomotion_pitch_floor"][2] < 0.


def test_manual_climb_jump_and_special_spin_are_excluded_but_yaw_rotation_is_not():
    data = scene(6)
    data["commands"][:, 0] = 5.
    data["commands"][:, 1] = torch.tensor([0., 3., 0., 0., 0., 0.])
    data["ordinary"][2] = False
    data["reference"].jumping[3] = True
    data["reference"].terrain_mode[4:] = torch.tensor([-1, 1])
    pitch(data, .2)
    data["gravity"][:, 1] = .1
    terms = evaluate(data, posture=POSTURE)
    for key in ("locomotion_pitch_floor", "locomotion_roll_floor"):
        assert terms[key][:2].lt(0.).all()
        assert terms[key][2:].eq(0.).all()


def test_acceleration_does_not_add_a_hidden_pitch_target_or_reward_phase_escape():
    data = scene(4)
    data["commands"][:, 0] = 3.
    data["reference"].acceleration[:] = torch.tensor([1.5, 0., -1.5, 0.])
    data["reference"].pitch[:] = torch.tensor([.1517, 0., -.1517, 0.])
    data["reference"].elapsed[:] = torch.tensor([0., 1., 2., 100.])
    pitch(data, math.atan(1.5 / 9.81))
    terms = evaluate(data, posture=POSTURE)
    for key in ("pitch_exp", "pitch_velocity", "locomotion_pitch_floor", "locomotion_roll_floor"):
        assert torch.equal(terms[key], terms[key][:1].expand(4)), key
    assert terms["pitch_exp"].gt(0.).all()
    assert terms["locomotion_pitch_floor"].lt(0.).all()  # finite soft cost, no hard rejection


def test_existing_hidden_reference_path_reuses_its_target_without_a_second_target():
    data = scene(2)
    data["commands"][:, 0] = 5.
    data["reference"].pitch[:] = torch.tensor([.2, -.2], dtype=torch.float64)
    pitch(data, [.2, -.2])
    terms = evaluate(data, settings=REFERENCE | {"visible_reference_only": False}, posture=POSTURE)
    torch.testing.assert_close(terms["pitch_exp"], torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(terms["pitch_velocity"], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(terms["locomotion_pitch_floor"], torch.zeros(2, dtype=torch.float64))


def test_floor_is_zero_centered_smooth_and_remains_a_reward_density():
    data = scene(3)
    data["commands"][:, 0] = 5.
    angle = torch.tensor([0., .01, .2], dtype=torch.float64, requires_grad=True)
    data["gravity"] = torch.stack((angle.sin(), torch.zeros_like(angle), -angle.cos()), -1)
    density = evaluate(data, posture=POSTURE)["locomotion_pitch_floor"]
    assert density[0] == 0.
    assert density.diff().lt(0.).all()
    slope = torch.autograd.grad(density.sum(), angle)[0]
    assert slope[0] == 0. and slope[1:].lt(0.).all()
    value = density[2].detach()
    assert value < -.1  # a per-second density, not already multiplied by .02
    torch.testing.assert_close((value.expand(50) * .02).sum(),
                               (value.expand(100) * .01).sum())
