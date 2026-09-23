"""The deployment document's NumPy codec must match the actual training interface."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.scut_observation import build_scut35
from wheeled_tasks.chassis.v5_control import V5Control

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("deployment_io", ROOT / "docs/examples/v5_policy_io.py")
CODEC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CODEC)
BUNDLE = ROOT / "model/纯底盘_v5/urdf"
MANIFEST = json.loads((BUNDLE / "manifest.json").read_text())
NOMINAL = np.array([MANIFEST["nominal_joint_pos"][n] for n in CODEC.CONTROL_ORDER], dtype=np.float32)


@pytest.mark.parametrize("jump,elapsed", [(False, 8.), (True, .25), (True, 8.), (True, -.2)])
def test_numpy_observation_matches_torch_with_wrapping_context_and_permutations(jump, elapsed):
    q = NOMINAL + np.array([6.1, -6.2, 125., 4.8, -4.9, -100.], dtype=np.float32)
    dq = np.array([.1, .2, 40., -.3, -.4, -50.], dtype=np.float32)
    omega = np.array([.2, -.3, 2.], dtype=np.float32)
    gravity = np.array([.1, -.2, -(1 - .1**2 - .2**2)**.5], dtype=np.float32)
    previous = np.array([.2, .5, -.8, 1.1, 3., -4.], dtype=np.float32)
    command = np.array([.4, 2., .28], dtype=np.float32)
    actual = CODEC.build_observation(q, dq, omega, gravity, command, NOMINAL, previous,
        lateral_command=.12, jump_request=jump, jump_apex_m=.1, jump_elapsed_s=elapsed)
    tensor = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))[None, :]
    expected = build_scut35(tensor(omega), tensor(gravity), tensor(command), tensor(q), tensor(dq),
        tensor(previous[CODEC.CONTROL_FROM_POLICY]), torch.from_numpy(NOMINAL), torch.tensor([jump]),
        torch.tensor([.1]), torch.tensor([elapsed]), torch.tensor([.12])).numpy()
    assert actual.dtype == np.float32 and actual.shape == (1, 35)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert (actual[0, 14:16] == 0).all()
    assert (actual[0, 29:32] == 0).all()


def test_numpy_decoding_and_joint_efforts_match_training_at_saturation():
    base = json.loads((ROOT / "contracts/v5_foundation_v1.json").read_text())
    settings = {**base["v5_control"], "leg_position_scale": .25, "wheel_action_clip": 9.}
    prior = json.loads((ROOT / base["control_math_source"]).read_text())
    control = V5Control(MANIFEST, json.loads((BUNDLE / "model_spec.json").read_text()),
        json.loads((BUNDLE / "fit_10mpa.json").read_text()), prior, settings, "cpu")
    q = NOMINAL + np.array([6.1, -.2, 0., .3, -6.2, 0.], dtype=np.float32)
    dq = np.array([2., -3., 50., 4., -5., 99.], dtype=np.float32)
    raw = np.array([4., -5., 1., -2., 12., -20.], dtype=np.float32)
    legs, wheels, clipped = CODEC.decode_actions(raw, q, NOMINAL, settings)
    expected_legs, expected_wheels, expected_clip = control.decode(
        torch.from_numpy(raw[CODEC.CONTROL_FROM_POLICY])[None], torch.from_numpy(q)[None])
    np.testing.assert_allclose(legs, expected_legs[0], atol=1e-6)
    np.testing.assert_allclose(wheels, expected_wheels[0], atol=1e-6)
    np.testing.assert_allclose(clipped, expected_clip[0, CODEC.POLICY_FROM_CONTROL], atol=1e-6)
    expected_effort = control.motor_efforts(torch.from_numpy(q)[None], torch.from_numpy(dq)[None],
                                          expected_legs, expected_wheels)[0].numpy()
    actual_effort = CODEC.reference_joint_torques(q, dq, legs, wheels, settings, prior["actuators"]["wheel"])
    np.testing.assert_allclose(actual_effort, expected_effort, atol=2e-5, rtol=1e-6)
    assert actual_effort[5] == 0.


def test_raw_acceleration_cannot_be_used_as_projected_gravity():
    with pytest.raises(ValueError, match="unit gravity"):
        CODEC.build_observation(NOMINAL, np.zeros(6), np.zeros(3), [0., 0., -9.81],
                                [0., 0., .305], NOMINAL, np.zeros(6))
