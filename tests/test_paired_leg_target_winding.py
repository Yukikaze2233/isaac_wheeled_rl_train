"""Verify coupled winding independently of any relative-angle projection."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.v5_control import V5Control

ROOT = Path(__file__).resolve().parents[1]


def controller(version=None):
    read = lambda path: json.loads((ROOT / path).read_text())
    cfg = read('configs/v7_formal_20261012/flat_tgs_continuation_055.json')
    settings = deepcopy(cfg['v5_control'])
    settings.pop('leg_winding_version', None)
    if version is not None:
        settings['leg_winding_version'] = version
    directory = cfg['asset_directory']
    return V5Control(read(directory + '/manifest.json'), read(directory + '/model_spec.json'),
                     read(directory + '/fit_10mpa.json'),
                     read(cfg['control_math_source']), settings, 'cpu')


def test_candidate_keeps_same_side_direction_at_the_angular_cut():
    old, new = controller(), controller('paired_common_v1')
    q = new.nominal.double().expand(1, -1).clone()
    q[:, new.LEGS] += torch.pi
    action = torch.zeros_like(q)
    action[:, new.LEGS] = action.new_tensor([.04, -.04, .04, -.04])
    legacy, wheels, clipped = old.decode(action, q)
    paired, new_wheels, new_clipped = new.decode(action, q)
    assert (legacy - q[:, old.LEGS]).unflatten(-1, (2, 2)).prod(-1).lt(0).all()
    assert (paired - q[:, new.LEGS]).unflatten(-1, (2, 2)).prod(-1).gt(0).all()
    desired = new.nominal[list(new.LEGS)] + clipped[:, new.LEGS] * .25
    torch.testing.assert_close(paired[:, 1::2] - paired[:, ::2],
                               desired[:, 1::2] - desired[:, ::2], atol=1e-12, rtol=0)
    torch.testing.assert_close(new_wheels, wheels, atol=0, rtol=0)
    torch.testing.assert_close(new_clipped, clipped, atol=0, rtol=0)
    torque = new.motor_efforts(q, torch.zeros_like(q), paired, new_wheels)
    assert torque[:, new.LEGS].unflatten(-1, (2, 2)).prod(-1).gt(0).all()
    assert torque[:, new.LEGS].abs().le(40).all()


@pytest.mark.parametrize('turns', [-20, -2, -1, 0, 1, 2, 20])
def test_full_common_revolutions_preserve_relative_targets(turns):
    c = controller('paired_common_v1')
    actions = torch.tensor([[.8, -.2, 5., -.1, .3, -5.]], dtype=torch.float64)
    q = c.nominal.double()[None].clone()
    q[:, c.LEGS] += turns * 2 * torch.pi
    legs, _, clipped = c.decode(actions, q)
    desired = c.nominal[list(c.LEGS)] + clipped[:, c.LEGS] * c.settings['leg_position_scale']
    torch.testing.assert_close(legs, desired + turns * 2 * torch.pi, atol=1e-12, rtol=0)


def test_default_remains_the_legacy_decoder_and_relative_targets_are_not_projected():
    c = controller()
    generator = torch.Generator().manual_seed(491)
    q = torch.randn(257, 6, generator=generator) * 20
    actions = torch.randn(257, 6, generator=generator) * 5
    actual, _, clipped = c.decode(actions, q)
    desired = c.nominal[list(c.LEGS)] + clipped[:, c.LEGS] * c.settings['leg_position_scale']
    delta = desired - q[:, c.LEGS]
    torch.testing.assert_close(actual, q[:, c.LEGS] + torch.atan2(delta.sin(), delta.cos()), atol=0, rtol=0)
    paired = controller('paired_common_v1')
    actions = torch.tensor([[-3., 3., 0., -3., 3., 0.]])
    legs, _, clipped = paired.decode(actions, paired.nominal[None])
    desired = paired.nominal[list(paired.LEGS)] + clipped[:, paired.LEGS] * .25
    torch.testing.assert_close(legs[:, 1::2] - legs[:, ::2],
                               desired[:, 1::2] - desired[:, ::2], atol=1e-6, rtol=0)


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError, match='winding version'):
        controller('unreviewed')
