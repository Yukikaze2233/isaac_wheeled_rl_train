"""Verify native probe catches unpaid/overwritten rewards and log dt mistakes."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from locomotion_payment_audit import LocomotionPaymentAudit
from wheeled_tasks.chassis import rewards
from wheeled_tasks.chassis.locomotion_quality_revision import POSTURE, STATIONARY
import json
from pathlib import Path

SETTINGS = json.loads((Path(__file__).resolve().parents[1] /
    'contracts/v6_task_campaign_speed_worker_v14.json').read_text())['reference_reward']


def exercise():
    observer = LocomotionPaymentAudit(dict(dense_tracking=dict(
        locomotion_posture=deepcopy(POSTURE), stand_still_lin_vel=deepcopy(STATIONARY))))
    # A released ordinary stand, fast acceleration, and world-reference translation.
    velocity = torch.tensor([[.1, .02, 0.], [3., .1, 0.], [0., .5, 0.]])
    gravity = torch.tensor([[0., 0., -1.], [.1, .1, -.99], [.1, .1, -.99]])
    commands = torch.tensor([[0., 0., .305], [3., 0., .305], [0., 1., .305]])
    ordinary = torch.tensor([True, True, False]); support = torch.ones(3)
    reference = SimpleNamespace(pitch=torch.zeros(3), jumping=torch.zeros(3, dtype=torch.bool),
        terrain_mode=torch.zeros(3, dtype=torch.long), elapsed=torch.zeros(3))
    observer.original_reference = rewards.reference_motion_terms
    observer.original_dense = lambda *args, **kwargs: None
    components = observer.reference(velocity, gravity, commands, support, ordinary,
        reference, SETTINGS, locomotion_posture=POSTURE)
    observer.dense(components, velocity, torch.zeros(3, 3), gravity,
        torch.full((3,), .305), commands, support, ordinary, reference,
        dict(stand_still_lin_vel=STATIONARY))
    # Stand cost reached the actual producer's terms; ordinary spin exception is zero.
    components['stand_still_lin_vel'] = observer.pending['stand_still_lin_vel'].clone()
    extras = dict(native_reward_components=components,
        log={'/reward/' + k: v.mean() * .02 for k, v in observer.pending.items()})
    return observer, extras


def test_actual_components_and_dt_payment_detect_all_three_terms():
    observer, extras = exercise()
    observer.observe(extras, .02)
    report = observer.report()
    assert report['steps'] == 1 and report['maximum_component_error'] == 0
    assert all(value > 0 for value in report['negative_lane_frames'].values())
    assert extras['native_reward_components']['stand_still_lin_vel'][2] == 0


@pytest.mark.parametrize('fault', ['missing_path', 'overwritten_cost', 'positive_bonus', 'forgot_dt', 'double_dt'])
def test_payment_audit_rejects_silent_reward_wiring_regressions(fault):
    observer, extras = exercise()
    if fault == 'missing_path':
        observer.pending.pop('stand_still_lin_vel')
    elif fault in ('overwritten_cost', 'positive_bonus'):
        extras['native_reward_components']['stand_still_lin_vel'].fill_(0. if fault == 'overwritten_cost' else 1.)
    else:
        extras['log']['/reward/stand_still_lin_vel'] *= 50. if fault == 'forgot_dt' else .02
    with pytest.raises((ValueError, AssertionError)):
        observer.observe(extras, .02)
