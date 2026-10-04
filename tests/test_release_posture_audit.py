"""Phase posture averages include failed stops and expose their population."""
from types import SimpleNamespace as NS
import math

import pytest
import torch

from wheeled_tasks.chassis import learning_audit


def test_posture_is_conditional_on_phase_and_failed_stops_remain_visible(monkeypatch):
    monkeypatch.setattr(learning_audit, "PolicyUpdateProbe", lambda *args: NS(metrics=lambda: {}))
    env = NS(num_envs=3, device="cpu", policy_dt=.02, commands=torch.tensor([[0., 0., .305]] * 3),
        cfg={}, skills=NS(batches=[(torch.tensor([0, 1]), dict(group_name="release", kind="release_stop")),
                                   (torch.tensor([2]), dict(group_name="ordinary", kind="constant"))]))
    audit = learning_audit.LearningAudit(env, NS(actor=None, storage=None))
    phase = torch.tensor([True, False, False])
    window = torch.tensor([True, True, False])
    data = dict(commands=env.commands.clone(), done=torch.zeros(3, dtype=torch.bool),
        reward=torch.zeros(3), height=torch.full((3,), .305),
        gravity=torch.tensor([[math.sin(.1), 0., -math.cos(.1)],
                              [math.sin(.4), 0., -math.cos(.4)], [0., 0., -1.]]),
        omega=torch.zeros(3, 3), velocity=torch.tensor([[.2, 0., 0.], [.4, 0., 0.], [0., 0., 0.]]),
        planar_speed_world=torch.tensor([.2, .4, 0.]), reasons={"fall": torch.zeros(3)},
        release_requested=window, braking_window=torch.zeros(3, dtype=torch.bool),
        settled_stop_window=window)
    for key in ("waiting", "driving", "braking", "settled", "accelerating", "cruising", "decelerating"):
        data["release_stop_" + key] = phase if key == "accelerating" else torch.zeros(3, dtype=torch.bool)
    data["release_stop_stationary_window"] = window
    for key in ("wait", "drive", "dwell", "brake_budget", "release_at", "end_at"):
        data["release_stop_" + key + "_s"] = torch.zeros(3)
    audit.observe(data, {})
    result = audit.drain()
    prefix = "LearningAudit/release/release_posture/"
    assert result[prefix + "accelerating/samples"] == 1
    assert result[prefix + "accelerating/pitch_abs_rad"] == pytest.approx(.1)
    assert result[prefix + "stationary_window/samples"] == 2
    assert result[prefix + "stationary_window/speed_m_s"] == pytest.approx(.3)
    assert result["LearningAudit/release/settled_stop/above_3cm_s_fraction"] == 1
    assert result["LearningAudit/release/release_stop/settled_fraction"] == 0
    assert not any(k.startswith("LearningAudit/ordinary/release_") for k in result)
