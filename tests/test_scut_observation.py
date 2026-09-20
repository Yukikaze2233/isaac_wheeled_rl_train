"""Deployment observation availability and SCUT timing/action contracts."""
import json
from pathlib import Path

import torch

from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.scut_observation import (
    CONTROL_FROM_POLICY, POLICY_FROM_CONTROL, build_scut35,
)

ROOT = Path(__file__).resolve().parents[1]


def frame(q=None, request=None):
    q = torch.zeros(2, 6) if q is None else q
    request = torch.tensor([False, True]) if request is None else request
    return build_scut35(torch.ones(2, 3), torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        torch.tensor([[.5, .6, .305]]).repeat(2, 1), q, torch.arange(6.).repeat(2, 1),
        torch.arange(6.).repeat(2, 1), torch.zeros(6), request, torch.full((2,), .06), torch.full((2,), .25))


def test_layout_contains_only_encoder_imu_and_command_channels():
    result = frame()
    assert result.shape == (2, 35)
    torch.testing.assert_close(result[0, :4], torch.tensor([.5, 0., .6, 1.525]))
    torch.testing.assert_close(result[0, 4:7], torch.full((3,), .5))
    torch.testing.assert_close(result[0, 7:10], torch.tensor([0., 0., -1.]))
    torch.testing.assert_close(result[0, 22:28], torch.tensor([0., 1., 3., 4., 2., 5.]))
    torch.testing.assert_close(result[0, 28:], torch.tensor([1., 0., 0., 0., 0., 0., 0.]))
    torch.testing.assert_close(result[1, 28:], torch.tensor([0., 0., 0., 0., 1., .3, .25]))


def test_wheel_encoder_phase_does_not_change_policy_observation():
    q = torch.zeros(2, 6)
    q[:, [2, 5]] = torch.tensor([37., -81.])
    torch.testing.assert_close(frame(q), frame())


def test_policy_to_motor_mapping_is_invertible():
    control = torch.arange(12.).reshape(2, 6)
    torch.testing.assert_close(control[:, POLICY_FROM_CONTROL][:, CONTROL_FROM_POLICY], control)


def test_single_frame_and_physical_time_match_scut_rollout():
    loader = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(loader("contracts/v5_scut35_v1.json"), loader)
    config = stage_contract(loader(plan["base_contract"]), plan, plan["stages"][0], 1024)
    assert config["actor_dim"] == config["actor_frame_dim"] == 35
    assert config["history_length"] == 1 and config["critic_dim"] == 81
    assert config["policy_dt"] / config["physics_dt"] == 4
    assert config["num_steps_per_env"] * config["policy_dt"] == .48
    assert config["signal_perturbations"]["max_delay_steps"] == 1
    assert config["learning_rate_schedule"] == "adaptive"
    assert config["critic_warmup_updates"] == 0
    assert config["total_updates"] == 8000
    assert all(c["command"][:2] == [0., 0.] for c in config["evaluation"]["cases"])
