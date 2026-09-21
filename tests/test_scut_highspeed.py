"""SCUT reward structure and physically bounded high-speed curriculum targets."""
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.motion_limits import project_commands, validate_command, wheel_speeds
from wheeled_tasks.chassis.scut_rewards import reward_terms
from wheeled_tasks.chassis.skill_commands import profile_command
from wheeled_tasks.chassis.task import corridor_mesh
from wheeled_tasks.v40.core import motor_torque_limit

ROOT = Path(__file__).resolve().parents[1]


def plan():
    loader = lambda name: json.loads((ROOT / name).read_text())
    return resolve_plan(loader("contracts/v5_scut35_v3_full.json"), loader), loader


def test_full_plan_reaches_five_mps_and_three_revolutions_per_second():
    p, load = plan()
    recipes = {r["name"]: r for r in p["stages"]}
    assert recipes["forward_5"]["command"][0] == 5.
    assert recipes["backward_5"]["command"][0] == -5.
    assert recipes["rotate_3rps"]["command"][1] == pytest.approx(6 * math.pi)
    for recipe in p["stages"]:
        config = stage_contract(load(p["base_contract"]), p, recipe, 4096)
        assert config["reward_profile"] == "scut_v14_flat_v5"
        assert config["termination_event_cost"] == 4.
        assert config["v5_control"]["wheel_action_clip"] == 9.
        assert max(c["episode_seconds"] for c in config["evaluation"]["cases"]) <= config["evaluation"]["episode_seconds"]
    config = stage_contract(load(p["base_contract"]), p, recipes["forward_5"], 4096)
    case = next(c for c in config["evaluation"]["cases"] if c["name"] == "forward_5")
    assert case["episode_seconds"] - case["warmup_seconds"] >= 10.


def test_independent_maxima_are_feasible_but_combined_maxima_are_not():
    p, load = plan()
    limits = p["motion_limits"]
    validate_command([5., 0., .305], limits)
    validate_command([0., 6 * math.pi, .305], limits)
    with pytest.raises(ValueError, match="wheel-speed"):
        validate_command([5., 6 * math.pi, .305], limits)
    prior = load("contracts/own_v40_v2.json")["actuators"]["wheel"]
    speeds = torch.tensor(wheel_speeds(5., 0.))
    reserve = motor_torque_limit(speeds, prior)
    assert (reserve > .3).all() and (reserve < .35).all()
    projected = project_commands(torch.tensor([[5., 6 * math.pi, .305]]), limits)[0]
    validate_command(projected.tolist(), limits)


def test_corridor_has_short_upward_faces_and_sufficient_extent():
    points, indices = corridor_mesh(160., 8.)
    vertices = torch.tensor(points)
    faces = vertices[torch.tensor(indices).reshape(-1, 3)]
    normals = torch.linalg.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    assert (normals[:, 2] > 0).all()
    assert (faces[:, :, 0].amax(-1) - faces[:, :, 0].amin(-1) <= 8.).all()
    assert vertices[:, 0].min() == -160 and vertices[:, 0].max() == 160


def rewards(vx=0., cmd_vx=0., yaw=0.):
    return reward_terms(torch.tensor([[vx, 0., 0.]]), torch.tensor([[0., 0., yaw]]),
        torch.tensor([[0., 0., -1.]]), torch.tensor([.305]), torch.tensor([[cmd_vx, yaw, .305]]),
        *[torch.zeros(1, 6) for _ in range(6)], torch.zeros(1, 2, 3), torch.ones(1),
        torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))


def test_scut_density_and_spin_zero_translation_penalty():
    assert sum(rewards().values()).item() == pytest.approx(5.)
    assert rewards(vx=.2, yaw=6 * math.pi)["stand_translation"].item() == pytest.approx(-.2)
    large = rewards(vx=0., cmd_vx=5.)
    smaller = rewards(vx=1., cmd_vx=5.)
    assert large["lin_vel_square"] < smaller["lin_vel_square"]


def test_fast_braking_profile_allows_time_to_reach_the_command():
    values = profile_command(torch.tensor([1., 13., 25., 37.]), [5., 0., .305],
                             {"kind": "start_stop", "segment_seconds": 12.})
    torch.testing.assert_close(values[:, 0], torch.tensor([5., 0., -5., 0.]))
