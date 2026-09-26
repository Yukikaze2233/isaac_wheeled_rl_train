"""Deployable references, actor migration, reward windows and capability gates."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.evaluation import capability_gate
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.policy_transfer import transfer_actor_state
from wheeled_tasks.chassis.references import CommandReference
from wheeled_tasks.chassis.rewards import reference_jump_terms, reference_motion_terms
from wheeled_tasks.chassis.robustness import V5SignalPerturbations
from wheeled_tasks.chassis.scut_observation import build_reference36
from wheeled_tasks.chassis.skill_commands import profile_command
from wheeled_tasks.chassis.task import Phase, choose_scene_groups

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "contracts/v6_manual_remediation_v1.json").read_text())


def reference_inputs(count):
    return [torch.full((count,), .305), torch.zeros(count), torch.zeros(count),
            torch.zeros(count), torch.full((count,), 7.), torch.zeros(count, dtype=torch.bool),
            torch.zeros(count), torch.full((count,), .06)]


def test_manual_step_is_clock_driven_with_smooth_restore_and_distinct_down_request():
    ref = CommandReference(3, "cpu", PLAN["command_reference"])
    args = reference_inputs(3)
    args[2][:] = torch.tensor([1., -1., 0.])
    args[3][:] = .5
    effective = ref.update(*args)
    torch.testing.assert_close(effective, torch.tensor([.4, .28, .305]))
    assert ref.terrain_mode.tolist() == [1., -1., 0.]
    args[3][:] = 7.5
    torch.testing.assert_close(ref.update(*args), args[0])
    assert not ref.terrain_mode.any()


def test_jump_reference_is_continuous_at_preload_release_and_impact():
    ref = CommandReference(2, "cpu", PLAN["command_reference"])
    args = reference_inputs(2)
    args[5][:] = True
    cfg = PLAN["command_reference"]
    speed = math.sqrt(2 * 9.81 * .06)
    release = cfg["preload_seconds"] + 2 * (cfg["preload_depth_m"] + cfg["release_offset_m"]) / speed
    impact = release + 2 * speed / 9.81
    for boundary in (cfg["preload_seconds"], release, impact, impact + cfg["landing_seconds"]):
        args[6][:] = torch.tensor([boundary - 1e-6, boundary + 1e-6])
        ref.update(*args)
        assert abs(float(ref.height[0] - ref.height[1])) < 1e-5
        assert abs(float(ref.vertical_velocity[0] - ref.vertical_velocity[1])) < 1e-4
    assert ref.height[1] == pytest.approx(.305)
    assert ref.vertical_velocity[1] == 0.


def test_reference_observation_contains_commands_only_and_noise_does_not_delay_them():
    ref = CommandReference(2, "cpu", PLAN["command_reference"])
    args = reference_inputs(2)
    args[1][:] = .6
    args[5][1] = True
    args[6][1] = .2
    ref.update(*args)
    z6, z3 = torch.zeros(2, 6), torch.zeros(2, 3)
    obs = build_reference36(z3, torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        torch.tensor([[.5, 0., .305]]).repeat(2, 1), z6, z6, z6, torch.zeros(6),
        args[5], args[-1], ref)
    assert obs.shape == (2, 36)
    torch.testing.assert_close(obs[:, 35], args[1])
    assert obs[0, 30:35].abs().sum() == 0.
    assert obs[1, 30] < 0 and obs[1, 31] < 0 and obs[1, 32] == 1
    perturb = V5SignalPerturbations(2, "cpu", {"frame_dim": 36, "max_delay_steps": 1}, torch.Generator().manual_seed(617))
    perturb.reset(torch.arange(2))
    noisy = perturb.observation(obs, 0)
    assert torch.equal(noisy[:, 28:], obs[:, 28:])


def test_actor_migration_retains_normal_output_and_exploration():
    source = {"actor_dim": 35, "actor_frame_dim": 35, "history_length": 1,
              "actor_observation_source": "scut35_encoders_imu_commands"}
    target = {**source, "actor_dim": 36, "actor_frame_dim": 36,
              "actor_observation_source": "encoders_imu_command_reference36",
              "actor_migration": PLAN["actor_migration"]}
    state = {"mlp.0.weight": torch.randn(256, 35), "mlp.0.bias": torch.randn(256),
             "distribution.std_param": torch.full((6,), .07)}
    migrated, receipt = transfer_actor_state(state, source, target)
    obs = torch.randn(32, 35)
    obs[:, 29:32] = 0.
    before = torch.nn.functional.linear(obs, state["mlp.0.weight"], state["mlp.0.bias"])
    after = torch.nn.functional.linear(torch.cat((obs, torch.zeros(32, 1)), -1), migrated["mlp.0.weight"], migrated["mlp.0.bias"])
    torch.testing.assert_close(before, after, atol=3e-6, rtol=1e-5)
    assert torch.equal(migrated["distribution.std_param"], state["distribution.std_param"])
    assert receipt["zero_initialized_columns"] == [29, 30, 31, 35]
    same, receipt = transfer_actor_state(migrated, target, target)
    assert same is migrated and receipt is None


def test_dynamic_pitch_reward_prefers_visible_acceleration_reference():
    ref = CommandReference(2, "cpu", PLAN["command_reference"])
    args = reference_inputs(2)
    args[1][:] = .6
    ref.update(*args)
    angle = math.atan(.6 / 9.81)
    gravity = torch.tensor([[0., 0., -1.], [math.sin(angle), 0., -math.cos(angle)]])
    reward = reference_motion_terms(torch.zeros(2, 3), gravity, torch.tensor([[.5, 0., .305]]).repeat(2, 1),
        torch.ones(2), torch.ones(2), ref, PLAN["reference_reward"])
    assert reward["pitch_exp"][1] > reward["pitch_exp"][0]
    assert reward["pitch_velocity"][1] > reward["pitch_velocity"][0]


def test_release_plateau_cannot_collect_push_height_reward_without_takeoff():
    ref = CommandReference(1, "cpu", PLAN["command_reference"])
    args = reference_inputs(1)
    args[5][:] = True
    args[6][:] = .8
    ref.update(*args)
    task = FullTaskSemantics(1, "cpu", .02, {})
    terms = reference_jump_terms(ref, task, torch.tensor([Phase.TAKEOFF]), torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([.34]), torch.zeros(1), torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]),
        torch.tensor([.28]), torch.zeros(1, 3), torch.tensor([[0., 0., .305]]))
    assert terms["jump_push_height"].item() == 0.
    assert terms["jump_release_shortfall"].item() < 0.


def test_com_apex_uses_first_support_loss_but_discards_unconfirmed_bounces():
    task = FullTaskSemantics(2, "cpu", .02, {"jump_apex_frame": "com_release"})
    mode = torch.full((2,), 4)
    request = torch.ones(2, dtype=torch.bool)
    task.observe_com(mode, torch.tensor([Phase.TAKEOFF, Phase.TAKEOFF]), torch.tensor([.32, .32]),
        torch.tensor([.8, .2]), requested=request, contacts=torch.zeros(2, 2, dtype=torch.bool))
    task.observe_com(mode, torch.tensor([Phase.FLIGHT, Phase.TAKEOFF]), torch.tensor([.334, .32]),
        torch.tensor([.6, 0.]), requested=request, contacts=torch.tensor([[False, False], [True, True]]))
    assert task.com_released.tolist() == [True, False]
    assert task.com_release_pending.tolist() == [True, False]
    assert task.com_release_speed[0] == pytest.approx(.8)
    task.observe_com(mode, torch.tensor([Phase.FLIGHT, Phase.TAKEOFF]), torch.tensor([.35, .32]),
        torch.zeros(2), requested=request, contacts=torch.tensor([[False, False], [True, True]]))
    assert task.com_rise.tolist() == pytest.approx([.03, 0.])


def test_velocity_profile_has_zero_endpoint_acceleration_and_bounded_peak():
    spec = {"kind": "velocity_curve", "segment_seconds": 4., "velocity_transition_seconds": 1.5}
    t = torch.linspace(0., 1.5, 151)
    values = profile_command(t, [.5, 0., .305], spec)[:, 0]
    derivative = values.diff() / .01
    assert values[0] == 0 and values[-1] == .5
    assert derivative.max() <= .501 and derivative[0] < .01 and derivative[-1] < .01


def test_remedial_plan_budget_no_oracle_and_stage_gates():
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(PLAN, load)
    configs = [stage_contract(load(plan["base_contract"]), plan, stage, 16384) for stage in plan["stages"]]
    assert sum(config["total_updates"] for config in configs) == 7000
    assert sum(config["total_updates"] * config["target_num_envs"] * 24 for config in configs) == 1622016000
    for config in configs:
        assert (config["actor_dim"], config["critic_dim"]) == (36, 114)
        assert not config["step_assist"]["enabled"]
        assert not config.get("performance_curriculum")
        assert config["evaluation"]["mode"] == "gate" and config["evaluation"]["promotion_case_names"]
        assert len(choose_scene_groups(config["scene_groups"], config["target_num_envs"])) == config["target_num_envs"]


def test_gate_does_not_require_unintroduced_capabilities_or_allow_lost_parent_passes():
    settings = {"promotion_case_names": ["stand", "forward"], "retention_case_names": ["stand", "forward", "fast"]}
    baseline = {"cases": {name: {"passed": name == "stand", "checks": {"mechanics": True}}
                          for name in ("stand", "forward", "fast")}, "rank_lower_is_better": [1, 0, 1]}
    candidate = deepcopy(baseline)
    candidate["passed"] = False
    candidate["cases"]["forward"]["passed"] = True
    assert capability_gate(candidate, baseline, settings)["passed"]
    candidate["cases"]["stand"]["passed"] = False
    assert not capability_gate(candidate, baseline, settings)["passed"]
