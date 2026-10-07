"""CPU checks of new scene preparation and actual actor command timing."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from chassis_suite_support import SCENARIOS, FALLEN_CASES, control_ownership, digest, prepare_playback, project_scene
from play_chassis_suite import apply_requested_command, validated_session
from wheeled_tasks.chassis.task import terrain_surfaces

SOURCE = ROOT / "models/v6_recovery_candidate_38000_20261008/policy.onnx.contract.json"
POLICY = ROOT / "models/v6_recovery_candidate_38000_20261008/policy.onnx"


@pytest.fixture(scope="module")
def source():
    value = json.loads(SOURCE.read_text())
    design = json.loads((ROOT / value["design_preflight"]["file"]).read_text())
    return value, design["stages"]["recovery"]["roles"]["full"]["contract"]


@pytest.mark.parametrize("scenario", tuple(SCENARIOS))
def test_all_profiles_keep_source_native_control_and_require_real_preflight(tmp_path, source, scenario):
    before = digest(SOURCE)
    original, config, _, control, seed, receipt = prepare_playback(ROOT, SOURCE, tmp_path / scenario, scenario)
    assert receipt["status"] == "static_passed"
    assert original == source[0]
    assert digest(SOURCE) == before
    assert config["requires_design_preflight"] is True
    assert config["recovery_training"] == source[1]["recovery_training"]
    assert (config["physics_dt"], config["policy_dt"]) == (.005, .02)
    assert (control["actuators"]["leg"]["kp"], control["actuators"]["leg"]["kd"]) == (160., 2.5)
    evidence = json.loads((tmp_path / scenario / "CPU_preflight.json").read_text())
    assert evidence["qualification"] is None and evidence["scene_run"] is False
    assert evidence["source_full_role"]["status"] == "static_passed"


def test_profile_rejects_effective_geometry_drift(tmp_path):
    from wheeled_tasks.chassis.preflight import design_preflight
    _, config, _, control, seed, _ = prepare_playback(ROOT, SOURCE, tmp_path / "stairs", "stairs_up_10mm")
    config["evaluation"]["cases"][0]["terrain_limits"]["stair_heights_m"][0] = .011
    with pytest.raises(ValueError):
        design_preflight(config, control, ROOT, stage=config["enabled_stages"][0], role="playback", seed=seed, num_envs=1)


def test_small_stair_geometry_is_three_real_rises_and_no_special_request(source):
    for mm in (5, 10, 20, 40):
        for direction in ("up", "down"):
            config = project_scene(source[1], f"stairs_{direction}_{mm:02d}mm", 290929)
            case = config["evaluation"]["cases"][0]
            surfaces = terrain_surfaces(case["terrain"], case["terrain_limits"], 1.)
            differences = [b.z0 - a.z0 for a, b in zip(surfaces[:3], surfaces[1:4])]
            assert all(abs(abs(rise) - mm / 1000.) < 1e-12 for rise in differences)
            assert all((rise > 0) == (direction == "up") for rise in differences)
            assert case["skill"]["kind"] == "constant" and case["skill"]["mode"] == 0
            assert "request_events" not in case["skill"] and "manual_step_mode" not in case["skill"]


def test_fallen_birth_is_preserved_exactly(source):
    by_name = {c["name"]: c for c in source[1]["evaluation"]["cases"]}
    for scenario, name in FALLEN_CASES.items():
        assert project_scene(source[1], scenario, 290929)["evaluation"]["cases"] == [by_name[name]]


def dummy_env(recovery=None):
    env = SimpleNamespace(commands=torch.tensor([[0., 0., .305]]), command_target=torch.zeros(1, 2),
        command_acceleration=torch.zeros(1), raw_height=torch.tensor([.305]), policy_dt=.02,
        history=SimpleNamespace(last_tick=12), mode=torch.zeros(1, dtype=torch.long), recovery=recovery,
        cfg=dict(command_slew=dict(vx_m_s2=1.5, yaw_rad_s2=4.)))
    env.update_targets = lambda: None
    env.update_command_reference = lambda: None
    env.get_observations = lambda: {"policy": env.commands.clone()}
    return env


def test_actor_observes_single_current_slew_and_end_target_is_pinned():
    env = dummy_env()
    previous = env.commands.clone()
    observation, command, acceleration = apply_requested_command(env, previous, [3., 5., .305], False)
    assert command[:2] == pytest.approx([.03, .08])
    assert acceleration == pytest.approx([1.5, 4.])
    assert observation["policy"].tolist()[0] == pytest.approx(command)
    assert torch.equal(env.command_target, env.commands[:, :2])
    # The native end-of-step slew has zero remaining increment.
    assert torch.equal(env.command_target - env.commands[:, :2], torch.zeros(1, 2))
    assert env.history.last_tick is None


def test_recovery_motion_waits_for_native_gate_then_uses_same_slew():
    recovery = SimpleNamespace(enabled=torch.tensor([True]), fsm=SimpleNamespace(motion_released=torch.tensor([False])),
        followup_command=torch.tensor([[0., 0., .305]]))
    env = dummy_env(recovery)
    previous = env.commands.clone()
    _, command, _ = apply_requested_command(env, previous, [3., 5., .4], False)
    assert command == pytest.approx([0., 0., .305])
    recovery.fsm.motion_released[0] = True
    _, command, _ = apply_requested_command(env, previous, [3., 5., .305], False)
    assert command == pytest.approx([.03, .08, .305])
    assert recovery.followup_command.tolist()[0] == pytest.approx(command)


def test_current_candidate_onnx_identity_and_abi(source):
    session, metadata = validated_session(POLICY, SOURCE, source[0])
    assert metadata["onnx_sha256"] == digest(POLICY)
    assert session.get_inputs()[0].shape[-1] == 35
    assert session.get_outputs()[0].shape[-1] == 6
