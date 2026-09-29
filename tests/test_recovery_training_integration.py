"""Check the physical-prelude adapter's control/learning and outcome boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.recovery import RecoveryPhase
from wheeled_tasks.chassis.recovery_training import RecoveryTraining
from wheeled_tasks.chassis.v5_control import V5Control

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rig(tmp_path):
    cfg = json.loads((ROOT / "contracts/v6_step_curriculum_training_v1.json").read_text())["runtime_base"]
    bundle = ROOT / cfg["asset_directory"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    spec = json.loads((bundle / "model_spec.json").read_text())
    fit = json.loads((bundle / "fit_10mpa.json").read_text())
    control = json.loads((ROOT / cfg["control_math_source"]).read_text())
    v5 = V5Control(manifest, spec, fit, control, cfg["v5_control"], "cpu")
    names = manifest["tree_joint_names"]
    nominal = torch.tensor([[manifest["nominal_joint_pos"][name] for name in names]]).repeat(2, 1)
    ids = [names.index(name) for name in manifest["control_joint_names"]]
    env = NS(num_envs=2, device="cpu", dt=.001, policy_dt=.02, tick=0, cfg=cfg, manifest=manifest,
        nominal=nominal, ids=ids, v5=v5, scene_groups=["recover", "ordinary"], kinds=["flat", "flat"],
        origins=torch.tensor([[10., 20., 0.], [30., 40., 0.]]), commands=torch.zeros(2, 3),
        raw_height=torch.zeros(2), command_target=torch.zeros(2, 2), command_acceleration=torch.zeros(2),
        actions=torch.zeros(2, 6),
        skills=NS(specs={"recover": {"recovery_pose": "upright", "recovery_followup_command": [.3, 0., .305]}}),
        robot=NS(joint_names=names, data=NS(joint_pos=NS(torch=nominal.clone()),
            root_link_lin_vel_w=NS(torch=torch.zeros(2, 3)),
            root_link_pose_w=NS(torch=torch.tensor([[0., 0., .305, 0., 0., 0., 1.]]).repeat(2, 1)))))
    table = {"control_joint_names": manifest["control_joint_names"],
        "asset_manifest_sha256": cfg["asset_manifest_sha256"], "axis_signs": [1., 1., -1., -1.],
        "wheel_axis_signs": [1., -1.],
        "prepare_reference": {"geometry_control6_rad": v5.nominal.tolist(),
            "pd_control6_rad": v5.nominal.tolist(), "hold_torque_nm": [0.] * 6, "kp": 160.},
        "targets": {name: {"control6_rad": v5.nominal.tolist()} for name in ("nominal", "fold", "thrust", "support")},
        "profiles": {"upright": {"enabled": True, "terrain": "flat", "root_position_m": [0., 0., .305],
            "root_quaternion_xyzw": [0., 0., 0., 1.], "full_joint_positions": manifest["nominal_joint_pos"],
            "initial_release_s": .2}}}
    (tmp_path / "profiles.json").write_text(json.dumps(table))
    (tmp_path / "lookup.json").write_bytes((ROOT / "contracts/v6_height_lookup_v1.json").read_bytes())
    settings = {"enabled": True, "profiles_file": "profiles.json", "height_lookup_file": "lookup.json",
        **{prefix + "_sha256": hashlib.sha256((tmp_path / file).read_bytes()).hexdigest()
           for prefix, file in (("profiles", "profiles.json"), ("height_lookup", "lookup.json"))}}
    recovery = RecoveryTraining(env, tmp_path, settings)
    recovery.reset(torch.arange(2))
    return env, recovery


def test_release_is_zero_active_effort_with_no_rl_credit(rig):
    env, recovery = rig
    q, dq = env.nominal[:, env.ids], torch.zeros(2, 6)
    env.v5.motor_efforts(q, dq, q[:, [0, 1, 3, 4]], dq[:, [2, 5]])
    recovery.begin_policy()
    result = recovery.torque(q, dq, torch.ones(2, 6))
    assert torch.equal(result[0], torch.zeros(6))
    assert torch.equal(result[1], torch.ones(6))
    assert recovery.learning_mask.tolist() == [False, True]
    assert recovery.pure_rl.tolist() == [False, True]
    assert env.commands[:, 0].tolist() == [0., 0.]


def test_whole_mixed_policy_interval_is_excluded(rig):
    env, recovery = rig
    recovery.release_finished[:] = True
    recovery.fsm.phase[0] = RecoveryPhase.BLEND
    recovery.begin_policy()
    recovery.fsm.phase[0] = RecoveryPhase.RL
    q, dq = env.nominal[:, env.ids], torch.zeros(2, 6)
    torque = env.v5.motor_efforts(q, dq, q[:, [0, 1, 3, 4]], dq[:, [2, 5]])
    recovery.torque(q, dq, torque)
    assert recovery.learning_mask.tolist() == [False, True]
    recovery.begin_policy()
    assert recovery.learning_mask.all()


def test_initial_profile_only_replaces_selected_lanes(rig):
    env, recovery = rig
    root, joint, velocity = torch.ones(2, 7), torch.ones_like(env.nominal), torch.ones(2, 6)
    recovery.initial_state(torch.arange(2), root, joint, velocity)
    assert torch.allclose(root[0, :3], torch.tensor([10., 20., .305]))
    assert root[1].eq(1).all() and joint[1].eq(1).all() and velocity[1].eq(1).all()
    assert torch.equal(joint[0], env.nominal[0]) and velocity[0].eq(0).all()


def test_entering_rl_or_inverted_quiet_pose_is_not_success(rig):
    env, recovery = rig
    recovery.release_finished[:] = True
    recovery.fsm.phase[0] = RecoveryPhase.RL
    recovery.fsm.motion_released[:] = False
    data = (torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
            torch.tensor([[0., 0., 1.], [0., 0., -1.]]), torch.zeros(2, 3),
            torch.full((2,), .305), torch.zeros(2, 3))
    for _ in range(120):
        assert not recovery.outcome(*data).any()
    assert recovery.true_stable_peak[0] == 0.
    # Real supported upright hold, followed by the requested forward movement.
    data[2][0, 2] = -1.
    for _ in range(51):
        recovery.outcome(*data)
    assert recovery.true_stable_peak[0] >= 1.
    recovery.fsm.motion_released[:] = True
    data[-1][0, 0] = .3
    for _ in range(51):
        recovery.outcome(*data)
    assert recovery.true_success.tolist() == [True, False]
    data[0][0, 1] = False
    assert not recovery.outcome(*data)[0]


def test_command_hold_and_release_do_not_reset_controller_history(rig):
    env, recovery = rig
    reference = recovery.fsm.script_targets.clone()
    recovery.fsm.motion_released[:] = True
    recovery.override_commands()
    assert env.commands[0, 0] == 0.  # Initial zero-torque release still holds commands.
    recovery.release_finished[:] = True
    recovery.override_commands()
    assert env.commands[0, 0] == pytest.approx(.3)
    assert torch.equal(reference, recovery.fsm.script_targets)
