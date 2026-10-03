"""Exercise frozen v8 reset selection through the real recovery adapter."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from test_recovery_training_integration import rig
from wheeled_tasks.chassis.recovery import RecoveryPhase as Phase
from wheeled_tasks.chassis.recovery_training import RecoveryTraining
from wheeled_tasks.chassis.v5_control import V5Control


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "contracts/v6_task_campaign_recovery_worker_v8.json"


@pytest.fixture
def v8_rig(rig):
    env, _ = rig
    config = json.loads(WORKER.read_text())
    env.cfg = config
    env.dt, env.policy_dt = config["physics_dt"], config["policy_dt"]
    env.task_actor_update = config["recovery_training"]["pose_curriculum"]["origin_actor_update"]
    env.generator = torch.Generator().manual_seed(715)
    env.scene_groups = ["recovery_front_down_90_forward", "stand_305mm"]
    env.skills.specs = deepcopy(config["skill_specs"])
    bundle = ROOT / config["asset_directory"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest() == config["asset_manifest_sha256"]
    env.manifest = manifest
    env.v5 = V5Control(manifest, json.loads((bundle / "model_spec.json").read_text()),
        json.loads((bundle / "fit_10mpa.json").read_text()),
        json.loads((ROOT / config["control_math_source"]).read_text()), config["v5_control"], "cpu")
    table_path = ROOT / config["recovery_training"]["profiles_file"]
    assert hashlib.sha256(table_path.read_bytes()).hexdigest() == config["recovery_training"]["profiles_sha256"]
    table = json.loads(table_path.read_text())
    recovery = RecoveryTraining(env, ROOT, config["recovery_training"])
    return env, recovery, table


def rebuild(env, groups, terrains, *, exact=False):
    env.scene_groups, env.kinds = list(groups), list(terrains)
    env.cfg["evaluation_exact_cases"] = exact
    return RecoveryTraining(env, ROOT, env.cfg["recovery_training"])


def initialize(env, recovery, ids=None):
    """Perform the same initial_state -> simulator write -> adapter reset order."""
    ids = torch.arange(env.num_envs) if ids is None else ids
    root = env.robot.data.root_link_pose_w.torch[ids].clone()
    joints = env.robot.data.joint_pos.torch[ids].clone()
    velocity = torch.zeros(len(ids), 6)
    recovery.initial_state(ids, root, joints, velocity)
    env.robot.data.root_link_pose_w.torch[ids] = root
    env.robot.data.joint_pos.torch[ids] = joints
    env.robot.data.root_link_lin_vel_w.torch[ids] = velocity[:, :3]
    recovery.reset(ids)
    return root, joints, velocity


def activate(env, recovery, phase=Phase.RL):
    initialize(env, recovery)
    # Finite upright sensor fixture: this tests control ownership, not dynamics.
    q = env.nominal[:, env.ids].clone()
    dq = torch.zeros_like(q)
    env.robot.data.joint_pos.torch[:] = env.nominal
    env.robot.data.root_link_pose_w.torch[:, 3:] = torch.tensor([0., 0., 0., 1.])
    recovery.encoders.reset(torch.arange(env.num_envs), q)
    recovery.release_finished[:] = True
    recovery.fsm.reset(torch.arange(env.num_envs), q, recovery.enabled)
    recovery.fsm.phase[0] = phase
    recovery.handover_count[0] = int(phase == Phase.RL)
    env.v5.motor_efforts(q, dq, q[:, list(env.v5.LEGS)], dq[:, list(env.v5.WHEELS)])
    return q, dq


def advance(recovery, q, dq, *, gravity=None, gyro=None):
    gravity = torch.tensor([[0., 0., -1.]]).repeat(len(q), 1) if gravity is None else gravity
    gyro = torch.zeros(len(q), 3) if gyro is None else gyro
    recovery.advance_reference(q, dq, gravity, gyro)


def test_real_v8_initial_state_gathers_only_active_selected_lanes(v8_rig):
    env, recovery, table = v8_rig
    assert recovery.settings["complete_started_blend"]
    assert recovery.pose_curriculum.settings["origin_actor_update"] == 400
    root, joints, velocity = torch.full((2, 7), 7.), torch.full_like(env.nominal, 8.), torch.full((2, 6), 9.)
    recovery.initial_state(torch.arange(2), root, joints, velocity)
    curriculum = recovery.pose_curriculum
    index = curriculum.profile_ids[0]
    expected = curriculum.root_pose_bank[index].clone()
    expected[:3] += env.origins[0]
    torch.testing.assert_close(root[0], expected)
    torch.testing.assert_close(joints[0], curriculum.joint_pose_bank[index])
    assert velocity[0].eq(0).all()
    assert root[1].eq(7).all() and joints[1].eq(8).all() and velocity[1].eq(9).all()
    assert curriculum.profile_ids[1] == -1
    assert curriculum.profile_names[index] != "front_down_90"
    assert curriculum.level_ids[0] in (0, 1)
    diagnostic = recovery.diagnostics()
    assert torch.equal(diagnostic["recovery_actual_profile_id"], curriculum.profile_ids)
    assert torch.equal(diagnostic["recovery_actual_level_id"], curriculum.level_ids)
    assert table["profiles"][curriculum.profile_names[index]]["geometry_valid"]


def test_reset_refreshes_same_surface_platform_targets_from_selected_bank(v8_rig):
    env, _, _ = v8_rig
    recovery = rebuild(env, ["recovery_double_step_first_riser_split_stand",
        "recovery_double_step_second_side_top_stand"], ["stairs", "stairs"])
    recovery.required_platform_height[:] = .99
    recovery.min_ending_x[:] = -5.
    recovery.goal_platform[:] = True
    initialize(env, recovery)
    indices = recovery.pose_curriculum.profile_ids
    for field in ("root_pose", "joint_pose", "initial_release", "required_platform_height", "min_ending_x"):
        torch.testing.assert_close(getattr(recovery, field), getattr(recovery.pose_curriculum, field + "_bank")[indices])
    assert recovery.required_platform_height.tolist() == pytest.approx([.2, .45])
    assert recovery.min_ending_x.tolist() == pytest.approx([.22, 1.02])
    assert recovery.initial_release.eq(0).all()
    assert not recovery.goal_platform.any()  # Standing on the target needs observed support evidence.
    assert recovery.release_finished.all()
    assert all("upright_top" in recovery.pose_curriculum.profile_names[i] for i in indices)


def test_reference_advance_does_not_resample_or_write_physical_state(v8_rig):
    env, recovery, _ = v8_rig
    q, dq = activate(env, recovery, Phase.PREPARE)
    physical = [env.robot.data.root_link_pose_w.torch, env.robot.data.joint_pos.torch,
                env.robot.data.root_link_lin_vel_w.torch]
    saved_physical = [value.clone() for value in physical]
    saved_profiles = recovery.pose_curriculum.profile_ids.clone()
    saved_counts = recovery.pose_curriculum.profile_reset_counts.clone()
    saved_pose = recovery.root_pose.clone()
    rng = env.generator.get_state().clone()
    env.task_actor_update += 6000
    for _ in range(8):
        advance(recovery, q, dq)
    for value, saved in zip(physical, saved_physical):
        torch.testing.assert_close(value, saved)
    assert torch.equal(recovery.root_pose, saved_pose)
    assert torch.equal(recovery.pose_curriculum.profile_ids, saved_profiles)
    assert torch.equal(recovery.pose_curriculum.profile_reset_counts, saved_counts)
    assert torch.equal(env.generator.get_state(), rng)


def test_partial_reset_preserves_other_profile_state_and_its_disjoint_counts(v8_rig):
    env, _, table = v8_rig
    recovery = rebuild(env, ["recovery_front_down_90_forward", "recovery_double_step_second_side_top_stand"],
                       ["flat", "stairs"])
    initialize(env, recovery)
    curriculum = recovery.pose_curriculum
    other_profile, other_level = curriculum.profile_ids[1].clone(), curriculum.level_ids[1].clone()
    other_root = env.robot.data.root_link_pose_w.torch[1].clone()
    other_joints = env.robot.data.joint_pos.torch[1].clone()
    stairs = torch.tensor([index for index, name in enumerate(curriculum.profile_names)
                           if table["profiles"][name]["terrain"] == "stairs"])
    other_counts = curriculum.profile_reset_counts[stairs].clone()
    old_total = curriculum.profile_reset_counts.sum().clone()
    env.task_actor_update += 6000
    for _ in range(6):
        initialize(env, recovery, torch.tensor([0]))
    assert curriculum.profile_ids[1] == other_profile and curriculum.level_ids[1] == other_level
    torch.testing.assert_close(env.robot.data.root_link_pose_w.torch[1], other_root)
    torch.testing.assert_close(env.robot.data.joint_pos.torch[1], other_joints)
    assert torch.equal(curriculum.profile_reset_counts[stairs], other_counts)
    assert curriculum.profile_reset_counts.sum() == old_total + 6


@pytest.mark.parametrize("group,terrain", [
    ("recovery_front_down_90_forward", "flat"),
    ("recovery_double_step_second_side_top_stand", "stairs"),
])
def test_exact_cases_bypasses_curriculum_and_retains_original_profile(v8_rig, group, terrain):
    env, _, table = v8_rig
    env.task_actor_update += 6000
    recovery = rebuild(env, [group, "stand_305mm"], [terrain, "flat"], exact=True)
    assert recovery.pose_curriculum is None
    rng = env.generator.get_state().clone()
    root, joints, _ = initialize(env, recovery)
    name = env.skills.specs[group]["recovery_pose"]
    profile = table["profiles"][name]
    expected = torch.tensor(profile["root_position_m"] + profile["root_quaternion_xyzw"])
    expected[:3] += env.origins[0]
    torch.testing.assert_close(root[0], expected)
    torch.testing.assert_close(joints[0], torch.tensor([profile["full_joint_positions"][n] for n in env.robot.joint_names]))
    assert recovery.initial_release[0] == pytest.approx(profile["initial_release_s"])
    assert recovery.profile_names[0] == name
    assert torch.equal(env.generator.get_state(), rng)


def test_wrapper_allows_started_blend_to_finish_past_eight_seconds(v8_rig):
    env, recovery, _ = v8_rig
    q, dq = activate(env, recovery, Phase.BLEND)
    recovery.elapsed[0] = 7.995
    recovery.fsm.age_ticks[0] = 1599
    for tick in range(40):
        if tick % 4 == 0:
            recovery.begin_policy()
        advance(recovery, q, dq)
        recovery.torque(q, dq, torch.ones_like(q))
        assert not recovery.failed[0]
        assert not recovery.learning_mask[0]
    assert recovery.elapsed[0] > 8.
    assert recovery.fsm.phase[0] == Phase.RL
    recovery.begin_policy()
    assert recovery.learning_mask[0]
    assert recovery.fsm.failure_code[0] == 0


def test_wrapper_still_expires_nonblending_script(v8_rig):
    env, recovery, _ = v8_rig
    activate(env, recovery, Phase.PREPARE)
    recovery.elapsed[0] = 8.
    assert recovery.failed[0]
    assert recovery.failure_reasons()["recovery_script"][0]


@pytest.mark.parametrize("invalid_sensor", ["gravity", "gyro"])
def test_blend_extension_does_not_hide_invalid_feedback(v8_rig, invalid_sensor):
    env, recovery, _ = v8_rig
    q, dq = activate(env, recovery, Phase.BLEND)
    recovery.elapsed[0] = 8.1
    recovery.begin_policy()
    gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    gyro = torch.zeros(2, 3)
    (gravity if invalid_sensor == "gravity" else gyro)[0, 0] = float("nan")
    advance(recovery, q, dq, gravity=gravity, gyro=gyro)
    effort = recovery.torque(q, dq, torch.ones_like(q))
    assert recovery.failed[0] and recovery.fsm.failure_code[0] == 1
    assert effort[0].eq(0).all()
    assert not recovery.learning_mask[0]


def test_begin_policy_hold_snapshot_matches_observed_command_at_release_boundary(v8_rig):
    env, recovery, _ = v8_rig
    q, dq = activate(env, recovery)
    recovery.fsm.motion_released[0] = False
    recovery.override_commands()
    recovery.begin_policy()
    assert recovery.policy_motion_hold.tolist() == [True, False]
    assert env.commands[0, :2].eq(0).all()
    recovery.fsm.stable_ticks[0] = 199
    advance(recovery, q, dq)
    assert recovery.fsm.motion_released[0]
    assert recovery.policy_motion_hold[0]  # The preceding action saw the held command.
    assert env.commands[0, :2].eq(0).all()
    recovery.override_commands()
    assert env.commands[0, 0] == pytest.approx(.3)
    assert recovery.policy_motion_hold[0]
    recovery.begin_policy()
    assert not recovery.policy_motion_hold[0]
    assert env.commands[0, 0] == pytest.approx(recovery.followup_command[0, 0])


def test_last_substep_actor_failure_keeps_terminal_learning_mask(v8_rig):
    env, recovery, _ = v8_rig
    q, dq = activate(env, recovery)
    recovery.begin_policy()
    for substep in range(4):
        gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
        if substep == 3:
            gravity[0] = torch.tensor([math.sin(math.pi / 3), 0., -.5])
        advance(recovery, q, dq, gravity=gravity)
        effort = recovery.torque(q, dq, torch.ones_like(q))
    assert recovery.failed[0] and recovery.fsm.failure_code[0] == 5
    assert effort[0].eq(0).all()
    assert recovery.learning_mask.tolist() == [True, True]
    assert recovery.failure_reasons()["recovery_actor_failure"][0]
    outgoing = recovery.learning_mask.clone()
    terminated = torch.tensor([True, False])
    assert recovery.bootstrap_mask(terminated).tolist() == [False, True]
    initialize(env, recovery, torch.tensor([0]))
    assert outgoing.tolist() == [True, True]
    assert recovery.bootstrap_mask(terminated).tolist() == [False, True]


def test_policy_interval_starting_in_blend_stays_excluded_after_handover(v8_rig):
    env, recovery, _ = v8_rig
    q, dq = activate(env, recovery, Phase.BLEND)
    recovery.fsm.phase_ticks[0] = 39
    recovery.begin_policy()
    for _ in range(4):
        advance(recovery, q, dq)
        recovery.torque(q, dq, torch.ones_like(q))
    assert recovery.fsm.phase[0] == Phase.RL
    assert recovery.learning_mask.tolist() == [False, True]
    recovery.begin_policy()
    assert recovery.learning_mask.tolist() == [True, True]
