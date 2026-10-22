"""Reset-state velocity dataflow only; no dynamic recovery qualification."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.application.runtime.recovery_training import RecoveryTraining


NAMES = tuple(f"joint_{index}" for index in range(18))


def make_recovery():
    recovery = RecoveryTraining.__new__(RecoveryTraining)
    recovery.env = NS(
        robot=NS(joint_names=list(reversed(NAMES))), cfg={},
        origins=torch.tensor([[10., 20., 0.], [30., 40., 0.], [50., 60., 0.]]),
    )
    recovery.settings = {}
    recovery.pose_curriculum = None
    recovery.enabled = torch.tensor([True, False, True])
    recovery.root_pose = torch.zeros(3, 7)
    recovery.joint_pose = torch.zeros(3, len(NAMES))
    recovery.initial_root_velocity = torch.zeros(3, 6)
    recovery.initial_joint_velocity = torch.zeros_like(recovery.joint_pose)
    recovery.initial_release = torch.zeros(3)
    recovery.required_platform_height = torch.zeros(3)
    recovery.min_ending_x = torch.zeros(3)
    return recovery


def profile():
    return {
        "root_position_m": [.1, .2, .3],
        "root_quaternion_xyzw": [0., 0., 0., 1.],
        "full_joint_positions": {name: index * .01 for index, name in enumerate(NAMES)},
        "full_joint_velocities": {name: index * -.02 for index, name in enumerate(NAMES)},
        "root_linear_velocity_m_s": [.4, -.5, .6],
        "root_angular_velocity_rad_s": [-.7, .8, -.9],
        "root_velocity_frame": "com_world",
    }


def outputs(count=3):
    return (torch.full((count, 7), 91.), torch.full((count, len(NAMES)), 92.),
            torch.full((count, 6), 93.), torch.full((count, len(NAMES)), 94.))


def test_frozen_profile_velocities_follow_native_joint_order_and_scatter_lanes():
    recovery = make_recovery()
    source = profile()
    recovery._load_initial_profile(0, source, "moving")
    recovery._load_initial_profile(2, source, "moving")
    root, joints, root_velocity, joint_velocity = outputs()
    recovery.initial_state(torch.arange(3), root, joints, root_velocity, joint_velocity)
    torch.testing.assert_close(root[0, :3], torch.tensor([10.1, 20.2, .3]))
    torch.testing.assert_close(root[2, :3], torch.tensor([50.1, 60.2, .3]))
    native_names = recovery.env.robot.joint_names
    for lane in (0, 2):
        torch.testing.assert_close(joints[lane], torch.tensor([
            source["full_joint_positions"][name] for name in native_names]))
        torch.testing.assert_close(joint_velocity[lane], torch.tensor([
            source["full_joint_velocities"][name] for name in native_names]))
        torch.testing.assert_close(root_velocity[lane], torch.tensor([.4, -.5, .6, -.7, .8, -.9]))
    for value, sentinel in zip((root, joints, root_velocity, joint_velocity), (91., 92., 93., 94.)):
        assert value[1].eq(sentinel).all()
    # Caller mutation cannot change the owned profile state.
    root_velocity.zero_()
    joint_velocity.zero_()
    assert recovery.initial_root_velocity[0].abs().sum() > 0.
    assert recovery.initial_joint_velocity[0].abs().sum() > 0.


def test_subset_reset_uses_selected_global_lane_and_preserves_other_output_rows():
    recovery = make_recovery()
    recovery._load_initial_profile(2, profile(), "moving")
    state = outputs(2)
    recovery.initial_state(torch.tensor([1, 2]), *state)
    assert state[0][0].eq(91.).all()
    assert state[1][0].eq(92.).all()
    assert state[2][0].eq(93.).all()
    assert state[3][0].eq(94.).all()
    torch.testing.assert_close(state[0][1, :3], torch.tensor([50.1, 60.2, .3]))
    torch.testing.assert_close(state[3][1], recovery.initial_joint_velocity[2])


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_legacy_zero_profile_preserves_four_argument_call(explicit_zero):
    recovery = make_recovery()
    source = profile()
    for key in ("root_linear_velocity_m_s", "root_angular_velocity_rad_s", "full_joint_velocities", "root_velocity_frame"):
        source.pop(key)
    if explicit_zero:
        source.update(root_linear_velocity_m_s=[0.] * 3, root_angular_velocity_rad_s=[0.] * 3,
                      full_joint_velocities=dict.fromkeys(NAMES, 0.))
    recovery._load_initial_profile(0, source, "legacy")
    root, joints, root_velocity, joint_velocity = outputs()
    recovery.initial_state(torch.arange(3), root, joints, root_velocity)
    assert root_velocity[[0, 2]].eq(0.).all()
    assert root_velocity[1].eq(93.).all()
    recovery.initial_state(torch.arange(3), root, joints, root_velocity, joint_velocity)
    assert joint_velocity[[0, 2]].eq(0.).all()
    assert joint_velocity[1].eq(94.).all()


def test_nonzero_joint_velocity_cannot_be_silently_dropped():
    recovery = make_recovery()
    recovery._load_initial_profile(0, profile(), "moving")
    root, joints, velocity, _ = outputs()
    before = tuple(value.clone() for value in (root, joints, velocity))
    with pytest.raises(ValueError, match="requires a joint_velocity reset output"):
        recovery.initial_state(torch.arange(3), root, joints, velocity)
    for value, original in zip((root, joints, velocity), before):
        assert torch.equal(value, original)


@pytest.mark.parametrize("frame", [None, "body", "link_world", "com_body", "world"])
def test_nonzero_root_velocity_requires_explicit_com_world_frame(frame):
    recovery = make_recovery()
    source = profile()
    if frame is None:
        source.pop("root_velocity_frame")
    else:
        source["root_velocity_frame"] = frame
    with pytest.raises(ValueError, match="root_velocity_frame=com_world"):
        recovery._load_initial_profile(0, source, "ambiguous")
    assert recovery.initial_root_velocity.eq(0.).all()
    assert recovery.initial_joint_velocity.eq(0.).all()


@pytest.mark.parametrize("field,bad", [
    ("root_position_m", [0., 0.]),
    ("root_quaternion_xyzw", [0., 0., 1.]),
    ("root_linear_velocity_m_s", [0., 0.]),
    ("root_angular_velocity_rad_s", [[0.], [0.], [0.]]),
    ("root_linear_velocity_m_s", [0., float("nan"), 0.]),
    ("root_angular_velocity_rad_s", [0., float("inf"), 0.]),
    ("full_joint_velocities", None),
    ("full_joint_positions", {"joint_0": 0.}),
])
def test_profile_rejects_malformed_or_nonfinite_state(field, bad):
    recovery = make_recovery()
    source = profile()
    source[field] = bad
    with pytest.raises(ValueError):
        recovery._load_initial_profile(0, source, "bad")


@pytest.mark.parametrize("field", ["full_joint_positions", "full_joint_velocities"])
@pytest.mark.parametrize("change", ["missing", "extra", "nested", "nonfinite"])
def test_full_named_joint_vectors_are_required(field, change):
    recovery = make_recovery()
    source = profile()
    if change == "missing":
        source[field].pop(NAMES[-1])
    elif change == "extra":
        source[field]["unknown_joint"] = 0.
    elif change == "nested":
        source[field] = {name: [value] for name, value in source[field].items()}
    else:
        source[field][NAMES[-1]] = float("inf")
    with pytest.raises(ValueError):
        recovery._load_initial_profile(0, source, "bad")


@pytest.mark.parametrize("prefix", [False, True])
def test_existing_curriculum_clears_velocity_on_selected_lanes(monkeypatch, prefix):
    recovery = make_recovery()
    for lane in (0, 2):
        recovery._load_initial_profile(lane, profile(), "moving")
    recovery.settings = {"prefix_curriculum": {"enabled": prefix}}
    monkeypatch.setattr("wheeled_tasks.chassis.application.runtime.recovery_training.stage_actor_updates", lambda env: 10000)
    calls = []
    bank = NS(
        reset=lambda ids, clock: torch.tensor([0, -1]),
        root_pose_bank=recovery.root_pose[:1].clone(), joint_pose_bank=recovery.joint_pose[:1].clone(),
        initial_release_bank=torch.zeros(1), required_platform_height_bank=torch.zeros(1),
        min_ending_x_bank=torch.zeros(1),
        apply_lesson=lambda owner, lanes: calls.append(lanes.tolist()),
    )
    recovery.pose_curriculum = bank
    original_root_velocity = recovery.initial_root_velocity[2].clone()
    original_joint_velocity = recovery.initial_joint_velocity[2].clone()
    # Only lane 0 is chosen from the zero-velocity bank; lane 2 is not reset.
    root, joints, velocity, joint_velocity = outputs(2)
    recovery.initial_state(torch.tensor([0, 1]), root, joints, velocity, joint_velocity)
    assert recovery.initial_root_velocity[0].eq(0.).all()
    assert recovery.initial_joint_velocity[0].eq(0.).all()
    assert velocity[0].eq(0.).all() and joint_velocity[0].eq(0.).all()
    assert velocity[1].eq(93.).all() and joint_velocity[1].eq(94.).all()
    assert torch.equal(recovery.initial_root_velocity[2], original_root_velocity)
    assert torch.equal(recovery.initial_joint_velocity[2], original_joint_velocity)
    assert calls == ([[0]] if prefix else [])


def test_joint_velocity_output_must_match_native_joint_shape():
    recovery = make_recovery()
    root, joints, velocity, _ = outputs()
    with pytest.raises(ValueError, match="must match the joint position tensor"):
        recovery.initial_state(torch.arange(3), root, joints, velocity, torch.zeros(3, 6))


def test_existing_frozen_profiles_keep_zero_velocity_without_frame_annotation():
    root = Path(__file__).resolve().parents[3]
    table = json.loads((root / "reports/v7_formal_training_20261012/recovery107_candidate/recovery_profiles_107_v2.json").read_text())
    recovery = make_recovery()
    recovery.env.robot.joint_names = list(reversed(table["tree_joint_names"]))
    for name, source in table["profiles"].items():
        recovery._load_initial_profile(0, source, name)
        assert recovery.initial_root_velocity[0].eq(0.).all()
        assert recovery.initial_joint_velocity[0].eq(0.).all()
        expected = [source["full_joint_positions"][key] for key in recovery.env.robot.joint_names]
        torch.testing.assert_close(recovery.joint_pose[0], torch.tensor(expected))
