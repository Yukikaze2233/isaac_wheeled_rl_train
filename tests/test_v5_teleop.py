from wheeled_tasks.chassis.teleop import KeyboardCommand
from types import SimpleNamespace

import pytest


def test_held_keys_have_correct_sign_and_release_does_not_reverse_command():
    command = KeyboardCommand()
    command.key("W", True)
    command.key("A", True)
    for _ in range(100):
        value = command.advance(.01)
    assert value[:2] == [.5, 1.]
    command.stop()
    command.key("W", False)
    command.key("A", False)
    assert command.advance(.01)[:2] == [0., 0.]


def test_opposing_keys_cancel_and_height_stays_in_trained_domain():
    command = KeyboardCommand()
    for name in ("W", "S", "A", "D"):
        command.key(name, True)
    assert command.advance(.1)[:2] == [0., 0.]
    command.key("Q", True)
    command.key("T", True)
    for _ in range(1000):
        command.advance(.01)
    assert command.height == .32
    command.stop()
    command.key("G", True)
    for _ in range(1000):
        command.advance(.01)
    assert command.height == .29


def test_character_events_do_not_access_enum_attributes_or_drive_robot():
    assert KeyboardCommand.event_fields(SimpleNamespace(type="CHAR", input="w")) == (None, None)
    enum_key = SimpleNamespace(name="W")
    enum_type = SimpleNamespace(name="KEY_PRESS")
    assert KeyboardCommand.event_fields(SimpleNamespace(type=enum_type, input=enum_key)) == ("W", "KEY_PRESS")
    assert KeyboardCommand.event_fields(SimpleNamespace(type="KEY_RELEASE", input="w")) == ("W", "KEY_RELEASE")


def test_fast_playback_reaches_full_speed_and_reverses_within_one_second():
    command = KeyboardCommand(vx_acceleration=6., yaw_acceleration=16.)
    command.vx_limit, command.yaw_limit = 3., 4.
    command.key("W", True)
    command.key("A", True)
    for _ in range(25):
        value = command.advance(.02)
    assert value[:2] == [3., 4.]
    command.key("W", False)
    command.key("S", True)
    previous = value[0]
    for _ in range(50):
        value = command.advance(.02)
        assert value[0] <= previous and previous - value[0] <= .12 + 1e-12
        previous = value[0]
    assert value[0] == -3.
    command.stop()
    assert command.advance(.02)[:2] == [0., 0.]


def test_keyboard_ramp_is_time_based_and_respects_contract_height_range():
    commands = [KeyboardCommand(6., 16., height_range=(.23, .43)) for _ in range(2)]
    for command, dt, ticks in zip(commands, (.01, .02), (25, 12)):
        command.vx_limit = 3.
        command.key("W", True)
        for _ in range(ticks):
            command.advance(dt)
    commands[1].advance(.01)
    assert commands[0].vx == pytest.approx(commands[1].vx)
    command = commands[0]
    command.stop()
    command.key("Q", True)
    for _ in range(1000):
        command.advance(.02)
    assert command.height == .43
    command.stop()
    command.key("E", True)
    for _ in range(1000):
        command.advance(.02)
    assert command.height == .23


@pytest.mark.parametrize("duration", [0., -1., float("nan")])
def test_invalid_duration_does_not_advance_keyboard_state(duration):
    command = KeyboardCommand(6., 16.)
    command.key("W", True)
    with pytest.raises(ValueError):
        command.advance(duration)
    assert command.vx == 0.


def test_manual_jump_rebases_only_task_state_and_selects_trained_reference():
    import importlib.util
    from pathlib import Path
    import torch
    from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
    from wheeled_tasks.chassis.task import FallConfirmation, PhaseTracker

    path = Path(__file__).resolve().parents[1] / "scripts/play_v5_grounded.py"
    spec = importlib.util.spec_from_file_location("manual_playback", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = {"jump_request_seconds": .8, "jump_apex_delta_m": .06}
    pose = torch.tensor([[1., 2., .305, 0., 0., 0., 1.]])
    before = pose.clone()
    env = SimpleNamespace(full_tasks=FullTaskSemantics(1, "cpu", .02, dict(settings)),
        cfg={"task_semantics": dict(settings)}, episode_length_buf=torch.tensor([500]),
        phase=PhaseTracker(1, "cpu", .02), fall_confirmation=FallConfirmation(1, "cpu", .02),
        robot=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose))),
        origins=torch.zeros(1, 3), success_hold=torch.ones(1), jump_requested=torch.zeros(1, dtype=torch.bool),
        mode=torch.zeros(1, dtype=torch.long), policy_dt=.02, update_targets=lambda: None)
    contract = {"task_semantics": settings, "skill_specs": {
        "jump_full": {"kind": "jump", "semantics": {"jump_apex_delta_m": .1, "jump_com_rise_m": .06}}}}
    module.begin_manual_jump(env, contract, .1)
    assert env.mode.item() == 4 and env.episode_length_buf.item() == 40
    assert env.full_tasks.cfg["jump_apex_delta_m"] == .1
    assert env.full_tasks.cfg["jump_com_rise_m"] == .06
    assert torch.equal(env.full_tasks.origin_xy, pose[:, :2])
    assert torch.equal(pose, before)
    with pytest.raises(ValueError, match="No trained"):
        module.begin_manual_jump(env, contract, .2)
