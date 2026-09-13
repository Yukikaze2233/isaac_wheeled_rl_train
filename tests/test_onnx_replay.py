"""CPU artifact and teleop transition tests; GUI evidence is recorded separately."""
from enum import Enum
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from play_v40_onnx import (
    DIAGNOSTIC_FLAGS, TERMINATION_FLAGS,
    KeyboardController, NativeKeyboard, ObservationCommandBridge, create_report, load_policy, main,
)


@pytest.fixture
def artifact(tmp_path):
    model = tmp_path / "policy.onnx"
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs_history", "weight"], ["actions"])], "fixture",
        [helper.make_tensor_value_info("obs_history", TensorProto.FLOAT, [1, 125])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 6])],
        [helper.make_tensor("weight", TensorProto.FLOAT, [125, 6], [0.1] * 750)],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(proto, model)
    expected = dict(contract_id="fixture", contract_sha256="contract", asset_manifest_sha256="asset",
                    actor_obs_dim=125, critic_obs_dim=29, action_dim=6, policy={}, stage="locomotion",
                    target_versions={"runtime": "CPU test fixture"})
    raw = json.dumps(expected).encode()
    (tmp_path / "run_manifest.json").write_bytes(raw)
    sidecar = dict(expected, run_manifest=expected,
                   run_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                   onnx_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                   input=dict(name="obs_history", dtype="float32", shape=[1, 125]),
                   output=dict(name="actions", dtype="float32", shape=[1, 6]))
    model.with_suffix(".onnx.json").write_text(json.dumps(sidecar))
    return model, expected, sidecar


def test_real_cpu_onnx_without_checkpoint(artifact):
    model, expected, _ = artifact
    session, _ = load_policy(model, expected)
    assert session.get_providers() == ["CPUExecutionProvider"]
    result = session.run(None, {"obs_history": np.ones((1, 125), dtype=np.float32)})[0]
    np.testing.assert_allclose(result, 12.5, rtol=1e-5)
    assert not (model.parent / "checkpoint.pt").exists()


@pytest.mark.parametrize("key", ["contract_sha256", "asset_manifest_sha256", "actor_obs_dim", "stage"])
def test_identity_rejected(artifact, key):
    model, expected, _ = artifact
    expected[key] = "wrong"
    with pytest.raises(ValueError, match="mismatch"):
        load_policy(model, expected)


def test_corrupt_model_rejected(artifact):
    model, expected, _ = artifact
    model.write_bytes(model.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="ONNX SHA256"):
        load_policy(model, expected)


def test_manifest_bytes_rejected(artifact):
    model, expected, _ = artifact
    (model.parent / "run_manifest.json").write_text(json.dumps(expected) + "\n")
    with pytest.raises(ValueError, match="manifest SHA256"):
        load_policy(model, expected)


def test_graph_signature_rejected_even_with_updated_hash(artifact):
    model, expected, sidecar = artifact
    proto = onnx.load(model)
    proto.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
    onnx.save(proto, model)
    sidecar["onnx_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    model.with_suffix(".onnx.json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="input signature"):
        load_policy(model, expected)


def test_sidecar_required(artifact):
    model, expected, _ = artifact
    model.with_suffix(".onnx.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_policy(model, expected)


def test_report_never_overwrites(tmp_path):
    report = tmp_path / "report"
    create_report(report)
    sentinel = report / "summary.json"
    sentinel.write_text("user data")
    with pytest.raises(FileExistsError):
        create_report(report)
    assert sentinel.read_text() == "user data"


BOUNDS = dict(vx=[-2., 2.], wz=[-2., 2.], height=[.28, .32])


def test_held_keys_release_opposition_and_stop():
    controller = KeyboardController(BOUNDS)
    original = controller.requested
    controller.press("W")
    controller.press("A")
    assert controller.requested == (1., 1., .30)
    controller.press("W")
    assert controller.requested == (1., 1., .30)
    controller.release("W")
    assert controller.requested == (0., 1., .30)
    controller.press("D")
    assert controller.requested == original
    controller.release("A")
    controller.press("S")
    assert controller.requested == (-1., -1., .30)
    controller.press("SPACE")
    controller.release("D")
    controller.release("S")
    assert controller.requested == original
    assert not controller.held


def test_height_discrete_clamped_and_stop_preserves_height():
    controller = KeyboardController(BOUNDS)
    controller.press("Q")
    controller.press("Q")
    assert controller.requested == (0., 0., .31)
    for _ in range(10):
        controller.release("Q")
        controller.press("Q")
    assert controller.requested[2] == .32
    for _ in range(10):
        controller.release("E")
        controller.press("E")
    controller.stop()
    assert controller.requested == (0., 0., .28)


@pytest.mark.parametrize("kwargs", [
    dict(linear_speed=2.01), dict(angular_speed=2.01), dict(linear_speed=0),
    dict(angular_speed=-1), dict(linear_speed=float("nan")), dict(angular_speed=float("inf")),
    dict(initial=(0., 0., .33)), dict(initial=(1., 0., .30)), dict(initial=(0., 0., float("nan"))),
])
def test_keyboard_rejects_invalid_commands(kwargs):
    with pytest.raises(ValueError):
        KeyboardController(BOUNDS, **kwargs)


def test_custom_keyboard_speeds():
    controller = KeyboardController(BOUNDS, linear_speed=2., angular_speed=.5)
    controller.press("S")
    controller.press("D")
    assert controller.requested == (-2., -.5, .30)


def test_120_rpm_conversion_and_exploratory_caps_preserve_low_initial_speed():
    controller = KeyboardController(BOUNDS, exploratory=True, max_linear_speed=5, max_angular_rpm=120)
    assert controller.rpm_to_rad_s(120) == pytest.approx(12.566370614359172)
    assert controller.rad_s_to_rpm(controller.max_angular_speed) == pytest.approx(120)
    assert controller.requested == (0., 0., .3)
    assert (controller.linear_speed, controller.angular_speed) == (1., 1.)
    controller.press("W")
    controller.press("A")
    for key in ("EQUAL", "RIGHT_BRACKET"):
        for _ in range(20):
            controller.press(key)
            controller.release(key)
    assert controller.requested == (5., controller.rpm_to_rad_s(120), .3)
    assert controller.out_of_training_domain(controller.requested)
    assert "EXPLORATORY" in controller.status_text and "requested OOD" in controller.status_text
    assert "12.5664 rad/s (120.00 rpm)" in controller.status_text
    for key in ("MINUS", "LEFT_BRACKET"):
        for _ in range(20):
            controller.press(key)
            controller.release(key)
    assert controller.requested == (0., 0., .3)
    assert not controller.out_of_training_domain(controller.requested)


def test_adjustment_repeat_opposing_keys_and_stop_do_not_accumulate():
    controller = KeyboardController(BOUNDS, exploratory=True, max_linear_speed=5, max_angular_rpm=120)
    controller.press("EQUAL")
    controller.press("EQUAL")
    assert controller.linear_speed == 1.5
    controller.press("RIGHT_BRACKET")
    controller.press("RIGHT_BRACKET")
    assert controller.angular_speed == pytest.approx(1 + math.pi / 3)
    controller.press("W")
    controller.press("S")
    controller.press("A")
    assert controller.requested[0] == 0
    controller.release("S")
    assert controller.requested[0] == 1.5
    controller.press("SPACE")
    controller.release("W")
    controller.release("A")
    assert controller.requested == (0., 0., .3)
    assert controller.linear_speed == 1.5


@pytest.mark.parametrize("caps", [dict(max_linear_speed=5), dict(max_angular_rpm=120)])
def test_standard_mode_rejects_ood_caps(caps):
    with pytest.raises(ValueError, match="require --exploratory"):
        KeyboardController(BOUNDS, **caps)


@pytest.mark.parametrize("field", ["max_linear_speed", "max_angular_rpm"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0])
def test_exploratory_caps_must_be_finite_positive(field, value):
    with pytest.raises(ValueError):
        KeyboardController(BOUNDS, exploratory=True, **{field: value})


def test_standard_adjustments_clamp_to_training_domain():
    controller = KeyboardController(BOUNDS)
    for _ in range(20):
        for key in ("NUMPAD_ADD", "RIGHT_BRACKET"):
            controller.press(key)
            controller.release(key)
    controller.press("W")
    controller.press("A")
    assert controller.requested == (2., 2., .3)
    assert not controller.out_of_training_domain(controller.requested)


def test_exploratory_metadata_does_not_mutate_contract_bytes(tmp_path):
    source = Path(__file__).resolve().parents[1] / "contracts/own_v40_v2.json"
    original = source.read_bytes()
    path = tmp_path / "contract.json"
    path.write_bytes(original)
    contract = json.loads(path.read_bytes())
    before = json.dumps(contract, sort_keys=True)
    controller = KeyboardController(contract["commands"]["stages"]["locomotion"],
                                    exploratory=True, max_linear_speed=5, max_angular_rpm=120)
    for _ in range(5):
        controller.press("EQUAL")
        controller.release("EQUAL")
    controller.press("W")
    metadata = controller.metadata
    assert metadata["exploratory"] is True
    assert metadata["requested_out_of_training_domain"] is True
    assert metadata["training_command_bounds"]["vx"] == (-2., 2.)
    assert metadata["max_angular_speed_rpm"] == pytest.approx(120)
    assert json.dumps(contract, sort_keys=True) == before
    assert path.read_bytes() == original == source.read_bytes()
    with pytest.raises(ValueError):
        controller.out_of_training_domain((float("nan"), 0., .3))


@pytest.mark.parametrize("kwargs", [dict(linear_speed=5), dict(angular_speed=3), dict(initial=(0., 0., .33))])
def test_exploratory_initial_settings_stay_in_training_domain(kwargs):
    with pytest.raises(ValueError):
        KeyboardController(BOUNDS, exploratory=True, max_linear_speed=5, max_angular_rpm=120, **kwargs)


def test_keyboard_headless_rejected_before_loading(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--onnx", "missing.onnx", "--report-dir", "unused", "--keyboard", "--headless"])
    assert exc.value.code == 2
    assert "requires GUI" in capsys.readouterr().err


def test_native_callback_repeat_focus_and_cleanup():
    # Exercise the real callback without importing Isaac or installing OS hooks.
    events = []
    types = Enum("KeyboardEventType", "KEY_PRESS KEY_REPEAT KEY_RELEASE CHAR")
    native = NativeKeyboard.__new__(NativeKeyboard)
    native.controller = KeyboardController(BOUNDS)
    native.record = lambda kind, **fields: events.append((kind, fields))
    native.event_types = types
    focused = [True]
    native.window = SimpleNamespace(is_focused=lambda: focused[0])

    def send(key, kind):
        return native.on_event(SimpleNamespace(input=SimpleNamespace(name=key), type=types[kind]))

    assert not send("W", "KEY_PRESS")
    send("Q", "KEY_PRESS")
    for _ in range(5):
        send("W", "KEY_REPEAT")
        send("Q", "KEY_REPEAT")
    assert native.controller.requested == (1., 0., .31)
    send("SPACE", "KEY_PRESS")
    send("W", "KEY_REPEAT")
    assert native.controller.requested == (0., 0., .31)
    send("W", "KEY_RELEASE")
    send("W", "KEY_PRESS")
    focused[0] = False
    native.on_focus(None)
    assert native.controller.requested == (0., 0., .31)
    assert events[-1][1]["event"] == "FOCUS_LOST"
    send("A", "KEY_PRESS")
    assert native.controller.requested == (0., 0., .31)
    assert send("Z", "KEY_PRESS")
    assert native.on_event(SimpleNamespace(type=types.CHAR))
    focused[0] = True
    send("EQUAL", "KEY_PRESS")
    send("EQUAL", "KEY_REPEAT")
    send("RIGHT_BRACKET", "KEY_PRESS")
    send("RIGHT_BRACKET", "KEY_REPEAT")
    assert native.controller.linear_speed == 1.5
    assert native.controller.angular_speed == 2.0  # Normal mode clamps 1 rad/s + 10 rpm.
    releases = []
    native.closed = False
    native.keyboard, native.subscription = "own-keyboard", 12
    native.input = SimpleNamespace(unsubscribe_to_keyboard_events=lambda *args: releases.append(args))
    native.focus_subscription = SimpleNamespace(unsubscribe=lambda: releases.append("focus"))
    native.close()
    native.close()
    assert releases == [("own-keyboard", 12), "focus"]


@pytest.fixture
def transition_env():
    # Reuse the repository's source-method loader; no copied sampler/reward/obs code.
    sys.path.insert(0, str(Path(__file__).parent / "v40"))
    from test_env_contract_static import isolated_tensor_method
    torch, core, sample = isolated_tensor_method("_sample_commands")
    contract = core.load_contract(Path(__file__).resolve().parents[1] / "contracts/own_v40_v2.json")
    q = torch.tensor([contract["joints"]["nominal_positions"]])
    zero3, zero6 = torch.zeros(1, 3), torch.zeros(1, 6)
    data = SimpleNamespace(
        root_com_lin_vel_b=SimpleNamespace(torch=zero3),
        root_com_ang_vel_b=SimpleNamespace(torch=zero3),
        projected_gravity_b=SimpleNamespace(torch=torch.tensor([[0., 0., -1.]])),
    )
    env = SimpleNamespace(
        commands=torch.tensor([[0., 0., .30]]), _evaluation_command_override=(0., 0., .30),
        _evaluation_command_pending=False, common_step_counter=0, contract=contract,
        cfg=SimpleNamespace(stage="locomotion"), num_envs=1, device="cpu",
        _command_ticks_left=torch.tensor([300]), _command_period_ticks=300,
        _commands_due=torch.zeros(1, dtype=torch.bool), history=core.NoisyHistoryStack(1, "cpu"),
        _joint_state=lambda: (q, zero6), _base_height=lambda: torch.tensor([.30]),
        robot=SimpleNamespace(data=data), actions=zero6.clone(), previous_actions=zero6.clone(),
        torques=zero6, _finite_state=torch.ones(1, dtype=torch.bool),
        reset_terminated=torch.zeros(1, dtype=torch.bool), extras={}, _episode_sums={}, _last_reward_tick=-1,
    )
    env._sample_commands = MethodType(sample, env)
    for name in ("_get_observations", "_get_rewards"):
        setattr(env, name, MethodType(isolated_tensor_method(name)[2], env))
    return env, torch


def test_command_boundary_uses_actual_rewards_observations_and_noisy_history(transition_env):
    env, torch = transition_env
    initial = env._get_observations()["policy"].clone()
    controller = KeyboardController(BOUNDS)
    changes = []
    original_sampler = env._sample_commands
    bridge = ObservationCommandBridge(env, controller, lambda *args, **fields: changes.append(fields))
    baseline_reward = env._get_rewards().clone()
    controller.press("W")
    # A callback during rendering cannot change the action's old command or same-tick history.
    torch.testing.assert_close(env.commands, torch.tensor([[0., 0., .30]]))
    torch.testing.assert_close(env._get_observations()["policy"], initial)
    env.common_step_counter += 1
    env.actions.fill_(.2)
    env.previous_actions.fill_(.2)  # Keep action-rate reward unchanged for this comparison.
    torch.testing.assert_close(env._get_rewards(), baseline_reward)
    assert not env._commands_due.any()  # The hook must run even for empty ids.
    moving = env._get_observations()["policy"].reshape(1, 5, 25)
    torch.testing.assert_close(moving[:, :-1], initial.reshape(1, 5, 25)[:, 1:])
    torch.testing.assert_close(moving[0, -1, 6:9], torch.tensor([1., 0., 1.5]))
    assert moving[0, -1, -6:].eq(.2).all()
    torch.testing.assert_close(initial[0, -19:-16], torch.tensor([0., 0., 1.5]))
    controller.stop()
    torch.testing.assert_close(env._get_observations()["policy"], moving.reshape(1, 125))
    assert env.commands[0, 0] == 1
    env.common_step_counter += 1
    # Reward still sees W even though stop is queued; only the following obs sees zero.
    assert env._get_rewards().item() < baseline_reward.item()
    assert env.extras["log"]["Tracking/vx_abs_error"].item() == 1
    stopped = env._get_observations()["policy"].reshape(1, 5, 25)
    torch.testing.assert_close(stopped[:, :-1], moving[:, 1:])
    assert stopped[0, -1, 6] == 0
    assert len(changes) == 2
    bridge.close()
    assert env._sample_commands == original_sampler


def test_auto_reset_sampler_resumes_manual_override_without_extra_history_reset(transition_env):
    env, torch = transition_env
    env._get_observations()
    controller = KeyboardController(BOUNDS)
    bridge = ObservationCommandBridge(env, controller, lambda *args, **kwargs: None)
    controller.press("A")
    env.common_step_counter = 1
    env._get_rewards()
    # Model the documented auto-reset boundary with real history reset + real sampler.
    ids = torch.tensor([0])
    env.history.reset(ids)
    env.commands.zero_()
    env._sample_commands(ids)
    reset_obs = env._get_observations()["policy"].reshape(1, 5, 25)
    assert reset_obs[0, :, 7].eq(1).all()
    assert env._evaluation_command_override == (0., 1., .30)
    controller.stop()
    env._sample_commands(ids)  # Same tick cannot mismatch the cached observation.
    assert env.commands[0, 1] == 1
    env.common_step_counter = 2
    env._get_observations()
    assert env.commands[0, 1] == 0
    bridge.close()


def test_ood_adjustments_keep_real_reward_history_and_reset_alignment(transition_env):
    env, torch = transition_env
    initial = env._get_observations()["policy"].clone()
    baseline = env._get_rewards().clone()
    controller = KeyboardController(BOUNDS, exploratory=True, max_linear_speed=5, max_angular_rpm=120)
    bridge = ObservationCommandBridge(env, controller, lambda *args, **kwargs: None)
    controller.press("W")
    controller.press("A")
    for _ in range(20):
        for key in ("EQUAL", "RIGHT_BRACKET"):
            controller.press(key)
            controller.release(key)
    torch.testing.assert_close(env._get_observations()["policy"], initial)
    env.common_step_counter = 1
    torch.testing.assert_close(env._get_rewards(), baseline)
    moving = env._get_observations()["policy"].reshape(1, 5, 25)
    torch.testing.assert_close(moving[:, :-1], initial.reshape(1, 5, 25)[:, 1:])
    expected = torch.tensor([5., 4 * math.pi, 1.5])
    torch.testing.assert_close(moving[0, -1, 6:9], expected, atol=0, rtol=0)
    # This float32 rounding exceeded the former absolute 1e-7 alignment tolerance.
    assert abs(float(expected[1]) - 4 * math.pi) > 1e-7
    env.common_step_counter = 2
    assert env._get_rewards().item() < baseline.item()
    env.history.reset(torch.tensor([0]))
    env.commands.zero_()
    env._sample_commands(torch.tensor([0]))
    reset = env._get_observations()["policy"].reshape(1, 5, 25)
    torch.testing.assert_close(reset[0, :, 6:9], expected.repeat(5, 1), atol=0, rtol=0)
    controller.stop()
    torch.testing.assert_close(env._get_observations()["policy"], reset.reshape(1, 125))
    assert env._evaluation_command_override == (5., 4 * math.pi, .3)
    env.common_step_counter = 3
    env._get_rewards()
    assert env.extras["log"]["Tracking/vx_abs_error"] == 5
    stopped = env._get_observations()["policy"].reshape(1, 5, 25)
    torch.testing.assert_close(stopped[:, :-1], reset[:, 1:])
    assert stopped[0, -1, 6:8].eq(0).all()
    bridge.close()


def test_entrypoint_ood_frame_counts_and_float32_alignment_cpu(
        transition_env, artifact, tmp_path, monkeypatch):
    """Run real ORT/core/CSV code with a CPU clock adapter, not simulated physics."""
    import play_v40_onnx as replay

    env, torch = transition_env
    model, expected, _ = artifact
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(env.contract))
    original_contract = contract_path.read_bytes()
    env.contract_sha256 = expected["contract_sha256"]
    env.asset_manifest_sha256 = expected["asset_manifest_sha256"]
    env.step_dt = .01
    env.set_evaluation_command = lambda command: None  # Fixture already initialized to this command.
    env.reset = lambda: (env._get_observations(), {})
    env.close = lambda: None
    env.sim = SimpleNamespace(has_gui=True)
    env.render_enabled = True
    env.viewport_camera_controller = SimpleNamespace(update_view_to_asset_root=lambda name: None)
    native = SimpleNamespace(closed=False, keyboard=None,
                             input=SimpleNamespace(get_keyboard_name=lambda key: "CPU-test-device"),
                             check_focus=lambda: None, update_status=lambda: None)

    def keyboard(controller, record):
        native.controller = controller
        return native

    native.close = lambda: setattr(native, "closed", True)

    def step(actions):
        env.common_step_counter += 1
        env.previous_actions = env.actions.clone()
        env.actions = actions
        reward_command = env.commands.clone()
        reward = env._get_rewards()
        if env.common_step_counter == 1:
            native.controller.press("W")
            native.controller.press("A")
            for _ in range(20):
                for key in ("EQUAL", "RIGHT_BRACKET"):
                    native.controller.press(key)
                    native.controller.release(key)
        elif env.common_step_counter == 3:
            native.controller.press("SPACE")
        env.snapshot = dict(
            sample_kind="pre_reset", diagnostic_flags_version=1,
            diagnostic_flags={key: torch.tensor([False]) for key in DIAGNOSTIC_FLAGS},
            termination_flags={key: torch.tensor([False]) for key in TERMINATION_FLAGS},
            terminated=torch.tensor([False]), timeout=torch.tensor([False]),
            episode_step=torch.tensor([env.common_step_counter]),
            episode_time_s=torch.tensor([env.common_step_counter * .01]),
            sustained_failure_ticks=torch.tensor([0]),
            command=reward_command, root_link_pos_w_m=torch.tensor([[0., 0., .3]]),
            root_com_lin_vel_b_m_s=torch.zeros(1, 3), root_com_ang_vel_b_rad_s=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]), non_wheel_net_force_max_n=torch.zeros(1),
            base_visual_clearance_lower_bound_m=torch.tensor([.1]),
        )
        return env._get_observations(), reward, torch.tensor([False]), torch.tensor([False]), {}

    env.step = step
    env.get_evaluation_snapshot = lambda: env.snapshot
    monkeypatch.setattr(replay, "NativeKeyboard", keyboard)
    monkeypatch.setattr(replay, "preflight", lambda args: ({"blockers": []}, env.contract, {}))
    monkeypatch.setattr(replay, "make_manifest", lambda *args: expected)
    monkeypatch.setattr(replay, "make_env", lambda args: env)
    monkeypatch.setattr(replay, "launch_app", lambda args: SimpleNamespace(
        app=SimpleNamespace(is_running=lambda: True, close=lambda **kwargs: None)))
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", SimpleNamespace(
        get_active_viewport=lambda: SimpleNamespace(resolution=(1280, 720), camera_path="CPU-test-camera"),
        capture_viewport_to_file=lambda *args, **kwargs: None))
    report = tmp_path / "replay"
    assert replay.main([
        "--keyboard", "--exploratory", "--max-linear-speed", "5", "--max-angular-rpm", "120",
        "--onnx", str(model), "--contract", str(contract_path), "--report-dir", str(report),
        "--max-steps", "4",
    ]) == 0
    summary = json.loads((report / "summary.json").read_text())
    assert summary["exploratory"] is True
    assert summary["out_of_training_domain_command_frames"] == 2
    assert summary["policy_steps"] == 4 and summary["reset_events"] == 0
    assert summary["keyboard_subscriptions_closed"]
    with (report / "command_changes.csv").open() as stream:
        changes = list(csv.DictReader(stream))
    assert [row["out_of_training_domain_command_frames"] for row in changes] == ["0", "0", "2", "2"]
    assert [row["out_of_training_domain"] for row in changes] == ["False", "True", "False", "False"]
    with (report / "telemetry.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["action_command_ood"] for row in rows] == ["False", "True", "True", "False"]
    assert contract_path.read_bytes() == original_contract


@pytest.mark.parametrize("height", [.28, .30, .32])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_headless_fixed_height_records_diagnostics_independently_of_termination(
        transition_env, artifact, tmp_path, monkeypatch, capsys, height, nonfinite):
    """Actual ORT + replay reporting, with CPU transition data and no GUI/physics."""
    import play_v40_onnx as replay

    env, torch = transition_env
    model, expected, _ = artifact
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(env.contract))
    env.contract_sha256 = expected["contract_sha256"]
    env.asset_manifest_sha256 = expected["asset_manifest_sha256"]
    env.step_dt = .01

    def set_command(command):
        env._evaluation_command_override = command
        env.commands[:] = torch.tensor(command)

    env.set_evaluation_command = set_command
    env.reset = lambda: (env._get_observations(), {})
    env.close = lambda: None

    def step(actions):
        env.common_step_counter += 1
        tick = env.common_step_counter
        env.actions = actions
        reward = env._get_rewards()
        terminal = nonfinite and tick == 2
        timeout = not nonfinite and tick == 2
        flags = {key: torch.tensor([False]) for key in DIAGNOSTIC_FLAGS}
        flags["non_wheel_contact"][:] = tick == 1
        flags["nonfinite"][:] = terminal
        reasons = {key: torch.tensor([False]) for key in TERMINATION_FLAGS}
        reasons["nonfinite"][:] = terminal
        env.snapshot = dict(
            sample_kind="pre_reset", diagnostic_flags_version=1, diagnostic_flags=flags,
            termination_flags=reasons, terminated=torch.tensor([terminal]), timeout=torch.tensor([timeout]),
            episode_step=torch.tensor([tick]), episode_time_s=torch.tensor([tick * .01]),
            sustained_failure_ticks=torch.tensor([0]), command=env.commands.clone(),
            root_link_pos_w_m=torch.tensor([[0., 0., float("nan") if terminal else height]]),
            root_com_lin_vel_b_m_s=torch.zeros(1, 3), root_com_ang_vel_b_rad_s=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]),
            non_wheel_net_force_max_n=torch.tensor([16. if tick == 1 else 0.]),
            base_visual_clearance_lower_bound_m=torch.tensor([.1]),
        )
        if timeout or terminal:
            env.history.reset(torch.tensor([0]))
            env._sample_commands(torch.tensor([0]))
        return env._get_observations(), reward, torch.tensor([terminal]), torch.tensor([timeout]), {}

    env.step = step
    env.get_evaluation_snapshot = lambda: env.snapshot
    monkeypatch.setattr(replay, "preflight", lambda args: ({"blockers": []}, env.contract, {}))
    monkeypatch.setattr(replay, "make_manifest", lambda *args: expected)
    monkeypatch.setattr(replay, "make_env", lambda args: env)

    def launch(args):
        assert args.headless and not args.keyboard
        return SimpleNamespace(app=SimpleNamespace(is_running=lambda: True, close=lambda **kw: None))

    monkeypatch.setattr(replay, "launch_app", launch)
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", None)
    monkeypatch.setattr(replay, "NativeKeyboard", lambda *a: pytest.fail("headless keyboard subscription"))
    monkeypatch.setattr(replay.time, "sleep", lambda *a: pytest.fail("headless must not GUI-pace"))
    report = tmp_path / "headless"
    assert replay.main([
        "--headless", "--onnx", str(model), "--contract", str(contract_path),
        "--report-dir", str(report), "--command", "0", "0", str(height), "--max-steps", "3",
    ]) == int(nonfinite)
    output = capsys.readouterr().out
    assert "SERVER_FINAL_ONNX_HEADLESS_READY" in output
    assert "SERVER_FINAL_ONNX_GUI_READY" not in output
    summary_text = (report / "summary.json").read_text()
    summary = json.loads(summary_text)
    assert "NaN" not in summary_text
    assert summary["screenshot_status"] == "disabled_headless"
    assert summary["diagnostic_frames"]["non_wheel_contact"] == 1
    assert summary["termination_flags"]["non_wheel_contact"] == 0
    assert summary["termination_resets"] == int(nonfinite)
    assert summary["timeout_resets"] == int(not nonfinite)
    assert summary["diagnostic_frames"]["nonfinite"] == int(nonfinite)
    assert summary["termination_flags"]["nonfinite"] == int(nonfinite)
    assert summary["all_numeric_finite"] is not nonfinite
    assert summary["status"] == ("failed" if nonfinite else "completed")
    assert summary["contact_semantics"].endswith("not_ground_pair")
    with (report / "telemetry.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary["diagnostic_frame_count"] == (2 if nonfinite else 3)
    assert rows[0]["diagnostic_non_wheel_contact"] == "True"
    assert rows[0]["termination_non_wheel_contact"] == rows[0]["terminated"] == "False"
    assert rows[0]["non_wheel_net_force_max_n"] == rows[0]["non_wheel_contact_n"] == "16.0"
    assert rows[1]["diagnostic_nonfinite"] == str(nonfinite)
    assert rows[1]["episode_step"] == "2"  # Pre-reset frame survives reset bookkeeping.
    assert all(float(row["action_cmd_height"]) == height for row in rows)
    assert all(float(row["reward_cmd_height"]) == pytest.approx(height) for row in rows)
    assert all(row["sample_kind"] == "pre_reset" for row in rows)
    assert not list(report.glob("*.png"))
