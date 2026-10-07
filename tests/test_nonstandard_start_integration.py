"""Execute production environment hooks with CPU physics/transport doubles."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.nonstandard_start import NonstandardStart, nonstandard_start_settings

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/wheeled_tasks/chassis/env.py"
BANK = json.loads((ROOT / "contracts/v6_nonstandard_start_profiles_v1.json").read_text())
JOINTS = BANK["tree_joint_names"]


def method(name):
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def mentions(node, expression):
    return expression in ast.unparse(node)


def execute(nodes, env, **values):
    scope = {"self": env, "torch": torch, "__package__": "wheeled_tasks.chassis", **values}
    module = ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope


def birth(count=2):
    settings = nonstandard_start_settings(100)
    settings["iteration_probability_schedule"] = [{"actor_update": 0, "probabilities": [0., 1., 0.]}]
    return NonstandardStart(count, "cpu", .005, settings, ROOT, JOINTS)


def init_hook(env, *, enabled=True, slew=True):
    settings = nonstandard_start_settings(100, enabled=enabled)
    cfg = {"nonstandard_start": settings, "asset_manifest_sha256": settings["asset_manifest_sha256"]}
    if slew:
        cfg["command_slew"] = {"vx_m_s2": 1.5, "yaw_rad_s2": 4.}
    block = next(node for node in method("__init__").body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "'nonstandard_start' in config")
    return execute([block], env, config=cfg, num_envs=len(env.kinds), device="cpu", root=ROOT)


def test_native_eligibility_only_ordinary_groups_excludes_world_spin_and_scripted_tasks():
    count = 7
    env = NS(kinds=["flat", "jump", "flat", "step_up", "stairs", "flat", "rough"],
        dt=.005, policy_dt=.02, robot=NS(joint_names=JOINTS), startup_report={},
        skills=NS(spin=torch.tensor([0, 0, 0, 0, 0, 1, 0], dtype=torch.bool)),
        recovery=NS(enabled=torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype=torch.bool)),
        step_jump=NS(enabled=torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.bool)),
        step_climb=NS(enabled=torch.tensor([0, 0, 0, 0, 1, 0, 0], dtype=torch.bool)))
    init_hook(env)
    assert env.nonstandard_start_eligible.tolist() == [True, False, False, False, False, False, True]
    env.nonstandard_start.settings["iteration_probability_schedule"] = [
        {"actor_update": 0, "probabilities": [0., 1., 0.]}]
    env.nonstandard_start.reset(torch.arange(count), 100, eligible=env.nonstandard_start_eligible)
    assert env.nonstandard_start.selected.tolist() == [True, False, False, False, False, False, True]


def test_native_constructor_requires_slew_only_when_training_feature_is_enabled():
    env = NS(kinds=["flat"], dt=.005, policy_dt=.02, robot=NS(joint_names=JOINTS),
             startup_report={}, skills=None, recovery=None, step_jump=None)
    with pytest.raises(ValueError, match="requires the existing command slew"):
        init_hook(env, slew=False)
    init_hook(env, enabled=False, slew=False)
    assert not env.nonstandard_start.enabled


def test_native_reset_places_full_pose_above_internal_rough_strip_and_writes_once():
    module = birth(3)
    writes = []
    robot = NS(
        write_root_link_pose_to_sim_index=lambda **kwargs: writes.append(("root", kwargs)),
        write_root_com_velocity_to_sim_index=lambda **kwargs: writes.append(("velocity", kwargs)),
        write_joint_position_to_sim_index=lambda **kwargs: writes.append(("q", kwargs)),
        write_joint_velocity_to_sim_index=lambda **kwargs: writes.append(("dq", kwargs)))
    env = NS(device="cpu", cfg={"task_campaign": {"domain": "speed"}}, task_actor_update=100,
        reset_positions=torch.tensor([[10., 20., .32], [30., 40., .82], [50., 60., .32]]),
        origins=torch.tensor([[10., 20., 0.], [30., 40., .5], [50., 60., 0.]]),
        surface_data=torch.tensor([[[-2., 2., 0., 0., 0.], [-.02, .02, .2, 0., 0.]]]).repeat(3, 1, 1),
        surface_valid=torch.ones(3, 2, dtype=torch.bool), generator=torch.Generator().manual_seed(44),
        skills=None, step_jump=None, recovery=None, nonstandard_start=module,
        nonstandard_start_eligible=torch.tensor([False, True, False]),
        nominal=torch.full((3, len(JOINTS)), 9.), joint_count=len(JOINTS), robot=robot)
    reset = method("reset")
    end = next(i for i, node in enumerate(reset.body)
               if mentions(node, "self.robot.write_joint_velocity_to_sim_index"))
    execute(reset.body[:end + 1], env, ids=torch.tensor([2, 1]), stage_actor_updates=lambda self: self.task_actor_update)
    assert [name for name, _ in writes] == ["root", "velocity", "q", "dq"]
    root = writes[0][1]["root_pose"]
    q = writes[2][1]["position"]
    assert root[0, 2] == pytest.approx(.32)
    assert q[0].eq(9.).all()
    row = module.profile_ids[1]
    assert root[1, 2] == pytest.approx(.5 + .2 + module.root_heights[row].item())
    torch.testing.assert_close(q[1], module.joint_positions[row])
    assert writes[1][1]["root_velocity"].eq(0.).all()
    assert writes[3][1]["velocity"].eq(0.).all()


def physics_hook(env):
    step = method("step")
    loop = next(node for node in step.body if isinstance(node, ast.For)
                and ast.unparse(node.target) == "substep")
    end = next(i for i, node in enumerate(loop.body) if mentions(node, "self.nonstandard_start.advance()"))
    selected = deepcopy(loop)
    selected.body = selected.body[:end + 1]
    history = next(node for node in step.body if isinstance(node, ast.If)
                   and mentions(node, "self.nonstandard_start.effective_action_history"))
    env.nonstandard_start.begin_policy()
    execute([selected, history], env, legs=torch.zeros(2, 4), wheels=torch.zeros(2, 2),
        update_robot_data=lambda native, dt: native.robot.update(dt))


def physics_env():
    module = birth()
    module.reset(torch.tensor([0]), 100)
    ids = [JOINTS.index(name) for name in BANK["control_joint_names"]]
    spring_ids = [JOINTS.index(name) for name in ("L_spring_slide", "R_spring_slide")]
    data = NS(joint_pos=NS(torch=torch.zeros(2, len(JOINTS))),
              joint_vel=NS(torch=torch.zeros(2, len(JOINTS))))
    writes, transports = [], []
    response = NS(enabled=True, apply_torque=lambda tau: tau + 7.)
    def transport(tau):
        transports.append(tau.clone())
        return tau + 2.
    robot = NS(data=data, set_joint_effort_target_index=lambda **kwargs: writes.append(kwargs["target"].clone()),
               write_data_to_sim=lambda: None, update=lambda dt: None)
    env = NS(nonstandard_start=module, robot=robot, ids=ids, spring_ids=spring_ids,
        decimation=4, dt=.005, signal_delay=None, usb_transport=None, recovery=None,
        is_v5=True, v5=NS(WHEELS=(2, 5), motor_efforts=lambda *args: torch.full((2, 6), 10.),
        spring_efforts=lambda q: torch.full_like(q, 27.), current_motor_bounds=torch.full((2, 6), 40.)),
        motor_strength=torch.full((2, 6), .8), spring_strength=torch.full((2, 1), .9), skills=None,
        command_transport=NS(apply_torque=transport), actuator_response=response,
        nominal=torch.zeros(2, len(JOINTS)), actions=torch.full((2, 6), 3.), jump_assist=None,
        sim=NS(step=lambda **kwargs: None), control_audit=NS(feedback_control=lambda: None))
    return env, writes, transports


def test_native_final_zero_gate_removes_delayed_filter_output_and_preserves_passive_springs():
    env, writes, transports = physics_env()
    for _ in range(10):
        physics_hook(env)
        assert not env.nonstandard_start.learning_mask[0] and env.nonstandard_start.learning_mask[1]
        assert env.actions[0].eq(0.).all()
    assert len(writes) == len(transports) == 40
    for effort, command in zip(writes, transports):
        assert effort[0, env.ids].eq(0.).all()  # Includes both powered wheels.
        assert command[0].eq(0.).all()  # Filter input history is also zero.
        assert effort[1, env.ids].eq(17.).all()
        torch.testing.assert_close(effort[:, env.spring_ids], torch.full((2, 2), 24.3))
    assert not env.nonstandard_start.holding.any()
    env.actions.fill_(2.)
    physics_hook(env)
    assert env.nonstandard_start.learning_mask.all()
    assert writes[-1][0, env.ids].eq(17.).all()
    assert env.actions.eq(2.).all()


def command_hook(env):
    node = next(node for node in method("step").body if isinstance(node, ast.If)
                and ast.unparse(node.test) == "self.cfg.get('command_slew')")
    execute([node], env)


def test_native_capture_target_copy_expires_and_existing_slew_restores_original_goal():
    module = birth()
    module.reset(torch.tensor([0]), 100)
    env = NS(nonstandard_start=module, commands=torch.tensor([[1., -1., .43], [2., 0., .23]]),
        command_target=torch.tensor([[5., -8.], [2., 0.]]), references=None, step_jump=None,
        skills=NS(spin=torch.zeros(2, dtype=torch.bool), update=lambda: None), policy_dt=.02,
        cfg={"command_slew": {"vx_m_s2": 1.5, "yaw_rad_s2": 4.}})
    original = env.command_target.clone()
    command_hook(env)
    torch.testing.assert_close(env.commands[0], torch.tensor([1., -1., .43]))
    assert torch.equal(original, env.command_target)
    module.episode_steps[0] = module.capture_steps
    command_hook(env)
    torch.testing.assert_close(env.commands[0], torch.tensor([1.03, -1.08, .43]))
    assert torch.equal(original, env.command_target)


def test_native_learning_mask_survives_autoreset_and_bootstrap_uses_new_hold():
    module = birth()
    module.reset(torch.tensor([0]), 100)
    module.remaining_steps[0] = 1
    module.begin_policy()
    module.apply_torque(torch.ones(2, 6))
    module.advance()
    env = NS(nonstandard_start=module)
    step = method("step")
    learning = next(node for node in step.body if isinstance(node, ast.If)
                    and mentions(node, "policy_learning_mask") and mentions(node, "nonstandard_start.metrics"))
    bootstrap = next(node for node in step.body if isinstance(node, ast.If)
                     and mentions(node, "policy_bootstrap_mask") and mentions(node, "nonstandard_start.bootstrap_mask"))
    extras = {"log": {}, "policy_learning_mask": torch.tensor([True, False])}
    execute([learning], env, extras=extras, stage_actor_updates=lambda self: 100)
    assert extras["policy_learning_mask"].tolist() == [False, False]
    # The previously pure-RL lane resets into a new passive birth interval.
    module.reset(torch.tensor([1]), 100)
    execute([bootstrap], env, extras=extras, terminated=torch.zeros(2, dtype=torch.bool))
    assert extras["policy_learning_mask"].tolist() == [False, False]
    assert extras["policy_bootstrap_mask"].tolist() == [True, False]


def test_train_runner_attaches_existing_native_mask_for_ordinary_birth_feature():
    from test_recovery_rollout_mask import algorithm
    tree = ast.parse((ROOT / "scripts/train_chassis.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "configure_recovery_rollout")
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "runner_mask_hook", "exec"), scope)
    alg = algorithm()
    adapter = scope["configure_recovery_rollout"](alg, NS(num_envs=2, cfg={"nonstandard_start": {"enabled": True}}))
    assert adapter is alg._recovery_rollout_mask
