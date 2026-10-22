"""Exercise the actual 40/41 drive boundary without constructing Isaac Sim."""
import ast
from types import MethodType, SimpleNamespace as NS

import torch

from wheeled_tasks.chassis.application.runtime.flat_episode_runtime import FlatEpisodeRuntime
from tests.support.support_nonstandard_start_integration import execute, method, physics_env, JOINTS


def test_jump_trace_binds_real_relocated_source_paths_and_hashes():
    import hashlib
    from wheeled_tasks.chassis.source_layout import module_source_path

    assignment = next(
        node for node in method('_initialize_jump_physics_trace').body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'sources' for target in node.targets)
    )
    scope = {'hashlib': hashlib, 'module_source_path': module_source_path}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<trace-sources>', 'exec'), scope)
    expected = {
        str(module_source_path(name)): hashlib.sha256(module_source_path(name).read_bytes()).hexdigest()
        for name in ('env.py', 'jump_physics_trace.py', 'jump_actuation_evidence.py',
                     'root_kinematics.py', 'training_schedule.py', 'skill_commands.py')
    }
    assert scope['sources'] == expected


def owner(count, ground):
    flat = FlatEpisodeRuntime(count, "cpu", None, None, JOINTS, birth_rejection={"mode": "abort"})
    generator = torch.Generator().manual_seed(421)
    flat.commands.reset(torch.arange(count), torch.zeros(count), ground, generator)
    flat.episode_id[:] = 7
    return flat, generator


def test_trace_retains_last_held_tick_before_owner_advances():
    env, writes, _ = physics_env()
    flat, generator = owner(2, torch.tensor([True, False]))
    env.flat_episode, env.nonstandard_start = flat, None
    env._pc_feedback_transport = env.scut_reward = None
    env.critic78 = False
    env.physical_tick = 0
    env.feedback_transport = env.torque_monitor = env.references = env.wheel_force_peak_n = None
    env.contact_views = []
    env.wheel_ids = [0, 1]
    env.contact_force = torch.zeros(2, 2, 3)
    env.contact_peak = torch.zeros(2)
    env.control_audit.physics_updated = lambda dt: None
    env.cfg = {"formal_physics_trace_substeps": [39, 40, 41]}
    scope = execute([method("_formal_physics_control_frame")], env)
    env._formal_physics_control_frame = MethodType(scope["_formal_physics_control_frame"], env)
    saved = []
    env._save_formal_physics_state = lambda name, control_frame: saved.append(
        (env.physical_tick, control_frame, flat.holding.clone()))
    loop = next(node for node in method("step").body if isinstance(node, ast.For)
                and ast.unparse(node.target) == "substep")
    for _ in range(11):
        flat.begin_policy()
        execute([loop], env, flat=flat, legs=torch.zeros(2, 4), wheels=torch.zeros(2, 2),
                update_robot_data=lambda env, dt: None)
        flat.advance(.02, torch.zeros(2), 0, generator)
    assert [tick for tick, _, _ in saved] == [39, 40, 41]
    assert [frame["holding_before"][0].item() for _, frame, _ in saved] == [True, True, False]
    assert [after[0].item() for _, _, after in saved] == [True, False, False]
    assert [frame["episode_physics_ticks_before"][0].item() for _, frame, _ in saved] == [38, 39, 40]
    for tick, frame, _ in saved:
        torch.testing.assert_close(frame["sent_effort"], writes[tick - 1])
        assert frame["episode_id_before"].eq(7).all()
    assert saved[1][1]["sent_effort"][0, env.ids].eq(0).all()
    assert saved[2][1]["sent_effort"][0, env.ids].eq(17).all()
    # Later target mutation must not retroactively alter the observed drive.
    env.torque.fill_(99.)
    assert saved[2][1]["sent_effort"][0, env.ids].eq(17).all()


def test_snapshot_saves_native_body_velocities_and_episode_identity(tmp_path):
    from pathlib import Path
    flat, _ = owner(1, torch.tensor([True]))
    tensor = lambda *shape: NS(torch=torch.zeros(*shape))
    data = NS(joint_pos=tensor(1, 18), joint_vel=tensor(1, 18), joint_acc=tensor(1, 18),
        applied_torque=tensor(1, 18), root_link_pose_w=tensor(1, 7),
        body_link_pose_w=tensor(1, 2, 7), body_com_pose_w=tensor(1, 2, 7),
        body_com_lin_vel_w=tensor(1, 2, 3), body_com_ang_vel_w=tensor(1, 2, 3))
    data.body_com_lin_vel_w.torch[0, 1, 0] = 3.
    data.body_com_ang_vel_w.torch[0, 1, 2] = 2.
    env = NS(robot=NS(data=data, joint_names=[str(i) for i in range(18)], body_names=["a", "b"]),
        cfg={"formal_physics_trace_directory": str(tmp_path), "formal_physics_trace_substeps": [40]},
        tick=9, physical_tick=40, flat_episode=flat, origins=torch.zeros(1, 3),
        closure_gap=lambda: torch.tensor([.006434]), closure_gap_peak=torch.tensor([.007]),
        critic_net_contact_force=torch.zeros(1, 2, 3), critic_applied_torque=torch.zeros(1, 6),
        _closure_body_ids=torch.tensor([[0, 1]]), _closure_points=torch.zeros(1, 2, 3), ids=list(range(6)))
    scope = execute([method("_save_formal_physics_state")], env, Path=Path)
    scope["_save_formal_physics_state"](env, "tick40", control_frame={"holding_before": torch.tensor([True])})
    saved = torch.load(tmp_path / "tick40.pt", weights_only=False)
    assert saved["physical_tick"] == 40 and saved["tick"] == 9
    assert saved["episode_id"].tolist() == [7]
    assert saved["body_com_linear_velocity"][0, 1, 0] == 3.
    assert saved["body_com_angular_velocity"][0, 1, 2] == 2.
    assert saved["control_frame"]["holding_before"].item()
    assert saved["closure_body_ids"].tolist() == [[0, 1]]
