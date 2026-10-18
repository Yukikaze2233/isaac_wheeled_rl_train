"""Shared CPU fixtures for current runtime tests."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from support_flat_episode_env_wiring import method, execute, bind_finish
from test_scut_flat_reward_runtime import env as reward_env, field
from wheeled_tasks.chassis.flat_episode_runtime import build_flat_episode_runtime
from wheeled_tasks.chassis.recovery_training import RecoveryTraining
from wheeled_tasks.chassis.recovery import RecoveryPhase
from wheeled_tasks.chassis.v5_control import V5Control
ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = 'reports/v7_formal_training_20261012/recovery107_candidate/'

def config():
    c = json.loads((ROOT / 'configs/v7_formal_20261012/flat_tgs_continuation_055.json').read_bytes())
    # Isolate controller ownership from optional training-course selectors.
    for key in ('formal_flat_refresh', 'formal_dynamic_nonsteady_birth',
                'formal_flat_rehearsal', 'formal_physics_trace_directory',
                'physx_solver_type', 'gas_pressure_application'):
        c.pop(key, None)
    c['formal_foundation_nominal_birth'] = True
    c['recovery_training'] = dict(enabled=True, reference_dt=0.005, blend_seconds=0.2, stable_seconds=1.0, max_script_seconds=8.0, reroute_tilt_rad=1.2217304763960306, reroute_support_lost_seconds=0.15, max_reroutes=1, learning_mask='pure_rl_intervals_only', reset_mode='initial_state_only', controller='encoder_imu_recovery_fsm', action_history='effective_reference', source_flat_handoff='scut0719_scripted_handoff_v1', wheel_balance=dict(enabled=True, pitch_kp_nm_rad=8.0, pitch_kd_nm_s_rad=1.5, wheel_damping_nm_s_rad=0.2, prepare_tilt_limit_deg=55.0, capture_tilt_limit_deg=85.0))
    for prefix, file in (('profiles', 'recovery_profiles_107_v2.json'), ('height_lookup', 'height_lookup_107.json')):
        path = CANDIDATE + file
        c['recovery_training'].update({prefix + '_file': path, prefix + '_sha256': hashlib.sha256((ROOT / path).read_bytes()).hexdigest()})
    c['scene_groups'] = [dict(name='recover', terrain=['flat'], fraction=0.5, recovery_pose='left_side_90'), dict(name='ordinary', terrain=['flat'], fraction=0.5)]
    return c

def production_reset(e, ids):
    reset = method('reset')
    end = next((i for i, node in enumerate(reset.body) if 'self.robot.write_joint_velocity_to_sim_index' in ast.unparse(node)))
    scope = execute(reset.body[:end + 1], e, ids=ids)
    start = next((i for i, node in enumerate(reset.body) if isinstance(node, ast.If) and ast.unparse(node.test) == 'flat is not None' and ('flat.commit_reset' in ast.unparse(node))))
    scope = {key: value for key, value in scope.items() if key not in ('self', 'torch', '__package__', '__builtins__')}
    execute(reset.body[start:], e, **scope)

@pytest.fixture
def rig():
    torch.set_num_threads(4)
    c = config()
    e = reward_env(2)
    e.round_trip_transport = None
    e.cfg = {**c, 'record_diagnostics': True}
    asset = ROOT / c['asset_directory']
    e.manifest = json.loads((asset / 'manifest.json').read_bytes())
    spec = json.loads((asset / 'model_spec.json').read_bytes())
    fit = json.loads((asset / 'fit_10mpa.json').read_bytes())
    control = json.loads((ROOT / c['control_math_source']).read_bytes())
    names = e.manifest['tree_joint_names']
    e.robot.joint_names = names
    e.nominal = torch.tensor([[e.manifest['nominal_joint_pos'][name] for name in names]]).repeat(2, 1)
    e.ids = [names.index(name) for name in e.manifest['control_joint_names']]
    for name in ('joint_pos', 'joint_vel', 'joint_acc', 'applied_torque'):
        setattr(e.robot.data, name, field(e.nominal.clone() if name == 'joint_pos' else torch.zeros_like(e.nominal)))
    e.robot.data.root_link_lin_vel_w = field(torch.zeros(2, 3))
    e.v5 = V5Control(e.manifest, spec, fit, control, c['v5_control'], 'cpu')
    e.device, e.dt, e.policy_dt, e.tick = ('cpu', 0.005, 0.02, 0)
    e.physical_tick = 0
    e.critic78, e.skills, e.full_tasks = (False, None, None)
    e.knee_ids = [names.index(name) for name in ('L_joint2', 'R_jonit2')]
    e.knee_bounds = {name: (-10.0, 10.0) for name in ('L_joint2', 'R_jonit2')}
    e.spring_ids = [names.index(name) for name in e.manifest['spring_joint_names']]
    e.nonwheel_ids = [0]
    e.scene_groups, e.kinds = (['recover', 'ordinary'], ['flat', 'flat'])
    e.raw_height = torch.zeros(2)
    e.command_target, e.command_acceleration = (torch.zeros(2, 2), torch.zeros(2))
    e.actions, e.mode, e.targets = (torch.zeros(2, 6), torch.zeros(2, dtype=torch.long), torch.zeros(2, 4))
    e.jump_requested = torch.zeros(2, dtype=torch.bool)
    e.generator = torch.Generator().manual_seed(9843)
    e.reset_positions = e.origins.clone()
    e.ground_height = lambda xy: torch.zeros(len(xy))
    e.update_targets = lambda: None
    e.recovery = RecoveryTraining(e, ROOT, c['recovery_training'])
    e.flat_episode = build_flat_episode_runtime(c, e.manifest, spec, ROOT, 2, 'cpu')
    e.flat_tree_to_native = [e.flat_episode.tree_joint_names.index(name) for name in names]
    e.writes = []

    def write(name, **kw):
        e.writes.append((name, deepcopy(kw)))
        ids = kw['env_ids']
        if name == 'root':
            e.robot.data.root_link_pose_w.torch[ids] = kw['root_pose']
        if name == 'velocity':
            e.robot.data.root_com_lin_vel_b.torch[ids] = kw['root_velocity'][:, :3]
        if name == 'q':
            e.robot.data.joint_pos.torch[ids] = kw['position']
        if name == 'dq':
            e.robot.data.joint_vel.torch[ids] = kw['velocity']
    e.robot.write_root_link_pose_to_sim_index = lambda **kw: write('root', **kw)
    e.robot.write_root_com_velocity_to_sim_index = lambda **kw: write('velocity', **kw)
    e.robot.write_joint_position_to_sim_index = lambda **kw: write('q', **kw)
    e.robot.write_joint_velocity_to_sim_index = lambda **kw: write('dq', **kw)
    bind_finish(e)
    e.reset = lambda ids: production_reset(e, ids)
    e.reset(torch.tensor([1, 0]))
    return e

def tick(e):
    e.flat_episode.begin_policy()
    e.recovery.begin_policy()
    for _ in range(4):
        e.flat_episode.advance_physics(0.005)
    e.tick += 1
    e.episode_length_buf += 1
    return e.finish()

def ready(e):
    e.robot.data.root_link_pose_w.torch[:] = torch.tensor([0.0, 0.0, 0.305, 0.0, 0.0, 0.0, 1.0])
    e.contact_force[:, 1:, 2] = 10.0
    e.recovery.release_finished[:] = True
    e.recovery.fsm.phase[0] = RecoveryPhase.RL
    e.recovery.fsm.enabled[0] = True
    e.recovery.fsm.motion_released[0] = False
