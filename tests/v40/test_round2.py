"""Round-two CPU contract/tensor/serialization regressions; no Isaac claims."""
from __future__ import annotations

import ast
from collections.abc import Sequence
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.v40 import core
from wheeled_tasks.v40.contract import DEFAULT_CONTRACT, audit_asset, is_round2, validate_contract
from wheeled_algo import v40_export, v40_job
from test_env_contract_static import load_cli
from test_export import make_checkpoint

ROOT = Path(__file__).resolve().parents[2]
V2_PATH = ROOT / 'contracts/own_v40_v2.json'


@pytest.fixture
def v2():
    return core.load_contract(V2_PATH)


def test_v1_default_and_shared_physical_identity(v2):
    v1 = core.load_contract()
    assert DEFAULT_CONTRACT.name == 'own_v40_v1.json'
    assert not is_round2(v1) and is_round2(v2)
    for key in ('asset', 'actuators', 'timing', 'policy'):
        assert v1[key] == v2[key]
    for key, value in v1['joints'].items():
        assert v2['joints'][key] == value
    assert v1['actions']['clip'] == 1 and v1['rewards']['termination_penalty'] == -5
    load_cli().check_training_baseline(v1)
    load_cli().check_training_baseline(v2)
    v1['actions'] = deepcopy(v2['actions'])
    with pytest.raises(ValueError, match='motor speed domain'):
        validate_contract(v1)
    v2['contract_id'] = 'own-v40-jointspace-h5-v3'
    with pytest.raises(ValueError, match='identity'):
        validate_contract(v2)


@pytest.mark.parametrize('bounds', [[float('nan'), 2], [2, -2]])
def test_v2_preflight_rejects_invalid_commands(v2, bounds):
    v2['commands']['stages']['locomotion']['vx'] = bounds
    with pytest.raises(ValueError):
        load_cli().check_training_baseline(v2)


def test_action_physical_limits_periodic_hips_and_linear_soft_reward(v2):
    q = torch.tensor([v2['joints']['nominal_positions']])
    actions = torch.full_like(q, 1000.)
    legs, wheels, clipped = core.decode_targets(actions, q, v2)
    assert clipped.eq(100).all() and wheels.eq(1000).all()
    for col, idx in ((1, 1), (3, 4)):
        hi = v2['joints']['knee_hard_limits'][v2['joints']['action_order'][idx]][1]
        assert legs[0, col].item() == pytest.approx(hi)
    action = torch.ones_like(q)
    legs, _, _ = core.decode_targets(action, q, v2)
    assert (legs[0, 0] - q[0, 0]).item() == pytest.approx(.5)
    v1legs, _, _ = core.decode_targets(action, q, core.load_contract())
    assert (v1legs[0, 0] - q[0, 0]).item() == pytest.approx(.15)
    wound = q.clone()
    wound[:, [0, 3]] += 20 * math.pi
    wound_legs, _, _ = core.decode_targets(action, wound, v2)
    torch.testing.assert_close(wound_legs[:, [0, 2]] - wound[:, [0, 3]],
                               legs[:, [0, 2]] - q[:, [0, 3]], atol=1e-5, rtol=1e-5)
    low_legs, _, _ = core.decode_targets(-actions, q, v2)
    for col, idx in ((1, 1), (3, 4)):
        lo = v2['joints']['knee_hard_limits'][v2['joints']['action_order'][idx]][0]
        assert low_legs[0, col].item() == pytest.approx(lo)
    velocity = torch.zeros_like(q)
    velocity[:, [2, 5]] = 1000
    torque = core.compute_torques(q, velocity, legs, wheels, v2)
    assert torque[:, [2, 5]].eq(0).all()
    assert torque[:, [0, 1, 3, 4]].abs().le(40).all()
    lo, hi = v2['joints']['knee_hard_limits']['L_joint2']
    q[:, 1] = hi
    zero3, zero6 = torch.zeros(1, 3), torch.zeros(1, 6)
    terms = core.compute_reward_terms(zero3, zero3, torch.tensor([[0., 0., -1.]]),
                                     torch.tensor([.32]), torch.tensor([[0., 0., .32]]),
                                     zero6, zero6, zero6, q, v2)
    assert terms['knee_soft_limit'].item() == pytest.approx((hi-lo)*.015 * -2 * .01, abs=1e-8)
    assert terms['height'].item() == pytest.approx(.02)
    assert terms['lateral_velocity'].item() == terms['zero_command_translation'].item() == 0


def adapter_class():
    """Compile actual env methods with a no-op base reset, not a copied implementation."""
    path = ROOT / 'src/wheeled_tasks/direct/v40_serial/env.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    names = {'_get_observations', '_get_dones', '_reset_idx', '_sample_commands',
             '_make_evaluation_snapshot', 'set_evaluation_command'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls.bases = [ast.Name(id='ResetBase', ctx=ast.Load())]

    class ResetBase:
        def _reset_idx(self, env_ids):
            self.episode_length_buf[env_ids] = 0

    namespace = dict(vars(core), ResetBase=ResetBase, Sequence=Sequence, math=math,
                     deepcopy=deepcopy, WHEEL_BODY_NAMES=['L_link3', 'R_link3'])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['V40Env']


def adapter(c):
    env = adapter_class()()
    n = 2
    q = torch.tensor(c['joints']['nominal_positions']).repeat(n, 1)
    data = SimpleNamespace(
        root_state_w=torch.zeros(n, 13), root_link_pos_w=torch.tensor([[0., 0., .32]]).repeat(n, 1),
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        root_lin_vel_b=torch.zeros(n, 3), root_ang_vel_b=torch.zeros(n, 3),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
        applied_torque=torch.zeros(n, 6), body_link_pos_w=torch.zeros(n, 2, 3),
        default_joint_pos=q.clone(), default_root_state=torch.zeros(n, 13),
    )
    data.default_root_state[:, 2] = .32
    data.default_root_state[:, 3] = 1
    env.__dict__.update(
        contract=c, num_envs=n, device='cpu', cfg=SimpleNamespace(stage='locomotion'),
        robot=SimpleNamespace(data=data, body_names=['L_link3', 'R_link3']),
        scene=SimpleNamespace(env_origins=torch.zeros(n, 3)),
        contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w_history=torch.zeros(n, 2, 7, 3))),
        _joint_ids=torch.arange(6), _leg_ids=torch.tensor([0, 1, 3, 4]), _knee_ids=torch.tensor([1, 4]),
        _nominal=q[0].clone(), _knee_limits=torch.tensor(list(c['joints']['knee_hard_limits'].values())),
        _wheel_body_ids=torch.tensor([5, 6]), _non_wheel_body_ids=torch.arange(5),
        actions=torch.zeros(n, 6), previous_actions=torch.zeros(n, 6), torques=torch.zeros(n, 6),
        leg_targets=q[:, [0, 1, 3, 4]].clone(), wheel_targets=torch.zeros(n, 2),
        commands=torch.tensor([[0., 0., .32]]).repeat(n, 1),
        _commands_due=torch.zeros(n, dtype=torch.bool), _command_ticks_left=torch.zeros(n, dtype=torch.long),
        _command_period_ticks=300, _invalid_actions=torch.zeros(n, dtype=torch.bool),
        _finite_state=torch.ones(n, dtype=torch.bool), _evaluation_command_override=None,
        _evaluation_command_pending=False, _evaluation_snapshot=None,
        _sustained_failure=core.SustainedFailure(n, 'cpu'), history=core.NoisyHistoryStack(n, 'cpu'),
        _episode_sums={}, extras={}, common_step_counter=0, _sim_step_counter=0,
        episode_length_buf=torch.zeros(n, dtype=torch.long), max_episode_length=2000,
        step_dt=.01, physics_dt=.005,
    )
    env._joint_state = lambda: (q, torch.zeros_like(q))
    env._base_height = lambda: data.root_link_pos_w[:, 2]
    env._base_visual_clearance = lambda: torch.full((n,), -.01)
    env._contact_magnitudes = lambda: torch.full((n, 7), 10.)
    env._named_indices = lambda actual, requested, kind: torch.tensor([actual.index(x) for x in requested])
    env.robot.write_root_pose_to_sim = lambda pose, env_ids: setattr(env, 'reset_pose', pose.clone())
    env.robot.write_root_velocity_to_sim = lambda velocity, env_ids: setattr(env, 'reset_velocity', velocity.clone())
    env.robot.write_joint_state_to_sim = lambda pos, vel, env_ids: setattr(env, 'reset_joint_pos', pos.clone())
    env.robot.set_joint_effort_target = lambda *args, **kwargs: None
    return env


def test_no_extra_done_sustained_failure_diagnostics_snapshot_and_reset(v2):
    env = adapter(v2)
    env._evaluation_command_override = (0., 0., .32)
    env.robot.data.root_link_pos_w[:, 2] = .10
    env._joint_state()[0][:, 1] = 1.
    for tick in range(1, 101):
        env.common_step_counter = tick
        env.robot.data.projected_gravity_b[:, 2] = 0.
        assert not env._get_dones()[0].any()
        assert not env._get_dones()[0].any()
    log = env.extras['log']
    for key in ('non_wheel_contact', 'knee_limit', 'low_height', 'base_visual_bounds_ground'):
        assert log['Diagnostic/' + key] == 1
        assert log['Termination/' + key] == 0
    env.common_step_counter = 101
    assert env._get_dones()[0].all()
    assert env._evaluation_snapshot['termination_flags']['tilt'].all()
    assert env._evaluation_snapshot['evaluation_settings']['observation_noise'] is False
    env._reset_idx([0])
    assert env._sustained_failure.count.tolist() == [0, 101]
    env.common_step_counter = 102
    env.robot.data.projected_gravity_b[1, 2] = -.1
    assert not env._get_dones()[0].any()
    assert env._sustained_failure.count.tolist() == [1, 0]
    env._invalid_actions[0] = True
    assert env._get_dones()[0].tolist() == [True, False]
    env.episode_length_buf[1] = 1999
    assert env._get_dones()[1].tolist() == [False, True]
    old = adapter(core.load_contract())
    assert old._get_dones()[0].all()  # V1 still terminates on contact/AABB immediately.


def test_noisy_observation_cache_clean_critic_and_partial_reset(v2):
    env = adapter(v2)
    first = env._get_observations()
    clean = first['critic'][:, :25]
    frame = first['policy'][:, -25:]
    assert not torch.equal(frame, clean)
    amplitude = torch.tensor([.1]*3 + [.05]*3 + [0.]*3 + [.02]*4 + [.15]*6 + [0.]*6)
    assert ((frame-clean).abs() <= amplitude + 1e-7).all()
    torch.testing.assert_close(first['critic'][:, -4:-1], env.robot.data.root_lin_vel_b)
    state = torch.random.get_rng_state().clone()
    second = env._get_observations()
    assert torch.equal(first['policy'], second['policy'])
    assert torch.equal(state, torch.random.get_rng_state())
    env.common_step_counter += 1
    advanced = env._get_observations()['policy'].reshape(2, 5, 25)
    torch.testing.assert_close(advanced[:, :-1], first['policy'].reshape(2, 5, 25)[:, 1:])
    env._reset_idx([0])
    reset = env._get_observations()['policy'].reshape(2, 5, 25)
    torch.testing.assert_close(reset[1], advanced[1])
    torch.testing.assert_close(reset[0], reset[0, -1:].expand(5, 25))


def test_seeded_training_resets_and_deterministic_evaluation(v2):
    env = adapter(v2)
    results = []
    for _ in range(2):
        torch.manual_seed(40)
        env._reset_idx(None)
        results.append((env.reset_velocity.clone(), env._get_observations()['policy']))
        torch.testing.assert_close(env.reset_joint_pos, env.robot.data.default_joint_pos)
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert results[0][0].abs().le(.5).all() and results[0][0].ne(0).any()
    env.set_evaluation_command((0., 0., .32))
    rng = torch.random.get_rng_state().clone()
    env._reset_idx(None)
    obs = env._get_observations()
    assert env.reset_velocity.eq(0).all()
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(obs['policy'][:, -25:], obs['critic'][:, :25])


def test_v2_export_and_frozen_provenance_reject_v1_checkpoint(v2, tmp_path):
    asset = audit_asset(v2)
    manifest = core.make_run_manifest(v2, asset)
    v40_export.validate_manifest(manifest)
    checkpoint, _ = make_checkpoint(manifest)
    checkpoint_path = tmp_path / 'model.pt'
    manifest_path = tmp_path / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    torch.save(checkpoint, checkpoint_path)
    report = v40_export.export_checkpoint(checkpoint_path, manifest_path, tmp_path / 'policy.onnx')
    assert report['contract_id'] == v2['contract_id']
    v40_job.snapshot_provenance(tmp_path, repo_root=ROOT, contract_path=V2_PATH,
                                contract=v2, asset=asset, manifest=manifest)
    assert json.loads((tmp_path / 'contract.json').read_text()) == v2
    assert {'src/wheeled_tasks/v40/core.py', 'src/wheeled_tasks/v40/contract.py',
            'src/wheeled_tasks/direct/v40_serial/env.py', 'scripts/train_v40.py',
            'src/wheeled_algo/v40_export.py'} <= set(v40_job.SOURCE_FILES)
    old_manifest = core.make_run_manifest(core.load_contract(), asset)
    old_checkpoint, _ = make_checkpoint(old_manifest)
    manifest_path.write_text(json.dumps(old_manifest))
    torch.save(old_checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match='checkpoint metadata mismatch'):
        load_cli().checked_checkpoint(checkpoint_path, manifest)
