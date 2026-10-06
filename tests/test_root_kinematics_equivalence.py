"""CPU semantic evidence; no PhysX, SimulationApp, CUDA launches, or PPO."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch
import warp as wp
wp.init()

ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(os.environ.get('ISAACLAB_SOURCE_ROOT', str(ROOT.parent / '.deployment_sources/IsaacLab-3.0.0-beta2/source')))
if not VENDOR.is_dir():
    raise unittest.SkipTest('Native vendor source is required for this explicit CPU equivalence check')
DATA_PATH = VENDOR/'isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py'
KERNEL_PATH = VENDOR/'isaaclab_physx/isaaclab_physx/assets/kernels.py'
PROXY_PATH = VENDOR/'isaaclab/isaaclab/utils/warp/proxy_array.py'
REPO = ROOT


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kernels = load('com020_vendor_kernels', KERNEL_PATH)
ProxyArray = load('com020_vendor_proxy', PROXY_PATH).ProxyArray
candidate = load('com020_root_kinematics', ROOT/'src/wheeled_tasks/chassis/root_kinematics.py')
observer = load('com020_recovery_observer', REPO/'src/wheeled_tasks/chassis/recovery_observer.py')

# CPU reference executes these vendor getter bodies verbatim from their AST.
# Synthetic backend buffers stand in for PhysX; production candidate uses no
# private timestamps or private buffers from the vendor data implementation.
METHODS = {'root_com_vel_w', 'root_link_pose_w', 'root_link_vel_w', 'body_com_pose_b',
           'root_link_lin_vel_w', 'root_link_lin_vel_b', 'root_link_quat_w',
           '_get_quat_from_transform', '_get_lin_vel_from_spatial_vector'}
original_ast = ast.parse(DATA_PATH.read_text())
original_class = next(x for x in original_ast.body if isinstance(x, ast.ClassDef) and x.name == 'ArticulationData')
selected = [x for x in original_class.body if isinstance(x, ast.FunctionDef) and x.name in METHODS]
assert {x.name for x in selected} == METHODS
reference_ast = ast.Module(body=[ast.ClassDef(name='OriginalGetters', bases=[], keywords=[], body=selected,
                                           decorator_list=[])], type_ignores=[])
namespace = {'wp': wp, 'ProxyArray': ProxyArray, 'shared_kernels': kernels}
exec(compile(ast.fix_missing_locations(reference_ast), str(DATA_PATH), 'exec'), namespace)
OriginalGetters = namespace['OriginalGetters']


class ModelView:
    def __init__(self, n, seed=620):
        self.generator = torch.Generator().manual_seed(seed)
        self.local = torch.randn(n, 19, 7, generator=self.generator) * .03
        q = torch.randn(n, 19, 4, generator=self.generator)
        self.local[:, :, 3:] = q/q.norm(dim=-1, keepdim=True)
        self.pose = torch.randn(n, 7, generator=self.generator)
        q = torch.randn(n, 4, generator=self.generator)
        self.pose[:, 3:] = q/q.norm(dim=-1, keepdim=True)
        self.velocity = torch.randn(n, 6, generator=self.generator)
        self.local_wp = wp.from_torch(self.local)
        self.pose_wp = wp.from_torch(self.pose)
        self.velocity_wp = wp.from_torch(self.velocity)
        self.com_reads = 0

    def get_coms(self):
        self.com_reads += 1
        return self.local_wp

    def get_root_velocities(self):
        return self.velocity_wp

    def get_root_transforms(self):
        return self.pose_wp


class Robot:
    def __init__(self, n, seed=620):
        self.root_view = ModelView(n, seed)
        d = self.data = OriginalGetters()
        d._root_view = self.root_view
        d.device = 'cpu'
        d._num_instances = n
        d._sim_timestamp = 0.
        for name, shape, dtype in [('root_com_vel_w', n, wp.spatial_vectorf),
                                  ('root_link_pose_w', n, wp.transformf),
                                  ('root_link_vel_w', n, wp.spatial_vectorf),
                                  ('root_link_lin_vel_b', n, wp.vec3f),
                                  ('body_com_pose_b', (n, 19), wp.transformf)]:
            setattr(d, '_'+name, NS(data=wp.zeros(shape, dtype=dtype, device='cpu'), timestamp=-1.))
            setattr(d, '_'+name+'_ta', None)
        d._root_link_lin_vel_w_ta = d._root_link_quat_w_ta = None
        self.calls = []
        self.fail_update = False

    def update(self, dt):
        if self.fail_update:
            raise RuntimeError('synthetic update failure')
        self.calls.append(dt)
        self.data._sim_timestamp += dt


def setup(n=48):
    robot = Robot(n)
    cache = candidate.StaticRootLinkKinematics(robot, n, 'cpu', kernels=kernels)
    return NS(robot=robot, root_kinematics=cache), cache


class Equivalence(unittest.TestCase):
    def assert_exact(self, left, right):
        self.assertTrue(torch.equal(left, right), f'max absolute difference {(left-right).abs().max().item()}')

    def compare(self, env, frame='world'):
        name = 'root_link_lin_vel_w' if frame == 'world' else 'root_link_lin_vel_b'
        expected = getattr(env.robot.data, name).torch
        actual = candidate.root_link_linear_velocity(env, frame=frame)
        self.assert_exact(actual, expected)
        return actual.clone()

    def test_4096_world_and_body_vendor_getters_bitwise(self):
        env, cache = setup(4096)
        self.compare(env)
        self.compare(env, 'body')
        self.assertEqual(cache.linear_velocity_world().stride(), (6, 1))
        self.assertEqual(cache.linear_velocity_body().stride(), (3, 1))

    def test_known_point_velocity_and_principal_frame(self):
        env, cache = setup(2)
        env.robot.root_view.local[:, 0, :3] = torch.tensor([.2, 0., 0.])
        env.robot.root_view.local[:, 0, 3:] = torch.tensor([0., .70710677, 0., .70710677])
        env.robot.root_view.pose[:, 3:] = torch.tensor([0., 0., 0., 1.])
        env.robot.root_view.velocity[:] = torch.tensor([0., 0., 0., 0., 0., 2.])
        cache.refresh_model()
        self.assert_exact(self.compare(env), torch.tensor([[0., -.4, 0.]]*2))
        self.compare(env, 'body')
        old = cache.linear_velocity_body().clone()
        env.robot.root_view.local[:, 0, 3:] = torch.tensor([.70710677, 0., 0., .70710677])
        cache.refresh_model()
        self.assert_exact(cache.linear_velocity_body(), old)

    def test_snapshot_detached_and_after_final_setters(self):
        env, cache = setup()
        before = cache.local_com.clone()
        # Simulate set_inertias changing principal axes followed by final set_coms.
        env.robot.root_view.local[:, 0, 3:] = torch.tensor([1., 0., 0., 0.])
        self.assert_exact(cache.local_com, before)
        cache.invalidate_model()
        with self.assertRaises(RuntimeError):
            cache.linear_velocity_world()
        with self.assertRaises(RuntimeError):
            cache.linear_velocity_body()
        env.robot.root_view.local[:, 0, :3] += .01
        cache.refresh_model()
        self.assert_exact(cache.local_com, env.robot.root_view.local[:, :1])
        self.compare(env)

    def test_same_epoch_push_keeps_old_world_and_body(self):
        env, cache = setup()
        world = self.compare(env)
        body = self.compare(env, 'body')
        env.robot.root_view.velocity[:, :3] += .15
        self.assert_exact(self.compare(env), world)
        self.assert_exact(self.compare(env, 'body'), body)
        candidate.update_robot_data(env, .005)
        self.assertFalse(torch.equal(self.compare(env), world))
        self.compare(env, 'body')

    def test_same_epoch_pose_write_separate_body_first_read(self):
        env, cache = setup()
        world = self.compare(env)
        env.robot.root_view.pose[:, 3:] = torch.tensor([0., 0., 1., 0.])
        self.assert_exact(self.compare(env), world)
        self.compare(env, 'body')  # old world transformed by current link quaternion
        candidate.update_robot_data(env, .005)
        self.compare(env)
        self.compare(env, 'body')

    def test_push_before_first_read_uses_current_public_com(self):
        env, cache = setup()
        env.robot.root_view.velocity[:, :3] += .22
        self.compare(env)
        self.compare(env, 'body')

    def test_partial_reset_advances_global_epoch_for_all_lanes(self):
        env, cache = setup()
        old = self.compare(env)
        self.compare(env, 'body')
        ids = torch.tensor([0, 3, 17])
        env.robot.root_view.velocity[ids] = 0
        env.robot.root_view.pose[ids, 3:] = torch.tensor([0., 0., 0., 1.])
        candidate.update_robot_data(env, .005)
        now = self.compare(env)
        self.compare(env, 'body')
        self.assert_exact(now[ids], torch.zeros(3, 3))
        self.assert_exact(now[1], old[1])
        self.assertEqual(cache.read_epoch, 1)

    def test_update_failure_does_not_advance_epoch(self):
        env, cache = setup()
        env.robot.fail_update = True
        with self.assertRaises(RuntimeError):
            candidate.update_robot_data(env, .005)
        self.assertEqual(cache.read_epoch, 0)
        self.assertEqual(env.robot.calls, [])

    def test_zero_dt_does_not_create_a_new_read_epoch(self):
        env, cache = setup()
        before = self.compare(env)
        env.robot.root_view.velocity[:, :3] += .15
        candidate.update_robot_data(env, 0.)
        self.assert_exact(self.compare(env), before)
        self.assertEqual(cache.read_epoch, 0)

    def test_observation_hook_runs_after_native_update_and_epoch(self):
        env, cache = setup()
        old = self.compare(env)
        original_update = cache.update_robot_data
        samples = []
        def observed_update(dt):
            original_update(dt)
            samples.append(self.compare(env, 'body'))
        cache.update_robot_data = observed_update
        env.robot.root_view.velocity[:, :3] += .15
        candidate.update_robot_data(env, .005)
        self.assertEqual(cache.read_epoch, 1)
        self.assertEqual(len(samples), 1)
        self.assertFalse(torch.equal(self.compare(env), old))

    def test_cpu_600_policy_ticks_2400_physics_imu_and_cache_history(self):
        env, cache = setup(16)
        baseline_imu = observer.SimulatedImu('cpu', 16, .005, noise=True)
        candidate_imu = observer.SimulatedImu('cpu', 16, .005, noise=True)
        baseline_imu.reset(self.compare(env))
        candidate_imu.reset(cache.linear_velocity_world())
        rng = torch.Generator().manual_seed(6617)
        saved_rng = rng.get_state().clone()
        for policy in range(600):
            for substep in range(4):
                env.robot.root_view.velocity.add_(torch.randn(16, 6, generator=rng)*.004)
                candidate.update_robot_data(env, .005)
                velocity = self.compare(env)
                pose = env.robot.data.root_link_pose_w.torch[:, 3:]
                expected = baseline_imu.sample(velocity, pose)
                actual = candidate_imu.sample(cache.linear_velocity_world(), pose)
                self.assert_exact(actual, expected)
                self.assert_exact(candidate_imu.previous_velocity, baseline_imu.previous_velocity)
            self.compare(env, 'body')
            if policy % 17 == 0:
                env.robot.root_view.velocity[::3, :2] += .15  # post-reward same-epoch push
                self.compare(env)
                self.compare(env, 'body')
            if policy % 31 == 0:
                env.robot.root_view.velocity[::5] = 0  # partial reset and evaluator second update
                candidate.update_robot_data(env, .005)
                self.compare(env)
                candidate.update_robot_data(env, .005)
                self.compare(env)
        self.assert_exact(candidate_imu.generator.get_state(), baseline_imu.generator.get_state())
        self.assertFalse(torch.equal(saved_rng, rng.get_state()))
        self.assertEqual(len(env.robot.calls), 2400+2*20)

    def test_static_model_removes_repeated_get_coms_cpu_semantics_only(self):
        env, cache = setup(16)
        for _ in range(100):
            candidate.update_robot_data(env, .005)
            cache.linear_velocity_world()
            cache.linear_velocity_body()
        self.assertEqual(env.robot.root_view.com_reads, 1)
        baseline = Robot(16)
        for _ in range(100):
            baseline.update(.005)
            baseline.data.root_link_lin_vel_w.torch
            baseline.data.root_link_lin_vel_b.torch
        self.assertEqual(baseline.root_view.com_reads, 100)

    def test_mock_fallback_frame_contract_and_policy_com_untouched(self):
        env, cache = setup()
        with self.assertRaises(ValueError):
            candidate.root_link_linear_velocity(env, frame='principal')
        com = env.robot.data.root_com_vel_w.torch.clone()
        self.compare(env)
        self.compare(env, 'body')
        self.assert_exact(env.robot.data.root_com_vel_w.torch, com)
        no_cache = NS(robot=env.robot)
        self.assert_exact(candidate.root_link_linear_velocity(no_cache), env.robot.data.root_link_lin_vel_w.torch)
        candidate.update_robot_data(no_cache, .005)


