"""Round4 scratch flat profile and one PPO-update-driven velocity-push curriculum.

Contract/cache validation is stdlib-only. No simulator import or implicit training.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from .round3 import Round3Commands

GROUND_ASSET_SHA256 = '78e9a1e72a8838a13d0f65c49cd487ab92e89233cd128b057730b5b5b4ca2164'
GROUND_ASSET_URL = ('https://omniverse-content-production.s3-us-west-2.amazonaws.com/'
                    'Assets/Isaac/6.0/Isaac/Environments/Grid/default_environment.usd')


def verify_ground_usd(path):
    """Verify a local copy of the exact official grid asset, without modifying it."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != GROUND_ASSET_SHA256:
        raise ValueError('ground USD SHA256 differs from the pinned official Sim6 asset')
    return {'path': str(path.resolve()), 'sha256': digest, 'size': len(raw), 'source_url': GROUND_ASSET_URL}


def validate_round4(contract):
    """Only the reviewed scratch/push profile may extend the strict B1 contract."""
    from .round3 import validate_round3_b1
    cfg = contract['round4']
    expected = {
        'profile': 'flat_full', 'initialization': 'scratch', 'num_envs': 1024,
        'planned_updates': 30000, 'rollout_steps_per_env': 48,
        'push': {
            'generator_seed': 2044, 'no_push_episode_probability': .5,
            'grace_seconds': 2.0, 'interval_seconds': [3.0, 5.0],
            'application': 'add_root_com_world_xy_velocity',
            'direction': 'uniform_world_xy', 'magnitude': 'uniform_0_to_max',
            'schedule': [[0, .1], [5000, .25], [10000, .5]],
        },
        'command_curriculum': {
            'schedule': [[0, .5, 1.0], [6000, 2.0, 2.0], [25000, 3.0, 6.0]],
            'bucket_probabilities': {'stand': .3, 'pure_yaw': .2, 'straight': .3, 'combined': .2},
            'track_m': .4373, 'wheel_radius_m': .06,
            'traction_fraction': .7, 'wheel_speed_fraction': .8,
        },
        'stabilization': {'wheel_deadzone_m_s': .018, 'wheel_scale_m_s': .1, 'zero_command_threshold': .05},
    }
    if not isinstance(cfg, dict) or set(cfg) != set(expected) or not isinstance(cfg.get('push'), dict):
        raise ValueError('Round4 profile fields mismatch')
    seed = cfg['push'].get('generator_seed')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Round4 push generator seed must be an integer in [0,2**63)')
    expected['push']['generator_seed'] = seed
    # Canonical JSON rejects bool-as-number, nonfinite and unknown nested fields.
    if json.dumps(cfg, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError('Round4 permits only the reviewed scratch flat push profile')
    inherited = deepcopy(contract)
    del inherited['round4']
    if inherited.get('round3', {}).get('stage') != 'B1':
        raise ValueError('Round4 must use the existing B1 material/command component')
    # Only these reviewed full-profile differences may be removed before B1 validation.
    changes = {
        'vx': [-3.0, 3.0], 'wz': [-6.0, 6.0],
    }
    for key, expected_range in changes.items():
        if inherited['commands']['stages']['locomotion'][key] != expected_range:
            raise ValueError('Round4 full command bounds must be vx±3 and wz±6')
        inherited['commands']['stages']['locomotion'][key] = [-2.0, 2.0]
    if inherited['round3']['spin_probability'] != .2:
        raise ValueError('Round4 full pure-yaw probability must be .2')
    inherited['round3']['spin_probability'] = .1
    for key, weight in (('body_angular_rate', -.05), ('wheel_quiet', -.02)):
        if inherited['rewards']['weights'].pop(key, None) != weight:
            raise ValueError('Round4 full stabilization weight mismatch')
    validate_round3_b1(inherited)


def effective_dynamic_from_readback(report, num_envs):
    """Use verified PhysX coefficients, not requested buckets, for command budgets."""
    if report.get('stage') != 'B1' or report.get('passed') is not True:
        raise ValueError('command curriculum requires a verified B1 material report')
    ids, values = report['wheel_env_ids'], report['wheel_physx_coefficients']
    if len(ids) != 2 * num_envs or len(values) != len(ids):
        raise ValueError('material readback must contain two wheel shapes per environment')
    result, counts = [None] * num_envs, [0] * num_envs
    for env_id, row in zip(ids, values):
        if type(env_id) is not int or not 0 <= env_id < num_envs or len(row) != 1 or len(row[0]) != 3:
            raise ValueError('invalid material readback layout')
        dynamic = row[0][1]
        if type(dynamic) not in (int, float) or not math.isfinite(dynamic):
            raise ValueError('nonfinite material readback')
        effective = .5 * (float(dynamic) + .5)
        if result[env_id] is not None and abs(result[env_id] - effective) > 1e-6:
            raise ValueError('B1 left/right friction readback differs')
        result[env_id] = effective
        counts[env_id] += 1
    if counts != [2] * num_envs:
        raise ValueError('missing/duplicate wheel material readback')
    return result


class FullManeuverCommands(Round3Commands):
    """Old-structure four-bucket commands with friction/curve sampling priors only."""

    def __init__(self, contract, stage, num_envs, device, effective_dynamic_mu):
        import torch
        super().__init__(contract, stage, num_envs, device)
        self.maneuver = deepcopy(contract['round4']['command_curriculum'])
        self.training_iteration = 0
        self.effective_dynamic_mu = torch.as_tensor(effective_dynamic_mu, device=device, dtype=torch.float32).clone()
        if (self.effective_dynamic_mu.shape != (num_envs,) or not torch.isfinite(self.effective_dynamic_mu).all()
                or (self.effective_dynamic_mu < .4 - 1e-6).any() or (self.effective_dynamic_mu > .6 + 1e-6).any()):
            raise ValueError('Round4 requires one actual B1 effective dynamic friction per environment')
        wheel = contract['actuators']['wheel']
        self.wheel_ground_speed_cap = (self.maneuver['wheel_speed_fraction'] * self.maneuver['wheel_radius_m']
                                       * wheel['motor_speed_rad_s'][-1] / wheel['gear_ratio'])
        self.traction_budget = self.maneuver['traction_fraction'] * self.effective_dynamic_mu * 9.81
        self.requested_velocity = torch.zeros(num_envs, 2, device=device)
        self.executed_velocity = torch.zeros_like(self.requested_velocity)
        self.traction_clipped = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.wheel_speed_clipped = torch.zeros_like(self.traction_clipped)

    def set_training_iteration(self, update):
        if type(update) is not int or update < self.training_iteration:
            raise ValueError('command curriculum requires monotonic completed PPO updates')
        self.training_iteration = update

    @property
    def caps(self):
        schedule = self.maneuver['schedule']
        for left, right in zip(schedule, schedule[1:]):
            if self.training_iteration <= right[0]:
                fraction = (self.training_iteration - left[0]) / (right[0] - left[0])
                return tuple(a + fraction * (b-a) for a, b in zip(left[1:], right[1:]))
        return tuple(schedule[-1][1:])

    def constrain(self, requested, env_ids):
        """Common scaling preserves signs/curvature; neither bound changes the solver."""
        import torch
        if requested.shape != (len(env_ids), 2) or not torch.isfinite(requested).all():
            raise ValueError('maneuver requests must be finite [env,2]')
        product = requested[:, 0].abs() * requested[:, 1].abs()
        traction_scale = torch.sqrt(self.traction_budget[env_ids] / product.clamp_min(1e-8)).clamp(max=1.)
        outer_speed = requested[:, 0].abs() + requested[:, 1].abs() * self.maneuver['track_m'] / 2
        speed_scale = (self.wheel_ground_speed_cap / outer_speed.clamp_min(1e-8)).clamp(max=1.)
        self.traction_clipped[env_ids] = traction_scale < 1.
        self.wheel_speed_clipped[env_ids] = speed_scale < 1.
        return requested * torch.minimum(traction_scale, speed_scale)[:, None]

    def sample_velocity(self, commands, env_ids):
        import torch
        if not self.mixed_velocity:
            super().sample_velocity(commands, env_ids)
            self.requested_velocity[env_ids] = commands[env_ids, :2]
            self.executed_velocity[env_ids] = commands[env_ids, :2]
            return
        if len(env_ids) == 0:
            return
        vx_cap, yaw_cap = self.caps
        requested = (2 * torch.rand(len(env_ids), 2, device=commands.device) - 1)
        requested *= requested.new_tensor([vx_cap, yaw_cap])
        draw = torch.rand(len(env_ids), device=commands.device)
        probabilities = self.maneuver['bucket_probabilities']
        a = probabilities['stand']
        b = a + probabilities['pure_yaw']
        c = b + probabilities['straight']
        stand, spin, straight = draw < a, (draw >= a) & (draw < b), (draw >= b) & (draw < c)
        requested[stand] = 0.
        requested[spin, 0] = 0.
        requested[straight, 1] = 0.
        self.velocity_bucket[env_ids] = torch.where(stand, 0, torch.where(spin, 1, torch.where(straight, 2, 3)))
        executed = self.constrain(requested, env_ids)
        self.requested_velocity[env_ids] = requested
        self.executed_velocity[env_ids] = executed
        commands[env_ids, :2] = executed

    def state(self):
        return {
            'completed_ppo_updates': self.training_iteration, 'vx_cap_m_s': self.caps[0], 'yaw_cap_rad_s': self.caps[1],
            'caps_apply_at_next_resampling': True, 'bucket_order': ['stand', 'pure_yaw', 'straight', 'combined'],
            'bucket_probabilities': deepcopy(self.maneuver['bucket_probabilities']),
            'wheel_ground_speed_cap_m_s': self.wheel_ground_speed_cap,
            'traction_fraction': self.maneuver['traction_fraction'],
            'effective_dynamic_mu': self.effective_dynamic_mu.cpu().tolist(),
            'requested_velocity_m_s_rad_s': self.requested_velocity.cpu().tolist(),
            'executed_velocity_m_s_rad_s': self.executed_velocity.cpu().tolist(),
            'traction_clipped': self.traction_clipped.cpu().tolist(),
            'wheel_speed_clipped': self.wheel_speed_clipped.cpu().tolist(),
            'scope': 'command sampling prior; not measured slip, ground support or a physical hard limit',
        }


class PushCurriculum:
    """Per-episode push masks/timers and an independent torch RNG, no physics writes."""

    def __init__(self, contract, num_envs, device):
        import torch
        self.config = deepcopy(contract['round4']['push'])
        self.policy_dt = contract['timing']['policy_dt']
        self.rollout_steps = contract['round4']['rollout_steps_per_env']
        self.device = device
        self.training_iteration = 0
        self.last_policy_tick = -1
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.config['generator_seed'])
        self.episode_push_enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.next_push_episode_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.grace_ticks = round(self.config['grace_seconds'] / self.policy_dt)
        self.interval_ticks = [round(x / self.policy_dt) for x in self.config['interval_seconds']]

    def set_training_iteration(self, iteration):
        if type(iteration) is not int or iteration < self.training_iteration:
            raise ValueError('completed PPO update count must be a monotonic nonnegative integer')
        self.training_iteration = iteration

    @property
    def max_delta_v(self):
        schedule = self.config['schedule']
        update = self.training_iteration
        for (left, low), (right, high) in zip(schedule, schedule[1:]):
            if update <= right:
                return low + (high - low) * (update - left) / (right - left)
        return schedule[-1][1]

    def _interval(self, count):
        import torch
        return torch.randint(self.interval_ticks[0], self.interval_ticks[1] + 1, (count,),
                             device=self.device, generator=self.generator)

    def reset(self, env_ids, *, automatic=True):
        import torch
        self.episode_push_enabled[env_ids] = False
        self.next_push_episode_step[env_ids] = -1
        if not automatic or len(env_ids) == 0:
            return
        enabled = torch.rand(len(env_ids), device=self.device, generator=self.generator) >= self.config['no_push_episode_probability']
        first = self.grace_ticks + self._interval(len(env_ids))
        self.episode_push_enabled[env_ids] = enabled
        self.next_push_episode_step[env_ids] = torch.where(enabled, first, -1)

    def sample(self, policy_tick, episode_steps, *, automatic=True):
        """None denotes an already-processed tick; empty tensors denote no events."""
        import torch
        if type(policy_tick) is not int or policy_tick < 0 or policy_tick < self.last_policy_tick:
            raise ValueError('push policy tick must be monotonic and nonnegative')
        if policy_tick == self.last_policy_tick:
            return None
        if episode_steps.shape != self.next_push_episode_step.shape or episode_steps.dtype not in (torch.int32, torch.int64):
            raise ValueError('push episode steps must be one integer per environment')
        self.last_policy_tick = policy_tick
        if not automatic:
            return torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, 2, device=self.device)
        due = self.episode_push_enabled & (episode_steps > self.grace_ticks) & (episode_steps >= self.next_push_episode_step)
        ids = due.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return ids, torch.empty(0, 2, device=self.device)
        angle = torch.rand(len(ids), device=self.device, generator=self.generator) * (2 * math.pi)
        magnitude = torch.rand(len(ids), device=self.device, generator=self.generator) * self.max_delta_v
        delta = magnitude[:, None] * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        self.next_push_episode_step[ids] = episode_steps[ids] + self._interval(len(ids))
        return ids, delta

    def state(self):
        """JSON inspection state; only iteration progress is supported for restoration."""
        return {
            'schema_version': 1, 'completed_ppo_updates': self.training_iteration,
            'max_delta_v_m_s': self.max_delta_v, 'generator_seed': self.config['generator_seed'],
            'generator_device': str(self.device), 'policy_dt_s': self.policy_dt,
            'rollout_steps_per_env': self.rollout_steps, 'last_policy_tick': self.last_policy_tick,
            'episode_push_enabled': self.episode_push_enabled.cpu().tolist(),
            'next_push_episode_step': self.next_push_episode_step.cpu().tolist(),
            'first_event_timing': 'grace 2s plus independent uniform 3..5s; subsequent interval 3..5s',
            'exact_trajectory_resume_supported': False,
            'resume_scope': 'set_training_iteration restores curriculum amplitude; push masks/timers/RNG restart on episode reset',
        }
