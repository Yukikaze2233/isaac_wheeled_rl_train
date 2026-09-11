"""Pure Torch V4 observation/history/action/reward math; no Isaac dependency.

Actor inputs are already normalized history (oldest first), not raw sensors.
All joint arrays use the six-name contract order. Dynamics remain research priors.
"""
from __future__ import annotations

import torch
from .contract import (
    REPO_ROOT, DEFAULT_CONTRACT, load_contract, validate_contract, contract_digest,
    audit_asset, validate_asset, make_run_manifest, is_round2,
)


def _matrix(value, width, name, batch=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width or not value.is_floating_point():
        raise ValueError(f'{name} must be floating Tensor[N,{width}]')
    if batch is not None and value.shape[0] != batch:
        raise ValueError(f'{name} batch mismatch')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} contains nonfinite values')
    return value


def _vector(value, batch, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f'{name} must be a tensor')
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1 or value.shape[0] != batch or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite floating Tensor[N] or Tensor[N,1]')
    return value


def _like(values, reference):
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class HistoryStack:
    """One advance per policy tick; reset envs repeat their first valid frame.

    Repeated queries at the same tick return the same history. A reset within
    that tick refreshes only those environments, never the other batch members.
    """
    def __init__(self, num_envs, device, length=5, dim=25):
        if type(num_envs) is not int or num_envs < 1 or type(length) is not int or length < 1 or type(dim) is not int or dim < 1:
            raise ValueError('History dimensions must be positive integers')
        self.num_envs, self.length, self.dim = num_envs, length, dim
        self.buffer = torch.zeros((num_envs, length, dim), device=device, dtype=torch.float32)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.last_tick = None

    def reset(self, env_ids):
        raw = torch.as_tensor(env_ids, device=self.buffer.device)
        if raw.ndim == 1 and raw.numel() == 0:
            return
        if raw.dtype not in (torch.int32, torch.int64) or raw.ndim != 1:
            raise ValueError('reset expects one-dimensional integer environment ids')
        ids = raw.to(torch.long)
        if ids.numel() and ((ids < 0).any() or (ids >= self.num_envs).any()):
            raise ValueError('reset environment id out of range')
        self.buffer[ids] = 0
        self.initialized[ids] = False

    def update(self, observation, tick):
        _matrix(observation, self.dim, 'history observation', self.num_envs)
        if observation.device != self.buffer.device or observation.dtype != self.buffer.dtype:
            raise ValueError('History uses float32 on its configured device')
        if type(tick) is not int or tick < 0 or self.last_tick is not None and tick < self.last_tick:
            raise ValueError('Policy tick must be a monotonic nonnegative integer')
        fresh = ~self.initialized
        if self.last_tick != tick:
            self.buffer[:, :-1] = self.buffer[:, 1:].clone()
            self.buffer[:, -1] = observation
        if fresh.any():
            self.buffer[fresh] = observation[fresh, None, :].expand(-1, self.length, -1)
            self.initialized[fresh] = True
        self.last_tick = tick
        # A snapshot, not a view that later updates could mutate in rollout storage.
        return self.buffer.reshape(self.num_envs, self.length * self.dim).clone()


def build_observation(angular_velocity, projected_gravity, commands3,
                      joint_pos6, joint_vel6, last_actions6, contract):
    q = _matrix(joint_pos6, 6, 'joint positions')
    n = q.shape[0]
    inputs = [(angular_velocity, 3, 'angular velocity'), (projected_gravity, 3, 'gravity'),
              (commands3, 3, 'commands'), (joint_vel6, 6, 'joint velocities'), (last_actions6, 6, 'last actions')]
    for tensor, width, name in inputs:
        _matrix(tensor, width, name, n)
        if tensor.device != q.device or tensor.dtype != q.dtype:
            raise ValueError('Observation inputs must share dtype/device')
    j, o = contract['joints'], contract['observations']
    error = q - _like(j['nominal_positions'], q)
    # Only continuous hips wrap; knees never wrap across a physical stop.
    error = error.clone()
    error[:, j['hip_indices']] = wrap_angle(error[:, j['hip_indices']])
    scales = o['scales']
    obs = torch.cat((angular_velocity * scales['angular_velocity'], projected_gravity,
                     commands3 * _like(scales['command'], q),
                     error[:, j['leg_indices']] * scales['joint_position'],
                     joint_vel6 * scales['joint_velocity'], last_actions6), dim=-1)
    if obs.shape != (n, 25):
        raise ValueError('Internal observation layout mismatch')
    return obs.clamp(-o['clip'], o['clip'])


def build_critic(observation25, linear_velocity_body, base_height):
    obs = _matrix(observation25, 25, 'critic proprioception')
    velocity = _matrix(linear_velocity_body, 3, 'critic true linear velocity', obs.shape[0])
    height = _vector(base_height, obs.shape[0], 'critic height')
    return torch.cat((obs, velocity, height[:, None]), dim=-1)


class NoisyHistoryStack(HistoryStack):
    """Cache the noisy actor frame per tick/reset; critic inputs stay external/clean."""

    def update_actor(self, observation, tick, contract, enabled=True):
        fresh = ~self.initialized
        rows = torch.ones_like(fresh) if self.last_tick != tick else fresh
        frame = self.buffer[:, -1].clone()
        if rows.any():
            values = observation[rows].clone()
            if enabled:
                noise = contract['observations']['noise']
                scales = contract['observations']['scales']
                amplitude = _like(
                    [noise['angular_velocity'] * scales['angular_velocity']] * 3
                    + [noise['gravity']] * 3 + [0.] * 3
                    + [noise['joint_position'] * scales['joint_position']] * 4
                    + [noise['joint_velocity'] * scales['joint_velocity']] * 6 + [0.] * 6,
                    values,
                )
                values += (2 * torch.rand_like(values) - 1) * amplitude
            clip = contract['observations']['clip']
            frame[rows] = values.clamp(-clip, clip)
        return self.update(frame, tick)


class SustainedFailure:
    """Count consecutive bad policy ticks once, with independent partial resets."""

    def __init__(self, num_envs, device):
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ticks = torch.full_like(self.count, -1)

    def reset(self, env_ids):
        self.count[env_ids] = 0
        self.ticks[env_ids] = -1

    def update(self, gravity_z, tick, contract):
        limits = contract['termination']
        fresh = self.ticks != tick
        bad = gravity_z > limits['failure_gravity_z']
        self.count[fresh] = torch.where(bad[fresh], self.count[fresh] + 1, 0)
        self.ticks[fresh] = tick
        return self.count > round(limits['failure_seconds'] / contract['timing']['policy_dt'])


def decode_targets(actions6, joint_pos6, contract):
    q = _matrix(joint_pos6, 6, 'joint positions')
    actions = _matrix(actions6, 6, 'policy actions', q.shape[0])
    if actions.dtype != q.dtype or actions.device != q.device:
        raise ValueError('Action and joint state dtype/device mismatch')
    a, j = contract['actions'], contract['joints']
    clipped = actions.clamp(-a['clip'], a['clip'])
    nominal = _like(j['nominal_positions'], q)
    desired = nominal[j['leg_indices']] + clipped[:, j['leg_indices']] * _like(a['leg_position_scales'], q)
    targets = desired.clone()
    for column, idx in enumerate(j['leg_indices']):
        name = j['action_order'][idx]
        if idx in j['hip_indices']:
            bounded = desired[:, column] if is_round2(contract) else desired[:, column].clamp(nominal[idx] - j['hip_soft_deviation'], nominal[idx] + j['hip_soft_deviation'])
            targets[:, column] = q[:, idx] + wrap_angle(bounded - q[:, idx])
        else:
            lo, hi = j['knee_hard_limits'][name]
            margin = 0.0 if is_round2(contract) else j['knee_soft_margin']
            targets[:, column] = desired[:, column].clamp(lo + margin, hi - margin)
    wheel = clipped[:, j['wheel_indices']] * a['wheel_velocity_scale']
    return targets, wheel, clipped


def motor_torque_limit(joint_speed, wheel_config):
    """Lookup motor-side curve at gear_ratio*joint_speed, then map torque back.

    No extrapolation of positive torque beyond the supplied speed domain.
    The default curve is explicitly an unverified linear research prior.
    """
    if not isinstance(joint_speed, torch.Tensor) or not joint_speed.is_floating_point() or not torch.isfinite(joint_speed).all():
        raise ValueError('joint_speed must be a finite floating tensor')
    ratio = wheel_config['gear_ratio']
    eta = wheel_config['gearbox_efficiency']
    if ratio <= 0 or not 0 < eta <= 1 or wheel_config['curve_side'] != 'motor':
        raise ValueError('Invalid motor-side curve domain')
    speeds = _like(wheel_config['motor_speed_rad_s'], joint_speed)
    torques = _like(wheel_config['motor_torque_nm'], joint_speed)
    if speeds.ndim != 1 or speeds.numel() < 2 or speeds.numel() != torques.numel() or not (speeds[1:] > speeds[:-1]).all() or not torch.isfinite(speeds).all() or not torch.isfinite(torques).all() or (torques < 0).any():
        raise ValueError('Invalid torque-speed samples')
    query = joint_speed.abs().reshape(-1).contiguous() * ratio
    idx = torch.searchsorted(speeds, query).clamp(1, speeds.numel() - 1)
    low, high = idx - 1, idx
    weight = ((query - speeds[low]) / (speeds[high] - speeds[low])).clamp(0, 1)
    motor_torque = torques[low] + weight * (torques[high] - torques[low])
    motor_torque = torch.where(query > speeds[-1], torch.zeros_like(motor_torque), motor_torque)
    return (motor_torque * ratio * eta).clamp(max=wheel_config['effort_limit']).reshape(joint_speed.shape)


def compute_torques(joint_pos6, joint_vel6, leg_targets4, wheel_targets2, contract):
    """Effective joint-space research forces, NOT identified DM motor commands.

    The physical chain/coaxial motor-to-joint mapping is still uncalibrated.
    Chain relocation alone neither changes FK nor proves coupling. Do not send
    these leg forces directly to real motors or call this hardware matching.
    """
    q = _matrix(joint_pos6, 6, 'joint positions')
    n = q.shape[0]
    v = _matrix(joint_vel6, 6, 'joint velocities', n)
    legs = _matrix(leg_targets4, 4, 'leg targets', n)
    wheels = _matrix(wheel_targets2, 2, 'wheel targets', n)
    ids, actuators = contract['joints'], contract['actuators']
    leg_cfg, wheel_cfg = actuators['leg'], actuators['wheel']
    torque = torch.zeros_like(q)
    leg_torque = leg_cfg['kp'] * (legs - q[:, ids['leg_indices']]) - leg_cfg['kd'] * v[:, ids['leg_indices']]
    torque[:, ids['leg_indices']] = leg_torque.clamp(-leg_cfg['effort_limit'], leg_cfg['effort_limit'])
    wheel_torque = wheel_cfg['kd'] * (wheels - v[:, ids['wheel_indices']])
    bound = motor_torque_limit(v[:, ids['wheel_indices']], wheel_cfg)
    torque[:, ids['wheel_indices']] = torch.maximum(torch.minimum(wheel_torque, bound), -bound)
    if not torch.isfinite(torque).all():
        raise ValueError('Nonfinite actuator effort')
    return torque


def compute_reward_terms(v_body3, w_body3, gravity3, height, commands3, actions6,
                         previous_actions6, torques6, joint_pos6, contract):
    """Weighted, policy-dt-scaled nonterminal rewards (do not multiply dt twice)."""
    v = _matrix(v_body3, 3, 'body linear velocity')
    n = v.shape[0]
    for value, width, name in [(w_body3, 3, 'body angular velocity'), (gravity3, 3, 'gravity'),
                                (commands3, 3, 'commands'), (actions6, 6, 'actions'),
                                (previous_actions6, 6, 'previous actions'), (torques6, 6, 'torques'),
                                (joint_pos6, 6, 'joint positions')]:
        _matrix(value, width, name, n)
    h = _vector(height, n, 'base height')
    r, j = contract['rewards'], contract['joints']
    soft = torch.zeros_like(h)
    for idx in j['knee_indices']:
        lo, hi = j['knee_hard_limits'][j['action_order'][idx]]
        if is_round2(contract):
            margin = (hi - lo) * (1 - j['soft_position_limit_factor']) / 2
            soft += (lo + margin - joint_pos6[:, idx]).clamp(min=0)
            soft += (joint_pos6[:, idx] - hi + margin).clamp(min=0)
        else:
            soft += (lo + j['knee_soft_margin'] - joint_pos6[:, idx]).clamp(min=0).square()
            soft += (joint_pos6[:, idx] - hi + j['knee_soft_margin']).clamp(min=0).square()
    caps = [contract['actuators']['wheel' if idx in j['wheel_indices'] else 'leg']['effort_limit'] for idx in range(6)]
    raw = {
        'velocity': torch.exp(-((v[:, 0] - commands3[:, 0]) / r['sigma_velocity']).square()),
        'yaw': torch.exp(-((w_body3[:, 2] - commands3[:, 1]) / r['sigma_yaw']).square()),
        'height': torch.exp(-((h - commands3[:, 2]) / r['sigma_height']).square()),
        'upright': gravity3[:, :2].square().sum(-1),
        'lateral_velocity': v[:, 1].square(), 'vertical_velocity': v[:, 2].square(),
        'action_rate': (actions6 - previous_actions6).square().sum(-1),
        'effort': (torques6 / _like(caps, torques6)).square().sum(-1),
        'knee_soft_limit': soft,
        # Nonzero yaw never disables zero-translation suppression.
        'zero_command_translation': (commands3[:, 0].abs() < r['zero_vx_threshold']).to(v.dtype) * v[:, :2].square().sum(-1),
    }
    return {name: value * r['weights'][name] * contract['timing']['policy_dt'] for name, value in raw.items()}
