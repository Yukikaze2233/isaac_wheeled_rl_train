"""Tensor observation buffers and output-side motor torque envelopes."""

import torch


def _matrix(value, width, name, batch=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width or not value.is_floating_point():
        raise ValueError(f'{name} must be floating Tensor[N,{width}]')
    if batch is not None and value.shape[0] != batch:
        raise ValueError(f'{name} batch mismatch')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} contains nonfinite values')
    return value


def _like(values, reference):
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


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


def motor_torque_limit(joint_speed, wheel_config):
    """Apply an explicitly wheel-side capacity cap, or convert a motor curve once.

    No extrapolation of positive torque beyond the supplied speed domain.
    The default curve is explicitly an unverified linear research prior.
    """
    if not isinstance(joint_speed, torch.Tensor) or not joint_speed.is_floating_point() or not torch.isfinite(joint_speed).all():
        raise ValueError('joint_speed must be a finite floating tensor')
    ratio = wheel_config['gear_ratio']
    eta = wheel_config['gearbox_efficiency']
    if wheel_config.get('curve_side') == 'wheel_output':
        # Engineering capacity probe only: a software cap is already wheel-side.
        # The ratio remains provenance and must not multiply the cap a second time.
        cap = wheel_config['effort_limit']
        if (wheel_config.get('curve_kind') != 'software_cap_only_not_identified'
                or wheel_config.get('scope') not in ('engineering_capacity_only', 'simulation_research_prior')
                or wheel_config.get('torque_reference') != 'wheel_output'
                or any(key in wheel_config for key in ('motor_torque_nm', 'motor_speed_rad_s'))
                or not isinstance(cap, (float, int)) or isinstance(cap, bool)
                or not 0 < cap < float('inf')):
            raise ValueError('Ambiguous or invalid engineering wheel-output torque cap')
        return torch.full_like(joint_speed, cap)
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
