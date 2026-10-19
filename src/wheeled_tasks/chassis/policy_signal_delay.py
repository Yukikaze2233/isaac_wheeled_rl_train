"""Additional policy-signal lag without delaying the host PD feedback path."""
from copy import deepcopy
import math


KEY = "policy_signal_delay"
SETTINGS = {
    "version": "policy-signal-lag-v1", "enabled": True,
    "clock": "physics_steps", "buffer_physics_samples": 10,
    "extra_episode_probability": .5, "observation_steps": [1, 2, 3],
    "target_steps": [1, 2], "jitter_s": 0., "resample": "episode_reset",
    "observation_application": "after_local_feedback_before_noise",
    "target_application": "before_PD_local_feedback_unchanged",
    "critic_action_age": "target_sample_through_downlink_in_physics_steps",
}


def validate_policy_signal_delay(contract):
    """Select an explicit extra robustness prior, separate from local RTT."""
    if KEY not in contract:
        return None
    value = contract[KEY]
    if not isinstance(value, dict) or set(value) != set(SETTINGS):
        raise ValueError("Policy signal delay requires complete versioned settings")
    for key, expected in SETTINGS.items():
        actual = value[key]
        if key == "extra_episode_probability":
            if type(actual) not in (int, float) or not math.isfinite(actual) or not 0 <= actual <= 1:
                raise ValueError("Extra policy-delay episode probability must be in [0,1]")
        elif (type(actual) is not type(expected) or actual != expected
                or isinstance(expected, list) and any(type(item) is not int for item in actual)):
            raise ValueError(f"Policy signal delay has unsupported {key}")
    if (contract.get("physics_dt") != .005 or contract.get("policy_dt") != .02
            or contract.get("actor_dim") != 35 or contract.get("critic_dim") != 78
            or contract.get("history_length") != 1):
        raise ValueError("Policy signal delay requires current-frame 35D/78D and the 200/50 Hz clocks")
    from .actor_sensor_noise import validate_actor_sensor_noise
    from .pc_communication import validate_pc_communication
    if (validate_actor_sensor_noise(contract) is None
            or "round_trip_transport" not in contract or validate_pc_communication(contract) is None):
        raise ValueError("Extra policy delay requires separate physical noise and shared local communication")
    return deepcopy(value)


class PolicySignalDelay:
    """Cache sensors and decoded targets on the physical clock.

    Four sensor groups are position, velocity, gyro and gravity. Target groups
    are legs and wheels. Each group has its own per-episode integer lag; one
    episode draw selects nominal or extra-lag coverage for all six groups.
    Source timestamps travel with each selected sample. Held policy targets
    retain their original 50 Hz generation time, independently of 200 Hz PD.
    """

    def __init__(self, count, device, contract, generator):
        import torch
        from .delay_buffer import GroupedDelayBuffer

        self.cfg = validate_policy_signal_delay(contract)
        if self.cfg is None or type(count) is not int or count <= 0:
            raise ValueError("Policy signal delay requires explicit selection and a positive batch")
        self.dt, self.physics_steps = float(contract["physics_dt"]), 0
        self.generator = generator
        self.feedback = torch.zeros(count, 18, device=device)
        if not isinstance(generator, torch.Generator) or generator.device != self.feedback.device:
            raise ValueError("Policy signal delay must share the environment RNG device")
        self.rows = torch.arange(count, device=device)
        size = self.cfg["buffer_physics_samples"]
        self.sensor_buffer = GroupedDelayBuffer(size, count, device, [0] * 6 + [1] * 6 + [2] * 3 + [3] * 3)
        self.sensor_time_buffer = GroupedDelayBuffer(size, count, device, [0, 1, 2, 3])
        self.target_buffer = GroupedDelayBuffer(size, count, device, [0] * 4 + [1] * 2)
        self.target_time_buffer = GroupedDelayBuffer(size, count, device, [0, 1, 0, 1])
        self.enabled = torch.zeros(count, dtype=torch.bool, device=device)
        self.observation_lags = torch.zeros(count, 4, dtype=torch.long, device=device)
        self.target_lags = torch.zeros(count, 2, dtype=torch.long, device=device)
        self.sensor_timestamp_s = torch.zeros(count, 4, dtype=torch.float64, device=device)
        self.observation_age_s = torch.zeros_like(self.sensor_timestamp_s)
        self.target_sample_s = torch.zeros(count, 2, dtype=torch.float64, device=device)
        self.target_policy_generated_s = torch.zeros_like(self.target_sample_s)
        self.selected_targets = torch.zeros(count, 6, device=device)
        self._target_tick = None

    def _sensors(self, sensors, timestamp_s):
        import torch

        if (sensors.shape != self.feedback.shape or sensors.dtype != torch.float32
                or sensors.device != self.feedback.device):
            raise ValueError("Policy delay needs float32 [environment,18] delivered sensors")
        if (timestamp_s.shape != (len(self.rows),) or timestamp_s.dtype != torch.float64
                or timestamp_s.device != sensors.device):
            raise ValueError("Policy delay needs the original local-feedback source timestamp")
        return timestamp_s[:, None].expand(-1, 4)

    def reset(self, ids, sensors, timestamp_s):
        """Redraw selected episodes and flush both values and source clocks."""
        import torch

        stamps = self._sensors(sensors, timestamp_s)
        if not len(ids):
            return
        self.enabled[ids] = torch.rand(len(ids), device=self.feedback.device,
            generator=self.generator) < self.cfg["extra_episode_probability"]
        self.observation_lags[ids] = torch.randint(1, 4, (len(ids), 4),
            device=self.feedback.device, generator=self.generator) * self.enabled[ids, None]
        self.target_lags[ids] = torch.randint(1, 3, (len(ids), 2),
            device=self.feedback.device, generator=self.generator) * self.enabled[ids, None]
        self.sensor_buffer.prime(sensors, ids)
        self.sensor_time_buffer.prime(stamps, ids)
        self.feedback[ids] = sensors[ids]
        self.sensor_timestamp_s[ids] = stamps[ids]
        self.observation_age_s[ids] = self.physics_steps * self.dt - stamps[ids]
        self.target_buffer.reset(ids)
        self.target_time_buffer.reset(ids)
        self.selected_targets[ids] = 0.
        self.target_sample_s[ids] = self.physics_steps * self.dt
        self.target_policy_generated_s[ids] = self.physics_steps * self.dt

    def observe(self, sensors, timestamp_s, physical_tick):
        """Append once after local feedback has delivered the new physics state."""
        if type(physical_tick) is not int or physical_tick != self.physics_steps + 1:
            raise ValueError("Policy sensor history must advance exactly once per physical step")
        stamps = self._sensors(sensors, timestamp_s)
        self.sensor_buffer.append(sensors)
        self.sensor_time_buffer.append(stamps)
        self.physics_steps = physical_tick
        self.feedback.copy_(self.sensor_buffer.read_groups(self.observation_lags))
        self.sensor_timestamp_s.copy_(self.sensor_time_buffer.read_groups(self.observation_lags))
        self.observation_age_s.copy_(physical_tick * self.dt - self.sensor_timestamp_s)

    def targets(self, legs, wheels, policy_generated_s, physical_tick):
        """Delay decoded proposals; leave actual PD feedback and clock untouched."""
        import torch

        if type(physical_tick) is not int or physical_tick != self.physics_steps or self._target_tick == physical_tick:
            raise ValueError("Policy target history must append once at the current physical boundary")
        count = len(self.rows)
        if (legs.shape != (count, 4) or wheels.shape != (count, 2)
                or legs.dtype != self.feedback.dtype or wheels.dtype != self.feedback.dtype
                or legs.device != self.feedback.device or wheels.device != self.feedback.device
                or policy_generated_s.shape != (count,) or policy_generated_s.dtype != torch.float64
                or policy_generated_s.device != self.feedback.device):
            raise ValueError("Policy delay requires decoded leg/wheel targets and original generation time")
        self.target_buffer.append(torch.cat((legs, wheels), -1))
        now = physical_tick * self.dt
        clock = torch.stack((policy_generated_s.new_full((count,), now),
            policy_generated_s.new_full((count,), now), policy_generated_s, policy_generated_s), -1)
        self.target_time_buffer.append(clock)
        self.selected_targets.copy_(self.target_buffer.read_groups(self.target_lags))
        stamps = self.target_time_buffer.read_groups(self.target_lags)
        self.target_sample_s.copy_(stamps[:, :2])
        self.target_policy_generated_s.copy_(stamps[:, 2:])
        self._target_tick = physical_tick
        return self.selected_targets[:, :4], self.selected_targets[:, 4:]

    def metrics(self):
        return {"/policy_delay/extra_episode_fraction": self.enabled.float().mean(),
            "/policy_delay/requested_observation_lag_max_s": self.observation_lags.amax() * self.dt,
            "/policy_delay/requested_target_lag_max_s": self.target_lags.amax() * self.dt,
            "/policy_delay/actual_observation_age_mean_s": self.observation_age_s.mean(),
            "/policy_delay/actual_observation_age_max_s": self.observation_age_s.amax(),
            "/policy_delay/selected_target_age_at_interval_end_mean_s": (self.physics_steps * self.dt - self.target_sample_s).mean()}

    def readback(self):
        return {"settings": deepcopy(self.cfg), "physical_tick": self.physics_steps,
            "extra_episode_enabled": self.enabled.clone(),
            "observation_lags_q_dq_gyro_gravity": self.observation_lags.clone(),
            "target_lags_legs_wheels": self.target_lags.clone(),
            "target_history_length": self.target_buffer.current_length.clone(),
            "sensor_output18": self.feedback.clone(), "sensor_timestamp_s": self.sensor_timestamp_s.clone(),
            "actual_observation_age_s": self.observation_age_s.clone(),
            "selected_targets_legs_then_wheels": self.selected_targets.clone(),
            "selected_target_sample_s": self.target_sample_s.clone(),
            "selected_target_policy_generated_s": self.target_policy_generated_s.clone(),
            "target_scope": "RL proposals before script or zero-effort overrides; those intervals remain excluded from PPO"}
