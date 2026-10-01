"""Physical-clock sensor age and setpoint latency using Isaac Lab native buffers."""
from copy import deepcopy

import torch


def validate_signal_delay(settings, physics_dt):
    if not isinstance(settings, dict) or settings.get("version") != "physical_signal_delay_v1":
        raise ValueError("Signal delay requires explicit physical_signal_delay_v1 settings")
    if type(settings.get("enabled")) is not bool or physics_dt != .005:
        raise ValueError("Signal delay requires an explicit enable and the 5 ms physical clock")
    if (settings.get("clock") != "physics_steps" or settings.get("jitter_clock") != "policy_steps"
            or settings.get("feedback_pd") != "fresh_physics_feedback"
            or settings.get("enabled_fraction") != .75 or settings.get("jitter_ms") != 5.):
        raise ValueError("Signal delay clock, clean population or feedback scope differs")
    profiles = settings.get("profiles", [])
    expected = ((.60, [0., 20.], [0., 15.]), (.35, [20., 40.], [15., 30.]),
                (.05, [40., 80.], [30., 60.]))
    if len(profiles) != len(expected):
        raise ValueError("Signal delay requires common, wide and engineering-tail populations")
    for profile, (probability, obs, act) in zip(profiles, expected, strict=True):
        if profile != dict(probability=probability, observation_ms=obs, action_ms=act):
            raise ValueError("Signal delay profile differs from its reviewed coverage")
    schedule = settings.get("schedule", {})
    if (schedule.get("clock") != "domain_actor_updates" or type(schedule.get("start")) is not int
            or type(schedule.get("end")) is not int or not 0 <= schedule["start"] < schedule["end"]):
        raise ValueError("Signal delay needs an explicit monotone domain-actor clock")
    return deepcopy(settings)


class PhysicalSignalDelay:
    """Delay policy sensors and decoded targets; leave local PD feedback current.

    Four sensor and two setpoint groups have independent ages. Native circular
    buffers advance only with physical time, never on repeated observation reads.
    Fresh commands/context/history and the privileged critic are not delayed.
    """
    def __init__(self, count, device, settings, physics_dt, generator):
        from isaaclab.utils.buffers import CircularBuffer
        self.cfg = validate_signal_delay(settings, physics_dt)
        self.dt_ms, self.generator = physics_dt * 1000., generator
        self.sensor_buffer = CircularBuffer(17, count, device)
        self.target_buffer = CircularBuffer(13, count, device)
        self.enabled = torch.zeros(count, dtype=torch.bool, device=device)
        self.profile = torch.zeros(count, dtype=torch.long, device=device)
        self.base_ms = torch.zeros(count, 6, device=device)
        self.bounds_ms = torch.zeros(count, 6, 2, device=device)
        self.lags = torch.zeros(count, 6, dtype=torch.long, device=device)
        self.feedback = torch.zeros(count, 18, device=device)
        self.probability_edges = torch.tensor([.60, .95], device=device)
        self.ranges = torch.tensor([[p["observation_ms"]] * 4 + [p["action_ms"]] * 2
                                   for p in self.cfg["profiles"]], device=device)
        self.strength, self.physics_steps = 0., 0

    def reset(self, ids, sensors):
        device = self.enabled.device
        self.sensor_buffer.reset(ids)
        self.target_buffer.reset(ids)
        self.enabled[ids] = torch.rand(len(ids), device=device, generator=self.generator) < .75
        draws = torch.rand(len(ids), device=device, generator=self.generator)
        self.profile[ids] = torch.bucketize(draws, self.probability_edges)
        ranges = self.ranges[self.profile[ids]]
        self.bounds_ms[ids] = ranges
        unit = torch.rand(len(ids), 6, device=device, generator=self.generator)
        self.base_ms[ids] = ranges[:, :, 0] + unit * (ranges[:, :, 1] - ranges[:, :, 0])
        self.lags[ids] = 0
        self.feedback[ids] = sensors[ids]

    def begin_policy(self, actor_update):
        schedule = self.cfg["schedule"]
        self.strength = min(1., max(0., (actor_update - schedule["start"]) / (schedule["end"] - schedule["start"])))
        jitter = (2. * torch.rand(self.base_ms.shape, device=self.enabled.device,
                                 generator=self.generator) - 1.) * self.cfg["jitter_ms"]
        ms = (self.base_ms + jitter).clamp(min=self.bounds_ms[:, :, 0], max=self.bounds_ms[:, :, 1])
        self.lags.copy_((ms * self.strength / self.dt_ms).round().long() * self.enabled[:, None])

    def observe(self, sensors):
        self.sensor_buffer.append(sensors)
        for column, span in enumerate((slice(0, 6), slice(6, 12), slice(12, 15), slice(15, 18))):
            self.feedback[:, span] = self.sensor_buffer[self.lags[:, column]][:, span]
        self.physics_steps += 1

    def targets(self, legs, wheels):
        self.target_buffer.append(torch.cat((legs, wheels), -1))
        delayed_legs = self.target_buffer[self.lags[:, 4]][:, :4]
        delayed_wheels = self.target_buffer[self.lags[:, 5]][:, 4:]
        # Native reset primes each cleared environment with its first new
        # sample, so no old episode target can leak through a delayed read.
        return delayed_legs, delayed_wheels

    def metrics(self):
        values = {"/delay/strength": self.strength,
                  "/delay/enabled_fraction": self.enabled.float().mean(),
                  "/delay/wide_fraction": ((self.profile == 1) & self.enabled).float().mean(),
                  "/delay/stress_fraction": ((self.profile == 2) & self.enabled).float().mean()}
        for column, name in enumerate(("joint_position", "joint_velocity", "gyro", "gravity", "leg_target", "wheel_target")):
            age = self.lags[:, column].float() * self.dt_ms
            values[f"/delay/{name}_mean_ms"] = age.mean()
            values[f"/delay/{name}_max_ms"] = age.max()
        return values
