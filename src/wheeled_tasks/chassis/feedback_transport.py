"""Fractional sensor-to-host feedback age on the physical simulation clock.

The complete one-way delay is an engineering prior, not a measured USB RTT.
Linear history reconstruction approximates sub-step sensor sampling; it does
not simulate acquisition, motor-bus events or packet loss. PC PD and the actor
consume the same cached feedback tensor, updated once per physics observation.
"""
import math

import torch

from .feedback_transport_config import validate_feedback_transport


def shared_activation_draw(count, device, seed, domain_draw=None):
    """Borrow an existing draw, or sample paired path activation with owned RNG.

    A dedicated stream leaves global RNGs and both transports' timing streams
    unchanged. The caller passes the returned per-lane draw to both paths.
    """
    if domain_draw is not None:
        return domain_draw
    generator = torch.Generator(device=device).manual_seed(seed ^ 0x464143)
    return torch.rand(count, device=device, generator=generator)


class FractionalFeedbackTransport:
    """Reconstruct chronological feedback using Isaac Lab's native history.

    Sensors are [six motor positions, six velocities, gyro xyz, gravity xyz].
    All eighteen channels share one sensor timestamp. Position interpolation
    follows the shortest angular arc; delayed gravity is normalized. Timing
    tensors use float64 to retain sub-ms precision over long training runs.
    """

    SEED_XOR = 0x464442

    def __init__(self, count, device, dt, config, seed, *, domain_draw=None):
        from .delay_buffer import GroupedDelayBuffer

        self.cfg = validate_feedback_transport(config, dt)
        self.dt, self.physics_steps = float(dt), 0
        self.low, self.high = self.cfg["delay_range_s"]
        self.fraction = self.cfg["enabled_fraction"] if self.cfg["enabled"] else 0.
        self.domain_draw = domain_draw
        if domain_draw is not None and domain_draw.shape != (count,):
            raise ValueError("Feedback activation needs one domain draw per environment")
        self.rows = torch.arange(count, device=device)
        self.history = GroupedDelayBuffer(math.ceil(self.high / self.dt) + 2, count, device, [0] * 18)
        self.feedback = torch.zeros(count, 18, device=device)
        self.enabled = torch.zeros(count, dtype=torch.bool, device=device)
        self.base_delay_s = torch.zeros(count, dtype=torch.float64, device=device)
        self.delay_s = torch.zeros_like(self.base_delay_s)
        self.sensor_timestamp_s = torch.zeros_like(self.base_delay_s)
        self.age_s = torch.zeros_like(self.base_delay_s)
        self.generator = torch.Generator(device=device)
        self.reset_generator = torch.Generator(device=device)
        self.reseed(seed)

    def reseed(self, seed):
        """Restart owned random streams without modifying other RNGs or state."""
        self.generator.manual_seed(seed ^ self.SEED_XOR)
        self.reset_generator.manual_seed(seed ^ self.SEED_XOR ^ 0xFFFF)

    def _check_sensors(self, sensors):
        if sensors.shape != self.feedback.shape or not sensors.is_floating_point():
            raise ValueError("Feedback transport needs a floating [environment, 18] sensor tensor")

    def prime(self, ids, sensors):
        """Flush only selected episode histories without advancing time or RNGs."""
        self._check_sensors(sensors)
        if not len(ids):
            return
        self.history.prime(sensors, ids)
        self.feedback[ids] = sensors[ids]
        self.sensor_timestamp_s[ids] = self.physics_steps * self.dt
        self.age_s[ids] = 0.

    def reset(self, ids, sensors):
        """Seed new episode feedback and redraw selected environments' prior."""
        self.prime(ids, sensors)
        count = len(ids)
        if not count:
            return
        if self.domain_draw is None:
            draw = torch.rand(count, generator=self.reset_generator, device=self.feedback.device)
        else:
            draw = self.domain_draw[ids]
        self.enabled[ids] = (draw < self.fraction) & self.cfg["enabled"]
        unit = torch.rand(count, generator=self.reset_generator, device=self.feedback.device, dtype=torch.float64)
        self.base_delay_s[ids] = self.low + (self.high - self.low) * unit
        self.delay_s[ids] = self.base_delay_s[ids] * self.enabled[ids]

    def set_enabled(self, ids, enabled):
        """Change selected activation while retaining chronological feedback."""
        self.enabled[ids] = enabled if self.cfg["enabled"] else False
        self.delay_s[ids] = self.base_delay_s[ids] * self.enabled[ids]
        if self.history._buffer is not None:
            self._reconstruct(ids)

    def _reconstruct(self, ids):
        now = self.physics_steps * self.dt
        available_s = (self.history.current_length - 1).clamp_min(0).to(self.age_s.dtype) * self.dt
        timestamp = torch.maximum(now - self.delay_s, self.sensor_timestamp_s)
        timestamp = torch.maximum(timestamp, now - available_s).clamp_max(now)
        age = (now - timestamp).clamp_min(0)
        delayed = self.history.read_groups_fractional((age / self.dt)[:, None], angular_columns=list(range(6)))
        gravity = delayed[:, 15:18]
        norm = gravity.norm(dim=-1, keepdim=True)
        previous_gravity = self.feedback[:, 15:18]
        fallback = previous_gravity / previous_gravity.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        # Antipodal endpoints have no unique linear direction. Keep the last
        # delivered direction instead of publishing a zero gravity vector.
        normalized = torch.where(norm > 1e-9, gravity / norm.clamp_min(1e-9), fallback)
        delayed[:, 15:18] = torch.where((age > 0)[:, None], normalized, gravity)
        self.feedback[ids] = delayed[ids]
        self.sensor_timestamp_s[ids] = timestamp[ids]
        self.age_s[ids] = age[ids]

    def observe(self, sensors):
        """Append one new physics sample and publish one shared feedback tensor."""
        self._check_sensors(sensors)
        self.history.append(sensors)
        self.physics_steps += 1
        innovation = (2 * torch.rand(len(self.rows), device=self.feedback.device,
            generator=self.generator, dtype=torch.float64) - 1) * self.cfg["jitter_s"]
        correlation = self.cfg["jitter_correlation"]
        self.delay_s.copy_((correlation * self.delay_s + (1 - correlation) * self.base_delay_s
            + innovation).clamp(self.low, self.high) * self.enabled)
        self._reconstruct(self.rows)

    def metrics(self):
        """Stable TensorBoard scalar tags; all timestamp and age units are seconds."""
        return {
            "/transport/feedback_enabled_fraction": self.enabled.float().mean(),
            "/transport/feedback_delay_mean_s": self.delay_s.mean(),
            "/transport/feedback_delay_max_s": self.delay_s.max(),
            "/transport/feedback_sensor_timestamp_mean_s": self.sensor_timestamp_s.mean(),
            "/transport/feedback_sensor_timestamp_min_s": self.sensor_timestamp_s.min(),
            "/transport/feedback_age_mean_s": self.age_s.mean(),
            "/transport/feedback_age_max_s": self.age_s.max(),
        }
