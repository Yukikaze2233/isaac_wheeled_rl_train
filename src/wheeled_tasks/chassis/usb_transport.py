"""Sub-physics-step USB RTT randomization for PC feedback and torque commands.

Past sensor values are linearly reconstructed; torque application preserves the
piecewise-constant command impulse within each physics step. Both are explicit
small-delay approximations, not a high-rate motor-bus or packet-loss simulator.
"""
import math

import torch


class SubstepUsbTransport:
    """Share one RTT budget across feedback and command transport, in seconds."""

    SEED_XOR = 0x555342

    def __init__(self, count, device, dt, config, seed):
        self.dt, self.cfg = float(dt), config
        profile = config["profile"]
        if profile["measurement_kind"] != "loopback_round_trip_time" or profile["units"] != "seconds":
            raise ValueError("USB transport requires an RTT profile in seconds")
        if profile["deployment_usb_mode"] != "HS":
            raise ValueError("This transport contract is calibrated for USB-HS")
        distribution = profile["distribution"]
        if distribution["kind"] != "piecewise_linear_quantile_surrogate":
            raise ValueError("Unsupported RTT prior")
        probabilities = distribution["probabilities"]
        knots = distribution["round_trip_time_s_by_size"][str(config["payload_size"])]
        if (len(probabilities) != len(knots) or len(knots) < 2
                or probabilities[0] != 0 or probabilities[-1] != 1
                or not all(math.isfinite(x) for x in [*probabilities, *knots])
                or any(a >= b for a, b in zip(probabilities, probabilities[1:]))
                or any(a > b for a, b in zip(knots, knots[1:])) or knots[0] < 0):
            raise ValueError("Invalid RTT quantiles")
        self.scale = float(config.get("rtt_scale", 1.))
        self.fraction = float(config.get("enabled_fraction", .5))
        self.split = tuple(config.get("uplink_fraction_range", [.25, .75]))
        maximum = knots[-1] * self.scale
        if (not math.isfinite(self.dt) or self.dt <= 0 or not math.isfinite(self.scale) or self.scale < 0
                or not 0 <= self.fraction <= 1 or len(self.split) != 2
                or not 0 <= self.split[0] <= self.split[1] <= 1 or maximum >= self.dt):
            raise ValueError("USB RTT must be finite, nonnegative and shorter than one physics step")
        self.probabilities = torch.tensor(probabilities, device=device)
        self.knots = torch.tensor(knots, device=device)
        self.generator = torch.Generator(device=device)
        self.reset_generator = torch.Generator(device=device)
        self.reseed(seed)
        self.current = torch.zeros(count, 18, device=device)
        self.previous = torch.zeros_like(self.current)
        self.feedback = torch.zeros_like(self.current)
        self.previous_torque = torch.zeros(count, 6, device=device)
        self.enabled = torch.zeros(count, dtype=torch.bool, device=device)
        self.split_fraction = torch.full((count,), .5, device=device)
        self.rtt = torch.zeros(count, device=device)
        self.uplink = torch.zeros_like(self.rtt)
        self.downlink = torch.zeros_like(self.rtt)
        self.rows = torch.arange(count, device=device)

    def inverse_cdf(self, probability):
        upper = torch.searchsorted(self.probabilities, probability.contiguous(), right=True).clamp(1, len(self.knots) - 1)
        lower = upper - 1
        weight = (probability - self.probabilities[lower]) / (self.probabilities[upper] - self.probabilities[lower])
        return (self.knots[lower] + weight * (self.knots[upper] - self.knots[lower])) * self.scale

    def reseed(self, seed):
        self.generator.manual_seed(seed ^ self.SEED_XOR)
        self.reset_generator.manual_seed(seed ^ self.SEED_XOR ^ 0xFFFF)

    def _sample(self, ids, generator=None):
        probability = torch.rand(len(ids), generator=generator or self.generator, device=self.current.device)
        self.rtt[ids] = self.inverse_cdf(probability) * self.enabled[ids]
        self.uplink[ids] = self.rtt[ids] * self.split_fraction[ids]
        self.downlink[ids] = self.rtt[ids] - self.uplink[ids]

    def prime(self, ids, sensors):
        """Seed post-reset feedback without advancing time or redrawing delays."""
        self.current[ids] = sensors[ids]
        self.previous[ids] = sensors[ids]
        self.feedback[ids] = sensors[ids]
        self.previous_torque[ids] = 0.

    def reset(self, ids, sensors):
        self.prime(ids, sensors)
        self.enabled[ids] = torch.rand(len(ids), generator=self.reset_generator, device=self.current.device) < self.fraction
        self.split_fraction[ids] = self.split[0] + (self.split[1] - self.split[0]) * torch.rand(
            len(ids), generator=self.reset_generator, device=self.current.device)
        self._sample(ids, self.reset_generator)

    def set_enabled(self, ids, enabled):
        self.enabled[ids] = enabled
        self._sample(ids, self.reset_generator)
        self._reconstruct()

    def _reconstruct(self):
        delta = self.current - self.previous
        # Wrapped motor coordinates must not interpolate across a 2*pi jump.
        delta[:, :6] = torch.atan2(delta[:, :6].sin(), delta[:, :6].cos())
        delayed = self.current - (self.uplink / self.dt)[:, None] * delta
        gravity = delayed[:, 15:18]
        delayed[:, 15:18] = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self.feedback.copy_(torch.where((self.uplink > 0)[:, None], delayed, self.current))

    def observe(self, sensors):
        """Advance exactly once after each physics step; jitter varies with time."""
        self.previous.copy_(self.current)
        self.current.copy_(sensors)
        self._sample(self.rows)
        self._reconstruct()

    def apply_torque(self, command):
        """Average old/new torque impulses, retaining the last sent command."""
        fraction = (self.downlink / self.dt)[:, None]
        applied = command + fraction * (self.previous_torque - command)
        self.previous_torque.copy_(command)
        return applied

    def metrics(self):
        return {"/transport/usb_rtt_mean_s": self.rtt.mean(),
                "/transport/usb_rtt_max_s": self.rtt.max(),
                "/transport/usb_enabled_rtt_mean_us": self.rtt.sum() * 1e6 / self.enabled.sum().clamp_min(1),
                "/transport/usb_rtt_max_us": self.rtt.max() * 1e6,
                "/transport/usb_feedback_age_mean_s": self.uplink.mean(),
                "/transport/usb_command_delay_mean_s": self.downlink.mean(),
                "/transport/usb_enabled_fraction": self.enabled.float().mean()}
