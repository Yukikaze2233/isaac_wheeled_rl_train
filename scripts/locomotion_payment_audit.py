"""Observe real environment reward calls during engineering probes only."""
from copy import deepcopy

import torch


class LocomotionPaymentAudit:
    """Check live inputs, returned components and dt-scaled logger payments.

    The wrappers call the unchanged production functions exactly once. No input,
    reward or environment state is replaced; this is never installed in PPO.
    """

    NAMES = ("stand_still_lin_vel", "locomotion_pitch_floor", "locomotion_roll_floor")

    def __init__(self, config):
        self.quality = {name: deepcopy(config["dense_tracking"][name])
            for name in ("stand_still_lin_vel", "locomotion_posture")}
        self.pending = {}
        self.steps = 0
        self.counts = {name: 0 for name in self.NAMES}
        self.sum_density = {name: 0. for name in self.NAMES}
        self.max_error = 0.
        self.module = None

    def install(self):
        from wheeled_tasks.chassis import rewards
        self.module = rewards
        self.original_reference = rewards.reference_motion_terms
        self.original_dense = rewards.apply_dense_tracking
        rewards.reference_motion_terms = self.reference
        rewards.apply_dense_tracking = self.dense

    def restore(self):
        if self.module is not None:
            self.module.reference_motion_terms = self.original_reference
            self.module.apply_dense_tracking = self.original_dense
            self.module = None

    def reference(self, velocity, gravity, commands, support, ordinary, reference, settings, *, locomotion_posture=None):
        if locomotion_posture != self.quality["locomotion_posture"]:
            raise ValueError("Live reference reward did not receive the frozen posture floor")
        terms = self.original_reference(velocity, gravity, commands, support, ordinary,
            reference, settings, locomotion_posture=locomotion_posture)
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        target = ((reference.terrain_mode != 0) & (reference.elapsed < 4.)) * commands[:, 0].sign() * settings.get("step_pitch_rad", .08)
        error = torch.sin(pitch - target)
        scale_squared = (2 * torch.exp(-commands[:, 0].square() / 3.) + 2).square()
        mask = ordinary.to(velocity.dtype) * ~reference.jumping * (reference.terrain_mode == 0)
        self.pending["locomotion_pitch_floor"] = -2 * mask * error.square() * (locomotion_posture["pitch_scale_floor"]**2 - support * scale_squared).clamp_min(0.)
        self.pending["locomotion_roll_floor"] = -2 * mask * gravity[:, 1].square() * (locomotion_posture["roll_scale_floor"]**2 - scale_squared).clamp_min(0.)
        return terms

    def dense(self, terms, velocity, omega, gravity, height, commands, support, ordinary, reference, settings, **kwargs):
        cost = settings.get("stand_still_lin_vel")
        if cost != self.quality["stand_still_lin_vel"]:
            raise ValueError("Live dense reward did not receive the frozen stationary cost")
        self.original_dense(terms, velocity, omega, gravity, height, commands, support,
            ordinary, reference, settings, **kwargs)
        active = (commands[:, 0].abs() < cost["command_threshold_m_s"]) & ~reference.jumping & ordinary.bool()
        self.pending["stand_still_lin_vel"] = cost["weight"] * velocity[:, :2].abs().sum(-1) * active

    def observe(self, extras, dt):
        if set(self.pending) != set(self.NAMES):
            raise ValueError("An actual environment reward path bypassed the quality calls")
        for name in self.NAMES:
            expected = self.pending[name]
            actual = extras["native_reward_components"][name]
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(extras["log"]["/reward/" + name], expected.mean() * dt,
                rtol=1e-6, atol=1e-7)
            if not torch.isfinite(actual).all() or (actual > 1e-7).any():
                raise ValueError("Quality cost is nonfinite or paid a positive bonus")
            self.max_error = max(self.max_error, float((actual - expected).abs().max()))
            self.counts[name] += int((actual < -1e-9).sum())
            self.sum_density[name] += float(actual.sum())
        self.pending.clear()
        self.steps += 1

    def report(self):
        return dict(status="actual_calls_components_and_dt_scaled_payments_verified", steps=self.steps,
            frozen_settings=self.quality, negative_lane_frames=self.counts,
            sum_reward_density=self.sum_density, maximum_component_error=self.max_error,
            scope="engineering_reward_wiring_only_not_policy_improvement")
