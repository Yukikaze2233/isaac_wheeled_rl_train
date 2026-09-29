"""Training-side, sensor-driven recovery references; no simulation state writes.

Routes adapt the ActivationBench fold/plant, sagittal orbit/thrust and side
swing experiments. Geometry-derived V6 poses are supplied by the caller, not
copied from V5. Call at 200 Hz; the caller evaluates PD at 1 kHz and blends
script/RL *torques* using ``blend``, never interpolates policy actions instead.
"""
from __future__ import annotations

from enum import IntEnum
import math

import torch


class RecoveryPhase(IntEnum):
    SELECT = 0
    FOLD = 1
    PLANT = 2
    ORBIT = 3
    THRUST = 4
    SIDE = 5
    CAPTURE = 6
    PREPARE = 7
    BLEND = 8
    RL = 9
    FAILED = 10


class RecoveryRoute(IntEnum):
    UPRIGHT = 0
    PLANT = 1
    ORBIT_POSITIVE = 2
    ORBIT_NEGATIVE = 3
    SIDE = 4


class RecoveryFSM:
    """Batched recovery using CONTROL order LH, LA, LW, RH, RA, RW.

    ``support`` and ``body_clear`` are deployable observer decisions, not true
    simulation contacts. ``estimated_height`` is the observer's supported-height
    estimate. Script targets contain four continuous motor angles and two wheel
    speeds (zero). ``enabled`` retains episode membership; inactive environments
    are pure RL with no command hold. FAILED is terminal until explicit reset.
    """
    LEG_IDS = (0, 1, 3, 4)
    WHEEL_IDS = (2, 5)
    DEFAULTS = {
        "dt": .005, "blend_seconds": .2, "stable_seconds": 1., "max_script_seconds": 8.,
        "prepare_speed_rad_s": 1., "push_speed_rad_s": 4., "orbit_speed_rad_s": 4.,
        "capture_speed_rad_s": 4., "side_speed_rad_s": 6., "side_angle_rad": math.pi / 2,
        "plant_angle_rad": math.radians(20.), "orbit_turns": 1.,
        "handover_height_min": .27, "handover_height_max": .36,
        "handover_tilt_rad": math.radians(8.), "handover_gyro_max": .75,
        "reroute_tilt_rad": math.radians(70.), "reroute_support_lost_seconds": .15,
        "max_reroutes": 1,
    }

    def __init__(self, count, device, config, nominal, fold, thrust, support, axis_signs):
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Recovery count must be a positive integer")
        config = dict(config or {})
        unknown = set(config) - self.DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unknown recovery configuration: {sorted(unknown)}")
        self.config = self.DEFAULTS | config
        if any(not math.isfinite(v) or v <= 0. for k, v in self.config.items() if k != "max_reroutes"):
            raise ValueError("Recovery parameters must be positive and finite")
        if type(self.config["max_reroutes"]) is not int or self.config["max_reroutes"] not in (0, 1):
            raise ValueError("Recovery permits at most one unsupported-pose reroute")
        self.dt = self.config["dt"]
        if self.dt != .005 or self.config["blend_seconds"] != .2:
            raise ValueError("Recovery requires 200 Hz references and a 200 ms torque blend")
        if self.config["stable_seconds"] < 1. or self.config["max_script_seconds"] > 8.:
            raise ValueError("Recovery requires >=1 s stable RL and <=8 s script including blend")
        if self.config["handover_height_min"] >= self.config["handover_height_max"]:
            raise ValueError("Invalid recovery handover height interval")
        if not self.config["handover_tilt_rad"] < self.config["reroute_tilt_rad"] < math.pi:
            raise ValueError("Recovery reroute tilt must be outside handover and below inversion")
        self.count, self.device = count, torch.device(device)
        for name, value in (("nominal", nominal), ("fold", fold), ("thrust", thrust), ("support", support)):
            value = torch.as_tensor(value, dtype=torch.float32, device=device).clone()
            if value.shape != (6,) or not torch.isfinite(value).all():
                raise ValueError(f"Recovery {name} must be a finite CONTROL-order six-vector")
            value[list(self.WHEEL_IDS)] = 0.
            setattr(self, name, value[list(self.LEG_IDS)])
        self.axis_signs = torch.as_tensor(axis_signs, dtype=torch.float32, device=device).clone()
        if (self.axis_signs.shape != (4,) or not torch.isfinite(self.axis_signs).all()
                or not self.axis_signs.abs().eq(1.).all()
                or not self.axis_signs[::2].eq(self.axis_signs[1::2]).all()):
            raise ValueError("Recovery root pairs require four +/-1 axis signs, equal within each pair")
        scalar = torch.zeros(count, device=device)
        self.enabled = scalar.bool()
        self.phase = torch.full((count,), int(RecoveryPhase.RL), dtype=torch.long, device=device)
        self.route = torch.zeros_like(self.phase)
        self.age_ticks = torch.zeros_like(self.phase)
        self.phase_ticks = torch.zeros_like(self.phase)
        self.ready_ticks = torch.zeros_like(self.phase)
        self.stable_ticks = torch.zeros_like(self.phase)
        self.plant_ticks = torch.zeros_like(self.phase)
        self.unsupported_ticks = torch.zeros_like(self.phase)
        self.reroute_count = torch.zeros_like(self.phase)
        self.failure_code = torch.zeros_like(self.phase)
        self.motion_released = torch.ones_like(self.enabled)
        self.script_targets = torch.zeros(count, 6, device=device)
        self.continuous_q = torch.zeros_like(self.script_targets)
        self.last_raw_q = torch.zeros_like(self.script_targets)
        self.continuous_pitch = scalar.clone()
        self.last_pitch = scalar.clone()
        self.pitch_initialized = self.enabled.clone()
        self.orbit_angle = scalar.clone()
        self.orbit_start = torch.zeros(count, 4, device=device)
        self.thrust_anchor = torch.zeros_like(self.orbit_start)
        self.plant_pitch = scalar.clone()
        self.side_start = torch.zeros_like(self.orbit_start)
        self.side_direction = torch.zeros_like(self.orbit_start)

    @staticmethod
    def paired_delta(goal, reference):
        """The hip chooses one winding for its entire hip/auxiliary pair."""
        difference = (goal - reference).reshape(-1, 2, 2)
        turns = (difference[..., :1] / math.tau).round()
        return (difference - turns * math.tau).reshape_as(reference)

    def reset(self, ids, q_control, active):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        q = torch.as_tensor(q_control, dtype=self.script_targets.dtype, device=self.device)
        active = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        if q.shape == (self.count, 6):
            q = q[ids]
        if active.shape == (self.count,):
            active = active[ids]
        if active.ndim == 0:
            active = active.expand(len(ids))
        if q.shape != (len(ids), 6) or active.shape != (len(ids),) or not torch.isfinite(q).all():
            raise ValueError("Recovery reset requires finite selected/full CONTROL angles and matching active mask")
        self.enabled[ids] = active
        self.phase[ids] = torch.where(active, int(RecoveryPhase.SELECT), int(RecoveryPhase.RL))
        self.motion_released[ids] = ~active
        self.script_targets[ids] = q
        self.script_targets[ids[:, None], torch.tensor(self.WHEEL_IDS, device=self.device)] = 0.
        self.continuous_q[ids] = q
        self.last_raw_q[ids] = q
        for name in ("route", "age_ticks", "phase_ticks", "ready_ticks", "stable_ticks", "plant_ticks",
                     "unsupported_ticks", "reroute_count", "failure_code", "continuous_pitch",
                     "last_pitch", "orbit_angle", "plant_pitch"):
            getattr(self, name)[ids] = 0
        self.pitch_initialized[ids] = False
        for name in ("orbit_start", "thrust_anchor", "side_start", "side_direction"):
            getattr(self, name)[ids] = 0.

    def _enter(self, mask, phase):
        self.phase[mask] = int(phase)
        self.phase_ticks[mask] = 0

    def _fail(self, mask, code):
        mask = mask & self.enabled & ~self.failed
        self._enter(mask, RecoveryPhase.FAILED)
        self.failure_code[mask] = code
        self.motion_released[mask] = False

    def _ticks(self, seconds):
        return math.ceil(seconds / self.dt - 1e-9)

    def update(self, q_control, dq_control, gravity, gyro, estimated_height, support, body_clear):
        """Advance exactly one 5 ms reference tick; never mutates caller inputs."""
        if (q_control.shape != (self.count, 6) or dq_control.shape != (self.count, 6)
                or gravity.shape != (self.count, 3) or gyro.shape != (self.count, 3)
                or any(value.shape != (self.count,) for value in (estimated_height, support, body_clear))):
            raise ValueError("Recovery update sensor shapes do not match environment count")
        valid = (torch.isfinite(q_control).all(-1) & torch.isfinite(dq_control).all(-1)
                 & torch.isfinite(gravity).all(-1) & torch.isfinite(gyro).all(-1)
                 & torch.isfinite(estimated_height) & (gravity.norm(dim=-1) > .5))
        self._fail(~valid, 1)
        q = torch.where(valid[:, None], q_control, self.last_raw_q)
        delta_q = q - self.last_raw_q
        self.continuous_q += torch.atan2(delta_q.sin(), delta_q.cos())
        self.last_raw_q.copy_(q)
        gravity = torch.nan_to_num(gravity)
        gravity = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.))
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        pitch_delta = pitch - self.last_pitch
        self.continuous_pitch = torch.where(self.pitch_initialized,
            self.continuous_pitch + torch.atan2(pitch_delta.sin(), pitch_delta.cos()), pitch)
        self.last_pitch.copy_(pitch)
        self.pitch_initialized |= valid
        self.age_ticks += self.enabled & ~self.failed & ~self.pure_rl
        self.phase_ticks += self.enabled & ~self.failed
        # A captured pose can fall away before handover. A sustained sensor-only
        # loss of support selects a new route once, without resetting elapsed
        # time, encoder winding, commanded targets or actuator history. Brief
        # contact loss and any state after torque blending never restart script.
        recapturable = (self.phase == int(RecoveryPhase.PREPARE)) | (self.phase == int(RecoveryPhase.CAPTURE))
        unsupported = (self.enabled & valid & recapturable & ~support
                       & (tilt > self.config["reroute_tilt_rad"]))
        self.unsupported_ticks = torch.where(unsupported, self.unsupported_ticks + 1, 0)
        reroute = (unsupported & (self.unsupported_ticks >= self._ticks(self.config["reroute_support_lost_seconds"]))
                   & (self.reroute_count < self.config["max_reroutes"]))
        self.reroute_count += reroute
        self.unsupported_ticks[reroute] = 0
        self._enter(reroute, RecoveryPhase.SELECT)
        selecting = self.phase == int(RecoveryPhase.SELECT)
        near_upright = (tilt < math.radians(15.)) & (estimated_height > .20)
        sagittal_route = torch.where(pitch >= 0., int(RecoveryRoute.ORBIT_POSITIVE), int(RecoveryRoute.ORBIT_NEGATIVE))
        selected = torch.where(tilt < math.radians(70.), int(RecoveryRoute.PLANT), sagittal_route)
        selected = torch.where((tilt >= math.radians(70.)) & (gravity[:, 1].abs() > .7), int(RecoveryRoute.SIDE), selected)
        selected = torch.where(near_upright, int(RecoveryRoute.UPRIGHT), selected)
        self.route[selecting] = selected[selecting]
        self._enter(selecting & near_upright, RecoveryPhase.PREPARE)
        self._enter(selecting & ~near_upright, RecoveryPhase.FOLD)
        phase = self.phase.clone()
        elapsed = self.phase_ticks * self.dt
        folding, planting, orbiting, thrusting, side = (phase == int(value) for value in (
            RecoveryPhase.FOLD, RecoveryPhase.PLANT, RecoveryPhase.ORBIT, RecoveryPhase.THRUST, RecoveryPhase.SIDE))
        preparing = phase == int(RecoveryPhase.PREPARE)
        capturing = phase == int(RecoveryPhase.CAPTURE)
        blending = phase == int(RecoveryPhase.BLEND)
        goal = self.nominal.expand(self.count, -1).clone()
        goal[folding] = self.fold
        aligned = (preparing | capturing | blending) & support & (tilt < math.radians(85.))
        extension = ((tilt - math.radians(12.)) / math.radians(43.)).clamp(0., 1.)
        aligned_goal = self.nominal + extension[:, None] * (self.support - self.nominal) - pitch[:, None] / self.axis_signs
        goal[aligned] = aligned_goal[aligned]
        plant_offset = (-pitch - self.config["plant_angle_rad"] * pitch.sign()).clamp(-2.2, 2.2)
        goal[planting] = self.fold
        sagittal_plant = planting & (gravity[:, 1].abs() < .5)
        goal[sagittal_plant] += plant_offset[sagittal_plant, None] / self.axis_signs
        direction = torch.where(self.route == int(RecoveryRoute.ORBIT_POSITIVE), 1., -1.)
        self.orbit_angle += orbiting * self.config["orbit_speed_rad_s"] * self.dt
        goal[orbiting] = (self.orbit_start + direction[:, None] * self.orbit_angle[:, None] / self.axis_signs)[orbiting]
        goal[thrusting] = (self.thrust_anchor - (self.continuous_pitch - self.plant_pitch)[:, None] / self.axis_signs)[thrusting]
        side_duration = self.config["side_angle_rad"] / self.config["side_speed_rad_s"]
        stroke = torch.where(elapsed <= side_duration, elapsed * self.config["side_speed_rad_s"],
            self.config["side_angle_rad"] - (elapsed - side_duration - .08).clamp_min(0.) * self.config["side_speed_rad_s"])
        stroke = stroke.clamp(0., self.config["side_angle_rad"])
        goal[side] = (self.side_start + self.side_direction * stroke[:, None] / self.axis_signs)[side]
        reference = self.script_targets[:, self.LEG_IDS]
        delta = self.paired_delta(goal, reference)
        directed = orbiting | thrusting | side
        delta[directed] = (goal - reference)[directed]
        speed = elapsed.new_full(elapsed.shape, self.config["prepare_speed_rad_s"])
        speed[preparing | thrusting] = self.config["push_speed_rad_s"]
        speed[capturing] = self.config["capture_speed_rad_s"]
        speed[orbiting] = self.config["orbit_speed_rad_s"]
        speed[side] = self.config["side_speed_rad_s"]
        # One fraction for all four roots preserves the intended coupled path.
        fraction = (speed * self.dt / delta.abs().amax(-1).clamp_min(1e-6)).clamp_max(1.)
        stepping = self.scripted | self.blending
        reference = reference + delta * fraction[:, None] * stepping[:, None]
        self.script_targets[:, self.LEG_IDS] = reference
        remaining = delta.abs().amax(-1) * (1. - fraction)
        measured_error = self.paired_delta(reference, self.continuous_q[:, self.LEG_IDS]).abs().amax(-1)
        folded = folding & (remaining < .02) & (measured_error < .2) & (elapsed >= .3)
        orbit_start = folded & ((self.route == int(RecoveryRoute.ORBIT_POSITIVE)) | (self.route == int(RecoveryRoute.ORBIT_NEGATIVE)))
        self.orbit_start[orbit_start] = reference[orbit_start]
        self.orbit_angle[orbit_start] = 0.
        self._enter(orbit_start, RecoveryPhase.ORBIT)
        side_start = folded & (self.route == int(RecoveryRoute.SIDE))
        self.side_start[side_start] = reference[side_start]
        # Body-frame gravity selects the lower side; no world wheel heights.
        left_direction = torch.where(gravity[:, 1] >= 0., -1., 1.)
        side_direction = torch.stack((left_direction, left_direction, -left_direction, -left_direction), -1)
        self.side_direction[side_start] = side_direction[side_start]
        self._enter(side_start, RecoveryPhase.SIDE)
        self._enter(folded & ~(orbit_start | side_start), RecoveryPhase.PLANT)
        converted = side & (elapsed > .05) & (gravity[:, 1].abs() < .65)
        self.route[converted] = selected[converted]
        self._enter(converted & near_upright, RecoveryPhase.PREPARE)
        self._enter(converted & ~near_upright, RecoveryPhase.FOLD)
        self._fail(side & ~converted & (elapsed >= 2 * side_duration + .4), 2)
        self.plant_ticks = torch.where(planting & support, self.plant_ticks + 1, 0)
        planted = planting & (remaining < .15) & (measured_error < .2) & (self.plant_ticks >= self._ticks(.06))
        self._enter(planted, RecoveryPhase.PREPARE)
        self._fail(planting & ~planted & (elapsed >= 2.5), 3)
        orbit_contact = orbiting & support & (tilt < math.radians(145.)) & (estimated_height > .12) & (self.orbit_angle > 3.7)
        self.thrust_anchor[orbit_contact] = (reference + self.paired_delta(self.thrust.expand_as(reference), reference))[orbit_contact]
        self.plant_pitch[orbit_contact] = self.continuous_pitch[orbit_contact]
        self._enter(orbit_contact, RecoveryPhase.THRUST)
        captured = (orbiting & (tilt < math.radians(30.)) & (estimated_height > .22) & support)
        captured |= thrusting & (tilt < math.radians(65.)) & (estimated_height > .24) & (gyro.norm(dim=-1) < 8.) & support
        self._enter(captured, RecoveryPhase.CAPTURE)
        self._fail(orbiting & ~orbit_contact & ~captured & (self.orbit_angle >= math.tau * self.config["orbit_turns"]), 4)
        self._enter(capturing & (tilt < math.radians(20.)) & support & (estimated_height > .22), RecoveryPhase.PREPARE)
        stable = (valid & support & body_clear & (tilt < self.config["handover_tilt_rad"])
            & (gyro.norm(dim=-1) < self.config["handover_gyro_max"])
            & (estimated_height > self.config["handover_height_min"])
            & (estimated_height < self.config["handover_height_max"])
            & (dq_control[:, self.WHEEL_IDS].abs().amax(-1) * .06 < .25)
            & (dq_control[:, self.LEG_IDS].abs().amax(-1) < 2.))
        ready = preparing & stable & (remaining < .02) & (measured_error < .12)
        self.ready_ticks = torch.where(ready, self.ready_ticks + 1, 0)
        self._enter(preparing & (self.ready_ticks >= self._ticks(.1)), RecoveryPhase.BLEND)
        self._enter(blending & (self.phase_ticks >= self._ticks(self.config["blend_seconds"])), RecoveryPhase.RL)
        holding_rl = self.enabled & (phase == int(RecoveryPhase.RL)) & ~self.motion_released
        self.stable_ticks = torch.where(holding_rl & stable, self.stable_ticks + 1, 0)
        self.motion_released |= holding_rl & (self.stable_ticks >= self._ticks(self.config["stable_seconds"]))
        takeover = (self.blending | (self.pure_rl & ~self.motion_released)) & self.enabled
        self._fail(takeover & (tilt > math.radians(45.)), 5)
        self._fail(self.enabled & ~self.pure_rl & (self.age_ticks >= self._ticks(self.config["max_script_seconds"])), 6)
        return self.script_targets

    @property
    def failed(self):
        return self.phase == int(RecoveryPhase.FAILED)

    @property
    def scripted(self):
        return self.enabled & (self.phase < int(RecoveryPhase.BLEND))

    @property
    def blending(self):
        return self.enabled & (self.phase == int(RecoveryPhase.BLEND))

    @property
    def pure_rl(self):
        return self.phase == int(RecoveryPhase.RL)

    @property
    def blend(self):
        fraction = (self.phase_ticks * self.dt / self.config["blend_seconds"]).clamp(0., 1.)
        return torch.where(self.pure_rl, 1., torch.where(self.blending, fraction, 0.))

    @property
    def motion_hold(self):
        return self.enabled & ~self.motion_released

    def diagnostics(self):
        return {"recovery_enabled": self.enabled.clone(), "recovery_phase": self.phase.clone(),
            "recovery_route": self.route.clone(), "recovery_failed": self.failed,
            "recovery_failure_code": self.failure_code.clone(), "recovery_blend": self.blend,
            "recovery_stable_seconds": self.stable_ticks * self.dt,
            "recovery_script_seconds": self.age_ticks * self.dt,
            "recovery_reroute_count": self.reroute_count.clone(),
            "recovery_motion_released": self.motion_released.clone()}
