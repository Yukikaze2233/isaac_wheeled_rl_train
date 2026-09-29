"""Chassis reward densities and bounded phase-aware stationary objectives.

Reference: SCUTRobotLab (MIT), b8ff79f, V14 env_cfg.py:869-914 and
wheelbipe25_v3/env.py:3505-3833. Values returned here are per-second densities.
The V5 environment retains its own real travel/closure/contact failure rules.
"""
import math

import torch
import torch.nn.functional as F

from .v5_control import V5Control


# Research freeze v1, not an identified physical optimum. All weights are reward
# per second; SI scales have the same unit as the corresponding error.
DENSE_TRACKING_V1 = {
    "enabled": True,
    "version": "v6-dense-visible35-v1",
    "height": {"broad_scale": .04, "broad_weight": 3., "fine_scale": .015, "fine_weight": 1.5},
    "velocity": {"broad_scale": .5, "broad_weight": 2., "fine_scale": .10, "fine_weight": .6},
    "lateral": {"broad_scale": .15, "broad_weight": .5, "fine_scale": .08, "fine_weight": .25},
    "yaw": {"broad_scale": 1., "broad_weight": 1., "fine_scale": .2, "fine_weight": .35},
    "stationary": {"broad_scale": .10, "broad_weight": 1.5, "fine_scale": .04, "fine_weight": .75},
    "stationary_vx_blend_m_s": .10,
    "stationary_yaw_blend_rad_s": .20,
    "tracking_peak_density": 1.,
    "ordinary_vertical_velocity_weight": 0.,
}


def validate_dense_tracking(settings):
    """Reject silent disabled precision, hidden derivative targets and bad scales."""
    if settings.get("enabled") is not True:
        raise ValueError("Dense tracking freeze requires enabled=true")
    if settings.get("version") != DENSE_TRACKING_V1["version"]:
        raise ValueError("Unknown dense tracking version")
    for axis in ("height", "velocity", "lateral", "yaw", "stationary"):
        for key in ("broad_scale", "broad_weight", "fine_scale", "fine_weight"):
            value = settings[axis][key]
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Dense tracking requires positive finite {axis}.{key}")
    for key in ("stationary_vx_blend_m_s", "stationary_yaw_blend_rad_s", "tracking_peak_density"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"Dense tracking requires positive finite {key}")
    if settings.get("ordinary_vertical_velocity_weight") != 0.:
        raise ValueError("Visible35 ordinary height has no identifiable derivative target")
    return settings


def dense_error_terms(error, settings):
    """Unclipped broad pseudo-Huber density and a bounded local precision cost.

    The rationalized form of sqrt(1+x*x)-1 avoids cancellation near zero.
    Fine saturation never switches the broad term off, nor rewards leaving a
    precision band. No dt is applied here: the environment applies policy_dt.
    """
    square = (error / settings["broad_scale"]).square()
    broad = -settings["broad_weight"] * square / (torch.sqrt(1. + square) + 1.)
    fine = settings["fine_weight"] * torch.expm1(-(error / settings["fine_scale"]).square())
    return broad, fine


def apply_climb_height_tracking(terms, scale):
    """Relax only posture-height objectives while ordinary climbing crosses a riser.

    Call after assembling the ordinary reward terms. Forward/yaw tracking,
    attitude, physical limits and effort costs remain intact. A completed climb
    restores scale=1; no additional reward is paid for remaining near the edge.
    """
    for name in ("track_height", "height_square", "height_wide_companion", "height_velocity_tracking",
                 "height_precision", "dense_height_broad", "dense_height_precision",
                 "landing_height_tracking", "landing_velocity_tracking", "vertical_velocity"):
        # Dense tracking's peak is constant, not height-dependent. Removing it
        # would make merely entering the transition zone cost one reward/s.
        if name == "track_height" and "dense_height_broad" in terms:
            continue
        if name in terms:
            terms[name] = terms[name] * scale


def apply_dense_tracking(terms, velocity, omega, gravity, height, commands, support,
                         ordinary, reference, settings, *, spin_velocity_error=None):
    """Replace legacy tracking once; retain its peak density and physical costs.

    Targets depend only on visible instantaneous commands. Ordinary rising and
    falling height commands can have identical 35D observations, so neither a
    hidden h-dot target nor an unconditional zero-vz penalty is added. Native
    command-clock jump and step reference objectives remain separate.
    """
    for name in ("track_lin_vel", "lin_vel_square", "track_yaw", "yaw_square", "track_height",
                 "height_square", "height_wide_companion", "height_velocity_tracking",
                 "velocity_huber", "velocity_wide", "yaw_precision", "velocity_precision",
                 "lateral_precision", "stand_precision", "height_precision",
                 "stationary_speed_precision", "stand_translation", "spin_translation"):
        terms.pop(name, None)
    ordinary = ordinary.to(velocity.dtype)
    jumping = reference.jumping if reference is not None else torch.zeros_like(height, dtype=torch.bool)
    height_gate = support * ~jumping
    # Continuous command-visible blend removes a reward cliff when braking
    # crosses the old |command| < .01 stationary classification boundary.
    quiet = (1. - torch.maximum(commands[:, 0].abs() / settings["stationary_vx_blend_m_s"],
                               commands[:, 1].abs() / settings["stationary_yaw_blend_rad_s"])).clamp(0., 1.)
    quiet = quiet * ordinary * ~jumping
    vx = velocity[:, 0] * (1. - gravity[:, 0].square()).clamp_min(0.).sqrt()
    errors = {"height": height - commands[:, 2], "velocity": vx - commands[:, 0],
              "lateral": velocity[:, 1], "yaw": omega[:, 2] - commands[:, 1],
              "stationary": torch.stack((vx - commands[:, 0], velocity[:, 1]), -1).norm(dim=-1)}
    gates = {"height": height_gate, "velocity": ordinary * (1. - quiet),
             "lateral": ordinary * (1. - quiet), "yaw": torch.ones_like(height),
             "stationary": quiet * support}
    for axis, error in errors.items():
        broad, fine = dense_error_terms(error, settings[axis])
        terms[f"dense_{axis}_broad"] = gates[axis] * broad
        terms[f"dense_{axis}_precision"] = gates[axis] * fine
    # SCUT's original three tracking peaks are kept exactly once. They are not
    # multiplied into negative costs: falling/leaning cannot hide tracking error.
    peak = settings["tracking_peak_density"]
    terms["track_lin_vel"] = ordinary * peak
    terms["track_yaw"] = torch.ones_like(height) * peak
    terms["track_height"] = height_gate * peak
    terms["vertical_velocity"] = torch.zeros_like(height)
    if spin_velocity_error is not None:
        broad, fine = dense_error_terms(spin_velocity_error, settings["velocity"])
        terms["dense_spin_velocity_broad"] = (1. - ordinary) * broad
        terms["dense_spin_velocity_precision"] = (1. - ordinary) * fine
        terms["track_lin_vel"] += (1. - ordinary) * peak


class StationaryAnchor:
    """Bounded residence reward; the anchor is privileged training state only."""
    def __init__(self, count, device, weight, band_m):
        if weight < 0 or band_m <= 0:
            raise ValueError("Stationary reward requires nonnegative weight and positive distance band")
        self.weight, self.band_m = weight, band_m
        self.position = torch.zeros(count, 2, device=device)
        self.valid = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.valid[ids] = False

    def reward(self, position, requested, supported):
        self.valid &= requested
        entering = requested & supported & ~self.valid
        self.position.copy_(torch.where(entering[:, None], position, self.position))
        self.valid |= entering
        distance_square = (position - self.position).square().sum(-1)
        return self.weight * torch.expm1(-distance_square / self.band_m**2) * requested * supported * self.valid


class CommandedHeightMargin:
    """Penalize excess stop proximity relative to a verified commanded posture."""

    def __init__(self, reference, device):
        self.heights = torch.tensor(reference["height_m"], device=device)
        self.risks = torch.tensor(reference["risk"], device=device)
        if (self.heights.ndim != 1 or len(self.heights) < 2 or self.risks.shape != self.heights.shape
                or not torch.isfinite(self.heights).all() or not torch.isfinite(self.risks).all()
                or not (self.heights.diff() > 0).all() or not ((self.risks >= 0) & (self.risks < 1)).all()):
            raise ValueError("Height margin requires increasing finite heights and risks in [0,1)")

    def excess(self, risk, height):
        height = height.clamp(self.heights[0], self.heights[-1]).contiguous()
        upper = torch.searchsorted(self.heights, height).clamp(1, len(self.heights) - 1)
        fraction = (height - self.heights[upper - 1]) / (self.heights[upper] - self.heights[upper - 1])
        allowed = torch.lerp(self.risks[upper - 1], self.risks[upper], fraction)[:, None]
        # The cost at the mechanical boundary remains one, even near an endpoint target.
        return (risk - allowed).clamp_min(0.) / (1. - allowed).clamp_min(1e-6)


def height_tracking_terms(height_error, velocity_error, support, moving, settings):
    """A wide companion preserves the original narrow kernel's precision signal."""
    return {
        "height_wide_companion": support * settings["wide_weight"] * torch.expm1(
            -height_error.square() / settings["wide_sigma_m"]**2),
        "height_velocity_tracking": support * moving * settings["velocity_weight"] * torch.exp(
            -velocity_error.square() / settings["velocity_sigma_m_s"]**2),
    }


def reward_terms(velocity, omega, gravity, height, commands, motor_velocity, motor_acceleration,
                  motor_torque, actions, previous_actions, before_previous_actions,
                  wheel_positions_b, support, ordinary_motion, undesired_contact, height_kernel_width_m=None):
    leg, wheel = list(V5Control.LEGS), list(V5Control.WHEELS)
    # Project body-forward velocity onto the horizontal plane as in V14.
    cos_pitch = torch.sqrt((1. - gravity[:, 0].square()).clamp_min(0.))
    ev = commands[:, 0] - velocity[:, 0] * cos_pitch
    ew = commands[:, 1] - omega[:, 2]
    eh = height - commands[:, 2]
    gx, gy = gravity[:, 0], gravity[:, 1]
    orientation_scale = 2. * torch.exp(-commands[:, 0].square() / 3.) + 2.
    delta = actions - previous_actions
    second = actions - 2 * previous_actions + before_previous_actions
    fork = wheel_positions_b[:, 0, 0] - wheel_positions_b[:, 1, 0]
    stationary = (commands[:, 0].abs() < .1) * ordinary_motion * support
    height_denominator = .001 if height_kernel_width_m is None else height_kernel_width_m.square()
    return {
        "track_lin_vel": ordinary_motion * torch.exp(-ev.clamp(-1., 1.).square() / .5),
        "lin_vel_square": -(ordinary_motion * (.25 * ev).square()),
        "track_yaw": torch.exp(-ew.clamp(-.8, .8).square() / .25),
        "yaw_square": -(.5 * ew).square(),
        "track_height": support * torch.exp(-eh.clamp(-.15, .15).square() / height_denominator),
        "height_square": -support * (10. * eh).square(),
        "pitch_exp": torch.exp(-gx.square() / .02),
        "roll_exp": torch.exp(-gy.square() / .01),
        "pitch_velocity": -2. * support * (orientation_scale * gx).square(),
        "roll_velocity": -2. * (orientation_scale * gy).square(),
        "stand_translation": -stationary * velocity[:, :2].abs().sum(-1),
        "leg_velocity": -.005 * motor_velocity[:, leg].square().sum(-1),
        "leg_acceleration": -5e-7 * motor_acceleration[:, leg].square().sum(-1),
        "wheel_velocity": -1e-5 * motor_velocity[:, wheel].square().sum(-1),
        "wheel_acceleration": -1e-8 * motor_acceleration[:, wheel].square().sum(-1),
        "motor_torque": -1e-4 * motor_torque.square().sum(-1),
        "wheel_power": -1e-4 * (motor_torque[:, wheel] * motor_velocity[:, wheel]).clamp_min(0).sum(-1),
        "vertical_velocity": -.5 * support * velocity[:, 2].square(),
        "horizontal_angular_velocity": -.05 * omega[:, :2].square().sum(-1),
        "action_rate": -.01 * delta.square().sum(-1),
        "leg_action_smoothness": -.05 * second[:, leg].square().sum(-1),
        "wheel_action_smoothness": -.01 * second[:, wheel].square().sum(-1),
        "no_fork": -(fork.abs() > .05).float(),
        "no_fork_square": -(5. * fork).square(),
        "undesired_contact": -2. * undesired_contact.float(),
    }


def reference_motion_terms(velocity, gravity, commands, support, ordinary, reference, settings):
    """Track a visible command derivative instead of penalizing intended lean."""
    ordinary = ordinary.to(velocity.dtype)
    error = commands[:, 0] - velocity[:, 0] * (1 - gravity[:, 0].square()).clamp_min(0).sqrt()
    pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
    pitch_error = torch.sin(pitch - reference.pitch)
    if settings.get("visible_reference_only"):
        # The 35D policy has no acceleration slot. A bounded stair lean is
        # determined by its visible mode, command and unsaturated command clock.
        climbing = (reference.terrain_mode != 0) & (reference.elapsed < 4.)
        target = climbing * commands[:, 0].sign() * settings.get("step_pitch_rad", .08)
        pitch_error = torch.sin(pitch - target)
    orientation_scale = 2 * torch.exp(-commands[:, 0].square() / 3.) + 2
    spinning = (commands[:, 0].abs() < .01) & (commands[:, 1].abs() > .1)
    return {
        "track_lin_vel": ordinary * torch.exp(-error.square() / .5),
        "velocity_huber": -ordinary * settings["velocity_huber_weight"] * F.smooth_l1_loss(
            error, torch.zeros_like(error), beta=settings["velocity_huber_delta_m_s"], reduction="none"),
        "velocity_wide": ordinary * settings["velocity_wide_weight"] * torch.exp(
            -error.square() / settings["velocity_wide_sigma_m_s"]**2),
        "pitch_exp": torch.exp(-pitch_error.square() / .02),
        "pitch_velocity": -2 * support * (orientation_scale * pitch_error).square(),
        "spin_translation": -1. * spinning * support * settings["spin_translation_weight"] * F.smooth_l1_loss(
            velocity[:, :2], torch.zeros_like(velocity[:, :2]), beta=.15, reduction="none").sum(-1),
    }


def apply_manual_tracking(terms, velocity, omega, gravity, commands, support, ordinary, reference, settings):
    """Gate positive densities, never let leaning erase error or safety costs."""
    requested = ((commands[:, :2].abs() < .01).all(-1) & ~reference.jumping
                 & (reference.terrain_mode == 0) & ordinary.bool())
    terms["stationary_speed_precision"] = (requested * support * settings["stationary_weight"]
        * torch.expm1(-velocity[:, :2].square().sum(-1) / settings["stationary_width_m_s"]**2))
    terms["yaw_precision"] = (.25 * ordinary * torch.expm1(
        -(commands[:, 1] - omega[:, 2]).square() / .1**2))
    denominator = torch.where(requested, settings["upright_denominator"],
                              max(.05, settings["upright_denominator"]))
    gate = torch.exp(-gravity[:, :2].square().sum(-1) / denominator)
    exempt = reference.jumping | ((reference.terrain_mode != 0) & (reference.elapsed < 4.))
    gate = torch.where(exempt, torch.ones_like(gate), gate)
    for name in ("track_lin_vel", "track_yaw", "track_height", "velocity_wide"):
        if name in terms:
            terms[name] *= gate


def precision_tracking_terms(velocity, gravity, height, commands, support, ordinary,
                             reference, wheel_positions_b, settings):
    """Bounded precision costs complement the existing wide tracking rewards.

    These costs stay outside the upright multiplier: leaning must not hide an
    error. Manual jump/step motion retains its phase-specific velocity objective.
    """
    locomotion = ordinary.bool() & ~reference.jumping & (reference.terrain_mode == 0)
    stationary = locomotion & (commands[:, :2].abs() < .01).all(-1)
    moving = locomotion & ~stationary
    vx = velocity[:, 0] * (1 - gravity[:, 0].square()).clamp_min(0).sqrt()
    error = commands[:, 0] - vx
    fork = (wheel_positions_b[:, 0, 0] - wheel_positions_b[:, 1, 0]).abs()
    excess = (fork - .05).clamp_min(0.)
    return {
        "velocity_precision": moving * settings["velocity_weight"] * torch.expm1(
            -(error / settings["velocity_width_m_s"]).square()),
        "lateral_precision": moving * settings["lateral_weight"] * torch.expm1(
            -(velocity[:, 1] / settings["lateral_width_m_s"]).square()),
        "stand_precision": stationary * support * settings["stationary_weight"] * torch.expm1(
            -velocity[:, :2].square().sum(-1) / settings["stationary_width_m_s"]**2),
        "height_precision": support * ~reference.jumping * settings["height_weight"] * torch.expm1(
            -((height - commands[:, 2]) / settings["height_width_m"]).square()),
        # Preserve the original geometric tolerance, remove its one-unit jump.
        "no_fork": settings["fork_weight"] * torch.expm1(
            -(excess / settings["fork_width_m"]).square()),
    }


def reference_jump_terms(reference, task, phase, contacts, height, com_vz, velocity,
                         gravity, extension, position, commands, *, landing_contact_time=None):
    """Phase-local densities; the release-height plateau cannot earn push reward."""
    from .task import Phase
    pre = (reference.phase == reference.PRELOAD) & ~task.com_released
    push = (reference.phase == reference.PUSH) & ~task.com_released
    airborne = reference.jumping & (phase == Phase.FLIGHT)
    landing = (reference.jumping & task.com_released & contacts.any(-1) & ~airborne
               & (reference.phase >= reference.LAND))
    height_error = task.com_displacement - reference.height_delta
    speed_error = com_vz - reference.vertical_velocity
    h_kernel = torch.exp(-(height_error / .035).square())
    v_kernel = torch.exp(-(speed_error / .5).square())
    after_push = reference.jumping & (reference.elapsed >= reference.push_end) & ~task.com_released
    shortfall = ((reference.release_speed - com_vz) / reference.release_speed.clamp_min(.3)).clamp(0., 2.)
    apex = reference.release_speed.square() / (2 * 9.81)
    # A token hop must not unlock the same long recovery reward as the requested
    # jump. This remains a dense progress factor, not a relaxed success predicate.
    landing = landing * (task.com_rise / apex.clamp_min(.001)).clamp(0., 1.) * (task.clear_air_time_peak >= .06)
    release_error = (task.com_release_speed - reference.release_speed) / reference.release_speed.clamp_min(.3)
    released_window = reference.jumping & task.com_released & (reference.phase <= reference.LAND)
    displacement = (position[:, :2] - task.origin_xy).norm(dim=-1)
    planar_error = velocity[:, :2] - torch.stack((commands[:, 0], torch.zeros_like(height)), -1)
    terms = {
        "jump_preload_height": pre * (3 * h_kernel - height_error.abs()),
        "jump_preload_velocity": pre * (1.5 * v_kernel - .25 * speed_error.abs()),
        "jump_push_height": push * 1.5 * h_kernel,
        "jump_push_velocity": push * (4 * v_kernel - shortfall),
        "jump_release_shortfall": -1. * after_push * shortfall,
        "jump_release_velocity_match": -1. * released_window * release_error.square().clamp_max(4.),
        "jump_tuck": 2 * airborne * torch.exp(-((extension - .16) / .04).square()),
        "jump_air_attitude": -1. * airborne * gravity[:, :2].square().sum(-1),
        "jump_landing_height": 2 * landing * (reference.phase == reference.LAND) * torch.exp(-(height_error / .04).square()),
        "jump_recovery_height": 2 * landing * (reference.phase == reference.RECOVER) * torch.exp(-((height - commands[:, 2]) / .035).square()),
        "jump_landing_vertical_velocity": landing * torch.exp(-(speed_error / .5).square()),
        "jump_landing_velocity": 2 * landing * torch.exp(-planar_error.square().sum(-1) / .04),
        "jump_landing_attitude": 2 * landing * torch.exp(-gravity[:, :2].square().sum(-1) / .02),
        "jump_landing_position": -1. * landing * (commands[:, 0].abs() < .01) * (displacement / .25).square().clamp_max(4.),
    }
    raised = getattr(reference, "landing_delta", torch.zeros_like(height)) != 0
    if not getattr(reference, "has_landing_delta", False) or not hasattr(reference, "step_landing_reference"):
        return terms
    # A high platform may be reached before the command-clock landing phase.
    # Use physical contact only in reward; the actor still gets preset commands.
    actual_landing = (reference.jumping & raised & task.com_released & contacts.any(-1)
                      & ((phase == Phase.LANDING) | (phase == Phase.RECOVERY)))
    contact_time = ((reference.elapsed - reference.flight_end).clamp_min(0.)
                    if landing_contact_time is None else landing_contact_time)
    landing_height, landing_speed = reference.step_landing_reference(contact_time)
    step_height_error = torch.where(actual_landing, task.com_displacement - landing_height, height_error)
    step_speed_error = torch.where(actual_landing, com_vz - landing_speed, speed_error)

    def dense(error, axis):
        broad, fine = dense_error_terms(error, DENSE_TRACKING_V1[axis])
        return 1. + broad + fine

    step_h = dense(step_height_error, "height")
    step_v = dense(step_speed_error, "velocity")
    valid_landing = (actual_landing * (task.com_rise / apex.clamp_min(.001)).clamp(0., 1.)
                     * (task.clear_air_time_peak >= .06))
    settled = contact_time >= reference.cfg["landing_seconds"]
    replacements = {
        "jump_preload_height": pre * 3 * step_h,
        "jump_preload_velocity": pre * 1.5 * step_v,
        "jump_push_height": push * 1.5 * step_h,
        "jump_push_velocity": push * (4 * step_v - shortfall),
        "jump_landing_height": 2 * valid_landing * ~settled * step_h,
        "jump_recovery_height": 2 * valid_landing * settled * step_h,
        "jump_landing_vertical_velocity": valid_landing * step_v,
        "jump_landing_velocity": 2 * valid_landing * torch.exp(-planar_error.square().sum(-1) / .04),
        "jump_landing_attitude": 2 * valid_landing * torch.exp(-gravity[:, :2].square().sum(-1) / .02),
        # Platform landing zones belong to the step task, not the takeoff radius.
        "jump_landing_position": torch.zeros_like(height),
    }
    for name, value in replacements.items():
        terms[name] = torch.where(raised, value, terms[name])
    return terms
