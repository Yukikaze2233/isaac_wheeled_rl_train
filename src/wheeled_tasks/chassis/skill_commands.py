"""Single-skill commands and airborne resets for the V5 SCUT-style curriculum.

Reference-frame velocity projection and tracking follow SCUTRobotLab V14
wheelbipe_V14/env.py (MIT, commit b8ff79f). The fixed reference heading here
represents a virtual gimbal; the V5 chassis asset has no actuated gimbal.
"""
import math

import torch

from .task import Phase
from .training_schedule import TrainingSchedule, actor_update_clock, stage_actor_updates
from .locomotion_difficulty import (
    bounded_height_sampling, difficulty_envelope, moving_spec,
    scale_planar_command, validate_difficulty,
)
from .release_stop import ReleaseStop, sample_release_stop_command
from .ordinary_large_stair import activation_fraction, command_profile, validate_row


def validate_planar_sampling(spec):
    standing = spec.get("standing_command_probability", 0.)
    if not isinstance(standing, (int, float)) or not math.isfinite(standing) or not 0 <= standing <= 1:
        raise ValueError("Standing command probability must be within [0, 1]")
    if "command_resampling_scope" in spec:
        bounds = spec.get("command_resampling_seconds")
        if (spec["command_resampling_scope"] != "skill" or not isinstance(bounds, (list, tuple))
                or len(bounds) != 2 or not all(math.isfinite(x) for x in bounds)
                or not 0 < bounds[0] <= bounds[1]):
            raise ValueError("Skill-local command duration requires finite positive declared bounds")
    bounds = spec.get("amplitude_range", [.4, 1.])
    if (len(bounds) != 2 or not all(math.isfinite(x) for x in bounds)
            or not 0 <= bounds[0] < bounds[1] <= 1.):
        raise ValueError("Command amplitude requires finite bounds within [0, 1]")
    joint = spec.get("planar_command_sampling")
    if joint is not None:
        for key in ("vx_range_m_s", "yaw_range_rad_s"):
            low, high = joint[key]
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("Planar command sampling requires finite increasing ranges")
        if not 0 <= joint["nominal_fraction"] <= 1:
            raise ValueError("Planar nominal fraction must be a probability")


def sample_planar_command(command, spec, random, *, evaluation=False, standing_out=None):
    """Keep legacy RNG semantics; independently sample only declared joint domains."""
    if evaluation:
        return command
    count = len(command)
    if spec.get("sample_amplitude", False):
        low, high = spec.get("amplitude_range", [.4, 1.])
        command[:, :2] *= (low + (high - low) * random(count))[:, None]
    if spec.get("sample_yaw_sign", True) and spec.get("kind") in ("rotate", "curve", "spin_translate"):
        command[:, 1] *= torch.where(random(count) < .5, -1., 1.)
    joint = spec.get("planar_command_sampling")
    if joint is not None:
        sampled = random(count) >= joint["nominal_fraction"]
        for axis, key in enumerate(("vx_range_m_s", "yaw_range_rad_s")):
            low, high = joint[key]
            values = low + (high - low) * random(count)
            command[:, axis] = torch.where(sampled, values, command[:, axis])
    standing = spec.get("standing_command_probability", 0.)
    if standing:
        selected = random(count) < standing
        command[selected, :2] = 0.
        if standing_out is not None:
            standing_out.copy_(selected)
    elif standing_out is not None:
        standing_out.zero_()
    return command


def scheduled_command(command, schedule, equivalent_iteration):
    """Ramp a command using the 4096-environment reference sample count."""
    if not schedule:
        return list(command)
    fraction = min(1., max(0., (equivalent_iteration - schedule["start"]) / schedule["ramp"]))
    return [initial + fraction * (target - initial)
            for initial, target in zip(schedule["initial"], command)]


def height_reference(time, profile):
    """Full-stroke, zero-end-velocity transitions with explicit endpoint dwell."""
    low, high = profile["height_range_m"]
    settings = profile["height_motion"]
    duration, dwell = settings["transition_seconds"], settings["dwell_seconds"]
    initial = settings.get("initial_height_m", .305)
    if not (low < high and low <= initial <= high and duration > 0 and dwell >= 0):
        raise ValueError("Invalid height reference domain or timing")
    u = (time / duration).clamp(0., 1.)
    endpoint = settings.get("endpoint")
    first_target = high if endpoint == "high" else low
    first_height = initial + (first_target - initial) * (3 * u.square() - 2 * u.pow(3))
    first_speed = (first_target - initial) * 6 * u * (1 - u) / duration
    if endpoint is not None:
        if endpoint not in ("low", "high"):
            raise ValueError("Height endpoint must be low or high")
        return first_height, first_speed
    clock = (time - duration).clamp_min(0.) % (2 * (duration + dwell))
    up = ((clock - dwell) / duration).clamp(0., 1.)
    down = ((clock - 2 * dwell - duration) / duration).clamp(0., 1.)
    height = low + (high - low) * (3 * up.square() - 2 * up.pow(3) - 3 * down.square() + 2 * down.pow(3))
    speed = (high - low) * 6 * (up * (1 - up) - down * (1 - down)) / duration
    return torch.where(time < duration, first_height, height), torch.where(time < duration, first_speed, speed)


def profile_command(time, command, profile):
    """Evaluate a deterministic command trajectory; time is in seconds."""
    result = torch.as_tensor(command, device=time.device, dtype=time.dtype).expand(len(time), -1).clone()
    kind = profile.get("kind", "constant")
    if profile.get("height_motion"):
        result[:, 2] = height_reference(time, profile)[0]
    elif kind == "height":
        low, high = profile.get("height_range_m", [.29, .32])
        result[:, 2] = (low + high) / 2 + (high - low) / 2 * torch.sin(math.tau * time / 6.)
    elif kind == "start_stop":
        segment = torch.floor(time / profile.get("segment_seconds", 3.)).long() % 4
        result[:, 0] *= torch.where(segment == 0, 1., torch.where(segment == 2, -1., 0.))
    elif kind == "velocity_curve":
        segment_seconds = profile["segment_seconds"]
        duration = profile["velocity_transition_seconds"]
        if not 0 < duration < segment_seconds:
            raise ValueError("Velocity curves require a bounded transition and a nonempty dwell")
        segment = torch.floor(time / segment_seconds).long() % 4
        target = torch.where(segment == 0, 1., torch.where(segment == 2, -1., 0.))
        initial = torch.where(segment == 1, 1., torch.where(segment == 3, -1., 0.))
        u = ((time % segment_seconds) / duration).clamp(0., 1.)
        result[:, 0] *= initial + (target - initial) * (3 * u.square() - 2 * u.pow(3))
    elif kind == "weave":
        segment = torch.floor(time / profile.get("segment_seconds", 1.)).long() % 4
        result[:, 1] *= torch.where((segment == 0) | (segment == 3), 1., -1.)
    elif kind == "height_pulse":
        pulse = profile["height_pulse"]
        clock = (time - pulse["start_seconds"]).remainder(pulse["period_seconds"])
        raised = (time >= pulse["start_seconds"]) & (clock < pulse["hold_seconds"])
        result[:, 2] = torch.where(raised, (result[:, 2] + pulse["bias_m"]).clamp_max(pulse["maximum_m"]), result[:, 2])
    return result


def reference_velocity(body_velocity, yaw):
    """Rotate planar body velocity into a fixed virtual-gimbal reference frame."""
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c * body_velocity[:, 0] - s * body_velocity[:, 1],
                        s * body_velocity[:, 0] + c * body_velocity[:, 1]), -1)


class SkillCommands:
    """Keep task sampling, command profiles and special resets out of physics."""

    def __init__(self, env, *, root=None):
        self.env = env
        self.schedule = TrainingSchedule(env)
        self.difficulty_cfg = validate_difficulty(env.cfg.get("locomotion_difficulty"))
        self.difficulty_enabled = self.difficulty_cfg is not None and not env.cfg.get("evaluation_exact_cases", False)
        specs = env.cfg.get("skill_specs", {})
        if env.cfg.get("evaluation_exact_cases"):
            specs = {c["name"]: {"command": c["command"], **c.get("skill", {})}
                     for c in env.cfg["evaluation"]["cases"]}
        self.specs = specs
        self.large_stair_curricula = {name: settings for name, spec in specs.items()
            if (settings := validate_row(spec)) is not None} if not env.cfg.get("evaluation_exact_cases") else {}
        self.large_stair_phase_offset = torch.zeros(env.num_envs, device=env.device) if self.large_stair_curricula else None
        self.large_stair_assessment = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.large_stair_actor_clock = None
        self.release_stop = (ReleaseStop(env.num_envs, env.device, env.cfg["release_stop"], env.policy_dt)
                             if "release_stop" in env.cfg else None)
        if any(spec.get("kind") == "release_stop" for spec in specs.values()) and self.release_stop is None:
            raise ValueError("Release-stop skill requires its explicit timing contract")
        self.has_spin = any(spec.get("kind") == "spin_translate" for spec in specs.values())
        self.has_airborne_resets = any(spec.get("kind") in ("airborne", "landing") for spec in specs.values())
        self.batches = [(torch.tensor([i for i, name in enumerate(env.scene_groups) if name == key],
                                     device=env.device, dtype=torch.long), {**spec, "group_name": key}) for key, spec in specs.items()]
        self.reference_target = torch.zeros(env.num_envs, 2, device=env.device)
        self.reference_filtered = torch.zeros_like(self.reference_target)
        self.spin = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.stable = torch.zeros_like(self.spin)
        self.landing = torch.zeros_like(self.spin)
        self.height_motion = torch.zeros_like(self.spin)
        self.height_velocity_reference = torch.zeros(env.num_envs, device=env.device)
        self.height_target = torch.full((env.num_envs,), .305, device=env.device)
        self.height_phase_offset = torch.zeros(env.num_envs, device=env.device)
        self.start_stop_delay = torch.zeros(env.num_envs, device=env.device)
        self.start_stop_delay_bounds = None
        delay = self.schedule.cfg.get("start_stop_start_delay_seconds")
        if delay is not None:
            if (not isinstance(delay, (list, tuple)) or len(delay) != 2
                    or any(type(value) not in (float, int) or not math.isfinite(value) for value in delay)
                    or not 0 <= delay[0] <= delay[1]):
                raise ValueError("Start/stop startup delay requires finite nonnegative bounds")
            if self.schedule.enabled:
                self.start_stop_delay_bounds = tuple(delay)
        default_push = getattr(env, "stage_cfg", {}).get("push_max", 0.)
        self.push_speed = torch.full((env.num_envs,), float(default_push), device=env.device)
        self.command_base = torch.zeros(env.num_envs, 3, device=env.device)
        self.manual_step_mode = torch.zeros(env.num_envs, device=env.device)
        self.manual_step_start = torch.full((env.num_envs,), torch.inf, device=env.device)
        self.manual_step_duration = torch.zeros(env.num_envs, device=env.device)
        self.sequence_cursor = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.vx_slew = torch.full((env.num_envs,), env.cfg.get("command_slew", {}).get("vx_m_s2", 1.5), device=env.device)
        for ids, spec in self.batches:
            validate_planar_sampling(spec)
            previous_time = -1.
            for entry in spec.get("command_sequence", []):
                if (not math.isfinite(entry["time_s"]) or entry["time_s"] < max(0., previous_time)
                        or entry["mode"] not in (0, 1, 2, 3, 4) or len(entry["command"]) != 3
                        or not all(math.isfinite(value) for value in entry["command"])
                        or entry.get("after_phase") not in (None, "ground_after_recovery")):
                    raise ValueError("Invalid command sequence timing, values or phase guard")
                previous_time = entry["time_s"]
            self.spin[ids] = spec.get("kind") == "spin_translate"
            self.landing[ids] = spec.get("kind") in ("airborne", "landing")
            self.height_motion[ids] = bool(spec.get("height_motion") or spec.get("height_transition_seconds"))
            self.push_speed[ids] = spec.get("push_m_s", default_push)
            self.vx_slew[ids] = spec.get("acceleration_m_s2", env.cfg.get("command_slew", {}).get("vx_m_s2", 1.5))
            if env.cfg.get("command_reference"):
                self.manual_step_mode[ids] = {"step_up": 1., "stairs": 1., "step_down": -1., "stairs_down": -1.}.get(spec.get("kind"), 0.)
        self.heading = None
        if env.cfg.get("scut_heading_command", {}).get("enabled") and not env.cfg.get("evaluation_exact_cases"):
            from .heading_command import HeadingCommand, configure_heading
            self.heading = HeadingCommand(env, self.batches, configure_heading(env.cfg, root))
        if self.large_stair_curricula and (not env.cfg.get("step_jump_task", {}).get("enabled")
                or (self.heading is not None and set(self.heading.names) & set(self.large_stair_curricula))):
            raise ValueError("Ordinary large stairs require native assessment and exclusive command ownership")

    def sample(self, ids, *, reset_height=True):
        env = self.env
        envelope = (difficulty_envelope(self.difficulty_cfg, stage_actor_updates(env))
                    if self.difficulty_enabled else None)
        integrated = bool(env.cfg.get("integrated_runtime_candidate")) and not env.cfg.get("evaluation_exact_cases")
        if reset_height:
            self.sequence_cursor[ids] = 0
        selected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        selected[ids] = True
        heading_yaw = self.yaw() if self.heading is not None else None
        for members, spec in self.batches:
            group = members[selected[members]]
            if not len(group) or "command" not in spec:
                continue
            if spec.get("kind") == "release_stop" and not reset_height:
                elapsed = env.episode_length_buf[group] * env.policy_dt
                cmd = self.release_stop.target(group, elapsed)
                env.command_target[group] = cmd[:, :2]
                env.command_clock[group] = 1e9
                continue
            if integrated:
                from .integrated_runtime import row_difficulty_envelope
                envelope = row_difficulty_envelope(env, spec["group_name"])
            schedule_cfg = (self.schedule.context(spec["group_name"], height_sampling=True)[0]
                            if integrated else self.schedule.cfg)
            command = spec["command"]
            if not env.cfg.get("evaluation_exact_cases"):
                command = scheduled_command(command, spec.get("command_curriculum"),
                    actor_update_clock(env) if self.schedule.enabled else
                    env.training_transitions / env.cfg.get("curriculum_reference_batch", 4096 * 24))
                if getattr(env, "performance_curriculum", None) is not None:
                    command = env.performance_curriculum.limit_command(spec["group_name"], command)
            cmd = env.commands.new_tensor(command).expand(len(group), -1).clone()
            if reset_height:
                self.height_phase_offset[group] = 0.
                if spec["group_name"] in self.large_stair_curricula:
                    self.large_stair_phase_offset[group] = 5. * env.random(len(group))
                self.start_stop_delay[group] = 0.
                if self.start_stop_delay_bounds is not None and spec.get("kind") in ("start_stop", "velocity_curve"):
                    low, high = self.start_stop_delay_bounds
                    self.start_stop_delay[group] = low + (high - low) * env.random(len(group))
                self.stable[group] = False
                if env.cfg.get("command_reference") and spec.get("kind") in ("step_up", "stairs", "step_down", "stairs_down"):
                    settings = env.cfg["command_reference"]
                    start = settings["step_request_range_s"]
                    hold = settings["step_hold_range_s"]
                    if env.cfg.get("evaluation_exact_cases"):
                        self.manual_step_start[group] = spec.get("manual_request_seconds", sum(start) / 2)
                        self.manual_step_duration[group] = spec.get("manual_hold_seconds", sum(hold) / 2)
                    else:
                        self.manual_step_start[group] = start[0] + (start[1] - start[0]) * env.random(len(group))
                        self.manual_step_duration[group] = hold[0] + (hold[1] - hold[0]) * env.random(len(group))
            else:
                cmd[:, 2] = env.commands[group, 2]
            if reset_height and spec.get("height_motion"):
                settings = spec["height_motion"]
                if settings.get("randomize_phase") and not env.cfg.get("evaluation_exact_cases"):
                    span = 3 * settings["transition_seconds"] + 2 * settings["dwell_seconds"]
                    self.height_phase_offset[group] = env.random(len(group)) * span
                cmd[:, 2], self.height_velocity_reference[group] = height_reference(self.height_phase_offset[group], spec)
            if reset_height and spec.get("height_sampling") and not env.cfg.get("evaluation_exact_cases"):
                settings = bounded_height_sampling(spec, envelope)
                low, high = settings["range_m"]
                if (self.schedule.enabled and schedule_cfg.get("stage", "S1") == "S1"
                        and schedule_cfg.get("height_sampling_schedule")):
                    cmd[:, 2] = self.schedule.sample_height(len(group), [low, high], group=spec["group_name"])
                else:
                    pick = env.random(len(group))
                    sampled = (self.schedule.sample_continuous_height(len(group), settings)
                               if self.schedule.enabled else low + (high - low) * env.random(len(group)))
                    nominal = settings["nominal_fraction"]
                    endpoints = settings["endpoint_fraction"]
                    sampled = torch.where(pick < nominal + endpoints / 2, low, sampled)
                    sampled = torch.where((pick >= nominal + endpoints / 2) & (pick < nominal + endpoints), high, sampled)
                    cmd[:, 2] = torch.where(pick < nominal, cmd[:, 2], sampled)
            if reset_height and spec.get("height_transition_seconds"):
                self.height_target[group] = cmd[:, 2]
                self.height_velocity_reference[group] = 0.
                probability = spec.get("height_sampling", {}).get("start_at_target_fraction", 0.)
                if probability and not env.cfg.get("evaluation_exact_cases"):
                    self.height_phase_offset[group] = (env.random(len(group)) < probability) * spec["height_transition_seconds"]
                cmd[:, 2] = torch.where(self.height_phase_offset[group] > 0., cmd[:, 2], .305)
            heading = self.heading if self.heading is not None and spec["group_name"] in self.heading.names else None
            standing = torch.empty(len(group), device=env.device, dtype=torch.bool) if heading is not None else None
            sample_planar_command(cmd, spec, env.random, evaluation=env.cfg.get("evaluation_exact_cases", False),
                standing_out=standing)
            if heading is not None:
                heading.capture(group, cmd, standing, envelope, spec, env.random)
                cmd = heading.resolve(group, heading_yaw[group], cmd[:, 2], transformed=False)
            if spec.get("kind") == "release_stop":
                sample_release_stop_command(cmd, spec, env.random)
            scale_planar_command(cmd, spec, envelope)
            curriculum = getattr(env, "performance_curriculum", None)
            changed = torch.zeros(len(group), dtype=torch.bool, device=env.device)
            activation = env.cfg.get("special_mode_activation", {}).get(spec.get("kind"))
            if activation and not env.cfg.get("evaluation_exact_cases"):
                reference_update = (actor_update_clock(env) if self.schedule.enabled else
                    env.training_transitions / env.cfg.get("curriculum_reference_batch", 4096 * 24))
                active = (self.stable[group] & (env.episode_length_buf[group] * env.policy_dt >= activation["min_episode_seconds"])
                          & (reference_update >= activation["start_reference_updates"]))
                changed = self.spin[group] != active
                self.spin[group] = active
                if curriculum is not None:
                    curriculum.reset_episodes(group[changed])
            if not env.cfg.get("evaluation_exact_cases") and hasattr(curriculum, "sample_commands"):
                cmd = curriculum.sample_commands(spec["group_name"], group, cmd, env.random,
                    reset=True if reset_height else changed)
                if activation:
                    curriculum.env_pool[group] = torch.where(self.spin[group], curriculum.env_pool[group], 0)
            if activation and not env.cfg.get("evaluation_exact_cases"):
                cmd[~self.spin[group], :2] = 0.
            if env.cfg.get("motion_limits"):
                from .motion_limits import project_commands
                cmd = project_commands(cmd, env.cfg["motion_limits"])
            self.command_base[group] = cmd
            if spec.get("kind") == "release_stop":
                end_ticks = self.release_stop.reset(group, cmd, env.random)
                declared_ticks = round(spec["episode_seconds"] / env.policy_dt)
                if bool((end_ticks > declared_ticks).any()):
                    raise ValueError("Release-stop dwell exceeds its declared episode duration")
                cmd = self.release_stop.target(group, env.episode_length_buf[group] * env.policy_dt)
            if self.start_stop_delay_bounds is not None and spec.get("kind") in ("start_stop", "velocity_curve"):
                # Delay the actual motion, not the episode clock: every brake
                # follows its own physically executed acceleration/motion.
                time = env.episode_length_buf[group] * env.policy_dt - self.start_stop_delay[group]
                cmd = profile_command(time.clamp_min(0.), cmd, spec)
                cmd[time < 0., :2] = 0.
            env.commands[group] = cmd
            env.command_target[group] = cmd[:, :2]
            env.mode[group] = spec.get("mode", int(any(value != 0 for value in command[:2])))
            if reset_height and spec.get("kind") in ("jump", "running_jump"):
                apex = self.schedule.sample_jump(group, spec)
                if self.schedule.enabled and getattr(env, "full_tasks", None) is not None:
                    for key, values in (("jump_apex_delta_m", apex),
                                        ("jump_com_rise_m", (apex - .015).clamp_min(.005))):
                        previous = env.full_tasks.cfg.get(key, .06 if key == "jump_apex_delta_m" else .005)
                        tensor = torch.as_tensor(previous, device=env.device).expand(env.num_envs).clone()
                        tensor[group] = values
                        env.full_tasks.cfg[key] = tensor
            env.command_clock[group] = 1e9
            scheduled_duration = (self.schedule.command_duration(len(group), spec.get("command_resampling_seconds"),
                per_skill=spec.get("command_resampling_scope") == "skill", group=spec["group_name"])
                                  if spec.get("kind") != "release_stop" else None)
            if scheduled_duration is not None:
                env.command_clock[group] = scheduled_duration
            elif spec.get("command_resampling_seconds") and not env.cfg.get("evaluation_exact_cases"):
                low, high = spec["command_resampling_seconds"]
                env.command_clock[group] = low + (high - low) * env.random(len(group))
            env.height_clock[group] = 1e9
            if reset_height:
                env.push_enabled[group] = spec.get("push_m_s", 0.) > 0
                env.push_clock[group] = spec.get("push_at_s", 3.) if spec.get("push_m_s", 0.) else 1e9
            if reset_height and env.cfg.get("signal_perturbations", {}).get("enabled") and not env.cfg.get("evaluation_exact_cases"):
                env.push_enabled[group] = env.random(len(group)) < .5
                env.push_clock[group] = 3. + env.random(len(group))
            self.reference_target[group] = env.commands.new_tensor(spec.get("reference_velocity", [0., 0.]))
            self.reference_target[group] *= self.spin[group, None]
            self.reference_filtered[group if reset_height else group[changed]] = 0.
            if not env.cfg.get("evaluation_exact_cases") and "episode_seconds" in spec:
                env.episode_limits[group] = round(spec["episode_seconds"] / env.policy_dt)
                if self.start_stop_delay_bounds is not None and spec.get("kind") in ("start_stop", "velocity_curve"):
                    env.episode_limits[group] += (self.start_stop_delay[group] / env.policy_dt).ceil().to(env.episode_limits.dtype)
        self.apply_command_sequence(ids)
        self.apply_large_stair_commands(ids)

    def apply_large_stair_commands(self, selected_ids=None):
        if not self.large_stair_curricula:
            return
        from .integrated_runtime import component_actor_updates
        env = self.env
        clock = component_actor_updates(env, "terrain")
        if self.large_stair_actor_clock is not None and clock < self.large_stair_actor_clock:
            raise ValueError("Ordinary stair component actor clock regressed")
        self.large_stair_actor_clock = clock
        selected = None
        if selected_ids is not None:
            selected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            selected[selected_ids] = True
        elapsed = env.episode_length_buf * env.policy_dt + self.large_stair_phase_offset
        for members, spec in self.batches:
            settings = self.large_stair_curricula.get(spec["group_name"])
            if settings is None:
                continue
            ids = members if selected is None else members[selected[members]]
            command, active = command_profile(elapsed[ids], settings, clock)
            self.large_stair_assessment[members] = active
            self.command_base[ids] = command
            env.command_target[ids] = command[:, :2]
            env.commands[ids, 2] = command[:, 2]
            env.mode[ids] = 0
            if selected_ids is not None:
                env.commands[ids] = command

    def large_stair_metrics(self):
        if not self.large_stair_curricula:
            return {}
        clock = self.large_stair_actor_clock
        return {"/curriculum/ordinary_large_stair/terrain_actor_updates": float(clock),
            **{"/curriculum/ordinary_large_stair/" + name + "/activation_fraction":
               activation_fraction(settings, clock) for name, settings in self.large_stair_curricula.items()}}

    def difficulty_runtime_state(self):
        result = dict(enabled=self.difficulty_enabled, settings=self.difficulty_cfg)
        audit = self.env.cfg.get("integrated_runtime_candidate")
        if audit and not self.env.cfg.get("evaluation_exact_cases"):
            result["row_components"] = {name: owner["difficulty_component"]
                for name, owner in audit["row_ownership"].items()}
        return result

    def verify_difficulty_samples(self):
        """Startup/resume probe of actual samples; never called in physics substeps."""
        state = self.difficulty_runtime_state()
        if not self.difficulty_enabled:
            return dict(**state, sample_check="exact_evaluation_bypass")
        if self.env.cfg.get("integrated_runtime_candidate"):
            from .integrated_runtime import row_difficulty_envelope, row_settings
            checks = {}
            for ids, spec in self.batches:
                name = spec["group_name"]
                envelope = row_difficulty_envelope(self.env, name)
                if envelope is None or not len(ids):
                    continue
                command = self.command_base[ids]
                maximum = command[:, :2].abs().amax(0).tolist()
                if (not bool(torch.isfinite(command).all()) or maximum[0] > envelope["vx_max_m_s"] + 1e-5
                        or maximum[1] > envelope["yaw_max_rad_s"] + 1e-5):
                    raise ValueError("Actual joint row commands bypass their component envelope")
                height = command[:, 2]
                if moving_spec(spec) and spec.get("height_sampling"):
                    low, high = envelope["motion_height_range_m"]
                    if float(height.min()) < low - 1e-5 or float(height.max()) > high + 1e-5:
                        raise ValueError("Actual joint row heights bypass their component envelope")
                _, clock = row_settings(self.env, name, "locomotion_difficulty")
                checks[name] = dict(actor_update=clock, envelope=envelope, sampled_planar_abs_max=maximum)
            return dict(**state, rows=checks, sample_check="passed")
        clock = stage_actor_updates(self.env)
        envelope = difficulty_envelope(self.difficulty_cfg, clock)
        command = self.command_base
        if not bool(torch.isfinite(command).all()):
            raise ValueError("Locomotion difficulty produced nonfinite commands")
        maximum = command[:, :2].abs().amax(dim=0).tolist()
        if (maximum[0] > envelope["vx_max_m_s"] + 1e-5
                or maximum[1] > envelope["yaw_max_rad_s"] + 1e-5):
            raise ValueError("Actual locomotion commands bypass the graded envelope")
        members = [ids for ids, spec in self.batches if len(ids) and moving_spec(spec)
                   and spec.get("height_sampling")]
        sampled_height = None
        if members:
            ids = torch.cat(members)
            values = self.height_target[ids]
            sampled_height = [float(values.min()), float(values.max())]
            low, high = envelope["motion_height_range_m"]
            if not bool(torch.isfinite(values).all()) or sampled_height[0] < low - 1e-5 or sampled_height[1] > high + 1e-5:
                raise ValueError("Actual moving-height targets bypass the graded envelope")
        return dict(**state, actor_update=clock, envelope=envelope,
            sampled_planar_abs_max=maximum, sampled_motion_height_range_m=sampled_height,
            sample_check="passed")

    def difficulty_metrics(self):
        if not self.difficulty_enabled:
            return {}
        if self.env.cfg.get("integrated_runtime_candidate"):
            from .integrated_runtime import component_actor_updates
            settings = self.env.cfg["integrated_runtime_candidate"]["component_runtime"]["speed"]["locomotion_difficulty"]
            envelope = difficulty_envelope(settings, component_actor_updates(self.env, "speed"))
        else:
            envelope = difficulty_envelope(self.difficulty_cfg, stage_actor_updates(self.env))
        low, high = envelope["motion_height_range_m"]
        return {"/curriculum/locomotion/grade": envelope["grade"],
                "/curriculum/locomotion/vx_max_m_s": envelope["vx_max_m_s"],
                "/curriculum/locomotion/yaw_max_rad_s": envelope["yaw_max_rad_s"],
                "/curriculum/locomotion/motion_height_min_m": low,
                "/curriculum/locomotion/motion_height_max_m": high}

    def start_stop_timing_state(self):
        """Check the initialized runtime samples, not only the config switch."""
        env = self.env
        if self.start_stop_delay_bounds is not None:
            low, high = self.start_stop_delay_bounds
            count = 0
            for ids, spec in self.batches:
                if spec.get("kind") not in ("start_stop", "velocity_curve") or not len(ids):
                    continue
                count += len(ids)
                delay = self.start_stop_delay[ids]
                expected = round(spec["episode_seconds"] / env.policy_dt) + (delay / env.policy_dt).ceil()
                if (not torch.isfinite(delay).all() or (delay < low).any() or (delay > high).any()
                        or (high > low and len(ids) > 1 and delay.min() == delay.max())
                        or not torch.equal(env.episode_limits[ids], expected.to(env.episode_limits.dtype))
                        or (env.episode_length_buf[ids] != 0).any()
                        or (env.command_target[ids[delay > 0.]] != 0.).any()
                        or (env.commands[ids[delay > 0.], :2] != 0.).any()):
                    raise ValueError("Start/stop runtime must preserve the real wait and full profile horizon")
            if not count:
                raise ValueError("Start/stop timing declared without a sampled start/stop task")
        return {"enabled": self.start_stop_delay_bounds is not None,
                "start_delay_seconds": (list(self.start_stop_delay_bounds)
                    if self.start_stop_delay_bounds is not None else None),
                "policy_dt": env.policy_dt}

    def apply_command_sequence(self, selected_ids=None):
        """Expose timed commands using the existing mode/command observation ABI."""
        env = self.env
        elapsed = env.episode_length_buf * env.policy_dt
        selected = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        if selected_ids is not None:
            selected.zero_()
            selected[selected_ids] = True
        for members, spec in self.batches:
            sequence = spec.get("command_sequence", [])
            if not sequence:
                continue
            ids = members[selected[members]]
            cursor = self.sequence_cursor[ids].clone()
            for index, entry in enumerate(sequence):
                if entry.get("after_phase") not in (None, "ground_after_recovery"):
                    raise ValueError("Unsupported command sequence phase guard")
                ready = (cursor == index) & (elapsed[ids] + 1e-6 >= entry["time_s"])
                if entry.get("after_phase") == "ground_after_recovery":
                    verified = getattr(env, "successful_jump_count", torch.zeros_like(self.schedule.accepted_request_count))
                    ready &= ((self.schedule.completed_jump_count[ids] > 0)
                              & (env.phase.phase[ids] == Phase.GROUND)
                              & ~self.schedule.request_active[ids]
                              & (verified[ids] >= self.schedule.accepted_request_count[ids]))
                self.sequence_cursor[ids[ready]] += 1
            for index, entry in enumerate(sequence):
                current = ids[self.sequence_cursor[ids] == index + 1]
                if not len(current):
                    continue
                command = env.commands.new_tensor(entry["command"])
                self.command_base[current] = command
                env.command_target[current] = command[:2]
                env.commands[current, 2] = command[2]
                env.mode[current] = entry["mode"]

    def sequence_pending(self):
        pending = self.schedule.pending_explicit_requests()
        for members, spec in self.batches:
            sequence = spec.get("command_sequence", [])
            if sequence:
                pending[members] |= self.sequence_cursor[members] < len(sequence)
        return pending

    def reset_pose(self, ids, root, velocity):
        """Only episode initialization may place a robot in the air."""
        env = self.env
        self.schedule.reset(ids, velocity)
        if env.cfg.get("command_reference") and not env.cfg.get("evaluation_exact_cases"):
            selected = self.manual_step_mode[ids] != 0
            count = int(selected.sum())
            if count:
                low, high = env.cfg["command_reference"].get("step_start_x_range_m", (-1.5, -1.5))
                root[selected, 0] = env.origins[ids[selected], 0] + low + (high - low) * env.random(count)
        if not self.has_airborne_resets:
            return
        for row, index in enumerate(ids.tolist()):
            spec = self.specs.get(env.scene_groups[index], {})
            if spec.get("kind") not in ("airborne", "landing"):
                continue
            drop = spec.get("drop_height_m", .08)
            if not env.cfg.get("evaluation_exact_cases") and spec.get("drop_height_range_m"):
                low, high = spec["drop_height_range_m"]
                drop = low + (high - low) * env.random(1)[0]
            root[row, 2] += drop
            pitch = spec.get("reset_pitch_rad", .08)
            root[row, 3:] = root.new_tensor([0., math.sin(pitch / 2), 0., math.cos(pitch / 2)])
            velocity[row, 0] = spec.get("reset_vx_m_s", 0.)
            velocity[row, 2] = -spec.get("descent_speed_m_s", .1)

    def reset_phase(self, ids):
        airborne = ids[self.landing[ids]]
        self.env.phase.phase[airborne] = Phase.FLIGHT

    def update(self):
        env = self.env
        elapsed = env.episode_length_buf * env.policy_dt
        yaw = self.yaw()
        for ids, spec in self.batches:
            if not len(ids):
                continue
            kind = spec.get("kind", "constant")
            if spec.get("height_motion"):
                env.commands[ids, 2], self.height_velocity_reference[ids] = height_reference(elapsed[ids] + self.height_phase_offset[ids], spec)
            elif spec.get("height_transition_seconds"):
                duration = spec["height_transition_seconds"]
                u = ((elapsed[ids] + self.height_phase_offset[ids]) / duration).clamp(0., 1.)
                delta = self.height_target[ids] - .305
                env.commands[ids, 2] = .305 + delta * (3 * u.square() - 2 * u.pow(3))
                self.height_velocity_reference[ids] = delta * 6 * u * (1 - u) / duration
            elif kind in ("height", "height_pulse"):
                cmd = profile_command(elapsed[ids], spec["command"], spec)
                env.commands[ids, 2] = cmd[:, 2]
            if kind in ("start_stop", "weave", "velocity_curve"):
                time = elapsed[ids] - self.start_stop_delay[ids]
                cmd = profile_command(time.clamp_min(0.), self.command_base[ids], spec)
                cmd[time < 0., :2] = 0.
                env.command_target[ids] = cmd[:, :2]
            elif kind == "release_stop":
                env.command_target[ids] = self.release_stop.target(ids, elapsed[ids])[:, :2]
            elif kind == "spin_translate":
                target = self.reference_target[ids]
                env.command_target[ids, 0] = yaw[ids].cos() * target[:, 0] + yaw[ids].sin() * target[:, 1]
                env.command_target[ids, 1] = self.command_base[ids, 1]
        self.apply_command_sequence()
        if self.heading is not None:
            self.heading.update(yaw)
        self.apply_large_stair_commands()

    def yaw(self):
        q = self.env.robot.data.root_link_pose_w.torch[:, 3:]
        return torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                           1 - 2 * (q[:, 1].square() + q[:, 2].square()))

    def observation_targets(self):
        env = self.env
        yaw = self.yaw()
        target = self.reference_target
        ids = self.spin
        env.targets[ids, 0] = yaw[ids].cos() * target[ids, 0] + yaw[ids].sin() * target[ids, 1]
        env.targets[ids, 1] = -yaw[ids].sin() * target[ids, 0] + yaw[ids].cos() * target[ids, 1]

    def velocity_reward(self, velocity, normal_reward):
        measured = reference_velocity(velocity, self.yaw())
        # A wheel pair cannot instantaneously realize lateral body velocity.
        # Evaluate translation over a spin timescale rather than rewarding slip.
        period = math.tau / self.command_base[:, 1].abs().clamp_min(1.)
        alpha = self.env.policy_dt / (period + self.env.policy_dt)
        self.reference_filtered += alpha[:, None] * (measured - self.reference_filtered)
        error = (self.reference_filtered - self.reference_target).square().sum(-1)
        return torch.where(self.spin, 2 * torch.exp(-error / .25), normal_reward)

    def completion(self, contact, stable):
        env = self.env
        recovered = (env.phase.phase == Phase.RECOVERY) | (env.phase.phase == Phase.GROUND)
        return self.landing & contact.all(-1) & stable & recovered

    def diagnostics(self, velocity, *, omega=None, height_error=None, gravity=None):
        env = self.env
        error = reference_velocity(velocity, self.yaw()) - self.reference_target
        elapsed = env.episode_length_buf * env.policy_dt
        stopped = torch.zeros_like(self.spin)
        stop_requested = torch.zeros_like(self.spin)
        waiting = torch.zeros_like(self.spin)
        profile_time = torch.zeros_like(elapsed)
        for ids, spec in self.batches:
            if spec.get("kind") in ("start_stop", "velocity_curve"):
                segment = spec.get("segment_seconds", 3.)
                settle = max(spec.get("stop_settle_seconds", 1.), spec.get("velocity_transition_seconds", 0.))
                time = elapsed[ids] - self.start_stop_delay[ids]
                waiting[ids] = time < 0.
                profile_time[ids] = time.clamp_min(0.)
                stop_requested[ids] = (time >= 0.) & (torch.floor(time / segment).long() % 2 == 1)
                stopped[ids] = stop_requested[ids] & (time % segment > settle)
        release_diagnostics = {}
        if self.release_stop is not None:
            release_diagnostics = self.release_stop.diagnostics(elapsed, env.commands, velocity,
                omega=omega, height_error=height_error, gravity=gravity)
            stop_requested |= release_diagnostics["release_requested"]
            stopped |= release_diagnostics["release_stop_stationary_window"]
            waiting |= release_diagnostics["release_stop_waiting"]
            profile_time = torch.where(self.release_stop.active,
                release_diagnostics["release_stop_profile_time_s"], profile_time)
        sequence_length = torch.zeros_like(self.sequence_cursor)
        for ids, spec in self.batches:
            sequence_length[ids] = len(spec.get("command_sequence", []))
        climb_diagnostics = {}
        if self.large_stair_curricula:
            eligible = torch.zeros_like(self.large_stair_assessment)
            for ids, spec in self.batches:
                if spec["group_name"] in self.large_stair_curricula:
                    eligible[ids] = True
            climb_diagnostics = dict(ordinary_large_stair_pretraining=eligible & ~self.large_stair_assessment,
                ordinary_large_stair_assessing=eligible & self.large_stair_assessment)
        return {"reference_velocity_error_vector": error * self.spin[:, None],
                "settled_stop_speed": velocity[:, :2].norm(dim=-1) * stopped,
                "braking_window": stop_requested & ~stopped,
                "settled_stop_window": stopped,
                "start_stop_waiting": waiting,
                "start_stop_profile_time_s": profile_time,
                "start_stop_start_delay_s": self.start_stop_delay.clone(),
                "command_sequence_steps": self.sequence_cursor.clone(),
                "command_sequence_length": sequence_length,
                **release_diagnostics,
                **climb_diagnostics,
                **self.schedule.diagnostics()}
