"""Single-skill commands and airborne resets for the V5 SCUT-style curriculum.

Reference-frame velocity projection and tracking follow SCUTRobotLab V14
wheelbipe_V14/env.py (MIT, commit b8ff79f). The fixed reference heading here
represents a virtual gimbal; the V5 chassis asset has no actuated gimbal.
"""
import math

import torch

from .task import Phase


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
    elif kind == "weave":
        segment = torch.floor(time / profile.get("segment_seconds", 1.)).long() % 4
        result[:, 1] *= torch.where((segment == 0) | (segment == 3), 1., -1.)
    return result


def reference_velocity(body_velocity, yaw):
    """Rotate planar body velocity into a fixed virtual-gimbal reference frame."""
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c * body_velocity[:, 0] - s * body_velocity[:, 1],
                        s * body_velocity[:, 0] + c * body_velocity[:, 1]), -1)


class SkillCommands:
    """Keep task sampling, command profiles and special resets out of physics."""

    def __init__(self, env):
        self.env = env
        specs = env.cfg.get("skill_specs", {})
        if env.cfg.get("evaluation_exact_cases"):
            specs = {c["name"]: {"command": c["command"], **c.get("skill", {})}
                     for c in env.cfg["evaluation"]["cases"]}
        self.specs = specs
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
        default_push = getattr(env, "stage_cfg", {}).get("push_max", 0.)
        self.push_speed = torch.full((env.num_envs,), float(default_push), device=env.device)
        self.command_base = torch.zeros(env.num_envs, 3, device=env.device)
        for ids, spec in self.batches:
            self.spin[ids] = spec.get("kind") == "spin_translate"
            self.landing[ids] = spec.get("kind") in ("airborne", "landing")
            self.height_motion[ids] = bool(spec.get("height_motion") or spec.get("height_transition_seconds"))
            self.push_speed[ids] = spec.get("push_m_s", default_push)

    def sample(self, ids, *, reset_height=True):
        env = self.env
        selected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        selected[ids] = True
        for members, spec in self.batches:
            group = members[selected[members]]
            if not len(group) or "command" not in spec:
                continue
            command = spec["command"]
            if not env.cfg.get("evaluation_exact_cases"):
                command = scheduled_command(command, spec.get("command_curriculum"),
                    env.training_transitions / env.cfg.get("curriculum_reference_batch", 4096 * 24))
                if getattr(env, "performance_curriculum", None) is not None:
                    command = env.performance_curriculum.limit_command(spec["group_name"], command)
            cmd = env.commands.new_tensor(command).expand(len(group), -1).clone()
            if reset_height:
                self.height_phase_offset[group] = 0.
                self.stable[group] = False
            else:
                cmd[:, 2] = env.commands[group, 2]
            if reset_height and spec.get("height_motion"):
                settings = spec["height_motion"]
                if settings.get("randomize_phase") and not env.cfg.get("evaluation_exact_cases"):
                    span = 3 * settings["transition_seconds"] + 2 * settings["dwell_seconds"]
                    self.height_phase_offset[group] = env.random(len(group)) * span
                cmd[:, 2], self.height_velocity_reference[group] = height_reference(self.height_phase_offset[group], spec)
            if reset_height and spec.get("height_sampling") and not env.cfg.get("evaluation_exact_cases"):
                settings = spec["height_sampling"]
                low, high = settings["range_m"]
                pick = env.random(len(group))
                sampled = low + (high - low) * env.random(len(group))
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
            if not env.cfg.get("evaluation_exact_cases") and spec.get("sample_amplitude", False):
                cmd[:, :2] *= (.4 + .6 * env.random(len(group)))[:, None]
            if (not env.cfg.get("evaluation_exact_cases") and spec.get("sample_yaw_sign", True)
                    and spec.get("kind") in ("rotate", "curve", "spin_translate")):
                cmd[:, 1] *= torch.where(env.random(len(group)) < .5, -1., 1.)
            curriculum = getattr(env, "performance_curriculum", None)
            changed = torch.zeros(len(group), dtype=torch.bool, device=env.device)
            activation = env.cfg.get("special_mode_activation", {}).get(spec.get("kind"))
            if activation and not env.cfg.get("evaluation_exact_cases"):
                reference_update = env.training_transitions / env.cfg.get("curriculum_reference_batch", 4096 * 24)
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
            env.commands[group] = cmd
            env.command_target[group] = cmd[:, :2]
            env.mode[group] = spec.get("mode", int(any(value != 0 for value in command[:2])))
            env.command_clock[group] = 1e9
            if spec.get("command_resampling_seconds") and not env.cfg.get("evaluation_exact_cases"):
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

    def reset_pose(self, ids, root, velocity):
        """Only episode initialization may place a robot in the air."""
        if not self.has_airborne_resets:
            return
        env = self.env
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
            elif kind == "height":
                cmd = profile_command(elapsed[ids], spec["command"], spec)
                env.commands[ids, 2] = cmd[:, 2]
            if kind in ("start_stop", "weave"):
                cmd = profile_command(elapsed[ids], self.command_base[ids], spec)
                env.command_target[ids] = cmd[:, :2]
            elif kind == "spin_translate":
                target = self.reference_target[ids]
                env.command_target[ids, 0] = yaw[ids].cos() * target[:, 0] + yaw[ids].sin() * target[:, 1]
                env.command_target[ids, 1] = self.command_base[ids, 1]

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

    def diagnostics(self, velocity):
        env = self.env
        error = reference_velocity(velocity, self.yaw()) - self.reference_target
        elapsed = env.episode_length_buf * env.policy_dt
        stopped = torch.zeros_like(self.spin)
        for ids, spec in self.batches:
            if spec.get("kind") == "start_stop":
                segment = spec.get("segment_seconds", 3.)
                stopped[ids] = (torch.floor(elapsed[ids] / segment).long() % 2 == 1) & (elapsed[ids] % segment > spec.get("stop_settle_seconds", 1.))
        return {"reference_velocity_error_vector": error * self.spin[:, None],
                "settled_stop_speed": velocity[:, :2].norm(dim=-1) * stopped}
