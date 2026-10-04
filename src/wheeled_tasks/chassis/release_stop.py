"""One reset-sampled drive followed by an abrupt zero request and long dwell.

This module produces targets and diagnostics only. Actual commands continue
through the existing frozen acceleration limits; it never writes actuator
effort, physical state, observations or rewards.
"""
from copy import deepcopy
import math

import torch


def release_stop_settings(enabled=True):
    if type(enabled) is not bool:
        raise ValueError("Release-stop enabled must be bool")
    return dict(version="release-stop-v1", enabled=enabled, training_only=True,
        clock="episode_elapsed_seconds", sample_at="episode_reset_only",
        request_semantics="abrupt_zero_target_existing_command_slew",
        initial_wait_seconds=[0., 4.], drive_seconds=[4., 8.],
        zero_request_dwell_seconds=[12., 16.], stationary_window_seconds=10.,
        stationary_warmup_seconds=2., command_epsilon=.001,
        settled_speed_m_s=.03, settled_yaw_rad_s=.05,
        settled_height_error_m=.01, settled_tilt_rad=.1,
        maximum_drive_count=1, unchanged_slew_m_s2_rad_s2=[1.5, 4.])


def validate_release_stop(settings):
    if not isinstance(settings, dict) or settings != release_stop_settings(settings.get("enabled")):
        raise ValueError("Release-stop settings differ from the explicit v1 rule")
    return deepcopy(settings)


def release_stop_runtime_expectation(settings, policy_dt):
    validate_release_stop(settings)
    if not math.isfinite(policy_dt) or policy_dt <= 0:
        raise ValueError("Release-stop policy dt must be positive")
    return dict(enabled=settings["enabled"], version=settings["version"], policy_dt=policy_dt,
        target_release_is_abrupt=True, actual_commands_use_existing_slew=True,
        reset_only_sample=True, maximum_drive_count=1, settings=deepcopy(settings))


def apply_release_stop_contract(config, domain):
    """Append an explicit 5% pool; preserve every old spec and relative quota."""
    if domain not in ("speed", "terrain") or "release_stop" in config:
        raise ValueError("Release-stop contract patch needs an unpatched speed/terrain worker")
    exact = bool(config.get("evaluation_exact_cases"))
    config["release_stop"] = release_stop_settings(enabled=not exact)
    vx_range, yaw_range = ([.5, 3.], [1., 6.]) if domain == "speed" else ([.3, 1.5], [.5, 2.])
    vx, yaw = vx_range[1], yaw_range[1]
    terrains = ["flat"] if domain == "speed" else ["flat", "rough", "material", "slope"]
    limits = deepcopy(config["terrain_limits"])
    if domain == "terrain":
        limits["slope_deg"] = min(limits.get("slope_deg", 8.), 8.)
        limits["roughness_m"] = min(limits.get("roughness_m", .02), .02)
    height = next((deepcopy(spec["height_sampling"]) for spec in config["skill_specs"].values()
                   if "height_sampling" in spec),
                  dict(range_m=[.23, .43], nominal_fraction=0., endpoint_fraction=.2))
    height["nominal_fraction"] = 0.
    new_groups = []
    for axis, command in (("translate", [vx, 0., .305]), ("yaw", [0., yaw, .305]),
                          ("combined", [vx, yaw, .305])):
        for continuous in (False, True):
            name = "release_stop_" + axis + ("__height_train" if continuous else "")
            if name in config["skill_specs"]:
                raise ValueError("Release-stop group name already exists")
            spec = dict(kind="release_stop", command=command, mode=1,
                sample_amplitude=False,
                sample_vx_sign=True, sample_yaw_sign=True, push_m_s=0.,
                release_stop_magnitude_ranges=dict(vx_m_s=list(vx_range), yaw_rad_s=list(yaw_range)),
                terrain_limits=deepcopy(limits), episode_seconds=32.)
            if continuous:
                spec.update(height_sampling=deepcopy(height), height_transition_seconds=4.)
            config["skill_specs"][name] = spec
            config["behavior_pool_membership"][name] = "release_stop"
            new_groups.append(dict(name=name, fraction=.05 / 6, terrain=list(terrains)))
    if not exact:
        for group in config["scene_groups"]:
            group["fraction"] *= .95
        config["scene_groups"].extend(new_groups)
        for pool in config["behavior_pool_fractions"]:
            config["behavior_pool_fractions"][pool] *= .95
        config["behavior_pool_fractions"]["release_stop"] = .05
    return config


def sample_release_stop_command(command, spec, random):
    """Independent magnitudes and signs; inactive axes remain exactly zero."""
    ranges = spec["release_stop_magnitude_ranges"]
    for axis, key in enumerate(("vx_m_s", "yaw_rad_s")):
        low, high = ranges[key]
        if not (0 < low <= high and math.isfinite(high)):
            raise ValueError("Release-stop magnitudes need positive finite bounds")
        if spec["command"][axis] != 0.:
            magnitude = low + (high - low) * random(len(command))
            sign = torch.where(random(len(command)) < .5, -1., 1.)
            command[:, axis] = magnitude * sign
        else:
            command[:, axis] = 0.
    return command


class ReleaseStop:
    """Finite per-lane target clock, compatible with partial episode resets."""
    def __init__(self, count, device, settings, policy_dt):
        self.settings = validate_release_stop(settings)
        self.count, self.device, self.policy_dt = count, torch.device(device), policy_dt
        release_stop_runtime_expectation(settings, policy_dt)
        self.enabled = settings["enabled"]
        self.active = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.wait = torch.zeros(count, device=self.device)
        self.drive = torch.zeros(count, device=self.device)
        self.requested_dwell = torch.zeros(count, device=self.device)
        self.dwell = torch.zeros(count, device=self.device)
        self.brake_budget = torch.zeros(count, device=self.device)
        self.release_at = torch.zeros(count, device=self.device)
        self.end_at = torch.zeros(count, device=self.device)
        self.drive_command = torch.zeros(count, 3, device=self.device)

    def reset(self, ids, command, random):
        """Select timing once; command is already sampled/projected by skills."""
        if not self.enabled:
            raise ValueError("Disabled release-stop cannot sample a training trajectory")
        if command.shape != (len(ids), 3):
            raise ValueError("Release-stop drive command must be [reset_count,3]")
        self.active[ids] = True
        self.drive_command[ids] = command
        for key, bounds in (("wait", "initial_wait_seconds"), ("drive", "drive_seconds"),
                            ("requested_dwell", "zero_request_dwell_seconds")):
            low, high = self.settings[bounds]
            getattr(self, key)[ids] = low + (high - low) * random(len(ids))
        rates = command.new_tensor(self.settings["unchanged_slew_m_s2_rad_s2"])
        self.brake_budget[ids] = (command[:, :2].abs() / rates).amax(-1)
        quiet = self.settings["stationary_warmup_seconds"] + self.settings["stationary_window_seconds"]
        self.dwell[ids] = torch.maximum(self.requested_dwell[ids], self.brake_budget[ids] + quiet)
        self.release_at[ids] = self.wait[ids] + self.drive[ids]
        self.end_at[ids] = self.release_at[ids] + self.dwell[ids]
        return (self.end_at[ids] / self.policy_dt).ceil().long()

    def target(self, ids, elapsed):
        """One drive interval only; both axes request exact zero after release."""
        driving = (elapsed >= self.wait[ids]) & (elapsed < self.release_at[ids])
        result = self.drive_command[ids].clone()
        result[:, :2] = torch.where(driving[:, None], result[:, :2], 0.)
        return result

    def diagnostics(self, elapsed, commands, velocity, *, omega=None, height_error=None, gravity=None):
        waiting = self.active & (elapsed < self.wait)
        driving = self.active & (elapsed >= self.wait) & (elapsed < self.release_at)
        released = self.active & (elapsed >= self.release_at)
        epsilon = self.settings["command_epsilon"]
        command_zero = (commands[:, :2].abs() < epsilon).all(-1)
        at_drive_target = (commands[:, :2] - self.drive_command[:, :2]).abs().amax(-1) < epsilon
        minimum_brake_complete = elapsed + 1e-6 >= self.release_at + self.brake_budget
        speed_quiet = velocity[:, :2].norm(dim=-1) <= self.settings["settled_speed_m_s"]
        stationary_window = released & minimum_brake_complete & command_zero
        settled = stationary_window & speed_quiet
        if omega is not None:
            settled &= omega[:, 2].abs() <= self.settings["settled_yaw_rad_s"]
        if height_error is not None:
            settled &= height_error.abs() <= self.settings["settled_height_error_m"]
        if gravity is not None:
            settled &= (-gravity[:, 2]).clamp(-1., 1.).acos() <= self.settings["settled_tilt_rad"]
        return dict(release_stop_active=self.active.clone(), release_requested=released,
            release_stop_waiting=waiting, release_stop_driving=driving,
            release_stop_accelerating=driving & ~at_drive_target,
            release_stop_cruising=driving & at_drive_target,
            release_stop_decelerating=released & ~command_zero,
            release_stop_braking=released & ~settled, release_stop_settled=settled,
            release_stop_stationary_window=stationary_window,
            release_stop_wait_s=self.wait.clone(), release_stop_drive_s=self.drive.clone(),
            release_stop_profile_time_s=(elapsed - self.wait).clamp_min(0.) * self.active,
            release_stop_release_at_s=self.release_at.clone(), release_stop_dwell_s=self.dwell.clone(),
            release_stop_brake_budget_s=self.brake_budget.clone(), release_stop_end_at_s=self.end_at.clone())

    def runtime_state(self):
        return release_stop_runtime_expectation(self.settings, self.policy_dt)
