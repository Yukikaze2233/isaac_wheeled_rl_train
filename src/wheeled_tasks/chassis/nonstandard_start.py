"""Reset-only closed-chain leg poses followed by a finite passive release.

Only the six active motor commands are suppressed. The caller keeps gravity,
passive gas springs, contact and actuator/transport history running normally.
The whole interval mask uses the existing RecoveryRolloutMask protocol; this
module does not change observations, actions, rewards or the PPO algorithm.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import torch


PROFILE_FILE = "contracts/v6_nonstandard_start_profiles_v1.json"
PROFILE_SHA256 = "2368ad15775e1974845f918bc74f9a2d756ee3f1940bf8f32c90941a1b937a0e"
ASSET_MANIFEST_SHA256 = "875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba"
LEVELS = ("nominal", "gentle", "wide")


def nonstandard_start_settings(origin_actor_update, enabled=True):
    """Create an explicit opt-in; an absent environment setting stays disabled."""
    if type(origin_actor_update) is not int or origin_actor_update < 0:
        raise ValueError("Nonstandard birth origin must be a nonnegative integer")
    if type(enabled) is not bool:
        raise ValueError("Nonstandard birth enabled must be bool")
    return {
        "version": "v6_nonstandard_start_v1",
        "enabled": enabled,
        "training_only": True,
        "clock": "domain_actor_updates",
        "origin_actor_update": origin_actor_update,
        "hold_seconds": .2,
        "capture_seconds": 1.5,
        "capture_vx_cap_m_s": 1.,
        "capture_yaw_cap_rad_s": 1.,
        "application": "zero_all_active_motor_effort_after_response",
        "state_writes": "once_at_episode_reset_only",
        "levels": list(LEVELS),
        "iteration_probability_schedule": [
            {"actor_update": 0, "probabilities": [.85, .15, 0.]},
            {"actor_update": 500, "probabilities": [.75, .20, .05]},
            {"actor_update": 1500, "probabilities": [.65, .20, .15]},
        ],
        "profiles_file": PROFILE_FILE,
        "profiles_sha256": PROFILE_SHA256,
        "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
    }


def validate_nonstandard_start(settings):
    expected = nonstandard_start_settings(0)
    if not isinstance(settings, dict) or settings.keys() != expected.keys():
        raise ValueError("Nonstandard birth settings require the complete v1 schema")
    for key in ("version", "training_only", "clock", "application", "state_writes",
                "levels", "profiles_file", "profiles_sha256", "asset_manifest_sha256"):
        if settings[key] != expected[key]:
            raise ValueError(f"Unsupported nonstandard birth {key}")
    if type(settings["enabled"]) is not bool:
        raise ValueError("Nonstandard birth enabled must be bool")
    origin = settings["origin_actor_update"]
    if type(origin) is not int or origin < 0:
        raise ValueError("Nonstandard birth origin must be a nonnegative integer")
    if settings["hold_seconds"] != .2:
        raise ValueError("Nonstandard birth v1 requires a 0.2 second passive hold")
    for key in ("capture_seconds", "capture_vx_cap_m_s", "capture_yaw_cap_rad_s"):
        if settings[key] != expected[key]:
            raise ValueError(f"Unsupported nonstandard birth {key}")
    schedule = settings["iteration_probability_schedule"]
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("Nonstandard birth needs an actor update schedule")
    previous = -1
    for point in schedule:
        if not isinstance(point, dict) or point.keys() != {"actor_update", "probabilities"}:
            raise ValueError("Invalid nonstandard birth schedule point")
        update, probabilities = point["actor_update"], point["probabilities"]
        if (type(update) is not int or update <= previous
                or not isinstance(probabilities, list) or len(probabilities) != 3
                or any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0
                       for p in probabilities)
                or abs(sum(probabilities) - 1.) > 1e-8):
            raise ValueError("Invalid nonstandard birth schedule probabilities")
        previous = update
    if schedule[0]["actor_update"] != 0:
        raise ValueError("Nonstandard birth schedule must start at relative update zero")


def birth_probabilities(settings, actor_update):
    """Absolute domain clock, with its sealed origin preserved across restarts."""
    if type(actor_update) is not int or actor_update < 0:
        raise ValueError("Nonstandard birth actor clock must be a nonnegative integer")
    relative = max(0, actor_update - settings["origin_actor_update"])
    schedule = settings["iteration_probability_schedule"]
    for left, right in zip(schedule, schedule[1:]):
        if relative < right["actor_update"]:
            alpha = (relative - left["actor_update"]) / (right["actor_update"] - left["actor_update"])
            return [(1 - alpha) * a + alpha * b
                    for a, b in zip(left["probabilities"], right["probabilities"])]
    return list(schedule[-1]["probabilities"])


def runtime_expectation(settings, policy_dt):
    validate_nonstandard_start(settings)
    if not math.isfinite(policy_dt) or policy_dt <= 0:
        raise ValueError("Nonstandard birth policy dt must be positive")
    return {
        "enabled": settings["enabled"],
        "version": settings["version"],
        "hold_seconds": settings["hold_seconds"],
        "capture_seconds": settings["capture_seconds"],
        "capture_vx_cap_m_s": settings["capture_vx_cap_m_s"],
        "capture_yaw_cap_rad_s": settings["capture_yaw_cap_rad_s"],
        "policy_dt": policy_dt,
        "clock": settings["clock"],
        "origin_actor_update": settings["origin_actor_update"],
        "state_writes": settings["state_writes"],
        "application": settings["application"],
        "whole_interval_learning_mask": True,
        "bootstrap_uses_returned_observation": True,
        "settings": deepcopy(settings),
    }


def support_height_for_footprint(surface_data, surface_valid, center_xy, bounds):
    """Exact maximum of the intersecting analytic terrain strips over XY.

    Surface columns are xmin, xmax, z_at_xmin, dx, dy. The affine maximum
    occurs at the clipped X endpoint chosen by dx and the Y bound chosen by dy.
    Testing only footprint corners would miss a raised strip inside its span.
    Heights and center XY use the same environment-local terrain frame.
    """
    if (surface_data.ndim != 3 or surface_data.shape[-1] != 5
            or surface_valid.shape != surface_data.shape[:2] or surface_valid.dtype != torch.bool
            or center_xy.shape != (surface_data.shape[0], 2)
            or bounds.shape != (surface_data.shape[0], 4)
            or any(t.device != surface_data.device for t in (surface_valid, center_xy, bounds))):
        raise ValueError("Nonstandard birth support needs terrain [N,S,5], mask [N,S], XY [N,2], bounds [N,4]")
    if surface_data.shape[1] == 0:
        return torch.zeros(surface_data.shape[0], device=surface_data.device, dtype=surface_data.dtype)
    x_lo = torch.maximum(center_xy[:, :1] + bounds[:, :1], surface_data[:, :, 0])
    x_hi = torch.minimum(center_xy[:, :1] + bounds[:, 1:2], surface_data[:, :, 1])
    intersects = (x_lo <= x_hi) & surface_valid
    x = torch.where(surface_data[:, :, 3] >= 0., x_hi, x_lo)
    y = center_xy[:, 1:2] + torch.where(surface_data[:, :, 4] >= 0., bounds[:, 3:4], bounds[:, 2:3])
    z = surface_data[:, :, 2] + (x - surface_data[:, :, 0]) * surface_data[:, :, 3] + y * surface_data[:, :, 4]
    maximum = z.masked_fill(~intersects, -torch.inf).amax(-1)
    return torch.where(intersects.any(-1), maximum, 0.)


class NonstandardStart:
    """A pose sampler and motor hold; never writes simulator state itself.

    Call reset() only for episode-reset IDs; initial_state() applies the chosen
    full configuration to local reset buffers. begin_policy(), apply_torque()
    after all response/transport filtering, and advance() once after each
    physical step define ownership. A partially held interval is excluded from
    actor and critic training. The next observation can already bootstrap RL.
    """

    def __init__(self, count, device, physics_dt, settings, root, joint_names,
                 *, evaluation=False, asset_manifest_sha256=None):
        validate_nonstandard_start(settings)
        if type(count) is not int or count < 1 or not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("Nonstandard birth needs a positive lane count and physics dt")
        steps = settings["hold_seconds"] / physics_dt
        if abs(steps - round(steps)) > 1e-7:
            raise ValueError("Nonstandard birth hold must contain whole physics steps")
        self.count, self.device, self.physics_dt = count, torch.device(device), physics_dt
        self.settings = deepcopy(settings)
        self.enabled = settings["enabled"] and not evaluation
        self.hold_steps = round(steps)
        capture_steps = settings["capture_seconds"] / physics_dt
        if abs(capture_steps - round(capture_steps)) > 1e-7:
            raise ValueError("Nonstandard birth capture must contain whole physics steps")
        self.capture_steps = round(capture_steps)
        if asset_manifest_sha256 is not None and asset_manifest_sha256 != settings["asset_manifest_sha256"]:
            raise ValueError("Nonstandard birth asset differs from the sealed geometry")
        path = root / settings["profiles_file"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != settings["profiles_sha256"]:
            raise ValueError("Nonstandard birth profile bank differs from its frozen hash")
        table = json.loads(raw)
        if (table.get("schema") != "v6_nonstandard_start_profiles_v1"
                or table.get("asset_manifest_sha256") != settings["asset_manifest_sha256"]
                or set(table.get("tree_joint_names", [])) != set(joint_names)
                or len(set(joint_names)) != len(joint_names)):
            raise ValueError("Nonstandard birth profile asset or joint set mismatch")
        self.profile_names = list(table["profiles"])
        positions, heights, footprints, level_ids = [], [], [], []
        for name in self.profile_names:
            row = table["profiles"][name]
            q, check = row["full_joint_positions"], row["geometry_validation"]
            if (not row["enabled"] or not row["geometry_valid"]
                    or row["state_writes"] != settings["state_writes"]
                    or row["root_quaternion_xyzw"] != [0., 0., 0., 1.]
                    or set(q) != set(joint_names)
                    or set(row["full_joint_velocities"].values()) != {0.}
                    or check["max_closure_gap_m"] > 1e-8
                    or check["minimum_terrain_clearance_m"] < .003
                    or any(not 40 - 1e-5 <= beta <= 105 + 1e-5 for beta in check["passive_knee_deg"])):
                raise ValueError(f"Invalid nonstandard birth full configuration: {name}")
            positions.append([q[joint] for joint in joint_names])
            heights.append(row["root_position_m"][2])
            footprints.append(row["collision_footprint_b_m"])
            level_ids.append(LEVELS.index(row["level"]))
        self.joint_positions = torch.tensor(positions, device=self.device)
        self.root_heights = torch.tensor(heights, device=self.device)
        self.footprints = torch.tensor(footprints, device=self.device)
        if not all(bool(torch.isfinite(t).all()) for t in (self.joint_positions, self.root_heights, self.footprints)):
            raise ValueError("Nonfinite nonstandard birth pose bank")
        self.bank_levels = torch.tensor(level_ids, dtype=torch.long, device=self.device)
        self.options = [torch.empty(0, dtype=torch.long, device=self.device)] + [
            (self.bank_levels == level).nonzero(as_tuple=True)[0] for level in (1, 2)]
        if any(len(self.options[level]) == 0 for level in (1, 2)):
            raise ValueError("Nonstandard birth bank must contain both levels")
        self.profile_ids = torch.full((count,), -1, dtype=torch.long, device=self.device)
        self.level_ids = torch.zeros(count, dtype=torch.long, device=self.device)
        self.remaining_steps = torch.zeros(count, dtype=torch.long, device=self.device)
        self.elapsed = torch.zeros(count, device=self.device)
        self.episode_steps = torch.zeros(count, dtype=torch.long, device=self.device)
        self.selected = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.held_interval = torch.zeros_like(self.selected)
        self.learning_mask = torch.ones_like(self.selected)
        self.profile_reset_counts = torch.zeros(len(self.profile_names), dtype=torch.long, device=self.device)
        self.level_reset_counts = torch.zeros(3, dtype=torch.long, device=self.device)
        self._policy_open = False

    @property
    def holding(self):
        return self.remaining_steps > 0

    @property
    def capturing(self):
        return self.selected & (self.episode_steps < self.capture_steps)

    def _ids(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or (len(ids) and (int(ids.min()) < 0 or int(ids.max()) >= self.count)):
            raise ValueError("Nonstandard birth reset IDs must be valid one-dimensional lane IDs")
        if len(ids) != len(torch.unique(ids)):
            raise ValueError("Nonstandard birth reset IDs must be unique")
        return ids

    def reset(self, ids, actor_update, *, eligible=None, generator=None):
        """Return bank row IDs (-1 is nominal), consuming RNG only at reset."""
        ids = self._ids(ids)
        probabilities = birth_probabilities(self.settings, actor_update)
        if eligible is None:
            eligible = torch.ones(self.count, dtype=torch.bool, device=self.device)
        if eligible.dtype != torch.bool or eligible.shape != (self.count,) or eligible.device != self.device:
            raise ValueError("Nonstandard birth eligibility must be bool [num_envs]")
        # This method may run in autoreset before extras are assembled. Preserve
        # the interval ownership result; reset changes only returned-state masks.
        self.profile_ids[ids] = -1
        self.level_ids[ids] = 0
        self.remaining_steps[ids] = 0
        self.elapsed[ids] = 0.
        self.episode_steps[ids] = 0
        self.selected[ids] = False
        lanes = ids[eligible[ids]] if self.enabled else ids[:0]
        if len(lanes):
            cdf = torch.tensor(probabilities, dtype=torch.float64, device=self.device).cumsum(0)
            cdf[-1] = 1.
            draw = torch.rand((len(lanes), 2), device=self.device, generator=generator)
            levels = torch.searchsorted(cdf, draw[:, 0].contiguous(), right=True).clamp_max(2)
            self.level_ids[lanes] = levels
            self.level_reset_counts += torch.bincount(levels, minlength=3)
            for level in (1, 2):
                mask = levels == level
                selected_lanes = lanes[mask]
                options = self.options[level]
                choices = options[(draw[mask, 1] * len(options)).long()]
                self.profile_ids[selected_lanes] = choices
                self.profile_reset_counts += torch.bincount(choices, minlength=len(self.profile_names))
            chosen = lanes[levels != 0]
            self.remaining_steps[chosen] = self.hold_steps
            self.selected[chosen] = True
        return self.profile_ids[ids].clone()

    def collision_footprints(self, ids):
        """Return selected body-frame XY bounds; nominal rows contain zeros."""
        ids = self._ids(ids)
        valid = self.profile_ids[ids] >= 0
        result = torch.zeros((len(ids), 4), device=self.device)
        result[valid] = self.footprints[self.profile_ids[ids][valid]]
        return result

    def initial_state(self, ids, root, joints, velocity, support_height, *, preserve_orientation=False):
        """Modify reset buffers once, above the caller's footprint support height.

        support_height is world Z of a conservative supporting plane over the
        selected collision footprint. A center-only query on rough terrain is
        insufficient. XY placement remains owned by the existing reset logic.
        """
        ids = self._ids(ids)
        if (root.shape != (len(ids), 7) or joints.shape != (len(ids), self.joint_positions.shape[1])
                or velocity.shape != (len(ids), 6) or support_height.shape != (len(ids),)
                or not bool(torch.isfinite(support_height).all())):
            raise ValueError("Invalid nonstandard birth reset buffers/support plane")
        valid = self.profile_ids[ids] >= 0
        bank = self.profile_ids[ids][valid]
        root[valid, 2] = support_height[valid] + self.root_heights[bank]
        if not preserve_orientation:
            root[valid, 3:7] = root.new_tensor([0., 0., 0., 1.])
        joints[valid] = self.joint_positions[bank]
        velocity[valid] = 0.

    def begin_policy(self):
        self.held_interval.zero_()
        self.learning_mask.copy_(~self.holding)
        self._policy_open = True

    def cap_commands(self, commands):
        """Cap a copy of visible commands or desired targets during capture.

        Keep the original skill command_target untouched. Pass its capped copy
        to the existing command slew, then cap visible commands before actor
        inference. When capture ends the same slew reaches the original target.
        Height, target resampling, route and acceleration limits stay intact.
        """
        if (commands.ndim != 2 or commands.shape[0] != self.count
                or commands.shape[1] not in (2, 3) or commands.device != self.device):
            raise ValueError("Nonstandard birth commands must be [N,2] or [N,3]")
        result = commands.clone()
        cap = commands.new_tensor([self.settings["capture_vx_cap_m_s"],
                                   self.settings["capture_yaw_cap_rad_s"]])
        result[:, :2] = torch.where(self.capturing[:, None],
                                   commands[:, :2].clamp(min=-cap, max=cap), commands[:, :2])
        return result

    def apply_torque(self, torque):
        """Final gate on [N,6] active effort, after PD and all drive response."""
        if not self._policy_open or torque.shape != (self.count, 6) or torque.device != self.device:
            raise ValueError("Nonstandard birth torque requires begin_policy and active effort [N,6]")
        held = self.holding
        self.held_interval |= held
        self.learning_mask &= ~held
        return torch.where(held[:, None], 0., torque)

    def advance(self):
        """Advance once after a physics step, including the last zero-torque step."""
        if not self._policy_open:
            raise ValueError("Nonstandard birth physics advance requires begin_policy")
        held = self.holding
        self.elapsed[held] += self.physics_dt
        self.remaining_steps.sub_(held.long())
        self.episode_steps.add_(self.selected.long())

    def effective_action_history(self, actions):
        if actions.shape != (self.count, 6) or actions.device != self.device:
            raise ValueError("Nonstandard birth action history must be [N,6]")
        return torch.where(self.held_interval[:, None], 0., actions)

    def bootstrap_mask(self, terminated):
        if terminated.shape != (self.count,) or terminated.dtype != torch.bool or terminated.device != self.device:
            raise ValueError("Nonstandard birth terminal mask must be bool [N]")
        return ~self.holding & ~terminated

    def runtime_state(self, policy_dt):
        result = runtime_expectation(self.settings, policy_dt)
        result["enabled"] = self.enabled
        return result

    def diagnostics(self):
        return {"nonstandard_start_" + key: value.clone() for key, value in {
            "selected": self.selected, "profile_id": self.profile_ids,
            "level_id": self.level_ids, "holding": self.holding,
            "hold_remaining_seconds": self.remaining_steps * self.physics_dt,
            "hold_elapsed_seconds": self.elapsed, "held_interval": self.held_interval,
            "capturing": self.capturing,
            "birth_elapsed_seconds": self.episode_steps * self.physics_dt,
            "learning_mask": self.learning_mask,
        }.items()}

    def metrics(self, actor_update):
        prefix = "/nonstandard_start/"
        result = {prefix + "relative_actor_updates": float(max(0, actor_update - self.settings["origin_actor_update"])),
                  prefix + "holding_fraction": self.holding.float().mean(),
                  prefix + "capturing_fraction": self.capturing.float().mean(),
                  prefix + "selected_fraction": self.selected.float().mean()}
        for i, (level, probability) in enumerate(zip(LEVELS, birth_probabilities(self.settings, actor_update))):
            result[prefix + level + "_probability"] = probability
            result[prefix + level + "_reset_count"] = self.level_reset_counts[i].clone()
        return result
