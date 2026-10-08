"""SCUT heading postprocessing on the native command lifecycle.

Heading equation: SCUTRobotLab b8ff79f3 commands.py:577–604 (MIT).
Wrap convention: Isaac Lab utils/math.py:100–122 (BSD-3-Clause).
Sampling, timers, RNG and physical control remain owned by SkillCommands/Env.
"""
from pathlib import Path

import torch

from .locomotion_difficulty import scale_planar_command
from .motion_limits import project_commands
from .scut_reuse_curriculum import COMMIT, NORMAL_KINDS, NORMAL_POOLS, SOURCES, _protected, read_sources


SCHEMA = "v6-scut-heading-command-v1"


def configure_heading(config, root=None):
    """Read the authenticated enabled source only for explicit training opt-in."""
    setting = config.get("scut_heading_command", {})
    if config.get("evaluation_exact_cases") or not setting.get("enabled", False):
        return None
    if setting.get("schema") != SCHEMA or setting.get("enabled") is not True:
        raise ValueError("Invalid SCUT heading opt-in schema")
    if not config.get("command_slew") or not config.get("motion_limits"):
        raise ValueError("SCUT heading requires existing command slew and motion limits")
    mapped = config.get("scut_reuse_candidate", {})
    profile = setting.get("profile")
    if (mapped.get("schema") != "scut-saved-command-reuse-v1"
            or setting.get("source_commit") != COMMIT or mapped.get("source_commit") != COMMIT
            or profile != mapped.get("profile")):
        raise ValueError("SCUT heading requires its authenticated native source map")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    sources = read_sources(root / "references/scut_curriculum_b8ff79f3")
    if profile not in sources:
        raise ValueError("Unknown saved SCUT heading profile")
    source = sources[profile]
    if (setting.get("source_files") != source["files"]
            or mapped.get("source_files") != source["files"]):
        raise ValueError("SCUT heading source pins differ from authenticated saved files")
    commands = source["commands"]
    if commands["heading_command"] is not True:
        raise ValueError("Saved SCUT heading command is not enabled")
    specs, membership = config["skill_specs"], config["behavior_pool_membership"]
    names = tuple(name for name, row in mapped["mapped_groups"].items() if row.get("kind") == "normal")
    if len(names) != 24:
        raise ValueError("SCUT heading requires the 24 declared ordinary groups")
    for name in names:
        spec, pool = specs.get(name), membership.get(name)
        if (spec is None or pool not in NORMAL_POOLS or spec.get("kind") not in NORMAL_KINDS
                or _protected(spec, pool) or spec.get("special_mode_activation")
                or spec.get("command_curriculum")):
            raise ValueError("SCUT heading owner overlaps a protected group: " + name)
        sampler = spec.get("planar_command_sampling", {})
        if (sampler.get("vx_range_m_s") != commands["ranges"]["lin_vel_x"]
                or sampler.get("yaw_range_rad_s") != commands["ranges"]["ang_vel_z"]
                or spec.get("standing_command_probability") != commands["rel_standing_envs"]
                or spec.get("command_resampling_scope") != "skill"
                or spec.get("command_resampling_seconds") != commands["resampling_time_range"]):
            raise ValueError("SCUT heading owner sampling differs from its source: " + name)
    if config.get("performance_curriculum"):
        raise ValueError("SCUT heading does not bypass a performance command producer")
    return dict(schema=SCHEMA, profile=profile, source_commit=COMMIT, source_files=source["files"],
        ordinary_names=names, heading_range=tuple(commands["ranges"]["heading"]),
        yaw_range=tuple(commands["ranges"]["ang_vel_z"]),
        stiffness=commands["heading_control_stiffness"], probability=commands["rel_heading_envs"])


def heading_source_files(config, root=None):
    """Declare every file read by this opt-in for the original source registry."""
    if configure_heading(config, root) is None:
        return []
    directory = "references/scut_curriculum_b8ff79f3"
    return ["src/wheeled_tasks/chassis/" + name + ".py" for name in (
        "heading_command", "scut_reuse_curriculum", "locomotion_difficulty", "motion_limits")] + [
        f"{directory}/pretrained/26_infantry/{profile}/{run}/params/{kind}.yaml"
        for profile, (run, _, _) in SOURCES.items() for kind in ("env", "agent")] + [
        f"{directory}/{name}" for name in ("source_manifest.json", "LICENSE", "THIRD_PARTY_NOTICES.md")]


def resolve_heading_commands(commands, *, heading_target, heading_w, is_heading_env,
                             is_standing_env, stiffness, yaw_range):
    """Resolve a compact ordinary-row tensor; leave height and inputs unchanged."""
    result = commands.clone()
    angles = heading_target - heading_w
    wrapped = (angles + torch.pi) % (2 * torch.pi)
    error = torch.where((wrapped == 0) & (angles > 0), torch.pi, wrapped - torch.pi)
    yaw = (stiffness * error).clamp(*yaw_range)
    result[:, 1] = torch.where(is_heading_env, yaw, result[:, 1])
    result[:, :2] = torch.where(is_standing_env[:, None], 0., result[:, :2])
    return result


class HeadingCommand:
    """Small state attachment; the original native sampler remains the only one."""

    def __init__(self, env, batches, settings):
        self.env, self.settings = env, settings
        self.names = frozenset(settings["ordinary_names"])
        if not self.names <= {spec["group_name"] for _, spec in batches}:
            raise ValueError("SCUT heading groups are absent from the native batches")
        self.ordinary_mask = torch.tensor([name in self.names for name in env.scene_groups],
            dtype=torch.bool, device=env.device)
        self.ids = self.ordinary_mask.nonzero(as_tuple=False).flatten()
        self.target = torch.zeros(env.num_envs, dtype=env.commands.dtype, device=env.device)
        self.is_heading = torch.zeros_like(self.ordinary_mask)
        self.is_standing = torch.zeros_like(self.ordinary_mask)
        self.raw_planar = torch.zeros_like(env.command_target)
        self.scales = torch.ones_like(env.command_target)
        self.feasibility_delta = torch.zeros_like(env.command_target)

    def validate_owners(self, *masks):
        for mask in masks:
            if mask is not None and bool((mask & self.ordinary_mask).any()):
                raise ValueError("SCUT heading overlaps a native special/FSM owner")

    def capture(self, ids, rawcommand, standing, envelope, spec, random):
        """Retain the original standing coin and draw only new heading metadata."""
        self.raw_planar[ids] = rawcommand[:, :2]
        self.is_standing[ids] = standing
        self.is_heading[ids] = False
        self.target[ids] = 0.
        selected = ids[~standing]
        if len(selected):
            low, high = self.settings["heading_range"]
            self.target[selected] = low + (high - low) * random(len(selected))
            self.is_heading[selected] = random(len(selected)) <= self.settings["probability"]
        scale = torch.ones_like(rawcommand)
        scale_planar_command(scale, spec, envelope)
        self.scales[ids] = scale[:, :2]

    def resolve(self, ids, heading_w, current_height, *, transformed=True):
        command = torch.cat((self.raw_planar[ids], current_height[:, None]), -1)
        command = resolve_heading_commands(command, heading_target=self.target[ids], heading_w=heading_w,
            is_heading_env=self.is_heading[ids], is_standing_env=self.is_standing[ids],
            stiffness=self.settings["stiffness"], yaw_range=self.settings["yaw_range"])
        if transformed:
            command[:, :2] *= self.scales[ids]
            command = project_commands(command, self.env.cfg["motion_limits"])
        return command

    def update(self, heading_w):
        desired = self.resolve(self.ids, heading_w[self.ids], self.env.commands[self.ids, 2])
        controlled = self.is_heading[self.ids] | self.is_standing[self.ids]
        self.env.command_target[self.ids] = torch.where(controlled[:, None],
            desired[:, :2], self.env.command_target[self.ids])

    def finish(self, commands, previous_planar, acceleration, limits):
        """Physical feasibility wins over slew; report the actual applied delta."""
        controlled = self.is_heading[self.ids] | self.is_standing[self.ids]
        before = commands[self.ids, :2]
        projected = project_commands(commands[self.ids], limits)[:, :2]
        applied = torch.where(controlled[:, None], projected, before)
        commands[self.ids, :2] = applied
        acceleration[self.ids] = torch.where(controlled,
            (applied[:, 0] - previous_planar[:, 0]) / self.env.policy_dt,
            acceleration[self.ids])
        self.feasibility_delta[self.ids] = applied - before

    def summary(self):
        return dict(schema=SCHEMA, profile=self.settings["profile"], source_commit=COMMIT,
            source_files=self.settings["source_files"], ordinary_groups=sorted(self.names),
            eligible_count=int(self.ordinary_mask.sum()), reference_dt=self.env.policy_dt,
            heading_range=list(self.settings["heading_range"]), yaw_range=list(self.settings["yaw_range"]),
            stiffness=self.settings["stiffness"], probability=self.settings["probability"],
            standing="existing single native draw; desired zero then native slew",
            final_projection="heading-owned lanes only; physical feasibility precedes slew bound",
            qualification=None)
