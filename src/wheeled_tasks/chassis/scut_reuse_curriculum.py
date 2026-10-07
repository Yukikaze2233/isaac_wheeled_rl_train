"""Opt-in command sampling from SCUT's saved, enabled V14 run settings.

Only ordinary training commands are mapped. Native geometry, dedicated skill
timing, rewards and learning state remain the caller's responsibility.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path


COMMIT = "b8ff79f3df855faf9dc92f4a282bd80c42649466"
SOURCES = {
    "rough_dash": ("2026-07-19_19-15-07",
        "da26c974517564480c8900b6adf339d1648c99902ea3d9bb86fa48387df302c4",
        "1dc253bcc36f7872e5f4885a2d48fb432c2cd5a149f20516318e298f8d2d3c05"),
    "rough_rotation_stair": ("2026-07-30_09-16-27",
        "a6117e55b3ebb5a03870981bd2d65d889c38850063008a9341d8346b997bb1ee",
        "bdf3cba49985e1683a8c43c9f1d28820ec9907ebadc6519f0022f2126f177d24"),
}
NORMAL_POOLS = frozenset({"translation", "yaw", "basic_translation", "basic_yaw"})
NORMAL_KINDS = frozenset({"forward", "backward", "rotate", "curve"})
STANDING_POOLS = frozenset({"stand_anchors", "stand_continuous"})


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_sources(repository):
    """Authenticate complete saved files, parsing Python tags as inert data."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    # These two tags occur in the saved configs; neither invokes a constructor.
    for tag in ("python/tuple", "python/object/apply:builtins.slice"):
        Loader.add_constructor("tag:yaml.org,2002:" + tag,
            lambda loader, node: loader.construct_sequence(node))
    sources = {}
    for profile, (run, env_sha, agent_sha) in SOURCES.items():
        files, data = {}, {}
        for name, expected in (("env", env_sha), ("agent", agent_sha)):
            relative = f"pretrained/26_infantry/{profile}/{run}/params/{name}.yaml"
            raw = (Path(repository) / relative).read_bytes()
            if _digest(raw) != expected:
                raise ValueError("SCUT saved source SHA differs: " + relative)
            files[name] = dict(file=relative, sha256=expected, bytes=len(raw))
            data[name] = yaml.load(raw, Loader=Loader)
        env = data["env"]
        commands = deepcopy(env["commands"])
        for key in ("goal_vel_visualizer_cfg", "current_vel_visualizer_cfg"):
            commands.pop(key, None)
        sources[profile] = dict(files=files, commands=commands,
            source_clock=dict(physics_dt=env["sim"]["dt"], decimation=env["decimation"],
                policy_dt=env["sim"]["dt"] * env["decimation"]),
            height_range_m=env["height_range"], default_height_cmd_m=env["default_height_cmd"],
            terrain_command_overrides=env["terrain_command_overrides"],
            rough_terrain_command_overrides_cfg=env.get("rough_terrain_command_overrides_cfg"),
            nonstandard_birth={key: deepcopy(env[key]) for key in (
                "use_leg_random_start", "use_predefined_leg_random_start", "predefined_reset_ground",
                "use_joint_vel_random_start", "leg_joint_vel_range", "wheel_joint_vel_range")},
            agent=data["agent"])
    return sources


def _range(values, name):
    if (not isinstance(values, (list, tuple)) or len(values) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not values[0] < values[1]):
        raise ValueError("Native sampling needs one finite increasing interval: " + name)
    return list(values)


def _height_range(source, native):
    low, high = _range(source, "SCUT height")
    native_low, native_high = _range(native, "native height")
    return _range([max(low, native_low), min(high, native_high)], "height intersection")


def _protected(spec, pool):
    return (pool in STANDING_POOLS or spec.get("kind") == "stand" or spec.get("zero_command")
        or (spec.get("mode", 0) == 0 and not any(spec.get("command", [0., 0.])[:2]))
        or spec.get("request_events") or spec.get("command_sequence") or spec.get("recovery_pose")
        or spec.get("mode", 0) == 4 or spec.get("height_motion")
        or spec.get("kind") in {"start_stop", "velocity_curve", "release_stop", "height",
            "recovery_handoff", "jump", "running_jump", "step_jump", "double_step_jump",
            "airborne", "landing", "spin", "spin_translate"})


def _sampling(spec, vx, yaw, height, duration, *, standing_probability=None):
    # The existing native samplers consume these fields without a new sampler.
    spec.update(sample_amplitude=False, sample_yaw_sign=False,
        planar_command_sampling=dict(vx_range_m_s=_range(vx, "vx m/s"),
            yaw_range_rad_s=_range(yaw, "yaw rad/s"), nominal_fraction=0.),
        height_sampling=dict(deepcopy(spec.get("height_sampling", {})), range_m=height,
            nominal_fraction=0., endpoint_fraction=0.,
            boundary_fraction_of_range=0., boundary_weight=1.),
        command_resampling_seconds=_range(duration, "resampling seconds"),
        command_resampling_scope="skill")
    if standing_probability is not None:
        spec["standing_command_probability"] = standing_probability


def apply(config, repository, *, profile="rough_dash", role="train", terrain_bindings=None):
    """Map normal continuous commands; exact explicit terrain bindings only.

    A terrain binding is ``{native_name: {override: source_name, vx_interval: i}}``.
    ``vx_interval`` explicitly selects one source interval when its distribution
    is disjoint. No terrain name, orientation or missing heading mode is guessed.
    """
    result = deepcopy(config)
    if role != "train":
        return result
    source = read_sources(repository)[profile]
    commands, changed = source["commands"], {}
    native_height = config["height_range_m"]
    normal_height = _height_range(source["height_range_m"], native_height)
    specs, membership = result["skill_specs"], result["behavior_pool_membership"]
    for name, spec in specs.items():
        pool = membership[name]
        if pool in NORMAL_POOLS and spec["kind"] in NORMAL_KINDS and not _protected(spec, pool):
            _sampling(spec, commands["ranges"]["lin_vel_x"], commands["ranges"]["ang_vel_z"],
                normal_height, commands["resampling_time_range"],
                standing_probability=commands["rel_standing_envs"])
            changed[name] = dict(source="commands.ranges", kind="normal",
                height_mapping="source/native intersection", height_range_m=normal_height)
    for name, binding in (terrain_bindings or {}).items():
        if name not in specs or _protected(specs[name], membership[name]):
            raise ValueError("Terrain mapping cannot replace a dedicated or zero-command skill: " + name)
        override = source["terrain_command_overrides"][binding["override"]]
        current = specs[name].get("planar_command_sampling")
        if current is None:
            raise ValueError("Terrain mapping needs its declared native planar sampler: " + name)
        vx = override["lin_vel_x"]
        if vx is None:
            vx = current["vx_range_m_s"]
        elif isinstance(vx[0], (list, tuple)):
            index = binding.get("vx_interval")
            if type(index) is not int or not 0 <= index < len(vx):
                raise ValueError("Disjoint source vx requires an explicit interval index")
            vx = vx[index]
        yaw = override.get("ang_vel_z_non_heading")
        if yaw is None:
            yaw = override.get("ang_vel_z") or current["yaw_range_rad_s"]
        heights = override["height_range"]
        if len(heights) != 1:
            raise ValueError("Disjoint terrain heights cannot be collapsed into one interval")
        _sampling(specs[name], vx, yaw, _height_range(heights[0], native_height),
            commands["resampling_time_range"])
        changed[name] = dict(source="terrain_command_overrides." + binding["override"],
            kind="explicit_terrain", binding=deepcopy(binding),
            inherited_null_fields=[k for k in ("lin_vel_x", "ang_vel_z", "ang_vel_z_non_heading")
                                   if override.get(k) is None])
    standing = math.fsum(row["fraction"] for row in result["scene_groups"]
        if membership[row["name"]] in STANDING_POOLS)
    if standing < commands["rel_standing_envs"] - 1e-12:
        raise ValueError("Continuous normal mapping requires the existing native 10 percent standing floor")
    # All unselected specs, rows, quotas, runtime settings and physics are exact.
    result["scut_reuse_candidate"] = dict(schema="scut-saved-command-reuse-v1", source_commit=COMMIT,
        profile=profile, source_files=deepcopy(source["files"]), mapped_groups=changed,
        preserved_groups=[name for name in specs if name not in changed],
        native_standing_fraction=standing, source_standing_probability=commands["rel_standing_envs"],
        standing_semantics="SCUT Bernoulli 0.1 per ordinary resample plus preserved native named anchors; total is not exactly 0.1",
        group_statistics="mapped legacy names now identify continuous mixtures, not their former fixed nominal command",
        runtime_ready=False, qualification=None, PPO_updates=0,
        remaining_integration=["named contracts and native runtime validation",
            "joint learning clock and staged mixed-pool runtime ownership",
            "heading controller and raycast are not mapped",
            "native command slew/feasibility and difficulty envelopes remain; sampled ranges are before these transforms",
            "height target retains native reset/transition timing; no SCUT raycast or height controller is copied",
            "SCUT disjoint special modes retain native high-motion skills",
            "native closed-chain pose bank retained; source leg dimensions/probabilities not copied"])
    return result


def source_evidence(repository):
    """Serializable source/configuration audit, not a deployment receipt."""
    return dict(schema="scut-reuse-source-v1", source_commit=COMMIT,
        profiles=read_sources(repository), mappings={
            "normal_planar": "commands.ranges → native planar_command_sampling; nominal_fraction=0",
            "dwell": "commands.resampling_time_range → per-skill command_resampling_seconds, seconds",
            "height": "height_range ∩ native height_range_m → native height_sampling, meters",
            "standing": "source rel_standing_envs=0.1 → ordinary standing_command_probability; additional native anchors stay zero",
            "terrain": "explicit saved terrain_command_overrides name and disjoint interval selection",
            "birth": "source enabled 0.2s zero torque/1.5s capture; existing native closed-chain birth retained"},
        exclusions=["SCUT spring/rigid-body geometry, actuator parameters and network sizes",
            "heading target/raycast observation, terrain regeneration and STEP FSM",
            "global resampling of start-stop, release-stop, recovery and requested jump sequences"],
        runtime_ready=False, PPO_updates=0, qualification=None)
