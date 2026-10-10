"""Actual native CPU commands for the opt-in pure-motion authoring contract.

No PhysX trajectory, learning update or policy qualification is inferred here.
"""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis import heading_command, integrated_runtime, motion_rehearsal
from wheeled_tasks.chassis.motion_limits import project_commands, wheel_speeds
from wheeled_tasks.chassis.skill_commands import SkillCommands


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source():
    recipe = json.loads((ROOT / "contracts/v6_integrated_runtime_recipe_v26.json").read_text())
    config = json.loads((ROOT / "contracts/v6_task_campaign_speed_worker_v26.json").read_text())
    config.update(deepcopy(recipe["domains"]["speed"]["patch"]))
    runtime = deepcopy(recipe["domains"]["speed"]["runtime"])
    # A valid zero-origin clock fixture for native routing; no checkpoint credit.
    selection = dict(domain="speed", lineage_id=recipe["lineage_id"],
                     plan_sha256="0" * 64, origin_checkpoint_sha256="1" * 64,
                     origin_recovery_actor_update=14188,
                     fresh_joint_actor_updates=0, fresh_domain_actor_updates=0)
    runtime["learning_source"] = selection
    runtime["clocks_at_source"] = integrated_runtime._clock_document(selection, {"speed": 0}, 0, 0)
    config["integrated_runtime_candidate"] = runtime
    return config


def environment(config, clock, groups, *, repeats=256, seed=1729):
    names = list(groups) * repeats
    count = len(names)
    generator = torch.Generator().manual_seed(seed)
    pose = torch.zeros(count, 7)
    pose[:, 6] = 1.
    env = NS(cfg=deepcopy(config), num_envs=count, device="cpu", scene_groups=names,
             task_actor_update=clock, global_actor_update=100000, stage_actor_update=clock,
             training_transitions=0, generator=generator,
             random=lambda n: torch.rand(n, generator=generator), policy_dt=.02,
             commands=torch.zeros(count, 3), command_target=torch.zeros(count, 2),
             mode=torch.zeros(count, dtype=torch.long), push_enabled=torch.zeros(count, dtype=torch.bool),
             phase=NS(phase=torch.zeros(count, dtype=torch.long)), full_tasks=None,
             robot=NS(data=NS(root_link_pose_w=NS(torch=pose))), references=None,
             step_assist=None, step_jump=None, nonstandard_start=None)
    for name in ("command_clock", "height_clock", "push_clock"):
        setattr(env, name, torch.zeros(count))
    for name in ("episode_limits", "episode_length_buf"):
        setattr(env, name, torch.zeros(count, dtype=torch.long))
    env.skills = SkillCommands(env, root=ROOT)
    return env


def indices(env, name):
    return torch.tensor([i for i, group in enumerate(env.scene_groups) if group == name])


def assert_pure_planar(env, *, applied=True):
    commands = env.commands[:, :2] if applied else env.command_target
    for base, command in motion_rehearsal.BASE_COMMANDS.items():
        for name in (base, base + "__height_train"):
            ids = indices(env, name)
            expected = commands.new_tensor(command[:2]).expand(len(ids), -1)
            torch.testing.assert_close(commands[ids], expected, rtol=0, atol=0)


def native_command_tick(env):
    """Run the exact native step command block without the GPU application."""
    file = ROOT / "src/wheeled_tasks/chassis/env.py"
    tree = ast.parse(file.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    block = next(node for node in step.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "self.cfg.get('command_slew')")
    module = ast.fix_missing_locations(ast.Module(body=[deepcopy(block)], type_ignores=[]))
    exec(compile(module, str(file), "exec"), {"self": env, "torch": torch,
                                           "__package__": "wheeled_tasks.chassis"})


def test_authoring_preserves_all_rows_weights_frontier_and_input(source):
    original = deepcopy(source)
    repaired = motion_rehearsal.apply(source)
    assert source == original
    assert repaired["scene_groups"] == source["scene_groups"]
    assert len(repaired["scene_groups"]) == 189
    for field in ("behavior_pool_membership", "behavior_pool_fractions", "integrated_sampling_candidate",
                  "locomotion_difficulty", "motion_limits", "command_slew", "release_stop"):
        assert repaired[field] == source[field]
    original_runtime, runtime = source["integrated_runtime_candidate"], repaired["integrated_runtime_candidate"]
    for field in runtime:
        if field != "row_ownership":
            assert runtime[field] == original_runtime[field]
    for name, spec in source["skill_specs"].items():
        if name not in motion_rehearsal.PURE_NAMES:
            assert repaired["skill_specs"][name] == spec
            assert runtime["row_ownership"][name] == original_runtime["row_ownership"][name]
        else:
            assert runtime["row_ownership"][name] == {
                **original_runtime["row_ownership"][name], "difficulty_component": None}
    assert motion_rehearsal.validate(repaired) == motion_rehearsal.motion_rehearsal_settings()
    configured = motion_rehearsal.configure_heading(repaired, ROOT)
    assert set(configured["ordinary_names"]) == set(motion_rehearsal.CONTINUOUS_NAMES)
    assert not set(configured["ordinary_names"]) & set(motion_rehearsal.PURE_NAMES)
    assert source == original
    # The inherited configurator cannot silently apply its old ownership.
    with pytest.raises(ValueError, match="24 declared"):
        heading_command.configure_heading(repaired, ROOT)


@pytest.mark.parametrize("clock", [0, 120, 240, 360, 3730])
def test_real_native_samples_keep_zero_axes_signs_heights_and_frontier(source, clock):
    repaired = motion_rehearsal.apply(source)
    groups = motion_rehearsal.PURE_NAMES + motion_rehearsal.CONTINUOUS_NAMES
    env = environment(repaired, clock, groups)
    env.skills.sample(torch.arange(env.num_envs))
    assert_pure_planar(env)
    assert_pure_planar(env, applied=False)
    pure_mask = torch.tensor([name in motion_rehearsal.PURE_NAMES for name in env.scene_groups])
    assert not env.skills.heading.ordinary_mask[pure_mask].any()
    assert env.skills.heading.ordinary_mask[~pure_mask].all()
    assert env.skills.heading.is_heading[~pure_mask].any()
    for base in motion_rehearsal.BASE_COMMANDS:
        nominal_ids = indices(env, base)
        torch.testing.assert_close(env.commands[nominal_ids, 2], torch.full((len(nominal_ids),), .305),
                                   rtol=0, atol=0)
        ids = indices(env, base + "__height_train")
        height = env.skills.height_target[ids]
        assert height.min().item() == pytest.approx(.23)
        assert height.max().item() == pytest.approx(.43)
        assert ((height > .23) & (height < .43)).any()
        assert env.skills.height_phase_offset[ids].eq(0.).any()
        assert env.skills.height_phase_offset[ids].eq(4.).any()
        assert integrated_runtime.row_difficulty_envelope(env, base) is None
        assert integrated_runtime.row_difficulty_envelope(env, base + "__height_train") is None
    # Extended rows still draw continuous joint commands and retain frontier ownership.
    continuous = env.skills.command_base[~pure_mask, :2]
    assert continuous[:, 0].gt(0).any() and continuous[:, 0].lt(0).any()
    assert continuous[:, 1].gt(0).any() and continuous[:, 1].lt(0).any()
    assert ((continuous[:, 0] != 0) & (continuous[:, 1] != 0)).any()
    frontier = integrated_runtime.row_difficulty_envelope(env, "forward_5")
    assert frontier["vx_max_m_s"] == pytest.approx({0: 2., 120: 3., 240: 4., 360: 5., 3730: 5.}[clock])
    limits = env.cfg["motion_limits"]
    left, right = wheel_speeds(env.commands[:, 0], env.commands[:, 1],
                             limits["wheel_radius_m"], limits["track_width_m"])
    assert torch.maximum(left.abs(), right.abs()).max() <= limits["max_wheel_speed_rad_s"] + 1e-5
    assert (env.commands[:, 0] * env.commands[:, 1]).abs().max() <= limits["lateral_acceleration_m_s2"] + 1e-5
    torch.testing.assert_close(project_commands(env.commands[pure_mask], limits), env.commands[pure_mask],
                               rtol=0, atol=0)


def test_native_heading_slew_and_partial_reset_cannot_rewrite_pure_axes(source):
    env = environment(motion_rehearsal.apply(source), 0,
                      motion_rehearsal.PURE_NAMES + motion_rehearsal.CONTINUOUS_NAMES, repeats=32)
    env.skills.sample(torch.arange(env.num_envs))
    original = env.commands.clone()
    selected = torch.arange(env.num_envs)[::3]
    env.skills.sample(selected)
    untouched = ~torch.isin(torch.arange(env.num_envs), selected)
    torch.testing.assert_close(env.commands[untouched], original[untouched], rtol=0, atol=0)
    assert_pure_planar(env)
    env.commands[:, :2] = 0.
    pose = env.robot.data.root_link_pose_w.torch
    pose[:, 5], pose[:, 6] = math.sin(1.2), math.cos(1.2)
    for _ in range(40):
        env.episode_length_buf += 1
        native_command_tick(env)
        assert_pure_planar(env, applied=False)
        for base, command in motion_rehearsal.BASE_COMMANDS.items():
            zero_axis = 1 if command[0] else 0
            for name in (base, base + "__height_train"):
                assert env.commands[indices(env, name), zero_axis].eq(0.).all()
    assert_pure_planar(env)
    # Transition completion reaches full-height targets without frontier clipping.
    env.episode_length_buf.fill_(200)
    env.skills.update()
    for base in motion_rehearsal.BASE_COMMANDS:
        ids = indices(env, base + "__height_train")
        torch.testing.assert_close(env.commands[ids, 2], env.skills.height_target[ids], rtol=0, atol=1e-7)


def test_original_sampler_is_a_negative_control_for_pure_axis_guarantee(source):
    env = environment(source, 0, ("forward_05", "rotate_1", "rotate_1__reverse_train"))
    env.skills.sample(torch.arange(env.num_envs))
    assert env.commands[indices(env, "forward_05"), 1].ne(0).any()
    assert env.commands[indices(env, "rotate_1"), 0].ne(0).any()
    assert env.commands[indices(env, "rotate_1__reverse_train"), 0].ne(0).any()


def test_candidate_source_inventory_adds_helper_and_preserves_inherited_list(source):
    original = deepcopy(source)
    inherited = heading_command.heading_source_files(source, ROOT)
    helper = "src/wheeled_tasks/chassis/motion_rehearsal.py"
    assert len(inherited) == 11 and helper not in inherited
    assert inherited[:4] == ["src/wheeled_tasks/chassis/" + name + ".py" for name in
                             ("heading_command", "scut_reuse_curriculum", "locomotion_difficulty", "motion_limits")]
    candidate = heading_command.heading_source_files(motion_rehearsal.apply(source), ROOT)
    assert candidate == [*inherited[:4], helper, *inherited[4:]]
    assert len(candidate) == len(set(candidate)) == 12
    assert all((ROOT / name).is_file() for name in candidate)
    assert source == original
    assert heading_command.heading_source_files(source, ROOT) == inherited


@pytest.mark.parametrize("mutate", [
    lambda c: c["skill_specs"]["forward_05"].update(standing_command_probability=.1),
    lambda c: c["skill_specs"]["rotate_1"].update(sample_yaw_sign=True),
    lambda c: c["skill_specs"]["backward_1__height_train"]["height_sampling"].update(range_m=[.28, .35]),
    lambda c: c["integrated_runtime_candidate"]["row_ownership"]["rotate_1"].update(difficulty_component="speed"),
    lambda c: c["scut_heading_command"]["ordinary_names"].append("rotate_1"),
    lambda c: c["scut_heading_command"].update(source_commit="0" * 40),
    lambda c: c["scut_reuse_candidate"]["source_files"]["env"].update(sha256="0" * 64),
    lambda c: c["skill_specs"]["curve_low"]["planar_command_sampling"].update(vx_range_m_s=[-1., 1.]),
])
def test_actual_command_or_ownership_mutations_are_rejected(source, mutate):
    config = motion_rehearsal.apply(source)
    mutate(config)
    with pytest.raises(ValueError):
        motion_rehearsal.configure_heading(config, ROOT)
