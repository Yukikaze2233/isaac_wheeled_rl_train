"""Authenticated heading commands on real CPU native command lifecycles.

No PhysX application, PPO update, fixed-evaluation qualification or GPU credit.
"""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis import heading_command as heading_module
from wheeled_tasks.chassis.skill_commands import SkillCommands, sample_planar_command


ROOT = Path(__file__).resolve().parents[1]
RECIPE = json.loads((ROOT / "contracts/v6_integrated_runtime_recipe_v24.json").read_text())
NORMAL = tuple(name for name, row in RECIPE["domains"]["speed"]["patch"]
               ["scut_reuse_candidate"]["mapped_groups"].items() if row["kind"] == "normal")
TERRAIN = tuple(name for name, row in RECIPE["domains"]["terrain"]["patch"]
                ["scut_reuse_candidate"]["mapped_groups"].items() if row["kind"] == "explicit_terrain")
ENV_SOURCE = ROOT / "src/wheeled_tasks/chassis/env.py"


def configuration(*, enabled=True, terrain=False):
    """Keep every authenticated normal owner even with a small CPU lane batch."""
    source = RECIPE["domains"]["speed"]["runtime"]["source_bindings"]["speed"]["file"]
    config = json.loads((ROOT / source).read_text())
    config.update(deepcopy(RECIPE["domains"]["speed"]["patch"]))
    if terrain:
        patch = RECIPE["domains"]["terrain"]["patch"]
        for name in TERRAIN:
            config["skill_specs"][name] = deepcopy(patch["skill_specs"][name])
            config["behavior_pool_membership"][name] = patch["behavior_pool_membership"][name]
            config["scut_reuse_candidate"]["mapped_groups"][name] = deepcopy(
                patch["scut_reuse_candidate"]["mapped_groups"][name])
    if enabled is not None:
        mapped = config["scut_reuse_candidate"]
        config["scut_heading_command"] = dict(schema="v6-scut-heading-command-v1", enabled=enabled,
            profile=mapped["profile"], source_commit=mapped["source_commit"],
            source_files=deepcopy(mapped["source_files"]))
    return config


def environment(config=None, groups=("forward_05",), *, repeats=1, seed=1729):
    config = configuration() if config is None else config
    names = list(groups) * repeats
    count = len(names)
    generator = torch.Generator().manual_seed(seed)
    pose = torch.zeros(count, 7)
    pose[:, 6] = 1.  # Native root pose stores the quaternion in xyzw order.
    env = NS(cfg=deepcopy(config), num_envs=count, device="cpu", scene_groups=names,
        global_actor_update=0, stage_actor_update=0, training_transitions=0,
        generator=generator, random=lambda n: torch.rand(n, generator=generator),
        policy_dt=.02, commands=torch.zeros(count, 3), command_target=torch.zeros(count, 2),
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


def set_yaw(env, value):
    value = torch.as_tensor(value, dtype=env.commands.dtype).expand(env.num_envs)
    q = env.robot.data.root_link_pose_w.torch[:, 3:]
    q.zero_()
    q[:, 2] = (value / 2).sin()
    q[:, 3] = (value / 2).cos()


def native_method(name):
    tree = ast.parse(ENV_SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def execute_native(nodes, env, **values):
    module = ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[]))
    scope = {"self": env, "torch": torch, "__package__": "wheeled_tasks.chassis", **values}
    exec(compile(module, str(ENV_SOURCE), "exec"), scope)
    return scope


def bind_native_resample(env):
    scope = execute_native([native_method("resample_commands")], env)
    env.resample_commands = scope["resample_commands"].__get__(env)


def native_command_tick(env):
    block = next(node for node in native_method("step").body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "self.cfg.get('command_slew')")
    execute_native([block], env)


def native_resampling_tick(env, done):
    nodes = native_method("step").body
    start = next(i for i, node in enumerate(nodes)
                 if ast.unparse(node) == "self.command_clock -= self.policy_dt")
    stop = next(i for i in range(start, len(nodes)) if isinstance(nodes[i], ast.If)
                and "self.resample_commands(" in ast.unparse(nodes[i]))
    execute_native(nodes[start:stop + 1], env, done=done)


def test_authenticated_native_scope_has_24_normal_owners_and_eight_protected_terrain_rows():
    config = configuration(terrain=True)
    original = deepcopy(config)
    protected = (*TERRAIN, "forward_5", "spin_translate_1rps", "release_stop_translate",
                 "recovery_upright_forward", "stand_305mm", "height_scan")
    env = environment(config, (*NORMAL, *protected))
    heading = env.skills.heading
    assert len(NORMAL) == 24 and len(TERRAIN) == 8
    assert heading is not None
    assert heading.ordinary_mask.tolist() == [True] * 24 + [False] * len(protected)
    assert heading.ids.tolist() == list(range(24))
    assert not heading.is_heading.any() and not heading.is_standing.any()
    assert heading.settings["heading_range"] == (-math.pi, math.pi)
    assert heading.settings["yaw_range"] == (-math.tau, math.tau)
    assert heading.settings["stiffness"] == 5. and heading.settings["probability"] == .5
    assert config == original


def test_real_protected_samplers_and_updates_keep_identical_commands_clocks_and_rng():
    groups = (*TERRAIN, "forward_5", "spin_translate_1rps", "release_stop_translate",
              "recovery_upright_forward", "stand_305mm", "height_scan")
    old = environment(configuration(enabled=False, terrain=True), groups)
    opted = environment(configuration(terrain=True), groups)
    assert not opted.skills.heading.ordinary_mask.any()
    ids = torch.arange(old.num_envs)
    for env in (old, opted):
        env.skills.sample(ids)
        env.skills.update()
    for name in ("commands", "command_target", "command_clock", "height_clock", "mode"):
        torch.testing.assert_close(getattr(old, name), getattr(opted, name), rtol=0, atol=0)
    torch.testing.assert_close(old.skills.height_target, opted.skills.height_target, rtol=0, atol=0)
    assert torch.equal(old.generator.get_state(), opted.generator.get_state())


@pytest.mark.parametrize("mutate", [
    lambda c: c["scut_heading_command"].update(source_commit="0" * 40),
    lambda c: c["scut_heading_command"]["source_files"]["env"].update(sha256="0" * 64),
    lambda c: c["scut_heading_command"]["source_files"]["env"].update(file="not-a-saved-source.yaml"),
    lambda c: c["scut_reuse_candidate"]["source_files"]["agent"].update(bytes=1),
    lambda c: c["skill_specs"]["forward_05"].update(recovery_pose="upright"),
    lambda c: c["skill_specs"]["forward_05"].update(standing_command_probability=.2),
    lambda c: c["scut_reuse_candidate"]["mapped_groups"].pop("forward_05"),
])
def test_opt_in_rejects_broken_source_pins_and_native_owner_mutations(mutate):
    config = configuration()
    mutate(config)
    with pytest.raises(ValueError):
        heading_module.configure_heading(config, ROOT)


def test_opt_in_authenticates_complete_saved_files(tmp_path):
    for relative in RECIPE["scut_archive_sources"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    file = tmp_path / "references/scut_curriculum_b8ff79f3" / configuration()[
        "scut_heading_command"]["source_files"]["env"]["file"]
    with file.open("ab") as stream:
        stream.write(b"\n# CPU authentication mutation\n")
    with pytest.raises(ValueError, match="SHA"):
        heading_module.configure_heading(configuration(), tmp_path)


def test_declared_source_registry_records_all_real_files_with_verified_sha256():
    config = configuration()
    files = heading_module.heading_source_files(config, ROOT)
    assert len(files) == len(set(files)) == 11
    assert all((ROOT / file).is_file() for file in files)
    trainer = ast.parse((ROOT / "scripts/train_chassis.py").read_text())
    main = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    registration = next(node for node in main.body if isinstance(node, ast.If)
                        and "heading_source_files(c, ROOT)" in ast.unparse(node))
    recording = next(node for node in main.body if isinstance(node, ast.Assign)
                     and ast.unparse(node.targets[0]) == "source_hashes")
    digest = lambda file: hashlib.sha256(Path(file).read_bytes()).hexdigest()
    scope = execute_native([registration, recording], NS(), c=config, ROOT=ROOT, digest=digest, source_files=[])
    assert scope["source_files"] == files
    assert scope["source_hashes"] == {file: digest(ROOT / file) for file in files}
    manifest = json.loads((ROOT / RECIPE["scut_archive_manifest"]["file"]).read_text())
    for file, pin in manifest["files"].items():
        archived = "references/scut_curriculum_b8ff79f3/" + file
        assert scope["source_hashes"][archived] == pin["sha256"]
    pin = RECIPE["scut_archive_manifest"]
    assert scope["source_hashes"][pin["file"]] == pin["sha256"]
    scope = execute_native([registration, recording], NS(), c=configuration(enabled=False),
                           ROOT=ROOT / "missing", digest=digest, source_files=[])
    assert scope["source_files"] == [] and scope["source_hashes"] == {}


def test_protected_row_cannot_replace_an_authenticated_normal_claim():
    config = configuration(terrain=True)
    mapped = config["scut_reuse_candidate"]["mapped_groups"]
    claim = mapped.pop("forward_05")
    mapped[TERRAIN[0]] = claim
    assert sum(row["kind"] == "normal" for row in mapped.values()) == 24
    with pytest.raises(ValueError, match="protected"):
        heading_module.configure_heading(config, ROOT)


@pytest.mark.parametrize("role", ["fixed_eval", "playback", "qualification", "preflight", "rollout_validation"])
def test_native_constructor_restricts_enabled_heading_to_training(role):
    block = next(node for node in native_method("__init__").body
                 if isinstance(node, ast.If) and "scut_heading_command" in ast.unparse(node.test))
    with pytest.raises(ValueError, match="explicit training"):
        execute_native([block], NS(), config=configuration(), run_role=role)
    exact = configuration()
    exact["evaluation_exact_cases"] = True
    execute_native([block], NS(), config=exact, run_role=role)
    execute_native([block], NS(), config=configuration(enabled=False), run_role=role)


def test_native_startup_validates_disjoint_fsm_owners_and_reports_real_heading_state():
    env = environment(groups=("forward_5", "forward_05"))
    env.recovery = NS(enabled=torch.tensor([True, False]))
    env.step_jump = NS(enabled=torch.tensor([False, False]))
    env.step_climb = NS(enabled=torch.tensor([False, False]))
    env.startup_report = {}
    block = next(node for node in native_method("__init__").body if isinstance(node, ast.If)
                 and "self.skills.heading.validate_owners(" in ast.unparse(node))
    execute_native([block], env)
    report = env.startup_report["scut_heading_command"]
    assert report["eligible_count"] == 1 and report["ordinary_groups"] == sorted(NORMAL)
    assert report["reference_dt"] == .02 and report["qualification"] is None
    assert report["source_files"] == env.cfg["scut_heading_command"]["source_files"]
    env.recovery.enabled[1] = True
    with pytest.raises(ValueError, match="native special/FSM owner"):
        execute_native([block], env)


def test_default_disabled_and_exact_eval_do_not_open_sources_or_change_native_rng(tmp_path):
    default, disabled = environment(configuration(enabled=None)), environment(configuration(enabled=False))
    ids = torch.arange(default.num_envs)
    for env in (default, disabled):
        assert env.skills.heading is None
        env.skills.sample(ids)
        env.skills.update()
    torch.testing.assert_close(default.commands, disabled.commands, rtol=0, atol=0)
    torch.testing.assert_close(default.command_target, disabled.command_target, rtol=0, atol=0)
    torch.testing.assert_close(default.command_clock, disabled.command_clock, rtol=0, atol=0)
    assert torch.equal(default.generator.get_state(), disabled.generator.get_state())
    missing = tmp_path / "not-a-source-tree"
    assert heading_module.configure_heading(configuration(enabled=None), missing) is None
    assert heading_module.configure_heading(configuration(enabled=False), missing) is None
    exact = configuration()
    exact["evaluation_exact_cases"] = True
    assert heading_module.configure_heading(exact, missing) is None
    exact["evaluation"] = {"cases": [{"name": "forward_05", "command": [.5, 0., .305]}]}
    fixed = environment(exact)
    assert fixed.skills.heading is None
    fixed.random = lambda _: (_ for _ in ()).throw(AssertionError("fixed case consumed RNG"))
    fixed.skills.sample(torch.tensor([0]))
    fixed.skills.update()
    torch.testing.assert_close(fixed.commands[0], torch.tensor([.5, 0., .305]), rtol=0, atol=0)


def test_standing_output_retains_exact_legacy_draw_and_rng_sequence():
    spec = configuration()["skill_specs"]["forward_05"]
    first = torch.Generator().manual_seed(3501)
    second = torch.Generator().manual_seed(3501)
    a = torch.tensor(spec["command"]).expand(4096, -1).clone()
    b = a.clone()
    standing = torch.empty(len(a), dtype=torch.bool)
    sample_planar_command(a, spec, lambda n: torch.rand(n, generator=first))
    sample_planar_command(b, spec, lambda n: torch.rand(n, generator=second), standing_out=standing)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(first.get_state(), second.get_state())
    assert torch.equal(standing, b[:, :2].eq(0.).all(-1))
    assert standing.float().mean().item() == pytest.approx(.1, abs=.02)
    untouched = torch.tensor([True, False, True])
    expected = untouched.clone()
    fixed = torch.tensor(spec["command"]).expand(3, -1).clone()
    sample_planar_command(fixed, spec,
        lambda _: (_ for _ in ()).throw(AssertionError("fixed sampler consumed RNG")),
        evaluation=True, standing_out=untouched)
    assert torch.equal(untouched, expected)


def test_capture_reuses_standing_and_includes_half_threshold_without_drawing_for_standing():
    env = environment(repeats=4)
    heading = env.skills.heading
    raw = torch.tensor([[1., .7, .30], [1., -.8, .31], [-1., 1., .32], [0., 0., .33]])
    standing = torch.tensor([False, False, False, True])
    draws = iter((torch.tensor([.5, .75, .25]), torch.tensor([.5, .50001, 0.])))
    counts = []
    def random(count):
        counts.append(count)
        result = next(draws)
        assert len(result) == count
        return result
    heading.capture(torch.arange(4), raw, standing, None,
                    env.cfg["skill_specs"]["forward_05"], random)
    assert counts == [3, 3]
    assert heading.is_heading.tolist() == [True, False, True, False]
    assert torch.equal(heading.is_standing, standing)
    torch.testing.assert_close(heading.target[:3], torch.tensor([0., math.pi / 2, -math.pi / 2]))
    resolved = heading.resolve(torch.arange(4), torch.zeros(4), raw[:, 2], transformed=False)
    torch.testing.assert_close(resolved[:, :2], torch.tensor([[1., 0.], [1., -.8], [-1., -math.tau], [0., 0.]]))
    torch.testing.assert_close(resolved[:, 2], raw[:, 2], rtol=0, atol=0)
    torch.testing.assert_close(raw[:, :2], torch.tensor([[1., .7], [1., -.8], [-1., 1.], [0., 0.]]), rtol=0, atol=0)


def test_wrap_gain_source_clip_and_small_error_have_exact_source_semantics():
    env = environment(repeats=7)
    heading = env.skills.heading
    raw = torch.tensor([[0., .7, .305]]).repeat(7, 1)
    heading.capture(torch.arange(7), raw, torch.zeros(7, dtype=torch.bool), None,
                    env.cfg["skill_specs"]["forward_05"], lambda n: torch.zeros(n))
    heading.target[:] = torch.tensor([-math.pi + .02, math.pi - .02, .001, .2, 2., math.pi, -math.pi])
    yaw = torch.tensor([math.pi - .02, -math.pi + .02, 0., 0., 0., 0., 0.])
    resolved = heading.resolve(torch.arange(7), yaw, raw[:, 2], transformed=False)
    torch.testing.assert_close(resolved[:, 1], torch.tensor([.2, -.2, .005, 1., math.tau, math.tau, -math.tau]),
                               rtol=0, atol=2e-6)
    torch.testing.assert_close(resolved[:, 0], raw[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(resolved[:, 2], raw[:, 2], rtol=0, atol=0)


def test_real_sampler_draws_heading_at_native_dwell_and_keeps_early_envelope():
    env = environment(repeats=8192)
    env.skills.sample(torch.arange(env.num_envs))
    heading = env.skills.heading
    assert heading.is_standing.float().mean().item() == pytest.approx(.1, abs=.015)
    assert not (heading.is_heading & heading.is_standing).any()
    assert heading.is_heading[~heading.is_standing].float().mean().item() == pytest.approx(.5, abs=.02)
    targets = heading.target[~heading.is_standing]
    assert targets.min() >= -math.pi and targets.max() <= math.pi
    assert targets.min() < -3. and targets.max() > 3.
    assert env.command_clock.min() >= 5. and env.command_clock.max() <= 15.
    assert env.command_clock.min() < 6. and env.command_clock.max() > 14.
    assert env.commands[heading.is_standing, :2].eq(0.).all()
    assert env.commands[:, 0].abs().max() <= 2. + 1e-6
    assert env.commands[:, 1].abs().max() <= math.tau + 1e-6
    assert env.commands[:, 2].min() >= .28 and env.commands[:, 2].max() <= .35
    assert (env.commands[:, 0] > 1.8).any() and (env.commands[:, 0] < -1.8).any()
    assert (env.commands[:, 0] * env.commands[:, 1]).abs().max() <= 2.06 + 1e-6


def test_real_policy_update_reacts_to_root_heading_without_rng_or_resampling():
    env = environment(repeats=128)
    env.skills.sample(torch.arange(env.num_envs))
    heading = env.skills.heading
    heading.target[heading.is_heading] = .2
    before = (heading.target.clone(), heading.is_heading.clone(), heading.is_standing.clone(),
              env.command_clock.clone(), env.commands[:, 2].clone(), env.generator.get_state())
    set_yaw(env, .1)
    env.skills.update()
    torch.testing.assert_close(env.command_target[heading.is_heading, 1], torch.full((int(heading.is_heading.sum()),), .5),
                               rtol=0, atol=1e-6)
    assert env.command_target[heading.is_standing].eq(0.).all()
    for actual, expected in zip((heading.target, heading.is_heading, heading.is_standing,
                                 env.command_clock, env.commands[:, 2], env.generator.get_state()), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_native_timed_subset_resample_retains_height_and_executed_command_and_skips_done():
    env = environment(groups=("forward_05__height_train",), repeats=64)
    bind_native_resample(env)
    env.resample_commands(torch.arange(env.num_envs), reset_height=True)
    env.commands[:, 0] = .2
    before_target = env.skills.heading.target.clone()
    before_heading = env.skills.heading.is_heading.clone()
    before_standing = env.skills.heading.is_standing.clone()
    before_height = env.commands[:, 2].clone()
    before_height_target = env.skills.height_target.clone()
    before_phase = env.skills.height_phase_offset.clone()
    before_planar = env.commands[:, :2].clone()
    expired = torch.arange(1, 32, 2)
    env.command_clock[expired] = .01
    done = torch.zeros(env.num_envs, dtype=torch.bool)
    done[expired[-1]] = True
    unchanged = torch.ones(env.num_envs, dtype=torch.bool)
    unchanged[expired[:-1]] = False
    native_resampling_tick(env, done)
    heading = env.skills.heading
    assert (heading.target[expired[:-1]] != before_target[expired[:-1]]).any()
    torch.testing.assert_close(heading.target[unchanged], before_target[unchanged], rtol=0, atol=0)
    torch.testing.assert_close(heading.is_heading[unchanged], before_heading[unchanged], rtol=0, atol=0)
    torch.testing.assert_close(heading.is_standing[unchanged], before_standing[unchanged], rtol=0, atol=0)
    assert env.command_clock[expired[:-1]].min() >= 5. and env.command_clock[expired[:-1]].max() <= 15.
    assert env.command_clock[expired[-1]] < 0.
    torch.testing.assert_close(env.commands[:, :2], before_planar, rtol=0, atol=0)
    torch.testing.assert_close(env.commands[:, 2], before_height, rtol=0, atol=0)
    torch.testing.assert_close(env.skills.height_target, before_height_target, rtol=0, atol=0)
    torch.testing.assert_close(env.skills.height_phase_offset, before_phase, rtol=0, atol=0)


def test_native_slew_then_final_projection_updates_actual_acceleration_once_and_only_owned_lanes(monkeypatch):
    env = environment(groups=("forward_5", "forward_05", "forward_05", "forward_05"))
    heading = env.skills.heading
    ids = heading.ids
    raw = torch.tensor([[2.7, 0., .30], [1., 0., .31], [2.7, 0., .32]])
    heading.capture(ids, raw, torch.zeros(3, dtype=torch.bool), None,
                    env.cfg["skill_specs"]["forward_05"], lambda n: torch.full((n,), .5))
    heading.is_heading[ids] = torch.tensor([True, False, True])
    env.commands[:] = torch.tensor([[6., 0., .33], [5.5, 0., .30], [6., 0., .31], [.32, math.tau, .32]])
    env.command_target[:] = env.commands[:, :2]
    env.command_target[ids[1]] = raw[1, :2]
    previous = env.commands.clone()
    calls = []
    projection_calls = []
    inside_finish = [False]
    original_project = heading_module.project_commands
    def project(commands, limits):
        if inside_finish[0]:
            projection_calls.append(commands.clone())
        return original_project(commands, limits)
    monkeypatch.setattr(heading_module, "project_commands", project)
    original_finish = heading.finish
    def finish(commands, previous_planar, acceleration, limits):
        calls.append((commands.clone(), previous_planar.clone()))
        inside_finish[0] = True
        try:
            return original_finish(commands, previous_planar, acceleration, limits)
        finally:
            inside_finish[0] = False
    monkeypatch.setattr(heading, "finish", finish)
    monkeypatch.setattr(heading_module, "read_sources",
                        lambda _: (_ for _ in ()).throw(AssertionError("policy tick read saved sources")))
    env.random = lambda _: (_ for _ in ()).throw(AssertionError("policy tick consumed RNG"))
    generator_state = env.generator.get_state()
    native_command_tick(env)
    assert len(calls) == 1
    assert len(projection_calls) == 1 and len(projection_calls[0]) == len(ids)
    torch.testing.assert_close(calls[0][1], previous[ids, :2], rtol=0, atol=0)
    torch.testing.assert_close(calls[0][0][ids, 0], torch.tensor([5.47, 5.97, .35]), rtol=0, atol=1e-6)
    assert env.commands[1, 0] == pytest.approx(5.4)
    assert env.commands[2, 0] == pytest.approx(5.97)  # Ordinary nonheading lane stays on native slew.
    torch.testing.assert_close(env.commands[2], calls[0][0][2], rtol=0, atol=0)
    assert env.commands[0, 0] == 6.  # Protected high-motion row is not projected by heading.
    assert abs(float(env.commands[3, 0] * env.commands[3, 1])) <= 2.06 + 1e-6
    assert env.command_acceleration[1] == pytest.approx(-5., abs=5e-5)
    controlled = ids[heading.is_heading[ids] | heading.is_standing[ids]]
    torch.testing.assert_close(env.command_acceleration[controlled],
                               (env.commands[controlled, 0] - previous[controlled, 0]) / env.policy_dt,
                               rtol=0, atol=1e-6)
    assert env.command_acceleration[2] == pytest.approx(-1.5)
    torch.testing.assert_close(env.commands[:, 2], previous[:, 2], rtol=0, atol=0)
    assert torch.equal(generator_state, env.generator.get_state())
