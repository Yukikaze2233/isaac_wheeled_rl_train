"""Accepted air reset metadata, actual command publication, and exact restart."""
import ast
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from tests.tasks.commands.test_final_rough_commands import owner, tick, ROOT, REFERENCE
from wheeled_tasks.chassis.tasks.recovery.final_rough_birth import FinalRoughBirthSampler
from wheeled_tasks.chassis.tasks.commands.final_rough_commands import AIR_STATE_VERSION, STATE_VERSION
from wheeled_tasks.chassis.tasks.terrain.final_rough_definition import source_manifest


def configured(count=4):
    commands, positions = owner(count, columns=[10] * count)
    fingerprint = FinalRoughBirthSampler(count, "cpu").definition_sha256
    commands.enable_air_birth_command_limits(birth_definition_sha256=fingerprint)
    return commands, positions, fingerprint


def reset(commands, rng, fingerprint, modes=None, ids=None, ground=None):
    ids = torch.arange(commands.num_envs) if ids is None else ids
    ground = torch.zeros(len(ids), dtype=torch.bool) if ground is None else ground
    kwargs = {} if modes is None else {"air_mode_index": modes, "birth_definition_sha256": fingerprint}
    commands.reset(ids, torch.zeros(len(ids)), ground, rng, **kwargs)


def source_oracle(n):
    source_manifest()
    tree = ast.parse((REFERENCE / "base_env.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    names = {"_set_predefined_reset_air_command_limits", "_clear_predefined_reset_air_command_limits",
        "_as_env_ids_tensor",
        "_parse_predefined_reset_air_limit_pair", "_apply_predefined_reset_air_limit_pair",
        "_apply_predefined_reset_air_command_limits", "_advance_predefined_reset_air_command_limit_timers",
        "_apply_predefined_reset_air_height_limits"}
    selected = [deepcopy(node) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    copied = ast.ClassDef(name="SourceLimits", bases=[], keywords=[], decorator_list=[], body=selected)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"torch": torch, "Mapping": Mapping, "Sequence": Sequence}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, copied], type_ignores=[])),
                 "frozen_air_command_methods", "exec"), scope)
    value = scope["SourceLimits"]()
    value.step_dt, value.common_step_counter, value.device = .02, 0, "cpu"
    prefix = "predefined_reset_air_command_limit_"
    setattr(value, prefix + "remaining_time", torch.zeros(n))
    setattr(value, prefix + "mode_id", torch.full((n,), -1, dtype=torch.long))
    setattr(value, prefix + "last_update_step", torch.full((n,), -1, dtype=torch.long))
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "height"):
        setattr(value, prefix + key, torch.full((n, 2), torch.nan))
    return value


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b) or (a.dtype.is_floating_point and torch.allclose(a, b, rtol=0, atol=0, equal_nan=True))
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_same(a[key], b[key])
    else:
        assert a == b


def test_countdown_and_last_priority_match_source_with_one_snapshot_per_public_frame():
    commands, positions, fingerprint = configured(3)
    rng = torch.Generator().manual_seed(71)
    reset(commands, rng, fingerprint, torch.tensor([0, -1, 0]))
    oracle = source_oracle(3)
    limits = commands.summary()["air_birth_command_limits"]["modes"][0]["command_limits"]
    oracle._set_predefined_reset_air_command_limits(torch.tensor([0, 2]), limits, mode_idx=0)
    first_expired = None
    saw_expiry_snapshot = False
    for step in range(160):
        if step:
            tick(commands, rng, positions)
        raw, height = commands.command3.clone(), commands.height.clone()
        public = torch.tensor([[3.5, 8., .9], [-4., -9., 0.], [-3.5, -8., -.1]])
        expected = public.clone()
        oracle.common_step_counter = step
        oracle.command = torch.stack((public[:, 0], torch.zeros(3), public[:, 1]), -1)
        oracle._apply_predefined_reset_air_command_limits()
        expected[:, 0], expected[:, 1] = oracle.command[:, 0], oracle.command[:, 2]
        expected[:, 2] = oracle._apply_predefined_reset_air_height_limits(public[:, 2])
        oracle._advance_predefined_reset_air_command_limit_timers()
        before_rng = rng.get_state().clone()
        commands.apply_air_birth_command_limits(public)
        assert torch.equal(public, expected)
        air = commands.diagnostics()["air_birth_command_limits"]
        assert torch.equal(air["remaining_s"], oracle.predefined_reset_air_command_limit_remaining_time)
        assert torch.equal(rng.get_state(), before_rng)
        assert torch.equal(commands.command3, raw) and torch.equal(commands.height, height)
        duplicate = torch.tensor([[3.5, 8., .9], [-4., -9., 0.], [-3.5, -8., -.1]])
        commands.apply_air_birth_command_limits(duplicate)
        assert torch.equal(duplicate, expected)
        assert_same(commands.diagnostics()["air_birth_command_limits"], air)
        if air["frame_active"][0] and air["remaining_s"][0] == 0.:
            saw_expiry_snapshot = True
        if not air["frame_active"][0] and first_expired is None:
            first_expired = step
    assert saw_expiry_snapshot and first_expired == 151
    assert commands.diagnostics()["air_birth_command_limits"]["mode_index"].tolist() == [0, -1, 0]


@pytest.mark.parametrize("bad", ["definition", "shape", "dtype", "mode", "ground"])
def test_bad_accepted_metadata_does_not_mutate_commands_clocks_or_rng(bad):
    commands, _, fingerprint = configured(3)
    rng = torch.Generator().manual_seed(13)
    before = commands.state_dict(rng)
    modes = torch.tensor([0, -1, 0])
    ground = torch.zeros(3, dtype=torch.bool)
    if bad == "definition":
        fingerprint = "0" * 64
    elif bad == "shape":
        modes = torch.tensor([0])
    elif bad == "dtype":
        modes = modes.float()
    elif bad == "mode":
        modes[0] = 1
    else:
        ground[0] = True
    with pytest.raises(ValueError):
        reset(commands, rng, fingerprint, modes, ground=ground)
    assert_same(commands.state_dict(rng), before)


def test_fixed_retained_flat_and_terrain_permissions_cannot_be_overwritten_by_air_metadata():
    commands, positions, fingerprint = configured(4)
    commands.set_training_rehearsal_requests([0], torch.tensor([[.5, 0., .2]]), torch.tensor([.3]))
    commands.set_source_flat_rehearsal([1], source_iteration_offset=10000)
    rng = torch.Generator().manual_seed(81)
    for index in (0, 1):
        before = commands.state_dict(rng)
        modes = torch.full((4,), -1, dtype=torch.long)
        modes[index] = 0
        with pytest.raises(ValueError, match="cannot own"):
            reset(commands, rng, fingerprint, modes)
        assert_same(commands.state_dict(rng), before)
    reset(commands, rng, fingerprint, torch.tensor([-1, -1, 0, -1]))
    public = torch.tensor([[.5, .2, .3], [4., 10., .6], [4., 10., .6], [4., 10., .6]])
    commands.apply_air_birth_command_limits(public)
    assert torch.equal(public[[0, 1, 3]], torch.tensor([[.5, .2, .3], [4., 10., .6], [4., 10., .6]]))
    assert public[2].tolist() == pytest.approx([2.5, torch.pi / 2, .43])
    blocked, _ = owner(2, columns=[0, 0])
    blocked.enable_air_birth_command_limits(birth_definition_sha256=fingerprint)
    with pytest.raises(ValueError, match="cannot own"):
        reset(blocked, torch.Generator().manual_seed(3), fingerprint, torch.tensor([0, -1]))


def test_enabled_restart_preserves_active_frame_timer_raw_caches_and_rng():
    commands, positions, fingerprint = configured(3)
    rng = torch.Generator().manual_seed(19)
    reset(commands, rng, fingerprint, torch.tensor([0, -1, 0]))
    public = torch.tensor([[4., 10., .9]] * 3)
    commands.apply_air_birth_command_limits(public)
    for _ in range(73):
        tick(commands, rng, positions)
        public.fill_(.4)
        commands.apply_air_birth_command_limits(public)
    saved = commands.state_dict(rng)
    assert saved["schema"] == AIR_STATE_VERSION
    restored, _, _ = configured(3)
    other = torch.Generator().manual_seed(997)
    restored.load_state_dict(saved, other)
    assert_same(restored.state_dict(other), saved)
    for _ in range(90):
        tick(commands, rng, positions)
        tick(restored, other, positions)
        a = torch.tensor([[4., -10., .9]] * 3)
        b = a.clone()
        commands.apply_air_birth_command_limits(a)
        restored.apply_air_birth_command_limits(b)
        assert torch.equal(a, b)
        assert_same(restored.state_dict(other), commands.state_dict(rng))
    old, _ = owner(3, columns=[10] * 3)
    old_rng = torch.Generator().manual_seed(43)
    old_state = old.state_dict(old_rng)
    assert old_state["schema"] == STATE_VERSION
    with pytest.raises(ValueError):
        old.load_state_dict(saved, old_rng)
    with pytest.raises(ValueError):
        restored.load_state_dict(old_state, other)


@pytest.mark.parametrize("corrupt", ["duration", "mode", "future_clock", "published_limit", "identity"])
def test_corrupt_air_restart_is_rejected_before_any_visible_write(corrupt):
    commands, _, fingerprint = configured(2)
    rng = torch.Generator().manual_seed(29)
    reset(commands, rng, fingerprint, torch.tensor([0, -1]))
    commands.apply_air_birth_command_limits(torch.tensor([[4., 10., .9]] * 2))
    saved = commands.state_dict(rng)
    bad = deepcopy(saved)
    data = bad["air_limits"]["tensors"]
    if corrupt == "duration":
        data["remaining_s"][0] = 4.
    elif corrupt == "mode":
        data["mode_index"][0] = -1
    elif corrupt == "future_clock":
        data["last_policy_step"][0] = 99
    elif corrupt == "published_limit":
        data["applied_public_command"][0, 0] = 8.
    else:
        bad["air_limits"]["definition_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        commands.load_state_dict(bad, rng)
    assert_same(commands.state_dict(rng), saved)


def test_partial_ordinary_reset_clears_only_its_air_context_and_publication_skips_fail_atomically():
    commands, positions, fingerprint = configured(3)
    rng = torch.Generator().manual_seed(31)
    reset(commands, rng, fingerprint, torch.tensor([0, 0, 0]))
    commands.apply_air_birth_command_limits(torch.tensor([[4., 10., .9]] * 3))
    before = commands.diagnostics()["air_birth_command_limits"]
    reset(commands, rng, fingerprint, ids=torch.tensor([1]))
    after = commands.diagnostics()["air_birth_command_limits"]
    for key in before:
        assert torch.equal(before[key][[0, 2]], after[key][[0, 2]])
    assert after["mode_index"][1] == -1 and after["remaining_s"][1] == 0
    tick(commands, rng, positions)
    tick(commands, rng, positions)
    state = commands.state_dict(rng)
    public = torch.tensor([[4., 10., .9]] * 3)
    original = public.clone()
    with pytest.raises(ValueError, match="skipped"):
        commands.apply_air_birth_command_limits(public)
    assert torch.equal(public, original)
    assert_same(commands.state_dict(rng), state)


def test_actual_environment_publish_applies_limits_after_stepup_and_recovery_overrides():
    commands, positions, fingerprint = configured(2)
    rng = torch.Generator().manual_seed(37)
    reset(commands, rng, fingerprint, torch.tensor([0, -1]))
    public = torch.tensor([[.5, .1, .3]] * 2)
    commands.apply_air_birth_command_limits(public)
    env = NS(flat_episode=None, generator=rng, policy_dt=.02, tick=1, commands=public,
             raw_height=torch.zeros(2), cfg={}, recovery=None)
    env._flat_heading = lambda: torch.zeros(2)
    env._flat_command_iteration = lambda: 0
    env.flat_episode = NS(commands=commands, holding=torch.zeros(2, dtype=torch.bool),
        advance=lambda *args, **kwargs: tick(commands, rng, positions),
        publish=lambda target, *args: target.copy_(torch.stack((commands.command3[:, 0], commands.command3[:, 2], commands.height), -1)),
        bootstrap_mask=lambda terminated: ~terminated)
    env.mode, env.targets, env.jump_requested = torch.ones(2), torch.zeros(2, 4), torch.zeros(2, dtype=torch.bool)
    env.v7_stepup = NS(advance=lambda height, *args, **kwargs: height + .6, metrics=lambda: {})

    def recovery_override():
        public[:, 0], public[:, 1] = 4., 10.

    env.recovery = NS(source_flat_handoff=True, policy_motion_hold=torch.zeros(2, dtype=torch.bool),
                      override_commands=recovery_override, bootstrap_mask=lambda terminated: ~terminated)
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/application/environment.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    fn = deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_advance_flat_episode"))
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "actual_publish_with_air_limits", "exec"), scope)
    scope[fn.name](env, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {"log": {}})
    assert public[0].tolist() == pytest.approx([2.5, torch.pi / 2, .43])
    assert public[1, :2].tolist() == [4., 10.]
    assert public[1, 2] > .6
    assert commands.diagnostics()["air_birth_command_limits"]["last_policy_step"].tolist() == [1, 1]


def test_actual_reset_publication_runs_after_targets_and_ages_only_the_reset_subset():
    commands, _, fingerprint = configured(3)
    rng = torch.Generator().manual_seed(41)
    reset(commands, rng, fingerprint, torch.tensor([0, 0, 0]))
    public = torch.tensor([[4., 10., .9]] * 3)
    commands.apply_air_birth_command_limits(public)
    before = commands.diagnostics()["air_birth_command_limits"]
    reset(commands, rng, fingerprint, torch.tensor([0]), ids=torch.tensor([1]))
    public[1] = torch.tensor([4., 10., .9])
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/application/environment.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "reset")
    node = next(node for node in fn.body if isinstance(node, ast.If)
                and "apply_air_birth_command_limits" in ast.unparse(node))
    update = next(node for node in fn.body if isinstance(node, ast.Expr)
                  and "self.update_targets()" == ast.unparse(node))
    assert fn.body.index(node) > fn.body.index(update)
    exec(compile(ast.Module(body=[deepcopy(node)], type_ignores=[]), "actual_reset_air_publication", "exec"),
         {"self": NS(commands=public), "flat": NS(commands=commands), "ids": torch.tensor([1])})
    after = commands.diagnostics()["air_birth_command_limits"]
    assert public[1].tolist() == pytest.approx([2.5, torch.pi / 2, .43])
    for key in before:
        assert torch.equal(before[key][[0, 2]], after[key][[0, 2]])
    assert after["remaining_s"][1] == before["remaining_s"][1]


def test_explicit_enable_is_identity_bound_and_rejected_after_first_reset_without_mutation():
    commands, _ = owner(2, columns=[10, 10])
    rng = torch.Generator().manual_seed(47)
    before = commands.state_dict(rng)
    with pytest.raises(ValueError, match="authenticated"):
        commands.enable_air_birth_command_limits(birth_definition_sha256="0" * 64)
    assert_same(commands.state_dict(rng), before)
    fingerprint = FinalRoughBirthSampler(2, "cpu").definition_sha256
    with pytest.raises(ValueError, match="configured"):
        reset(commands, rng, fingerprint, torch.tensor([0, -1]))
    assert_same(commands.state_dict(rng), before)
    reset(commands, rng, fingerprint)
    current = commands.state_dict(rng)
    with pytest.raises(ValueError, match="fresh owner"):
        commands.enable_air_birth_command_limits(birth_definition_sha256=fingerprint)
    assert_same(commands.state_dict(rng), current)
