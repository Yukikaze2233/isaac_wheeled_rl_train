"""CPU-only source-AST oracle; no IsaacLab package/Env/App/model import."""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from heading_candidate import resolve_heading_commands


HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "inputs001.json").read_text())


def bound_source(key):
    ref = MANIFEST["inputs"][key]
    text = Path(ref["file"]).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == ref["sha256"]
    return text


def original_function(key, name, namespace, *, owner=None):
    tree = ast.parse(bound_source(key))
    scope = tree if owner is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = deepcopy(next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, MANIFEST["inputs"][key]["file"], "exec"), namespace)
    return namespace[name]


def source_oracle(commands, target, world, heading, standing, special, ordinary, *, enabled=True,
                  stiffness=5., yaw_range=(-2 * math.pi, 2 * math.pi)):
    wrap = original_function("native_wrap_dependency", "wrap_to_pi", {"torch": torch})
    update = original_function("scut_selector", "_update_command",
                               {"torch": torch, "math_utils": SimpleNamespace(wrap_to_pi=wrap)},
                               owner="SpecialModeUniformVelocityCommand")
    # Source has [vx, vy, yaw]; native has [vx, yaw, height].
    velocity = torch.stack((commands[:, 0], torch.zeros_like(commands[:, 0]), commands[:, 1]), -1)
    source = SimpleNamespace(vel_command_b=velocity,
        cfg=SimpleNamespace(heading_command=enabled, heading_control_stiffness=stiffness,
                            ranges=SimpleNamespace(ang_vel_z=yaw_range)),
        special_mode_id=special, is_heading_env=heading & ordinary,
        is_standing_env=standing & ordinary, heading_target=target,
        robot=SimpleNamespace(data=SimpleNamespace(heading_w=world)))
    update(source)
    return torch.stack((velocity[:, 0], velocity[:, 2], commands[:, 2]), -1)


def test_oracle_is_the_source_special_class_method_not_a_same_named_method():
    update = original_function("scut_selector", "_update_command", {"torch": torch},
                               owner="SpecialModeUniformVelocityCommand")
    assert update.__code__.co_firstlineno == 577
    assert "special_mode_id" in update.__code__.co_names


def arguments(size=4096, dtype=torch.float64):
    generator = torch.Generator(device="cpu").manual_seed(20261008)
    commands = torch.rand(size, 3, dtype=dtype, generator=generator)
    commands[:, 0] = commands[:, 0] * 5.4 - 2.7
    commands[:, 1] = commands[:, 1] * (4 * math.pi) - 2 * math.pi
    commands[:, 2] = commands[:, 2] * .19 + .23
    return commands, dict(
        heading_target=(torch.rand(size, dtype=dtype, generator=generator) * 20 - 10) * math.pi,
        heading_w=(torch.rand(size, dtype=dtype, generator=generator) * 20 - 10) * math.pi,
        is_heading_env=torch.rand(size, generator=generator) < .5,
        is_standing_env=torch.rand(size, generator=generator) < .1,
        special_mode_id=torch.randint(-1, 4, (size,), generator=generator),
        ordinary_mask=torch.rand(size, generator=generator) < .5)


def expected(commands, args, **kwargs):
    return source_oracle(commands, args["heading_target"], args["heading_w"],
        args["is_heading_env"], args["is_standing_env"], args["special_mode_id"],
        args["ordinary_mask"], **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enabled", [False, True])
def test_actual_source_method_exact_cpu_tensor_equivalence(dtype, enabled):
    commands, args = arguments(dtype=dtype)
    before = commands.clone(), {k: v.clone() for k, v in args.items()}
    output = resolve_heading_commands(commands, **args, heading_command=enabled)
    assert torch.equal(output, expected(commands, args, enabled=enabled))
    assert torch.equal(output[:, 2], commands[:, 2])
    assert torch.equal(output[~args["ordinary_mask"]], commands[~args["ordinary_mask"]])
    assert torch.equal(commands, before[0])
    assert all(torch.equal(args[k], value) for k, value in before[1].items())


@pytest.mark.parametrize("angle", [-5 * math.pi, -math.pi, 0., math.pi, 5 * math.pi,
                                     math.pi - 1e-9, math.pi + 1e-9, -math.pi - 1e-9])
def test_actual_wrap_pi_boundary_and_multi_turn(angle):
    commands, args = arguments(size=1)
    args.update(heading_target=torch.tensor([angle], dtype=torch.float64),
                heading_w=torch.zeros(1, dtype=torch.float64),
                is_heading_env=torch.ones(1, dtype=torch.bool),
                is_standing_env=torch.zeros(1, dtype=torch.bool),
                special_mode_id=torch.full((1,), -1), ordinary_mask=torch.ones(1, dtype=torch.bool))
    result = resolve_heading_commands(commands, **args, stiffness=1., yaw_range=(-20., 20.))
    assert torch.equal(result, expected(commands, args, stiffness=1., yaw_range=(-20., 20.)))
    if angle == math.pi:
        assert result[0, 1] == math.pi
    elif angle == -math.pi:
        assert result[0, 1] == -math.pi


def test_special_skip_and_standing_last_and_height_preserved():
    commands, args = arguments(size=3)
    args.update(is_heading_env=torch.ones(3, dtype=torch.bool),
        special_mode_id=torch.tensor([0, 1, -1]), is_standing_env=torch.tensor([False, True, True]),
        ordinary_mask=torch.ones(3, dtype=torch.bool))
    result = resolve_heading_commands(commands, **args)
    assert torch.equal(result, expected(commands, args))
    assert torch.equal(result[0], commands[0])
    assert torch.equal(result[1:, :2], torch.zeros(2, 2, dtype=commands.dtype))
    assert torch.equal(result[:, 2], commands[:, 2])


@pytest.mark.parametrize("domain", ["speed", "terrain", "jump", "stepjump"])
def test_real_frozen_row_scope_keeps_dedicated_and_eight_terrain_bindings(domain):
    recipe = json.loads(bound_source("recipe"))["domains"][domain]["patch"]
    names = [r["name"] for r in recipe["scene_groups"]]
    mapped = recipe["scut_reuse_candidate"]["mapped_groups"]
    ordinary = torch.tensor([mapped.get(n, {}).get("kind") == "normal" for n in names])
    assert ordinary.sum() == 24
    explicit = [n for n, value in mapped.items() if value["kind"] == "explicit_terrain"]
    assert len(explicit) == (0 if domain == "speed" else 8)
    commands, args = arguments(size=len(names))
    args.update(is_heading_env=torch.ones(len(names), dtype=torch.bool),
        is_standing_env=torch.ones(len(names), dtype=torch.bool), ordinary_mask=ordinary,
        special_mode_id=torch.full((len(names),), -1))
    output = resolve_heading_commands(commands, **args)
    assert torch.equal(output[~ordinary], commands[~ordinary])
    assert torch.equal(output[ordinary, :2], torch.zeros(24, 2, dtype=commands.dtype))
    assert torch.equal(output[:, 2], commands[:, 2])


def test_native_original_scut35_keeps_abi_and_only_receives_resolved_yaw():
    namespace = {"torch": torch, "POLICY_FROM_CONTROL": (0, 1, 3, 4, 2, 5)}
    build = original_function("native_observation", "build_scut35", namespace)
    commands, args = arguments(size=16)
    args.update(ordinary_mask=torch.ones(16, dtype=torch.bool),
                special_mode_id=torch.full((16,), -1))
    output = resolve_heading_commands(commands, **args)
    zeros = torch.zeros(16, 6, dtype=commands.dtype)
    obs = build(torch.zeros(16, 3, dtype=commands.dtype), torch.zeros(16, 3, dtype=commands.dtype),
        output, zeros, zeros, zeros, zeros, torch.zeros(16, dtype=torch.bool),
        torch.zeros(16, dtype=commands.dtype), torch.zeros(16, dtype=commands.dtype))
    assert obs.shape == (16, 35)
    assert torch.equal(obs[:, 0], output[:, 0])
    assert torch.equal(obs[:, 2], output[:, 1])
    assert torch.equal(obs[:, 3], output[:, 2] * 5.)
    assert torch.equal(obs[:, 16:22], zeros)
    assert torch.equal(obs[:, 22:28], zeros)


@pytest.mark.parametrize("bad", ["command_width", "mask_dtype", "world_shape"])
def test_reject_wrong_native_abi(bad):
    commands, args = arguments(size=2)
    if bad == "command_width":
        commands = commands[:, :2]
    elif bad == "mask_dtype":
        args["ordinary_mask"] = args["ordinary_mask"].long()
    else:
        args["heading_w"] = args["heading_w"][:, None]
    with pytest.raises(ValueError):
        resolve_heading_commands(commands, **args)
