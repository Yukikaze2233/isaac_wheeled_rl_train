"""CPU evidence for final active definitions and the frozen source closure."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis.tasks.terrain import final_rough_definition as final


def _assignment(tree, name):
    return next(node for node in tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and ((isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == name)
                     or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                         and node.target.id == name)))


class _Configs:
    def __getattr__(self, name):
        return lambda **kwargs: {"class": name, **kwargs}


def _active_geometry_oracle():
    tree = ast.parse((final.REFERENCE / "terrains.py").read_text())
    generator = _assignment(tree, "RM_ROUGH_TERRAINS_CFG")
    entries = next(item.value for item in generator.value.keywords if item.arg == "sub_terrains")
    dependencies = [item.id for item in entries.values]
    nodes = [_assignment(tree, name) for name in dependencies] + [generator]
    env = {"terrain_gen": _Configs(), "TerrainGeneratorCfg": lambda **kwargs: kwargs}
    for node in nodes:
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                env.setdefault(call.func.id, lambda **kwargs: kwargs)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "frozen_active_geometry", "exec"), env)
    return env["RM_ROUGH_TERRAINS_CFG"]


def _yaml_archive():
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    for tag in ("python/tuple", "python/object/apply:builtins.slice"):
        Loader.add_constructor("tag:yaml.org,2002:" + tag, lambda loader, node: loader.construct_sequence(node))
    return yaml.load((final.REFERENCE / "rough_dash_env.yaml").read_bytes(), Loader=Loader)


def test_authenticated_files_are_git_blobs_and_source_closure_has_no_missing_dependencies():
    manifest = final.source_manifest()
    assert manifest["source_commit"] == final.SOURCE_COMMIT
    assert manifest["registered_environment"] == "Robotics-Wheelbipe-V14-Rough-v1"
    files = final.final_rough_source_files()
    assert len(files) == len(set(files)) == 17
    assert all((final.REFERENCE.parents[1] / item).is_file() for item in files)
    for name, entry in manifest["files"].items():
        data = (final.REFERENCE / name).read_bytes()
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == entry["git_blob"]
    registration = ast.parse((final.REFERENCE / "registration.py").read_text())
    calls = [node.value for node in registration.body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
             and node.value.func.attr == "register"]
    selected = next(call for call in calls if any(item.arg == "id" and isinstance(item.value, ast.Constant)
        and item.value.value == "Robotics-Wheelbipe-V14-Rough-v1" for item in call.keywords))
    assert "WheelbipeV14RoughEnvCfg_v1" in ast.unparse(selected)


def test_active_ast_geometry_agrees_with_archive_without_selecting_its_rewards():
    definition = final.load_final_rough_definition()
    actual = definition["terrain"]["terrain_generator"]
    oracle = _active_geometry_oracle()
    assert oracle["curriculum"] is False and actual["curriculum"] is True
    for field, value in oracle.items():
        if field in ("sub_terrains", "curriculum"):
            continue
        assert actual[field] == list(value) if isinstance(value, tuple) else actual[field] == value
    assert list(actual["sub_terrains"]) == list(oracle["sub_terrains"])
    for name, source in oracle["sub_terrains"].items():
        for key, value in source.items():
            if key == "class":
                continue
            assert actual["sub_terrains"][name][key] == list(value) if isinstance(value, tuple) else actual["sub_terrains"][name][key] == value
    archive = _yaml_archive()
    assert actual == archive["terrain"]["terrain_generator"]
    assert len([weight for weight in archive["rewards"].values() if weight != 0]) == 33
    assert "rewards" not in definition and "rewards" not in json.dumps(definition)
    assert (actual["size"], actual["num_rows"], actual["num_cols"], actual["border_width"]) == ([9., 9.], 10, 13, 10.)
    assert Counter(actual["sub_terrains"].keys()) == Counter(oracle["sub_terrains"].keys())
    assert sum(item["proportion"] for item in actual["sub_terrains"].values()) == pytest.approx(1.3)
    assert actual["sub_terrains"]["tiny_step"]["bar_height_range"] == [.02, .06]
    assert actual["sub_terrains"]["cliff_inv_stair_slope_short_for_rm"]["height_offset_range"] == [.3, .4]
    assert actual["sub_terrains"]["slope_for_rm_high"]["border_width"] == 2.


def test_source_defaults_keep_generation_difficulty_separate_from_performance_progression():
    terrain = final.load_final_rough_definition()["terrain"]
    assert terrain["max_init_terrain_level"] is None
    assert terrain["performance_curriculum"] is None
    assert terrain["boundary_reset_cfg"] == {"enabled": True, "margin": .5, "use_inner_terrain_area": False}
    assert terrain["height_offset_curriculum_cfg"] == {
        "enabled": False, "interval": 500, "max_iteration": 5000, "num_levels": 11,
        "steps_per_iteration": 24, "random_reset_up_to_current_level": False,
        "random_reset_after_max": True, "randomize_type_on_random_reset": True}


def test_final_commands_have_final_zero_start_modes_and_full_source_ranges():
    commands = final.load_final_rough_definition()["commands"]
    assert commands["height_range"] == [.20, .42]
    assert commands["ranges"] == {"lin_vel_x": [-2.7, 2.7], "lin_vel_y": [0., 0.],
                                  "ang_vel_z": [-2 * math.pi, 2 * math.pi], "heading": [-math.pi, math.pi]}
    assert commands["resampling_time_range"] == [5., 15.]
    assert (commands["rel_standing_envs"], commands["rel_heading_envs"], commands["heading_control_stiffness"]) == (.1, .5, 5.)
    assert commands["special_mode_min_episode_time"] == 5.
    assert commands["special_mode_require_stable"] is False
    modes = commands["special_modes"]
    assert list(modes) == ["spin_low", "spin_mid", "dash", "zero_cmd"]
    assert [mode["rel_envs"] for mode in modes.values()] == [.15, .15, .2, .1]
    assert all(mode["iteration_start"] == 0 and mode["iteration_end"] == -1 for mode in modes.values())
    assert all(mode["height_range"] is None and mode["height_wave"] is None and mode["height_step"] is None for mode in modes.values())
    assert modes["spin_mid"]["ranges"]["ang_vel_z"] == [[3.25 * math.pi, 4.5 * math.pi], [-4.5 * math.pi, -3.25 * math.pi]]
    assert modes["dash"]["ranges"]["lin_vel_x"] == [[2., 3.], [-3., -2.]]
    assert modes["dash"]["disable_jump_takeoff"] and modes["zero_cmd"]["disable_jump_takeoff"]
    assert commands["height_special_modes_cfg"]["enabled"] is False


def test_only_active_overrides_and_explicit_birth_defaults_are_retained():
    definition = final.load_final_rough_definition()
    overrides, birth = definition["overrides"], definition["birth"]
    assert len(overrides) == 10 and "plane_for_rm" not in overrides and "high_stair_for_rm" not in overrides
    fields = {"height_range", "lin_vel_x", "lin_vel_y", "ang_vel_z", "ang_vel_z_heading",
              "ang_vel_z_non_heading", "reset_heading_axis_aligned_only", "disable_predefined_reset_air",
              "disable_predefined_reset_ground", "disable_jump_takeoff", "disable_special_mode"}
    assert all(set(item) == fields for item in overrides.values())
    assert overrides["inv_slope_for_rm_low"]["height_range"] == [[.22, .42]]
    assert all(item["disable_predefined_reset_ground"] is None for item in overrides.values())
    assert overrides["high_speed_stair_for_rm"]["lin_vel_x"] == [[1.5, 2.7], [-2.7, -1.5]]
    assert overrides["high_speed_stair_for_rm"]["ang_vel_z_heading"] is None
    assert overrides["high_speed_stair_for_rm"]["ang_vel_z_non_heading"] == [-.1, .1]
    assert overrides["high_speed_stair_for_rm"]["disable_predefined_reset_air"] is True
    assert overrides["tiny_step"]["height_range"] == [[.22, .32]]
    assert birth["reset_heading_axis_aligned_only"] is False
    assert birth["disable_predefined_reset_air"] is birth["disable_predefined_reset_ground"] is False
    assert birth["axis_aligned_reset_headings_rad"] == [0., math.pi / 2, math.pi, -math.pi / 2]
    # The frozen source uses parentheses around a single dictionary, no tuple
    # comma. Preserve that representation instead of inventing an extra mode.
    air = birth["predefined_reset_air"]
    assert air["enabled"] is True and air["modes"]["prob"] == .3
    assert air["modes"]["pose_range"]["z"] == [.12, .42]
    assert air["modes"]["command_limits"]["height"] == [.18, .43]


@pytest.mark.parametrize("name", ["source_manifest.json", "terrains.py", "cfg_utils.py", "height_field.py",
                                 "archive_terrain_defaults.json", "asset_cfg.py", "reset_events.py",
                                 "archive_root_defaults.json"])
def test_any_source_tampering_is_rejected(name, tmp_path):
    shutil.copytree(final.REFERENCE, tmp_path / "reference")
    path = tmp_path / "reference" / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA differs"):
        final.load_final_rough_definition(tmp_path / "reference")


def test_loader_returns_detached_data_and_entry_remains_torch_sdk_free():
    definition = final.load_final_rough_definition()
    snapshot = deepcopy(definition)
    definition["terrain"]["terrain_generator"]["num_cols"] = 14
    definition["commands"]["special_modes"]["dash"]["rel_envs"] = 1.
    assert final.load_final_rough_definition() == snapshot
    code = "from wheeled_tasks.chassis.tasks.terrain.final_rough_definition import load_final_rough_definition; import sys; load_final_rough_definition(); assert not any(n=='torch' or n.startswith(('torch.','isaaclab','isaacsim','warp','omni','pxr')) for n in sys.modules)"
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_final_birth_definition_resolves_inheritance_and_post_init_without_changing_commands():
    commands = final.load_final_rough_definition()
    birth = final.load_final_rough_birth_definition()
    assert final.load_final_rough_definition() == commands
    assert birth["source_commit"] == final.SOURCE_COMMIT
    assert all(birth[key] is True for key in ("use_leg_random_start", "use_predefined_leg_random_start",
                                             "use_joint_vel_random_start"))
    assert birth["leg_length_range"] == [.13, .32]
    assert birth["leg_angle_range"] == [-math.pi / 2, 3 * math.pi / 4]
    assert birth["leg_joint_vel_range"] == [-math.pi / 2, math.pi / 2]
    assert birth["wheel_joint_vel_range"] == [-50., 50.]
    assert birth["wheel_angle_range"] == [-2 * math.pi, 2 * math.pi]
    ground = birth["predefined_reset_ground"]
    assert [mode["prob"] for mode in ground["modes"].values()] == [.2, .1]
    assert ground["prob"] == .2  # Legacy scalar is unused when modes is present.
    assert (ground["start_root_height"], ground["start_reset_time"], ground["zero_torque_time_s"]) == (.25, 1.5, .2)
    assert birth["predefined_reset_air"]["modes"]["prob"] == .3
    root = birth["source_root_init_state"]
    assert root == {"pos": [0., 0., .38], "rot": [1., 0., 0., 0.],
                    "lin_vel": [0., 0., 0.], "ang_vel": [0., 0., 0.]}
    offset = birth["predefined_reset_air"]["modes"]["pose_range"]["z"]
    assert [root["pos"][2] + z for z in offset] == pytest.approx([.50, .80])
    assert birth["root_reset_event"] == {"pose_range": {"roll": [-.15, .15],
        "pitch": [-.15, .15], "yaw": [-3.14, 3.14]}, "velocity_range": {}}
    birth["predefined_reset_ground"]["modes"]["positive"]["prob"] = 1.
    assert final.load_final_rough_birth_definition()["predefined_reset_ground"]["modes"]["positive"]["prob"] == .2
