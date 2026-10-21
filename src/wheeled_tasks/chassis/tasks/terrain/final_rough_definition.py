"""Authenticated final Rough-v1 data, without importing the source task or SDK.

Active Python configuration is read as AST. The archived terrain serialization
supplies source-era SDK defaults only; explicitly declared values must agree
with that archive. Archived reward fields are never loaded.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


VERSION = "v7-final-rough-definition-v1"
FINAL_TERRAIN_SOURCE = "v7_final_b8ff79f3"
SOURCE_COMMIT = "b8ff79f3df855faf9dc92f4a282bd80c42649466"
REFERENCE = Path(__file__).resolve().parents[5] / "references/v7_final_terrain_b8ff79f3"
MANIFEST_SHA256 = "8e018e64de8ac599dabc283665cba5c83a029fe4ce7579b3eda1387d9fdb5233"

_FUNCTIONS = {
    "MeshPlaneTerrainCfg": "isaaclab.terrains.trimesh.mesh_terrains:flat_terrain",
    "HfPyramidSlopedTerrainCfg": "isaaclab.terrains.height_field.hf_terrains:pyramid_sloped_terrain",
    "HfInvertedPyramidSlopedTerrainCfg": "isaaclab.terrains.height_field.hf_terrains:pyramid_sloped_terrain",
    "HfPyramidStairsTerrainCfg": "isaaclab.terrains.height_field.hf_terrains:pyramid_stairs_terrain",
    "HfInvertedPyramidStairsTerrainCfg": "isaaclab.terrains.height_field.hf_terrains:pyramid_stairs_terrain",
    "HfRandomUniformTerrainCfg": "isaaclab.terrains.height_field.hf_terrains:random_uniform_terrain",
    "MeshCustomGridBarsTerrainCfg": "agent_world.terrains.height_field:mesh_grid_bars_terrain",
    "HfCliffInvertedPyramidStairsTerrainCfg": "agent_world.terrains.height_field:cliff_inverted_pyramid_stairs_terrain",
}


def source_manifest(root=REFERENCE):
    """Authenticate the complete frozen source closure on every read."""
    root = Path(root)
    raw = (root / "source_manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Final terrain source manifest SHA differs")
    manifest = json.loads(raw)
    if manifest["source_commit"] != SOURCE_COMMIT:
        raise ValueError("Final terrain source commit differs")
    for group in ("files", "derived"):
        for name, entry in manifest[group].items():
            data = (root / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("Final terrain source SHA differs: " + name)
            if group == "files":
                blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                if blob != entry["git_blob"]:
                    raise ValueError("Final terrain source Git blob differs: " + name)
    return manifest


def final_rough_source_files():
    """Repository-relative frozen dependencies for snapshot and resume binding."""
    manifest = source_manifest()
    prefix = "references/v7_final_terrain_b8ff79f3/"
    return [prefix + "source_manifest.json", *[prefix + name for name in manifest["files"]],
            *[prefix + name for name in manifest["derived"]]]


def _path(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (*_path(node.value), node.attr)
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return (*_path(node.value), node.slice.value)
    raise ValueError("Unsupported final configuration AST path")


def _assignments(scope):
    result = {}
    for node in scope.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        else:
            continue
        result[_path(target)] = node.value
    return result


def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _post_init(cls):
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")


def _value(node, names=None):
    """A data-only evaluator for the pinned configuration expression subset."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_value(item, names) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_value(key, names): _value(value, names) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _value(node.operand, names)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _value(node.left, names), _value(node.right, names)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
    if isinstance(node, ast.Attribute) and _path(node) == ("torch", "pi"):
        return math.pi
    if isinstance(node, ast.Name) and names is not None and (node.id,) in names:
        return _value(names[(node.id,)], names)
    if isinstance(node, ast.Call) and not node.args and all(item.arg is not None for item in node.keywords):
        return {"_config_class": _path(node.func)[-1],
                **{item.arg: _value(item.value, names) for item in node.keywords}}
    raise ValueError("Unsupported final configuration AST expression: " + ast.dump(node))


def _data(value):
    if isinstance(value, dict):
        return {key: _data(item) for key, item in value.items() if key != "_config_class"}
    if isinstance(value, list):
        return [_data(item) for item in value]
    return value


def _check_declared(actual, declared, label):
    for name, expected in declared.items():
        if name == "_config_class":
            continue
        if name in actual and isinstance(expected, dict):
            _check_declared(actual[name], expected, label + "." + name)
        elif name not in actual or actual[name] != _data(expected):
            raise ValueError("Final active AST differs from terrain archive: " + label + "." + name)


def load_final_rough_definition(root=REFERENCE):
    """Return fresh effective terrain, command, override, and birth dictionaries."""
    root = Path(root)
    source_manifest(root)
    trees = {name: ast.parse((root / (name + ".py")).read_text())
             for name in ("terrains", "cfg_utils", "env_cfg", "base_env_cfg", "commands", "terrain_command_manager")}
    cfg_names, terrain_names = _assignments(trees["cfg_utils"]), _assignments(trees["terrains"])
    env, base = trees["env_cfg"], trees["base_env_cfg"]
    flat = _class(env, "WheelbipeV14FlatEnvCfg")
    flat_v1 = _class(env, "WheelbipeV14FlatEnvCfg_v1")
    rough = _class(env, "WheelbipeV14RoughEnvCfg_v1")
    if [_path(item)[-1] for item in rough.bases] != [flat_v1.name]:
        raise ValueError("Final rough configuration inheritance differs")
    helper = next(item for item in trees["cfg_utils"].body
                  if isinstance(item, ast.FunctionDef) and item.name == "_apply_v14_rough_runtime_cfg")
    helper_assignments = _assignments(helper)
    fallback = any(isinstance(item, ast.Assign) and isinstance(item.value, ast.Attribute)
                   and _path(item.value) == ("mdp", "RM_ROUGH_TERRAINS_CFG")
                   and [_path(target) for target in item.targets] == [("raw_rough_terrain_gen",)]
                   for item in ast.walk(helper))
    if (_value(helper_assignments[("cfg", "curriculum")]) is not None
            or _value(helper_assignments[("rough_terrain_gen", "curriculum")]) is not True
            or not fallback):
        raise ValueError("Final rough runtime helper differs")
    terrain = json.loads((root / "archive_terrain_defaults.json").read_text())["terrain"]
    declared = _value(terrain_names[("RM_ROUGH_TERRAINS_CFG",)], terrain_names)
    declared["curriculum"] = True
    sub_terrains = declared.pop("sub_terrains")
    _check_declared(terrain["terrain_generator"], declared, "generator")
    if list(terrain["terrain_generator"]["sub_terrains"]) != list(sub_terrains):
        raise ValueError("Final active terrain keys differ from terrain archive")
    for name, fields in sub_terrains.items():
        actual = terrain["terrain_generator"]["sub_terrains"][name]
        if actual["function"] != _FUNCTIONS[fields["_config_class"]]:
            raise ValueError("Final terrain function differs: " + name)
        _check_declared(actual, fields, name)
    base_fields = _assignments(_class(base, "Wheelbipe25v3BaseEnvCfg"))
    importer = _value(base_fields[("terrain",)])
    importer["terrain_type"] = "generator"
    _check_declared(terrain, importer, "importer")
    terrain["boundary_reset_cfg"] = _value(_assignments(rough)[("rough_terrain_boundary_reset_cfg",)])
    terrain["height_offset_curriculum_cfg"] = _value(cfg_names[("V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG",)])
    terrain["performance_curriculum"] = None

    commands = _value(_assignments(_post_init(flat))[("self", "commands")])
    defaults = {path[0]: _value(value) for path, value in _assignments(_class(trees["commands"], "SpecialModeEntryCfg")).items()
                if not isinstance(value, ast.Name)}
    commands["special_modes"] = {name: {**deepcopy(defaults), **fields}
                                  for name, fields in commands["special_modes"].items()}
    for path, value in _assignments(_post_init(flat_v1)).items():
        if path[:3] == ("self", "commands", "special_modes"):
            target = commands["special_modes"]
            for key in path[3:-1]:
                target = target[key]
            result = _value(value)
            target[path[-1]] = {**deepcopy(defaults), **result} if len(path) == 4 else result
    commands["height_range"] = _value(_assignments(flat)[("height_range",)])
    commands["height_special_modes_cfg"] = _value(_assignments(_post_init(flat))[("self", "height_command_special_modes_cfg")])
    commands["terrain_command_switch_hold_steps"] = _value(_assignments(flat)[("terrain_command_switch_hold_steps",)])

    override_defaults = {path[0]: _value(value) for path, value in _assignments(
        _class(trees["terrain_command_manager"], "TerrainCommandOverrideCfg")).items()}
    overrides = _value(cfg_names[("V14_ROUGH_TERRAIN_COMMAND_OVERRIDES",)])
    overrides = {name: {**deepcopy(override_defaults), **_data(value)} for name, value in overrides.items()
                 if name in sub_terrains}
    headings = _value(base_fields[("reset_heading_axis_aligned_candidates_deg",)])
    birth = {"reset_heading_axis_aligned_only": _value(base_fields[("reset_heading_axis_aligned_only",)]),
             "reset_heading_axis_aligned_candidates_deg": headings,
             "axis_aligned_reset_headings_rad": [value / 180. * math.pi for value in headings],
             "disable_predefined_reset_air": False,
             "disable_predefined_reset_ground": False,
             "predefined_reset_air": _value(_assignments(flat_v1)[("predefined_reset_air",)])}
    return _data({"terrain": terrain, "commands": commands, "overrides": overrides, "birth": birth})


def load_final_rough_birth_definition(root=REFERENCE):
    """Read the effective final reset distribution, without selecting a course.

    The flat-v1 post-init changes the inherited ground weights. Air root poses
    add to the asset default, whereas ground poses replace its root height.
    These source values are requests; they do not qualify this chassis's IK,
    collision placement, velocity reference point, or native reset ownership.
    The existing terrain/command definition and restart identity stay separate.
    """
    root = Path(root)
    definition = load_final_rough_definition(root)
    trees = {name: ast.parse((root / (name + ".py")).read_text())
             for name in ("env_cfg", "base_env_cfg", "cfg_utils", "asset_cfg")}
    classes = {node.name: node for name in ("base_env_cfg", "env_cfg")
               for node in trees[name].body if isinstance(node, ast.ClassDef)}
    lineage, name = [], "WheelbipeV14RoughEnvCfg_v1"
    while name in classes:
        cls = classes[name]
        lineage.append(cls)
        if len(cls.bases) != 1:
            raise ValueError("Final birth configuration requires its single inheritance chain")
        name = _path(cls.bases[0])[-1]
    fields = {"use_leg_random_start", "use_predefined_leg_random_start",
              "use_joint_vel_random_start", "leg_length_range", "leg_angle_range",
              "wheel_angle_range", "leg_joint_vel_range", "wheel_joint_vel_range"}
    effective = {}
    ground = _data(_value(_assignments(trees["cfg_utils"])[("V14_PREDEFINED_RESET_GROUND",)]))
    for cls in reversed(lineage):
        for path, value in _assignments(cls).items():
            if len(path) == 1 and path[0] in fields:
                effective[path[0]] = _value(value)
        post = next((node for node in cls.body if isinstance(node, ast.FunctionDef)
                     and node.name == "__post_init__"), None)
        if post is not None:
            for path, value in _assignments(post).items():
                if len(path) == 2 and path[0] == "self" and path[1] in fields:
                    effective[path[1]] = _value(value)
                elif path[:2] == ("self", "predefined_reset_ground"):
                    target = ground
                    for key in path[2:-1]:
                        target = target[key]
                    target[path[-1]] = _value(value)
    if set(effective) != fields:
        raise ValueError("Final birth configuration is missing inherited reset fields")
    asset = _assignments(trees["asset_cfg"])[("Wheelbipe_V14_2_CFG",)]
    init = next(item.value for item in asset.keywords if item.arg == "init_state")
    declared = {item.arg: _value(item.value) for item in init.keywords
                if item.arg in ("pos", "rot", "lin_vel", "ang_vel")}
    defaults = json.loads((root / "archive_root_defaults.json").read_text())["init_state"]
    _check_declared(defaults, declared, "root_init_state")
    event = _assignments(_class(trees["env_cfg"], "EventCfgV14"))[("reset_base",)]
    if _path(next(item.value for item in event.keywords if item.arg == "func")) != (
            "mdp", "reset_root_state_uniform_vel_b"):
        raise ValueError("Final birth root reset event differs")
    root_event = _value(next(item.value for item in event.keywords if item.arg == "params"))
    return _data({"version": "v7-final-rough-birth-definition-v1",
        "source_commit": SOURCE_COMMIT, "source_manifest_sha256": MANIFEST_SHA256,
        **effective, "source_root_init_state": defaults, "root_reset_event": root_event,
        "predefined_reset_air": definition["birth"]["predefined_reset_air"],
        "predefined_reset_ground": ground,
        "reset_heading_axis_aligned_candidates_deg": definition["birth"][
            "reset_heading_axis_aligned_candidates_deg"],
        "terrain_overrides": {key: {field: value[field] for field in (
            "reset_heading_axis_aligned_only", "disable_predefined_reset_air",
            "disable_predefined_reset_ground")} for key, value in definition["overrides"].items()}})
