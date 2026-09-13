"""CPU import-boundary regression: fake USD + real tensors, never import Isaac/pxr."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from wheeled_tasks.v40.contract import audit_asset, load_contract

ROOT = Path(__file__).resolve().parents[2]
FLOAT32_MAX = torch.finfo(torch.float32).max


@pytest.fixture
def joint_contract():
    return load_contract()["joints"]


@pytest.fixture
def adapter():
    path = ROOT / "src/wheeled_world/assets/v40.py"
    names = {"normalize_v40_usd_joint_limits", "validate_v40_physx_joint_limits"}
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(normalize=namespace["normalize_v40_usd_joint_limits"],
                           validate=namespace["validate_v40_physx_joint_limits"])


class FakeAttribute:
    def __init__(self, stage, value):
        self.stage = stage
        self.value = value

    def Get(self):
        return self.value

    def Set(self, value):
        self.stage.edits += 1
        if self.stage.write_failure:
            return False
        if not self.stage.stale_readback:
            self.value = value
        return True


class FakeJoint:
    def __init__(self, prim):
        self.prim = prim

    def GetLowerLimitAttr(self):
        return self.prim.lower

    def GetUpperLimitAttr(self):
        return self.prim.upper

    CreateLowerLimitAttr = GetLowerLimitAttr
    CreateUpperLimitAttr = GetUpperLimitAttr


SCHEMA = SimpleNamespace(Joint=object(), RevoluteJoint=FakeJoint)


class FakePrim:
    def __init__(self, stage, path, limits=None):
        self.path = path
        self.is_joint = limits is not None
        self.is_revolute = self.is_joint
        self.schemas = ["PhysxJointAPI", "PhysxLimitAPI:angular"] if self.is_joint else []
        # These unrelated properties must survive the adapter unchanged.
        self.properties = {"effort": 100, "velocity": 1, "axis": "Z", "drive_stiffness": 0}
        if limits is not None:
            self.lower, self.upper = [FakeAttribute(stage, value) for value in limits]

    def GetPath(self):
        return self.path

    def GetName(self):
        return self.path.rsplit("/", 1)[1]

    def IsValid(self):
        return True

    def IsA(self, schema):
        return self.is_joint if schema is SCHEMA.Joint else self.is_revolute

    def GetAppliedSchemas(self):
        return self.schemas


class FakeStage:
    def __init__(self, contract, num_envs=2):
        self.edits = 0
        self.stale_readback = False
        self.write_failure = False
        self.env_paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
        self.prims = {}
        for env in self.env_paths:
            root = env + "/Robot"
            self.prims[root] = FakePrim(self, root)
            # Deliberately shuffled imported joint order and nested prim paths.
            for name in reversed(contract["action_order"]):
                radians = contract["knee_hard_limits"].get(name, [-3.14, 3.14])
                # USD revolute attributes are float32 degrees, not radians.
                degrees = [struct.unpack("f", struct.pack("f", math.degrees(v)))[0] for v in radians]
                path = root + "/joints/" + name
                self.prims[path] = FakePrim(self, path, degrees)

    def Traverse(self):
        return list(self.prims.values())

    def GetPrimAtPath(self, path):
        return self.prims.get(path)


def test_continuous_normalized_all_clones_knees_and_other_properties_untouched(adapter, joint_contract):
    stage = FakeStage(joint_contract)
    other = FakePrim(stage, "/World/unrelated/Robot/joints/L_joint3", [-179.908, 179.908])
    stage.prims[other.path] = other
    knees_before = {path: (p.lower.Get(), p.upper.Get()) for path, p in stage.prims.items()
                    if p.GetName() in joint_contract["knee_hard_limits"]}
    properties_before = {path: (dict(p.properties), list(p.schemas)) for path, p in stage.prims.items()}
    result = adapter.normalize(stage, stage.env_paths, joint_contract, usd_physics=SCHEMA)
    assert result["env_count"] == 2 and stage.edits == 16
    for path, prim in stage.prims.items():
        assert (prim.properties, prim.schemas) == properties_before[path]
        if path in knees_before:
            assert (prim.lower.Get(), prim.upper.Get()) == knees_before[path]
        elif prim.is_joint and prim is not other:
            assert (prim.lower.Get(), prim.upper.Get()) == (-math.inf, math.inf)
    assert (other.lower.Get(), other.upper.Get()) == (-179.908, 179.908)
    assert adapter.normalize(stage, stage.env_paths, joint_contract, usd_physics=SCHEMA) == result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "unknown", "wrong_schema", "additional_limit", "bad_knee", "nan_knee", "missing_root",
])
def test_all_usd_clones_validated_before_edits(adapter, joint_contract, mutation):
    stage = FakeStage(joint_contract)
    root = stage.env_paths[1] + "/Robot"
    path = root + "/joints/L_joint3"
    if mutation == "missing":
        del stage.prims[path]
    elif mutation in ("duplicate", "unknown"):
        name = "L_joint3" if mutation == "duplicate" else "extra_joint"
        path = root + "/other/" + name
        stage.prims[path] = FakePrim(stage, path, [-180., 180.])
    elif mutation == "wrong_schema":
        stage.prims[path].is_revolute = False
    elif mutation == "additional_limit":
        stage.prims[path].schemas.append("PhysicsLimitAPI:rotZ")
    elif mutation == "missing_root":
        del stage.prims[root]
    else:
        stage.prims[root + "/joints/L_joint2"].lower.value = math.nan if mutation == "nan_knee" else -180.
    with pytest.raises(ValueError):
        adapter.normalize(stage, stage.env_paths, joint_contract, usd_physics=SCHEMA)
    assert stage.edits == 0


@pytest.mark.parametrize("failure", ["stale_readback", "write_failure"])
def test_usd_failed_authoring_or_composition_is_fatal(adapter, joint_contract, failure):
    stage = FakeStage(joint_contract)
    setattr(stage, failure, True)
    with pytest.raises(RuntimeError, match="readback mismatch|failed writing"):
        adapter.normalize(stage, stage.env_paths, joint_contract, usd_physics=SCHEMA)


def solver_fixture(contract):
    import warp as wp
    wp.init()
    names = list(reversed(contract["action_order"]))
    limits = torch.tensor([contract["knee_hard_limits"].get(name, [-math.inf, math.inf]) for name in names])
    limits = limits.repeat(2, 1, 1)
    # A correct-looking cached tensor must NOT mask wrong actual solver values.
    robot = SimpleNamespace(joint_names=names, data=SimpleNamespace(joint_pos_limits=limits.clone()),
                            root_view=SimpleNamespace(get_dof_limits=lambda: wp.from_torch(limits)))
    return robot, limits


def test_solver_reads_actual_tensor_by_name_for_every_clone(adapter, joint_contract):
    robot, limits = solver_fixture(joint_contract)
    before = limits.clone()
    report = adapter.validate(robot, joint_contract, num_envs=2)
    assert report["shape"] == [2, 6, 2]
    assert report["limits_rad_env0"]["R_joint3"] == ["-inf", "+inf"]
    assert report["limit_encoding_env0"]["R_joint3"] == "ieee_infinity"
    assert report["limits_rad_env0"]["L_joint2"] == limits[0, robot.joint_names.index("L_joint2")].tolist()
    assert report["limit_encoding_env0"]["L_joint2"] == "finite_knee"
    torch.testing.assert_close(limits, before)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("native_envs", [(0,), (1,), (0, 1)])
def test_physx_float32_extrema_accepted_and_reported_without_rewriting(adapter, joint_contract, native_envs):
    robot, limits = solver_fixture(joint_contract)
    # Live server readback: exact +/-FLT_MAX, including all four continuous joints.
    assert FLOAT32_MAX == 3.4028234663852886e38
    for env_id in native_envs:
        for name in joint_contract["continuous"]:
            limits[env_id, robot.joint_names.index(name)] = torch.tensor([-FLOAT32_MAX, FLOAT32_MAX])
    before = limits.clone()
    report = adapter.validate(robot, joint_contract, num_envs=2)
    assert report["env_count"] == 2 and report["shape"] == [2, 6, 2]
    for name in joint_contract["action_order"]:
        if name not in joint_contract["continuous"]:
            encoding = "finite_knee"
            raw = limits[0, robot.joint_names.index(name)].tolist()
        elif 0 in native_envs:
            encoding, raw = "physx_float32_extrema", [-FLOAT32_MAX, FLOAT32_MAX]
        else:
            encoding, raw = "ieee_infinity", ["-inf", "+inf"]
        assert report["limits_rad_env0"][name] == raw
        assert report["limit_encoding_env0"][name] == encoding
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    torch.testing.assert_close(limits, before)


@pytest.mark.parametrize("name", ["L_joint1", "R_joint1", "L_joint3", "R_joint3"])
@pytest.mark.parametrize("bounds", [
    [-3.1399999, 3.1399999], [-1e9, 1e9], [-1e10, 1e10], [-math.inf, 3.14],
    [-3.14, math.inf], [math.nan, math.inf], [math.inf, -math.inf],
    [-FLOAT32_MAX, 1e10], [-1e10, FLOAT32_MAX], [math.nan, FLOAT32_MAX],
    [-FLOAT32_MAX, math.nan], [FLOAT32_MAX, -FLOAT32_MAX],
    [-math.inf, FLOAT32_MAX], [-FLOAT32_MAX, math.inf],
])
def test_live_hardstop_and_other_noncontinuous_solver_limits_fail(adapter, joint_contract, name, bounds):
    robot, limits = solver_fixture(joint_contract)
    limits[1, robot.joint_names.index(name)] = torch.tensor(bounds)
    with pytest.raises(RuntimeError, match=f"env 1, {name}"):
        adapter.validate(robot, joint_contract, num_envs=2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("column", [0, 1])
def test_native_extrema_must_be_exact_not_nearby_finite_values(adapter, joint_contract, dtype, column):
    robot, limits = solver_fixture(joint_contract)
    limits = limits.to(dtype)
    import warp as wp
    robot.root_view.get_dof_limits = lambda: wp.from_torch(limits)
    index = robot.joint_names.index("R_joint3")
    limits[1, index] = torch.tensor([-FLOAT32_MAX, FLOAT32_MAX], dtype=dtype)
    # The native sentinel stays float32 FLT_MAX even when the API data is widened.
    adapter.validate(robot, joint_contract, num_envs=2)
    limits[1, index, column] = torch.nextafter(limits[1, index, column], limits.new_tensor(0.))
    with pytest.raises(RuntimeError, match="env 1, R_joint3"):
        adapter.validate(robot, joint_contract, num_envs=2)


@pytest.mark.parametrize("name", ["L_joint2", "R_jonit2"])
@pytest.mark.parametrize("bounds", [[-math.inf, math.inf], [-FLOAT32_MAX, FLOAT32_MAX],
                                    [-3.14, 3.14], [0., 0.], [math.nan, .5]])
def test_physical_knee_limits_must_remain_finite_and_exact(adapter, joint_contract, name, bounds):
    robot, limits = solver_fixture(joint_contract)
    limits[1, robot.joint_names.index(name)] = torch.tensor(bounds)
    with pytest.raises(RuntimeError, match=f"env 1, {name}"):
        adapter.validate(robot, joint_contract, num_envs=2)


@pytest.mark.parametrize("mutation", ["missing_name", "duplicate_name", "unknown_name", "shape", "env_count"])
def test_solver_mismatched_names_and_tensor_shape_fail(adapter, joint_contract, mutation):
    robot, limits = solver_fixture(joint_contract)
    if mutation == "missing_name":
        robot.joint_names.pop()
    elif mutation == "duplicate_name":
        robot.joint_names[0] = robot.joint_names[1]
    elif mutation == "unknown_name":
        robot.joint_names[0] = "other"
    elif mutation == "shape":
        import warp as wp
        robot.root_view.get_dof_limits = lambda: wp.from_torch(limits[0])
    with pytest.raises(RuntimeError, match="mismatch"):
        adapter.validate(robot, joint_contract, num_envs=1 if mutation == "env_count" else 2)


def test_hash_bound_urdf_still_preserves_source_continuous_effort_velocity():
    audited = audit_asset(load_contract())
    for filename in ("source.urdf", "robot.urdf"):
        joints = ET.parse(audited["directory"] / filename).getroot().findall("joint")
        continuous = [j for j in joints if j.get("type") == "continuous"]
        assert len(continuous) == 4
        for joint in continuous:
            limit = joint.find("limit")
            assert [float(limit.get(key)) for key in ("lower", "upper", "effort", "velocity")] == [-3.14, 3.14, 100., 1.]
    assert audited["raw_manifest"]["collision_validation"]["passed"] is False


def test_startup_and_checker_use_the_boundary_and_fresh_solver_check():
    source = (ROOT / "src/wheeled_tasks/direct/v40_serial/env.py").read_text()
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef))
    methods = {node.name: ast.get_source_segment(source, node) for node in cls.body if isinstance(node, ast.FunctionDef)}
    setup = methods["_setup_scene"]
    assert setup.index("clone_environments(") < setup.index("normalize_v40_usd_joint_limits(")
    assert "self.sim.stage" in setup and "self.sim.reset(" not in setup
    init = methods["__init__"]
    assert init.index("super().__init__(") < init.index("self.check_physics_joint_limits()") < init.index("self.actions =")
    assert "validate_v40_physx_joint_limits(self.robot" in methods["check_physics_joint_limits"]
    checker = (ROOT / "scripts/check_v40_env.py").read_text()
    assert checker.index("obs, _ = env.reset()") < checker.index("env.check_physics_joint_limits()")
    assert checker.count("env.check_physics_joint_limits()") == 2
