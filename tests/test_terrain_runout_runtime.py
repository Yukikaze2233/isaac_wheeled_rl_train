"""CPU USD/API and fake-env readback tests, not native simulator acceptance."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sysconfig
from types import MethodType, SimpleNamespace

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
import pytest
import torch

from wheeled_tasks.chassis import terrain_runout_revision as recipe
from wheeled_tasks.chassis import terrain_runout_runtime as runtime
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import validate_design_contract, validate_runtime_state
from wheeled_tasks.chassis.task import Surface, choose_scene_groups, corridor_mesh
from wheeled_tasks.chassis.terrain_runout import append_route_runout, validate_runout


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / recipe.SOURCE_WORKER_FILE
WORKER = json.loads(WORKER_PATH.read_text())
SETTINGS = dict(enabled=True, support_end_x_m=6.2, positive_boundary_x_m=5.6)


def enabled_config():
    config = deepcopy(WORKER)
    config["terrain_route_runout"] = dict(SETTINGS)
    return config


def native_ground_height():
    # Execute the actual pure query body without importing/initializing Isaac.
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/env.py").read_text())
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "ground_height")
    module = ast.fix_missing_locations(ast.Module(body=[deepcopy(fn)], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, "native_ground_height_cpu", "exec"), namespace)
    return namespace["ground_height"]


def surface_prim(stage, env_path, index, surface, *, width=4., mesh=False, corridor=False):
    path = env_path + f"/Terrain/surface_{index}"
    size, position, quaternion = surface.box(width=width)
    if corridor:
        size, position, quaternion = (1., 1., 1.), (0., 0., 0.), (1., 0., 0., 0.)
    parent = UsdGeom.Xform.Define(stage, path)
    parent.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*position))
    parent.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(quaternion[0], Gf.Vec3d(*quaternion[1:])))
    geometry_path = path + "/geometry/mesh"
    if corridor:
        prim = UsdGeom.Mesh.Define(stage, geometry_path)
        points, indices = corridor_mesh(160., width)
        prim.CreatePointsAttr(points)
        prim.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
        prim.CreateFaceVertexIndicesAttr(indices)
    elif mesh:
        prim = UsdGeom.Mesh.Define(stage, geometry_path)
        prim.CreatePointsAttr([(x, y, z) for z in (-.5, .5)
                              for x in (-.5, .5) for y in (-.5, .5)])
        prim.CreateFaceVertexCountsAttr([4] * 6)
        prim.CreateFaceVertexIndicesAttr([0, 1, 3, 2, 4, 6, 7, 5,
            0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
    else:
        prim = UsdGeom.Cube.Define(stage, geometry_path)
        prim.CreateSizeAttr(1.)
    UsdGeom.Xformable(prim).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*size))
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim()).CreateCollisionEnabledAttr(True)
    material = UsdShade.Material.Define(stage, path + "/Material")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(surface.friction)
    api.CreateDynamicFrictionAttr(surface.friction)
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material, materialPurpose="physics")
    return geometry_path, material.GetPath()


def fake_env(dtype=torch.float64):
    """Four CPU lanes with real in-memory USD, deliberately permuted clones.

    Only the readback plumbing and actual pure ground_height are exercised.
    PhysX sensor data, rigid dynamics, contacts and PPO are not run.
    """
    stage = Usd.Stage.CreateInMemory()
    groups = ["cross_slope", "stand_305mm", "slope_up", "body_climb_06"]
    kinds = ["cross_slope", "flat", "slope_up", "step_up"]
    clones = [2, 0, 3, 1]
    mask = torch.tensor([True, False, True, False])
    surfaces = [
        append_route_runout([Surface(-4., 0., cross_slope=.2, friction=.3),
            Surface(0., 4., cross_slope=.2, friction=.3)], SETTINGS),
        [Surface(-160., 160., friction=.5)],
        append_route_runout([Surface(-4., 0., friction=.4),
            Surface(0., 2., slope=.25, friction=.4),
            Surface(2., 4., z0=.5, friction=.4)], SETTINGS),
        [Surface(-4., 4., z0=.1, friction=.6)],
    ]
    paths = [f"/World/envs/env_{index}" for index in clones]
    origins = [(1660. + (index % 2) * 332., 1539. + (index // 2) * 9., 0.)
               for index in clones]
    for lane, path in enumerate(paths):
        UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set(origins[lane])
        UsdGeom.Xform.Define(stage, path + "/Terrain")
        for index, surface in enumerate(surfaces[lane]):
            surface_prim(stage, path, index, surface,
                width=8. if kinds[lane] == "flat" else 4., mesh=(lane == 2),
                corridor=kinds[lane] == "flat")
    maximum = max(map(len, surfaces))
    data = torch.zeros(4, maximum, 5, dtype=dtype)
    valid = torch.zeros(4, maximum, dtype=torch.bool)
    mu = torch.zeros(4, maximum, dtype=dtype)
    for lane, lane_surfaces in enumerate(surfaces):
        data[lane, :len(lane_surfaces)] = torch.tensor([
            [s.x0, s.x1, s.z0, s.slope, s.cross_slope] for s in lane_surfaces], dtype=dtype)
        valid[lane, :len(lane_surfaces)] = True
        mu[lane, :len(lane_surfaces)] = torch.tensor([s.friction for s in lane_surfaces], dtype=dtype)
    bodies = ["left_wheel", "right_wheel"]
    views = []
    for count in sorted(set(map(len, surfaces))):
        ids = [i for i, s in enumerate(surfaces) if len(s) == count]
        sensor_paths = [paths[lane] + "/Robot/" + body for lane in ids for body in bodies]
        filter_paths = [[paths[lane] + f"/Terrain/surface_{j}/geometry/mesh"
                         for j in range(count)] for lane in ids for body in bodies]
        view = SimpleNamespace(filter_count=count, sensor_count=len(sensor_paths),
            sensor_paths=sensor_paths, filter_paths=filter_paths)
        views.append((torch.tensor(ids), view))
    config = enabled_config()
    config["scene_groups"] = [{"name": name, "fraction": .25, "terrain": [kind]}
                               for name, kind in zip(groups, kinds)]
    # The synthetic body-climb label is explicitly ineligible; it is not an
    # independent released scene catalogue or a physical body-climb test.
    config["skill_specs"][groups[3]] = {"kind": "body_climb", "mode": 1}
    env = SimpleNamespace(route_runout_settings=dict(SETTINGS), route_runout_mask=mask,
        clone_indices=clones, env_paths=paths, scene_groups=groups, kinds=kinds,
        surfaces=surfaces, surface_data=data, surface_valid=valid, surface_mu=mu,
        sim=SimpleNamespace(stage=stage), device="cpu", cfg=config,
        contact_domain=SimpleNamespace(enabled=True), contact_views=views,
        origins=torch.tensor(origins, dtype=dtype),
        robot=SimpleNamespace(body_names=bodies), body_count=len(bodies), num_envs=4)
    env.ground_height = MethodType(native_ground_height(), env)
    return env


def observation():
    return {**dict.fromkeys(("clone_order_verified", "surface_tensor_verified", "query_verified",
            "collision_geometry_verified", "collision_material_verified",
            "contact_view_population_verified", "boundary_scope_verified"), True),
        "selected_environment_count": 2, "collision_count": 9, "contact_view_count": 3,
        "query_sample_count": 50, "max_surface_tensor_error_m": 0.,
        "max_collision_top_error_m": 0., "max_query_error_m": 0.,
        "max_collision_friction_error": 0.,
        "boundary_evidence_scope": "shared_boundary_function_probes_and_frozen_env_callsite_no_episode_outcomes"}


def basic_runtime():
    return dict(physics_dt=.005, policy_dt=.02, decimation=4, pc_control_dt=.005,
        seed=617, modules={}, leg_kp=160., leg_kd=2.5, wheel_velocity_p=.6,
        precision_tracking=False)


def test_byte_bound_real_v18_expected_selects_54_and_1724():
    assert hashlib.sha256(WORKER_PATH.read_bytes()).hexdigest() == recipe.SOURCE_WORKER_SHA
    config = enabled_config()
    before = deepcopy(config)
    result = runtime.runtime_expectation(config)
    rows = choose_scene_groups(WORKER["scene_groups"], 4096)
    names = {row["name"] for row in WORKER["scene_groups"]
             if WORKER["skill_specs"][row["name"]].get("mode") in (2, 3)}
    expected = Counter(name for name, kind in rows if name in names)
    assert result["selected_group_count"] == len(names) == 54
    assert result["selected_environment_count"] == sum(expected.values()) == 1724
    assert result["selected_group_counts"] == dict(expected)
    assert result["selected_groups"] == sorted(names)
    assert result["corridor_spacing_m"] == 332.
    assert result["corridor_y_spacing_m"] == 9.
    assert config == before


@pytest.mark.parametrize("mutation", ["absent", "disabled", "fixed", "recovery"])
def test_expectation_noops_and_observer_off_require_no_native_objects(mutation):
    config = enabled_config()
    if mutation == "absent":
        config.pop("terrain_route_runout")
    elif mutation == "disabled":
        config["terrain_route_runout"]["enabled"] = False
    elif mutation == "fixed":
        config["evaluation_exact_cases"] = True
    else:
        config["task_campaign"]["domain"] = "recovery"
    assert runtime.runtime_expectation(config) == {"enabled": False}
    assert runtime.observe_runtime(SimpleNamespace(route_runout_settings=None)) == {"enabled": False}


def test_enabled_expectation_cannot_have_zero_route_population():
    config = enabled_config()
    config["scene_groups"] = [{"name": "stand_305mm", "fraction": 1., "terrain": ["flat"]}]
    with pytest.raises(ValueError, match="selected native route"):
        runtime.runtime_expectation(config, num_envs=1)


def test_installed_official_physx_tensor_api_exposes_readback_properties():
    directory = Path(sysconfig.get_path("purelib")) / "isaacsim/extscache"
    files = sorted(directory.glob("omni.physics.tensors-*/omni/physics/tensors/api.py"))
    assert files, "The bound Isaac CPU environment must contain its official tensor API"
    tree = ast.parse(files[-1].read_text())
    contact = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "RigidContactView")
    properties = {node.name: node for node in contact.body
                  if isinstance(node, ast.FunctionDef) and any(
                      isinstance(item, ast.Name) and item.id == "property"
                      for item in node.decorator_list)}
    for name in ("sensor_count", "filter_count", "sensor_paths", "filter_paths"):
        assert name in properties
        returned = next(node.value for node in properties[name].body
                        if isinstance(node, ast.Return))
        assert isinstance(returned, ast.Attribute) and returned.attr == name
        assert isinstance(returned.value, ast.Attribute) and returned.value.attr == "_backend"
    # This establishes installed API semantics, not actual PhysX view creation.


@pytest.mark.parametrize("mesh", [False, True])
@pytest.mark.parametrize("surface", [Surface(4., 6.2, z0=.2),
    Surface(4., 6.2, z0=.2, slope=.2), Surface(4., 6.2, z0=.2, slope=-.2),
    Surface(4., 6.2, z0=.2, cross_slope=.2)])
def test_real_pxr_relative_transform_and_cube_mesh_physics_material(mesh, surface):
    stage = Usd.Stage.CreateInMemory()
    env_path = "/World/envs/env_12"
    env = UsdGeom.Xform.Define(stage, env_path)
    env.AddTranslateOp().Set((20000., -2000., 100.))
    path, material = surface_prim(stage, env_path, 0, surface, mesh=mesh)
    matrix, reset = UsdGeom.XformCache().ComputeRelativeTransform(
        stage.GetPrimAtPath(path), env.GetPrim())
    assert reset is False
    points, static, dynamic = runtime._mesh_top_vertices(stage, path, env_path)
    assert len(points) == 4
    assert min(p[0] for p in points) == pytest.approx(surface.x0, abs=1e-6)
    assert max(p[0] for p in points) == pytest.approx(surface.x1, abs=1e-6)
    assert min(p[1] for p in points) == pytest.approx(-2., abs=1e-6)
    assert max(p[1] for p in points) == pytest.approx(2., abs=1e-6)
    for x, y, z in points:
        assert z == pytest.approx(surface.height(x, y), abs=1e-6)
    assert static == dynamic == pytest.approx(surface.friction)


@pytest.mark.parametrize("fault", ["absent", "disabled", "material", "reset", "empty_mesh"])
def test_real_pxr_missing_collision_material_or_reset_is_rejected(fault):
    stage = Usd.Stage.CreateInMemory()
    env_path = "/World/envs/env_12"
    UsdGeom.Xform.Define(stage, env_path)
    path, material = surface_prim(stage, env_path, 0, Surface(4., 6.2), mesh=True)
    prim = stage.GetPrimAtPath(path)
    if fault == "absent":
        stage.RemovePrim(path)
    elif fault == "disabled":
        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    elif fault == "material":
        stage.RemovePrim(material)
    elif fault == "reset":
        UsdGeom.Xformable(prim).SetResetXformStack(True)
    else:
        UsdGeom.Mesh(prim).GetPointsAttr().Set([])
    with pytest.raises(ValueError):
        runtime._mesh_top_vertices(stage, path, env_path)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_fake_env_reads_real_usd_native_query_and_nontrivial_clone_order(dtype):
    env = fake_env(dtype)
    state = runtime.observe_runtime(env)
    observed = state.pop("runtime_observed")
    assert state == runtime.runtime_expectation(env.cfg, num_envs=4)
    runtime.validate_observation(observed)
    assert observed["collision_count"] == sum(map(len, env.surfaces)) == 9
    assert observed["contact_view_count"] == 3
    assert observed["query_sample_count"] >= 2 * 15 + (3 + 4) * 9
    assert observed["boundary_evidence_scope"] == runtime.BOUNDARY_EVIDENCE_SCOPE
    assert observed["contact_view_binding_scope"] == (
        "actual_sensor_order_and_filter_path_membership_not_filter_slot_order")
    assert observed["surface_count_distribution"] == {1: 2, 3: 1, 4: 1}
    assert observed["max_query_error_m"] < 1e-6


@pytest.mark.parametrize("fault", ["clone_duplicate", "clone_path", "valid_count", "valid_gap",
    "surface_data", "surface_data_nan", "friction", "friction_nan", "query", "query_nan", "query_old_segment",
    "material", "material_nan", "collision_height", "collision_short", "collision_narrow",
    "contact_count", "contact_duplicate_lane", "contact_sensor_path", "contact_foreign_filter",
    "origin_nan"])
def test_fake_env_rejects_broken_independent_readback(fault):
    env = fake_env()
    if fault == "clone_duplicate":
        env.clone_indices[0] = env.clone_indices[1]
    elif fault == "clone_path":
        env.env_paths[0] = env.env_paths[1]
    elif fault == "valid_count":
        env.surface_valid[0, 0] = False
    elif fault == "valid_gap":
        env.surface_valid[0, 0] = False
        env.surface_valid[0, 3] = True
    elif fault == "surface_data":
        env.surface_data[0, 0, 2] += .01
    elif fault == "surface_data_nan":
        env.surface_data[0, 0, 2] = torch.nan
    elif fault == "friction":
        env.surface_mu[0, 0] += .01
    elif fault == "friction_nan":
        env.surface_mu[0, 0] = torch.nan
    elif fault in ("query", "query_nan"):
        original = env.ground_height
        env.ground_height = lambda xy: original(xy) + (.01 if fault == "query" else torch.nan)
    elif fault == "query_old_segment":
        original = env.ground_height
        env.ground_height = lambda xy: original(xy) + torch.where(
            xy[:, 0] < 0., xy.new_tensor(.01), xy.new_tensor(0.))
    elif fault in ("material", "material_nan"):
        prim = env.sim.stage.GetPrimAtPath(env.env_paths[0] + "/Terrain/surface_2/Material")
        UsdPhysics.MaterialAPI(prim).GetStaticFrictionAttr().Set(.9 if fault == "material" else float("nan"))
    elif fault in ("collision_height", "collision_short", "collision_narrow"):
        path = env.env_paths[0] + "/Terrain/surface_2"
        parent = UsdGeom.Xformable(env.sim.stage.GetPrimAtPath(path))
        prim = env.sim.stage.GetPrimAtPath(path + "/geometry/mesh")
        translate = parent.GetOrderedXformOps()[0]
        scale = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
        old_position, old_scale = translate.Get(), scale.Get()
        if fault == "collision_height":
            translate.Set(Gf.Vec3d(old_position[0], old_position[1], 1.))
        elif fault == "collision_short":
            translate.Set(Gf.Vec3d(4.05, old_position[1], old_position[2]))
            scale.Set(Gf.Vec3d(.1, old_scale[1], old_scale[2]))
        else:
            scale.Set(Gf.Vec3d(old_scale[0], .1, old_scale[2]))
    elif fault == "contact_count":
        env.contact_views[-1][1].filter_count -= 1
    elif fault == "contact_duplicate_lane":
        env.contact_views.append(env.contact_views[-1])
    elif fault == "contact_sensor_path":
        env.contact_views[0][1].sensor_paths[0] = "/World/envs/env_999/Robot/left_wheel"
    elif fault == "contact_foreign_filter":
        env.contact_views[0][1].filter_paths[0][0] = "/World/envs/env_999/Terrain/surface_0/geometry/mesh"
    else:
        env.origins[0, 0] = torch.nan
    with pytest.raises(ValueError):
        runtime.observe_runtime(env)


def test_fake_contact_backend_flat_path_list_is_membership_not_slot_order():
    env = fake_env()
    for ids, view in env.contact_views:
        view.filter_paths = sorted({path for row in view.filter_paths for path in row}, reverse=True)
    observed = runtime.observe_runtime(env)["runtime_observed"]
    assert observed["contact_view_population_verified"] is True
    assert observed["contact_view_binding_scope"].endswith("not_filter_slot_order")


@pytest.mark.parametrize("scope", [None, "native_episode_outcome_verified"])
def test_observation_cannot_overclaim_boundary_evidence(scope):
    actual = observation()
    actual["boundary_evidence_scope"] = scope
    with pytest.raises(ValueError, match="static probe scope"):
        runtime.validate_observation(actual)


@pytest.mark.parametrize("name", ["clone_order_verified", "surface_tensor_verified", "query_verified",
    "collision_geometry_verified", "collision_material_verified", "contact_view_population_verified",
    "boundary_scope_verified"])
@pytest.mark.parametrize("bad", [None, False, 1])
def test_observation_flags_require_actual_explicit_true(name, bad):
    actual = observation()
    actual[name] = bad
    with pytest.raises(ValueError, match="independent native geometry"):
        runtime.validate_observation(actual)


@pytest.mark.parametrize("name", ["selected_environment_count", "collision_count",
    "contact_view_count", "query_sample_count"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.])
def test_observation_counts_require_positive_integers(name, bad):
    actual = observation()
    actual[name] = bad
    with pytest.raises(ValueError, match="actual positive count"):
        runtime.validate_observation(actual)


@pytest.mark.parametrize("name", ["max_surface_tensor_error_m", "max_collision_top_error_m",
    "max_query_error_m", "max_collision_friction_error"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1, .000101, True])
def test_observation_errors_require_finite_nonnegative_within_tolerance(name, bad):
    actual = observation()
    actual[name] = bad
    with pytest.raises(ValueError, match="tolerance"):
        runtime.validate_observation(actual)


def test_runtime_preflight_accepts_fake_observation_but_labels_no_functional_probe():
    env = fake_env()
    expected = basic_runtime()
    expected["modules"]["terrain_route_runout"] = True
    expected["terrain_route_runout"] = runtime.runtime_expectation(env.cfg, 4)
    actual = deepcopy(expected)
    actual["terrain_route_runout"] = runtime.observe_runtime(env)
    result = validate_runtime_state(expected, actual)
    assert result["status"] == "runtime_configuration_passed"
    assert result["functional_transport_precision_randomization_probes"] == "not_run"


@pytest.mark.parametrize("fault", ["missing_expectation", "missing_module", "missing_readback",
    "failed_readback", "population_mismatch", "disabled_claim"])
def test_runtime_preflight_rejects_undeclared_or_missing_native_readback(fault):
    expected = basic_runtime()
    expected["modules"]["terrain_route_runout"] = True
    expected["terrain_route_runout"] = runtime.runtime_expectation(enabled_config())
    actual = deepcopy(expected)
    actual["terrain_route_runout"]["runtime_observed"] = observation()
    actual["terrain_route_runout"]["runtime_observed"]["selected_environment_count"] = 1724
    if fault == "missing_expectation":
        expected.pop("terrain_route_runout")
    elif fault == "missing_module":
        expected["modules"].pop("terrain_route_runout")
    elif fault == "missing_readback":
        actual["terrain_route_runout"].pop("runtime_observed")
    elif fault == "failed_readback":
        actual["terrain_route_runout"]["runtime_observed"]["query_verified"] = False
    elif fault == "population_mismatch":
        actual["terrain_route_runout"]["runtime_observed"]["selected_environment_count"] = 1723
    else:
        expected["modules"]["terrain_route_runout"] = False
        expected["terrain_route_runout"] = {"enabled": False}
        actual = deepcopy(expected)
        actual["terrain_route_runout"]["runtime_observed"] = observation()
    with pytest.raises(ValueError, match="Runtime preflight rejected"):
        validate_runtime_state(expected, actual)


def test_legacy_runtime_without_new_field_and_disabled_projection_keep_old_behavior():
    legacy = basic_runtime()
    assert validate_runtime_state(legacy, deepcopy(legacy))["status"] == "runtime_configuration_passed"
    legacy["modules"]["terrain_route_runout"] = False
    legacy["terrain_route_runout"] = {"enabled": False}
    assert validate_runtime_state(legacy, deepcopy(legacy))["status"] == "runtime_configuration_passed"


@pytest.mark.parametrize("fault", ["none", "missing_contract", "missing_runtime", "missing_module",
    "wrong_population", "enabled_fixed"])
def test_static_preflight_requires_independent_complete_role_expectations(fault):
    config = deepcopy(WORKER)
    recipe.apply_revision(config)
    design = recipe.revised_design(ROOT)
    control = load_chassis_control(ROOT / config["control_math_source"])
    role = "full" if fault == "enabled_fixed" else "train"
    row = design["stages"]["terrain"]["roles"][role]
    if fault == "enabled_fixed":
        from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
        names = [case["name"] for case in row["contract"]["evaluation"]["cases"]]
        config["evaluation"] = select_evaluation_cases(config["evaluation"], names)
        config["evaluation"]["seed"] = row["invocation"]["seed"]["value"]
        config = fixed_suite_contract(config)
        config["terrain_route_runout"]["enabled"] = True
    if fault == "missing_contract":
        row["contract"].pop("terrain_route_runout")
    elif fault == "missing_runtime":
        row["runtime"].pop("terrain_route_runout")
    elif fault == "missing_module":
        row["runtime"]["modules"].pop("terrain_route_runout")
    elif fault == "wrong_population":
        row["runtime"]["terrain_route_runout"]["selected_environment_count"] = 1723
    seed_spec = row["invocation"]["seed"]
    seed = seed_spec["value"] if seed_spec["rule"] == "fixed" else seed_spec["root_seed"]
    args = dict(stage="terrain", role=role, seed=seed,
                num_envs=row["invocation"]["num_envs"], block_index=0)
    if fault == "none":
        assert validate_design_contract(config, design, control, **args)["status"] == "static_passed"
    else:
        with pytest.raises(ValueError, match="terrain_route_runout"):
            validate_design_contract(config, design, control, **args)
