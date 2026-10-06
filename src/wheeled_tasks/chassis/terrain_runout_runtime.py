"""Startup geometry readback for the optional terrain training runout.

This code runs once at construction. It never changes rewards, success, contact
forces, actor inputs or curriculum clocks, and does not run in physics substeps.
"""
from collections import Counter
import math

from .task import choose_scene_groups
from .terrain_runout import apply_route_boundary, route_runout_selection, validate_runout


BOUNDARY_EVIDENCE_SCOPE = (
    "shared_boundary_function_probes_and_frozen_env_callsite_no_episode_outcomes")


def runtime_expectation(config, num_envs=4096):
    """Independent static population declaration, before native construction."""
    settings = validate_runout(config)
    if settings is None:
        return {"enabled": False}
    rows = choose_scene_groups(config["scene_groups"], num_envs)
    counts = Counter(group for group, kind in rows
                     if route_runout_selection(config, group, kind))
    if not counts:
        raise ValueError("Enabled terrain runout requires selected native route rows")
    return dict(enabled=True, settings=settings,
        selected_group_count=len(counts), selected_environment_count=sum(counts.values()),
        selected_groups=sorted(counts), selected_group_counts=dict(sorted(counts.items())),
        negative_boundary_x_m=-3.6, lateral_boundary_y_m=1.7,
        corridor_spacing_m=2 * config.get("flat_half_length_m", 44.) + 12.,
        corridor_y_spacing_m=config.get("flat_floor_width_m", 4.) + 1.)


def validate_observation(observed):
    """Reject missing or failed readbacks; this is an implementation check."""
    required = ("clone_order_verified", "surface_tensor_verified", "query_verified",
                "collision_geometry_verified", "collision_material_verified",
                "contact_view_population_verified", "boundary_scope_verified")
    if not isinstance(observed, dict) or any(observed.get(key) is not True for key in required):
        raise ValueError("Terrain runout requires independent native geometry readback")
    if observed.get("boundary_evidence_scope") != BOUNDARY_EVIDENCE_SCOPE:
        raise ValueError("Terrain runout boundary evidence must declare its static probe scope")
    for key in ("selected_environment_count", "collision_count", "contact_view_count",
                "query_sample_count"):
        if type(observed.get(key)) is not int or observed[key] <= 0:
            raise ValueError(f"Terrain runout {key} must be an actual positive count")
    for key in ("max_surface_tensor_error_m", "max_collision_top_error_m", "max_query_error_m",
                "max_collision_friction_error"):
        value = observed.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e-4:
            raise ValueError(f"Terrain runout {key} exceeded native readback tolerance")


def _mesh_top_vertices(stage, path, env_path):
    """Read collision geometry in its env frame, avoiding large-world subtraction."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    prim = stage.GetPrimAtPath(path)
    if not prim or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
        raise ValueError("Terrain runout collision mesh is absent or disabled")
    if prim.IsA(UsdGeom.Cube):
        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
        vertices = [(x, y, half) for x in (-half, half) for y in (-half, half)]
    elif prim.IsA(UsdGeom.Mesh):
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            raise ValueError("Terrain runout collision mesh has no vertices")
        top = max(float(p[2]) for p in points)
        vertices = [tuple(p) for p in points if abs(float(p[2]) - top) < 1e-7]
        if len(vertices) < 4:
            raise ValueError("Terrain runout collision mesh has no box top face")
    else:
        raise ValueError("Terrain runout requires the existing native cuboid collision geometry")
    transform, reset = UsdGeom.XformCache(Usd.TimeCode.Default()).ComputeRelativeTransform(
        prim, stage.GetPrimAtPath(env_path))
    if reset:
        raise ValueError("Terrain runout collider resets the environment transform")
    top_points = [tuple(transform.Transform(Gf.Vec3d(*point))) for point in vertices]
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if not material:
        raise ValueError("Terrain runout collider has no bound physics material")
    api = UsdPhysics.MaterialAPI(material.GetPrim())
    static, dynamic = api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get()
    if static is None or dynamic is None:
        raise ValueError("Terrain runout material lacks friction readback")
    return top_points, float(static), float(dynamic)


def observe_runtime(env):
    """Compare actual sorted lanes, tensors, USD collisions and contact views.

    Nothing is copied from a frozen design expectation. Native construction has
    already selected and sorted the lanes and applied contact-domain friction.
    """
    settings = env.route_runout_settings
    if settings is None:
        return {"enabled": False}
    import torch
    mask = env.route_runout_mask.detach().cpu().tolist()
    clone = list(env.clone_indices)
    count = len(mask)
    if sorted(clone) != list(range(count)) or len(env.env_paths) != count:
        raise ValueError("Terrain runout clone order is ambiguous")
    for path, index in zip(env.env_paths, clone, strict=True):
        if int(path.split("/env_")[1].split("/")[0]) != index:
            raise ValueError("Terrain runout sorted path and clone index differ")
    selected = Counter(group for group, active in zip(env.scene_groups, mask, strict=True) if active)
    if not selected:
        raise ValueError("Terrain runout actual selected population is empty")
    data = env.surface_data.detach().cpu()
    valid = env.surface_valid.detach().cpu()
    friction = env.surface_mu.detach().cpu()
    if (data.shape[:2] != valid.shape or friction.shape != valid.shape
            or data.shape != (count, valid.shape[1], 5)
            or not bool(torch.isfinite(data).all() & torch.isfinite(friction).all())):
        raise ValueError("Terrain runout surface tensor is malformed or nonfinite")
    tensor_error, top_error, material_error = 0., 0., 0.
    collider_count = 0
    query_points, query_expected = [], []
    surface_distribution = Counter()
    for lane, surfaces in enumerate(env.surfaces):
        surface_distribution[len(surfaces)] += 1
        expected_valid = torch.arange(valid.shape[1]) < len(surfaces)
        if not torch.equal(valid[lane], expected_valid):
            raise ValueError("Terrain runout surface tensor validity differs from actual lane")
        if mask[lane]:
            last, extension = surfaces[-2:]
            if (abs(last.x1 - 4.) > 1e-9 or abs(extension.x0 - 4.) > 1e-9
                    or abs(extension.x1 - settings["support_end_x_m"]) > 1e-9):
                raise ValueError("Terrain runout actual support endpoints differ")
            if (last.slope, last.cross_slope, last.friction) != (
                    extension.slope, extension.cross_slope, extension.friction):
                raise ValueError("Terrain runout extension differs from terminal plane/material")
        for index, surface in enumerate(surfaces):
            if not all(math.isfinite(value) for value in (surface.x0, surface.x1,
                    surface.z0, surface.slope, surface.cross_slope, surface.friction)):
                raise ValueError("Terrain runout actual Surface is nonfinite")
            expected = data.new_tensor((surface.x0, surface.x1, surface.z0,
                                        surface.slope, surface.cross_slope))
            tensor_error = max(tensor_error, float((data[lane, index] - expected).abs().max()),
                               abs(float(friction[lane, index]) - surface.friction))
            path = env.env_paths[lane] + f"/Terrain/surface_{index}/geometry/mesh"
            points, static, dynamic = _mesh_top_vertices(env.sim.stage, path, env.env_paths[lane])
            if (not points or any(len(point) != 3 or not all(math.isfinite(v) for v in point)
                                  for point in points)
                    or not math.isfinite(static) or not math.isfinite(dynamic)):
                raise ValueError("Terrain runout collision geometry or material is nonfinite")
            collider_count += 1
            width = env.cfg["flat_floor_width_m"] if env.kinds[lane] in ("flat", "jump") else 4.
            for observed, expected in ((min(p[0] for p in points), surface.x0),
                    (max(p[0] for p in points), surface.x1),
                    (min(p[1] for p in points), -width / 2),
                    (max(p[1] for p in points), width / 2)):
                if abs(observed - expected) > 1e-4:
                    raise ValueError("Terrain runout collision top does not cover its support interval")
            for x, y, z in points:
                if not surface.x0 - 1e-4 <= x <= surface.x1 + 1e-4 or abs(y) > width / 2 + 1e-4:
                    raise ValueError("Terrain runout collision top lies outside its support interval")
                top_error = max(top_error, abs(z - surface.height(x, y)))
            mu = surface.friction if env.contact_domain is not None else 2 * surface.friction - .5
            material_error = max(material_error, abs(static - mu), abs(dynamic - mu))
        # Every selected lane is independently sampled around the real seam,
        # boundary and terminal edge, including its index-dependent rough patch.
        queries = [(x, y) for x in (3.99, 4., 4.01,
            settings["positive_boundary_x_m"], settings["support_end_x_m"])
            for y in (-1.7, 0., 1.7)] if mask[lane] else []
        if mask[lane]:
            for surface in surfaces:
                inset = min(.001, (surface.x1 - surface.x0) / 4)
                queries.extend((x, y) for x in (surface.x0 + inset,
                    (surface.x0 + surface.x1) / 2, surface.x1 - inset)
                    for y in (-1.7, 0., 1.7))
        query_points.append(queries)
        query_expected.append([max((s.height(x, y) for s in surfaces if s.x0 <= x <= s.x1),
                                  default=0.) for x, y in query_points[-1]])
    query_count = sum(map(len, query_points))
    max_queries = max(map(len, query_points))
    query_valid = torch.tensor([[True] * len(points) + [False] * (max_queries - len(points))
                               for points in query_points], device=env.device)
    query_points = [points + [(0., 0.)] * (max_queries - len(points)) for points in query_points]
    query_expected = [values + [0.] * (max_queries - len(values)) for values in query_expected]
    points = torch.tensor(query_points, device=env.device, dtype=env.surface_data.dtype)
    actual_height = torch.stack([env.ground_height(points[:, index, :])
                                 for index in range(points.shape[1])], -1)
    target_height = torch.tensor(query_expected, device=env.device, dtype=actual_height.dtype)
    if not bool(torch.isfinite(actual_height).all() & torch.isfinite(target_height).all()):
        raise ValueError("Terrain runout ground query is nonfinite")
    query_error = float((actual_height - target_height).abs()[query_valid].max())
    view_lanes = []
    for ids, view in env.contact_views:
        lanes = ids.detach().cpu().tolist()
        view_lanes.extend(lanes)
        if any(view.filter_count != len(env.surfaces[lane]) for lane in lanes):
            raise ValueError("Terrain runout native contact view misses a collision filter")
        expected_sensors = [env.env_paths[lane] + "/Robot/" + body
                            for lane in lanes for body in env.robot.body_names]
        if list(view.sensor_paths) != expected_sensors or view.sensor_count != len(expected_sensors):
            raise ValueError("Terrain runout native contact sensor paths or order differ")
        filters = set()
        def collect(value):
            if isinstance(value, str):
                filters.add(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            else:
                raise ValueError("Terrain runout contact filter path has an unknown shape")
        collect(view.filter_paths)
        expected_filters = {env.env_paths[lane] + f"/Terrain/surface_{index}/geometry/mesh"
                            for lane in lanes for index in range(len(env.surfaces[lane]))}
        if filters != expected_filters:
            raise ValueError("Terrain runout native contact filter path membership differs")
    if sorted(view_lanes) != list(range(count)):
        raise ValueError("Terrain runout native contact views omit or repeat a lane")
    base_limit = env.surface_data.new_full((count,), 3.6)
    positive = apply_route_boundary(base_limit.new_ones(count), base_limit,
                                   env.route_runout_mask, settings)
    negative = apply_route_boundary(-base_limit.new_ones(count), base_limit,
                                   env.route_runout_mask, settings)
    if (not bool((positive[env.route_runout_mask] == settings["positive_boundary_x_m"]).all())
            or not bool((positive[~env.route_runout_mask] == base_limit[~env.route_runout_mask]).all())
            or not bool((negative == base_limit).all())):
        raise ValueError("Terrain runout actual boundary path changed another direction or lane")
    origins = env.origins.detach().cpu()
    if origins.shape != (count, 3) or not bool(torch.isfinite(origins).all()):
        raise ValueError("Terrain runout scene origins are malformed or nonfinite")
    spacings = []
    for axis in (0, 1):
        values = sorted(set(float(value) for value in origins[:, axis]))
        if len(values) < 2:
            raise ValueError("Terrain runout population cannot observe separated scene spacing")
        spacings.append(min(b - a for a, b in zip(values, values[1:])))
    state = dict(enabled=True, settings=dict(settings), selected_group_count=len(selected),
        selected_environment_count=sum(selected.values()), selected_groups=sorted(selected),
        selected_group_counts=dict(sorted(selected.items())), negative_boundary_x_m=-3.6,
        lateral_boundary_y_m=1.7, corridor_spacing_m=spacings[0], corridor_y_spacing_m=spacings[1])
    observed = dict(clone_order_verified=True, surface_tensor_verified=True, query_verified=True,
        collision_geometry_verified=True, collision_material_verified=True,
        contact_view_population_verified=True, boundary_scope_verified=True,
        boundary_evidence_scope=BOUNDARY_EVIDENCE_SCOPE,
        selected_environment_count=sum(selected.values()), collision_count=collider_count,
        contact_view_count=len(env.contact_views), query_sample_count=query_count,
        contact_view_binding_scope="actual_sensor_order_and_filter_path_membership_not_filter_slot_order",
        surface_count_distribution=dict(sorted(surface_distribution.items())),
        max_surface_tensor_error_m=tensor_error, max_collision_top_error_m=top_error,
        max_query_error_m=query_error, max_collision_friction_error=material_error)
    validate_observation(observed)
    state["runtime_observed"] = observed
    return state
