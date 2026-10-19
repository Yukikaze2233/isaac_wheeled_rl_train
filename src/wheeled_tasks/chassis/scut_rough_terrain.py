"""0730 reference rough geometry, using the SDK generator and its actual mesh.

This adapter supplies terrain geometry only. It does not enable skill state
machines, terrain progression, or change the flat reward/command contract.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


REFERENCE = Path(__file__).resolve().parents[3] / "references/scut_curriculum_b8ff79f3"
ENV_FILE = "pretrained/26_infantry/rough_rotation_stair/2026-07-30_09-16-27/params/env.yaml"
ENV_SHA = "a6117e55b3ebb5a03870981bd2d65d889c38850063008a9341d8346b997bb1ee"
GRID_FILE = "source/agent_world/agent_world/terrains/height_field.py"
GRID_SHA = "5f444236205f6ab4a56e0c76387b388d5d2d5a7a83ab63386f5cf15ae5b68d0e"
SOURCE_COMMIT = "b8ff79f3df855faf9dc92f4a282bd80c42649466"
BOUNDARY_MARGIN_M = .5
REHEARSAL_BOUNDS_M = ((-200., 100., 0.), (200., 500., 0.))
REHEARSAL_PRIM_PATH = "/World/flat_rehearsal"
REHEARSAL_COLLIDER_KEY = "formal_flat_rehearsal_collider"
NATIVE_REHEARSAL_PLANE = "native-plane-v1"


def _rehearsal_origins(values, num_envs):
    """Explicit pre-clone assignments only; no population or command sampling."""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("Rehearsal origins require an env_id to world XYZ mapping")
    result = {}
    low, high = REHEARSAL_BOUNDS_M
    for env_id, origin in values.items():
        if type(env_id) is not int or not 0 <= env_id < num_envs:
            raise ValueError("Rehearsal env_id is outside the pre-clone environment population")
        xyz = np.asarray(origin, dtype=float)
        if (xyz.shape != (3,) or not np.isfinite(xyz).all() or xyz[2] != 0.
                or not all(low[i] + BOUNDARY_MARGIN_M < xyz[i] < high[i] - BOUNDARY_MARGIN_M
                           for i in (0, 1))):
            raise ValueError("Rehearsal origins must lie inside the finite plane at world z=0")
        result[env_id] = xyz.tolist()
    return result


def _rehearsal_plane_mesh():
    low, high = REHEARSAL_BOUNDS_M
    points = np.asarray([[low[0], low[1], 0.], [high[0], low[1], 0.],
                         [high[0], high[1], 0.], [low[0], high[1], 0.]], dtype=np.float32)
    return points, np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def _author_rehearsal_plane(stage, points, faces, material_path, *, native_plane=False):
    """Author the selected collider, preserving the shared physics material."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    if stage.GetPrimAtPath(REHEARSAL_PRIM_PATH).IsValid():
        raise ValueError("Rehearsal mesh path already exists")
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material:
        raise ValueError("Original rough physics material is missing")
    if native_plane:
        # Large static triangles have different wheel contact behavior from a
        # native plane. Keep the bounded z=0 query region for birth placement;
        # collision groups restrict the infinite collider to rehearsal lanes.
        low, high = np.asarray(REHEARSAL_BOUNDS_M)
        plane = UsdGeom.Plane.Define(stage, REHEARSAL_PRIM_PATH)
        plane.CreateAxisAttr("Z")
        plane.CreateWidthAttr(float(high[0] - low[0]))
        plane.CreateLengthAttr(float(high[1] - low[1]))
        UsdGeom.Xformable(plane).AddTranslateOp().Set(Gf.Vec3d(*((low + high) / 2.)))
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
        return str(plane.GetPath())
    mesh = UsdGeom.Mesh.Define(stage, REHEARSAL_PRIM_PATH)
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
    return str(mesh.GetPath())


def _read_bound(relative, expected, root=REFERENCE):
    raw = (Path(root) / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("reference rough source SHA differs: " + relative)
    return raw


def source_parameters(root=REFERENCE):
    """Read the saved effective terrain block; Python YAML tags remain data."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    for tag in ("python/tuple", "python/object/apply:builtins.slice"):
        Loader.add_constructor("tag:yaml.org,2002:" + tag,
                               lambda loader, node: loader.construct_sequence(node))
    env = yaml.load(_read_bound(ENV_FILE, ENV_SHA, root), Loader=Loader)
    return deepcopy(env["terrain"])


def _grid_config_class(root=REFERENCE):
    # Use the pinned source implementation, including its count/width sampling.
    # Do not import the source robot task or the unrelated 0719 dash terrain.
    raw = _read_bound(GRID_FILE, GRID_SHA, root).decode()
    spec = importlib.util.spec_from_file_location("_scut0730_grid_terrain", Path(root) / GRID_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass annotation resolution and callable serialization
    try:
        # SDK 3 exposes utils.configclass as a module after importing built-in
        # terrain configs. Adapt only this import; pinned geometry is unchanged.
        raw = raw.replace("from isaaclab.utils import configclass\n",
                          "from isaaclab.utils.configclass import configclass\n", 1)
        exec(compile(raw, str(Path(root) / GRID_FILE), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module.MeshCustomGridBarsTerrainCfg


def generator_config(seed, root=REFERENCE):
    """Build real SDK configuration classes from the authenticated archive."""
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.height_field.hf_terrains_cfg import (
        HfPyramidSlopedTerrainCfg, HfPyramidStairsTerrainCfg, HfRandomUniformTerrainCfg,
    )
    from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

    terrain = source_parameters(root)
    raw = deepcopy(terrain["terrain_generator"])
    raw.pop("class_type")
    # Archive seed=null inherited the process seed. Make that seed explicit in
    # the generated geometry receipt; this is not a different difficulty range.
    raw["seed"] = int(seed)
    classes = {
        "agent_world.terrains.height_field:mesh_grid_bars_terrain": _grid_config_class(root),
        "isaaclab.terrains.height_field.hf_terrains:pyramid_sloped_terrain": HfPyramidSlopedTerrainCfg,
        "isaaclab.terrains.height_field.hf_terrains:pyramid_stairs_terrain": HfPyramidStairsTerrainCfg,
        "isaaclab.terrains.height_field.hf_terrains:random_uniform_terrain": HfRandomUniformTerrainCfg,
        "isaaclab.terrains.trimesh.mesh_terrains:flat_terrain": MeshPlaneTerrainCfg,
    }
    sub_terrains = {}
    for name, saved in raw.pop("sub_terrains").items():
        fields = deepcopy(saved)
        kind = fields.pop("function")
        if kind not in classes:
            raise ValueError("Unsupported archived rough geometry: " + kind)
        sub_terrains[name] = classes[kind](**fields)
    return TerrainGeneratorCfg(sub_terrains=sub_terrains, **raw)


def column_names(parameters):
    generator = parameters["terrain_generator"]
    entries = generator["sub_terrains"]
    weights = np.asarray([entry["proportion"] for entry in entries.values()], dtype=float)
    cumulative = np.cumsum(weights / weights.sum())
    names = list(entries)
    return [names[int(np.flatnonzero(i / generator["num_cols"] + .001 < cumulative)[0])]
            for i in range(generator["num_cols"])]


@dataclass
class SharedRough:
    origins_m: list
    collision_paths: list
    global_prim_paths: list
    material_paths: list
    importer: object
    warp_mesh: object
    raycast: object
    bounds_m: list
    mesh_sha256: str
    seed: int
    rehearsal_origins_by_env: dict | None = None
    rehearsal_bounds_m: list | None = None
    source_rough_mesh_sha256: str | None = None
    rehearsal_mesh_sha256: str | None = None
    max_init_terrain_level: int | None = None
    mixed_rough_assignment: dict | None = None
    rehearsal_collider: str | None = None

    def distribute_mixed_rough_origins(self):
        """Allocate terrain columns within the rough population before cloning.

        The importer initially spreads all lanes over columns. Removing the
        flat roles afterwards biases contiguous rough lanes toward the last
        columns. Keep their sampled rows, but apply the same column rule to
        the rough population itself. No mesh, difficulty or RNG draw changes.
        """
        if not self.rehearsal_origins_by_env or self.mixed_rough_assignment is not None:
            raise ValueError("Mixed rough origins require one pre-clone allocation")
        importer = self.importer
        ids = torch.tensor([i for i in range(len(self.origins_m))
                            if i not in self.rehearsal_origins_by_env],
                           dtype=torch.long, device=importer.device)
        if not ids.numel():
            raise ValueError("Mixed terrain requires a nonempty rough population")
        columns = importer.terrain_origins.shape[1]
        before = torch.bincount(importer.terrain_types[ids], minlength=columns)
        types = torch.div(torch.arange(len(ids), device=importer.device),
                          len(ids) / columns, rounding_mode="floor").long()
        origins = importer.terrain_origins[importer.terrain_levels[ids], types]
        importer.terrain_types[ids] = types
        importer.env_origins[ids] = origins
        for env_id, origin in zip(ids.cpu().tolist(), origins.cpu().tolist()):
            self.origins_m[env_id] = origin
        self.mixed_rough_assignment = {
            "version": "v7-rough-population-columns-v1",
            "rough_lane_count": len(ids),
            "column_counts_before": before.cpu().tolist(),
            "column_counts_after": torch.bincount(types, minlength=columns).cpu().tolist(),
            "row_counts": torch.bincount(importer.terrain_levels[ids],
                                         minlength=importer.terrain_origins.shape[0]).cpu().tolist(),
            "scope": "actual_rough_birth_origins_before_native_clone_ordering",
            "flat_origins_unchanged": True,
        }

    @property
    def query_bounds_m(self):
        if self.rehearsal_bounds_m is None:
            return self.bounds_m
        return [np.minimum(self.bounds_m[0], self.rehearsal_bounds_m[0]).tolist(),
                np.maximum(self.bounds_m[1], self.rehearsal_bounds_m[1]).tolist()]

    def height(self, world_xy):
        """World z on rough triangles and the assigned rehearsal query region."""
        shape = world_xy.shape[:-1]
        starts = torch.empty((*shape, 3), device=world_xy.device, dtype=torch.float32)
        starts[..., :2] = world_xy
        starts[..., 2] = self.query_bounds_m[1][2] + 1.
        directions = torch.zeros_like(starts)
        directions[..., 2] = -1.
        hits = self.raycast(starts, directions, self.warp_mesh)[0][..., 2]
        # Beyond the complete mesh, the existing boundary timeout owns reset.
        # Interior misses stay nonfinite instead of fabricating a floor.
        outside = self._outside(world_xy, 0.)
        return torch.where(outside, torch.zeros_like(hits), hits).to(world_xy.dtype)

    def _outside(self, world_xy, margin):
        low, high = self.bounds_m
        outside = ((world_xy[..., 0] < low[0] + margin)
                | (world_xy[..., 0] > high[0] - margin)
                | (world_xy[..., 1] < low[1] + margin)
                | (world_xy[..., 1] > high[1] - margin))
        if self.rehearsal_bounds_m is not None:
            low, high = self.rehearsal_bounds_m
            outside &= ((world_xy[..., 0] < low[0] + margin)
                        | (world_xy[..., 0] > high[0] - margin)
                        | (world_xy[..., 1] < low[1] + margin)
                        | (world_xy[..., 1] > high[1] - margin))
        return outside

    def boundary(self, world_xyz):
        return self._outside(world_xyz[..., :2], BOUNDARY_MARGIN_M)

    def summary(self):
        result = dict(type="rough", source="scut0730", shared=True,
                    source_commit=SOURCE_COMMIT, source_env_sha256=ENV_SHA,
                    source_grid_sha256=GRID_SHA, geometry_seed=self.seed,
                    collision_paths=self.collision_paths, mesh_sha256=self.mesh_sha256,
                    mesh_bounds_world_m=self.bounds_m,
                    boundary_margin_m=BOUNDARY_MARGIN_M, boundary_scope="whole_generated_terrain",
                    height_query="vertical_ray_on_actual_collision_mesh_world_z",
                    row_count=10, column_count=14, tile_size_m=[9., 9.], border_width_m=5.,
                    column_names=column_names(source_parameters()),
                    max_init_terrain_level=self.max_init_terrain_level,
                    initial_level_scope="source_all_10_rows" if self.max_init_terrain_level is None
                        else "local_initial_row_cap_only_geometry_unchanged",
                    initial_level_counts=torch.bincount(self.importer.terrain_levels, minlength=10).cpu().tolist(),
                    initial_type_counts=torch.bincount(self.importer.terrain_types, minlength=14).cpu().tolist(),
                    training_terrain_progression=False, skill_state_machines_enabled=False,
                    scope="geometry_support_only_not_rough_behavior_qualification")
        if self.rehearsal_bounds_m is not None:
            result.update(boundary_scope="union_of_rough_and_separate_finite_rehearsal_plane",
                          boundary_regions_world_m=[self.bounds_m, self.rehearsal_bounds_m],
                          query_mesh_bounds_world_m=self.query_bounds_m,
                          mesh_sha256_scope="concatenated_actual_native_collision_meshes",
                          rough_mesh_bounds_world_m=self.bounds_m,
                          rough_mesh_sha256=self.source_rough_mesh_sha256,
                          initial_assignment_counts_scope="source_importer_before_explicit_rehearsal_override",
                          rehearsal_plane={
                              "source": "explicit_fixed_flat_rehearsal_geometry_not_scut_generated",
                              "collision_path": REHEARSAL_PRIM_PATH,
                              "bounds_world_m": self.rehearsal_bounds_m,
                              "size_m": [400., 400.], "mesh_sha256": self.rehearsal_mesh_sha256,
                              "material_paths_reused": self.material_paths,
                              "origins_by_preclone_env_id": self.rehearsal_origins_by_env,
                              "command_sampling": "caller_owned_not_assigned_by_geometry",
                              "query_geometry": "concatenated_actual_native_mesh_triangles",
                              "internal_tile_boundaries": False,
                          })
        if self.mixed_rough_assignment is not None:
            result["mixed_rough_assignment"] = deepcopy(self.mixed_rough_assignment)
            result["initial_assignment_counts_scope"] = (
                "importer_after_rough_population_distribution_before_flat_origin_override")
        if self.rehearsal_collider == NATIVE_REHEARSAL_PLANE:
            result.update(
                height_query="rough_native_mesh_ray_and_equivalent_bounded_z0_plane_query",
                mesh_sha256_scope="rough_native_mesh_and_equivalent_bounded_plane_query_triangles",
                boundary_scope="union_of_rough_and_assigned_rehearsal_query_region")
            result["rehearsal_plane"].update(
                native_geometry="Plane", native_infinite=True,
                query_geometry="bounded_z0_triangles_equivalent_inside_rehearsal_region")
        return result


def create_rough_terrain(stage, settings, num_envs, device, seed, physics_material, *,
                         rehearsal_origins_by_env=None, max_init_terrain_level=None,
                         rehearsal_collider=None):
    """Keep reference geometry; optionally cap initial rows or add a separate plane."""
    if settings != {"type": "rough", "source": "scut0730"}:
        raise ValueError("Rough terrain requires the unchanged scut0730 archived geometry")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("Rough terrain requires a positive environment count")
    if max_init_terrain_level is not None and (
            type(max_init_terrain_level) is not int or not 0 <= max_init_terrain_level <= 9):
        raise ValueError("Rough initial terrain level must be an integer from 0 to 9 or None")
    rehearsal_origins = _rehearsal_origins(rehearsal_origins_by_env, num_envs)
    if rehearsal_collider is not None and (
            rehearsal_collider != NATIVE_REHEARSAL_PLANE or not rehearsal_origins):
        raise ValueError("Native rehearsal collider requires explicit flat origins and native-plane-v1")
    from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import Usd, UsdGeom, UsdPhysics

    class RetainedMeshImporter(TerrainImporter):
        def import_mesh(self, name, mesh):
            super().import_mesh(name, mesh)
            self.query_mesh = mesh

    cfg = TerrainImporterCfg(prim_path="/World/ground", terrain_type="generator",
                             num_envs=num_envs, terrain_generator=generator_config(seed),
                             env_spacing=4., use_terrain_origins=True,
                             max_init_terrain_level=max_init_terrain_level, physics_material=physics_material,
                             debug_vis=False)
    # Both SDK uniform noise and reference bars use NumPy's legacy global RNG.
    # Restore it after construction so geometry does not perturb later draws.
    np_state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        importer = RetainedMeshImporter(cfg)
    finally:
        np.random.set_state(np_state)
    if torch.device(importer.device) != torch.device(device):
        raise ValueError("Terrain and robot device differ")
    colliders = [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(cfg.prim_path))
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if len(colliders) != 1 or stage.GetPrimAtPath(colliders[0]).GetTypeName() != "Mesh":
        raise RuntimeError("reference rough requires one native generated triangle mesh")
    # Explicit triangle collision and visualization, without convex conversion.
    prim = stage.GetPrimAtPath(colliders[0])
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    UsdGeom.Mesh(prim).CreateSubdivisionSchemeAttr("none")
    mesh = importer.query_mesh
    points = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    mesh_sha = hashlib.sha256(points.tobytes() + faces.tobytes()).hexdigest()
    if not rehearsal_origins:
        return SharedRough(importer.env_origins.cpu().tolist(), colliders, [cfg.prim_path],
                       [cfg.prim_path + "/terrain/physicsMaterial"], importer,
                       convert_to_warp_mesh(points, faces, device=device), raycast_mesh,
                       mesh.bounds.tolist(), mesh_sha, int(seed),
                       max_init_terrain_level=max_init_terrain_level)
    low, high = mesh.bounds
    patch_low, patch_high = REHEARSAL_BOUNDS_M
    if not any(high[i] < patch_low[i] or patch_high[i] < low[i] for i in (0, 1)):
        raise ValueError("Rehearsal plane must be disjoint from the actual rough mesh bounds")
    patch_points, patch_faces = _rehearsal_plane_mesh()
    material_path = cfg.prim_path + "/terrain/physicsMaterial"
    patch_path = _author_rehearsal_plane(stage, patch_points, patch_faces, material_path,
                                       native_plane=rehearsal_collider == NATIVE_REHEARSAL_PLANE)
    query_points = np.concatenate((points, patch_points))
    query_faces = np.concatenate((faces, patch_faces + len(points)))
    query_sha = hashlib.sha256(query_points.tobytes() + query_faces.tobytes()).hexdigest()
    patch_sha = hashlib.sha256(patch_points.tobytes() + patch_faces.tobytes()).hexdigest()
    origins = importer.env_origins.cpu().tolist()
    for env_id, origin in rehearsal_origins.items():
        origins[env_id] = list(origin)
    return SharedRough(origins, colliders + [patch_path], [cfg.prim_path, patch_path],
                       [material_path], importer,
                       convert_to_warp_mesh(query_points, query_faces, device=device), raycast_mesh,
                       mesh.bounds.tolist(), query_sha, int(seed), rehearsal_origins,
                       [list(patch_low), list(patch_high)], mesh_sha, patch_sha,
                       max_init_terrain_level=max_init_terrain_level,
                       rehearsal_collider=rehearsal_collider)
