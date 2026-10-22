"""Final source geometry bound to the current SDK and retained collision mesh.

This builds terrain only. Command ownership and physical state machines are
explicit callers; historical SDK triangle identity and behavior are unclaimed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

from wheeled_tasks.chassis.tasks.terrain.final_rough_definition import (
    FINAL_TERRAIN_SOURCE, REFERENCE, SOURCE_COMMIT, VERSION as DEFINITION_VERSION,
    load_final_rough_definition, source_manifest,
)
from wheeled_tasks.chassis.simulation.terrain.rough_terrain import (
    REHEARSAL_BOUNDS_M, REHEARSAL_PRIM_PATH, SharedRough,
    _author_rehearsal_plane, _rehearsal_origins, _rehearsal_plane_mesh,
)


VERSION = "v7-final-rough-shared-mesh-v1"
SDK_COMPATIBILITY_ADAPTATION = "configclass_import_module_to_callable_only"


def _custom_config_classes(root=REFERENCE):
    source_manifest(root)
    path = Path(root) / "height_field.py"
    raw = path.read_text()
    old = "from isaaclab.utils import configclass\n"
    if raw.count(old) != 1:
        raise ValueError("Final custom geometry configclass import differs")
    # SDK 3 exposes this package name as a module after terrain imports.
    # Only its import path changes; all frozen geometry functions stay intact.
    raw = raw.replace(old, "from isaaclab.utils.configclass import configclass\n", 1)
    spec = importlib.util.spec_from_file_location("_v7_final_terrain_height_field", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module.MeshCustomGridBarsTerrainCfg, module.HfCliffInvertedPyramidStairsTerrainCfg


def generator_config(seed, root=REFERENCE):
    """Instantiate genuine SDK classes from authenticated active definitions."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Final geometry seed must be an unsigned 32-bit integer")
    definition = load_final_rough_definition(root)
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.height_field.hf_terrains_cfg import (
        HfPyramidSlopedTerrainCfg, HfPyramidStairsTerrainCfg, HfRandomUniformTerrainCfg,
    )
    from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

    bars, cliff = _custom_config_classes(root)
    classes = {
        "agent_world.terrains.height_field:mesh_grid_bars_terrain": bars,
        "agent_world.terrains.height_field:cliff_inverted_pyramid_stairs_terrain": cliff,
        "isaaclab.terrains.height_field.hf_terrains:pyramid_sloped_terrain": HfPyramidSlopedTerrainCfg,
        "isaaclab.terrains.height_field.hf_terrains:pyramid_stairs_terrain": HfPyramidStairsTerrainCfg,
        "isaaclab.terrains.height_field.hf_terrains:random_uniform_terrain": HfRandomUniformTerrainCfg,
        "isaaclab.terrains.trimesh.mesh_terrains:flat_terrain": MeshPlaneTerrainCfg,
    }
    raw = deepcopy(definition["terrain"]["terrain_generator"])
    raw.pop("class_type")
    raw["seed"] = seed
    # Source seed=None uses the process seed; binding that caller seed makes
    # actual generated geometry reproducible without changing difficulty.
    sub_terrains = {}
    for name, saved in raw.pop("sub_terrains").items():
        fields = deepcopy(saved)
        kind = fields.pop("function")
        if kind not in classes:
            raise ValueError("Unsupported final geometry function: " + kind)
        sub_terrains[name] = classes[kind](**fields)
    return TerrainGeneratorCfg(sub_terrains=sub_terrains, **raw)


def column_names(parameters=None):
    """Current SDK curriculum columns, using its NumPy proportion arithmetic."""
    if parameters is None:
        parameters = load_final_rough_definition()["terrain"]
    generator = parameters["terrain_generator"]
    entries = generator["sub_terrains"]
    proportions = np.array([entry["proportion"] for entry in entries.values()])
    proportions /= np.sum(proportions)
    cumulative = np.cumsum(proportions)
    names = list(entries)
    return [names[int(np.flatnonzero(col / generator["num_cols"] + .001 < cumulative)[0])]
            for col in range(generator["num_cols"])]


@dataclass
class SharedFinalRough(SharedRough):
    definition: dict = field(default_factory=load_final_rough_definition, repr=False)

    def boundary(self, world_xyz):
        cfg = self.definition["terrain"]["boundary_reset_cfg"]
        if not cfg["enabled"]:
            return torch.zeros(world_xyz.shape[:-1], dtype=torch.bool, device=world_xyz.device)
        if cfg["use_inner_terrain_area"]:
            raise ValueError("Final frozen boundary requires whole generated mesh bounds")
        return self._outside(world_xyz[..., :2], cfg["margin"])

    def summary(self):
        terrain = self.definition["terrain"]
        generator, boundary = terrain["terrain_generator"], terrain["boundary_reset_cfg"]
        rows, cols = generator["num_rows"], generator["num_cols"]
        manifest = source_manifest()
        result = dict(type="rough", source=FINAL_TERRAIN_SOURCE, shared=True,
            version=VERSION, definition_version=DEFINITION_VERSION, source_commit=SOURCE_COMMIT,
            source_terrain_sha256=manifest["files"]["terrains.py"]["sha256"],
            source_grid_sha256=manifest["files"]["height_field.py"]["sha256"],
            geometry_seed=self.seed, collision_paths=self.collision_paths, mesh_sha256=self.mesh_sha256,
            mesh_bounds_world_m=self.bounds_m, boundary_enabled=boundary["enabled"],
            boundary_margin_m=boundary["margin"], boundary_scope="whole_generated_terrain",
            height_query="vertical_ray_on_actual_collision_mesh_world_z", row_count=rows,
            column_count=cols, terrain_key_count=len(generator["sub_terrains"]),
            terrain_keys=list(generator["sub_terrains"]), column_names=column_names(terrain),
            tile_size_m=generator["size"], border_width_m=generator["border_width"],
            max_init_terrain_level=self.max_init_terrain_level,
            initial_level_scope="source_all_10_rows" if self.max_init_terrain_level is None
                else "local_initial_row_cap_only_geometry_unchanged",
            initial_level_counts=torch.bincount(self.importer.terrain_levels, minlength=rows).cpu().tolist(),
            initial_type_counts=torch.bincount(self.importer.terrain_types, minlength=cols).cpu().tolist(),
            generator_curriculum=generator["curriculum"],
            height_offset_curriculum_enabled=terrain["height_offset_curriculum_cfg"]["enabled"],
            training_terrain_progression=False, skill_state_machines_enabled=False,
            sdk_compatibility_adaptation=SDK_COMPATIBILITY_ADAPTATION,
            historical_sdk_triangle_identity_verified=False, native_contact_qualified=False,
            scope="final_source_geometry_on_current_sdk_not_behavior_qualification")
        if self.rehearsal_bounds_m is not None:
            result.update(boundary_scope="union_of_rough_and_separate_finite_rehearsal_plane",
                boundary_regions_world_m=[self.bounds_m, self.rehearsal_bounds_m],
                query_mesh_bounds_world_m=self.query_bounds_m,
                mesh_sha256_scope="concatenated_actual_native_collision_meshes",
                rough_mesh_bounds_world_m=self.bounds_m, rough_mesh_sha256=self.source_rough_mesh_sha256,
                initial_assignment_counts_scope="source_importer_before_explicit_rehearsal_override",
                rehearsal_plane={"source": "explicit_fixed_flat_rehearsal_geometry",
                    "collision_path": REHEARSAL_PRIM_PATH, "bounds_world_m": self.rehearsal_bounds_m,
                    "size_m": [400., 400.], "mesh_sha256": self.rehearsal_mesh_sha256,
                    "material_paths_reused": self.material_paths,
                    "origins_by_preclone_env_id": self.rehearsal_origins_by_env,
                    "command_sampling": "caller_owned_not_assigned_by_geometry",
                    "query_geometry": "concatenated_actual_native_mesh_triangles",
                    "internal_tile_boundaries": False})
        return result


def create_final_rough_terrain(stage, settings, num_envs, device, seed, physics_material, *,
                               rehearsal_origins_by_env=None, max_init_terrain_level=None):
    """Author final geometry and retain exactly the native collision triangles."""
    if settings != {"type": "rough", "source": FINAL_TERRAIN_SOURCE}:
        raise ValueError("Final rough terrain requires its explicit frozen source")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("Final rough terrain requires a positive environment count")
    if max_init_terrain_level is not None and (
            type(max_init_terrain_level) is not int or not 0 <= max_init_terrain_level <= 9):
        raise ValueError("Final initial terrain level must be an integer from 0 to 9 or None")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Final geometry seed must be an unsigned 32-bit integer")
    definition = load_final_rough_definition()
    rehearsal_origins = _rehearsal_origins(rehearsal_origins_by_env, num_envs)
    from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import Usd, UsdGeom, UsdPhysics

    class RetainedMeshImporter(TerrainImporter):
        def import_mesh(self, name, mesh):
            super().import_mesh(name, mesh)
            self.query_mesh = mesh

    importer_fields = definition["terrain"]
    cfg = TerrainImporterCfg(prim_path=importer_fields["prim_path"], terrain_type="generator",
        collision_group=importer_fields["collision_group"], num_envs=num_envs,
        terrain_generator=generator_config(seed), env_spacing=importer_fields["env_spacing"],
        use_terrain_origins=importer_fields["use_terrain_origins"],
        max_init_terrain_level=max_init_terrain_level, physics_material=physics_material,
        debug_vis=importer_fields["debug_vis"])
    # Frozen bars and SDK noise use the legacy global NumPy stream. Preserve
    # caller RNG state while binding the complete geometry to the supplied seed.
    np_state = np.random.get_state()
    try:
        np.random.seed(seed)
        importer = RetainedMeshImporter(cfg)
    finally:
        np.random.set_state(np_state)
    if torch.device(importer.device) != torch.device(device):
        raise ValueError("Terrain and robot device differ")
    colliders = [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(cfg.prim_path))
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if len(colliders) != 1 or stage.GetPrimAtPath(colliders[0]).GetTypeName() != "Mesh":
        raise RuntimeError("Final rough requires one native generated triangle mesh")
    prim = stage.GetPrimAtPath(colliders[0])
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    UsdGeom.Mesh(prim).CreateSubdivisionSchemeAttr("none")
    mesh = importer.query_mesh
    points = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    mesh_sha = hashlib.sha256(points.tobytes() + faces.tobytes()).hexdigest()
    if not rehearsal_origins:
        return SharedFinalRough(importer.env_origins.cpu().tolist(), colliders, [cfg.prim_path],
            [cfg.prim_path + "/terrain/physicsMaterial"], importer,
            convert_to_warp_mesh(points, faces, device=device), raycast_mesh,
            mesh.bounds.tolist(), mesh_sha, seed, max_init_terrain_level=max_init_terrain_level,
            definition=definition)
    low, high = mesh.bounds
    patch_low, patch_high = REHEARSAL_BOUNDS_M
    if not any(high[i] < patch_low[i] or patch_high[i] < low[i] for i in (0, 1)):
        raise ValueError("Rehearsal plane must be disjoint from the actual rough mesh bounds")
    patch_points, patch_faces = _rehearsal_plane_mesh()
    material_path = cfg.prim_path + "/terrain/physicsMaterial"
    patch_path = _author_rehearsal_plane(stage, patch_points, patch_faces, material_path)
    query_points = np.concatenate((points, patch_points))
    query_faces = np.concatenate((faces, patch_faces + len(points)))
    query_sha = hashlib.sha256(query_points.tobytes() + query_faces.tobytes()).hexdigest()
    patch_sha = hashlib.sha256(patch_points.tobytes() + patch_faces.tobytes()).hexdigest()
    origins = importer.env_origins.cpu().tolist()
    for env_id, origin in rehearsal_origins.items():
        origins[env_id] = list(origin)
    return SharedFinalRough(origins, colliders + [patch_path], [cfg.prim_path, patch_path],
        [material_path], importer, convert_to_warp_mesh(query_points, query_faces, device=device), raycast_mesh,
        mesh.bounds.tolist(), query_sha, seed, rehearsal_origins, [list(patch_low), list(patch_high)],
        mesh_sha, patch_sha, max_init_terrain_level=max_init_terrain_level, definition=definition)
