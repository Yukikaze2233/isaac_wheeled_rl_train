"""Shared reference terrain adapters; native imports are delayed until authoring."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class SharedPlane:
    origins_m: list
    collision_paths: list
    global_prim_paths: list
    material_paths: list
    env_spacing_m: float

    def height(self, world_xy):
        return torch.zeros_like(world_xy[..., 0])

    def boundary(self, world_xyz):
        return torch.zeros_like(world_xyz[..., 0], dtype=torch.bool)

    def raycast(self, starts, directions):
        """Analytic forward-ray intersections with the actual infinite z=0 plane."""
        if (not isinstance(starts, torch.Tensor) or not isinstance(directions, torch.Tensor)
                or starts.shape != directions.shape or starts.shape[-1] != 3
                or starts.device != directions.device or starts.dtype != directions.dtype
                or not starts.is_floating_point()):
            raise ValueError("Plane rays require matching floating point [...,3] starts and directions")
        dz = directions[..., 2]
        distance = -starts[..., 2] / torch.where(dz != 0., dz, torch.ones_like(dz))
        valid = (torch.isfinite(starts).all(-1) & torch.isfinite(directions).all(-1)
                 & (dz != 0.) & (distance >= 0.) & torch.isfinite(distance))
        hits = starts + distance[..., None] * directions
        hits[..., 2] = 0.
        hits = torch.where(valid[..., None], hits, torch.full_like(hits, torch.inf))
        return (hits,)

    def summary(self):
        return {"type": "plane", "source": "scut0719", "shared": True,
                "env_spacing_m": self.env_spacing_m, "xy_boundary_enabled": False,
                "ground_height_world_m": 0., "collision_paths": self.collision_paths,
                "robot_collision_isolation": "inverted_per_environment_collision_groups"}


def plane_origins(num_envs, spacing):
    if num_envs < 1 or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Shared plane requires positive count and finite spacing")
    columns = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / columns)
    return [((i % columns - (columns - 1) / 2) * spacing,
             (i // columns - (rows - 1) / 2) * spacing, 0.) for i in range(num_envs)]


def create_shared_terrain(stage, settings, num_envs, device, seed, physics_material,
                          *, rehearsal_origins_by_env=None, max_init_terrain_level=None,
                          rehearsal_collider=None):
    """Return one world terrain with exact contact-filter collider paths."""
    if settings["type"] == "rough":
        if settings.get("source") == "v7_final_b8ff79f3":
            if rehearsal_collider is not None:
                raise ValueError("Native rehearsal selection applies only to the retained small rough course")
            from wheeled_tasks.chassis.simulation.terrain.final_rough_terrain import create_final_rough_terrain
            return create_final_rough_terrain(stage, settings, num_envs, device, seed, physics_material,
                rehearsal_origins_by_env=rehearsal_origins_by_env,
                max_init_terrain_level=max_init_terrain_level)
        from wheeled_tasks.chassis.simulation.terrain.rough_terrain import create_rough_terrain
        return create_rough_terrain(stage, settings, num_envs, device, seed, physics_material,
            **({"rehearsal_collider": rehearsal_collider} if rehearsal_collider is not None else {}),
            **({"max_init_terrain_level": max_init_terrain_level}
               if max_init_terrain_level is not None else {}),
            **({"rehearsal_origins_by_env": rehearsal_origins_by_env}
               if rehearsal_origins_by_env is not None else {}))
    if rehearsal_origins_by_env is not None or rehearsal_collider is not None:
        raise ValueError("Rehearsal patch is an explicit rough training adaptation")
    if max_init_terrain_level is not None:
        raise ValueError("Initial terrain level applies only to rough terrain")
    if settings["type"] != "plane" or settings.get("source") != "scut0719":
        raise ValueError("Unknown formal shared terrain")
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdPhysics

    spacing = float(settings["env_spacing_m"])
    origins = plane_origins(num_envs, spacing)
    path = "/World/ground"
    # As in TerrainImporter, size changes the visible mesh. The physical
    # collider in GroundPlaneCfg is a Plane, not a finite lane mesh or box.
    cfg = sim_utils.GroundPlaneCfg(size=(2.0e6, 2.0e6), physics_material=physics_material)
    cfg.func(path, cfg)
    colliders = [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(path))
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if len(colliders) != 1 or stage.GetPrimAtPath(colliders[0]).GetTypeName() != "Plane":
        raise RuntimeError(f"Shared ground requires one native Plane collider, got {colliders}")
    return SharedPlane(origins, colliders, [path], [path + "/physicsMaterial"], spacing)


def isolate_shared_terrain_robots(stage, physics_scene_path, env_paths, global_paths):
    """Use the SDK's collision groups, independent of spatial separation."""
    from isaaclab import cloner
    cloner.filter_collisions(stage, physics_scene_path, "/World/collisions",
                             env_paths, global_paths=global_paths)
    scene = stage.GetPrimAtPath(physics_scene_path)
    if scene.GetAttribute("physxScene:invertCollisionGroupFilter").Get() is not True:
        raise RuntimeError("Shared terrain robot collision isolation was not authored")
    for i, path in enumerate(env_paths):
        group_path = f"/World/collisions/group{i}"
        group = stage.GetPrimAtPath(group_path)
        includes = [str(p) for p in group.GetRelationship("collection:colliders:includes").GetTargets()]
        allowed = [str(p) for p in group.GetRelationship("physics:filteredGroups").GetTargets()]
        if includes != [path] or set(allowed) != {group_path, "/World/collisions/global_group"}:
            raise RuntimeError(f"Shared terrain collision group readback differs: {group_path}")
    return {"inverted_filter": True, "environment_groups_verified": len(env_paths),
            "cross_environment_collisions_enabled": False, "global_paths": list(global_paths),
            "scope": "authored_USD_collision_groups_before_simulation_reset"}
