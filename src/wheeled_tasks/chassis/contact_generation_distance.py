"""Explicit contact generation distance with preserved other ground pairs."""
import hashlib
import math
import re

import numpy as np
import torch


VERSION = "v7_base_terrain_contact_distance_v1"
UPPER_LINK_VERSION = "v7_base_upper_link_terrain_contact_distance_v2"
LEG_LINK_VERSION = "v7_base_leg_link_terrain_contact_distance_v3"
ROUGH_WHEEL_VERSION = "v7_rough_wheel_terrain_contact_distance_v4"
ROUGH_WHEEL_MARGIN_VERSION = "v7_rough_wheel_terrain_contact_distance_v5"
ROUGH_WHEEL_VERSIONS = (ROUGH_WHEEL_VERSION, ROUGH_WHEEL_MARGIN_VERSION)
SUPPORTED_VERSIONS = (VERSION, UPPER_LINK_VERSION, LEG_LINK_VERSION, *ROUGH_WHEEL_VERSIONS)
ROUGH_WHEEL_MARGIN_M = .005
KEY = "formal_contact_generation_distance"
OFFSET_M = 1e-5
# Native static-shape readback at 5 ms, from the retained OVD recording.
# These are geometry-bound references, not universal automatic mesh defaults.
MESH_RECORDING_SHA256 = "ea0b6b30f50dfee3986c5760466090a54a3d9b981ec0c73822a0180d24775384"
MESH_REFERENCE_OFFSETS = {
    "b5ae55b9342712ddb600311541ece9509602b10a05857206b221520d52dc31fe": .041999995708465576,
    "dda81a9f2482ee6d51dceee90e4078369cf2676b4e91d8fcdb19fa0a0e6d99bb": .0004905000096186996,
}


def selected(config):
    if KEY not in config:
        return False
    if (config[KEY] not in SUPPORTED_VERSIONS or config.get("formal_v7_flat") is not True
            or config.get("physics_dt") != .005 or config.get("physx_solver_type") != 1
            or config.get("position_iterations") != 12 or config.get("velocity_iterations") != 6):
        raise ValueError("Contact distance requires its explicit V7 version and 200Hz/TGS12/6")
    if (config[KEY] in ROUGH_WHEEL_VERSIONS
            and config.get("formal_terrain", {}).get("type") not in ("rough", "plane")):
        raise ValueError("Rough wheel contact distance requires explicit terrain selection")
    return True


class ContactGenerationDistance:
    """Author static shapes before parsing, then compensate native robot shapes.

    Only contactOffset changes. Rest distances, material friction, collision
    geometry, friction activation distance and solver settings stay unchanged.
    Static terrain has no rigid-body tensor view: an automatic Plane uses its
    documented 20 mm default. Automatic meshes require retained native evidence
    for their exact geometry. Mixed meshes compensate only their assigned
    robots; inverted collision groups exclude the other terrain collider.
    """

    def __init__(self, stage, collision_paths, *, terrain=None, retain_flat_parent_pairs=False,
                 version=VERSION):
        from pxr import Sdf, UsdGeom, UsdPhysics

        if version not in SUPPORTED_VERSIONS:
            raise ValueError("Unknown contact generation distance version")
        self.version = version
        if UsdGeom.GetStageMetersPerUnit(stage) != 1. or not collision_paths:
            raise ValueError("Contact distance requires explicit terrain colliders in meter units")
        self.rough_wheel_path = None
        if version in ROUGH_WHEEL_VERSIONS:
            from .scut_rough_terrain import SharedRough
            first = stage.GetPrimAtPath(collision_paths[0])
            if len(collision_paths) == 1 and first.IsA(UsdGeom.Plane):
                pass
            elif isinstance(terrain, SharedRough) and first.IsA(UsdGeom.Mesh):
                self.rough_wheel_path = collision_paths[0]
            else:
                raise ValueError("Rough wheel contact distance requires its retained rough mesh")
        if len(set(collision_paths)) != len(collision_paths):
            raise ValueError("Terrain collider paths must be unique")
        if (type(retain_flat_parent_pairs) is not bool
                or retain_flat_parent_pairs and (terrain is None or len(collision_paths) != 2
                    or collision_paths[1] != "/World/flat_rehearsal")):
            raise ValueError("Parent plane pair distances require the explicit mixed flat terrain")
        pending = []
        records = []
        for path in collision_paths:
            prim = stage.GetPrimAtPath(path)
            if (not prim or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True
                    or not (prim.IsA(UsdGeom.Plane) or prim.IsA(UsdGeom.Mesh))):
                raise ValueError("Contact distance requires enabled static Plane or Mesh colliders")
            ancestor = prim
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError("Contact distance cannot rewrite dynamic or kinematic terrain")
                ancestor = ancestor.GetParent()
            contact = prim.GetAttribute("physxCollision:contactOffset")
            rest = prim.GetAttribute("physxCollision:restOffset")
            old = contact.Get() if contact and contact.HasAuthoredValueOpinion() else None
            source = "authored_USD"
            if old is None or old == -math.inf:
                if prim.IsA(UsdGeom.Plane):
                    old, source = .02, "documented_Plane_default"
                elif terrain is not None:
                    old, source = self._recorded_mesh_offset(prim)
                else:
                    raise ValueError("Automatic mesh contact distance needs resolved evidence before adaptation")
            rest_value = rest.Get() if rest and rest.HasAuthoredValueOpinion() else None
            if (not math.isfinite(old) or old <= OFFSET_M
                    or rest_value not in (None, -math.inf, 0.)):
                raise ValueError("Contact distance requires a positive original offset and zero resting offset")
            records.append({"path": path, "shape": prim.GetTypeName(),
                "original_contact_offset_m": old, "original_value_source": source,
                "rest_offset_m": 0.})
            if retain_flat_parent_pairs and path == "/World/flat_rehearsal":
                records[-1].update(preserved_pair_reference_offset_m=.02,
                    pair_reference_source="inherited_parent_infinite_Plane_documented_default")
            pending.append(prim)
        self.original_terrain_offset = records[0]["original_contact_offset_m"]
        self.terrain_by_env = None
        if any(record["original_contact_offset_m"] != self.original_terrain_offset for record in records):
            if terrain is None:
                raise ValueError("Different terrain offsets require explicit environment ownership")
            self.terrain_by_env = self._mesh_ownership(stage, terrain, collision_paths)
            for prim in pending:
                relation = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
                if relation and relation.GetTargets():
                    raise ValueError("Mixed contact terrain requires unfiltered original colliders")
            self._isolate_owned_terrain(stage, collision_paths, self.terrain_by_env)
        self.original_by_path = {r["path"]: r.get("preserved_pair_reference_offset_m",
                                                r["original_contact_offset_m"]) for r in records}
        for prim, record in zip(pending, records, strict=True):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            attribute = prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float)
            attribute.Set(OFFSET_M)
            record["contact_offset_m"] = attribute.Get()
        self.terrain_offset = records[0]["contact_offset_m"]
        self.receipt = {"version": version, "terrain": records,
            "terrain_readback_scope": "authored_USD_before_parsing; static shapes have no rigid-body tensor view",
            "base_offset_m": OFFSET_M, "rest_offsets_changed": False,
            "materials_changed": False, "native_applied": False}
        if self.terrain_by_env is not None:
            self.receipt.update(
                assigned_terrain_counts={p: self.terrain_by_env.count(p) for p in collision_paths},
                terrain_filter_scope="inverted collision groups, each clone linked only to its assigned terrain; before parsing",
                preserved_pair_scope="each robot and its assigned terrain only",
                mesh_recording_sha256=MESH_RECORDING_SHA256)
        if retain_flat_parent_pairs:
            self.receipt["flat_parent_pair_scope"] = (
                "Unmodified flat pairs retain the old infinite-plane 20mm reference")

    @staticmethod
    def _isolate_owned_terrain(stage, paths, ownership):
        """Split the existing global group without expanding per-shape pairs.

        FilteredPairs against thousands of articulation hierarchies makes
        native scene attachment unbounded in the full-size mixed scene. The
        SDK's inverted group representation expresses the same exclusions
        while retaining one collision group per environment.
        """
        from pxr import Usd, UsdPhysics

        prefix = "/World/collisions"
        original = prefix + "/global_group"
        split = prefix + "/flat_terrain_group"
        clone_groups = [prefix + f"/group{i}" for i in range(len(ownership))]

        def targets(prim, name):
            return [str(p) for p in prim.GetRelationship(name).GetTargets()]

        scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        global_prim = stage.GetPrimAtPath(original)
        if (len(scenes) != 1 or scenes[0].GetAttribute(
                "physxScene:invertCollisionGroupFilter").Get() is not True
                or not global_prim or stage.GetPrimAtPath(split)
                or set(targets(global_prim, "physics:filteredGroups")) != {original, *clone_groups}):
            raise ValueError("Mixed terrain requires the original inverted global collision group")
        includes = targets(global_prim, "collection:colliders:includes")
        if (len(paths) != 2 or set(ownership) != set(paths)
                or len(includes) != 2 or any(not any(
                    path == root or path.startswith(root + "/") for root in includes) for path in paths)):
            raise ValueError("Mixed terrain groups do not match the original global colliders")
        prims = [stage.GetPrimAtPath(path) for path in clone_groups]
        for i, (prim, path) in enumerate(zip(prims, clone_groups, strict=True)):
            if (not prim or targets(prim, "collection:colliders:includes") != [f"/World/envs/env_{i}"]
                    or set(targets(prim, "physics:filteredGroups")) != {path, original}
                    or not stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")):
                raise ValueError("Mixed terrain requires unchanged per-clone collision isolation")

        terrain_groups = dict(zip(paths, (original, split), strict=True))
        expected = {}
        for path, group_path in terrain_groups.items():
            group = UsdPhysics.CollisionGroup.Define(stage, group_path)
            collection = Usd.CollectionAPI.Apply(group.GetPrim(), "colliders")
            collection.CreateExpansionRuleAttr().Set("expandPrims")
            collection.CreateIncludesRel().SetTargets([path])
            expected[group_path] = [group_path] + [g for g, p in zip(clone_groups, ownership, strict=True) if p == path]
            group.CreateFilteredGroupsRel().SetTargets(expected[group_path])
        for prim, own_group, terrain in zip(prims, clone_groups, ownership, strict=True):
            expected[own_group] = [own_group, terrain_groups[terrain]]
            prim.GetRelationship("physics:filteredGroups").SetTargets(expected[own_group])
        for path, allowed in expected.items():
            if targets(stage.GetPrimAtPath(path), "physics:filteredGroups") != allowed:
                raise RuntimeError("Assigned terrain collision group readback differs")

    @staticmethod
    def _recorded_mesh_offset(prim):
        from pxr import Usd, UsdGeom

        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or not counts.size or not np.all(counts == 3)
                or faces.size != counts.size * 3 or not np.array_equal(transform, np.eye(4))):
            raise ValueError("Recorded mesh contact reference requires unchanged world-space triangles")
        digest = hashlib.sha256(points.tobytes() + faces.tobytes()).hexdigest()
        if digest not in MESH_REFERENCE_OFFSETS:
            raise ValueError("Automatic mesh contact distance has no resolved evidence for this geometry")
        return MESH_REFERENCE_OFFSETS[digest], "native_OVD_mesh_sha256:" + digest

    @staticmethod
    def _mesh_ownership(stage, terrain, collision_paths):
        from pxr import UsdGeom
        from .scut_rough_terrain import REHEARSAL_PRIM_PATH, SharedRough

        if (not isinstance(terrain, SharedRough) or len(collision_paths) != 2
                or collision_paths[1] != REHEARSAL_PRIM_PATH
                or not terrain.rehearsal_origins_by_env):
            raise ValueError("Mixed contact offsets require explicit rough and rehearsal assignments")
        bounds = []
        for path in collision_paths:
            points = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get())
            bounds.append((points.min(axis=0), points.max(axis=0)))
        if not any(bounds[0][1][i] < bounds[1][0][i] or bounds[1][1][i] < bounds[0][0][i]
                   for i in (0, 1)):
            raise ValueError("Mixed contact terrain regions must be disjoint")
        result = []
        for env_id, origin in enumerate(terrain.origins_m):
            index = int(env_id in terrain.rehearsal_origins_by_env)
            low, high = bounds[index]
            if not all(low[i] < origin[i] < high[i] for i in (0, 1)):
                raise ValueError("Robot origin is outside its assigned contact terrain")
            result.append(collision_paths[index])
        if set(result) != set(collision_paths):
            raise ValueError("Mixed contact adaptation requires robots assigned to both terrains")
        return tuple(result)

    def apply_to_robot(self, robot, physics_sim_view):
        import warp as wp
        from .privileged_runtime import wheel_shape_layout

        if self.receipt["native_applied"]:
            raise RuntimeError("Contact distance compensation must not be applied twice")
        layout = wheel_shape_layout(robot, physics_sim_view)
        base = robot.body_names.index("base_link")
        start = sum(layout.shapes_per_body[:base])
        end = start + layout.shapes_per_body[base]
        if end - start != 8:
            raise ValueError("Contact distance expects the bound eight base collision shapes")
        ranges = {"base_link": (start, end)}
        if self.version in (UPPER_LINK_VERSION, LEG_LINK_VERSION, *ROUGH_WHEEL_VERSIONS):
            names = ("L_link1", "R_link1")
            if self.version in (LEG_LINK_VERSION, *ROUGH_WHEEL_VERSIONS):
                names += ("L_link2", "R_link2")
            for name in names:
                if robot.body_names.count(name) != 1:
                    raise ValueError("Leg-link contact repair requires all bound paired links")
                index = robot.body_names.index(name)
                first = sum(layout.shapes_per_body[:index])
                if layout.shapes_per_body[index] != 1:
                    raise ValueError("Leg-link contact repair expects one collision shape per bound link")
                ranges[name] = (first, first + 1)
        view = robot.root_view
        old = wp.to_torch(view.get_contact_offsets()).clone()
        rest = wp.to_torch(view.get_rest_offsets()).clone()
        if (old.shape != (view.count, layout.total_shapes) or rest.shape != old.shape
                or not bool(torch.isfinite(old).all()) or bool((old <= 0.).any())
                or bool((rest != 0.).any())):
            raise ValueError("Native shape offsets differ from the homogeneous robot layout")
        references = torch.full((view.count, 1), self.original_terrain_offset,
                                dtype=old.dtype, device=old.device)
        if self.terrain_by_env is not None:
            matches = [re.fullmatch(r"/World/envs/env_(\d+)/Robot(?:/.*)?", str(path))
                       for path in view.prim_paths]
            if (any(match is None for match in matches)
                    or view.count != len(self.terrain_by_env)
                    or sorted(int(match[1]) for match in matches) != list(range(view.count))):
                raise ValueError("Mixed contact compensation requires a complete native clone permutation")
            native_terrain = [self.terrain_by_env[int(match[1])] for match in matches]
            references[:, 0] = torch.tensor([self.original_by_path[path] for path in native_terrain],
                                            dtype=old.dtype, device=old.device)
        updated = old + (references - self.terrain_offset)
        for first, last in ranges.values():
            updated[:, first:last] = OFFSET_M
        wheel_rows = None
        if self.rough_wheel_path is not None:
            if (tuple(robot.body_names[i] for i in layout.body_ids) != ("L_link3", "R_link3")
                    or len(layout.shape_ranges) != 2):
                raise ValueError("Rough wheel contact distance requires both named tire shape ranges")
            wheel_rows = (torch.tensor([path == self.rough_wheel_path for path in native_terrain],
                                      device=old.device, dtype=torch.bool)
                          if self.terrain_by_env is not None else
                          torch.ones(view.count, dtype=torch.bool, device=old.device))
            # The 5 mm margin avoids the captured deep-overlap event at 200 Hz.
            # Neither revision inherits the much larger mesh margin.
            for first, last in layout.shape_ranges:
                margin = (torch.full_like(old[:, first:last], ROUGH_WHEEL_MARGIN_M)
                          if self.version == ROUGH_WHEEL_MARGIN_VERSION else old[:, first:last])
                updated[:, first:last] = torch.where(wheel_rows[:, None],
                    margin, updated[:, first:last])
        indices = torch.arange(view.count, dtype=torch.int32)
        view.set_contact_offsets(wp.from_torch(updated.contiguous()), wp.from_torch(indices))
        actual = wp.to_torch(view.get_contact_offsets()).clone()
        if not torch.equal(actual, updated) or not torch.equal(wp.to_torch(view.get_rest_offsets()), rest):
            raise RuntimeError("Native contact distance or resting offset readback differs")
        other = torch.ones_like(old, dtype=torch.bool)
        for first, last in ranges.values():
            other[:, first:last] = False
        if wheel_rows is not None:
            for first, last in layout.shape_ranges:
                other[:, first:last] &= ~wheel_rows[:, None]
        before_pairs = (old + references)[other]
        after_pairs = (actual + self.terrain_offset)[other]
        if not torch.allclose(before_pairs, after_pairs, atol=1e-8, rtol=1e-6):
            raise RuntimeError("Unmodified ground contact generation distance was not preserved")
        self.receipt.update(native_applied=True, articulation_count=view.count,
            shapes_per_articulation=layout.total_shapes, base_shape_range=[start, end],
            original_first_articulation_offsets_m=old[0].cpu().tolist(),
            actual_first_articulation_offsets_m=actual[0].cpu().tolist(),
            other_pair_distance_max_error_m=float((before_pairs - after_pairs).abs().max()),
            base_pair_contact_distance_m=float(actual[0, start] + self.terrain_offset),
            repaired_shape_ranges={name: list(bounds) for name, bounds in ranges.items()},
            repaired_pair_contact_distance_m={
                name: float(actual[0, first] + self.terrain_offset)
                for name, (first, _) in ranges.items()},
            native_readback_scope="all articulation shapes through native tensor getters",
            capability_qualified=False)
        if wheel_rows is not None:
            self.receipt.update(
                rough_wheel_scope=("5mm tire margins on assigned rough mesh; flat parent pairs retained"
                    if self.version == ROUGH_WHEEL_MARGIN_VERSION else
                    "own native tire margins on assigned rough mesh; flat parent pairs retained"),
                rough_wheel_articulation_count=int(wheel_rows.sum()),
                rough_wheel_terrain_path=self.rough_wheel_path,
                rough_wheel_shape_ranges={robot.body_names[body]: list(bounds)
                    for body, bounds in zip(layout.body_ids, layout.shape_ranges, strict=True)})
            if self.version == ROUGH_WHEEL_MARGIN_VERSION:
                self.receipt["rough_wheel_contact_offset_m"] = ROUGH_WHEEL_MARGIN_M
        elif self.version in ROUGH_WHEEL_VERSIONS:
            self.receipt.update(rough_wheel_articulation_count=0,
                rough_wheel_scope="No rough mesh: retain all tire ground pairs and paired-link repair")
        if self.terrain_by_env is not None:
            self.receipt["native_group_readback"] = {
                path: {"count": native_terrain.count(path),
                       "reference_terrain_contact_offset_m": self.original_by_path[path],
                       "first_actual_robot_offsets_m": actual[native_terrain.index(path)].cpu().tolist()}
                for path in self.original_by_path}
