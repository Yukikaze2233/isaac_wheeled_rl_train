"""Contact-offset selection, atomic terrain authoring and pair preservation."""
from types import SimpleNamespace as NS
import hashlib

import numpy as np
import pytest
import torch
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from wheeled_tasks.chassis.contact_generation_distance import (
    ContactGenerationDistance, KEY, OFFSET_M, VERSION, UPPER_LINK_VERSION, LEG_LINK_VERSION,
    ROUGH_WHEEL_VERSION, ROUGH_WHEEL_MARGIN_VERSION, ROUGH_WHEEL_MARGIN_M,
    ROUGH_WHEEL_IMPACT_VERSION, ROUGH_WHEEL_IMPACT_MARGIN_M, selected,
)


def mixed_terrain(monkeypatch):
    from wheeled_tasks.chassis import contact_generation_distance as contact
    from wheeled_tasks.chassis.scut_rough_terrain import SharedRough, _rehearsal_plane_mesh

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    patch, faces = _rehearsal_plane_mesh()
    rough = np.asarray([[-10., -10., 0.], [10., -10., .1],
                        [10., 10., 0.], [-10., 10., .1]], dtype=np.float32)
    paths = ["/World/ground/terrain/mesh", "/World/flat_rehearsal"]
    references = dict(contact.MESH_REFERENCE_OFFSETS)
    references[hashlib.sha256(rough.tobytes() + faces.tobytes()).hexdigest()] = .042
    monkeypatch.setattr(contact, "MESH_REFERENCE_OFFSETS", references)
    for path, points in zip(paths, (rough, patch), strict=True):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points.tolist())
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    origins = [[0., 0., 0.], [0., 300., 0.], [1., 0., 0.]]
    for env_id in range(3):
        prim = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/Robot").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(prim)
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene").GetPrim()
    scene.CreateAttribute("physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool).Set(True)
    global_path = "/World/collisions/global_group"
    clone_groups = [f"/World/collisions/group{i}" for i in range(3)]
    for i, path in enumerate([global_path, *clone_groups]):
        group = UsdPhysics.CollisionGroup.Define(stage, path)
        collection = Usd.CollectionAPI.Apply(group.GetPrim(), "colliders")
        collection.CreateExpansionRuleAttr().Set("expandPrims")
        collection.CreateIncludesRel().SetTargets(paths if i == 0 else [f"/World/envs/env_{i - 1}"])
        group.CreateFilteredGroupsRel().SetTargets([global_path, *clone_groups] if i == 0 else [path, global_path])
    shared = SharedRough(origins, paths, [], [], None, None, None,
                         [rough.min(0).tolist(), rough.max(0).tolist()], "unused", 1,
                         rehearsal_origins_by_env={1: origins[1]})
    return stage, shared


def terrain(mesh=False, offset=None):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    shape = (UsdGeom.Mesh if mesh else UsdGeom.Plane).Define(stage, "/ground")
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    if offset is not None:
        shape.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        shape.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
    return stage, shape.GetPrim()


def test_candidate_is_explicit_and_keeps_the_selected_solver():
    config = {"formal_v7_flat": True, "physics_dt": .005, "physx_solver_type": 1,
              "position_iterations": 12, "velocity_iterations": 6}
    assert not selected(config)
    assert selected({**config, KEY: VERSION})
    assert selected({**config, KEY: UPPER_LINK_VERSION})
    assert selected({**config, KEY: LEG_LINK_VERSION})
    assert selected({**config, KEY: ROUGH_WHEEL_VERSION, "formal_terrain": {"type": "rough"}})
    assert selected({**config, KEY: ROUGH_WHEEL_VERSION, "formal_terrain": {"type": "plane"}})
    assert selected({**config, KEY: ROUGH_WHEEL_MARGIN_VERSION, "formal_terrain": {"type": "rough"}})
    assert selected({**config, KEY: ROUGH_WHEEL_MARGIN_VERSION, "formal_terrain": {"type": "plane"}})
    assert selected({**config, KEY: ROUGH_WHEEL_IMPACT_VERSION, "formal_terrain": {"type": "rough"}})
    assert selected({**config, KEY: ROUGH_WHEEL_IMPACT_VERSION, "formal_terrain": {"type": "plane"}})
    with pytest.raises(ValueError, match="explicit terrain selection"):
        selected({**config, KEY: ROUGH_WHEEL_VERSION})
    for change in ({KEY: True}, {"velocity_iterations": 0}, {"physics_dt": .001}):
        with pytest.raises(ValueError, match="200Hz/TGS12/6"):
            selected({**config, KEY: VERSION, **change})


def test_unsupported_mesh_rejected_before_any_ground_is_rewritten():
    stage, plane = terrain()
    mesh = UsdGeom.Mesh.Define(stage, "/mesh").GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh)
    with pytest.raises(ValueError, match="resolved evidence"):
        ContactGenerationDistance(stage, ["/ground", "/mesh"])
    assert not plane.GetAttribute("physxCollision:contactOffset").HasAuthoredValueOpinion()
    assert not mesh.GetAttribute("physxCollision:contactOffset").HasAuthoredValueOpinion()


def test_explicit_mesh_offset_has_no_plane_default_assumption():
    stage, mesh = terrain(mesh=True, offset=.012)
    value = ContactGenerationDistance(stage, ["/ground"])
    assert value.original_terrain_offset == pytest.approx(.012)
    assert value.receipt["terrain"][0]["original_value_source"] == "authored_USD"
    assert mesh.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(OFFSET_M)


def test_native_all_clone_update_preserves_other_pairs_and_rest(monkeypatch):
    import warp as wp
    from wheeled_tasks.chassis import privileged_runtime

    wp.init()
    stage, _ = terrain()
    candidate = ContactGenerationDistance(stage, ["/ground"])
    old = torch.tensor([[.003] * 8 + [.002, .008, .01, .02]], dtype=torch.float32).repeat(3, 1)
    current = old.clone()
    rest = torch.zeros_like(old)
    calls = []

    def setter(values, ids):
        values, ids = wp.to_torch(values), wp.to_torch(ids).long()
        assert ids.tolist() == [0, 1, 2]
        current[ids] = values[ids]
        calls.append(1)

    view = NS(count=3, get_contact_offsets=lambda: wp.from_torch(current),
              get_rest_offsets=lambda: wp.from_torch(rest), set_contact_offsets=setter)
    robot = NS(root_view=view, body_names=["base_link", "other"])
    monkeypatch.setattr(privileged_runtime, "wheel_shape_layout",
        lambda robot, physics: NS(shapes_per_body=(8, 4), total_shapes=12))
    candidate.apply_to_robot(robot, None)
    torch.testing.assert_close(current[:, :8], torch.full((3, 8), OFFSET_M))
    torch.testing.assert_close(current[:, 8:] + candidate.terrain_offset, old[:, 8:] + .02)
    assert not rest.any() and calls == [1]
    assert candidate.receipt["native_applied"]
    assert candidate.receipt["articulation_count"] == 3
    assert candidate.receipt["other_pair_distance_max_error_m"] <= 4e-9
    with pytest.raises(RuntimeError, match="twice"):
        candidate.apply_to_robot(robot, None)
    assert calls == [1]


@pytest.mark.parametrize("mixed", [False, True])
def test_upper_link_repair_preserves_wheel_pairs_in_both_terrain_orders(monkeypatch, mixed):
    import warp as wp
    from wheeled_tasks.chassis import privileged_runtime

    stage, shared = mixed_terrain(monkeypatch) if mixed else (terrain()[0], None)
    candidate = ContactGenerationDistance(stage, shared.collision_paths if mixed else ["/ground"],
        terrain=shared, retain_flat_parent_pairs=mixed, version=UPPER_LINK_VERSION)
    names = ["R_link1", "base_link", "left_wheel", "L_link1", "right_wheel"]
    counts = (1, 8, 2, 1, 1)
    old = torch.arange(1, 40, dtype=torch.float32).reshape(3, 13) * .0001
    actual, rest = old.clone(), torch.zeros_like(old)

    def setter(values, ids):
        actual[wp.to_torch(ids).long()] = wp.to_torch(values)

    view = NS(count=3, get_contact_offsets=lambda: wp.from_torch(actual),
              get_rest_offsets=lambda: wp.from_torch(rest), set_contact_offsets=setter,
              prim_paths=[f"/World/envs/env_{i}/Robot/base_link" for i in (2, 0, 1)])
    monkeypatch.setattr(privileged_runtime, "wheel_shape_layout",
        lambda robot, physics: NS(shapes_per_body=counts, total_shapes=13))
    candidate.apply_to_robot(NS(root_view=view, body_names=names), None)
    repaired = [*range(9), 11]
    wheels = [9, 10, 12]
    torch.testing.assert_close(actual[:, repaired], torch.full((3, 10), OFFSET_M))
    references = torch.tensor([.042, .042, .02] if mixed else [.02] * 3)[:, None]
    torch.testing.assert_close(actual[:, wheels] + candidate.terrain_offset,
                               old[:, wheels] + references, rtol=0., atol=4e-9)
    assert not rest.any()
    assert candidate.receipt["version"] == UPPER_LINK_VERSION
    assert candidate.receipt["repaired_shape_ranges"] == {
        "base_link": [1, 9], "L_link1": [11, 12], "R_link1": [0, 1]}
    assert all(v == pytest.approx(2 * OFFSET_M)
               for v in candidate.receipt["repaired_pair_contact_distance_m"].values())


@pytest.mark.parametrize("mixed", [False, True])
def test_full_leg_pair_repair_preserves_tires_and_auxiliary_links(monkeypatch, mixed):
    import warp as wp
    from wheeled_tasks.chassis import privileged_runtime

    stage, shared = mixed_terrain(monkeypatch) if mixed else (terrain()[0], None)
    candidate = ContactGenerationDistance(stage, shared.collision_paths if mixed else ["/ground"],
        terrain=shared, retain_flat_parent_pairs=mixed, version=LEG_LINK_VERSION)
    names = ["R_link2", "R_link1", "base_link", "left_wheel", "L_link2", "L_link1", "right_wheel", "RR_link4"]
    counts = (1, 1, 8, 2, 1, 1, 1, 1)
    old = torch.arange(1, 49, dtype=torch.float32).reshape(3, 16) * .0001
    actual, rest = old.clone(), torch.zeros_like(old)

    def setter(values, ids):
        actual[wp.to_torch(ids).long()] = wp.to_torch(values)

    view = NS(count=3, get_contact_offsets=lambda: wp.from_torch(actual),
              get_rest_offsets=lambda: wp.from_torch(rest), set_contact_offsets=setter,
              prim_paths=[f"/World/envs/env_{i}/Robot/base_link" for i in (2, 0, 1)])
    monkeypatch.setattr(privileged_runtime, "wheel_shape_layout",
        lambda robot, physics: NS(shapes_per_body=counts, total_shapes=16))
    candidate.apply_to_robot(NS(root_view=view, body_names=names), None)
    repaired, preserved = [*range(10), 12, 13], [10, 11, 14, 15]
    torch.testing.assert_close(actual[:, repaired], torch.full((3, 12), OFFSET_M))
    references = torch.tensor([.042, .042, .02] if mixed else [.02] * 3)[:, None]
    torch.testing.assert_close(actual[:, preserved] + candidate.terrain_offset,
                               old[:, preserved] + references, rtol=0., atol=4e-9)
    assert not rest.any()
    assert candidate.receipt["repaired_shape_ranges"] == {
        "base_link": [2, 10], "L_link1": [13, 14], "R_link1": [1, 2],
        "L_link2": [12, 13], "R_link2": [0, 1]}


def test_upper_link_layout_mismatch_cannot_write_native_offsets(monkeypatch):
    from wheeled_tasks.chassis import privileged_runtime

    stage, _ = terrain()
    candidate = ContactGenerationDistance(stage, ["/ground"], version=UPPER_LINK_VERSION)
    robot = NS(body_names=["base_link", "L_link1", "R_link1"])
    monkeypatch.setattr(privileged_runtime, "wheel_shape_layout",
        lambda robot, physics: NS(shapes_per_body=(8, 2, 1), total_shapes=11))
    with pytest.raises(ValueError, match="one collision shape"):
        candidate.apply_to_robot(robot, None)


@pytest.mark.parametrize("version", [ROUGH_WHEEL_VERSION, ROUGH_WHEEL_MARGIN_VERSION, ROUGH_WHEEL_IMPACT_VERSION])
def test_rough_wheel_revision_retains_flat_pairs_in_native_clone_order(monkeypatch, version):
    import warp as wp
    from wheeled_tasks.chassis import privileged_runtime

    stage, shared = mixed_terrain(monkeypatch)
    candidate = ContactGenerationDistance(stage, shared.collision_paths, terrain=shared,
        retain_flat_parent_pairs=True, version=version)
    names = ["R_link2", "R_link1", "base_link", "L_link3", "L_link2", "L_link1", "R_link3", "RR_link4"]
    counts = (1, 1, 8, 2, 1, 1, 1, 1)
    old = torch.arange(1, 49, dtype=torch.float32).reshape(3, 16) * .0001
    actual, rest = old.clone(), torch.zeros_like(old)

    def setter(values, ids):
        actual[wp.to_torch(ids).long()] = wp.to_torch(values)

    view = NS(count=3, get_contact_offsets=lambda: wp.from_torch(actual),
        get_rest_offsets=lambda: wp.from_torch(rest), set_contact_offsets=setter,
        prim_paths=[f"/World/envs/env_{i}/Robot/base_link" for i in (2, 0, 1)])
    monkeypatch.setattr(privileged_runtime, "wheel_shape_layout", lambda robot, physics:
        NS(shapes_per_body=counts, total_shapes=16, body_ids=(3, 6), shape_ranges=((10, 12), (14, 15))))
    candidate.apply_to_robot(NS(root_view=view, body_names=names), None)
    torch.testing.assert_close(actual[:, [*range(10), 12, 13]], torch.full((3, 12), OFFSET_M))
    if version == ROUGH_WHEEL_IMPACT_VERSION:
        assert ROUGH_WHEEL_IMPACT_MARGIN_M == .02
        expected = torch.full((2, 3), .02)
    elif version == ROUGH_WHEEL_MARGIN_VERSION:
        expected = torch.full((2, 3), ROUGH_WHEEL_MARGIN_M)
    else:
        expected = old[:2, [10, 11, 14]]
    torch.testing.assert_close(actual[:2, [10, 11, 14]], expected)
    torch.testing.assert_close(actual[2, [10, 11, 14]] + candidate.terrain_offset,
                               old[2, [10, 11, 14]] + .02, rtol=0., atol=4e-9)
    torch.testing.assert_close(actual[:, 15] + candidate.terrain_offset,
                               old[:, 15] + old.new_tensor([.042, .042, .02]), rtol=0., atol=4e-9)
    assert not rest.any()
    assert candidate.receipt["rough_wheel_articulation_count"] == 2
    assert candidate.receipt["rough_wheel_shape_ranges"] == {"L_link3": [10, 12], "R_link3": [14, 15]}
    assert candidate.receipt["native_group_readback"][shared.collision_paths[1]]["count"] == 1
    if version == ROUGH_WHEEL_IMPACT_VERSION:
        assert candidate.receipt["rough_wheel_contact_offset_m"] == .02
        assert candidate.receipt["rough_wheel_scope"].startswith("20mm")


@pytest.mark.parametrize("version", [ROUGH_WHEEL_VERSION, ROUGH_WHEEL_MARGIN_VERSION, ROUGH_WHEEL_IMPACT_VERSION])
def test_rough_wheel_revision_keeps_the_parent_plane_tire_distances(monkeypatch, version):
    import warp as wp
    from wheeled_tasks.chassis import privileged_runtime

    stage, _ = terrain()
    candidate = ContactGenerationDistance(stage, ["/ground"], version=version)
    names = ["base_link", "L_link1", "R_link1", "L_link2", "R_link2", "L_link3", "R_link3", "RR_link4"]
    old = torch.full((1, 15), .0005)
    actual, rest = old.clone(), torch.zeros_like(old)

    def setter(values, ids):
        actual[wp.to_torch(ids).long()] = wp.to_torch(values)

    view = NS(count=1, get_contact_offsets=lambda: wp.from_torch(actual),
        get_rest_offsets=lambda: wp.from_torch(rest), set_contact_offsets=setter)
    monkeypatch.setattr(privileged_runtime, "wheel_shape_layout", lambda robot, physics:
        NS(shapes_per_body=(8, 1, 1, 1, 1, 1, 1, 1), total_shapes=15,
           body_ids=(5, 6), shape_ranges=((12, 13), (13, 14))))
    candidate.apply_to_robot(NS(root_view=view, body_names=names), None)
    torch.testing.assert_close(actual[:, :12], torch.full((1, 12), OFFSET_M))
    torch.testing.assert_close(actual[:, 12:] + candidate.terrain_offset,
                               old[:, 12:] + .02, rtol=0., atol=4e-9)
    assert candidate.receipt["rough_wheel_articulation_count"] == 0
    assert not rest.any()


@pytest.mark.parametrize("retain_parent", [False, True])
def test_mixed_terrain_preserves_assigned_pairs_in_native_clone_order(monkeypatch, retain_parent):
    import warp as wp
    from wheeled_tasks.chassis import privileged_runtime

    wp.init()
    stage, shared = mixed_terrain(monkeypatch)
    candidate = ContactGenerationDistance(stage, shared.collision_paths, terrain=shared,
                                          retain_flat_parent_pairs=retain_parent)
    for path in shared.collision_paths:
        assert not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.FilteredPairsAPI)
    groups = [f"/World/collisions/group{i}" for i in range(3)]
    rough, flat = "/World/collisions/global_group", "/World/collisions/flat_terrain_group"
    allowed = {path: {str(p) for p in stage.GetPrimAtPath(path).GetRelationship(
        "physics:filteredGroups").GetTargets()} for path in [*groups, rough, flat]}
    for i, group in enumerate(groups):
        assert allowed[group] == {group, flat if i == 1 else rough}
        for j, other in enumerate(groups):
            assert (other in allowed[group] or group in allowed[other]) == (i == j)
        for terrain, expected in ((rough, i != 1), (flat, i == 1)):
            assert (terrain in allowed[group]) == expected
            assert (group in allowed[terrain]) == expected
    for terrain, path in ((rough, shared.collision_paths[0]), (flat, shared.collision_paths[1])):
        assert [str(p) for p in stage.GetPrimAtPath(terrain).GetRelationship(
            "collection:colliders:includes").GetTargets()] == [path]
    old = torch.tensor([[.003] * 8 + [.002, .008], [.004] * 8 + [.003, .009],
                        [.005] * 8 + [.004, .01]])
    current, rest = old.clone(), torch.zeros_like(old)
    calls = []

    def setter(values, ids):
        current[wp.to_torch(ids).long()] = wp.to_torch(values)
        calls.append(1)

    view = NS(count=3, get_contact_offsets=lambda: wp.from_torch(current),
              get_rest_offsets=lambda: wp.from_torch(rest), set_contact_offsets=setter,
              prim_paths=[f"/World/envs/env_{i}/Robot/base_link" for i in (2, 0, 1)])
    robot = NS(root_view=view, body_names=["base_link", "other"])
    monkeypatch.setattr(privileged_runtime, "wheel_shape_layout",
        lambda robot, physics: NS(shapes_per_body=(8, 2), total_shapes=10))
    candidate.apply_to_robot(robot, None)
    references = torch.tensor([.042, .042, .02 if retain_parent else .0004905000096186996])[:, None]
    torch.testing.assert_close(current[:, 8:] + candidate.terrain_offset, old[:, 8:] + references,
                               rtol=0., atol=4e-9)
    torch.testing.assert_close(current[:, :8], torch.full((3, 8), OFFSET_M))
    assert not rest.any() and calls == [1]
    groups = candidate.receipt["native_group_readback"]
    assert groups[shared.collision_paths[0]]["count"] == 2
    assert groups[shared.collision_paths[1]]["count"] == 1
    if retain_parent:
        record = candidate.receipt["terrain"][1]
        assert record["original_contact_offset_m"] == pytest.approx(.0004905000096186996)
        assert record["preserved_pair_reference_offset_m"] == .02


def test_unknown_mesh_geometry_cannot_borrow_native_reference(monkeypatch):
    stage, shared = mixed_terrain(monkeypatch)
    patch = UsdGeom.Mesh(stage.GetPrimAtPath(shared.collision_paths[1]))
    points = np.asarray(patch.GetPointsAttr().Get()).copy()
    points[0, 0] += 1.
    patch.GetPointsAttr().Set(points.tolist())
    with pytest.raises(ValueError, match="no resolved evidence"):
        ContactGenerationDistance(stage, shared.collision_paths, terrain=shared)
    for path in shared.collision_paths:
        prim = stage.GetPrimAtPath(path)
        assert not prim.GetAttribute("physxCollision:contactOffset").HasAuthoredValueOpinion()
        assert not prim.GetRelationship("physics:filteredPairs").HasAuthoredTargets()


def test_mixed_origin_mismatch_rejected_before_authoring(monkeypatch):
    stage, shared = mixed_terrain(monkeypatch)
    shared.origins_m[1] = [0., 0., 0.]
    with pytest.raises(ValueError, match="outside its assigned"):
        ContactGenerationDistance(stage, shared.collision_paths, terrain=shared)
    assert all(not stage.GetPrimAtPath(path).GetAttribute(
        "physxCollision:contactOffset").HasAuthoredValueOpinion() for path in shared.collision_paths)


def test_mixed_existing_collision_isolation_must_match_before_rewrite(monkeypatch):
    stage, shared = mixed_terrain(monkeypatch)
    group = stage.GetPrimAtPath("/World/collisions/group1")
    group.GetRelationship("physics:filteredGroups").SetTargets(["/World/collisions/group0"])
    with pytest.raises(ValueError, match="per-clone collision isolation"):
        ContactGenerationDistance(stage, shared.collision_paths, terrain=shared)
    assert not stage.GetPrimAtPath("/World/collisions/flat_terrain_group")
    assert all(not stage.GetPrimAtPath(path).GetAttribute(
        "physxCollision:contactOffset").HasAuthoredValueOpinion() for path in shared.collision_paths)


def test_missing_clone_never_reaches_native_setter(monkeypatch):
    import warp as wp
    from wheeled_tasks.chassis import privileged_runtime

    stage, shared = mixed_terrain(monkeypatch)
    candidate = ContactGenerationDistance(stage, shared.collision_paths, terrain=shared)
    old = torch.full((3, 10), .001)
    rest = torch.zeros_like(old)
    view = NS(count=3, get_contact_offsets=lambda: wp.from_torch(old),
              get_rest_offsets=lambda: wp.from_torch(rest),
              prim_paths=["/World/envs/env_0/Robot"] * 3,
              set_contact_offsets=lambda *args: pytest.fail("Ambiguous clone offsets were written"))
    monkeypatch.setattr(privileged_runtime, "wheel_shape_layout",
        lambda robot, physics: NS(shapes_per_body=(8, 2), total_shapes=10))
    with pytest.raises(ValueError, match="complete native clone permutation"):
        candidate.apply_to_robot(NS(root_view=view, body_names=["base_link", "other"]), None)
