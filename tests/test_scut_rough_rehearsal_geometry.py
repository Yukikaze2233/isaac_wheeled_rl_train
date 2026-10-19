"""CPU USD and actual Warp queries; no Kit/PhysX/GPU stepping or policy work."""
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from wheeled_tasks.chassis import scut_rough_terrain as rough
from wheeled_tasks.chassis.scut_rough_support import RoughBirthSupport, mesh_upper_bounds


@pytest.fixture
def terrains(monkeypatch):
    sdk = next((parent / ".deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab"
                for parent in Path(__file__).resolve().parents
                if (parent / ".deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab").is_dir()), None)
    if sdk is None:
        pytest.skip("CPU SDK mesh query source snapshot unavailable")
    monkeypatch.syspath_prepend(str(sdk))
    import isaaclab.terrains

    stages = []

    class Importer:
        def __init__(self, cfg):
            self.device = "cpu"
            self.env_origins = torch.tensor(np.random.uniform(-1., 1., (cfg.num_envs, 3)))
            self.terrain_levels = torch.arange(cfg.num_envs) % 10
            self.terrain_types = torch.arange(cfg.num_envs) % 14
            self.import_mesh("terrain", trimesh.Trimesh(
                vertices=[[-50., -68., .4], [50., -68., .4], [50., 68., .4], [-50., 68., .4]],
                faces=[[0, 1, 2], [0, 2, 3]], process=False))

        def import_mesh(self, name, mesh):
            stage = stages[-1]
            native = UsdGeom.Mesh.Define(stage, "/World/ground/terrain/mesh")
            native.CreatePointsAttr(np.asarray(mesh.vertices, dtype=np.float32).tolist())
            native.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
            native.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
            UsdPhysics.CollisionAPI.Apply(native.GetPrim())
            UsdShade.Material.Define(stage, "/World/ground/terrain/physicsMaterial")

    monkeypatch.setattr(isaaclab.terrains, "TerrainImporter", Importer)
    monkeypatch.setattr(rough, "generator_config", lambda seed: None)

    def create(assignments=None, *, collider=None):
        stage = Usd.Stage.CreateInMemory()
        stages.append(stage)
        return stage, rough.create_rough_terrain(
            stage, {"type": "rough", "source": "scut0730"}, 3, "cpu", 66, None,
            rehearsal_origins_by_env=assignments, rehearsal_collider=collider)

    return create


@pytest.mark.parametrize("mapping", [{3: (0., 300., 0.)}, {True: (0., 300., 0.)},
                                     {0: (0., 80., 0.)}, {0: (200., 300., 0.)},
                                     {0: (0., 300., 1.)}, {0: (0., float("nan"), 0.)}])
def test_bad_explicit_origins_fail_before_native_import(mapping):
    with pytest.raises(ValueError, match="Rehearsal"):
        rough.create_rough_terrain(None, {"type": "rough", "source": "scut0730"},
                                  3, "cpu", 66, None, rehearsal_origins_by_env=mapping)


def test_opt_in_preserves_rough_mesh_assignments_rng_and_reuses_material(terrains):
    np.random.seed(713)
    before = np.random.get_state()
    default_stage, default = terrains()
    empty_stage, empty = terrains({})
    stage, mixed = terrains({1: (-4., 302., 0.)})
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert default.summary() == empty.summary()
    assert not default_stage.GetPrimAtPath(rough.REHEARSAL_PRIM_PATH)
    assert not empty_stage.GetPrimAtPath(rough.REHEARSAL_PRIM_PATH)
    assert default.origins_m == empty.origins_m
    assert mixed.origins_m[0] == default.origins_m[0] and mixed.origins_m[2] == default.origins_m[2]
    assert mixed.origins_m[1] == [-4., 302., 0.]
    torch.testing.assert_close(mixed.importer.env_origins, default.importer.env_origins)
    torch.testing.assert_close(mixed.importer.terrain_levels, default.importer.terrain_levels)
    torch.testing.assert_close(mixed.importer.terrain_types, default.importer.terrain_types)
    assert mixed.bounds_m == default.bounds_m
    assert mixed.source_rough_mesh_sha256 == default.mesh_sha256
    assert mixed.mesh_sha256 != default.mesh_sha256
    for attr in ("points", "faceVertexIndices", "faceVertexCounts"):
        assert (stage.GetPrimAtPath("/World/ground/terrain/mesh").GetAttribute(attr).Get()
                == default_stage.GetPrimAtPath("/World/ground/terrain/mesh").GetAttribute(attr).Get())
    patch = stage.GetPrimAtPath(rough.REHEARSAL_PRIM_PATH)
    assert patch.GetTypeName() == "Mesh" and patch.HasAPI(UsdPhysics.CollisionAPI)
    assert not patch.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MeshCollisionAPI(patch).GetApproximationAttr().Get() == "none"
    material, _ = UsdShade.MaterialBindingAPI(patch).ComputeBoundMaterial("physics")
    assert str(material.GetPath()) == "/World/ground/terrain/physicsMaterial"
    native_points = np.asarray(UsdGeom.Mesh(patch).GetPointsAttr().Get(), dtype=np.float32)
    native_faces = np.asarray(UsdGeom.Mesh(patch).GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    assert mixed.rehearsal_mesh_sha256 == hashlib.sha256(native_points.tobytes() + native_faces.tobytes()).hexdigest()
    # The actual Warp query retains original vertices/faces and appends exactly
    # these native points/faces; no infinite fallback plane is introduced.
    query_points = mixed.warp_mesh.points.numpy()
    query_faces = mixed.warp_mesh.indices.numpy().reshape(-1, 3)
    np.testing.assert_array_equal(query_points[-4:], native_points)
    np.testing.assert_array_equal(query_faces[-2:] - len(default.warp_mesh.points.numpy()), native_faces.reshape(-1, 3))
    assert mixed.collision_paths[-1] == mixed.global_prim_paths[-1] == rough.REHEARSAL_PRIM_PATH
    assert mixed.summary()["boundary_scope"] == "union_of_rough_and_separate_finite_rehearsal_plane"
    assert mixed.summary()["rehearsal_plane"]["size_m"] == [400., 400.]
    np.testing.assert_array_equal(native_points.min(0), [-200., 100., 0.])
    np.testing.assert_array_equal(native_points.max(0), [200., 500., 0.])


def test_two_actual_regions_have_support_but_gap_has_none(terrains):
    _, mixed = terrains({1: (0., 300., 0.)})
    xyz = torch.tensor([[4.5, 9., 1.], [-180., 120., 1.], [180., 480., 1.],
                        [0., 80., 1.], [199.6, 300., 1.], [0., 99.9, 1.]])
    assert mixed.boundary(xyz).tolist() == [False, False, False, True, True, True]
    direction = torch.zeros_like(xyz)
    direction[:, 2] = -1.
    hits = mixed.raycast(xyz, direction, mixed.warp_mesh)[0]
    torch.testing.assert_close(hits[:3, 2], torch.tensor([.4, 0., 0.]))
    assert torch.isinf(hits[3]).all() and torch.isinf(hits[5]).all()
    torch.testing.assert_close(mixed.height(xyz[:3, :2]), hits[:3, 2])
    lower = xyz[:4, :2].double() - .1
    upper = lower + .2
    support = mesh_upper_bounds(mixed.warp_mesh, mixed.query_bounds_m, lower, upper)
    torch.testing.assert_close(support[:3], torch.tensor([.4, 0., 0.], dtype=torch.float64), atol=1e-8, rtol=0.)
    assert support[3].isneginf()

    transform = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    transform[:, 2, 3] = -.2
    placement = NS(device=torch.device("cpu"), dtype=torch.float64, clearance_m=.003,
        kinematics=NS(frames=lambda q: {"wheel": transform.expand(len(q), -1, -1)}),
        _shapes=[("wheel", 0, "cylinder", torch.eye(4, dtype=torch.float64), (.1, .04))])
    pending = NS(joint_position=torch.zeros(2, 18, dtype=torch.float64),
        root_pose=torch.tensor([[0., 0., .223, 0., 0., 0., 1.],
                                [-126., 174., .223, 0., 0., 0., 1.]], dtype=torch.float64),
        placement_raise_m=torch.full((2,), .223, dtype=torch.float64),
        minimum_shape_clearance_m=torch.full((2,), .003, dtype=torch.float64),
        requests=NS(source_root_rpy_rad=torch.zeros(2, 3)))
    birth = RoughBirthSupport(placement, mixed)
    birth.adjust(pending)
    torch.testing.assert_close(pending.root_pose[:, 2], torch.tensor([.623, .223], dtype=torch.float64), atol=1e-8, rtol=0.)
    assert birth.summary()["query_geometry"] == "rough_and_finite_rehearsal_plane_triangle_union"
    pending.root_pose[1, 1] = 80.
    with pytest.raises(RuntimeError, match="no finite mesh support"):
        birth.adjust(pending)


def test_native_plane_retains_query_support_material_and_original_rough(terrains):
    _, original = terrains({1: (0., 300., 0.)})
    stage, native = terrains({1: (0., 300., 0.)}, collider=rough.NATIVE_REHEARSAL_PLANE)
    patch = stage.GetPrimAtPath(rough.REHEARSAL_PRIM_PATH)
    assert patch.IsA(UsdGeom.Plane) and patch.HasAPI(UsdPhysics.CollisionAPI)
    assert not patch.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdGeom.Plane(patch).GetAxisAttr().Get() == "Z"
    matrix = UsdGeom.Xformable(patch).GetLocalTransformation()
    assert list(matrix.ExtractTranslation()) == [0., 300., 0.]
    material, _ = UsdShade.MaterialBindingAPI(patch).ComputeBoundMaterial("physics")
    assert str(material.GetPath()) == "/World/ground/terrain/physicsMaterial"
    assert native.origins_m == original.origins_m
    assert native.source_rough_mesh_sha256 == original.source_rough_mesh_sha256
    assert native.mesh_sha256 == original.mesh_sha256  # Query mesh, not Plane geometry.
    np.testing.assert_array_equal(native.warp_mesh.points.numpy(), original.warp_mesh.points.numpy())
    xy = torch.tensor([[0., 300.], [190., 490.], [0., 0.]])
    torch.testing.assert_close(native.height(xy), original.height(xy))
    receipt = native.summary()["rehearsal_plane"]
    assert receipt["native_geometry"] == "Plane" and receipt["native_infinite"] is True
    assert receipt["query_geometry"] == "bounded_z0_triangles_equivalent_inside_rehearsal_region"


def test_native_plane_authoring_uses_original_material_without_mesh_cooking():
    stage = Usd.Stage.CreateInMemory()
    material_path = "/World/ground/terrain/physicsMaterial"
    UsdShade.Material.Define(stage, material_path)
    points, faces = rough._rehearsal_plane_mesh()
    path = rough._author_rehearsal_plane(stage, points, faces, material_path, native_plane=True)
    prim = stage.GetPrimAtPath(path)
    assert prim.IsA(UsdGeom.Plane) and not prim.IsA(UsdGeom.Mesh)
    assert prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert prim.GetAttribute("physics:collisionEnabled").Get() is True
    assert list(UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()) == [0., 300., 0.]
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    assert str(material.GetPath()) == material_path


@pytest.mark.parametrize("origins,collider", [(None, rough.NATIVE_REHEARSAL_PLANE),
                                            ({0: [0., 300., 0.]}, "unreviewed")])
def test_native_plane_requires_explicit_assignment_before_sdk_import(origins, collider):
    with pytest.raises(ValueError, match="Native rehearsal collider"):
        rough.create_rough_terrain(None, {"type": "rough", "source": "scut0730"},
            3, "cpu", 66, None, rehearsal_origins_by_env=origins, rehearsal_collider=collider)
