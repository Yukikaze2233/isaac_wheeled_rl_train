"""CPU geometry checks, without a native scene or training process."""
from collections import Counter
from pathlib import Path
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from wheeled_tasks.chassis import final_rough_terrain as final


@pytest.fixture
def sdk(monkeypatch):
    location = next((parent / ".deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab"
                     for parent in Path(__file__).resolve().parents
                     if (parent / ".deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab/isaaclab/terrains").is_dir()), None)
    if location is None:
        pytest.skip("CPU SDK source snapshot unavailable")
    monkeypatch.syspath_prepend(str(location))


def test_columns_include_the_two_big_stairs_and_cliff_as_final_active_keys():
    names = final.column_names()
    assert names == ["low_speed_stair_for_rm", "tiny_step", "tiny_step", "slope_for_rm_high",
        "inv_slope_for_rm_low", "high_speed_stair_for_rm", "stair_slope_for_rm_high",
        "inv_stair_slope_for_rm_low", "inv_stair_slope_for_rm_high", "inv_stair_slope_for_rm_high",
        "plane_for_rm", "random_uniform_for_rm", "cliff_inv_stair_slope_short_for_rm"]
    assert len(names) == 13 and len(Counter(names)) == 11


def test_all_real_sdk_and_frozen_functions_generate_finite_final_meshes(sdk):
    generator = final.generator_config(66)
    generator.validate()
    assert (generator.seed, generator.num_rows, generator.num_cols) == (66, 10, 13)
    assert generator.size == [9., 9.] and generator.border_width == 10.
    assert generator.curriculum is True and generator.difficulty_range == [0., 1.]
    assert len(generator.sub_terrains) == 11
    state = np.random.get_state()
    try:
        np.random.seed(66)
        for name, cfg in generator.sub_terrains.items():
            for difficulty in (.05, .95):
                meshes, origin = cfg.function(difficulty, cfg)
                mesh = trimesh.util.concatenate(meshes)
                assert np.isfinite(mesh.vertices).all(), name
                assert np.isfinite(origin).all(), name
                assert len(mesh.faces) > 0, name
                assert mesh.bounds[0, :2] == pytest.approx([0., 0.]), name
                assert mesh.bounds[1, :2] == pytest.approx([9., 9.]), name
                assert cfg.size == [9., 9.], name
                if name == "tiny_step":
                    assert mesh.bounds[1, 2] == pytest.approx(.02 + difficulty * .04)
                    assert cfg.force_even_counts and cfg.randomize_bar_count_difficulty
                    assert cfg.randomize_bar_width_difficulty
                if name == "high_speed_stair_for_rm" and difficulty == .95:
                    assert mesh.bounds[1, 2] == pytest.approx(.77)
                if name == "cliff_inv_stair_slope_short_for_rm":
                    assert mesh.bounds[1, 2] == pytest.approx(round((.3 + .1 * difficulty) / .005) * .005)
                    assert cfg.height_offset_curriculum_scale_by_difficulty is False
                    assert "_v7_final_terrain_height_field" in cfg.function.__module__
    finally:
        np.random.set_state(state)


def _shared(*, rehearsal=False, max_level=None):
    calls = []
    mesh = object()

    def raycast(starts, dirs, passed_mesh):
        calls.append((starts.clone(), dirs.clone(), passed_mesh))
        hits = starts.clone()
        hits[..., 2] = .3
        hits[1, 2] = float("inf")
        return hits, None, None, None

    importer = SimpleNamespace(terrain_levels=torch.tensor([0, 9]), terrain_types=torch.tensor([0, 12]))
    kwargs = {"rehearsal_origins_by_env": {1: [-190., 110., 0.]},
              "rehearsal_bounds_m": [[-200., 100., 0.], [200., 500., 0.]],
              "source_rough_mesh_sha256": "rough-sha", "rehearsal_mesh_sha256": "plane-sha"} if rehearsal else {}
    shared = final.SharedFinalRough([], ["/World/ground/terrain"], ["/World/ground"],
        ["/World/ground/terrain/physicsMaterial"], importer, mesh, raycast,
        [[-55., -68.5, -.5], [55., 68.5, 1.]], "actual-sha", 66,
        max_init_terrain_level=max_level, **kwargs)
    return shared, calls, mesh


def test_actual_mesh_height_keeps_world_z_interior_misses_and_input_dtype():
    shared, calls, mesh = _shared()
    xy = torch.tensor([[4.5, 9.], [0., 0.], [55.5, 0.]], dtype=torch.float64)
    height = shared.height(xy)
    assert height.dtype == xy.dtype
    assert height[0] == pytest.approx(.3) and torch.isinf(height[1]) and height[2] == 0
    assert calls[0][2] is mesh
    assert torch.equal(calls[0][0][:, :2], xy.float())
    assert torch.all(calls[0][0][:, 2] == 2.)
    assert torch.all(calls[0][1] == torch.tensor([0., 0., -1.]))


def test_boundary_uses_final_whole_mesh_margin_and_rehearsal_union():
    shared, _, _ = _shared()
    points = torch.tensor([[4.5, 9., 0.], [54.5, 68., 0.], [54.501, 0., 0.], [0., -68.001, 0.]])
    assert shared.boundary(points).tolist() == [False, False, True, True]
    shared, _, _ = _shared(rehearsal=True)
    points = torch.tensor([[0., 0., 0.], [-190., 110., 0.], [-199.5, 110., 0.], [-199.501, 110., 0.], [100., 80., 0.]])
    assert shared.boundary(points).tolist() == [False, False, False, True, True]
    assert shared.query_bounds_m == [[-200., -68.5, -.5], [200., 500., 1.]]


@pytest.mark.parametrize("rehearsal", [False, True])
def test_summary_reports_actual_13_columns_and_no_behavior_or_historical_sdk_claim(rehearsal):
    shared, _, _ = _shared(rehearsal=rehearsal)
    summary = shared.summary()
    assert summary["source"] == final.FINAL_TERRAIN_SOURCE
    assert (summary["row_count"], summary["column_count"], summary["terrain_key_count"]) == (10, 13, 11)
    assert summary["border_width_m"] == 10. and summary["boundary_margin_m"] == .5
    assert summary["initial_level_counts"] == [1] + [0] * 8 + [1]
    assert summary["initial_type_counts"] == [1] + [0] * 11 + [1]
    assert summary["historical_sdk_triangle_identity_verified"] is False
    assert summary["training_terrain_progression"] is summary["native_contact_qualified"] is False
    assert summary["height_offset_curriculum_enabled"] is False
    assert summary["skill_state_machines_enabled"] is False
    if rehearsal:
        assert summary["boundary_scope"] == "union_of_rough_and_separate_finite_rehearsal_plane"
        assert summary["rough_mesh_sha256"] == "rough-sha"
        assert summary["rehearsal_plane"]["mesh_sha256"] == "plane-sha"


@pytest.mark.parametrize("settings,num_envs,seed,level", [
    ({"type": "rough", "source": "scut0730"}, 2, 66, None),
    ({"type": "rough", "source": final.FINAL_TERRAIN_SOURCE, "unknown": True}, 2, 66, None),
    ({"type": "rough", "source": final.FINAL_TERRAIN_SOURCE}, 0, 66, None),
    ({"type": "rough", "source": final.FINAL_TERRAIN_SOURCE}, True, 66, None),
    ({"type": "rough", "source": final.FINAL_TERRAIN_SOURCE}, 2, True, None),
    ({"type": "rough", "source": final.FINAL_TERRAIN_SOURCE}, 2, -1, None),
    ({"type": "rough", "source": final.FINAL_TERRAIN_SOURCE}, 2, 66, 10),
    ({"type": "rough", "source": final.FINAL_TERRAIN_SOURCE}, 2, 66, True),
])
def test_invalid_factory_settings_reject_before_any_native_import(settings, num_envs, seed, level):
    with pytest.raises(ValueError):
        final.create_final_rough_terrain(None, settings, num_envs, "cpu", seed, None,
                                        max_init_terrain_level=level)


@pytest.mark.parametrize("seed", [None, True, -1, 2**32, 1.5])
def test_invalid_seed_rejects_before_sdk(seed):
    with pytest.raises(ValueError, match="32-bit integer"):
        final.generator_config(seed)


@pytest.mark.parametrize("rehearsal,level", [(False, None), (True, 0)])
def test_factory_retains_exact_collision_triangles_and_restores_numpy_rng(monkeypatch, rehearsal, level):
    """Exercise adapter ownership with CPU doubles; no native scene is opened."""
    events, converted = [], []
    points = np.array([[-55., -68.5, 0.], [55., -68.5, 0.], [55., 68.5, .2], [-55., 68.5, .2]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)

    class Importer:
        def __init__(self, cfg):
            events.append(("cfg", cfg))
            self.device = "cpu"
            self.env_origins = torch.tensor([[0., 0., .2], [1., 1., .2]])
            self.terrain_levels, self.terrain_types = torch.tensor([0, 1]), torch.tensor([0, 12])
            np.random.rand(5)
            self.import_mesh("terrain", mesh)

        def import_mesh(self, name, passed_mesh):
            events.append(("collision_mesh", passed_mesh))

    class Prim:
        def GetPath(self):
            return "/World/ground/terrain"

        def GetTypeName(self):
            return "Mesh"

        def HasAPI(self, api):
            return True

    class MeshAPI:
        def __init__(self, prim):
            pass

        def CreateSubdivisionSchemeAttr(self, value):
            events.append(("subdivision", value))

    class CollisionAPI:
        @staticmethod
        def Apply(prim):
            return CollisionAPI()

        def CreateApproximationAttr(self, value):
            events.append(("approximation", value))

    def convert(vertices, triangles, *, device):
        converted.append((vertices.copy(), triangles.copy(), device))
        return "actual-triangle-query"

    stage = SimpleNamespace(GetPrimAtPath=lambda path: Prim())
    monkeypatch.setitem(sys.modules, "isaaclab.terrains", SimpleNamespace(TerrainImporter=Importer,
        TerrainImporterCfg=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.warp", SimpleNamespace(
        convert_to_warp_mesh=convert, raycast_mesh=lambda *args: None))
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        Usd=SimpleNamespace(PrimRange=lambda prim: [Prim()]),
        UsdGeom=SimpleNamespace(Mesh=MeshAPI),
        UsdPhysics=SimpleNamespace(CollisionAPI=object(), MeshCollisionAPI=CollisionAPI)))
    monkeypatch.setattr(final, "generator_config", lambda seed: ("final-source-generator", seed))
    monkeypatch.setattr(final, "_author_rehearsal_plane", lambda stage, points, faces, material:
        events.append(("rehearsal_material", material)) or final.REHEARSAL_PRIM_PATH)
    np.random.seed(2468)
    expected_rng = np.random.get_state()
    material = object()
    shared = final.create_final_rough_terrain(stage,
        {"type": "rough", "source": final.FINAL_TERRAIN_SOURCE}, 2, "cpu", 66, material,
        rehearsal_origins_by_env={1: [-190., 110., 0.]} if rehearsal else None,
        max_init_terrain_level=level)
    actual_rng = np.random.get_state()
    assert actual_rng[0] == expected_rng[0] and np.array_equal(actual_rng[1], expected_rng[1])
    assert actual_rng[2:] == expected_rng[2:]
    cfg = events[0][1]
    assert cfg.terrain_generator == ("final-source-generator", 66)
    assert cfg.collision_group == -1 and cfg.use_terrain_origins and cfg.env_spacing == 4.
    assert cfg.max_init_terrain_level == level and cfg.physics_material is material
    assert events[1] == ("collision_mesh", mesh)
    assert ("subdivision", "none") in events and ("approximation", "none") in events
    query_points, query_faces, device = converted[0]
    assert device == "cpu" and query_points.dtype == np.float32 and query_faces.dtype == np.int32
    assert np.array_equal(query_points[:len(points)], points.astype(np.float32))
    assert np.array_equal(query_faces[:len(faces)], faces.astype(np.int32))
    assert shared.warp_mesh == "actual-triangle-query"
    assert shared.mesh_sha256 == hashlib.sha256(query_points.tobytes() + query_faces.tobytes()).hexdigest()
    assert shared.bounds_m == mesh.bounds.tolist()
    assert shared.max_init_terrain_level == level
    if rehearsal:
        patch_points, patch_faces = final._rehearsal_plane_mesh()
        assert np.array_equal(query_points[len(points):], patch_points)
        assert np.array_equal(query_faces[len(faces):], patch_faces + len(points))
        assert shared.origins_m[1] == [-190., 110., 0.]
        assert shared.source_rough_mesh_sha256 == hashlib.sha256(points.astype(np.float32).tobytes() + faces.astype(np.int32).tobytes()).hexdigest()
        assert ("rehearsal_material", "/World/ground/terrain/physicsMaterial") in events
    else:
        assert len(query_points) == len(points) and len(query_faces) == len(faces)
