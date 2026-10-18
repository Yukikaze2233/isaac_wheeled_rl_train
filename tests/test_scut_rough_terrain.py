"""CPU checks: pinned source, actual SDK mesh functions, and shared boundaries."""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from wheeled_tasks.chassis import scut_rough_terrain as rough


def test_pinned_0730_geometry_has_no_0719_dash_high_platforms():
    terrain = rough.source_parameters()
    cfg = terrain["terrain_generator"]
    assert (cfg["size"], cfg["num_rows"], cfg["num_cols"]) == ([9., 9.], 10, 14)
    assert cfg["curriculum"] and cfg["difficulty_range"] == [0., 1.]
    assert cfg["border_width"] == 5. and terrain["max_init_terrain_level"] is None
    assert Counter(rough.column_names(terrain)) == {
        "tiny_step_rot": 2, "slope_for_rm_low": 1, "inv_slope_for_rm_low": 1,
        "stair_slope_for_rm_low": 1, "stair_slope_for_rm_mid": 2,
        "inv_stair_slope_for_rm_low": 1, "inv_stair_slope_for_rm_mid": 2,
        "plane_for_rm_rot": 3, "random_uniform_for_rm": 1}
    assert max(c.get("step_height_range", [0, 0])[1] for c in cfg["sub_terrains"].values()) == .04
    assert all("height_offset_range" not in c for c in cfg["sub_terrains"].values())
    assert rough._read_bound(rough.GRID_FILE, rough.GRID_SHA)


def test_source_tamper_is_rejected(tmp_path):
    target = tmp_path / rough.ENV_FILE
    target.parent.mkdir(parents=True)
    target.write_bytes((rough.REFERENCE / rough.ENV_FILE).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA differs"):
        rough.source_parameters(tmp_path)


def test_whole_terrain_boundary_and_world_height_query():
    calls = []
    mesh = object()

    def raycast(starts, directions, passed_mesh):
        calls.append((starts.clone(), directions.clone(), passed_mesh))
        result = starts.clone()
        result[..., 2] = starts[..., 0] * .1
        result[1, 2] = float("inf")  # interior miss is not silently a floor
        return result, None, None, None

    terrain = rough.SharedRough([], [], [], [], None, mesh, raycast,
                               [[-50., -68., -1.], [50., 68., 2.]], "sha", 66)
    xy = torch.tensor([[4.5, 9.], [0., 0.], [50.5, 0.]], dtype=torch.float64)
    z = terrain.height(xy)
    assert z.dtype == xy.dtype and z[0].item() == pytest.approx(.45)
    assert torch.isinf(z[1]) and z[2] == 0
    assert calls[0][2] is mesh
    assert torch.equal(calls[0][0][:, :2], xy.float())
    assert torch.all(calls[0][0][:, 2] == 3.)
    assert torch.all(calls[0][1] == torch.tensor([0., 0., -1.]))
    # Tile seams and 5m global border interior are traversable. Only the whole
    # terrain edge minus .5m times out; there is no per-origin/per-tile bound.
    points = torch.tensor([[4.5, 9., 3.], [45.1, 0., 0.], [49.5, 67.5, 0.],
                           [49.501, 0., 0.], [0., -67.501, 0.]])
    assert terrain.boundary(points).tolist() == [False, False, False, True, True]


def test_actual_sdk_and_scut_functions_generate_all_archived_mesh_types(monkeypatch):
    sdk = next((parent / ".deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab/isaaclab/terrains"
                for parent in Path(__file__).resolve().parents
                if (parent / ".deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab/isaaclab/terrains").is_dir()), None)
    if sdk is None:
        pytest.skip("CPU SDK source snapshot unavailable")
    monkeypatch.syspath_prepend(str(sdk.parent.parent))
    generator = rough.generator_config(66)
    generator.validate()
    assert generator.seed == 66 and len(generator.sub_terrains) == 9
    assert generator.size == [9., 9.]
    assert (generator.horizontal_scale, generator.vertical_scale, generator.slope_threshold) == (.1, .005, .5)
    np_state = np.random.get_state()
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
                if name == "tiny_step_rot":
                    assert mesh.bounds[1, 2] == pytest.approx(.01 + difficulty * .02)
                if name == "plane_for_rm_rot":
                    assert mesh.bounds[:, 2] == pytest.approx([0., 0.])
    finally:
        np.random.set_state(np_state)


def test_unreviewed_settings_fail_before_native_import():
    with pytest.raises(ValueError, match="unchanged scut0730"):
        rough.create_rough_terrain(None, {"type": "rough", "source": "scut0719"},
                                   1, "cpu", 66, None)


@pytest.mark.parametrize("level", [-1, 10, True, 0., "0"])
def test_invalid_initial_level_fails_before_native_import(level):
    with pytest.raises(ValueError, match="initial terrain level"):
        rough.create_rough_terrain(None, {"type": "rough", "source": "scut0730"},
                                   1, "cpu", 66, None, max_init_terrain_level=level)


def test_shared_terrain_routes_initial_level_only_to_rough(monkeypatch):
    from wheeled_tasks.chassis.formal_terrain import create_shared_terrain
    calls = []
    monkeypatch.setattr(rough, "create_rough_terrain", lambda *args, **kwargs: calls.append(kwargs))
    settings = {"type": "rough", "source": "scut0730"}
    create_shared_terrain(None, settings, 1, "cpu", 66, None)
    create_shared_terrain(None, settings, 1, "cpu", 66, None, max_init_terrain_level=0)
    assert calls == [{}, {"max_init_terrain_level": 0}]
    with pytest.raises(ValueError, match="only to rough"):
        create_shared_terrain(None, {"type": "plane", "source": "scut0719", "env_spacing_m": 4.},
                              1, "cpu", 66, None, max_init_terrain_level=0)
