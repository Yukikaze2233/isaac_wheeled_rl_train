"""Reproduce population truncation using the SDK's contiguous column rule."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import reorder_fixed_flat_rehearsal
from wheeled_tasks.chassis.simulation.terrain.rough_terrain import SharedRough, column_names, source_parameters
from wheeled_tasks.chassis.application.configuration.small_rough_course import ROLES, allocate
from wheeled_tasks.chassis.tasks.episodes import choose_scene_groups


ROOT = Path(__file__).resolve().parents[3]


def mixed_geometry():
    config = json.loads((ROOT / "configs/v7_formal_20261012/mixed_tracking_repair_063.json").read_text())
    groups = [name for name, _ in choose_scene_groups(config["scene_groups"], 4096)]
    spec = allocate(config, ROOT, 4096, groups)
    tiles = torch.zeros(10, 14, 3)
    tiles[:, :, 0] = (torch.arange(10) * 9. - 40.5)[:, None]
    tiles[:, :, 1] = (torch.arange(14) * 9. - 58.5)[None, :]
    # This is the imported SDK assignment, rather than round-robin columns.
    types = torch.div(torch.arange(4096), 4096 / 14, rounding_mode="floor").long()
    levels = torch.zeros(4096, dtype=torch.long)
    importer = SimpleNamespace(device="cpu", terrain_origins=tiles, terrain_types=types,
                               terrain_levels=levels, env_origins=tiles[levels, types].clone())
    origins = importer.env_origins.tolist()
    for lane, origin in spec["origins_by_env"].items():
        origins[lane] = list(origin)
    terrain = SharedRough(origins, [], [], [], importer, None, None,
                          [[-50., -68., 0.], [50., 68., 0.]], "unchanged", 290929,
                          rehearsal_origins_by_env=spec["origins_by_env"],
                          rehearsal_bounds_m=[[-200., 100., 0.], [200., 500., 0.]],
                          max_init_terrain_level=0)
    return terrain, spec


def test_4096_population_covers_all_columns_without_changing_flat_or_rows():
    terrain, spec = mixed_geometry()
    rough_ids = [i for i, role in enumerate(spec["population_index"])
                 if role == ROLES.index("source_rough")]
    names = column_names(source_parameters())
    previous = Counter(names[int(terrain.importer.terrain_types[i])] for i in rough_ids)
    assert sorted(previous.values()) == [366, 658]
    assert len(previous) == 2
    old_flat = {i: terrain.origins_m[i][:] for i in spec["origins_by_env"]}
    old_rows = terrain.importer.terrain_levels.clone()
    old_tiles = terrain.importer.terrain_origins.clone()
    rng = torch.get_rng_state().clone()

    terrain.distribute_mixed_rough_origins()

    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(terrain.importer.terrain_levels, old_rows)
    assert torch.equal(terrain.importer.terrain_origins, old_tiles)
    assert all(terrain.origins_m[i] == origin for i, origin in old_flat.items())
    after = terrain.mixed_rough_assignment
    assert after["rough_lane_count"] == 1024
    assert sum(after["column_counts_after"]) == 1024
    assert min(after["column_counts_after"]) == 73 and max(after["column_counts_after"]) == 74
    assert after["row_counts"] == [1024] + [0] * 9
    assert len({names[int(terrain.importer.terrain_types[i])] for i in rough_ids}) == 9
    actual = torch.tensor([terrain.origins_m[i] for i in rough_ids])
    torch.testing.assert_close(actual, terrain.importer.env_origins[rough_ids])
    torch.testing.assert_close(actual, old_tiles[old_rows[rough_ids], terrain.importer.terrain_types[rough_ids]])

    # Native views use lexical prim ordering, not pre-clone integer ordering.
    order = sorted(range(4096), key=lambda i: f"/World/envs/env_{i}/Robot")
    reordered = reorder_fixed_flat_rehearsal(spec, order)
    actual = torch.tensor([terrain.origins_m[i] for i in order])
    rough_mask = torch.tensor(reordered["population_index"]) == ROLES.index("source_rough")
    resolved_columns = ((actual[rough_mask, 1] + 63.) / 9.).floor().long()
    assert torch.bincount(resolved_columns, minlength=14).tolist() == after["column_counts_after"]
    assert (actual[~rough_mask, 1] > 100.).all()
    summary = terrain.summary()
    assert summary["mixed_rough_assignment"] == after
    assert summary["training_terrain_progression"] is False


def test_reallocation_requires_an_explicit_mixed_population_and_runs_once():
    terrain, _ = mixed_geometry()
    terrain.distribute_mixed_rough_origins()
    with pytest.raises(ValueError, match="one pre-clone allocation"):
        terrain.distribute_mixed_rough_origins()
    terrain, _ = mixed_geometry()
    terrain.rehearsal_origins_by_env = {}
    with pytest.raises(ValueError, match="one pre-clone allocation"):
        terrain.distribute_mixed_rough_origins()
