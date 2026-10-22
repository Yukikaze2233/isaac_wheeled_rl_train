"""CPU checks for geometric exposure, terminal ownership and missing intervals."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.application.evaluation.rough_evaluation import RoughTerrainMeasurements


def fixture(initial_xy):
    n = len(initial_xy)
    grid = torch.zeros(3, 2, 3)
    grid[:, :, 0] = torch.tensor([4.5, 13.5, 22.5])[:, None]
    grid[:, :, 1] = torch.tensor([4.5, 13.5])[None, :]
    summary = {"type": "rough", "column_names": ["rough_a", "rough_b"],
        "tile_size_m": [9., 9.], "mesh_bounds_world_m": [[-5., -5., -1.], [32., 23., 1.]],
        "source": "CPU fixture", "source_commit": "pinned fixture", "source_env_sha256": "fixture",
        "mesh_sha256": "fixture mesh", "geometry_seed": 66}
    pose = torch.zeros(n, 7)
    pose[:, :2] = torch.tensor(initial_xy)
    env = NS(num_envs=n, shared_terrain=NS(summary=lambda: deepcopy(summary), importer=NS(terrain_origins=grid)),
        flat_episode=NS(episode_id=torch.zeros(n, dtype=torch.long),
            commands=NS(summary=lambda: {"terrain_profiles": {"rough_a": [.2, .4], "rough_b": [.2, .4]}})),
        episode_length_buf=torch.zeros(n, dtype=torch.long), robot=NS(data=NS(root_link_pose_w=NS(torch=pose))))
    return RoughTerrainMeasurements.from_env(env, .02)


def diagnostics(xy, *, episode=0, tick=1, valid=True, done=False, boundary=False, owners=None):
    n = len(xy)
    scalar = lambda value, dtype: torch.full((n,), value, dtype=dtype)
    position = torch.tensor([[x, y, .8] for x, y in xy])
    return {"episode_id": scalar(episode, torch.long), "episode_tick": scalar(tick, torch.long),
        "measurement_valid_mask": scalar(valid, torch.bool), "root_link_position_world": position,
        "root_yaw_rad": scalar(0., torch.float32),
        "effective_command": torch.tensor([[1., 0., .4]]).repeat(n, 1),
        "root_COM_velocity_body": torch.tensor([[-.5, 0., 0.]]).repeat(n, 1),
        "angular_velocity_body": torch.zeros(n, 3), "height": scalar(.5, torch.float32),
        "root_height_world": scalar(.8, torch.float32), "reward_ground_height_world": scalar(.3, torch.float32),
        "rough_base_scan_valid_hit_count": scalar(9, torch.long),
        "done": scalar(done, torch.bool), "terminated": scalar(done and not boundary, torch.bool),
        "timeout": scalar(boundary, torch.bool), "boundary": scalar(boundary, torch.bool),
        "command_override_active": scalar(False, torch.bool),
        "source_flat_episode": {"terrain_key_ids": torch.tensor(owners if owners is not None else [0] * n)},
        "reasons": {"source_attitude_or_contact": scalar(done and not boundary, torch.bool),
                    "boundary": scalar(boundary, torch.bool)}}


def test_world_cell_exposure_is_independent_of_command_owner_and_prepare_is_pure():
    measured = fixture([[8.9, 4.5], [4.5, 8.9]])
    before = measured.report()
    data = diagnostics([[9.1, 4.5], [4.5, 9.1]], owners=[1, 0])
    data["root_yaw_rad"][1] = torch.pi / 2
    data["effective_command"][1, 0] = -1.
    pending = measured.prepare(data, [0, 1])
    assert measured.report() == before
    data["root_link_position_world"].zero_()  # queued readings own their CPU values
    measured.accept(pending)
    result = measured.report()
    a, b = (result["by_terrain_type"][name] for name in ("rough_a", "rough_b"))
    assert a["command_owner_frames"] == {"rough_b": 1}
    assert b["command_owner_frames"] == {"rough_a": 1}
    assert a["cross_tile_observations"] == b["cross_tile_observations"] == 1
    assert result["by_row"]["1"]["valid_frames"] == 1
    assert len(result["unobserved_type_rows"]) == 4
    assert {"terrain_type": "rough_a", "row": 1} not in result["unobserved_type_rows"]
    assert a["valid_lane_seconds"] == .02
    assert a["root_link_planar_path_m"] == pytest.approx(.2, abs=1e-6)
    assert a["signed_command_direction_progress_m"] == pytest.approx(.2, abs=1e-6)
    assert b["signed_command_direction_progress_m"] == pytest.approx(-.2, abs=1e-6)
    assert a["tracking"]["vx"]["signed_error_mean"] == -1.5
    assert a["tracking"]["height"]["rmse"] == pytest.approx(.1)


def test_terminal_included_but_new_episode_invalid_gap_and_finished_lane_do_not_add_paths():
    measured = fixture([[1., 1.], [1., 1.]])
    measured.accept(measured.prepare(diagnostics([[1.2, 1.], [10., 10.]], done=True), [0]))
    measured.accept(measured.prepare(diagnostics([[20., 1.], [20., 20.]], episode=1), [0]))
    measured.accept(measured.prepare(diagnostics([[20.1, 1.], [21., 20.]], episode=1, tick=2), [0]))
    measured.accept(measured.prepare(diagnostics([[999., 1.], [22., 20.]], episode=1, tick=3, valid=False), [0]))
    measured.accept(measured.prepare(diagnostics([[25., 1.], [23., 20.]], episode=1, tick=4), [0]))
    result = measured.report()
    a = result["by_terrain_type"]["rough_a"]
    assert a["accepted_frames"] == 4
    assert a["root_link_planar_path_m"] == pytest.approx(.3, abs=2e-6)
    assert a["cross_tile_observations"] == 0
    assert a["terminated_episodes"] == 1
    assert a["termination_reasons"] == {"source_attitude_or_contact": 1}
    assert result["by_terrain_type"]["outside_mesh"]["invalid_frames"] == 1
    assert result["by_terrain_type"]["rough_b"]["accepted_frames"] == 0  # quota-complete lane ignored


def test_border_outside_missing_scan_and_stationary_horizon_are_measurements_only():
    measured = fixture([[-1., 4.5], [-5.1, 4.5], [4.5, 4.5]])
    data = diagnostics([[-1., 4.5], [-5.1, 4.5], [4.5, 4.5]], done=True, boundary=True)
    data["effective_command"][:, 0] = 0.
    data["rough_base_scan_valid_hit_count"][1] = 0
    data["timeout"][2], data["boundary"][2], data["reasons"]["boundary"][2] = True, False, False
    measured.accept(measured.prepare(data, [0, 1, 2]))
    result = measured.report()
    assert result["by_terrain_type"]["border"]["boundary_episodes"] == 1
    outside = result["by_terrain_type"]["outside_mesh"]
    assert outside["scan_hit_count_frames"]["0"] == 1
    stationary = result["by_terrain_type"]["rough_a"]
    assert stationary["timeout_episodes"] == 1 and stationary["boundary_episodes"] == 0
    assert stationary["root_link_planar_path_m"] == 0.
    assert stationary["by_command_kind"]["standing"]["valid_frames"] == 1
    assert result["assessment"] == "measurement_only" and "passed" not in stationary
    assert "not a stair pass" in result["scope"]["capability"]
