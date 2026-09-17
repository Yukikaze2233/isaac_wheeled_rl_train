"""Independent task-state and terrain geometry checks without Isaac imports."""
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.task import Phase, PhaseTracker, choose_terrains, phase_reward_masks, terrain_surfaces


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "contracts/chassis_full_v1.json").read_text())


def test_full_budget_interface_and_scene_coverage():
    assert sum(s["updates"] for s in CONFIG["stages"]) == 100000
    assert CONFIG["actor_dim"] == 5 * (25 + 5 + 4 + 2 + 5 + 1)
    assert CONFIG["critic_dim"] == 42 + 3 + 1 + 2 + 14 + 14
    for stage in CONFIG["stages"]:
        families = choose_terrains(stage, 64)
        assert families.count("flat") >= math.ceil(64 * 0.4)
        assert set(stage["terrain"]) <= set(families)


def test_contact_debounce_takeoff_flight_landing_recovery():
    tracker = PhaseTracker(1, "cpu")
    contact = torch.tensor([[True, True]])
    request, stable = torch.tensor([True]), torch.tensor([False])
    tracker.update(contact, request, stable)
    assert tracker.phase.item() == Phase.TAKEOFF
    for _ in range(2):
        tracker.update(~contact, request, stable)
        assert tracker.phase.item() == Phase.TAKEOFF
    tracker.update(~contact, request, stable)
    assert tracker.phase.item() == Phase.FLIGHT and tracker.flew.item()
    tracker.update(contact, request, stable)
    assert tracker.phase.item() == Phase.LANDING
    for _ in range(7):
        tracker.update(contact, request, stable)
    assert tracker.phase.item() == Phase.RECOVERY
    for _ in range(55):
        tracker.update(contact, ~request, ~stable)
    assert tracker.phase.item() == Phase.GROUND


def test_passive_drop_does_not_consume_future_jump_request():
    tracker = PhaseTracker(2, "cpu")
    for _ in range(4):
        tracker.update(torch.zeros(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
                       torch.zeros(2, dtype=torch.bool))
    assert (tracker.phase == Phase.FLIGHT).all()
    assert not tracker.flew.any()
    tracker.reset(torch.tensor([0]))
    assert tracker.phase.tolist() == [Phase.GROUND, Phase.FLIGHT]


def test_flight_disables_ground_and_landing_rewards():
    masks = phase_reward_masks(torch.tensor(list(Phase)))
    assert masks["ground"].tolist() == [True, False, False, False, False]
    assert masks["flight"].tolist() == [False, False, True, False, False]
    assert masks["landing"].tolist() == [False, False, False, True, True]


@pytest.mark.parametrize("kind", ["flat", "material", "slope", "rough", "low_step", "step_up", "step_down", "stairs", "jump"])
def test_terrain_tile_is_contiguous_and_box_top_matches_height(kind):
    surfaces = terrain_surfaces(kind, CONFIG["terrain_limits"], 1., index=3)
    assert surfaces[0].x0 == -4 and surfaces[-1].x1 == 4
    for a, b in zip(surfaces, surfaces[1:]):
        assert a.x1 == pytest.approx(b.x0)
    for surface in surfaces:
        size, center, quat = surface.box()
        assert min(size) > 0
        # Independently rotate the local upper face midpoint about Y.
        angle = 2 * math.atan2(quat[2], quat[0])
        x = center[0] + math.sin(angle) * size[2] / 2
        z = center[2] + math.cos(angle) * size[2] / 2
        assert z == pytest.approx(surface.height(x), abs=1e-12)
        assert 2 * surface.friction - 0.5 >= 0


def test_step_direction_and_material_boundaries():
    limits = CONFIG["terrain_limits"]
    up = terrain_surfaces("step_up", limits, 1)
    down = terrain_surfaces("step_down", limits, 1)
    assert up[1].height(1) - up[0].height(-1) == pytest.approx(0.03)
    assert down[1].height(1) - down[0].height(-1) == pytest.approx(-0.05)
    material = terrain_surfaces("material", limits, 1)
    assert [s.friction for s in material] == pytest.approx([0.5, 0.3, 0.7])
