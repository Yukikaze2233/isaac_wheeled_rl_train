"""Specialist isolation, invariant regression scenarios and motion semantics."""
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.full_curriculum import stage_contract
from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.skill_commands import profile_command, reference_velocity
from wheeled_tasks.chassis.skill_curriculum import SKILLS
from wheeled_tasks.chassis.task import Phase, Surface, choose_scene_groups, terrain_surfaces

ROOT = Path(__file__).resolve().parents[1]


def contracts():
    base = json.loads((ROOT / "contracts/v5_locomotion_v2.json").read_text())
    plan = json.loads((ROOT / "contracts/v5_scut_skills_v3.json").read_text())
    return base, plan


def test_specialists_cover_all_skills_and_have_one_training_group():
    base, plan = contracts()
    assert {r["skill"] for r in plan["stages"] if "skill" in r} == set(SKILLS)
    for recipe in plan["stages"]:
        c = stage_contract(base, plan, recipe, 256)
        if "skill" in recipe:
            assert len(c["scene_groups"]) == 1
            assert set(name for name, _ in choose_scene_groups(c["scene_groups"], 64)) == {recipe["name"]}
        cases = c["evaluation"]["cases"]
        assert len({case["name"] for case in cases}) == len(cases)
        assert all(case["anchor"] for case in cases[:7])
        for group in c["scene_groups"]:
            for kind in group["terrain"]:
                assert terrain_surfaces(kind, c["terrain_limits"], 1.)


def test_previous_step_case_keeps_its_original_height():
    base, plan = contracts()
    recipe = next(r for r in plan["stages"] if r["name"] == "step_up_06")
    config = stage_contract(base, plan, recipe, 256)
    cases = {c["name"]: c for c in config["evaluation"]["cases"]}
    assert cases["step_up_03"]["terrain_limits"]["step_up_m"] == .03
    assert cases["step_up_06"]["terrain_limits"]["step_up_m"] == .06
    assert cases["step_up_03"]["anchor"]


def test_start_stop_contains_forward_brake_reverse_and_brake():
    times = torch.tensor([1., 4., 7., 10.])
    result = profile_command(times, [.5, 0., .305], {"kind": "start_stop"})
    torch.testing.assert_close(result[:, 0], torch.tensor([.5, 0., -.5, 0.]))
    assert result[:, 2].tolist() == pytest.approx([.305] * 4)


def test_height_profile_and_rotating_reference_are_consistent():
    result = profile_command(torch.tensor([0., 1.5, 4.5]), [0., 0., .305], {"kind": "height"})
    torch.testing.assert_close(result[:, 2], torch.tensor([.305, .32, .29]))
    body = torch.tensor([[1., 0.], [1., 0.]])
    world = reference_velocity(body, torch.tensor([0., math.pi / 2]))
    torch.testing.assert_close(world, torch.tensor([[1., 0.], [0., 1.]]), atol=1e-6, rtol=0)


def test_cross_slope_collision_top_matches_analytic_surface():
    surface = Surface(-4., 4., cross_slope=.1)
    size, center, q = surface.box()
    w, x, y, z = q
    normal = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    top = [center[i] + normal[i] * size[2] / 2 for i in range(3)]
    assert top[2] == pytest.approx(surface.height(top[0], top[1]), abs=1e-9)
    assert surface.height(0., .2) - surface.height(0., -.2) == pytest.approx(.04)


def test_jump_com_gate_rejects_leg_extension_without_airborne_com_rise():
    task = FullTaskSemantics(1, "cpu", .01, {"jump_com_rise_m": .02})
    mode = torch.tensor([4])
    air = torch.tensor([Phase.FLIGHT])
    task.observe_com(mode, air, torch.tensor([.30]), torch.tensor([.7]))
    task.observe_com(mode, air, torch.tensor([.31]), torch.tensor([.2]))
    assert task.com_rise.item() == pytest.approx(.01)
    task.clear_air_time_peak[:] = .1
    task.release_velocity[:] = .7
    task.height_peak[:] = .4
    args = (mode, torch.tensor([[0., 0., .305]]), torch.zeros(1, 2), torch.tensor([.65]),
            torch.ones(1, 2, dtype=torch.bool), torch.ones(1, dtype=torch.bool),
            torch.tensor([Phase.RECOVERY]), torch.tensor([.305]))
    assert not task.completion(*args).item()
    task.observe_com(mode, air, torch.tensor([.33]), torch.tensor([0.]))
    assert task.completion(*args).item()


def test_mixed_jump_references_support_per_case_targets():
    task = FullTaskSemantics(2, "cpu", .01, {"jump_apex_delta_m": torch.tensor([.06, .1]),
                                           "release_height_offset_m": torch.tensor([.025, .025])})
    h, v = task.jump_reference(torch.full((2,), .305), torch.full((2,), Phase.FLIGHT),
                              torch.zeros(2), torch.full((2,), .05), torch.zeros(2))
    assert h[1] > h[0] and v[1] > v[0]
