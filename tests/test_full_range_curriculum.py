"""Full-domain coverage, physically bound references and continuous dense objectives."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.rewards import CommandedHeightMargin, height_tracking_terms
from wheeled_tasks.chassis.skill_commands import SkillCommands, height_reference
from wheeled_tasks.chassis.task import choose_scene_groups
from wheeled_tasks.chassis.v5_control import V5Control

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v5_full_range_v54.json"), load)
    return plan, load(plan["base_contract"])


def test_complete_ground_domain_is_available_before_any_training_update():
    plan, base = configuration()
    config = stage_contract(base, plan, plan["stages"][0], 6144)
    assert plan["initialization"].startswith("scratch_")
    assert len(plan["stages"]) == 5
    assert sum(s["updates"] for s in plan["stages"]) == 52000
    specs = config["skill_specs"]
    assert specs["height_full"]["height_range_m"] == [.21, .35]
    assert specs["forward_5"]["command"][0] == 5.
    assert specs["backward_5"]["command"][0] == -5.
    assert specs["rotate_3rps"]["command"][1] == pytest.approx(6 * torch.pi)
    assert not any(s.get("command_curriculum") for s in specs.values())
    for name in ("forward_5", "backward_5", "rotate_3rps", "spin_translate_3rps"):
        assert specs[name]["height_sampling"]["range_m"] == [.21, .35]
    cases = {c["name"]: c for c in config["evaluation"]["cases"]}
    assert cases["height_full_hold_low"]["command"][2] == .21
    assert cases["height_full_hold_high"]["command"][2] == .35
    assert cases["forward_05_height_210mm"]["skill"]["height_transition_seconds"] == 6.
    assert cases["height_full"]["height_velocity_mae_m_s_max"] == .03
    assert config["height_range_m"] == [.21, .35]


def test_later_phases_retain_full_domain_and_terrain_contact_targets():
    plan, base = configuration()
    for recipe in plan["stages"]:
        config = stage_contract(base, plan, recipe, 6144)
        assert sum(g["fraction"] for g in config["scene_groups"]) == pytest.approx(1.)
        assert len({c["name"] for c in config["evaluation"]["cases"]}) == len(config["evaluation"]["cases"])
        assert config["height_workspace"]["height_m"][0] == .21
        assert config["landing_tracking"]["impact_weight"] > 0
        assert config["task_semantics"]["preload_depth_m"] == .06
        assert config["task_semantics"]["jump_tuck_extension_m"] == .16
        for count in (128, 6144):
            scenes = choose_scene_groups(config["scene_groups"], count)
            assert len(scenes) == count
            assert {name for name, _ in scenes} == {g["name"] for g in config["scene_groups"]}
    terrain = next(s for s in plan["stages"] if s["kind"] == "terrain")
    config = stage_contract(base, plan, terrain, 6144)
    assert config["skill_specs"]["step_up_10"]["terrain_limits"]["step_up_m"] == .1
    assert config["skill_specs"]["step_down_15"]["terrain_limits"]["step_down_m"] == .15
    assert config["skill_specs"]["cross_slope"]["command"][2] == .28


def test_reference_visits_both_endpoints_with_zero_speed_and_dwell():
    plan, base = configuration()
    profile = stage_contract(base, plan, plan["stages"][0], 64)["skill_specs"]["height_full"]
    times = torch.tensor([0., 6., 7., 8., 11., 14., 15., 16., 22., 23., 30.], dtype=torch.float64)
    height, speed = height_reference(times, profile)
    assert height.tolist() == pytest.approx([.305, .21, .21, .21, .28, .35, .35, .35, .21, .21, .35])
    assert speed[[0, 1, 2, 3, 5, 6, 7, 8, 9, 10]].abs().max() < 1e-12
    assert speed[4] == pytest.approx(.035)
    dense_time = torch.linspace(0., 38., 3801, dtype=torch.float64)
    h, v = height_reference(dense_time, profile)
    assert h.min() >= .21 - 1e-12 and h.max() <= .35 + 1e-12
    step = 1e-5
    derivative = (height_reference(dense_time + step, profile)[0] - height_reference(dense_time - step, profile)[0]) / (2 * step)
    assert torch.allclose(v, derivative, atol=2e-7, rtol=1e-4)


@pytest.mark.parametrize("endpoint,target", [("low", .21), ("high", .35)])
def test_endpoint_hold_uses_a_continuous_initial_transition(endpoint, target):
    profile = {"height_range_m": [.21, .35], "height_motion": {
        "transition_seconds": 6., "dwell_seconds": 2., "initial_height_m": .305, "endpoint": endpoint}}
    h, v = height_reference(torch.tensor([0., 3., 6., 18.]), profile)
    assert h.tolist() == pytest.approx([.305, (.305 + target) / 2, target, target])
    assert v[[0, 2, 3]].abs().max() == 0.


def test_height_conditioned_margin_retains_hard_boundary_cost():
    reference = json.loads((ROOT / "contracts/v5_height_workspace_v1.json").read_text())
    margin = CommandedHeightMargin(reference, "cpu")
    height = torch.tensor([.21, .305, .35])
    risk = torch.tensor([[reference["risk"][0]] * 2, [0., 0.], [reference["risk"][-1]] * 2])
    assert margin.excess(risk, height).abs().max() < 1e-6
    assert torch.allclose(margin.excess(torch.ones(3, 2), height), torch.ones(3, 2))
    assert torch.allclose(margin.excess(torch.full((1, 2), .4), torch.tensor([.305])), torch.full((1, 2), .4))


def test_wide_companion_has_no_peak_shift_and_keeps_a_far_field_signal():
    plan, _ = configuration()
    error = torch.tensor([0., .1], requires_grad=True)
    terms = height_tracking_terms(error, torch.zeros(2), torch.ones(2), torch.ones(2), plan["height_tracking"])
    assert terms["height_wide_companion"][0] == 0.
    derivative = torch.autograd.grad(terms["height_wide_companion"].sum(), error)[0]
    assert derivative[1] < -1.
    assert terms["height_velocity_tracking"].tolist() == pytest.approx([.5, .5])


def test_sampling_actually_reaches_the_full_height_domain_and_preserves_push_cap():
    count = 8
    profile = {"kind": "forward", "mode": 1, "command": [.5, 0., .305], "push_m_s": .3,
               "height_sampling": {"range_m": [.21, .35], "nominal_fraction": .25, "endpoint_fraction": .4},
               "height_transition_seconds": 6.}
    pose = torch.zeros(count, 7)
    pose[:, 6] = 1.
    env = SimpleNamespace(num_envs=count, device="cpu", cfg={"skill_specs": {"move": profile}},
        stage_cfg={"push_max": 0.}, scene_groups=["move"] * count, commands=torch.zeros(count, 3),
        command_target=torch.zeros(count, 2), mode=torch.zeros(count, dtype=torch.long), training_transitions=0,
        command_clock=torch.zeros(count), height_clock=torch.zeros(count), push_clock=torch.zeros(count),
        push_enabled=torch.zeros(count, dtype=torch.bool), policy_dt=.02,
        episode_length_buf=torch.full((count,), 600),
        robot=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose))),
        random=lambda n: torch.linspace(0., .99, n))
    commands = SkillCommands(env)
    commands.sample(torch.arange(count))
    assert torch.all(env.commands[:, 2] == .305)
    commands.update()
    assert env.commands[:, 2].min() == pytest.approx(.21)
    assert env.commands[:, 2].max() == pytest.approx(.35)
    assert torch.all(commands.push_speed == .3)


def test_workspace_reference_matches_mechanics_and_existing_drive_limits():
    module_spec = importlib.util.spec_from_file_location("workspace_test", ROOT / "tools/analyze_v5_spring_limits.py")
    audit = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(audit)
    bundle = ROOT / "model/纯底盘_v5/urdf"
    saved = json.loads((ROOT / "contracts/v5_height_workspace_v2.json").read_text())
    report = audit.height_workspace(bundle, [.23, .305, .43], saved["knee_working_margin_rad"])
    assert all(row["inside_existing_motor_and_action_limits"] for row in report["rows"])
    assert report["reward_reference"]["risk"] == pytest.approx([saved["risk"][0], 0., saved["risk"][-1]], abs=1e-8)
    spec = json.loads((bundle / "model_spec.json").read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    fit = json.loads((bundle / "fit_10mpa.json").read_text())
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v5_full_usb_v59.json"), load)
    base = load(plan["base_contract"])
    config = stage_contract(base, plan, plan["stages"][0], 64)
    control = V5Control(manifest, spec, fit, json.loads((ROOT / config["control_math_source"]).read_text()), config["v5_control"], "cpu")
    for row in report["rows"]:
        pose = audit.balanced_standing_pose(spec, row["knee_inner_deg"])["joint_positions"]
        risk = control.working_margin_risk(torch.tensor([[pose[n] for n in ("L_joint2", "R_jonit2")]]),
            torch.tensor([[pose[n] for n in manifest["spring_joint_names"]]]), saved["knee_working_margin_rad"])
        assert risk.max() == pytest.approx(row["reference_risk"], abs=2e-6)


def test_phase_replacement_keeps_the_original_catalog_and_rejects_duplicate_additions():
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = load("contracts/v5_full_range_v54.json")
    broken = deepcopy(plan)
    broken["catalog_additions"].append({"name": "stand", "skill": "stand"})
    with pytest.raises(ValueError, match="Duplicate skill"):
        resolve_plan(broken, load)
