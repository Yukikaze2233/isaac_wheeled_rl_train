"""Optional ordinary stairs: native sampling, useful preteach and physical credit."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis import integrated_runtime as runtime
from wheeled_tasks.chassis.ordinary_large_stair import (
    POOL, ROWS, activation_fraction, command_profile, row_settings,
    sampling_candidate, validate_row,
)
from wheeled_tasks.chassis.skill_commands import SkillCommands
from wheeled_tasks.chassis.step_climb import StepClimbTask
from wheeled_tasks.chassis.task import choose_scene_groups, terrain_surfaces


ROOT = Path(__file__).resolve().parents[1]


def source():
    return json.loads((ROOT / "contracts/v6_task_campaign_terrain_worker_v23.json").read_text())


def candidate(**kwargs):
    return sampling_candidate(source(), enabled=True, **kwargs)


def environment(spec, count=16):
    generator = torch.Generator().manual_seed(20261009)
    env = SimpleNamespace(num_envs=count, device="cpu", task_actor_update=0,
        global_actor_update=90000, stage_actor_update=70000, training_transitions=900000000,
        scene_groups=["climb"] * count, cfg={"skill_specs": {"climb": spec},
            "task_campaign": {"domain": "terrain"}, "step_jump_task": {"enabled": True}},
        random=lambda n: torch.rand(n, generator=generator), policy_dt=.02,
        commands=torch.zeros(count, 3), command_target=torch.zeros(count, 2),
        mode=torch.zeros(count, dtype=torch.long), push_enabled=torch.zeros(count, dtype=torch.bool))
    for name in ("command_clock", "height_clock", "push_clock", "episode_limits", "episode_length_buf"):
        setattr(env, name, torch.zeros(count))
    pose = torch.zeros(count, 7)
    pose[:, 6] = 1.
    env.robot = SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose)))
    return env


def test_opt_in_preserves_all_unrelated_rows_and_explicit_donor_tradeoff():
    original = source()
    assert sampling_candidate(original) == original
    patched = sampling_candidate(original, enabled=True)
    assert original == source()
    assert patched["behavior_pool_fractions"][POOL] == .10
    assert patched["behavior_pool_fractions"]["untriggered_steps"] == .42
    assert patched["behavior_pool_fractions"]["surface_traversal"] == .02
    assert patched["behavior_pool_fractions"]["surface_height_joint"] == .02
    old_groups = {row["name"]: row for row in original["scene_groups"]}
    new_groups = {row["name"]: row for row in patched["scene_groups"]}
    for name, old in old_groups.items():
        pool = original["behavior_pool_membership"][name]
        assert patched["skill_specs"][name] == original["skill_specs"][name]
        expected = deepcopy(old)
        if pool in ("surface_traversal", "surface_height_joint"):
            expected["fraction"] *= .02 / original["behavior_pool_fractions"][pool]
        assert new_groups[name] == expected
    selected = choose_scene_groups(patched["scene_groups"], 4096)
    assert len(selected) == 4096
    assert set(name for name, _ in selected) == set(new_groups)
    assert len(new_groups) == len(old_groups) + 8
    assert all(patched["behavior_pool_membership"][name] == POOL for name, *_ in ROWS)


def test_native_builder_and_step_climb_agree_on_cumulative_double_geometry():
    patched = candidate()
    names = [name for name, *_ in ROWS]
    task = StepClimbTask(names, patched["skill_specs"], "cpu", .02)
    for index, (name, rises, treads, _, _) in enumerate(ROWS):
        spec = patched["skill_specs"][name]
        surfaces = terrain_surfaces("stairs", spec["terrain_limits"], 1., index)
        tops = [sum(rises[:i + 1]) for i in range(len(rises))]
        assert task.tops[index, :len(rises)].tolist() == pytest.approx(tops)
        assert [surface.z0 for surface in surfaces[1:-1]] == pytest.approx(tops)
        assert [surface.x1 - surface.x0 for surface in surfaces[1:-1]] == pytest.approx(treads)
        assert surfaces[0].x0 == -4. and surfaces[0].x1 == 0.
        assert surfaces[-1].z0 == pytest.approx(tops[-1])
        assert spec["mode"] == 0 and spec["kind"] == "body_climb"
        assert "step_jump" not in spec and "trigger_source" not in spec["body_climb"]
    assert task.tops[-1].tolist() == pytest.approx([.2, .45])
    assert task.edges[-1].tolist() == pytest.approx([0., .8])


@pytest.mark.parametrize("allowance", [1000, 8000, 15000])
def test_every_grade_is_monotone_and_final_exposure_precedes_budget_end(allowance):
    settings = [row_settings(i, origin_actor_update=123, actor_allowance=allowance) for i in range(8)]
    ends = [row["activation_end"] for row in settings]
    assert ends == sorted(ends) and len(set(ends)) == 8
    assert ends[-1] <= min(4000, int(.35 * allowance))
    for row in settings:
        assert row["activation_start"] < row["activation_end"]
        values = [activation_fraction(row, 123 + tick) for tick in range(allowance + 1)]
        assert values == sorted(values) and values[-1] == 1.
        assert activation_fraction(row, 123 + ends[-1]) == 1.
    assert activation_fraction(settings[-1], 123) == 0.


def test_preactivation_exposes_forward_reverse_and_real_stop_without_jump():
    settings = row_settings(7, origin_actor_update=0, actor_allowance=15000)
    time = torch.tensor([0., .99, 1., 1.99, 2., 4.99, 5.])
    command, active = command_profile(time, settings, 0)
    assert not active
    assert command[:, 0].tolist() == pytest.approx([.1, .1, -.1, -.1, 0., 0., .1])
    assert command[:, 1].eq(0).all() and command[:, 2].eq(.305).all()
    command, active = command_profile(time, settings, 3900)
    assert active and command[:, 0].eq(.2).all()
    command, active = command_profile(time, settings, 4000)
    assert active and command[:, 0].eq(.3).all()


def test_real_skill_sampler_has_one_command_owner_and_continues_pure_policy_prep():
    spec = candidate()["skill_specs"][ROWS[-1][0]]
    env = environment(spec)
    commands = SkillCommands(env)
    commands.sample(torch.arange(env.num_envs))
    assert commands.large_stair_phase_offset.unique().numel() > 1
    commands.large_stair_phase_offset.zero_()
    for elapsed, speed in ((.5, .1), (1.5, -.1), (3., 0.)):
        env.episode_length_buf[:] = elapsed / env.policy_dt
        commands.update()
        assert env.command_target[:, 0].eq(speed).all()
        assert env.mode.eq(0).all() and not commands.large_stair_assessment.any()
    assert not commands.schedule.request_active.any()
    assert not hasattr(commands, "policy_learning_mask")
    env.task_actor_update = 4000
    commands.update()
    assert env.command_target[:, 0].eq(.3).all() and commands.large_stair_assessment.all()
    assert not commands.schedule.request_active.any()
    # Episode resets do not reset the curriculum or turn completed teaching off.
    env.episode_length_buf.zero_()
    commands.sample(torch.tensor([0, 1]))
    assert commands.large_stair_assessment.all() and env.command_target[:, 0].eq(.3).all()
    env.task_actor_update = 3999
    with pytest.raises(ValueError, match="regressed"):
        commands.update()


def test_integrated_clock_ignores_critic_consumption_and_preserves_component_rehearsal():
    spec = candidate(origin_actor_update=50)["skill_specs"][ROWS[0][0]]
    env = environment(spec)
    selection = dict(domain="terrain", fresh_joint_actor_updates=15050,
        fresh_domain_actor_updates=50, lineage_id="cpu-test", plan_sha256="a" * 64,
        origin_checkpoint_sha256="b" * 64, origin_recovery_actor_update=14188)
    clocks = runtime._clock_document(selection, {"speed": 0, "terrain": 15000}, 15050, 50)
    env.cfg["integrated_runtime_candidate"] = dict(schema=runtime.SCHEMA, domain="terrain",
        clocks_at_source=clocks, learning_source=selection)
    with pytest.raises(ValueError, match="regressed"):
        runtime.component_actor_updates(env, "terrain")
    env.task_actor_update = 50
    assert runtime.component_actor_updates(env, "terrain") == 50
    env.global_actor_update += 10  # Critic-only/global accounting is not the course clock.
    env.training_transitions += 4096 * 24 * 10
    assert runtime.component_actor_updates(env, "terrain") == 50
    env.task_actor_update = 1250
    assert runtime.component_actor_updates(env, "terrain") == 1250
    assert activation_fraction(spec["ordinary_large_stair_curriculum"], 1250) == 1.
    later = dict(selection, domain="jump", fresh_joint_actor_updates=27000,
        fresh_domain_actor_updates=1000)
    env.cfg["task_campaign"]["domain"] = "jump"
    env.cfg["integrated_runtime_candidate"].update(domain="jump", learning_source=later,
        clocks_at_source=runtime._clock_document(later,
            {"speed": 0, "terrain": 15000, "jump": 26000}, 27000, 1000))
    env.task_actor_update = 1001
    assert runtime.component_actor_updates(env, "terrain") == 12001
    assert activation_fraction(spec["ordinary_large_stair_curriculum"], 12001) == 1.


def test_inactive_climb_pays_no_special_reward_or_failed_climb_and_rearming_cannot_fake_crossing():
    spec = candidate()["skill_specs"][ROWS[3][0]]
    task = StepClimbTask(["climb"], {"climb": spec}, "cpu", .02)
    wheels = torch.tensor([[[.10, 0., .26], [.10, 0., .26]]])
    mask = torch.tensor([False])
    task.set_assessment_mask(mask, wheels)
    for _ in range(20):
        task.observe(wheels, torch.tensor([[False, False]]), torch.tensor([True]))
    assert not task.flight_violation.any() and not task.success.any()
    assert task.height_tracking_scale(wheels).eq(1).all()
    assert all(value.eq(0).all() for value in task.reward_terms(wheels, torch.tensor([[0., 0., -1.]])).values())
    assert not task.diagnostics()["body_climb_enabled"].any()
    assert task.diagnostics()["body_climb_required_count"].eq(0).all()
    task.set_assessment_mask(torch.tensor([True]), wheels)
    for _ in range(20):
        task.observe(wheels, torch.tensor([[True, True]]), torch.tensor([True]))
    assert not task.success.any()  # Activation on top did not manufacture an edge crossing.
    wheels[:, :, 0] = -.15
    task.observe(wheels, torch.tensor([[True, True]]), torch.tensor([True]))
    wheels[:, :, 0] = .10
    for _ in range(20):
        task.observe(wheels, torch.tensor([[True, True]]), torch.tensor([True]))
    assert task.success.item()
    for _ in range(4):
        task.observe(wheels, torch.tensor([[False, False]]), torch.tensor([True]))
    assert task.flight_violation.item() and not task.success.item()


def test_arming_before_first_physical_step_retains_its_real_riser_crossing():
    spec = candidate()["skill_specs"][ROWS[0][0]]
    task = StepClimbTask(["climb"], {"climb": spec}, "cpu", .02)
    before = torch.tensor([[[-.01, 0., .06], [-.01, 0., .06]]])
    task.set_assessment_mask(torch.tensor([False]), before)
    task.observe(before, torch.tensor([[True, True]]), torch.tensor([True]))
    task.set_assessment_mask(torch.tensor([True]), before)
    after = torch.tensor([[[.07, 0., .11], [.07, 0., .11]]])
    task.observe(after, torch.tensor([[True, True]]), torch.tensor([True]))
    assert task.front_passed.all()
    for _ in range(20):
        task.observe(after, torch.tensor([[True, True]]), torch.tensor([True]))
    assert task.success.item()


@pytest.mark.parametrize("field,value", [("mode", 4), ("command_sequence", []),
    ("planar_command_sampling", {}), ("command_curriculum", {}), ("reference_velocity", [1., 0.])])
def test_second_command_owner_or_jump_request_is_rejected(field, value):
    spec = candidate()["skill_specs"][ROWS[0][0]]
    spec[field] = value
    with pytest.raises(ValueError, match="another command owner"):
        validate_row(spec)


def test_fixed_evaluation_retains_actual_forward_climb_and_bypasses_training_clock():
    spec = candidate()["skill_specs"][ROWS[-1][0]]
    env = environment(spec)
    env.cfg.update(evaluation_exact_cases=True, evaluation={"cases": [dict(name="climb",
        command=spec["command"], skill=spec)]})
    commands = SkillCommands(env)
    commands.sample(torch.arange(env.num_envs))
    assert not commands.large_stair_curricula
    assert env.command_target[:, 0].eq(.3).all() and env.mode.eq(0).all()


@pytest.mark.parametrize("kwargs", [dict(actor_allowance=999), dict(origin_actor_update=-1),
    dict(actor_allowance=True), dict(origin_actor_update=1.5)])
def test_invalid_budget_or_origin_is_not_inferred_or_refunded(kwargs):
    with pytest.raises(ValueError, match="actor origin"):
        candidate(**kwargs)
