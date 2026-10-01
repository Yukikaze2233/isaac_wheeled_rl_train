"""Coverage holes, terminal waiting incentives and once-only value adaptation."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.rewards import episode_event_terms
from wheeled_tasks.chassis.skill_commands import sample_planar_command, validate_planar_sampling
from wheeled_tasks.chassis.task import episode_outcome

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_v6_rewards import calibration_suite, discounted_return, trajectory_terms
from run_task_campaign import validate_contracts
from train_chassis import resume_critic_warmup


def read(name):
    return json.loads((ROOT / name).read_text())


def test_named_repair_preserves_physics_tracking_quotas_and_every_case_threshold():
    old, new = read("contracts/v6_task_campaign_v3.json"), read("contracts/v6_task_campaign_v4.json")
    result = campaign.validate_plan_migration(old, new, new["revision_migration"]["run_directory"])
    assert result["consumption"] == "preserved_no_refund"
    assert len(validate_contracts(new)) == 30
    for before, after in zip(old["domains"], new["domains"], strict=True):
        source, target = read(before["contract"]), read(after["contract"])
        assert campaign.validate_task_resume(source, target)["task_clock"] == "restored"
        assert campaign.validate_task_resume(target, target)["task_clock"] == "restored"
        for key in ("physics_dt", "policy_architecture", "control_math_sha256", "dense_tracking",
                    "landing_tracking", "scene_groups", "behavior_pool_fractions", "evaluation", "motion_limits"):
            assert source[key] == target[key]
        for key, value in (("termination_event_cost", 4.), ("success_event_bonus", 100.),
                           ("learning_rate", 2e-4), ("physics_dt", .001), ("checkpoint_interval", 100)):
            broken = deepcopy(target); broken[key] = value
            with pytest.raises(ValueError):
                campaign.validate_task_resume(source, broken)


def test_continuous_commands_include_held_low_speed_and_independent_signed_combinations():
    flat = read("contracts/v6_task_campaign_flat_worker_v4.json")
    rng = torch.Generator().manual_seed(43)
    random = lambda n: torch.rand(n, generator=rng)
    count = 50000
    for name in ("forward_continuous_low", "backward_continuous_low", "yaw_continuous_low"):
        spec = flat["skill_specs"][name]
        validate_planar_sampling(spec)
        command = torch.tensor(spec["command"]).repeat(count, 1)
        sampled = sample_planar_command(command, spec, random)
        axis = 1 if name.startswith("yaw") else 0
        values = sampled[:, axis].abs()
        assert values.min() >= .05 and values.max() <= 1.
        assert ((values >= .1) & (values <= .3)).float().mean() > .19
    spec = flat["skill_specs"]["curve_low"]
    sampled = sample_planar_command(torch.tensor(spec["command"]).repeat(count, 1), spec, random)
    anchors = sampled[:, :2].eq(torch.tensor(spec["command"][:2])).all(-1)
    assert .48 < anchors.float().mean() < .52
    varied = sampled[~anchors]
    for vx_sign in (-1, 1):
        for yaw_sign in (-1, 1):
            quadrant = (varied[:, 0] * vx_sign > 0) & (varied[:, 1] * yaw_sign > 0)
            assert .23 < quadrant.float().mean() < .27
    assert abs(torch.corrcoef(varied[:, :2].T)[0, 1]) < .03
    groups = {g["name"]:g for g in flat["scene_groups"]}
    assert all(g["fraction"] == pytest.approx(.02) for n,g in groups.items() if n.startswith("stand_") and n != "stand_continuous")


def test_evaluation_uses_exact_commands_and_legacy_rng_stream_is_unchanged():
    generator = torch.Generator().manual_seed(31)
    initial = generator.get_state()
    random = lambda n: torch.rand(n, generator=generator)
    nominal = torch.tensor([[1., .6, .305]]).repeat(20, 1)
    spec = dict(kind="curve", sample_amplitude=True, planar_command_sampling=dict(
        vx_range_m_s=[-1.,1.], yaw_range_rad_s=[-.6,.6], nominal_fraction=.5))
    torch.testing.assert_close(sample_planar_command(nominal.clone(), spec, random, evaluation=True), nominal)
    assert torch.equal(initial, generator.get_state())
    legacy = dict(kind="curve", sample_amplitude=True)
    actual = sample_planar_command(nominal.clone(), legacy, random)
    generator.set_state(initial)
    expected = nominal.clone(); expected[:, :2] *= (.4 + .6 * random(20))[:, None]
    expected[:, 1] *= torch.where(random(20) < .5, -1., 1.)
    assert torch.equal(actual, expected)
    for bounds in ([float("nan"),1.],[-.1,1.],[1.,1.]):
        with pytest.raises(ValueError):
            validate_planar_sampling(dict(amplitude_range=bounds))


def test_event_units_do_not_turn_boundary_or_timeout_into_failure():
    zero = torch.zeros(4, dtype=torch.bool)
    failed, timeout, success = episode_outcome(dict(fall=torch.tensor([True,False,False,False]),
        boundary=torch.tensor([False,True,False,False])), torch.tensor([False,False,False,True]),
        torch.tensor([False,False,True,False]))
    terms = episode_event_terms(success, failed, dict(termination_event_cost=200., success_event_bonus=30.))
    assert sum(terms.values()).tolist() == [-200.,0.,0.,30.]
    assert timeout.tolist() == [False,True,True,False]
    assert episode_event_terms(zero, zero, {})["success_event"].eq(0).all()


def test_complete_now_beats_waiting_and_small_failure_cost_fails_the_envelope():
    for vx, yaw in ((0.,0.),(.3,0.),(0.,1.)):
        command = torch.tensor([[vx,yaw,.305]],dtype=torch.float64).repeat(501,1)
        velocity = torch.zeros_like(command)
        # Compare the native supported state at the goal with idle-before-goal.
        terms = trajectory_terms(command,velocity,command[:,2],stationary_anchor=False)
        density = torch.stack(list(terms.values())).sum(0)
        assert discounted_return(density[:1],.02,.99,success_bonus=30.) > discounted_return(density,.02,.99,success_bonus=30.)
    baseline = torch.full((201,),5.,dtype=torch.float64)
    assert discounted_return(baseline[:1],.02,.99,success_bonus=5.) < discounted_return(baseline,.02,.99,success_bonus=5.)
    # Near completion: even retaining a 12/s reward forever plus both remaining
    # 2-point platform events (un-discounted upper bound) cannot beat +30.
    assert 12. * .02 / (1. - .99) + 4. < 30.
    repaired,_ = calibration_suite(terminal_cost=200.,stationary_anchor=False)
    unsafe,_ = calibration_suite(terminal_cost=4.,stationary_anchor=False)
    assert repaired["all_checks_passed"]
    assert not unsafe["termination"]["failure_worse_than_survival_and_bootstrap"]


def test_value_warmup_runs_once_and_faults_restore_its_remaining_updates():
    config = dict(resume_critic_warmup_updates=10)
    info = dict(contract_sha256="v3",successful_updates_total=1620,critic_warmup_remaining=0)
    assert resume_critic_warmup(config,info,contract_sha256="v4",task_change=True) == 10
    info.update(contract_sha256="v4",critic_warmup_remaining=7,successful_updates_total=1623)
    assert resume_critic_warmup(config,info,contract_sha256="v4",task_change=True) == 7
    info["critic_warmup_remaining"] = 0
    assert resume_critic_warmup(config,info,contract_sha256="v4",task_change=True) == 0
    with pytest.raises(ValueError,match="SHA"):
        resume_critic_warmup(config,info,task_change=True)
