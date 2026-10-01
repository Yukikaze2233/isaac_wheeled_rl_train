"""Native-buffer clock, episode isolation and authenticated task-delay migration."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.signal_delay import PhysicalSignalDelay, validate_signal_delay
from wheeled_tasks.chassis.learning_audit import StationaryWindowAudit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner
from train_chassis import resume_critic_warmup


def make(count=4, domain="speed", seed=20):
    return PhysicalSignalDelay(count, "cpu", campaign.signal_delay_settings(domain),
                               .005, torch.Generator().manual_seed(seed))


def test_native_sensor_and_target_delay_advance_each_physics_tick_not_each_policy():
    delay = make()
    delay.reset(torch.arange(4), torch.zeros(4, 18))
    delay.lags[:] = torch.tensor([1, 2, 3, 4, 2, 1])
    for tick in range(10):
        value = torch.full((4, 18), float(tick))
        delay.observe(value)
        legs, wheels = delay.targets(torch.full((4, 4), float(tick)), torch.full((4, 2), -float(tick)))
    assert delay.physics_steps == 10
    assert delay.feedback[0].tolist() == [8.] * 6 + [7.] * 6 + [6.] * 3 + [5.] * 3
    assert legs.eq(7.).all() and wheels.eq(-8.).all()
    cached = delay.feedback.clone()
    for _ in range(20):
        torch.testing.assert_close(delay.feedback, cached)
    assert delay.physics_steps == 10


def test_partial_reset_does_not_leak_previous_episode_or_reset_other_environments():
    delay = make()
    delay.reset(torch.arange(4), torch.zeros(4, 18))
    for tick in range(5):
        delay.observe(torch.full((4, 18), 9.))
        delay.targets(torch.full((4, 4), 9.), torch.full((4, 2), 9.))
    sensors = torch.zeros(4, 18); sensors[0] = -8.
    delay.reset(torch.tensor([0]), sensors)
    delay.lags[:] = 4
    delay.observe(sensors)
    legs, wheels = delay.targets(torch.full((4, 4), -7.), torch.full((4, 2), -6.))
    assert delay.feedback[0].eq(-8.).all() and delay.feedback[1].eq(9.).all()
    assert legs[0].eq(-7.).all() and wheels[0].eq(-6.).all()
    assert legs[1].eq(9.).all() and wheels[1].eq(9.).all()


def test_seeded_wide_coverage_clean_population_curriculum_and_independent_jitter():
    left, right = make(4096), make(4096)
    for d in (left, right):
        d.reset(torch.arange(4096), torch.zeros(4096, 18))
        d.begin_policy(1000)
    torch.testing.assert_close(left.lags, right.lags)
    assert .72 < left.enabled.float().mean() < .78
    assert left.lags[~left.enabled].eq(0).all()
    assert left.lags[:, :4].max() == 16 and left.lags[:, 4:].max() == 12
    assert (left.lags[:, 0] != left.lags[:, 1]).any()
    before = left.lags.clone(); left.begin_policy(1001)
    assert (left.lags != before).any()
    flat = make(domain="flat");flat.reset(torch.arange(4), torch.zeros(4, 18))
    flat.begin_policy(1400);assert flat.lags.eq(0).all() and flat.strength == 0.
    flat.begin_policy(2000);assert flat.strength == .5
    flat.begin_policy(2500);assert flat.strength == 1.


@pytest.mark.parametrize("field,value", [("clock", "policy_steps"), ("feedback_pd", "stale"),
                                         ("enabled_fraction", 1.), ("enabled", None)])
def test_delay_definition_rejects_clock_scope_and_silent_enable_changes(field, value):
    config = campaign.signal_delay_settings("flat");config[field] = value
    with pytest.raises(ValueError):validate_signal_delay(config, .005)


def test_named_migration_preserves_physics_reward_budget_and_requires_real_delay_enable():
    before = json.loads((ROOT / "contracts/v6_task_campaign_v5.json").read_text())
    plan = json.loads((ROOT / "contracts/v6_task_campaign_v6.json").read_text())
    campaign.validate_plan_migration(before, plan, plan["revision_migration"]["run_directory"])
    assert len(runner.validate_contracts(plan)) == 30
    for old, new in zip(before["domains"], plan["domains"], strict=True):
        source = json.loads((ROOT / old["contract"]).read_text())
        config = json.loads((ROOT / new["contract"]).read_text())
        campaign.validate_task_resume(source, config)
        campaign.validate_task_resume(config, config)
        for key in ("dense_tracking", "reference_reward", "physics_dt", "v5_control", "actuator_response",
                    "command_transport", "usb_transport", "dynamics_randomization", "skill_specs",
                    "new_asset_curriculum", "behavior_pool_fractions", "policy_architecture", "learning_rate"):
            assert source[key] == config[key]
        for key, value in (("enabled", False), ("enabled_fraction", 1.), ("jitter_ms", 20.)):
            bad = deepcopy(config);bad["signal_delay"][key] = value
            with pytest.raises(ValueError):campaign.validate_task_resume(source, bad)
        infos = dict(contract_sha256=old["sha256"], task_campaign=dict(domain=new["name"]),
                     successful_updates_total=2100, critic_warmup_remaining=0)
        assert resume_critic_warmup(config, infos, contract_sha256=new["sha256"], task_change=True) == 10
        infos["contract_sha256"] = new["sha256"];infos["critic_warmup_remaining"] = 3
        assert resume_critic_warmup(config, infos, contract_sha256=new["sha256"], task_change=True) == 3


def data(tick, *, moving=False, done=False):
    return dict(commands=torch.tensor([[.5 if moving else 0., 0., .305]]),
        done=torch.tensor([done]), height=torch.tensor([.305]), position=torch.tensor([[tick * .0002, 0., 0.]]),
        planar_speed_world=torch.tensor([.01]))


def test_stationary_windows_measure_exact_10_seconds_after_entry_and_survive_logger_drain():
    audit = StationaryWindowAudit(1, "cpu", .02)
    for tick in range(100):audit.observe(data(tick))
    assert audit.report([0])["samples"] == 0
    audit.clear_report()
    for tick in range(100, 600):audit.observe(data(tick))
    result = audit.report([0])
    assert result["samples"] == 1
    assert result["endpoint_mean_m"] == pytest.approx(.1)
    assert result["peak_max_m"] == pytest.approx(.1)
    assert result["mean_speed_m_s"] == pytest.approx(.01, abs=1e-7)
    audit.clear_report();assert audit.report([0])["samples"] == 0


def test_stationary_windows_discard_partial_moving_or_reset_sequences():
    audit = StationaryWindowAudit(1, "cpu", .02)
    for tick in range(400):audit.observe(data(tick))
    audit.observe(data(400, moving=True))
    for tick in range(401, 950):audit.observe(data(tick))
    assert audit.report([0])["samples"] == 0
    audit.observe(data(950, done=True))
    for tick in range(951, 1500):audit.observe(data(tick))
    assert audit.report([0])["samples"] == 0
