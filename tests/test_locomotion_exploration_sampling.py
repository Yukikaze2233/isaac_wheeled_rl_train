"""CPU-only contracts for the independent native exploration probe."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from locomotion_diagnostic_suite import DEFAULT_SUITE, HIGH_SPEED_SUITE, get_suite

FILE = Path(__file__).resolve().parents[1] / "scripts/locomotion_exploration_sampling.py"
SPEC = importlib.util.spec_from_file_location("locomotion_exploration_sampling", FILE)
sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampling)

CONFIG = {
    "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
    "actor": {"class_name": "MLPModel", "hidden_dims": [256, 128, 64], "activation": "elu",
              "obs_normalization": False,
              "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"}},
}
SOURCE = {"actor_dim": 35, "critic_dim": 81, "action_dim": 6}
SAVED_STD = torch.tensor([.25, .3, .2, .27, .8, .75])
NAMES = tuple(name for name in sampling.CASES for _ in range(4))


def observations(count=12):
    return TensorDict({"policy": torch.zeros(count, 35), "critic": torch.zeros(count, 81)}, batch_size=[count])


@pytest.fixture
def saved():
    args = deepcopy(CONFIG["actor"])
    args.pop("class_name")
    actor = MLPModel(observations(1), CONFIG["obs_groups"], "actor", 6, **args)
    with torch.no_grad():
        actor.distribution.std_param.copy_(SAVED_STD)
    return {"actor_state_dict": deepcopy(actor.state_dict())}


def actor_for(saved):
    return sampling.load_native_actor(saved, CONFIG, SOURCE, "cpu")


def flags(count=12, **lanes):
    masks = {name: [False] * count for name in ("done", "terminated", "timeouts", "success", "boundary", "blocked")}
    for name, ids in lanes.items():
        for lane in ids:
            masks[name][lane] = True
    return dict(done=masks["done"], terminated=masks["terminated"], timeouts=masks["timeouts"],
                success=masks["success"], reasons={"boundary": masks["boundary"], "blocked": masks["blocked"]})


def test_strict_load_preserves_saved_std_and_actor(saved):
    original = deepcopy(saved)
    actor = actor_for(saved)
    assert torch.equal(actor.distribution.std_param, SAVED_STD)
    assert all(torch.equal(value, actor.state_dict()[name]) for name, value in original["actor_state_dict"].items())
    assert all(torch.equal(value, saved["actor_state_dict"][name]) for name, value in original["actor_state_dict"].items())
    assert actor.distribution is not None and not actor.training


@pytest.mark.parametrize("change", ("missing_std", "nonfinite_std", "extra_key", "castable_dtype"))
def test_saved_actor_failures_are_not_silently_reinitialized(saved, change):
    changed = deepcopy(saved)
    if change == "missing_std":
        del changed["actor_state_dict"]["distribution.std_param"]
    elif change == "nonfinite_std":
        changed["actor_state_dict"]["distribution.std_param"][0] = float("nan")
    elif change == "extra_key":
        changed["actor_state_dict"]["unexpected"] = torch.zeros(1)
    else:
        changed["actor_state_dict"]["distribution.std_param"] = SAVED_STD.double()
    with pytest.raises((ValueError, RuntimeError)):
        actor_for(changed)


@pytest.mark.parametrize("which", ("ABI", "groups", "architecture", "distribution"))
def test_fixed_native_factory_rejects_changed_interface(saved, which):
    config, source = deepcopy(CONFIG), deepcopy(SOURCE)
    if which == "ABI":
        source["actor_dim"] = 34
    elif which == "groups":
        config["obs_groups"]["actor"] = ["critic"]
    elif which == "architecture":
        config["actor"]["hidden_dims"] = [128, 128, 64]
    else:
        config["actor"]["distribution_cfg"]["std_type"] = "log"
    with pytest.raises(ValueError):
        sampling.load_native_actor(saved, config, source, "cpu")


def test_native_private_sample_replay_and_global_rng_restoration(saved):
    actor = actor_for(saved)
    first = sampling.NativeActionSampler(actor, 93001, "cpu")
    second = sampling.NativeActionSampler(actor, 93001, "cpu")
    global_before = torch.random.get_rng_state().clone()
    for tick in range(2):
        a, b = first.draw(observations()), second.draw(observations())
        assert torch.equal(a["mu"], b["mu"])
        assert torch.equal(a["sample"], b["sample"])
        assert torch.equal(a["potential_sample_minus_mean"], a["sample"] - a["mu"])
        assert torch.equal(a["effective_std"], SAVED_STD.expand(12, -1))
        assert a["rng_before_sha256"] == b["rng_before_sha256"]
        assert a["rng_after_sha256"] == b["rng_after_sha256"]
        assert a["draw_counter"] == tick + 1
        assert a["sample_element_count"] == (tick + 1) * 72
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert torch.equal(actor.distribution.std_param, SAVED_STD)


def test_draw_uses_original_native_sampler_and_refreshes_mean(saved, monkeypatch):
    actor = actor_for(saved)
    obs = observations()
    expected = actor(obs, stochastic_output=False)
    actor.distribution.update(expected + 99.)  # Deterministic forward leaves this stale.
    native = actor.distribution.sample
    calls = []
    private = sampling.NativeActionSampler(actor, 123, "cpu")
    initial = private.generator.get_state().clone()

    def checked_native():
        assert torch.equal(actor.distribution.mean, expected)
        calls.append(True)
        return native()

    monkeypatch.setattr(actor.distribution, "sample", checked_native)
    result = private.draw(obs)
    assert calls == [True]
    # Same native Normal.sample with the same RNG state, not a custom epsilon draw.
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(initial)
        original_sample = native()
    assert torch.equal(result["sample"], original_sample)


def test_different_private_seed_changes_sample_not_mean(saved):
    actor = actor_for(saved)
    a = sampling.NativeActionSampler(actor, 1, "cpu").draw(observations())
    b = sampling.NativeActionSampler(actor, 2, "cpu").draw(observations())
    assert torch.equal(a["mu"], b["mu"])
    assert not torch.equal(a["sample"], b["sample"])


def test_effective_std_uses_native_clamp_without_mutating_raw_std(saved):
    changed = deepcopy(saved)
    raw = torch.tensor([-1., 0., 1e-8, 1e8, .2, .3])
    changed["actor_state_dict"]["distribution.std_param"] = raw
    actor = actor_for(changed)
    result = sampling.NativeActionSampler(actor, 9, "cpu").draw(observations())
    expected = raw.clamp(*actor.distribution.std_range).expand(12, -1)
    assert torch.equal(result["effective_std"], expected)
    assert torch.equal(actor.distribution.std_param, raw)


def test_sampling_exception_restores_global_rng_and_does_not_advance_receipt(saved, monkeypatch):
    actor = actor_for(saved)
    sampler = sampling.NativeActionSampler(actor, 93001, "cpu")
    global_before, private_before = torch.random.get_rng_state().clone(), sampler.generator.get_state().clone()

    def broken_sample():
        torch.randn(12, 6)
        raise RuntimeError("injected native sampling failure")

    monkeypatch.setattr(actor.distribution, "sample", broken_sample)
    with pytest.raises(RuntimeError, match="injected"):
        sampler.draw(observations())
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert torch.equal(private_before, sampler.generator.get_state())
    assert sampler.draw_counter == 0 and sampler.sample_element_count == 0


def test_ledger_accepts_actual_interleaved_case_order_and_fixed_N4():
    ledger = sampling.FirstCohortLedger(sampling.CASES * 4)
    summary = ledger.summary(physical_tick=0)
    assert ledger.live == (True,) * 12
    assert summary["fixed_birth_denominator"] == 12
    for name in sampling.CASES:
        assert summary["cases"][name]["fixed_birth_denominator"] == 4
    with pytest.raises(ValueError):
        sampling.FirstCohortLedger(NAMES[:-1])


def test_first_native_terminal_locks_out_repeated_done_and_censors_actual_time():
    ledger = sampling.FirstCohortLedger(NAMES)
    observed = flags(done=[0, 4, 8, 9], terminated=[0], success=[4], timeouts=[8, 9], boundary=[8])
    assert ledger.observe(**observed, physical_tick=4) == (0, 4, 8, 9)
    assert ledger.observe(**observed, physical_tick=8) == ()
    summary = ledger.summary(physical_tick=9)
    assert summary["actual_simulated_seconds"] == .045
    assert summary["partitions"] == {"failure": 1, "success": 1, "boundary_truncation": 1,
                                     "timeout": 1, "right_censored_unfinished": 8}
    assert sum(summary["partitions"].values()) == 12
    assert summary["rows"][0]["terminal"]["physical_tick"] == 4
    assert summary["rows"][0]["exposure_seconds"] == .02
    assert summary["rows"][1]["censor_time_s"] == .045
    assert ledger.live[0] is False and ledger.live[1] is True
    assert ledger.rows[1]["partition"] is None  # summary does not close a live slot.


def test_native_20s_timeout_is_not_external_right_censor():
    ledger = sampling.FirstCohortLedger(NAMES)
    ledger.observe(**flags(done=list(range(12)), timeouts=list(range(12))), physical_tick=4000)
    summary = ledger.summary(physical_tick=4000)
    assert summary["partitions"]["timeout"] == 12
    assert summary["partitions"]["right_censored_unfinished"] == 0
    assert all(row["terminal"]["time_s"] == 20. for row in summary["rows"])


def test_native_failure_precedence_keeps_overlapping_original_flags():
    ledger = sampling.FirstCohortLedger(NAMES)
    ledger.observe(**flags(done=[0], terminated=[0], success=[0], timeouts=[0], boundary=[0]), physical_tick=4)
    row = ledger.summary(physical_tick=4)["rows"][0]
    assert row["partition"] == "failure"
    assert row["terminal"]["success"] is True and row["terminal"]["reasons"]["boundary"] is True


def test_zero_completed_windows_are_null_and_partial_values_never_zero_fill():
    ledger = sampling.FirstCohortLedger(NAMES)
    empty = ledger.summary(physical_tick=4)
    for case in empty["cases"].values():
        for window in case["windows"].values():
            assert window["complete_count"] == 0 and window["denominator"] == 4
            assert window["mean"] is None
    ledger.record_window(0, "stand", .02, complete=True, observed_seconds=13.)
    ledger.record_window(1, "stand", .13, complete=False, observed_seconds=5.)
    window = ledger.summary(physical_tick=4000)["cases"]["stand_305mm"]["windows"]["stand"]
    assert window["complete_count"] == 1 and window["complete_fraction"] == .25
    assert window["recorded_count"] == 2 and window["mean"] == .02
    assert window["completed_observed_seconds"] == 13.
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record_window(0, "stand", .03, complete=True, observed_seconds=13.)


def test_clock_and_native_done_inconsistency_rejected():
    ledger = sampling.FirstCohortLedger(NAMES)
    with pytest.raises(ValueError, match="missing done"):
        ledger.observe(**flags(terminated=[0]), physical_tick=4)
    ledger.observe(**flags(), physical_tick=4)
    with pytest.raises(ValueError, match="advance"):
        ledger.observe(**flags(), physical_tick=4)
    with pytest.raises(ValueError, match="summary clock"):
        ledger.summary(physical_tick=3)


def high_names():
    return tuple(name for name in get_suite(HIGH_SPEED_SUITE).case_names for _ in range(4))


def test_high_suite_fixed_first_cohort_and_partial_windows_are_not_reopened():
    suite = get_suite(HIGH_SPEED_SUITE)
    names = high_names()
    ledger = sampling.FirstCohortLedger(names, suite_name=HIGH_SPEED_SUITE)
    assert len(names) == 64 and suite.physics_steps == 6400
    observed = flags(count=64, done=[0, 8, 9, 10], terminated=[0], success=[8],
                     timeouts=[9, 10], boundary=[9])
    assert ledger.observe(**observed, physical_tick=1600) == (0, 8, 9, 10)
    assert ledger.observe(**observed, physical_tick=2000) == ()
    ledger.record_window(0, "stand", .2, complete=False, observed_seconds=6.)
    ledger.record_window(8, "stop", .04, complete=False, observed_seconds=1.)
    summary = ledger.summary(physical_tick=2400)
    assert summary["suite_name"] == HIGH_SPEED_SUITE and summary["suite"] == suite.as_dict()
    assert summary["fixed_birth_denominator"] == 64
    assert tuple(summary["cases"]) == suite.case_names
    assert summary["partitions"] == {"failure": 1, "success": 1, "boundary_truncation": 1,
                                      "timeout": 1, "right_censored_unfinished": 60}
    assert summary["rows"][0]["exposure_seconds"] == 8.
    assert summary["rows"][1]["censor_time_s"] == 12.
    for name, case in summary["cases"].items():
        assert case["fixed_birth_denominator"] == 4
        assert set(case["windows"]) == set(suite.primary_windows[name])
        assert all(window["complete_count"] == 0 and window["mean"] is None
                   for window in case["windows"].values())
    assert ledger.rows[1]["partition"] is None


def test_high_native_32s_timeout_and_clock_bound():
    ledger = sampling.FirstCohortLedger(high_names(), suite_name=HIGH_SPEED_SUITE)
    ledger.observe(**flags(count=64, done=list(range(64)), timeouts=list(range(64))), physical_tick=6400)
    summary = ledger.summary(physical_tick=6400)
    assert summary["partitions"]["timeout"] == 64
    assert summary["partitions"]["right_censored_unfinished"] == 0
    assert summary["actual_simulated_seconds"] == 32.
    with pytest.raises(ValueError, match="within named suite"):
        ledger.observe(**flags(count=64), physical_tick=6401)
    with pytest.raises(ValueError, match="summary clock"):
        ledger.summary(physical_tick=6401)


@pytest.mark.parametrize("change", ("unknown_suite", "missing", "renamed", "duplicate", "wrong_default", "replicas"))
def test_catalog_identity_cannot_expand_cases_or_population(change):
    names = list(high_names())
    kwargs = {"suite_name": HIGH_SPEED_SUITE}
    if change == "unknown_suite":
        kwargs["suite_name"] = "unregistered_speed_suite"
    elif change == "missing":
        names.pop()
    elif change == "renamed":
        names[:4] = ["unregistered_stand"] * 4
    elif change == "duplicate":
        names[:4] = names[4:8]
    elif change == "wrong_default":
        kwargs["suite_name"] = DEFAULT_SUITE
    else:
        kwargs["replicas"] = 5
    with pytest.raises(ValueError):
        sampling.FirstCohortLedger(names, **kwargs)


def test_catalog_primary_windows_and_duration_cannot_expand():
    ledger = sampling.FirstCohortLedger(high_names(), suite_name=HIGH_SPEED_SUITE)
    with pytest.raises(ValueError, match="not declared"):
        ledger.record_window(0, "stop", .2, complete=True, observed_seconds=10.)
    with pytest.raises(ValueError, match="not declared"):
        ledger.record_window(8, "made_up_window", .2, complete=True, observed_seconds=10.)
    with pytest.raises(ValueError, match="duration"):
        ledger.record_window(0, "stand", .2, complete=True, observed_seconds=33.)
    assert ledger.rows[0]["windows"] == {}


def test_native_saved_actor_sampling_supports_named_high_population_without_state_change(saved):
    actor = actor_for(saved)
    before = deepcopy(actor.state_dict())
    sampler = sampling.NativeActionSampler(actor, 93001, "cpu")
    result = sampler.draw(observations(64))
    assert result["mu"].shape == result["sample"].shape == result["effective_std"].shape == (64, 6)
    assert sampler.sample_element_count == 384 and sampler.draw_counter == 1
    assert all(torch.equal(value, actor.state_dict()[key]) for key, value in before.items())
