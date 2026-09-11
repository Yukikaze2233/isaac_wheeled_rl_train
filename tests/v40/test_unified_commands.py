"""Unified command sampling and transition regressions using actual env methods."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.v40.core import load_contract, make_run_manifest, validate_contract
from wheeled_tasks.v40.contract import audit_asset
from wheeled_algo.v40_metrics import InvalidTrajectory, default_cases, validate_command
from test_env_contract_static import isolated_tensor_method, load_cli
from test_export import make_checkpoint
from test_round2 import V2_PATH, adapter


@pytest.fixture
def contract():
    return load_contract(V2_PATH)


def sampler(contract, count):
    _, _, sample = isolated_tensor_method("_sample_commands")
    env = SimpleNamespace(
        contract=contract, cfg=SimpleNamespace(stage="locomotion"), device="cpu",
        commands=torch.full((count, 3), -99.), _command_period_ticks=300,
        _command_ticks_left=torch.full((count,), 7, dtype=torch.long),
        _commands_due=torch.ones(count, dtype=torch.bool),
    )
    return env, sample


@pytest.mark.parametrize("probability", [-.01, 1.01, True, "0.1", float("nan"), float("inf")])
def test_invalid_standing_probability_rejected(contract, probability):
    contract["commands"]["stages"]["locomotion"]["standing_probability"] = probability
    with pytest.raises(ValueError, match="probability"):
        validate_contract(contract)


def test_seeded_mixture_probability_and_height_are_independent(contract):
    env, sample = sampler(contract, 10000)
    torch.manual_seed(41)
    sample(env, torch.arange(10000))
    standing = (env.commands[:, :2] == 0).all(-1)
    assert .08 < standing.float().mean() < .12
    assert env.commands[:, :2].abs().le(2).all()
    assert env.commands[:, 2].ge(.28).all() and env.commands[:, 2].le(.32).all()
    assert env.commands[standing, 2].std() > .005
    assert env.commands[~standing, 2].std() > .005
    original = env.commands.clone()
    torch.manual_seed(41)
    sample(env, torch.arange(10000))
    torch.testing.assert_close(env.commands, original, atol=0, rtol=0)


def test_all_standing_partial_resample_preserves_other_envs_and_height(contract):
    contract["commands"]["stages"]["locomotion"]["standing_probability"] = 1.
    env, sample = sampler(contract, 5)
    ids = torch.tensor([1, 4])
    sample(env, ids)
    assert env.commands[ids, :2].eq(0).all()
    assert env.commands[ids, 2].ge(.28).all()
    assert env.commands[[0, 2, 3]].eq(-99).all()
    assert env._command_ticks_left.tolist() == [7, 300, 7, 7, 300]
    assert env._commands_due.tolist() == [True, False, True, True, False]
    rng = torch.random.get_rng_state().clone()
    sample(env, torch.tensor([], dtype=torch.long))
    assert torch.equal(rng, torch.random.get_rng_state())


def test_old_contracts_and_zero_probability_preserve_uniform_rng(contract):
    old_v2 = deepcopy(contract)
    old_v2["commands"]["stages"]["locomotion"].pop("standing_probability")
    contract["commands"]["stages"]["locomotion"]["standing_probability"] = 0.
    for config in (load_contract(), old_v2, contract):
        validate_contract(config)
        env, sample = sampler(config, 5)
        torch.manual_seed(42)
        expected = torch.stack([torch.rand(5) for _ in range(3)], dim=1)
        expected_rng = torch.random.get_rng_state().clone()
        for column, key in enumerate(("vx", "wz", "height")):
            low, high = config["commands"]["stages"]["locomotion"][key]
            expected[:, column] = low + (high - low) * expected[:, column]
        torch.manual_seed(42)
        sample(env, torch.arange(5))
        torch.testing.assert_close(env.commands, expected, atol=0, rtol=0)
        assert torch.equal(expected_rng, torch.random.get_rng_state())


def test_evaluation_override_bypasses_standing_mixture_and_rng(contract):
    contract["commands"]["stages"]["locomotion"]["standing_probability"] = 1.
    env, sample = sampler(contract, 2)
    env._evaluation_command_override = (1., -.5, .30)
    rng = torch.random.get_rng_state().clone()
    sample(env, torch.arange(2))
    torch.testing.assert_close(env.commands, torch.tensor([[1., -.5, .30]]).repeat(2, 1))
    assert torch.equal(rng, torch.random.get_rng_state())


def test_move_stop_move_keeps_history_and_rewards_use_executed_command(contract):
    contract["commands"]["stages"]["locomotion"]["standing_probability"] = 1.
    env = adapter(contract)
    env.commands[0] = torch.tensor([1., .5, .30])
    before = env._get_observations()["policy"].reshape(2, 5, 25).clone()
    env._command_ticks_left[:] = torch.tensor([1, 9])
    env._last_reward_tick = -1
    env.reset_terminated = torch.zeros(2, dtype=torch.bool)
    _, core, get_rewards = isolated_tensor_method("_get_rewards")
    old_commands = env.commands.clone()
    q, _ = env._joint_state()
    data = env.robot.data
    terms = core.compute_reward_terms(data.root_lin_vel_b, data.root_ang_vel_b,
        data.projected_gravity_b, env._base_height(), old_commands, env.actions,
        env.previous_actions, env.torques, q, contract)
    torch.testing.assert_close(get_rewards(env), sum(terms.values()))
    torch.testing.assert_close(env.commands, old_commands)
    assert env.extras["log"]["Command/standing_fraction"] == .5
    assert env._commands_due.tolist() == [True, False]
    env.common_step_counter += 1
    stopped = env._get_observations()["policy"].reshape(2, 5, 25)
    torch.testing.assert_close(stopped[:, :-1], before[:, 1:])
    assert stopped[0, -1, 6:8].eq(0).all()
    assert stopped[0, -1, 8] >= 1.4  # Height command was not cleared.
    torch.testing.assert_close(env.commands[1], old_commands[1])
    rng = torch.random.get_rng_state().clone()
    torch.testing.assert_close(env._get_observations()["policy"], stopped.reshape(2, 125))
    assert torch.equal(rng, torch.random.get_rng_state())
    contract["commands"]["stages"]["locomotion"]["standing_probability"] = 0.
    env._commands_due[0] = True
    env.common_step_counter += 1
    moving = env._get_observations()["policy"].reshape(2, 5, 25)
    torch.testing.assert_close(moving[:, :-1], stopped[:, 1:])
    assert moving[0, -1, 6:8].ne(0).any()


def test_evaluation_mixture_domain_and_stationary_height_coverage(contract):
    cases = default_cases(contract, "locomotion")
    commands = [tuple(case["command"]) for case in cases]
    assert len(commands) == len(set(commands))
    assert {(0., 0., .28), (0., 0., .30), (0., 0., .32)} <= set(commands)
    bounds = contract["commands"]["stages"]["locomotion"]
    bounds["vx"], bounds["wz"] = [1., 2.], [.5, 1.]
    validate_contract(contract)
    assert validate_command(contract, "locomotion", (0., 0., .30)) == (0., 0., .30)
    with pytest.raises(InvalidTrajectory):
        validate_command(contract, "locomotion", (0., .5, .30))
    bounds["standing_probability"] = 0.
    with pytest.raises(InvalidTrajectory):
        validate_command(contract, "locomotion", (0., 0., .30))


def test_old_v2_checkpoint_cannot_silently_resume_new_mixture(contract, tmp_path):
    old = deepcopy(contract)
    old["commands"]["stages"]["locomotion"].pop("standing_probability")
    asset = audit_asset(contract)
    manifest = make_run_manifest(old, asset)
    checkpoint, _ = make_checkpoint(manifest)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_cli().checked_checkpoint(checkpoint_path, make_run_manifest(contract, asset))
