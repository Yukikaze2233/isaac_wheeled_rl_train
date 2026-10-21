"""Explicit final rough learning-state admission on CPU; no course release.

The frozen runner configuration and weight loader are AST oracles. Current RSL
loading is exercised without collecting native observations or PPO updates.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from tests.application.training.test_formal_stage_finetune_entry import algorithm, execute_actual_resume_load
from wheeled_tasks.chassis.application.training.policy_transfer import formal_stage_load_kwargs
from wheeled_tasks.chassis.application.configuration.training_profile import (
    ALGORITHM, FINAL_ROUGH_OPTIMIZER, FINAL_ROUGH_STAGE_VERSION, FINAL_RUNNER_REFERENCE,
    FINAL_RUNNER_SOURCE_COMMIT, configure_runner, effective_profile_algorithm,
    profile_identity, profile_source_files, restore_profile_clocks, restore_rng,
    validate_formal_learning_state_resume, validate_profile_selection,
    verify_learning_rate, verify_runner,
)


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "configs/v7_formal_20261012"
REFERENCE = ROOT / FINAL_RUNNER_REFERENCE


class LayoutEnv:
    """Only the observed35/78 layout needed to construct the CPU runner."""
    cfg = {}
    num_envs = 2
    num_actions = 6
    device = "cpu"

    def get_observations(self):
        return TensorDict({"policy": torch.zeros(2, 35), "critic": torch.zeros(2, 78)}, batch_size=[2])
















def test_frozen_loader_restores_actor_critic_std_without_optimizer_or_old_iteration(tmp_path):
    path = REFERENCE / "train_finetune.py"
    loader = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)
                  and node.name == "_load_policy_for_finetune")
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(loader)], type_ignores=[])), str(path), "exec"), scope)
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Linear(35, 6)
            self.critic = torch.nn.Linear(78, 1)
            self.std = torch.nn.Parameter(torch.ones(6))
    source, target = Policy(), Policy()
    with torch.no_grad():
        for value in source.parameters():
            value.fill_(.37)
    checkpoint = tmp_path / "source-policy.pt"
    torch.save({"model_state_dict": source.state_dict(), "optimizer_state_dict": {"sentinel": 1}, "iter": 273}, checkpoint)
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-4)
    runner = NS(device="cpu", alg=NS(policy=target, optimizer=optimizer, learning_rate=1e-4), current_learning_iteration=19)
    scope["_load_policy_for_finetune"](runner, str(checkpoint))
    assert all(torch.equal(value, source.state_dict()[name]) for name, value in target.state_dict().items())
    assert not optimizer.state and runner.alg.learning_rate == optimizer.param_groups[0]["lr"] == 1e-4
    assert runner.current_learning_iteration == 0


def test_inherit_rng_uses_saved_torch_and_command_streams():
    env = NS(generator=torch.Generator().manual_seed(17), round_trip_transport=None)
    rng = torch.Generator().manual_seed(31).get_state()
    command_rng = torch.Generator().manual_seed(47).get_state()
    before = torch.get_rng_state()
    try:
        receipt = restore_rng(torch, env, {"rng_state": rng, "env_rng_state": command_rng}, seed=99)
        assert receipt["mode"] == "inherit" and receipt["parent_rng_restored"] is True
        assert torch.equal(torch.get_rng_state(), rng) and torch.equal(env.generator.get_state(), command_rng)
    finally:
        torch.set_rng_state(before)
