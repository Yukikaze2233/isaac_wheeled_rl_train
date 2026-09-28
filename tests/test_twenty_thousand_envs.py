"""The 20k entry must keep actual-update budgets and PPO batch divisibility."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    def load(name):
        return json.loads((ROOT / name).read_text())

    plan = resolve_plan(load('contracts/v6_scut35_budget_retry_v1.json'), load)
    plan.update(target_num_envs=20000, ppo_minibatch_samples=48000)
    # Unified evaluation materializes later stages too; their explicit caps
    # must belong to the same new 20k design instead of an inherited 16k plan.
    for stage in plan['stages']:
        stage['num_envs'] = 20000
    recipe = {**deepcopy(plan['stages'][0]), 'num_envs': 20000, 'updates': 1000}
    return load(plan['base_contract']), plan, recipe


def test_actual_update_budget_and_ten_minibatches_at_twenty_thousand():
    base, plan, recipe = inputs()
    config = stage_contract(base, plan, recipe, 20000)
    assert config['target_num_envs'] == 20000
    assert config['total_updates'] == 1000
    assert config['num_steps_per_env'] == 24
    assert config['num_mini_batches'] == 10
    assert config['target_num_envs'] * config['num_steps_per_env'] == 480000


def test_old_minibatch_size_cannot_silently_round_at_twenty_thousand():
    base, plan, recipe = inputs()
    plan['ppo_minibatch_samples'] = 49152
    with pytest.raises(ValueError):
        stage_contract(base, plan, recipe, 20000)


def test_explicit_stage_ceiling_still_rejects_unsupported_counts():
    base, plan, recipe = inputs()
    recipe['num_envs'] = 20001
    with pytest.raises(ValueError, match='Stage environment ceiling'):
        stage_contract(base, plan, recipe, 20001)
