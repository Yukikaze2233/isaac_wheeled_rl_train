"""Shared CPU fixtures for current runtime tests."""
from types import SimpleNamespace
import torch
from support_flat_episode_env_wiring import TREE
from support_flat_episode_runtime import component_config
from wheeled_tasks.chassis.flat_episode_runtime import FlatEpisodeRuntime

def make_config(seed=17):
    config = component_config()
    config.update(record_diagnostics=True, auto_reset=True, physics_dt=0.005, policy_dt=0.02, episode_seconds=20, seed=seed)
    return config

def make_env(setup):
    seed = setup.protocol['environment_seed']
    count = len(setup.lane_case_ids)
    return SimpleNamespace(cfg=make_config(seed), num_envs=count, physical_tick=0, generator=torch.Generator().manual_seed(seed), startup_report={}, stage_actor_update=0, global_actor_update=0, episode_length_buf=torch.zeros(count, dtype=torch.long), flat_episode=FlatEpisodeRuntime(count, 'cpu', None, None, TREE, birth_rejection={'mode': 'abort'}))
