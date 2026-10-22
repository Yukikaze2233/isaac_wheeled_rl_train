"""Exercise the scene readback against the actual source command clock."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.tasks.commands.flat_command_sampling import ScutFlatCommands


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def entry():
    saved = sys.path[:]
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("nonsteady_scene_cpu_entry", ROOT / "src/wheeled_tasks/chassis/application/training/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path[:] = saved


def source_payload(step, *, reset=False):
    commands = ScutFlatCommands(2, "cpu")
    generator = torch.Generator().manual_seed(91)
    commands.reset(torch.arange(2), torch.zeros(2), torch.tensor([True, False]), generator)
    for _ in range(step):
        commands.begin_policy()
        for _ in range(4):
            commands.advance_physics(.005)
    owner = {name: value.clone() for name, value in commands.diagnostics().items()}
    owner["episode_id"] = torch.zeros(2, dtype=torch.long)
    terminated = torch.zeros(2, dtype=torch.bool)
    done = torch.tensor([False, reset])
    learning = commands.learning_mask.clone()
    if reset:
        commands.reset(torch.tensor([1]), torch.zeros(1), torch.ones(1, dtype=torch.bool), generator)
    after = {name: value.clone() for name, value in commands.diagnostics().items()}
    after["episode_id"] = done.long()
    info = {"diagnostics": {"source_flat_episode": owner, "terminated": terminated},
            "returned_episode_state": {"source_flat_episode": after, "reset_mask": done.clone()},
            "policy_learning_mask": learning,
            "policy_bootstrap_mask": commands.bootstrap_mask(terminated)}
    observations = {"policy": torch.zeros(2, 35), "critic": torch.zeros(2, 78)}
    return SimpleNamespace(num_envs=2, physical_tick=4 * step), observations, done, info


@pytest.mark.parametrize("step", [1, 9, 10, 11, 12])
def test_source_gate_and_returned_bootstrap_have_distinct_meanings(entry, step):
    env, observations, done, info = source_payload(step)
    frame = entry.nonsteady_scene_validation_frame(env, observations, done, info, step)
    assert bool(frame["policy_learning_mask"][0]) is (step > 10)
    assert bool(frame["returned_bootstrap_mask"][0]) is (step >= 10)
    if step == 10:
        assert not bool(frame["pre_reset_owner"]["holding"][0])
        assert not bool(frame["policy_learning_mask"][0])
        assert bool(frame["returned_bootstrap_mask"][0])
    observations["critic"].fill_(42)
    assert not bool(frame["observations"]["critic"].any())


def test_newborn_owner_learning_mask_does_not_replace_bootstrap(entry):
    env, observations, done, info = source_payload(11, reset=True)
    frame = entry.nonsteady_scene_validation_frame(env, observations, done, info, 11)
    assert bool(frame["policy_learning_mask"][1])
    assert bool(frame["returned_owner"]["learning_mask"][1])
    assert bool(frame["returned_owner"]["holding"][1])
    assert not bool(frame["returned_bootstrap_mask"][1])
    assert bool(frame["returned_reset_mask"][1])


def test_mixed_scene_combines_recovery_ownership_without_weakening_flat_gate(entry):
    env, observations, done, info = source_payload(11)
    info["diagnostics"]["source_flat_episode"]["scripted_recovery_birth"] = torch.tensor([False, True])
    info["diagnostics"]["recovery_interval_learning_mask"] = torch.tensor([True, False])
    info["returned_episode_state"]["recovery_bootstrap_mask"] = torch.tensor([True, False])
    info["policy_learning_mask"][1] = False
    info["policy_bootstrap_mask"][1] = False
    frame = entry.nonsteady_scene_validation_frame(env, observations, done, info, 11)
    assert frame["policy_learning_mask"].tolist() == [True, False]
    assert frame["ownership_components"]["recovery_interval_learning_mask"].tolist() == [True, False]
    info["policy_learning_mask"][1] = True
    with pytest.raises(RuntimeError, match="ownership masks differ"):
        entry.nonsteady_scene_validation_frame(env, observations, done, info, 11)


def test_native_noise_readback_uses_actual_encoded_actor_and_clean_critic(entry):
    import json
    from wheeled_tasks.chassis.observations.actor_sensor_noise import ActorSensorNoise
    from wheeled_tasks.chassis.observations.actor import build_sensor18

    config = json.loads((ROOT / "configs/v7_formal_20261012/motion_retention_review.json").read_text())
    env, observations, done, info = source_payload(11)
    env.actor_sensor_noise = ActorSensorNoise(2, "cpu", config, torch.Generator().manual_seed(91))
    env.v5 = SimpleNamespace(nominal=torch.zeros(6))
    env._actor_sensor_noise_clean_frame = torch.zeros(2, 35)
    noisy = env.actor_sensor_noise.apply(torch.zeros(2, 18), 11)
    observations["policy"][:, 4:22] = build_sensor18(noisy[:, 12:15], noisy[:, 15:18],
        noisy[:, :6], noisy[:, 6:12], env.v5.nominal)
    frame = entry.nonsteady_scene_validation_frame(env, observations, done, info, 11)
    assert frame["actor_sensor_noise"]["noise_vector_draws"] == 2
    assert torch.equal(frame["actor_sensor_noise"]["sensor_output18"], noisy)
    for name, column in (("critic", 7), ("policy", 0), ("policy", 24)):
        observations[name][0, column] = .25
        with pytest.raises(RuntimeError, match="isolated to the actor sensor slice"):
            entry.nonsteady_scene_validation_frame(env, observations, done, info, 11)
        observations[name][0, column] = 0.


@pytest.mark.parametrize("mutate", [
    lambda obs, info: obs.update(critic=torch.zeros(2, 81)),
    lambda obs, info: obs.update(policy=torch.zeros(2, 35, dtype=torch.float64)),
    lambda obs, info: obs["critic"].fill_(float("nan")),
    lambda obs, info: info["policy_learning_mask"].fill_(True),
    lambda obs, info: info["policy_bootstrap_mask"].zero_(),
])
def test_invalid_abi_or_owner_masks_are_rejected_before_publishing(entry, mutate):
    env, observations, done, info = source_payload(10)
    mutate(observations, info)
    with pytest.raises(RuntimeError):
        entry.nonsteady_scene_validation_frame(env, observations, done, info, 10)
