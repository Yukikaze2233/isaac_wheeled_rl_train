"""Rough scene readback preserves the two sides of native autoreset."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import time

import pytest
import torch


def load_entry():
    path = Path(__file__).resolve().parents[2] / "src/wheeled_tasks/chassis/application/training/train.py"
    spec = importlib.util.spec_from_file_location("rough_scene_entry", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry, path


def scene_env(*, initial_level=0, interior_births=0):
    grid = torch.zeros(10, 14, 3)
    grid[:, :, 0] = ((torch.arange(10) - 4.5) * 9)[:, None]
    grid[:, :, 1] = ((torch.arange(14) - 6.5) * 9)[None, :]
    origins = grid[0, :8].clone()
    origins[-2:] = torch.tensor([[-2., 300., 0.], [2., 300., 0.]])
    origins[:interior_births] = grid[1, 1:1 + interior_births]
    root_pose = torch.zeros(8, 7)
    root_pose[:, :3] = origins
    root_pose[:, 2] += .32
    root_pose[:, 6] = 1.
    reset_calls = []
    env = NS(tick=0, physical_tick=0, decimation=4, num_envs=8,
        cfg={"formal_terrain": {"type": "rough"}, "formal_rough_initial_level": initial_level},
        shared_terrain=NS(importer=NS(terrain_origins=grid)), origins=origins,
        reset_positions=root_pose[:, :3].clone(),
        flat_episode=NS(episode_id=torch.arange(8)),
        robot=NS(data=NS(root_link_pose_w=NS(torch=root_pose))),
        reset=lambda ids: reset_calls.append(ids.clone()))
    return env, reset_calls


@pytest.mark.parametrize("interior_births", [0, 1])
def test_declared_row0_preserves_births_when_seam_probe_is_unavailable(interior_births):
    entry, _ = load_entry()
    env, resets = scene_env(interior_births=interior_births)
    before = [value.clone() for value in (env.origins, env.reset_positions,
        env.robot.data.root_link_pose_w.torch, env.flat_episode.episode_id)]
    result = entry.prepare_rough_scene_validation(env)
    assert result["seam_probe_status"] == "skipped"
    assert result["seam_probe_eligible_births"] == interior_births
    assert result["births_preserved"] and result["seam_probe_env_ids"] == []
    assert not resets
    for old, new in zip(before, (env.origins, env.reset_positions,
            env.robot.data.root_link_pose_w.torch, env.flat_episode.episode_id)):
        assert torch.equal(old, new)


@pytest.mark.parametrize("initial_level", [None, False, 0., 1])
def test_unavailable_seam_births_still_fail_without_explicit_integer_row0(initial_level):
    entry, _ = load_entry()
    env, resets = scene_env(initial_level=initial_level)
    with pytest.raises(ValueError, match="two robots on interior tiles"):
        entry.prepare_rough_scene_validation(env)
    assert not resets


def test_available_interior_births_keep_original_seam_probe():
    entry, _ = load_entry()
    env, resets = scene_env(initial_level=None, interior_births=2)
    origins = env.origins.clone()
    old_positions = env.reset_positions.clone()
    result = entry.prepare_rough_scene_validation(env)
    assert result["seam_probe_env_ids"] == [0, 1]
    assert len(resets) == 1 and resets[0].tolist() == [0, 1]
    torch.testing.assert_close(env.reset_positions[:2, :2],
        origins[:2, :2] + torch.tensor([[4.6, 0.], [0., 4.6]]))
    assert torch.equal(env.origins, origins)
    assert torch.equal(env.reset_positions[2:], old_positions[2:])


def test_skipped_seam_probe_does_not_skip_existing_scene_step_loop(tmp_path):
    entry, path = load_entry()
    env, resets = scene_env()
    calls = []

    def step(action):
        calls.append(action.clone())
        env.tick += 1
        env.physical_tick += 4
        return {}, torch.zeros(8), torch.zeros(8, dtype=torch.bool), {"diagnostics": {}}

    env.step = step
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    branch = next(node for node in ast.walk(main) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "args.validate_scene"
        and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == "scene_started" for target in child.targets) for child in node.body))
    scope = {"env": env, "c": env.cfg, "torch": torch, "time": time, "report": {},
        "args": NS(validate_scene=1, num_envs=8, device="cpu", run_dir=tmp_path),
        "budget": NS(check=lambda: None), "budget_owner": None,
        "prepare_rough_scene_validation": entry.prepare_rough_scene_validation,
        "rough_scene_validation_frame": lambda *args: {"readback_called": True}}
    exec(compile(ast.Module(body=branch.body, type_ignores=[]), str(path), "exec"), scope)
    assert not resets and len(calls) == 1 and not calls[0].any()
    assert scope["report"]["status"] == "scene_validated"
    assert scope["report"]["scene_diagnostics"]["steps_completed"] == 1
    assert scope["report"]["rough_scene_validation"]["seam_probe_status"] == "skipped"
    assert scope["report"]["rough_scene_validation"]["readback_files"] == ["rough_scene_step_0001.pt"]
    assert torch.load(tmp_path / "rough_scene_step_0001.pt", weights_only=True)["readback_called"]


def test_terminal_diagnostic_is_not_paired_with_newborn_critic_height():
    from wheeled_tasks.chassis.tasks.rewards.flat_reward import ROUGH_WEIGHTS

    path = Path(__file__).resolve().parents[2] / "src/wheeled_tasks/chassis/application/training/train.py"
    spec = importlib.util.spec_from_file_location("rough_scene_entry", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    pre_episode = torch.tensor([7, 7])
    returned_episode = torch.tensor([7, 8])
    done = torch.tensor([False, True])
    root_pose = torch.zeros(2, 7)
    root_pose[:, 2] = torch.tensor([.6, .4])
    env = NS(physical_tick=8, robot=NS(data=NS(root_link_pose_w=NS(torch=root_pose))))
    source = {"terrain_key_ids": torch.tensor([0, 2]),
        "placement_raise_m": torch.tensor([.001, .01]),
        "minimum_shape_clearance_m": torch.tensor([.001, .001])}
    diagnostics = {"episode_id": pre_episode, "episode_tick": torch.tensor([2, 2]),
        "root_link_position_world": torch.tensor([[0., 0., .6], [9., 0., 1.8]]),
        "root_height_world": torch.tensor([.6, 1.8]),
        "reward_ground_height_world": torch.tensor([.2, 1.4]),
        "rough_base_scan_valid_hit_count": torch.tensor([9, 8]),
        "height": torch.tensor([.4, .4]), "boundary": torch.tensor([False, False]),
        "measurement_valid_mask": torch.tensor([True, True]),
        "source_flat_episode": source, "reasons": {"numeric": torch.tensor([False, False])}}
    returned = {"episode_id": returned_episode, "episode_tick": torch.tensor([2, 0]),
        "reset_mask": done, "source_flat_episode": source}
    critic = torch.zeros(2, 78)
    critic[:, 31] = root_pose[:, 2] * 5
    policy = torch.zeros(2, 35)
    payments = {name: torch.tensor([i, i + 1.]) * .001 for i, name in enumerate(ROUGH_WEIGHTS)}
    diagnostics["reward"] = sum(payments.values())
    info = {"diagnostics": diagnostics, "returned_episode_state": returned,
        "native_reward_components": payments,
        "native_reward_component_units": "already_dt_scaled_payment_per_policy_step",
        "log": {"/reward/" + name: value.mean() for name, value in payments.items()}}
    info["log"]["/reward/total"] = diagnostics["reward"].mean()
    frame = entry.rough_scene_validation_frame(env, {"policy": policy, "critic": critic}, done, info, 2)
    assert frame["same_episode_mask"].tolist() == [True, False]
    assert frame["pre_reset"]["episode_id"].tolist() == [7, 7]
    assert frame["returned"]["episode_id"].tolist() == [7, 8]
    assert frame["pre_reset"]["root_height_world"][1].item() > 1.
    assert frame["returned"]["root_height_world"][1].item() < 1.
    assert torch.equal(frame["returned"]["critic_height_minus_expected"], torch.zeros(2))
    assert frame["critic_height_index"] == 31
    assert frame["physical_tick"] == 8
    assert frame["observations"] == {"policy": {"shape": [2, 35], "finite": True},
                                      "critic": {"shape": [2, 78], "finite": True}}
    assert frame["reward_payments"]["component_count"] == 26
    assert frame["reward_payments"]["component_means"] == {
        name: float(value.mean()) for name, value in payments.items()}
    assert frame["reward_payments"]["per_lane_payment_sum_max_abs_error"] == 0.
    assert abs(frame["reward_payments"]["component_mean_sum_minus_total_mean"]) < 1.e-6
    policy[0, 0], critic[1, 5] = float("nan"), float("inf")
    invalid = entry.rough_scene_validation_frame(env, {"policy": policy, "critic": critic}, done, info, 2)
    assert not invalid["observations"]["policy"]["finite"]
    assert not invalid["observations"]["critic"]["finite"]
    # Later native/cache writes must not mutate the already captured frame.
    root_pose[:, 2].zero_()
    pre_episode.zero_()
    assert frame["pre_reset"]["episode_id"].tolist() == [7, 7]
    assert bool((frame["returned"]["root_height_world"] > 0).all())
