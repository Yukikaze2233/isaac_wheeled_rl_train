"""Replay production command sampling on 20k CPU lanes, without Isaac or PPO.

This verifies clocks/coverage and full episode lengths, not motor behavior,
physical parking or reward convergence. Run with PYTHONPATH=src.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from wheeled_tasks.chassis.skill_commands import SkillCommands
from wheeled_tasks.chassis.task import choose_scene_groups


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/v6_s1_landing_repair_worker_v1.json"
torch.set_num_threads(1)


def environment(stagger):
    cfg = json.loads(CONTRACT.read_text())
    if stagger:
        cfg["new_asset_curriculum"]["start_stop_start_delay_seconds"] = [0., 8.]
    count = 20000
    pose = torch.zeros(count, 7)
    pose[:, 6] = 1.
    generator = torch.Generator().manual_seed(290927)
    e = SimpleNamespace(cfg=cfg, num_envs=count, device="cpu",
        scene_groups=[g for g, _ in choose_scene_groups(cfg["scene_groups"], count)],
        commands=torch.zeros(count, 3), command_target=torch.zeros(count, 2),
        mode=torch.zeros(count, dtype=torch.long), policy_dt=.02,
        stage_actor_update=368, global_actor_update=368,
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        episode_limits=torch.zeros(count, dtype=torch.long),
        command_clock=torch.zeros(count), height_clock=torch.zeros(count), push_clock=torch.zeros(count),
        push_enabled=torch.zeros(count, dtype=torch.bool),
        robot=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose))),
        random=lambda n: torch.rand(n, generator=generator))
    e.skills = SkillCommands(e)
    e.skills.sample(torch.arange(count))
    return e


def run(stagger, updates):
    e = environment(stagger)
    s = e.skills
    if stagger:
        runtime = s.start_stop_timing_state()
    else:
        runtime = {"enabled": False}
    members = {spec["group_name"]: ids for ids, spec in s.batches}
    stop_ids = members["start_stop_05"]
    delay = s.start_stop_delay[stop_ids].clone()
    horizons = e.episode_limits[stop_ids].clone()
    resets = torch.zeros(e.num_envs, dtype=torch.long)
    velocity = torch.zeros(e.num_envs, 3)
    batches = []
    for update in range(updates):
        samples = {key: 0 for key in ("braking_window", "settled_stop_window", "start_stop_waiting")}
        for _ in range(24):
            e.episode_length_buf += 1
            s.update()
            data = s.diagnostics(velocity)
            for key in samples:
                samples[key] += int(data[key][stop_ids].sum())
            # Fail if random wait itself is mistaken for successful stopping.
            assert not (data["start_stop_waiting"] & (data["braking_window"] | data["settled_stop_window"])).any()
            assert not e.command_target[stop_ids[data["start_stop_waiting"][stop_ids]]].any()
            done = e.episode_length_buf >= e.episode_limits
            ids = done.nonzero().flatten()
            if len(ids):
                resets[ids] += 1
                e.episode_length_buf[ids] = 0
                s.sample(ids)
            e.command_clock -= e.policy_dt
            resample = (e.command_clock <= 0) & ~done
            if resample.any():
                ids = resample.nonzero().flatten()
                before = s.start_stop_delay[ids].clone()
                s.sample(ids, reset_height=False)
                assert torch.equal(before, s.start_stop_delay[ids])
        batches.append({"update": update + 1, **samples})
    return {"updates": updates, "simulated_command_seconds": updates * 24 * .02,
            "num_envs": e.num_envs, "start_stop_envs": len(stop_ids), "runtime": runtime,
            "initial_delay_range_s": [float(delay.min()), float(delay.max())],
            "initial_episode_range_s": [float(horizons.min()) * .02, float(horizons.max()) * .02],
            "batches_with_braking": sum(b["braking_window"] > 0 for b in batches),
            "batches_with_settled_stop": sum(b["settled_stop_window"] > 0 for b in batches),
            "last_five_batches": batches[-5:],
            "completed_command_episodes": {name: {"total": int(resets[ids].sum()),
                "per_lane_min": int(resets[ids].min()), "per_lane_max": int(resets[ids].max())}
                for name, ids in members.items()},
            "batches": batches}


result = {"scope": "Production CPU command-clock replay; no physics, policy, reward or acceptance claim",
          "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (CONTRACT, ROOT / "src/wheeled_tasks/chassis/skill_commands.py",
                           ROOT / "src/wheeled_tasks/chassis/training_schedule.py")},
          "legacy_40": run(False, 40), "stagger_40": run(True, 40),
          "stagger_persistent_222": run(True, 222)}
assert result["legacy_40"]["batches_with_braking"] == 7
assert result["legacy_40"]["batches_with_settled_stop"] == 14
assert result["stagger_40"]["batches_with_braking"] > result["legacy_40"]["batches_with_braking"]
assert all(result["stagger_persistent_222"]["completed_command_episodes"][group]["per_lane_min"] >= 3
           for group in ("stand_continuous", "height_scan", "start_stop_05"))
output = Path(__file__).with_name("start_stop_timing_probe.json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: {k: v for k, v in value.items() if k != "batches"}
                  for key, value in result.items() if key in ("legacy_40", "stagger_40", "stagger_persistent_222")}, indent=2))
