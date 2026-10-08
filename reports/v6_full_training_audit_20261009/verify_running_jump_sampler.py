#!/usr/bin/env python3
"""CPU proof of frozen v26 target availability; no simulator or training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch

from wheeled_tasks.chassis import integrated_runtime as runtime
from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.skill_commands import SkillCommands
from wheeled_tasks.chassis.training_schedule import jump_bin_probabilities


ROOT = Path(runtime.__file__).resolve().parents[3]
COMMIT = "e5bd9c1367b875d7f229b9a1930647013d023372"
RECIPE_SHA = "d1a1de9829bb1c798f627e6605d4cc57ca669b400136b568e46b9b73708df2c5"
NAMES = ("running_jump_03", "running_jump_06", "running_jump_10")
COUNT = 10000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_env(domain, recipe):
    source_file, expected, _ = runtime.sampling.SOURCE_BINDINGS[domain]
    assert sha(ROOT / source_file) == expected
    cfg = json.loads((ROOT / source_file).read_text())
    cfg.update(deepcopy(recipe["domains"][domain]["patch"]))
    audit = deepcopy(recipe["domains"][domain]["runtime"])
    # Synthetic CPU source clocks satisfy the real runtime clock validator.
    # They are not a real parent admission or a claimed training endpoint.
    starts = dict(speed=0, terrain=15000, jump=30000)
    if domain == "stepjump":
        starts["stepjump"] = 40000
    joint = starts[domain]
    selection = dict(domain=domain, lineage_id=recipe["lineage_id"],
        plan_sha256="0" * 64, origin_checkpoint_sha256="0" * 64,
        origin_recovery_actor_update=14188, fresh_joint_actor_updates=joint,
        fresh_domain_actor_updates=0)
    audit.update(learning_source=selection,
        clocks_at_source=runtime._clock_document(selection, starts, joint, 0))
    cfg["integrated_runtime_candidate"] = audit
    # Project only the three real native rows into a command-sampler fixture.
    # Their protected command owner never participates in heading.
    cfg["skill_specs"] = {name: cfg["skill_specs"][name] for name in NAMES}
    cfg["scut_heading_command"]["enabled"] = False
    total = COUNT * len(NAMES)
    groups = [name for name in NAMES for _ in range(COUNT)]
    env = SimpleNamespace(cfg=cfg, device="cpu", num_envs=total, scene_groups=groups,
        policy_dt=cfg["policy_dt"], task_actor_update=0, global_actor_update=90000,
        stage_actor_update=50000, successful_updates_total=0,
        random=lambda count: (torch.arange(count, dtype=torch.float32) + .5) / count,
        commands=torch.zeros(total, 3), command_target=torch.zeros(total, 2),
        mode=torch.zeros(total, dtype=torch.long), push_enabled=torch.zeros(total, dtype=torch.bool))
    for field in ("command_clock", "height_clock", "push_clock", "episode_limits", "episode_length_buf"):
        setattr(env, field, torch.zeros(total))
    semantics = deepcopy(cfg["task_semantics"])
    for key in {key for name in NAMES for key in cfg["skill_specs"][name]["semantics"]}:
        semantics[key] = torch.tensor([cfg["skill_specs"][name]["semantics"].get(key,
            cfg["task_semantics"].get(key, 0.)) for name in groups])
    env.full_tasks = FullTaskSemantics(total, "cpu", env.policy_dt, semantics)
    return env


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert commit == COMMIT, commit
    recipe_file = ROOT / "contracts/v6_integrated_runtime_recipe_v26.json"
    assert sha(recipe_file) == RECIPE_SHA
    recipe = json.loads(recipe_file.read_text())
    result = dict(schema="frozen-v26-running-jump-sampler-cpu-v1", source_root=str(ROOT),
        source_commit=commit, recipe_sha256=RECIPE_SHA, sample_count_per_row=COUNT,
        sampling_method="deterministic_uniform_CDF_grid_midpoints",
        scope="CPU target availability and command/task-reference wiring; synthetic clocks, no actual exposure, success or qualification",
        domains={}, source_files={file: sha(ROOT / file) for file in (
            "src/wheeled_tasks/chassis/training_schedule.py", "src/wheeled_tasks/chassis/skill_commands.py",
            "src/wheeled_tasks/chassis/integrated_runtime.py", "src/wheeled_tasks/chassis/full_tasks.py",
            "src/wheeled_tasks/chassis/env.py", "scripts/train_chassis.py")})
    for domain in ("jump", "stepjump"):
        env = build_env(domain, recipe)
        skills = SkillCommands(env)
        ids = torch.arange(env.num_envs)
        rows = []
        for update in (0, 2000, 6000):
            env.task_actor_update = update
            skills.sample(ids)
            for index, name in enumerate(NAMES):
                members = slice(index * COUNT, (index + 1) * COUNT)
                spec = env.cfg["skill_specs"][name]
                settings, clock = skills.schedule.context(name)
                jump = settings["jump_curriculum"]
                bins = torch.tensor(jump["bins_m"])
                expected = jump_bin_probabilities(jump, actor_update=clock)
                apex = skills.schedule.jump_apex[members]
                counts = [(apex == value).sum().item() for value in bins]
                probabilities = [count / COUNT for count in counts]
                assert all(abs(a - b) <= .0002 for a, b in zip(probabilities, expected))
                assert all(count > 0 for count in counts)
                assert env.mode[members].eq(4).all()
                assert torch.allclose(env.commands[members, 0], torch.full((COUNT,), .4))
                assert env.commands[members, 1].eq(0).all()
                torch.testing.assert_close(env.full_tasks.cfg["jump_apex_delta_m"][members], apex)
                torch.testing.assert_close(env.full_tasks.cfg["jump_com_rise_m"][members], (apex - .015).clamp_min(.005))
                rows.append(dict(task_actor_update=update, group=name,
                    request_component=env.cfg["integrated_runtime_candidate"]["row_ownership"][name]["request_component"],
                    component_actor_update=clock, bins_m=jump["bins_m"], probabilities=expected,
                    CPU_sample_counts=counts, empirical_probabilities=probabilities,
                    high_15_to_30_probability=sum(expected[3:]), vx_target_m_s=.4,
                    horizontal_goal_m=spec["semantics"]["jump_forward_distance_m"],
                    landing_radius_m=spec["semantics"]["jump_landing_radius_m"],
                    landing_speed_max_m_s=spec["semantics"]["jump_landing_speed_max_m_s"]))
        # Simulate completed critic-only update counters; no actor commit occurred.
        env.task_actor_update = 0
        before = [skills.schedule.context(name)[1] for name in NAMES]
        env.successful_updates_total += 10
        env.global_actor_update += 10
        env.stage_actor_update += 10
        after = [skills.schedule.context(name)[1] for name in NAMES]
        assert before == after
        result["domains"][domain] = dict(rows=rows, critic_only_clock_before=before,
            critic_only_clock_after=after,
            projected_fixture="native three-row SkillCommands.sample; heading disabled only outside these protected rows")
    # Frozen examinations intentionally do not sample their named target.
    env = build_env("jump", recipe)
    env.cfg["evaluation_exact_cases"] = True
    env.cfg["evaluation"] = dict(cases=[dict(name=name, command=spec["command"], skill=spec)
        for name, spec in env.cfg["skill_specs"].items()])
    env.task_actor_update = 6000
    skills = SkillCommands(env)
    skills.sample(torch.arange(env.num_envs))
    result["fixed_eval_targets_m"] = {}
    for index, name in enumerate(NAMES):
        members = slice(index * COUNT, (index + 1) * COUNT)
        apex = skills.schedule.jump_apex[members]
        expected = env.cfg["skill_specs"][name]["semantics"]["jump_apex_delta_m"]
        assert apex.eq(expected).all()
        result["fixed_eval_targets_m"][name] = expected
    result["passed"] = True
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
