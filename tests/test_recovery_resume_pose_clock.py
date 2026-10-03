"""Exercise the learner's real startup blocks without loading Isaac/PhysX."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_recovery_pose_curriculum import make_env
from wheeled_tasks.chassis.recovery_curriculum import RecoveryResetCurriculum
from wheeled_tasks.chassis.recovery_revision import pose_curriculum


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/train_chassis.py"


def startup_clock_block():
    tree = ast.parse(ENTRY.read_text())
    body = next(value for node in ast.walk(tree) for field, value in ast.iter_fields(node)
                if field in ("body", "orelse") and isinstance(value, list)
                and any(isinstance(item, ast.Assign)
                        and isinstance(item.targets[0], ast.Attribute)
                        and item.targets[0].attr == "training_transitions" for item in value))
    start = next(index for index, item in enumerate(body)
                 if isinstance(item, ast.Assign)
                 and isinstance(item.targets[0], ast.Attribute)
                 and item.targets[0].attr == "training_transitions")
    end = next(index for index, item in enumerate(body)
               if isinstance(item, ast.Assign)
               and isinstance(item.targets[0], ast.Name)
               and item.targets[0].id == "before_actor")
    return compile(ast.Module(body=body[start:end], type_ignores=[]), str(ENTRY), "exec")


def resume_startup(env, *, actor_update=6400, resume=True, bins=(), domain="recovery",
                   requested_actor_update=None):
    events = []

    def record_reset(ids):
        events.append(("reset", env.task_actor_update, ids.clone()))
        env.recovery.pose_curriculum.reset(ids, env.task_actor_update)

    env.reset = record_reset
    env.resample_commands = lambda ids, **kwargs: events.append(
        ("commands", env.task_actor_update, ids.clone(), kwargs))
    env.update_targets = lambda: events.append(("targets", env.task_actor_update))
    env.update_command_reference = lambda: events.append(("reference", env.task_actor_update))
    args = SimpleNamespace(resume="sealed.pt" if resume else None,
        actor_updates_consumed=actor_update, stage_actor_update=None,
        global_actor_update_offset=13000, task_actor_update=actor_update if domain == "recovery" else 0,
        num_envs=env.num_envs, device="cpu")
    if requested_actor_update is not None:
        args.task_actor_update = requested_actor_update
    infos = dict(stage_actor_update=actor_update, task_actor_update=actor_update,
                 task_campaign=dict(domain=domain))
    namespace = dict(env=env, args=args, infos=infos,
        c=dict(task_campaign=dict(domain="recovery")), confirmed_bins=list(bins),
        report=dict(parent_training_transitions=12345), torch=torch)
    exec(startup_clock_block(), namespace)
    return events, namespace["report"]


def recovery_environment(count=4096):
    table = json.loads((ROOT / "contracts/v6_recovery_profiles_v2.json").read_text())
    env = make_env(table, count=count, only_profile="front_down_90")
    env.task_actor_update = 0
    env.recovery = SimpleNamespace(pose_curriculum=RecoveryResetCurriculum(env, table, pose_curriculum(400)))
    return env


def test_resumed_first_episode_uses_restored_pose_distribution():
    env = recovery_environment()
    sampler = env.recovery.pose_curriculum
    sampler.reset(torch.arange(env.num_envs), 0)
    assert not (sampler.level_ids == 2).any()
    events, report = resume_startup(env)
    assert [event[0] for event in events] == ["reset"]
    assert events[0][1] == env.task_actor_update == 6400
    assert env.stage_actor_update == 6400 and env.global_actor_update == 19400
    assert env.training_transitions == 12345
    assert report["task_actor_update"] == 6400
    fractions = torch.bincount(sampler.level_ids, minlength=3).float() / env.num_envs
    assert fractions.tolist() == pytest.approx([.2, .2, .6], abs=.02)


def test_migration_to_recovery_starts_at_its_own_actor_clock():
    env = recovery_environment()
    events, report = resume_startup(env, actor_update=13000, domain="flat")
    assert events[0][1] == report["task_actor_update"] == 0
    assert not (env.recovery.pose_curriculum.level_ids == 2).any()
    assert env.global_actor_update == 26000


@pytest.mark.parametrize("recovery", [None, SimpleNamespace(pose_curriculum=None)])
def test_nonrecovery_and_exact_eval_preserve_command_only_startup(recovery):
    env = SimpleNamespace(num_envs=8, task_actor_update=0, recovery=recovery)
    events, _ = resume_startup(env, actor_update=5000)
    assert [event[0] for event in events] == ["commands", "targets", "reference"]
    assert all(event[1] == 5000 for event in events)
    assert events[0][3] == dict(reset_height=True)


def test_fresh_run_does_not_repeat_constructor_reset():
    env = recovery_environment(count=10)
    events, _ = resume_startup(env, actor_update=0, resume=False, domain="flat")
    assert events == []
    assert env.recovery.pose_curriculum.profile_reset_counts.sum().item() == 0


def test_fresh_confirmed_bins_keep_command_only_startup():
    env = recovery_environment(count=10)
    events, _ = resume_startup(env, actor_update=0, resume=False, bins=(.1,), domain="flat")
    assert [event[0] for event in events] == ["commands", "targets", "reference"]
    assert all(event[1] == 0 for event in events)
    assert env.recovery.pose_curriculum.profile_reset_counts.sum().item() == 0
    assert env.confirmed_jump_bins_m == [.1]


def test_clock_mismatch_rejects_before_any_pose_or_command_reset():
    env = recovery_environment(count=10)
    before = env.recovery.pose_curriculum.profile_reset_counts.clone()
    with pytest.raises(ValueError, match="Task curriculum clock"):
        resume_startup(env, requested_actor_update=1234)
    assert torch.equal(env.recovery.pose_curriculum.profile_reset_counts, before)


def manifest_sources(config):
    tree = ast.parse(ENTRY.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignments = [index for index, node in enumerate(main.body)
                   if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "source_files"]
    start, end = assignments[0], assignments[-1]
    namespace = dict(c=config, manifest=dict(model_kind="test"), ROOT=ROOT, json=json,
        args=SimpleNamespace(transfer=False, resume_physics_change=False,
                             resume_response_change=False, resume_task_change=False))
    block = ast.Module(body=main.body[start:end + 1], type_ignores=[])
    exec(compile(block, str(ENTRY), "exec"), namespace)
    return namespace["source_files"]


@pytest.mark.parametrize("exact,enabled,revision,expected_pose,expected_revision", [
    (False, True, "task-worker-revision-v8", True, True),
    (True, True, "task-worker-revision-v8", False, True),
    (False, False, "task-worker-revision-v8", False, True),
    (False, True, "task-worker-revision-v6", True, False),
])
def test_active_curriculum_and_revision_have_checkpoint_source_pins(
        exact, enabled, revision, expected_pose, expected_revision):
    config = dict(control_math_source="contracts/control.json", evaluation_exact_cases=exact,
        recovery_training=dict(enabled=True, profiles_file="contracts/profiles.json",
            height_lookup_file="contracts/heights.json", pose_curriculum=dict(enabled=enabled)),
        task_campaign=dict(revision_migration=dict(schema=revision)))
    sources = manifest_sources(config)
    assert ("src/wheeled_tasks/chassis/recovery_curriculum.py" in sources) == expected_pose
    assert ("src/wheeled_tasks/chassis/recovery_revision.py" in sources) == expected_revision
    assert len(sources) == len(set(sources))


def test_historical_source_list_does_not_gain_inactive_recovery_modules():
    config = dict(control_math_source="contracts/control.json")
    unchanged = deepcopy(config)
    sources = manifest_sources(config)
    assert config == unchanged
    assert not any(name.endswith(("recovery_curriculum.py", "recovery_revision.py")) for name in sources)
