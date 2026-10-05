"""Execute production environment hooks with native CPU task and lesson objects."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_nonstandard_start_integration import init_hook
from test_recovery_prefix_curriculum import learner
from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.jump_landing import JUMP_LANDING_V1, JumpLandingWindow
from wheeled_tasks.chassis.preflight import validate_runtime_state
from wheeled_tasks.chassis.task import PhaseTracker


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/wheeled_tasks/chassis/env.py"


def method(name):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChassisEnv")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(nodes, env, **values):
    scope = dict(self=env, torch=torch, __package__="wheeled_tasks.chassis", **values)
    tree = ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[]))
    exec(compile(tree, str(SOURCE), "exec"), scope)
    return scope


def bind_method(env, name):
    scope = execute([method(name)], env, stage_actor_updates=lambda self: self.task_actor_update)
    return scope[name].__get__(env)


@pytest.fixture
def table():
    return json.loads((ROOT / "contracts/v6_recovery_profiles_v2.json").read_text())


def recovery_env(table, *, clock=8294):
    env, sampler = learner(table, count=100, clock=clock)
    env.cfg["recovery_training"] = {"enabled": True, "prefix_curriculum": sampler.settings}
    env.task_actor_update = clock
    flags = torch.ones(100, dtype=torch.bool)
    diagnostic = {"recovery_holding_rl": flags,
        "recovery_stable_peak_seconds": torch.full((100,), .25),
        "recovery_stable_break_count": torch.zeros(100, dtype=torch.long),
        **{"recovery_stable_" + k + "_pass": flags for k in
            ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed")}}
    env.recovery = NS(pose_curriculum=sampler, enabled=flags, prelude=~flags,
        learning_mask=flags, true_success=sampler.level_ids < 2,
        fsm=NS(diagnostics=lambda: diagnostic, config={"complete_started_blend": True}))
    env.recovery_takeover_settings = None
    return env


def test_native_logging_records_truth_before_reset_and_does_not_credit_l3_with_easy(table):
    env = recovery_env(table)
    hook = bind_method(env, "_recovery_metrics")
    done = torch.ones(100, dtype=torch.bool)
    success = env.recovery.true_success
    failed = ~success
    timeouts = torch.zeros_like(done)
    log = hook(done, success, failed, timeouts)
    sampler = env.recovery.pose_curriculum
    assert log["/recovery/lesson/prefix_stand/success_fraction"] == 1
    assert log["/recovery/full_chain_success"] == 0
    assert log["/recovery/full_chain_population"] == sampler.full_chain_mask.sum()
    assert log["/recovery/lesson/original_chain/failures"] == failed.sum()
    assert log["/recovery/hold_gate/support_pass_fraction"] == 1
    count = sampler.outcome_counts.clone()
    hook(done, success, failed, timeouts)
    assert torch.equal(sampler.outcome_counts, count)
    sampler.reset(torch.arange(100), 2294)
    new = hook(torch.zeros_like(done), torch.zeros_like(done), torch.zeros_like(done), timeouts)
    assert new["/recovery/full_chain_population"] == 0
    assert new["/recovery/full_chain_success"] == 0
    assert torch.equal(sampler.outcome_counts, count)


def test_native_exact_eval_preserves_original_population_statistic(table):
    env = recovery_env(table)
    env.recovery.pose_curriculum = None
    env.cfg["recovery_training"].pop("prefix_curriculum")
    env.cfg["evaluation_exact_cases"] = True
    flags = torch.zeros(100, dtype=torch.bool)
    result = bind_method(env, "_recovery_metrics")(flags, flags, flags, flags)
    assert result["/recovery/full_chain_success"] == env.recovery.true_success.float().mean()
    assert "/recovery/full_chain_population" not in result
    assert not any(key.startswith("/recovery/lesson/") for key in result)


def runtime_hook(env, settings):
    node = next(n for n in ast.walk(method("__init__")) if isinstance(n, ast.If)
        and ast.unparse(n.test) == "config.get('recovery_training', {}).get('enabled')"
        and "actual['modules']" in ast.unparse(n))
    expected = json.loads((ROOT / "contracts/v6_task_campaign_recovery_expectations_v17.json").read_text())["stages"]["recovery"]["roles"]["train"]["runtime"]
    actual = deepcopy(expected)
    # Unrelated module fields are fixture constants; execute the production
    # recovery instance assertion and test its own native presence/settings.
    execute([node], env, config={"recovery_training": settings}, actual=actual)
    return expected, actual


def test_native_startup_asserts_actual_prefix_instance_and_exact_settings(table):
    env = recovery_env(table)
    env.startup_report = {}
    expected, actual = runtime_hook(env, env.cfg["recovery_training"])
    assert actual["modules"]["recovery_prefix_curriculum"] is True
    assert env.startup_report["recovery_prefix_curriculum"] == actual["recovery_prefix_curriculum"]
    assert validate_runtime_state(expected, actual)["status"] == "runtime_configuration_passed"
    for bad in ("enabled", "origin_actor_update"):
        broken = deepcopy(actual)
        if bad == "enabled":
            broken["modules"]["recovery_prefix_curriculum"] = False
        else:
            broken["recovery_prefix_curriculum"]["settings"][bad] = 0
        with pytest.raises(ValueError, match="recovery_prefix_curriculum"):
            validate_runtime_state(expected, broken)
    env.recovery.pose_curriculum = None
    _, absent = runtime_hook(env, env.cfg["recovery_training"])
    with pytest.raises(ValueError, match="recovery_prefix_curriculum"):
        validate_runtime_state(expected, absent)


def jumping_env(*, enabled=True):
    count = 2
    env = NS(jump_cycle_reset_on_accepted=enabled,
        full_tasks=FullTaskSemantics(count, "cpu", .02, {}),
        jump_landing=JumpLandingWindow(count, "cpu", .02, deepcopy(JUMP_LANDING_V1)),
        phase=PhaseTracker(count, "cpu", .02), success_hold=torch.ones(count),
        sequence_success_latched=torch.ones(count, dtype=torch.bool))
    env._reset_jump_evidence = bind_method(env, "_reset_jump_evidence")
    for value in vars(env.full_tasks).values():
        if isinstance(value, torch.Tensor):
            value.fill_(1)
    env.jump_landing.outcome_active[:] = True
    env.jump_landing._request_ticks[:] = 200
    env.phase.flew[:] = True
    return env


def test_native_second_step_request_clears_own_observers_and_preserves_other_lane():
    env = jumping_env()
    local = torch.tensor([[.8, .0, .45], [1.4, .0, .70]])
    env._reset_jump_evidence(torch.tensor([0]), local)
    assert not env.full_tasks.com_released[0]
    assert env.full_tasks.com_start_height[0] == 0
    assert env.full_tasks.com_release_height[0] == 0
    assert env.full_tasks.previous_x[0] == .8
    assert not env.jump_landing.outcome_active[0] and env.jump_landing._request_ticks[0] == 0
    assert not env.phase.flew[0] and env.success_hold[0] == 0
    assert env.full_tasks.com_released[1] and env.jump_landing.outcome_active[1]
    assert env.jump_landing._request_ticks[1] == 200 and env.phase.flew[1]


def test_legacy_step_request_retains_old_observer_ownership_when_flag_absent():
    env = jumping_env(enabled=False)
    env._reset_jump_evidence(torch.tensor([0]), torch.zeros(2, 3))
    assert env.jump_landing.outcome_active[0] and env.jump_landing._request_ticks[0] == 200
    assert env.full_tasks.com_release_height[0] == 1


def test_native_flat_accepted_request_uses_uniform_reset_only_for_new_flag():
    block = next(n for n in ast.walk(method("update_targets")) if isinstance(n, ast.If)
        and ast.unparse(n.test) == "len(accepted_jump)")
    env = jumping_env()
    execute([block], env, accepted_jump=torch.tensor([0]), local=torch.zeros(2, 3))
    assert not env.jump_landing.outcome_active[0]
    assert env.full_tasks.previous_x[0] == 0
    env = jumping_env(enabled=False)
    execute([block], env, accepted_jump=torch.tensor([0]), local=torch.zeros(2, 3))
    assert env.jump_landing.outcome_active[0]
    assert env.full_tasks.previous_x[0] == 1


def test_flag_default_and_runtime_state_require_explicit_new_contract_switch():
    assignment = next(n for n in method("__init__").body if isinstance(n, ast.Assign)
        and "self.jump_cycle_reset_on_accepted" in ast.unparse(n))
    env = NS()
    execute([assignment], env, config={})
    assert env.jump_cycle_reset_on_accepted is False
    execute([assignment], env, config={"jump_cycle_reset_on_accepted": True})
    assert env.jump_cycle_reset_on_accepted is True


def test_v17_birth_mask_excludes_flat_mode4_sequence_pool_and_keeps_ordinary(table):
    from wheeled_tasks.chassis.nonstandard_start import nonstandard_start_settings
    groups = ["stand", "sequence", "forward", "raised_request"]
    settings = nonstandard_start_settings(100)
    cfg = dict(nonstandard_start=settings, asset_manifest_sha256=settings["asset_manifest_sha256"],
        command_slew={"vx_m_s2": 1.5, "yaw_rad_s2": 4.},
        task_campaign={"domain": "jump", "revision_migration": {"schema": "task-worker-revision-v17"}},
        behavior_pool_membership=dict(zip(groups,
            ["stand_anchors", "request_sequences", "basic_translation", "flat_jumps"])))
    node = next(n for n in method("__init__").body if isinstance(n, ast.If)
        and ast.unparse(n.test) == "'nonstandard_start' in config")
    env = NS(kinds=["flat"] * 4, scene_groups=groups, mode=torch.tensor([0, 4, 0, 4]),
        dt=.005, policy_dt=.02, robot=NS(joint_names=table["tree_joint_names"]),
        startup_report={}, skills=NS(spin=torch.zeros(4, dtype=torch.bool)),
        recovery=None, step_jump=None)
    execute([node], env, config=cfg, num_envs=4, device="cpu", root=ROOT)
    assert env.nonstandard_start_eligible.tolist() == [True, False, True, False]
    assert env.startup_report["nonstandard_start_eligibility"]["request_pool_count"] == 2
    assert env.startup_report["nonstandard_start_eligibility"]["eligible_count"] == 2
    cfg["behavior_pool_membership"]["sequence"] = "stand_anchors"
    with pytest.raises(ValueError, match="exclude every active request lane"):
        execute([node], env, config=cfg, num_envs=4, device="cpu", root=ROOT)
