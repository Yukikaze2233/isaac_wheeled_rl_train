"""CPU command-owner checks for opt-in initial-heading evaluation feedback."""
import ast
import math
from pathlib import Path

import pytest
import torch

from tests.support.support_flat_evaluation_setup import make_env
from tests.support.support_flat_episode_env_wiring import bind_finish
from tests.support.support_flat_policy_evaluation import fixed
from tests.simulation.runtime.test_flat_reward_runtime import env as reward_env
from wheeled_tasks.chassis.tasks.commands.flat_command_sampling import ScutFlatCommands
from wheeled_tasks.chassis.application.evaluation.flat_evaluation_setup import FlatEvaluationSetup
from wheeled_tasks.chassis.application.evaluation.heading_hold_evaluation import MODE, install_heading_hold


def protocol():
    values = [(f"vx_{speed}", [speed, 0., .305]) for speed in (.5, -.5, 1.35, -1.35, 2.7, -2.7)]
    values += [("standing", [0., 0., .305]), ("spin", [0., 2., .305]),
               ("combined", [1., .1, .305])]
    result = fixed(heading_hold=MODE)
    result["cases"] = [dict(case_id=name, base_request=request) for name, request in values]
    return result


def setup_env(*, enabled=True):
    p = protocol()
    if not enabled:
        p.pop("heading_hold")
    lanes = tuple(case["case_id"] for case in p["cases"])
    setup = FlatEvaluationSetup(p, lanes, 864, 9147)
    env = make_env(setup)
    setup.apply_before_first_reset(env)
    return env


def reset(env, heading=None, ids=None):
    n = env.num_envs
    heading = torch.zeros(n) if heading is None else heading
    ids = torch.arange(n) if ids is None else ids
    env.flat_episode.commands.reset(ids, heading, torch.zeros(n, dtype=torch.bool), env.generator, 9147)


def advance(env, heading):
    commands = env.flat_episode.commands
    for _ in range(4):
        commands.advance_physics()
    commands.advance(.02, heading, 9147, env.generator)


def test_default_setup_has_no_instance_hooks_or_heading_owner():
    env = setup_env(enabled=False)
    commands = env.flat_episode.commands
    assert not hasattr(env, "evaluation_heading_hold")
    assert not ({"reset", "_postprocess", "diagnostics"} & set(vars(commands)))
    assert commands.reset.__func__ is ScutFlatCommands.reset


def test_explicit_setup_selects_only_six_straight_cases_without_extra_rng_draws():
    env, baseline = setup_env(), setup_env(enabled=False)
    owner = env.evaluation_heading_hold
    assert owner.selected.tolist() == [True] * 6 + [False] * 3
    torch.testing.assert_close(env.generator.get_state(), baseline.generator.get_state())
    heading = torch.linspace(-1., 1., env.num_envs)
    reset(env, heading)
    reset(baseline, heading)
    torch.testing.assert_close(owner.target, heading)
    torch.testing.assert_close(env.generator.get_state(), baseline.generator.get_state())
    before = env.flat_episode.commands.command3.clone()
    advance(env, heading + .04)
    advance(baseline, heading + .04)
    torch.testing.assert_close(env.generator.get_state(), baseline.generator.get_state())
    torch.testing.assert_close(env.flat_episode.commands.command3[:6, 2], torch.full((6,), -.2))
    torch.testing.assert_close(env.flat_episode.commands.command3[6:], before[6:])
    assert owner.postprocess_calls == 1 and owner.last_read_physics_step == 4


def test_original_request_and_effective_correction_remain_separate_detached_readings():
    env = setup_env()
    reset(env)
    advance(env, torch.full((env.num_envs,), .1))
    commands = env.flat_episode.commands
    diagnostics = commands.diagnostics()
    torch.testing.assert_close(diagnostics["base_request_command3"][:6], commands.fixed_request_command3[:6])
    torch.testing.assert_close(diagnostics["command3"][:6, 2], torch.full((6,), -.5))
    assert not diagnostics["override_active"].any()
    diagnostics["base_request_command3"].zero_()
    assert commands.fixed_request_command3[:6, 0].abs().min() > 0.


def test_timed_resample_does_not_disable_feedback_or_replace_episode_target():
    env, baseline = setup_env(), setup_env(enabled=False)
    initial = torch.linspace(-.8, .8, env.num_envs)
    reset(env, initial)
    reset(baseline, initial)
    for target in (env, baseline):
        target.flat_episode.commands.time_left.fill_(.001)
    advance(env, initial + .1)
    advance(baseline, initial + .1)
    commands = env.flat_episode.commands
    assert bool(commands.is_heading[:6].all())
    assert bool((commands.command_counter == 2).all())
    assert bool(((commands.time_left >= 5.) & (commands.time_left <= 15.)).all())
    torch.testing.assert_close(env.evaluation_heading_hold.target, initial)
    torch.testing.assert_close(commands.heading_target[:6], initial[:6])
    torch.testing.assert_close(commands.command3[:6, 2], torch.full((6,), -.5))
    torch.testing.assert_close(env.generator.get_state(), baseline.generator.get_state())
    advance(env, initial + .2)
    torch.testing.assert_close(commands.command3[:6, 2], torch.full((6,), -1.))


@pytest.mark.parametrize("initial,current,expected", [
    (math.pi - .03, -math.pi + .02, -.25),
    (-math.pi + .03, math.pi - .02, .25),
    (0., -2., 2. * math.pi), (0., 2., -2. * math.pi),
])
def test_wrapped_gain_and_symmetric_limits(initial, current, expected):
    env = setup_env()
    reset(env, torch.full((env.num_envs,), initial))
    advance(env, torch.full((env.num_envs,), current))
    torch.testing.assert_close(env.flat_episode.commands.command3[:6, 2], torch.full((6,), expected),
                               atol=2e-6, rtol=1e-6)


def test_partial_episode_reset_reanchors_only_reset_lanes_and_initial_command_is_zero():
    env = setup_env()
    reset(env)
    advance(env, torch.full((env.num_envs,), .1))
    ids = torch.tensor([0, 3])
    reset(env, torch.full((env.num_envs,), 1.2), ids)
    assert env.evaluation_heading_hold.target.tolist() == pytest.approx([1.2, 0., 0., 1.2, 0., 0., 0., 0., 0.])
    torch.testing.assert_close(env.flat_episode.commands.command3[ids, 2], torch.zeros(2))
    advance(env, torch.full((env.num_envs,), 1.3))
    torch.testing.assert_close(env.flat_episode.commands.command3[ids, 2], torch.full((2,), -.5))


def test_install_requires_fresh_owner_and_does_not_silently_replace_one():
    env = setup_env()
    with pytest.raises(ValueError, match="already has an owner"):
        install_heading_hold(env, protocol(), tuple(c["case_id"] for c in protocol()["cases"]))


def test_production_finish_reads_original_request_and_pays_previous_effective_command():
    setup = setup_env()
    reset(setup)
    e = reward_env(setup.num_envs)
    e.flat_episode, e.generator = setup.flat_episode, setup.generator
    e.policy_dt = .02
    e.raw_height = torch.zeros(e.num_envs)
    e.mode = torch.ones(e.num_envs, dtype=torch.long)
    e.targets = torch.zeros(e.num_envs, 4)
    e.jump_requested = torch.zeros(e.num_envs, dtype=torch.bool)
    e.flat_episode.episode_id.zero_()
    e.flat_episode.publish(e.commands, e.mode, e.targets, e.jump_requested)
    bind_finish(e)
    e.get_observations = lambda: {"command": e.commands.clone()}
    for yaw in (.1, .12):
        preceding = e.commands.clone()
        e.flat_episode.begin_policy()
        for _ in range(4):
            e.flat_episode.advance_physics(.005)
        e.robot.data.root_link_pose_w.torch[:, 5] = math.sin(yaw / 2.)
        e.robot.data.root_link_pose_w.torch[:, 6] = math.cos(yaw / 2.)
        obs, _, done, extras = e.finish()
        assert not done.any()
        diagnostics = extras["diagnostics"]
        torch.testing.assert_close(diagnostics["effective_command"], preceding)
        torch.testing.assert_close(diagnostics["base_request"][:6, 1], torch.zeros(6))
        torch.testing.assert_close(obs["command"][:6, 1], torch.full((6,), -5. * yaw))
        torch.testing.assert_close(diagnostics["source_flat_episode"]["base_request_command3"][:6, 2],
                                   torch.zeros(6))


def test_physical_source_pays_previous_command_then_advances_before_returned_observation():
    path = Path(__file__).resolve().parents[3] / "src/wheeled_tasks/chassis/application/environment.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and any(isinstance(call, ast.Call) and ast.unparse(call.func) == "source.pay"
                            for call in ast.walk(node)))
    calls = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(function)
             if isinstance(node, ast.Call)]
    first = lambda name: next(line for line, value in calls if value == name)
    assert first("source.pay") < first("self._advance_flat_episode") < first("self.get_observations")
