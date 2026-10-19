"""Bounded restored-policy entry consumers; CPU stubs do not qualify native recovery."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict

from support_scut_flat_scripted_recovery import config, rig, tick
from wheeled_tasks.chassis.flat_evaluation_runtime import _stock_actor
from wheeled_algo.training_control import PlannedStop

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'scripts/train_chassis.py'


@pytest.fixture
def entry():
    spec = importlib.util.spec_from_file_location('restored_policy_entry', ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    return NS(**(dict(validate_restored_policy=3, resume=Path('/fixture/parent.pt'),
        validate_scene=0, preflight_only=False, capture_numerical_failure=True) | overrides))


def test_explicit_full_recovery_resume_is_required_and_default_is_unchanged(entry):
    c = config()
    c.update(formal_learning_state_resume=True, record_diagnostics=True)
    assert entry.validate_restored_policy_request(args(), c, ROOT)
    assert not entry.validate_restored_policy_request(args(validate_restored_policy=0), {}, ROOT)
    for overrides in (dict(resume=None), dict(validate_scene=1), dict(preflight_only=True),
                      dict(capture_numerical_failure=False), dict(validate_restored_policy=10001),
                      dict(validate_restored_policy=-1), dict(validate_restored_policy=True)):
        with pytest.raises(ValueError):
            entry.validate_restored_policy_request(args(**overrides), c, ROOT)
    for changes in (dict(formal_learning_state_resume=False), dict(record_diagnostics=False),
                    dict(recovery_training={})):
        with pytest.raises(ValueError):
            entry.validate_restored_policy_request(args(), c | changes, ROOT)


def consumer_fixture(e, *, fail_at=None):
    e.episode_length_buf.zero_()
    e.physical_tick, e.stage_actor_update, e.global_actor_update = 0, 0, 5500
    e._numerical_failure_capture = NS(stop_on_first_event=True, recorded=False,
        summary={'enabled': True, 'recorded': False, 'worker_abort_raised': False},
        observe_original_policy_end=lambda *args: None)
    e.get_observations = lambda: TensorDict({'policy': torch.zeros(2,35), 'critic':torch.zeros(2,78)}, batch_size=[2])
    actor = _stock_actor(None)
    with torch.no_grad():
        for name, param in actor.named_parameters():
            if 'std' not in name:
                param.zero_()
        actor.mlp[-1].bias.fill_(.17)
    before = deepcopy(actor.state_dict())
    received = []
    def step(action):
        received.append(action.clone())
        if fail_at == len(received):
            e.physical_tick += 1
            e._numerical_failure_capture.recorded = True
            e._numerical_failure_capture.summary.update(recorded=True, worker_abort_raised=True)
            raise FloatingPointError('first 5ms numerical crossing fixture')
        e.physical_tick += 4
        e.scut_reward.begin_policy(action)
        return tick(e)  # Actual recovery/source reward transition producer, SDK physics double.
    e.step = step
    def forbidden(*a, **k):
        raise AssertionError('No optimizer, reserve, learn or checkpoint is allowed')
    runner = NS(get_inference_policy=lambda: actor, learn=forbidden, save=forbidden,
                alg=NS(actor=actor, update=forbidden))
    return runner, received, before


def test_mean_actor_and_real_producer_journal_have_zero_ppo_credit(entry, rig):
    runner, received, before = consumer_fixture(rig)
    report, checks = {}, []
    entry.validate_restored_policy(runner, rig, 3, report, check_runtime=lambda: checks.append(True))
    assert len(checks) == len(received) == 3
    assert all(torch.allclose(a, torch.full((2,6), .17)) for a in received)
    assert all(torch.equal(value, before[name]) for name, value in runner.alg.actor.state_dict().items())
    assert report['status'] == 'window_completed' and report['physical_steps_completed'] == 12
    assert report['completed_policy_steps'] == 3 and report['qualification'] is False
    assert report['ppo_updates'] == report['actor_direct_loss_credit'] == 0
    assert report['checkpoint_written'] is False
    totals = report['physical_event_totals']
    assert totals['/RecoveryEvidence/policy_intervals'] == 3
    assert totals['/RecoveryEvidence/observed_episodes'] >= 1
    assert totals['/RecoveryEvidence/actor_direct_loss_intervals'] == 0
    assert report['recovery_evidence']['completed_actor_updates'] == 0
    assert report['recovery_evidence']['critic_only_updates'] == 0
    assert (rig.stage_actor_update, rig.global_actor_update) == (0, 5500)
    json.dumps(report, allow_nan=False)


def test_first_substep_failure_preserves_completed_window_only(entry, rig):
    runner, received, _ = consumer_fixture(rig, fail_at=2)
    report = {}
    with pytest.raises(FloatingPointError, match='first 5ms'):
        entry.validate_restored_policy(runner, rig, 3, report, check_runtime=lambda: None)
    assert len(received) == 2 and report['completed_policy_steps'] == 1
    assert report['physical_steps_completed'] == 5
    assert report['numerical_failure_capture']['recorded']
    assert report['status'] == 'failed_or_interrupted'
    assert report['physical_event_totals']['/RecoveryEvidence/policy_intervals'] == 1
    assert report['ppo_updates'] == 0
    json.dumps(report, allow_nan=False)


def test_read_only_behavior_publishes_fixed_cases_without_granting_ppo_credit(entry, rig):
    runner, _, _ = consumer_fixture(rig)
    scalars, observed = [], []
    runner.logger = NS(init_logging_writer=lambda: None,
        writer=NS(add_scalar=lambda *args: scalars.append(args), flush=lambda: None))
    values = {'groups': {'ordinary': {'vx_mae_m_s': .4}},
        'fixed_rehearsal': {'groups': {'stand_h0200': {'vx_mae_m_s': .1}}}}
    metrics = NS(report=lambda: deepcopy(values), observe=lambda data: observed.append(data))
    report = {}
    entry.validate_restored_policy(runner, rig, 3, report,
        check_runtime=lambda: None, behavior_metrics=metrics)
    assert ('ParentInference/fixed17/stand_h0200/vx_mae_m_s', .1, 3) in scalars
    assert len(observed) == 3
    assert all(data['measurement_valid_mask'].dtype == torch.bool for data in observed)
    assert report['ppo_updates'] == report['actor_direct_loss_credit'] == 0
    assert report['checkpoint_written'] is False


def test_validation_branch_exits_before_learning_hooks_and_checkpoint(entry, tmp_path):
    tree = ast.parse(ENTRY.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    node = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'restored_policy_validation')
    text = ENTRY.read_text()
    assert text.index('runner.load(str(args.resume), **resume_load_kwargs)') < text.index('if restored_policy_validation:')
    assert text.index('report["rng_restore"] = restore_rng') < text.index('if restored_policy_validation:')
    assert text.index('env.reset(ids)', text.index('report["rng_restore"] = restore_rng')) < text.index('if restored_policy_validation:')
    assert text.index('if restored_policy_validation:') < text.index('recovery_rollout = configure_recovery_rollout')
    assert text.index('if restored_policy_validation:') < text.index('initial = args.run_dir / "initial_state"') < text.index('runner.learn(args.updates)')
    calls = []
    def observe(runner, env, steps, report, *, check_runtime, behavior_metrics=None):
        check_runtime()
        calls.append('mean_window')
        report.update(status='window_completed', completed_policy_steps=steps, ppo_updates=0)
    scope = dict(restored_policy_validation=True, resume_load_kwargs={},
        budget=NS(check=lambda: calls.append('wall_check')),
        budget_owner=NS(check_runtime=lambda: calls.append('shared_budget_check')),
        report={}, runner=NS(alg=object()), env=object(), args=NS(validate_restored_policy=3, run_dir=tmp_path),
        checkpoint=object(), fresh_critic_state=None, metrics=None,
        validate_restored_policy=observe, PlannedStop=PlannedStop, json=json,
        verify_learning_state_restore=lambda *a, **kw: calls.append('full_state_unchanged') or {'actor_exact':True})
    with pytest.raises(PlannedStop, match='restored_policy_validation_completed'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(ENTRY), 'exec'), scope)
    assert calls == ['wall_check', 'shared_budget_check', 'mean_window', 'full_state_unchanged']
    assert list(tmp_path.iterdir()) == [tmp_path / 'restored_policy_validation.json']


def test_only_read_only_mean_policy_check_can_reduce_formal_population():
    main = next(n for n in ast.parse(ENTRY.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    population_guard = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                           and 'args.num_envs != 4096' in ast.unparse(n.test))
    def reject(message):
        raise ValueError(message)
    for population in (16, 256, 480):
        # Both PPO and sampled-policy validation leave this flag false.
        with pytest.raises(ValueError, match='4096'):
            exec(compile(ast.Module(body=[population_guard], type_ignores=[]), str(ENTRY), 'exec'),
                 {'args': NS(preflight_only=False, validate_scene=0, num_envs=population),
                  'restored_policy_validation': False, 'parser': NS(error=reject)})
        exec(compile(ast.Module(body=[population_guard], type_ignores=[]), str(ENTRY), 'exec'),
             {'args': NS(preflight_only=False, validate_scene=0, num_envs=population),
              'restored_policy_validation': True, 'parser': NS(error=reject)})
    exec(compile(ast.Module(body=[population_guard], type_ignores=[]), str(ENTRY), 'exec'),
         {'args': NS(preflight_only=False, validate_scene=0, num_envs=4096),
          'restored_policy_validation': True, 'parser': NS(error=reject)})
