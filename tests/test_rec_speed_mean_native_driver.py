"""Only pure CPU input/constructor/event checks; no native simulator import."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('isolated_mean_driver', ROOT / 'scripts/run_rec_speed_mean_replay.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
GOLDEN = ROOT.parent / 'serial-dispatch-source028-20261009'
if GOLDEN.is_dir():
    d.prep.frozen_runtime(GOLDEN)


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    if not GOLDEN.is_dir():
        pytest.skip('Exact local e5 runtime unavailable')
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    out = tmp_path_factory.mktemp('native-driver')
    values = {}
    for protocol in d.PROTOCOLS:
        path = out / protocol
        receipt = d.prep.prepare_projection(GOLDEN, path, protocol=protocol)
        config = d.prep.read(path / 'interactive_unscored_contract.json')
        control = load_chassis_control(GOLDEN / config['control_math_source'])
        values[protocol] = (config, control, receipt['projection'])
    return values, design_preflight


def module(original):
    return SimpleNamespace(__name__='wheeled_tasks.chassis.env',
        __file__=str(GOLDEN / 'src/wheeled_tasks/chassis/env.py'), design_preflight=original)


def invoke(target, config, control, **kwargs):
    args = dict(stage='recovery', role='baseline', seed=d.SEED, num_envs=28, block_index=0)
    args.update(kwargs)
    root = args.pop('root', GOLDEN)
    return target.design_preflight(config, control, root, **args)


def test_five_protocols_keep_seven_actual_births_and_full_original_authentication(prepared):
    values, _ = prepared
    source = d.prep.read(GOLDEN / d.prep.REC)
    original = {c['name']: c for c in source['evaluation']['cases']}
    for protocol, (config, _, receipt) in values.items():
        assert [c['name'] for c in config['evaluation']['cases']] == list(d.SCENES)
        assert config['evaluation']['episodes_per_case'] == 4
        assert (receipt['role'], receipt['num_envs'], receipt['block_index']) == ('baseline', 28, 0)
        for case in config['evaluation']['cases'][3:]:
            assert case == original[case['name']]
            assert case['episode_seconds'] == 20.
        duration = next(c['episode_seconds'] for c in config['evaluation']['cases'] if c['name'] == 'stand_305mm')
        assert duration == (10. if protocol == 'nominal' else 22. if protocol.startswith('yaw') else 17.)
        assert config['recovery_training']['side_speed_rad_s'] == 1.5
        assert config['recovery_training']['complete_started_blend'] is True
        assert config['auto_reset'] is False


def test_scoped_di_one_exact_constructor_deepcopy_and_finally_restore(prepared):
    values, original = prepared
    config, control, receipt = values['nominal']
    target = module(original)
    with d.scoped_bootstrap(target, GOLDEN, config, control, receipt) as calls:
        returned = invoke(target, config, control)
        returned['runtime_expected']['physics_dt'] = 999
        assert receipt['runtime_expected']['physics_dt'] == .005
        assert calls == [dict(stage='recovery', role='baseline', seed=d.SEED, num_envs=28, block_index=0)]
        with pytest.raises(ValueError):
            invoke(target, config, control)
    assert target.design_preflight is original
    with pytest.raises(RuntimeError, match='constructor failure'):
        with d.scoped_bootstrap(target, GOLDEN, config, control, receipt):
            invoke(target, config, control)
            raise RuntimeError('constructor failure')
    assert target.design_preflight is original


@pytest.mark.parametrize('mismatch', ('config', 'control', 'root', 'stage', 'role', 'seed', 'num_envs', 'block_index'))
def test_scoped_di_rejects_every_invocation_mismatch(prepared, mismatch, tmp_path):
    values, original = prepared
    config, control, receipt = values['nominal']
    original_calls = []
    def original_guard(*args, **kwargs):
        original_calls.append((args, kwargs))
        raise ValueError('Original production validator was invoked')
    target = module(original_guard)
    bad_config, bad_control, kwargs = deepcopy(config), deepcopy(control), {}
    if mismatch == 'config':
        bad_config['observation_noise_enabled'] = False
    elif mismatch == 'control':
        bad_control['actuators']['leg']['kp'] = 159.
    else:
        kwargs[mismatch] = dict(root=tmp_path, stage='speed', role='full', seed=d.SEED + 1,
                                num_envs=44, block_index=1)[mismatch]
    with pytest.raises(ValueError, match='Original production validator'):
        with d.scoped_bootstrap(target, GOLDEN, config, control, receipt):
            invoke(target, bad_config, bad_control, **kwargs)
    assert target.design_preflight is original_guard
    assert len(original_calls) == 1
    assert original_calls[0][0] == (bad_config, bad_control, kwargs.get('root', GOLDEN))


def test_scoped_di_refuses_changed_receipt_and_restores_on_missing_call(prepared):
    values, original = prepared
    config, control, receipt = values['nominal']
    target = module(original)
    bad = deepcopy(receipt)
    bad['runtime_expected']['modules']['command_transport'] = False
    with pytest.raises(ValueError, match='authenticated isolated'):
        with d.scoped_bootstrap(target, GOLDEN, config, control, bad):
            pass
    assert target.design_preflight is original


def test_duplicate_exact_call_really_delegates_to_original_validator(prepared):
    values, _ = prepared
    config, control, receipt = values['nominal']
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return {'original_validator_result': True}
    target = module(original)
    with d.scoped_bootstrap(target, GOLDEN, config, control, receipt) as isolated:
        assert invoke(target, config, control)['role'] == 'baseline'
        assert invoke(target, config, control) == {'original_validator_result': True}
        assert len(isolated) == 1
    assert len(calls) == 1 and target.design_preflight is original
    with pytest.raises(ValueError, match='did not consume'):
        with d.scoped_bootstrap(target, GOLDEN, config, control, receipt):
            pass
    assert target.design_preflight is original


def test_scoped_di_revalidates_profile_bytes_at_call(prepared):
    values, original = prepared
    config, control, receipt = values['nominal']
    path = Path(config['design_preflight']['file'])
    before = path.read_bytes()
    target = module(original)
    try:
        with pytest.raises(ValueError):
            with d.scoped_bootstrap(target, GOLDEN, config, control, receipt):
                path.write_bytes(before + b' ')
                invoke(target, config, control)
    finally:
        path.write_bytes(before)
    assert target.design_preflight is original


def diagnostic(done):
    from wheeled_tasks.chassis.evaluation import RequestProtocolMetrics
    value = {'done': torch.tensor(done), 'terminated': torch.tensor(done), 'success': torch.zeros(len(done), dtype=torch.bool),
             'reasons': {'boundary': torch.tensor(done)}}
    for key in RequestProtocolMetrics.RECOVERY_COUNTERS:
        value['recovery_' + key] = torch.zeros(len(done))
    value['recovery_enabled'] = torch.ones(len(done), dtype=torch.bool)
    value['recovery_holding_rl'] = torch.ones(len(done), dtype=torch.bool)
    value['recovery_stable_peak_seconds'] = torch.full((len(done),), .005)
    for key in d.CONDITIONS:
        value['recovery_stable_' + key + '_pass'] = torch.ones(len(done), dtype=torch.bool)
    return value


def ownership(phase='RL'):
    return dict(phase=phase, pure_rl=True, motion_released=False, ownership='rl_hold')


@pytest.mark.parametrize('protocol,release,target', (
    ('vx_positive', 5., [.5, 0.]), ('vx_negative', 5., [-.5, 0.]),
    ('yaw_positive', 10., [0., 1.]), ('yaw_negative', 10., [0., -1.])))
def test_command_release_and_full_quiet_are_unconditional_on_robot_speed(protocol, release, target):
    assert d.request(protocol, 'stand_305mm', release - .02)[:2] == target
    assert d.request(protocol, 'stand_305mm', release) == [0., 0., .305]
    for group in d.prep.FALLEN:
        assert d.request(protocol, group, 2.) == [0., 0., .305]
    events = d.NativeEvents(['stand_305mm'], protocol)
    data = diagnostic([False])
    owner = [ownership()]
    _, quiet = events.observe(release, data, torch.tensor([True]), owner, [[0., 0.]], [[.1, 0.]])
    assert not bool(quiet[0])
    assert events.rows[0]['first_zero_request_s'] == release
    assert events.rows[0]['first_applied_zero_s'] is None
    events.observe(release + .02, data, torch.tensor([True]), owner, [[0., 0.]], [[1e-8, 0.]])
    assert events.rows[0]['first_applied_zero_s'] is None  # Exact command zero, not robot-speed filtering.
    for tick in range(500):
        _, quiet = events.observe(release + .34 + tick * .02, data, torch.tensor([True]), owner, [[0., 0.]], [[0., 0.]])
        assert bool(quiet[0])
    row = events.report()['replicas'][0]
    assert row['full_ten_second_quiet_window'] is True
    assert row['quiet_observed_seconds'] == pytest.approx(10.)
    assert row['first_applied_zero_s'] == pytest.approx(release + .34)
    _, quiet = events.observe(release + 11., data, torch.tensor([True]), owner, [[0., 0.]], [[0., 0.]])
    assert not bool(quiet[0])  # Exactly ten seconds, no late favorable samples.


def test_early_terminal_censors_quiet_and_native_denominator_once():
    events = d.NativeEvents(['stand_305mm'] * 4, 'vx_positive')
    for tick in range(100):
        events.observe(5.34 + tick * .02, diagnostic([tick == 99] * 4), torch.tensor([False] * 4),
                       [ownership('BLEND')] * 4, [[0., 0.]] * 4, [[0., 0.]] * 4)
    events.observe(10., diagnostic([True] * 4), torch.tensor([True] * 4),
                   [ownership()] * 4, [[0., 0.]] * 4, [[0., 0.]] * 4)
    report = events.report()
    assert report['groups']['stand_305mm']['native_recovery_ended_count'] == 4
    assert all(not row['full_ten_second_quiet_window'] for row in report['replicas'])
    assert all(row['quiet_observed_seconds'] == pytest.approx(2.) for row in report['replicas'])
    assert all(row['policy_owned_seconds'] == 0. for row in report['replicas'])
    assert all(row['terminal_reasons'] == ['boundary'] for row in report['replicas'])
    assert report['qualification'] is None


def test_rest_origin_excludes_moving_displacement_with_existing_metrics():
    from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
    metrics = EpisodeMetrics(['stand_305mm'], 'cpu', .02)
    quiet_metrics = EpisodeMetrics(['stand_305mm'], 'cpu', .02)
    events = d.NativeEvents(['stand_305mm'], 'vx_positive')
    data = dict(episode_ticks=torch.tensor([250]), position=torch.tensor([[4., 0., .305]]),
        velocity=torch.tensor([[.12, 0., 0.]]), omega=torch.zeros(1, 3), height=torch.tensor([.305]),
        commands=torch.tensor([[0., 0., .305]]), reward=torch.zeros(1), motor_effort=torch.zeros(1, 6),
        gravity=torch.tensor([[0., 0., -1.]]), gap=torch.zeros(1), done=torch.tensor([False]),
        terminated=torch.tensor([False]), success=torch.tensor([False]), reasons={'boundary': torch.tensor([False])})
    data.update(diagnostic([False]))
    d.observe_interval(events, metrics, quiet_metrics, 5.34, data, torch.tensor([True]), [ownership()],
                       [[0., 0., .305]], torch.tensor([[3.998, 0.]]))
    group = quiet_metrics.report()['groups']['stand_305mm']
    assert group['stand_drift_max_m'] == pytest.approx(.002, abs=1e-6)
    assert group['planar_speed_mean_m_s'] == pytest.approx(.12)


def test_driver_import_and_default_cpu_path_have_no_native_dependency():
    assert 'isaaclab.app' not in sys.modules
    assert 'wheeled_tasks.chassis.env' not in sys.modules
    assert 'wheeled_tasks.chassis.eval_env' not in sys.modules
    with pytest.raises(ValueError):
        d.request('unknown', 'stand_305mm', 0.)


def test_real_per_case_timeout_bounds_full_yaw_quiet_window():
    # e5's global max_episode_length may remain20s; its exact-case limits can
    # legitimately extend only stand305 to22s without mutating any timeout.
    env = SimpleNamespace(num_envs=3, max_episode_length=1000,
                          episode_limits=torch.tensor([500, 1000, 1100]))
    assert d.native_tick_limit(env) == 1101
    env.episode_limits[-1] = 0
    with pytest.raises(ValueError, match='episode limits'):
        d.native_tick_limit(env)
