"""CPU authentication and independent-child guards; no Kit or scene is started."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('independent_mean_driver', ROOT / 'scripts/run_rec_speed_mean_replay.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
GOLDEN = ROOT.parent / 'serial-dispatch-source028-20261009'
MODELS = ROOT.parent.parent / 'isaac_wheeled_rl_schedule/models'
PAIR = tuple((MODELS / name / 'model.pt', MODELS / name / 'policy.onnx') for name in
    ('v6_recovery_candidate_41393_20261009', 'v6_speed_periodic_42000_actor597_20261009'))


@pytest.fixture(scope='module')
def authenticated():
    if not GOLDEN.is_dir() or not all(path.is_file() for pair in PAIR for path in pair):
        pytest.skip('Frozen e5 runtime and the two sealed original endpoints are required')
    root = d.prep.frozen_runtime(GOLDEN)
    contracts, identities = zip(*(d.prep.authenticate_candidate(root, *pair) for pair in PAIR))
    abi = d.prep.shared_abi(contracts)
    d.prep.validate_pair(identities)
    return root, contracts, identities, abi


@pytest.fixture
def projection(tmp_path, authenticated):
    root = authenticated[0]
    directory = tmp_path / 'shared_nominal'
    receipt = d.prep.prepare_projection(root, directory, protocol='nominal')
    d.prep.write(directory / 'projection_receipt.json', receipt)
    pin = d.prep.digest(directory / 'projection_receipt.json')
    return directory, pin


def mode(**changes):
    values = dict(candidate_index=0, execute=True, protocol='nominal', interface_smoke=False,
        prepared_projection=Path('/prepared'), projection_receipt_sha256='a' * 64)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('changes', (
    {'candidate_index': -1}, {'candidate_index': 2}, {'candidate_index': True},
    {'execute': False}, {'protocol': None}, {'interface_smoke': True},
    {'prepared_projection': None}, {'projection_receipt_sha256': None},
    {'candidate_index': None}))
def test_single_mode_requires_both_pins_and_explicit_native_child(changes):
    with pytest.raises(ValueError):
        d.validate_single_mode(mode(**changes))


def test_default_mode_requires_no_prepared_flags():
    d.validate_single_mode(mode(candidate_index=None, execute=False, protocol=None,
        prepared_projection=None, projection_receipt_sha256=None))
    d.validate_single_mode(mode(candidate_index=1))


@pytest.mark.parametrize('filename', ('interactive_unscored_contract.json',
    'interactive_unscored_expectations.json', 'projection_changes.json', 'projection_receipt.json'))
def test_prepared_bytes_tampering_is_rejected(projection, authenticated, filename):
    directory, pin = projection
    path = directory / filename
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA mismatch'):
        d.verify_prepared_projection(authenticated[0], directory, 'nominal', pin)


def test_repinning_inner_files_cannot_bypass_external_receipt_pin(projection, authenticated):
    directory, pin = projection
    path = directory / 'interactive_unscored_contract.json'
    config = d.prep.read(path)
    config['observation_noise_enabled'] = not config['observation_noise_enabled']
    d.prep.write(path, config)
    receipt_path = directory / 'projection_receipt.json'
    receipt = d.prep.read(receipt_path)
    receipt['config_sha256'] = d.prep.digest(path)
    d.prep.write(receipt_path, receipt)
    with pytest.raises(ValueError, match='receipt SHA mismatch'):
        d.verify_prepared_projection(authenticated[0], directory, 'nominal', pin)


@pytest.mark.parametrize('protocol,pin', (('yaw_positive', None), ('nominal', 'bad'), ('nominal', '0' * 64)))
def test_prepared_protocol_and_receipt_pin_are_explicit(projection, authenticated, protocol, pin):
    directory, original = projection
    with pytest.raises(ValueError):
        d.verify_prepared_projection(authenticated[0], directory, protocol, original if pin is None else pin)


def test_externally_bound_design_path_and_source_projection_are_required(projection, authenticated):
    directory, _ = projection
    path = directory / 'interactive_unscored_contract.json'
    config = d.prep.read(path)
    config['design_preflight']['file'] = str(directory / 'other_expectations.json')
    d.prep.write(path, config)
    receipt_path = directory / 'projection_receipt.json'
    receipt = d.prep.read(receipt_path)
    receipt['config_sha256'] = d.prep.digest(path)
    d.prep.write(receipt_path, receipt)
    with pytest.raises(ValueError, match='exact expectations path'):
        d.verify_prepared_projection(authenticated[0], directory, 'nominal', d.prep.digest(receipt_path))
    config['design_preflight']['file'] = str(directory / 'interactive_unscored_expectations.json')
    config['observation_noise_enabled'] = not config['observation_noise_enabled']
    d.prep.write(path, config)
    receipt['config_sha256'] = d.prep.digest(path)
    d.prep.write(receipt_path, receipt)
    with pytest.raises(ValueError, match='full-role source projection'):
        d.verify_prepared_projection(authenticated[0], directory, 'nominal', d.prep.digest(receipt_path))


def result_fixture(output, projection, authenticated, index):
    root, _, identities, abi = authenticated
    directory, pin = projection
    config, receipt, binding = d.verify_prepared_projection(root, directory, 'nominal', pin)
    from wheeled_tasks.chassis.preflight import validate_runtime_state
    from wheeled_tasks.chassis.recovery_script_revision import runtime_expectation
    runtime = validate_runtime_state(receipt['projection']['runtime_expected'],
        deepcopy(receipt['projection']['runtime_expected']))
    timing = dict(physics_dt=.005, policy_dt=.02, decimation=4, pc_control_dt=.005,
        target_update='once_per_policy_step', feedback_control='after_each_physical_state_refresh',
        runtime_substep_verification='pending_first_step')
    groups = [name for name in d.SCENES for _ in range(4)]
    from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
    from wheeled_tasks.chassis.evaluation import RequestProtocolMetrics, recovery_episode_diagnostics
    first_metrics = EpisodeMetrics(groups, 'cpu', .02).report()
    quiet_metrics = deepcopy(first_metrics)
    for report in (first_metrics, quiet_metrics):
        for group in report['groups'].values():
            group['termination_reasons'] = dict(fall=0, closure_gap=0, recovery_script=0)
    for name, group in first_metrics['groups'].items():
        frames = 4000 if name in d.prep.FALLEN else 2000
        group.update(frames=frames, observed_frames=frames, episodes=4, timeouts=4)
    event_groups = {}
    for name in d.SCENES:
        count = 4 if name in d.prep.FALLEN else 0
        event_groups[name] = dict(native_recovery_ended_count=count,
            native_recovery_episodes=[{key: 0. for key in RequestProtocolMetrics.RECOVERY_COUNTERS} for _ in range(count)],
            native_recovery_diagnostics=[])
        event_groups[name]['native_recovery_diagnostics'] = recovery_episode_diagnostics(
            {'recovery_episodes': event_groups[name]['native_recovery_episodes']}, count)
    measurement = dict(identity=identities[index], original_candidate_index=index, policy_ticks=1000,
        control_substep_readback=dict(policy_steps=1000, physics_steps=4000, target_decodes=1,
            pd_updates=4, feedback_refreshes=4, physics_dt=.005, policy_dt=.02),
        right_censored=0, reset_fingerprint=dict(groups=groups, actor_observation_sha256='1' * 64,
            clone_indices=list(range(28)), fixed_birth_samples=[[0.] * 5] * 28, root_pose=[[0.] * 7] * 28,
            motor_positions=[[0.] * 18] * 28, masses=[[1.] * 19] * 28, friction=[1.] * 28),
        native_events=dict(run_scope='interactive_unscored', qualification=None, groups=event_groups,
            replicas=[dict(case=name, right_censored=False, terminal_s=20. if name in d.prep.FALLEN else 10.) for name in groups]),
        native_episode_metrics=first_metrics, fixed_quiet_window_metrics=quiet_metrics,
        fixed_quiet_window_target_group=None)
    result = dict(status='native_mean_replay_completed_unscored', run_scope='interactive_unscored',
        protocol='nominal', PPO_updates=0, optimizer_constructed=False, qualification=None,
        world_mode='independent_world', selected_candidate_index=index, selected_identity=identities[index],
        original_candidates=list(identities), shared_abi=abi, prepared_projection=binding,
        native_constructor=dict(pid=1234 + index, count=1, fresh_process=True, runtime_root=str(root),
            source_runtime_commit=d.prep.E5, environment_class='wheeled_tasks.chassis.eval_env.FixedCaseEnv'),
        bootstrap_call=[dict(stage=d.environment_stage(config), role='baseline', seed=d.SEED, num_envs=28, block_index=0)],
        native_readback=dict(runtime_preflight=runtime, qualification=None, control_timing=timing,
            controller=dict(runtime_expectation(d.prep.read(root / d.prep.REC)), complete_started_blend=True)),
        startup_report=dict(runtime_preflight=runtime, control_timing=timing, num_envs=28,
            scene_group_counts={name: 4 for name in d.SCENES}, design_preflight=receipt['projection']),
        candidates=[measurement])
    output.mkdir(exist_ok=True)
    d.prep.write(output / f'candidate_{index:02d}.json', measurement)
    traces = []
    for time_s in (.02, 10., 20.):
        traces.append(dict(time_s=time_s, active=[name in d.prep.FALLEN or time_s <= 10. for name in groups],
            done=[time_s == (20. if name in d.prep.FALLEN else 10.) for name in groups],
            terminated=[False] * 28, final_policy_mask=[True] * 28,
            requested=[[0., 0., .305]] * 28, applied=[[0., 0., .305]] * 28,
            velocity=[[0.] * 3] * 28, omega=[[0.] * 3] * 28, gravity=[[0., 0., -1.]] * 28,
            position=[[0., 0., .305]] * 28, height=[.305] * 28))
    d.prep.write(output / f'candidate_{index:02d}_trace.json', traces)
    d.prep.write(output / 'native_replay.json', result)
    return result


def verify(output, projection, authenticated, index=0):
    root, _, identities, abi = authenticated
    return d.verify_single_child_completion(output, 'nominal', index, identities, root, *projection, abi)


@pytest.mark.parametrize('index', (0, 1))
def test_complete_single_result_binds_original_index_and_native_runtime(tmp_path, projection, authenticated, index):
    output = tmp_path / ('child' + str(index))
    expected = result_fixture(output, projection, authenticated, index)
    assert verify(output, projection, authenticated, index) == expected
    with pytest.raises(ValueError, match='failed or incomplete'):
        d.verify_child_completion(output, 'nominal')  # Existing dual completion guard stays strict.


@pytest.mark.parametrize('mutation', (
    'failed', 'wrong_index', 'wrong_identity', 'pair_reversed', 'empty', 'PPO_false',
    'optimizer', 'qualification', 'runtime_missing', 'runtime_field', 'startup_runtime_field',
    'bootstrap_twice', 'constructor_twice', 'prepared_pin', 'substeps', 'censored', 'wrong_group',
    'empty_metrics', 'missing_terminal', 'empty_ended_ledger', 'missing_ledger_counters', 'fake_original_index',
    'wrong_denominator', 'late_terminal', 'fabricated_diagnostic'))
def test_zero_exit_style_single_report_cannot_hide_failure_or_mismatch(tmp_path, projection, authenticated, mutation):
    output = tmp_path / 'child'
    result = result_fixture(output, projection, authenticated, 0)
    if mutation == 'failed': result['status'] = 'failed'
    elif mutation == 'wrong_index': result['selected_candidate_index'] = 1
    elif mutation == 'wrong_identity': result['candidates'][0]['identity'] = result['original_candidates'][1]
    elif mutation == 'pair_reversed': result['original_candidates'].reverse()
    elif mutation == 'empty': result['candidates'] = []
    elif mutation == 'PPO_false': result['PPO_updates'] = False
    elif mutation == 'optimizer': result['optimizer_constructed'] = True
    elif mutation == 'qualification': del result['qualification']
    elif mutation == 'runtime_missing': del result['native_readback']['runtime_preflight']['observed']
    elif mutation == 'runtime_field': del result['native_readback']['runtime_preflight']['observed']['command_transport_model']
    elif mutation == 'startup_runtime_field': result['startup_report']['runtime_preflight']['observed']['leg_kp'] = 159.
    elif mutation == 'bootstrap_twice': result['bootstrap_call'] *= 2
    elif mutation == 'constructor_twice': result['native_constructor']['count'] = 2
    elif mutation == 'prepared_pin': result['prepared_projection']['receipt_sha256'] = '0' * 64
    elif mutation == 'substeps': result['candidates'][0]['control_substep_readback']['feedback_refreshes'] = 3
    elif mutation == 'censored': result['candidates'][0]['right_censored'] = 1
    elif mutation == 'wrong_group': result['candidates'][0]['native_events']['replicas'][0]['case'] = 'wrong'
    elif mutation == 'empty_metrics': result['candidates'][0]['native_episode_metrics']['groups'] = {name: {} for name in d.SCENES}
    elif mutation == 'missing_terminal': result['candidates'][0]['native_events']['replicas'][0]['terminal_s'] = None
    elif mutation == 'empty_ended_ledger': result['candidates'][0]['native_events']['groups'][d.prep.FALLEN[0]]['native_recovery_episodes'] = []
    elif mutation == 'missing_ledger_counters': result['candidates'][0]['native_events']['groups'][d.prep.FALLEN[0]]['native_recovery_episodes'][0] = {}
    elif mutation == 'fake_original_index': result['candidates'][0]['original_candidate_index'] = False
    elif mutation == 'wrong_denominator': result['candidates'][0]['native_episode_metrics']['groups']['stand_305mm']['frames'] += 1
    elif mutation == 'late_terminal': result['candidates'][0]['native_events']['replicas'][0]['terminal_s'] = 20.
    elif mutation == 'fabricated_diagnostic': result['candidates'][0]['native_events']['groups'][d.prep.FALLEN[0]]['native_recovery_diagnostics'][0]['reason'] = 'fabricated_success'
    d.prep.write(output / 'native_replay.json', result)
    with pytest.raises(ValueError): verify(output, projection, authenticated)


@pytest.mark.parametrize('filename', ('candidate_00.json', 'candidate_00_trace.json'))
def test_single_result_requires_actual_selected_files(tmp_path, projection, authenticated, filename):
    output = tmp_path / 'child'
    result_fixture(output, projection, authenticated, 0)
    (output / filename).unlink()
    with pytest.raises(OSError): verify(output, projection, authenticated)


@pytest.mark.parametrize('mutation', ('missing_time', 'missing_velocity', 'nonfinite_velocity', 'missing_masks', 'missing_terminal_frame'))
def test_trace_must_support_actual_requested_and_physical_comparison(tmp_path, projection, authenticated, mutation):
    output = tmp_path / 'child'
    result_fixture(output, projection, authenticated, 0)
    path = output / 'candidate_00_trace.json'
    rows = d.prep.read(path)
    if mutation == 'missing_time': del rows[0]['time_s']
    elif mutation == 'missing_velocity': del rows[0]['velocity']
    elif mutation == 'nonfinite_velocity': rows[0]['velocity'][0][0] = float('nan')
    elif mutation == 'missing_masks': del rows[0]['done']
    elif mutation == 'missing_terminal_frame': rows.pop()
    import json
    path.write_text(json.dumps(rows))  # Deliberately allows the invalid NaN fixture.
    with pytest.raises(ValueError): verify(output, projection, authenticated)


def arguments(output):
    argv = ['--runtime-root', str(GOLDEN), '--output', str(output)]
    for pt, onnx in PAIR: argv += ['--candidate', str(pt), str(onnx)]
    return argv


def test_two_child_entries_share_projection_and_default_remains_cpu_only(tmp_path, projection, authenticated, monkeypatch):
    root, contracts, identities, _ = authenticated
    auth_calls, native_calls = [], []
    def authenticate(observed_root, pt, onnx):
        assert observed_root == root
        index = PAIR.index((Path(pt), Path(onnx)))
        auth_calls.append(index)
        return deepcopy(contracts[index]), deepcopy(identities[index])
    monkeypatch.setattr(d.prep, 'authenticate_candidate', authenticate)
    directory, pin = projection
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    def native(args, observed_root, configs, observed_pair):
        assert tuple(observed_pair) == identities and observed_root == root
        assert configs == directory and args.prepared_binding['receipt_sha256'] == pin
        native_calls.append(args.candidate_index)
        result = result_fixture(args.output, projection, authenticated, args.candidate_index)
        return result
    monkeypatch.setattr(d, 'run_protocol', native)
    for index in (0, 1):
        output = tmp_path / ('native' + str(index))
        report = d.main(arguments(output) + ['--execute', '--protocol', 'nominal', '--candidate-index', str(index),
            '--prepared-projection', str(directory), '--projection-receipt-sha256', pin])
        assert report['selected_candidate_index'] == index
        assert not (output / 'nominal').exists()
    assert native_calls == [0, 1] and auth_calls == [0, 1, 0, 1]
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    output = tmp_path / 'default_cpu'
    report = d.main(arguments(output))
    assert report['status'] == 'CPU_prepared_scene_not_run' and report['scene_run'] is False
    assert 'world_mode' not in report and len(report['candidates']) == 2
    assert all((output / protocol / 'projection_receipt.json').is_file() for protocol in d.PROTOCOLS)
    assert native_calls == [0, 1] and auth_calls == [0, 1, 0, 1, 0, 1]
    assert 'isaaclab.app' not in sys.modules
    assert 'wheeled_tasks.chassis.env' not in sys.modules


def test_missing_pin_and_out_of_range_index_fail_before_authentication(tmp_path, monkeypatch):
    monkeypatch.setattr(d.prep, 'frozen_runtime', lambda *_: pytest.fail('Invalid CLI reached runtime authentication'))
    with pytest.raises(ValueError, match='explicitly pinned'):
        d.main(arguments(tmp_path / 'missing') + ['--execute', '--protocol', 'nominal', '--candidate-index', '0'])
    with pytest.raises(SystemExit):
        d.main(arguments(tmp_path / 'range') + ['--execute', '--protocol', 'nominal', '--candidate-index', '2'])
    assert not (tmp_path / 'missing').exists() and not (tmp_path / 'range').exists()
