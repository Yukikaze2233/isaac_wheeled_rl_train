"""Synthetic CPU reader tests; these fixtures are not native physics evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).with_name('compare_clean_exploration.py')
SPEC = importlib.util.spec_from_file_location('clean_exploration_comparison', PATH)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from locomotion_exploration_sampling import FirstCohortLedger
sys.path.insert(0, str(ROOT / 'src'))

NAMES = tuple(case for case in reader.CASES for _ in range(4))


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def actions(arm):
    for tick in range(1, 1001):
        mu, sample = [[0.] * 6] * 12, [[.1] * 6] * 12
        yield {'policy_tick': tick, 'phase': ['stand'] * 12,
            'requested_command': [[0., 0., .305]] * 12, 'visible_command': [[0., 0., .305]] * 12,
            'rng_before_sha256': str(tick - 1), 'rng_after_sha256': str(tick),
            'draw_counter': tick, 'sample_element_count': tick * 72,
            'mu': mu, 'sample': sample, 'sent_policy_action': mu if arm == 'mean' else sample}


def frames(arm, *, ticks=range(1, 4001)):
    for tick in ticks:
        for lane, case in enumerate(NAMES):
            live = arm == 'mean' or lane != 0 or tick <= 8
            requested = [0.] * 6
            if lane == 0 and arm == 'mean' and tick > 8:
                requested[0] = 2.
            if lane == 0 and arm == 'stochastic' and tick <= 8:
                requested[1] = 2.
            yield {'lane': lane, 'case': case, 'physics_tick': tick, 'first_cohort': live,
                'phase': 'stand' if case == 'stand_305mm' else 'steady',
                'requested_motor_effort': requested, 'current_motor_bounds': [1.] * 6,
                'root_link_velocity_body_m_s': [.1, 0., 0.],
                'root_link_velocity_world_m_s': [.1, 0., 0.],
                'root_angular_velocity_body_rad_s': [0., 0., .03],
                'command': [0., 0., .305], 'height_m': .305,
                'pitch_deg': 2., 'roll_deg': 3., 'root_position_local_m': [.1, 0., .305],
                'phase_anchor_xy_m': [0., 0.]}


@pytest.fixture
def pair(tmp_path, monkeypatch):
    paths = {arm: tmp_path / arm for arm in ('mean', 'stochastic')}
    for arm, path in paths.items():
        path.mkdir()
        ledger = FirstCohortLedger(NAMES)
        failed = [arm == 'stochastic' and lane == 0 for lane in range(12)]
        ledger.observe(done=failed, terminated=failed, timeouts=[False] * 12,
                       success=[False] * 12, reasons={'boundary': [False] * 12}, physical_tick=8)
        ledger.observe(done=[True] * 12, terminated=failed, timeouts=[not x for x in failed],
                       success=[False] * 12, reasons={'boundary': [False] * 12}, physical_tick=4000)
        cohort = ledger.summary(physical_tick=4000)
        report = {'arm': arm, 'status': 'native_probe_completed', 'execution_status': 'completed',
            'policy_steps': 1000, 'physics_steps': 4000, 'formal_updates_consumed': 0,
            'actor_tensors_unchanged': True, 'source_identity_final_verified': True,
            'sealed_inputs_final_verified': True, 'source_revision': 'a' * 40,
            'source_authentication': {'synthetic': True},
            'input_authentication': {'files': {'checkpoint': {'sha256': 'b' * 64}, 'contract': {'sha256': 'c' * 64}}},
            'sampling_seed': 93001, 'native_policy_action_order': ['p'] * 6,
            'native_control_order': ['c'] * 6, 'draw_counter': 1000, 'sample_element_count': 72000,
            'first_cohort': cohort}
        write(path / 'first_cohort.json', {**cohort, 'lane_phase_metrics': []})
        write(path / 'birth.json', {'synthetic': True})
        write(path / 'clean_design_expectations.json', {'synthetic': True})
        design_sha = reader.sha(path / 'clean_design_expectations.json')
        write(path / 'clean_effective_contract.json', {'design_preflight': {
            'file': str(path / 'clean_design_expectations.json'), 'sha256': design_sha}, 'physics_dt': .005})
        raw_sha = reader.sha(path / 'clean_effective_contract.json')
        report['fixture'] = {'effective_contract_sha256': raw_sha, 'source_contract_sha256': 'c' * 64,
                             'design_sha256': design_sha}
        write(path / 'clean_fixture_receipt.json', report['fixture'])
        audit = {'contract_path': str(path / 'clean_effective_contract.json'), 'effective_contract_sha256': raw_sha,
                 'status': 'passed', 'effective_contract_sha_matches': True, 'runtime_config_differences': [],
                 'allowed_constructor_overrides': {'stage': 'speed', 'num_envs': 12, 'seed': 190619, 'terrain_level': 1.}}
        write(path / 'runtime_config_validation.json', audit)
        write(path / 'runtime.json', {'scene_groups': NAMES, 'birth': {'synthetic': True}, 'runtime_config_validation': audit})
        write(path / 'report.json', report)
        for name in ('actions.jsonl', 'physical_frames.jsonl'):
            (path / name).write_text('synthetic records supplied by CPU fixture\n')

    def records(path):
        arm = Path(path).parent.name
        return actions(arm) if Path(path).name == 'actions.jsonl' else frames(arm)
    monkeypatch.setattr(reader, 'records', records)
    return paths


def test_common_saturation_excludes_longer_survivor_tail(pair):
    result = reader.compare(pair['mean'], pair['stochastic'])
    assert result['common_exposure_physics_ticks_per_lane'] == [8] + [4000] * 11
    own = result['first_cohort_torque_saturation']['mean']['stand_305mm']
    shared = result['common_exposure_torque_saturation']
    assert own['count'] == 16000 and own['hit_counts'] == [3992, 0, 0, 0, 0, 0]
    assert shared['mean']['stand_305mm']['count'] == shared['stochastic']['stand_305mm']['count'] == 12008
    assert shared['mean']['stand_305mm']['hit_counts'] == [0] * 6
    assert shared['stochastic']['stand_305mm']['hit_counts'] == [0, 8, 0, 0, 0, 0]
    assert shared['stochastic']['stand_305mm']['saturation_fraction_control_order'][1] == pytest.approx(8 / 12008)
    assert result['qualification'] is None and result['formal_updates_consumed'] == 0
    bindings = result['authenticated_arm_local_bindings']
    assert bindings['mean']['raw_effective_contract_sha256'] != bindings['stochastic']['raw_effective_contract_sha256']
    assert bindings['mean']['canonical_effective_contract_sha256'] == bindings['stochastic']['canonical_effective_contract_sha256']


@pytest.mark.parametrize('change,match', [
    (lambda audit: audit.update(contract_path='/other/clean_effective_contract.json'), 'contract path'),
    (lambda audit: audit.update(effective_contract_sha256='0' * 64), 'contract SHA'),
    (lambda audit: audit['allowed_constructor_overrides'].update(terrain_level=.5), 'config validation differ'),
    (lambda audit: audit['runtime_config_differences'].append({'path': 'physics_dt'}), 'config validation differ'),
])
def test_runtime_output_normalization_preserves_sha_and_all_other_fields(pair, change, match):
    path = pair['stochastic']
    runtime = reader.read(path / 'runtime.json')
    change(runtime['runtime_config_validation'])
    write(path / 'runtime.json', runtime)
    write(path / 'runtime_config_validation.json', runtime['runtime_config_validation'])
    with pytest.raises(ValueError, match=match):
        reader.compare(pair['mean'], path)


@pytest.mark.parametrize('change,match', [
    (lambda row: row.update(case='forward_0p5'), 'lane/case/replica'),
    (lambda row: row['terminal'].update(physical_tick=12), 'first terminal'),
    (lambda row: row.update(partition='timeout'), 'partition'),
])
def test_native_first_terminal_metadata_cannot_be_relabelled(pair, change, match):
    path = pair['stochastic']
    cohort = reader.read(path / 'first_cohort.json')
    change(cohort['rows'][0])
    write(path / 'first_cohort.json', cohort)
    report = reader.read(path / 'report.json')
    report['first_cohort'] = {k: v for k, v in cohort.items() if k != 'lane_phase_metrics'}
    write(path / 'report.json', report)
    with pytest.raises(ValueError, match=match):
        reader.compare(pair['mean'], path)


def test_export_marks_empty_case_without_zero_filling_values(tmp_path, monkeypatch):
    import torch.utils.tensorboard
    scalars, texts = [], []

    class Writer:
        def __init__(self, path, **kwargs):
            Path(path).mkdir()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def add_scalar(self, name, value, step):
            scalars.append((name, value, step))
        def add_text(self, name, value, step):
            texts.append(value)

    monkeypatch.setattr(torch.utils.tensorboard, 'SummaryWriter', Writer)
    write(tmp_path / 'report.json', {'fixture_role': 'clean'})
    rows = list(frames('stochastic', ticks=(4, 8, 12)))
    # The native first-terminal frame at tick 8 remains eligible; tick 12 has
    # no live stand replicas in this synthetic event-export edge case.
    for row in rows:
        if row['physics_tick'] == 12 and row['case'] == 'stand_305mm':
            row['first_cohort'] = False
    monkeypatch.setattr(reader, 'records', lambda path: iter(deepcopy(rows)))
    exported = reader.export_events(tmp_path, 'stochastic')
    prefix = 'ExplorationProbe/clean/stochastic/stand_305mm/'
    assert (prefix + 'first_cohort_live_count', 0, 12) in scalars
    assert not any(name == prefix + 'planar_speed_m_s' and step == 12 for name, _, step in scalars)
    assert (prefix + 'first_cohort_live_count', 4, 8) in scalars
    assert exported['sampling_hz'] == 50 and exported['server_started'] is False
    assert any('Self-survivor instantaneous 50Hz sampled' in text for text in texts)


def pipeline_capture(arm, policies=2):
    """Native CPU delay buffers, real RNG and real encoder; no simulated physics."""
    import torch
    from probe_locomotion_exploration import PhysicalRecorder
    from wheeled_tasks.chassis.signal_delay import PhysicalSignalDelay
    from wheeled_tasks.chassis.task_campaign import signal_delay_settings
    from wheeled_tasks.chassis.scut_observation import build_sensor18
    delay = PhysicalSignalDelay(12, 'cpu', signal_delay_settings('speed'), .005,
                                torch.Generator().manual_seed(190619))
    initial = torch.zeros(12, 18)
    delay.reset(torch.arange(12), initial)
    runtime = {'scene_groups': list(NAMES), 'signal_delay': {
        'instance_present': True, 'class_name': 'PhysicalSignalDelay', 'cfg': delay.cfg,
        'task_actor_update': 4185, 'strength': 0., 'policy_tick': 0, 'physics_tick': 0,
        'physics_steps_observed': 0, 'quantum_ms': 5.,
        'enabled': delay.enabled.tolist(), 'profile': delay.profile.tolist(), 'base_ms': delay.base_ms.tolist(),
        'bounds_ms': delay.bounds_ms.tolist(), 'lags_physics_steps': delay.lags.tolist(),
        'ages_ms': (delay.lags * 5.).tolist(), 'feedback_sensor18': delay.feedback.tolist(),
        'raw_jitter_draw_coverage': 'not_covered_native_local_not_persisted'}}
    birth = {'fixture_role': 'pipeline_age_only', 'task_actor_update': 4185,
             'scene_groups': list(NAMES), 'v5_nominal6': [0.] * 6}
    cohort = {'rows': [{'exposure_physics_ticks': policies * 4} for _ in NAMES]}
    owner = SimpleNamespace(delay=delay, env=SimpleNamespace(scene_groups=NAMES))
    state_rows = lambda: PhysicalRecorder.delay_state_rows(owner)
    reset_rows = []
    for state in state_rows():
        reset_rows.append({**state, 'event': 'native_reset_feedback', 'append_physics_tick': 0,
            'source_physics_tick': [0] * 4, 'actual_age_steps': [0] * 4, 'actual_age_ms': [0.] * 4})
    latest = deepcopy(reset_rows)
    journal, action_rows, physical_rows = [], [], []
    for policy in range(1, policies + 1):
        start = (policy - 1) * 4
        actor_sensor = []
        for lane, row in enumerate(latest):
            actor_sensor.append({**deepcopy(row), 'feedback18': delay.feedback[lane].tolist(),
                'consumed_at_physics_tick': start, 'actor_observation_columns_affected': [4, 22],
                'pd_feedback_scope': 'fresh_physics_feedback'})
        policy_input = torch.zeros(12, 35)
        policy_input[:, 4:22] = build_sensor18(delay.feedback[:, 12:15], delay.feedback[:, 15:18],
                                            delay.feedback[:, :6], delay.feedback[:, 6:12], torch.zeros(6))
        action_rows.append({'policy_tick': policy, 'policy_input': policy_input.tolist(), 'actor_sensor_delay': actor_sensor})
        delay.begin_policy(4185)
        for lane, row in enumerate(state_rows()):
            journal.append({**row, 'event': 'policy_lag_sample', 'lane': lane, 'case': NAMES[lane],
                'policy_tick': policy, 'physics_tick': start, 'first_cohort': True, 'native_call_once': True, 'actor_update': 4185})
        for tick in range(start + 1, start + 5):
            targets = torch.full((12, 6), float(tick) + (.25 if arm == 'stochastic' else 0.))
            legs, wheels = delay.targets(targets[:, :4], targets[:, 4:])
            target_rows = []
            for lane, row in enumerate(state_rows()):
                ages = [min(row['requested_lag_steps'][c], tick - 1) for c in (4, 5)]
                target_rows.append({**row, 'event': 'target_applied', 'lane': lane, 'case': NAMES[lane],
                    'policy_tick': policy, 'physics_tick': tick - 1, 'first_cohort': True, 'native_call_once': True,
                    'append_physics_tick': tick - 1, 'applied_physics_tick': tick, 'history_samples': tick,
                    'actual_age_steps': ages, 'actual_age_ms': [v * 5. for v in ages],
                    'source_physics_tick': [tick - 1 - v for v in ages],
                    'age_scope': 'native stored lag clipped by actual append history; first append primes cleared history',
                    'input_targets6': targets[lane].tolist(), 'delayed_targets6': legs[lane].tolist() + wheels[lane].tolist()})
            journal.extend(target_rows)
            sensors = torch.full((12, 18), tick * .01 + (.25 if arm == 'stochastic' else 0.))
            before = initial if tick == 1 else previous_sensors
            for lane in range(12):
                physical_rows.append({'lane': lane, 'physics_tick': tick, 'pd_feedback_matches_fresh_physics': True,
                    'decode_q': before[lane, :6].tolist(), 'pd_q': before[lane, :6].tolist(), 'pd_dq': before[lane, 6:12].tolist(),
                    'q6': sensors[lane, :6].tolist(), 'dq6': sensors[lane, 6:12].tolist(),
                    'root_angular_velocity_body_rad_s': sensors[lane, 12:15].tolist(), 'projected_gravity_b': sensors[lane, 15:18].tolist(),
                    'pd_leg_targets': legs[lane].tolist(), 'pd_wheel_targets': wheels[lane].tolist(),
                    'signal_delay': {'target': deepcopy(target_rows[lane]),
                                    'observation_feedback_before_native_refresh': deepcopy(latest[lane]),
                                    'native_observation_refresh_follows_this_row': True}})
            delay.observe(sensors)
            sensor_rows = []
            for lane, row in enumerate(state_rows()):
                ages = [min(row['requested_lag_steps'][c], tick - 1) for c in range(4)]
                sensor_rows.append({**row, 'event': 'observation_refresh', 'lane': lane, 'case': NAMES[lane],
                    'policy_tick': policy, 'physics_tick': tick, 'first_cohort': True, 'native_call_once': True,
                    'append_physics_tick': tick, 'history_samples': tick, 'actual_age_steps': ages,
                    'actual_age_ms': [v * 5. for v in ages], 'source_physics_tick': [tick - v for v in ages],
                    'age_scope': 'native stored lag clipped by actual append history; first append primes cleared history',
                    'input_sensors18': sensors[lane].tolist(), 'feedback18': delay.feedback[lane].tolist()})
            journal.extend(sensor_rows)
            latest, previous_sensors = sensor_rows, sensors
    return {'runtime': runtime, 'config': {'signal_delay': delay.cfg}, 'cohort': cohort,
            'birth': birth, 'journal': journal, 'actions': action_rows, 'frames': physical_rows,
            'encoder': build_sensor18}


def pipeline_pair(tmp_path, monkeypatch, change=None):
    captures = {arm: pipeline_capture(arm) for arm in ('mean', 'stochastic')}
    if change:
        change(captures['stochastic'])
    paths = {arm: tmp_path / arm for arm in captures}
    for arm, path in paths.items():
        path.mkdir()
        write(path / 'birth.json', captures[arm]['birth'])
    monkeypatch.setattr(reader, 'records', lambda path: iter(deepcopy(captures[Path(path).parent.name]['journal'])))
    reports = {arm: {'signal_delay_native_method_counts': {'begin_policy': 2, 'targets': 8, 'observe': 8}} for arm in captures}
    audits = reader.validate_pipeline_journals(paths, reports, {a: c['runtime'] for a, c in captures.items()},
        {a: c['config'] for a, c in captures.items()}, {a: c['cohort'] for a, c in captures.items()}, policy_steps=2)
    return captures, audits


def test_pipeline_native_history_pairs_metadata_but_not_arm_trajectory(tmp_path, monkeypatch):
    captures, audits = pipeline_pair(tmp_path, monkeypatch)
    assert captures['mean']['journal'][12]['delayed_targets6'] != captures['stochastic']['journal'][12]['delayed_targets6']
    for arm, audit in audits.items():
        for row in captures[arm]['actions']:
            audit.check_action(row, captures[arm]['encoder'])
        for row in captures[arm]['frames']:
            audit.check_physical(row)
        summary = audit.summary()
        assert summary['native_lane_event_counts'] == {'policy_lag_sample': 24, 'target_applied': 96, 'observation_refresh': 96}
        assert summary['effective_history_age_min_ms_control_group_order'] == [0.] * 6
        assert summary['raw_jitter_draw_coverage'].startswith('not_covered')
        assert 'no claim of complete 80ms' in summary['coverage_scope']
        first_target = captures[arm]['journal'][12]
        assert first_target['actual_age_steps'] == [0, 0] and first_target['source_physics_tick'] == [0, 0]


@pytest.mark.parametrize('field,value,match', [
    ('source_physics_tick', [1, 1], 'effective age/source/history'),
    ('actual_age_steps', [1, 1], 'effective age/source/history'),
    ('history_samples', 2, 'effective age/source/history'),
    ('delayed_targets6', [999.] * 6, 'grouped history output'),
    ('profile', 2, 'sample/profile'),
    ('native_call_once', False, 'event/call clock'),
    ('raw_jitter_ms', [0.] * 6, 'raw-jitter semantics'),
])
def test_pipeline_refuses_false_delay_provenance(tmp_path, monkeypatch, field, value, match):
    with pytest.raises(ValueError, match=match):
        pipeline_pair(tmp_path, monkeypatch, lambda c: c['journal'][12].update({field: value}))


def test_pipeline_actor_and_pd_sources_cannot_be_relabelled(tmp_path, monkeypatch):
    captures, audits = pipeline_pair(tmp_path, monkeypatch)
    audit, data = audits['mean'], captures['mean']
    action = deepcopy(data['actions'][1])
    action['actor_sensor_delay'][0]['source_physics_tick'][0] += 1
    with pytest.raises(ValueError, match='latest native sensor feedback'):
        audit.check_action(action, data['encoder'])
    action = deepcopy(data['actions'][0])
    action['policy_input'][0][4] += .5
    with pytest.raises(ValueError, match='sensor18 encoding'):
        audit.check_action(action, data['encoder'])
    frame = deepcopy(data['frames'][0])
    frame['signal_delay']['observation_feedback_before_native_refresh']['event'] = 'observation_refresh'
    with pytest.raises(ValueError, match='previous sensor provenance'):
        audit.check_physical(frame)
    audit.check_physical(data['frames'][0])
    frame = deepcopy(data['frames'][12])
    frame['pd_q'] = [999.] * 6
    with pytest.raises(ValueError, match='previous refreshed physical'):
        audit.check_physical(frame)


def test_role_rejects_missing_delayed_identity_and_cross_role_pair(pair):
    with pytest.raises(ValueError, match='Missing role'):
        reader.role_identity({}, {'signal_delay': {'instance_present': True}}, {'signal_delay': {'enabled': True}})
    path = pair['stochastic']
    config = reader.read(path / 'clean_effective_contract.json')
    config['signal_delay'] = {'enabled': True}
    write(path / 'clean_effective_contract.json', config)
    report = reader.read(path / 'report.json')
    raw = reader.sha(path / 'clean_effective_contract.json')
    report.update(fixture_role='pipeline_age_only', task_actor_update=4185)
    report['fixture'].update(fixture_role='pipeline_age_only', effective_contract_sha256=raw)
    write(path / 'report.json', report)
    write(path / 'clean_fixture_receipt.json', report['fixture'])
    runtime = reader.read(path / 'runtime.json')
    runtime.update(fixture_role='pipeline_age_only', task_actor_update=4185)
    runtime['runtime_config_validation']['effective_contract_sha256'] = raw
    write(path / 'runtime.json', runtime)
    write(path / 'runtime_config_validation.json', runtime['runtime_config_validation'])
    with pytest.raises(ValueError, match='matching fixture role'):
        reader.compare(pair['mean'], pair['stochastic'])
