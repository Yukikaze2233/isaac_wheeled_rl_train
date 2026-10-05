"""CPU stream contracts; one generated high-suite stream is reused read-only."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

SPEC = importlib.util.spec_from_file_location('analyze_reward_payment',
    Path(__file__).with_name('analyze_reward_payment.py'))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
from locomotion_diagnostic_suite import DEFAULT_SUITE, HIGH_SPEED_SUITE, get_suite, suite_schedule


def fixture(tmp_path, mutate=None, *, suite_name=DEFAULT_SUITE, named=False):
    suite = get_suite(suite_name)
    cases = [case for case in suite.case_names for _ in range(suite.replicas)]
    plans = {case: suite_schedule(case, suite_name) for case in suite.case_names}
    receipt = dict(fixture_role='pipeline_age_only', num_envs=suite.num_envs,
        replicas_per_case=suite.replicas, horizon_seconds=suite.horizon_s,
        policy_ticks=suite.policy_steps, physics_ticks=suite.physics_steps, case_profiles=suite.profiles)
    capture = dict(fixture_role='pipeline_age_only', native_policy_steps_captured=suite.policy_steps,
        native_event_calls=suite.policy_steps, rows_written=suite.num_envs * suite.policy_steps,
        closed=True, incomplete_policy_interval=False)
    report = dict(status='native_probe_completed', formal_updates_consumed=0,
        actor_tensors_unchanged=True, sealed_inputs_final_verified=True,
        source_identity_final_verified=True, reward_payment_capture_requested=True,
        policy_steps=suite.policy_steps, physics_steps=suite.physics_steps, fixture_role='pipeline_age_only',
        source_revision='a' * 40, arm='mean', fixture=receipt, reward_payment_capture=capture)
    if named:
        report.update(suite_name=suite_name, diagnostic_suite=suite.as_dict())
        receipt.update(suite_name=suite_name, diagnostic_suite=suite.as_dict())
        capture.update(suite_name=suite_name, suite=suite.as_dict())
    totals = {scope: defaultdict(float) for scope in ('all_native', 'first_cohort')}
    first_cohort_rows = 0
    (tmp_path / 'birth.json').write_text(json.dumps(dict(scene_groups=cases)))
    with (tmp_path / 'policy_reward_frames.jsonl').open('w') as stream:
        for tick in range(1, suite.policy_steps + 1):
            for lane, case in enumerate(cases):
                # Lane0 fails at tick3 in settle; later native failure payment persists,
                # while first-cohort exposure must never reopen or omit tick3.
                active = lane != 0 or tick <= 3
                failed = lane == 0 and tick >= 3
                events = dict(success_event=0., failure_event=-200. if failed else 0.)
                row = dict(schema='native-reward-payment-v1', fixture_role='pipeline_age_only',
                    lane=lane, case=case, policy_tick=tick, physics_tick_start=(tick - 1) * 4,
                    physics_tick_end=tick * 4, policy_dt_s=.02, physics_dt_s=.005,
                    first_cohort=active, after_first_terminal=not active, phase=plans[case][tick - 1][0],
                    density_per_s=dict(tracking=3., stationary=-.5),
                    event_payment=events, reward_total=.05 + sum(events.values()),
                    terminal=dict(done=failed or tick == suite.policy_steps),
                    state=dict(velocity=[.1, 0., 0.], commands=[0., 0., .305]))
                if named:
                    row['suite_name'] = suite_name
                if mutate is not None:
                    mutate(row)
                stream.write(json.dumps(row) + '\n')
                for scope in ('all_native', 'first_cohort') if row['first_cohort'] else ('all_native',):
                    totals[scope]['density_total'] += sum(row['density_per_s'].values()) * .02
                    totals[scope]['event_total'] += sum(row['event_payment'].values())
                    totals[scope]['reward_total'] += row['reward_total']
                first_cohort_rows += row['first_cohort']
    capture.update(first_cohort_rows=first_cohort_rows,
                   payment_totals={scope: dict(total) for scope, total in totals.items()})
    (tmp_path / 'report.json').write_text(json.dumps(report))
    (tmp_path / 'clean_fixture_receipt.json').write_text(json.dumps(receipt))
    return tmp_path


def test_discrete_failure_not_scaled_by_dt_and_first_terminal_included(tmp_path):
    result = audit.analyze(fixture(tmp_path))
    assert result['totals']['all_recorded_rows'] == 12000
    assert result['totals']['first_cohort_rows'] == 11003
    assert result['maximum_reconciliation_residual'] < 1e-9
    assert result['legacy_low_suite_metadata'] and result['suite_name'] == DEFAULT_SUITE
    stand = [row for row in result['phases'] if row['case'] == 'stand_305mm']
    assert sum(row['first_cohort_lane_policy_samples'] for row in stand) == 3003
    settle = next(row for row in stand if row['trajectory_phase'] == 'settle')
    assert settle['native_event_total_reward']['failure_event'] == -200.
    assert sum(row['native_event_total_reward']['failure_event'] for row in stand) == -200.
    assert all(row['native_event_total_reward']['failure_event'] == 0. for row in stand if row is not settle)
    assert all(row['density_mean_per_s'] == dict(stationary=-.5, tracking=3.) for row in stand)
    assert result['first_cohort_exposure']['first_terminal_policy_ticks'] == [3] + [1000] * 11


def test_event_accidentally_multiplied_by_dt_is_rejected(tmp_path):
    def mutate(row):
        if row['lane'] == 0 and row['terminal']['done']:
            row['reward_total'] = .05 - 200. * .02
    with pytest.raises(ValueError, match='total disagrees'):
        audit.analyze(fixture(tmp_path, mutate))


def test_first_terminal_omission_is_rejected(tmp_path):
    def mutate(row):
        if row['lane'] == 0 and row['policy_tick'] == 3:
            row.update(first_cohort=False, after_first_terminal=True)
    with pytest.raises(ValueError, match='First terminal'):
        audit.analyze(fixture(tmp_path, mutate))


@pytest.mark.parametrize('fault', ['missing_row', 'clock', 'density', 'nan', 'source'])
def test_missing_or_untrusted_payment_data_rejected(tmp_path, fault):
    def mutate(row):
        if row['lane'] == 0 and row['policy_tick'] == 50:
            if fault == 'clock':
                row['physics_tick_end'] += 1
            elif fault == 'density':
                row['density_per_s'].pop('stationary')
            elif fault == 'nan':
                row['density_per_s']['stationary'] = float('nan')
    directory = fixture(tmp_path, mutate)
    if fault == 'missing_row':
        path = directory / 'policy_reward_frames.jsonl'
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines[:500] + lines[501:]) + '\n')
    elif fault == 'source':
        path = directory / 'report.json'
        report = json.loads(path.read_text()); report['source_identity_final_verified'] = False
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit.analyze(directory)


def test_export_stops_metric_population_after_native_done(tmp_path):
    class Writer:
        def __init__(self):
            self.rows = []
        def add_scalar(self, tag, value, tick):
            self.rows.append((tag, value, tick))
    writer = Writer()
    audit.analyze(fixture(tmp_path), writer)
    values = {tick: value for tag, value, tick in writer.rows
              if tag == 'NativePayment/stand_305mm/first_cohort_population'}
    assert values[3] == 4 and values[4] == 3
    paid = {tick: value for tag, value, tick in writer.rows
            if tag == 'NativePayment/stand_305mm/event_payment/failure_event'}
    assert paid[3] == -50 and paid[4] == 0


@pytest.fixture(scope='module')
def high_stream(tmp_path_factory):
    return fixture(tmp_path_factory.mktemp('high_native_payment'), suite_name=HIGH_SPEED_SUITE, named=True)


def test_complete_high_suite_stream_counts_phases_and_first_terminal(high_stream):
    result = audit.analyze(high_stream)
    suite = get_suite(HIGH_SPEED_SUITE)
    assert suite.num_envs == 64 and suite.policy_steps == 1600
    assert result['suite_name'] == HIGH_SPEED_SUITE and result['diagnostic_suite'] == suite.as_dict()
    assert not result['legacy_low_suite_metadata']
    assert result['totals']['all_recorded_rows'] == 102400
    assert result['totals']['first_cohort_rows'] == 100803
    assert result['first_cohort_exposure']['first_terminal_policy_ticks'] == [3] + [1600] * 63
    assert result['first_cohort_exposure']['lane_policy_samples'] == [3] + [1600] * 63
    assert {row['case'] for row in result['phases']} == set(suite.case_names)
    assert sum(row['native_event_total_reward']['failure_event'] for row in result['phases']) == -200.
    assert sum(row['first_cohort_lane_policy_samples'] for row in result['phases']) == 100803
    assert result['maximum_reconciliation_residual'] < 1e-9 and result['qualification'] is None
    forward = [row for row in result['phases'] if row['case'] == 'forward_5_430mm']
    assert next(row for row in forward if row['trajectory_phase'] == 'steady')['first_cohort_lane_policy_samples'] == 800
    assert next(row for row in forward if row['trajectory_phase'] == 'stop')['first_cohort_lane_policy_samples'] == 2000
    assert 'height_settle' in {row['trajectory_phase'] for row in forward}
    cancel = [row for row in result['phases'] if row['case'] == 'cancel_forward_5_305mm']
    assert next(row for row in cancel if row['trajectory_phase'] == 'steady')['first_cohort_lane_policy_samples'] == 200
    assert 'cancelled_braking' in {row['trajectory_phase'] for row in cancel}


def copy_metadata(source, directory):
    for name in ('report.json', 'birth.json', 'clean_fixture_receipt.json'):
        shutil.copyfile(source / name, directory / name)
    (directory / 'policy_reward_frames.jsonl').symlink_to(source / 'policy_reward_frames.jsonl')
    return directory


@pytest.mark.parametrize('fault', ('unknown_suite', 'no_report_metadata', 'changed_catalog', 'fixture',
                                  'capture_suite', 'capture_calls', 'capture_population', 'birth_population',
                                  'actual_clock', 'capture_first_cohort', 'capture_total'))
def test_high_metadata_population_clocks_and_payment_receipt_rejected(high_stream, tmp_path, fault):
    directory = copy_metadata(high_stream, tmp_path)
    report = json.loads((directory / 'report.json').read_text())
    if fault == 'unknown_suite':
        report['suite_name'] = 'custom_unbounded'
    elif fault == 'no_report_metadata':
        report.pop('suite_name'); report.pop('diagnostic_suite')
    elif fault == 'changed_catalog':
        report['diagnostic_suite']['policy_steps'] = 1601
    elif fault == 'fixture':
        report['fixture']['num_envs'] = 65
    elif fault == 'capture_suite':
        report['reward_payment_capture']['suite_name'] = DEFAULT_SUITE
    elif fault == 'capture_calls':
        report['reward_payment_capture']['native_event_calls'] = 1599
    elif fault == 'capture_population':
        report['reward_payment_capture']['rows_written'] = 102399
    elif fault == 'birth_population':
        path = directory / 'birth.json'
        birth = json.loads(path.read_text()); birth['scene_groups'].pop()
        path.write_text(json.dumps(birth))
    elif fault == 'actual_clock':
        report['physics_steps'] = 6399
    elif fault == 'capture_first_cohort':
        report['reward_payment_capture']['first_cohort_rows'] += 1
    else:
        report['reward_payment_capture']['payment_totals']['all_native']['reward_total'] += 1.
    (directory / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit.analyze(directory)


@pytest.mark.parametrize('fault', ('row_suite', 'row_without_suite', 'phase', 'clock', 'lane', 'case'))
def test_high_first_row_cannot_spoof_suite_or_native_trajectory(high_stream, tmp_path, fault):
    directory = copy_metadata(high_stream, tmp_path)
    path = directory / 'policy_reward_frames.jsonl'
    with path.open() as stream:
        row = json.loads(next(stream))
    path.unlink()  # Replace this test's symlink; the shared 102400 rows remain untouched.
    if fault == 'row_suite':
        row['suite_name'] = DEFAULT_SUITE
    elif fault == 'row_without_suite':
        row.pop('suite_name')
    elif fault == 'phase':
        row['phase'] = 'made_up_steady'
    elif fault == 'clock':
        row['physics_tick_end'] = 5
    elif fault == 'lane':
        row['lane'] = 1
    else:
        row['case'] = 'arbitrary_5m_s'
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError):
        audit.analyze(directory)
