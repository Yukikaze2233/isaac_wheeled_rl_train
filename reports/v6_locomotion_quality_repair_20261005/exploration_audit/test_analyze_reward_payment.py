import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('analyze_reward_payment',
    Path(__file__).with_name('analyze_reward_payment.py'))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture(tmp_path, mutate=None):
    cases = [case for case in ('stand', 'forward', 'backward') for _ in range(4)]
    report = dict(status='native_probe_completed', formal_updates_consumed=0,
        actor_tensors_unchanged=True, sealed_inputs_final_verified=True,
        source_identity_final_verified=True, reward_payment_capture_requested=True,
        policy_steps=1000, physics_steps=4000, fixture_role='pipeline_age_only',
        source_revision='a' * 40, arm='mean')
    (tmp_path / 'report.json').write_text(json.dumps(report))
    (tmp_path / 'birth.json').write_text(json.dumps(dict(scene_groups=cases)))
    with (tmp_path / 'policy_reward_frames.jsonl').open('w') as stream:
        for tick in range(1, 1001):
            for lane, case in enumerate(cases):
                # Lane0 fails at tick3; later native failure payment persists,
                # while first-cohort exposure must never reopen or omit tick3.
                active = lane != 0 or tick <= 3
                failed = lane == 0 and tick >= 3
                events = dict(success_event=0., failure_event=-200. if failed else 0.)
                row = dict(schema='native-reward-payment-v1', fixture_role='pipeline_age_only',
                    lane=lane, case=case, policy_tick=tick, physics_tick_start=(tick - 1) * 4,
                    physics_tick_end=tick * 4, policy_dt_s=.02, physics_dt_s=.005,
                    first_cohort=active, after_first_terminal=not active, phase='stand',
                    density_per_s=dict(tracking=3., stationary=-.5),
                    event_payment=events, reward_total=.05 + sum(events.values()),
                    terminal=dict(done=failed), state=dict(velocity=[.1, 0., 0.], commands=[0., 0., .305]))
                if mutate is not None:
                    mutate(row)
                stream.write(json.dumps(row) + '\n')
    return tmp_path


def test_discrete_failure_not_scaled_by_dt_and_first_terminal_included(tmp_path):
    result = audit.analyze(fixture(tmp_path))
    assert result['totals']['all_recorded_rows'] == 12000
    assert result['totals']['first_cohort_rows'] == 11003
    assert result['maximum_reconciliation_residual'] < 1e-9
    stand = next(row for row in result['phases'] if row['case'] == 'stand')
    assert stand['first_cohort_lane_policy_samples'] == 3003
    assert stand['native_event_total_reward']['failure_event'] == -200.
    assert stand['density_mean_per_s'] == dict(stationary=-.5, tracking=3.)


def test_event_accidentally_multiplied_by_dt_is_rejected(tmp_path):
    def mutate(row):
        if row['terminal']['done']:
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
              if tag == 'NativePayment/stand/first_cohort_population'}
    assert values[3] == 4 and values[4] == 3
    paid = {tick: value for tag, value, tick in writer.rows
            if tag == 'NativePayment/stand/event_payment/failure_event'}
    assert paid[3] == -50 and paid[4] == 0
