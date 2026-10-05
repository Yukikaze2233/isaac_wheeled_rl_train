"""Stream actual native payment rows; keep densities and discrete events separate."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from locomotion_diagnostic_suite import DEFAULT_SUITE, get_suite, suite_schedule


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(type(value) in (float, int) and math.isfinite(value), 'Nonfinite or invalid payment')
    return float(value)


def authenticated_suite(directory, report):
    """Legacy reports may omit suite metadata only for the original low suite."""
    named = 'suite_name' in report or 'diagnostic_suite' in report
    suite = get_suite(report.get('suite_name', DEFAULT_SUITE))
    expected = suite.as_dict()
    if named:
        require(report.get('suite_name') == suite.name and report.get('diagnostic_suite') == expected,
                'Incomplete or changed named suite report')
    else:
        require(suite.name == DEFAULT_SUITE, 'Only legacy low suite may omit metadata')
    fixture = report.get('fixture')
    require(isinstance(fixture, dict), 'Native fixture receipt missing')
    receipt_path = directory / 'clean_fixture_receipt.json'
    require(receipt_path.exists() and json.loads(receipt_path.read_text()) == fixture,
            'Embedded and saved fixture receipts differ')
    capture = report.get('reward_payment_capture')
    require(isinstance(capture, dict), 'Native payment capture summary missing')
    for owner, field, description in ((fixture, 'diagnostic_suite', 'fixture'),
                                      (capture, 'suite', 'capture')):
        declared = 'suite_name' in owner or field in owner
        if named or declared:
            require(owner.get('suite_name') == suite.name and owner.get(field) == expected,
                    f'Named suite {description} metadata differs')
    require(report['fixture_role'] in ('clean', 'pipeline_age_only')
            and fixture.get('fixture_role') == capture.get('fixture_role') == report['fixture_role'],
            'Native fixture/capture role differs')
    require(fixture.get('num_envs') == suite.num_envs and fixture.get('replicas_per_case') == suite.replicas
            and fixture.get('horizon_seconds') == suite.horizon_s
            and fixture.get('policy_ticks') == suite.policy_steps and fixture.get('physics_ticks') == suite.physics_steps
            and fixture.get('case_profiles') == suite.profiles, 'Fixture population or clocks differ from named suite')
    require(capture.get('native_policy_steps_captured') == capture.get('native_event_calls') == suite.policy_steps
            and capture.get('rows_written') == suite.num_envs * suite.policy_steps
            and capture.get('closed') is True and capture.get('incomplete_policy_interval') is False,
            'Native capture summary does not cover named suite')
    return suite, named, capture


def analyze(directory, writer=None):
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    require(report['status'] == 'native_probe_completed' and report['formal_updates_consumed'] == 0,
            'Only completed inference probes are accepted')
    require(report['actor_tensors_unchanged'] and report['sealed_inputs_final_verified']
            and report['source_identity_final_verified'], 'Incomplete source/actor/input authentication')
    require(report.get('reward_payment_capture_requested') is True, 'Native capture was not requested')
    suite, named, capture = authenticated_suite(directory, report)
    require(report['policy_steps'] == suite.policy_steps and report['physics_steps'] == suite.physics_steps,
            'Wrong actual completed exposure')
    birth = json.loads((directory / 'birth.json').read_text())
    cases = tuple(birth['scene_groups'])
    require(Counter(cases) == Counter({name: suite.replicas for name in suite.case_names}),
            'Expected named suite cases with N4 each')
    plans = {name: suite_schedule(name, suite.name) for name in suite.case_names}
    phases, totals = defaultdict(lambda: {'samples': 0, 'density': defaultdict(float),
        'events': defaultdict(float), 'reward': 0.}), defaultdict(float)
    density_keys = None
    max_residual = 0.
    rows = 0
    batch = []
    terminal_seen = [False] * suite.num_envs
    first_terminal_ticks = [None] * suite.num_envs
    path = directory / 'policy_reward_frames.jsonl'

    def export_batch(batch):
        if writer is None:
            return
        for case in suite.case_names:
            members = [row for row in batch if row['case'] == case and row['first_cohort']]
            tick = batch[0]['policy_tick']
            prefix = 'NativePayment/' + case + '/'
            writer.add_scalar(prefix + 'first_cohort_population', len(members), tick)
            if not members:
                continue
            for key in density_keys:
                writer.add_scalar(prefix + 'density_per_s/' + key,
                    sum(row['density_per_s'][key] for row in members) / len(members), tick)
            for key in ('success_event', 'failure_event'):
                writer.add_scalar(prefix + 'event_payment/' + key,
                    sum(row['event_payment'][key] for row in members) / len(members), tick)
            writer.add_scalar(prefix + 'reward_per_policy_step',
                sum(row['reward_total'] for row in members) / len(members), tick)
            writer.add_scalar(prefix + 'body_planar_speed_m_s', sum(
                math.hypot(*row['state']['velocity'][:2]) for row in members) / len(members), tick)
            writer.add_scalar(prefix + 'vx_command_m_s', sum(
                row['state']['commands'][0] for row in members) / len(members), tick)

    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row['schema'] == 'native-reward-payment-v1', 'Wrong row schema')
            require(row['fixture_role'] == report['fixture_role'], 'Mixed fixture role')
            require(row.get('suite_name', DEFAULT_SUITE) == suite.name
                    and (not named or 'suite_name' in row), 'Payment row suite differs')
            lane, tick = rows % suite.num_envs, rows // suite.num_envs + 1
            require(tick <= suite.policy_steps, 'Payment exceeds named suite horizon')
            require(row['lane'] == lane and row['case'] == cases[lane]
                    and row['policy_tick'] == tick, 'Missing, duplicated, or reordered payment frame')
            require(row['phase'] == plans[cases[lane]][tick - 1][0], 'Payment phase differs from named suite trajectory')
            require(row['physics_tick_start'] == 4 * (tick - 1)
                    and row['physics_tick_end'] == 4 * tick
                    and row['policy_dt_s'] == .02 and row['physics_dt_s'] == .005,
                    'Payment clocks are not native 50Hz/200Hz')
            require(type(row['first_cohort']) is bool
                    and row['after_first_terminal'] is (not row['first_cohort']), 'Invalid cohort mask')
            require(row['first_cohort'] is (not terminal_seen[lane]),
                    'First terminal frame omitted or closed cohort reopened')
            require(type(row['terminal']['done']) is bool, 'Native terminal mask missing')
            if row['terminal']['done']:
                if not terminal_seen[lane]:
                    first_terminal_ticks[lane] = tick
                terminal_seen[lane] = True
            keys = set(row['density_per_s'])
            if density_keys is None:
                density_keys = sorted(keys)
            require(keys == set(density_keys) and keys, 'Missing native density terms')
            require(set(row['event_payment']) == {'success_event', 'failure_event'}, 'Missing native events')
            density = {key: finite(value) for key, value in row['density_per_s'].items()}
            events = {key: finite(value) for key, value in row['event_payment'].items()}
            actual = finite(row['reward_total'])
            reconstruction = sum(density.values()) * .02 + sum(events.values())
            residual = abs(reconstruction - actual)
            max_residual = max(max_residual, residual)
            require(residual <= 2e-5 + abs(actual) * 2e-6, 'Native total disagrees with density/event audit')
            totals['all_recorded_rows'] += 1
            totals['all_recorded_reward'] += actual
            totals['all_recorded_density_payment'] += sum(density.values()) * .02
            totals['all_recorded_event_payment'] += sum(events.values())
            if row['first_cohort']:
                value = phases[row['case'], row['phase']]
                value['samples'] += 1
                value['reward'] += actual
                for key, item in density.items():
                    value['density'][key] += item
                for key, item in events.items():
                    value['events'][key] += item
                totals['first_cohort_rows'] += 1
                totals['first_cohort_reward'] += actual
                totals['first_cohort_density_payment'] += sum(density.values()) * .02
                totals['first_cohort_event_payment'] += sum(events.values())
            batch.append(row)
            rows += 1
            if len(batch) == suite.num_envs:
                export_batch(batch)
                batch.clear()
    require(rows == suite.num_envs * suite.policy_steps and not batch, 'Incomplete named suite native payment capture')
    require(capture.get('first_cohort_rows') == totals['first_cohort_rows'], 'Capture first-cohort row count differs')
    payments = capture.get('payment_totals')
    require(isinstance(payments, dict), 'Capture payment totals missing')
    for scope, prefix in (('all_native', 'all_recorded'), ('first_cohort', 'first_cohort')):
        require(isinstance(payments.get(scope), dict), 'Capture payment scope missing')
        for field, suffix in (('density_total', 'density_payment'), ('event_total', 'event_payment'),
                              ('reward_total', 'reward')):
            require(math.isclose(finite(payments[scope].get(field)), totals[prefix + '_' + suffix],
                                 abs_tol=5e-4, rel_tol=2e-6), 'Capture payment total differs from native rows')
    phase_rows = []
    for (case, phase), value in sorted(phases.items()):
        count = value['samples']
        phase_rows.append({'case': case, 'trajectory_phase': phase, 'first_cohort_lane_policy_samples': count,
            'lane_seconds': count * .02, 'density_mean_per_s': {
                key: value['density'][key] / count for key in density_keys},
            'density_integral_reward': {key: value['density'][key] * .02 for key in density_keys},
            'native_event_total_reward': dict(value['events']), 'native_total_reward': value['reward']})
    return {'schema': 'native-payment-analysis-v1', 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': report['source_revision'], 'fixture_role': report['fixture_role'], 'arm': report['arm'],
        'suite_name': suite.name, 'diagnostic_suite': suite.as_dict(), 'legacy_low_suite_metadata': not named,
        'formal_updates_consumed': 0, 'qualification': None, 'input_report_sha256': digest(directory / 'report.json'),
        'payment_frames_sha256': digest(path), 'birth_sha256': digest(directory / 'birth.json'),
        'analyzer_sha256': digest(__file__), 'totals': dict(totals), 'density_terms': density_keys,
        'maximum_reconciliation_residual': max_residual, 'phases': phase_rows,
        'first_cohort_exposure': {'fixed_birth_denominator': suite.num_envs, 'replicas_per_case': suite.replicas,
            'first_terminal_policy_ticks': first_terminal_ticks,
            'lane_policy_samples': [tick if tick is not None else suite.policy_steps for tick in first_terminal_ticks],
            'phase_samples_scope': 'actual surviving first cohort; first native terminal included; no complete-window claim'},
        'interpretation': 'Actual paid components on fixed trajectories, not optimizer causality or qualification.',
        'tensorboard_semantics': '50Hz surviving-first-cohort means with actual population N<=4; no after-terminal zero substitution.'}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--events', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refuse overwriting an existing analysis')
    writer = None
    if args.events is not None:
        require(not args.events.exists(), 'Refuse overwriting TensorBoard events')
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.events))
    try:
        result = analyze(args.directory, writer)
    finally:
        if writer is not None:
            writer.close()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'rows': result['totals']['all_recorded_rows'],
        'max_residual': result['maximum_reconciliation_residual']}))


if __name__ == '__main__':
    main()
