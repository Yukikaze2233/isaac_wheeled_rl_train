"""Stream actual native payment rows; keep densities and discrete events separate."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


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


def analyze(directory, writer=None):
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    require(report['status'] == 'native_probe_completed' and report['formal_updates_consumed'] == 0,
            'Only completed inference probes are accepted')
    require(report['actor_tensors_unchanged'] and report['sealed_inputs_final_verified']
            and report['source_identity_final_verified'], 'Incomplete source/actor/input authentication')
    require(report.get('reward_payment_capture_requested') is True, 'Native capture was not requested')
    require(report['policy_steps'] == 1000 and report['physics_steps'] == 4000,
            'Wrong actual completed exposure')
    birth = json.loads((directory / 'birth.json').read_text())
    cases = birth['scene_groups']
    require(len(cases) == 12 and len(set(cases)) == 3
            and all(cases.count(case) == 4 for case in cases), 'Expected three cases with N4 each')
    phases, totals = defaultdict(lambda: {'samples': 0, 'density': defaultdict(float),
        'events': defaultdict(float), 'reward': 0.}), defaultdict(float)
    density_keys = None
    max_residual = 0.
    rows = 0
    batch = []
    terminal_seen = [False] * 12
    path = directory / 'policy_reward_frames.jsonl'

    def export_batch(batch):
        if writer is None:
            return
        for case in set(cases):
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
            lane, tick = rows % 12, rows // 12 + 1
            require(row['lane'] == lane and row['case'] == cases[lane]
                    and row['policy_tick'] == tick, 'Missing, duplicated, or reordered payment frame')
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
            batch.append(row)
            rows += 1
            if len(batch) == 12:
                export_batch(batch)
                batch.clear()
    require(rows == 12000 and not batch, 'Incomplete 20s native payment capture')
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
        'formal_updates_consumed': 0, 'qualification': None, 'input_report_sha256': digest(directory / 'report.json'),
        'payment_frames_sha256': digest(path), 'birth_sha256': digest(directory / 'birth.json'),
        'analyzer_sha256': digest(__file__), 'totals': dict(totals), 'density_terms': density_keys,
        'maximum_reconciliation_residual': max_residual, 'phases': phase_rows,
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
