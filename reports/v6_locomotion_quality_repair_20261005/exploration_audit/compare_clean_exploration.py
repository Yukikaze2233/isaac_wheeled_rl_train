"""Verify and summarize the two fresh native inference arms; never train."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def compare(mean, stochastic):
    directories = {'mean': Path(mean), 'stochastic': Path(stochastic)}
    reports = {arm: read(path / 'report.json') for arm, path in directories.items()}
    for arm, report in reports.items():
        require(report['arm'] == arm and report['status'] == 'native_probe_completed'
                and report['execution_status'] == 'completed', f'{arm} arm did not complete')
        require(report['policy_steps'] == 1000 and report['physics_steps'] == 4000
                and report['formal_updates_consumed'] == 0
                and report['actor_tensors_unchanged'] and report['source_identity_final_verified']
                and report['sealed_inputs_final_verified'], f'{arm} identity/timing/inference invariant failed')
    a, b = reports.values()
    for key in ('source_authentication', 'input_authentication', 'sampling_seed',
                'native_policy_action_order', 'native_control_order'):
        require(a[key] == b[key], f'Paired {key} differs')
    require(a['sampling_seed'] == 93001 and a['draw_counter'] == b['draw_counter'] == 1000
            and a['sample_element_count'] == b['sample_element_count'] == 72000, 'Private sample clock differs')
    configs = [read(path / 'clean_effective_contract.json') for path in directories.values()]
    # The only allowed materialized difference is the location of each arm's
    # otherwise byte-identical independent design, not a physical parameter.
    for config in configs:
        config['design_preflight'].pop('file')
    require(configs[0] == configs[1], 'Effective physical fixtures differ')
    runtimes = {arm: read(path / 'runtime.json') for arm, path in directories.items()}
    require(runtimes['mean'] == runtimes['stochastic'], 'Fresh public births/materials differ')
    require(sha(directories['mean'] / 'clean_design_expectations.json')
            == sha(directories['stochastic'] / 'clean_design_expectations.json'), 'Independent design differs')
    action_pairs = 0
    for x, y in zip(records(directories['mean'] / 'actions.jsonl'), records(directories['stochastic'] / 'actions.jsonl'), strict=True):
        action_pairs += 1
        for key in ('policy_tick', 'phase', 'requested_command', 'visible_command',
                    'rng_before_sha256', 'rng_after_sha256', 'draw_counter', 'sample_element_count'):
            require(x[key] == y[key], f'Paired command/private stream differs at {action_pairs}: {key}')
        require(x['sent_policy_action'] == x['mu'] and y['sent_policy_action'] == y['sample'],
                'Arm did not execute its declared native action')
    require(action_pairs == 1000, 'Paired policy trace is incomplete')
    cohorts = {arm: read(path / 'first_cohort.json') for arm, path in directories.items()}
    for cohort in cohorts.values():
        require(cohort['fixed_birth_denominator'] == 12 and len(cohort['rows']) == 12,
                'Fixed first-cohort denominator changed')
    shared_ticks = [min(x['exposure_physics_ticks'], y['exposure_physics_ticks'])
                    for x, y in zip(cohorts['mean']['rows'], cohorts['stochastic']['rows'], strict=True)]
    common = {}
    torque = {}
    trace_receipts = {}
    for arm, directory in directories.items():
        grouped = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'sums': defaultdict(float)}))
        hits = defaultdict(lambda: {'count': 0, 'hit_counts': [0] * 6})
        counts = [0] * 12
        for row in records(directory / 'physical_frames.jsonl'):
            lane, tick = row['lane'], row['physics_tick']
            counts[lane] += 1
            require(tick == counts[lane] and tick <= 4000, 'Physical trace has a gap/repeated tick')
            require(row['first_cohort'] == (tick <= cohorts[arm]['rows'][lane]['exposure_physics_ticks']),
                    'Physical first-cohort ownership differs from native outcome')
            if row['first_cohort']:
                value = hits[row['case']]
                value['count'] += 1
                for axis, (requested, bound) in enumerate(zip(row['requested_motor_effort'], row['current_motor_bounds'], strict=True)):
                    require(bound > 0 and math.isfinite(requested), 'Invalid native effort bounds')
                    value['hit_counts'][axis] += abs(requested) >= bound - 1e-6
            if tick > shared_ticks[lane]:
                continue
            value = grouped[row['case']][row['phase']]
            value['count'] += 1
            velocity, command = row['root_link_velocity_body_m_s'], row['command']
            samples = {'planar_speed_m_s': math.hypot(*row['root_link_velocity_world_m_s'][:2]),
                'vx_mae_m_s': abs(velocity[0] - command[0]),
                'yaw_mae_rad_s': abs(row['root_angular_velocity_body_rad_s'][2] - command[1]),
                'height_mae_m': abs(row['height_m'] - command[2]),
                'pitch_abs_deg': abs(row['pitch_deg']), 'roll_abs_deg': abs(row['roll_deg'])}
            for key, sample in samples.items():
                value['sums'][key] += sample
        require(counts == [4000] * 12, 'Whole-population 20s physical trace incomplete')
        common[arm] = {case: {phase: {'frames': value['count'], 'aggregate_exposure_s': value['count'] * .005,
            'means': {key: total / value['count'] for key, total in value['sums'].items()}}
            for phase, value in phases.items()} for case, phases in grouped.items()}
        torque[arm] = {case: {**value, 'saturation_fraction_control_order':
            [hits / value['count'] for hits in value['hit_counts']]} for case, value in hits.items()}
        trace_receipts[arm] = {name: sha(directory / name) for name in
            ('report.json', 'runtime.json', 'birth.json', 'actions.jsonl', 'physical_frames.jsonl', 'first_cohort.json')}
    return {'schema': 'clean-native-mean-stochastic-comparison-v1', 'status': 'paired_diagnostic_verified',
        'written_at_utc': datetime.now(timezone.utc).isoformat(), 'qualification': None,
        'hardware_deployment_ready': False, 'formal_updates_consumed': 0,
        'source_revision': a['source_revision'], 'PT_SHA': a['input_authentication']['files']['checkpoint']['sha256'],
        'private_stream_paired_ticks': action_pairs, 'fresh_public_births_exactly_matched': True,
        'PhysX_internal_state_restored': False, 'common_exposure_physics_ticks_per_lane': shared_ticks,
        'cases': {arm: cohort['cases'] for arm, cohort in cohorts.items()},
        'lane_phase_metrics': {arm: cohort['lane_phase_metrics'] for arm, cohort in cohorts.items()},
        'common_exposure_metrics': common, 'first_cohort_torque_saturation': torque, 'artifact_SHA256': trace_receipts,
        'scope': 'One N4-per-case clean same-PT pair isolates its native action sampling contribution; '
                 'no training-noise/delay/DR or global reward attribution; completed-window means are conditional on completion.'}


def export_events(directory, arm):
    from torch.utils.tensorboard import SummaryWriter
    path = Path(directory)
    target = path / 'trace_tensorboard'
    require(not target.exists(), 'TensorBoard trace already exists')
    with SummaryWriter(str(target), max_queue=200, flush_secs=10) as writer:
        group = []
        previous_tick = None
        def flush():
            if not group or group[0]['physics_tick'] % 4:
                return
            by_case = defaultdict(list)
            for row in group:
                if row['first_cohort']:
                    by_case[row['case']].append(row)
            step = group[0]['physics_tick']
            for case, rows in by_case.items():
                prefix = f'ExplorationProbe/clean/{arm}/{case}'
                values = {'planar_speed_m_s': [math.hypot(*row['root_link_velocity_world_m_s'][:2]) for row in rows],
                    'vx_error_m_s': [abs(row['root_link_velocity_body_m_s'][0] - row['command'][0]) for row in rows],
                    'pitch_abs_deg': [abs(row['pitch_deg']) for row in rows],
                    'roll_abs_deg': [abs(row['roll_deg']) for row in rows],
                    'phase_anchor_displacement_m': [math.dist(row['root_position_local_m'][:2], row['phase_anchor_xy_m']) for row in rows]}
                writer.add_scalar(prefix + '/first_cohort_live_count', len(rows), step)
                for key, samples in values.items():
                    writer.add_scalar(prefix + '/' + key, sum(samples) / len(samples), step)
        for row in records(path / 'physical_frames.jsonl'):
            if previous_tick is not None and row['physics_tick'] != previous_tick:
                flush()
                group = []
            previous_tick = row['physics_tick']
            group.append(row)
        flush()
        writer.add_text('ExplorationProbe/clock', 'step=physical tick; 5ms/tick, not PPO update. '
                        'Curves include first-cohort frames only; phase anchors change at each named phase.', 0)
    return {'arm': arm, 'path': str(target.resolve()), 'server_started': False,
            'event_SHA256': {p.name: sha(p) for p in target.iterdir() if p.is_file()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mean', type=Path, required=True)
    parser.add_argument('--stochastic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-events', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'Comparison artifact already exists')
    result = compare(args.mean, args.stochastic)
    if args.export_events:
        result['tensorboard_exports'] = [export_events(args.mean, 'mean'), export_events(args.stochastic, 'stochastic')]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()), 'cases': result['cases']}))
