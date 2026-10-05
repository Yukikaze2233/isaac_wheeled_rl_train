"""Verify and summarize the two fresh native inference arms; never train."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

CASES = ('stand_305mm', 'forward_0p5', 'backward_0p5')
PARTITIONS = ('failure', 'success', 'boundary_truncation', 'timeout', 'right_censored_unfinished')


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


def validate_cohort(cohort, runtime):
    """Bind each fixed lane to its first native terminal, not later done counts."""
    require(cohort['fixed_birth_denominator'] == 12 and len(cohort['rows']) == 12
            and cohort['actual_physics_tick'] == 4000
            and cohort['actual_simulated_seconds'] == 20., 'Fixed first-cohort clock/denominator changed')
    names = runtime['scene_groups']
    require(len(names) == 12 and Counter(names) == {case: 4 for case in CASES}, 'Fixed case layout changed')
    seen = Counter()
    for lane, row in enumerate(cohort['rows']):
        case = names[lane]
        require(row['lane'] == lane and row['case'] == case and row['replica'] == seen[case]
                and row['birth_id'] == 1, 'First-cohort lane/case/replica identity differs')
        seen[case] += 1
        tick = row['exposure_physics_ticks']
        require(type(tick) is int and 0 <= tick <= 4000 and tick % 4 == 0
                and row['exposure_seconds'] == tick * .005, 'Native first-cohort exposure clock differs')
        terminal = row['terminal']
        if row['partition'] == 'right_censored_unfinished':
            require(terminal is None and tick == 4000 and row['censor_time_s'] == 20.,
                    'External censoring differs from actual final exposure')
            continue
        require(isinstance(terminal, dict) and terminal['physical_tick'] == tick and tick > 0
                and terminal['time_s'] == tick * .005 and terminal['done'] is True,
                'Native first terminal differs from cohort exposure')
        require(all(type(terminal[key]) is bool for key in ('terminated', 'success', 'timeout'))
                and all(type(flag) is bool for flag in terminal['reasons'].values()), 'Invalid native outcome flags')
        truncated = terminal['reasons'].get('boundary', False) or terminal['reasons'].get('blocked', False)
        partition = ('failure' if terminal['terminated'] else 'success' if terminal['success'] else
                     'boundary_truncation' if truncated else 'timeout' if terminal['timeout'] else None)
        require(partition == row['partition'], 'Native first-terminal partition differs')
    require(cohort['partitions'] == {part: sum(row['partition'] == part for row in cohort['rows'])
                                     for part in PARTITIONS}, 'Native cohort partition totals differ')


def saturation_summary(hits, scope):
    return {case: {**hits[case], 'scope': scope,
        'saturation_fraction_control_order': ([n / hits[case]['count'] for n in hits[case]['hit_counts']]
                                               if hits[case]['count'] else None)} for case in CASES}


def normalize_runtime_binding(directory, report, runtime, config):
    """Authenticate arm-local output bindings before canonical comparison.

    Only the design file path is removed from the configuration. Only the
    runtime validation's authenticated contract path and its derived raw SHA
    are normalized; every other actual runtime/configuration field remains.
    """
    directory = Path(directory).resolve()
    contract_path, design_path = directory / 'clean_effective_contract.json', directory / 'clean_design_expectations.json'
    audit = runtime['runtime_config_validation']
    require(audit == read(directory / 'runtime_config_validation.json'), 'Runtime validation file differs from native readback')
    require(Path(audit['contract_path']).resolve() == contract_path and contract_path.is_file(),
            'Runtime contract path is not this arm verified output')
    raw_sha, design_sha = sha(contract_path), sha(design_path)
    fixture = report['fixture']
    require(audit['status'] == 'passed' and audit['effective_contract_sha_matches'] is True
            and audit['effective_contract_sha256'] == raw_sha == fixture['effective_contract_sha256']
            and fixture == read(directory / 'clean_fixture_receipt.json'), 'Runtime/fixture contract SHA binding differs')
    require(fixture['source_contract_sha256'] == report['input_authentication']['files']['contract']['sha256'],
            'Fixture source contract differs from authenticated input')
    require(Path(config['design_preflight']['file']).resolve() == design_path
            and config['design_preflight']['sha256'] == design_sha == fixture['design_sha256'],
            'Configuration design path/SHA binding differs')
    canonical = deepcopy(config)
    canonical['design_preflight'].pop('file')
    canonical_sha = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode()).hexdigest()
    normalized = deepcopy(runtime)
    normalized['runtime_config_validation']['contract_path'] = '<verified-arm-output>/clean_effective_contract.json'
    normalized['runtime_config_validation']['effective_contract_sha256'] = canonical_sha
    return canonical, normalized, {'raw_effective_contract_sha256': raw_sha, 'canonical_effective_contract_sha256': canonical_sha,
                                  'design_sha256': design_sha, 'only_removed_config_path': 'design_preflight.file',
                                  'normalized_runtime_fields': ['runtime_config_validation.contract_path',
                                                                'runtime_config_validation.effective_contract_sha256']}


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
    runtimes = {arm: read(path / 'runtime.json') for arm, path in directories.items()}
    canonical, normalized, bindings = {}, {}, {}
    for arm, directory in directories.items():
        canonical[arm], normalized[arm], bindings[arm] = normalize_runtime_binding(directory, reports[arm], runtimes[arm],
            read(directory / 'clean_effective_contract.json'))
    require(canonical['mean'] == canonical['stochastic'], 'Effective physical fixtures differ')
    require(normalized['mean'] == normalized['stochastic'], 'Fresh public births/materials/config validation differ')
    require(sha(directories['mean'] / 'clean_design_expectations.json')
            == sha(directories['stochastic'] / 'clean_design_expectations.json'), 'Independent design differs')
    action_pairs = 0
    previous_rng_after = None
    for x, y in zip(records(directories['mean'] / 'actions.jsonl'), records(directories['stochastic'] / 'actions.jsonl'), strict=True):
        action_pairs += 1
        for key in ('policy_tick', 'phase', 'requested_command', 'visible_command',
                    'rng_before_sha256', 'rng_after_sha256', 'draw_counter', 'sample_element_count'):
            require(x[key] == y[key], f'Paired command/private stream differs at {action_pairs}: {key}')
        require(x['policy_tick'] == x['draw_counter'] == action_pairs
                and x['sample_element_count'] == action_pairs * 72, 'Native policy/private RNG trace clock differs')
        require(previous_rng_after is None or x['rng_before_sha256'] == previous_rng_after,
                'Private RNG stream restarted or skipped')
        previous_rng_after = x['rng_after_sha256']
        require(x['sent_policy_action'] == x['mu'] and y['sent_policy_action'] == y['sample'],
                'Arm did not execute its declared native action')
    require(action_pairs == 1000, 'Paired policy trace is incomplete')
    cohorts = {arm: read(path / 'first_cohort.json') for arm, path in directories.items()}
    for arm, cohort in cohorts.items():
        validate_cohort(cohort, runtimes[arm])
        require({key: value for key, value in cohort.items() if key != 'lane_phase_metrics'}
                == reports[arm]['first_cohort'], 'First-cohort file differs from final native report')
    shared_ticks = [min(x['exposure_physics_ticks'], y['exposure_physics_ticks'])
                    for x, y in zip(cohorts['mean']['rows'], cohorts['stochastic']['rows'], strict=True)]
    common = {}
    torque = {}
    common_torque = {}
    trace_receipts = {}
    for arm, directory in directories.items():
        grouped = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'sums': defaultdict(float)}))
        hits = defaultdict(lambda: {'count': 0, 'hit_counts': [0] * 6})
        shared_hits = defaultdict(lambda: {'count': 0, 'hit_counts': [0] * 6})
        counts = [0] * 12
        for row in records(directory / 'physical_frames.jsonl'):
            lane, tick = row['lane'], row['physics_tick']
            require(type(lane) is int and 0 <= lane < 12 and type(tick) is int
                    and row['case'] == cohorts[arm]['rows'][lane]['case'], 'Physical lane/case identity differs')
            counts[lane] += 1
            require(tick == counts[lane] and tick <= 4000, 'Physical trace has a gap/repeated tick')
            require(row['first_cohort'] == (tick <= cohorts[arm]['rows'][lane]['exposure_physics_ticks']),
                    'Physical first-cohort ownership differs from native outcome')
            if row['first_cohort']:
                value = hits[row['case']]
                value['count'] += 1
                require(len(row['requested_motor_effort']) == len(row['current_motor_bounds']) == 6,
                        'Native six-axis effort record differs')
                flags = []
                for axis, (requested, bound) in enumerate(zip(row['requested_motor_effort'], row['current_motor_bounds'], strict=True)):
                    # Native wheel torque-speed limits may legitimately reach 0.
                    require(bound >= 0 and math.isfinite(bound) and math.isfinite(requested), 'Invalid native effort bounds')
                    flags.append(abs(requested) >= bound - 1e-6)
                    value['hit_counts'][axis] += flags[-1]
            if tick > shared_ticks[lane]:
                continue
            shared_value = shared_hits[row['case']]
            shared_value['count'] += 1
            for axis, hit in enumerate(flags):
                shared_value['hit_counts'][axis] += hit
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
        torque[arm] = saturation_summary(hits, 'each arm own first-cohort exposure; survivor durations may differ')
        common_torque[arm] = saturation_summary(shared_hits, 'paired per-lane min(first-native-terminal) exposure')
        trace_receipts[arm] = {name: sha(directory / name) for name in
            ('report.json', 'runtime.json', 'birth.json', 'actions.jsonl', 'physical_frames.jsonl', 'first_cohort.json',
             'runtime_config_validation.json', 'clean_effective_contract.json', 'clean_fixture_receipt.json',
             'clean_design_expectations.json')}
    return {'schema': 'clean-native-mean-stochastic-comparison-v1', 'status': 'paired_diagnostic_verified',
        'written_at_utc': datetime.now(timezone.utc).isoformat(), 'qualification': None,
        'hardware_deployment_ready': False, 'formal_updates_consumed': 0,
        'source_revision': a['source_revision'], 'PT_SHA': a['input_authentication']['files']['checkpoint']['sha256'],
        'private_stream_paired_ticks': action_pairs, 'fresh_public_births_exactly_matched': True,
        'authenticated_arm_local_bindings': bindings,
        'PhysX_internal_state_restored': False, 'common_exposure_physics_ticks_per_lane': shared_ticks,
        'cases': {arm: cohort['cases'] for arm, cohort in cohorts.items()},
        'lane_phase_metrics': {arm: cohort['lane_phase_metrics'] for arm, cohort in cohorts.items()},
        'common_exposure_metrics': common, 'first_cohort_torque_saturation': torque,
        'common_exposure_torque_saturation': common_torque,
        'torque_saturation_definition': 'abs(native pre-cap requested effort) >= instantaneous bound - 1e-6; '
                                       'six control-order axes; includes zero drive bounds',
        'artifact_SHA256': trace_receipts,
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
            if not group:
                return
            require(len(group) == 12 and [row['lane'] for row in group] == list(range(12))
                    and Counter(row['case'] for row in group) == {case: 4 for case in CASES},
                    'TensorBoard export requires one complete ordered N12 physical tick')
            if group[0]['physics_tick'] % 4:
                return
            by_case = defaultdict(list)
            for row in group:
                if row['first_cohort']:
                    by_case[row['case']].append(row)
            step = group[0]['physics_tick']
            for case in CASES:
                rows = by_case[case]
                prefix = f'ExplorationProbe/clean/{arm}/{case}'
                writer.add_scalar(prefix + '/first_cohort_live_count', len(rows), step)
                if not rows:
                    # Missing observations have no value, not a zero-speed win.
                    continue
                values = {'planar_speed_m_s': [math.hypot(*row['root_link_velocity_world_m_s'][:2]) for row in rows],
                    'vx_error_m_s': [abs(row['root_link_velocity_body_m_s'][0] - row['command'][0]) for row in rows],
                    'pitch_abs_deg': [abs(row['pitch_deg']) for row in rows],
                    'roll_abs_deg': [abs(row['roll_deg']) for row in rows],
                    'phase_anchor_displacement_m': [math.dist(row['root_position_local_m'][:2], row['phase_anchor_xy_m']) for row in rows]}
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
                        'Self-survivor instantaneous 50Hz sampled curves: every fourth physical frame only, '
                        'each arm own first cohort. They are neither 200Hz time averages nor paired common-exposure '
                        'means. A case with no live lanes has live_count=0 and no value samples. '
                        'Phase anchors change at each named phase.', 0)
    return {'arm': arm, 'path': str(target.resolve()), 'server_started': False,
            'curve_scope': 'self-survivor instantaneous 50Hz sampled; not paired common-exposure means',
            'sampling_stride_physics_ticks': 4, 'sampling_hz': 50,
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
