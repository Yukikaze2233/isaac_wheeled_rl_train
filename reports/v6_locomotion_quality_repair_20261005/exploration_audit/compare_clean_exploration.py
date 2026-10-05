"""Verify and summarize the two fresh native inference arms; never train."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

CASES = ('stand_305mm', 'forward_0p5', 'backward_0p5')
PARTITIONS = ('failure', 'success', 'boundary_truncation', 'timeout', 'right_censored_unfinished')
FIXTURE_ROLES = ('clean', 'pipeline_age_only')
DELAY_STATE_KEYS = ('enabled', 'profile', 'base_ms', 'bounds_ms', 'requested_lag_steps',
                    'requested_lag_ms', 'strength', 'raw_jitter_ms', 'raw_jitter_coverage',
                    'jitter_metric_scope', 'scaled_enabled_base_ms',
                    'effective_quantized_lag_delta_from_scaled_base_ms')
DELAY_GROUPS = ((0, 6), (6, 12), (12, 15), (15, 18))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def finite_vector(value, size, message):
    require(isinstance(value, list) and len(value) == size
            and all(type(x) in (int, float) and math.isfinite(x) for x in value), message)


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


def role_identity(report, runtime, config):
    """Old clean outputs remain readable; delayed roles require explicit identity."""
    role = report.get('fixture_role', 'clean')
    require(role in FIXTURE_ROLES, 'Unsupported diagnostic fixture role')
    if 'fixture_role' not in report:
        require(config.get('signal_delay', {}).get('enabled', False) is False
                and not runtime.get('signal_delay'), 'Missing role cannot identify a delayed fixture')
    else:
        require(report['fixture']['fixture_role'] == runtime['fixture_role'] == role,
                'Fixture role differs between invocation/receipt/native runtime')
        clock = 4185 if role == 'pipeline_age_only' else 0
        require(report['task_actor_update'] == runtime['task_actor_update'] == clock,
                'Diagnostic task-actor clock differs from its role')
    require(config.get('signal_delay', {}).get('enabled', False) is (role == 'pipeline_age_only'),
            'Native signal-delay enable differs from fixture role')
    if role == 'clean' and 'fixture_role' in report:
        require(runtime['signal_delay']['instance_present'] is False
                and runtime['signal_delay']['cfg'] == config['signal_delay'], 'Clean role has native signal delay')
    return role


class PipelineJournalAudit:
    """Authenticate recorded native ages/history, without drawing jitter again.

    The bounded list histories check the original grouped gather's provenance;
    they do not execute a simulator, mutate native buffers, or infer raw jitter.
    Link hashes bind those journal outputs to the actor and physical recorder.
    """
    def __init__(self, runtime, config, cohort, birth):
        import torch
        state = runtime['signal_delay']
        cfg = config['signal_delay']
        require(state['instance_present'] is True and state['class_name'] == 'PhysicalSignalDelay'
                and state['cfg'] == cfg and state['task_actor_update'] == 4185
                and state['strength'] == state['policy_tick'] == state['physics_tick']
                == state['physics_steps_observed'] == 0 and state['quantum_ms'] == 5.,
                'Pipeline native birth/config/clock differs')
        require(state['raw_jitter_draw_coverage'] == 'not_covered_native_local_not_persisted',
                'Raw jitter must not be represented as an observed draw')
        self.names, self.exposure = runtime['scene_groups'], [r['exposure_physics_ticks'] for r in cohort['rows']]
        require(birth['fixture_role'] == 'pipeline_age_only' and birth['task_actor_update'] == 4185
                and birth['scene_groups'] == self.names, 'Pipeline birth role/layout differs')
        finite_vector(birth['v5_nominal6'], 6, 'Native nominal motor vector differs')
        self.nominal = birth['v5_nominal6']
        self.state, self.cfg, self.minimum, self.maximum = state, cfg, [], []
        for field in ('enabled', 'profile', 'base_ms', 'bounds_ms', 'lags_physics_steps', 'ages_ms', 'feedback_sensor18'):
            require(len(state[field]) == 12, 'Pipeline native birth population differs')
        for lane in range(12):
            require(type(state['enabled'][lane]) is bool and type(state['profile'][lane]) is int
                    and 0 <= state['profile'][lane] < len(cfg['profiles']), 'Native delay enabled/profile differs')
            p = cfg['profiles'][state['profile'][lane]]
            bounds = [p['observation_ms']] * 4 + [p['action_ms']] * 2
            require(state['bounds_ms'][lane] == bounds, 'Native delay profile bounds differ')
            finite_vector(state['base_ms'][lane], 6, 'Native base delays invalid')
            finite_vector(state['feedback_sensor18'][lane], 18, 'Native reset sensor feedback invalid')
            require(all(lo <= v <= hi for v, (lo, hi) in zip(state['base_ms'][lane], bounds, strict=True))
                    and state['lags_physics_steps'][lane] == [0] * 6
                    and state['ages_ms'][lane] == [0.] * 6, 'Native birth lag/base differs')
            base, ranges = torch.tensor(state['base_ms'][lane]), torch.tensor(bounds)
            enabled = int(state['enabled'][lane])
            self.minimum.append((torch.maximum(base - cfg['jitter_ms'], ranges[:, 0]) / 5.).round().long().mul(enabled).tolist())
            self.maximum.append((torch.minimum(base + cfg['jitter_ms'], ranges[:, 1]) / 5.).round().long().mul(enabled).tolist())
        self.history = {kind: [deque(maxlen=17 if kind == 'sensor' else 13) for _ in range(12)]
                        for kind in ('sensor', 'target')}
        self.reset_rows = [self.reset_row(lane) for lane in range(12)]
        self.latest_sensor = deepcopy(self.reset_rows)
        self.current_lags = [[0] * 6 for _ in range(12)]
        self.actor_links, self.target_links, self.sensor_links, self.sensor_inputs = {}, {}, {}, {}
        self.previous_physical = {}
        self.counts, self.policy_records = Counter(), 0
        self.lag_min, self.lag_max = [math.inf] * 6, [0.] * 6
        self.age_min, self.age_max = [math.inf] * 6, [0.] * 6
        self.actor_encoding_max_error = 0.

    def reset_row(self, lane):
        state = self.state
        return {'event': 'native_reset_feedback', 'append_physics_tick': 0,
            'source_physics_tick': [0] * 4, 'actual_age_steps': [0] * 4, 'actual_age_ms': [0.] * 4,
            'raw_jitter_ms': None, 'raw_jitter_coverage': 'not_covered_native_draw_is_not_retained',
            'enabled': state['enabled'][lane], 'profile': state['profile'][lane],
            'base_ms': state['base_ms'][lane], 'bounds_ms': state['bounds_ms'][lane],
            'requested_lag_steps': [0] * 6, 'requested_lag_ms': [0.] * 6, 'strength': 0.,
            'jitter_metric_scope': 'stored quantized lag minus scaled base; includes clipping/rounding/enabled masking',
            'scaled_enabled_base_ms': [0.] * 6,
            'effective_quantized_lag_delta_from_scaled_base_ms': [0.] * 6}

    def consume(self, row, event, lane, policy_tick, physics_tick):
        require(row['event'] == event and row['lane'] == lane and row['case'] == self.names[lane]
                and row['policy_tick'] == policy_tick and row['physics_tick'] == physics_tick
                and row['native_call_once'] is True, 'Native delay journal event/call clock differs')
        actual_tick = physics_tick + (event == 'target_applied')
        live = physics_tick < self.exposure[lane] if event == 'policy_lag_sample' else actual_tick <= self.exposure[lane]
        require(row['first_cohort'] is live, 'Native delay journal first-terminal ownership differs')
        require(row['enabled'] is self.state['enabled'][lane] and row['profile'] == self.state['profile'][lane]
                and row['base_ms'] == self.state['base_ms'][lane] and row['bounds_ms'] == self.state['bounds_ms'][lane]
                and row['strength'] == 1. and row['raw_jitter_ms'] is None
                and row['raw_jitter_coverage'] == 'not_covered_native_draw_is_not_retained'
                and row['jitter_metric_scope'] == self.reset_rows[lane]['jitter_metric_scope'],
                'Native delay sample/profile/raw-jitter semantics differs')
        lags = row['requested_lag_steps']
        require(len(lags) == 6 and all(type(v) is int and lo <= v <= hi for v, lo, hi in
                zip(lags, self.minimum[lane], self.maximum[lane], strict=True)), 'Native lag outside jitter/quantization range')
        base = row['base_ms'] if row['enabled'] else [0.] * 6
        require(row['requested_lag_ms'] == [v * 5. for v in lags]
                and row['scaled_enabled_base_ms'] == base
                and row['effective_quantized_lag_delta_from_scaled_base_ms'] == [v * 5. - b for v, b in zip(lags, base, strict=True)],
                'Native quantized lag/delta differs; delta is not raw jitter')
        self.counts[event] += 1
        if event == 'policy_lag_sample':
            require(row['actor_update'] == 4185, 'Native begin-policy actor clock differs')
            previous = self.latest_sensor[lane]
            feedback = previous.get('feedback18', self.state['feedback_sensor18'][lane])
            self.actor_links[policy_tick, lane] = fingerprint(previous), feedback
            self.current_lags[lane] = lags
            for column, value in enumerate(row['requested_lag_ms']):
                self.lag_min[column] = min(self.lag_min[column], value)
                self.lag_max[column] = max(self.lag_max[column], value)
            return
        require(lags == self.current_lags[lane], 'Native physical lag changed between policy calls')
        sensor = event == 'observation_refresh'
        append_tick = physics_tick
        tick = actual_tick
        columns = range(4) if sensor else (4, 5)
        valid = [min(lags[c], tick - 1) for c in columns]
        require(row['append_physics_tick'] == append_tick and row['history_samples'] == tick
                and row['actual_age_steps'] == valid and row['actual_age_ms'] == [v * 5. for v in valid]
                and row['source_physics_tick'] == [append_tick - v for v in valid]
                and row['age_scope'] == 'native stored lag clipped by actual append history; first append primes cleared history',
                'Native effective age/source/history clock differs')
        if not sensor:
            require(row['applied_physics_tick'] == tick, 'Native delayed target application clock differs')
        kind, input_key, output_key = ('sensor', 'input_sensors18', 'feedback18') if sensor else ('target', 'input_targets6', 'delayed_targets6')
        size = 18 if sensor else 6
        finite_vector(row[input_key], size, 'Native delay journal input invalid')
        finite_vector(row[output_key], size, 'Native delay journal output invalid')
        history = self.history[kind][lane]
        history.append(row[input_key])
        groups = DELAY_GROUPS if sensor else ((0, 4), (4, 6))
        expected = [v for age, (start, end) in zip(valid, groups, strict=True) for v in history[-1 - age][start:end]]
        require(row[output_key] == expected, 'Native grouped history output does not match its source')
        for column, value in zip(columns, row['actual_age_ms'], strict=True):
            self.age_min[column] = min(self.age_min[column], value)
            self.age_max[column] = max(self.age_max[column], value)
        if sensor:
            self.latest_sensor[lane] = row
            self.sensor_links[tick, lane] = fingerprint(row)
            self.sensor_inputs[tick, lane] = row[input_key]
        else:
            self.target_links[tick, lane] = fingerprint(row), row[output_key]

    def check_action(self, row, encoder):
        import torch
        policy_tick = row['policy_tick']
        values, provenance = row['policy_input'], row['actor_sensor_delay']
        require(len(values) == len(provenance) == 12, 'Delayed actor population differs')
        feedback = []
        for lane, (frame, observed) in enumerate(zip(values, provenance, strict=True)):
            finite_vector(frame, 35, 'Delayed actor ABI is not finite 35D')
            require(observed['consumed_at_physics_tick'] == (policy_tick - 1) * 4
                    and observed['actor_observation_columns_affected'] == [4, 22]
                    and observed['pd_feedback_scope'] == 'fresh_physics_feedback', 'Delayed actor consumption clock/scope differs')
            expected_sha, sensor = self.actor_links[policy_tick, lane]
            copied = deepcopy(observed)
            for key in ('consumed_at_physics_tick', 'actor_observation_columns_affected', 'pd_feedback_scope'):
                copied.pop(key)
            if policy_tick == 1:
                copied.pop('feedback18')
            require(fingerprint(copied) == expected_sha and observed['feedback18'] == sensor,
                    'Actor did not consume latest native sensor feedback')
            feedback.append(sensor)
        sensors = torch.tensor(feedback)
        encoded = encoder(sensors[:, 12:15], sensors[:, 15:18], sensors[:, :6], sensors[:, 6:12], torch.tensor(self.nominal))
        supplied = torch.tensor(values)[:, 4:22]
        error = float((encoded - supplied).abs().amax())
        self.actor_encoding_max_error = max(self.actor_encoding_max_error, error)
        require(torch.allclose(encoded, supplied, rtol=1e-6, atol=2e-6), 'Actor sensor18 encoding differs from native delayed feedback')

    def check_physical(self, row):
        tick, lane = row['physics_tick'], row['lane']
        delay = row['signal_delay']
        target_sha, targets = self.target_links[tick, lane]
        sensor_sha = self.sensor_links[tick - 1, lane] if tick > 1 else fingerprint(self.reset_rows[lane])
        require(delay['native_observation_refresh_follows_this_row'] is True
                and fingerprint(delay['target']) == target_sha
                and fingerprint(delay['observation_feedback_before_native_refresh']) == sensor_sha,
                'Physical recorder target/previous sensor provenance differs')
        require(row['pd_feedback_matches_fresh_physics'] is True, 'Pipeline physical PD feedback is not fresh')
        previous = self.previous_physical.get(lane)
        if previous is not None:
            require(row['pd_q'] == previous[0] and row['pd_dq'] == previous[1],
                    'PD did not consume the previous refreshed physical q/dq')
        else:
            require(row['pd_q'] == row['decode_q'], 'Initial PD feedback differs from fresh native decode')
        self.previous_physical[lane] = row['q6'], row['dq6']
        require(row['q6'] + row['dq6'] + row['root_angular_velocity_body_rad_s'] + row['projected_gravity_b']
                == self.sensor_inputs[tick, lane], 'Native sensor append differs from refreshed physical state')
        require(row['pd_leg_targets'] + row['pd_wheel_targets'] == targets,
                'Physical PD did not use native delayed targets')

    def summary(self):
        enabled = self.state['enabled']
        return {'native_lane_event_counts': dict(self.counts), 'quantum_ms': 5.,
            'enabled_count': sum(enabled), 'fixed_birth_denominator': 12,
            'sampled_profile_counts_all_lanes': dict(Counter(self.state['profile'])),
            'sampled_profile_counts_enabled_lanes': dict(Counter(p for p, on in zip(self.state['profile'], enabled, strict=True) if on)),
            'requested_age_min_ms_control_group_order': self.lag_min, 'requested_age_max_ms_control_group_order': self.lag_max,
            'effective_history_age_min_ms_control_group_order': self.age_min, 'effective_history_age_max_ms_control_group_order': self.age_max,
            'actor_sensor18_encoding_max_abs_error': self.actor_encoding_max_error,
            'raw_jitter_draw_coverage': 'not_covered_native_local_not_persisted',
            'coverage_scope': 'actual N12 reset samples and quantized ages only; no claim of complete 80ms tail/profile coverage',
            'target_feedback_history_verified': True, 'fresh_200Hz_PD_verified': True}


def validate_pipeline_journals(directories, reports, runtimes, configs, cohorts, *, policy_steps=1000):
    audits = {arm: PipelineJournalAudit(runtimes[arm], configs[arm], cohorts[arm], read(path / 'birth.json'))
              for arm, path in directories.items()}
    streams = {arm: iter(records(path / 'signal_delay_frames.jsonl')) for arm, path in directories.items()}
    def group(event, policy_tick, physics_tick):
        for lane in range(12):
            pair = {}
            for arm, stream in streams.items():
                row = next(stream, None)
                require(row is not None, 'Native delay journal is incomplete')
                audits[arm].consume(row, event, lane, policy_tick, physics_tick)
                pair[arm] = {key: row[key] for key in (*DELAY_STATE_KEYS, 'event', 'lane', 'case', 'policy_tick', 'physics_tick')}
                if event != 'policy_lag_sample':
                    pair[arm].update({key: row[key] for key in ('append_physics_tick', 'history_samples', 'actual_age_steps',
                                                             'actual_age_ms', 'source_physics_tick')})
            require(pair['mean'] == pair['stochastic'], 'Paired native lag/profile/effective-age stream differs')
    for policy_tick in range(1, policy_steps + 1):
        start = (policy_tick - 1) * 4
        group('policy_lag_sample', policy_tick, start)
        for tick in range(start + 1, start + 5):
            group('target_applied', policy_tick, tick - 1)
            group('observation_refresh', policy_tick, tick)
    require(all(next(stream, None) is None for stream in streams.values()), 'Native delay journal has extra calls')
    for arm, report in reports.items():
        require(report['signal_delay_native_method_counts'] == {'begin_policy': policy_steps, 'targets': policy_steps * 4, 'observe': policy_steps * 4},
                'Native delay report call counts differ from full trace')
    return audits


def compare(mean, stochastic, fixture_role=None):
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
    canonical, normalized, bindings, roles = {}, {}, {}, {}
    for arm, directory in directories.items():
        canonical[arm], normalized[arm], bindings[arm] = normalize_runtime_binding(directory, reports[arm], runtimes[arm],
            read(directory / 'clean_effective_contract.json'))
        roles[arm] = role_identity(reports[arm], runtimes[arm], canonical[arm])
    role = roles['mean']
    require(role == roles['stochastic'] and (fixture_role is None or role == fixture_role),
            'Only one matching fixture role may be paired')
    require(canonical['mean'] == canonical['stochastic'], 'Effective physical fixtures differ')
    require(normalized['mean'] == normalized['stochastic'], 'Fresh public births/materials/config validation differ')
    require(sha(directories['mean'] / 'clean_design_expectations.json')
            == sha(directories['stochastic'] / 'clean_design_expectations.json'), 'Independent design differs')
    cohorts = {arm: read(path / 'first_cohort.json') for arm, path in directories.items()}
    for arm, cohort in cohorts.items():
        validate_cohort(cohort, runtimes[arm])
        require({key: value for key, value in cohort.items() if key != 'lane_phase_metrics'}
                == reports[arm]['first_cohort'], 'First-cohort file differs from final native report')
    delay_audits, native_encoder = {}, None
    if role == 'pipeline_age_only':
        source = Path(__file__).resolve().parents[3] / 'src'
        native_file = 'src/wheeled_tasks/chassis/scut_observation.py'
        require(sha(source.parent / native_file) == a['source_authentication']['actual_source_files_sha256'][native_file],
                'Native actor sensor encoder source differs from captured source')
        sys.path.insert(0, str(source))
        from wheeled_tasks.chassis.scut_observation import build_sensor18
        native_encoder = build_sensor18
        delay_audits = validate_pipeline_journals(directories, reports, runtimes, canonical, cohorts)
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
        for arm, row in (('mean', x), ('stochastic', y)):
            if role == 'pipeline_age_only':
                delay_audits[arm].check_action(row, native_encoder)
            else:
                require(row.get('actor_sensor_delay') is None, 'Clean arm unexpectedly consumed delayed sensor provenance')
    require(action_pairs == 1000, 'Paired policy trace is incomplete')
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
            if role == 'pipeline_age_only':
                delay_audits[arm].check_physical(row)
            else:
                require('signal_delay' not in row, 'Clean arm unexpectedly recorded native signal delay')
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
        if role == 'pipeline_age_only':
            trace_receipts[arm]['signal_delay_frames.jsonl'] = sha(directory / 'signal_delay_frames.jsonl')
    return {'schema': 'native-mean-stochastic-comparison-v2', 'status': 'paired_diagnostic_verified',
        'fixture_role': role,
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
        'pipeline_delay_validation': ({arm: audit.summary() for arm, audit in delay_audits.items()} if delay_audits else None),
        'torque_saturation_definition': 'abs(native pre-cap requested effort) >= instantaneous bound - 1e-6; '
                                       'six control-order axes; includes zero drive bounds',
        'artifact_SHA256': trace_receipts,
        'scope': f'One N4-per-case same-PT pair within {role}; native action sampling is paired within this role. '
                 'No cross-role generator pairing or strict single-physical-intervention claim for pipeline minus clean; '
                 'no global reward attribution; completed-window means are conditional on completion.'}


def export_events(directory, arm, fixture_role=None):
    from torch.utils.tensorboard import SummaryWriter
    path = Path(directory)
    declared = read(path / 'report.json').get('fixture_role', 'clean')
    require(declared in FIXTURE_ROLES and (fixture_role is None or fixture_role == declared),
            'TensorBoard fixture role differs from actual invocation')
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
                prefix = f'ExplorationProbe/{declared}/{arm}/{case}'
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
            'fixture_role': declared,
            'curve_scope': 'self-survivor instantaneous 50Hz sampled; not paired common-exposure means',
            'sampling_stride_physics_ticks': 4, 'sampling_hz': 50,
            'event_SHA256': {p.name: sha(p) for p in target.iterdir() if p.is_file()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mean', type=Path, required=True)
    parser.add_argument('--stochastic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-events', action='store_true')
    parser.add_argument('--fixture-role', choices=FIXTURE_ROLES,
                        help='Assert the recorded role; never change or synthesize a fixture')
    args = parser.parse_args()
    require(not args.output.exists(), 'Comparison artifact already exists')
    result = compare(args.mean, args.stochastic, args.fixture_role)
    if args.export_events:
        result['tensorboard_exports'] = [export_events(args.mean, 'mean', result['fixture_role']),
                                        export_events(args.stochastic, 'stochastic', result['fixture_role'])]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()), 'cases': result['cases']}))
