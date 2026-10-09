"""Read completed native worlds, verify identity, and summarize observed behavior."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import math
import sys


BASE = Path(__file__).resolve().parent
RUN = BASE / 'rec_speed_mean43500_independent_compare011'
OUTPUT = BASE / 'periodic43500_compare011_root_review001.json'


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def regular_rows(trace):
    return [row for row in trace
            if abs((row['time_s'] - .02) / .2 - round((row['time_s'] - .02) / .2)) < 1e-7]


def movement(trace):
    samples = []
    for row in regular_rows(trace):
        if not 1.02 - 1e-7 <= row['time_s'] <= 9.82 + 1e-7:
            continue
        for lane in range(4, 8):
            if row['active'][lane] and abs(row['requested'][lane][1]) > .001:
                gravity = row['gravity'][lane]
                samples.append(dict(requested=row['requested'][lane][1],
                    applied=row['applied'][lane][1], actual=row['omega'][lane][2],
                    pitch_deg=math.degrees(math.asin(max(-1., min(1., gravity[0])))),
                    roll_deg=math.degrees(math.atan2(-gravity[1], -gravity[2])),
                    tilt_deg=math.degrees(math.acos(max(-1., min(1., -gravity[2]))))))
    assert len(samples) == 180
    return dict(sample_count=len(samples), sample_rate_hz=5., time_range_s=[1.02, 9.82],
        requested_yaw_mean_rad_s=mean(x['requested'] for x in samples),
        applied_yaw_mean_rad_s=mean(x['applied'] for x in samples),
        actual_yaw_mean_rad_s=mean(x['actual'] for x in samples),
        request_yaw_mae_rad_s=mean(abs(x['actual'] - x['requested']) for x in samples),
        applied_yaw_mae_rad_s=mean(abs(x['actual'] - x['applied']) for x in samples),
        posture={key: dict(mean_absolute=mean(abs(x[key]) for x in samples),
                           peak_absolute=max(abs(x[key]) for x in samples))
                 for key in ('pitch_deg', 'roll_deg', 'tilt_deg')})


def recovery(data):
    result = {}
    for name, group in data['native_events']['groups'].items():
        if not name.startswith('recovery_'):
            continue
        episodes = group['native_recovery_episodes']
        assert len(episodes) == group['native_recovery_ended_count'] == 4
        diagnostics = group['native_recovery_diagnostics']
        result[name] = dict(ended=len(episodes),
            success=sum(x['true_success'] == 1. for x in episodes),
            motion_released=sum(x['motion_released'] == 1. for x in episodes),
            failure_codes=[x['failure_code'] for x in episodes],
            diagnostic_reasons=[x['reason'] for x in diagnostics])
    assert len(result) == 4
    return result


def summarize(data, trace, protocol):
    groups = data['native_episode_metrics']['groups']
    fields = ('planar_speed_mean_cm_s', 'vx_mae_m_s', 'yaw_mae_rad_s', 'height_mae_m',
              'tilt_max_deg', 'stand_drift_max_m', 'frames', 'episodes', 'failures',
              'boundary_truncations')
    stands = {name: {key: groups[name][key] for key in fields}
              for name in ('stand_230mm', 'stand_305mm', 'stand_430mm')}
    replicas = data['native_events']['replicas']
    assert len(replicas) == 28 and data['right_censored'] == 0
    assert all(not row['right_censored'] and row['terminal_s'] is not None for row in replicas)
    recovered = recovery(data)
    result = dict(first_episode_stands=stands, ended_first_episodes=28, right_censored=0,
        recovery=recovered, recovery_success=sum(x['success'] for x in recovered.values()),
        recovery_ended=sum(x['ended'] for x in recovered.values()),
        released_lanes=[i for i, x in enumerate(replicas) if i >= 12 and x['first_blend_s'] is not None],
        script_timeout_lanes=[i for i, x in enumerate(replicas)
                              if i >= 12 and x['first_blend_s'] is None],
        max_closure_gap_mm=1000 * max(x['closure_gap_max_m'] for x in groups.values()))
    if protocol != 'nominal':
        result['movement_fixed_5Hz'] = movement(trace)
        quiet = data['fixed_quiet_window_metrics']['groups']['stand_305mm']
        assert quiet['frames'] == 2000 and quiet['observed_sim_seconds'] == 40.
        result['quiet_native_50Hz'] = {key: quiet[key] for key in fields if key in quiet}
    return result


def prefix(left, right, traces):
    rows = [{round(row['time_s'], 9): row for row in regular_rows(trace)} for trace in traces]
    compared = 0
    for lane in range(12, 28):
        ends = []
        for data in (left, right):
            row = data['native_events']['replicas'][lane]
            ends.append(row['first_blend_s'] if row['first_blend_s'] is not None else row['terminal_s'])
        assert ends[0] == ends[1]
        for time_s in sorted(rows[0].keys() & rows[1].keys()):
            a, b = rows[0][time_s], rows[1][time_s]
            if time_s >= ends[0] or not (a['active'][lane] and b['active'][lane]):
                continue
            for field in ('position', 'velocity', 'omega', 'gravity', 'height'):
                assert a[field][lane] == b[field][lane], (lane, time_s, field)
            compared += 1
    return dict(compared_lane_time_samples=compared, physical_fields_exact_equal=True,
        first_blend_or_terminal_times_exact_equal=True,
        solver_internal_state_equivalence_proven=False)


def main():
    assert not OUTPUT.exists() and (RUN / 'completion.json').exists() and not (RUN / 'failure.json').exists()
    launcher = module(BASE / 'run_periodic43500_compare011.py', 'review_launcher011')
    launcher.check_adapter(launcher.ADAPTER_COMMIT)
    sys.path.insert(0, str(launcher.ADAPTER / 'scripts'))
    driver = module(launcher.ADAPTER / 'scripts/run_rec_speed_mean_replay.py', 'review_driver011')
    root = driver.prep.frozen_runtime(launcher.RUNTIME)
    contracts, identities = zip(*(driver.prep.authenticate_candidate(root, *candidate)
                                   for candidate in launcher.PAIR))
    abi = driver.prep.shared_abi(contracts)
    driver.prep.validate_pair(identities)
    launcher.check_candidate_bytes(identities)
    intent, complete = read(RUN / 'intent.json'), read(RUN / 'completion.json')
    assert complete['intent_sha256'] == pin(RUN / 'intent.json')['sha256']
    assert complete['status'] == 'independent_native_motion_completed_unscored'
    assert complete['PPO_updates'] == 0 and not complete['optimizer_constructed']
    assert complete['qualification'] is None and not complete['hardware_deployment_ready']
    pids, process_checks, evidence, results = [], [], [], []
    previous_ended = None
    for plan in intent['plans']:
        protocol, projection = plan['protocol'], Path(plan['projection'])
        candidates, traces = [], []
        for index in (0, 1):
            child = RUN / protocol / f'world_{index:02d}'
            external = read(child.with_suffix('.process.json'))
            execution = read(child.with_suffix('.execution.json'))
            assert execution['exit_code'] == 0 and execution['process'] == external
            assert type(external['starttime']) is int and external['starttime'] > 0
            assert external['command'] == launcher.child_command(launcher.ADAPTER / 'scripts/run_rec_speed_mean_replay.py',
                root, protocol, projection, plan['receipt_sha256'], index)[1]
            assert previous_ended is None or previous_ended <= external['started_at_utc']
            previous_ended = execution['ended_at_utc']
            actual = driver.verify_single_child_completion(child, protocol, index, identities,
                root, projection, plan['receipt_sha256'], abi)
            assert actual['native_constructor']['pid'] == external['pid']
            assert actual['candidates'][0]['identity'] == identities[index]
            pids.append(external['pid'])
            data = read(child / f'candidate_{index:02d}.json')
            trace = read(child / f'candidate_{index:02d}_trace.json')
            assert data == actual['candidates'][0]
            candidates.append(data)
            traces.append(trace)
            process_checks.append(dict(protocol=protocol, index=index, pid=external['pid'],
                starttime=external['starttime'], exit_code=0, full_argv_exact=True,
                strict_single_validator_passed=True, actual_constructor_pid_equal=True))
            for path in (child / 'native_replay.json', child / f'candidate_{index:02d}.json',
                         child / f'candidate_{index:02d}_trace.json',
                         child.with_suffix('.process.json'), child.with_suffix('.execution.json')):
                evidence.append(pin(path))
        assert candidates[0]['reset_fingerprint'] == candidates[1]['reset_fingerprint']
        results.append(dict(protocol=protocol, initial_fingerprints_exact_equal=True,
            candidates=[dict(index=i, **summarize(candidates[i], traces[i], protocol)) for i in (0, 1)],
            preblend_physical_prefix=prefix(*candidates, traces)))
    assert len(pids) == len(set(pids)) == 6
    result = dict(schema='periodic-speed-native-compare011-root-review-v1',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='six_actual_fresh_worlds_verified_unscored',
        helper=pin(__file__), intent=pin(RUN / 'intent.json'), completion=pin(RUN / 'completion.json'),
        adapter_commit=launcher.ADAPTER_COMMIT, runtime_commit=driver.prep.E5,
        actual_source_identities=identities, process_checks=process_checks,
        distinct_serial_native_processes=True, evidence=evidence, protocols=results,
        reviewer_actions=dict(native_runs=0, PPO_updates=0, optimizer_constructed=False,
                             remote_writes=0, training_signals=0),
        qualification=None, hardware_deployment_ready=False, new_model_stronger_established=False,
        unique_training_cause_established=False, solver_internal_state_equivalence_proven=False,
        scope='Nominal stand and yaw ±1, 4 replicas per scene; not the full high-speed translation domain.')
    with OUTPUT.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], output=str(OUTPUT)), ensure_ascii=False))


if __name__ == '__main__':
    main()
