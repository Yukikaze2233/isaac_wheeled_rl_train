"""Root-only readback and recomputation of the four completed native spin worlds."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('spin013_runner', BASE / 'run_spin_geometry43500_compare013.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.check_adapter(runner.ADAPTER_COMMIT)
import sys
sys.path.insert(0, str(runner.ADAPTER / 'scripts'))
import run_rec_speed_mean_replay as driver
root = driver.prep.frozen_runtime(runner.RUNTIME)
contracts, identities = zip(*(driver.prep.authenticate_candidate(root, *pair) for pair in runner.PAIR))
abi = driver.prep.shared_abi(contracts)
intent = driver.prep.read(runner.OUTPUT / 'intent.json')
completion = driver.prep.read(runner.OUTPUT / 'completion.json')
assert intent['original_candidates'] == list(identities)
assert completion['intent_sha256'] == runner.digest(runner.OUTPUT / 'intent.json')
assert completion['status'] == 'independent_native_motion_completed_unscored'
report = dict(schema='native-spin-root-review-v1', reviewer='root', independent_agent_review=False,
    source_commit=runner.ADAPTER_COMMIT, runtime_commit=driver.prep.E5, PPO_updates=0,
    training_signals=0, qualification=None, hardware_deployment_ready=False,
    initial_fingerprints_equal=True, solver_internal_state_equivalence_proven=False,
    completion_sha256=runner.digest(runner.OUTPUT / 'completion.json'), comparisons=[])
for plan in intent['plans']:
    rows = []
    for i in (0, 1):
        directory = runner.OUTPUT / plan['protocol'] / f'world_{i:02d}'
        value = driver.verify_single_child_completion(directory, plan['protocol'], i, identities,
            root, plan['projection'], plan['receipt_sha256'], abi)
        external = driver.prep.read(directory.parent / f'world_{i:02d}.process.json')
        execution = driver.prep.read(directory.parent / f'world_{i:02d}.execution.json')
        assert execution['exit_code'] == 0 and value['native_constructor']['pid'] == external['pid']
        assert type(external['starttime']) is int and external['command'] == execution['process']['command']
        assert value['geometry_observer_sha256'] == runner.GEOMETRY_SHA
        measurement = value['candidates'][0]
        geometry = measurement['spin_geometry']
        frames = geometry['policy_trace']
        nominal = measurement['native_episode_metrics']['groups']['stand_305mm']
        quiet = measurement['fixed_quiet_window_metrics']['groups']['stand_305mm']
        n = len(frames)
        mean = lambda key: sum(f[key] for f in frames) / n if n else None
        target_frames = [f for f in frames if abs(f['applied'][1] - f['requested'][1]) <= .1]
        nt = len(target_frames)
        row = dict(candidate_index=i, consumed_updates=identities[i]['consumed_updates'],
            native_pid=external['pid'], starttime=external['starttime'],
            completed_first_episodes=sum(g['episodes'] for g in measurement['native_episode_metrics']['groups'].values()),
            right_censored=measurement['right_censored'], spin_policy_frames=n, applied_target_frames=nt,
            planar_speed_mean_m_s=mean('planar_speed_world_m_s'), actual_yaw_mean_rad_s=mean('yaw_actual_rad_s'),
            applied_yaw_mean_rad_s=sum(f['applied'][1] for f in frames)/n if n else None,
            requested_yaw_rad_s=math.copysign(math.tau, 1 if plan['protocol']=='spin_positive' else -1),
            leg_length_mae_m=sum(abs(f['leg_length_delta_m']) for f in frames)/n if n else None,
            leg_angle_mae_rad=sum(abs(f['leg_angle_delta_rad']) for f in frames)/n if n else None,
            root_displacement_max_m=max((f['root_displacement_m'] for f in frames), default=None),
            wheel_midpoint_displacement_max_m=max((f['wheel_midpoint_displacement_m'] for f in frames), default=None),
            both_wheel_force_above_2n_frames=sum(all(v>2 for v in f['wheel_contact_force_n']) for f in frames),
            neither_wheel_force_above_2n_frames=sum(all(v<=2 for v in f['wheel_contact_force_n']) for f in frames),
            target_subset_actual_yaw_mean_rad_s=sum(f['yaw_actual_rad_s'] for f in target_frames)/nt if nt else None,
            target_subset_planar_speed_mean_m_s=sum(f['planar_speed_world_m_s'] for f in target_frames)/nt if nt else None,
            whole_episode_tilt_max_deg=nominal['tilt_max_deg'],
            whole_episode_height_mae_m=nominal['height_mae_m'],
            terminal_first_episode_time_mean_s=nominal['terminal_episode_time_mean_s'],
            physical_failures=nominal['failures'], timeouts=nominal['timeouts'],
            termination_reasons=nominal['termination_reasons'], quiet_frames=quiet['frames'],
            quiet_planar_speed_mean_m_s=quiet['planar_speed_mean_m_s'] if quiet['frames'] else None,
            closure_gap_peak_m=max(g['closure_gap_max_m'] for g in measurement['native_episode_metrics']['groups'].values()),
            native_result_file=str(directory/'native_replay.json'), native_result_sha256=runner.digest(directory/'native_replay.json'),
            geometry_trace_sha256=driver.canonical_sha(frames))
        assert row['completed_first_episodes'] == 28 and row['right_censored'] == 0
        rows.append(row)
    assert rows[0]['native_pid'] != rows[1]['native_pid']
    report['comparisons'].append(dict(protocol=plan['protocol'], rows=rows))
assert len({r['native_pid'] for c in report['comparisons'] for r in c['rows']}) == 4
report['scope'] = 'Requested yaw-only 1..10s, native pure-policy first-episode samples; slew/early termination retained. Whole-episode tilt and quiet windows use their own actual denominators.'
report['limits'] = 'One fixed seed, four replicas per scene; no causal reward ablation, statistical multi-training-seed result, 3rev/s result or deployment qualification.'
path = BASE/'spin_geometry43500_compare013_root_review001.json'
assert not path.exists()
path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
print(json.dumps(report,ensure_ascii=False,allow_nan=False))
