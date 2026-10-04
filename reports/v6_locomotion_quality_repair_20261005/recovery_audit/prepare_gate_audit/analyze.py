#!/usr/bin/env python3
"""CPU-only audit of existing prefix 002; never imports Isaac or contacts SSH.

Frozen classes are read from Git, hash-authenticated against the probe receipt.
Counterfactuals advance cloned controller snapshots on identical held inputs;
they do not predict a physical trajectory or restore PhysX internal state.
"""
from __future__ import annotations

import ast
from collections import Counter
import difflib
import hashlib
import json
import math
from pathlib import Path
import subprocess
import types

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RAW = OUT.parent / 'native_prefix_002'
REVISION = 'b4ce80964d47c0127e0b3d9929ad35a620bbf204'
CONTROL = [0, 2, 8, 1, 3, 10]
LEGS = [0, 1, 3, 4]
WHEELS = [2, 5]
DT = .005


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_lines(name):
    return [json.loads(line) for line in (RAW / name).read_text().splitlines()]


def frozen(path, report):
    data = subprocess.check_output(['git', 'show', f'{REVISION}:{path}'], cwd=ROOT)
    assert sha(data) == report['source_files_sha256'][path], path
    return data.decode()


def module(source, name):
    result = types.ModuleType(name)
    # No package import or simulator dependency occurs in these pure classes.
    import sys
    sys.modules[name] = result
    exec(compile(source, f'<frozen:{name}>', 'exec'), result.__dict__)
    return result


def gravity(pose):
    x, y, z, w = torch.tensor(pose[3:], dtype=torch.float32)
    return torch.stack((2 * (w*y - x*z), -2 * (y*z + w*x), -(1 - 2*(x*x+y*y))))


def attitude(g):
    g = g / g.norm()
    return dict(tilt_deg=math.degrees(math.acos(float((-g[2]).clamp(-1, 1)))),
                pitch_deg=math.degrees(math.atan2(float(g[0]), float(-g[2]))),
                gravity_roll_deg=math.degrees(math.atan2(float(-g[1]), float(-g[2]))))


def longest(values):
    run = best = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def tensor_slice(snapshot, key, lane):
    return snapshot[key][lane:lane+1].clone()


def scalar_summary(frame, previous, reconstruction):
    dq = torch.tensor(previous['dq18'])[CONTROL]
    return {'sim_time_s': frame['sim_time_s'], 'physical_tick': frame['physical_tick'],
            **attitude(gravity(previous['root_pose_world_xyzw'])),
            'root_z_m_prephysics': previous['root_pose_world_xyzw'][2],
            'height_if_grounded_m': reconstruction['height'],
            'specific_acceleration_m_s2': reconstruction['acceleration_norm'],
            'continuous_control_q_rad': reconstruction['q'],
            'control_dq_rad_s_prephysics': dq.tolist(),
            'gyro_body_rad_s_prephysics': previous['root_angular_velocity_body_rad_s'],
            'script_targets6': frame['script_targets6'], 'applied_torque6_nm': frame['applied_torque6'],
            'script_target_step_max_rad': max(abs(frame['script_targets6'][i]-previous['script_targets6'][i]) for i in LEGS),
            'gates': frame['gates'], 'ready_ticks': frame['ready_ticks'],
            'failed_gates': [k for k, v in frame['gates'].items() if not v]}


def main():
    torch.set_num_threads(1)
    report = json.loads((RAW / 'report.json').read_text())
    contract = json.loads((RAW / 'effective_contract.json').read_text())
    assert report['frozen_physical_control_sources']['revision'] == REVISION
    snapshots = torch.load(RAW / 'event_snapshots.pt', map_location='cpu', weights_only=False)['snapshots']
    frames, policy, events = (load_lines(name) for name in ('physical_frames.jsonl', 'policy_frames.jsonl', 'events.jsonl'))
    src = {name: frozen(f'src/wheeled_tasks/chassis/{name}.py', report)
           for name in ('recovery', 'continuous_encoders', 'mechanism_observer', 'recovery_observer', 'step_trigger', 'recovery_training')}
    fsm_module = module(src['recovery'], 'prefix_audit_fsm')
    encoder_module = module(src['continuous_encoders'], 'prefix_audit_encoders')
    geometry_module = module(src['mechanism_observer'], 'prefix_audit_geometry')
    observer_module = module(src['recovery_observer'], 'prefix_audit_observers')
    node = next(n for n in ast.parse(src['step_trigger']).body if isinstance(n, ast.FunctionDef) and n.name == 'estimate_height_if_grounded')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen:height-function>', 'exec'), ns)
    estimate = ns['estimate_height_if_grounded']
    settings = contract['recovery_training']
    tables = {}
    for key in ('profiles', 'height_lookup'):
        data = (ROOT / settings[key+'_file']).read_bytes()
        assert sha(data) == settings[key+'_sha256']
        tables[key] = json.loads(data)
    model = geometry_module.ClosedChainHeightModel(tables['height_lookup'], 'cpu')
    limits = torch.stack([s['delta_rad'][[0, -1]] for s in model.sides])
    results = {}
    for lane in range(4):
        lane_frames = [f for f in frames if f['lane'] == lane]
        birth = next(e for e in events if e['lane'] == lane and e['event'] == 'birth_complete')
        birth_s = snapshots[birth['snapshot_index']]
        previous = {'q18': birth_s['physics.joint_pos'][lane].tolist(),
                    'dq18': birth_s['physics.joint_vel'][lane].tolist(),
                    'root_pose_world_xyzw': birth_s['physics.root_link_pose_w'][lane].tolist(),
                    'root_linear_velocity_world_m_s': birth_s['physics.root_link_lin_vel_w'][lane].tolist(),
                    'root_angular_velocity_body_rad_s': birth_s['physics.root_com_ang_vel_b'][lane].tolist(),
                    'raw_fsm_phase_used': 9,
                    'script_targets6': birth_s['fsm.script_targets'][lane].tolist()}
        encoder = encoder_module.ContinuousEncoderAngles(1, 'cpu', limits)
        encoder.reset([0], torch.tensor(previous['q18'])[CONTROL][None])
        probe = observer_module.WheelSupportProbe('cpu', 1, DT)
        previous_pre_velocity = torch.tensor(previous['root_linear_velocity_world_m_s'])
        released = False
        replay = []
        mismatch = Counter()
        for frame in lane_frames:
            raw_q = torch.tensor(previous['q18'], dtype=torch.float32)[CONTROL][None]
            encoder.update(raw_q)
            if frame['release_finished_used'] and not released:
                encoder.realign(torch.tensor([0]))
            released = frame['release_finished_used']
            q = encoder.project(raw_q)
            g = gravity(previous['root_pose_world_xyzw'])[None]
            dq = torch.tensor(previous['dq18'])[CONTROL][None]
            gyro = torch.tensor(previous['root_angular_velocity_body_rad_s'])[None]
            velocity = torch.tensor(previous['root_linear_velocity_world_m_s'])
            acceleration_norm = ((velocity - previous_pre_velocity) / DT + torch.tensor([0., 0., 9.81])).norm()
            previous_pre_velocity = velocity
            points, valid = model.wheel_positions(q)
            height, valid_height = estimate(points, g, model.wheel_radius_m, grounded=valid,
                                             sensor_valid=valid, wheel_axle_b=model.wheel_axles_b)
            factors = {'geometry_and_height_valid': bool(valid_height.all()),
                       'height_above_120mm': bool(height.amin() > .12),
                       'two_wheel_height_difference_below_35mm': bool((height[:,0]-height[:,1]).abs() < .035),
                       'specific_acceleration_between_6_16': bool(6 < acceleration_norm < 16),
                       'gyro_norm_below_8': bool(gyro.norm() < 8)}
            plausible = torch.tensor([all(factors.values())])
            confirmed = probe.observe(dq[:, WHEELS], gyro, plausible)
            # Body-clear is computed before this tick's FSM transition.
            prev_phase = previous['raw_fsm_phase_used'] if released else 9
            if released and not previous.get('release_finished_used', False):
                prev_phase = 0  # the release boundary resets FSM before observer use
            body_clear = bool((confirmed | (prev_phase in (8, 9))) & plausible & (height.amin() > .27))
            reconstructed_gates = {'valid': bool(torch.isfinite(q).all() and torch.isfinite(dq).all()
                and torch.isfinite(g).all() and torch.isfinite(gyro).all() and torch.isfinite(height.amin()) and g.norm() > .5),
                'support': bool(plausible), 'body_clear': body_clear,
                'tilt': bool(torch.acos((-g[:,2]).clamp(-1,1)) < math.radians(8)),
                'gyro': bool(gyro.norm() < .75), 'height': bool(.27 < height.amin() < .36),
                'wheel_speed': bool(dq[:,WHEELS].abs().max() * .06 < .25),
                'leg_speed': bool(dq[:,LEGS].abs().max() < 2.)}
            for gate, passed in reconstructed_gates.items():
                if passed != frame['gates'][gate]:
                    mismatch[gate] += 1
            probe.command(frame['reference_tick'] - 1,
                          torch.tensor([released and frame['raw_fsm_phase_used'] < 8]) & plausible, confirmed)
            reconstructed = {'q': q[0].tolist(), 'height': float(height.amin()),
                             'acceleration_norm': float(acceleration_norm),
                             'support_factors': factors, 'probe_confirmed': bool(confirmed),
                             'support': bool(plausible), 'body_clear': body_clear}
            summary = scalar_summary(frame, previous, reconstructed)
            summary['observer_reconstruction'] = reconstructed
            replay.append((frame, summary))
            previous = frame
        assert not mismatch, (lane, mismatch)
        by_tick = {f['physical_tick']: s for f, s in replay}
        encoder_error = max(max(abs(a-b) for a,b in zip(by_tick[p['physical_tick']]['continuous_control_q_rad'],
                           p['native_recovery']['recovery_continuous_q'])) for p in policy if p['lane'] == lane)
        assert encoder_error < 2e-6, (lane, encoder_error)
        preparing = [(f, s) for f, s in replay if f['effective_control_ownership'] == 'script' and f['phase_used'] == 7]
        assert preparing
        gates = list(preparing[0][0]['gates'])
        counter = Counter(tuple(s['failed_gates']) for _, s in preparing)
        by_interval = []
        for f, s in preparing:
            if not by_interval or f['physical_tick'] != by_interval[-1][-1][0]['physical_tick'] + 1:
                by_interval.append([])
            by_interval[-1].append((f, s))
        intervals = []
        for interval in by_interval:
            vals = [s for _, s in interval]
            bins = []
            for start in range(0, len(interval), 50):
                selected = interval[start:start+50]
                ss = [s for _, s in selected]
                bins.append({'from_s': ss[0]['sim_time_s'], 'through_s': ss[-1]['sim_time_s'],
                             'pitch_deg_first_last': [ss[0]['pitch_deg'], ss[-1]['pitch_deg']],
                             'tilt_deg_max': max(s['tilt_deg'] for s in ss),
                             'gate_pass_ticks': {g: sum(s['gates'][g] for s in ss) for g in gates},
                             'leg_speed_max_rad_s': max(max(abs(s['control_dq_rad_s_prephysics'][i]) for i in LEGS) for s in ss),
                             'wheel_torque_at_4p5_cap_lane_ticks': sum(any(abs(s['applied_torque6_nm'][i]) >= 4.49999 for i in WHEELS) for s in ss)})
            intervals.append({'first': vals[0], 'last': vals[-1], 'ticks': len(vals), 'quarter_second_bins': bins})
        policy_last = next(p for p in reversed(policy) if p['lane'] == lane)
        results[str(lane)] = {'case': lane_frames[0]['case'], 'prepare_ticks': len(preparing),
            'full_eight_gate_ticks': sum(all(f['gates'].values()) for f, _ in preparing),
            'max_ready_ticks': max(f['ready_ticks'] for f, _ in preparing),
            'gate_pass_ticks': {g: sum(f['gates'][g] for f, _ in preparing) for g in gates},
            'gate_longest_pass_seconds': {g: longest([f['gates'][g] for f, _ in preparing])*DT for g in gates},
            'all_except_one_ticks': {g: sum(all(v for k, v in f['gates'].items() if k != g) for f, _ in preparing) for g in gates},
            'fewest_simultaneous_failed_gates': min(len(s['failed_gates']) for _, s in preparing),
            'best_frames': [s for _, s in preparing if len(s['failed_gates']) == min(len(t['failed_gates']) for _, t in preparing)][:6],
            'failed_gate_combinations': [{'gates': list(k), 'ticks': v} for k, v in counter.most_common(12)],
            'support_false_factors_ticks': {k: sum(not s['observer_reconstruction']['support_factors'][k] for _, s in preparing) for k in factors},
            'observer_reconstruction_mismatches_all_physics_ticks': dict(mismatch),
            'encoder_reconstruction_max_error_rad_at_policy_samples': encoder_error,
            'support_false_due_only_to_acceleration_ticks': sum(not s['observer_reconstruction']['support_factors']['specific_acceleration_between_6_16']
                and all(v for k,v in s['observer_reconstruction']['support_factors'].items() if k != 'specific_acceleration_between_6_16') for _,s in preparing),
            'body_clear_false_decomposition_ticks': {'support_false': sum(not s['gates']['support'] for _, s in preparing),
                'support_true_height_not_above_270mm': sum(s['gates']['support'] and s['height_if_grounded_m'] <= .27 for _, s in preparing),
                'support_true_height_above_270mm_probe_unconfirmed': sum(s['gates']['support'] and s['height_if_grounded_m'] > .27 and not s['observer_reconstruction']['probe_confirmed'] for _, s in preparing)},
            'max_applied_leg_torque_nm': max(max(abs(s['applied_torque6_nm'][i]) for i in LEGS) for _, s in preparing),
            'prepare_target_slew_above_1_rad_s_ticks': sum(s['script_target_step_max_rad'] > 1.001*DT for _,s in preparing),
            'prepare_target_slew_at_4_rad_s_ticks': sum(abs(s['script_target_step_max_rad']/DT-4.) < .002 for _,s in preparing),
            'prepare_target_slew_max_rad_s': max(s['script_target_step_max_rad']/DT for _,s in preparing),
            'wheel_torque_at_cap_ticks': sum(any(abs(s['applied_torque6_nm'][i]) >= 4.49999 for i in WHEELS) for _, s in preparing),
            'terminal_native_recovery': policy_last['native_recovery'], 'intervals': intervals}
    old_line = 'speed[preparing | thrusting] = self.config["push_speed_rad_s"]'
    new_line = 'speed[thrusting] = self.config["push_speed_rad_s"]'
    assert src['recovery'].count(old_line) == 1
    candidate_source = src['recovery'].replace(old_line, new_line)
    candidate_module = module(candidate_source, 'prefix_audit_fsm_candidate')
    cfg = {k: v for k, v in settings.items() if k in fsm_module.RecoveryFSM.DEFAULTS}
    cfg['dt'] = settings['reference_dt']
    table = tables['profiles']
    targets = [table['targets'][name]['control6_rad'] for name in ('nominal', 'fold', 'thrust', 'support')]
    targets[0] = targets[3] = table['prepare_reference']['pd_control6_rad']
    counterfactual = []
    for event in events:
        if not (event['event'].startswith('phase_') or event['event'] == 'terminal_pre_autoreset'):
            continue
        snapshot, lane = snapshots[event['snapshot_index']], event['lane']
        phase = int(snapshot['fsm.phase'][lane])
        if phase not in (1, 2, 3, 4, 5, 6, 7, 8):
            continue
        fsm_pair = [m.RecoveryFSM(1, 'cpu', cfg, *targets, table['axis_signs']) for m in (fsm_module, candidate_module)]
        for fsm in fsm_pair:
            fsm.enabled.fill_(True)
            for key, value in snapshot.items():
                if key.startswith('fsm.'):
                    name = key[4:]
                    if hasattr(fsm, name):
                        setattr(fsm, name, tensor_slice(snapshot, key, lane))
            for gate in fsm.stable_pass_masks:
                fsm.stable_pass_masks[gate].copy_(tensor_slice(snapshot, 'gate.'+gate, lane))
        reference = fsm_pair[0].script_targets.clone()
        inputs = [tensor_slice(snapshot, key, lane) for key in ('encoders.q', 'control.pd_dq',
                  'physics.projected_gravity_b', 'physics.root_com_ang_vel_b', 'recovery.observed_height',
                  'recovery.observed_support', 'gate.body_clear')]
        before_inputs = [t.clone() for t in inputs]
        for fsm in fsm_pair:
            fsm.update(*inputs)
        assert all(torch.equal(a, b) for a, b in zip(inputs, before_inputs))
        same_gates = all(torch.equal(fsm_pair[0].stable_pass_masks[g], fsm_pair[1].stable_pass_masks[g]) for g in fsm_pair[0].stable_pass_masks)
        assert same_gates
        if phase != 7:
            assert all(torch.equal(v, getattr(fsm_pair[1], k)) for k, v in vars(fsm_pair[0]).items() if torch.is_tensor(v)), (event, phase)
        row = {'event_snapshot_index': event['snapshot_index'], 'case': event['case'], 'sim_time_s': event['sim_time_s'],
            'starting_phase': phase, 'held_identical_inputs': True, 'all_sensor_gate_masks_identical': same_gates,
            'reference_step_max_rad_baseline': float((fsm_pair[0].script_targets-reference)[:,LEGS].abs().max()),
            'reference_step_max_rad_candidate': float((fsm_pair[1].script_targets-reference)[:,LEGS].abs().max()),
            'target_max_difference_rad': float((fsm_pair[0].script_targets-fsm_pair[1].script_targets).abs().max()),
            'next_PD_requested_leg_torque_max_difference_nm_at_identical_feedback':
                float(160.*(fsm_pair[0].script_targets-fsm_pair[1].script_targets)[:,LEGS].abs().max()),
            'resulting_phase_baseline_candidate': [int(f.phase) for f in fsm_pair]}
        counterfactual.append(row)
    assert len(counterfactual) > 0 and any(r['starting_phase'] == 7 for r in counterfactual)
    candidate_diff = ''.join(difflib.unified_diff(src['recovery'].splitlines(True), candidate_source.splitlines(True),
                  fromfile='a/src/wheeled_tasks/chassis/recovery.py', tofile='b/src/wheeled_tasks/chassis/recovery.py'))
    (OUT / 'candidate_prepare_slew.diff').write_text(candidate_diff)
    evidence = {'schema': 'v6-prepare-gate-audit-v1', 'status': 'CPU_read_only_analysis_complete',
        'source_revision': REVISION, 'formal_updates_consumed': 0, 'qualification': 'diagnostic_only',
        'checkpoint': report['checkpoint'], 'input_sha256': {n: sha((RAW/n).read_bytes()) for n in
             ('report.json','cohort_summary.json','physical_frames.jsonl','policy_frames.jsonl','events.jsonl','event_snapshots.pt','effective_contract.json')},
        'authenticated_source_sha256': {f'src/wheeled_tasks/chassis/{k}.py': sha(v.encode()) for k,v in src.items()},
        'sampling_alignment': 'Frame gates/targets are preintegration inputs; q/root in the same physical frame are postintegration. Audit shifts q/root/velocity by one physics tick and reconstructs support factors with exact frozen CPU geometry/encoders/probe.',
        'settings': {'dt': DT, 'ready_ticks_required': 20, 'prepare_speed_config_rad_s': 1.,
                     'actual_PREPARE_speed_rad_s': 4., 'candidate_PREPARE_speed_rad_s': 1.,
                     'THRUST_speed_rad_s_unchanged': 4., 'wrapper_timeout_s_includes_release': 8., 'initial_release_s': 1.},
        'lane_results': results, 'CPU_counterfactual': {'scope': 'one next FSM update from cloned recorded controller snapshots; identical held sensor inputs, not a future physical simulation',
            'candidate_change': old_line+' -> '+new_line, 'rows': counterfactual,
            'tested_snapshot_count': len(counterfactual), 'physics_restore_claim': False},
        'limits': ['One exact-seed four-pose cohort, no statistical success estimate.',
                   'Support/body_clear are deployable estimators, not true contact; cached contact snapshots do not bind body indices to names.',
                   'No physical candidate replay was run; CPU target-rate change does not establish improved stabilization.',
                   'Current v14 actor and live training metrics are outside this old-flat prefix audit.']}
    (OUT / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'lane_prepare': {k:{x:v[x] for x in ('prepare_ticks','full_eight_gate_ticks','max_ready_ticks','fewest_simultaneous_failed_gates','observer_reconstruction_mismatches_all_physics_ticks','support_false_due_only_to_acceleration_ticks','encoder_reconstruction_max_error_rad_at_policy_samples','wheel_torque_at_cap_ticks')} for k,v in results.items()},
          'CPU_counterfactual_snapshot_count': len(counterfactual),
          'CPU_PREPARE_counterfactual': [r for r in counterfactual if r['starting_phase'] == 7]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
