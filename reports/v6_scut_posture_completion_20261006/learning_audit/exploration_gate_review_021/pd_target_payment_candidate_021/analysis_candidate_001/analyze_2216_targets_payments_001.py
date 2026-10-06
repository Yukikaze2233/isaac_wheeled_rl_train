#!/usr/bin/env python3
"""Read only a root-sealed complete2216 native journal; produce bounded summaries."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
MANIFEST_SHA = 'e59c5f3bf90cb1e032673b32d0396b96568c203ac0893e8730bedf310bb21682'
GATES = ('valid', 'support', 'body_clear', 'tilt', 'gyro', 'height', 'wheel_speed', 'leg_speed')
TERMS = ('leg_velocity', 'wheel_velocity', 'horizontal_angular_velocity')
LEGS, WHEELS = (0, 1, 3, 4), (2, 5)
AXES = ('L_joint1', 'LL_joint1', 'L_joint3', 'R_joint1', 'RR_joint1', 'R_joint3')


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*p
    lo = int(index)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def stats(values):
    require(all(math.isfinite(x) for x in values), 'Nonfinite native selected value')
    if not values:
        return dict(n=0, mean=None, min=None, max=None, p95=None)
    return dict(n=len(values), mean=math.fsum(values)/len(values), min=min(values),
                max=max(values), p95=quantile(values, .95))


def regime(sample):
    fsm = sample['fsm']
    if not fsm['release_finished']:
        return 'passive_release'
    phase = fsm['phase_after']
    if phase < 8:
        return 'script'
    return {8:'blend', 9:'pure_rl', 10:'failed'}.get(phase, 'unknown')


def policy_regime(row):
    order = list(dict.fromkeys(regime(s) for s in row['substeps']))
    return order[0] if len(order) == 1 else 'mixed_'+'_'.join(order)


def summarize(rows):
    substeps = [(row, substep) for row in rows for substep in row['substeps']]
    regimes = Counter(regime(sample) for _, sample in substeps)
    phase_before = Counter(str(sample['fsm']['phase_before']) for _, sample in substeps)
    phase_after = Counter(str(sample['fsm']['phase_after']) for _, sample in substeps)
    targets = {}
    for label in ('all', 'passive_release', 'script', 'blend', 'pure_rl', 'failed'):
        members = [(row,s) for row,s in substeps if label == 'all' or regime(s) == label]
        if not members:
            continue
        axes = {}
        for index in range(4):
            control = LEGS[index]
            axes[AXES[control]] = dict(units='rad',
                decode_target=stats([row['decode']['leg_targets_rad'][index] for row,s in members]),
                delayed_PD_target=stats([s['pd']['actual_PD_leg_targets_rad'][index] for row,s in members]),
                delayed_minus_decoded=stats([s['pd']['actual_PD_leg_targets_rad'][index]-row['decode']['leg_targets_rad'][index] for row,s in members]),
                actual_continuous_PD_error=stats([s['pd']['actual_PD_leg_targets_rad'][index]-s['pd']['continuous_control_q_rad'][control] for row,s in members]),
                fresh_speed_rad_s=stats([s['pd']['fresh_motor_dq_rad_s'][control] for row,s in members]),
                original_RL_PD_request_nm=stats([s['pd']['original_RL_PD_preclip_request_nm'][control] for row,s in members]),
                original_RL_PD_clipped_nm=stats([s['pd']['original_RL_PD_clipped_torque_nm'][control] for row,s in members]))
        for index in range(2):
            control = WHEELS[index]
            axes[AXES[control]] = dict(units='rad/s',
                decode_target=stats([row['decode']['wheel_targets_rad_s'][index] for row,s in members]),
                delayed_PD_target=stats([s['pd']['actual_PD_wheel_targets_rad_s'][index] for row,s in members]),
                delayed_minus_decoded=stats([s['pd']['actual_PD_wheel_targets_rad_s'][index]-row['decode']['wheel_targets_rad_s'][index] for row,s in members]),
                target_minus_fresh_speed=stats([s['pd']['actual_PD_wheel_targets_rad_s'][index]-s['pd']['fresh_motor_dq_rad_s'][control] for row,s in members]),
                fresh_speed_rad_s=stats([s['pd']['fresh_motor_dq_rad_s'][control] for row,s in members]),
                original_RL_PD_request_nm=stats([s['pd']['original_RL_PD_preclip_request_nm'][control] for row,s in members]),
                original_RL_PD_clipped_nm=stats([s['pd']['original_RL_PD_clipped_torque_nm'][control] for row,s in members]))
        rim_target = [.06*max(abs(x) for x in s['pd']['actual_PD_wheel_targets_rad_s']) for row,s in members]
        rim_actual = [.06*max(abs(s['pd']['fresh_motor_dq_rad_s'][i]) for i in WHEELS) for row,s in members]
        targets[label] = dict(native_substep_samples=len(members), phase_is_after_original_FSM_before_physics=True,
            axes=axes, delayed_target_peak_wheel_rim_m_s=stats(rim_target),
            fresh_actual_peak_wheel_rim_m_s=stats(rim_actual),
            delayed_rim_target_gt_025_m_s_count=sum(x > .25 for x in rim_target),
            target_threshold_comparison_is_not_a_native_gate_or_inevitable_failure=True,
            actual_leg_target_error_is_continuous_difference_not_short_arc=True)
    hold = [(row,s) for row,s in substeps if s['fsm']['native_holding_rl'] and s['fsm']['release_finished']]
    failures = {name: sum(not s['fsm']['native_gate_masks'][name] for row,s in hold) for name in GATES}
    joint = sum(all(s['fsm']['native_gate_masks'].values()) for row,s in hold)
    require(all(s['fsm']['native_stable_pass'] == all(s['fsm']['native_gate_masks'].values()) for row,s in substeps),
            'Saved native stable_pass differs from its direct eight masks')
    streaks, active_streak = [], None
    for row,s in hold:
        reference = s['reference_and_PD_before_physical_step']
        if s['fsm']['native_stable_pass']:
            if active_streak is None or active_streak['last_reference'] != reference-1:
                if active_streak:
                    streaks.append(active_streak)
                active_streak = dict(first_reference=reference, last_reference=reference, samples=1)
            else:
                active_streak['last_reference'] = reference
                active_streak['samples'] += 1
        elif active_streak:
            streaks.append(active_streak)
            active_streak = None
    if active_streak:
        streaks.append(active_streak)
    for streak in streaks:
        streak.update(native_dwell_sample_seconds=streak['samples']*.005,
            first_input_physical_step=streak['first_reference']-1,
            last_input_physical_step=streak['last_reference']-1)
    peak = max(streaks, key=lambda s:s['samples']) if streaks else None
    native_ticks = [s['fsm']['stable_ticks'] for row,s in substeps]
    native_peaks = [s['fsm']['stable_peak_ticks'] for row,s in substeps]
    native_breaks = [s['fsm']['stable_break_count'] for row,s in substeps]
    break_events = []
    previous = 0
    for row,s in substeps:
        current = s['fsm']['stable_break_count']
        if current > previous:
            break_events.append(dict(reference=s['reference_and_PD_before_physical_step'],
                input_physical_step=s['input_physical_step'], increment=current-previous,
                failed_gates=[name for name in GATES if not s['fsm']['native_gate_masks'][name]],
                native_holding_rl=s['fsm']['native_holding_rl'], regime=regime(s)))
        previous = current
    hold_summary = dict(direct_native_holding_samples=len(hold), failed_gate_counts_nonexclusive=failures,
        joint_pass_samples=joint, marginals_must_not_be_multiplied=True,
        first_reference=hold[0][1]['reference_and_PD_before_physical_step'] if hold else None,
        last_reference=hold[-1][1]['reference_and_PD_before_physical_step'] if hold else None,
        first_input_physical_step=hold[0][1]['input_physical_step'] if hold else None,
        last_input_physical_step=hold[-1][1]['input_physical_step'] if hold else None,
        native_final_stable_ticks=native_ticks[-1], native_peak_ticks=max(native_peaks),
        native_peak_seconds=max(native_peaks)*.005, required_stable_ticks=200,
        native_final_break_count=native_breaks[-1], break_events=break_events,
        break_failed_gate_counts_nonexclusive={name: sum(event['increment'] for event in break_events if name in event['failed_gates']) for name in GATES},
        longest_direct_mask_streak=peak,
        direct_mask_streak_matches_native_peak=(peak['samples'] if peak else 0) == max(native_peaks),
        counter_scope='Existing native counters are primary; direct-mask streak arithmetic does not run another FSM.')
    payment_groups = {}
    selected_groups = defaultdict(list)
    for row in rows:
        selected_groups[policy_regime(row)].append(row)
    for label,members in selected_groups.items():
        paid = [row for row in members if row['hold_producer']['native_hold_cost_mask']]
        eligible = [row for row in members if row['policy_end']['whole_interval_actor_valid']]
        payment_groups[label] = dict(policy_intervals=len(members), policy_seconds=len(members)*.02,
            whole_interval_actor_valid_count=len(eligible), native_hold_cost_paid_intervals=len(paid),
            all_actual_three_motion_components={name: dict(
                sum_payment=math.fsum(row['policy_end']['actual_component_payments'][name] for row in members),
                sum_actor_eligible_payment=math.fsum(row['policy_end']['actual_component_payments'][name] for row in eligible)) for name in TERMS},
            actual_native_hold_payment={name: dict(
                sum_payment=math.fsum(row['policy_end']['actual_component_payments'][name] for row in paid),
                end_state_density=stats([row['policy_end']['actual_component_densities'][name] for row in paid])) for name in TERMS},
            native_original_broad_fine={axis: {part: math.fsum(.02*row['hold_producer']['native_diagnostics'][axis+'/'+part] for row in paid)
                for part in ('broad_density','fine_density')} for axis in ('leg','wheel','gyro')},
            all_three_components_are_not_called_hold_payments_when_native_mask_false=True)
    return dict(policy_rows=len(rows), direct_native_substeps=len(substeps),
        substep_regimes=dict(regimes), phase_before_counts=dict(phase_before),phase_after_counts=dict(phase_after),
        targets_by_native_control_regime=targets, native_hold=hold_summary,
        actual_three_payments_by_policy_regime=payment_groups,
        policy_transition_bins_are_explicit=True,
        first_policy_end=rows[0]['policy_end'], final_policy_end=rows[-1]['policy_end'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-output', type=Path, required=True)
    parser.add_argument('--process-receipt', type=Path, required=True)
    parser.add_argument('--root-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    native_output, output = args.native_output.resolve(), args.output.resolve()
    require(HERE in output.parents and not output.exists(), 'New analysis-only output required')
    # Never open a journal until root has sealed actual lifecycle and readback.
    process, root_review = read(args.process_receipt), read(args.root_review)
    run_plan = read(native_output/'run_plan.json')
    require(process['process_exit_observed'] is True and process['process_exit_code'] == 0
            and process.get('process_present_after_wait') is False
            and process['candidate_manifest_sha256'] == MANIFEST_SHA
            and process['command'] == run_plan['native_command']
            and process['PPO_updates'] == process['formal_updates_consumed'] == 0,
            'Actual completed native process required before journal read')
    require(root_review['status'] in ('native_measurement_complete_not_qualified',
            'native_measurement_complete_trajectory_diverged_not_qualified')
            and root_review['native_integrity']['native_terminal_integrity_passed'] is True
            and root_review['qualification'] is None, 'Root complete native readback required')
    spec = importlib.util.spec_from_file_location('targets_bound_readback', CANDIDATE/'run_target_payment_021.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    side, capture, manifest = driver.verify_candidate(MANIFEST_SHA)
    native = native_output/'native'
    summary, report = read(native/'pd_target_payment_summary.json'), read(native/'report.json')
    require(root_review['target_payment_capture'] == summary
            and root_review['artifacts_sha256'][str(native/'report.json')] == sha(native/'report.json')
            and root_review['artifacts_sha256'][str(native/'pd_target_payment_2216.jsonl.gz')]
                == sha(native/'pd_target_payment_2216.jsonl.gz'),
            'Root readback is not bound to these actual native artifacts')
    require(summary['complete'] and summary['hooks_restored'] and summary['policy_steps'] == 600
            and summary['expected_physical_steps'] == 2400 and summary['error'] is None,
            'Complete600/2400 observer summary missing')
    require(report['pd_target_payment_summary_sha256'] == sha(native/'pd_target_payment_summary.json')
            and summary['trace_artifact_sha256'] == sha(native/'pd_target_payment_2216.jsonl.gz'),
            'Native journal/summary integrity binding differs')
    files = [Path(__file__), CANDIDATE/'candidate_manifest.json', CANDIDATE/'run_target_payment_021.py',
        Path(args.process_receipt), Path(args.root_review), native_output/'run_plan.json']
    files += [native/name for name in ('identity.json','report.json','pd_target_payment_summary.json',
        'pd_target_payment_2216.jsonl.gz','mechanical_summary.json','side_override_preclose.json')]
    before = {str(p.resolve()):sha(p) for p in files}
    with gzip.open(native/'pd_target_payment_2216.jsonl.gz','rt') as stream:
        rows = [json.loads(line) for line in stream]
    journal = driver.validate_journal(rows, summary)
    result = summarize(rows)
    after = {str(p.resolve()):sha(p) for p in files}
    require(before == after, 'Actual input changed during bounded analysis')
    output.mkdir()
    receipt = dict(schema='source021-native2216-target-payment-analysis-v1',
        status='completed_native_mean_first_birth_readback_not_qualified',
        candidate_manifest_sha256=MANIFEST_SHA, native_context=manifest['context'],
        root_native_trajectory_reproduced=root_review['same_trajectory_reproduced'],
        journal_clock_validation=journal, analysis=result,
        inputs_before_sha256=before, inputs_after_sha256=after, all_inputs_unchanged=True,
        source_callback_producers_or_policy_calls=0, App_GPU_SSH_run=False, PPO_updates=0,
        qualification=None, app_close_returned=root_review['app_close_returned'],
        limitations=['Only original three motion/hold components were captured per lane; no five-positive-peaks, complete-negative-payment or criticV audit is inferred.',
            'Original decode input/targets are distinguished from delayed actual PD targets; targets are not physical motion or std.',
            'FSM support/height/body-clear fields are native proxy sensors, not a new true-contact gate.',
            'Eight gate marginals and 50Hz payment density are distinct from joint1s dwell and per-episode success.',
            'Missing release/success at12s without done is right-censored; existing thresholds remain unchanged.',
            'GPU aggregate scatter_add differences and gzip hashes are not per-lane physical/reward equality evidence.'])
    with (output/'report.json').open('x') as f:
        json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    hold=result['native_hold'];end=result['final_policy_end']
    lines=['# 2216 native mean target/payment observation','',
        f"Read {result['policy_rows']} first-birth policy intervals and {result['direct_native_substeps']} direct native reference samples. The last gate input is physics{journal['last_reference_input_physical_step']}; the last payment uses physics{journal['last_payment_physical_end_step']}. Each payment density is weighted once by20ms.",'',
        f"Native hold samples: {hold['direct_native_holding_samples']}; joint-pass samples: {hold['joint_pass_samples']}; peak: {hold['native_peak_seconds']:.3f}s; breaks: {hold['native_final_break_count']}. Gate failure counts are nonexclusive and cannot be multiplied into a1s probability.",'',
        f"Final done={end['done']}, motion_released={end['native_motion_released']}, success={end['success']}; right-censored unfinished={journal['right_censored_unfinished']}.",'',
        '| Native interval regime | Policy intervals | Actor-valid | Native hold-paid | Leg payment | Wheel payment | Gyro payment |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label,value in result['actual_three_payments_by_policy_regime'].items():
        sums=[value['actual_native_hold_payment'][name]['sum_payment'] for name in TERMS]
        lines.append(f"| {label} | {value['policy_intervals']} | {value['whole_interval_actor_valid_count']} | {value['native_hold_cost_paid_intervals']} | {sums[0]:.6f} | {sums[1]:.6f} | {sums[2]:.6f} |")
    lines += ['', 'The three-axis values above are original native hold payments. Unmasked script/blend component values and explicit transition bins remain in JSON. PD requests are captured before recovery mixing and are not final shaft efforts. No five-peak reward, full negative ledger, critic value or new model quality conclusion follows this observation.','',
        'Input/report hashes and per-axis decoded/delayed targets, continuous PD errors, native gate breaks and original broad/fine values are retained in report.json.']
    with (output/'summary.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=receipt['status'],report_sha256=sha(output/'report.json'),
        summary_sha256=sha(output/'summary.md'), policy_rows=len(rows), native_peak_seconds=hold['native_peak_seconds'],
        native_breaks=hold['native_final_break_count'],qualification=None)))


if __name__ == '__main__':
    main()
