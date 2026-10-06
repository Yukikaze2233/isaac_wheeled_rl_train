#!/usr/bin/env python3
"""Supplement the completed historical2216 readback; never operate native code."""
from pathlib import Path
import collections
import gzip
import hashlib
import importlib.util
import json
import math

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
OUTPUT = HERE / 'actual_analysis_001'
NATIVE = CANDIDATE / 'native_target_payment_001/native'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    paths = [Path(__file__), HERE/'analyze_2216_targets_payments_001.py',
        CANDIDATE/'candidate_manifest.json', CANDIDATE/'native_target_payment_001.process.json',
        CANDIDATE/'native_target_payment_001/target_payment_review.json', NATIVE/'report.json',
        NATIVE/'identity.json', NATIVE/'pd_target_payment_2216.jsonl.gz',
        OUTPUT/'report.json', OUTPUT/'summary.md']
    before = {str(p): sha(p) for p in paths}
    spec = importlib.util.spec_from_file_location('bound_2216_analysis', paths[1])
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    review, process = read(paths[4]), read(paths[3])
    base = read(OUTPUT/'report.json')
    assert review['same_trajectory_reproduced'] is True
    assert review['original_birth_private_sampling_std_exact'] is True
    assert all(v['native_values_bitwise_identical'] for v in review['baseline_native_value_comparison'].values())
    assert process['process_exit_observed'] is True and process['process_exit_code'] == 0
    assert process['process_present_after_wait'] is False
    assert base['all_inputs_unchanged'] and base['qualification'] is None
    assert all(sha(Path(p)) == h for p, h in base['inputs_after_sha256'].items())
    with gzip.open(NATIVE/'pd_target_payment_2216.jsonl.gz', 'rt') as f:
        rows = [json.loads(line) for line in f]
    pairs = [(r, s) for r in rows for s in r['substeps']]
    hold = [(r, s) for r, s in pairs if s['fsm']['native_holding_rl'] and s['fsm']['release_finished']]
    assert len(rows) == 600 and len(pairs) == 2400 and len(hold) == 1416
    assert all(r['lane'] == 2216 and r['birth_id'] == 1 and r['old_episode']['lesson_id'] == 2
        and r['old_episode']['profile_id'] == 21 for r in rows)
    gates, terms = a.GATES, a.TERMS

    def describe(members):
        refs = [s['reference_and_PD_before_physical_step'] for r, s in members]
        rawrim = [.06*max(abs(x) for x in r['decode']['wheel_targets_rad_s']) for r, s in members]
        delayrim = [.06*max(abs(x) for x in s['pd']['actual_PD_wheel_targets_rad_s']) for r, s in members]
        actualrim = [.06*max(abs(s['pd']['fresh_motor_dq_rad_s'][i]) for i in a.WHEELS) for r, s in members]
        legdelta = [max(abs(x-y) for x, y in zip(s['pd']['actual_PD_leg_targets_rad'], r['decode']['leg_targets_rad'])) for r, s in members]
        wheeldelta = [max(abs(x-y) for x, y in zip(s['pd']['actual_PD_wheel_targets_rad_s'], r['decode']['wheel_targets_rad_s'])) for r, s in members]
        legerror = [max(abs(s['pd']['actual_PD_leg_targets_rad'][i]-s['pd']['continuous_control_q_rad'][j]) for i, j in enumerate(a.LEGS)) for r, s in members]
        return dict(reference_range=[min(refs), max(refs)], reference_samples=len(members),
            input_physical_range=[min(refs)-1, max(refs)-1],
            input_physical_seconds_range=[(min(refs)-1)*.005, (max(refs)-1)*.005],
            raw_decode_peak_wheel_rim_target_m_s=a.stats(rawrim),
            delayed_PD_peak_wheel_rim_target_m_s=a.stats(delayrim),
            actual_fresh_peak_wheel_rim_speed_m_s=a.stats(actualrim),
            raw_decode_rim_target_gt_gate_count=sum(x > .25 for x in rawrim),
            delayed_rim_target_gt_gate_count=sum(x > .25 for x in delayrim),
            raw_delay_max_axis_wheel_difference_rad_s=a.stats(wheeldelta),
            raw_delay_max_axis_leg_difference_rad=a.stats(legdelta),
            max_leg_continuous_PD_error_rad=a.stats(legerror),
            max_leg_fresh_speed_rad_s=a.stats([max(abs(s['pd']['fresh_motor_dq_rad_s'][i]) for i in a.LEGS) for r, s in members]),
            gyro_norm_rad_s=a.stats([math.sqrt(math.fsum(x*x for x in s['fsm']['gyro_rad_s'])) for r, s in members]),
            original_RL_max_leg_PD_request_nm=a.stats([max(abs(s['pd']['original_RL_PD_preclip_request_nm'][i]) for i in a.LEGS) for r, s in members]),
            native_failed_gate_counts_nonexclusive={g: sum(not s['fsm']['native_gate_masks'][g] for r, s in members) for g in gates},
            native_joint_pass_samples=sum(s['fsm']['native_stable_pass'] for r, s in members),
            legs_target_lag_steps_distribution=dict(collections.Counter(s['delay']['target_lags_physics_steps'][0] for r, s in members)),
            wheels_target_lag_steps_distribution=dict(collections.Counter(s['delay']['target_lags_physics_steps'][1] for r, s in members)))

    stages = {k: describe([(r, s) for r, s in pairs if a.regime(s) == k])
        for k in ('passive_release', 'script', 'blend', 'pure_rl')}
    stages['native_hold_only'] = describe(hold)
    breaks = base['analysis']['native_hold']['break_events']
    windows = []
    for lo, hi in ((985, 1200), (1201, 1600), (1601, 2000), (2001, 2400)):
        members = [(r, s) for r, s in hold if lo <= s['reference_and_PD_before_physical_step'] <= hi]
        paid = [r for r in rows if r['hold_producer']['native_hold_cost_mask'] and lo <= r['policy_end']['physical_end_step'] <= hi]
        broken = [b for b in breaks if lo <= b['reference'] <= hi]
        d = describe(members)
        d.update(native_break_events=len(broken),
            native_break_failed_gate_counts_nonexclusive={g: sum(b['increment'] for b in broken if g in b['failed_gates']) for g in gates},
            hold_paid_policy_intervals=len(paid), hold_paid_seconds=len(paid)*.02,
            payment_end_physical_range=[paid[0]['policy_end']['physical_end_step'], paid[-1]['policy_end']['physical_end_step']],
            actual_native_dt_density_payments={t: math.fsum(r['policy_end']['actual_component_payments'][t] for r in paid) for t in terms})
        windows.append(d)
    paid = [r for r in rows if r['hold_producer']['native_hold_cost_mask']]
    assert len(paid) == 354 and all(r['policy_end']['whole_interval_actor_valid'] for r in paid)
    assert not any(r['policy_end']['done'] or r['policy_end']['native_motion_released'] or r['policy_end']['success'] for r in rows)
    payments = {t: math.fsum(r['policy_end']['actual_component_payments'][t] for r in paid) for t in terms}
    assert all(payments[t] == base['analysis']['actual_three_payments_by_policy_regime']['pure_rl']['actual_native_hold_payment'][t]['sum_payment'] for t in terms)
    residual = {t: math.fsum(w['actual_native_dt_density_payments'][t] for w in windows)-payments[t] for t in terms}
    refs = {b['reference'] for b in breaks}
    break_inputs = {'all': describe([(r, s) for r, s in hold if s['reference_and_PD_before_physical_step'] in refs])}
    for gate in ('wheel_speed', 'gyro', 'leg_speed'):
        refs = {b['reference'] for b in breaks if gate in b['failed_gates']}
        break_inputs[gate] = describe([(r, s) for r, s in hold if s['reference_and_PD_before_physical_step'] in refs])
    after = {str(p): sha(p) for p in paths}
    assert before == after
    result = dict(schema='independent-native2216-target-payment-readback-v1',
        status='complete_historical_mean_mechanism_observation_not_qualified',
        source_commit='d0dce6181df83b03fe166abcbadef3c81de2e57e', parent_actor=3076,
        parent_model_sha256='286b9e26cdd7fe8726f9deb1ce5f24e005d5d918d3abc2b122fdb85b6b671c31',
        configuration_override='SIDE speed6 to1.5 only; no CAPTURE gyro candidate',
        identity_birth_private_RNG_saved_std_exact=True, same_trajectory_reproduced=True,
        baseline_projection=review['baseline_native_value_comparison'],
        native_policy_steps=600, native_physical_steps=2400, native_process_exit_observed=True,
        native_process_exit_code=0, app_close_returned=None, PPO_updates=0,
        formal_updates_consumed=0, qualification=None,
        native_control_regime_segments=stages, native_hold_time_windows=windows,
        native_breaks=base['analysis']['native_hold'], native_break_input_descriptions=break_inputs,
        actor_owned_native_hold_payer_policy_range=[247, 600], actor_owned_hold_paid_policy_intervals=354,
        actor_owned_hold_paid_seconds=7.08, native_original_component_density_dt_payments=payments,
        segment_payment_sum_residual=residual, right_censored_at12_seconds=True,
        clock_and_units=dict(native_reference_k='uses physical state k-1; physics k occurs later',
            gate_sample_dt_s=.005, payment_end='density uses physical4n; last gate input4n-1', payment_dt_s=.02,
            component_payment_precision='Native float32 component density copied then weighted by20ms in Python. This is not a second observation of float32 total-reward summands; inherited original payment/event reconciliation is authoritative.',
            control_axes=list(a.AXES), substep_regime='After original FSM. Entering-RL reference984 is not holding/whole-actor eligible yet.',
            script_blend_PD='Original primary RL requests before recovery mixing; not applied shaft torque.',
            gate_counts='Nonexclusive direct masks; joint samples counted separately. No marginal product.',
            target_vs_gate='Target threshold comparison is descriptive. Native wheel predicate uses actual dq.'),
        minimal_next_step_proposal=dict(status='proposal_only_no_source_or_native_change',
            diagnosis='Mean already requests oscillating wheel targets during hold and actual wheel/gyro/leg gates repeatedly fail while costs are active. This2216 mean result cannot be attributed solely to stochastic exploration. Delay, policy, physical response and reward weighting remain competing contributors.',
            bounded_identification='Before additional costs or std changes, compare same sealed mean actor with one engineering target-delay variant only during actor-owned hold. Preserve observation delay, source, initial birth/RNG, PD, thresholds, costs and SIDE1.5. Later state/RNG can diverge legitimately; only initial/pre-intervention prefix is exact.',
            conditional_repair='If target-lag contrast materially improves joint dwell, author a bounded hold-target delay curriculum with the existing mechanism. Do not deploy a bypass/clamp from this observation. Otherwise inspect learned settling direction/gyro response with existing active terms.',
            observable_DoD=['Authenticated same source/seal/initial private stream and pre-intervention prefix.',
                'Actual600/2400, cleanup/exit and zeroPPO independently sealed.',
                'Direct200tick/1s gate, release/followup and censoring reported separately.',
                'Original ownership/payments retained; no duplicate costs or lower3mm/1s.'],
            reward_weight_tuning='Not justified here: criticV, advantages, gradients, five positive peaks and full negative ledger were not captured.'),
        limitations=['Historical REC3076 mean; no current live model or quality qualification.',
            'Native support/body/height masks are proxies, not reconstructed actual contact gates.',
            'No criticV, five positive peaks or full negative payment ledger captured.',
            'Gzip headers and global scatter_add tails are not per-lane trajectory differences.'],
        input_sha256_before=before, input_sha256_after=after, all_inputs_unchanged=True,
        new_App_GPU_SSH_or_producer_calls=0)
    with (OUTPUT/'independent_actual_review_001.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    lines = ['# 2216 mean target/payment readback', '',
        'Actual exit0 and600/2400, source021/ownREC3076/SIDE1.5. Birth/private RNG/std match. Decoded4096 births,3756 events and9 global peaks equal SIDE001 bitwise. No PPO or production change; qualification null; App.close return unknown.', '',
        '| Native segment | 5ms samples | Raw wheel target rim mean | Delayed target mean | Fresh wheel mean | Gyro norm mean | Max leg speed mean |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for k in ('script', 'blend', 'pure_rl'):
        d = stages[k]
        vals = [d[x]['mean'] for x in ('raw_decode_peak_wheel_rim_target_m_s', 'delayed_PD_peak_wheel_rim_target_m_s', 'actual_fresh_peak_wheel_rim_speed_m_s', 'gyro_norm_rad_s', 'max_leg_fresh_speed_rad_s')]
        lines.append(f"| {k} | {d['reference_samples']} | {vals[0]:.4f}m/s | {vals[1]:.4f}m/s | {vals[2]:.4f}m/s | {vals[3]:.4f}rad/s | {vals[4]:.4f}rad/s |")
    h = stages['native_hold_only']
    lines += ['', 'Hold inputs physical984–2399,1416 samples: joint pass295; peak16 ticks/80ms versus required200/1s;82 breaks. Nonexclusive failures wheel787/gyro596/legs332/tilt39; at breaks32/58/17/0. Other four masks passed. At12s it is unfinished right-censoring.', '',
        f"During hold, raw wheel target rim mean{h['raw_decode_peak_wheel_rim_target_m_s']['mean']:.4f}m/s; delayed{h['delayed_PD_peak_wheel_rim_target_m_s']['mean']:.4f}; fresh actual{h['actual_fresh_peak_wheel_rim_speed_m_s']['mean']:.4f}. Delayed target exceeds.25 in{h['delayed_rim_target_gt_gate_count']}/1416; this is not the native actual-speed gate. Max-axis wheel decode-to-delay delta mean{h['raw_delay_max_axis_wheel_difference_rad_s']['mean']:.3f}rad/s/max{h['raw_delay_max_axis_wheel_difference_rad_s']['max']:.3f}; leg delta mean{h['raw_delay_max_axis_leg_difference_rad']['mean']:.5f}rad.", '',
        '| Hold input window | Samples | Joint pass | Breaks | Wheel/gyro/leg failures | Leg/wheel/gyro20ms payment |',
        '| --- | ---: | ---: | ---: | --- | --- |']
    for d in windows:
        lo, hi = d['input_physical_seconds_range']
        fail, paid = d['native_failed_gate_counts_nonexclusive'], d['actual_native_dt_density_payments']
        lines.append(f"| {lo:.3f}–{hi:.3f}s | {d['reference_samples']} | {d['native_joint_pass_samples']} | {d['native_break_events']} | {fail['wheel_speed']}/{fail['gyro']}/{fail['leg_speed']} | {paid[terms[0]]:.4f}/{paid[terms[1]]:.4f}/{paid[terms[2]]:.4f} |")
    lines += ['', f"354 actor-owned intervals/7.08s paid original density×.02: leg{payments[terms[0]]:.6f},wheel{payments[terms[1]]:.6f},gyro{payments[terms[2]]:.6f}. Script/blend paid no native hold costs. Payment uses physics4n; last gate uses4n−1. Python dt weighting is diagnostic precision; inherited original total reward/event reconciliation passed.", '',
        'Mean is already unstable, so stochastic exploration alone cannot explain this lane. Requests, delayed targets, physical wheel speed and gyro/leg failures coexist; no unique cause follows. The smallest proposed identification is one hold target-delay contrast with the same historical actor and other thresholds/physics/rewards/std fixed. This is a future0PPO proposal, not an implemented repair. Do not add duplicate taxes or tune weights without critic/advantage evidence; keep1s/3mm.', '',
        'No five-positive-peak, full-negative ledger or criticV conclusions are added. Original inputs and frozen candidate bytes remain unchanged.']
    with (OUTPUT/'independent_actual_review_001.md').open('x') as f:
        f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'], json_sha256=sha(OUTPUT/'independent_actual_review_001.json'),
        md_sha256=sha(OUTPUT/'independent_actual_review_001.md'), native_hold_segment=h,
        window_breaks=[d['native_break_events'] for d in windows], payments=payments)))


if __name__ == '__main__':
    main()
