#!/usr/bin/env python3
"""Read only a completed, root-reviewed SIDE counterfactual and actual003.

Reuses003's mechanical/contact/effort readback, never instantiates physics or
learning. File hashes and native first-birth denominators bound every outcome.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
BASELINE = CANDIDATE.parent/'mechanical_capture_candidate_021/native_capture_003'
CAPTURE_HELPER = BASELINE.parent/'analysis_candidate_001.py'
CAPTURE_HELPER_SHA = '5e8ac734d72e788bfa54d41c56000de4d4185cb725581ed292c402eec2c6c8a4'
SIDE_MANIFEST_SHA = '2d04ffc45d6d12e7f3bac610f24f7c3f88e2af2493af24ee4f8cefc611b487c1'
LANES = (1708, 2216)


def require(value, message):
    if not bool(value): raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def completed_admission(capture_root, process_path, review_sha):
    """Verify exit/review BEFORE opening any captured gzip stream."""
    process = read(process_path)
    require(process['process_exit_observed'] is True and process['process_exit_code'] == 0
        and process['process_present_after_wait'] is False, 'Actual native process has not completed/exited')
    path = capture_root/'side_speed_review.json'
    require(sha(path) == review_sha, 'Root review SHA differs')
    review = read(path)
    require(review['status'] == 'native_counterfactual_integrity_checked_not_functionally_qualified'
        and review['PPO_updates'] == review['formal_updates_consumed'] == 0
        and review['qualification'] is None, 'Root has not accepted native diagnostic integrity')
    for path, digest in review['inputs_sha256'].items():
        require(sha(path) == digest, 'Root-reviewed actual input changed: '+path)
    require(review['inputs_sha256'].get(str(process_path.resolve())) == sha(process_path), 'Actual exit receipt unbound by root review')
    return review


def first_birth_summary(rows):
    require(len(rows) == 4096 and {row['lane'] for row in rows} == set(range(4096))
        and all(row['birth_id'] == 1 for row in rows), 'First-birth4096 denominator differs')
    result = dict(fixed_birth_denominator=4096, partitions=dict(Counter(row['partition'] for row in rows)), by_lesson={})
    for lesson in sorted({row['lesson'] for row in rows}):
        group = [row for row in rows if row['lesson'] == lesson]
        result['by_lesson'][lesson] = dict(fixed_birth_denominator=len(group),
            partitions=dict(Counter(row['partition'] for row in group)),
            ever_handed_over=sum(row['first_handover_reference_tick'] >= 0 for row in group),
            ever_motion_released=sum(row['first_motion_release_reference_tick'] >= 0 for row in group))
    return result


def sum_scalars(value):
    return sum(sum_scalars(item) for item in value) if isinstance(value, list) else float(value)


def body_normal_force(frame, body):
    contact = frame['actual_terrain_contact_readback']
    if contact is None: return 0.
    return sum(sum_scalars(patch['force_scalar_n']) for row in contact['sensor_body_rows_with_observed_force_or_patch']
        if row['body_name'] == body for patch in row['normal_patches'])


def compact_frame(frame):
    if frame is None: return None
    effort = frame['effort_reference_readback']
    return {key: frame[key] for key in ('physical_step', 'simulated_s', 'phase', 'route', 'ownership',
        'pin_argmax', 'pin_constraint', 'pin_max_gap_m', 'actual_terrain_contact_readback')} | dict(
        RR_joint1_rotary_effort_reference=(None if effort is None else effort['rotary_axes'][4]),
        RR_link1_normal_patch_force_sum_n=body_normal_force(frame, 'RR_link1'))


def selected_case(raw, converted, entry, first, layout, lane):
    rows = [row for row in raw if row['state']['lane'] == lane]
    require(len(rows) == len(converted), 'Selected chronology shape differs')
    peak_index = max(range(len(converted)), key=lambda index: converted[index]['pin_max_gap_m'])
    rr_index = layout['full_joint_names'].index('RR_joint1')
    max_dq_index = max(range(len(rows)), key=lambda index: abs(rows[index]['state']['full_dq'][rr_index]))
    max_force_index = max(range(len(converted)), key=lambda index: body_normal_force(converted[index], 'RR_link1'))
    crossing = entry['first_substep_above_original3mm']
    previous = None if crossing is None or crossing['physical_step'] == 0 else converted[crossing['physical_step']-1]
    terminal = entry['native_first_terminal']
    actual = first['recovery_failure_code']
    return dict(lane=lane, profile=entry['actual_original_profile'],
        captured_first_birth_physical_steps=len(rows)-1, selected_first_birth_record=first,
        phase_ownership_transitions=entry['phase_route_ownership_transitions'],
        first_substep_guard=compact_frame(crossing), immediately_previous5ms=compact_frame(previous),
        selected_first_birth_peak=compact_frame(converted[peak_index]),
        maximum_abs_RR_joint1_dq_rad_s=abs(rows[max_dq_index]['state']['full_dq'][rr_index]),
        maximum_dq_physical_step=rows[max_dq_index]['physical_step'],
        maximum_RR_link1_normal_patch_force_sum_n=body_normal_force(converted[max_force_index], 'RR_link1'),
        maximum_contact_force_physical_step=converted[max_force_index]['physical_step'],
        native_first_terminal=terminal,
        native_script_failure_code=actual,
        native_SIDE_timeout_recorded=bool(first['recovery_failed'] and actual == 2),
        ever_handed_over=first['first_handover_reference_tick'] >= 0,
        ever_motion_released=first['first_motion_release_reference_tick'] >= 0,
        success_partition=first['partition'] == 'success',
        right_censored_unfinished=first['partition'] == 'right_censored_unfinished')


def case_readback(root, helper):
    native = root/'native'
    layout, summary, report, teaching = (read(native/name) for name in
        ('mechanical_layout.json', 'mechanical_summary.json', 'report.json', 'teaching_summary.json'))
    require(summary['complete'] and summary['hooks_restored'] and summary['policy_steps'] == 600
        and summary['physics_steps'] == 2400 and layout['solver'] == [64, 32] and layout['guard_m'] == .003,
        'Completed original physical/guard scope differs')
    raw = helper.rows(native/'mechanical_5ms.jsonl.gz')
    peaks = helper.rows(native/'mechanical_global_peaks.jsonl.gz')
    first = helper.rows(native/'first_birth_rows.jsonl.gz')
    helper.finite_tree(raw); helper.finite_tree(peaks); helper.finite_tree(first)
    require(peaks and max(row['gap_m'] for row in peaks) == summary['full_window_max_pin_gap_m'], 'Global substep peak differs')
    independent = first_birth_summary(first)
    for key in ('fixed_birth_denominator', 'partitions', 'by_lesson'):
        require(independent[key] == report['first_birth'][key], 'Independent native birth ledger differs: '+key)
    selected = []
    for lane, origin in zip(LANES, layout['selected_env_origins_world_m']):
        entry, converted = helper.lane_readback(raw, layout, summary, lane, origin)
        row = next(row for row in first if row['lane'] == lane)
        selected.append(selected_case(raw, converted, entry, row, layout, lane))
    return dict(selected=selected,
        global_actual_substep_peak={key: max(peaks, key=lambda row: row['gap_m'])[key] for key in
            ('physical_step', 'simulated_s', 'lane', 'pin', 'birth_id', 'gap_m')},
        global_policy_end_peak_m=teaching['native_formal_probe_structural_audit']['max_pin_gap_m'],
        all_birth_guard_terminal_lanes=teaching['native_formal_probe_structural_audit']['counts']['closure_guard_terminal_lanes'],
        independently_reconciled_all4096_first_births=independent,
        native_teaching_summary=teaching,
        observation_only=True, PPO_updates=0, qualification=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--exit-receipt', type=Path, required=True)
    parser.add_argument('--root-review-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    actual, output = args.capture_root.resolve(), args.output.resolve()
    require(CANDIDATE in actual.parents and HERE in output.parents and not output.exists(), 'New isolated analysis output required')
    driver = load(CANDIDATE/'run_side_speed_021.py', 'SIDE_completed_driver')
    driver.verify_candidate(SIDE_MANIFEST_SHA)
    root_review = completed_admission(actual, args.exit_receipt, args.root_review_sha)
    require(sha(CAPTURE_HELPER) == CAPTURE_HELPER_SHA, 'Reusable003 analysis helper changed')
    helper = load(CAPTURE_HELPER, 'SIDE_readonly003_analysis')
    paths = [Path(__file__).resolve(), CAPTURE_HELPER, actual/'side_speed_review.json', args.exit_receipt.resolve()]
    names = ['mechanical_layout.json', 'mechanical_summary.json', 'mechanical_5ms.jsonl.gz', 'mechanical_global_peaks.jsonl.gz',
        'first_birth_rows.jsonl.gz', 'events.jsonl.gz', 'report.json', 'teaching_summary.json', 'payment_summary.json']
    paths += [root/'native'/name for root in (BASELINE, actual) for name in names]
    inputs = {str(path): sha(path) for path in paths}
    base, trial = case_readback(BASELINE, helper), case_readback(actual, helper)
    for path, digest in inputs.items(): require(sha(path) == digest, 'Completed input changed while reading')
    output.mkdir(parents=True)
    result = dict(schema='source021-SIDE-actual-single-variable-outcome-comparison-v1',
        actual_inputs_sha256=inputs, root_integrity_review_sha256=args.root_review_sha,
        preSIDE_bitwise=root_review['pre_SIDE'], native_coupled_reference=root_review['coupled_reference'],
        baseline6_rad_s=base, counterfactual1p5_rad_s=trial,
        PPO_updates=0, formal_updates_consumed=0, qualification=None,
        limitations=['Native600/12s unfinished first births remain right-censored, not successes or failures',
            'Global substep peaks and policy-end peaks are distinct',
            'All-birth guard events and fixed4096 first-birth outcomes have different denominators',
            'Selected shaft speeds and normal contact-force maxima are observed, not causal attribution',
            'No illegal contact/sole solver cause/functional qualification is inferred',
            'Actual self-contact pair identity, overflow flag and randomized massCOMfriction draw remain unavailable',
            'Old unlocalized stochastic730mm is unrelated to this localized mean counterfactual'])
    helper.write(output/'comparison.json', result)
    for path, digest in inputs.items(): require(sha(path) == digest, 'Completed input changed during publication')
    print(json.dumps(dict(output=str(output/'comparison.json'), sha256=sha(output/'comparison.json'), qualification=None)))
    return 0


if __name__ == '__main__': raise SystemExit(main())
