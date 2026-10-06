#!/usr/bin/env python3
"""Same SIDE1.5/REC3076 mean600 producer; only three observed lanes change."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
SIDE = HERE.parent/'mechanical_side_speed_candidate_021'
SIDE_SHA = '2d04ffc45d6d12e7f3bac610f24f7c3f88e2af2493af24ee4f8cefc611b487c1'
REFERENCE = SIDE/'native_SIDE_001/native'
PYTHON = '/home/yukikaze/isaacsim60-venv/bin/python'
LANES = (881, 1877, 2121)


def require(value, message):
    if not bool(value): raise ValueError(message)


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_candidate(expected_sha):
    side = load(SIDE/'run_side_speed_021.py', 'remaining_frozen_SIDE_driver')
    capture, _ = side.verify_candidate(SIDE_SHA)
    path = HERE/'candidate_manifest.json'
    require(capture.sha(path) == expected_sha, 'Remaining observer manifest SHA differs')
    manifest = capture.read(path)
    require(manifest['schema'] == 'source021-remaining-mechanical-observer-candidate-v1', 'Candidate schema differs')
    for relative, digest in manifest['files_sha256'].items():
        name = Path(relative)
        require(not name.is_absolute() and '..' not in name.parts, 'Candidate path escape')
        require(capture.sha(HERE/name) == digest, 'Candidate bytes changed: '+relative)
    for name, digest in manifest['external_sha256'].items():
        require(Path(name).is_absolute() and capture.sha(name) == digest, 'Bound actual evidence changed: '+name)
    return side, capture, manifest


def add_native_reference_telemetry(mechanical):
    """Identical SIDE reference telemetry for the new observation selection."""
    require(tuple(mechanical.LANES) == LANES and mechanical.LEVELS == (2, 2, 1)
        and mechanical.TRACE_STEPS == 2400 and mechanical.GUARD_M == .003,
        'Private observer selection/clock/guard differs')
    parent = mechanical.MechanicalCapture
    class WithNativeReferences(parent):
        def control(self, target):
            result = super().control(target)
            fsm = self.env.recovery.fsm
            result['side_reference_audit'] = [mechanical.plain(dict(
                phase=int(fsm.phase[lane]), phase_ticks=int(fsm.phase_ticks[lane]),
                side_start_rad=fsm.side_start[lane], side_direction=fsm.side_direction[lane],
                axis_signs=fsm.axis_signs, native_config_side_speed_rad_s=fsm.config['side_speed_rad_s'],
                source='native FSM state at the original effort setter; no update/reconstruction'))
                for lane in LANES]
            return result
    mechanical.MechanicalCapture = WithNativeReferences


def decoded_rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream: yield json.loads(line)


def compare_native_outputs(reference, native, compare_bits):
    """No tolerance and no filtering of birth, event, reward or payment values."""
    counts = {}
    for name in ('first_birth_rows.jsonl.gz', 'events.jsonl.gz', 'mechanical_global_peaks.jsonl.gz'):
        a, b = list(decoded_rows(reference/name)), list(decoded_rows(native/name))
        mismatch = compare_bits(a, b)
        require(mismatch is None, 'Native decoded output differs: '+name+' '+str(mismatch))
        counts[name] = len(a)
    require(counts['first_birth_rows.jsonl.gz'] == 4096, 'Full4096 firstbirth population not present')
    for name in ('identity.json', 'payment_summary.json', 'teaching_summary.json',
                 'side_override_applied.json', 'side_override_preclose.json'):
        a, b = json.loads((reference/name).read_text()), json.loads((native/name).read_text())
        mismatch = compare_bits(a, b)
        require(mismatch is None, 'Native identity/payment/teaching/override differs: '+name+' '+str(mismatch))
    a, b = json.loads((reference/'report.json').read_text()), json.loads((native/'report.json').read_text())
    ignored = ('started_at_utc', 'observer_included_wall_seconds', 'mechanical_summary_sha256')
    for value in (a, b):
        for key in ignored: value.pop(key, None)
    mismatch = compare_bits(a, b)
    require(mismatch is None, 'Native report differs outside observer/time receipt: '+str(mismatch))
    return dict(status='original_SIDE001_all4096_birth_event_payment_native_values_exact',
        row_counts=counts, report_exclusions=list(ignored), tolerance_used=False,
        gzip_header_bytes_are_not_semantic_evidence=True)


def review(output, candidate_sha, process_receipt):
    side, capture, manifest = verify_candidate(candidate_sha)
    helper = load(SIDE/'side_speed_capture.py', 'remaining_frozen_SIDE_helper_review')
    pair = capture.load(capture.ORCHESTRATOR, 'remaining_original_pair_review')
    process = capture.read(process_receipt); plan = capture.read(output/'run_plan.json')
    require(process['process_exit_observed'] and process['process_exit_code'] == 0
        and process.get('process_present_after_wait') is False, 'Diagnostic exit not externally observed')
    require(process['candidate_manifest_sha256'] == candidate_sha
        and process['PPO_updates'] == process['formal_updates_consumed'] == 0
        and process['command'] == plan['native_command'], 'Process argv/zeroPPO/manifest differs')
    native = output/'native'
    integrity = pair.diagnostic_review(pair.bound_driver(), native)
    reproduction = compare_native_outputs(REFERENCE, native, helper.compare_bits)
    summary = capture.read(native/'mechanical_summary.json'); layout = capture.read(native/'mechanical_layout.json')
    require(summary['complete'] and summary['policy_steps'] == 600 and summary['physics_steps'] == 2400
        and summary['trace_cutoff_physics_steps'] == 2400 and summary['hooks_restored'], 'Capture incomplete')
    require(layout['lanes'] == list(LANES) and layout['profiles'] == manifest['selection']['profiles'], 'Observed lane binding differs')
    require(not summary['contact_count_at_capacity_indicators'], 'Contact capacity indicator; preserve/reject incomplete contact scope')
    for lane, policy_tick, level, profile in zip(LANES, (222, 104, 251), (2,2,1), (8,18,19)):
        key = str(lane)
        require(summary['lane_row_counts'][key] == policy_tick*4+1, 'Selected firstbirth trace incomplete: '+key)
        terminal = summary['native_first_terminals'][key]
        require(terminal['policy_tick'] == policy_tick and terminal['reasons']['closure_gap']
            and terminal['fields']['recovery_handover_count'] == 0
            and not terminal['fields']['recovery_pure_rl']
            and terminal['fields']['recovery_actual_level_id'] == level
            and terminal['fields']['recovery_actual_profile_id'] == profile,
            'Actual firstbirth script/closure terminal or level/profile differs: '+key)
        require(summary['pre_autoreset_snapshots'][key]['state']['birth_id'] == 1,
            'Selected preautoreset birth differs: '+key)
    trace = {lane:[] for lane in LANES}
    for row in decoded_rows(native/'mechanical_5ms.jsonl.gz'):
        state = row['state']; lane = state['lane']
        require(lane in trace and state['birth_id'] == 1, 'Unknown selected lane or later birth in trace')
        trace[lane].append(row['physical_step'])
    require(all(trace[lane] == list(range(summary['lane_row_counts'][str(lane)])) for lane in LANES),
        'Selected initial/substep trace has missing or duplicate physical steps')
    require(not any((native/name).exists() for name in ('side_observer_failure.json', 'mechanical_capture_failure.json')),
        'Native observer failure sidecar exists')
    paths = [p for p in native.iterdir() if p.is_file()] + [Path(process_receipt), output/'run_plan.json', HERE/'candidate_manifest.json']
    return dict(schema='source021-remaining-mechanical-native-readback-v1',
        status='same_SIDE_native_reproduced_measurement_complete_not_qualified', native_integrity=integrity,
        reproduction=reproduction, selected_capture=summary,
        inputs_sha256={str(p): capture.sha(p) for p in paths},
        original_control_source_RNG_births_clocks_guard_preserved=True, PPO_updates=0,
        formal_updates_consumed=0, qualification=None,
        limitations=['New contacts/state locate a failure but do not prove causal force decomposition.',
            'Existing views identify terrain contacts; self-contact identity/overflow flags remain unavailable.',
            'No PPO learning, new qualification or formal deployment follows this observation.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu', 'native', 'review'), default='cpu')
    parser.add_argument('--candidate-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--process-receipt', type=Path)
    args = parser.parse_args(argv); output = args.output.resolve()
    require(HERE in output.parents, 'New candidate-only output required')
    side, capture, manifest = verify_candidate(args.candidate_sha)
    if args.mode == 'review':
        require(args.process_receipt is not None and output.is_dir(), 'Actual output/process receipt required')
        try: capture.write(output/'remaining_capture_review.json', review(output, args.candidate_sha, args.process_receipt))
        except BaseException:
            capture.write(output/'remaining_capture_rejection.json', dict(status='preserved_actual_measurement_rejected',
                error=traceback.format_exc(), PPO_updates=0, qualification=None)); return 2
        return 0
    require(args.process_receipt is None and not output.exists(), 'New output required')
    pair = capture.load(capture.ORCHESTRATOR, 'remaining_original_pair')
    plan, admission = pair.load_plan(capture.PAIR)
    helper = load(SIDE/'side_speed_capture.py', 'remaining_frozen_SIDE_override')
    output.mkdir(parents=True)
    resources = None if args.mode == 'cpu' else pair.local_GPU_empty()
    source = str(Path(admission['source_root'])/'src/wheeled_tasks/chassis/recovery.py')
    command = [PYTHON, '-B', str(HERE/'run_remaining_capture_021.py'), '--mode', 'native',
        '--candidate-sha', args.candidate_sha, '--output', str(output)]
    capture.write(output/'run_plan.json', dict(schema='source021-remaining-mechanical-run-plan-v1', mode=args.mode,
        candidate_manifest_sha256=args.candidate_sha, inherited_SIDE_manifest_sha256=SIDE_SHA,
        inherited_admission_sha256=plan['admission_sha256'], source_commit=plan['source_commit'],
        checkpoint_sha256=plan['checkpoint_sha256'], source_recovery_sha256=capture.sha(source),
        actual_SIDE001_report_sha256=capture.sha(REFERENCE/'report.json'), native_command=command,
        resource_check=resources, runtime_requested=dict(lanes=4096, policy_steps=600, physics_steps=2400,
            selected_lanes=list(LANES), policy_dt=.02, physics_dt=.005, solver=[64,32], guard_m=.003, arm='mean'),
        sole_control_override={'RecoveryFSM.config.side_speed_rad_s': {'from':6., 'to':1.5}},
        sole_difference_from_actual_SIDE001='measurement-selected-lanes/levels/profiles/three-lane-telemetry-slicing',
        PPO_updates=0, formal_updates_consumed=0, qualification=None))
    driver = pair.bound_driver()
    if args.mode == 'native':
        mechanical = capture.load(HERE/'mechanical_remaining_capture.py', 'remaining_private_capture')
        add_native_reference_telemetry(mechanical)
        original_factory = driver.make_observer
        def factory(base, producer, composition, native_args, identity, report, binding, writer):
            parent = original_factory(base, producer, composition, native_args, identity, report, binding, writer)
            return side.make_side_observer(parent, capture, mechanical, helper, writer,
                admission['source_root'], capture.sha(source))
        driver.make_observer = factory
    original_args = ['--admission', str(capture.PAIR/'admission.json'), '--admission-sha', plan['admission_sha256'],
        '--arm', 'mean', '--output', str(output/'native')]
    if args.mode == 'cpu': original_args.append('--preflight-only')
    code = driver.main(original_args)
    verify_candidate(args.candidate_sha)
    if code != 0: return code
    if args.mode == 'cpu':
        report = capture.read(output/'native/report.json')
        require(report['status'] == 'CPU_preflight_passed_scene_not_run' and not report['native_scene_run'], 'CPU preflight failed')
        capture.write(output/'CPU_admission.json', dict(status='CPU_authenticated_observer_selection_not_native_tested',
            candidate_manifest_sha256=args.candidate_sha, source_commit=plan['source_commit'],
            original_identity_sha256=capture.sha(output/'native/identity.json'), native_scene_run=False,
            runtime_override_applied=False, PPO_updates=0, qualification=None))
    return 0


if __name__ == '__main__': raise SystemExit(main())
