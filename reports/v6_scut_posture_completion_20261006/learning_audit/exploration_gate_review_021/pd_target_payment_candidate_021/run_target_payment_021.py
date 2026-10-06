#!/usr/bin/env python3
"""Artifact-only original REC3076 mean/SIDE1.5 target and hold-payment observation."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
SIDE = HERE.parent/'mechanical_side_speed_candidate_021'
SIDE_SHA = '2d04ffc45d6d12e7f3bac610f24f7c3f88e2af2493af24ee4f8cefc611b487c1'
BASELINE = SIDE/'native_SIDE_001/native'
PYTHON = '/home/yukikaze/isaacsim60-venv/bin/python'


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_candidate(expected_sha):
    side = load(SIDE/'run_side_speed_021.py', 'target_frozen_SIDE_driver')
    capture, _ = side.verify_candidate(SIDE_SHA)
    path = HERE/'candidate_manifest.json'
    require(capture.sha(path) == expected_sha, 'Target/payment candidate manifest differs')
    manifest = capture.read(path)
    require(manifest['schema'] == 'source021-native2216-target-payment-candidate-v1', 'Candidate schema differs')
    for name, digest in manifest['files_sha256'].items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and capture.sha(HERE/p) == digest,
                'Candidate path/bytes differ: '+name)
    for name, digest in manifest['external_sha256'].items():
        require(Path(name).is_absolute() and capture.sha(name) == digest, 'Bound input changed: '+name)
    return side, capture, manifest


def authenticate_original_callbacks(env, parent, source_root, capture):
    """Resolve only classes/modules already constructed by the original native parent."""
    root = Path(source_root)/'src/wheeled_tasks/chassis'
    for owner, module_name, class_name, filename, methods in (
            (env.v5, 'wheeled_tasks.chassis.v5_control', 'V5Control', 'v5_control.py', ('decode', 'motor_efforts')),
            (env.signal_delay, 'wheeled_tasks.chassis.signal_delay', 'PhysicalSignalDelay', 'signal_delay.py', ('targets',)),
            (env.recovery.fsm, 'wheeled_tasks.chassis.recovery', 'RecoveryFSM', 'recovery.py', ())):
        module = sys.modules.get(module_name)
        require(module is not None and type(owner).__module__ == module_name
                and getattr(module, class_name, None) is type(owner), 'Constructed native class differs: '+class_name)
        require(Path(inspect.getsourcefile(type(owner))).resolve() == root/filename,
                'Native source path differs: '+class_name)
        for name in methods:
            callback = getattr(owner, name)
            require(callback.__self__ is owner and callback.__func__ is getattr(type(owner), name),
                    'Native callback owner/implementation differs: '+name)
    fsm = env.recovery.fsm
    require(parent.original_update.__self__ is fsm
            and parent.original_update.__func__ is type(fsm).update,
            'Original55 native FSM callback binding differs')
    callback = fsm.update
    closure = dict(zip(callback.__code__.co_freevars, (cell.cell_contents for cell in callback.__closure__)))
    original55 = HERE.parent.parent/'probe_prefix_mean_stochastic_019.py'
    require(callback.__code__.co_name == 'update'
            and callback.__code__.co_qualname == 'NativeObserver.__init__.<locals>.update'
            and Path(inspect.getsourcefile(callback)).resolve() == original55
            and capture.sha(original55) == '55f4511284ae09010ba9d79af2a747b70b1f9da3df6b4f713323340b64648b9d'
            and closure.get('self') is parent
            and closure.get('r') is env.recovery,
            'Original55 FSM observer chain differs')
    rewards = sys.modules.get('wheeled_tasks.chassis.rewards')
    require(rewards is not None and Path(rewards.__file__).resolve() == root/'rewards.py'
            and parent.cost_prior.__module__ == rewards.__name__
            and Path(inspect.getsourcefile(parent.cost_prior)).resolve() == root/'rewards.py',
            'Constructed native reward producer differs')
    # The frozen source inventory/pair admission was authenticated transitively;
    # keep actual source hashes in the already written run plan as well.
    require(capture.sha(root/'recovery.py') == '71ee882bd6d660fd5d7b82b0075e12ad1fee71fba196884dde1fd17692797d42',
            'Frozen source021 FSM changed')
    return rewards


def validate_journal(rows, summary):
    require(len(rows) == summary['first_birth_policy_rows']
            and [row['policy_tick'] for row in rows] == list(range(1, len(rows)+1)), 'First birth journal has gaps')
    gates = {'valid', 'support', 'body_clear', 'tilt', 'gyro', 'height', 'wheel_speed', 'leg_speed'}
    terms = {'leg_velocity', 'wheel_velocity', 'horizontal_angular_velocity'}
    for row in rows:
        n = row['policy_tick']
        require(row['lane'] == 2216 and row['birth_id'] == 1 and row['first_birth_live']
                and row['counts'] == dict(decode=1, delay=4, pd=4, fsm=4, cost=1)
                and row['old_episode']['lesson_id'] == 2 and row['old_episode']['profile_id'] == 21,
                'First birth labels/actual single-call cadence changed')
        decode = row['decode']
        require(decode['input_physical_step'] == 4*(n-1) and decode['policy_dt_s'] == .02
                and len(decode['leg_targets_rad']) == 4 and len(decode['wheel_targets_rad_s']) == 2
                and len(decode['continuous_encoder_input_rad']) == 6, 'Decode dimensions/clock differ')
        require(len(row['substeps']) == 4, 'Four original substeps missing')
        for k, substep in enumerate(row['substeps'], 1):
            physical = 4*(n-1)+k
            require(substep['substep'] == k and substep['reference_and_PD_before_physical_step'] == physical
                    and substep['input_physical_step'] == physical-1
                    and substep['postphysics_step'] == physical and substep['reference_dt_s'] == .005,
                    'PD/reference sensor clock differs')
            delay, pd, fsm = substep['delay'], substep['pd'], substep['fsm']
            require(delay['decoded_legs_rad'] == decode['leg_targets_rad']
                    and delay['decoded_wheels_rad_s'] == decode['wheel_targets_rad_s']
                    and pd['actual_PD_leg_targets_rad'] == delay['delayed_legs_rad']
                    and pd['actual_PD_wheel_targets_rad_s'] == delay['delayed_wheels_rad_s'],
                    'Original decoded/delayed/PD targets conflated')
            require(set(fsm['native_gate_masks']) == gates
                    and all(type(value) is bool for value in fsm['native_gate_masks'].values())
                    and type(fsm['stable_ticks']) is int and type(fsm['stable_peak_ticks']) is int
                    and type(fsm['stable_break_count']) is int, 'Native gate/counter snapshot missing')
            require(all(len(pd[name]) == 6 for name in ('continuous_control_q_rad', 'fresh_motor_dq_rad_s',
                    'original_RL_PD_preclip_request_nm', 'original_RL_PD_clipped_torque_nm', 'native_motor_bounds_nm')),
                    'Original PD argument/output dimensions differ')
        producer, end = row['hold_producer'], row['policy_end']
        require(producer['physical_end_step'] == end['physical_end_step'] == 4*n
                and producer['last_gate_input_physical_step'] == 4*n-1
                and producer['density_dt_s'] == end['density_dt_s'] == .02,
                'Native end-state payment clock differs')
        require(set(end['actual_component_densities']) == set(end['actual_component_payments']) == terms
                and end['actual_component_densities'] == producer['actual_component_densities']
                and end['whole_interval_actor_valid'] == producer['whole_interval_actor_owned']
                and end['old_episode_attribution'] == row['old_episode'], 'Original terminal payment attribution differs')
        require(all(end['actual_component_payments'][name] == .02*end['actual_component_densities'][name]
                    for name in terms), 'Native20ms density/payment units differ')
        require(not end['done'] or n == len(rows), 'First birth reopened after original done')
    require(bool(rows) and summary['right_censored_unfinished'] == (not rows[-1]['policy_end']['done']),
            'First-cohort terminal/right-censoring differs')
    return dict(policy_rows=len(rows), direct_native_gate_samples=4*len(rows),
        last_reference_input_physical_step=4*len(rows)-1, last_payment_physical_end_step=4*len(rows),
        right_censored_unfinished=summary['right_censored_unfinished'], no_recomputed_gate_delay_PD_reward=True)


def review(output, candidate_sha, process_receipt):
    side, capture, manifest = verify_candidate(candidate_sha)
    pair = capture.load(capture.ORCHESTRATOR, 'target_original_pair_review')
    process, plan = capture.read(process_receipt), capture.read(output/'run_plan.json')
    require(process['process_exit_observed'] is True and process['process_exit_code'] == 0
            and process.get('process_present_after_wait') is False, 'Actual exit not externally observed')
    require(process['candidate_manifest_sha256'] == candidate_sha and process['command'] == plan['native_command']
            and process['PPO_updates'] == process['formal_updates_consumed'] == 0, 'Process identity/scope differs')
    native = output/'native'
    integrity = pair.diagnostic_review(pair.bound_driver(), native)
    report, old = capture.read(native/'report.json'), capture.read(BASELINE/'report.json')
    require(capture.read(native/'identity.json') == capture.read(BASELINE/'identity.json'), 'Same ownREC3076 identity differs')
    for key in ('initial_physical_birth_tensor_sha256', 'initial_physical_birth_sha256', 'initial_env_rng_sha256',
                'native_group_population', 'sampling_rng_initial_sha256', 'sampling_rng_final_sha256',
                'first_native_sampling', 'final_native_sampling', 'native_effective_std'):
        require(report[key] == old[key], 'Original physical birth/private sampling/std changed: '+key)
    summary = capture.read(native/'pd_target_payment_summary.json')
    require(summary['complete'] and summary['policy_steps'] == 600 and summary['expected_physical_steps'] == 2400
            and summary['hooks_restored'] and summary['unfinished_interval'] is None and summary['error'] is None
            and summary['original_calls_attempted'] == summary['original_calls_completed']
            == dict(decode=600, delay=2400, pd=2400, fsm=2400, cost=600)
            and report['pd_target_payment_capture_complete'], 'Target/payment capture incomplete')
    mechanical = capture.read(native/'mechanical_summary.json')
    rate = capture.read(native/'side_override_preclose.json')
    require(mechanical['complete'] and mechanical['policy_steps'] == 600
            and mechanical['physics_steps'] == mechanical['trace_cutoff_physics_steps'] == 2400
            and mechanical['hooks_restored'] and not mechanical['contact_count_at_capacity_indicators'],
            'Inherited actual600/2400 physics measurement incomplete')
    require(rate['complete'] and rate['config_restored'] and rate['override_hooks_restored'] and not rate['errors']
            and rate['sole_changed_keys'] == ['side_speed_rad_s']
            and rate['before'] | {'side_speed_rad_s': 1.5} == rate['effective'], 'SIDE1.5/restoration changed')
    require(not any((native/name).exists() for name in ('pd_target_payment_failure.json',
            'side_observer_failure.json', 'mechanical_capture_failure.json', 'teaching_observer_failure.json',
            'payment_observer_failure.json')), 'Native observer failure sidecar')
    import gzip
    def decoded(name):
        with gzip.open(native/name, 'rt') as stream:
            return [json.loads(line) for line in stream]
    rows = decoded('pd_target_payment_2216.jsonl.gz')
    require(summary['trace_artifact_sha256'] == capture.sha(native/'pd_target_payment_2216.jsonl.gz'),
            'Actual trace artifact does not match sealed summary')
    journal = validate_journal(rows, summary)
    # No bitwise equality gate is applied to scatter_add aggregates or gzip
    # hashes. Each inherited payment is reconciled to its same-run actual return.
    helper = load(SIDE/'side_speed_capture.py', 'target_semantic_compare')
    reproduction = {}
    for name in ('first_birth_rows.jsonl.gz', 'events.jsonl.gz', 'mechanical_global_peaks.jsonl.gz'):
        with gzip.open(BASELINE/name, 'rt') as stream:
            baseline = [json.loads(line) for line in stream]
        actual = decoded(name)
        mismatch = helper.compare_bits(baseline, actual)
        reproduction[name] = dict(decoded_rows=len(actual), native_values_bitwise_identical=mismatch is None,
                                  first_mismatch=mismatch)
    # Preserve changed trajectories rather than attributing them to this read-only
    # observer. This diagnostic integrity receipt never grants a causal/quality result.
    reproduced = all(v['native_values_bitwise_identical'] for v in reproduction.values())
    return dict(schema='source021-native2216-target-payment-readback-v1',
        status=('native_measurement_complete_not_qualified' if reproduced
                else 'native_measurement_complete_trajectory_diverged_not_qualified'), native_integrity=integrity,
        original_birth_private_sampling_std_exact=True, target_payment_capture=summary,
        journal_clock_and_attribution=journal,
        baseline_native_value_comparison=reproduction,
        same_trajectory_reproduced=reproduced,
        aggregate_payment_bitwise_equality_required=False, gzip_hash_is_not_native_value_equality=True,
        artifacts_sha256={str(p): capture.sha(p) for p in list(native.iterdir()) if p.is_file()},
        PPO_updates=0, formal_updates_consumed=0, qualification=None, app_close_returned=None,
        limitations=['Per-lane actual returns/targets/masks are distinct from all-lane scatter_add summaries.',
            'Native masks use physics k-1; payments use physics4n end-state and original20ms density weighting.',
            'Changed physical outcomes remain preserved; no new recovery/causal/mechanical qualification.',
            'Existing terrain contact views lack self-contact identity/overflow flag and randomized massCOM/friction draws.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu', 'native', 'review'), default='cpu')
    parser.add_argument('--candidate-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--process-receipt', type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(HERE in output.parents, 'Candidate-only new output required')
    side, capture, manifest = verify_candidate(args.candidate_sha)
    if args.mode == 'review':
        require(args.process_receipt is not None and output.is_dir(), 'Actual output/process receipt required')
        try:
            capture.write(output/'target_payment_review.json', review(output, args.candidate_sha, args.process_receipt))
        except BaseException:
            capture.write(output/'target_payment_rejection.json', dict(status='preserved_actual_capture_rejected',
                error=traceback.format_exc(), qualification=None, PPO_updates=0))
            return 2
        return 0
    require(args.process_receipt is None and not output.exists(), 'New output required')
    pair = capture.load(capture.ORCHESTRATOR, 'target_original_pair')
    plan, admission = pair.load_plan(capture.PAIR)
    source_root = admission['source_root']
    root = Path(source_root)/'src/wheeled_tasks/chassis'
    output.mkdir(parents=True)
    resources = None if args.mode == 'cpu' else pair.local_GPU_empty()
    command = [PYTHON, '-B', str(HERE/'run_target_payment_021.py'), '--mode', 'native',
               '--candidate-sha', args.candidate_sha, '--output', str(output)]
    capture.write(output/'run_plan.json', dict(schema='source021-native2216-target-payment-run-plan-v1',
        mode=args.mode, candidate_manifest_sha256=args.candidate_sha, inherited_SIDE_manifest_sha256=SIDE_SHA,
        inherited_admission_sha256=plan['admission_sha256'], source_commit=plan['source_commit'],
        checkpoint_sha256=plan['checkpoint_sha256'], resource_check=resources, native_command=command,
        source_callbacks_sha256={name: capture.sha(root/name) for name in
            ('env.py', 'v5_control.py', 'signal_delay.py', 'recovery.py', 'recovery_training.py', 'rewards.py')},
        inherited_control_override={'side_speed_rad_s': 1.5}, new_control_override=None,
        runtime_requested=dict(lanes=4096, policy_steps=600, physics_steps=2400, policy_dt=.02,
            physics_dt=.005, solver=[64, 32], guard_m=.003, arm='mean', target_payment_lane=2216, birth_id=1),
        no_policy_readback_or_resampling=True, no_second_decode_delay_FSM_reward=True,
        PPO_updates=0, formal_updates_consumed=0, qualification=None))
    driver = pair.bound_driver()
    if args.mode == 'native':
        mechanical = capture.load(side.CAPTURE/'mechanical_capture.py', 'target_private_capture003')
        side_helper = load(SIDE/'side_speed_capture.py', 'target_SIDE_override')
        side_helper.expand_capture_module(mechanical)
        target_helper = load(HERE/'target_payment_capture.py', 'target_payment_observer')
        original_factory = driver.make_observer
        def factory(base, producer, composition, native_args, identity, report, binding, writer):
            parent = original_factory(base, producer, composition, native_args, identity, report, binding, writer)
            inherited = side.make_side_observer(parent, capture, mechanical, side_helper, writer,
                source_root, capture.sha(root/'recovery.py'))
            return target_helper.make_observer(inherited,
                lambda env, observer: authenticate_original_callbacks(env, observer, source_root, capture),
                mechanical.HookSet, writer, capture.sha)
        driver.make_observer = factory
    inherited = ['--admission', str(capture.PAIR/'admission.json'), '--admission-sha', plan['admission_sha256'],
                 '--arm', 'mean', '--output', str(output/'native')]
    if args.mode == 'cpu':
        inherited.append('--preflight-only')
    code = driver.main(inherited)
    verify_candidate(args.candidate_sha)
    if code:
        return code
    if args.mode == 'cpu':
        report = capture.read(output/'native/report.json')
        require(report['status'] == 'CPU_preflight_passed_scene_not_run' and not report['native_scene_run'],
                'Original App-free preflight failed')
        capture.write(output/'CPU_admission.json', dict(status='CPU_original_identity_authenticated_observer_not_native_tested',
            candidate_manifest_sha256=args.candidate_sha, original_identity_sha256=capture.sha(output/'native/identity.json'),
            native_scene_run=False, runtime_observer_applied=False, PPO_updates=0, qualification=None))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
