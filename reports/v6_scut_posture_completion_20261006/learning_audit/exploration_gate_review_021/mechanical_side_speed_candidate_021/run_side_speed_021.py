#!/usr/bin/env python3
"""0 PPO, same REC3076 mean native600 loop; only SIDE6->1.5 rad/s."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
CAPTURE = HERE.parent/'mechanical_capture_candidate_021'
CAPTURE_SHA = '34092556be7830a1f075b7d618c3010516fee94bfbcdf3d8b31444efd0b4abbf'
BASELINE = CAPTURE/'native_capture_003/native'
PYTHON = '/home/yukikaze/isaacsim60-venv/bin/python'


def require(value, message):
    if not bool(value): raise ValueError(message)


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def frozen_capture():
    # No Isaac/env import; this module's factory is the admitted003 lifecycle.
    module = load(CAPTURE/'run_capture_021.py', 'side_frozen_capture003')
    module.verify_candidate(CAPTURE_SHA)
    return module


def verify_candidate(expected_sha):
    capture = frozen_capture()
    path = HERE/'candidate_manifest.json'
    require(capture.sha(path) == expected_sha, 'SIDE candidate manifest SHA differs')
    manifest = capture.read(path)
    require(manifest['schema'] == 'source021-SIDE-speed-artifact-candidate-v1', 'SIDE candidate schema differs')
    for relative, digest in manifest['files_sha256'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Candidate path escape')
        require(capture.sha(HERE/path) == digest, 'Candidate bytes changed: '+relative)
    for path, digest in manifest['external_sha256'].items():
        require(Path(path).is_absolute() and capture.sha(path) == digest, 'Bound evidence changed: '+path)
    return capture, manifest


def authenticate_fsm(fsm, source_root, source_sha, sha, recovery=None):
    """Only the already constructed, exact frozen native RecoveryFSM owner."""
    name = 'wheeled_tasks.chassis.recovery'
    module = sys.modules.get(name)
    require(type(fsm).__module__ == name and module is not None
        and getattr(module, 'RecoveryFSM', None) is type(fsm), 'Native FSM was not constructed/registered')
    source = Path(inspect.getsourcefile(type(fsm))).resolve()
    expected = Path(source_root)/'src/wheeled_tasks/chassis/recovery.py'
    require(source == expected and sha(source) == source_sha, 'Native FSM source identity differs')
    require(fsm.count == 4096 and fsm.DEFAULTS['side_speed_rad_s'] == 6., 'Native FSM baseline/population differs')
    require(recovery is not None, 'Constructed recovery settings binding absent')
    expected_config = fsm.DEFAULTS | {key: value for key, value in recovery.settings.items() if key in fsm.DEFAULTS}
    expected_config['dt'] = recovery.reference_dt
    require(fsm.config == expected_config, 'Instance FSM config differs from its native settings construction')


def make_side_observer(parent, capture, mechanical, helper, writer, source_root, source_sha):
    base = capture.make_capture_observer(parent, None, mechanical, writer)
    class WithSideSpeed(base):
        def __init__(self, env, output):
            self.side_override = None
            super().__init__(env, output)
            try:
                self.side_override = helper.SideOverride(env, mechanical.HookSet(), writer, output,
                    lambda fsm: authenticate_fsm(fsm, source_root, source_sha, capture.sha, env.recovery))
                self.side_override.apply()
            except BaseException:
                self._before_close(); raise

        def _before_close(self):
            try:
                if self.side_override is not None:
                    errors = self.side_override.cleanup()
                    require(not errors, 'SIDE override cleanup ownership differs: '+str(errors))
            finally: super()._before_close()

        def finish(self, output):
            try:
                errors = self.side_override.finish()
                require(not errors, 'SIDE override preclose restoration rejected: '+str(errors))
                # Remove outer step hook before003 restores its inner callbacks.
                result = super().finish(output)
            except BaseException:
                writer(output/'side_observer_failure.json', dict(status='SIDE_finish_rejected_preclose',
                    error=traceback.format_exc(), qualification=None, PPO_updates=0))
                self._before_close(); raise
            result.update(isolated_SIDE_override_complete=True,
                isolated_SIDE_override_summary_sha256=capture.sha(output/'side_override_preclose.json'),
                isolated_SIDE_override_runtime_value_rad_s=1.5,
                original_runtime_preflight_precedes_explicit_diagnostic_override=True,
                observer_trace_first_birth_physics_steps=2400)
            return result
    return WithSideSpeed


def review(output, candidate_sha, process_receipt):
    capture, manifest = verify_candidate(candidate_sha)
    helper = load(HERE/'side_speed_capture.py', 'side_readback_helper')
    pair = capture.load(capture.ORCHESTRATOR, 'side_original_pair_readback')
    process = capture.read(process_receipt)
    require(process['process_exit_observed'] is True and process['process_exit_code'] == 0
        and process.get('process_present_after_wait') is False, 'Actual exclusive diagnostic exit not observed')
    require(process['candidate_manifest_sha256'] == candidate_sha
        and process['PPO_updates'] == process['formal_updates_consumed'] == 0, 'Actual process scope/manifest differs')
    plan = capture.read(output/'run_plan.json')
    require(process['command'] == plan['native_command'], 'Process argv differs from requested diagnostic')
    native = output/'native'
    integrity = pair.diagnostic_review(pair.bound_driver(), native)
    report, baseline = capture.read(native/'report.json'), capture.read(BASELINE/'report.json')
    require(capture.read(native/'identity.json') == capture.read(BASELINE/'identity.json'), 'Same ownREC3076 full source/seal identity differs')
    for key in ('initial_physical_birth_tensor_sha256', 'initial_physical_birth_sha256', 'initial_env_rng_sha256',
            'native_group_population', 'sampling_rng_initial_sha256', 'sampling_rng_final_sha256',
            'first_native_sampling', 'final_native_sampling', 'native_effective_std'):
        require(report[key] == baseline[key], 'Same initial birth/private sampling differs: '+key)
    summary = capture.read(native/'mechanical_summary.json')
    override = capture.read(native/'side_override_preclose.json')
    require(summary['complete'] and summary['policy_steps'] == 600 and summary['physics_steps'] == 2400
        and summary['trace_cutoff_physics_steps'] == 2400 and summary['hooks_restored'], 'Mechanical full-window capture incomplete')
    require(not summary['contact_count_at_capacity_indicators'], 'Contact buffer capacity indicator; contact scope incomplete')
    require(override['complete'] and override['config_restored'] and override['override_hooks_restored']
        and override['policy_intervals_completed'] == 600 and not override['errors']
        and override['sole_changed_keys'] == ['side_speed_rad_s']
        and override['before'] | {'side_speed_rad_s': 1.5} == override['effective']
        and report['isolated_SIDE_override_complete'], 'Runtime SIDE override receipt incomplete')
    require(not any((native/name).exists() for name in ('side_observer_failure.json', 'mechanical_capture_failure.json')),
        'Native SIDE/capture failure sidecar exists')
    pre = helper.pre_side_review(BASELINE/'mechanical_5ms.jsonl.gz', native/'mechanical_5ms.jsonl.gz')
    coupled = helper.coupled_reference_review(native/'mechanical_5ms.jsonl.gz')
    capture.write(output/'pre_SIDE_bitwise_review.json', pre)
    capture.write(output/'coupled_reference_review.json', coupled)
    require(pre['status'] == 'preSIDE_equal', 'preSIDE native selected frames differ; preserve but reject single-variable interpretation')
    require(coupled['status'] == 'paired_reference_preserved', 'Native SIDE reference coupling differs')
    inputs = list(BASELINE.glob('mechanical*')) + list(native.glob('*')) + [Path(process_receipt), output/'run_plan.json',
        output/'pre_SIDE_bitwise_review.json', output/'coupled_reference_review.json', HERE/'candidate_manifest.json']
    return dict(schema='source021-SIDE-single-variable-native-readback-v1',
        status='native_counterfactual_integrity_checked_not_functionally_qualified',
        native_integrity=integrity, initial_birth_private_sampling_std_exact=True,
        pre_SIDE=pre, coupled_reference=coupled, override=override, mechanical_capture=summary,
        inputs_sha256={str(path): capture.sha(path) for path in inputs if path.is_file()},
        PPO_updates=0, formal_updates_consumed=0, qualification=None,
        limitations=['12s first-birth survival/unfinished handoff remains right-censored',
            'A changed guard/contact outcome is evidence of this rate counterfactual, not complete recovery qualification',
            'Same selected preSIDE bitwise check does not require postSIDE/reset histories to match',
            'Self-contact pair identity/overflow flag/actual randomized massCOMfriction draw unavailable',
            'Kit.close may exit Python; process exit belongs to external root receipt'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu', 'native', 'review'), default='cpu')
    parser.add_argument('--candidate-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--process-receipt', type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(HERE in output.parents, 'Candidate-only output directory required')
    capture, manifest = verify_candidate(args.candidate_sha)
    if args.mode == 'review':
        require(args.process_receipt is not None and output.is_dir(), 'Actual native output/process receipt required')
        try: capture.write(output/'side_speed_review.json', review(output, args.candidate_sha, args.process_receipt))
        except BaseException:
            capture.write(output/'side_speed_rejection.json', dict(status='preserved_actual_counterfactual_rejected',
                error=traceback.format_exc(), PPO_updates=0, qualification=None)); return 2
        return 0
    require(args.process_receipt is None and not output.exists(), 'New output required; no execution under review receipt')
    pair = capture.load(capture.ORCHESTRATOR, 'side_original_pair')
    plan, admission = pair.load_plan(capture.PAIR)
    helper = load(HERE/'side_speed_capture.py', 'side_override021')
    output.mkdir(parents=True)
    resources = None if args.mode == 'cpu' else pair.local_GPU_empty()
    source = str(Path(admission['source_root'])/'src/wheeled_tasks/chassis/recovery.py')
    native_command = [PYTHON, '-B', str(HERE/'run_side_speed_021.py'), '--mode', 'native',
        '--candidate-sha', args.candidate_sha, '--output', str(output)]
    capture.write(output/'run_plan.json', dict(schema='source021-SIDE-speed-run-plan-v1', mode=args.mode,
        candidate_manifest_sha256=args.candidate_sha, inherited_capture003_manifest_sha256=CAPTURE_SHA,
        inherited_pair_invocation_sha256=capture.sha(capture.PAIR/'invocation.json'),
        inherited_admission_sha256=plan['admission_sha256'], source_commit=plan['source_commit'],
        actual_REC3076_checkpoint_sha256=plan['checkpoint_sha256'], native_recovery_source_sha256=capture.sha(source),
        actual_baseline003_report_sha256=capture.sha(BASELINE/'report.json'), native_command=native_command,
        resource_check=resources, sole_control_override={'RecoveryFSM.config.side_speed_rad_s': {'from': 6., 'to': 1.5}},
        runtime_requested=dict(lanes=4096, policy_steps=600, physics_steps=2400, policy_dt=.02, physics_dt=.005,
            solver=[64, 32], guard_m=.003, selected_lanes=[1708, 2216], arm='mean'),
        measurement_expansion=dict(from_first260_to_first2400_physics_steps=True, first_birth_only=True,
            no_extra_physical_steps=True), PPO_updates=0, formal_updates_consumed=0, qualification=None))
    driver = pair.bound_driver()
    if args.mode == 'native':
        mechanical = capture.load(CAPTURE/'mechanical_capture.py', 'side_private_capture003')
        helper.expand_capture_module(mechanical)
        original_factory = driver.make_observer
        def factory(base, producer, composition, native_args, identity, report, binding, writer):
            parent = original_factory(base, producer, composition, native_args, identity, report, binding, writer)
            return make_side_observer(parent, capture, mechanical, helper, writer, admission['source_root'], capture.sha(source))
        driver.make_observer = factory
    original_args = ['--admission', str(capture.PAIR/'admission.json'), '--admission-sha', plan['admission_sha256'],
        '--arm', 'mean', '--output', str(output/'native')]
    if args.mode == 'cpu': original_args.append('--preflight-only')
    code = driver.main(original_args)
    verify_candidate(args.candidate_sha)
    if code != 0: return code
    if args.mode == 'cpu':
        report = capture.read(output/'native/report.json')
        require(report['status'] == 'CPU_preflight_passed_scene_not_run' and not report['native_scene_run'], 'App-free native preflight failed')
        capture.write(output/'CPU_admission.json', dict(status='CPU_authenticated_runtime_override_not_applied_scene_not_run',
            candidate_manifest_sha256=args.candidate_sha, source_commit=plan['source_commit'],
            original_identity_sha256=capture.sha(output/'native/identity.json'),
            native_scene_run=False, runtime_override_applied=False, PPO_updates=0, qualification=None))
    # Native postclose Python continuation is not relied on. Root records exit,
    # GPU ownership and argv, then invokes the explicit read-only review mode.
    return 0


if __name__ == '__main__': raise SystemExit(main())
