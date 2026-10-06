#!/usr/bin/env python3
"""Bound local REC3076 mean/stochastic observations; never PPO or SSH.

prepare/preflight/review are CPU-only. run is explicit, serial, GPU0-only.
The frozen native002/55/payment loop remains the only simulator/controller.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent.parent
DRIVER = HERE.parent / 'hold_cost_candidate_021/validate_native_teaching_021_002.py'
PYTHON = '/home/yukikaze/isaacsim60-venv/bin/python'
COMMIT = 'd0dce6181df83b03fe166abcbadef3c81de2e57e'
ARCHIVE = '2c019ba1fd56053af75bf631258397176de8c5b40f36d35ccdf54d32f5b21ee9'
PT_SHA = '286b9e26cdd7fe8726f9deb1ce5f24e005d5d918d3abc2b122fdb85b6b671c31'
WORKER_SHA = '7918e7850e16b96edfa54a8854d3f3571e8101dfa6527cad78a06b4e15baea95'
PROOF_SHA = 'e7af7199d4bd64ef7dd527d2cc56cc83d46dbac9a7e6c0fdc5c0fd5fa56aa30e'
BOUND = {
    str(DRIVER): '4fed2e4f85f4d8e92c2af804aa98975dc13381c0a6bfcb02f7f49f92d35c84e9',
    str(HERE.parent / 'probe_prefix_mean_stochastic_019.py'): '55f4511284ae09010ba9d79af2a747b70b1f9da3df6b4f713323340b64648b9d',
    str(HERE.parent / 'observe_prefix_native_payments_019.py'): '51580e8bd7c08025edd69588e70e62a2970cb6afb5f97f355d85ddf48460a8ac',
    str(HERE.parent / 'probe_prefix_mean_stochastic_payments_019.py'): 'a9e7bc648f1a8e34d802695124a3faf72998a55fe278a0000cc5cf1b803a4ca5',
}
ARMS = ('mean', 'stochastic')
COVERAGE_ONLY = 'Mechanisms reconciled, but actual L0 forward and both yaw exposure remains a native coverage gap'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def bound_driver():
    require(all(sha(p) == expected for p, expected in BOUND.items()), 'Frozen observer dependency changed')
    spec = importlib.util.spec_from_file_location('REC3076_frozen_native002', DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(output, checkpoint):
    require(not output.exists(), 'Pair output exists; preserve it and inspect')
    bound_driver()
    proof_path = REPORT / 'source_repair_021/first_PPO_verification_021.json'
    require(sha(proof_path) == PROOF_SHA, 'Actual first-PPO proof bytes differ')
    proof = read(proof_path)
    require(all(row['status'] == 'passed' for row in proof['checks'].values()), 'First-PPO proof incomplete')
    rows = [row for path, row in proof['evidence']['immutable_checkpoints'].items()
            if '/update_00030007/model.pt' in path]
    require(len(rows) == 1 and rows[0]['sha256'] == PT_SHA, 'Actual own REC3076 parent differs')
    seal = read(checkpoint.with_suffix('.checkpoint.json'))
    require(sha(checkpoint) == PT_SHA and seal == rows[0]['seal'], 'Downloaded PT or full seal differs')
    require(sha(checkpoint.with_suffix('.contract.json')) == seal['contract_sha256'] == WORKER_SHA,
            'Downloaded native v19 parent contract differs')
    require(seal['task_actor_update'] == 3076 and seal['learning_updates'] == 30007
            and seal['actor_updates_consumed'] == 29857 and seal['stage_actor_update'] == 16444
            and seal['learning_lineage_updates'] == 16514, 'REC3076 clocks differ')
    meta = read(REPORT / 'source_repair_021/source_candidate_021.json')
    require(meta['source_commit'] == COMMIT and meta['archive_sha256'] == ARCHIVE, 'Frozen source header differs')
    source = Path(meta['clean_root']).resolve()
    inventory = Path(meta['source_inventory_file']).resolve()
    require(sha(inventory) == meta['source_inventory_sha256'] == 'c3f3f17a3747b27e2982793a796fff29fe622d3406b9d2d614a682bbaf34d068',
            'Frozen source inventory differs')
    files = read(inventory)
    require(all(sha(source / name) == digest for name, digest in files.items()), 'Actual clean source differs')
    worker = source / 'contracts/v6_task_campaign_recovery_worker_v19.json'
    require(sha(worker) == WORKER_SHA, 'Recovery worker differs')
    admission = dict(schema='source021-native-teaching-admission-v1', context='sealed_recovery_resume',
        source_commit=COMMIT, source_root=str(source), source_manifest=str(inventory),
        source_manifest_sha256=sha(inventory), contract_relative=str(worker.relative_to(source)),
        contract_sha256=WORKER_SHA, checkpoint=str(checkpoint), checkpoint_sha256=PT_SHA,
        checkpoint_contract_sha256=sha(checkpoint.with_suffix('.contract.json')),
        checkpoint_seal_sha256=sha(checkpoint.with_suffix('.checkpoint.json')), seed=290927, block_index=0)
    output.mkdir(parents=True)
    parent = output / 'admission.json'
    write(parent, admission)
    commands = {arm: [PYTHON, '-B', str(DRIVER), '--admission', str(parent), '--admission-sha', sha(parent),
                     '--arm', arm, '--output', str(output / arm)] for arm in ARMS}
    plan = dict(schema='source021-sealed-REC3076-pair-plan-v1', source_commit=COMMIT,
        archive_sha256=ARCHIVE, context='sealed_recovery_resume', checkpoint_sha256=PT_SHA,
        admission_sha256=sha(parent), first_PPO_proof_sha256=PROOF_SHA,
        checkpoint_clocks=seal, bound_observer_sha256=BOUND, orchestrator_sha256=sha(Path(__file__)),
        commands=commands, serial_arm_order=list(ARMS), local_GPU=0, policy_steps=600, physical_steps=2400,
        lanes=4096, formal_updates_consumed=0, PPO_updates=0, qualification=None,
        sole_arm_difference='original55 sends native mean versus sample; both draw private noise identically',
        valid_negative='Full600 native integrity may complete while L0 exposure gate remains false',
        downstream_env_RNG='May diverge after policy-dependent auto-reset; initial birth and private sampling must match',
        app_close_returned=None)
    write(output / 'invocation.json', plan)
    return plan


def load_plan(output):
    plan = read(output / 'invocation.json')
    require(plan['orchestrator_sha256'] == sha(Path(__file__)) and plan['bound_observer_sha256'] == BOUND,
            'Pair orchestrator or frozen dependency declaration changed')
    bound_driver()
    require(plan['source_commit'] == COMMIT and plan['checkpoint_sha256'] == PT_SHA
            and plan['context'] == 'sealed_recovery_resume', 'Pair identity differs')
    admission = bound_driver().read_admission(output / 'admission.json', plan['admission_sha256'])
    for path, key in ((Path(admission['checkpoint']), 'checkpoint_sha256'),
                      (Path(admission['checkpoint']).with_suffix('.contract.json'), 'checkpoint_contract_sha256'),
                      (Path(admission['checkpoint']).with_suffix('.checkpoint.json'), 'checkpoint_seal_sha256'),
                      (Path(admission['source_manifest']), 'source_manifest_sha256')):
        require(sha(path) == admission[key], 'Pair input changed: ' + key)
    require(plan['commands'] == {arm: [PYTHON, '-B', str(DRIVER), '--admission', str(output / 'admission.json'),
            '--admission-sha', plan['admission_sha256'], '--arm', arm, '--output', str(output / arm)] for arm in ARMS},
            'Pair native arguments changed')
    return plan, admission


def diagnostic_review(module, output):
    """Keep strict native identity checks; classify only its last exposure rejection."""
    strict = None
    try:
        strict = module.review_output(output)
    except ValueError as error:
        require(str(error) == COVERAGE_ONLY, 'Native integrity rejection: ' + str(error))
    report, identity = read(output / 'report.json'), read(output / 'identity.json')
    payment, teaching = read(output / 'payment_summary.json'), read(output / 'teaching_summary.json')
    integrity = module.validate_terminal(report, payment, teaching)
    require(identity['qualification'] is None and identity['PPO_updates'] == report['formal_updates_consumed'] == 0,
            'Observation claims learning or qualification')
    exposure = integrity['actual_L0_followup_exposure_complete']
    require((strict is not None) == bool(exposure), 'Exposure rejection and native result differ')
    return dict(native_terminal_integrity_passed=True, strict_engineering_exposure_admission_passed=bool(exposure),
        status='native_complete_exposure_observed' if exposure else 'native_complete_valid_exposure_negative',
        report_sha256=sha(output / 'report.json'), identity_sha256=sha(output / 'identity.json'),
        payment_sha256=sha(output / 'payment_summary.json'), teaching_sha256=sha(output / 'teaching_summary.json'),
        first_birth=report['first_birth'], actual_nonzero_goal_lane_intervals=teaching['exposure_counts'],
        raw_instantaneous_hold_gates=report['raw_instantaneous_hold_gates'], policy_diagnostics=report['policy_diagnostics'],
        preclose_hooks_restored=True, app_close_returned=None, qualification=None, PPO_updates=0)


def process_present(pid):
    return (Path('/proc') / str(pid)).exists()


def process_starttime(pid):
    text = (Path('/proc') / str(pid) / 'stat').read_text()
    return int(text[text.rfind(')') + 2:].split()[19])


def local_GPU_empty():
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                            text=True, capture_output=True, check=True)
    require(not result.stdout.strip(), 'Local GPU compute owner exists; do not launch')
    return dict(queried_at_utc=datetime.now(timezone.utc).isoformat(), device=0, compute_owners=[], raw=result.stdout)


def preflight(output):
    plan, admission = load_plan(output)
    identities = []
    for arm in ARMS:
        target = output / ('CPU_' + arm)
        require(not target.exists(), 'CPU preflight exists; preserve evidence')
        command = plan['commands'][arm][:-1] + [str(target), '--preflight-only']
        with (output / ('CPU_' + arm + '.stdout.log')).open('x') as stream:
            result = subprocess.run(command, cwd=admission['source_root'],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'), stdout=stream, stderr=subprocess.STDOUT)
        report = read(target / 'report.json')
        require(result.returncode == 0 and report['status'] == 'CPU_preflight_passed_scene_not_run'
                and report['native_scene_run'] is False, 'Native002 CPU preflight failed')
        identities.append(read(target / 'identity.json'))
    require(identities[0] == identities[1] and identities[0]['forced_engineering_actor_update'] is None
            and identities[0]['checkpoint_clocks']['task_actor_update'] == 3076, 'CPU arm identities/clocks differ')
    load_plan(output)
    receipt = dict(status='both_CPU_preflights_passed_no_scene', invocation_sha256=sha(output / 'invocation.json'),
        identity_sha256=sha(output / 'CPU_mean/identity.json'), native_scene_run=False, PPO_updates=0, qualification=None)
    write(output / 'CPU_preflight.json', receipt)
    return receipt


def bound_arm_review(output, arm):
    receipt = diagnostic_review(bound_driver(), output / arm)
    require(read(output / arm / 'report.json')['arm'] == arm, 'Native applied arm differs')
    require(read(output / arm / 'identity.json') == read(output / 'CPU_mean/identity.json'),
            'Native parent/source/std/clocks differ from the admitted CPU identity')
    return receipt


def pair_review(output):
    plan, _ = load_plan(output)
    reports, identities, receipts = [], [], []
    for arm in ARMS:
        receipt = read(output / (arm + '.process.json'))
        require(receipt['process_exit_observed'] is True and not process_present(receipt['pid']), 'Previous native process still exists')
        review = bound_arm_review(output, arm)
        require(receipt['readback']['report_sha256'] == review['report_sha256'], 'Recorded native report changed')
        reports.append(read(output / arm / 'report.json')); identities.append(read(output / arm / 'identity.json')); receipts.append(review)
    require(identities[0] == identities[1], 'Native arm identity, source, parent, std or initial clocks differ')
    for key in ('initial_physical_birth_tensor_sha256', 'initial_physical_birth_sha256', 'initial_env_rng_sha256',
                'native_group_population', 'sampling_rng_initial_sha256', 'sampling_rng_final_sha256',
                'first_native_sampling', 'final_native_sampling', 'native_effective_std'):
        require(reports[0][key] == reports[1][key], 'Native pair initial birth/private sampling differs: ' + key)
    result = dict(schema='source021-sealed-REC3076-pair-readback-v1', status='paired_native_diagnostic_complete',
        source_commit=COMMIT, checkpoint_sha256=PT_SHA, identity_sha256=sha(output / 'mean/identity.json'),
        same_initial_birth=True, same_private_sampling_stream=True, same_saved_std=True,
        arms=dict(zip(ARMS, receipts)), formal_updates_consumed=0, PPO_updates=0, qualification=None,
        causal_scope='Same sealed actor/initial scene pair separates applied mean versus learned Gaussian sampling; no claim of learned reward benefit.',
        App_close_returned=None, unfinished_scope='12s survival is right censoring, not failure')
    write(output / 'pair_review.json', result)
    return result


def run(output, arm, max_seconds):
    plan, admission = load_plan(output)
    require(read(output / 'CPU_preflight.json')['invocation_sha256'] == sha(output / 'invocation.json'), 'CPU preflight missing or changed')
    require(60 <= max_seconds <= 1200, 'Bounded lifecycle must be60..1200seconds')
    chosen = ARMS if arm == 'pair' else (arm,)
    for name in chosen:
        if name == 'stochastic':
            previous = read(output / 'mean.process.json')
            require(previous['process_exit_observed'] and not process_present(previous['pid']), 'Mean process not confirmed exited')
            bound_arm_review(output, 'mean')
        require(not (output / name).exists(), 'Native arm exists; never rerun or overwrite')
        resources = local_GPU_empty()
        write(output / (name + '.request.json'), dict(command=plan['commands'][name], resources=resources,
              invocation_sha256=sha(output / 'invocation.json'), requested_at_utc=datetime.now(timezone.utc).isoformat()))
        start = time.monotonic()
        with (output / (name + '.stdout.log')).open('x') as stream:
            child = subprocess.Popen(plan['commands'][name], cwd=admission['source_root'],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1', OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1'),
                stdout=stream, stderr=subprocess.STDOUT)
            starttime = process_starttime(child.pid)
            write(output / (name + '.started.json'), dict(pid=child.pid, command=plan['commands'][name],
                  starttime_ticks=starttime))
            try:
                exit_code = child.wait(timeout=max_seconds)
            except subprocess.TimeoutExpired:
                evidence = None
                try:
                    evidence = bound_arm_review(output, name)
                except Exception:
                    pass
                write(output / (name + '.lifecycle_limit.json'), dict(pid=child.pid, starttime_ticks=starttime,
                    command=plan['commands'][name], status='root_attention_process_not_signalled',
                    complete_preclose=evidence is not None, preclose_readback=evidence,
                    root_may_reclaim_exclusive_diagnostic_only=evidence is not None,
                    app_close_returned=None, next_arm_started=False, automatic_signal_sent=False))
                raise RuntimeError('Lifecycle limit: no signal sent; root must inspect/reclaim only this local diagnostic process')
        require(exit_code == 0, 'Native process exit differs; preserve raw output and inspect')
        receipt = bound_arm_review(output, name)
        require(not process_present(child.pid), 'Native process has not exited')
        local_GPU_empty()
        write(output / (name + '.process.json'), dict(pid=child.pid, command=plan['commands'][name],
            process_exit_observed=True, process_exit_code=exit_code, process_wall_seconds=time.monotonic()-start,
            readback=receipt, app_close_returned=None, formal_updates_consumed=0, PPO_updates=0))
        load_plan(output)
    if all((output / (name + '.process.json')).exists() for name in ARMS):
        return pair_review(output)
    return read(output / (chosen[-1] + '.process.json'))


def collect_reclaimed(output, arm, exit_evidence):
    """CPU collection only after root records the exclusive child's real exit.

    This never signals or invents an exit code. Root's separate receipt must
    name the exact started PID/argv/starttime and say it observed that exit.
    """
    plan, _ = load_plan(output)
    limited = read(output / (arm + '.lifecycle_limit.json'))
    started = read(output / (arm + '.started.json'))
    root_exit = read(exit_evidence)
    require(limited['complete_preclose'] is True and limited['root_may_reclaim_exclusive_diagnostic_only'] is True,
            'No authenticated600 preclose; reclamation is not admitted')
    require(all(root_exit[key] == started[key] == limited[key] for key in ('pid', 'command', 'starttime_ticks'))
            and root_exit['process_exit_observed'] is True and type(root_exit['process_exit_code']) is int,
            'Root exit receipt differs from the exclusive local diagnostic')
    require(not process_present(started['pid']), 'Exclusive native process still exists')
    local_GPU_empty()
    result = bound_arm_review(output, arm)
    require(result['report_sha256'] == limited['preclose_readback']['report_sha256'], 'Preclose evidence changed after timeout')
    receipt = dict(pid=started['pid'], command=plan['commands'][arm], process_exit_observed=True,
        process_exit_code=root_exit['process_exit_code'], root_exit_evidence_sha256=sha(exit_evidence),
        root_reclaimed_after_complete_preclose=True, readback=result, app_close_returned=None,
        formal_updates_consumed=0, PPO_updates=0)
    write(output / (arm + '.process.json'), receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'preflight', 'run', 'review', 'collect-reclaimed'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=HERE / 'immutable_REC3076/model.pt')
    p.add_argument('--arm', choices=('pair', *ARMS), default='pair')
    p.add_argument('--max-arm-wall-seconds', type=int, default=600)
    p.add_argument('--exit-evidence', type=Path)
    args = p.parse_args(argv)
    output, checkpoint = args.output.resolve(), args.checkpoint.resolve()
    if args.mode == 'prepare':
        result = prepare(output, checkpoint)
    elif args.mode == 'preflight':
        result = preflight(output)
    elif args.mode == 'run':
        result = run(output, args.arm, args.max_arm_wall_seconds)
    elif args.mode == 'collect-reclaimed':
        require(args.arm in ARMS and args.exit_evidence is not None, 'One arm and independent root exit receipt required')
        result = collect_reclaimed(output, args.arm, args.exit_evidence.resolve())
    else:
        result = pair_review(output)
    print(json.dumps(dict(status=result.get('status', 'prepared_not_run'), output=str(output),
                         formal_updates_consumed=0, qualification=None), indent=2))


if __name__ == '__main__':
    main()
