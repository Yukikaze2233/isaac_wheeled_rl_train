"""Authenticate three frozen projections; --execute runs six serial worlds."""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback


BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parents[2]
ADAPTER = WORKSPACE / '.worktrees/periodic-replay-source-20261009'
RUNTIME = WORKSPACE / '.worktrees/serial-dispatch-source028-20261009'
MODELS = BASE.parents[1] / 'models'
OUTPUT = BASE / 'rec_speed_mean43500_independent_compare011'
PYTHON = Path('/home/yukikaze/isaacsim60-venv/bin/python')
ADAPTER_COMMIT = 'beef273c326823dca4403ab6e4b1c28a6a1b70a7'
DRIVER_SHA = 'a86c9c448da7e92f4ad5ecb52b3e2145d3eea7fbcdf151e075638dab15a36cff'
PREP_SHA = '257eb179cba81df3cc786c6c9d98383cd41b6319b78c48825e18b17efbab3328'
PROJECTIONS = (
    ('nominal', '03bf51b6b9f45fc2aaa450af7600d14e68ff767dd192b652775b4be5e8840e9d'),
    ('yaw_positive', 'b6c2f058c92505400cf9a85342a759d61f110d3970879ce789f515f3cbdd7ba1'),
    ('yaw_negative', '162a3f08a6341ecc7706f65e71675e5315c7c8237486b03fd564db71f843919e'),
)
PAIR = tuple((MODELS / name / 'model.pt', MODELS / name / 'policy.onnx') for name in (
    'v6_recovery_candidate_41393_20261009', 'v6_speed_periodic_43500_actor2097_20261009'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, data):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def check_adapter(commit):
    require(commit == ADAPTER_COMMIT, 'Only the reviewed frozen adapter commit is accepted')
    actual = subprocess.check_output(['git', '-C', str(ADAPTER), 'rev-parse', 'HEAD'], text=True).strip()
    require(actual == commit, 'Adapter HEAD differs from the reviewed commit')
    dirty = subprocess.check_output(['git', '-C', str(ADAPTER), 'status', '--porcelain',
                                     '--untracked-files=no'], text=True).strip()
    require(not dirty, 'Tracked adapter files changed')
    require(digest(ADAPTER / 'scripts/run_rec_speed_mean_replay.py') == DRIVER_SHA,
            'Reviewed driver SHA differs')
    require(digest(ADAPTER / 'scripts/prepare_rec_speed_mean_replay.py') == PREP_SHA,
            'Reviewed preparation module SHA differs')


def check_candidate_bytes(identities):
    for identity in identities:
        for path_key, pin_key in (('checkpoint', 'checkpoint_sha256'), ('onnx', 'onnx_sha256'),
                                  ('original_contract', 'original_contract_sha256')):
            require(digest(identity[path_key]) == identity[pin_key], 'Authenticated candidate bytes changed')
        require(digest(Path(identity['checkpoint']).with_suffix('.checkpoint.json')) == identity['seal_sha256'],
                'Authenticated candidate seal changed')
        require(Path(identity['onnx'] + '.contract.json').read_bytes()
                == Path(identity['original_contract']).read_bytes(), 'Original ONNX contract bytes changed')


def child_command(driver, root, protocol, projection, receipt_sha, index):
    child = OUTPUT / protocol / f'world_{index:02d}'
    command = [str(PYTHON), '-B', str(driver), '--runtime-root', str(root),
        '--execute', '--protocol', protocol, '--device', 'cpu', '--candidate-index', str(index),
        '--prepared-projection', str(projection), '--projection-receipt-sha256', receipt_sha,
        '--output', str(child)]
    for checkpoint, policy in PAIR:
        command.extend(['--candidate', str(checkpoint), str(policy)])
    return child, command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-commit', required=True)
    parser.add_argument('--execute', action='store_true', help='Explicitly run six serial native processes')
    args = parser.parse_args(argv)
    check_adapter(args.adapter_commit)
    require(not OUTPUT.exists(), 'A new exclusive output directory is required; no retry')
    driver = ADAPTER / 'scripts/run_rec_speed_mean_replay.py'
    sys.path.insert(0, str(driver.parent))
    spec = importlib.util.spec_from_file_location('independent_compare011_driver', driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = module.prep.frozen_runtime(RUNTIME)
    contracts, identities = zip(*(module.prep.authenticate_candidate(root, *candidate) for candidate in PAIR))
    abi = module.prep.shared_abi(contracts)
    module.prep.validate_pair(identities)
    check_candidate_bytes(identities)
    plans = []
    for protocol, receipt_sha in PROJECTIONS:
        projection = BASE / ('rec_speed_mean42000_nominal002' if protocol == 'nominal' else f'rec_speed_mean42000_{protocol}003') / protocol
        _, _, binding = module.verify_prepared_projection(root, projection, protocol, receipt_sha)
        plans.append(dict(protocol=protocol, projection=str(projection), receipt_sha256=receipt_sha,
                          binding=binding, commands=[child_command(driver, root, protocol, projection,
                              receipt_sha, index)[1] for index in (0, 1)]))
    intent = dict(schema='independent-native-motion-intent-v1', adapter_commit=args.adapter_commit,
        runner_sha256=digest(__file__), driver_sha256=DRIVER_SHA, preparation_sha256=PREP_SHA,
        runtime_commit=module.prep.E5, original_candidates=identities, shared_abi=abi, plans=plans,
        mode='six_fresh_serial_native_processes', PPO_updates=0, training_signals=0,
        optimizer_constructed=False, qualification=None, hardware_deployment_ready=False)
    if not args.execute:
        print(json.dumps(dict(status='CPU_authenticated_scene_not_run', output=str(OUTPUT),
                              scene_run=False, intent=intent), ensure_ascii=False), flush=True)
        return intent
    check_adapter(args.adapter_commit)
    require(digest(__file__) == intent['runner_sha256'], 'Runner bytes changed before execution')
    OUTPUT.mkdir()
    intent['started_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write(OUTPUT / 'intent.json', intent)
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1')
    protocols_completed, all_pids = [], []
    current_protocol = current_index = None
    try:
        for plan in plans:
            current_protocol = plan['protocol']
            directory = OUTPUT / current_protocol
            directory.mkdir()
            results = []
            for current_index in (0, 1):
                check_adapter(args.adapter_commit)
                check_candidate_bytes(identities)
                require(digest(__file__) == intent['runner_sha256'], 'Runner bytes changed')
                _, _, binding = module.verify_prepared_projection(root, plan['projection'],
                    current_protocol, plan['receipt_sha256'])
                require(binding == plan['binding'], 'Shared prepared projection binding changed')
                child, command = child_command(driver, root, current_protocol, Path(plan['projection']),
                                               plan['receipt_sha256'], current_index)
                started = time.monotonic()
                with (directory / f'world_{current_index:02d}.log').open('x') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               cwd=ADAPTER, env=environment)
                    receipt = dict(pid=process.pid, command=command, protocol=current_protocol,
                        selected_candidate_index=current_index, driver_sha256=DRIVER_SHA,
                        prepared_projection=binding, started_at_utc=datetime.datetime.now(
                            datetime.timezone.utc).isoformat())
                    try:
                        stat = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
                        receipt['starttime'] = int(stat[19])
                    except (OSError, ValueError, IndexError) as error:
                        receipt.update(starttime=None, process_read_error=str(error))
                    try:
                        write(directory / f'world_{current_index:02d}.process.json', receipt)
                        print(json.dumps(dict(status='native_child_live', protocol=current_protocol,
                            index=current_index, pid=process.pid)), flush=True)
                    finally:
                        code = process.wait()
                execution = dict(exit_code=code, wall_seconds=time.monotonic() - started, process=receipt,
                    ended_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                write(directory / f'world_{current_index:02d}.execution.json', execution)
                require(code == 0, f'Native child {current_protocol}/{current_index} failed; no retry')
                require(type(receipt['starttime']) is int, 'Actual process starttime was not captured')
                result = module.verify_single_child_completion(child, current_protocol, current_index,
                    identities, root, plan['projection'], plan['receipt_sha256'], abi)
                require(result['native_constructor']['pid'] == process.pid, 'Native constructor PID differs')
                require(process.pid not in all_pids, 'Fresh children must have distinct recorded PIDs')
                check_adapter(args.adapter_commit)
                check_candidate_bytes(identities)
                require(digest(__file__) == intent['runner_sha256'], 'Runner bytes changed')
                all_pids.append(process.pid)
                results.append(dict(index=current_index, result_file=str(child / 'native_replay.json'),
                    result_sha256=digest(child / 'native_replay.json'), native_constructor=result['native_constructor'],
                    external_process=receipt, reset_fingerprint=result['candidates'][0]['reset_fingerprint']))
                print(json.dumps(dict(status='native_child_report_verified', protocol=current_protocol,
                    index=current_index, pid=process.pid, exit_code=code)), flush=True)
            require(results[0]['reset_fingerprint'] == results[1]['reset_fingerprint'],
                    'Original paired native reset fingerprints differ')
            complete = dict(protocol=current_protocol, prepared_projection=plan['binding'], children=results,
                initial_fingerprints_equal=True, solver_internal_state_equivalence_proven=False,
                PPO_updates=0, optimizer_constructed=False, qualification=None, hardware_deployment_ready=False)
            write(directory / 'completion.json', complete)
            protocols_completed.append(complete)
        write(OUTPUT / 'completion.json', dict(status='independent_native_motion_completed_unscored',
            intent_sha256=digest(OUTPUT / 'intent.json'), protocols=protocols_completed,
            solver_internal_state_equivalence_proven=False, PPO_updates=0, optimizer_constructed=False,
            qualification=None, hardware_deployment_ready=False))
    except Exception:
        write(OUTPUT / 'failure.json', dict(status='independent_native_motion_failed_no_retry',
            protocol=current_protocol, candidate_index=current_index, error=traceback.format_exc(),
            completed_protocols=protocols_completed, intent_sha256=digest(OUTPUT / 'intent.json'),
            PPO_updates=0, optimizer_constructed=False, qualification=None))
        raise
    print(json.dumps(dict(status='independent_native_motion_completed_unscored', output=str(OUTPUT))), flush=True)
    return intent


if __name__ == '__main__':
    main()
