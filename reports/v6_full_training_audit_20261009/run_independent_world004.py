"""Run two fresh, serial local native worlds with one shared pinned projection."""
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

base = Path(__file__).resolve().parent
workspace = base.parents[2]
adapter = workspace / '.worktrees/moving-jump-source-20261009'
runtime = workspace / '.worktrees/serial-dispatch-source028-20261009'
models = base.parents[1] / 'models'
projection = base / 'rec_speed_mean42000_nominal002/nominal'
output = base / 'rec_speed_mean42000_independent_world004'
python = Path('/home/yukikaze/isaacsim60-venv/bin/python')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

args = argparse.ArgumentParser()
args.add_argument('--adapter-commit', required=True)
args = args.parse_args()
assert subprocess.check_output(['git', '-C', str(adapter), 'rev-parse', 'HEAD'], text=True).strip() == args.adapter_commit
assert not subprocess.check_output(['git', '-C', str(adapter), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
assert not output.exists()
driver = adapter / 'scripts/run_rec_speed_mean_replay.py'
sys.path.insert(0, str(driver.parent))
spec = importlib.util.spec_from_file_location('independent_world004_driver', driver)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pair = [(models / name / 'model.pt', models / name / 'policy.onnx') for name in (
    'v6_recovery_candidate_41393_20261009', 'v6_speed_periodic_42000_actor597_20261009')]
root = module.prep.frozen_runtime(runtime)
contracts, identities = zip(*(module.prep.authenticate_candidate(root, *candidate) for candidate in pair))
abi = module.prep.shared_abi(contracts)
module.prep.validate_pair(identities)
pin = digest(projection / 'projection_receipt.json')
assert pin == '03bf51b6b9f45fc2aaa450af7600d14e68ff767dd192b652775b4be5e8840e9d'
_, _, binding = module.verify_prepared_projection(root, projection, 'nominal', pin)
driver_sha = digest(driver)
output.mkdir()
intent = dict(schema='independent-native-pair-intent-v1', adapter_commit=args.adapter_commit,
    driver_sha256=driver_sha, runtime_commit=module.prep.E5, original_candidates=identities,
    shared_abi=abi, prepared_projection=binding, mode='two_fresh_serial_native_processes',
    PPO_updates=0, training_signals=0, optimizer_constructed=False, qualification=None,
    hardware_deployment_ready=False, started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
write(output / 'intent.json', intent)
results = []
environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1')
for index in (0, 1):
    assert digest(driver) == driver_sha
    child = output / f'world_{index:02d}'
    command = [str(python), '-B', str(driver), '--runtime-root', str(root),
        '--execute', '--protocol', 'nominal', '--device', 'cpu', '--candidate-index', str(index),
        '--prepared-projection', str(projection), '--projection-receipt-sha256', pin,
        '--output', str(child)]
    for checkpoint, policy in pair:
        command.extend(['--candidate', str(checkpoint), str(policy)])
    started = time.monotonic()
    with (output / f'world_{index:02d}.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=adapter, env=environment)
        stat = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
        receipt = dict(pid=process.pid, starttime=int(stat[19]), command=command,
            selected_candidate_index=index, driver_sha256=driver_sha, prepared_projection=binding)
        write(output / f'world_{index:02d}.process.json', receipt)
        print(json.dumps(dict(status='native_child_live', index=index, pid=process.pid)), flush=True)
        code = process.wait()
    execution = dict(exit_code=code, wall_seconds=time.monotonic()-started, process=receipt,
        ended_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(output / f'world_{index:02d}.execution.json', execution)
    if code != 0:
        raise RuntimeError(f'Independent native child {index} failed; no retry')
    result = module.verify_single_child_completion(child, 'nominal', index, identities, root, projection, pin, abi)
    assert result['native_constructor']['pid'] == process.pid
    assert digest(driver) == driver_sha
    results.append(dict(index=index, result_file=str(child/'native_replay.json'),
        result_sha256=digest(child/'native_replay.json'), native_constructor=result['native_constructor'],
        reset_fingerprint=result['candidates'][0]['reset_fingerprint']))
    print(json.dumps(dict(status='native_child_report_verified', index=index, exit_code=code)), flush=True)
assert results[0]['native_constructor']['pid'] != results[1]['native_constructor']['pid']
assert results[0]['reset_fingerprint'] == results[1]['reset_fingerprint']
write(output / 'completion.json', dict(status='independent_native_pair_completed_unscored',
    intent_sha256=digest(output/'intent.json'), children=results, initial_fingerprints_equal=True,
    solver_internal_state_equivalence_proven=False, PPO_updates=0, optimizer_constructed=False,
    qualification=None, hardware_deployment_ready=False))
print(json.dumps(dict(status='independent_native_pair_completed_unscored', output=str(output))), flush=True)
