"""One bounded native REC diagnostic; no retries or training signals."""
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
repo = base.parents[1]
workspace = repo.parent
adapter = workspace / '.worktrees/moving-jump-source-20261009'
runtime = workspace / '.worktrees/serial-dispatch-source028-20261009'
projection = base / 'rec_speed_mean42000_nominal002/nominal'
output = base / 'recovery_script_diagnostic007'
tool = repo / 'scripts/replay_recovery_script_diagnostic.py'
python = Path('/home/yukikaze/isaacsim60-venv/bin/python')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def load_tool():
    spec = importlib.util.spec_from_file_location('script_diagnostic006_tool', tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

parser = argparse.ArgumentParser()
parser.add_argument('--diagnostic-commit', required=True)
args = parser.parse_args()
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == args.diagnostic_commit
for path in (tool, Path(__file__).resolve()):
    committed = subprocess.check_output(['git', '-C', str(repo), 'show', 'HEAD:' + str(path.relative_to(repo))])
    assert hashlib.sha256(committed).hexdigest() == digest(path)
assert not output.exists()
module = load_tool()
driver = module.load_driver(adapter)
root = driver.prep.frozen_runtime(runtime)
pair = [(repo / 'models' / name / 'model.pt', repo / 'models' / name / 'policy.onnx') for name in (
    'v6_recovery_candidate_41393_20261009', 'v6_speed_periodic_42000_actor597_20261009')]
contracts, identities = zip(*(driver.prep.authenticate_candidate(root, *candidate) for candidate in pair))
abi = driver.prep.shared_abi(contracts)
driver.prep.validate_pair(identities)
_, _, binding = driver.verify_prepared_projection(root, projection, 'nominal', module.PROJECTION_SHA)
source_pins = {str(path): digest(path) for path in (tool, Path(__file__).resolve(),
    adapter / 'scripts/run_rec_speed_mean_replay.py', adapter / 'scripts/prepare_rec_speed_mean_replay.py',
    *(runtime / name for name in ('src/wheeled_tasks/chassis/env.py', 'src/wheeled_tasks/chassis/eval_env.py',
        'src/wheeled_tasks/chassis/recovery.py', 'src/wheeled_tasks/chassis/recovery_training.py',
        'src/wheeled_tasks/chassis/preflight.py', 'src/wheeled_tasks/chassis/torque_monitor.py',
        'src/wheeled_tasks/chassis/continuous_encoders.py', 'src/wheeled_tasks/chassis/recovery_observer.py',
        'src/wheeled_tasks/chassis/command_transport.py', 'src/wheeled_tasks/chassis/actuator_response.py')))}
for name, expected in (
    ('native_replay.json', '7e3669677cc6a8726a71b16777464f1653d0c28d1b2949f83c3b5f6ba0058d36'),
    ('candidate_00_trace.json', '1b327bcc09f3c4dad971b0accd8f3256646ca1d8700a612f9a67179b76d99aa2')):
    path = base / 'rec_speed_mean42000_independent_world004/world_00' / name
    assert digest(path) == expected
    source_pins[str(path)] = expected
output.mkdir()
write(output / 'intent.json', dict(schema='native-five-lane-script-diagnostic-intent-v1',
    diagnostic_commit=args.diagnostic_commit, runtime_commit=driver.prep.E5,
    adapter_commit=module.ADAPTER_COMMIT, source_pins=source_pins, candidates=identities,
    shared_abi=abi, prepared_projection=binding, selected_candidate_index=0,
    method='one_fresh_native_CPU_world_observation_only', PPO_updates=0,
    optimizer_constructed=False, training_signals=0, qualification=None,
    started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
command = [str(python), '-B', str(tool), '--execute', '--adapter-root', str(adapter),
    '--runtime-root', str(runtime), '--projection', str(projection), '--output', str(output / 'native')]
for checkpoint, policy in pair:
    command += ['--candidate', str(checkpoint), str(policy)]
environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1')
wall = time.monotonic()
with (output / 'native.log').open('x') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=repo, env=environment)
    receipt = module.wait_process_identity(process, command)
    receipt['actual_command'] = receipt['command']
    write(output / 'process.json', receipt)
    print(json.dumps(dict(status='native_diagnostic_child_live', pid=process.pid)), flush=True)
    code = process.wait()
write(output / 'execution.json', dict(exit_code=code, wall_seconds=time.monotonic()-wall,
    process=receipt, ended_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
assert code == 0, 'Native diagnostic failed; preserve evidence, no retry'
result = driver.verify_single_child_completion(output / 'native', 'nominal', 0, identities,
    root, projection, module.PROJECTION_SHA, abi)
assert result['native_constructor']['pid'] == process.pid
diagnostic = json.loads((output / 'native/script_diagnostic.json').read_text())
assert diagnostic['status'] == 'native_script_diagnostic_completed_unscored'
assert diagnostic['native_result_sha256'] == digest(output / 'native/native_replay.json')
assert diagnostic['sidecar']['sha256'] == digest(output / 'native/script_prefix_200hz.npz')
assert all(digest(path) == pin for path, pin in source_pins.items())
baseline = json.loads((base / 'rec_speed_mean42000_independent_world004/world_00/native_replay.json').read_text())
native = result['candidates'][0]
baseline_native = baseline['candidates'][0]
original_trace = json.loads((base / 'rec_speed_mean42000_independent_world004/world_00/candidate_00_trace.json').read_text())
observed_trace = json.loads((output / 'native/candidate_00_trace.json').read_text())
comparison = dict(reset_fingerprint_equal=native['reset_fingerprint'] == baseline_native['reset_fingerprint'],
    full_policy_trace_equal=observed_trace == original_trace,
    native_events_equal=native['native_events'] == baseline_native['native_events'],
    native_episode_metrics_equal=native['native_episode_metrics'] == baseline_native['native_episode_metrics'],
    quiet_metrics_equal=native['fixed_quiet_window_metrics'] == baseline_native['fixed_quiet_window_metrics'],
    hidden_solver_state_equivalence_proven=False)
write(output / 'comparison_to_unobserved004.json', comparison)
assert all(comparison[key] for key in comparison if key != 'hidden_solver_state_equivalence_proven'), \
    'Recorded replay differs from original native behavior; do not grant observation equivalence'
write(output / 'completion.json', dict(status='native_script_diagnostic_completed_unscored',
    native_constructor=result['native_constructor'], diagnostic_sha256=digest(output / 'native/script_diagnostic.json'),
    original_trace_comparison=comparison, PPO_updates=0, training_signals=0,
    optimizer_constructed=False, qualification=None, hardware_deployment_ready=False))
print(json.dumps(dict(status='native_diagnostic_completed_unscored', PPO_updates=0)), flush=True)
