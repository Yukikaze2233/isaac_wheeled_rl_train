"""Actual bounded CPU check of the isolated 9fb99fd heading implementation."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/tmp/scut-heading-027-worktree')
OUT = Path(__file__).resolve().parent
TESTS = ('tests/test_scut_heading_native.py', 'tests/test_scut_reuse_curriculum.py',
         'tests/test_skill_command_duration.py', 'tests/test_locomotion_difficulty.py')
OWNED = ('src/wheeled_tasks/chassis/scut_heading_command.py',
         'src/wheeled_tasks/chassis/skill_commands.py', 'src/wheeled_tasks/chassis/env.py',
         'src/wheeled_tasks/chassis/integrated_production.py', 'scripts/train_chassis.py',
         'tests/test_scut_heading_native.py')
INPUTS = sorted(set([str(p.relative_to(ROOT)) for p in (ROOT/'src/wheeled_tasks/chassis').glob('*.py')]
    + [str(p.relative_to(ROOT)) for p in (ROOT/'references/scut_curriculum_b8ff79f3').rglob('*') if p.is_file()]
    + list(TESTS) + ['scripts/train_chassis.py', 'contracts/v6_integrated_runtime_recipe_v24.json',
                    'contracts/v6_task_campaign_speed_worker_v18.json',
                    'contracts/v6_task_campaign_speed_worker_v10.json']))
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():
    return {name: digest(ROOT/name) for name in INPUTS}
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
before = hashes()
command = ['/home/yukikaze/isaacsim60-venv/bin/python', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *TESTS]
env = dict(os.environ, PYTHONPATH=str(ROOT/'src') + ':' + str(ROOT/'tests'), PYTHONDONTWRITEBYTECODE='1')
begin = time.monotonic()
result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=300)
wall = time.monotonic()-begin
stdout, stderr = OUT/'cpu001.stdout.txt', OUT/'cpu001.stderr.txt'
stdout.write_text(result.stdout)
stderr.write_text(result.stderr)
after = hashes()
report = dict(schema='v6-scut-heading-native-cpu-receipt-v1', started_utc=started,
    cwd=str(ROOT), command=command, exit_code=result.returncode, wall_seconds=wall,
    baseline_commit='9fb99fd02ee2f9c3cab04df32dfe7c051a9d37b6',
    input_sha256_before=before, input_sha256_after=after, inputs_unchanged=before==after,
    owned_files={name: digest(ROOT/name) for name in OWNED},
    stdout=dict(file=str(stdout), sha256=digest(stdout)), stderr=dict(file=str(stderr), sha256=digest(stderr)),
    native_started=False, GPU_started=False, SSH_actions=False, PPO_updates=0,
    v24_recipe_modified=False, production_wiring=False, runtime_ready=False, qualification=None,
    limitations=['Real CPU native functions and AST hooks, not PhysX execution or model validation.',
                 'No native app, GPU, SSH, checkpoint, campaign or running queue action.',
                 'Enabled training has additional target/membership draws; legacy/eval preserve their original draw path.',
                 'Final physical feasibility can override nominal slew; actual applied acceleration is recomputed.'])
receipt=OUT/'cpu001.receipt.json'
receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(exit_code=result.returncode, wall_seconds=wall, inputs_unchanged=before==after,
    stdout=result.stdout[-3000:], receipt=str(receipt), receipt_sha256=digest(receipt)), ensure_ascii=False))
