"""Run only the bounded local model comparison, retaining its real exit status."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


root = Path(__file__).resolve().parents[2]
output = Path(__file__).resolve().parent
cmd = ['/home/yukikaze/isaacsim60-venv/bin/python', '-B', str(output/'compare_candidates_four_cases_v3.py'),
    '--source-root', '/home/yukikaze/Documents/workspace/robot_rl/.worktrees/completed-speed-resume-20261010',
    '--source-revision', 'f004d2687db0f9e6351cb68c8c2f142154e30364',
    '--driver', str(output/'compare_four_cases_v3.py'),
    '--output', str(output/'native004_four_cases')]
for model in ('v6_recovery_candidate_41374_20261009', 'v6_recovery_candidate_38000_20261008',
              'v6_flat_candidate_14020_20261003', 'v6_speed_candidate_45500_actor3730_20261011'):
    cmd += ['--onnx', str(root/'models'/model/'policy.onnx')]
env = os.environ.copy()
env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    OMNI_KIT_ACCEPT_EULA='YES', LD_LIBRARY_PATH='/home/yukikaze/.local/lib/compat',
    TMPDIR='/home/yukikaze/.cache/kit-tmp')


def available():
    return next(int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines()
                if row.startswith('MemAvailable:'))


if (output/'native004_four_cases').exists() or (output/'native004_four_cases.process.json').exists():
    raise ValueError('Comparison output already exists')
if available() < 6*1024**3:
    raise RuntimeError('Insufficient available memory for the local scene')
started, minimum, reason = time.monotonic(), available(), None
with (output/'native004_four_cases.log').open('xb') as stream:
    process = subprocess.Popen(cmd, cwd=root, env=env, stdout=stream,
        stderr=subprocess.STDOUT, start_new_session=True)
    while process.poll() is None:
        current = available()
        minimum = min(minimum, current)
        if current < 1.5*1024**3 or time.monotonic()-started > 420:
            reason = 'memory_floor' if current < 1.5*1024**3 else 'wall_timeout'
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        time.sleep(2)
record = dict(at_utc=datetime.now(timezone.utc).isoformat(), command=cmd,
    pid=process.pid, exit_code=process.returncode, guard_stop_reason=reason,
    minimum_available_bytes=minimum, wall_seconds=time.monotonic()-started,
    other_processes_signalled=False, PPO_updates=0)
(output/'native004_four_cases.process.json').write_text(json.dumps(record, indent=2)+'\n')
report = json.loads((output/'native004_four_cases'/'report.json').read_text())
record['report_status'] = report['status']
record['completed_comparison'] = record['exit_code'] == 0 and report['status'] == 'completed_bounded_unscored' and len(report.get('candidates', [])) == 4
(output/'native004_four_cases.process.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
raise SystemExit(0 if record['completed_comparison'] else 1)
