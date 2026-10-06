#!/usr/bin/env python3
"""Copy pinned read-only artifacts, read existing TB on remote CPU, preserve bytes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reader-sha256', required=True)
    p.add_argument('--updates', type=int, choices=(50, 100), nargs='+', required=True)
    p.add_argument('--end-update', type=int, required=True)
    p.add_argument('--observation', required=True)
    args = p.parse_args()
    if not args.observation.replace('_', '').isalnum():
        raise ValueError('Observation must be an alphanumeric identifier')
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    sources = [here / 'read_recovery_window_021.py',
               here.parent / 'read_recovery_window_020.py',
               here.parent / 'read_recovery_window_019.py',
               root / 'source_repair_021/first_PPO_verification_021.json',
               root / 'source_repair_021/frozen_source_inventory_021.json']
    if digest(sources[0]) != args.reader_sha256:
        raise ValueError('Candidate changed since independent CPU review')
    if digest(sources[1]) != '521b10353125272aa0f21ff9dc986c65b958ea7dd225051cfc6cfdd269a215e5':
        raise ValueError('Frozen020 changed')
    if digest(sources[2]) != 'ffefbaae2e1cbf30b520224fc286a8adbc194708570e4ce0fcb00fa7f86cdb17':
        raise ValueError('Frozen019 changed')
    remote_dir = '/tmp/readonly_recovery_window_021_' + args.observation
    destination = here / args.observation
    destination.mkdir(exist_ok=False)
    ssh = ['ssh', '-p', '2222', '-oBatchMode=yes', '-oConnectTimeout=10', 'kaiser@192.168.64.234']
    subprocess.run(ssh + ['mkdir ' + shlex.quote(remote_dir)], check=True, timeout=30)
    subprocess.run(['scp', '-P', '2222', '-oBatchMode=yes', '-oConnectTimeout=10',
                    *map(str, sources), 'kaiser@192.168.64.234:' + remote_dir + '/'], check=True, timeout=45)
    remote_output = remote_dir + '/report.json'
    argv = ['/home/kaiser/robot-rl-sim60/env/bin/python', '-B', remote_dir + '/read_recovery_window_021.py',
            '--reader020', remote_dir + '/read_recovery_window_020.py',
            '--reader019', remote_dir + '/read_recovery_window_019.py',
            '--launch', '/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/recovery_quality_launch_021.json',
            '--first-ppo', remote_dir + '/first_PPO_verification_021.json',
            '--inventory', remote_dir + '/frozen_source_inventory_021.json',
            '--updates', *map(str, args.updates), '--end-update', str(args.end_update),
            '--output', remote_output]
    result = subprocess.run(ssh + [shlex.join(argv)], text=True, capture_output=True, timeout=55)
    (destination / 'stdout.log').write_text(result.stdout)
    (destination / 'stderr.log').write_text(result.stderr)
    # Even failed/pending observations are preserved. Exit0 alone is not a pass.
    subprocess.run(['scp', '-P', '2222', '-oBatchMode=yes', '-oConnectTimeout=10',
                    'kaiser@192.168.64.234:' + remote_output, str(destination / 'report.json')], check=True, timeout=30)
    script = '''
import hashlib,json
from pathlib import Path
root=Path(REMOTE_DIR)
print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}))
'''.replace('REMOTE_DIR', repr(remote_dir))
    hashes = subprocess.run(ssh + ['/home/kaiser/robot-rl-sim60/env/bin/python', '-B', '-'],
                            input=script, text=True, capture_output=True, check=True, timeout=30)
    actual = json.loads(hashes.stdout)
    expected = {file.name: digest(file) for file in sources}
    expected['report.json'] = digest(destination / 'report.json')
    if actual != expected:
        raise ValueError('Remote/source/copied result bytes differ')
    report = json.loads((destination / 'report.json').read_text())
    receipt = dict(read_at_utc=datetime.now(timezone.utc).isoformat(), remote_directory=remote_dir,
                   command=argv, exitcode=result.returncode, copied_sha256=expected,
                   source_and_copied_bytes_exact=True, report_status=report['status'],
                   report_complete=report['complete'], inspector_sha256=digest(__file__),
                   scope='separate_readonly_artifacts_existing_remote_TB_CPU_no_training_edits_no_GPU_no_PPO')
    (destination / 'invocation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(directory=str(destination), status=report['status'], complete=report['complete'],
                         error=report.get('error'), report_sha256=expected['report.json'])))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
