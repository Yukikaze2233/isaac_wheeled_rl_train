#!/usr/bin/env python3
"""Read the existing 021 native owner/progress and immutable std on CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    proof_path = root / 'first_PPO_verification_021.json'
    proof = json.loads(proof_path.read_text())
    rows = [dict(path=path, expected_sha256=row['sha256'])
            for path, row in proof['evidence']['immutable_checkpoints'].items()]
    payload = dict(rows=rows, source_commit=proof['source_commit']) if 'source_commit' in proof else dict(
        rows=rows, source_commit='d0dce6181df83b03fe166abcbadef3c81de2e57e')
    program = '''
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import torch
torch.set_num_threads(1)
payload = PAYLOAD
run = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
worker = run / 'train/recovery/attempt_005/train'
progress = json.loads((worker / 'progress.json').read_text())
owners = []
for pid in (3506636, 3506645, 3506652):
    p = Path('/proc') / str(pid) / 'cmdline'
    if not p.exists():
        raise RuntimeError('021 owner missing: ' + str(pid))
    command = p.read_bytes().decode().rstrip('\\0').split('\\0')
    if not any('source_d0dce6181df8' in s for s in command):
        raise RuntimeError('021 owner source mismatch')
    owners.append(dict(pid=pid, command=command))
std = []
for row in payload['rows']:
    p = Path(row['path'])
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != row['expected_sha256']:
        raise RuntimeError('Immutable checkpoint hash mismatch: ' + str(p))
    state = torch.load(p, map_location='cpu', weights_only=False)
    value = state['actor_state_dict']['distribution.std_param'].tolist()
    std.append(dict(path=str(p), sha256=h, task_actor_update=state['infos']['task_actor_update'],
        learning_updates=state['infos']['successful_updates_total'],
        std_parameter=value))
account = json.loads((run / 'train/consumption.json').read_text())
print(json.dumps(dict(read_at_utc=datetime.now(timezone.utc).isoformat(),
    source_commit=payload['source_commit'], progress=progress, account=account,
    owners=owners, immutable_checkpoint_std=std, qualification=None,
    scope='readonly_CPU_immutable_seals_and_native_progress_no_model_inference_no_GPU_no_PPO')))
'''.replace('PAYLOAD', repr(payload))
    result = subprocess.run([
        'ssh', '-p', '2222', '-oBatchMode=yes', '-oConnectTimeout=10',
        'kaiser@192.168.64.234', '/home/kaiser/robot-rl-sim60/env/bin/python', '-B', '-'
    ], input=program, text=True, capture_output=True, timeout=55)
    if result.returncode:
        raise RuntimeError(result.stderr)
    report = json.loads(result.stdout)
    report['local_inspector_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['first_PPO_proof_sha256'] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite an observation')
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(output=str(args.output), read_at_utc=report['read_at_utc'],
        task_actor_update=report['progress']['task_actor_update'],
        actor_updates_in_block=report['progress']['actor_updates_in_block'],
        consumed=report['account']['consumed_updates'],
        std_rows=report['immutable_checkpoint_std']), ensure_ascii=False))


if __name__ == '__main__':
    main()
