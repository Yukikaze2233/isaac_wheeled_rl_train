"""Unpack and authenticate a frozen v25 source; no process or account changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


data = json.load(sys.stdin)
root = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
target = root / 'source_integrated_v25_20261008'
archive = root / 'source_integrated_v25_20261008.tar.gz'
if data['source_root'] != str(target) or data['source_archive'] != str(archive):
    raise ValueError('Only the named integrated source may be staged')
if sha(root / 'job.json') != data['old_job_sha256'] or sha(root / 'source_revision_023.json') != data['old_source_revision_sha256']:
    raise ValueError('Production predecessor changed before staging')
for pin in data['old_owners'].values():
    tail = (Path('/proc') / str(pin['pid']) / 'stat').read_text().rpartition(')')[2].split()
    if tail[0] == 'Z' or int(tail[19]) != pin['starttime']:
        raise ValueError('Production predecessor owner changed before staging')
if sha(archive) != data['source_archive_sha256']:
    raise ValueError('Source archive SHA differs')
receipt_path = root / 'integrated_source_staging_027.json'
if target.exists() and not receipt_path.exists():
    raise ValueError('Partial or unknown target exists; never overwrite it')
if not target.exists():
    with tarfile.open(archive, 'r:gz') as tar:
        if tar.pax_headers.get('comment') != data['source_commit']:
            raise ValueError('Git archive frozen commit differs')
        members = tar.getmembers()
        for item in members:
            name = Path(item.name)
            if name.is_absolute() or '..' in name.parts or not (item.isfile() or item.isdir()):
                raise ValueError('Source archive contains an unexpected path or member type')
        expected = {item.name for item in members if item.isfile()}
        if expected != set(data['inventory']):
            raise ValueError('Source archive inventory differs')
        target.mkdir()
        tar.extractall(target, filter='data')
actual = {str(p.relative_to(target)):dict(sha256=sha(p), size=p.stat().st_size)
          for p in target.rglob('*') if p.is_file()}
if actual != data['inventory']:
    raise ValueError('Unpacked source differs from its complete frozen inventory')
old = root / 'source_6f07ec298e01'
legacy_contracts = {str(p.relative_to(old)):sha(p) for p in (old / 'contracts').glob('*.json')}
if any(sha(target / name) != digest for name, digest in legacy_contracts.items()):
    raise ValueError('Frozen legacy contract bytes changed')
if sha(target / 'contracts/v6_task_campaign_v25.json') != data['plan_sha256']:
    raise ValueError('Frozen target plan SHA differs')
receipt = dict(status='frozen_joint_source_staged_not_launched', source_commit=data['source_commit'],
    source_root=str(target), source_archive=str(archive), source_archive_sha256=sha(archive),
    source_files=len(actual), old_contracts_preserved=len(legacy_contracts),
    plan_sha256=data['plan_sha256'], inventory_sha256=data['inventory_sha256'],
    staged_at_utc=datetime.now(timezone.utc).isoformat(), PPO_updates=0,
    production_enabled=False, signals=0, qualification=None)
if receipt_path.exists():
    old_receipt = json.loads(receipt_path.read_text())
    for key in ('source_commit','source_root','source_archive_sha256','source_files','plan_sha256','inventory_sha256'):
        if old_receipt[key] != receipt[key]:
            raise ValueError('Existing staging receipt differs from this frozen source')
    receipt = old_receipt
else:
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
print(json.dumps(receipt))
