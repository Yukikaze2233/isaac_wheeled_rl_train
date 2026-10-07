"""Stage an authenticated frozen source; never invoke a training entry point."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import tarfile

ROOT = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
COMMIT = 'b0d2800a338fa11099b355ece48c0baaccbb1b02'
ARCHIVE_SHA = 'd5786ade35a056583380acdd9391409a57ca53caddeefbb0c04792f92ab45546'
UPLOAD = ROOT / 'source_b0d2800a338f.tar.gz.upload_20261007T0021Z'
ARCHIVE = ROOT / 'source_b0d2800a338f.tar.gz'
SOURCE = ROOT / 'source_b0d2800a338f'
OLD_SOURCE = ROOT / 'source_6f07ec298e01'
RECEIPT = ROOT / 'terrain_v22_staging_024.json'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


require(not RECEIPT.exists(), 'Existing immutable staging receipt; inspect before retry')
require(not ARCHIVE.exists() and not SOURCE.exists(), 'Staging target already exists')
require(sha(UPLOAD) == ARCHIVE_SHA, 'Uploaded archive SHA differs')
names = set()
with tarfile.open(UPLOAD, 'r:*') as archive:
    require(archive.pax_headers.get('comment') == COMMIT, 'Archive source commit differs')
    members = archive.getmembers()
    for member in members:
        path = Path(member.name)
        require(not path.is_absolute() and '..' not in path.parts
                and (member.isfile() or member.isdir()), 'Unsupported archive path/type')
        if member.isfile():
            require(member.name not in names, 'Duplicate source path')
            names.add(member.name)
    require('scripts/run_task_campaign.py' in names
            and 'contracts/v6_task_campaign_v22.json' in names, 'Missing native source')
    SOURCE.mkdir()
    archive.extractall(SOURCE, members=members, filter='data')
    inventory = {}
    for member in members:
        if member.isfile():
            with archive.extractfile(member) as stream:
                expected = hashlib.sha256(stream.read()).hexdigest()
            actual = sha(SOURCE / member.name)
            require(actual == expected, 'Extracted bytes differ: ' + member.name)
            inventory[member.name] = actual
old_contracts = list((OLD_SOURCE / 'contracts').rglob('*.json'))
new_contracts = list((SOURCE / 'contracts').rglob('*.json'))
require(len(old_contracts) == 231 and len(new_contracts) == 235, 'Frozen contract inventory differs')
for path in old_contracts:
    require(sha(path) == sha(SOURCE / path.relative_to(OLD_SOURCE)),
            'Previously frozen contract changed: ' + str(path))
old_plan = json.loads((OLD_SOURCE / 'contracts/v6_task_campaign_v21.json').read_text())
new_plan = json.loads((SOURCE / 'contracts/v6_task_campaign_v22.json').read_text())
require(old_plan['domain_completion'] == new_plan['domain_completion'], 'Schedule changed')
for old, new in zip(old_plan['domains'], new_plan['domains'], strict=True):
    require(old['name'] == new['name'], 'Domain order changed')
    if old['name'] != 'terrain':
        require(old == new, 'Other domain configuration changed')
    require(sha(SOURCE / new['contract']) == new['sha256'], 'Worker hash differs')
require(sha(SOURCE / 'contracts/v6_task_campaign_v22.json') ==
        'd3e99634980d845acf209906a02d18dbc360cb3fe4c71d8b941eaef078ed32dd', 'Plan hash differs')
os.link(UPLOAD, ARCHIVE)
UPLOAD.unlink()
result = dict(status='frozen_v22_source_staged_not_launched', source_commit=COMMIT,
              source_root=str(SOURCE), source_archive=str(ARCHIVE),
              source_archive_sha256=ARCHIVE_SHA, source_files=len(inventory),
              old_contracts_preserved=231, contracts=235,
              plan_sha256=sha(SOURCE / 'contracts/v6_task_campaign_v22.json'),
              inventory_sha256=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
              schedule_unchanged=True, other_five_domains_unchanged=True,
              staged_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              PPO_updates=0, production_enabled=False, qualification=None)
with RECEIPT.open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result))
