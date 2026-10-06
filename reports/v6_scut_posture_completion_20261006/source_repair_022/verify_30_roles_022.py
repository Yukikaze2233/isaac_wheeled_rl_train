from pathlib import Path
import hashlib,json,sys,tarfile
root=Path(sys.argv[1]);archive=Path(sys.argv[2]);commit=sys.argv[3];expected_sha=sys.argv[4]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(archive)==expected_sha
names=set()
with tarfile.open(archive,'r:gz') as t:
 assert t.pax_headers['comment']==commit
 for m in t:
  assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and (m.isfile() or m.isdir())
  if m.isdir():continue
  assert m.name not in names;names.add(m.name);assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==sha(root/m.name),m.name
sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from run_task_campaign import validate_contracts
from wheeled_tasks.chassis.recovery_script_revision import validate_frozen_workers,revised_plan,revised_manifest
plan=json.loads((root/'contracts/v6_task_campaign_v20.json').read_text());assert plan==revised_plan(root)
assert json.loads((root/'contracts/v6_task_campaign_migration_v20.json').read_text())==revised_manifest(root)
worker=validate_frozen_workers(root);roles=validate_contracts(plan)
assert len(roles)==30 and all(x['status']=='static_passed' for x in roles)
assert len({(x['stage'],x['role']) for x in roles})==30
print(json.dumps(dict(status='passed',source_commit=commit,archive_sha256=expected_sha,archive_header_commit=commit,clean_root=str(root),all_selected_source_bytes_match_archive=True,verified_file_count=len(names),role_count=len(roles),receipts=roles,frozen_recovery_worker=worker,formal_updates_consumed=0,qualification=None)))
