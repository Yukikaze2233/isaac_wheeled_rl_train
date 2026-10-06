from pathlib import Path
import sys,json,hashlib,time,tarfile
from copy import deepcopy
out=Path(__file__).resolve().parent;m=json.loads((out/'source_candidate_020.json').read_text());root=Path(m['clean_root']);sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from train_chassis import preflight
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight
from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis import ordinary_posture_revision as recipe
from wheeled_tasks.chassis.task_campaign import validate_plan_migration
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();t=time.monotonic();plan=json.loads((root/m['contract_file']).read_text());source=json.loads((root/recipe.SOURCE_PLAN_FILE).read_text());migration=validate_plan_migration(source,plan,plan['revision_migration']['run_directory']);workers=recipe.validate_frozen_workers(root);receipts=[]
with tarfile.open(m['archive_local']) as a:
 assert a.pax_headers['comment']==m['source_commit'];files={str(p.relative_to(root)):sha(p) for folders in ['src','scripts','contracts'] for p in (root/folders).rglob('*') if p.is_file() and p.suffix in ['.py','.json']};seen={member.name:hashlib.sha256(a.extractfile(member).read()).hexdigest() for member in a if member.isfile() and member.name in files};assert seen==files
for row in plan['domains']:
 assert sha(root/row['contract'])==row['sha256'];c,manifest=preflight(root/row['contract']);binding=c['design_preflight'];design=json.loads((root/binding['file']).read_text());control=load_chassis_control(root/c['control_math_source'])
 for role,details in design['stages'][binding['stage']]['roles'].items():
  rule=details['invocation']['seed'];seed=rule if type(rule)==int else rule['value'] if rule['rule']=='fixed' else rule['root_seed'];count=details['invocation']['num_envs']
  projected=deepcopy(c)
  if role!='train':projected['evaluation']=deepcopy(details['contract']['evaluation'])
  derived=projected if role=='train' else fixed_suite_contract(projected)
  r=design_preflight(derived,control,root,stage=c['enabled_stages'][0],role=role,seed=seed,num_envs=count,block_index=0);assert r['status']=='static_passed';receipts.append(r)
assert len(receipts)==30;after={n:sha(root/n) for n in files};assert files==after
d=dict(status='passed',source_commit=m['source_commit'],archive_sha256=m['archive_sha256'],archive_header_commit=m['source_commit'],all_selected_source_bytes_match_archive=True,clean_root=str(root),role_count=30,migration=migration,frozen_workers=workers,duration_seconds=time.monotonic()-t,receipts=receipts,formal_updates_consumed=0,qualification=None);(out/'clean_30_roles_020.json').write_text(json.dumps(d,indent=2)+'\n');print(dict(status='passed',role_count=len(receipts),duration_seconds=d['duration_seconds']))
