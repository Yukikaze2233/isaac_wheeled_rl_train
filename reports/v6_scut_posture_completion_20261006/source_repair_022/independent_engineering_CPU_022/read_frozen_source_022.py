from pathlib import Path
import json,hashlib,subprocess,tarfile,sys,ast
R=Path('/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule')
D=R/'reports/v6_scut_posture_completion_20261006/source_repair_022'
OUT=D/'independent_engineering_CPU_022';OUT.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
M=D/'source_candidate_022.json';m=json.loads(M.read_text());C=Path(m['clean_root']);a=Path(m['archive_local'])
I=Path(m['source_inventory_file']);inv=json.loads(I.read_text());oldI=D.parent/'source_repair_021/frozen_source_inventory_021.json';oldinv=json.loads(oldI.read_text())
A=R/'reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/contract_script_revision_candidate_022'
CM=A/'manifest_022.json';cm=json.loads(CM.read_text());candidate=Path(cm['candidate_root'])
inputs=[M,I,oldI,CM,A/'source_overlay_022.patch',A/'overlay_files_022.list',D/'freeze_source_022.py',a]+[C/name for name in cm['overlay_files']]
before={str(p):sha(p) for p in inputs}
assert sha(a)==m['archive_sha256'] and a.stat().st_size==m['archive_size_bytes']
assert sha(I)==m['source_inventory_sha256'] and sha(CM)==m['candidate_manifest_sha256']
commit=m['source_commit'];parent=m['previous_source_commit']
def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True)
assert git('rev-list','--parents','-n','1',commit).strip().split()==[commit,parent]
assert parent==cm['base_commit']=='d0dce6181df83b03fe166abcbadef3c81de2e57e'
diffs=git('diff-tree','--no-commit-id','--name-status','-r',parent,commit).splitlines();changed={line.split('\t')[1] for line in diffs}
assert changed==set(cm['overlay_files']) and len(changed)==15
files=git('ls-tree','-r','--name-only','-z',commit).rstrip('\0').split('\0');assert set(files)==set(inv) and len(files)==2632
for name,h in inv.items():assert sha(C/name)==h,name
pins=cm['source_files_sha256']|cm['contracts_sha256']|{'tests/test_recovery_script_revision.py':cm['test_sha256']}
assert set(pins)==changed
assert inv==oldinv|pins
assert all(sha(C/name)==sha(candidate/name)==h for name,h in pins.items())
# One forward archive traversal; no repeated random member seeking.
archive_files={};archive_dirs=set();header=None
with tarfile.open(a,'r|gz') as stream:
 for member in stream:
  rel=Path(member.name);assert not rel.is_absolute() and '..' not in rel.parts
  assert member.isfile() or member.isdir()
  if header is None:header=stream.pax_headers.get('comment') or member.pax_headers.get('comment')
  if member.isdir():archive_dirs.add(member.name);continue
  assert member.name not in archive_files
  f=stream.extractfile(member);h=hashlib.sha256()
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  archive_files[member.name]=h.hexdigest()
assert header==commit and archive_files==inv
assert (C/'v40_train_local.sh').exists()
# No entropy or foreign playback change can enter if the exact child delta and 15 pins hold.
assert ast.dump(next(n for n in ast.parse((C/'scripts/train_chassis.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main'),include_attributes=False)==ast.dump(next(n for n in ast.parse((Path(cm['base_root'])/'scripts/train_chassis.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main'),include_attributes=False)
sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from wheeled_tasks.chassis import recovery_script_revision as recipe,task_campaign as campaign
from run_task_campaign import validate_contracts
from train_chassis import resume_critic_warmup
plan=json.loads((C/recipe.PLAN_FILE).read_text());oldplan=json.loads((C/recipe.SOURCE_PLAN_FILE).read_text());new=recipe.canonical_worker(root=C);old=recipe.read_bound(recipe.SOURCE_WORKER_FILE,recipe.SOURCE_WORKER_SHA,C)
roles=validate_contracts(plan);assert len(roles)==30 and all(x['status']=='static_passed' for x in roles)
(OUT/'static_30_roles_022.json').write_text(json.dumps(roles,indent=2)+'\n')
realmigration=campaign.validate_plan_migration(oldplan,plan,plan['revision_migration']['run_directory']);resume=campaign.validate_task_resume(old,new)
# Real historical seal is engineering parent3076, never the next formal parent selector.
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from locomotion_exploration_sampling import load_native_actor
from wheeled_tasks.chassis.policy_transfer import verify_learning_state_restore,resume_budget
P=R/'reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/immutable_REC3076'
pt=P/'model.pt';contract=P/'model.contract.json';sealfile=P/'model.checkpoint.json'
parentpins={str(pt):'286b9e26cdd7fe8726f9deb1ce5f24e005d5d918d3abc2b122fdb85b6b671c31',str(contract):recipe.SOURCE_WORKER_SHA,str(sealfile):'14b8b10c429b819faeb49e6f640dde74fbd0ae1b4f0c928d4cd7f2ac84a2eff2'}
assert all(sha(p)==h for p,h in parentpins.items());before.update(parentpins)
saved=torch.load(pt,map_location='cpu',weights_only=True);infos=saved['infos'];seal=json.loads(sealfile.read_text())
assert seal['sealed'] is True and seal['checkpoint_sha256']==sha(pt) and infos['contract_sha256']==recipe.SOURCE_WORKER_SHA
assert infos['task_campaign']==old['task_campaign']==seal['task_campaign']
assert infos['task_actor_update']==seal['task_actor_update']==3076
assert infos['successful_updates_total']==seal['learning_updates']==30007
assert json.loads(contract.read_text())==old
cfg={'obs_groups':{'actor':['policy'],'critic':['critic']},'actor':{'class_name':'MLPModel','hidden_dims':[256,128,64],'activation':'elu','obs_normalization':False,'distribution_cfg':{'class_name':'GaussianDistribution','std_type':'scalar','init_std':new['initial_noise_std']}}}
actor=load_native_actor(saved,cfg,new,'cpu')
obs=TensorDict({'policy':torch.zeros(1,35),'critic':torch.zeros(1,81)},batch_size=[1])
critic=MLPModel(obs,cfg['obs_groups'],'critic',1,hidden_dims=[256,128,64],activation='elu',obs_normalization=False)
alg=PPO(actor,critic,None,learning_rate=new['learning_rate']);alg.load(saved,load_cfg=None,strict=True)
restore=verify_learning_state_restore(alg,saved)
budget=resume_budget(infos,30007,4096*24)
assert budget['parent_updates']==30007 and budget['learning_lineage_updates']==16514
assert torch.equal(alg.actor.state_dict()['distribution.std_param'],saved['actor_state_dict']['distribution.std_param'])
assert resume_critic_warmup(new,infos,contract_sha256=recipe.digest(C/recipe.WORKER_FILE),task_change=True)==10
for remaining in (10,7,1,0):
 same=dict(infos,contract_sha256=recipe.digest(C/recipe.WORKER_FILE),critic_warmup_remaining=remaining)
 assert resume_critic_warmup(new,same,contract_sha256=recipe.digest(C/recipe.WORKER_FILE),task_change=True)==remaining
records={'recovery/CPU_readonly_previous/train':dict(domain='recovery',consumed_updates=300,actor_updates=290)}
b=campaign.account_updates(plan,records);adapt=campaign.update_attempt(plan,records,'recovery/CPU_readonly_revision/train','recovery',dict(successful_updates=10,attempted_updates=10,actor_updates_in_block=0));v=campaign.account_updates(plan,adapt)
assert v['consumed_updates']==b['consumed_updates']+10 and v['actor_updates_consumed']==b['actor_updates_consumed']
nextactor=campaign.update_attempt(plan,adapt,'recovery/CPU_readonly_revision/train','recovery',dict(successful_updates=11,attempted_updates=11,actor_updates_in_block=1));v2=campaign.account_updates(plan,nextactor)
assert v2['consumed_updates']==b['consumed_updates']+11 and v2['actor_updates_consumed']==b['actor_updates_consumed']+1
# Actual class binding uses class definition even with an instance wrapper.
from wheeled_tasks.chassis.recovery import RecoveryFSM
fields={k:v for k,v in new['recovery_training'].items() if k in RecoveryFSM.DEFAULTS};pose=torch.zeros(6)
fsm=RecoveryFSM(1,'cpu',fields,pose,pose,pose,pose,torch.ones(4));original=fsm.update
fsm.update=lambda *args,**kwargs:original(*args,**kwargs)
runtime=recipe.native_runtime_binding(fsm,new,C);assert runtime==recipe.runtime_expectation(new)
assert runtime['side_speed_rad_s']==1.5 and runtime['handover_gyro_max_rad_s']==.75
assert all(sha(p)==h for p,h in before.items())
review=dict(schema='source022-independent-frozen-engineering-CPU-review-v1',status='passed_engineering_CPU_only_native_pending',source_commit=commit,archive_sha256=sha(a),archive_header_commit=header,archive_files=len(archive_files),inventory_files=len(inv),baseline_files=len(oldinv),clean_root=str(C),overlay_count=15,exact_child_delta=diffs,all_files_SHA_matched=True,no_unowned_entropy_or_playback_change=True,full_clean_shell_restored=True,static_roles=30,static_roles_report_sha256=sha(OUT/'static_30_roles_022.json'),plan_migration=realmigration,task_resume=resume,historical_engineering_parent=dict(task_actor_update=3076,consumption=30007,learning_lineage_updates=16514,stage_actor_update=infos['stage_actor_update'],global_actor_update=infos['global_actor_update'],actor_updates_consumed=infos['actor_updates_consumed'],files_sha256=parentpins,formal_next_parent='must_select_latest_own_REC_seal_at_deployment_not_this_diagnostic_parent'),native_CPU_fullstate_restore=restore,raw_std_preserved=True,critic_adaptation_updates=10,critic_only_consumption_fixture=10,actor_next_consumption_fixture=11,native_runtime_class_binding=runtime,instance_observer_closure_accepted=True,inputs_sha256_before=before,inputs_sha256_after={p:sha(p) for p in before},native_scene_run=False,SSH=False,GPU=False,PPO_updates=0,qualification=None,limitations=['Engineering CPU/identity qualification does not certify actual 5ms physics or recovery ability.','The actual native run and formal ownership/seal/launch remain root-only.','Existing failed first freeze is preserved; review uses current exact committed archive and full clean tree.'])
(OUT/'review.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'review':str(OUT/'review.json'),'sha256':sha(OUT/'review.json'),'status':review['status']},indent=2))
