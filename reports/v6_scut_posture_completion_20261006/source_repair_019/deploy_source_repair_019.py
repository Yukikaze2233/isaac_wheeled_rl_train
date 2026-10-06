#!/usr/bin/env python3
"""Emit three separately reviewed remote steps for a same-contract source repair.

This program never connects, signals, or launches. All invocations must explicitly
use --emit-remote-code. Copy the prepare readback's revision/predecessor JSON to
the two local files before emitting launch. A launch receipt is never retryable.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tarfile

OUT = Path(__file__).resolve().parent
BASE = "/home/kaiser/robot-rl-sim60"
RUN = BASE + "/experiments/v6-task-campaign-20261001T110045Z-282358"
SESSION = "v6-task-campaign-20261001T110045Z-282358-r19"
COMMIT = "712d701473f38ef6797725f8efdda9e6beb85e87"
PREVIOUS = "4ba339299c203cf2aba77183ec81f2f017dc9bf9"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evidence(paths):
    meta, tests, roles, native = (read(paths[name]) for name in paths)
    assert meta["source_commit"] == COMMIT and meta["previous_source_commit"] == PREVIOUS
    assert meta["source_remote"] == RUN + "/source_712d701473f3"
    assert meta["archive_remote"] == meta["source_remote"] + ".tar.gz"
    for item in (tests, roles, native):
        assert item["status"] == "passed"
        assert item["source_commit"] == COMMIT and item["archive_sha256"] == meta["archive_sha256"]
    assert tests["exitcode"] == 0 and roles["role_count"] == 30
    assert tests["source_files_unchanged"] is tests["exact_bytes_match_archive"] is True
    assert tests["source_files_before_sha256"] == tests["source_files_after_sha256"]
    assert roles["all_selected_source_bytes_match_archive"] is True
    assert roles["archive_header_commit"] == COMMIT
    assert len(roles["receipts"]) == 30 and all(item["status"] == "static_passed" for item in roles["receipts"])
    assert sha(meta["archive_local"]) == meta["archive_sha256"]
    # A native envelope must carry the complete original report, not only flags.
    probe = native["domains"]["recovery"] if "domains" in native else native["report"]
    clean = Path(meta["clean_root"])
    sys.path.insert(0, str(clean / "src"))
    sys.path.insert(0, str(clean / "scripts"))
    from wheeled_tasks.chassis.preflight import validate_runtime_state
    from probe_scut_curriculum import source_snapshot
    from recovery_prefix_support import BASELINE_PT_SHA, BASELINE_CONTRACT_SHA
    with tarfile.open(meta["archive_local"], "r:gz") as archive:
        assert archive.pax_headers.get("comment") == COMMIT
        def archived(name):
            return archive.extractfile(name).read()
        plan = json.loads(archived("contracts/v6_task_campaign_v18.json"))
        assert hashlib.sha256(archived("contracts/v6_task_campaign_v18.json")).hexdigest() == meta["plan_sha256"]
        assert hashlib.sha256(archived("contracts/v6_task_campaign_migration_v18.json")).hexdigest() == meta["manifest_sha256"]
        pins = {row["name"]: row["sha256"] for row in plan["domains"] if row["name"] != "flat"}
        assert pins == meta["worker_hashes"]
        assert {item["domain"]: item["target_sha256"] for item in roles["frozen_workers"]} == pins
        for row in plan["domains"]:
            assert hashlib.sha256(archived(row["contract"])).hexdigest() == row["sha256"]
        for name, checksum in tests["source_files_before_sha256"].items():
            assert hashlib.sha256(archived(name)).hexdigest() == sha(clean / name) == checksum
        row = next(row for row in plan["domains"] if row["name"] == "recovery")
        config = json.loads(archived(row["contract"]))
        binding = config["design_preflight"]
        role = json.loads(archived(binding["file"]))["stages"][binding["stage"]]["roles"]["train"]
        rule = role["invocation"]["seed"]
        seed = rule if type(rule) is int else rule["root_seed"] + probe["block_index"]
        assert probe["schema"] == "scut-curriculum-native-probe-v18"
        assert probe["status"] == "engineering_probe_completed" and probe["domain"] == "recovery"
        assert probe["run_role"] == "train" and probe["num_envs"] == role["invocation"]["num_envs"] == 4096
        assert probe["seed"] == plan["seed"] == seed and probe["block_index"] == 0
        assert probe["target_contract_sha256"] == row["sha256"]
        assert probe["source_files_unchanged"] is probe["target_contract_unchanged"] is True
        assert probe["source_files_changed"] == []
        assert probe["forced_engineering_actor_update"] == config["ordinary_posture_curriculum"]["origin_actor_update"]
        snapshot = source_snapshot(config)
        assert probe["source_files_sha256"] == snapshot
        for name, checksum in snapshot.items():
            assert hashlib.sha256(archived(name)).hexdigest() == checksum
        checkpoint = probe["checkpoint"]
        assert checkpoint["checkpoint_sha256"] == BASELINE_PT_SHA
        assert checkpoint["checkpoint_contract_sha256"] == BASELINE_CONTRACT_SHA
        assert checkpoint["seal_and_infos_authenticated"] is checkpoint["actor_only_deterministic_inference"] is True
        assert probe["formal_updates_consumed"] == probe["optimizer_steps"] == 0 and probe["qualification"] is None
        assert probe["policy_steps"] == 600 and probe["physics_steps"] == 2400
        preflight = probe["startup"]["runtime_preflight"]
        assert preflight["status"] == "runtime_configuration_passed"
        observed = preflight["observed"]
        expected = deepcopy(role["runtime"])
        if expected.get("seed") == {"source": "effective_seed"}:
            expected["seed"] = seed
        validate_runtime_state(expected, observed)
        assert observed["seed"] == seed and observed["modules"]["ordinary_posture_curriculum"] is True
        proof = probe["ordinary_posture_native_proof"]
        assert proof["actual_native_velocity_delta_observed"] is proof["actual_root_rotation_observed"] is True
        initial, final = proof["initial"], proof["final"]
        assert initial["eligible_count"] == final["eligible_count"] == 1449
        assert initial["native_root_quaternion_readback_passed"] is initial["geometry_bound"] is True
        assert initial["birth_selected_count"] > 0 and initial["birth_abs_angle_max_rad"] > 0
        assert final["push_events"] > 0 and final["push_velocity_delta_max_m_s"] > 0
        groups = probe["startup"]["scene_group_counts"]
        assert sum(groups.values()) == 4096 and min(groups.values()) > 0
        timing = probe["control_timing"]
        assert timing["pd_updates"] == timing["feedback_refreshes"] == 4 and timing["target_decodes"] == 1
        assert (timing["physics_dt"], timing["policy_dt"]) == (.005, .02)
    return dict(source=meta, evidence_sha256={name: sha(path) for name, path in paths.items()})


COMMON = r'''
import hashlib,json,pathlib,sys
r=pathlib.Path(RUN);src=pathlib.Path(META['source_remote'])
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def entry(argv):
 for value in argv[1:]:
  if value in (b'-c',b'-m'):return None
  if value.endswith(b'.py'):return pathlib.Path(value.decode())
 return None
def owners():
 result=[]
 for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:
   argv=p.read_bytes().split(b'\0');script=entry(argv)
   if script and script.name in ('run_task_campaign.py','train_chassis.py','evaluate_chassis.py','chassis_remote_job.py') and any(x==str(r).encode() or x==str(r/'train').encode() or x.startswith(str(r/'train').encode()+b'/') for x in argv):result.append(dict(pid=int(p.parent.name),command=[x.decode() for x in argv if x]))
  except (OSError,PermissionError):pass
 return result
sys.path.insert(0,str(src/'src'));sys.path.insert(0,str(src/'scripts'))
from chassis_remote_job import validate_source_revision
from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count,checkpoint_contract_path
from wheeled_tasks.chassis.task_campaign import account_updates,validate_task_resume
def latest(domain):
 candidates=[]
 for p in (r/'train'/domain).glob('attempt_*/train/**/*.checkpoint.json'):
  if any(part.endswith('.pending') for part in p.parts):continue
  seal=read(p);cp=p.with_suffix('').with_suffix('.pt')
  assert type(seal.get('learning_updates')) is int and seal['learning_updates']>=0
  candidates.append((seal['learning_updates'],cp.name=='model_final.pt',cp))
 return max(candidates,key=lambda item:item[:2])[2] if candidates else None
def parents(plan):
 result={}
 for row in plan['domains']:
  domain=row['name'];cp=latest(domain)
  if cp is None:
   assert domain=='stepjump','Existing domain lost all sealed parents'
   continue
  seal_path=cp.with_suffix('.checkpoint.json');seal=read(seal_path)
  contract=checkpoint_contract_path(cp)
  assert cp.resolve().is_relative_to((r/'train'/domain).resolve())
  assert seal['sealed'] is True and seal['task_campaign']['domain']==domain
  assert checkpoint_update_count(cp)==seal['learning_updates']
  assert sha(cp)==seal['checkpoint_sha256'] and sha(contract)==seal['contract_sha256']
  # Older unvisited domains retain their own older worker state and migrate
  # through the already-frozen v18 path when the existing campaign reaches them.
  validate_task_resume(read(contract),read(src/row['contract']))
  result[domain]=dict(path=str(cp),sha256=sha(cp),seal=seal,seal_sha256=sha(seal_path),contract=str(contract),contract_sha256=sha(contract))
 return result
def authenticate_quiescent():
 job=read(r/'job.json');assert not pathlib.Path('/proc/'+str(job['pid'])+'/cmdline').exists(),'Former supervisor still live'
 assert not owners(),'Live campaign worker/wrapper/supervisor or orphan'
 old=pathlib.Path(RUN)/'source_4ba339299c20'
 oldrevision=read(r/'source_revision_018.json')
 assert oldrevision['source_commit']==META['previous_source_commit'] and pathlib.Path(oldrevision['new_source_root'])==old
 cmd=job['command']
 assert cmd.count(str(old/'scripts/run_task_campaign.py'))==1
 assert cmd.count('--contract')==cmd.count('--run-dir')==1
 assert pathlib.Path(cmd[cmd.index('--contract')+1])==old/'contracts/v6_task_campaign_v18.json'
 assert pathlib.Path(cmd[cmd.index('--run-dir')+1])==r/'train'
 assert sha(old/'contracts/v6_task_campaign_v18.json')==sha(src/'contracts/v6_task_campaign_v18.json')==META['plan_sha256']
 # Source-only means every existing contract stays byte-for-byte identical.
 oldcontracts={str(p.relative_to(old)):sha(p) for p in (old/'contracts').rglob('*') if p.is_file()}
 newcontracts={str(p.relative_to(src)):sha(p) for p in (src/'contracts').rglob('*') if p.is_file()}
 assert oldcontracts==newcontracts,'Source repair changed contract bytes or catalogue'
 state=read(r/'train/campaign_state.json');account=read(r/'train/consumption.json');plan=read(src/'contracts/v6_task_campaign_v18.json')
 assert state['status']=='stopped'
 assert state['contract_sha256']==account['contract_sha256']==META['plan_sha256']
 assert account['rollback_refunds_consumed_updates'] is False and account['run_directory']==str(r/'train')
 calculated=account_updates(plan,account['attempts'])
 assert calculated=={k:account[k] for k in calculated}=={k:state[k] for k in calculated}
 # Recompute every reservation from persisted worker progress without writing.
 from wheeled_tasks.chassis.task_campaign import update_attempt
 replay=account['attempts']
 for p in (r/'train').glob('*/attempt_*/train/progress.json'):
  name=str(p.parent.relative_to(r/'train'));replay=update_attempt(plan,replay,name,p.parts[-4],read(p))
 assert replay==account['attempts'],'Final progress has uncharged work or stale account'
 bound=parents(plan);rec=bound['recovery'];cp=pathlib.Path(rec['path'])
 assert cp.name=='model_final.pt' and cp.parent==r/'train/recovery/attempt_002/train'
 completion=read(cp.parent/'completion.json');progress=read(cp.parent/'progress.json')
 assert completion['status']=='stopped' and completion['successful_updates']>0 and completion['actor_updates_in_block']>0
 assert completion['task_actor_update']==progress['task_actor_update']==rec['seal']['task_actor_update']>2294
 worker=next(row for row in plan['domains'] if row['name']=='recovery')
 assert rec['contract_sha256']==worker['sha256']==META['worker_hashes']['recovery']
 assert rec['seal']['learning_updates']<=account['consumed_updates'] and rec['seal']['actor_updates_consumed']<=account['actor_updates_consumed']
 import torch
 from train_chassis import resume_critic_warmup
 payload=torch.load(cp,map_location='cpu',weights_only=False);infos=payload['infos']
 assert all(payload.get(k) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict')),'Not full learning state'
 assert payload['optimizer_state_dict']['state'],'Missing restored Adam state'
 for key in ('stage_actor_update','task_actor_update','actor_updates_consumed','learning_lineage_updates'):
  assert infos[key]==rec['seal'][key]
 assert infos['successful_updates_total']==rec['seal']['learning_updates']
 assert infos['contract_sha256']==worker['sha256'] and infos['task_campaign']['domain']=='recovery'
 assert infos['critic_warmup_remaining']==resume_critic_warmup(read(src/worker['contract']),infos,contract_sha256=worker['sha256'],task_change=True)==0
 rec['authenticated_infos']={k:infos[k] for k in ('contract_sha256','successful_updates_total','stage_actor_update','task_actor_update','actor_updates_consumed','learning_lineage_updates','critic_warmup_remaining')}
 rec['completion_sha256']=sha(cp.parent/'completion.json');rec['progress_sha256']=sha(cp.parent/'progress.json')
 return job,state,account,bound,old
'''

PREPARE = r'''
import fcntl,tempfile
from datetime import datetime,timezone
with (r/'source_repair_019.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/'source_repair_launch_019.json').exists(),'Already dispatched; inspect'
 job,state,account,bound,old=authenticate_quiescent()
 command=list(job['command']);command[command.index(str(old/'scripts/run_task_campaign.py'))]=str(src/'scripts/run_task_campaign.py')
 command[command.index('--contract')+1]=str(src/'contracts/v6_task_campaign_v18.json')
 revision=dict(schema='task-source-revision-v1',previous_job_sha256=sha(r/'job.json'),command=command,
  source_commit=META['source_commit'],old_source_root=str(old),new_source_root=str(src),old_contract_sha256=META['plan_sha256'],new_contract_sha256=META['plan_sha256'],source_archive=META['archive_remote'],source_archive_sha256=META['archive_sha256'])
 with tempfile.NamedTemporaryFile(mode='w',dir=r,prefix='source_revision_019_validation_',suffix='.pending',delete=False) as f:
  json.dump(revision,f);pending=pathlib.Path(f.name)
 try:validate_source_revision(r,command,job,pending)
 finally:pending.unlink()
 predecessor=dict(schema='task-source-repair-predecessor-019-v1',prepared_at_utc=datetime.now(timezone.utc).isoformat(),source_commit=META['source_commit'],previous_source_commit=META['previous_source_commit'],evidence_sha256=EVIDENCE_SHA,job=job,campaign_state=state,consumption=account,
  file_sha256={name:sha(r/name) for name in ('job.json','train/campaign_state.json','train/consumption.json')},actual_parents=bound,expected_critic_warmup=0,qualification=None)
 proof=r/'predecessor_019.json'
 if proof.exists():
  prior=read(proof);assert all(prior[k]==predecessor[k] for k in predecessor if k!='prepared_at_utc')
 else:
  with proof.open('x') as f:json.dump(predecessor,f,indent=2);f.write(chr(10))
 path=r/'source_revision_019.json'
 if path.exists():assert read(path)==revision
 else:
  with path.open('x') as f:json.dump(revision,f,indent=2);f.write(chr(10))
 print(json.dumps(dict(status='prepared_same_v18_source_only',revision=revision,predecessor=read(proof))))
'''

LAUNCH = r'''
import fcntl,shlex,subprocess
from datetime import datetime,timezone
with (r/'source_repair_019.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/'source_repair_launch_019.json').exists(),'Launch receipt exists; inspect instead of retrying'
 assert not (r/'job.resume_019.log').exists(),'Ambiguous prior launch log; inspect'
 assert read(r/'source_revision_019.json')==REVISION and read(r/'predecessor_019.json')==PREDECESSOR
 assert PREDECESSOR['evidence_sha256']==EVIDENCE_SHA
 for name,checksum in PREDECESSOR['file_sha256'].items():assert sha(r/name)==checksum
 job,state,account,bound,old=authenticate_quiescent()
 assert job==PREDECESSOR['job'] and state==PREDECESSOR['campaign_state'] and account==PREDECESSOR['consumption']
 assert bound==PREDECESSOR['actual_parents'],'Prepared full parent no longer latest or immutable'
 validate_source_revision(r,REVISION['command'],job,r/'source_revision_019.json')
 socket=BASE+'/tmux.sock'
 assert subprocess.run(['tmux','-S',socket,'has-session','-t',SESSION],capture_output=True).returncode!=0,'r19 exists'
 wrapper=[BASE+'/env/bin/python','-B',str(src/'scripts/chassis_remote_job.py'),'--run-root',str(r),'--resume-existing','--source-revision-receipt',str(r/'source_revision_019.json'),'--',*REVISION['command']]
 inner='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && '+shlex.join(wrapper)+' > '+shlex.quote(str(r/'job.resume_019.log'))+' 2>&1'
 receipt=dict(schema='task-source-repair-launch-019-v1',status='launch_admitted_pending_tmux',admitted_at_utc=datetime.now(timezone.utc).isoformat(),tmux=SESSION,source_commit=META['source_commit'],source_archive_sha256=META['archive_sha256'],plan_sha256=META['plan_sha256'],run_root=str(r),source_root=str(src),command=REVISION['command'],wrapper_command=wrapper,evidence_sha256=EVIDENCE_SHA,parent=bound['recovery'],expected_critic_warmup=0,qualification=None)
 path=r/'source_repair_launch_019.json'
 with path.open('x') as f:json.dump(receipt,f,indent=2);f.write(chr(10))
 result=subprocess.run(['tmux','-S',socket,'new-session','-d','-s',SESSION,'-c',str(src),'bash','-lc',inner],capture_output=True,text=True)
 receipt.update(status='launched_waiting_for_real_restore_and_PPO' if result.returncode==0 else 'tmux_launch_failed_inspect_before_retry',tmux_exit_code=result.returncode,tmux_stderr=result.stderr[-2000:],launched_at_utc=datetime.now(timezone.utc).isoformat())
 pending=path.with_suffix('.tmp');pending.write_text(json.dumps(receipt,indent=2)+chr(10));pending.replace(path)
 print(json.dumps(receipt));assert result.returncode==0,result.stderr
'''

READBACK = r'''
# Read-only inspection: this step creates no receipt/lock/log and cannot launch.
revision=read(r/'source_revision_019.json');predecessor=read(r/'predecessor_019.json')
assert revision['schema']=='task-source-revision-v1' and revision['source_commit']==META['source_commit']
assert revision['source_archive_sha256']==META['archive_sha256'] and sha(pathlib.Path(revision['source_archive']))==META['archive_sha256']
assert revision['old_contract_sha256']==revision['new_contract_sha256']==META['plan_sha256']
assert predecessor['evidence_sha256']==EVIDENCE_SHA
for domain,bound in predecessor['actual_parents'].items():
 cp=pathlib.Path(bound['path']);assert sha(cp)==bound['sha256']
 assert sha(cp.with_suffix('.checkpoint.json'))==bound['seal_sha256'] and sha(checkpoint_contract_path(cp))==bound['contract_sha256']
job=read(r/'job.json');launch_path=r/'source_repair_launch_019.json';launch=read(launch_path) if launch_path.exists() else None
if launch is None:
 assert job==predecessor['job'] and sha(r/'job.json')==revision['previous_job_sha256']
else:
 assert launch['source_commit']==META['source_commit'] and launch['evidence_sha256']==EVIDENCE_SHA
 if job!=predecessor['job']:
  assert job['command']==revision['command']
  assert sha(r/job['previous_job'])==revision['previous_job_sha256']
  assert read(r/job['source_revision_receipt'])==revision
state=read(r/'train/campaign_state.json');account=read(r/'train/consumption.json')
assert state['contract_sha256']==account['contract_sha256']==META['plan_sha256']
assert account['consumed_updates']>=predecessor['consumption']['consumed_updates']
assert account['actor_updates_consumed']>=predecessor['consumption']['actor_updates_consumed']
observed_owners=owners()
assert sum(entry([x.encode() for x in o['command']]).name=='run_task_campaign.py' for o in observed_owners)<=1,'Duplicate supervisor'
assert sum(entry([x.encode() for x in o['command']]).name=='train_chassis.py' for o in observed_owners)<=1,'Duplicate training worker'
assert sum(entry([x.encode() for x in o['command']]).name=='chassis_remote_job.py' for o in observed_owners)<=1,'Duplicate wrapper'
if launch is not None and job!=predecessor['job']:
 for owner in observed_owners:
  assert entry([x.encode() for x in owner['command']]).resolve().is_relative_to(src.resolve()),'Owner runs an unexpected source'
print(json.dumps(dict(status='read_only_source_repair_inspection_not_PPO_verification',revision=revision,predecessor=predecessor,launch=launch,job=job,campaign_state=state,consumption=account,owners=observed_owners,latest_parents={row['name']:str(latest(row['name'])) for row in read(src/'contracts/v6_task_campaign_v18.json')['domains']},qualification=None)))
'''


def remote_code(mode, evidence, revision=None, predecessor=None):
    values = dict(RUN=RUN, BASE=BASE, SESSION=SESSION, META=evidence["source"], EVIDENCE_SHA=evidence["evidence_sha256"])
    if mode == "launch":
        assert revision["schema"] == "task-source-revision-v1"
        assert revision["source_commit"] == predecessor["source_commit"] == COMMIT
        assert revision["source_archive_sha256"] == evidence["source"]["archive_sha256"]
        assert revision["new_source_root"] == evidence["source"]["source_remote"]
        assert revision["old_source_root"] == RUN + "/source_4ba339299c20"
        assert revision["source_archive"] == evidence["source"]["archive_remote"]
        assert revision["new_contract_sha256"] == revision["old_contract_sha256"] == evidence["source"]["plan_sha256"]
        assert predecessor["evidence_sha256"] == evidence["evidence_sha256"]
        values.update(REVISION=revision, PREDECESSOR=predecessor)
    return "\n".join(name + "=" + repr(value) for name, value in values.items()) + "\n" + COMMON + {"prepare": PREPARE, "launch": LAUNCH, "readback": READBACK}[mode]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "readback"))
    parser.add_argument("--emit-remote-code", action="store_true", required=True, help="Output Python only; never execute it")
    parser.add_argument("--source-candidate", type=Path, default=OUT / "source_candidate_019.json")
    parser.add_argument("--clean-tests", type=Path, default=OUT / "clean_tests_019.json")
    parser.add_argument("--clean-30-roles", type=Path, default=OUT / "clean_30_roles_019.json")
    parser.add_argument("--native-recovery", type=Path, default=OUT / "native_recovery_019.json")
    parser.add_argument("--revision", type=Path, default=OUT / "source_revision_019.json")
    parser.add_argument("--predecessor", type=Path, default=OUT / "predecessor_019.json")
    args = parser.parse_args()
    evidence = load_evidence(dict(source_candidate=args.source_candidate, clean_tests=args.clean_tests, clean_30_roles=args.clean_30_roles, native_recovery=args.native_recovery))
    print(remote_code(args.mode, evidence, read(args.revision) if args.mode == "launch" else None, read(args.predecessor) if args.mode == "launch" else None))


if __name__ == "__main__":
    main()
