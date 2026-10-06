#!/usr/bin/env python3
"""Emit one authenticated supervisor SIGTERM; emitter never connects or signals."""
import argparse
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
PREVIOUS = '723732adae37f12f8488f9537717664415a98d10'
RUN = '/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358'
REMOTE = r'''
import datetime,fcntl,hashlib,json,os,pathlib,signal
r=pathlib.Path(RUN);old=r/'source_723732adae37'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def entry(a):
 for x in a[1:]:
  if x in ('-c','-m'):return None
  if x.endswith('.py'):return pathlib.Path(x)
def process(pid):
 p=pathlib.Path('/proc')/str(pid)
 args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 stat=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,ppid=int(stat[1]),starttime_ticks=int(stat[19]),command=args)
with (r/'recovery_entropy_023.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (r/'stop_requested_023.json').exists(),'Already requested: inspect, do not repeat'
 assert not (r/'recovery_entropy_launch_023.json').exists(),'Already dispatched'
 assert sha(pathlib.Path(META['archive_remote']))==META['archive_sha256'],'Candidate upload not authenticated'
 job=read(r/'job.json');revision=read(r/'source_revision_022.json')
 assert job['command']==revision['command'] and job.get('source_revision_receipt') is None
 assert revision['schema']=='task-plan-revision-v20' and revision['source_commit']==PREVIOUS
 assert pathlib.Path(revision['new_source_root'])==old
 assert sha(pathlib.Path(revision['source_archive']))==revision['source_archive_sha256']
 resumed=read(r/'same_contract_recovery_20261007_002.json')
 assert resumed['schema']=='source022-same-contract-failed-physical-recovery-v1' and resumed['qualification'] is None
 assert resumed['source_commit']==PREVIOUS and resumed['new_critic_warmup']==0
 assert resumed['wrapper_command'][resumed['wrapper_command'].index('--')+1:]==job['command']
 assert resumed['session'].endswith('-r22-recovery002')
 assert sha(pathlib.Path(resumed['runtime_shell']))==resumed['runtime_shell_sha256']=='c76cc88f7273a400430ded6251ebc32d984a36fed366733a27ed76e2c94dcff7'
 kinds={'chassis_remote_job.py':'wrapper','run_task_campaign.py':'supervisor','train_chassis.py':'worker','evaluate_chassis.py':'evaluator'}
 owners=[]
 for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:
   item=process(int(p.parent.name));e=entry(item['command'])
   if e and e.name in kinds and any(x==str(r) or x==str(r/'train') or x.startswith(str(r/'train')+'/') for x in item['command']):
    assert e.parent==old/'scripts','Different source owner'
    item.update(kind=kinds[e.name],entry=str(e));owners.append(item)
  except OSError:pass
 assert sorted(x['kind'] for x in owners)==['supervisor','worker','wrapper'],'Duplicate/missing/evaluation owner'
 by={x['kind']:x for x in owners}
 assert by['worker']['ppid']==by['supervisor']['pid'] and by['supervisor']['ppid']==by['wrapper']['pid']
 assert by['supervisor']['command']==job['command'] and by['supervisor']['pid']==job['pid']
 assert by['wrapper']['command']==resumed['wrapper_command']
 assert {x['pid']:x['starttime_ticks'] for x in owners}==EXPECTED_STARTTIMES,'Owner changed since trusted bounded read'
 worker=by['worker'];command=worker['command']
 assert command[command.index('--run-dir')+1]==str(r/'train/recovery/attempt_008/train')
 assert command[command.index('--contract')+1]==str(old/'contracts/v6_task_campaign_recovery_worker_v20.json')
 invocation=read(r/'train/recovery/attempt_008/invocation.json')
 assert invocation['command']==command
 assert invocation['source_checkpoint']==resumed['parent_checkpoint'] and invocation['source_checkpoint_sha256']==resumed['parent_checkpoint_sha256']
 assert command[command.index('--resume')+1]==resumed['parent_checkpoint']
 assert int(command[command.index('--consumed-updates')+1])==resumed['parent_consumption']
 assert int(command[command.index('--actor-updates-consumed')+1])==resumed['parent_actor_consumption']
 assert int(command[command.index('--task-actor-update')+1])==resumed['parent_task_actor_update']
 progress=read(r/'train/recovery/attempt_008/train/progress.json')
 assert progress['pid']==worker['pid'] and progress['optimizer_phase']=='actor_and_critic'
 assert progress['successful_updates']==progress['actor_updates_in_block'] and progress['successful_updates']>0
 assert read(r/'train/campaign_state.json')['status']=='running'
 assert resumed['plan_sha256']==sha(old/'contracts/v6_task_campaign_v20.json')
 receipt=dict(schema='source023-authenticated-graceful-stop-request-v1',admission_sha256=ADMISSION_SHA,
   previous_source_commit=PREVIOUS,requested_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   signal='SIGTERM',target_pid=by['supervisor']['pid'],target='supervisor_only',owners=owners,progress=progress,
   source_commit=META['source_commit'],archive_sha256=META['archive_sha256'],qualification=None,refund=False)
 with (r/'stop_requested_023.json').open('x') as f:
  json.dump(receipt,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
 directory=os.open(r,os.O_DIRECTORY)
 try:os.fsync(directory)
 finally:os.close(directory)
 latest=process(by['supervisor']['pid'])
 assert latest['starttime_ticks']==by['supervisor']['starttime_ticks'] and latest['command']==job['command'],'Supervisor identity changed before signal'
 os.kill(by['supervisor']['pid'],signal.SIGTERM)
 print(json.dumps(receipt))
'''


def emit(meta, admission, live_read, admission_sha):
    assert meta['previous_source_commit'] == PREVIOUS and meta['contract_file'] == 'contracts/v6_task_campaign_v21.json'
    assert admission['status'] == 'passed' and admission['qualification'] is None
    assert admission['source_commit'] == meta['source_commit'] and admission['archive_sha256'] == meta['archive_sha256']
    assert live_read['qualification'] is None
    expected = {x['pid']: x['starttime_ticks'] for x in live_read['owners']}
    assert len(expected) == 3 and all(type(k) is int and type(v) is int and v > 0 for k, v in expected.items())
    values = dict(RUN=RUN, PREVIOUS=PREVIOUS, META=meta, ADMISSION_SHA=admission_sha, EXPECTED_STARTTIMES=expected)
    return '\n'.join(k + '=' + repr(v) for k, v in values.items()) + '\n' + REMOTE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--emit-remote-code', action='store_true', required=True)
    p.add_argument('--source', type=Path, default=OUT/'source_candidate_023.json')
    p.add_argument('--admission', type=Path, default=OUT/'admission_023.json')
    p.add_argument('--owner-read', type=Path, default=OUT.parent/'failure_audit_20261007/attempt008_live_read002.json')
    a = p.parse_args()
    print(emit(json.loads(a.source.read_text()), json.loads(a.admission.read_text()),
               json.loads(a.owner_read.read_text()), hashlib.sha256(a.admission.read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
