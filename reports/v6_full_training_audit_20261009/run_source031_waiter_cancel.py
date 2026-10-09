#!/usr/bin/env python3
"""Transport one frozen observer-cancellation helper; never signal training."""
import argparse, hashlib, json, shlex, subprocess, time
from pathlib import Path
REPORT=Path(__file__).resolve().parent
RUN=Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
PY='/home/kaiser/robot-rl-sim60/env/bin/python'
SSH=['ssh','-p','2222','-oBatchMode=yes','-oConnectTimeout=10','kaiser@192.168.64.234']
HELPER=REPORT/'source031_queue/cancel_waiting_source030.py'
HELPER_SHA='66b506724fb8a076edd20ddaeeaa1c3f6ea475956ba688db0d4d15711615e10d'
PEER=REPORT/'source031_queue_transition_independent_review001.json'
PEER_SHA='425439226a2f14581a231c9f76a096c7fa027219ddfa8df304a6685b233d5d80'
REMOTE=RUN/'cancel_waiting_source030_66b50672.py'
STAGE_SHA='b7dae15f3878aa92039580387bc98e526078fc4f617064a35c18da8994f93828'
PREFLIGHT_SHA='9c14b08e90dc4eb7a8f3933ba29035e025b3b36083cae37de64002445bc9a83f'
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=['transfer','dryrun','apply']);args=a.parse_args()
    assert digest(HELPER)==HELPER_SHA and digest(PEER)==PEER_SHA
    prefix=REPORT/('source031_waiter_cancel_'+args.mode+'001')
    assert not any(prefix.with_suffix(s).exists() for s in ['.intent.json','.stdout','.stderr','.json','.execution.json'])
    frozen=subprocess.check_output(['git','log','-1','--format=%H','--',str(Path(__file__).resolve())],cwd=REPORT,text=True).strip()
    assert subprocess.check_output(['git','show',frozen+':'+str(Path(__file__).resolve().relative_to(REPORT.parent.parent))],cwd=REPORT)==Path(__file__).read_bytes()
    if args.mode=='apply':
        dry=json.loads((REPORT/'source031_waiter_cancel_dryrun001.json').read_text())
        assert dry['status']=='validated_waiting_source030_not_cancelled'
        ex=json.loads((REPORT/'source031_waiter_cancel_dryrun001.execution.json').read_text())
        assert ex['status']=='completed' and ex['exit_code']==0
    intent=dict(mode=args.mode,helper_sha256=HELPER_SHA,peer_sha256=PEER_SHA,launcher_commit=frozen,launcher_sha256=digest(__file__),training_signals_by_transport=0,qualification=None)
    with prefix.with_suffix('.intent.json').open('x') as f:json.dump(intent,f,indent=2);f.write('\n')
    code="import hashlib,json,sys\nfrom pathlib import Path\np=Path("+repr(str(REMOTE))+")\n"
    if args.mode=='transfer':
        code+="assert not p.exists()\np.parent.mkdir(exist_ok=True)\nwith p.open('xb') as f:f.write("+repr(HELPER.read_bytes())+")\np.chmod(0o644)\n"
        code+="with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=="+repr(HELPER_SHA)+"\nprint(json.dumps(dict(status='frozen_waiter_cancel_helper_transferred',sha256="+repr(HELPER_SHA)+")))\n"
    else:
        code+="assert not p.is_symlink() and p.is_file()\nwith p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=="+repr(HELPER_SHA)+"\n"
        command=[str(REMOTE),'--run-root',str(RUN),'--new-stage-sha256',STAGE_SHA,'--new-preflight-sha256',PREFLIGHT_SHA]+(['--apply'] if args.mode=='apply' else [])
        code+="import runpy\nsys.argv="+repr(command)+"\nrunpy.run_path(str(p),run_name='__main__')\n"
    start=time.monotonic();r=subprocess.run([*SSH,shlex.join(['env','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',PY,'-B','-'])],input=code,text=True,capture_output=True,timeout=180)
    prefix.with_suffix('.stdout').write_text(r.stdout);prefix.with_suffix('.stderr').write_text(r.stderr)
    prefix.with_suffix('.execution.json').write_text(json.dumps(dict(status='completed' if r.returncode==0 else 'failed_retained_no_retry',exit_code=r.returncode,wall_seconds=time.monotonic()-start,stdin_sha256=hashlib.sha256(code.encode()).hexdigest(),training_signals_by_transport=0,qualification=None),indent=2)+'\n')
    if r.returncode:raise RuntimeError('Action refused or failed; inspect the same handles and retained evidence, never retry')
    d=json.loads(r.stdout);prefix.with_suffix('.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(mode=args.mode,status=d['status'],training_signals=d.get('training_signals',0))))
if __name__=='__main__':main()
