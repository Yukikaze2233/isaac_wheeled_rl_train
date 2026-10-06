#!/usr/bin/env python3
"""Emit the exact source022 readonly observation; run only with explicit --mode run.

No launch, stop, model, simulator, GPU, PPO or TensorBoard server. The run mode
copies isolated readers/metadata, invokes one remote CPU reader and preserves
pending/failure output. A transport timeout is unknown; there is no retry.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REPORT=HERE.parent.parent
RUN_ROOT='/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358'
SSH=['ssh','-p','2222','-oBatchMode=yes','-oConnectTimeout=10','kaiser@192.168.64.234']
SCP=['scp','-P','2222','-oBatchMode=yes','-oConnectTimeout=10']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)

def recipe(manifest,observation):
    require(observation.replace('_','').isalnum() and observation.isascii(),'Observation must be ASCII alphanumeric with underscores')
    files={Path(k):v for k,v in manifest['files_sha256'].items()}
    require(all(p.is_file() and digest(p)==h for p,h in files.items()),'Candidate/admission/dependency bytes changed since CPU freeze')
    sources=[HERE/'read_recovery_window_022.py',HERE.parent/'recovery_window_candidate_021/read_recovery_window_021.py',
        HERE.parent/'read_recovery_window_020.py',HERE.parent/'read_recovery_window_019.py',
        REPORT/'source_repair_022/first_PPO_verification_022.readonly002.json',REPORT/'source_repair_022/frozen_source_inventory_022.json',
        REPORT/'source_repair_022/predecessor_022.json',REPORT/'source_repair_022/source_revision_022.json']
    require(all(p in files for p in sources),'Manifest lacks actual reader/admission sources')
    remote='/tmp/readonly_recovery_window_022_'+observation
    output=remote+'/report.json'
    argv=['/home/kaiser/robot-rl-sim60/env/bin/python','-B',remote+'/read_recovery_window_022.py',
        '--reader021',remote+'/read_recovery_window_021.py','--reader020',remote+'/read_recovery_window_020.py','--reader019',remote+'/read_recovery_window_019.py',
        '--launch',RUN_ROOT+'/recovery_quality_launch_022.json','--first-ppo',remote+'/first_PPO_verification_022.readonly002.json',
        '--inventory',remote+'/frozen_source_inventory_022.json','--predecessor',remote+'/predecessor_022.json','--revision',remote+'/source_revision_022.json',
        '--updates','100','--end-update','100','--window-mode','after_critic_baseline','--output',output]
    return dict(schema='source022-readonly-first100-recipe-v1',observation=observation,remote_directory=remote,remote_output=output,
        source_files=[str(p) for p in sources],copied_source_sha256={p.name:files[p] for p in sources},command=argv,
        native_window=dict(parent_consumption30802=30802,critic_warmup=10,end_completed_update=110,end_actor_update=100,
            baseline_event_step=30811,dense_events=[30812,30911],task_actor_updates=[3872,3971],expected_lane_frames=9830400),
        source_commit=manifest['source_commit'],qualification=None,PPO_actions=0,execute_count_limit=1)

def exclusive_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=HERE/'candidate_manifest.json')
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--observation',required=True)
    p.add_argument('--mode',choices=('emit','run'),default='emit');args=p.parse_args(argv)
    require(digest(args.manifest)==args.manifest_sha256,'Reviewed candidate manifest SHA differs')
    manifest=json.loads(args.manifest.read_text());plan=recipe(manifest,args.observation)
    destination=HERE/args.observation;destination.mkdir(exist_ok=False)
    exclusive_json(destination/'recipe.json',plan)
    if args.mode=='emit':
        print(json.dumps(dict(status='emitted_not_executed',directory=str(destination),command=plan['command'],PPO_actions=0)))
        return 0
    started=datetime.now(timezone.utc).isoformat();phase='create_isolated_remote_directory'
    try:
        subprocess.run(SSH+['mkdir '+shlex.quote(plan['remote_directory'])],check=True,timeout=30)
        phase='copy_pinned_reader_bundle'
        subprocess.run(SCP+plan['source_files']+['kaiser@192.168.64.234:'+plan['remote_directory']+'/'],check=True,timeout=45)
        phase='one_remote_CPU_reader'
        result=subprocess.run(SSH+[shlex.join(plan['command'])],text=True,capture_output=True,timeout=60)
        (destination/'stdout.log').write_text(result.stdout);(destination/'stderr.log').write_text(result.stderr)
        phase='preserve_result_bytes'
        subprocess.run(SCP+['kaiser@192.168.64.234:'+plan['remote_output'],str(destination/'report.json')],check=True,timeout=30)
        source_hash_code="""import hashlib,json
from pathlib import Path
root=Path(REMOTE)
print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}))
""".replace('REMOTE',repr(plan['remote_directory']))
        hashes=subprocess.run(SSH+['/home/kaiser/robot-rl-sim60/env/bin/python','-B','-'],input=source_hash_code,text=True,capture_output=True,check=True,timeout=30)
        expected={**plan['copied_source_sha256'],'report.json':digest(destination/'report.json')}
        require(json.loads(hashes.stdout)==expected,'Remote/source/copied report bytes differ')
        report=json.loads((destination/'report.json').read_text())
        receipt=dict(schema='source022-readonly-first100-execution-v1',started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
            command=plan['command'],exit_code=result.returncode,source_and_copied_bytes_exact=True,copied_sha256=expected,
            report_status=report['status'],report_complete=report['complete'],runner_sha256=digest(__file__),qualification=None,PPO_actions=0,
            status='reader_completed_not_qualification',scope='one_existing_native_TB_CPU_reader_no_training_changes_no_GPU_no_PPO')
        exclusive_json(destination/'invocation.json',receipt)
        print(json.dumps(dict(status=report['status'],complete=report['complete'],directory=str(destination),report_sha256=expected['report.json'])))
        return result.returncode
    except Exception as error:
        # No retry/kill/relaunch. SSH timeout may leave a remote reader alive;
        # only subsequent independent readback can establish its final outcome.
        receipt=dict(schema='source022-readonly-first100-execution-v1',status='transport_or_readback_failed_outcome_not_retried',
            started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),phase=phase,error=type(error).__name__+': '+str(error),
            remote_reader_outcome='unknown' if phase=='one_remote_CPU_reader' else 'not_inferred',retry_count=0,PPO_actions=0,qualification=None)
        exclusive_json(destination/'invocation.json',receipt);raise

if __name__=='__main__':raise SystemExit(main())
