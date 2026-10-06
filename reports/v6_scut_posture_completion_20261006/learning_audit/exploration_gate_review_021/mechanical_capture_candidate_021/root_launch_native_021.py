#!/usr/bin/env python3
"""Launch one admitted local diagnostic child; never signal or run PPO."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
CANDIDATE_SHA='6c912170f15ab3436e4aad7ed2e522194219134fe123f80cb9377ce851b71812'
PYTHON='/home/yukikaze/isaacsim60-venv/bin/python'
OUT=HERE/'native_capture_001'

def write(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')

def main():
    if OUT.exists():
        raise RuntimeError('Preserve existing native output; no repeat launch')
    spec=importlib.util.spec_from_file_location('root_bound_capture',HERE/'run_capture_021.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.verify_candidate(CANDIDATE_SHA)
    independent=HERE/'independent_review_021_002.json'
    review=json.loads(independent.read_text())
    if review['candidate_manifest_sha256']!=CANDIDATE_SHA or review['status']!='no_confirmed_execution_blocker_in_frozen_candidate_CPU_review' or not review['bound_input_SHA_all_match'] or review['CPU_evidence']['passed']!=26:
        raise RuntimeError('Independent review not passed for this exact candidate')
    pair=mod.load(mod.ORCHESTRATOR,'root_admitted_pair')
    resources=pair.local_GPU_empty()
    command=[PYTHON,'-B',str(HERE/'run_capture_021.py'),'--mode','native','--candidate-sha',CANDIDATE_SHA,'--output',str(OUT)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMNI_KIT_ACCEPT_EULA='YES',PYTHONDONTWRITEBYTECODE='1')
    begun=time.monotonic()
    with (HERE/'native_capture_001.stdout.log').open('x') as stream:
        child=subprocess.Popen(command,cwd=str(HERE.parents[4]),env=env,stdout=stream,stderr=subprocess.STDOUT)
        proc=Path('/proc')/str(child.pid)
        stat=(proc/'stat').read_text()
        starttime=int(stat[stat.rfind(')')+2:].split()[19])
        start=dict(pid=child.pid,command=command,starttime_ticks=starttime,started_at_utc=datetime.now(timezone.utc).isoformat(),
            resources=resources,candidate_manifest_sha256=CANDIDATE_SHA,independent_review_sha256=hashlib.sha256(independent.read_bytes()).hexdigest(),
            root_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),PPO_updates=0,formal_updates_consumed=0,qualification=None)
        write(HERE/'native_capture_001.started.json',start)
        try:
            code=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            write(HERE/'native_capture_001.lifecycle_limit.json',dict(**start,status='root_attention_process_not_signalled',automatic_signal_sent=False))
            raise RuntimeError('600s lifecycle limit; exclusive child was not signalled')
    exit_receipt=dict(**start,process_exit_observed=True,process_exit_code=code,process_wall_seconds=time.monotonic()-begun,
        process_present_after_wait=proc.exists(),finished_at_utc=datetime.now(timezone.utc).isoformat(),App_close_returned=None)
    write(HERE/'native_capture_001.process.json',exit_receipt)
    print(json.dumps({k:exit_receipt[k] for k in ('pid','process_exit_observed','process_exit_code','process_wall_seconds','PPO_updates')},ensure_ascii=False))
    if code:
        raise RuntimeError('Native diagnostic exited nonzero; preserve raw log/trace')

if __name__=='__main__':main()
