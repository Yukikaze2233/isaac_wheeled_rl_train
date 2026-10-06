from pathlib import Path
import hashlib,json,os,subprocess,time
out=Path(__file__).resolve().parent;m=json.loads((out/'source_candidate_021.json').read_text());root=Path(m['clean_root']);inventory=json.loads((out/'frozen_source_inventory_021.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/n)==v for n,v in inventory.items())
old=json.loads((out.parent/'source_repair_020/clean_tests_020.json').read_text());cmd=old['command']+['tests/test_recovery_quality_revision.py'];start=time.monotonic()
with (out/'clean_tests_021.stdout.log').open('x') as f:
 q=subprocess.run(cmd,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')+':'+str(root/'scripts'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT)
unchanged=all(sha(root/n)==v for n,v in inventory.items());receipt=dict(status='passed' if q.returncode==0 and unchanged else 'failed',source_commit=m['source_commit'],archive_sha256=m['archive_sha256'],exitcode=q.returncode,duration_seconds=time.monotonic()-start,command=cmd,source_files_unchanged=unchanged,exact_bytes_match_archive=unchanged,formal_updates_consumed=0,qualification=None)
(out/'clean_tests_021.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(q.returncode or int(not unchanged))
