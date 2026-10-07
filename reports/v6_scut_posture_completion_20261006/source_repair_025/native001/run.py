import subprocess,json,pathlib,time,sys
base=pathlib.Path(__file__).resolve().parent
root=base.parents[3]
record=json.loads((base.parent/'source_commit.json').read_text())
output=pathlib.Path('/tmp/small-stair-native-v23-001')
argv=['/home/yukikaze/isaacsim60-venv/bin/python','-u',str(base/'probe.py'),'--source',record['clean_source_directory'],'--output',str(output)]
start=time.monotonic()
with (base/'stdout.log').open('w') as out,(base/'stderr.log').open('w') as err:
 process=subprocess.run(argv,stdout=out,stderr=err,cwd=root)
r=json.loads((output/'report.json').read_text())
receipt={'process_exit_code':process.returncode,'native_report_status':r['status'],'wrapper_elapsed_seconds':time.monotonic()-start,'source_commit':record['source_commit'],'argv':argv}
(base/'process_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(base/'actual_report.json').write_bytes((output/'report.json').read_bytes())
print(json.dumps(receipt),flush=True)
sys.exit(0 if r['status']=='native_readback_and_one_interval_passed' else 1)
