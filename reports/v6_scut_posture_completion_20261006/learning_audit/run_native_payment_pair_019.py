import json,subprocess,os,pathlib,datetime,hashlib,time
base=pathlib.Path(__file__).resolve().parent
inv=json.loads((base/'payment_native_invocation_019.json').read_text())
for n,h in inv['artifact_sha256'].items():
 assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h
record={'schema':'root-native-payment-pair-process-019-v1','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'serial':True,'formal_updates_consumed':0,'qualified':None,'arms':{}}
def save(): (base/'payment_native_process_019.json').write_text(json.dumps(record,indent=2)+'\n')
for arm in ['mean','stochastic']:
 command=inv['commands'][arm];out=pathlib.Path(command[command.index('--output')+1]);assert not out.exists()
 source=pathlib.Path(command[command.index('--source-root')+1]);assert source.is_dir()
 start=time.monotonic()
 with (base/('native_payment_'+arm+'_001.stdout.log')).open('w') as log:
  process=subprocess.Popen(command,cwd=source,env={**os.environ,**inv['environment']},stdout=log,stderr=subprocess.STDOUT)
  record['arms'][arm]={'pid':process.pid,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'status':'running'};save()
  print(arm,'started',process.pid,flush=True)
  rc=process.wait(timeout=1200)
 r=json.loads((out/'report.json').read_text())
 payment=json.loads((out/'payment_summary.json').read_text()) if (out/'payment_summary.json').exists() else {}
 ok=rc==0 and r.get('status')=='native600_completed_preclose' and r.get('actual_payment_complete') is True and payment.get('complete') is True and payment.get('native_lane_intervals')==2457600 and payment.get('event_producer_calls')==600
 ok=ok and not any((out/n).exists() for n in ['payment_observer_failure.json','payment_admission_failure.json','payment_postclose_or_admission_error.json'])
 record['arms'][arm].update(exit_code=rc,wall_seconds=time.monotonic()-start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='native_completed_exact_payment_preclose' if ok else 'failed_no_admission',report_sha256=hashlib.sha256((out/'report.json').read_bytes()).hexdigest(),payment_sha256=hashlib.sha256((out/'payment_summary.json').read_bytes()).hexdigest() if payment else None,app_close_returned_unknown=not(out/'postclose.json').exists())
 save();print(arm,record['arms'][arm]['status'],rc,flush=True)
 if not ok:raise SystemExit(2)
record['status']='two_serial_native_payment_observations_completed_no_PPO';save()
