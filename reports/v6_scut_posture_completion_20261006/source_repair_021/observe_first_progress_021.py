#!/usr/bin/env python3
"""Read-only bounded copy of native first-update bookkeeping, never a trainer."""
import argparse
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout-seconds',type=int,default=900);a=p.parse_args()
assert a.source_root.is_absolute() and a.run_root.is_absolute() and 1<=a.timeout_seconds<=900
assert not a.output.exists();a.output.mkdir();start=time.monotonic();rows={};worker=None
while time.monotonic()-start<a.timeout_seconds:
 matches=[]
 for path in (a.run_root/'train/recovery').glob('attempt_*/invocation.json'):
  try:
   q=json.loads(path.read_text());cmd=q.get('command',[])
   if str(a.source_root/'scripts/train_chassis.py') in cmd:matches.append(path.parent/'train')
  except (OSError,json.JSONDecodeError):continue
 assert len(matches)<=1,'Duplicate source021 worker invocations'
 if matches:
  if worker is None:worker=matches[0]
  assert worker==matches[0]
  path=worker/'progress.json'
  try:raw=path.read_bytes();value=json.loads(raw)
  except (OSError,json.JSONDecodeError):value={}
  update=value.get('successful_updates')
  if type(update) is int and 1<=update<=11 and update not in rows:
   target=a.output/f'update_{update:02d}.json';target.write_bytes(raw)
   rows[update]=dict(path=str(target),sha256=hashlib.sha256(raw).hexdigest(),successful_updates=update,captured_at_utc=datetime.now(timezone.utc).isoformat(),source_progress=str(path))
   print(json.dumps(dict(captured_update=update,optimizer_phase=value.get('optimizer_phase'),task_actor_update=value.get('task_actor_update'))),flush=True)
  index=dict(schema='source021-native-first-progress-copies-v1',status='complete' if set(rows)==set(range(1,12)) else 'collecting',source_root=str(a.source_root),worker=str(worker),rows=[rows[n] for n in sorted(rows)],qualification=None,PPO_actions=0,read_only=True)
  tmp=a.output/'index.tmp';tmp.write_text(json.dumps(index,indent=2)+'\n');tmp.replace(a.output/'index.json')
  if type(update) is int and update>=11:break
 time.sleep(1)
index=dict(schema='source021-native-first-progress-copies-v1',status='complete' if set(rows)==set(range(1,12)) else 'incomplete',source_root=str(a.source_root),worker=str(worker) if worker else None,rows=[rows[n] for n in sorted(rows)],qualification=None,PPO_actions=0,read_only=True)
tmp=a.output/'index.tmp';tmp.write_text(json.dumps(index,indent=2)+'\n');tmp.replace(a.output/'index.json');print(json.dumps(dict(status=index['status'],captured=len(rows))),flush=True)
