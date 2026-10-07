import pathlib,json,datetime,hashlib
r=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_009/train')
rows=[]
for p in sorted((r/'checkpoints').glob('update_*'))[-3:]:
 rows.append({'directory':str(p),'files':[{ 'name':f.name,'bytes':f.stat().st_size} for f in p.iterdir() if f.is_file()], 'metadata':{f.name:json.loads(f.read_text()) for f in p.glob('*.json') if f.stat().st_size<250000}})
p=r/'progress.json'
x = {'read_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'progress': json.loads(p.read_text()), 'latest_checkpoints': rows, 'PPO_actions': 0, 'qualification': None}

import subprocess
argv=['tmux','-S','/home/kaiser/robot-rl-sim60/tmux.sock','list-panes','-a','-F','#{session_name}\t#{window_index}\t#{pane_id}\t#{pane_pid}\t#{pane_current_command}\t#{pane_dead}']
tm=subprocess.run(argv,capture_output=True,text=True,timeout=10)
x['tmux']={'argv':argv,'exit_code':tm.returncode,'stdout':tm.stdout,'stderr':tm.stderr}
if rows:
 d=pathlib.Path(rows[-1]['directory']);checks={}
 for f in sorted(d.iterdir()):
  if f.is_file() and (f.name=='model.pt' or (f.suffix=='.json' and f.stat().st_size<250000)):
   a=f.stat();raw=f.read_bytes();b=f.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
   checks[f.name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'stat_unchanged':True,'inode':a.st_ino,'mtime_ns':a.st_mtime_ns}
 x['latest_seal_file_checks']=checks
x['scope']='latest three immutable checkpoint metadata; SHA/stat only latest; tmux list read-only; no tensor validation or qualification'
x['remote_writes']=0
print(json.dumps(x))
