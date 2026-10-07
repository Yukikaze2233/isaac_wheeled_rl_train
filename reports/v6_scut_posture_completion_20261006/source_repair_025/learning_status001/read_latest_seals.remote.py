import pathlib,json,datetime,hashlib
r=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_009/train')
rows=[]
for p in sorted((r/'checkpoints').glob('update_*'))[-3:]:
 rows.append({'directory':str(p),'files':[{ 'name':f.name,'bytes':f.stat().st_size} for f in p.iterdir() if f.is_file()], 'metadata':{f.name:json.loads(f.read_text()) for f in p.glob('*.json') if f.stat().st_size<250000}})
p=r/'progress.json'
print(json.dumps({'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'progress':json.loads(p.read_text()),'latest_checkpoints':rows,'PPO_actions':0,'qualification':None}))
