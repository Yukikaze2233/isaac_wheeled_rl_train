import sys,json,pathlib,hashlib
p=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-pair-fit-20260929')
sys.path.insert(0,str(p/'diagnostic-fix-v1/scripts'))
from identify_pair_dynamics import Replay
bundle=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-capacity-check-20260929/frozen/model/纯底盘_v6_232mm/urdf')
out={'scope':'restore audit pose after tangent finite difference; forward dynamics unchanged','script_sha256':hashlib.sha256((p/'diagnostic-fix-v1/scripts/identify_pair_dynamics.py').read_bytes()).hexdigest(),'sides':{}}
for side,run in [('left','run_20260928T073044Z'),('right','run_20260928T144606Z')]:
 r=Replay(p/run,side=side,physics_dt=.001,bundle=bundle,coordinate_sign=-1.)
 rows=r.static_load_audit()
 old=json.loads((p/(side+'-final-asset-1ms')/'static_load.json').read_text())
 if isinstance(old,dict): print('dict_keys',old.keys()); raise ValueError('Unexpected prior artifact shape')
 prior={s['segment']:s for s in old}
 delta=[row['knee_inner_deg']-prior[row['segment']]['knee_inner_deg'] for row in rows]
 other=[max(abs(a-b) for a,b in zip(row['predicted_hold_nm'],prior[row['segment']]['predicted_hold_nm'])) for row in rows]
 out['sides'][side]={'segment_count':len(rows),'max_inner_angle_delta_deg':max(map(abs,delta)),'max_predicted_load_delta_nm':max(other),'segment0':next(row for row in rows if row['segment']==0)}
 assert max(map(abs,delta))<.001 and max(other)<1e-9
 (p/'diagnostic-fix-v1'/f'{side}-static_load.json').write_text(json.dumps(rows,indent=2)+'\n')
(p/'diagnostic-fix-v1/verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
