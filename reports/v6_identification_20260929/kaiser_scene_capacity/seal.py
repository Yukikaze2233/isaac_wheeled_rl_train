import hashlib,json,pathlib,collections,statistics,math
p=pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-capacity-check-20260929')
d=json.loads((p/'scene/completion.json').read_text()); r=json.loads((p/'resources.json').read_text())
s=d['startup']; metrics=d['metrics']; launch=json.loads((p/'launch.json').read_text())
result={k:d[k] for k in ['status','started_at','finished_at','contract_id','contract_sha256','asset_manifest_sha256','control_math_sha256','source_sha256','successful_updates','scene_capacity']}
result['launch']=launch
result['startup']={k:v for k,v in s.items() if k not in ['contact_filters','terrain_families','terrain_collision_paths','mass_kg_each']}
for key in ['contact_filters','terrain_families','terrain_collision_paths']:
 result['startup'][key+'_count']=len(s[key])
result['startup']['mass_kg_range']=[min(s['mass_kg_each']),max(s['mass_kg_each'])]
result['metrics']={k:v for k,v in metrics.items() if k!='terrain_families'}
result['metrics']['terrain_family_counts']=dict(collections.Counter(metrics['terrain_families']))
# Keep per-environment arrays in the hashed raw completion, not the compact receipt.
for section,key in [('startup','solver_order'),('metrics','final_relative_height_m'),('metrics','final_filtered_wheel_force_n')]:
 values=result[section].pop(key)
 columns=list(zip(*values)) if values and isinstance(values[0],list) else [values]
 result[section][key+'_summary']={'environments':len(values),'columns':[{'min':min(c),'max':max(c),'mean':statistics.fmean(c),'all_finite':all(math.isfinite(v) for v in c)} for c in columns]}

result['resources']={k:v for k,v in r.items() if k not in ['samples','command','inputs_sha256']}
result['resources']['sample_count']=len(r['samples'])
result['resources']['min_host_available_ram_kib']=min(t['host_available_ram_kib'] for t in r['samples'] if 'host_available_ram_kib' in t)
result['torque_monitor']=json.loads((p/'scene/torque_monitor.json').read_text())
result['raw_artifacts']={}
for name in ['scene/completion.json','scene/startup.json','scene/torque_monitor.json','scene.log','launch.json','resources.json','run.py']:
 f=p/name
 result['raw_artifacts'][name]={'remote_path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
result['scope_notes']=['Physics scene capacity only: no PPO allocation or optimizer updates.', 'Non-wheel-contact terminations require separate behavior/collision review.', 'Functional transport/randomization probes were not run; modules are explicitly disabled in this probe.']
(p/'capacity_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','scene_capacity','resources']},indent=2))
