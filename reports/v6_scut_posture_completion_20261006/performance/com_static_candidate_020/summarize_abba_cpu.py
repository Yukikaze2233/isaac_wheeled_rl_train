#!/usr/bin/env python3
"""CPU-only descriptive summary after four authenticated terminal arms."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
TARGET=Path('/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/performance/com_static_candidate_020')
ORDER=['A1_baseline','B1_candidate','B2_candidate','A2_baseline']
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def binding(p):
 p=Path(p)
 return {'path':str(p),'size_bytes':p.stat().st_size,'sha256':digest(p)}
def main():
 arms=[];shared=None
 for label in ORDER:
  invocation=HERE/'invocations'/(label+'.invocation.json');i=read(invocation)
  p=HERE/'native'/label/'report.json';s=p.with_name('root_com_candidate_receipt.json')
  review=HERE/'reviews'/(label+'.review.json')
  if not p.is_file() or read(p).get('status') not in ('engineering_probe_completed','failed','failed_source_changed'):
   print('Not all arms terminal; summary not written: '+label);return 2
  if not review.is_file(): print('Missing independent review; summary not written: '+label);return 2
  r,x,v=read(p),read(s),read(review)
  assert v['status']=='passed' and v['errors']==[], label+' independent review failed'
  assert v['report_sha256']==digest(p) and v['receipt_sha256']==digest(s), label+' reviewed evidence changed'
  assert v['invocation_sha256']==digest(invocation),label+' invocation changed'
  assert v['mode']==i['mode']=='throughput' and v['variant']==i['variant']
  assert r['status']=='engineering_probe_completed' and x['mode_complete'] is True
  assert x['written_before_app_close'] is True and x['throughput_contains_vendor_comparison'] is False
  assert r['num_envs']==4096 and r['policy_steps']==600 and r['physics_steps']==2400
  assert r['formal_updates_consumed']==r['optimizer_steps']==0 and r['qualification'] is None
  assert x['measured_env_steps']==600 and x['warmup_steps_excluded']==50
  assert x['driver_sha256']==i['driver']['sha256']==digest(HERE/'native_driver.py')
  assert x['patch_manifest_sha256']==i['patch_manifest']['sha256']==digest(HERE/'patch_manifest.json')
  common={k:i[k] for k in ['protocol','environment','driver','patch_manifest','checkpoint','contract_bindings','source_base_commit','source_patches']}
  if shared is None:shared=common
  else:assert common==shared,label+' trial protocol/authentication differs'
  # Parent invocation copies are part of the evidence chain when present.
  copied=TARGET/(label+'.invocation.json')
  if copied.exists():assert digest(copied)==digest(invocation),label+' parent invocation copy differs'
  arm={'name':label,'variant':i['variant'],'probe_loop_wall_seconds':r['wall_seconds'],
       'warmed_env_step_median_s':x['warmed_env_step_median_s'],'warmed_env_step_p95_s':x['warmed_env_step_p95_s'],
       'warmed_env_step_samples':x['measured_env_steps']-x['warmup_steps_excluded'],
       'process_wall_seconds_excluded':x['separate_process_wall_seconds'],
       'source_verified_file_count':v['source_file_pins_verified'],
       'bindings':{'invocation':binding(invocation),'report':binding(p),'preclose_receipt':binding(s),'CPU_review':binding(review)}}
  post=s.with_name('root_com_candidate_postclose_receipt.json')
  if post.exists():arm['bindings']['optional_postclose_receipt']=binding(post)
  arms.append(arm)
 a=[x for x in arms if x['variant']=='baseline'];b=[x for x in arms if x['variant']=='candidate']
 assert len(a)==len(b)==2
 def stats(rows,key):
  values=[x[key] for x in rows]
  return {'values':values,'mean':statistics.mean(values),'min':min(values),'max':max(values),
          'within_variant_range':max(values)-min(values)}
 metrics={}
 for key in ['probe_loop_wall_seconds','warmed_env_step_median_s','warmed_env_step_p95_s']:
  sa,sb=stats(a,key),stats(b,key)
  metrics[key]={'baseline':sa,'candidate':sb,'candidate_reduction_fraction':(sa['mean']-sb['mean'])/sa['mean'],
                'baseline_over_candidate_ratio':sa['mean']/sb['mean'],
                'candidate_below_both_baseline_samples':sb['max']<sa['min'],
                'observed_ranges_overlap':not(sb['max']<sa['min'] or sa['max']<sb['min'])}
 result={'schema':'com-static-local-native-ABBA-paired-review-v1','status':'passed_native_protocol_only',
         'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'order':ORDER,'arms':arms,'metrics':metrics,
         'authenticated_shared_protocol':shared,
         'prerequisite_semantic_review':binding(HERE/'reviews'/'semantic_001.review.json'),
         'patch_manifest':binding(HERE/'patch_manifest.json'),'verified_source_bytes':binding(HERE/'verified_sources.json'),
         'reviewer_source':binding(__file__),
         'primary_measure':'native probe loop wall_seconds includes actor/structural audit and 600 policy steps; excludes app startup',
         'secondary_measure':'mean of two per-trial env.step median/p95 values after excluding first 50/600 steps; not pooled sample statistics',
         'excluded':['semantic vendor-comparison run','app/asset startup','archive/cold-stop','full PPO/Adam/logger time'],
         'statistical_limit':'Two samples per variant are descriptive only; no confidence interval or general throughput guarantee is inferred.',
         'local_native_measurement_only':True,'native_or_GPU_launched_by_reviewer':False,
         'Kaiser_PPO_speedup':None,'PPO_updates':0,'qualification':None}
 TARGET.mkdir(parents=True,exist_ok=True)
 destination=TARGET/'paired_summary_independent_CPU.json'
 if destination.exists(): raise RuntimeError('Refusing to overwrite prior paired summary: '+str(destination))
 destination.write_text(json.dumps(result,indent=2)+'\n')
 lines=['# COM static 020 local native ABBA independent review','',
        'Four sequential fresh-process arms passed the authenticated native protocol. Each used 4096 environments, 600 policy steps and 2400 physics steps, zero PPO and no qualification. Pre-close terminal sidecars and independent CPU reviews passed.','',
        '| Arm | Probe loop (s) | Warm env.step median (ms) | Warm p95 (ms) |',
        '| --- | ---: | ---: | ---: |']
 for x in arms:lines.append(f"| {x['name']} | {x['probe_loop_wall_seconds']:.6f} | {1000*x['warmed_env_step_median_s']:.6f} | {1000*x['warmed_env_step_p95_s']:.6f} |")
 m=metrics['probe_loop_wall_seconds'];lines+=['',
   f"Baseline mean {m['baseline']['mean']:.6f}s; candidate mean {m['candidate']['mean']:.6f}s. Candidate reduction {100*m['candidate_reduction_fraction']:.3f}% in these two local samples per variant.",
   f"Baseline endpoint spread {m['baseline']['within_variant_range']:.6f}s; candidate spread {m['candidate']['within_variant_range']:.6f}s. Observed ranges overlap: {m['observed_ranges_overlap']}.",'',
   'Semantic comparisons, app startup and archive/cold-stop costs are excluded. Warm summaries use 550 samples per arm and are not pooled distributions. This native probe includes structural audit and inference; it does not measure Kaiser PPO, optimizer or logger throughput. Two samples per variant do not establish a general performance guarantee or skill/deployment qualification.','',
   f"Machine-readable evidence binding: [paired_summary_independent_CPU.json]({destination})."]
 md=TARGET/'paired_summary_independent_CPU.md'
 if md.exists():raise RuntimeError('Refusing to overwrite prior paired Markdown summary')
 md.write_text('\n'.join(lines)+'\n')
 print(json.dumps({'summary':str(destination),'summary_sha256':digest(destination),'Markdown':str(md),
                   'primary_metrics':m,'Kaiser_PPO_speedup':None,'qualification':None},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
