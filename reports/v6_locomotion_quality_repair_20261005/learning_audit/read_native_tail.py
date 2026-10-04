#!/usr/bin/env python3
"""Read a bounded native TensorBoard tail; no GPU, evaluator, or TB server."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

BASE = "/home/kaiser/robot-rl-sim60"
RUN = BASE + "/experiments/v6-task-campaign-20261001T110045Z-282358"
DIRECTORY = RUN + "/train/speed/attempt_002/train"
SOURCE_COMMIT = "b4ce80964d47c0127e0b3d9929ad35a620bbf204"
CONSUMED_ORIGIN, ACTOR_ORIGIN, CRITIC_UPDATES = 23868, 3563, 10
STAND = ("stand_230mm", "stand_305mm", "stand_430mm")
MOTION = ("forward_05", "backward_05", "rotate_1", "rotate_1__reverse_train")
RELEASE = tuple(name + suffix for name in
    ("release_stop_translate", "release_stop_yaw", "release_stop_combined")
    for suffix in ("", "__height_train"))
COUNTS = ("frames", "observed_frames", "episodes", "failures", "timeouts", "boundary_truncations")
AVERAGES = ("planar_speed_mean_m_s", "vx_mae_m_s", "vx_actual_mean_m_s",
    "vx_command_mean_m_s", "yaw_mae_rad_s", "height_actual_mean_m",
    "height_command_mean_m", "height_mae_m")
REMOTE = r'''
import json, math, pathlib, re, struct, time
from datetime import datetime, timezone
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c

def inspect(settings):
    begin=time.monotonic(); directory=pathlib.Path(settings['directory'])
    paths=list(directory.glob('events.out.tfevents.*'))
    if not paths: raise RuntimeError('No native event file in actual v14 speed attempt')
    event=max(paths,key=lambda p:p.stat().st_mtime_ns); size=event.stat().st_size
    offset=max(0,size-settings['tail_bytes'])
    with event.open('rb') as stream:
        stream.seek(offset); data=stream.read(min(settings['tail_bytes'],size-offset))
    groups='(?:'+'|'.join(re.escape(g) for g in settings['groups'])+')'
    counts='(?:'+'|'.join(settings['counts'])+')'
    averages='(?:'+'|'.join(settings['averages'])+'|tilt_max_deg)'
    expression=(r'(?:Behavior/[A-Za-z0-9_]+/'+counts+
        r'|Behavior/'+groups+'/'+averages+
        r'|LearningAudit/'+groups+r'/(?:stationary_10s/[A-Za-z0-9_]+|release_posture/[A-Za-z0-9_]+/[A-Za-z0-9_]+|planar_speed_m_s|height_error_m|command_vx_m_s|velocity/signed_forward_m_s)'
        r'|Timing/[A-Za-z0-9_]+|(?:Episode/)?/(?:delay|transport)/[A-Za-z0-9_]+|Policy/mean_std)')
    pattern=re.compile(expression.encode()+rb'(?![A-Za-z0-9_/])')
    seen=set(); series={}; unresolved=0
    for match in pattern.finditer(data):
        located=False
        # Reuse the native first-PPO reader's TFRecord header and payload CRC
        # validation. Tail offset may cut the first record; it is never accepted.
        for pos in range(max(0,match.start()-128),max(0,match.start()-11)):
            length,crc=struct.unpack_from('<QI',data,pos); end=pos+16+length
            if not 0<length<1024**2 or end>len(data) or end-4<match.end(): continue
            if masked_crc32c(data[pos:pos+8])!=crc: continue
            payload=data[pos+12:end-4]
            if masked_crc32c(payload)!=struct.unpack_from('<I',data,end-4)[0]: continue
            located=True
            if pos in seen: break
            seen.add(pos); item=Event.FromString(payload)
            for value in item.summary.value:
                if pattern.fullmatch(value.tag.encode()) and value.HasField('simple_value'):
                    number=float(value.simple_value)
                    if not math.isfinite(number): raise ValueError('Nonfinite scalar '+value.tag)
                    series.setdefault(value.tag,{})[int(item.step)]=[number,float(item.wall_time)]
            break
        unresolved+=int(not located)
    progress_path=directory/'progress.json'
    if progress_path.stat().st_size>1024**2: raise RuntimeError('Progress metadata unexpectedly large')
    progress=json.loads(progress_path.read_text())
    invocation_path=directory.parent/'invocation.json'
    if invocation_path.stat().st_size>1024**2: raise RuntimeError('Invocation metadata unexpectedly large')
    invocation=json.loads(invocation_path.read_text()); command=invocation.get('command',[])
    expected_entry=settings['source_remote']+'/scripts/train_chassis.py'
    expected_contract=settings['source_remote']+'/contracts/v6_task_campaign_speed_worker_v14.json'
    if expected_entry not in command or '--contract' not in command or command[command.index('--contract')+1]!=expected_contract:
        raise RuntimeError('Actual worker invocation differs from frozen v14 source')
    keep=('updated_at','pid','successful_updates','actor_updates_in_block','task_actor_update','optimizer_phase','num_envs')
    return dict(queried_at_utc=datetime.now(timezone.utc).isoformat(),
        event=dict(file=str(event),file_bytes_at_open=size,read_offset=offset,read_bytes=len(data),
            limit_bytes=settings['tail_bytes'],crc_valid_records=len(seen),unlocated_matches=unresolved,
            coverage='bounded_native_tail_not_reservoir_not_full_run',tail_sha256=__import__('hashlib').sha256(data).hexdigest()),
        source_invocation=dict(entry_script=expected_entry,contract=expected_contract),
        progress={k:progress.get(k) for k in keep},series=series,read_wall_seconds=time.monotonic()-begin)
print(json.dumps(inspect(SETTINGS),allow_nan=False))
'''


def number(series, tag, step):
    row = series.get(tag, {}).get(step)
    return row[0] if row is not None else None


def phase_stats(series, prefix, steps):
    observed = [s for s in steps if number(series, prefix + "samples", s) is not None]
    count = sum(number(series, prefix + "samples", s) for s in observed)
    result = dict(samples=count, observed_update_records=len(observed),
                  expected_update_records=len(steps), complete_update_coverage=len(observed) == len(steps))
    for tag, values in series.items():
        if not tag.startswith(prefix) or tag == prefix + "samples":
            continue
        key = tag[len(prefix):]
        # Phase fractions use a different denominator from conditional pose
        # means. Keep only pose means and their explicit lane-frame weights.
        if key == "fraction":
            continue
        used = [s for s in observed if number(series, prefix + "samples", s) > 0 and s in values]
        if not count:
            result[key] = None
        elif sum(number(series, prefix + "samples", s) for s in used) != count:
            result[key] = None
        elif key.endswith("_max_m"):
            result[key] = max(values[s][0] for s in used) if used else None
        else:
            result[key] = sum(values[s][0] * number(series, prefix + "samples", s) for s in used) / count
    for key in ("pitch_abs_rad", "roll_abs_rad"):
        if result.get(key) is not None:
            result[key.replace("rad", "deg")] = result[key] * 180. / 3.141592653589793
    return result


def window(series, a, b):
    expected = list(range(a + 1, b + 1))
    result = dict(event_steps_inclusive=[a + 1, b], cumulative_baseline_event_step=a,
        worker_updates=[a + 1 - CONSUMED_ORIGIN + 1, b - CONSUMED_ORIGIN + 1],
        actor_updates=[ACTOR_ORIGIN + a + 1 - CONSUMED_ORIGIN + 1 - CRITIC_UPDATES,
                       ACTOR_ORIGIN + b - CONSUMED_ORIGIN + 1 - CRITIC_UPDATES],
        policy_sim_seconds_per_environment=(b-a)*24*.02,
        interpretation="Actor has only just restarted; this short exploratory training window has no quality or qualification conclusion.",
        groups={}, release_stop={}, timing={}, totals={})
    names = sorted({tag.split('/')[1] for tag in series if tag.startswith('Behavior/')})
    totals = {key: 0 for key in COUNTS}
    count_coverage = []
    for name in names:
        prefix = "Behavior/" + name + "/"
        counts = {key: (number(series, prefix+key, b)-number(series, prefix+key, a))
                  for key in COUNTS if number(series, prefix+key, a) is not None and number(series, prefix+key, b) is not None}
        if len(counts) == len(COUNTS):
            count_coverage.append(name)
            for key in COUNTS:
                totals[key] += counts[key]
        if name not in STAND + MOTION + RELEASE:
            continue
        group = dict(counts)
        n = counts.get("frames", 0)
        for key in AVERAGES:
            old, new = number(series, prefix+key, a), number(series, prefix+key, b)
            group[key] = ((new * number(series,prefix+"frames",b) - old * number(series,prefix+"frames",a)) / n
                          if old is not None and new is not None and n > 0 else None)
        group["failures_per_ended_episode"] = counts.get("failures",0)/counts["episodes"] if counts.get("episodes") else None
        group["boundary_per_ended_episode"] = counts.get("boundary_truncations",0)/counts["episodes"] if counts.get("episodes") else None
        group["tilt_max_deg_since_worker_birth"] = number(series,prefix+"tilt_max_deg",b)
        group["stationary_10s"] = phase_stats(series,"LearningAudit/"+name+"/stationary_10s/",expected)
        result["groups"][name] = group
    totals["group_count_with_complete_counter_endpoints"] = len(count_coverage)
    totals["failures_per_ended_episode"] = totals["failures"]/totals["episodes"] if totals["episodes"] else None
    totals["counter_scope"] = "Native episode counters differenced across endpoints; boundary truncations are separate from failures."
    result["totals"] = totals
    for name in RELEASE:
        result["release_stop"][name] = {
            "completed_quiet_10s_windows": result["groups"].get(name,{}).get("stationary_10s"),
            "posture_phases": {phase: phase_stats(series,f"LearningAudit/{name}/release_posture/{phase}/",expected)
                for phase in ("accelerating", "cruising", "decelerating", "stationary_window")}}
    for tag, values in series.items():
        if tag.startswith("Timing/") or tag == "Policy/mean_std" or "/delay/" in tag or "/transport/" in tag:
            used = [values[s][0] for s in expected if s in values]
            result["timing" if tag.startswith("Timing/") else "runtime_scalars"] = result.get("timing" if tag.startswith("Timing/") else "runtime_scalars",{})
            result["timing" if tag.startswith("Timing/") else "runtime_scalars"][tag] = dict(
                records=len(used), expected_records=len(expected), complete_update_coverage=len(used)==len(expected),
                mean=statistics.mean(used) if used else None, median=statistics.median(used) if used else None)
    return result


def latest_snapshot(series):
    """Keep available evidence when the byte cap excludes a second endpoint."""
    behavior_steps=sorted(series.get('Behavior/stand_305mm/frames',{}))
    result=dict(interpretation='Actor has only just restarted; no quality or qualification conclusion.',
                latest_cumulative_behavior=None,latest_native_rollout_window=None)
    if behavior_steps:
        endpoint=max(behavior_steps)
        groups={}
        names=sorted({t.split('/')[1] for t in series if t.startswith('Behavior/')})
        all_counts={k:0 for k in COUNTS}; coverage=[]
        for name in names:
            prefix='Behavior/'+name+'/'
            counts={k:number(series,prefix+k,endpoint) for k in COUNTS}
            if all(v is not None for v in counts.values()):
                coverage.append(name)
                for k,v in counts.items(): all_counts[k]+=v
            if name in STAND+MOTION+RELEASE:
                groups[name]={**counts,**{k:number(series,prefix+k,endpoint) for k in AVERAGES},
                    'tilt_max_deg_since_worker_birth':number(series,prefix+'tilt_max_deg',endpoint)}
        result['latest_cumulative_behavior']=dict(event_step=endpoint,
            worker_successful_updates=endpoint-CONSUMED_ORIGIN+1,
            actor_updates_since_restart=max(0,endpoint-CONSUMED_ORIGIN+1-CRITIC_UPDATES),
            task_actor_update=ACTOR_ORIGIN+max(0,endpoint-CONSUMED_ORIGIN+1-CRITIC_UPDATES),
            native_event_wall_time_utc=datetime.fromtimestamp(series['Behavior/stand_305mm/frames'][endpoint][1],timezone.utc).isoformat(),
            scope='Cumulative since this worker started, INCLUDING critic-only updates; not a tail-window difference.',
            groups=groups,all_groups_counts=all_counts,groups_with_complete_counters=len(coverage))
    dense_steps=sorted(series.get('Timing/iteration_wall_seconds',{}))
    if dense_steps:
        a,b=min(dense_steps),max(dense_steps)
        expected=list(range(a,b+1))
        groups={}
        for name in STAND+MOTION:
            group={}
            for metric in ('planar_speed_m_s','height_error_m','command_vx_m_s','velocity/signed_forward_m_s'):
                values=series.get(f'LearningAudit/{name}/{metric}',{})
                used=[values[s][0] for s in expected if s in values]
                group[metric]=dict(mean=statistics.mean(used) if used else None,records=len(used),
                    expected_records=len(expected),complete_update_coverage=len(used)==len(expected))
            group['completed_quiet_10s_windows']=phase_stats(series,f'LearningAudit/{name}/stationary_10s/',expected)
            groups[name]=group
        release={name:dict(completed_quiet_10s_windows=phase_stats(series,f'LearningAudit/{name}/stationary_10s/',expected),
            posture_phases={phase:phase_stats(series,f'LearningAudit/{name}/release_posture/{phase}/',expected)
                for phase in ('accelerating','cruising','decelerating','stationary_window')}) for name in RELEASE}
        timing={}
        for tag,values in series.items():
            if tag.startswith('Timing/') or tag=='Policy/mean_std' or '/delay/' in tag or '/transport/' in tag:
                used=[values[s][0] for s in expected if s in values]
                timing[tag]=dict(records=len(used),expected_records=len(expected),complete_update_coverage=len(used)==len(expected),
                    mean=statistics.mean(used) if used else None,median=statistics.median(used) if used else None)
        result['latest_native_rollout_window']=dict(event_steps_inclusive=[a,b],
            worker_updates=[a-CONSUMED_ORIGIN+1,b-CONSUMED_ORIGIN+1],
            actor_updates_since_restart=[max(0,a-CONSUMED_ORIGIN+1-CRITIC_UPDATES),max(0,b-CONSUMED_ORIGIN+1-CRITIC_UPDATES)],
            policy_sim_seconds_per_environment=len(expected)*24*.02,
            observed_timing_steps=dense_steps,contiguous_timing_coverage=dense_steps==expected,
            scope='Only CRC-validated records within the bounded tail; per-tag completeness is reported.',
            groups=groups,release_stop=release,timing_and_runtime_scalars=timing)
    return result


def validated_counts(row, label):
    result={}
    for key in COUNTS:
        value=row.get(key)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or value!=round(value):
            raise ValueError(f'{label}: missing or invalid counter {key}')
        result[key]=int(value)
    if result['frames']>result['observed_frames'] or result['failures']>result['episodes'] or result['boundary_truncations']>result['timeouts']:
        raise ValueError(f'{label}: inconsistent counter denominators')
    return result


def interval_phase_stats(series, prefix, expected):
    observed=phase_stats(series,prefix,expected)
    if observed['complete_update_coverage']:
        return observed
    # Summing the visible tail cannot recover completed windows outside it.
    return dict(samples=None,mean_speed_m_s=None,endpoint_mean_m=None,peak_mean_m=None,peak_max_m=None,
        complete_update_coverage=False,observed_update_records=observed['observed_update_records'],
        expected_update_records=len(expected),scope='Partial tail only; full interval completed-window statistics unavailable.',
        observed_tail_statistics=observed)


def baseline_difference(baseline, report, series):
    """Difference validated event cumulative sums, never lifetime maxima."""
    current=report.get('latest_cumulative_behavior')
    if current is None:
        raise ValueError('No current complete native Behavior endpoint in bounded tail')
    if baseline.get('source_commit')!=report.get('source_commit') or report.get('source_commit')!=SOURCE_COMMIT:
        raise ValueError('Cross-source baseline rejected')
    old_event=baseline.get('event',{}).get('file'); new_event=report.get('event',{}).get('file')
    if old_event!=new_event or str(Path(new_event or '').parent)!=DIRECTORY:
        raise ValueError('Cross-attempt/event baseline rejected')
    for label,row in (('baseline',baseline),('current',current)):
        worker=row.get('worker_successful_updates'); actor=row.get('actor_updates_since_restart')
        if not isinstance(worker,int) or worker<=CRITIC_UPDATES or row.get('event_step')!=CONSUMED_ORIGIN+worker-1 or actor!=worker-CRITIC_UPDATES or row.get('task_actor_update')!=ACTOR_ORIGIN+actor:
            raise ValueError(f'{label}: inconsistent update clocks')
    a,b=baseline['event_step'],current['event_step']
    if b<=a or datetime.fromisoformat(current['native_event_wall_time_utc'])<=datetime.fromisoformat(baseline['native_event_wall_time_utc']):
        raise ValueError('Baseline/current clocks did not increase')
    if datetime.fromisoformat(report['queried_at_utc'])<=datetime.fromisoformat(baseline['queried_at_utc']):
        raise ValueError('Snapshot query clock did not increase')
    expected_groups=set(STAND+MOTION+RELEASE)
    if set(baseline.get('groups',{}))!=expected_groups or set(current.get('groups',{}))!=expected_groups:
        raise ValueError('Missing selected group endpoint denominators')
    old_coverage=baseline.get('groups_with_complete_counters'); new_coverage=current.get('groups_with_complete_counters')
    if not isinstance(old_coverage,int) or old_coverage<len(expected_groups) or old_coverage!=new_coverage or new_coverage!=report.get('expected_groups'):
        raise ValueError('Incomplete or changed all-group counter coverage')
    old_total=validated_counts(baseline.get('all_groups_counts',{}),'baseline/all')
    new_total=validated_counts(current.get('all_groups_counts',{}),'current/all')
    totals={k:new_total[k]-old_total[k] for k in COUNTS}
    if any(v<0 for v in totals.values()):
        raise ValueError('All-group counter rollback')
    if totals['observed_frames']!=(b-a)*report['progress']['num_envs']*24 or totals['frames']<=0:
        raise ValueError('Insufficient or inconsistent all-group frame denominator')
    result=dict(event_steps_inclusive=[a+1,b],baseline_event_step=a,
        baseline_worker_updates=baseline['worker_successful_updates'],baseline_actor_updates=baseline['actor_updates_since_restart'],
        baseline_includes_critic_updates=CRITIC_UPDATES,
        worker_updates_inclusive=[baseline['worker_successful_updates']+1,current['worker_successful_updates']],
        actor_updates_since_restart_inclusive=[baseline['actor_updates_since_restart']+1,current['actor_updates_since_restart']],
        task_actor_updates_inclusive=[baseline['task_actor_update']+1,current['task_actor_update']],
        updates=b-a,policy_sim_seconds_per_environment=(b-a)*24*.02,all_updates_actor_and_critic=True,
        interpretation='Pure actor interval after a cumulative baseline that included critic warmup; exploration and resets remain. No quality or qualification conclusion.',
        groups={},all_groups_counts=totals,complete_counter_groups=new_coverage,
        maxima_scope='Lifetime tilt maxima are retained only in the cumulative endpoint; no interval maximum is inferred.')
    expected=list(range(a+1,b+1))
    for name in sorted(expected_groups):
        old=baseline['groups'][name]; new=current['groups'][name]
        old_counts=validated_counts(old,'baseline/'+name); new_counts=validated_counts(new,'current/'+name)
        counts={k:new_counts[k]-old_counts[k] for k in COUNTS}
        if any(v<0 for v in counts.values()):
            raise ValueError('Group counter rollback: '+name)
        if old_counts['frames']<=0 or counts['frames']<=0 or counts['frames']>counts['observed_frames']:
            raise ValueError('Insufficient frame denominator: '+name)
        if new_counts['observed_frames']*baseline['worker_successful_updates']!=old_counts['observed_frames']*current['worker_successful_updates']:
            raise ValueError('Changed group population denominator: '+name)
        group=dict(counts)
        for key in AVERAGES:
            before,after=old.get(key),new.get(key)
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (before,after)):
                raise ValueError('Missing cumulative mean denominator: '+name+'/'+key)
            group[key]=(after*new_counts['frames']-before*old_counts['frames'])/counts['frames']
        group['failures_per_ended_episode']=counts['failures']/counts['episodes'] if counts['episodes'] else None
        group['boundary_per_ended_episode']=counts['boundary_truncations']/counts['episodes'] if counts['episodes'] else None
        group['stationary_10s']=interval_phase_stats(series,f'LearningAudit/{name}/stationary_10s/',expected)
        result['groups'][name]=group
    result['all_groups_counts']['failures_per_ended_episode']=totals['failures']/totals['episodes'] if totals['episodes'] else None
    values=series.get('Policy/mean_std',{}); used=[values[s][0] for s in expected if s in values]
    result['exploration_std_observed_tail']=dict(mean=statistics.mean(used) if used else None,records=len(used),
        expected_interval_records=len(expected),complete_interval_coverage=len(used)==len(expected),
        event_steps=sorted(s for s in expected if s in values),scope='Only current native tail; no missing interval values imputed.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='kaiser@192.168.64.234')
    parser.add_argument('--port',type=int,default=2222)
    parser.add_argument('--tail-mib',type=int,choices=(8,16),default=16)
    parser.add_argument('--window-updates',type=int,default=10)
    parser.add_argument('--windows',type=int,default=2)
    parser.add_argument('--baseline',type=Path,help='Saved native cumulative baseline from the same event and frozen source')
    parser.add_argument('--summary-output',type=Path,help='Compact summary; defaults to output filename with .summary.json')
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name(
        'v14_native_snapshot_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'))
    parser.add_argument('--emit-remote-code',action='store_true')
    args=parser.parse_args()
    if args.window_updates<=0 or args.window_updates%10 or not 1<=args.windows<=3:
        parser.error('Window updates must be a positive multiple of 10; windows must be 1–3')
    settings=dict(directory=DIRECTORY,tail_bytes=args.tail_mib*1024**2,groups=STAND+MOTION+RELEASE,counts=COUNTS,averages=AVERAGES,
        source_remote=RUN+'/source_b4ce80964d47')
    code='SETTINGS='+repr(settings)+'\n'+REMOTE
    if args.emit_remote_code:
        print(code); return
    report=dict(schema='bounded-v14-native-learning-audit-v1',status='pending',
        source_commit=SOURCE_COMMIT,qualification=False,skill_qualification=None,curve_source='native_remote_tensorboard_events_only',
        scope='read_only_existing_training_no_GPU_no_evaluation_no_tensorboard_server',
        metric_semantics=dict(behavior='Cumulative event means differenced with frames at 10-update endpoints; includes exploration and episode resets.',
            stationary_10s='Complete 10s quiet windows, after 2s zero visible command warmup, height error <=10mm; anchored at eligible-window entry. Zeros without samples are excluded.',
            release_posture='Conditional native phase means weighted by policy-rate lane-frame samples; these sample counts are not completed-window counts.',
            pitch_roll_scope='Native separated pitch/roll exists only for release_stop phases. Other groups expose cumulative lifetime tilt maxima, not separate pitch/roll.',
            observation_rate_hz=50,feedback_PD_hz=200),windows=[])
    command='source '+shlex.quote(BASE+'/bin/sim60-runtime.sh')+' && export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 && '+shlex.join([BASE+'/env/bin/python','-B','-'])
    try:
        remote=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p',str(args.port),args.host,
            shlex.join(['bash','-lc',command])],input=code,capture_output=True,text=True,timeout=60)
        if remote.returncode: raise RuntimeError(remote.stderr[-2000:])
        data=json.loads(remote.stdout); series={tag:{int(s):v for s,v in rows.items()} for tag,rows in data.pop('series').items()}
        report.update(data)
        steps=sorted(series.get('Behavior/stand_305mm/frames',{}))
        pairs=[]
        if steps:
            end=max(steps)
            for i in reversed(range(args.windows)):
                b=end-i*args.window_updates; a=b-args.window_updates
                if a in steps and b in steps and a>=CONSUMED_ORIGIN+CRITIC_UPDATES-1:
                    pairs.append((a,b))
        report['windows']=[window(series,a,b) for a,b in pairs]
        report.update(latest_snapshot(series))
        report['status']='bounded_native_startup_snapshot_complete'
        report['behavior_window_difference_status']='complete' if pairs else 'insufficient_native_endpoint_coverage_within_byte_cap'
        report['native_behavior_endpoint_steps_in_tail']=steps
        report['requested_window_updates']=args.window_updates
        report['expected_groups']=len({t.split('/')[1] for t in series if t.startswith('Behavior/')})
        if args.baseline:
            raw=args.baseline.read_bytes()
            if len(raw)>1024**2: raise ValueError('Baseline exceeds small-metadata limit')
            baseline=json.loads(raw)
            if 'latest_cumulative_behavior' in baseline:
                baseline={**{k:baseline.get(k) for k in ('source_commit','queried_at_utc','event')},
                    **(baseline['latest_cumulative_behavior'] or {})}
            report['baseline_provenance']=dict(file=str(args.baseline.resolve()),sha256=hashlib.sha256(raw).hexdigest(),
                queried_at_utc=baseline.get('queried_at_utc'),event=baseline.get('event'))
            try:
                report['baseline_interval']=baseline_difference(baseline,report,series)
                report['baseline_interval_status']='validated_native_cumulative_difference'
            except ValueError as error:
                report['baseline_interval_status']='rejected'
                report['baseline_interval']=None
                report['baseline_interval_error']=str(error)
                report['status']='baseline_interval_rejected'
    except (subprocess.TimeoutExpired,RuntimeError,json.JSONDecodeError,OSError,ValueError) as error:
        report['status']='readonly_snapshot_failed'
        report['error']=str(error)
        report['queried_at_utc']=datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.baseline:
        summary_path=args.summary_output or args.output.with_suffix('.summary.json')
        summary_path.parent.mkdir(parents=True,exist_ok=True)
        summary={key:report.get(key) for key in ('schema','status','source_commit','qualification','skill_qualification',
            'curve_source','scope','queried_at_utc','progress','event','baseline_provenance','baseline_interval_status',
            'baseline_interval_error','baseline_interval','latest_native_rollout_window','error')}
        summary['source_snapshot']=str(args.output.resolve())
        summary_path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],output=str(args.output),windows=len(report['windows']),
        progress=report.get('progress'),read_bytes=report.get('event',{}).get('read_bytes')),ensure_ascii=False))


if __name__=='__main__': main()
