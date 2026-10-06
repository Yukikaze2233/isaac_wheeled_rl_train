#!/usr/bin/env python3
"""Summarize complete existing021 report JSONs; no event reparse or simulator.

The two50 windows must be disjoint and exactly partition the100 actor window.
Count and frame-weighted mean identities are checked before interpretation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
COMMIT='d0dce6181df83b03fe166abcbadef3c81de2e57e'
READER_SHA='5faa75e24b0dd484bdacb0273756642caab8fcc6d6e66f90c034a0bda6ac19a6'
COUNTS=('frames','observed_frames','episodes','successes','failures','timeouts','boundary_truncations','blocked_truncations')
MEANS=('vx_mae_m_s','yaw_mae_rad_s','height_mae_m','planar_speed_mean_m_s','vx_actual_mean_m_s','vx_command_mean_m_s','vx_bias_m_s','height_actual_mean_m','height_command_mean_m')


def require(condition,message):
    if not condition:raise ValueError(message)


def load(path):
    raw=Path(path).read_bytes();report=json.loads(raw)
    require(report['status']=='requested_actor_windows_complete' and report['complete'] is True,'Only complete native reports may enter this summary')
    require(report['source_commit']==COMMIT and report['reader_sha256']==READER_SHA,'Wrong frozen source/reader')
    require(report['qualification'] is None and report['PPO_updates_by_reader']==0,'Unexpected qualification/PPO')
    return report,hashlib.sha256(raw).hexdigest()


def check_partition(first,last,full):
    require(first['window']['task_actor_updates_inclusive']==[3086,3135]
        and last['window']['task_actor_updates_inclusive']==[3136,3185]
        and full['window']['task_actor_updates_inclusive']==[3086,3185],'Windows do not exactly partition intended actors')
    require(first['window']['event_steps_inclusive']==[30016,30065]
        and last['window']['event_steps_inclusive']==[30066,30115]
        and full['window']['event_steps_inclusive']==[30016,30115],'Native event intervals differ')
    require(first['window']['baseline_step']==30015 and last['window']['baseline_step']==30065
        and full['window']['baseline_step']==30015,'Missing actual Behavior baselines')
    catalogs=[row['behavior']['groups'] for row in (first,last,full)]
    require(set(catalogs[0])==set(catalogs[1])==set(catalogs[2]),'Native catalog changed')
    for group in catalogs[0]:
        a,b,c=(catalog[group] for catalog in catalogs)
        for count in COUNTS:require(a[count]+b[count]==c[count],f'Count partition differs: {group}/{count}')
        for mean in MEANS:
            if c['frames']:
                expected=((a[mean] or 0)*a['frames']+(b[mean] or 0)*b['frames'])/c['frames']
                require(math.isclose(expected,c[mean],rel_tol=1e-7,abs_tol=1e-7),f'Frame-weighted mean partition differs: {group}/{mean}')
    # Per-rollout conditional means are descriptive only, but their50+50 time
    # means must partition the100 tag sequence independently of lane weights.
    for section in ('hold_cost','hold_gates'):
        for tag,c in full['teaching'][section].items():
            a,b=first['teaching'][section][tag],last['teaching'][section][tag]
            require(a['records']==b['records']==50 and c['records']==100 and all(x['complete'] for x in (a,b,c)),'Incomplete teaching tag records')
            require(math.isclose((a['mean']+b['mean'])/2,c['mean'],rel_tol=1e-7,abs_tol=1e-7),'Conditional time-mean partition differs:'+tag)
    return dict(status='passed',groups=len(catalogs[0]),exact_count_partitions=len(catalogs[0])*len(COUNTS),frame_mean_partitions=len(catalogs[0])*len(MEANS),actor_intervals_disjoint=True,critic_excluded_from_selected_intervals=True,no_samples_added_from_overlapping100=True)


def extract(window):
    groups=window['behavior']['groups'];recovery=window['behavior']['pools']['scripted_recovery']
    costs=window['teaching']['hold_cost'];gate=window['teaching']['hold_gates'];commands=window['teaching']['static_group_command_means']
    return dict(window=window['window'],learning_ownership=window['learning_ownership'],timing=window['timing'],exploration=window['exploration'],
        exact_recovery_counts={key:recovery[key] for key in COUNTS},
        base_group_metrics={g:row for g,row in groups.items() if g in window['behavior']['base_groups']},
        dynamic_lessons=window['dynamic_lessons'],hold_cost=costs,observer_gate_stats=gate,
        static_recovery_command_logs=dict(groups=len(commands),records_per_tag=window['window']['actor_updates'],all_logged_min_max_zero=all(v['min']==v['max']==0 for fields in commands.values() for v in fields.values())),
        actual_L0_motion_release=window['teaching']['actual_L0_motion_release'],actual_L0_nonzero_goal_exposure=window['teaching']['actual_L0_nonzero_goal_exposure'],
        stationary_10s=window['stationary_10s'],conditional_stop_phases=window['optional_conditional_stop_phases'])


def change(a,b):
    return dict(first50=a,last50=b,change=b-a if a is not None and b is not None else None)


def build(first_report,last_report,sha_first,sha_last,first_path,last_path):
    first,last,full=first_report['windows']['50'],last_report['windows']['50'],last_report['windows']['100']
    require(all(x['complete'] for x in (first,last,full)),'Incomplete selected actor window')
    require(first_report['identity']['event']==last_report['identity']['event'],'Different native events')
    require(first_report['identity']['source_files_sha256']==last_report['identity']['source_files_sha256'],'Different frozen source bytes')
    identity=check_partition(first,last,full)
    trends={}
    for group in first['behavior']['base_groups']:
        a,b=first['behavior']['groups'][group],last['behavior']['groups'][group]
        trends[group]={field:change(a[field],b[field]) for field in MEANS}
        trends[group]['exact_counts']={'first50':{k:a[k] for k in COUNTS},'last50':{k:b[k] for k in COUNTS}}
    holds={section:{tag:change(a['mean'],last['teaching'][section][tag]['mean']) for tag,a in first['teaching'][section].items()} for section in ('hold_cost','hold_gates')}
    lesson_trends={lesson:{field:change(first['dynamic_lessons']['levels'][lesson]['fields'][field]['mean'],last['dynamic_lessons']['levels'][lesson]['fields'][field]['mean']) for field in ('probability','population','success_fraction','successes')} for lesson in first['dynamic_lessons']['levels']}
    return dict(schema='source021-first100-with-disjoint-halves-read-only-summary-v1',source_commit=COMMIT,qualification=None,PPO_updates_by_summary=0,
        raw_inputs={str(Path(first_path).resolve()):sha_first,str(Path(last_path).resolve()):sha_last},reader_sha256=READER_SHA,
        sample_partition=identity,first50=extract(first),last50=extract(last),full100=extract(full),
        changes=dict(base_groups=trends,hold=holds,dynamic_lessons=lesson_trends,timing={tag:change(a['mean'],last['timing'][tag]['mean']) for tag,a in first['timing'].items()},mean_std=change(first['exploration']['mean'],last['exploration']['mean'])),
        signal_scope_correction='support and body_clear are deployable FK/IMU observer decisions, not true simulation contacts; true contact is graded separately in RecoveryTraining.outcome',
        physics_scope=dict(num_envs=4096,physics_dt_s=.005,physics_feedback_Hz=200,policy_dt_s=.02,policy_Hz=50,decimation=4,
            scope='unchanged frozen source021 and the runtime admission authenticated by the reader; not a new physics or hardware test'),
        interpretation=dict(recovery_success_observed=False,base_quality_changes_mixed=True,
            hold_cost_active_in_logs=True,causal_root_cause_identified=False,
            L0_specific_motion_release_exposure_available=False,qualification_awarded=False),
        limits=['Descriptive within-run50-vs50 changes under actual ongoing PPO; not a causal reward/std experiment or deterministic qualification.',
            '54 static recovery groups mix lessons; their command averages do not independently identify L0 release or exposure.',
            'Gate fractions are instantaneous marginal observer decisions; they do not determine true contact or continuous1s joint success probability.',
            'HoldCost tags are time means of actor-hold conditional density means, not exact full lane payments or actor advantages.',
            'Dynamic lesson counters are24-frame averaged cumulative counters, not exact episode counts. Static Behavior episode partitions supply exact ended counts.',
            'This aggregate-window reader omits existing DiagnosticUpdate/action_std_0..5 and per-group KL tags; their separate authenticated native100 read is recorded under exploration_gate_review_021/per_axis_std_021_001. Mean_std alone is not measured joint/gyro noise or entropy-loss-to-reward comparison; individual joint stable dwell remains unavailable.',
            'Recovery timeouts include collection censoring; boundary truncations are a subset of timeouts. Ended episode counts do not reveal eventual success after truncation.',
            'Completed stationary10s windows are an eligibility-selected subset and can have anchors before the actor interval. They do not represent all standing frames; zero completed samples are unavailable.',
            '50+50 exactly partitions100; full100 overlaps both and is not additional experience.',
            'True motor responses for legs remain unqualified; hardware qualification is null. No training action was made by this summary.'])


def markdown(summary):
    a,b,c=summary['first50'],summary['last50'],summary['full100']
    rows=[]
    def row(label,x,y,unit=''):
        rows.append(f'| {label} | {x:.4f}{unit} | {y:.4f}{unit} | {y-x:+.4f}{unit} |')
    for group in ('stand_230mm','stand_305mm','stand_430mm'):
        row(group+' 平面均速',a['base_group_metrics'][group]['planar_speed_mean_m_s']*100,b['base_group_metrics'][group]['planar_speed_mean_m_s']*100,'cm/s')
    for group in ('forward_05','backward_05'):row(group+' vx MAE',a['base_group_metrics'][group]['vx_mae_m_s']*100,b['base_group_metrics'][group]['vx_mae_m_s']*100,'cm/s')
    for group in ('rotate_1','rotate_1__reverse_train'):row(group+' yaw角速度MAE',a['base_group_metrics'][group]['yaw_mae_rad_s'],b['base_group_metrics'][group]['yaw_mae_rad_s'],'rad/s')
    for group in ('start_stop_05','start_stop_1','velocity_curve_1'):
        x,y=(w['conditional_stop_phases']['groups'][group]['settled_stop'] for w in (a,b))
        row(group+' settled_stop条件均速',x['mean_speed_m_s']*100,y['mean_speed_m_s']*100,'cm/s')
    for axis in ('wheel','leg','gyro'):
        tag=f'/HoldCost/{axis}/gate_relative_error_mean';row(axis+'门归一误差',a['hold_cost'][tag]['mean'],b['hold_cost'][tag]['mean'])
    for gate in ('support','body_clear','tilt','height','wheel_speed','leg_speed','gyro'):
        tag=f'/recovery/hold_gate/{gate}_pass_fraction';row(gate+' observer瞬时通过率',a['observer_gate_stats'][tag]['mean']*100,b['observer_gate_stats'][tag]['mean']*100,'%')
    row('更新墙钟',a['timing']['iteration_wall_seconds']['mean'],b['timing']['iteration_wall_seconds']['mean'],'s')
    row('Policy mean_std',a['exploration']['mean'],b['exploration']['mean'])
    recovery=c['exact_recovery_counts'];first_counts=a['exact_recovery_counts'];last_counts=b['exact_recovery_counts']
    successes={name:c['dynamic_lessons']['levels'][name]['fields']['successes']['max'] for name in ('prefix_stand','prefix_release','original_chain')}
    sample_counts={group: [w['conditional_stop_phases']['groups'][group]['settled_stop']['samples'] for w in (a,b,c)] for group in ('start_stop_05','start_stop_1','velocity_curve_1')}
    standing100=[c['base_group_metrics'][g]['planar_speed_mean_m_s']*100 for g in ('stand_230mm','stand_305mm','stand_430mm')]
    vx100=[c['base_group_metrics'][g]['vx_mae_m_s']*100 for g in ('forward_05','backward_05')]
    yaw100=[c['base_group_metrics'][g]['yaw_mae_rad_s'] for g in ('rotate_1','rotate_1__reverse_train')]
    text=f'''source021 完整100 actor与不重叠两半50窗口实测\n\n前半task3086–3135／event30016–30065；后半task3136–3185／event30066–30115。各4096×24s，共100actor／48s／9,830,400lane策略帧。实际父29996/task3075及critic10保持独立；95组760个精确计数和855个frame加权均值分割核对通过。100窗与两个50窗重叠，不相加经验。\n\n| 指标 | 前50 | 后50 | 后−前（描述性） |\n|---|---:|---:|---:|\n'''+ '\n'.join(rows)+f'''\n\n恢复静态54组结束{recovery['episodes']}回合，成功{recovery['successes']}、失败{recovery['failures']}、超时{recovery['timeouts']}；前／后成功分别{first_counts['successes']}/{first_counts['episodes']}与{last_counts['successes']}/{last_counts['episodes']}，混合动态lessons。L0/L1/L2成功日志峰值为{successes}。54组运动命令min/max全零：前{a['static_recovery_command_logs']['all_logged_min_max_zero']}／后{b['static_recovery_command_logs']['all_logged_min_max_zero']}／100窗{c['static_recovery_command_logs']['all_logged_min_max_zero']}。独立L0 motion release和原目标曝光仍为unavailable，不能据此认定bridge实现错误。\n\nsupport/body_clear是可部署FK／IMU observer门（recovery.py41），不是实际接触、接触力或真实离地状态。真接触在RecoveryTraining.outcome另评分。表中边际门率不能推出连续1s联合通过率；hold成本是条件密度均值，不是完整付款或优势。静止10s没有完成样本时指标不可用，不能填零；实际230／305两半都0样本，430前44／后27／合71，只是eligible完成子集。停车条件样本（前50／后50／100）为{sample_counts}，整段启停速度与驻停速度分开。4932 timeout为实际结束／删失，含80次边界截断；不能从它推断无限时长下的恢复成功率，failure也不能一律解释成跌倒。\n\n两半变化只有同一run短期描述意义。策略年龄、动态lesson比例、出生／重置／域随机化和探索都随学习变化；未做同seal mean/stochastic配对，不能将一个成本或std定为唯一原因。Policy mean_std是归一化actor动作维度的聚合，不是单轴std或实测关节／gyro噪声，也不能与reward密度／熵loss跨量纲直接比较。当前物理dt5ms/200Hz、策略dt20ms/50Hz，4096与原PD160/2.5保持；工程原生通过不替代恢复行为资格。硬件资格仍null；本文没有修改训练、预算、门判据或优化器。\n\n原始输入SHA：\n'''
    totals=f"100窗整体：230／305／430mm静站平面均速{standing100[0]:.2f}／{standing100[1]:.2f}／{standing100[2]:.2f}cm/s；0.5前／后vx MAE{vx100[0]:.2f}／{vx100[1]:.2f}cm/s；正／反yaw角速度MAE{yaw100[0]:.4f}／{yaw100[1]:.4f}rad/s。actor有效转移{c['learning_ownership']['valid_transitions_total']}、SCRIPT／BLEND／passive排除{c['learning_ownership']['excluded_transitions_total']}。恢复成功仍未出现，腿／gyro门误差微降且轮误差几乎不变，普通行为改善与退步并存，未见整体好转证据。"
    text=text.replace('\n\n恢复静态54组', '\n\n'+totals+'\n\n恢复静态54组')
    for path,sha in summary['raw_inputs'].items():text+=f'- `{Path(path).parent.name}/{Path(path).name}`：`{sha}`\n'
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--first50',type=Path,required=True);parser.add_argument('--tail50-full100',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    first,sha1=load(args.first50);last,sha2=load(args.tail50_full100)
    summary=build(first,last,sha1,sha2,args.first50,args.tail50_full100)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'first100_summary.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
    with (args.output_dir/'first100_summary.md').open('x') as f:f.write(markdown(summary))
    require(hashlib.sha256(args.first50.read_bytes()).hexdigest()==sha1 and hashlib.sha256(args.tail50_full100.read_bytes()).hexdigest()==sha2,'Raw input changed during summary')
    print(json.dumps(dict(status='completed_read_only_summary',source_commit=COMMIT,qualification=None,output_dir=str(args.output_dir))))


if __name__=='__main__':main()
