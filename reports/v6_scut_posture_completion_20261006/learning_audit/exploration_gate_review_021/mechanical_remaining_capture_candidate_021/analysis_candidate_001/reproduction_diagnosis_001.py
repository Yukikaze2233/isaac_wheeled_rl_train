"""Read-only diagnosis of a preserved strict-reproduction rejection; no App."""
from __future__ import annotations
import collections
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
REFERENCE = CANDIDATE.parent / 'mechanical_side_speed_candidate_021/native_SIDE_001/native'
ACTUAL = CANDIDATE / 'native_remaining_001/native'
OUT = HERE / 'reproduction_diagnosis_001'
MANIFEST_SHA = 'af75f849a42e7447b35cbdb7c5d21628c27ae04a5b976a7c5d6e2897cbb698a2'

def require(condition, message):
    if not condition: raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path): return json.loads(Path(path).read_text())

def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream: yield json.loads(line)

def ordered_bits(value):
    raw = struct.unpack('>Q', struct.pack('>d', value))[0]
    return (~raw & ((1 << 64)-1)) if raw >> 63 else raw | (1 << 63)

def diffs(a, b, path=''):
    if type(a) is not type(b):
        yield dict(path=path, kind='type', reference=str(type(a)), actual=str(type(b))); return
    if isinstance(a, dict):
        for key in a:
            if key not in b: yield dict(path=path+'/'+key, kind='missing_actual_key')
            else: yield from diffs(a[key], b[key], path+'/'+key)
        for key in b:
            if key not in a: yield dict(path=path+'/'+key, kind='new_actual_key')
    elif isinstance(a, list):
        if len(a) != len(b):
            yield dict(path=path, kind='length', reference=len(a), actual=len(b)); return
        for i, (old, new) in enumerate(zip(a, b)): yield from diffs(old, new, path+'/'+str(i))
    elif isinstance(a, float):
        if struct.pack('>d', a) != struct.pack('>d', b):
            require(math.isfinite(a) and math.isfinite(b), 'Nonfinite changed summary value')
            absolute = abs(b-a)
            yield dict(path=path, kind='float_bits', reference=a, actual=b,
                absolute_difference=absolute, signed_difference=b-a,
                relative_difference=absolute/max(abs(a), abs(b), 1e-300),
                signed_zero_only=(a == b == 0),
                double_ULPs=(None if a == b == 0 else abs(ordered_bits(a)-ordered_bits(b))))
    elif a != b: yield dict(path=path, kind='scalar', reference=a, actual=b)

def exact(a, b): return next(diffs(a, b), None) is None

def main():
    require(not OUT.exists(), 'Never overwrite a diagnosis')
    paths = [p for directory in (REFERENCE, ACTUAL) for p in directory.iterdir() if p.is_file()]
    paths += [CANDIDATE/'native_remaining_001.process.json', CANDIDATE/'native_remaining_001/run_plan.json',
        CANDIDATE/'native_remaining_001/remaining_capture_rejection.json', CANDIDATE/'candidate_manifest.json',
        CANDIDATE/'run_remaining_capture_021.py', HERE/'read_remaining_capture_001.py', Path(__file__)]
    inputs = {str(p.resolve()): sha(p) for p in paths}
    spec = importlib.util.spec_from_file_location('remaining_diagnosis_frozen_driver', CANDIDATE/'run_remaining_capture_021.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    driver.verify_candidate(MANIFEST_SHA)  # Pure CPU identity authentication, not old review/native.
    process = read(CANDIDATE/'native_remaining_001.process.json')
    plan = read(CANDIDATE/'native_remaining_001/run_plan.json')
    require(process['process_exit_observed'] is True and process['process_exit_code'] == 0
        and process['process_present_after_wait'] is False, 'No authenticated completed process')
    require(process['command'] == plan['native_command'] and process['candidate_manifest_sha256'] == MANIFEST_SHA
        and process['PPO_updates'] == process['formal_updates_consumed'] == 0, 'Process identity/accounting differs')
    preserved_rejection = read(CANDIDATE/'native_remaining_001/remaining_capture_rejection.json')
    require(preserved_rejection['status'] == 'preserved_actual_measurement_rejected', 'Original rejection missing')
    stream_checks = {}
    for name in ('first_birth_rows.jsonl.gz', 'events.jsonl.gz', 'mechanical_global_peaks.jsonl.gz'):
        a, b = list(rows(REFERENCE/name)), list(rows(ACTUAL/name))
        require(exact(a, b), 'Decoded native trajectory/birth/event values differ: '+name)
        stream_checks[name] = dict(decoded_values_float_bits_exact=True, reference_rows=len(a), actual_rows=len(b),
            reference_raw_sha256=sha(REFERENCE/name), actual_raw_sha256=sha(ACTUAL/name),
            decompressed_bytes_sha256_exact=(hashlib.sha256(gzip.decompress((REFERENCE/name).read_bytes())).hexdigest()
                == hashlib.sha256(gzip.decompress((ACTUAL/name).read_bytes())).hexdigest()))
    require(stream_checks['first_birth_rows.jsonl.gz']['actual_rows'] == 4096, 'Missing full birth population')
    unchanged = {}
    for name in ('identity.json', 'teaching_summary.json', 'side_override_applied.json', 'side_override_preclose.json'):
        require(exact(read(REFERENCE/name), read(ACTUAL/name)), 'Native identity/teaching/override differs: '+name)
        unchanged[name] = dict(decoded_values_float_bits_exact=True, raw_bytes_exact=sha(REFERENCE/name)==sha(ACTUAL/name))
    old_payment, payment = read(REFERENCE/'payment_summary.json'), read(ACTUAL/'payment_summary.json')
    payment_diff = list(diffs(old_payment, payment))
    require(payment_diff and all(d['kind'] == 'float_bits' and d['path'].startswith('/rows/')
        and '/components/' in d['path'] for d in payment_diff), 'Nonaggregate payment difference')
    fields = collections.Counter(d['path'].rsplit('/', 1)[-1] for d in payment_diff)
    zero_diffs = [d for d in payment_diff if d['signed_zero_only']]
    numeric_diffs = [d for d in payment_diff if not d['signed_zero_only']]
    require(all(d['path'].rsplit('/',1)[-1] in ('minimum','maximum') for d in zero_diffs), 'Unexpected signedzero field')
    require(all(d['path'].rsplit('/',1)[-1] in ('sum','mean','sum_payment','mean_payment','standard_deviation')
        for d in numeric_diffs), 'Native nonsummary/extrema numeric difference')
    maxima_by_field = {key:max((d for d in payment_diff if d['path'].endswith('/'+key)),
        key=lambda d:d['absolute_difference']) for key in fields}
    event_checks = {name: all(exact(a['components'][name], b['components'][name])
        for a,b in zip(old_payment['rows'], payment['rows'])) for name in ('success_event','failure_event')}
    require(all(event_checks.values()), 'Discrete event aggregates changed')
    require(payment['complete'] and payment['policy_steps'] == payment['event_producer_calls'] == 600
        and payment['native_lane_intervals'] == 4096*600 and payment['max_reconciliation_residual'] == 0,
        'Payment capture/integrity incomplete')
    aggregates = {}
    for scope in ('all_native_intervals', 'first_birth_intervals'):
        selected = [[r for r in p['rows'] if r['scope']==scope] for p in (old_payment, payment)]
        totals = {}
        for name in ('returned_immediate_reward','actor_eligible_immediate_reward','success_event','failure_event'):
            a,b = [math.fsum(r['components'][name]['sum'] for r in selected_rows) for selected_rows in selected]
            totals[name] = dict(reference_sum=a,actual_sum=b,absolute_difference=abs(b-a),unit='reward')
        aggregates[scope] = dict(reference_intervals=sum(r['policy_lane_intervals'] for r in selected[0]),
            actual_intervals=sum(r['policy_lane_intervals'] for r in selected[1]),totals=totals)
    report, old_report = read(ACTUAL/'report.json'), read(REFERENCE/'report.json')
    report_diffs = list(diffs(old_report, report))
    allowed_report_paths = {'/started_at_utc','/observer_included_wall_seconds','/mechanical_summary_sha256',
        '/payment_summary_sha256'} | {'/artifact_sha256/'+name for name in
            ('events.jsonl.gz','first_birth_rows.jsonl.gz','mechanical_global_peaks.jsonl.gz','mechanical_5ms.jsonl.gz')}
    require({d['path'] for d in report_diffs} <= allowed_report_paths, 'Report behavior/physics/RNG/source value changed')
    for base,value in ((ACTUAL,report),(REFERENCE,old_report)):
        for name,digest in value['artifact_sha256'].items(): require(sha(base/name)==digest, 'Report raw artifact hash differs')
        for name,key in (('identity.json','identity_sha256'),('payment_summary.json','payment_summary_sha256'),
            ('teaching_summary.json','teaching_summary_sha256'),('mechanical_summary.json','mechanical_summary_sha256'),
            ('side_override_preclose.json','isolated_SIDE_override_summary_sha256')):
            require(sha(base/name)==value[key], 'Native own content binding differs: '+name)
    require(report['status']=='native600_completed_preclose' and report['policy_steps_completed']==600
        and report['physical_steps_completed']==2400 and report['simulated_seconds']==12
        and report['actor_tensors_and_saved_std_unchanged'] and report['checkpoint_unchanged']
        and report['source_files_unchanged'] and report['curriculum_clocks_unchanged']
        and report['learning_transition_clock_unchanged'] and report['payment_hooks_restored_before_app_close']
        and report['mechanical_hooks_restored'] and report['qualification'] is None
        and report['PPO_updates']==report['formal_updates_consumed']==report['optimizer_instances']==0,
        'Native terminal engineering scope differs')
    summary, layout = read(ACTUAL/'mechanical_summary.json'), read(ACTUAL/'mechanical_layout.json')
    require(summary['complete'] and summary['hooks_restored'] and summary['policy_steps']==600
        and summary['physics_steps']==summary['trace_cutoff_physics_steps']==2400
        and not summary['contact_count_at_capacity_indicators'], 'Mechanical incomplete/capacity ambiguity')
    require(layout['lanes']==[881,1877,2121] and layout['profiles']==
        ['front_down_90','step200_lip_near_shell','step200_upright_top'] and layout['num_envs']==4096
        and layout['physics_dt_s']==.005 and layout['policy_dt_s']==.02 and layout['guard_m']==.003,
        'Mechanical measurement selection differs')
    trace = collections.defaultdict(list)
    for row in rows(ACTUAL/'mechanical_5ms.jsonl.gz'): trace[row['state']['lane']].append(row)
    lane_checks={}
    for lane,tick,level,profile_id,profile,group in zip((881,1877,2121),(222,104,251),(2,2,1),(8,18,19),
        layout['profiles'],('recovery_front_down_90_forward','recovery_step200_lip_near_shell_stand',
            'recovery_step200_prone_belly_top_forward')):
        selected=trace[lane];terminal=summary['native_first_terminals'][str(lane)]
        require(len(selected)==summary['lane_row_counts'][str(lane)]==tick*4+1, 'Lane missing trace')
        for i,row in enumerate(selected):
            require(row['physical_step']==i and row['simulated_s']==i*.005 and row['state']['birth_id']==1
                and row['state']['group']==group, 'Lane firstbirth trace/clock/group differs')
            if i: require(row['policy_tick']==(i-1)//4+1 and row['substep']==(i-1)%4+1, 'Policy mapping differs')
        require(terminal['policy_tick']==tick and terminal['reasons']['closure_gap']
            and terminal['fields']['recovery_actual_level_id']==level
            and terminal['fields']['recovery_actual_profile_id']==profile_id
            and terminal['fields']['recovery_handover_count']==0 and not terminal['fields']['recovery_pure_rl'],
            'Actual firstbirth terminal/lesson differs')
        require(exact(selected[-1]['state'],summary['pre_autoreset_snapshots'][str(lane)]['state']),
            'Terminal saved substep vs preautoreset state differs')
        lane_checks[str(lane)]=dict(rows=len(selected),physical_step_range=[0,tick*4],native_first_terminal_policy_tick=tick,
            continuous_5ms_and_firstbirth=True,actual_lesson_level=level,actual_sampled_profile=profile,
            behavior_group=group,terminal_preautoreset_state_exact=True,no_handover_before_closure_guard=True)
    require(set(trace)=={881,1877,2121}, 'Unexpected observed lane')
    for path,digest in inputs.items():require(sha(path)==digest, 'Input modified during readonly diagnosis')
    OUT.mkdir()
    detail=OUT/'payment_all_float_bit_differences.jsonl.gz'
    with gzip.open(detail,'wt') as stream:
        for d in payment_diff:stream.write(json.dumps(d,separators=(',',':'),allow_nan=False)+'\n')
    result=dict(schema='source021-remaining-reproduction-diagnosis-v1',status='actual_strict_reproduction_rejected_mechanical_observation_authentic',
        strict_whole_reproduction=False,original_rejection_preserved=True,inputs_sha256=inputs,
        decoded_native_stream_checks=stream_checks,unchanged_identity_teaching_overrides=unchanged,
        payment=dict(float_bit_differences=len(payment_diff),field_counts=dict(fields),signed_zero_extrema=len(zero_diffs),
            numeric_roundoff_differences=len(numeric_diffs),maxima_by_field=maxima_by_field,
            all_difference_locations_are_aggregated_components=True,counts_partitions_catalog_nonrow_fields_exact=True,
            event_aggregates_exact=event_checks,scope_total_sums=aggregates,
            maximum_absolute_difference=max(d['absolute_difference'] for d in payment_diff),
            maximum_relative_difference=max(d['relative_difference'] for d in numeric_diffs),
            reconciliation_residual_both_zero=old_payment['max_reconciliation_residual']==payment['max_reconciliation_residual']==0,
            detailed_differences_path=str(detail),detailed_differences_sha256=sha(detail),
            aggregation_implementation='NativePaymentObserver uses GPU float64 scatter_add_ of values/squares and scatter_reduce_ extrema; summary sqrt(E[x²]−mean²).',
            attribution='Aggregate-roundoff pattern is consistent with nonordered GPU reductions; per-lane density samples were not serialized so their exact equivalence is unproved.'),
        native_report=dict(all_differences=report_diffs,own_raw_sha_bindings_verified=True,
            all_nonprovenance_behavior_RNG_source_physics_fields_float_bits_exact=True,
            comparison_note='Original strict report excludes only three fields; raw gzip hashes/payment SHA also genuinely differ and remain explicit, not silently accepted.'),
        mechanical_selected_lane_integrity=lane_checks,actual_native_steps=[600,2400],process_wall_seconds=process['process_wall_seconds'],
        qualification=None,PPO_updates=0,formal_updates_consumed=0,
        limitations=['Decoded all4096 firstbirth/event/global-record states match; no all4096 continuous5ms state trace exists.',
            'New selected3lane trajectories have no identical selected-lane5ms reference trace in SIDE001; only shared native outputs match.',
            'No strict whole-output pass, per-sample density equivalence, physical-cause attribution, PPO benefit or deployment follows this diagnosis.'])
    (OUT/'reproduction_diagnosis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    admission=dict(schema='source021-remaining-mechanical-analysis-admission-v1',status='completed_native_mechanical_data_admitted_for_descriptive_readonly_analysis',
        strict_whole_reproduction=False,qualification=None,PPO_updates=0,formal_updates_consumed=0,
        authentic_process_exit_observed=True,authentic_process_exit_code=0,actual_policy_steps=600,actual_physics_steps=2400,
        candidate_manifest_sha256=MANIFEST_SHA,source_commit='d0dce6181df83b03fe166abcbadef3c81de2e57e',
        actual_input_sha256=inputs,diagnosis_path=str(OUT/'reproduction_diagnosis.json'),diagnosis_sha256=sha(OUT/'reproduction_diagnosis.json'),
        preserved_strict_rejection_path=str(CANDIDATE/'native_remaining_001/remaining_capture_rejection.json'),
        selected_lanes=[881,1877,2121],selected_lanes_5ms_complete=True,
        full4096_firstbirth_events_global_record_states_float_bits_exact=True,identity_teaching_override_float_bits_exact=True,
        original_control_source_RNG_birth_clocks_report_values_exact=True,payment_aggregate_float_bits_exact=False,
        purpose='Describe authenticated actual selected-lane mechanical trajectories only; not a replacement strict reproduction review.')
    (OUT/'analysis_admission.json').write_text(json.dumps(admission,indent=2,allow_nan=False)+'\n')
    (OUT/'reproduction_diagnosis.md').write_text(
        '# Remaining capture：保留严格拒绝，机械观察数据可只读分析\n\n'
        f'实际进程 exit0，{process["process_wall_seconds"]:.3f}s，4096×600 策略／2400 物理步，0 PPO。原 strict review 的拒绝不改写。\n\n'
        f'全4096首出生、{stream_checks["events.jsonl.gz"]["actual_rows"]}个事件、{stream_checks["mechanical_global_peaks.jsonl.gz"]["actual_rows"]}条全域纪录峰值状态的解压字节和解析 float bits 均与 SIDE001 相同。identity、teaching 和两份 SIDE override 回执相同；report 的行为、物理、初末 RNG、源和控制字段一致。其 gzip 文件 SHA、付款汇总 SHA、测量汇总 SHA 和时间确实不同，已逐个绑定实际文件。\n\n'
        f'付款汇总共有 {len(payment_diff)} 个 float-bit 差异：{dict(fields)}。其中 {len(zero_diffs)} 个是±0 极值，其余是 sum／mean／付款 sum／付款 mean／SD 归约尾位；最大绝对差 {max(d["absolute_difference"] for d in payment_diff):.17g}。计数、mask分区、目录、离散事件全部相同，两臂付款对账残差均为0。原观察器使用 GPU float64 scatter_add_；差异形态与归约舍入吻合，但没有逐lane density原始流，不能证明其逐样本完全一致。\n\n'
        '三lane新trace分别889／417／1005行，firstbirth连续5ms、终止前状态与pre-autoreset逐位一致。881／1877是L2，2121严格保留L1及实际step200_upright_top出生。SIDE001并未观察这些lane完整子步，不能把这些新trace称为已有逐子步对照相同。\n\n'
        '新增 analysis_admission 只授权描述这份真实机械数据，strict_whole_reproduction=false、qualification=null。不绕过原拒绝，不触发GPU复验、正式训练或部署。\n')
    for path,digest in inputs.items():require(sha(path)==digest, 'Input modified during publication')
    print(json.dumps(dict(output=str(OUT),diagnosis_sha256=sha(OUT/'reproduction_diagnosis.json'),
        admission_sha256=sha(OUT/'analysis_admission.json'),payment_diff_count=len(payment_diff),strict_whole_reproduction=False)))

if __name__=='__main__': main()
