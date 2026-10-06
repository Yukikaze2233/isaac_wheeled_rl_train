#!/usr/bin/env python3
"""Read only three preserved JSONs; classify posture evidence without reward inversion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent.parent
INPUTS = {
    HERE.parent / 'latest900_read001/report.json': 'cc6e87e2c2863cb5071c00a86d7cbfae963f2d072f401fee0559279eba929222',
    AUDIT / 'tb_http_course_review_022/read001/curves.json': '17beece01856be02cba92d1f194dac772e2fdfccace2649a9979441486e0c6a7',
    AUDIT / 'tb_http_course_review_022/read002/curves.json': 'b630b5e68508eb59e3c9c647a6531c12ab45b87c1b8acde5be77c76c268bcd82',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert all(sha(path) == value for path, value in INPUTS.items())
documents = [json.loads(path.read_text()) for path in INPUTS]
report = documents[0]
assert report['complete'] and report['status'] == 'requested_actor_windows_complete'
assert report['qualification'] is None and report['PPO_updates_by_reader'] == 0
window = report['windows']['100']
assert window['window']['end_update'] == 900 and window['window']['actor_updates'] == 100
groups = window['behavior']['groups']
assert len(groups) == 95 and all(group['complete'] for group in groups.values())
postures = {name: group['posture'] for name, group in groups.items()}
assert all(posture['interval_max_available'] is False and posture['separated_pitch_roll_available'] is False for posture in postures.values())
posture_fields = sorted(set(key for posture in postures.values() for key in posture))
assert posture_fields == ['interval_max_available', 'separated_pitch_roll_available',
    'tilt_max_deg_since_observer_birth_after', 'tilt_max_deg_since_observer_birth_before']
selected = ('stand_230mm', 'stand_305mm', 'stand_430mm', 'forward_05', 'backward_05',
    'recovery_upright_stand', 'recovery_crouched_stand', 'recovery_left_side_90_stand',
    'recovery_right_side_90_stand', 'recovery_pitch_forward_45_stand', 'recovery_prone_back_legs_above_stand')
gates = {}
for name in ('valid', 'support', 'body_clear', 'tilt', 'gyro', 'height', 'wheel_speed', 'leg_speed'):
    tag = '/recovery/hold_gate/' + name + '_pass_fraction'
    row = window['teaching']['hold_gates'][tag]
    assert row['complete'] and row['records'] == row['expected_records'] == 100
    assert 0 <= row['min'] <= row['mean'] <= row['max'] <= 1
    gates[name] = dict(tag=tag, mean_fraction=row['mean'], mean_percent=row['mean'] * 100,
        min_logged_rollout_fraction=row['min'], max_logged_rollout_fraction=row['max'], records=100)

curve_tags = sorted(tag for document in documents[1:] for tag in document['curves'])
assert len(curve_tags) == 15
saved_measured_angle_curves = [tag for tag in curve_tags if any(axis in tag for axis in ('pitch', 'roll', 'tilt'))]
assert saved_measured_angle_curves == []
recovery = [posture for name, posture in postures.items() if report['identity']['group_pools'][name] == 'scripted_recovery']
assert len(recovery) == 54
reward_section = window['optional_posture_reward_density']
assert reward_section['scope'] == 'Reward densities/costs, not measured pitch/roll angles.'

result = dict(schema='source022-saved-posture-evidence-audit-v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
    status='saved_evidence_classified_current_angles_unavailable', source_commit=report['source_commit'],
    input_sha256={str(path): value for path, value in INPUTS.items()}, input_after_sha256={str(path): sha(path) for path in INPUTS},
    inputs_unchanged=True, script_sha256=sha(__file__),
    latest_window=dict(actor_updates=[801, 900], REC_actor_updates=[4672, 4771], dense_native_logged_updates=[31612, 31711],
        baseline_native_logged_update=31611, expected_lane_frames=9830400, policy_seconds_per_environment=48),
    availability_scope='Only the saved latest900 report and two saved15-tag HTTP JSONs; no server, source or raw event was queried.',
    measured_posture_evidence=dict(groups=95, native_fields=posture_fields,
        quantity='Two endpoint snapshots of maximum tilt in degrees since observer birth',
        interval_max_available_count=0, separated_pitch_roll_available_count=0,
        selected_groups={name: postures[name] for name in selected},
        recovery54_groups_historical_after_range_deg=[min(p['tilt_max_deg_since_observer_birth_after'] for p in recovery),
            max(p['tilt_max_deg_since_observer_birth_after'] for p in recovery)],
        interpretation='Historical since-birth extrema are measured angles but are not current angles, latest100 averages or latest100 interval extrema. Do not subtract extrema to produce an angle.'),
    recovery_hold_gates=dict(scope='100-rollout time means of separately logged instantaneous RL-hold conditional pass fractions',
        gates=gates, mean_hold_population=window['teaching']['hold_gates']['/recovery/hold_population']['mean'],
        native_scope_statements=window['teaching']['scopes'],
        continuously_stable_1s_joint_probability=None,
        limitation='Marginal gate means provide neither simultaneous joint pass rate nor continuous1s dwell; tilt79.3783% does not establish posture recovery.'),
    historical_stability_statistic=dict(tag='/recovery/stable_peak_seconds_mean',
        mean_seconds=window['teaching']['hold_gates']['/recovery/stable_peak_seconds_mean']['mean'],
        scope='Time-mean of the logged historical stable-peak statistic; not current dwell, maximum individual dwell or1s completion probability.'),
    saved_HTTP=dict(tags=curve_tags, measured_pitch_roll_tilt_curve_tags=saved_measured_angle_curves,
        posture_related_gate_curves=[tag for tag in curve_tags if '/hold_gate/' in tag],
        sampling='TensorBoard scalar API reservoir; absent from these queried files does not prove absence from all server tags.'),
    reward_entries=dict(scope=reward_section['scope'], groups=len(reward_section['groups']),
        field_names=sorted(set(field for row in reward_section['groups'].values() for field in row)),
        inversion_performed=False, measured_angle_derived_from_reward=False),
    missing_actual_posture_quantities=['Current signed pitch/roll per lane or group',
        'Current tilt angle or latest100 tilt mean/quantiles', 'Latest100 interval maximum tilt',
        'Latest100 measured pitch/roll absolute angle means or quantiles', 'Measured body gyro norm/axis mean and quantiles',
        'Posture angle time series for standing, forward/backward and recovery',
        'Continuous joint eight-gate1s dwell distribution/completion probability'],
    SSH_actions=0, GPU_actions=0, PPO_updates=0, qualification=None)

rows = ['现有保存材料不能给出当前俯仰／横滚角各是多少度。真实角度字段只有95组的“观察器出生以来最大倾角”两端快照；全部明确不提供最新100窗最大值或分开的pitch／roll。',
    '', '| 组 | 历史倾角峰：窗口前 (°) | 窗口后 (°) |', '|---|---:|---:|']
for name in selected[:5]:
    row = postures[name]
    rows.append(f"| {name} | {row['tilt_max_deg_since_observer_birth_before']:.4f} | {row['tilt_max_deg_since_observer_birth_after']:.4f} |")
rows.extend(['', f"54个恢复组的窗口后历史峰范围为{result['measured_posture_evidence']['recovery54_groups_historical_after_range_deg'][0]:.4f}–180°；它覆盖先前历史，不能理解成当前全部倒置。静站305mm的历史峰从61.4741到62.2479°，仍不是当前姿态或窗口均值。",
    '', '最新actor801–900的恢复RL-hold八门如下。每项是单门即时通过比例的100次rollout时间均值，不是联合连续1s稳定率。',
    '', '| 门 | 通过比例均值 |', '|---|---:|'])
for name, row in gates.items():
    rows.append(f"| {name} | {row['mean_percent']:.4f}% |")
rows.extend(['', 'tilt约79.38%只能说明该边际门的通过比例。gyro约23.41%、轮速约14.78%、腿速约21.25%；不能相乘或把79%当接稳成功率。原生历史stable_peak时间均值约8.176ms，也不代表当前连续稳定时长或1s完成概率。',
    '', '两份保存的TB HTTP JSON共15条曲线，姿态相关仅gyro／轮速／腿速边际门；没有pitch／roll／tilt角曲线。这里只判断已查询材料，不声称服务器所有tag均缺失。',
    '', 'pitch_exp、roll_exp、pitch_velocity、roll_velocity等条目属于reward density／cost。没有从奖励反演角度或角速度。当前／窗口内姿态角均值、分位数、曲线、独立窗口峰以及连续八门联合1s概率均缺失。',
    '', '仅本机JSON审计，原三份JSON逐SHA保持；0SSH／0GPU／0PPO，qualification=null。', ''])
for name, body in (('posture_audit.json', json.dumps(result, indent=2, allow_nan=False) + '\n'), ('posture_audit.md', '\n'.join(rows))):
    with (HERE / name).open('x') as output:
        output.write(body)
assert all(sha(path) == value for path, value in INPUTS.items())
print(json.dumps(dict(status=result['status'], JSON_sha256=sha(HERE / 'posture_audit.json'), markdown=str(HERE / 'posture_audit.md'),
    input_files_unchanged=True, measured_current_pitch_roll_available=False, tilt_gate_mean_percent=gates['tilt']['mean_percent'], gyro_gate_mean_percent=gates['gyro']['mean_percent'])))
