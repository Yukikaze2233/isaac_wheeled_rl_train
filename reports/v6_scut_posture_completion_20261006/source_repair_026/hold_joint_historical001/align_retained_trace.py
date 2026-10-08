"""Align retained historical native FSM inputs with their physical endpoint.

Offline method check only: source021 REC3076, SIDE 1.5 counterfactual,
lane2216/birth1. No current source023/v24 or aggregate causal inference.
"""
from collections import Counter
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import math
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'learning_audit/exploration_gate_review_021/pd_target_payment_candidate_021/native_target_payment_001/native'
OUT = Path(__file__).resolve().parent
PINS = {
    'pd_target_payment_2216.jsonl.gz': 'eedd429ecfb2d85e52f3cce3c4849c93c41d712b701644a1cc59a8f72dd53100',
    'mechanical_5ms.jsonl.gz': '3b80d7c1a39e517687704540e799e67aeb05b00711754747d7715cf7c004749e',
    'mechanical_layout.json': '80620ab94b6cfd8f6e73d768333437b65b65717ea4e3a55a9a931de5ba30862b',
    'pd_target_payment_summary.json': '904095ec18b9c7482da79c1243dc40d2e768ffc748d6bb59cc0a4241359c328a',
}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    started = time.monotonic()
    assert all(sha(BASE / n) == h for n, h in PINS.items())
    layout = json.loads((BASE / 'mechanical_layout.json').read_text())
    control_ids = [layout['full_joint_names'].index(n) for n in layout['control_joint_names']]
    mapping = next(r for r in layout['contact_mapping'] if r['lane'] == 2216)
    sensors = dict(zip(layout['body_names'], mapping['sensor_indices'], strict=True))
    wheel_sensors = [sensors[n] for n in ('L_link3', 'R_link3')]
    physical = {}
    with gzip.open(BASE / 'mechanical_5ms.jsonl.gz', 'rt') as f:
        for line in f:
            row = json.loads(line)
            if row['state']['lane'] != 2216 or row['state']['birth_id'] != 1:
                continue
            step = row['physical_step']
            assert step not in physical
            physical[step] = row
    assert sorted(physical) == list(range(2401))
    with gzip.open(BASE / 'pd_target_payment_2216.jsonl.gz', 'rt') as f:
        policy = [json.loads(line) for line in f]
    assert len(policy) == 600
    gate = [(r, s) for r in policy for s in r['substeps']]
    assert len(gate) == 2400
    joints, truth, breaks = Counter(), Counter(), []
    mismatches = {'fresh_dq': 0.0, 'gyro': 0.0, 'gravity': 0.0}
    previous = None
    held = joint_pass = contact_available = 0
    for number, (r, s) in enumerate(gate, 1):
        assert r['lane'] == 2216 and r['birth_id'] == 1 and r['first_birth_live']
        assert s['reference_and_PD_before_physical_step'] == number
        assert s['input_physical_step'] == number - 1
        assert s['postphysics_step'] == number
        prior = physical[number - 1]
        fsm, state = s['fsm'], prior['state']
        assert fsm['native_stable_pass'] == all(fsm['native_gate_masks'].values())
        for name, actual, observed in (
            ('fresh_dq', [state['full_dq'][i] for i in control_ids], fsm['motor_dq_rad_s']),
            ('gyro', state['gyro_body'], fsm['gyro_rad_s']),
            ('gravity', state['gravity_body'], fsm['gravity']),
        ):
            mismatches[name] = max(mismatches[name], max(abs(a-b) for a, b in zip(actual, observed, strict=True)))
        contacts = prior['contacts']
        contact_result = None
        if contacts is not None:
            rows = {row['sensor_index']: row for row in contacts['sensor_body_rows']}
            def norm(sensor):
                force = [math.fsum(v[i] for v in rows[sensor]['matrix_force_n']) for i in range(3)]
                return math.sqrt(math.fsum(v*v for v in force))
            forces = [norm(i) for i in wheel_sensors]
            contact_result = {'wheel_filtered_force_n': forces,
                              'both_wheels_over_2N': all(v > 2.0 for v in forces),
                              'nonwheel_filtered_over_5N': any(norm(i) > 5.0 for i in sensors.values() if i not in wheel_sensors),
                              'physical_endpoint': number-1,
                              'scope': 'Recorded terrain-filtered matrix sum; no self-contact identity'}
        failed = tuple(k for k, passed in fsm['native_gate_masks'].items() if not passed)
        if fsm['native_holding_rl']:
            held += 1
            joint_pass += bool(fsm['native_stable_pass'])
            joints['+'.join(failed) if failed else 'all_pass'] += 1
            if contact_result:
                contact_available += 1
                truth[str((fsm['native_gate_masks']['support'], contact_result['both_wheels_over_2N']))] += 1
        if previous is not None and fsm['stable_break_count'] > previous['stable_break_count']:
            breaks.append(dict(reference_call=number, input_physical_step=number-1,
                               previous_stable_ticks=previous['stable_ticks'],
                               stable_ticks_after=fsm['stable_ticks'],
                               increment=fsm['stable_break_count']-previous['stable_break_count'],
                               failed_gates=list(failed), native_holding_rl=fsm['native_holding_rl'],
                               aligned_contact=contact_result))
        previous = fsm
    assert sum(e['increment'] for e in breaks) == previous['stable_break_count'] == 82
    summary = json.loads((BASE / 'pd_target_payment_summary.json').read_text())
    assert summary['complete'] and summary['right_censored_unfinished']
    report = dict(schema='historical-native-hold-joint-clock-alignment-v1',
                  created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_scope='source021 REC3076/SIDE1.5 diagnostic counterfactual; lane2216/birth1 only',
                  current_source023_or_v24=False, input_sha256={str(BASE/n): h for n, h in PINS.items()},
                  aligned_reference_inputs=2400, physical_rows=2401, policy_rows=600,
                  clock_rule='FSM reference k consumes physical endpoint k-1; initial endpoint0 has no contact interval',
                  maximum_input_alignment_absolute_difference=mismatches,
                  holding_samples=held, native_joint_pass_samples=joint_pass,
                  joint_failed_gate_combinations=dict(joints),
                  hold_samples_with_recorded_contact=contact_available,
                  support_proxy_vs_both_wheel_filtered_contact_counts=dict(truth),
                  native_break_count=82, break_events=breaks,
                  right_censored_unfinished=True, qualification=None, PPO_updates=0,
                  limitations=['Single selected historical birth, not pooled training statistics or current failure cause.',
                               'Terrain-filtered contact classification only; no self-contact or overflow conclusion.',
                               'Native masks/counters retained; no new reward, stability gate, simulator step or RNG.',
                               'Gate failures overlap and are temporally correlated; do not multiply marginals into1s success probability.'],
                  cpu_wall_seconds=time.monotonic()-started)
    assert all(sha(BASE/n) == h for n, h in PINS.items())
    with (OUT/'report.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('source_scope','maximum_input_alignment_absolute_difference',
          'holding_samples','native_joint_pass_samples','joint_failed_gate_combinations',
          'support_proxy_vs_both_wheel_filtered_contact_counts','native_break_count','cpu_wall_seconds')}))


if __name__ == '__main__':
    main()
