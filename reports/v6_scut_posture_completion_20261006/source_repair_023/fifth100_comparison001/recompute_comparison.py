"""Five disjoint saved native windows; root reuse of reviewed arithmetic."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
PRIOR = SOURCE / 'fourth100_comparison001/recompute_comparison.py'
spec = importlib.util.spec_from_file_location('prior_window_recomputation', PRIOR)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
reader = prior.reader


def main():
    receipt = json.loads((SOURCE / 'fifth100_window_023.read001.receipt.json').read_text())
    assert receipt['exit_code'] == 0 and receipt['complete'] is True
    inputs = dict(prior.INPUTS)
    inputs['fifth100'] = ('fifth100_window_023.read001.json', [6836, 6935], [34040, 34139],
                          receipt['report_sha256'])
    assert reader.sha(prior.REUSED) == '03a0792406fbf164ab49b36fbc2ad8a63cb3e2eb3deeb0de76ae895f799be719'
    rows = {}
    for name, (filename, rec, tb, checksum) in inputs.items():
        path = SOURCE / filename
        assert reader.sha(path) == checksum
        rows[name] = reader.audit(name, path, '100', rec, tb)
        assert rows[name]['source_commit'] == '6f07ec298e01d2686216787cfba34b394151f1aa'
        assert '/attempt_009/train/' in rows[name]['reported_crc_evidence']['file']
    values = list(rows.values())
    for before, after in zip(values, values[1:]):
        assert before['clock']['end_update'] == after['clock']['baseline_update']
        assert before['clock']['end_step'] + 1 == after['clock']['start_step']
        assert before['clock']['task_actor_updates_inclusive'][1] + 1 == after['clock']['task_actor_updates_inclusive'][0]
    latest = rows['fifth100']
    comparison = dict(scope='five_disjoint_sampled_training_windows_not_paired_policy_qualification',
                      snapshots=rows, fifth_minus_fourth=reader.paired(latest, rows['fourth100']),
                      fifth_minus_first=reader.paired(latest, rows['first100']))
    review = dict(status='saved_native_report_arithmetic_consistent',
                  review_scope='Root recomputation reusing independently reviewed arithmetic; no fresh independent review claimed',
                  complete=True, input_sha256={name: value[3] for name, value in inputs.items()},
                  reused_arithmetic_sha256=reader.sha(prior.REUSED), group_count=95, pool_count=7,
                  observed_frames_per_window=9830400, actor_updates_per_window=100, overlap_actor_updates=0,
                  fifth_REC_window=[6836, 6935], fifth_TB_window=[34040, 34139],
                  fifth_recovery_counts=latest['arithmetic']['pool_counts']['scripted_recovery'],
                  L2_scope='instantaneous_population_success_not_ended_episode_rate',
                  recovery_pool_scope='54_static_groups_mix_L0_L1_L2',
                  comparison_scope='stochastic_curriculum_DR_exposure_not_causal_entropy_AB',
                  SSH_calls=0, GPU_calls=0, PPO_updates=0, qualification=None)
    for name, result in (('comparison.json', comparison), ('review.json', review)):
        (HERE / name).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(review=review, stand=latest['ordinary_stand'], motion=latest['motion'],
                          yaw=latest['yaw_mae_rad_s'], stop=latest['conditional_stop_phases'],
                          hold=latest['hold_gate_means'], std=latest['policy_mean_std'],
                          wall=latest['iteration_timing']['iteration_wall_seconds'],
                          timing=latest['iteration_timing'],
                          valid_fraction=latest['learning_ownership']['valid_fraction'],
                          L2_success=latest['full_chain_success']), ensure_ascii=False))


if __name__ == '__main__':
    main()
