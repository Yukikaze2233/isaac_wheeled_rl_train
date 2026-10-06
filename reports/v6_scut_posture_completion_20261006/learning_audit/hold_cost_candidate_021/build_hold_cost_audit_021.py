"""Reduce already published real native ledgers; no simulator/model execution."""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).parent
AUDIT = HERE.parent
BASE = Path('/tmp/scut-com-frozen-020-w14p1d1x')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(fields, name):
    s = fields[name]
    mean, std = s['mean'], s['standard_deviation']
    return dict(mean=mean, standard_deviation=std, rms=math.sqrt(mean * mean + std * std),
                minimum=s['minimum'], maximum=s['maximum'])


def main():
    bindings = {}; arms = {}
    limits = {'wheel_rim_max_m_s': .25, 'leg_max_rad_s': 2., 'gyro_all_norm_rad_s': .75}
    dq_fields = ['dq_L_joint1_rad_s', 'dq_LL_joint1_rad_s', 'dq_L_joint3_rad_s',
                 'dq_R_joint1_rad_s', 'dq_RR_joint1_rad_s', 'dq_R_joint3_rad_s']
    gyro_fields = ['gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']
    taxes = ['leg_velocity', 'wheel_velocity', 'horizontal_angular_velocity',
             'dense_yaw_broad', 'dense_yaw_precision', 'dense_stationary_broad',
             'dense_stationary_precision', 'stand_still_lin_vel']
    for arm in ('mean', 'stochastic'):
        directory = AUDIT / ('native_payment_' + arm + '_001')
        summary = json.loads((directory / 'payment_summary.json').read_text())
        report = json.loads((directory / 'report.json').read_text())
        identity = json.loads((directory / 'identity.json').read_text())
        assert report['status'] == 'native600_completed_preclose'
        assert report['payment_summary_sha256'] == sha(directory / 'payment_summary.json')
        assert report['identity_sha256'] == sha(directory / 'identity.json')
        assert summary['status'] == 'native_payments_reconciled_preclose'
        assert summary['complete'] and summary['policy_steps'] == 600 and summary['event_producer_calls'] == 600
        assert summary['native_lane_intervals'] == 2457600 and summary['max_reconciliation_residual'] == 0
        assert report['policy_steps_completed'] == 600 and report['physical_steps_completed'] == 2400
        assert report['runtime_preflight_passed'] and report['payment_hooks_restored_before_app_close']
        assert report['PPO_updates'] == report['optimizer_steps'] == report['formal_updates_consumed'] == 0
        for key in ('actor_tensors_and_saved_std_unchanged', 'curriculum_clocks_unchanged',
                    'learning_transition_clock_unchanged', 'checkpoint_unchanged', 'source_files_unchanged',
                    'seal_contract_source_manifest_wrapper_unchanged', 'payment_artifacts_unchanged',
                    'initial_birth_and_private_stream_match_original2579'):
            assert report[key]
        source_reward = identity['source_root'] + '/src/wheeled_tasks/chassis/rewards.py'
        assert identity['source_files_sha256'][source_reward] == sha(BASE / 'src/wheeled_tasks/chassis/rewards.py')
        bindings[arm] = dict(directory=str(directory),
            payment_summary_sha256=sha(directory / 'payment_summary.json'),
            report_sha256=sha(directory / 'report.json'), identity_sha256=sha(directory / 'identity.json'),
            source_commit=identity['source_commit_label'],
            unchanged_reward_sha256=identity['source_files_sha256'][source_reward],
            initial_physical_birth_sha256=report['initial_physical_birth_sha256'],
            initial_env_rng_sha256=report['initial_env_rng_sha256'],
            sampling_rng_initial_sha256=report['sampling_rng_initial_sha256'],
            native_effective_std=report['native_effective_std'],
            policy_steps=600, physics_steps=2400, native_lane_intervals=2457600,
            max_payment_residual=0., PPO_updates=0, qualification=None)
        rows = []
        for r in summary['rows']:
            if r['scope'] != 'first_birth_intervals' or r['payment_partition'] != 'recovery_actor_valid_still_holding':
                continue
            f = r['components']
            rows.append(dict(lesson=r['lesson'], original_goal=r['original_goal'],
                lane_intervals=r['policy_lane_intervals'], lane_seconds=r['lane_seconds'],
                immediate_reward_per_policy_interval=stat(f, 'actor_eligible_immediate_reward'),
                immediate_reward_divided_by_policy_dt=f['actor_eligible_immediate_reward']['mean'] / .02,
                positive_tracking_density=sum(f[k]['mean'] for k in ('track_lin_vel','track_yaw','track_height')),
                positive_orientation_density=sum(f[k]['mean'] for k in ('pitch_exp','roll_exp')),
                old_motion_and_stationary_density={k:stat(f,k) for k in taxes},
                physical_motion={k:{**stat(f,k), 'gate_limit':limit,
                    'mean_divided_by_gate':f[k]['mean']/limit} for k,limit in limits.items()},
                physical_planar_speed=stat(f,'planar_speed_world_m_s'),
                signed_CONTROL_velocity={k:stat(f,k) for k in dq_fields},
                signed_body_gyro={k:stat(f,k) for k in gyro_fields},
                last_feedback_pass_fraction={k:f['sensor_'+k+'_pass_at_last_reference']['mean']
                    for k in ('valid','support','body_clear','tilt','gyro','height','wheel_speed','leg_speed')},
                last_feedback_all_eight_pass_fraction=f['sensor_all_eight_pass_at_last_reference']['mean'],
                sensor_hold_peak_seconds=stat(f,'sensor_hold_peak_s'),
                actual_contact_hold_peak_seconds=stat(f,'actual_contact_hold_peak_s')))
        arms[arm] = rows
    assert bindings['mean']['initial_physical_birth_sha256'] == bindings['stochastic']['initial_physical_birth_sha256']
    assert bindings['mean']['initial_env_rng_sha256'] == bindings['stochastic']['initial_env_rng_sha256']
    assert bindings['mean']['sampling_rng_initial_sha256'] == bindings['stochastic']['sampling_rng_initial_sha256']
    result = dict(schema='hold-cost-actual-payment-audit-021-v1',
        status='evidence_bound_proposal_not_native_candidate_validated_not_deployed',
        bindings=bindings, first_birth_actor_valid_still_holding=arms,
        conclusions=[
            'Existing quiet costs are active and negative in both real native arms; absence of payment is disproved.',
            'Signed joint/gyro means hide sizeable RMS and maxima; body planar quiet alone is not the settling gate.',
            'The measured stochastic arm has more wheel/leg/allgyro motion and fewer last-feedback passes than mean under the same initial birth/private stream; this is not sufficient entropy/optimizer causality.',
            'The new gate-relative broad+fine coefficients are proposals using native widths and previously reviewed cost shapes; no PPO benefit has been established.'],
        limits=[
            'Each first-born still-holding bin is conditional on native actor ownership/hold duration; mean and stochastic memberships differ.',
            'Physical end-of-policy dq/gyro and final delayed feedback gates have different clocks; they are not the same instantaneous stream.',
            'Last-feedback pass fractions are not independent per-tick probabilities or continuous-one-second success rates.',
            'Ledgers contain moments, not all state traces. A nonlinear replacement expected payment cannot be computed from aggregate means.',
            'Immediate mask-eligible reward is not PPO advantage or optimizer credit. Reward-per-second cannot be compared to entropy-loss units.',
            'Observed source is sealed historical2579/source019; current live source020 behavior requires its own authenticated window.'],
        candidate_physical_widths={'leg':{'broad_rad_s':2.,'fine_rad_s':.5},
            'wheel':{'broad_rim_m_s':.25,'fine_rim_m_s':.1},
            'gyro':{'broad_rad_s':.75,'fine_rad_s':.2}},
        native_candidate_validation_remaining=True, PPO_updates=0, GPU_runs_by_this_audit=0,
        production_changes=[], qualification=None)
    (HERE / 'actual_payment_audit_021.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'sha256':sha(HERE/'actual_payment_audit_021.json'),
        'first_born_hold_bins':{arm:len(rows) for arm,rows in arms.items()}},indent=2))


if __name__ == '__main__': main()
