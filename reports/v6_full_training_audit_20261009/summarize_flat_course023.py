"""Summarize one complete rollout window and authenticated native comparisons."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import statistics
import subprocess
import yaml


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'flat_course_metrics023.json'
RAW = BASE / 'source029_tb2300_exact002.json'
ROUTING = BASE / 'source029_flat_runtime_clock002.json'
REVIEW = BASE / 'periodic43500_compare011_root_review001.json'
FIRST, LAST, BASELINE = 43603, 43702, 43602


def load(path):
    return json.loads(Path(path).read_text())


def pin(path):
    return dict(file=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


raw, routing, review = load(RAW), load(ROUTING), load(REVIEW)
assert raw['status'] == 'read_only_native_event_exact_window'
assert raw['actor_window']['actor_TB_steps'] == [FIRST, LAST]
assert review['status'] == 'six_actual_fresh_worlds_verified_unscored'
assert routing['owner_before'] == routing['owner_after'] == raw['owner_before']
assert all(routing['metadata_sha256'][Path(path).name] == digest
           for path, digest in raw['metadata_sha256'].items() if Path(path).name != 'source_revision_029.json')


def points(tag, *, baseline=False):
    section = raw['timing'] if tag.startswith('Timing/') else raw['selected']
    values = {int(row[1]): row[2] for row in section.get(tag, [])
              if (BASELINE if baseline else FIRST) <= int(row[1]) <= LAST}
    if not tag.startswith('Behavior/'):
        assert set(values) == set(range(BASELINE if baseline else FIRST, LAST + 1)), tag
    assert all(math.isfinite(v) for v in values.values()), tag
    return values


def weighted(tag, weight):
    values, weights = points(tag), points(weight)
    count = math.fsum(weights.values())
    return math.fsum(values[t] * weights[t] for t in values) / count if count else None


def count_delta(group, metric):
    values = points(f'Behavior/{group}/{metric}', baseline=True)
    assert BASELINE in values and LAST in values
    difference = values[LAST] - values[BASELINE]
    assert difference >= 0 and difference == round(difference)
    return int(difference)


names = ['stand_230mm', 'stand_305mm', 'stand_430mm', 'forward_2', 'forward_3',
         'forward_5', 'backward_2', 'backward_3', 'backward_5', 'rotate_1rps',
         'rotate_3rps', 'spin_translate_3rps', 'forward_3__height_train',
         'rotate_3rps__height_train', 'release_stop_translate', 'release_stop_yaw',
         'release_stop_combined']
groups = {}
for name in names:
    prefix = f'LearningAudit/{name}/'
    weights = prefix + 'rollout_samples'
    count = math.fsum(points(weights).values())
    item = dict(rollout_samples=count, population_from_24_frame_rollouts=count / 100 / 24)
    for metric in ('command_vx_m_s', 'command_yaw_rad_s', 'velocity/signed_forward_m_s',
                   'planar_speed_m_s', 'height_error_m'):
        item[metric] = weighted(prefix + metric, weights)
    if not name.startswith('stand_'):
        item['ended'] = count_delta(name, 'episodes')
        item['physically_terminated'] = count_delta(name, 'failures')
        item['boundary_truncated'] = count_delta(name, 'boundary_truncations')
        item['physical_failure_fraction_of_ended'] = (item['physically_terminated'] / item['ended']
                                                       if item['ended'] else None)
        frames = points(f'Behavior/{name}/frames', baseline=True)
        item['scored_frame_delta'] = count_delta(name, 'frames')
        assert item['scored_frame_delta'] == count
        for metric in ('vx_mae_m_s', 'yaw_mae_rad_s', 'height_mae_m'):
            cumulative = points(f'Behavior/{name}/{metric}', baseline=True)
            assert BASELINE in cumulative and LAST in cumulative
            # These means are exported as float scalars; cancellation adds
            # rounding error. Native per-frame counts remain exact integers.
            item['reconstructed_' + metric] = ((cumulative[LAST] * frames[LAST]
                - cumulative[BASELINE] * frames[BASELINE]) / item['scored_frame_delta'])
    if name.startswith('release_stop_'):
        stop = prefix + 'settled_stop/samples'
        item['settled_stop_samples'] = math.fsum(points(stop).values())
        item['settled_stop_speed_m_s'] = weighted(prefix + 'settled_stop/mean_speed_m_s', stop)
        phases = {}
        for phase in ('accelerating', 'cruising', 'decelerating', 'stationary_window'):
            root = prefix + f'release_posture/{phase}/'
            samples = root + 'samples'
            phases[phase] = dict(samples=math.fsum(points(samples).values()),
                pitch_abs_mean_deg=math.degrees(weighted(root + 'pitch_abs_rad', samples)),
                roll_abs_mean_deg=math.degrees(weighted(root + 'roll_abs_rad', samples)))
        item['conditional_posture'] = phases
    groups[name] = item

reference_repo = BASE.parents[2] / 'scut-wheeled-legged-rl'
revision = 'b8ff79f3df855faf9dc92f4a282bd80c42649466'
reference_path = 'pretrained/26_infantry/flat_and_rotation/2026-07-29_23-20-03/params/env.yaml'
reference_bytes = subprocess.check_output(['git', '-C', str(reference_repo), 'show', revision + ':' + reference_path])
reference = yaml.load(reference_bytes, Loader=yaml.BaseLoader)
command = reference['commands']
reference_fields = {key: command[key] for key in ('resampling_time_range', 'rel_standing_envs',
    'rel_heading_envs', 'heading_command', 'ranges', 'special_modes', 'special_mode_min_episode_time')}
reference_fields['reward_weights'] = {key: reference['rewards'][key] for key in (
    'stand_still_lin_vel', 'no_fork', 'no_fork_square', 'flat_orientation_x_v',
    'flat_orientation_y_v', 'track_lin_vel_xy', 'track_lin_vel_xy_square',
    'track_ang_vel_z', 'track_ang_vel_z_square')}
posture_floor = ['pitch_exp', 'roll_exp', 'locomotion_pitch_floor', 'locomotion_roll_floor',
                 'pitch_velocity', 'roll_velocity']
for name in ('forward_5', 'rotate_1rps', 'rotate_3rps'):
    groups[name]['posture_reward_density_mean_per_s'] = {
        term: weighted(f'LearningAudit/{name}/reward_density/{term}',
                       f'LearningAudit/{name}/rollout_samples') for term in posture_floor}

result = dict(schema='flat-course-readonly-audit-v1',
    at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='complete100_rollout_and_actual_native_worlds_summarized',
    input_pins=[pin(path) for path in (RAW, ROUTING, REVIEW)], actor_window=raw['actor_window'],
    scope=dict(rollout='sampled actions and domain randomization; not deterministic qualification',
        native='six serial fresh CPU worlds; fixed means; original physical completion predicates',
        behavioral_deltas='exact endpoint counter differences; reconstructed MAE uses float cumulative scalars',
        rollout_tilt_max='not reconstructed: cumulative maximum cannot yield a window maximum'),
    timing_mean_seconds={key: statistics.fmean(points(key).values()) for key in raw['timing'] if key.endswith('seconds')},
    groups=groups, latest_scalar_snapshots=routing['scalars'],
    native_protocols=review['protocols'], native_review=pin(REVIEW),
    command_source=dict(commit=revision, file=reference_path,
        sha256=hashlib.sha256(reference_bytes).hexdigest(), fields=reference_fields),
    concerns=dict(difficulty='native origin 0; 360 updates expands velocity, yaw and moving-height together',
        population='189 routed rows; selected nominal high-motion rows have 11 environments each',
        symmetry='no_fork controls wheel fore-aft mismatch; full mirrored leg length/angle symmetry was not recorded',
        completed_motor_response='unqualified leg data is not credited as an identified actuator'),
    unique_training_cause_established=False, new_model_stronger_established=False,
    hardware_deployment_ready=False, qualification=None, PPO_updates_by_analysis=0,
    remote_writes_by_analysis=0)
assert not OUTPUT.exists()
OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(status=result['status'], groups=len(groups), file=str(OUTPUT))))
