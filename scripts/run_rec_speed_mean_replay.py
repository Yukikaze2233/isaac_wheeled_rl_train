#!/usr/bin/env python3
"""Isolated, unscored native mean replay; CPU preparation is the default.

Only --execute starts native replay. The default starts five serial processes;
each reuses the unchanged e5 FixedCaseEnv for both authenticated ONNX models.
The hidden single-candidate child uses one fresh world and a shared pinned projection.
No runner, optimizer, train admission, new FSM or qualification is constructed.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_rec_speed_mean_replay as prep

PROTOCOLS = ('nominal', 'vx_positive', 'vx_negative', 'yaw_positive', 'yaw_negative')
EXPLICIT_PROTOCOLS = (*PROTOCOLS, 'spin_positive', 'spin_negative')
SCENES = (*prep.STANDS, *prep.FALLEN)
SEED = 290929
COUNT = 28
CONDITIONS = ('valid', 'support', 'body_clear', 'tilt', 'gyro', 'height', 'wheel_speed', 'leg_speed')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def environment_stage(config):
    stages = config['enabled_stages']
    if len(stages) != 1 or stages[0] not in {row['name'] for row in config['stages']}:
        raise ValueError('Replay requires one actual enabled environment stage')
    return stages[0]


def verify_child_completion(output, protocol):
    result = prep.read(Path(output) / 'native_replay.json')
    if (result.get('status') != 'native_mean_replay_completed_unscored'
            or result.get('protocol') != protocol or result.get('PPO_updates') != 0
            or result.get('optimizer_constructed') is not False
            or result.get('qualification', 'missing') is not None
            or len(result.get('candidates', [])) != 2):
        raise ValueError('Native child report is failed or incomplete; process exit zero is insufficient')
    return result


def verify_prepared_projection(root, directory, protocol, receipt_sha256):
    """Read one shared, externally pinned CPU projection without rebinding paths."""
    if (protocol not in EXPLICIT_PROTOCOLS or not isinstance(receipt_sha256, str)
            or len(receipt_sha256) != 64
            or any(c not in '0123456789abcdef' for c in receipt_sha256)):
        raise ValueError('Shared prepared projection requires an explicit receipt SHA256 and protocol')
    directory = Path(directory).resolve(strict=True)
    paths = {key: directory / name for key, name in (
        ('receipt', 'projection_receipt.json'), ('config', 'interactive_unscored_contract.json'),
        ('profile', 'interactive_unscored_expectations.json'), ('changes', 'projection_changes.json'))}
    pins = {key: prep.digest(path) for key, path in paths.items()}
    if pins['receipt'] != receipt_sha256:
        raise ValueError('Shared prepared projection receipt SHA mismatch')
    receipt, config, profile, changes = (prep.read(paths[key]) for key in ('receipt', 'config', 'profile', 'changes'))
    if any(receipt.get(key + '_sha256') != pins[name] for key, name in
           (('config', 'config'), ('profile', 'profile'), ('projection_changes', 'changes'))):
        raise ValueError('Shared prepared projection file SHA mismatch')
    if config.get('design_preflight') != dict(file=str(paths['profile']), sha256=pins['profile'], stage='recovery'):
        raise ValueError('Shared prepared projection must retain its exact expectations path binding')
    if (profile.get('reset_protocol') != protocol or profile.get('run_scope') != 'interactive_unscored'
            or profile.get('qualification', 'missing') is not None
            or profile.get('baseline_score', 'missing') is not None
            or profile.get('internal_validation_role') != 'baseline'
            or profile.get('source_role') != 'full'
            or profile.get('original_source_controller_readback_required') is not True):
        raise ValueError('Shared prepared projection scope or protocol differs')
    source, _, full, design, _, control, original = prep.original_full(root, root / prep.REC)
    *_, flat_receipt = prep.original_full(root, root / prep.FLAT)
    settings = source['recovery_training']
    if (prep.digest(root / prep.REC) != prep.REC_SHA
            or settings != design['stages']['recovery']['roles']['train']['contract']['recovery_training']
            or settings.get('side_speed_rad_s') != 1.5 or settings.get('complete_started_blend') is not True
            or profile.get('source_contract_sha256') != prep.REC_SHA
            or profile.get('source_design_sha256') != source['design_preflight']['sha256']
            or profile.get('controller_source_role') != 'train'
            or profile.get('controller_source_sha256') != source['design_preflight']['sha256']):
        raise ValueError('Shared prepared projection lost the original authenticated controller')
    cases = prep.selected_cases(source, protocol)
    expected_config = prep.projection(full, cases, settings)
    expected_config['design_preflight'] = config['design_preflight']
    expected_changes = {key: dict(before=full.get(key), after=config.get(key))
        for key in sorted(full.keys() | config.keys())
        if key != 'design_preflight' and full.get(key) != config.get(key)}
    if (config != expected_config or changes != expected_changes or receipt.get('case_names') != list(SCENES)
            or receipt.get('source_full_role') != original
            or receipt.get('independent_stand_full_role') != flat_receipt
            or receipt.get('control_protocol') != settings
            or receipt.get('observation_noise_enabled') != config['observation_noise_enabled']
            or receipt.get('randomization') != {key: config[key] for key in
                ('usb_transport', 'command_transport', 'signal_delay', 'signal_perturbations',
                 'dynamics_randomization', 'contact_domain')}):
        raise ValueError('Shared prepared projection differs from its full-role source projection')
    from wheeled_tasks.chassis.preflight import validate_design_contract, validate_response_evidence
    from wheeled_tasks.chassis.response_migration import validate_recovery_configuration, validate_step_task_configuration
    authenticated = validate_design_contract(config, profile, control, stage='recovery',
        role='baseline', seed=SEED, num_envs=COUNT, block_index=0)
    authenticated['response_evidence'] = validate_response_evidence(config, root)
    authenticated['recovery_configuration'] = validate_recovery_configuration(settings, root)
    authenticated['step_task_configuration'] = validate_step_task_configuration(config['step_jump_task'], root)
    authenticated['coverage'] = dict(independent_design=True, response_evidence=True, recovery_configuration=True,
        step_task_configuration=True, original_full_budgets=True, projected_production_budget=False,
        train_chassis_admission=False, native_runtime_readback=False)
    if receipt.get('projection') != authenticated:
        raise ValueError('Shared prepared projection receipt differs from native CPU authentication')
    # The same frozen paths and bytes must survive all CPU authentication reads.
    if any(prep.digest(path) != pins[key] for key, path in paths.items()):
        raise ValueError('Shared prepared projection changed during authentication')
    binding = dict(directory=str(directory), protocol=protocol, receipt_sha256=pins['receipt'],
        config_sha256=pins['config'], profile_sha256=pins['profile'],
        projection_changes_sha256=pins['changes'])
    return config, receipt, binding


def validate_single_mode(args):
    index = args.candidate_index
    prepared = args.prepared_projection is not None or args.projection_receipt_sha256 is not None
    if index is None:
        if prepared:
            raise ValueError('Prepared projection arguments require candidate-index')
        return
    if (type(index) is not int or index not in (0, 1) or not args.execute
            or args.protocol not in EXPLICIT_PROTOCOLS or args.interface_smoke
            or args.prepared_projection is None or args.projection_receipt_sha256 is None):
        raise ValueError('candidate-index requires execute, one protocol and an explicitly pinned prepared projection')


def verify_single_measurement(measurement, protocol, config):
    """Require usable ended ledgers and metrics, without scoring task outcomes."""
    events, fingerprint = measurement['native_events'], measurement['reset_fingerprint']
    seconds = {case['name']: case['episode_seconds'] for case in config['evaluation']['cases']}
    if measurement['policy_ticks'] > round(max(seconds.values()) / .02) + 1:
        raise ValueError('Single native child exceeded the original native case bound')
    if events.get('run_scope') != 'interactive_unscored' or events.get('qualification', 'missing') is not None:
        raise ValueError('Single native child event scope differs')
    for row in events['replicas']:
        terminal = row.get('terminal_s')
        if (type(terminal) not in (int, float) or not math.isfinite(terminal) or terminal <= 0
                or terminal > min(seconds[row['case']], measurement['policy_ticks'] * .02) + 1e-8):
            raise ValueError('Single native child requires actual terminal times')
    from wheeled_tasks.chassis.evaluation import RequestProtocolMetrics, recovery_episode_diagnostics
    for name, group in events['groups'].items():
        ended = 4 if name in prep.FALLEN else 0
        if (type(group.get('native_recovery_ended_count')) is not int
                or group['native_recovery_ended_count'] != ended
                or len(group.get('native_recovery_episodes', [])) != ended
                or len(group.get('native_recovery_diagnostics', [])) != ended):
            raise ValueError('Single native child requires the complete native recovery ended ledger')
        for episode in group['native_recovery_episodes']:
            if any(type(episode.get(key)) not in (int, float) or not math.isfinite(episode[key])
                   for key in RequestProtocolMetrics.RECOVERY_COUNTERS):
                raise ValueError('Single native child recovery ended ledger lacks native counters')
        for diagnostic in group['native_recovery_diagnostics']:
            if (not isinstance(diagnostic.get('stage'), str) or not diagnostic['stage']
                    or not isinstance(diagnostic.get('reason'), str) or not diagnostic['reason']
                    or not isinstance(diagnostic.get('failed_checks'), list)):
                raise ValueError('Single native child recovery ended diagnostics are incomplete')
        if group['native_recovery_diagnostics'] != recovery_episode_diagnostics(
                {'recovery_episodes': group['native_recovery_episodes']}, ended):
            raise ValueError('Single native child diagnostics differ from the original native ended ledger')
    for key in ('native_episode_metrics', 'fixed_quiet_window_metrics'):
        report = measurement[key]
        if report.get('sample_rate_hz') != 50. or report.get('warmup_seconds') != 0.:
            raise ValueError('Single native child requires original 50 Hz metrics without warmup')
        for name, group in report['groups'].items():
            for counter in ('frames', 'observed_frames', 'episodes', 'failures', 'timeouts', 'successes', 'boundary_truncations'):
                if type(group.get(counter)) is not int or group[counter] < 0:
                    raise ValueError('Single native child requires actual metric counters')
            if key == 'native_episode_metrics' and (group['frames'] <= 0 or group['episodes'] != 4):
                raise ValueError('Single native child requires four ended first episodes per scene')
            if key == 'native_episode_metrics':
                frames = sum(round(row['terminal_s'] / .02) for row in events['replicas'] if row['case'] == name)
                if group['frames'] != frames or group['observed_frames'] != frames:
                    raise ValueError('Single native child metric denominator differs from actual terminal frames')
            for metric in ('vx_actual_mean_m_s', 'vx_mae_m_s', 'yaw_mae_rad_s', 'height_mae_m',
                           'planar_speed_mean_m_s', 'stand_drift_max_m', 'tilt_max_deg', 'closure_gap_max_m'):
                if type(group.get(metric)) not in (int, float) or not math.isfinite(group[metric]):
                    raise ValueError('Single native child requires complete finite physical metrics')
            if not isinstance(group.get('termination_reasons'), dict) or not group['termination_reasons']:
                raise ValueError('Single native child requires native termination reason counts')
            if any(type(value) is not int or value < 0 for value in group['termination_reasons'].values()):
                raise ValueError('Single native child termination reason counters differ')
    digest = fingerprint.get('actor_observation_sha256')
    if (not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
            or any(len(fingerprint.get(key, [])) != COUNT for key in
                ('clone_indices', 'fixed_birth_samples', 'root_pose', 'motor_positions', 'masses', 'friction'))):
        raise ValueError('Single native child requires its actual native reset fingerprint')
    target = None if protocol == 'nominal' else 'stand_305mm'
    if measurement.get('fixed_quiet_window_target_group', 'missing') != target:
        raise ValueError('Single native child quiet metric target differs')


def verify_single_child_completion(output, protocol, candidate_index, identities, root,
                                   prepared_projection, projection_receipt_sha256, shared_abi):
    """Authenticate one independent world; never relax the existing pair guard."""
    if type(candidate_index) is not int or candidate_index not in (0, 1) or len(identities) != 2:
        raise ValueError('Single native child requires the original pair and a valid selected index')
    root = prep.frozen_runtime(root)
    config, receipt, binding = verify_prepared_projection(root, prepared_projection, protocol, projection_receipt_sha256)
    result = prep.read(Path(output) / 'native_replay.json')
    canonical_sha(result)  # Reject nonfinite JSON anywhere in the physical result.
    candidates = result.get('candidates', [])
    if (result.get('status') != 'native_mean_replay_completed_unscored'
            or result.get('run_scope') != 'interactive_unscored' or result.get('protocol') != protocol
            or type(result.get('PPO_updates')) is not int or result['PPO_updates'] != 0
            or result.get('optimizer_constructed') is not False
            or result.get('qualification', 'missing') is not None
            or result.get('world_mode') != 'independent_world'
            or type(result.get('selected_candidate_index')) is not int
            or result['selected_candidate_index'] != candidate_index
            or result.get('selected_identity') != identities[candidate_index]
            or result.get('original_candidates') != list(identities)
            or result.get('shared_abi') != shared_abi or result.get('prepared_projection') != binding
            or len(candidates) != 1 or candidates[0].get('identity') != identities[candidate_index]
            or type(candidates[0].get('original_candidate_index')) is not int
            or candidates[0].get('original_candidate_index') != candidate_index):
        raise ValueError('Single native child report is failed, mismatched or incomplete; exit zero is insufficient')
    constructor = result.get('native_constructor', {})
    if (type(constructor.get('pid')) is not int or constructor['pid'] <= 0
            or type(constructor.get('count')) is not int or constructor['count'] != 1
            or constructor.get('fresh_process') is not True
            or constructor.get('runtime_root') != str(Path(root).resolve())
            or constructor.get('source_runtime_commit') != prep.E5
            or constructor.get('environment_class') != 'wheeled_tasks.chassis.eval_env.FixedCaseEnv'
            or result.get('bootstrap_call') != [dict(stage=environment_stage(config), role='baseline',
                seed=SEED, num_envs=COUNT, block_index=0)]):
        raise ValueError('Single native child lacks its independent authenticated native constructor')
    from wheeled_tasks.chassis.preflight import validate_runtime_state
    from wheeled_tasks.chassis.recovery_script_revision import runtime_expectation
    readback, startup = result.get('native_readback', {}), result.get('startup_report', {})
    for runtime in (readback.get('runtime_preflight', {}), startup.get('runtime_preflight', {})):
        if runtime.get('status') != 'runtime_configuration_passed' or not isinstance(runtime.get('observed'), dict):
            raise ValueError('Single native child requires complete native runtime readback')
        validate_runtime_state(receipt['projection']['runtime_expected'], runtime['observed'])
    expected_controller = dict(runtime_expectation(prep.read(root / prep.REC)), complete_started_blend=True)
    timing = dict(physics_dt=.005, policy_dt=.02, decimation=4, pc_control_dt=.005,
        target_update='once_per_policy_step', feedback_control='after_each_physical_state_refresh',
        runtime_substep_verification='pending_first_step')
    if (readback.get('qualification', 'missing') is not None or readback.get('controller') != expected_controller
            or readback.get('control_timing') != timing or startup.get('control_timing') != timing
            or startup.get('num_envs') != COUNT
            or startup.get('scene_group_counts') != {name: 4 for name in SCENES}
            or startup.get('design_preflight') != receipt['projection']):
        raise ValueError('Single native child controller, timing or constructor readback differs')
    measurement = candidates[0]
    ticks, audit = measurement.get('policy_ticks'), measurement.get('control_substep_readback', {})
    fingerprint = measurement.get('reset_fingerprint', {})
    replicas = measurement.get('native_events', {}).get('replicas', [])
    if (type(ticks) is not int or ticks <= 0
            or audit != dict(policy_steps=ticks, physics_steps=4 * ticks, target_decodes=1,
                pd_updates=4, feedback_refreshes=4, physics_dt=.005, policy_dt=.02)
            or measurement.get('right_censored') != 0
            or set(measurement.get('native_events', {}).get('groups', {})) != set(SCENES)
            or len(replicas) != COUNT or any(sum(row.get('case') == name for row in replicas) != 4 for name in SCENES)
            or any(row.get('right_censored') is not False for row in replicas)
            or set(measurement.get('native_episode_metrics', {}).get('groups', {})) != set(SCENES)
            or set(measurement.get('fixed_quiet_window_metrics', {}).get('groups', {})) != set(SCENES)
            or len(fingerprint.get('groups', [])) != COUNT
            or any(fingerprint['groups'].count(name) != 4 for name in SCENES)):
        raise ValueError('Single native child physical result or actual substep readback is incomplete')
    verify_single_measurement(measurement, protocol, config)
    if 'geometry_observer_sha256' in result:
        from chassis_geometry_observer import validate_report
        if result['geometry_observer_sha256'] != prep.digest(Path(__file__).with_name('chassis_geometry_observer.py')):
            raise ValueError('Native geometry observer source differs')
        validate_report(measurement.get('spin_geometry', {}), COUNT)
    candidate_file = Path(output) / f'candidate_{candidate_index:02d}.json'
    traces = prep.read(Path(output) / f'candidate_{candidate_index:02d}_trace.json')
    if prep.read(candidate_file) != measurement or not isinstance(traces, list) or not traces:
        raise ValueError('Single native child selected candidate result or trace is incomplete')
    previous = 0.
    for row in traces:
        if (not isinstance(row, dict) or type(row.get('time_s')) not in (int, float)
                or not previous < row['time_s'] <= ticks * .02 + 1e-8):
            raise ValueError('Single native child trace requires monotonic actual sample times')
        previous = row['time_s']
        for key in ('active', 'done', 'terminated', 'final_policy_mask'):
            if len(row.get(key, [])) != COUNT or any(type(value) is not bool for value in row[key]):
                raise ValueError('Single native child trace requires actual replica masks')
        for key in ('requested', 'applied', 'velocity', 'omega', 'gravity', 'position'):
            if (len(row.get(key, [])) != COUNT or any(not isinstance(values, list) or len(values) != 3
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                    for values in row[key])):
                raise ValueError('Single native child trace requires finite physical vectors')
        if len(row.get('height', [])) != COUNT or any(type(value) not in (int, float)
                or not math.isfinite(value) for value in row['height']):
            raise ValueError('Single native child trace requires actual physical heights')
    if not math.isclose(traces[-1]['time_s'], ticks * .02, rel_tol=0., abs_tol=1e-8):
        raise ValueError('Single native child trace does not reach its actual final policy tick')
    for index, replica in enumerate(replicas):
        if not any(math.isclose(row['time_s'], replica['terminal_s'], rel_tol=0., abs_tol=1e-8)
                   and row['active'][index] and row['done'][index] for row in traces):
            raise ValueError('Single native child trace omits an actual terminal frame')
    return result


def request(protocol, group, time_s):
    if protocol not in EXPLICIT_PROTOCOLS or group not in SCENES:
        raise ValueError('Unknown isolated seven-scene protocol')
    height = int(group[6:9]) / 1000. if group in prep.STANDS else .305
    if group != 'stand_305mm' or protocol == 'nominal':
        return [0., 0., height]
    release = 10. if protocol.startswith(('yaw', 'spin')) else 5.
    target = dict(vx_positive=[.5, 0.], vx_negative=[-.5, 0.],
                  yaw_positive=[0., 1.], yaw_negative=[0., -1.],
                  spin_positive=[0., math.tau], spin_negative=[0., -math.tau])[protocol]
    return [*(target if 1. <= time_s < release else [0., 0.]), height]


@contextmanager
def scoped_bootstrap(module, root, config, control, receipt):
    """Constructor-only DI in this isolated process; production files stay intact.

    A single exact independent baseline call receives a deep copy of the
    separately validated receipt. All other calls use the original validator. The native
    constructor's *original* validate_runtime_state remains untouched.
    """
    root = Path(root).resolve(strict=True)
    stage_name = environment_stage(config)
    profile = Path(config['design_preflight']['file']).resolve(strict=True)
    config_pin, control_pin = canonical_sha(config), canonical_sha(control)
    control_path = root / config['control_math_source']
    profile_pin = config['design_preflight']['sha256']
    from wheeled_tasks.chassis.preflight import validate_design_contract
    authenticated = validate_design_contract(config, prep.read(profile), control, stage='recovery',
        role='baseline', seed=SEED, num_envs=COUNT, block_index=0)
    if (module.__name__ != 'wheeled_tasks.chassis.env'
            or Path(module.__file__).resolve() != root / 'src/wheeled_tasks/chassis/env.py'
            or prep.digest(profile) != profile_pin
            or prep.digest(control_path) != config['control_math_sha256']
            or prep.read(profile).get('run_scope') != 'interactive_unscored'
            or prep.read(profile).get('qualification', 'missing') is not None
            or receipt.get('role') != 'baseline' or receipt.get('status') != 'static_passed'
            or any(receipt.get(key) != value for key, value in authenticated.items())
            or set(c['name'] for c in config['evaluation']['cases']) != set(SCENES)
            or config.get('auto_reset') is not False or config.get('evaluation_exact_cases') is not True):
        raise ValueError('Not the authenticated isolated constructor profile')
    original = module.design_preflight
    calls = []

    def independent(configuration, observed_control, observed_root, *, stage, role, seed,
                    num_envs, block_index=0):
        if (calls or canonical_sha(configuration) != config_pin
                or canonical_sha(observed_control) != control_pin
                or Path(observed_root).resolve() != root
                or (stage, role, seed, num_envs, block_index) != (stage_name, 'baseline', SEED, COUNT, 0)
                or prep.digest(profile) != profile_pin
                or prep.digest(control_path) != config['control_math_sha256']):
            return original(configuration, observed_control, observed_root, stage=stage,
                            role=role, seed=seed, num_envs=num_envs, block_index=block_index)
        calls.append(dict(stage=stage, role=role, seed=seed, num_envs=num_envs, block_index=block_index))
        return deepcopy(receipt)

    module.design_preflight = independent
    try:
        yield calls
        if len(calls) != 1:
            raise ValueError('Native constructor did not consume its single bootstrap receipt')
    finally:
        module.design_preflight = original


def apply_request(env, previous, requested):
    """Reuse the existing suite's vectorized command protocol and slew once."""
    from wheeled_tasks.chassis.playback_command import advance_playback_command
    proposed = previous.new_tensor(requested)
    if env.recovery is not None:
        held = env.recovery.enabled & ~env.recovery.fsm.motion_released
        proposed[held, :2] = 0.
        proposed[held, 2] = .305
    applied = advance_playback_command(previous, proposed, env.policy_dt,
                                       env.cfg['command_slew'], env.cfg.get('motion_limits'))
    env.command_acceleration.copy_((applied[:, 0] - previous[:, 0]) / env.policy_dt)
    previous.copy_(applied)
    env.commands.copy_(applied)
    env.command_target.copy_(applied[:, :2])
    env.raw_height.copy_(applied[:, 2])
    if env.recovery is not None:
        env.recovery.followup_command.copy_(applied)
    env.mode.copy_((applied[:, :2].abs().sum(-1) > .01).long())
    env.history.last_tick = None
    env.update_targets()
    env.update_command_reference()
    return env.get_observations(), applied.clone()


class NativeEvents:
    """Observer only: native ended ledgers, final mask and fixed quiet window."""
    def __init__(self, groups, protocol, dt=.02):
        import torch
        self.groups, self.protocol, self.dt = list(groups), protocol, dt
        self.native = prep.ReplayEvents(groups, dt)
        self.quiet_ticks = torch.zeros(len(groups), dtype=torch.long)
        self.quiet_started = torch.zeros(len(groups), dtype=torch.bool)
        self.quiet_first = torch.zeros_like(self.quiet_started)
        self.rows = [dict(case=name, first_zero_request_s=None, first_applied_zero_s=None,
                         first_blend_s=None, quiet_completed_s=None, terminal_reasons=[],
                         condition_pass_policy_frames={key: 0 for key in CONDITIONS},
                         holding_rl_condition_pass_policy_frames={key: 0 for key in CONDITIONS},
                         holding_rl_policy_frames=0, native_continuous_peak_s=0.,
                         sim_truth_hold_peak_s=0.) for name in groups]

    def observe(self, time_s, diagnostic, mask, owners, requested, applied):
        import torch
        active = self.native.alive.clone()
        self.quiet_first.zero_()
        quiet = torch.zeros_like(active)
        release = 10. if self.protocol.startswith(('yaw', 'spin')) else 5.
        for i, row in enumerate(self.rows):
            if not bool(active[i]):
                continue
            if owners[i]['phase'] == 'BLEND' and row['first_blend_s'] is None:
                row['first_blend_s'] = time_s + self.dt
            for key in CONDITIONS:
                passed = bool(diagnostic['recovery_stable_' + key + '_pass'][i])
                row['condition_pass_policy_frames'][key] += int(passed)
                row['holding_rl_condition_pass_policy_frames'][key] += int(passed and bool(diagnostic['recovery_holding_rl'][i]))
            row['holding_rl_policy_frames'] += int(bool(diagnostic['recovery_holding_rl'][i]))
            row['native_continuous_peak_s'] = max(row['native_continuous_peak_s'],
                                                float(diagnostic['recovery_stable_peak_seconds'][i]))
            row['sim_truth_hold_peak_s'] = max(row['sim_truth_hold_peak_s'], float(diagnostic['recovery_rl_stable_s'][i]))
            if self.protocol != 'nominal' and row['case'] == 'stand_305mm' and time_s >= release:
                zero_request = all(float(v) == 0. for v in requested[i][:2])
                zero_applied = all(float(v) == 0. for v in applied[i][:2])
                if zero_request and row['first_zero_request_s'] is None:
                    row['first_zero_request_s'] = time_s
                if zero_request and zero_applied:
                    if not bool(self.quiet_started[i]):
                        self.quiet_first[i] = True
                        self.quiet_started[i] = True
                        row['first_applied_zero_s'] = time_s
                    if int(self.quiet_ticks[i]) < round(10. / self.dt):
                        quiet[i] = True
                        self.quiet_ticks[i] += 1
                        if int(self.quiet_ticks[i]) == round(10. / self.dt):
                            row['quiet_completed_s'] = time_s + self.dt
                else:
                    # A violated command-defined window cannot silently restart
                    # after a different driven trajectory and receive credit.
                    if bool(self.quiet_started[i]):
                        raise ValueError('Actual planar command left the frozen quiet window')
            if bool(diagnostic['done'][i]):
                row['terminal_reasons'] = [key for key, value in diagnostic['reasons'].items() if bool(value[i])]
                if bool(diagnostic['success'][i]) and not bool(diagnostic['terminated'][i]):
                    row['terminal_reasons'].append('native_task_success_unscored')
        self.native.observe(time_s, diagnostic, mask, owners, requested, applied)
        return active, quiet

    def report(self):
        result = self.native.report()
        for i, native in enumerate(result['replicas']):
            native.update(self.rows[i], quiet_observed_seconds=int(self.quiet_ticks[i]) * self.dt,
                          full_ten_second_quiet_window=int(self.quiet_ticks[i]) == round(10. / self.dt))
        result['condition_counts_scope'] = 'Last physical tick snapshot at policy rate; native peak/dwell uses all physical ticks'
        return result


def to_cpu(value):
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    return value.detach().cpu() if hasattr(value, 'detach') else value


def observe_interval(events, metrics, quiet_metrics, time_s, diagnostic, mask, owners, requested, origin_xy):
    """Connect the original metrics to the true pre-step quiet-window origin."""
    active, quiet = events.observe(time_s, diagnostic, mask, owners, requested, diagnostic['commands'].tolist())
    metrics.observe(diagnostic, active)
    quiet_metrics.origin_xy[events.quiet_first] = origin_xy[events.quiet_first]
    quiet_metrics.observe(diagnostic, quiet)
    return active, quiet


def native_reset(env, seed):
    """Birth-only native reset with paired observation and transport RNG streams."""
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env.device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    if env.command_transport is not None:
        env.command_transport.generator.manual_seed(seed ^ 0x43414E)
        env.command_transport.tick = 0  # A fresh transport epoch at episode reset only.
    return env.reset_suite()


def reset_fingerprint(env, observation):
    result = dict(groups=list(env.scene_groups), clone_indices=list(env.clone_indices),
        actor_observation_sha256=hashlib.sha256(observation['policy'].detach().cpu().numpy().tobytes()).hexdigest(),
        fixed_birth_samples=env.fixed_reset_samples.detach().cpu().tolist(),
        root_pose=env.robot.data.root_link_pose_w.torch.detach().cpu().tolist(),
        motor_positions=env.robot.data.joint_pos.torch.detach().cpu().tolist(),
        masses=env.body_mass.detach().cpu().tolist(), friction=env.surface_mu.detach().cpu().tolist(),
        observation_noise_flag=env.cfg['observation_noise_enabled'],
        signal_perturbation_module=env.perturbations is not None, signal_delay_module=env.signal_delay is not None)
    if env.command_transport is not None:
        transport = env.command_transport
        result['command_transport'] = {key: getattr(transport, key).detach().cpu().tolist()
            for key in ('enabled', 'base_delay', 'loss_probability', 'can_phase')}
    return result


def verify_physics(env, root, config):
    # Native constructor has already executed its unchanged runtime validator.
    observed = env.startup_report.get('runtime_preflight', {})
    if observed.get('status') != 'runtime_configuration_passed':
        raise ValueError('Unchanged native validate_runtime_state did not pass')
    if (env.dt, env.policy_dt, env.decimation, env.v5.settings['leg_kp'], env.v5.settings['leg_kd'],
        env.robot.cfg.spawn.articulation_props.solver_position_iteration_count,
        env.robot.cfg.spawn.articulation_props.solver_velocity_iteration_count,
        config['closure_gap_termination_m']) != (.005, .02, 4, 160., 2.5, 64, 32, .003):
        raise ValueError('Actual physics, feedback PD, solver or closure guard differs')
    if env.num_envs != COUNT or len(env.scene_groups) != COUNT:
        raise ValueError('This replay is exactly seven scenes by four fixed replicas')
    if any(env.scene_groups.count(group) != 4 for group in SCENES):
        raise ValueError('Native clone allocation lost a fixed scene or replica')
    seconds = {case['name']: case['episode_seconds'] for case in config['evaluation']['cases']}
    if env.episode_limits.detach().cpu().tolist() != [round(seconds[group] / .02) for group in env.scene_groups]:
        raise ValueError('Native per-case timeout differs from the independent frozen projection')
    controller = prep.verify_common_controller(root, env.recovery.fsm)
    return dict(runtime_preflight=deepcopy(observed), controller=controller,
                control_timing=deepcopy(env.startup_report['control_timing']), qualification=None)


def native_tick_limit(env):
    """The native per-case timeout, including 22s yaw, owns the loop bound."""
    if env.episode_limits.numel() != env.num_envs or bool((env.episode_limits <= 0).any()):
        raise ValueError('Invalid native per-case episode limits')
    return int(env.episode_limits.max()) + 1


def run_protocol(args, root, configs, identities):
    """Actual lazy native bootstrap; only invoked by explicit --execute children."""
    import numpy as np
    from play_chassis_suite import validated_session
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    selected_index = getattr(args, 'candidate_index', None)
    if selected_index is None:
        config = prep.read(configs / 'interactive_unscored_contract.json')
        profile_receipt = prep.read(configs / 'projection_receipt.json')
    else:
        # Recheck the shared bytes immediately before the native bootstrap.
        config, profile_receipt, binding = verify_prepared_projection(root, configs, args.protocol,
            args.projection_receipt_sha256)
        if binding != args.prepared_binding:
            raise ValueError('Shared prepared projection changed before native bootstrap')
    control = load_chassis_control(root / config['control_math_source'])
    from train_chassis import preflight
    _, manifest = preflight(root / prep.REC)
    result = dict(run_scope='interactive_unscored', qualification=None, protocol=args.protocol,
        original_candidates=identities, optimizer_constructed=False, PPO_updates=0,
        production_design_preflight_used_for_projection=False, status='native_bootstrap_pending', candidates=[])
    if selected_index is not None:
        result.update(world_mode='independent_world', selected_candidate_index=selected_index,
            selected_identity=identities[selected_index], shared_abi=args.shared_abi,
            prepared_projection=binding)
    launcher = env = writer = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False))
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
        from chassis_suite_support import control_ownership
        import wheeled_tasks.chassis.env as native_module
        torch.set_num_threads(4)
        with scoped_bootstrap(native_module, root, config, control, profile_receipt['projection']) as calls:
            env = FixedCaseEnv(config, manifest, control, root, stage_name=environment_stage(config), num_envs=COUNT,
                              seed=SEED, level=1., device=args.device, run_role='baseline', block_index=0)
        result['bootstrap_call'] = calls
        result['native_readback'] = verify_physics(env, root, config)
        result['startup_report'] = deepcopy(env.startup_report)
        if selected_index is not None:
            result['native_constructor'] = dict(pid=os.getpid(), count=1, fresh_process=True,
                runtime_root=str(root), source_runtime_commit=prep.E5,
                environment_class='wheeled_tasks.chassis.eval_env.FixedCaseEnv')
        writer = SummaryWriter(str(args.output / 'events'))
        reference_birth = None
        for index, identity in enumerate(identities):
            if selected_index is not None and index != selected_index:
                continue
            session, _ = validated_session(Path(identity['onnx']), Path(identity['original_contract']),
                                           prep.read(identity['original_contract']))
            observations = native_reset(env, SEED)
            fingerprint = reset_fingerprint(env, observations)
            if reference_birth is None:
                reference_birth = fingerprint
            elif fingerprint != reference_birth:
                raise ValueError('Models did not receive identical native birth/observation/DR streams')
            previous = env.commands.clone()
            events = NativeEvents(env.scene_groups, args.protocol, env.policy_dt)
            metrics = EpisodeMetrics(env.scene_groups, 'cpu', env.policy_dt)
            quiet_metrics = EpisodeMetrics(env.scene_groups, 'cpu', env.policy_dt)
            geometry_observer = None
            if getattr(args, 'geometry_trace', False):
                from chassis_geometry_observer import SpinGeometryObserver
                geometry_observer = SpinGeometryObserver(env.scene_groups)
                result['geometry_observer_sha256'] = prep.digest(Path(__file__).with_name('chassis_geometry_observer.py'))
            traces = []
            start = time.monotonic()
            for tick in range(native_tick_limit(env)):
                if not launcher.app.is_running() or env.sim.is_stopped():
                    raise RuntimeError('Native timeline stopped before bounded protocol completion')
                time_s = tick * env.policy_dt
                requested = [request(args.protocol, group, time_s) for group in env.scene_groups]
                observations, applied = apply_request(env, previous, requested)
                window_origin = (env.robot.data.root_link_pose_w.torch[:, :2] - env.origins[:, :2]).detach().cpu().clone()
                if geometry_observer is not None:
                    wheel_origin = env.wheel_centers()[:, :, :2].mean(1).detach().cpu().clone()
                with torch.no_grad():
                    output = session.run(None, {'obs': observations['policy'].detach().cpu().numpy().astype(np.float32)})[0]
                    if output.shape != (COUNT, 6) or not np.isfinite(output).all():
                        raise ValueError('Invalid native mean ONNX actions')
                    observations, _, _, extras = env.step(torch.as_tensor(output, device=args.device))
                data = to_cpu(extras['diagnostics'])  # Native pre-reset diagnostic snapshot.
                mask = to_cpu(extras['policy_learning_mask'])
                owners = [control_ownership(env.recovery, i) for i in range(COUNT)]
                # A stopping displacement is measured from the actual zero-
                # command window origin; preceding movement never enters it.
                active, quiet = observe_interval(events, metrics, quiet_metrics, time_s, data, mask, owners,
                                                 requested, window_origin)
                if geometry_observer is not None:
                    geometry = geometry_observer.snapshot(env)
                    wheel_after = env.wheel_centers()[:, :, :2].mean(1).detach().cpu()
                    geometry_observer.observe(time_s, data, active, mask, requested, data['commands'],
                        geometry, window_origin, wheel_origin, wheel_after)
                if tick % 10 == 0 or bool((data['done'] & active).any()):
                    traces.append(dict(time_s=time_s + env.policy_dt, active=active.tolist(),
                        requested=requested, applied=data['commands'].tolist(), final_policy_mask=mask.tolist(),
                        owners=owners, velocity=data['velocity'].tolist(), omega=data['omega'].tolist(),
                        gravity=data['gravity'].tolist(), height=data['height'].tolist(), position=data['position'].tolist(),
                        done=data['done'].tolist(), terminated=data['terminated'].tolist(),
                        conditions={key: data['recovery_stable_' + key + '_pass'].tolist() for key in CONDITIONS},
                        native_continuous_peak_s=data['recovery_stable_peak_seconds'].tolist()))
                    if geometry_observer is not None:
                        traces[-1]['geometry'] = {key: value.tolist() for key, value in geometry.items()}
                if not bool(events.native.alive.any()):
                    break
            measurement = dict(identity=identity, reset_fingerprint=fingerprint, native_events=events.report(),
                native_episode_metrics=metrics.report(), fixed_quiet_window_metrics=quiet_metrics.report(),
                fixed_quiet_window_target_group=None if args.protocol == 'nominal' else 'stand_305mm',
                metric_scope='All first-episode frames (warmup=0); separate command-defined quiet window also warmup=0; no formal exam score',
                physics_wall_seconds=time.monotonic() - start, policy_ticks=tick + 1,
                control_substep_readback=deepcopy(env.control_audit.last), right_censored=int(events.native.alive.sum()))
            if geometry_observer is not None:
                measurement['spin_geometry'] = geometry_observer.report()
                for replica, row in enumerate(measurement['spin_geometry']['replicas']):
                    if not row['policy_frames']:
                        continue
                    for key in ('policy_frames', 'planar_speed_mean_m_s', 'requested_yaw_mae_rad_s',
                        'applied_yaw_mae_rad_s', 'leg_length_mae_m', 'leg_angle_mae_rad', 'root_displacement_max_m'):
                        writer.add_scalar(f'Unscored/{args.protocol}/geometry/replica_{replica}/{key}',
                            row[key], identity['consumed_updates'])
            if selected_index is not None:
                measurement['original_candidate_index'] = index
            prep.write(args.output / f'candidate_{index:02d}.json', measurement)
            prep.write(args.output / f'candidate_{index:02d}_trace.json', traces)
            for name, group in measurement['native_episode_metrics']['groups'].items():
                for key in ('planar_speed_mean_m_s', 'height_mae_m', 'yaw_mae_rad_s', 'tilt_max_deg', 'stand_drift_max_m'):
                    writer.add_scalar(f'Unscored/{args.protocol}/{name}/{key}', group[key], index)
            result['candidates'].append(measurement)
        result['status'] = 'native_mean_replay_completed_unscored'
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        prep.write(args.output / 'native_replay.json', result)
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            try:
                launcher.app.close()
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True, type=Path)
    parser.add_argument('--candidate', required=True, action='append', nargs=2, metavar=('PT', 'ONNX'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--interface-smoke', action='store_true', help='CPU-only same-contract ABI smoke, no native comparison')
    parser.add_argument('--execute', action='store_true', help='Explicitly start five serial native replay processes, 0 PPO')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--protocol', choices=EXPLICIT_PROTOCOLS, help=argparse.SUPPRESS)
    parser.add_argument('--geometry-trace', action='store_true', help='Observe native body geometry at policy rate, without control changes')
    parser.add_argument('--candidate-index', type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument('--prepared-projection', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--projection-receipt-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    validate_single_mode(args)
    if args.interface_smoke and (args.execute or args.protocol):
        raise ValueError('Interface smoke cannot execute a physical parent/child comparison')
    root = prep.frozen_runtime(args.runtime_root)
    if len(args.candidate) != 2 or args.output.exists():
        raise ValueError('Two authentic candidates and a new exclusive output directory are required')
    contracts, identities = zip(*(prep.authenticate_candidate(root, *pair) for pair in args.candidate))
    abi = prep.shared_abi(contracts)
    prep.validate_pair(identities, interface_smoke=args.interface_smoke)
    if args.candidate_index is not None:
        _, _, args.prepared_binding = verify_prepared_projection(root, args.prepared_projection,
            args.protocol, args.projection_receipt_sha256)
        args.prepared_projection = Path(args.prepared_binding['directory'])
        args.shared_abi = abi
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rec-speed-mean-native-replay-v1', run_scope='interactive_unscored', qualification=None,
        status='CPU_prepared_scene_not_run', optimizer_constructed=False, PPO_updates=0, scene_run=False,
        source_runtime_commit=prep.E5, driver_sha256=prep.digest(__file__),
        adapter_sha256=prep.digest(prep.__file__), candidates=identities, shared_abi=abi,
        endpoint_selection=prep.endpoint_selection(root, None if args.interface_smoke else identities[1]['consumed_updates']),
        scenes=list(SCENES), replicas_per_scene=4, num_envs=COUNT, protocols=list(PROTOCOLS),
        geometry_trace_enabled=args.geometry_trace,
        source_files_sha256={name: prep.digest(root / name) for name in (
            'src/wheeled_tasks/chassis/env.py', 'src/wheeled_tasks/chassis/eval_env.py',
            'src/wheeled_tasks/chassis/recovery.py', 'src/wheeled_tasks/chassis/recovery_training.py',
            'src/wheeled_tasks/chassis/preflight.py', 'src/wheeled_tasks/chassis/episode_metrics.py',
            'src/wheeled_tasks/chassis/playback_command.py', 'scripts/chassis_suite_support.py')},
        production_design_preflight_used_for_projection=False, internal_validation_role='baseline',
        baseline_score=None, coverage_unverified=['Actual native bootstrap/readback and physical behavior until execution',
            'These seven original fixed births do not add broad 20–80ms sensor/action-age or friction/payload stress profiles',
            '28 environments are an unscored functional replay, not a 4096-env performance measurement'])
    if args.candidate_index is None:
        for protocol in ((args.protocol,) if args.protocol else PROTOCOLS):
            directory = args.output / protocol
            receipt = prep.prepare_projection(root, directory, protocol=protocol)
            prep.write(directory / 'projection_receipt.json', receipt)
    else:
        report.update(world_mode='independent_world', selected_candidate_index=args.candidate_index,
            selected_identity=identities[args.candidate_index], prepared_projection=args.prepared_binding)
    prep.write(args.output / 'replay_preparation.json', report)
    if args.protocol:
        if not args.execute:
            raise ValueError('Native child protocol requires explicit execute')
        configs = args.output / args.protocol if args.candidate_index is None else args.prepared_projection
        native = run_protocol(args, root, configs, identities)
        if args.candidate_index is not None:
            try:
                verify_single_child_completion(args.output, args.protocol, args.candidate_index, identities,
                    root, configs, args.projection_receipt_sha256, abi)
            except (ValueError, OSError) as error:
                report.update(status='independent_native_replay_failed_no_retry', scene_run=True,
                    child_report_error=str(error))
                prep.write(args.output / 'replay_preparation.json', report)
                raise
        report.update(status=native['status'], scene_run=True,
                      native_result_file=str(args.output / 'native_replay.json'))
        prep.write(args.output / 'replay_preparation.json', report)
    elif args.execute:
        report['native_processes'] = []
        for protocol in PROTOCOLS:
            output = args.output / (protocol + '_native')
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--runtime-root', str(root),
                       '--output', str(output), '--execute', '--protocol', protocol, '--device', args.device]
            if args.geometry_trace:
                command.append('--geometry-trace')
            for checkpoint, policy in args.candidate:
                command.extend(['--candidate', str(checkpoint), str(policy)])
            wall = time.monotonic()
            with (args.output / (protocol + '.native.log')).open('x') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            report['native_processes'].append(dict(protocol=protocol, output=str(output),
                exit_code=completed.returncode, wall_seconds=time.monotonic() - wall))
            prep.write(args.output / 'replay_preparation.json', report)
            failure = None
            if completed.returncode == 0:
                try:
                    verify_child_completion(output, protocol)
                except (ValueError, OSError) as error:
                    failure = str(error)
            if completed.returncode or failure:
                report.update(status='serial_native_replay_failed_no_retry', failed_protocol=protocol, scene_run=True)
                if failure:
                    report['child_report_error'] = failure
                prep.write(args.output / 'replay_preparation.json', report)
                raise RuntimeError('Serial native replay failed; no retry: ' + protocol)
        report.update(status='five_serial_native_protocols_completed_unscored', scene_run=True)
        prep.write(args.output / 'replay_preparation.json', report)
    print(json.dumps(dict(status=report['status'], output=str(args.output), qualification=None)))
    return report


if __name__ == '__main__':
    main()
