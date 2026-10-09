#!/usr/bin/env python3
"""Isolated, unscored native mean replay; CPU preparation is the default.

Only --execute starts five *serial* native subprocesses. Each reuses the
unchanged e5 FixedCaseEnv for both independently authenticated ONNX models.
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


def request(protocol, group, time_s):
    if protocol not in PROTOCOLS or group not in SCENES:
        raise ValueError('Unknown isolated seven-scene protocol')
    height = int(group[6:9]) / 1000. if group in prep.STANDS else .305
    if group != 'stand_305mm' or protocol == 'nominal':
        return [0., 0., height]
    release = 10. if protocol.startswith('yaw') else 5.
    target = dict(vx_positive=[.5, 0.], vx_negative=[-.5, 0.],
                  yaw_positive=[0., 1.], yaw_negative=[0., -1.])[protocol]
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
        release = 10. if self.protocol.startswith('yaw') else 5.
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
    config = prep.read(configs / 'interactive_unscored_contract.json')
    profile_receipt = prep.read(configs / 'projection_receipt.json')
    control = load_chassis_control(root / config['control_math_source'])
    from train_chassis import preflight
    _, manifest = preflight(root / prep.REC)
    result = dict(run_scope='interactive_unscored', qualification=None, protocol=args.protocol,
        original_candidates=identities, optimizer_constructed=False, PPO_updates=0,
        production_design_preflight_used_for_projection=False, status='native_bootstrap_pending', candidates=[])
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
        writer = SummaryWriter(str(args.output / 'events'))
        reference_birth = None
        for index, identity in enumerate(identities):
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
            traces = []
            start = time.monotonic()
            for tick in range(native_tick_limit(env)):
                if not launcher.app.is_running() or env.sim.is_stopped():
                    raise RuntimeError('Native timeline stopped before bounded protocol completion')
                time_s = tick * env.policy_dt
                requested = [request(args.protocol, group, time_s) for group in env.scene_groups]
                observations, applied = apply_request(env, previous, requested)
                window_origin = (env.robot.data.root_link_pose_w.torch[:, :2] - env.origins[:, :2]).detach().cpu().clone()
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
                if tick % 10 == 0 or bool((data['done'] & active).any()):
                    traces.append(dict(time_s=time_s + env.policy_dt, active=active.tolist(),
                        requested=requested, applied=data['commands'].tolist(), final_policy_mask=mask.tolist(),
                        owners=owners, velocity=data['velocity'].tolist(), omega=data['omega'].tolist(),
                        gravity=data['gravity'].tolist(), height=data['height'].tolist(), position=data['position'].tolist(),
                        done=data['done'].tolist(), terminated=data['terminated'].tolist(),
                        conditions={key: data['recovery_stable_' + key + '_pass'].tolist() for key in CONDITIONS},
                        native_continuous_peak_s=data['recovery_stable_peak_seconds'].tolist()))
                if not bool(events.native.alive.any()):
                    break
            measurement = dict(identity=identity, reset_fingerprint=fingerprint, native_events=events.report(),
                native_episode_metrics=metrics.report(), fixed_quiet_window_metrics=quiet_metrics.report(),
                fixed_quiet_window_target_group=None if args.protocol == 'nominal' else 'stand_305mm',
                metric_scope='All first-episode frames (warmup=0); separate command-defined quiet window also warmup=0; no formal exam score',
                physics_wall_seconds=time.monotonic() - start, policy_ticks=tick + 1,
                control_substep_readback=deepcopy(env.control_audit.last), right_censored=int(events.native.alive.sum()))
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
    parser.add_argument('--protocol', choices=PROTOCOLS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.interface_smoke and (args.execute or args.protocol):
        raise ValueError('Interface smoke cannot execute a physical parent/child comparison')
    root = prep.frozen_runtime(args.runtime_root)
    if len(args.candidate) != 2 or args.output.exists():
        raise ValueError('Two authentic candidates and a new exclusive output directory are required')
    contracts, identities = zip(*(prep.authenticate_candidate(root, *pair) for pair in args.candidate))
    abi = prep.shared_abi(contracts)
    prep.validate_pair(identities, interface_smoke=args.interface_smoke)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rec-speed-mean-native-replay-v1', run_scope='interactive_unscored', qualification=None,
        status='CPU_prepared_scene_not_run', optimizer_constructed=False, PPO_updates=0, scene_run=False,
        source_runtime_commit=prep.E5, driver_sha256=prep.digest(__file__),
        adapter_sha256=prep.digest(prep.__file__), candidates=identities, shared_abi=abi,
        endpoint_selection=prep.endpoint_selection(root),
        scenes=list(SCENES), replicas_per_scene=4, num_envs=COUNT, protocols=list(PROTOCOLS),
        source_files_sha256={name: prep.digest(root / name) for name in (
            'src/wheeled_tasks/chassis/env.py', 'src/wheeled_tasks/chassis/eval_env.py',
            'src/wheeled_tasks/chassis/recovery.py', 'src/wheeled_tasks/chassis/recovery_training.py',
            'src/wheeled_tasks/chassis/preflight.py', 'src/wheeled_tasks/chassis/episode_metrics.py',
            'src/wheeled_tasks/chassis/playback_command.py', 'scripts/chassis_suite_support.py')},
        production_design_preflight_used_for_projection=False, internal_validation_role='baseline',
        baseline_score=None, coverage_unverified=['Actual native bootstrap/readback and physical behavior until execution',
            'These seven original fixed births do not add broad 20–80ms sensor/action-age or friction/payload stress profiles',
            '28 environments are an unscored functional replay, not a 4096-env performance measurement'])
    for protocol in ((args.protocol,) if args.protocol else PROTOCOLS):
        directory = args.output / protocol
        receipt = prep.prepare_projection(root, directory, protocol=protocol)
        prep.write(directory / 'projection_receipt.json', receipt)
    prep.write(args.output / 'replay_preparation.json', report)
    if args.protocol:
        if not args.execute:
            raise ValueError('Native child protocol requires explicit execute')
        native = run_protocol(args, root, args.output / args.protocol, identities)
        report.update(status=native['status'], scene_run=True,
                      native_result_file=str(args.output / 'native_replay.json'))
        prep.write(args.output / 'replay_preparation.json', report)
    elif args.execute:
        report['native_processes'] = []
        for protocol in PROTOCOLS:
            output = args.output / (protocol + '_native')
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--runtime-root', str(root),
                       '--output', str(output), '--execute', '--protocol', protocol, '--device', args.device]
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
