#!/usr/bin/env python3
"""CPU-only preparation of an explicitly unscored, same-scene mean replay.

No simulator, optimizer, campaign entry or physical run is available here.
The internal baseline role validates an independent projection; it awards no
baseline score. Original model contracts still undergo their full-role check.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

E5 = 'e5bd9c1367b875d7f229b9a1930647013d023372'
REC = 'contracts/v6_task_campaign_recovery_worker_v21.json'
SPEED = 'contracts/v6_task_campaign_speed_worker_v26.json'
FLAT = 'contracts/v6_task_campaign_flat_worker_v6.json'
REC_SHA = 'eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2'
SPEED_SHA = '941fb9e1f7a8ced3aa83fca76e82b7cf3064b594602d1e2900e56e0df0cc6b64'
PARENT_SHA = 'e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967'
STANDS = ('stand_230mm', 'stand_305mm', 'stand_430mm')
FALLEN = ('recovery_left_side_90_stand', 'recovery_right_side_90_stand',
          'recovery_prone_belly_legs_above_stand', 'recovery_prone_back_legs_above_stand')
BRAKES = {'brake_vx_positive': [.5, 0., .305], 'brake_vx_negative': [-.5, 0., .305],
          'brake_yaw_positive': [0., 1., .305], 'brake_yaw_negative': [0., -1., .305]}
ABI_KEYS = ('actor_dim', 'critic_dim', 'action_dim', 'actor_frame_dim', 'history_length',
            'actor_layout', 'manual_context35', 'actor_observation_source', 'policy_action_order',
            'policy_architecture', 'v5_control', 'physics_dt', 'policy_dt', 'control_math_sha256',
            'asset_manifest_sha256', 'command_reference', 'motion_limits', 'command_slew',
            'actor_reserved_frame30')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def frozen_runtime(root):
    root = Path(root).resolve(strict=True)
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != E5:
        raise ValueError('Replay runtime must be the actual frozen e5 source')
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', E5, '--',
                    'src', 'scripts', 'contracts', 'model'], check=True, capture_output=True)
    sys.path[:0] = [str(root / 'scripts'), str(root / 'src')]
    # Imports below use this immutable runtime, never a dirty working GUI tree.
    return root


def original_full(root, path):
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
    from wheeled_tasks.chassis.preflight import design_preflight
    source, manifest = preflight(path)
    binding = source['design_preflight']
    design_path = root / binding['file']
    if digest(design_path) != binding['sha256']:
        raise ValueError('Original full-role design SHA mismatch')
    design = read(design_path)
    row = design['stages'][binding['stage']]['roles']['full']
    if row['invocation']['seed']['rule'] != 'fixed':
        raise ValueError('Original full-role seed must be fixed')
    seed = row['invocation']['seed']['value']
    config = deepcopy(source)
    config['evaluation'] = select_evaluation_cases(source['evaluation'],
        [c['name'] for c in row['contract']['evaluation']['cases']])
    config['evaluation']['seed'] = seed
    config = fixed_suite_contract(config)
    control = load_chassis_control(root / source['control_math_source'])
    receipt = design_preflight(config, control, root, stage=source['enabled_stages'][0],
        role='full', seed=seed, num_envs=row['invocation']['num_envs'])
    return source, manifest, config, design, row, control, receipt


def shared_abi(contracts):
    reference = contracts[0]
    for contract in contracts[1:]:
        for key in ABI_KEYS:
            if key not in reference or key not in contract or reference[key] != contract[key]:
                raise ValueError('Shared replay ABI differs: ' + key)
    if (reference['physics_dt'], reference['policy_dt'], reference['actor_dim'],
        reference['critic_dim'], reference['action_dim'], reference['history_length']) != (.005, .02, 35, 81, 6, 1):
        raise ValueError('Shared replay requires 50 Hz policy / 200 Hz feedback PD / manual35')
    return {key: deepcopy(reference[key]) for key in ABI_KEYS}


def authenticate_candidate(root, checkpoint, policy):
    import numpy as np
    import torch
    import export_chassis_candidate as exporter
    from play_chassis_suite import validated_session
    checkpoint, policy = Path(checkpoint).resolve(strict=True), Path(policy).resolve(strict=True)
    seal_path = checkpoint.with_suffix('.checkpoint.json')
    if not seal_path.is_file() or read(seal_path).get('sealed') is not True:
        raise ValueError('Every candidate requires its immutable per-file checkpoint seal')
    exporter.ROOT = root
    authenticated = exporter.authenticate(checkpoint)
    _, saved, updates, contract_path, contract, _, agent, manifest_path, _ = authenticated
    if Path(str(policy) + '.contract.json').read_bytes() != contract_path.read_bytes():
        raise ValueError('ONNX sidecar must retain the original checkpoint contract bytes')
    session, metadata = validated_session(policy, contract_path, contract)
    if metadata.get('source_checkpoint_sha256') != digest(checkpoint):
        raise ValueError('ONNX provenance differs from the authenticated PT')
    clock_keys = ('contract_id', 'contract_sha256', 'asset_manifest_sha256', 'control_math_sha256',
        'stage', 'successful_updates_total', 'learning_lineage_updates', 'critic_warmup_remaining',
        'stage_actor_update', 'global_actor_update_offset', 'global_actor_update', 'actor_updates_consumed',
        'task_actor_update', 'confirmed_jump_bins_m', 'batch_transitions', 'training_transitions')
    clock_infos = {key: saved['infos'][key] for key in clock_keys if key in saved['infos']}
    if metadata.get('checkpoint_infos') != clock_infos:
        raise ValueError('ONNX checkpoint clock metadata differs from the sealed PT')
    actor = exporter.load_actor(saved, contract, agent)
    vectors, _ = exporter.abi_vectors(contract, read(manifest_path))
    with torch.no_grad():
        expected = actor.as_onnx(verbose=False)(vectors).cpu().numpy()
    observed = session.run(None, {'obs': vectors.numpy().astype(np.float32)})[0]
    if observed.shape != expected.shape or not np.isfinite(observed).all():
        raise ValueError('Invalid deterministic ONNX output')
    error = float(np.abs(observed - expected).max())
    if error > 1e-5:
        raise ValueError('ONNX does not reproduce the original native mean actor')
    _, _, _, _, _, _, receipt = original_full(root, contract_path)
    return contract, dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        seal_sha256=digest(seal_path), original_contract_sha256=digest(contract_path),
        original_contract=str(contract_path.resolve()),
        onnx=str(policy), onnx_sha256=digest(policy), checkpoint_infos=clock_infos, task_campaign=saved['infos'].get('task_campaign'),
        integrated_lineage=saved['infos'].get('integrated_lineage'),
        consumed_updates=updates, original_full_role=receipt, native_mean_max_abs_error=error)


def selected_cases(mapping, protocol=None):
    cases = {c['name']: c for c in mapping['evaluation']['cases']}
    selected = [deepcopy(cases[name]) for name in (*STANDS, *FALLEN)]
    if protocol is not None:
        if protocol not in ('nominal', 'vx_positive', 'vx_negative', 'yaw_positive', 'yaw_negative'):
            raise ValueError('Unknown seven-scene reset protocol')
        if protocol != 'nominal':
            case = selected[1]
            case.update(episode_seconds=22. if protocol.startswith('yaw') else 17., stationary=False,
                        command=[0., 0., .305], task='survive')
            case['skill'] = dict(kind='constant', command=[0., 0., .305], mode=0, sample_amplitude=False)
        return selected
    for name in BRAKES:
        case = deepcopy(cases['stand_305mm'])
        case.update(name=name, reset_seed_key=name, episode_seconds=17., stationary=False,
                    command=[0., 0., .305], task='survive')
        case['skill'] = dict(kind='constant', command=[0., 0., .305], mode=0, sample_amplitude=False)
        selected.append(case)
    return selected


def projection(mapping, cases, recovery_settings):
    result = deepcopy(mapping)
    names = [c['name'] for c in cases]
    evaluation = result['evaluation']
    evaluation.update(cases=deepcopy(cases), canonical_case_names=names,
        protocol_id='rec_speed_mean_interactive_unscored_v1', episodes_per_case=4)
    for key in ('promotion_case_names', 'protected_case_names', 'retention_case_names', 'quick_case_names'):
        evaluation[key] = []
    # This independent unscored profile is not a v20 campaign worker. The
    # authentic source full role above retains the old exact-case side=6 rule.
    result['task_campaign'].pop('revision_migration', None)
    result.update(scene_groups=[dict(name=c['name'], fraction=1 / len(cases), terrain=[c['terrain']]) for c in cases],
        skill_specs={c['name']: deepcopy(c['skill']) for c in cases},
        recovery_training=deepcopy(recovery_settings), record_diagnostics=True,
        auto_reset=False, monitor_applied_effort=True, evaluation_exact_cases=True,
        evaluation_long_corridors=True, playback_open_ground=False,
        flat_triangle_mesh=False, flat_half_length_m=44.)
    return result


def requested_command(name, time_s):
    """Literal operator release at 5s; the existing slew still determines applied command."""
    if name in BRAKES and 1. <= time_s < 5.:
        return list(BRAKES[name])
    if name in STANDS:
        return [0., 0., int(name[6:9]) / 1000.]
    if name not in BRAKES and name not in FALLEN:
        raise ValueError('Unknown fixed replay case')
    return [0., 0., .305]


def verify_common_controller(root, fsm):
    """Future driver must call this on its actual native instance before env.step."""
    from wheeled_tasks.chassis.recovery_script_revision import native_runtime_binding
    source = read(Path(root) / REC)
    if digest(Path(root) / REC) != REC_SHA:
        raise ValueError('Common controller source contract changed')
    observed = native_runtime_binding(fsm, source, root)
    for key in ('complete_started_blend', 'blend_seconds', 'stable_seconds', 'max_script_seconds'):
        if fsm.config[key] != source['recovery_training'][key]:
            raise ValueError('Common controller differs: ' + key)
    return dict(observed, complete_started_blend=fsm.config['complete_started_blend'])


def prepare_projection(root, output, protocol=None):
    from wheeled_tasks.chassis.preflight import validate_design_contract, validate_response_evidence
    from wheeled_tasks.chassis.response_migration import validate_recovery_configuration, validate_step_task_configuration
    from wheeled_tasks.chassis.recovery_prefix_curriculum import runtime_expectation as prefix_expectation
    source, _, full, design, row, control, original = original_full(root, root / REC)
    _, _, _, _, flat_row, _, flat_receipt = original_full(root, root / FLAT)
    binding = source['design_preflight']
    train_row = design['stages'][binding['stage']]['roles']['train']
    settings = deepcopy(train_row['contract']['recovery_training'])
    if settings != source['recovery_training'] or settings.get('side_speed_rad_s') != 1.5 or settings.get('complete_started_blend') is not True:
        raise ValueError('Common replay must use the real REC21 controller, not its old fixed-exam side speed')
    runtime_cases = selected_cases(source, protocol)
    independent = deepcopy(row['contract'])
    independent['evaluation']['cases'] += [deepcopy(c) for c in flat_row['contract']['evaluation']['cases']
        if c['name'] in ('stand_230mm', 'stand_430mm')]
    expected_cases = selected_cases(independent, protocol)
    if runtime_cases != expected_cases:
        raise ValueError('Fixed births differ from independently authenticated full rows')
    config = projection(full, runtime_cases, settings)
    expected = deepcopy(row)
    expected['contract'] = projection(independent, expected_cases, settings)
    expected['invocation']['num_envs'] = len(runtime_cases) * 4
    # No production worker migration is authenticated on a projected profile.
    # The separate original-source native class readback is mandatory in a
    # future driver and is tested below with a real CPU RecoveryFSM instance.
    expected['runtime'].pop('recovery_script_revision', None)
    expected['runtime']['recovery_complete_started_blend'] = True
    expected['runtime']['recovery_pose_curriculum'] = None
    expected['runtime']['modules']['recovery_prefix_curriculum'] = False
    expected['runtime']['recovery_prefix_curriculum'] = prefix_expectation(settings['prefix_curriculum'], enabled=False)
    profile = {key: deepcopy(value) for key, value in design.items() if key != 'stages'}
    profile.update(run_scope='interactive_unscored', qualification=None,
        reset_protocol=protocol,
        source_role='full', internal_validation_role='baseline',
        baseline_score=None, baseline_name_is_internal_only=True,
        source_contract_sha256=digest(root / REC), source_design_sha256=binding['sha256'],
        controller_source_role='train', controller_source_sha256=binding['sha256'],
        original_source_controller_readback_required=True,
        production_migration_removed_only_in_unscored_projection=True,
        allowed_changes=['evaluation', 'scene_groups', 'skill_specs', 'recovery_training',
            'record_diagnostics', 'auto_reset', 'monitor_applied_effort', 'evaluation_long_corridors',
            'playback_open_ground', 'flat_triangle_mesh', 'flat_half_length_m', 'invocation.num_envs',
            'runtime.recovery_script_revision', 'runtime.recovery_complete_started_blend',
            'runtime.recovery_pose_curriculum', 'runtime.recovery_prefix_curriculum',
            'runtime.modules.recovery_prefix_curriculum', 'task_campaign.revision_migration'],
        stages={binding['stage']: {'roles': {'baseline': expected}}})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    profile_path = output / 'interactive_unscored_expectations.json'
    write(profile_path, profile)
    config['design_preflight'] = dict(file=str(profile_path.resolve()), sha256=digest(profile_path), stage=binding['stage'])
    # This API validates the independent test design, not a production worker
    # or new campaign budget. Original full-role receipts above cover budgets.
    receipt = validate_design_contract(config, profile, control, stage=binding['stage'],
        role='baseline', seed=row['invocation']['seed']['value'], num_envs=len(runtime_cases) * 4)
    receipt['response_evidence'] = validate_response_evidence(config, root)
    receipt['recovery_configuration'] = validate_recovery_configuration(settings, root)
    receipt['step_task_configuration'] = validate_step_task_configuration(config['step_jump_task'], root)
    receipt['coverage'] = dict(independent_design=True, response_evidence=True, recovery_configuration=True,
        step_task_configuration=True, original_full_budgets=True, projected_production_budget=False,
        train_chassis_admission=False, native_runtime_readback=False)
    changes = {key: dict(before=full.get(key), after=config.get(key))
        for key in sorted(full.keys() | config.keys())
        if key != 'design_preflight' and full.get(key) != config.get(key)}
    write(output / 'projection_changes.json', changes)
    write(output / 'interactive_unscored_contract.json', config)
    return dict(source_full_role=original, independent_stand_full_role=flat_receipt, projection=receipt,
        profile_sha256=digest(profile_path), config_sha256=digest(output / 'interactive_unscored_contract.json'),
        projection_changes_sha256=digest(output / 'projection_changes.json'),
        case_names=[c['name'] for c in runtime_cases], control_protocol=settings,
        observation_noise_enabled=config['observation_noise_enabled'],
        randomization={k: config[k] for k in ('usb_transport', 'command_transport', 'signal_delay',
            'signal_perturbations', 'dynamics_randomization', 'contact_domain')})


class ReplayEvents:
    """Pre-reset diagnostics and whole-policy-interval ownership, not a new FSM.

    For a future native driver: observe the terminal frame, then exclude that
    replica forever (auto_reset=False). Native ledgers and metrics remain the
    source of physical completion and errors; this adds only exposure/timing.
    """
    def __init__(self, groups, policy_dt=.02):
        import torch
        from wheeled_tasks.chassis.evaluation import RequestProtocolMetrics
        self.groups, self.dt = list(groups), policy_dt
        self.alive = torch.ones(len(groups), dtype=torch.bool)
        self.protocol = RequestProtocolMetrics(groups, 'cpu', policy_dt)
        self.rows = [dict(case=name, policy_owned_seconds=0., first_pure_rl_s=None,
            first_motion_release_s=None, first_zero_request_s=None, first_applied_zero_s=None,
            quiet_seconds=0., terminal_s=None) for name in groups]

    def observe(self, time_s, diagnostics, final_policy_mask, owners, requested, applied):
        # time_s is policy start; owners/diagnostics are the pre-reset end snapshot.
        import torch
        active = self.alive.clone()
        self.protocol.observe(diagnostics, active)
        for index, row in enumerate(self.rows):
            if not bool(active[index]):
                continue
            owner = owners[index]
            if bool(final_policy_mask[index]):
                row['policy_owned_seconds'] += self.dt
            if owner['pure_rl'] and owner['ownership'] != 'failed_disarmed':
                if row['first_pure_rl_s'] is None:
                    row['first_pure_rl_s'] = time_s + self.dt
                if owner['motion_released'] and row['first_motion_release_s'] is None:
                    row['first_motion_release_s'] = time_s + self.dt
            if row['case'] in BRAKES and time_s >= 5.:
                zero_request = all(float(x) == 0. for x in requested[index][:2])
                zero_applied = all(abs(float(x)) <= 1e-6 for x in applied[index][:2])
                if zero_request and row['first_zero_request_s'] is None:
                    row['first_zero_request_s'] = time_s
                if zero_request and zero_applied:
                    if row['first_applied_zero_s'] is None:
                        row['first_applied_zero_s'] = time_s
                    row['quiet_seconds'] += self.dt
                else:
                    row['quiet_seconds'] = 0.
                row['quiet_seconds'] = min(10., row['quiet_seconds'])
            if bool(diagnostics['done'][index]):
                row['terminal_s'] = time_s + self.dt
        self.alive &= ~diagnostics['done'].cpu().bool()
        return active

    def report(self):
        from wheeled_tasks.chassis.evaluation import recovery_episode_diagnostics
        result = []
        for row, alive in zip(self.rows, self.alive.tolist()):
            item = deepcopy(row)
            item['right_censored'] = alive
            result.append(item)
        groups = {}
        for name, native in self.protocol.recovery_episodes.items():
            groups[name] = dict(native_recovery_ended_count=len(native),
                native_recovery_episodes=deepcopy(native),
                native_recovery_diagnostics=recovery_episode_diagnostics({'recovery_episodes': native}, len(native)))
        return dict(run_scope='interactive_unscored', qualification=None, replicas=result, groups=groups)


def endpoint_selection(root):
    """Select the first real periodic seal after 500 actor updates, not a new save."""
    speed = read(root / SPEED)
    if digest(root / SPEED) != SPEED_SHA or (speed['checkpoint_interval'],
            speed['task_campaign']['evaluation_interval']) != (500, 0):
        raise ValueError('Native periodic endpoint requires the exact frozen speed26 schedule')
    requested_actor = 500
    parent_total, parent_actor, parent_stage, parent_lineage, critic = 41393, 41223, 27556, 27646, 10
    requested_total = parent_total + critic + requested_actor
    native_total = ((requested_total + 499) // 500) * 500
    native_actor = native_total - parent_total - critic
    return dict(schema='first-native-periodic-after-500-actor-v1',
        source_runtime_commit=E5, contract_file=SPEED, contract_sha256=SPEED_SHA,
        checkpoint_interval=500, task_evaluation_interval=0,
        save_clock='train_chassis completed=parent_updates+successful_updates; completed%checkpoint_interval',
        save_source_file='scripts/train_chassis.py', save_source_sha256=digest(root / 'scripts/train_chassis.py'),
        unavailable_requested_endpoint=dict(task_actor_update=requested_actor,
            successful_updates_total=requested_total, native_periodic_seal=False,
            reason='41903 is not divisible by 500; evaluation_interval=0 supplies no actor500 assessment seal'),
        selected_endpoint=dict(task_actor_update=native_actor, successful_updates_total=native_total,
            actor_updates_consumed=parent_actor+native_actor, stage_actor_update=parent_stage+native_actor,
            learning_lineage_updates=parent_lineage+critic+native_actor, critic_warmup_remaining=0),
        parent_checkpoint_sha256=PARENT_SHA, critic_updates_paid=critic,
        physical_comparison_complete=False, new_checkpoint_request=False)


def validate_pair(identities, *, interface_smoke=False):
    if len(identities) != 2:
        raise ValueError('Exactly two model identities are required')
    parent, child = identities
    if interface_smoke:
        if any(item['original_contract_sha256'] != REC_SHA for item in identities):
            raise ValueError('Interface smoke permits only existing same REC21 contract exports')
        return
    if (parent['checkpoint_sha256'] != PARENT_SHA or parent['original_contract_sha256'] != REC_SHA
            or parent['checkpoint_infos']['task_actor_update'] != 14188):
        raise ValueError('Real comparison requires the exact REC41393 parent')
    expected = dict(task_actor_update=597, successful_updates_total=42000, stage_actor_update=28153,
        learning_lineage_updates=28253, actor_updates_consumed=41820, critic_warmup_remaining=0)
    if child['original_contract_sha256'] != SPEED_SHA or any(
            child['checkpoint_infos'].get(key) != value for key, value in expected.items()):
        raise ValueError('Real comparison requires genuine periodic speed597 clocks, including 10 critic updates')
    meta = child.get('integrated_lineage') or {}
    required = dict(lineage_id='source028-joint-serial-v1',
        plan_sha256='db693283b744400ed5b6b586055f1514b56aea9106580c40a3bc3ee882d36d92',
        origin_checkpoint_sha256=PARENT_SHA, domain='speed', joint_actor_updates=597, domain_actor_updates=597,
        domain_complete=False)
    if meta != required:
        raise ValueError('Speed597 does not authenticate the admitted REC41393 learning origin')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True, type=Path)
    parser.add_argument('--candidate', required=True, action='append', nargs=2, metavar=('CHECKPOINT', 'ONNX'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--interface-smoke', action='store_true', help='Only test two existing same-contract exports; no retention comparison')
    args = parser.parse_args(argv)
    root = frozen_runtime(args.runtime_root)
    if len(args.candidate) != 2:
        raise ValueError('Exactly two independently authenticated inputs are required')
    if args.output.exists():
        raise FileExistsError('Preparation output must be new')
    contracts, identities = zip(*(authenticate_candidate(root, *pair) for pair in args.candidate))
    abi = shared_abi(contracts)
    validate_pair(identities, interface_smoke=args.interface_smoke)
    projection_receipt = prepare_projection(root, args.output)
    files = ('scripts/export_chassis_candidate.py', 'scripts/play_chassis_suite.py',
        'scripts/chassis_suite_support.py', 'src/wheeled_tasks/chassis/preflight.py',
        'src/wheeled_tasks/chassis/recovery.py', 'src/wheeled_tasks/chassis/evaluation.py',
        'src/wheeled_tasks/chassis/eval_env.py')
    result = dict(schema='rec-speed-mean-replay-preparation-v1', status='CPU_inputs_and_unscored_design_validated_scene_not_run',
        run_scope='interactive_unscored', qualification=None, baseline_score=None,
        internal_validation_role='baseline', interface_smoke=args.interface_smoke,
        physical_driver_implemented=False, scene_run=False, PPO_updates=0,
        frozen_runtime_commit=E5, frozen_runtime_root=str(root),
        runtime_files_sha256={name: digest(root / name) for name in files},
        adapter_sha256=digest(__file__), candidates=identities, shared_abi=abi,
        endpoint_selection=endpoint_selection(root),
        projection=projection_receipt, command_protocol=dict(stand_seconds=10., fallen_seconds=20.,
            brake=dict(zero_request_seconds=5., quiet_seconds=10., end_seconds=17.,
                       quiet_window='10 continuous seconds after actual slew-limited command reaches zero'),
            brake_targets=BRAKES, policy='native deterministic mean, no exploration samples'),
        pending=['Authentic REC41393 export and sealed native speed597 PT/ONNX inputs must be supplied by the caller',
                 'Standalone native replay driver: confirm independent projection bootstrap (not production design_preflight), actual controller readback, and existing EpisodeMetrics/ReplayEvents',
                 'Physical same-condition mean-policy comparison; no current skill conclusion'])
    write(args.output / 'preparation.json', result)
    print(json.dumps(dict(status=result['status'], output=str(args.output), qualification=None)))
    return result


if __name__ == '__main__':
    main()
