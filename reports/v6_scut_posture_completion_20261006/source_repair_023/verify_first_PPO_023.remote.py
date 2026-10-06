#!/usr/bin/env python3
"""Emit a read-only v21 recovery first-PPO inspector; never connect or launch.

Adapted from verify_first_PPO_018.py (migration) and verify_first_PPO_020.py
(source, latest full seal and no refund). The emitted program reads local CPU
checkpoints, native receipts, ledger files and /proc on its executing host.
Missing evidence is pending; neither failed nor pending results permit restart.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import tarfile


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b):
    import torch
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def finite(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def full_learning_state_check(saved):
    """Require the complete native 9 actor, 8 critic and 17 Adam parameter state."""
    from collections import Counter
    import torch
    actor=saved.get('actor_state_dict', {}); critic=saved.get('critic_state_dict', {})
    optimizer=saved.get('optimizer_state_dict', {})
    require(isinstance(actor,dict) and len(actor)==9, 'Expected all 9 actor/std tensors')
    require(isinstance(critic,dict) and len(critic)==8, 'Expected all 8 critic tensors')
    tensors=list(actor.values())+list(critic.values())
    require(all(torch.is_tensor(x) and x.is_floating_point() and x.numel()>0 and bool(torch.isfinite(x).all()) for x in tensors),
            'Invalid/nonfinite actor or critic tensor')
    stdkeys=[key for key in actor if key == 'distribution.std_param']
    require(len(stdkeys)==1 and actor[stdkeys[0]].shape==(6,) and bool((actor[stdkeys[0]]>0).all()), 'Missing/invalid 6-axis actor std')
    groups=optimizer.get('param_groups', []); states=optimizer.get('state', {})
    require(isinstance(groups,list) and groups and isinstance(states,dict), 'Missing Adam group/state')
    params=[value for group in groups for value in group.get('params', [])]
    require(len(params)==17 and all(type(x) is int and x>=0 for x in params) and len(set(params))==17 and set(states)==set(params),
            'Adam does not cover exactly the unique 17 native parameters')
    require(finite(optimizer), 'Nonfinite Adam state/group')
    moments=[]
    for param in params:
        state=states[param]
        require(isinstance(state,dict) and {'step','exp_avg','exp_avg_sq'} <= set(state), 'Incomplete Adam moments/step')
        mean,square,step=state['exp_avg'],state['exp_avg_sq'],state['step']
        require(torch.is_tensor(mean) and torch.is_tensor(square) and mean.is_floating_point() and
                mean.shape==square.shape and mean.dtype==square.dtype and mean.numel()>0 and
                bool(torch.isfinite(mean).all()) and bool(torch.isfinite(square).all()) and bool((square>=0).all()),
                'Invalid Adam moment tensors')
        require((torch.is_tensor(step) and step.numel()==1 and bool(torch.isfinite(step).all()) and step.item()>=0) or
                (type(step) in (int,float) and math.isfinite(step) and step>=0), 'Invalid Adam step')
        moments.append(mean)
    require(Counter((tuple(x.shape),str(x.dtype)) for x in moments)==Counter((tuple(x.shape),str(x.dtype)) for x in tensors),
            'Adam moment shapes/dtypes do not cover actor and critic parameters')
    return {'actor_tensors':9,'critic_tensors':8,'std_axes':6,'Adam_parameter_states':17,'all_learning_tensors_finite':True}


def entropy_zero(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value == 0.0,
            'Native entropy readback is not numeric zero: ' + label)
    return float(value)


def native_entropy_readbacks(config, agent, resume, progress):
    values = {'contract': config.get('entropy_coef'),
              'agent_config': agent.get('algorithm', {}).get('entropy_coef'),
              'resume_effective': resume.get('effective_entropy_coef'),
              'progress': progress.get('entropy_coef')}
    for label, value in values.items():
        entropy_zero(value, label)
    require(resume.get('entropy_coef_source') == progress.get('entropy_coef_source') == 'contract',
            'Native entropy did not bind contract')
    return {**values, 'source': 'contract', 'parent_entropy_coef': resume.get('parent_entropy_coef'),
            'startup_scope': 'startup.json precedes runner creation; actual agent_config, resume receipt and sealed initial state provide entropy evidence'}


def native_warmup(source):
    """Execute only the existing pure function; do not import the native runner."""
    path = Path(source) / 'scripts/train_chassis.py'
    function = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'resume_critic_warmup')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['resume_critic_warmup']


def phase_checks(parent, initial, updates, *, warmup, consumption_origin, actor_consumption_origin):
    """Check every charged critic update, then the first actual actor update.

    'updates' must contain immutable snapshots 1..warmup+1. This supports a
    same-v21 restart with its saved remaining value, including remaining zero.
    """
    import torch
    require(type(warmup) is int and 0 <= warmup <= 10, 'Invalid remaining critic warmup')
    task = parent['infos']['task_actor_update']
    stage = parent['infos']['stage_actor_update']
    lineage = parent['infos']['learning_lineage_updates']
    for name in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
        require(equal(parent[name], initial[name]), 'Initial full state differs: ' + name)
    require(len(initial['actor_state_dict']) == 9, 'Expected all nine actor/std tensors')

    def clocks(saved, index, actors):
        infos = saved['infos']
        expected = dict(successful_updates_total=consumption_origin + index,
                        actor_updates_consumed=actor_consumption_origin + actors,
                        task_actor_update=task + actors, stage_actor_update=stage + actors,
                        learning_lineage_updates=lineage + index,
                        critic_warmup_remaining=max(0, warmup - index))
        require(all(infos.get(k) == v for k, v in expected.items()), 'Clock/remaining mismatch at update ' + str(index))
        return expected

    clocks(initial, 0, 0)
    require(set(updates) == set(range(1, warmup + 2)), 'Missing or extra required prefix snapshots')
    previous = initial
    rows = []
    for index in range(1, warmup + 1):
        saved = updates[index]
        require(equal(initial['actor_state_dict'], saved['actor_state_dict']), 'Actor/std moved in critic-only update ' + str(index))
        require(not equal(previous['critic_state_dict'], saved['critic_state_dict']), 'Critic did not change at update ' + str(index))
        require(not equal(previous['optimizer_state_dict'], saved['optimizer_state_dict']), 'Adam did not change at update ' + str(index))
        rows.append(dict(index=index, phase='critic_only', clocks=clocks(saved, index, 0)))
        previous = saved
    first = updates[warmup + 1]
    require(initial['actor_state_dict'].keys() == first['actor_state_dict'].keys(), 'Actor tensor catalog changed')
    changed = [k for k, value in initial['actor_state_dict'].items()
               if not torch.equal(value, first['actor_state_dict'][k])]
    require(len(changed) == 9, 'First actor update did not change all nine tensors including std')
    require(not equal(previous['critic_state_dict'], first['critic_state_dict']), 'First actor update did not change critic')
    require(not equal(previous['optimizer_state_dict'], first['optimizer_state_dict']), 'First actor update did not change Adam')
    rows.append(dict(index=warmup + 1, phase='actor_and_critic', clocks=clocks(first, warmup + 1, 1), actor_changed=changed))
    return rows


def bounded_phase_checks(parent, initial, first_critic, first_actor, *, warmup,
                         consumption_origin, actor_consumption_origin):
    """Native checkpoint500 evidence: initial, first critic and first actor only."""
    import torch
    require(type(warmup) is int and 0 <= warmup <= 10, 'Invalid remaining critic warmup')
    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
        require(equal(parent[key], initial[key]), 'Initial full state differs: ' + key)
    require(len(initial['actor_state_dict']) == 9, 'Expected nine actor/std tensors')
    def clocks(saved, index, actors):
        expected = dict(successful_updates_total=consumption_origin + index,
                        actor_updates_consumed=actor_consumption_origin + actors,
                        task_actor_update=parent['infos']['task_actor_update'] + actors,
                        stage_actor_update=parent['infos']['stage_actor_update'] + actors,
                        learning_lineage_updates=parent['infos']['learning_lineage_updates'] + index,
                        critic_warmup_remaining=max(0, warmup - index))
        require(all(saved['infos'].get(k) == v for k, v in expected.items()), 'Clock/remaining mismatch at update ' + str(index))
    clocks(initial, 0, 0)
    if warmup:
        require(first_critic is not None, 'First critic snapshot absent')
        clocks(first_critic, 1, 0)
        require(equal(initial['actor_state_dict'], first_critic['actor_state_dict']), 'Actor/std moved at first critic update')
        require(not equal(initial['critic_state_dict'], first_critic['critic_state_dict']) and
                not equal(initial['optimizer_state_dict'], first_critic['optimizer_state_dict']), 'First critic/Adam update absent')
    clocks(first_actor, warmup + 1, 1)
    require(initial['actor_state_dict'].keys() == first_actor['actor_state_dict'].keys(), 'Actor catalog differs')
    changed = [k for k, value in initial['actor_state_dict'].items() if not torch.equal(value, first_actor['actor_state_dict'][k])]
    require(len(changed) == 9, 'First actor update did not change all nine actor/std tensors')
    baseline = first_critic if warmup else initial
    require(not equal(baseline['critic_state_dict'], first_actor['critic_state_dict']) and
            not equal(baseline['optimizer_state_dict'], first_actor['optimizer_state_dict']), 'First actor prefix critic/Adam did not change')
    return dict(critic_tensor_observed_indices=[1] if warmup else [], first_actor_index=warmup + 1,
                actor_changed=changed, intermediate_actor_tensor_observation='unobserved_not_sealed_by_native_checkpoint500',
                critic_Adam_between_observed_seals='changed; intermediate changes are not individually tensor-observed')


def progress_phase_checks(rows, parent, *, warmup, consumption_origin, actor_consumption_origin, global_actor_update_offset=0):
    required = set(range(1, warmup + 2))
    if not required <= set(rows):
        raise FileNotFoundError('Direct progress capture prefix incomplete: ' + str(sorted(required - set(rows))))
    for index in sorted(required):
        value = rows[index]; actors = int(index > warmup)
        expected = dict(successful_updates=index, parent_updates=consumption_origin,
                        actor_updates_in_block=actors,
                        global_budget_actor_updates=global_actor_update_offset + actor_consumption_origin + actors,
                        task_actor_update=parent['infos']['task_actor_update'] + actors,
                        stage_actor_update=parent['infos']['stage_actor_update'] + actors,
                        optimizer_phase='actor_and_critic' if actors else 'critic_warmup')
        require(all(value.get(k) == v for k, v in expected.items()), 'Direct progress phase/clock differs at update ' + str(index))
        entropy_zero(value.get('entropy_coef'), 'progress update ' + str(index))
        require(value.get('entropy_coef_source') == 'contract', 'Direct progress native entropy source differs at update ' + str(index))
    return dict(observed_completed_updates=sorted(required), source='independent_readonly_native_progress_capture_not_TensorBoard',
                actor_consumption_source='actual_native_global_budget_actor_updates_minus_authenticated_argv_global_offset',
                actor_tensor_scope='only initial/first_critic/first_actor sealed tensors observed')


def progress_capture(index, *, source, directory, parent, warmup, consumption_origin, actor_consumption_origin, global_actor_update_offset=0):
    """Authenticate copied native bytes; partial observer coverage stays pending."""
    rows = {}
    require(index['schema'] == 'source023-native-first-progress-copies-v1' and
            Path(index['source_root']) == Path(source) and index['qualification'] is None and
            index['PPO_actions'] == 0 and index['read_only'] is True,
            'Independent progress capture identity differs')
    if index.get('worker') is None:
        raise FileNotFoundError('Progress observer has not found the native worker')
    require(Path(index['worker']) == Path(directory), 'Independent progress capture worker differs')
    for captured in index['rows']:
        path = Path(captured['path'])
        require(path.is_absolute() and sha(path) == captured['sha256'], 'Independent progress capture bytes changed')
        stamp = datetime.fromisoformat(captured['captured_at_utc'])
        require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, 'Progress capture has no UTC timestamp')
        value = read(path); completed = value['successful_updates']
        require(type(completed) is int and captured['successful_updates'] == completed and
                captured['source_progress'] == str(Path(directory) / 'progress.json'),
                'Independent progress capture source/update differs')
        require(completed not in rows, 'Duplicate independent progress capture')
        rows[completed] = value
    proof = progress_phase_checks(rows, parent, warmup=warmup, consumption_origin=consumption_origin,
                                  actor_consumption_origin=actor_consumption_origin,
                                  global_actor_update_offset=global_actor_update_offset)
    proof['observer_status'] = index.get('status')
    return proof


def crc_event_prefix(directory, *, origin, warmup, task_origin, ordinary_origin, prefix_origin, max_bytes):
    """Read a bounded event prefix; no explicit phase tag exists in source023."""
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
    files = sorted(Path(directory).glob('events.out.tfevents.*'), key=lambda p: p.stat().st_mtime_ns)
    if not files:
        raise FileNotFoundError('No native event file')
    event = files[0]  # Native first prefix, not a moving multi-GB tail.
    with event.open('rb') as stream:
        data = stream.read(max_bytes)
    pattern = re.compile(rb'(?:Timing/iteration_wall_seconds|(?:Episode/)?/(?:ordinary_posture|recovery/lesson)/relative_actor_updates)')
    series, seen = {}, set()
    for match in pattern.finditer(data):
        for pos in range(max(0, match.start() - 128), max(0, match.start() - 11)):
            length, crc = struct.unpack_from('<QI', data, pos); end = pos + length + 16
            if not 0 < length < 1024 ** 2 or end > len(data) or end - 4 < match.end(): continue
            if masked_crc32c(data[pos:pos + 8]) != crc: continue
            payload = data[pos + 12:end - 4]
            if masked_crc32c(payload) != struct.unpack_from('<I', data, end - 4)[0]: continue
            if pos in seen: break
            seen.add(pos); item = Event.FromString(payload)
            for value in item.summary.value:
                if pattern.fullmatch(value.tag.encode()) and value.HasField('simple_value'):
                    number = float(value.simple_value)
                    require(math.isfinite(number), 'Nonfinite CRC scalar')
                    old = series.setdefault(value.tag, {}).get(int(item.step))
                    require(old is None or old == number, 'Conflicting duplicate CRC scalar')
                    series[value.tag][int(item.step)] = number
            break
    timing = series.get('Timing/iteration_wall_seconds', {})
    steps = set(range(origin, origin + warmup + 1))
    if not steps <= set(timing):
        raise FileNotFoundError('CRC timing prefix not fully published within bounded prefix read')
    require(all(timing[step] > 0 for step in steps), 'Nonpositive native optimizer iteration wall time')
    clocks = [(tag, values) for tag, values in series.items() if tag.endswith('/relative_actor_updates')]
    critic_steps = set(range(origin, origin + warmup))
    actual = {}
    for family, curriculum_origin in (('ordinary_posture', ordinary_origin), ('recovery/lesson', prefix_origin)):
        matches = [(tag, values) for tag, values in clocks if '/' + family + '/' in tag]
        if not matches or not any(critic_steps <= set(values) for _, values in matches):
            raise FileNotFoundError('CRC actor clock prefix incomplete: ' + family)
        for tag, values in matches:
            if critic_steps <= set(values):
                require(all(math.isclose(values[s], max(0, task_origin - curriculum_origin), abs_tol=1e-5) for s in critic_steps),
                        'Actor curriculum clock advanced in critic prefix: ' + family)
                actual[tag] = {str(s): values[s] for s in sorted(critic_steps)}
                break
    return dict(file=str(event), bytes_read=len(data), max_bytes=max_bytes, CRC_valid_records=len(seen),
                optimizer_iteration_steps=sorted(steps), critic_actor_clock_values=actual,
                phase_source='inferred_from_authenticated_original_warmup_control_path_and_actor_clock; no_TensorBoard_optimizer_phase_tag',
                first_actor_tensor_update_source='separate_immutable_checkpoint_not_event_derived')


def preserve_account(plan, predecessor, account, state, calculate):
    before, old_state = predecessor['consumption'], predecessor['campaign_state']
    require(calculate(plan, account['attempts']) == {k: account[k] for k in calculate(plan, account['attempts'])}, 'Account arithmetic differs')
    require(account['rollback_refunds_consumed_updates'] is False, 'Refund enabled')
    require(account['consumed_updates'] >= before['consumed_updates'] and
            account['actor_updates_consumed'] >= before['actor_updates_consumed'], 'Consumption rewound')
    require(all(account['attempts'].get(k) == v for k, v in before['attempts'].items()), 'Old charged attempt changed')
    others = tuple(r['name'] for r in plan['domains'] if r['name'] != 'recovery')
    require(all(account['domain_consumed_updates'][n] == before['domain_consumed_updates'][n] for n in others), 'Other domain charged')
    require(all(state['domains'].get(n) == old_state['domains'].get(n) for n in others), 'Other domain record changed')
    for key in ('completion_history', 'reopening'):
        require(state['domains']['recovery'].get(key) == old_state['domains']['recovery'].get(key), 'Recovery history/reopening changed: ' + key)
    require(state['domains']['recovery'].get('domain_qualified') is None, 'Unassessed recovery became qualified')


def process_entry(arguments):
    for value in arguments[1:]:
        if value in (b'-c', b'-m'):
            return None
        if value.endswith(b'.py'):
            return Path(value.decode())
    return None


def inspect(settings):
    import torch
    torch.set_num_threads(1)
    root = Path(settings['run']); train = root / 'train'
    meta, predecessor, revision = settings['source'], settings['predecessor'], settings['revision']
    source = Path(meta['source_remote'])
    bound = predecessor['actual_parents']['recovery']; parent_path = Path(bound['path'])
    before = predecessor['consumption']; origin = before['consumed_updates']
    mode = settings.get('mode', 'v20_to_v21')
    result = dict(schema='source023-first-ppo-readonly-verification-v1',
                  verified_at_utc=datetime.now(timezone.utc).isoformat(), source_commit=meta['source_commit'],
                  scope='full_state_restore_critic_consumption_and_first_actor_update_not_behavior_quality',
                  qualification=None, checks={}, evidence={'engineering_limitations': settings.get('engineering_limitations', {})})
    checks, evidence = result['checks'], result['evidence']

    def section(name, function):
        try:
            value = function()
            checks[name] = dict(status='passed', detail=value)
        except FileNotFoundError as error:
            checks[name] = dict(status='pending', detail=str(error))
        except Exception as error:
            checks[name] = dict(status='failed', detail=f'{type(error).__name__}: {error}')

    def failed():
        return any(x['status'] != 'passed' for x in checks.values())

    def finish():
        statuses = [x['status'] for x in checks.values()]
        result['status'] = 'failed' if 'failed' in statuses else 'pending' if 'pending' in statuses else 'first_actual_actor_PPO_verified'
        result['check_counts'] = {n: statuses.count(n) for n in ('passed', 'pending', 'failed')}
        return result

    def identity():
        require(mode == 'v20_to_v21', 'Unknown transition mode')
        require(revision['source_commit'] == predecessor['source_commit'] == meta['source_commit'], 'Commit differs')
        require(read(root / settings.get('revision_file', 'source_revision_023.json')) == revision and
                read(root / settings.get('predecessor_file', 'predecessor_023.json')) == predecessor, 'Prepared admission changed')
        require(Path(revision['new_source_root']) == source and revision['new_contract_sha256'] == meta['plan_sha256'], 'Prepared source/plan differs')
        archive = Path(revision['source_archive'])
        require(sha(archive) == meta['archive_sha256'] == revision['source_archive_sha256'], 'Archive SHA differs')
        names = set()
        with tarfile.open(archive, 'r:gz') as tar:
            require(tar.pax_headers.get('comment') == meta['source_commit'], 'Archive commit differs')
            for member in tar:
                rel = Path(member.name)
                require(not rel.is_absolute() and '..' not in rel.parts and (member.isfile() or member.isdir()), 'Unsafe archive member')
                if member.isdir():
                    continue
                require(member.name not in names, 'Duplicate archive member')
                names.add(member.name)
                require(hashlib.sha256(tar.extractfile(member).read()).hexdigest() == sha(source / member.name), 'Archived source differs: ' + member.name)
        old_source = Path(revision['old_source_root'])
        old_files = {str(p.relative_to(old_source)): sha(p) for p in (old_source / 'contracts').rglob('*') if p.is_file()}
        new_files = {str(p.relative_to(source)): sha(p) for p in (source / 'contracts').rglob('*') if p.is_file()}
        require(all(new_files.get(k) == v for k, v in old_files.items()), 'Old frozen contract bytes changed')
        if mode == 'v20_to_v21':
            require(set(new_files) - set(old_files) == {'contracts/v6_task_campaign_v21.json', 'contracts/v6_task_campaign_migration_v21.json',
                    'contracts/v6_task_campaign_recovery_worker_v21.json', 'contracts/v6_task_campaign_recovery_expectations_v21.json'}, 'Unexpected new contract files')
        require(sha(source / 'contracts/v6_task_campaign_v21.json') == meta['plan_sha256'], 'Target plan SHA differs')
        launch = read(root / settings.get('launch_file', 'recovery_entropy_launch_023.json')); job = read(root / 'job.json')
        require(launch['source_commit'] == meta['source_commit'] and launch['evidence_sha256'] == predecessor['evidence_sha256'], 'Launch evidence differs')
        require(job['command'] == revision['command'] and read(root / job['source_revision_receipt']) == revision and
                sha(root / job['previous_job']) == revision['previous_job_sha256'], 'Actual job/previous job identity differs')
        evidence.update(job=job, launch=launch, frozen_archive_file_count=len(names))
        return dict(verified_source_files=len(names), all_old_contracts_preserved=True)

    section('frozen_source_predecessor_and_actual_job_identity', identity)
    if failed():
        return finish()
    sys.path[:0] = [str(source / 'src'), str(source / 'scripts')]
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count
    from wheeled_tasks.chassis.task_campaign import account_updates, validate_task_resume, validate_plan_migration
    from wheeled_tasks.chassis.recovery_entropy_revision import revised_plan, revised_manifest, validate_frozen_workers
    from wheeled_tasks.chassis.preflight import validate_runtime_state
    plan = read(source / 'contracts/v6_task_campaign_v21.json')
    old_plan = read(source / ('contracts/v6_task_campaign_v20.json' if mode == 'v20_to_v21' else 'contracts/v6_task_campaign_v21.json'))
    worker = next(r for r in plan['domains'] if r['name'] == 'recovery'); config = read(source / worker['contract'])
    state = read(train / 'campaign_state.json'); account = read(train / 'consumption.json')

    def canonical_migration():
        require(plan == revised_plan(source) and read(source / 'contracts/v6_task_campaign_migration_v21.json') == revised_manifest(source), 'Canonical v21 recipe differs')
        require(validate_frozen_workers(source)['domain'] == 'recovery', 'Revision is not recovery only')
        require(state['contract_sha256'] == account['contract_sha256'] == meta['plan_sha256'], 'Campaign account not v21')
        shared_account = Path(plan['budget_source_stage']) / (plan['campaign_id'] + '.account.json')
        require(read(shared_account) == account, 'Shared authoritative claim and campaign ledger differ; take a fresh bounded read')
        if mode == 'v20_to_v21':
            migration = validate_plan_migration(old_plan, plan, train)
            actual = read(train / 'revisions/v20_to_v21_account_migration.json')
            require(all(actual.get(k) == v for k, v in migration.items()), 'Native account migration receipt differs')
        old_rows = {r['name']: r for r in old_plan['domains']}
        require(all(r == old_rows[r['name']] for r in plan['domains'] if r['name'] != 'recovery'), 'One of other five workers changed')
        require(all(r['parent_domain'] == old_rows[r['name']]['parent_domain'] for r in plan['domains']), 'Parent tree changed')
        require(plan['total_ceiling'] == old_plan['total_ceiling'] == 121000, 'Campaign ceiling changed')
        require(config.get('entropy_coef') == 0.0, 'Canonical recovery entropy is not zero')
        return dict(other_workers=5, total_ceiling=121000, reopening=None)

    section('recovery_only_canonical_migration_five_workers_and_parent_tree', canonical_migration)
    section('complete_old_account_and_domain_history_no_refund', lambda: preserve_account(plan, predecessor, account, state, account_updates))

    def load_checkpoint(path, expected_contract):
        seal = read(path.with_suffix('.checkpoint.json')); contract = checkpoint_contract_path(path)
        require(seal['sealed'] is True and sha(path) == seal['checkpoint_sha256'] and
                sha(contract) == seal['contract_sha256'] == expected_contract, 'Immutable PT/contract seal differs: ' + str(path))
        require(checkpoint_update_count(path) == seal['learning_updates'], 'Native checkpoint clock validation failed')
        saved = torch.load(path, map_location='cpu', weights_only=False); infos = saved['infos']
        evidence.setdefault('full_learning_state_integrity', {})[str(path)] = full_learning_state_check(saved)
        for key in ('task_actor_update', 'stage_actor_update', 'actor_updates_consumed', 'learning_lineage_updates'):
            require(infos[key] == seal[key], 'Checkpoint infos/clock differs: ' + key)
        require(infos['contract_sha256'] == expected_contract and infos['successful_updates_total'] == seal['learning_updates'] and
                infos['task_campaign']['domain'] == 'recovery', 'Checkpoint provenance differs')
        require(all(finite(saved[k]) for k in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict')), 'Nonfinite learning state')
        require(saved['optimizer_state_dict']['state'], 'Empty Adam learning state')
        if expected_contract == worker['sha256']:
            entropy_zero(infos.get('entropy_coef'), 'checkpoint infos ' + str(path))
            entropy_zero(seal.get('entropy_coef'), 'checkpoint seal ' + str(path))
        for key in ('rng_state', 'env_rng_state'):
            require(torch.is_tensor(infos.get(key)) and infos[key].dtype == torch.uint8 and infos[key].numel() > 0, 'Missing RNG state: ' + key)
        require(isinstance(infos.get('cuda_rng_states'), list) and infos['cuda_rng_states'] and
                all(torch.is_tensor(x) and x.dtype == torch.uint8 and x.numel() > 0 for x in infos['cuda_rng_states']), 'Missing CUDA RNG state')
        evidence.setdefault('immutable_checkpoints', {})[str(path)] = dict(sha256=sha(path), seal=seal)
        return saved

    parent = None; warmup = None

    def captured_parents():
        nonlocal parent, warmup
        require(set(predecessor['actual_parents']) == {'flat', 'recovery', 'speed', 'terrain', 'jump'}, 'Incomplete five own-domain parent inventory')
        for name, captured in predecessor['actual_parents'].items():
            cp = Path(captured['path']); seal = read(cp.with_suffix('.checkpoint.json')); contract = checkpoint_contract_path(cp)
            require(cp.resolve().is_relative_to((train / name).resolve()) and sha(cp) == captured['sha256'] and
                    seal == captured['seal'] and sha(cp.with_suffix('.checkpoint.json')) == captured['seal_sha256'] and
                    sha(contract) == captured['contract_sha256'], 'Captured own-domain parent changed: ' + name)
            require(checkpoint_update_count(cp) == seal['learning_updates'], 'Captured parent clock invalid: ' + name)
        require(parent_path.name == 'model_final.pt' and parent_path.parent == train / 'recovery/attempt_008/train', 'Not actual stopped attempt008 full predecessor seal')
        require(bound['seal']['learning_updates'] == origin and bound['seal']['actor_updates_consumed'] == before['actor_updates_consumed'], 'Stopped parent is not latest charged state')
        # Only attempts captured in the pre-stop ledger can compete with this
        # parent. New first-PPO snapshots must not invalidate latest-parent proof.
        previous_attempts = {str(Path(k).parent) for k in before['attempts'] if k.startswith('recovery/')}
        candidates = []
        for p in (train / 'recovery').glob('attempt_*/train/**/*.checkpoint.json'):
            if any(x.endswith('.pending') for x in p.parts):
                continue
            attempt = str(p.parents[0].relative_to(train)).split('/train')[0]
            if attempt not in previous_attempts:
                continue
            seal = read(p); cp = p.with_suffix('').with_suffix('.pt')
            candidates.append((seal['learning_updates'], cp.name == 'model_final.pt', cp))
        require(candidates and max(candidates, key=lambda row: row[:2])[2] == parent_path, 'Captured predecessor is not latest sealed own-domain learning state')
        parent = load_checkpoint(parent_path, bound['contract_sha256'])
        require(all(parent['infos'][k] == v for k, v in bound['authenticated_infos'].items()), 'Prepared authenticated parent infos differ')
        validate_task_resume(read(checkpoint_contract_path(parent_path)), config)
        warmup = native_warmup(source)(config, parent['infos'], contract_sha256=worker['sha256'], task_change=True)
        require(warmup == predecessor['expected_critic_warmup'] and
                (warmup == 10 if mode == 'v20_to_v21' else warmup == parent['infos']['critic_warmup_remaining']), 'Warmup does not use original migration/remaining logic')
        evidence['clocks'] = dict(consumption_origin=origin, actor_consumption_origin=before['actor_updates_consumed'],
                restored_checkpoint_learning_updates=bound['seal']['learning_updates'], restored_task_actor_origin=bound['seal']['task_actor_update'],
                expected_critic_only_updates=warmup, first_actor_consumption_update=origin + warmup + 1)
        evidence['RNG_restore_scope'] = 'Full RNG presence plus frozen restore path; reset/resampling consumes RNG before initial_state, so no post-reset bitwise RNG identity claim.'
        return dict(latest_full_parent=str(parent_path), warmup_remaining=warmup)

    section('latest_immutable_full_parent_and_original_warmup_remaining_logic', captured_parents)
    if parent is None or warmup is None:
        return finish()
    invocations = []
    for path in (train / 'recovery').glob('attempt_*/invocation.json'):
        row = read(path); command = row.get('command', [])
        if process_entry([str(x).encode() for x in command]) == source / 'scripts/train_chassis.py' and '--contract' in command and command[command.index('--contract') + 1] == str(source / worker['contract']):
            key = str((path.parent / 'train').relative_to(train))
            if key not in before['attempts']:
                invocations.append((path, row))
    if not invocations:
        checks['unique_new_recovery_invocation'] = dict(status='pending', detail='No new frozen v21 recovery invocation')
        return finish()
    if len(invocations) != 1:
        checks['unique_new_recovery_invocation'] = dict(status='failed', detail=len(invocations))
        return finish()
    path, invocation = invocations[0]; directory = path.parent / 'train'; command = invocation['command']

    def argument(flag):
        return command[command.index(flag) + 1]

    def invocation_check():
        endpoint = plan['domain_completion']['schedule']['end_actor_updates']['recovery']
        task = bound['seal']['task_actor_update']
        expected = min(worker['updates'] - before['domain_consumed_updates']['recovery'],
                       max(0, endpoint - task) + (warmup if task < endpoint else 0))
        require(Path(argument('--run-dir')) == directory and invocation['source_checkpoint'] == str(parent_path) and
                invocation['source_checkpoint_sha256'] == bound['sha256'] and Path(argument('--resume')) == parent_path, 'Invocation parent differs')
        require(int(argument('--consumed-updates')) == origin and int(argument('--task-actor-update')) == task and
                int(argument('--stage-actor-update')) == bound['seal']['stage_actor_update'] and
                int(argument('--actor-updates-consumed')) == before['actor_updates_consumed'], 'Invocation clock differs')
        require(invocation['requested_updates'] == expected and invocation['scheduled_end_actor_update'] == endpoint, 'Requested updates do not include only remaining adaptation and actor budget')
        evidence.update(actual_worker_directory=str(directory), invocation=invocation)
        return dict(requested_updates=expected, warmup=warmup, endpoint=endpoint)

    section('actual_new_attempt_full_parent_and_separate_consumption_budget', invocation_check)

    def runtime():
        resume = read(directory / 'resume_verification.json'); startup = read(directory / 'startup.json'); progress = read(directory / 'progress.json')
        pinned = startup['source_sha256']
        required = ('scripts/train_chassis.py', 'src/wheeled_tasks/chassis/env.py', 'src/wheeled_tasks/chassis/rewards.py',
                    'src/wheeled_tasks/chassis/recovery.py', 'src/wheeled_tasks/chassis/recovery_prefix_curriculum.py',
                    'src/wheeled_tasks/chassis/recovery_script_revision.py', 'src/wheeled_tasks/chassis/preflight.py', 'src/wheeled_tasks/chassis/recovery_entropy_revision.py')
        require(all(n in pinned for n in required), 'Missing required new runtime source snapshot')
        for name, digest in pinned.items():
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe source snapshot path')
            require(sha(source / name) == sha(directory / 'source' / name) == digest, 'Actual PPO source snapshot differs: ' + name)
        for name, digest in settings.get('runtime_pins', {}).items():
            require(pinned.get(name) == digest, 'Prepared candidate runtime SHA differs: ' + name)
        native = startup['startup']; preflight = native['runtime_preflight']; observed = preflight['observed']
        entropy_evidence = native_entropy_readbacks(config, read(directory / 'agent_config.json'), resume, progress)
        from wheeled_tasks.chassis.recovery_entropy_revision import revision_binding
        require(config['task_campaign']['entropy_revision_migration'] == revision_binding() and progress['task_campaign']['entropy_revision_migration'] == revision_binding(), 'Native entropy revision binding differs')
        evidence['actual_native_entropy'] = entropy_evidence
        require(preflight['status'] == 'runtime_configuration_passed', 'Native runtime preflight did not pass')
        validate_runtime_state(native['design_preflight']['runtime_expected'], observed)
        require(all(resume.get(k) is True for k in ('actor_exact', 'critic_exact', 'optimizer_exact')) and
                resume['parent_checkpoint_sha256'] == bound['sha256'] and resume['parent_updates'] == origin and
                resume['restored_checkpoint_updates'] == bound['seal']['learning_updates'] and resume['task_clock'] == 'restored' and
                resume['source_contract_sha256'] == bound['contract_sha256'] and resume['target_contract_sha256'] == worker['sha256'] and
                resume['effective_learning_rate'] == resume['restored_optimizer_learning_rate'] == 1e-4, 'Actual full restore receipt differs')
        require(progress['num_envs'] == 4096 and (config['actor_dim'], config['critic_dim'], config['action_dim']) == (35, 81, 6) and
                (observed['physics_dt'], observed['policy_dt'], observed['decimation'], observed['pc_control_dt']) == (.005, .02, 4, .005), 'Control/population dimensions differ')
        require(all(observed['modules'][k] is True for k in ('command_transport', 'signal_delay', 'recovery_training',
                'recovery_prefix_curriculum', 'recovery_hold_cost', 'ordinary_posture_curriculum')), 'Native module missing/disabled')
        require(observed['seed'] == int(argument('--seed')) == plan['seed'] and int(argument('--block-index')) == 0,
                'Native seed/block differs')
        prefix = config['recovery_training']['prefix_curriculum']
        require(observed['recovery_prefix_curriculum']['settings'] == prefix and prefix['goal_bridge'] == 'native-original-followup-after-stable-gate-v1', 'Native original goal bridge not bound')
        require(observed['recovery_hold_cost'] == config['recovery_hold_cost'], 'Actual hold cost settings differ')
        expected_script_revision = dict(
            source_recovery_sha256='04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6',
            side_speed_rad_s=1.5,
            capture_prepare_gyro_gate='native-current-tick-norm-below-existing-handover-gyro-max',
            handover_gyro_max_rad_s=.75)
        require(sha(source / 'src/wheeled_tasks/chassis/recovery.py') == expected_script_revision['source_recovery_sha256'] and
                config['recovery_training'].get('side_speed_rad_s') == expected_script_revision['side_speed_rad_s'] and
                observed.get('recovery_script_revision') == native.get('recovery_script_revision') ==
                native['design_preflight']['runtime_expected'].get('recovery_script_revision') == expected_script_revision,
                'Real native CAPTURE/SIDE source/config startup binding absent or differs')
        evidence['actual_recovery_script_revision'] = expected_script_revision
        from wheeled_tasks.chassis.recovery_quality_revision import HOLD_GATE_BINDING
        require(observed['recovery_hold_cost_gate_binding'] == HOLD_GATE_BINDING and
                native['recovery_hold_cost_gate_binding'] == HOLD_GATE_BINDING, 'Native FSM/geometry binding absent or differs')
        require(sum(native['scene_group_counts'].values()) == 4096 and min(native['scene_group_counts'].values()) > 0, 'Native population invalid')
        completed, actors = progress['successful_updates'], progress['actor_updates_in_block']
        require(completed - actors == min(warmup, completed) and progress['parent_updates'] == origin and
                progress['task_actor_update'] == bound['seal']['task_actor_update'] + actors and
                progress['stage_actor_update'] == bound['seal']['stage_actor_update'] + actors, 'Progress critic/actor accounting differs')
        old_config = read(source / 'contracts/v6_task_campaign_recovery_worker_v20.json')
        require(prefix['origin_actor_update'] == old_config['recovery_training']['prefix_curriculum']['origin_actor_update'] and
                observed['ordinary_posture_curriculum']['origin_actor_update'] == old_config['ordinary_posture_curriculum']['origin_actor_update'], 'Curriculum origin reset')
        admission = read(source / config['actuator_response_evidence']['file'])
        require(admission['adopted_joint_names'] == ['L_joint3', 'R_joint3'] and admission['leg_response'].startswith('candidate_not_adopted') and
                observed['actuator_response'] == config['actuator_response'], 'Wheel/leg response scope changed')
        evidence.update(resume_verification=resume, progress=progress, runtime_preflight=preflight)
        return dict(source_snapshot_files=len(pinned), actual_goal_bridge_bound=True, native_hold_cost_gate_binding=HOLD_GATE_BINDING)

    section('actual_native_full_restore_new_modules_origins_and_clocks', runtime)

    def learning():
        initial = load_checkpoint(directory / 'initial_state/model_final.pt', worker['sha256'])
        first_critic = load_checkpoint(directory / f'checkpoints/update_{origin + 1:08d}/model.pt', worker['sha256']) if warmup else None
        first_actor = load_checkpoint(directory / f'checkpoints/update_{origin + warmup + 1:08d}/model.pt', worker['sha256'])
        rows = bounded_phase_checks(parent, initial, first_critic, first_actor, warmup=warmup, consumption_origin=origin,
                                    actor_consumption_origin=before['actor_updates_consumed'])
        evidence['actual_optimizer_prefix'] = rows
        return dict(critic_only_tensor_seals_verified=int(warmup > 0), first_actor_consumption_update=origin + warmup + 1,
                    all_nine_actor_tensors_changed=True, critic_and_adam_changed=True)

    section('initial_first_critic_and_first_nine_tensor_actor_immutable_seals', learning)
    def native_events():
        proof = crc_event_prefix(directory, origin=origin, warmup=warmup, task_origin=bound['seal']['task_actor_update'],
                                 ordinary_origin=config['ordinary_posture_curriculum']['origin_actor_update'],
                                 prefix_origin=config['recovery_training']['prefix_curriculum']['origin_actor_update'],
                                 max_bytes=settings.get('event_prefix_bytes', 64 * 1024 ** 2))
        evidence['native_CRC_update_and_actor_clock_prefix'] = proof
        return proof
    section('native_CRC_optimizer_iterations_and_held_actor_curriculum_clock', native_events)
    if settings.get('phase_progress_capture'):
        def phase_capture():
            proof = progress_capture(read(settings['phase_progress_capture']), source=source, directory=directory,
                                     parent=parent, warmup=warmup, consumption_origin=origin,
                                     actor_consumption_origin=before['actor_updates_consumed'],
                                     global_actor_update_offset=int(argument('--global-actor-update-offset')))
            evidence['direct_native_progress_phases'] = proof
            return proof
        section('direct_native_progress_phase_and_consumption_prefix', phase_capture)

    def ownership():
        kinds = {'chassis_remote_job.py': 'wrapper', 'run_task_campaign.py': 'supervisor', 'train_chassis.py': 'worker', 'evaluate_chassis.py': 'evaluator'}
        owners = []
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = p.read_bytes().split(b'\0'); entry = process_entry(args)
                if not entry or entry.name not in kinds or not any(x == str(root).encode() or x == str(train).encode() or x.startswith(str(train).encode() + b'/') for x in args):
                    continue
                ppid = int(next(line.split()[1] for line in (p.parent / 'status').read_text().splitlines() if line.startswith('PPid:')))
                owners.append(dict(pid=int(p.parent.name), ppid=ppid, kind=kinds[entry.name], entry_script=str(entry), command=[x.decode() for x in args if x]))
            except (OSError, PermissionError, StopIteration):
                continue
        if not owners:
            raise FileNotFoundError('No live campaign owner; checkpoint evidence does not prove current ownership')
        require(sorted(r['kind'] for r in owners) == ['supervisor', 'worker', 'wrapper'], 'Duplicate/evaluation owner found')
        by = {r['kind']: r for r in owners}
        require(by['worker']['ppid'] == by['supervisor']['pid'] and by['supervisor']['ppid'] == by['wrapper']['pid'] and
                by['supervisor']['command'] == revision['command'] and by['worker']['command'] == command and
                by['supervisor']['pid'] == evidence['job']['pid'], 'Owner chain/argv differs')
        require(all(r['entry_script'] == str(source / 'scripts' / {'wrapper': 'chassis_remote_job.py', 'supervisor': 'run_task_campaign.py', 'worker': 'train_chassis.py'}[r['kind']]) for r in owners), 'Owner source differs')
        evidence['campaign_python_entry_owners'] = owners
        return owners

    section('unique_native_owner_chain_no_parallel_evaluator', ownership)
    return finish()


def settings_for(args):
    source, predecessor, revision = read(args.source), read(args.predecessor), read(args.revision)
    admission = read(args.admission)
    require(admission['status'] == 'passed' and admission['qualification'] is None, 'Admission not passed/unassessed')
    require(source['source_commit'] == predecessor['source_commit'] == revision['source_commit'] == admission['source_commit'], 'Dynamic commit binding differs')
    require(source['archive_sha256'] == admission['archive_sha256'], 'Archive admission differs')
    require(predecessor['evidence_sha256'] == admission['evidence_sha256'], 'Prepared evidence differs')
    require(source['previous_source_commit'] == predecessor['previous_source_commit'] == '723732adae37f12f8488f9537717664415a98d10', 'Previous deployed source differs')
    require(source['contract_file'] == 'contracts/v6_task_campaign_v21.json' and revision['schema'] == 'task-plan-revision-v21', 'Target revision identity differs')
    require(predecessor['schema'] == 'task-recovery-entropy-predecessor-023-v1' and predecessor['expected_entropy_coef'] == 0.0 and predecessor['expected_critic_warmup'] == 10, 'Prepared native entropy/warmup differs')
    return dict(run=str(Path(predecessor['actual_parents']['recovery']['path']).parents[4]),
                source=source, predecessor=predecessor, revision=revision, mode=args.mode,
                engineering_limitations=admission.get('engineering_limitations', {}),
                phase_progress_capture=str(args.phase_progress_capture) if args.phase_progress_capture else None,
                event_prefix_bytes=args.event_prefix_mib * 1024 ** 2,
                runtime_pins={n: row['candidate'] for n, row in source.get('changed_runtime_files', {}).items()
                              if n in ('scripts/train_chassis.py', 'src/wheeled_tasks/chassis/env.py',
                                       'src/wheeled_tasks/chassis/rewards.py', 'src/wheeled_tasks/chassis/recovery.py',
                                       'src/wheeled_tasks/chassis/recovery_prefix_curriculum.py',
                                       'src/wheeled_tasks/chassis/recovery_script_revision.py',
                                       'src/wheeled_tasks/chassis/preflight.py', 'src/wheeled_tasks/chassis/recovery_entropy_revision.py')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--emit-remote-code', action='store_true', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--admission', type=Path, required=True)
    p.add_argument('--predecessor', type=Path, required=True)
    p.add_argument('--revision', type=Path, required=True)
    p.add_argument('--mode', choices=('v20_to_v21',), default='v20_to_v21')
    p.add_argument('--phase-progress-capture', type=Path)
    p.add_argument('--event-prefix-mib', type=int, choices=range(1, 129), default=64)
    args = p.parse_args(); settings = settings_for(args)
    # Emission prints a read-only inspector, not a shell or network command.
    text = Path(__file__).read_text().split('\nif __name__ ==')[0]
    print(text + '\nprint(json.dumps(inspect(json.loads(' + repr(json.dumps(settings)) + ')), allow_nan=False))\n')


if __name__ == "__main__":
    main()
