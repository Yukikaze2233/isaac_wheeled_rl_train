#!/usr/bin/env python3
"""Emit a read-only v19 recovery first-PPO inspector; never connect or launch.

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
    same-v19 restart with its saved remaining value, including remaining zero.
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
    return dict(observed_completed_updates=sorted(required), source='independent_readonly_native_progress_capture_not_TensorBoard',
                actor_consumption_source='actual_native_global_budget_actor_updates_minus_authenticated_argv_global_offset',
                actor_tensor_scope='only initial/first_critic/first_actor sealed tensors observed')


def progress_capture(index, *, source, directory, parent, warmup, consumption_origin, actor_consumption_origin, global_actor_update_offset=0):
    """Authenticate copied native bytes; partial observer coverage stays pending."""
    rows = {}
    require(index['schema'] == 'source021-native-first-progress-copies-v1' and
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
    """Read a bounded event prefix; no explicit phase tag exists in source021."""
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
    mode = settings.get('mode', 'v18_to_v19')
    result = dict(schema='source021-first-ppo-readonly-verification-v1',
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
        require(mode in ('v18_to_v19', 'same_v19'), 'Unknown transition mode')
        require(revision['source_commit'] == predecessor['source_commit'] == meta['source_commit'], 'Commit differs')
        require(read(root / settings.get('revision_file', 'source_revision_021.json')) == revision and
                read(root / settings.get('predecessor_file', 'predecessor_021.json')) == predecessor, 'Prepared admission changed')
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
        if mode == 'v18_to_v19':
            require(set(new_files) - set(old_files) == {'contracts/v6_task_campaign_v19.json', 'contracts/v6_task_campaign_migration_v19.json',
                    'contracts/v6_task_campaign_recovery_worker_v19.json', 'contracts/v6_task_campaign_recovery_expectations_v19.json'}, 'Unexpected new contract files')
        require(sha(source / 'contracts/v6_task_campaign_v19.json') == meta['plan_sha256'], 'Target plan SHA differs')
        launch = read(root / settings.get('launch_file', 'recovery_quality_launch_021.json')); job = read(root / 'job.json')
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
    from wheeled_tasks.chassis.recovery_quality_revision import revised_plan, revised_manifest, validate_frozen_workers
    from wheeled_tasks.chassis.preflight import validate_runtime_state
    plan = read(source / 'contracts/v6_task_campaign_v19.json')
    old_plan = read(source / ('contracts/v6_task_campaign_v18.json' if mode == 'v18_to_v19' else 'contracts/v6_task_campaign_v19.json'))
    worker = next(r for r in plan['domains'] if r['name'] == 'recovery'); config = read(source / worker['contract'])
    state = read(train / 'campaign_state.json'); account = read(train / 'consumption.json')

    def canonical_migration():
        require(plan == revised_plan(source) and read(source / 'contracts/v6_task_campaign_migration_v19.json') == revised_manifest(source), 'Canonical v19 recipe differs')
        require(validate_frozen_workers(source)['domain'] == 'recovery', 'Revision is not recovery only')
        require(state['contract_sha256'] == account['contract_sha256'] == meta['plan_sha256'], 'Campaign account not v19')
        shared_account = Path(plan['budget_source_stage']) / (plan['campaign_id'] + '.account.json')
        require(read(shared_account) == account, 'Shared authoritative claim and campaign ledger differ; take a fresh bounded read')
        if mode == 'v18_to_v19':
            migration = validate_plan_migration(old_plan, plan, train)
            actual = read(train / 'revisions/v18_to_v19_account_migration.json')
            require(all(actual.get(k) == v for k, v in migration.items()), 'Native account migration receipt differs')
        old_rows = {r['name']: r for r in old_plan['domains']}
        require(all(r == old_rows[r['name']] for r in plan['domains'] if r['name'] != 'recovery'), 'One of other five workers changed')
        require(all(r['parent_domain'] == old_rows[r['name']]['parent_domain'] for r in plan['domains']), 'Parent tree changed')
        require(plan['total_ceiling'] == old_plan['total_ceiling'] == 121000, 'Campaign ceiling changed')
        return dict(other_workers=5, total_ceiling=121000, reopening=None)

    section('recovery_only_canonical_migration_five_workers_and_parent_tree', canonical_migration)
    section('complete_old_account_and_domain_history_no_refund', lambda: preserve_account(plan, predecessor, account, state, account_updates))

    def load_checkpoint(path, expected_contract):
        seal = read(path.with_suffix('.checkpoint.json')); contract = checkpoint_contract_path(path)
        require(seal['sealed'] is True and sha(path) == seal['checkpoint_sha256'] and
                sha(contract) == seal['contract_sha256'] == expected_contract, 'Immutable PT/contract seal differs: ' + str(path))
        require(checkpoint_update_count(path) == seal['learning_updates'], 'Native checkpoint clock validation failed')
        saved = torch.load(path, map_location='cpu', weights_only=False); infos = saved['infos']
        for key in ('task_actor_update', 'stage_actor_update', 'actor_updates_consumed', 'learning_lineage_updates'):
            require(infos[key] == seal[key], 'Checkpoint infos/clock differs: ' + key)
        require(infos['contract_sha256'] == expected_contract and infos['successful_updates_total'] == seal['learning_updates'] and
                infos['task_campaign']['domain'] == 'recovery', 'Checkpoint provenance differs')
        require(all(finite(saved[k]) for k in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict')), 'Nonfinite learning state')
        require(saved['optimizer_state_dict']['state'], 'Empty Adam learning state')
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
        require(parent_path.name == 'model_final.pt', 'Not a full final predecessor seal')
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
                (warmup == 10 if mode == 'v18_to_v19' else warmup == parent['infos']['critic_warmup_remaining']), 'Warmup does not use original migration/remaining logic')
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
        checks['unique_new_recovery_invocation'] = dict(status='pending', detail='No new frozen v19 recovery invocation')
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
                    'src/wheeled_tasks/chassis/recovery_quality_revision.py', 'src/wheeled_tasks/chassis/preflight.py')
        require(all(n in pinned for n in required), 'Missing required new runtime source snapshot')
        for name, digest in pinned.items():
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe source snapshot path')
            require(sha(source / name) == sha(directory / 'source' / name) == digest, 'Actual PPO source snapshot differs: ' + name)
        for name, digest in settings.get('runtime_pins', {}).items():
            require(pinned.get(name) == digest, 'Prepared candidate runtime SHA differs: ' + name)
        native = startup['startup']; preflight = native['runtime_preflight']; observed = preflight['observed']
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
        from wheeled_tasks.chassis.recovery_quality_revision import HOLD_GATE_BINDING
        require(observed['recovery_hold_cost_gate_binding'] == HOLD_GATE_BINDING and
                native['recovery_hold_cost_gate_binding'] == HOLD_GATE_BINDING, 'Native FSM/geometry binding absent or differs')
        require(sum(native['scene_group_counts'].values()) == 4096 and min(native['scene_group_counts'].values()) > 0, 'Native population invalid')
        completed, actors = progress['successful_updates'], progress['actor_updates_in_block']
        require(completed - actors == min(warmup, completed) and progress['parent_updates'] == origin and
                progress['task_actor_update'] == bound['seal']['task_actor_update'] + actors and
                progress['stage_actor_update'] == bound['seal']['stage_actor_update'] + actors, 'Progress critic/actor accounting differs')
        old_config = read(source / 'contracts/v6_task_campaign_recovery_worker_v18.json')
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
    return dict(run=str(Path(predecessor['actual_parents']['recovery']['path']).parents[4]),
                source=source, predecessor=predecessor, revision=revision, mode=args.mode,
                engineering_limitations=admission.get('engineering_limitations', {}),
                phase_progress_capture=str(args.phase_progress_capture) if args.phase_progress_capture else None,
                event_prefix_bytes=args.event_prefix_mib * 1024 ** 2,
                runtime_pins={n: row['candidate'] for n, row in source.get('changed_runtime_files', {}).items()
                              if n in ('scripts/train_chassis.py', 'src/wheeled_tasks/chassis/env.py',
                                       'src/wheeled_tasks/chassis/rewards.py', 'src/wheeled_tasks/chassis/recovery.py',
                                       'src/wheeled_tasks/chassis/recovery_prefix_curriculum.py',
                                       'src/wheeled_tasks/chassis/recovery_quality_revision.py',
                                       'src/wheeled_tasks/chassis/preflight.py')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--emit-remote-code', action='store_true', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--admission', type=Path, required=True)
    p.add_argument('--predecessor', type=Path, required=True)
    p.add_argument('--revision', type=Path, required=True)
    p.add_argument('--mode', choices=('v18_to_v19', 'same_v19'), default='v18_to_v19')
    p.add_argument('--phase-progress-capture', type=Path)
    p.add_argument('--event-prefix-mib', type=int, choices=range(1, 129), default=64)
    args = p.parse_args(); settings = settings_for(args)
    # Emission prints a read-only inspector, not a shell or network command.
    text = Path(__file__).read_text().split('\nif __name__ ==')[0]
    print(text + '\nprint(json.dumps(inspect(json.loads(' + repr(json.dumps(settings)) + ')), allow_nan=False))\n')


print(json.dumps(inspect(json.loads('{"run": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358", "source": {"schema": "task-recovery-quality-021-source-candidate-v1", "status": "frozen_pending_native_not_deployed", "source_commit": "d0dce6181df83b03fe166abcbadef3c81de2e57e", "previous_source_commit": "7f315f30741be01d70046a11848443038c14b240", "archive_local": "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/source_d0dce6181df8.tar.gz", "archive_sha256": "2c019ba1fd56053af75bf631258397176de8c5b40f36d35ccdf54d32f5b21ee9", "archive_size_bytes": 107957017, "clean_root": "/tmp/scut-recovery-frozen-021-qv1ukmbr", "source_remote": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_d0dce6181df8", "archive_remote": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_d0dce6181df8.tar.gz", "contract_file": "contracts/v6_task_campaign_v19.json", "plan_sha256": "dbce7bf39105bb284ee9927f06239ff8ce767bcadbab896f2b653897c19649f9", "manifest_sha256": "08c5e2bf9c36d710791593c802b73716e1c0debe2f2fc466b54a8a474e694666", "changed_runtime_files": {"src/wheeled_tasks/chassis/env.py": {"previous": "d8a8bbaa91f7e7f8055031d714ce97657e950ae93a949b70be5c4cd2ec1d183b", "candidate": "ea40933b811549edb6a4fe3d2d5963b4a05a698e653cfead45518c5e45496c0e"}, "src/wheeled_tasks/chassis/recovery.py": {"previous": "03765b4d6c0cc87f307bfc4063bd410f86a0f901b65338403f0eff061eac2272", "candidate": "71ee882bd6d660fd5d7b82b0075e12ad1fee71fba196884dde1fd17692797d42"}, "src/wheeled_tasks/chassis/rewards.py": {"previous": "371899baec0fc9aac75ee05ff5f24e6d9fb9eb40e529ffaf8eee4567eda4fec3", "candidate": "b6f402f275114719bb8fe30a5c5a43a27edabc36d3e3a58e7c9f393022293c2d"}, "src/wheeled_tasks/chassis/recovery_prefix_curriculum.py": {"previous": "c11cfe901744d943e27ac940c48e668be628c95ce57d6ea3f1461bca055b8bd4", "candidate": "2188f108930a6f8898ee145c397342ef8a235bfd0007c1c82f7a54f878253766"}, "src/wheeled_tasks/chassis/recovery_quality_revision.py": {"previous": null, "candidate": "6ac2ce49d57c1061ed03a519e7290904fb831d33dca6f64cbc351a0b9d9b4972"}, "src/wheeled_tasks/chassis/preflight.py": {"previous": "f3bd4216b1ed8f089905e94879d3ab3972b31474f84b86f65285877294ddf772", "candidate": "d99492798eb75281dc0446dc5bf458a1f37fbd8919bb6d70c4f7cfb32a4e483b"}, "src/wheeled_tasks/chassis/evaluation.py": {"previous": "77b86f6e8230fda71e9e8ce93c14b3a3430dbecfbe9da6b1534f3264e069fd25", "candidate": "5f9112d7e7645dbce1220f8f0baecaca4ffaa6bdb748e08016b8a17527de75b1"}, "src/wheeled_tasks/chassis/task_campaign.py": {"previous": "6ed468cf95a026037305d923d39dec5ca617baf3977614f04ba8b65b3dad573a", "candidate": "a4aaad8cc7606061ebb69ba680bc96bc5bc9a46beeefa279e427cf348a98453f"}, "scripts/train_chassis.py": {"previous": "e7e6f711e238b72ed258ca6793f8cb7c88977a29572c4f9d26877743167d1773", "candidate": "2bce70a0274e9ed1012e32e4d9985adeece6029f601b843119466870d0e3aeaf"}, "scripts/run_task_campaign.py": {"previous": "be75e6689f9cc3ed9aa379d537b031df23c873e3808daad682b35b6b40146d70", "candidate": "3bc78465e68411a67a449b98aaefaee9203494ad63b486d2b23a3721e800d036"}, "scripts/chassis_remote_job.py": {"previous": "4e123655d20225215ce430491d02a7a0b5071d19f4c6f2aca81256e251483e79", "candidate": "39dbe81872a29e9b5b16d69a126db3b609a02d2d27ea06aa354e673fcf3251d3"}, "scripts/probe_scut_curriculum.py": {"previous": "934559f851ab3b7aa3761313cbd457b3acaf914905f03d9b8fbca5956d9c99de", "candidate": "21c1cb926ff35f0fed1d9d52a6a37d0db65f8113b57c6263af46b589be7dabae"}, "tests/test_recovery_quality_revision.py": {"previous": null, "candidate": "5e210fd62ec479e86013743ffff0397045ba1505e5c6c09fb5b48e548501826b"}, "contracts/v6_task_campaign_recovery_expectations_v19.json": {"previous": null, "candidate": "f84a73d47f90f941a4520472cd62576b8e55b784ccc0b0643f328b7b7b84d423"}, "contracts/v6_task_campaign_recovery_worker_v19.json": {"previous": null, "candidate": "7918e7850e16b96edfa54a8854d3f3571e8101dfa6527cad78a06b4e15baea95"}, "contracts/v6_task_campaign_v19.json": {"previous": null, "candidate": "dbce7bf39105bb284ee9927f06239ff8ce767bcadbab896f2b653897c19649f9"}, "contracts/v6_task_campaign_migration_v19.json": {"previous": null, "candidate": "08c5e2bf9c36d710791593c802b73716e1c0debe2f2fc466b54a8a474e694666"}}, "source_inventory_file": "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/frozen_source_inventory_021.json", "source_inventory_sha256": "c3f3f17a3747b27e2982793a796fff29fe622d3406b9d2d614a682bbaf34d068", "qualification": null, "formal_updates_consumed": 0}, "predecessor": {"schema": "task-recovery-quality-predecessor-021-v1", "source_commit": "d0dce6181df83b03fe166abcbadef3c81de2e57e", "previous_source_commit": "7f315f30741be01d70046a11848443038c14b240", "evidence_sha256": {"/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/clean_tests_021.json": "726f70e8efb27c99768dda6767c44fa81994906172af7cd173ad0b7bb9ed4373", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/clean_30_roles_021.json": "ed5943ebb4597128c9d1a86dc95146e321b280e73a7c7d698acb6f9f9911c32f", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/remote_30_roles_021.json": "6fd8f60c224ee419f156f01dfd3cad5fe33c38c36ed97295ea01e530230c8d88", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/role_inheritance_review_021.json": "18fcb0c68a117f3ce67ed9f716fe69502f06ef1d8d0a41833428217171a38fb5", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/native_teaching_readback_021_002.json": "c8b687af3830f1ff0925453c7169a0328fda1a316dbcbd22f6ee59df7b5693bb", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/source_candidate_021.json": "6019bcd0ada9d4512c6cfa65c7db7b577b6d0a7941b37dd384109c27f6d9c306", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/frozen_source_inventory_021.json": "c3f3f17a3747b27e2982793a796fff29fe622d3406b9d2d614a682bbaf34d068", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/clean_tests_021.stdout.log": "4f66f95e56a802054dddec5eba6401c5b5234b6a08d837a25d4aaa58cd583c4a", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/native_observer_failed_021_001.json": "cbba5b6d50e179695a9db94be635aa35d2e6179e5ad23aecc55f865f09367910", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/native_engineering_parent_021_002.json": "7be1ae04b8ba831e778641e9b00aa09c840e7282c8a0b930788660ffca8dc13a", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/native_teaching_021_002.invocation.json": "f22cc182f376d44f21594af97da1355612645e475d6449f1c7c503b571f52901", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/independent_deploy_review_021.json": "ab220106de5b2f29ef59fbcbfcbc484cecc6f5142356ea4cf8b3f57be7b53e4f", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/learning_audit/hold_cost_candidate_021/validate_native_teaching_021_002.py": "4fed2e4f85f4d8e92c2af804aa98975dc13381c0a6bfcb02f7f49f92d35c84e9", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/independent_native_teaching_CPU_021_002.json": "83a34e61ef00fc2e76e0567e35ed6e004b8f879fbfc1eb2623291447e94854f2", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/independent_native_teaching_review_021_002.json": "ea1f3cc5bce49f7b4aee0de6b3d38226f579e173217aa544f30d6a46aeb3ffa6", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/independent_stop_emitter_review_021_002.json": "40684422106d0459e64875b3fe53dc183bd36da2787e2548857d475bda88001d", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/learning_audit/hold_cost_candidate_021/native_observer_CPU_verification_021_002.json": "a1b046d0c1dde7a8002218d79fcc0aa786fc0ac27f7b587fce4a9fb45d936ecb", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/learning_audit/hold_cost_candidate_021/actual_engineering_review_021_002.json": "b82e9d014b2f5130295576036f80eee82d4aa050f8a08c16c037ed5ddfa106b2", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/learning_audit/hold_cost_candidate_021/solver_independent_native_teaching_review_021.json": "d8d07148557c905eec932c28b754e057987c3356c1111ed1c40907590a373073", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/deploy_recovery_quality_021.py": "e8d057c125c066e5df653bc133f12ec546843073149056b90be0947f317b4194", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/emit_stop_previous_021.py": "3cddee45f7c90eee1f6c240adb264ce87cd6c77e0b2c8ad57415b49cc4784045", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/observe_first_progress_021.py": "4eba41c9295f44bf40c57108f46bb78f37261aad85de57a6d933f2d2dc65de07", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/native_teaching_021_002/report.json": "8342e52706c968c502287ded58e85042b3f5225d5a5e70e20427a966ce404f12", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/native_teaching_021_002/identity.json": "d9e4a4773e2a332f3abe36afffca3898714966d0ad84ec9744ecb81ca023ba44", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/native_teaching_021_002/payment_summary.json": "942916a3b1e01f8d28a5813c282d6b0015c4022da4186e25bb78e0c99f55a16e", "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_021/native_teaching_021_002/teaching_summary.json": "2ed4f0362b220623726b999eace28e465c2e71e5071bc22e328864836afa1039"}, "job": {"pid": 3494087, "command": ["/home/kaiser/robot-rl-sim60/env/bin/python", "-B", "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_7f315f30741b/scripts/run_task_campaign.py", "--contract", "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_7f315f30741b/contracts/v6_task_campaign_v18.json", "--num-envs", "4096", "--device", "cuda:0", "--research", "--max-runtime-seconds", "15552000", "--run-dir", "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train"], "log": "train.resume_017.log", "previous_job": "job.previous_017.json", "source_revision_receipt": "job.source_revision_017.json", "started_at": "2026-10-06T01:45:50.779182+00:00"}, "campaign_state": {"status": "stopped", "campaign_id": "v6-task-domains-20261001", "contract_sha256": "027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022", "domains": {"flat": {"status": "domain_schedule_finished_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/flat/attempt_007/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/flat/attempt_007/train/model_final.pt", "candidate_checkpoint_sha256": "7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f", "parent_domain": "A", "consumed_updates": 13020, "scope": "unassessed_individual_task_research_candidate", "schedule_completion": {"clock": "domain_actor_updates", "end_actor_update": 13000, "actual_actor_update": 13000, "reason": "iteration_endpoint", "automatic_advance": true, "domain_qualified": null}}, "recovery": {"status": "domain_reopened_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_001/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_001/train/model_final.pt", "candidate_checkpoint_sha256": "2119674b64db65c0570a4b093067430cba90ceecdcd28c540d97c76b7f35572f", "parent_domain": "flat", "consumed_updates": 2314, "scope": "unassessed_individual_task_research_candidate", "completion_history": [{"plan_revision": "v15", "record": {"status": "domain_review_finished_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_001/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_001/train/model_final.pt", "candidate_checkpoint_sha256": "2119674b64db65c0570a4b093067430cba90ceecdcd28c540d97c76b7f35572f", "parent_domain": "flat", "consumed_updates": 2314, "scope": "unassessed_individual_task_research_candidate", "finish_review": {"schema": "task-domain-review-v1", "campaign_id": "v6-task-domains-20261001", "plan_sha256": "7300675623ce15d533ed9fd0b4c2755bd8f9f412b25b4c172b5174d0240edcd3", "domain": "recovery", "decision": "finish_domain", "reason": "Preserve the recovery research branch after verified clean mean-policy velocity/yaw regression; continue speed from its unchanged flat parent while redesigning takeover. No qualification or consumption refund.", "checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_001/train/checkpoints/update_00016000/model.pt", "checkpoint_sha256": "9d0ac50c9aae3c4cfab40cbb60873c9976cd06cd8c0e566991a1ed9b6212067f", "evidence": [{"kind": "curve_review", "file": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/reviews/recovery16000_curve_review_20261003.json", "sha256": "f2a924cd31cec9a1023bac24a1aa2de4cfbc763be30c837e4f7d277272008d9b"}, {"kind": "behavior_review", "file": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/reviews/recovery16000_behavior_review_20261003.json", "sha256": "1bb972c5319b34b0afa4fd5e70b19ce1ee55b34440dd26b73975df0f5d2eeb58"}], "consumed_updates_at_review": 16334, "domain_qualified": null, "continuation": "seal_latest_full_learning_state; preserve_all_consumption", "token": "239126e16dcaf6e5276ba572f10dec3a9aec12e5b422b0d76a784f8e0c15e365"}, "schedule_completion": {"clock": "domain_actor_updates", "end_actor_update": 15000, "actual_actor_update": 2294, "reason": "review", "automatic_advance": true, "domain_qualified": null}}}], "reopening": {"schema": "task-domain-reopen-v18", "source_plan_sha256": "3d4ceb76c212fb47b3380d980b26afda19d0203785f556c5aa2fb44b5b353e61", "source_contract_sha256": "f6aa77c6808c9746dc8dee431e8b3331b6b0f883db701335e57069e7df738ee8", "source_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_001/train/model_final.pt", "source_checkpoint_sha256": "2119674b64db65c0570a4b093067430cba90ceecdcd28c540d97c76b7f35572f", "origin_actor_update": 2294, "end_actor_update": 15000, "actor_updates_added": 12706, "critic_adaptation_updates": 10, "consumption": "preserved_no_refund", "outcome": "unassessed_research_candidate"}}, "speed": {"status": "domain_reopened_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_002/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_002/train/model_final.pt", "candidate_checkpoint_sha256": "81920e8b8275f9e1c865ce7efc0b3cff0f2d098e9e0a1f1daeb7c02dfc9cd7b5", "parent_domain": "flat", "consumed_updates": 5093, "scope": "unassessed_individual_task_research_candidate", "completion_history": [{"plan_revision": "v12", "record": {"status": "domain_schedule_finished_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_001/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_001/train/model_final.pt", "candidate_checkpoint_sha256": "123841b609439939e9c9c8962c0171330892276446a692c936c8f33375ce11da", "parent_domain": "flat", "consumed_updates": 3583, "scope": "unassessed_individual_task_research_candidate", "schedule_completion": {"clock": "domain_actor_updates", "end_actor_update": 3563, "actual_actor_update": 3563, "reason": "iteration_endpoint", "automatic_advance": true, "domain_qualified": null}}}, {"plan_revision": "v15", "record": {"status": "domain_schedule_finished_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_002/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_002/train/model_final.pt", "candidate_checkpoint_sha256": "81920e8b8275f9e1c865ce7efc0b3cff0f2d098e9e0a1f1daeb7c02dfc9cd7b5", "parent_domain": "flat", "consumed_updates": 5093, "scope": "unassessed_individual_task_research_candidate", "completion_history": [{"plan_revision": "v12", "record": {"status": "domain_schedule_finished_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_001/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_001/train/model_final.pt", "candidate_checkpoint_sha256": "123841b609439939e9c9c8962c0171330892276446a692c936c8f33375ce11da", "parent_domain": "flat", "consumed_updates": 3583, "scope": "unassessed_individual_task_research_candidate", "schedule_completion": {"clock": "domain_actor_updates", "end_actor_update": 3563, "actual_actor_update": 3563, "reason": "iteration_endpoint", "automatic_advance": true, "domain_qualified": null}}}], "reopening": {"schema": "task-domain-reopen-v14", "source_plan_sha256": "e79e94872f041ddf935ad961c59093b827b0b50f3519c9234718118cf1b59298", "source_contract_sha256": "37837b286de10a66404ef155ae84ead613a90f76375d4abe6874a042eae6b406", "origin_actor_update": 3563, "end_actor_update": 5063, "actor_updates_added": 1500, "consumption": "preserved_no_refund", "outcome": "unassessed_research_candidate"}, "schedule_completion": {"clock": "domain_actor_updates", "end_actor_update": 5063, "actual_actor_update": 5063, "reason": "iteration_endpoint", "automatic_advance": true, "domain_qualified": null}}}], "reopening": {"schema": "task-domain-reopen-v18", "source_plan_sha256": "3d4ceb76c212fb47b3380d980b26afda19d0203785f556c5aa2fb44b5b353e61", "source_contract_sha256": "f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3", "source_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_002/train/model_final.pt", "source_checkpoint_sha256": "81920e8b8275f9e1c865ce7efc0b3cff0f2d098e9e0a1f1daeb7c02dfc9cd7b5", "origin_actor_update": 5063, "end_actor_update": 15000, "actor_updates_added": 9937, "critic_adaptation_updates": 10, "consumption": "preserved_no_refund", "outcome": "unassessed_research_candidate"}}, "terrain": {"status": "domain_reopened_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_002/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_002/train/model_final.pt", "candidate_checkpoint_sha256": "762b5688a3eab8af87ea6a87e552fae46f005eb1b858561c00b6853d30b3e0a1", "parent_domain": "flat", "consumed_updates": 2206, "scope": "unassessed_individual_task_research_candidate", "completion_history": [{"plan_revision": "v12", "record": {"status": "domain_schedule_finished_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_000/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_000/train/model_final.pt", "candidate_checkpoint_sha256": "fe429597ebb565644b2774747f558a8f70fbea53e033819096bbf9fe2a927134", "parent_domain": "flat", "consumed_updates": 1196, "scope": "unassessed_individual_task_research_candidate", "schedule_completion": {"clock": "domain_actor_updates", "end_actor_update": 1186, "actual_actor_update": 1186, "reason": "iteration_endpoint", "automatic_advance": true, "domain_qualified": null}}}, {"plan_revision": "v15", "record": {"status": "domain_schedule_finished_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_002/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_002/train/model_final.pt", "candidate_checkpoint_sha256": "762b5688a3eab8af87ea6a87e552fae46f005eb1b858561c00b6853d30b3e0a1", "parent_domain": "flat", "consumed_updates": 2206, "scope": "unassessed_individual_task_research_candidate", "completion_history": [{"plan_revision": "v12", "record": {"status": "domain_schedule_finished_unassessed", "domain_qualified": null, "assessment_status": "not_run", "accepted_stage": null, "deployment_checkpoint": null, "final_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_000/train/model_final.pt", "candidate_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_000/train/model_final.pt", "candidate_checkpoint_sha256": "fe429597ebb565644b2774747f558a8f70fbea53e033819096bbf9fe2a927134", "parent_domain": "flat", "consumed_updates": 1196, "scope": "unassessed_individual_task_research_candidate", "schedule_completion": {"clock": "domain_actor_updates", "end_actor_update": 1186, "actual_actor_update": 1186, "reason": "iteration_endpoint", "automatic_advance": true, "domain_qualified": null}}}], "reopening": {"schema": "task-domain-reopen-v14", "source_plan_sha256": "e79e94872f041ddf935ad961c59093b827b0b50f3519c9234718118cf1b59298", "source_contract_sha256": "4337842f9689ec2a4ecd0994dfb39ef821c95a9868a91f445de52f2a0afc456e", "origin_actor_update": 1186, "end_actor_update": 2186, "actor_updates_added": 1000, "consumption": "preserved_no_refund", "outcome": "unassessed_research_candidate"}, "schedule_completion": {"clock": "domain_actor_updates", "end_actor_update": 2186, "actual_actor_update": 2186, "reason": "iteration_endpoint", "automatic_advance": true, "domain_qualified": null}}}], "reopening": {"schema": "task-domain-reopen-v18", "source_plan_sha256": "3d4ceb76c212fb47b3380d980b26afda19d0203785f556c5aa2fb44b5b353e61", "source_contract_sha256": "843c848d28557907fbe3a842b271d0a7736fdf8f56b97ec0f51ba98591d15a41", "source_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_002/train/model_final.pt", "source_checkpoint_sha256": "762b5688a3eab8af87ea6a87e552fae46f005eb1b858561c00b6853d30b3e0a1", "origin_actor_update": 2186, "end_actor_update": 15000, "actor_updates_added": 12814, "critic_adaptation_updates": 10, "consumption": "preserved_no_refund", "outcome": "unassessed_research_candidate"}}}, "evaluations": {}, "started_at": "2026-10-01T11:00:56.021425+00:00", "elapsed_seconds": 375646.9725846901, "assessment_seconds": 0.0, "training_process_seconds": 375601.8376095038, "accepted_stage": null, "deployment_checkpoint": null, "automatic_promotion": false, "consumed_updates": 29996, "actor_updates_consumed": 29856, "domain_consumed_updates": {"flat": 13020, "recovery": 3105, "speed": 5093, "terrain": 2206, "jump": 5572, "stepjump": 0}, "remaining_updates": 91004, "successful_updates": 29996, "finished_at": "2026-10-06T03:13:37.049876+00:00", "ppo_process_active_seconds": 375601.8376095038, "non_training_seconds": 45.134966470242944, "revision_migration": {"schema": "task-plan-revision-v18", "campaign_id": "v6-task-domains-20261001", "source_plan_sha256": "3d4ceb76c212fb47b3380d980b26afda19d0203785f556c5aa2fb44b5b353e61", "target_plan_sha256": "027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022", "run_directory": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train", "actor_critic_optimizer": "preserved", "task_clock": "restored", "consumption": "preserved_no_refund", "consumed_updates": 29205, "actor_updates_consumed": 29075, "domain_consumed_updates": {"flat": 13020, "recovery": 2314, "speed": 5093, "terrain": 2206, "jump": 5572, "stepjump": 0}, "remaining_updates": 91795, "attempts": {"flat/attempt_000/train": {"domain": "flat", "consumed_updates": 0, "actor_updates": 0}, "flat/attempt_001/train": {"domain": "flat", "consumed_updates": 30, "actor_updates": 30}, "flat/attempt_002/train": {"domain": "flat", "consumed_updates": 514, "actor_updates": 514}, "flat/attempt_003/train": {"domain": "flat", "consumed_updates": 144, "actor_updates": 144}, "flat/attempt_004/train": {"domain": "flat", "consumed_updates": 263, "actor_updates": 253}, "flat/attempt_005/train": {"domain": "flat", "consumed_updates": 481, "actor_updates": 481}, "flat/attempt_006/train": {"domain": "flat", "consumed_updates": 1321, "actor_updates": 1311}, "flat/attempt_007/train": {"domain": "flat", "consumed_updates": 10267, "actor_updates": 10267}, "recovery/attempt_000/train": {"domain": "recovery", "consumed_updates": 400, "actor_updates": 390}, "recovery/attempt_001/train": {"domain": "recovery", "consumed_updates": 1914, "actor_updates": 1904}, "speed/attempt_000/train": {"domain": "speed", "consumed_updates": 1694, "actor_updates": 1684}, "speed/attempt_001/train": {"domain": "speed", "consumed_updates": 1889, "actor_updates": 1879}, "terrain/attempt_000/train": {"domain": "terrain", "consumed_updates": 1196, "actor_updates": 1186}, "jump/attempt_000/train": {"domain": "jump", "consumed_updates": 2581, "actor_updates": 2571}, "jump/attempt_001/train": {"domain": "jump", "consumed_updates": 174, "actor_updates": 164}, "speed/attempt_002/train": {"domain": "speed", "consumed_updates": 1510, "actor_updates": 1500}, "terrain/attempt_002/train": {"domain": "terrain", "consumed_updates": 1010, "actor_updates": 1000}, "jump/attempt_002/train": {"domain": "jump", "consumed_updates": 2817, "actor_updates": 2817}}, "domain_reopening": {"schema": "task-domain-reopening-v18", "domains": {"recovery": 2294, "speed": 5063, "terrain": 2186}, "end_actor_updates": {"recovery": 15000, "speed": 15000, "terrain": 15000}, "reopening": {"recovery": {"schema": "task-domain-reopen-v18", "source_plan_sha256": "3d4ceb76c212fb47b3380d980b26afda19d0203785f556c5aa2fb44b5b353e61", "source_contract_sha256": "f6aa77c6808c9746dc8dee431e8b3331b6b0f883db701335e57069e7df738ee8", "source_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_001/train/model_final.pt", "source_checkpoint_sha256": "2119674b64db65c0570a4b093067430cba90ceecdcd28c540d97c76b7f35572f", "origin_actor_update": 2294, "end_actor_update": 15000, "actor_updates_added": 12706, "critic_adaptation_updates": 10, "consumption": "preserved_no_refund", "outcome": "unassessed_research_candidate"}, "speed": {"schema": "task-domain-reopen-v18", "source_plan_sha256": "3d4ceb76c212fb47b3380d980b26afda19d0203785f556c5aa2fb44b5b353e61", "source_contract_sha256": "f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3", "source_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_002/train/model_final.pt", "source_checkpoint_sha256": "81920e8b8275f9e1c865ce7efc0b3cff0f2d098e9e0a1f1daeb7c02dfc9cd7b5", "origin_actor_update": 5063, "end_actor_update": 15000, "actor_updates_added": 9937, "critic_adaptation_updates": 10, "consumption": "preserved_no_refund", "outcome": "unassessed_research_candidate"}, "terrain": {"schema": "task-domain-reopen-v18", "source_plan_sha256": "3d4ceb76c212fb47b3380d980b26afda19d0203785f556c5aa2fb44b5b353e61", "source_contract_sha256": "843c848d28557907fbe3a842b271d0a7736fdf8f56b97ec0f51ba98591d15a41", "source_checkpoint": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_002/train/model_final.pt", "source_checkpoint_sha256": "762b5688a3eab8af87ea6a87e552fae46f005eb1b858561c00b6853d30b3e0a1", "origin_actor_update": 2186, "end_actor_update": 15000, "actor_updates_added": 12814, "critic_adaptation_updates": 10, "consumption": "preserved_no_refund", "outcome": "unassessed_research_candidate"}}, "consumption": "preserved_no_refund", "prior_results": "append_only_completion_history"}}, "consumed_review_tokens": ["239126e16dcaf6e5276ba572f10dec3a9aec12e5b422b0d76a784f8e0c15e365"], "reason": "Campaign stopped; all observed work remains charged"}, "consumption": {"campaign_id": "v6-task-domains-20261001", "run_directory": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train", "contract_sha256": "027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022", "attempts": {"flat/attempt_000/train": {"domain": "flat", "consumed_updates": 0, "actor_updates": 0}, "flat/attempt_001/train": {"domain": "flat", "consumed_updates": 30, "actor_updates": 30}, "flat/attempt_002/train": {"domain": "flat", "consumed_updates": 514, "actor_updates": 514}, "flat/attempt_003/train": {"domain": "flat", "consumed_updates": 144, "actor_updates": 144}, "flat/attempt_004/train": {"domain": "flat", "consumed_updates": 263, "actor_updates": 253}, "flat/attempt_005/train": {"domain": "flat", "consumed_updates": 481, "actor_updates": 481}, "flat/attempt_006/train": {"domain": "flat", "consumed_updates": 1321, "actor_updates": 1311}, "flat/attempt_007/train": {"domain": "flat", "consumed_updates": 10267, "actor_updates": 10267}, "recovery/attempt_000/train": {"domain": "recovery", "consumed_updates": 400, "actor_updates": 390}, "recovery/attempt_001/train": {"domain": "recovery", "consumed_updates": 1914, "actor_updates": 1904}, "speed/attempt_000/train": {"domain": "speed", "consumed_updates": 1694, "actor_updates": 1684}, "speed/attempt_001/train": {"domain": "speed", "consumed_updates": 1889, "actor_updates": 1879}, "terrain/attempt_000/train": {"domain": "terrain", "consumed_updates": 1196, "actor_updates": 1186}, "jump/attempt_000/train": {"domain": "jump", "consumed_updates": 2581, "actor_updates": 2571}, "jump/attempt_001/train": {"domain": "jump", "consumed_updates": 174, "actor_updates": 164}, "speed/attempt_002/train": {"domain": "speed", "consumed_updates": 1510, "actor_updates": 1500}, "terrain/attempt_002/train": {"domain": "terrain", "consumed_updates": 1010, "actor_updates": 1000}, "jump/attempt_002/train": {"domain": "jump", "consumed_updates": 2817, "actor_updates": 2817}, "recovery/attempt_002/train": {"domain": "recovery", "consumed_updates": 174, "actor_updates": 164}, "recovery/attempt_003/train": {"domain": "recovery", "consumed_updates": 300, "actor_updates": 300}, "recovery/attempt_004/train": {"domain": "recovery", "consumed_updates": 317, "actor_updates": 317}}, "rollback_refunds_consumed_updates": false, "consumed_updates": 29996, "actor_updates_consumed": 29856, "domain_consumed_updates": {"flat": 13020, "recovery": 3105, "speed": 5093, "terrain": 2206, "jump": 5572, "stepjump": 0}, "remaining_updates": 91004}, "actual_parents": {"flat": {"path": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/flat/attempt_007/train/model_final.pt", "sha256": "7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f", "seal": {"schema": "chassis-learning-checkpoint-v1", "sealed": true, "checkpoint_sha256": "7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f", "contract_sha256": "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7", "learning_updates": 14020, "learning_lineage_updates": 13398, "stage_actor_update": 13368, "global_actor_update_offset": 0, "global_actor_update": 13368, "actor_updates_consumed": 13980, "task_campaign": {"campaign_id": "v6-task-domains-20261001", "domain": "flat", "source_contract_file": "contracts/v6_s1_A_fixed_v1.json", "source_contract_sha256": "8c1b5b2b8c5c8352b00d1db8a966a5698df3c4756d0d4bb756c96646d1c1a6fc", "updates": 20000, "clock": "domain_actor_updates", "learning_audit": true, "evaluation_interval": 0, "initialization": "full_learning_state", "parent_candidate_scope": "research_not_accepted", "execution_mode": "continuous_review_or_cap_then_next", "revision_migration": {"schema": "task-worker-revision-v6", "source_contract_file": "contracts/v6_task_campaign_flat_worker_v5.json", "source_contract_sha256": "3296c141acde1317025402679675f2c53050a61e8f29122a3b4de9c93e2fb401"}}, "task_actor_update": 13000, "confirmed_jump_bins_m": [], "gate_result": "not_assessed", "role": "learning_state", "budget_consumption_is_separate_from_checkpoint_lineage": true}, "seal_sha256": "f3652ce3210148af0eb075d26e1980dffb815fd8d3864587577ab9f3509089b9", "contract": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/flat/attempt_007/train/model_final.contract.json", "contract_sha256": "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7"}, "recovery": {"path": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_004/train/model_final.pt", "sha256": "21c3cf70caadaafe94e16a8b8d0174939694017c569b0498e59b06480cf6be45", "seal": {"schema": "chassis-learning-checkpoint-v1", "sealed": true, "checkpoint_sha256": "21c3cf70caadaafe94e16a8b8d0174939694017c569b0498e59b06480cf6be45", "contract_sha256": "7b5ecc7acc6871e072c2ca131d2ad2fa169dfd321b8a6c875cf2df643c5fd013", "learning_updates": 29996, "learning_lineage_updates": 16503, "stage_actor_update": 16443, "global_actor_update_offset": 0, "global_actor_update": 16443, "actor_updates_consumed": 29856, "task_campaign": {"campaign_id": "v6-task-domains-20261001", "domain": "recovery", "source_contract_file": "contracts/v6_task_campaign_flat_worker_v6.json", "source_contract_sha256": "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7", "updates": 20000, "clock": "domain_actor_updates", "learning_audit": true, "evaluation_interval": 0, "initialization": "full_learning_state", "parent_candidate_scope": "research_not_accepted", "execution_mode": "continuous_review_or_cap_then_next", "revision_migration": {"schema": "task-worker-revision-v18", "source_contract_file": "contracts/v6_task_campaign_recovery_worker_v8.json", "source_contract_sha256": "f6aa77c6808c9746dc8dee431e8b3331b6b0f883db701335e57069e7df738ee8"}, "worker_revision_id": "v6-task-campaign-recovery-worker-v18"}, "task_actor_update": 3075, "confirmed_jump_bins_m": [], "gate_result": "not_assessed", "role": "learning_state", "budget_consumption_is_separate_from_checkpoint_lineage": true}, "seal_sha256": "33affd2b9d6b6988f2ee7855d80983bb3ba2ea08617f44c1fb6f4188e530522f", "contract": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_004/train/model_final.contract.json", "contract_sha256": "7b5ecc7acc6871e072c2ca131d2ad2fa169dfd321b8a6c875cf2df643c5fd013", "authenticated_infos": {"contract_sha256": "7b5ecc7acc6871e072c2ca131d2ad2fa169dfd321b8a6c875cf2df643c5fd013", "successful_updates_total": 29996, "stage_actor_update": 16443, "task_actor_update": 3075, "actor_updates_consumed": 29856, "learning_lineage_updates": 16503, "critic_warmup_remaining": 0}}, "speed": {"path": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_002/train/model_final.pt", "sha256": "81920e8b8275f9e1c865ce7efc0b3cff0f2d098e9e0a1f1daeb7c02dfc9cd7b5", "seal": {"schema": "chassis-learning-checkpoint-v1", "sealed": true, "checkpoint_sha256": "81920e8b8275f9e1c865ce7efc0b3cff0f2d098e9e0a1f1daeb7c02dfc9cd7b5", "contract_sha256": "f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3", "learning_updates": 25378, "learning_lineage_updates": 18491, "stage_actor_update": 18431, "global_actor_update_offset": 0, "global_actor_update": 18431, "actor_updates_consumed": 25258, "task_campaign": {"campaign_id": "v6-task-domains-20261001", "domain": "speed", "source_contract_file": "contracts/v6_task_campaign_flat_worker_v6.json", "source_contract_sha256": "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7", "updates": 20000, "clock": "domain_actor_updates", "learning_audit": true, "evaluation_interval": 0, "initialization": "full_learning_state", "parent_candidate_scope": "research_not_accepted", "execution_mode": "continuous_review_or_cap_then_next", "revision_migration": {"schema": "task-worker-revision-v14", "source_contract_file": "contracts/v6_task_campaign_speed_worker_v10.json", "source_contract_sha256": "37837b286de10a66404ef155ae84ead613a90f76375d4abe6874a042eae6b406", "origin_actor_update": 3563}, "theme": "mixed_locomotion", "sampling_revision": {"schema": "mixed-locomotion-sampling-v1", "scope": "constant_stationary_low_high_sampling_only_preserve_all_129_groups", "policy_scope": "one_policy_stationary_low_high", "allocation_clock": "constant_all_actor_updates", "flat_membership_file": "contracts/v6_task_campaign_flat_worker_v6.json", "flat_membership_sha256": "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7", "group_counts": {"basic_translation": 16, "basic_yaw": 8, "high_motion": 72, "basic_transitions": 6, "high_transitions": 16, "stand_anchors": 9, "stand_continuous": 1, "height": 1}, "integer_allocation_population": 4096, "integer_allocation": "existing_choose_scene_groups_largest_remainders_minimum_one"}}, "task_actor_update": 5063, "confirmed_jump_bins_m": [], "gate_result": "not_assessed", "role": "learning_state", "budget_consumption_is_separate_from_checkpoint_lineage": true}, "seal_sha256": "49c56cef7cb9adff8fc9126440bbf23083e4bd078152ab2a1f4910697e620ffd", "contract": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_002/train/model_final.contract.json", "contract_sha256": "f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3"}, "terrain": {"path": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_002/train/model_final.pt", "sha256": "762b5688a3eab8af87ea6a87e552fae46f005eb1b858561c00b6853d30b3e0a1", "seal": {"schema": "chassis-learning-checkpoint-v1", "sealed": true, "checkpoint_sha256": "762b5688a3eab8af87ea6a87e552fae46f005eb1b858561c00b6853d30b3e0a1", "contract_sha256": "843c848d28557907fbe3a842b271d0a7736fdf8f56b97ec0f51ba98591d15a41", "learning_updates": 26388, "learning_lineage_updates": 15604, "stage_actor_update": 15554, "global_actor_update_offset": 0, "global_actor_update": 15554, "actor_updates_consumed": 26258, "task_campaign": {"campaign_id": "v6-task-domains-20261001", "domain": "terrain", "source_contract_file": "contracts/v6_task_campaign_flat_worker_v6.json", "source_contract_sha256": "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7", "updates": 20000, "clock": "domain_actor_updates", "learning_audit": true, "evaluation_interval": 0, "initialization": "full_learning_state", "parent_candidate_scope": "research_not_accepted", "execution_mode": "continuous_review_or_cap_then_next", "revision_migration": {"schema": "task-worker-revision-v15", "source_contract_file": "contracts/v6_task_campaign_terrain_worker_v14.json", "source_contract_sha256": "35014b99f4c8ac7423df386bdddf809699c10fc75be56e6a73002f8f532be1d0", "accepted_resume_sources": [{"source_contract_file": "contracts/v6_task_campaign_terrain_worker_v14.json", "source_contract_sha256": "35014b99f4c8ac7423df386bdddf809699c10fc75be56e6a73002f8f532be1d0"}, {"source_contract_file": "contracts/v6_task_campaign_terrain_worker_v6.json", "source_contract_sha256": "4337842f9689ec2a4ecd0994dfb39ef821c95a9868a91f445de52f2a0afc456e"}]}}, "task_actor_update": 2186, "confirmed_jump_bins_m": [], "gate_result": "not_assessed", "role": "learning_state", "budget_consumption_is_separate_from_checkpoint_lineage": true}, "seal_sha256": "8df1a1f28847cc717717eb5b041b42d81fb762e90890bd14db03d5616dcffc46", "contract": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_002/train/model_final.contract.json", "contract_sha256": "843c848d28557907fbe3a842b271d0a7736fdf8f56b97ec0f51ba98591d15a41"}, "jump": {"path": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/jump/attempt_002/train/model_final.pt", "sha256": "d346b979148c50cb891a0d513a4b2bf1d1dcfccc86dfe09dce6f6614d42fe793", "seal": {"schema": "chassis-learning-checkpoint-v1", "sealed": true, "checkpoint_sha256": "d346b979148c50cb891a0d513a4b2bf1d1dcfccc86dfe09dce6f6614d42fe793", "contract_sha256": "bece6ea0672b13e7613118fda375ce27e03a0fab39ce687c8fff99c66ef93b9d", "learning_updates": 29205, "learning_lineage_updates": 18970, "stage_actor_update": 18920, "global_actor_update_offset": 0, "global_actor_update": 18920, "actor_updates_consumed": 29075, "task_campaign": {"campaign_id": "v6-task-domains-20261001", "domain": "jump", "source_contract_file": "contracts/v6_task_campaign_flat_worker_v6.json", "source_contract_sha256": "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7", "updates": 20000, "clock": "domain_actor_updates", "learning_audit": true, "evaluation_interval": 0, "initialization": "full_learning_state", "parent_candidate_scope": "research_not_accepted", "execution_mode": "continuous_review_or_cap_then_next", "revision_migration": {"schema": "task-worker-revision-v12", "source_contract_file": "contracts/v6_task_campaign_jump_worker_v6.json", "source_contract_sha256": "9c5e14a1aa6da1301ccc6c04b890c858e6104999aa7f5bea5bd0def3af1afa93"}}, "task_actor_update": 5552, "confirmed_jump_bins_m": [], "gate_result": "not_assessed", "role": "learning_state", "budget_consumption_is_separate_from_checkpoint_lineage": true}, "seal_sha256": "174403cfafc29dcdb88aa9de1b75d2e6989c75672ba08b3229c552e68cd00753", "contract": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/jump/attempt_002/train/model_final.contract.json", "contract_sha256": "bece6ea0672b13e7613118fda375ce27e03a0fab39ce687c8fff99c66ef93b9d"}}, "expected_critic_warmup": 10, "reopening": null, "qualification": null}, "revision": {"schema": "task-plan-revision-v19", "previous_job_sha256": "edd1e05bc8a0992dfb1f9e928deb5713eedf384beb2e1ab698cec6362e2f8b21", "command": ["/home/kaiser/robot-rl-sim60/env/bin/python", "-B", "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_d0dce6181df8/scripts/run_task_campaign.py", "--contract", "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_d0dce6181df8/contracts/v6_task_campaign_v19.json", "--num-envs", "4096", "--device", "cuda:0", "--research", "--max-runtime-seconds", "15552000", "--run-dir", "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train"], "source_commit": "d0dce6181df83b03fe166abcbadef3c81de2e57e", "old_source_root": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_7f315f30741b", "new_source_root": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_d0dce6181df8", "new_contract_relative_path": "contracts/v6_task_campaign_v19.json", "old_contract_sha256": "027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022", "new_contract_sha256": "dbce7bf39105bb284ee9927f06239ff8ce767bcadbab896f2b653897c19649f9", "source_archive": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_d0dce6181df8.tar.gz", "source_archive_sha256": "2c019ba1fd56053af75bf631258397176de8c5b40f36d35ccdf54d32f5b21ee9"}, "mode": "v18_to_v19", "engineering_limitations": {}, "phase_progress_capture": "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/first_progress_021/index.json", "event_prefix_bytes": 67108864, "runtime_pins": {"src/wheeled_tasks/chassis/env.py": "ea40933b811549edb6a4fe3d2d5963b4a05a698e653cfead45518c5e45496c0e", "src/wheeled_tasks/chassis/recovery.py": "71ee882bd6d660fd5d7b82b0075e12ad1fee71fba196884dde1fd17692797d42", "src/wheeled_tasks/chassis/rewards.py": "b6f402f275114719bb8fe30a5c5a43a27edabc36d3e3a58e7c9f393022293c2d", "src/wheeled_tasks/chassis/recovery_prefix_curriculum.py": "2188f108930a6f8898ee145c397342ef8a235bfd0007c1c82f7a54f878253766", "src/wheeled_tasks/chassis/recovery_quality_revision.py": "6ac2ce49d57c1061ed03a519e7290904fb831d33dca6f64cbc351a0b9d9b4972", "src/wheeled_tasks/chassis/preflight.py": "d99492798eb75281dc0446dc5bf458a1f37fbd8919bb6d70c4f7cfb32a4e483b", "scripts/train_chassis.py": "2bce70a0274e9ed1012e32e4d9985adeece6029f601b843119466870d0e3aeaf"}}')), allow_nan=False))

