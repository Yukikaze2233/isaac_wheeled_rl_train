"""Authorized early entry restores real CPU seals; no environment or PPO step."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis import integrated_runtime as runtime
from wheeled_tasks.chassis.integrated_preflight import _clocks
from wheeled_tasks.chassis.integrated_training import critic_warmup_remaining
from test_integrated_lineage import fixture, checkpoint, metadata, write
from test_integrated_runtime import seal_clocks


@pytest.fixture
def early(fixture):
    f = fixture
    f.manifest['lineage_id'] = 'source028-joint-serial-v1'
    authorization = dict(schema=lineage.EARLY_SCHEMA, lineage_id=f.manifest['lineage_id'],
        decision='seal_current_REC_then_same_actor_serial', run_root=str(f.run),
        source_commit='6f07ec298e01d2686216787cfba34b394151f1aa',
        source_plan_sha256=f.manifest['campaign_plan']['sha256'],
        recovery_contract_sha256=f.manifest['recovery_entry']['contract_sha256'],
        planned_actor_endpoint=15000, minimum_actor_update=13868, qualification=None)
    f.authorization_path = f.source / 'contracts/v6_integrated_early_entry_authorization_v26.json'
    write(f.authorization_path, authorization)
    f.manifest['recovery_entry']['early_stop'] = dict(schema=lineage.EARLY_SCHEMA,
        authorization_file=str(f.authorization_path.relative_to(f.source)),
        authorization_sha256=lineage.digest(f.authorization_path))
    write(f.path, f.manifest)
    f.binding, _ = lineage.load_plan(f.path, lineage.digest(f.path), f.source, f.run)
    f.rec = checkpoint(f, 'recovery', count=24000, task=13870, lineage=23000, final=True)
    completion_path = Path(f.rec['path']).parent / 'completion.json'
    completion = json.loads(completion_path.read_text())
    completion.update(status='stopped', reason='signal', checkpoint_sha256=f.rec['sha256'])
    write(completion_path, completion)
    state = dict(status='stopped', campaign_id='v6-task-domains-20261001',
                 contract_sha256=f.manifest['campaign_plan']['sha256'])
    snapshots = f.run / 'integrated_early_origin_v26'
    f.stopped_state = snapshots / 'campaign_state.json'
    f.stopped_completion = snapshots / 'completion.json'
    for path in [f.stopped_state, f.stopped_completion, f.run / 'campaign_state.json', f.run / 'completion.json']:
        write(path, state)
    f.rec_record = dict(status='authorized_REC_early_stop_unassessed', domain_qualified=None,
        early_stop_protocol=lineage.EARLY_PROTOCOL,
        early_stop_authorization_sha256=lineage.digest(f.authorization_path),
        early_stop_campaign_state=dict(file=str(f.stopped_state), sha256=lineage.digest(f.stopped_state)),
        early_stop_campaign_completion=dict(file=str(f.stopped_completion), sha256=lineage.digest(f.stopped_completion)),
        candidate_checkpoint=f.rec['path'], candidate_checkpoint_sha256=f.rec['sha256'])
    f.entry = lineage.admit_entry(f.binding, f.rec, f.rec_record)
    return f


def test_actual_early_origin_and_old_consumption_are_preserved(early):
    f = early
    selected = lineage.select_learning_source(f.binding, 'speed', f.entry)
    assert selected['checkpoint'] == f.rec
    assert selected['mode'] == 'rebase_from_authorized_REC'
    assert selected['origin_recovery_actor_update'] == 13870
    assert selected['fresh_joint_actor_updates'] == selected['fresh_domain_actor_updates'] == 0
    assert selected['actor_critic_optimizer'] == 'preserved'
    allowance = lineage.remaining_budget(f.binding, selected, f.attempts, requested_actor_updates=2)
    assert allowance['historical_and_joint_domain_consumed_updates'] == 5000
    assert allowance['charged_critic_updates'] == 10
    assert allowance['requested_total_updates'] == 12
    assert allowance['rollback_refunds_consumed_updates'] is False
    assert critic_warmup_remaining(dict(selection=selected), {}) == 10


def test_actual_origin_advances_in_runtime_and_independent_preflight(early):
    selected = lineage.select_learning_source(early.binding, 'speed', early.entry)
    document = runtime.source_clocks(early.binding, selected)
    assert document['component_actor_updates'] == dict(recovery=13870, speed=0)
    advanced = runtime.advance_clock_document(document, 7)
    assert advanced['origin_recovery_actor_update'] == 13870
    assert advanced['component_actor_updates'] == dict(recovery=13877, speed=7)
    assert _clocks(dict(clocks_at_source=document), 7) == advanced['component_actor_updates']


def test_replay_uses_stopped_origin_after_live_campaign_advances(early):
    write(early.run / 'campaign_state.json', dict(status='running', active_joint_domain='speed'))
    selected = lineage.select_learning_source(early.binding, 'speed', early.entry)
    assert selected['checkpoint'] == early.rec
    assert runtime.source_clocks(early.binding, selected)['component_actor_updates']['recovery'] == 13870


@pytest.mark.parametrize('patch', ['authorization_SHA', 'protocol', 'fake_normal_record',
    'qualified', 'wrong_candidate', 'live_snapshot_path', 'snapshot_SHA', 'worker_completed', 'worker_clock'])
def test_bad_early_evidence_is_rejected(early, patch):
    f = early
    record = deepcopy(f.rec_record)
    if patch == 'authorization_SHA':
        record['early_stop_authorization_sha256'] = '0' * 64
    elif patch == 'protocol':
        record['early_stop_protocol'] = 'worker_only_stop'
    elif patch == 'fake_normal_record':
        record['status'] = 'domain_schedule_finished_unassessed'
    elif patch == 'qualified':
        record['domain_qualified'] = True
    elif patch == 'wrong_candidate':
        record['candidate_checkpoint_sha256'] = '0' * 64
    elif patch == 'live_snapshot_path':
        record['early_stop_campaign_state']['file'] = str(f.run / 'campaign_state.json')
    elif patch == 'snapshot_SHA':
        write(f.stopped_state, dict(status='running'))
    else:
        path = Path(f.rec['path']).parent / 'completion.json'
        completion = json.loads(path.read_text())
        completion['status' if patch == 'worker_completed' else 'task_actor_update'] = 'completed' if patch == 'worker_completed' else 13869
        write(path, completion)
    with pytest.raises(ValueError):
        lineage.admit_entry(f.binding, f.rec, record)


@pytest.mark.parametrize('field,value', [('decision', 'finish_any_domain'), ('run_root', '/tmp/another'),
    ('source_commit', '0' * 40), ('source_plan_sha256', '0' * 64),
    ('recovery_contract_sha256', '0' * 64), ('planned_actor_endpoint', 13870),
    ('minimum_actor_update', True), ('qualification', True)])
def test_bad_frozen_authorization_is_rejected_even_with_updated_pin(early, field, value):
    f = early
    authorization = json.loads(f.authorization_path.read_text())
    authorization[field] = value
    write(f.authorization_path, authorization)
    f.manifest['recovery_entry']['early_stop']['authorization_sha256'] = lineage.digest(f.authorization_path)
    write(f.path, f.manifest)
    with pytest.raises(ValueError):
        lineage.load_plan(f.path, lineage.digest(f.path), f.source, f.run)


def test_newer_rec_final_prevents_rollback(early):
    checkpoint(early, 'recovery', count=24001, task=13871, lineage=23001, final=True, suffix='002')
    with pytest.raises(ValueError, match='latest final'):
        lineage.admit_entry(early.binding, early.rec, early.rec_record)


def test_tampered_selection_origin_is_rejected(early):
    selected = lineage.select_learning_source(early.binding, 'speed', early.entry)
    selected['origin_recovery_actor_update'] = 15000
    with pytest.raises(ValueError):
        runtime.source_clocks(early.binding, selected)


def test_own_resume_keeps_early_origin_and_does_not_repeat_warmup(early):
    f = early
    meta = metadata(f, 'speed', joint_actor=2, domain_actor=2)
    own = checkpoint(f, 'speed', count=24012, task=2, lineage=23012, meta=meta, warmup=0)
    selected = lineage.select_learning_source(f.binding, 'speed', f.entry, own_candidates=[own])
    assert selected['mode'] == 'resume_joint_own'
    assert selected['origin_recovery_actor_update'] == 13870
    assert critic_warmup_remaining(dict(selection=selected), {'critic_warmup_remaining': 0}) == 0


def test_normal_path_keeps_legacy_clock_bytes_and_rejects_early_clock(fixture):
    selected = lineage.select_learning_source(fixture.binding, 'speed', fixture.entry)
    assert selected['mode'] == 'rebase_from_normal_REC'
    assert 'origin_recovery_actor_update' not in selected
    document = runtime.source_clocks(fixture.binding, selected)
    assert 'origin_recovery_actor_update' not in document
    assert document['component_actor_updates'] == dict(recovery=15000, speed=0)
    record = deepcopy(fixture.rec_record)
    record['status'] = 'authorized_REC_early_stop_unassessed'
    with pytest.raises(ValueError):
        lineage.admit_entry(fixture.binding, fixture.rec, record)


@pytest.mark.parametrize('actual', [13867, 15000])
def test_actor_outside_authorized_early_interval_is_rejected(early, actual):
    f = early
    reference = checkpoint(f, 'recovery', count=24000, task=actual, lineage=23000, final=True)
    path = Path(reference['path']).parent / 'completion.json'
    completion = json.loads(path.read_text())
    completion.update(status='stopped', checkpoint_sha256=reference['sha256'])
    write(path, completion)
    record = dict(f.rec_record, candidate_checkpoint=reference['path'],
                  candidate_checkpoint_sha256=reference['sha256'])
    with pytest.raises(ValueError, match='planned-stop'):
        lineage.admit_entry(f.binding, reference, record)


def test_completed_joint_parent_inherits_real_early_origin(early):
    f = early
    selected = lineage.select_learning_source(f.binding, 'speed', f.entry)
    clocks = runtime.advance_clock_document(runtime.source_clocks(f.binding, selected), 2)
    meta = metadata(f, 'speed', joint_actor=2, domain_actor=2, complete=True)
    parent = checkpoint(f, 'speed', count=24012, task=2, lineage=23012, meta=meta, final=True)
    parent = seal_clocks(parent, clocks)
    record = dict(status='joint_domain_finished_unassessed', candidate_checkpoint=parent['path'],
                  candidate_checkpoint_sha256=parent['sha256'])
    terrain = lineage.select_learning_source(f.binding, 'terrain', f.entry,
        parent=parent, parent_record=record)
    assert terrain['mode'] == 'inherit_completed_joint_parent'
    assert terrain['origin_recovery_actor_update'] == 13870
    assert runtime.source_clocks(f.binding, terrain)['component_actor_updates'] == dict(recovery=13872, speed=2, terrain=0)
