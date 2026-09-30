"""Reward-only repair preserves A's learning state and charges the latest ledger."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis import continuation
from wheeled_tasks.chassis.policy_transfer import validate_reward_resume

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_chassis_continuation import Continuation, validate_contracts, validate_plan


def read(path):
    return json.loads((ROOT / path).read_text())


def test_only_declared_landing_reward_change_is_accepted():
    source = read('contracts/v6_s1_A_fixed_v1.json')
    target = continuation.worker_contract(source, {'file': 'design.json'}, continuation.REPAIR_REVISION)
    receipt = validate_reward_resume(source, target)
    assert receipt['reward'] == 'finite_passive_landing_cost'
    assert receipt['actor_critic_optimizer'] == 'preserved'
    assert target['checkpoint_first_update'] == 739
    assert target['landing_tracking'] == {
        **source['landing_tracking'], 'passive_semantics': 'finite_task_cost_v1', 'window_seconds': .6}
    for key in ('dense_tracking', 'new_asset_curriculum', 'evaluation', 'actuator_response',
                'usb_transport', 'v5_control', 'landing_tracking'):
        changed = deepcopy(target)
        changed[key]['unexpected'] = True
        with pytest.raises(ValueError, match='undeclared contract fields'):
            validate_reward_resume(source, changed)
    for key, value in [('physics_dt', .005), ('actor_dim', 36), ('learning_rate', 2e-4),
                       ('resume_critic_warmup_updates', 10)]:
        changed = deepcopy(target)
        changed[key] = value
        with pytest.raises(ValueError, match='undeclared contract fields'):
            validate_reward_resume(source, changed)


def test_A_clock_is_distinct_from_latest_consumption(monkeypatch):
    source = read('contracts/v6_s1_A_fixed_v1.json')
    target = continuation.worker_contract(source, {}, continuation.REPAIR_REVISION)
    args = SimpleNamespace(resume_reward_change=True, resume='A.pt', consumed_updates=738,
        actor_updates_consumed=718, stage_actor_update=368, updates=40, global_actor_update_offset=0,
        learning_rate_scale=1., num_envs=20000, seed=290927, block_index=0)
    monkeypatch.setattr(continuation, 'digest', lambda _: continuation.CHECKPOINT_SHA)
    continuation.validate_worker(target, args)
    for key, value in [('consumed_updates', 698), ('actor_updates_consumed', 678),
                       ('stage_actor_update', 408), ('updates', 262), ('learning_rate_scale', .5)]:
        changed = deepcopy(args)
        setattr(changed, key, value)
        with pytest.raises(ValueError):
            continuation.validate_worker(target, changed)
    assert continuation.charged_updates([], continuation.REPAIR_REVISION) == 738
    assert continuation.charged_updates([{'successful_updates': 13, 'actor_updates_in_block': 13}],
                                       continuation.REPAIR_REVISION) == 751
    assert continuation.charged_updates([{'successful_updates': 40, 'actor_updates_in_block': 40}],
                                       continuation.REPAIR_REVISION) == 778


def test_repair_plan_protection_and_all_entry_roles():
    plan = read('contracts/v6_s1_landing_repair_v1.json')
    validate_plan(plan)
    assert len(validate_contracts(plan)) == 5
    for key, value in [('source_consumed_updates', 698), ('source_actor_consumed_updates', 678),
                       ('budget_source_stage', plan['source_stage']), ('budget_source_files_sha256', {}),
                       ('protected_case_names', plan['protected_case_names'][1:]),
                       ('inherited_recovery_attempts', []), ('automatic_formal_resume', True)]:
        changed = deepcopy(plan)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_plan(changed)


def test_repair_claim_is_owned_by_latest_budget_source(tmp_path):
    task = Continuation(SimpleNamespace(contract=ROOT / 'contracts/v6_s1_landing_repair_v1.json',
        run_dir=tmp_path / 'run', max_runtime_seconds=100, transfer=None, stage='S1',
        num_envs=20000, device='cpu'))
    assert task.claim.parent == Path(task.plan['budget_source_stage'])
    assert task.claim.parent != Path(task.plan['source_stage'])
    task.root.mkdir()
    task.claim = tmp_path / 'claim.json'
    task.claimed = True
    task.account('A_landing_repair', {'successful_updates': 13, 'actor_updates_in_block': 13})
    assert task.report['successful_updates'] == 751
    assert task.report['actor_updates_consumed'] == 731
    with pytest.raises(ValueError):
        task.account('A_landing_repair', {'successful_updates': 0, 'actor_updates_in_block': 0})
    with pytest.raises(FileExistsError):
        task.claim.open('x')
    command = task.train_command(task.plan['stages'][0])
    assert command[command.index('--consumed-updates') + 1] == '738'
    assert command[command.index('--actor-updates-consumed') + 1] == '718'
    assert command[command.index('--stage-actor-update') + 1] == '368'


def budget_fixture(tmp_path):
    task = object.__new__(Continuation)
    stage = tmp_path / 'budget'
    checkpoint = stage / 'A_continue/train/model_final.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'immutable latest checkpoint')
    contract = checkpoint.with_suffix('.contract.json')
    contract.write_text('{}')
    seal = dict(schema='chassis-learning-checkpoint-v1', sealed=True,
        checkpoint_sha256=continuation.digest(checkpoint), contract_sha256=continuation.digest(contract),
        learning_updates=738, actor_updates_consumed=718, learning_lineage_updates=418, stage_actor_update=408)
    checkpoint.with_suffix('.checkpoint.json').write_text(json.dumps(seal))
    ledger = dict(consumed_updates=738, actor_updates_consumed=718)
    completed = dict(status='continuation_review_required', successful_updates=738,
        actor_updates_consumed=718, inherited_recovery_attempts=[{'consumed_updates': 498}],
        protected_case_names=['stand_230mm'], arms={'A_continue': {'checkpoint_sha256': continuation.digest(checkpoint)}})
    (stage / 'completion.json').write_text(json.dumps(completed))
    (stage / 'consumption.json').write_text(json.dumps(ledger))
    hashes = {name: continuation.digest(stage / name) for name in ('completion.json', 'consumption.json')}
    task.plan = dict(budget_source_stage=str(stage), budget_source_files_sha256=hashes,
        source_consumed_updates=738, source_actor_consumed_updates=718,
        inherited_recovery_attempts=completed['inherited_recovery_attempts'], protected_case_names=['stand_230mm'])
    origin = dict(completion_status=completed['status'], consumption=ledger,
        inherited_recovery_attempts=completed['inherited_recovery_attempts'], protected_case_names=['stand_230mm'])
    return task, origin, checkpoint


def test_budget_authentication_rejects_changed_artifacts_and_wrong_seal(tmp_path):
    task, origin, checkpoint = budget_fixture(tmp_path)
    task.authenticate_budget_source(origin)
    path = checkpoint.with_suffix('.checkpoint.json')
    seal = json.loads(path.read_text())
    seal['learning_updates'] = 698
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match='spent-budget checkpoint seal'):
        task.authenticate_budget_source(origin)
    seal['learning_updates'] = 738
    path.write_text(json.dumps(seal))
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='manifest'):
        task.authenticate_budget_source(origin)


def test_budget_authentication_rejects_refund_even_if_origin_is_rewritten(tmp_path):
    task, origin, _ = budget_fixture(tmp_path)
    stage = Path(task.plan['budget_source_stage'])
    ledger = dict(consumed_updates=698, actor_updates_consumed=678)
    (stage / 'consumption.json').write_text(json.dumps(ledger))
    origin['consumption'] = ledger
    task.plan['budget_source_files_sha256']['consumption.json'] = continuation.digest(stage / 'consumption.json')
    with pytest.raises(ValueError, match='refund work'):
        task.authenticate_budget_source(origin)
