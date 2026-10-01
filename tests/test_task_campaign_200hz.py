"""Prevent silent control-clock changes, budget rebasing and repeated supervisor I/O."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.control_contract import load_chassis_control, validate_chassis_control
from wheeled_tasks.chassis.preflight import SubstepControlAudit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_task_campaign as runner

def read(name):
    return json.loads((ROOT / name).read_text())

def test_named_timing_revision_preserves_all_domain_caps_and_rejects_other_changes():
    old, new = read('contracts/v6_task_campaign_v2.json'), read('contracts/v6_task_campaign_v3.json')
    receipt = campaign.validate_plan_migration(old, new, new['revision_migration']['run_directory'])
    assert receipt['consumption'] == 'preserved_no_refund'
    assert len(runner.validate_contracts(new)) == 30
    for before, after in zip(old['domains'], new['domains'], strict=True):
        source, target = read(before['contract']), read(after['contract'])
        assert campaign.validate_task_resume(source, target)['task_clock'] == 'restored'
        for key in ('dense_tracking', 'v5_control', 'actuator_response', 'command_transport',
                    'skill_specs', 'scene_groups', 'position_iterations', 'velocity_iterations'):
            assert source[key] == target[key]
        for key, value in [('physics_dt', .002), ('learning_rate', 2e-4), ('position_iterations', 12),
                           ('policy_dt', .005), ('checkpoint_interval', 100)]:
            bad = deepcopy(target); bad[key] = value
            with pytest.raises(ValueError):
                campaign.validate_task_resume(source, bad)

def test_coarse_control_cannot_claim_hardware_or_five_fresh_feedbacks():
    original = load_chassis_control(ROOT / 'contracts/v6_wheel_response_control_v1.json')
    bad = deepcopy(original); bad['timing'].update(physics_dt=.005, pc_control_dt=.005, decimation=4)
    with pytest.raises(ValueError):
        validate_chassis_control(bad)
    coarse = load_chassis_control(ROOT / 'contracts/v6_wheel_response_control_200hz_v1.json')
    for key, value in [('hardware_equivalence_verified', True), ('fresh_feedback_per_policy_step', 20),
                       ('hardware_pd_dt', .005)]:
        bad = deepcopy(coarse); bad['simulation_timing_approximation'][key] = value
        with pytest.raises(ValueError):
            validate_chassis_control(bad)
    assert original['actuators'] == coarse['actuators']

def test_real_execution_audit_counts_four_new_feedback_pd_steps():
    audit = SubstepControlAudit(.005, .02)
    audit.begin_policy(); audit.targets_decoded()
    for _ in range(4):
        audit.feedback_control(); audit.physics_updated(.005)
    actual = audit.end_policy()
    assert actual['pd_updates'] == actual['feedback_refreshes'] == 4
    assert actual['target_decodes'] == 1
    audit.begin_policy(); audit.targets_decoded(); audit.feedback_control()
    with pytest.raises(RuntimeError):
        audit.feedback_control()

def test_supervisor_skips_unchanged_receipts_but_copies_new_worker_and_atomic_replacements(tmp_path):
    job = object.__new__(runner.TaskCampaign)
    job.root = tmp_path / 'campaign'; job.root.mkdir()
    job.plan = {'contract_id': campaign.REVISION_V3}; job.active = {'name': 'flat'}
    job.report = {'consumed_updates': 1010, 'actor_updates_consumed': 990}
    accounts, copies = [], []
    job.account = lambda directory, record: accounts.append(record.copy())
    job.copy_atomic = lambda source, name: copies.append((str(source), name))
    directory = tmp_path / 'worker'; directory.mkdir()
    runner.write_json(directory / 'progress.json', {'pid': 1, 'successful_updates': 10})
    runner.write_json(directory / 'startup.json', {'startup': True})
    assert job.publish(directory) and job.publish(directory)
    assert len(accounts) == len(copies) == 1
    runner.write_json(directory / 'progress.json', {'pid': 1, 'successful_updates': 11})
    job.publish(directory)
    assert len(accounts) == 2 and len(copies) == 1
    second = tmp_path / 'new_worker'; second.mkdir()
    runner.write_json(second / 'progress.json', {'pid': 2, 'successful_updates': 1})
    runner.write_json(second / 'startup.json', {'startup': True})
    job.publish(second)
    assert len(accounts) == 3 and len(copies) == 2
