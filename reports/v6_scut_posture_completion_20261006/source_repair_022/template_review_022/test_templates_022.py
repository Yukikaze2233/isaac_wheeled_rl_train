"""CPU-only emitter review; emitted remote code is compiled, never executed."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
OUT = HERE.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deploy = load(OUT / 'deploy_script_repair_022.py', '_review022_deploy')
verifier = load(OUT / 'verify_first_PPO_022.remote.py', '_review022_verifier')
old_test = load(OUT.parent / 'first_PPO_validator_021/test_verify_first_PPO_021.py', '_review022_original_phase_fixtures')


def synthetic_inputs():
    meta = dict(previous_source_commit=deploy.PREVIOUS, contract_file='contracts/v6_task_campaign_v20.json',
                source_commit='synthetic_future_commit', archive_sha256='synthetic_archive_sha',
                source_remote='/synthetic/source022', plan_sha256='synthetic_plan_sha', manifest_sha256='synthetic_migration_sha',
                archive_remote='/synthetic/source022.tar.gz')
    evidence = dict(status='passed', qualification=None, source_commit=meta['source_commit'],
                    archive_sha256=meta['archive_sha256'], evidence_sha256={'synthetic_record': 'synthetic_record_sha'})
    revision = dict(schema='task-plan-revision-v20', source_commit=meta['source_commit'])
    predecessor = dict(schema='task-recovery-quality-predecessor-022-v1', source_commit=meta['source_commit'],
                       previous_source_commit=deploy.PREVIOUS, expected_critic_warmup=10,
                       reopening=None, qualification=None, evidence_sha256=evidence['evidence_sha256'])
    return meta, evidence, revision, predecessor


def test_verifier_import_has_no_old_inspect_or_io_launch_execution():
    tree = ast.parse((OUT / 'verify_first_PPO_022.remote.py').read_text())
    assert isinstance(tree.body[-1], ast.If)
    assert ast.unparse(tree.body[-1].test) == "__name__ == '__main__'"
    # Imports and function definitions only before the explicit main guard.
    assert all(isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or
               isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
               for n in tree.body[:-1])
    calls = {getattr(n.func, 'attr', getattr(n.func, 'id', None))
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not {'Popen', 'run', 'kill', 'killpg', 'launch', 'learn', 'backward', 'step', 'save'} & calls
    assert 'synthetic_future_commit' not in (OUT / 'verify_first_PPO_022.remote.py').read_text()


def test_dynamic_emit_both_modes_compile_without_running_remote():
    meta, evidence, revision, predecessor = synthetic_inputs()
    for mode in ('prepare', 'launch'):
        code = deploy.emit(mode, meta, evidence, revision, predecessor)
        compile(code, 'unexecuted-remote-' + mode, 'exec')
        assert "revision['schema']=='task-plan-revision-v19'" in code
        assert "old=r/'source_d0dce6181df8'" in code
        assert 'source_revision_021.json' in code
        assert 'recovery_quality_022.lock' in code
    assert deploy.SESSION.endswith('-r22')


def test_first_PPO_CLI_emits_dynamic_inputs_without_inspecting_them(tmp_path, monkeypatch, capsys):
    meta, admission, revision, predecessor = synthetic_inputs()
    predecessor['actual_parents'] = {'recovery': {'path': '/synthetic/run/train/recovery/attempt_005/train/model_final.pt'}}
    paths = {}
    for name, data in [('source', meta), ('admission', admission), ('predecessor', predecessor), ('revision', revision)]:
        paths[name] = tmp_path / (name + '.json')
        paths[name].write_text(json.dumps(data))
    monkeypatch.setattr('sys.argv', [str(OUT / 'verify_first_PPO_022.remote.py'), '--emit-remote-code',
        '--source', str(paths['source']), '--admission', str(paths['admission']),
        '--predecessor', str(paths['predecessor']), '--revision', str(paths['revision']),
        '--phase-progress-capture', '/synthetic/run/first_progress_022/index.json'])
    monkeypatch.setattr(verifier, 'inspect', lambda _: pytest.fail('Emitter executed remote inspection'))
    verifier.main()
    emitted = capsys.readouterr().out
    compile(emitted, 'unexecuted-first-PPO-dynamic-inspector', 'exec')
    assert 'synthetic_future_commit' in emitted
    assert '/synthetic/run/first_progress_022/index.json' in emitted
    assert '"mode": "v19_to_v20"' in emitted
    assert '"actual_parents": {"recovery": {"path": "/synthetic/run/' in emitted


@pytest.mark.parametrize('fault', ['wrong_previous', 'wrong_contract', 'failed_admission', 'qualified_admission',
                                  'wrong_archive', 'old_revision_schema', 'old_predecessor_schema',
                                  'wrong_predecessor_previous', 'wrong_warmup', 'reopening', 'qualified_parent', 'changed_evidence'])
def test_emitter_wrong_identity_or_admission_fails_closed(fault):
    meta, evidence, revision, predecessor = synthetic_inputs()
    if fault == 'wrong_previous': meta['previous_source_commit'] = 'old020'
    elif fault == 'wrong_contract': meta['contract_file'] = 'contracts/v6_task_campaign_v19.json'
    elif fault == 'failed_admission': evidence['status'] = 'pending'
    elif fault == 'qualified_admission': evidence['qualification'] = True
    elif fault == 'wrong_archive': evidence['archive_sha256'] = 'other'
    elif fault == 'old_revision_schema': revision['schema'] = 'task-plan-revision-v19'
    elif fault == 'old_predecessor_schema': predecessor['schema'] = 'task-recovery-quality-predecessor-021-v1'
    elif fault == 'wrong_predecessor_previous': predecessor['previous_source_commit'] = 'old020'
    elif fault == 'wrong_warmup': predecessor['expected_critic_warmup'] = 0
    elif fault == 'reopening': predecessor['reopening'] = {'automatic': True}
    elif fault == 'qualified_parent': predecessor['qualification'] = True
    elif fault == 'changed_evidence': predecessor['evidence_sha256'] = {'changed': 'digest'}
    with pytest.raises(AssertionError):
        deploy.emit('launch', meta, evidence, revision, predecessor)


def test_full_state_ten_critic_then_first_actor_native_accounting_helpers():
    parent, initial, rows = old_test.phase_fixture(10)
    proof = verifier.phase_checks(parent, initial, rows, warmup=10,
        consumption_origin=10003, actor_consumption_origin=9861)
    assert [x['phase'] for x in proof] == ['critic_only'] * 10 + ['actor_and_critic']
    bounded = verifier.bounded_phase_checks(parent, initial, rows[1], rows[11], warmup=10,
        consumption_origin=10003, actor_consumption_origin=9861)
    assert bounded['intermediate_actor_tensor_observation'] == 'unobserved_not_sealed_by_native_checkpoint500'


def test_readonly_progress_uses_real_critic_phases_and_actor11():
    parent, rows = old_test.progress_fixture(10)
    proof = verifier.progress_phase_checks(rows, parent, warmup=10, consumption_origin=10003,
        actor_consumption_origin=9861, global_actor_update_offset=44)
    assert proof['observed_completed_updates'] == list(range(1, 12))
    altered = copy.deepcopy(rows)
    altered[10]['optimizer_phase'] = 'actor_and_critic'
    with pytest.raises(ValueError, match='phase/clock'):
        verifier.progress_phase_checks(altered, parent, warmup=10, consumption_origin=10003,
            actor_consumption_origin=9861, global_actor_update_offset=44)


def test_complete_ledger_and_other_five_domains_preserved():
    fixture = old_test.ledger_fixture()
    verifier.preserve_account(*fixture)
    plan, predecessor, account, state, calculate = fixture
    account['rollback_refunds_consumed_updates'] = True
    with pytest.raises(ValueError, match='Refund'):
        verifier.preserve_account(plan, predecessor, account, state, calculate)


def test_stop_is_emit_only_and_current_owner_ids_and_attempt_are_exact():
    tree = ast.parse((OUT / 'emit_stop_previous_022.py').read_text())
    top_calls = {getattr(n.func, 'attr', getattr(n.func, 'id', None))
                 for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not {'kill', 'killpg', 'Popen', 'run'} & top_calls
    remote = next(n.value.args[0].value for n in tree.body if isinstance(n, ast.Expr) and
                  isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'print' and
                  n.value.args and isinstance(n.value.args[0], ast.Constant) and 'os.kill' in str(n.value.args[0].value))
    compile(remote, 'unexecuted-stop-remote', 'exec')
    assert "expected={3506636:'chassis_remote_job.py',3506645:'run_task_campaign.py',3506652:'train_chassis.py'}" in remote
    assert "train/recovery/attempt_005/train/progress.json" in remote
    assert "os.kill(3506645,signal.SIGTERM)" in remote
    assert "assert not (r/'stop_requested_022.json').exists()" in remote


def test_real_script_runtime_binding_is_checked_not_claimed_by_static_label():
    text = (OUT / 'verify_first_PPO_022.remote.py').read_text()
    assert "observed.get('recovery_script_revision') == native.get('recovery_script_revision')" in text
    assert "config['recovery_training'].get('side_speed_rad_s')" in text
    assert '04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6' in text
    assert "'src/wheeled_tasks/chassis/recovery_script_revision.py'" in text


def test_duplicate_launch_guards_and_observer_new_worker_scope_preserved():
    assert "assert not (r/'recovery_quality_launch_022.json').exists() and not (r/'job.resume_022.log').exists()" in deploy.LAUNCH
    assert "fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)" in deploy.LAUNCH
    assert "assert not owners()" in deploy.COMMON
    observer = (OUT / 'observe_first_progress_022.py').read_text()
    assert "assert len(matches)<=1,'Duplicate source022 worker invocations'" in observer
    assert "str(a.source_root/'scripts/train_chassis.py') in cmd" in observer
    assert 'source022-native-first-progress-copies-v1' in observer
    assert not {'kill', 'Popen', 'learn'} & {getattr(n.func, 'attr', getattr(n.func, 'id', None))
        for n in ast.walk(ast.parse(observer)) if isinstance(n, ast.Call)}
