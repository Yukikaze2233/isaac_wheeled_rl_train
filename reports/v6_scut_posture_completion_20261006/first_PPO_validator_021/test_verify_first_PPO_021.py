"""CPU fixtures for the read-only inspector, never real run evidence."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('first_ppo021', HERE / 'verify_first_PPO_021.py')
MOD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)
SOURCE = Path('/tmp/scut-recovery-frozen-021-qv1ukmbr')
BASE = Path('/tmp/scut-com-frozen-020-w14p1d1x')


def phase_fixture(remaining=10):
    parent = dict(actor_state_dict={('std' if i == 8 else 'net.' + str(i)): torch.tensor([float(i)]) for i in range(9)},
                  critic_state_dict={'value': torch.tensor([0.])},
                  optimizer_state_dict={'state': {9: {'step': torch.tensor(2.), 'exp_avg': torch.tensor(.1)}}, 'param_groups': [{'lr': .0001}]},
                  infos=dict(successful_updates_total=9876, actor_updates_consumed=9800,
                             task_actor_update=3001, stage_actor_update=18000,
                             learning_lineage_updates=18099, critic_warmup_remaining=0))
    initial = deepcopy(parent)
    initial['infos'].update(successful_updates_total=10003, actor_updates_consumed=9861, critic_warmup_remaining=remaining)
    rows = {}
    for i in range(1, remaining + 2):
        saved = deepcopy(initial)
        actors = int(i > remaining)
        saved['infos'].update(successful_updates_total=10003 + i, actor_updates_consumed=9861 + actors,
                              task_actor_update=3001 + actors, stage_actor_update=18000 + actors,
                              learning_lineage_updates=18099 + i, critic_warmup_remaining=max(0, remaining - i))
        saved['critic_state_dict']['value'] += i
        saved['optimizer_state_dict']['state'][9]['step'] += i
        if actors:
            for value in saved['actor_state_dict'].values():
                value += .1
        rows[i] = saved
    return parent, initial, rows


def check_fixture(parent, initial, rows, remaining):
    return MOD.phase_checks(parent, initial, rows, warmup=remaining,
                            consumption_origin=10003, actor_consumption_origin=9861)


@pytest.mark.parametrize('remaining', [0, 1, 4, 9, 10])
def test_every_charged_critic_seal_and_first_actor_with_separate_clocks(remaining):
    parent, initial, rows = phase_fixture(remaining)
    result = check_fixture(parent, initial, rows, remaining)
    assert len(result) == remaining + 1
    assert result[-1]['clocks']['successful_updates_total'] == 10003 + remaining + 1
    assert len(result[-1]['actor_changed']) == 9 and 'std' in result[-1]['actor_changed']
    assert all(r['clocks']['task_actor_update'] == 3001 and r['clocks']['actor_updates_consumed'] == 9861 for r in result[:-1])


@pytest.mark.parametrize('index', list(range(1, 11)))
def test_actor_or_std_moving_in_any_of_ten_critic_updates_rejected(index):
    parent, initial, rows = phase_fixture()
    rows[index]['actor_state_dict']['std'] += .01
    with pytest.raises(ValueError, match='Actor/std moved'):
        check_fixture(parent, initial, rows, 10)


@pytest.mark.parametrize('fault', ['initial_Adam', 'critic_frozen', 'Adam_frozen', 'actor_one_frozen', 'actor_std_frozen',
                                  'actor_consumption_advanced_early', 'remaining_restarted', 'missing_update_10', 'missing_first_actor'])
def test_full_state_and_phase_failures_are_not_silent(fault):
    parent, initial, rows = phase_fixture()
    if fault == 'initial_Adam': initial['optimizer_state_dict']['state'][9]['step'] += 1
    elif fault == 'critic_frozen': rows[6]['critic_state_dict'] = deepcopy(rows[5]['critic_state_dict'])
    elif fault == 'Adam_frozen': rows[6]['optimizer_state_dict'] = deepcopy(rows[5]['optimizer_state_dict'])
    elif fault == 'actor_one_frozen': rows[11]['actor_state_dict']['net.2'] = deepcopy(initial['actor_state_dict']['net.2'])
    elif fault == 'actor_std_frozen': rows[11]['actor_state_dict']['std'] = deepcopy(initial['actor_state_dict']['std'])
    elif fault == 'actor_consumption_advanced_early': rows[10]['infos']['actor_updates_consumed'] += 1
    elif fault == 'remaining_restarted': rows[5]['infos']['critic_warmup_remaining'] = 10
    elif fault == 'missing_update_10': rows.pop(10)
    elif fault == 'missing_first_actor': rows.pop(11)
    with pytest.raises(ValueError): check_fixture(parent, initial, rows, 10)


def test_native_resume_function_is_original020_ast():
    def function(path):
        return next(n for n in ast.parse((path / 'scripts/train_chassis.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'resume_critic_warmup')
    assert ast.dump(function(BASE), include_attributes=False) == ast.dump(function(SOURCE), include_attributes=False)


@pytest.mark.parametrize('remaining', [0, 1, 6, 10])
def test_same_v19_uses_saved_remaining_instead_of_restarting10(remaining):
    config = MOD.read(SOURCE / 'contracts/v6_task_campaign_recovery_worker_v19.json')
    config_sha = MOD.sha(SOURCE / 'contracts/v6_task_campaign_recovery_worker_v19.json')
    infos = dict(contract_sha256=config_sha, critic_warmup_remaining=remaining, successful_updates_total=32000)
    assert MOD.native_warmup(SOURCE)(config, infos, contract_sha256=config_sha, task_change=True) == remaining


def test_v18_to_v19_uses_inherited10_even_when_parent_remaining0():
    config = MOD.read(SOURCE / 'contracts/v6_task_campaign_recovery_worker_v19.json')
    infos = dict(contract_sha256=MOD.sha(SOURCE / 'contracts/v6_task_campaign_recovery_worker_v18.json'),
                 critic_warmup_remaining=0, successful_updates_total=32000)
    assert config['resume_critic_warmup_updates'] == 10
    assert MOD.native_warmup(SOURCE)(config, infos, contract_sha256=MOD.sha(SOURCE / 'contracts/v6_task_campaign_recovery_worker_v19.json'), task_change=True) == 10


def ledger_fixture():
    domains = ['flat', 'recovery', 'speed', 'terrain', 'jump', 'stepjump']
    plan = dict(domains=[dict(name=n, updates=20000) for n in domains], historical_consumed_updates=1000,
                historical_actor_consumed_updates=980, total_ceiling=121000)
    attempts = {'flat/attempt_000/train': dict(domain='flat', consumed_updates=100, actor_updates=90),
                'recovery/attempt_004/train': dict(domain='recovery', consumed_updates=200, actor_updates=190)}
    # Use original frozen accounting function as the only arithmetic source.
    tree = ast.parse((SOURCE / 'src/wheeled_tasks/chassis/task_campaign.py').read_text())
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'account_updates')
    scope = {}; exec(compile(ast.Module(body=[f], type_ignores=[]), 'account_updates', 'exec'), scope)
    calculate = scope['account_updates']
    before = dict(attempts=attempts, rollback_refunds_consumed_updates=False, **calculate(plan, attempts))
    old_state = dict(domains={n: dict(domain_qualified=None, completion_history=[{'old': n}]) for n in domains})
    predecessor = dict(consumption=deepcopy(before), campaign_state=deepcopy(old_state))
    account = deepcopy(before); account['attempts']['recovery/attempt_005/train'] = dict(domain='recovery', consumed_updates=11, actor_updates=1)
    account.update(calculate(plan, account['attempts']))
    return plan, predecessor, account, deepcopy(old_state), calculate


def test_original_full_ledger_calculations_and_other_five_states_preserved():
    MOD.preserve_account(*ledger_fixture())


@pytest.mark.parametrize('fault', ['refund', 'old_charge', 'other_domain_charge', 'other_state', 'new_reopening', 'qualified'])
def test_ledger_refund_reopening_and_other_domain_mutations_rejected(fault):
    plan, predecessor, account, state, calculate = ledger_fixture()
    if fault == 'refund': account['rollback_refunds_consumed_updates'] = True
    elif fault == 'old_charge': account['attempts']['recovery/attempt_004/train']['consumed_updates'] -= 1; account.update(calculate(plan, account['attempts']))
    elif fault == 'other_domain_charge': account['attempts']['speed/attempt_007/train'] = dict(domain='speed', consumed_updates=1, actor_updates=1); account.update(calculate(plan, account['attempts']))
    elif fault == 'other_state': state['domains']['terrain']['completion_history'] = []
    elif fault == 'new_reopening': state['domains']['recovery']['reopening'] = {'origin_actor_update': 0}
    elif fault == 'qualified': state['domains']['recovery']['domain_qualified'] = True
    with pytest.raises(ValueError): MOD.preserve_account(plan, predecessor, account, state, calculate)


def test_python_entry_ignores_embedded_wrapper_argv_and_code_strings():
    assert MOD.process_entry([b'python', b'-B', b'/root/chassis_remote_job.py', b'--', b'python', b'/root/train_chassis.py']).name == 'chassis_remote_job.py'
    assert MOD.process_entry([b'python', b'-c', b'/root/train_chassis.py']) is None


def test_inspector_no_connection_launch_optimizer_or_process_mutation_api():
    tree = ast.parse((HERE / 'verify_first_PPO_021.py').read_text())
    calls = {getattr(n.func, 'attr', getattr(n.func, 'id', None)) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not {'system', 'Popen', 'run', 'kill', 'killpg', 'signal', 'launch', 'learn', 'backward', 'step', 'save', 'write_text', 'write_bytes', 'mkdir', 'unlink'} & calls
    assert 'subprocess' not in (HERE / 'verify_first_PPO_021.py').read_text()
    compile(tree, str(HERE / 'verify_first_PPO_021.py'), 'exec')


def test_original_020_and_018_sources_are_declared_and_no_fixed_historical_clock():
    text = (HERE / 'verify_first_PPO_021.py').read_text()
    assert 'verify_first_PPO_018.py' in text and 'verify_first_PPO_020.py' in text
    assert '29679' not in text and '2758' not in text and 'attempt_004' not in text
    assert "mode == 'v18_to_v19'" in text and "'same_v19'" in text


@pytest.mark.parametrize('remaining', [0, 1, 4, 10])
def test_actual_native500_bounded_seals_never_claim_intermediate_actor_tensors(remaining):
    parent, initial, rows = phase_fixture(remaining)
    proof = MOD.bounded_phase_checks(parent, initial, rows.get(1) if remaining else None, rows[remaining+1],
        warmup=remaining, consumption_origin=10003, actor_consumption_origin=9861)
    assert proof['first_actor_index'] == remaining + 1
    assert proof['critic_tensor_observed_indices'] == ([1] if remaining else [])
    assert proof['intermediate_actor_tensor_observation'] == 'unobserved_not_sealed_by_native_checkpoint500'


@pytest.mark.parametrize('fault', ['initial_Adam','critic_actor','critic_Adam','actor_std','actor_clock'])
def test_bounded_real_seal_faults_rejected(fault):
    parent, initial, rows = phase_fixture()
    if fault == 'initial_Adam': initial['optimizer_state_dict']['state'][9]['step'] += 1
    elif fault == 'critic_actor': rows[1]['actor_state_dict']['std'] += .1
    elif fault == 'critic_Adam': rows[1]['optimizer_state_dict'] = deepcopy(initial['optimizer_state_dict'])
    elif fault == 'actor_std': rows[11]['actor_state_dict']['std'] = deepcopy(initial['actor_state_dict']['std'])
    elif fault == 'actor_clock': rows[11]['infos']['actor_updates_consumed'] -= 1
    with pytest.raises(ValueError):
        MOD.bounded_phase_checks(parent, initial, rows[1], rows[11], warmup=10,
            consumption_origin=10003, actor_consumption_origin=9861)


def progress_fixture(remaining=10):
    parent, _, _ = phase_fixture(remaining)
    return parent, {i: dict(successful_updates=i, parent_updates=10003,
        actor_updates_in_block=int(i > remaining), global_budget_actor_updates=44+9861+int(i>remaining),
        task_actor_update=3001+int(i>remaining), stage_actor_update=18000+int(i>remaining),
        optimizer_phase='actor_and_critic' if i>remaining else 'critic_warmup') for i in range(1,remaining+2)}


@pytest.mark.parametrize('remaining', [0, 1, 6, 10])
def test_direct_native_progress_actual_schema_and_remaining(remaining):
    parent, rows = progress_fixture(remaining)
    assert all('actor_updates_consumed' not in row for row in rows.values())
    proof=MOD.progress_phase_checks(rows,parent,warmup=remaining,consumption_origin=10003,
        actor_consumption_origin=9861,global_actor_update_offset=44)
    assert proof['observed_completed_updates']==list(range(1,remaining+2))
    assert 'not_TensorBoard' in proof['source']


@pytest.mark.parametrize('fault', ['phase','actor_clock','task_clock','parent_clock','missing'])
def test_native_progress_fault_or_missing_coverage(fault):
    parent,rows=progress_fixture()
    if fault=='phase':rows[10]['optimizer_phase']='actor_and_critic'
    elif fault=='actor_clock':rows[5]['global_budget_actor_updates']+=1
    elif fault=='task_clock':rows[3]['task_actor_update']+=1
    elif fault=='parent_clock':rows[8]['parent_updates']-=1
    elif fault=='missing':rows.pop(7)
    with pytest.raises(FileNotFoundError if fault=='missing' else ValueError):
        MOD.progress_phase_checks(rows,parent,warmup=10,consumption_origin=10003,
            actor_consumption_origin=9861,global_actor_update_offset=44)


def event_fixture(tmp_path, *, fault=None, warmup=10):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
    data=bytearray()
    for index in range(warmup+1):
        values={'Timing/iteration_wall_seconds':15.0,
            'Episode//ordinary_posture/relative_actor_updates':2901.,
            'Episode//recovery/lesson/relative_actor_updates':707.}
        if fault=='advance' and index==5:values['Episode//recovery/lesson/relative_actor_updates']+=1
        if fault=='negative' and index==0:values['Timing/iteration_wall_seconds']=0.
        if fault=='nonfinite' and index==0:values['Timing/iteration_wall_seconds']=float('nan')
        for tag,value in values.items():
            event=Event(step=10003+index,wall_time=1.)
            event.summary.value.add(tag=tag,simple_value=value)
            payload=event.SerializeToString();length=struct.pack('<Q',len(payload))
            data.extend(length+struct.pack('<I',masked_crc32c(length))+payload+struct.pack('<I',masked_crc32c(payload)))
    if fault=='badcrc':data[8:12]=b'BAD!'
    path=tmp_path/'events.out.tfevents.fixture';path.write_bytes(data)
    return path


@pytest.mark.parametrize('warmup', [0, 1, 10])
def test_CRC_prefix_actual_native_tags_no_optimizer_phase_claim(tmp_path,warmup):
    path=event_fixture(tmp_path,warmup=warmup)
    proof=MOD.crc_event_prefix(tmp_path,origin=10003,warmup=warmup,task_origin=3001,
        ordinary_origin=100,prefix_origin=2294,max_bytes=path.stat().st_size)
    assert len(proof['optimizer_iteration_steps'])==warmup+1
    assert 'no_TensorBoard_optimizer_phase_tag' in proof['phase_source']
    assert proof['first_actor_tensor_update_source']=='separate_immutable_checkpoint_not_event_derived'


@pytest.mark.parametrize('fault', ['badcrc','partial','advance','negative','nonfinite'])
def test_CRC_prefix_invalid_or_unpublished_record_is_not_evidence(tmp_path,fault):
    path=event_fixture(tmp_path,fault=fault)
    limit=path.stat().st_size//2 if fault=='partial' else path.stat().st_size
    with pytest.raises(FileNotFoundError if fault in ('badcrc','partial') else ValueError):
        MOD.crc_event_prefix(tmp_path,origin=10003,warmup=10,task_origin=3001,
            ordinary_origin=100,prefix_origin=2294,max_bytes=limit)


@pytest.mark.parametrize('fault', [None,'byte_tamper','duplicate','source','missing'])
def test_progress_copies_SHA_identity_no_synthetic_phases(tmp_path,fault):
    parent,rows=progress_fixture();directory=tmp_path/'native_worker'
    index=dict(schema='source021-native-first-progress-copies-v1',status='complete',source_root=str(SOURCE),
        worker=str(directory),rows=[],qualification=None,PPO_actions=0,read_only=True)
    for i,value in rows.items():
        path=tmp_path/f'update_{i:02d}.json';path.write_text(json.dumps(value))
        index['rows'].append(dict(path=str(path),sha256=MOD.sha(path),successful_updates=i,
            captured_at_utc='2026-10-06T01:00:00+00:00',source_progress=str(directory/'progress.json')))
    if fault=='byte_tamper':(tmp_path/'update_03.json').write_text('{}')
    elif fault=='duplicate':index['rows'].append(index['rows'][0])
    elif fault=='source':index['source_root']='/tmp/unauthenticated'
    elif fault=='missing':index['rows'].pop(5)
    args=dict(source=SOURCE,directory=directory,parent=parent,warmup=10,consumption_origin=10003,
        actor_consumption_origin=9861,global_actor_update_offset=44)
    if fault:
        with pytest.raises(FileNotFoundError if fault=='missing' else ValueError):MOD.progress_capture(index,**args)
    else:assert len(MOD.progress_capture(index,**args)['observed_completed_updates'])==11


def test_original_actor_freeze_update_save_paths_remain_source020_AST_exact():
    def functions(source,name):
        return [n for n in ast.walk(ast.parse((source/'scripts/train_chassis.py').read_text()))
            if isinstance(n,ast.FunctionDef) and n.name==name]
    for name in ('update','save'):
        old,new=functions(BASE,name),functions(SOURCE,name)
        assert len(old)==len(new)==1
        assert ast.dump(old[0],include_attributes=False)==ast.dump(new[0],include_attributes=False)


def test_dynamic_cli_emit_compiles_without_connection_or_inspection(tmp_path, monkeypatch, capsys):
    source=dict(source_commit='a'*40,archive_sha256='b'*64,changed_runtime_files={})
    parent=dict(source_commit='a'*40,evidence_sha256='c'*64,
        actual_parents={'recovery':dict(path='/run/example/train/recovery/attempt_123/train/model_final.pt')})
    revision=dict(source_commit='a'*40)
    admission=dict(source_commit='a'*40,archive_sha256='b'*64,evidence_sha256='c'*64,
        status='passed',qualification=None)
    argv=['verify_first_PPO_021.py','--emit-remote-code']
    for name,value in [('source',source),('predecessor',parent),('revision',revision),('admission',admission)]:
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));argv.extend(['--'+name,str(path)])
    argv.extend(['--phase-progress-capture','/run/example/first_progress_021/index.json'])
    monkeypatch.setattr('sys.argv',argv);monkeypatch.setattr(MOD,'inspect',lambda *_:pytest.fail('Emitter must not inspect or connect'))
    MOD.main();code=capsys.readouterr().out;compile(code,'<readonly-emitted-inspector>','exec')
    assert '"run": "/run/example"' in code and 'first_progress_021/index.json' in code
