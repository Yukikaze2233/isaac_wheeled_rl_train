"""CPU-only synthetic protocol checks; no remote code, launch, or native runs."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('deploy020', HERE / 'deploy_source_repair_020.py')
emitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emitter)
clean = Path(json.loads((HERE / 'source_candidate_020.json').read_text())['clean_root'])
sys.path[:0] = [str(clean / 'src'), str(clean / 'scripts')]
from wheeled_tasks.chassis.task_campaign import account_updates


def functions_from_common(*names):
    tree = ast.parse(emitter.COMMON)
    return ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def protocol(tmp_path):
    root = tmp_path / 'run'
    old, new = root / 'source_712d701473f3', root / 'source_7f315f30741b'
    plan = json.loads((clean / 'contracts/v6_task_campaign_v18.json').read_text())
    worker_row = next(row for row in plan['domains'] if row['name'] == 'recovery')
    worker = json.loads((clean / worker_row['contract']).read_text())
    for directory in (old, new):
        put(directory / 'contracts/v6_task_campaign_v18.json', plan)
        put(directory / worker_row['contract'], worker)
        (directory / worker_row['contract']).write_bytes((clean / worker_row['contract']).read_bytes())
    plan_sha, worker_sha = emitter.sha(new / 'contracts/v6_task_campaign_v18.json'), emitter.sha(new / worker_row['contract'])
    previous = root / 'job.previous_015.json'
    put(previous, {'old_owner': 'synthetic_only'})
    archive = Path(str(old) + '.tar.gz')
    with tarfile.open(archive, 'w:gz', format=tarfile.PAX_FORMAT, pax_headers={'comment': emitter.PREVIOUS}) as stream:
        data = b'synthetic CPU fixture only'
        member = tarfile.TarInfo('fixture.txt'); member.size = len(data)
        stream.addfile(member, io.BytesIO(data))
    command = ['/synthetic/python', '-B', str(old / 'scripts/run_task_campaign.py'), '--contract', str(old / 'contracts/v6_task_campaign_v18.json'), '--num-envs', '4096', '--device', 'cuda:0', '--run-dir', str(root / 'train')]
    oldrevision = dict(schema='task-source-revision-v1', source_commit=emitter.PREVIOUS,
        new_source_root=str(old), old_contract_sha256=plan_sha, new_contract_sha256=plan_sha,
        command=command, source_archive=str(archive), source_archive_sha256=emitter.sha(archive),
        previous_job_sha256=emitter.sha(previous))
    put(root / 'source_revision_019.json', oldrevision)
    put(root / 'job.source_revision_015.json', oldrevision)
    job = dict(pid=999999999, command=command, previous_job=previous.name, source_revision_receipt='job.source_revision_015.json')
    put(root / 'job.json', job)
    name = 'recovery/attempt_003/train'
    attempts = {name: dict(domain='recovery', consumed_updates=2500, actor_updates=2500)}
    calculated = account_updates(plan, attempts)
    account = dict(contract_sha256=plan_sha, rollback_refunds_consumed_updates=False,
        run_directory=str(root / 'train'), attempts=attempts, **calculated)
    state = dict(status='stopped', contract_sha256=plan_sha, **calculated)
    put(root / 'train/consumption.json', account); put(root / 'train/campaign_state.json', state)
    directory = root / 'train' / name
    progress = dict(successful_updates=2500, attempted_updates=2500, actor_updates_in_block=2500, task_actor_update=2469)
    put(directory / 'progress.json', progress)
    put(directory / 'completion.json', dict(status='stopped', **progress))
    seal = dict(task_actor_update=2469, stage_actor_update=15937,
        actor_updates_consumed=calculated['actor_updates_consumed'], learning_updates=calculated['consumed_updates'],
        learning_lineage_updates=3530)
    infos = dict(**{k:v for k,v in seal.items() if k != 'learning_updates'},
        successful_updates_total=seal['learning_updates'], contract_sha256=worker_sha,
        task_campaign={'domain':'recovery'}, critic_warmup_remaining=0,
        rng_state=torch.get_rng_state(), env_rng_state=torch.get_rng_state(), cuda_rng_states=[torch.get_rng_state()])
    payload = dict(actor_state_dict={'weight':torch.ones(1)}, critic_state_dict={'weight':torch.ones(1)},
        optimizer_state_dict={'state':{0:{'step':torch.tensor(1.)}},'param_groups':[]}, infos=infos)
    cp = directory / 'model_final.pt'; torch.save(payload, cp)
    bound = {'recovery':dict(path=str(cp), seal=seal, contract_sha256=worker_sha)}
    namespace = dict(pathlib=__import__('pathlib'), RUN=str(root), r=root, src=new,
        META=dict(previous_source_commit=emitter.PREVIOUS, plan_sha256=plan_sha, worker_hashes={'recovery':worker_sha}),
        read=emitter.read, sha=emitter.sha, owners=lambda:[], account_updates=account_updates,
        parents=lambda _:deepcopy(bound))
    exec(compile(functions_from_common('authenticate_quiescent'), '<synthetic-authentication-020>', 'exec'), namespace)
    return dict(namespace=namespace, root=root, old=old, new=new, cp=cp, payload=payload, bound=bound)


def test_authenticates_full_latest_019_same_contract_without_warmup(protocol):
    result = protocol['namespace']['authenticate_quiescent']()
    assert result[3]['recovery']['authenticated_infos']['critic_warmup_remaining'] == 0
    assert result[4] == protocol['old']


@pytest.mark.parametrize('fault', ['owners', 'wrong_job_command', 'revision_copy', 'old_archive',
    'contract_inventory', 'state_status', 'refund', 'account_clock', 'uncharged_progress',
    'old_parent_attempt', 'actor_state', 'critic_state', 'adam_state', 'cpu_rng', 'env_rng',
    'cuda_rng', 'infos_clock', 'warmup'])
def test_refuses_inconsistent_or_incomplete_resume(protocol, fault):
    root, ns = protocol['root'], protocol['namespace']
    if fault == 'owners': ns['owners'] = lambda:[{'pid':123,'command':['synthetic-wrapper']}]
    elif fault == 'wrong_job_command':
        item=emitter.read(root/'job.json'); item['command'][2]='/wrong/source/runner.py'; put(root/'job.json',item)
    elif fault == 'revision_copy': put(root/'job.source_revision_015.json', {'schema':'wrong'})
    elif fault == 'old_archive': Path(str(protocol['old'])+'.tar.gz').write_bytes(b'corrupt')
    elif fault == 'contract_inventory': put(protocol['new']/'contracts/extra.json',{})
    elif fault == 'state_status':
        item=emitter.read(root/'train/campaign_state.json');item['status']='running';put(root/'train/campaign_state.json',item)
    elif fault == 'refund':
        item=emitter.read(root/'train/consumption.json');item['rollback_refunds_consumed_updates']=True;put(root/'train/consumption.json',item)
    elif fault == 'account_clock':
        item=emitter.read(root/'train/consumption.json');item['consumed_updates']+=1;put(root/'train/consumption.json',item)
    elif fault == 'uncharged_progress':
        path=protocol['cp'].parent/'progress.json'; item=emitter.read(path);item['attempted_updates']+=1;put(path,item)
    elif fault == 'old_parent_attempt':
        item=deepcopy(protocol['bound']);item['recovery']['path']=str(root/'train/recovery/attempt_002/train/model_final.pt');ns['parents']=lambda _:item
    else:
        payload=protocol['payload']
        if fault == 'actor_state': payload['actor_state_dict']={}
        elif fault == 'critic_state': payload['critic_state_dict']={}
        elif fault == 'adam_state': payload['optimizer_state_dict']['state']={}
        elif fault == 'cpu_rng': del payload['infos']['rng_state']
        elif fault == 'env_rng': del payload['infos']['env_rng_state']
        elif fault == 'cuda_rng': payload['infos']['cuda_rng_states']=[]
        elif fault == 'infos_clock': payload['infos']['learning_lineage_updates']+=1
        elif fault == 'warmup': payload['infos']['critic_warmup_remaining']=10
        torch.save(payload,protocol['cp'])
    with pytest.raises((AssertionError,ValueError,KeyError)):
        ns['authenticate_quiescent']()


def test_latest_selects_immutable_final_at_latest_completed_learning_clock(tmp_path):
    namespace=dict(r=tmp_path, read=emitter.read)
    exec(compile(functions_from_common('latest'), '<synthetic-latest-020>','exec'),namespace)
    for relative, count in [('recovery/attempt_002/train/model_final.checkpoint.json',10),
        ('recovery/attempt_003/train/model_11.checkpoint.json',11),
        ('recovery/attempt_003/train/model_final.checkpoint.json',11),
        ('recovery/attempt_003/train/export.pending/model_99.checkpoint.json',99)]:
        put(tmp_path/'train'/relative,{'learning_updates':count})
    assert namespace['latest']('recovery') == tmp_path/'train/recovery/attempt_003/train/model_final.pt'


def test_all_emit_modes_compile_and_never_execute():
    evidence=dict(source=dict(source_commit=emitter.COMMIT,previous_source_commit=emitter.PREVIOUS,
        source_remote=emitter.RUN+'/source_7f315f30741b',archive_remote=emitter.RUN+'/source_7f315f30741b.tar.gz',
        archive_sha256='a'*64,plan_sha256='b'*64),evidence_sha256={'all':'c'*64})
    revision=dict(schema='task-source-revision-v1',source_commit=emitter.COMMIT,
        new_source_root=evidence['source']['source_remote'],old_source_root=emitter.RUN+'/source_712d701473f3',
        source_archive=evidence['source']['archive_remote'],source_archive_sha256='a'*64,
        new_contract_sha256='b'*64,old_contract_sha256='b'*64)
    predecessor=dict(source_commit=emitter.COMMIT,evidence_sha256=evidence['evidence_sha256'])
    for mode in ('prepare','launch','readback'):
        code=emitter.remote_code(mode,evidence,revision,predecessor)
        compile(code,'<not-executed-020-'+mode+'>','exec')
    assert emitter.SESSION.endswith('-r20')
    assert 'source_repair_020.lock' in emitter.PREPARE and 'source_repair_020.lock' in emitter.LAUNCH
    assert 'LOCK_EX|fcntl.LOCK_NB' in emitter.PREPARE and 'LOCK_EX|fcntl.LOCK_NB' in emitter.LAUNCH
    assert "with path.open('x')" in emitter.LAUNCH
    assert emitter.LAUNCH.index("with path.open('x')") < emitter.LAUNCH.index("'new-session'")
    assert "'source_revision_019.json'" in emitter.COMMON
    assert "'attempt_003" not in emitter.COMMON  # path is intentionally run-root anchored
    assert 'attempt_003/train' in emitter.COMMON
    assert all(token not in emitter.READBACK for token in ['subprocess.run','open(\'x\')','flock(','new-session'])


def test_incomplete_primary_native_evidence_is_rejected_without_remote_emit(tmp_path):
    paths={}
    meta=emitter.read(HERE/'source_candidate_020.json')
    payloads=dict(source_candidate=meta,clean_tests=emitter.read(HERE/'clean_tests_020.json'),
        clean_30_roles=emitter.read(HERE/'clean_30_roles_020.json'),
        native_domains=dict(status='in_progress',source_commit=emitter.COMMIT,archive_sha256=meta['archive_sha256'],formal_updates_consumed=0,qualification=None,domains={}))
    for key,value in payloads.items():
        path=tmp_path/(key+'.json');put(path,value);paths[key]=path
    with pytest.raises(AssertionError,match='Incomplete primary frozen020 evidence'):
        emitter.load_evidence(paths)
