"""CPU checks for sealed periodic endpoints; no simulator or learning runner."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('periodic_mean_driver', ROOT / 'scripts/run_rec_speed_mean_replay.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
r = d.prep
GOLDEN = ROOT.parent / 'serial-dispatch-source028-20261009'
MODELS = ROOT.parent.parent / 'isaac_wheeled_rl_schedule/models'
PARENT = MODELS / 'v6_recovery_candidate_41393_20261009'
CHILDREN = {42000: MODELS / 'v6_speed_periodic_42000_actor597_20261009',
            43500: MODELS / 'v6_speed_periodic_43500_actor2097_20261009'}


@pytest.fixture(scope='module')
def sealed_endpoints():
    paths = (PARENT, *CHILDREN.values())
    if not GOLDEN.is_dir() or not all((p / name).is_file() for p in paths for name in ('model.pt', 'policy.onnx')):
        pytest.skip('Frozen e5 and original sealed REC41393/speed42000/speed43500 exports are required')
    root = r.frozen_runtime(GOLDEN)
    parent = r.authenticate_candidate(root, PARENT / 'model.pt', PARENT / 'policy.onnx')
    children = {total: r.authenticate_candidate(root, p / 'model.pt', p / 'policy.onnx')
                for total, p in CHILDREN.items()}
    return root, parent, children


@pytest.mark.parametrize('total,actor,consumed,stage,lineage', (
    (42000, 597, 41820, 28153, 28253), (43500, 2097, 43320, 29653, 29753)))
def test_original_full_sealed_exports_authenticate(total, actor, consumed, stage, lineage, sealed_endpoints):
    root, parent, children = sealed_endpoints
    child = children[total]
    r.shared_abi([parent[0], child[0]])
    r.validate_pair([parent[1], child[1]])
    selected = r.endpoint_selection(root, child[1]['consumed_updates'])
    assert selected['selected_endpoint'] == dict(task_actor_update=actor, successful_updates_total=total,
        actor_updates_consumed=consumed, stage_actor_update=stage,
        learning_lineage_updates=lineage, critic_warmup_remaining=0)
    assert selected['first_periodic_endpoint']['successful_updates_total'] == 42000
    assert child[1]['original_full_role'] and child[1]['native_mean_max_abs_error'] <= 1e-5
    assert selected['new_checkpoint_request'] is False and selected['physical_comparison_complete'] is False


@pytest.mark.parametrize('key', ('task_actor_update', 'stage_actor_update', 'learning_lineage_updates',
                                'actor_updates_consumed', 'critic_warmup_remaining'))
def test_one_clock_disagreement_rejects_sealed_child(key, sealed_endpoints):
    _, parent, children = sealed_endpoints
    wrong = deepcopy(children[43500][1])
    wrong['checkpoint_infos'][key] += 1
    with pytest.raises(ValueError, match='periodic speed26'):
        r.validate_pair([parent[1], wrong])


@pytest.mark.parametrize('total', (41903, 41500, 43490, 43501, 56500, 61500, 43500., True, None))
def test_nonperiodic_early_overbudget_or_noninteger_endpoint_rejected(total, sealed_endpoints):
    _, parent, children = sealed_endpoints
    wrong = deepcopy(children[43500][1])
    wrong['checkpoint_infos']['successful_updates_total'] = total
    with pytest.raises(ValueError, match='periodic speed26'):
        r.validate_pair([parent[1], wrong])


@pytest.mark.parametrize('key,value', (
    ('origin_checkpoint_sha256', 'wrong'), ('plan_sha256', 'wrong'), ('lineage_id', 'wrong'),
    ('domain', 'terrain'), ('joint_actor_updates', 2096), ('domain_actor_updates', 2097.),
    ('domain_complete', True), ('domain_complete', 0)))
def test_original_learning_origin_and_incomplete_domain_remain_exact(key, value, sealed_endpoints):
    _, parent, children = sealed_endpoints
    wrong = deepcopy(children[43500][1])
    wrong['integrated_lineage'][key] = value
    with pytest.raises(ValueError, match='learning origin'):
        r.validate_pair([parent[1], wrong])


@pytest.mark.parametrize('value', (0, 43500., True))
def test_seal_total_must_equal_checkpoint_total(value, sealed_endpoints):
    _, parent, children = sealed_endpoints
    wrong = deepcopy(children[43500][1])
    wrong['consumed_updates'] = value
    with pytest.raises(ValueError, match='per-file seal clock'):
        r.validate_pair([parent[1], wrong])


@pytest.mark.parametrize('sealed', (None, False))
def test_completion_cannot_replace_per_file_seal(sealed, tmp_path, sealed_endpoints):
    root = sealed_endpoints[0]
    checkpoint, policy = tmp_path / 'model.pt', tmp_path / 'policy.onnx'
    checkpoint.write_bytes(b'candidate')
    policy.write_bytes(b'policy')
    r.write(tmp_path / 'completion.json', dict(status='checkpoint_sealed', successful_updates=43500))
    if sealed is not None:
        r.write(checkpoint.with_suffix('.checkpoint.json'), dict(sealed=sealed))
    with pytest.raises(ValueError, match='immutable per-file checkpoint seal'):
        r.authenticate_candidate(root, checkpoint, policy)


def test_actor_only_checkpoint_with_valid_seal_is_not_full_learning_state(tmp_path, sealed_endpoints):
    root = sealed_endpoints[0]
    checkpoint, policy = tmp_path / 'model.pt', tmp_path / 'policy.onnx'
    torch.save(dict(actor_state_dict={}, infos={}), checkpoint)
    policy.write_bytes(b'policy')
    contract = tmp_path / 'model.contract.json'
    shutil.copyfile(CHILDREN[43500] / 'contract.json', contract)
    r.write(checkpoint.with_suffix('.checkpoint.json'), dict(schema='chassis-learning-checkpoint-v1', sealed=True,
        checkpoint_sha256=r.digest(checkpoint), contract_sha256=r.digest(contract), learning_updates=43500))
    with pytest.raises(ValueError, match='full learning checkpoint'):
        r.authenticate_candidate(root, checkpoint, policy)


def test_authenticated_seal_and_infos_clock_disagreement_rejected(tmp_path, sealed_endpoints):
    root = sealed_endpoints[0]
    original = CHILDREN[43500]
    for name in ('model.pt', 'contract.json', 'model.checkpoint.json', 'agent_config.json'):
        shutil.copyfile(original / name, tmp_path / name)
    seal = r.read(tmp_path / 'model.checkpoint.json')
    seal['learning_updates'] -= 500
    r.write(tmp_path / 'model.checkpoint.json', seal)
    with pytest.raises(ValueError, match='seal learning clock'):
        r.authenticate_candidate(root, tmp_path / 'model.pt', original / 'policy.onnx')


def test_admitted_budget_is_smaller_than_contract_training_cap(sealed_endpoints):
    root = sealed_endpoints[0]
    assert r.read(root / r.SPEED)['task_campaign']['updates'] == 20000
    assert r.SPEED_ACTOR_CAP == 14897
    assert r.periodic_speed_clocks(56000)['task_actor_update'] == 14597
    with pytest.raises(ValueError, match='admitted cap'):
        r.periodic_speed_clocks(56500)


def test_cpu_entrypoints_report_authenticated_43500(tmp_path, sealed_endpoints, monkeypatch):
    root, parent, children = sealed_endpoints
    pairs = [(PARENT / 'model.pt', PARENT / 'policy.onnx'),
             (CHILDREN[43500] / 'model.pt', CHILDREN[43500] / 'policy.onnx')]
    authenticated = [parent, children[43500]]
    def authenticate(observed_root, checkpoint, policy):
        assert observed_root == root
        return deepcopy(authenticated[pairs.index((Path(checkpoint), Path(policy)))])
    monkeypatch.setattr(r, 'authenticate_candidate', authenticate)
    monkeypatch.setattr(d, 'run_protocol', lambda *_: pytest.fail('CPU preparation attempted native replay'))
    for entry, name in ((r.main, 'prepare'), (d.main, 'driver')):
        argv = ['--runtime-root', str(root), '--output', str(tmp_path / name)]
        for checkpoint, policy in pairs:
            argv += ['--candidate', str(checkpoint), str(policy)]
        result = entry(argv)
        assert result['endpoint_selection']['selected_endpoint']['successful_updates_total'] == 43500
        assert result['endpoint_selection']['selected_endpoint']['task_actor_update'] == 2097
        assert result['qualification'] is None and result['scene_run'] is False and result['PPO_updates'] == 0
    assert 'isaaclab.app' not in sys.modules and 'wheeled_tasks.chassis.env' not in sys.modules
