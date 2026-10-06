"""CPU-only tests of identity, valid negatives and pair integrity; no native run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope='module')
def module():
    spec = importlib.util.spec_from_file_location('REC3076_pair_fixture', HERE / 'run_sealed_REC3076_pair_021.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def terminal(module, tmp_path):
    # Real completed engineering002 metadata seeds only this classification
    # fixture. It does not pretend to be a REC3076 native outcome.
    source = module.REPORT / 'source_repair_021/native_teaching_021_002'
    for name in ('report.json', 'identity.json', 'payment_summary.json', 'teaching_summary.json'):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    return tmp_path


def replace_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def set_negative(module, output):
    teaching = module.read(output / 'teaching_summary.json')
    teaching['exposure_counts'] = [[[0, 0, 0] for _ in range(3)] for _ in range(2)]
    teaching['L0_all_three_nonzero_followups_observed'] = False
    replace_json(output / 'teaching_summary.json', teaching)
    report = module.read(output / 'report.json')
    report['teaching_summary_sha256'] = module.sha(output / 'teaching_summary.json')
    replace_json(output / 'report.json', report)


def test_actual_sealed_own_REC3076_prepare_is_bound_not_flat_or_forced(module, tmp_path):
    checkpoint = HERE / 'immutable_REC3076/model.pt'
    plan = module.prepare(tmp_path / 'pair', checkpoint.resolve())
    assert plan['checkpoint_sha256'] == module.PT_SHA
    assert plan['checkpoint_clocks']['task_actor_update'] == 3076
    assert plan['checkpoint_clocks']['learning_updates'] == 30007
    assert module.read(tmp_path / 'pair/admission.json')['context'] == 'sealed_recovery_resume'
    assert plan['commands']['mean'][-3:-1] == ['mean', '--output']
    assert plan['commands']['stochastic'][-3:-1] == ['stochastic', '--output']
    module.load_plan(tmp_path / 'pair')
    with pytest.raises(ValueError, match='exists'):
        module.prepare(tmp_path / 'pair', checkpoint.resolve())


def test_strict_exposure_and_valid_zero_exposure_are_separate(module, terminal):
    driver = module.bound_driver()
    positive = module.diagnostic_review(driver, terminal)
    assert positive['native_terminal_integrity_passed'] and positive['strict_engineering_exposure_admission_passed']
    set_negative(module, terminal)
    negative = module.diagnostic_review(driver, terminal)
    assert negative['native_terminal_integrity_passed'] and not negative['strict_engineering_exposure_admission_passed']
    assert negative['status'] == 'native_complete_valid_exposure_negative'
    assert negative['first_birth']['censor_is_not_failure']


@pytest.mark.parametrize('failure', ['residual', 'count', 'sidecar', 'qualification', 'hash'])
def test_no_integrity_failure_is_reclassified_as_a_valid_negative(module, terminal, failure):
    set_negative(module, terminal)
    report = module.read(terminal / 'report.json')
    if failure == 'residual':
        payment = module.read(terminal / 'payment_summary.json');payment['max_reconciliation_residual'] = 1
        replace_json(terminal / 'payment_summary.json', payment)
        report['payment_summary_sha256'] = module.sha(terminal / 'payment_summary.json')
    elif failure == 'count':
        report['policy_steps_completed'] = 599
    elif failure == 'sidecar':
        (terminal / 'teaching_observer_failure.json').write_text('{}')
    elif failure == 'qualification':
        identity = module.read(terminal / 'identity.json');identity['qualification'] = 'bad_credit'
        replace_json(terminal / 'identity.json', identity)
        report['identity_sha256'] = module.sha(terminal / 'identity.json')
    else:
        report['teaching_summary_sha256'] = '0' * 64
    replace_json(terminal / 'report.json', report)
    with pytest.raises(ValueError):
        module.diagnostic_review(module.bound_driver(), terminal)


def test_pair_requires_same_initial_birth_and_noise_but_allows_end_env_divergence(module, terminal, tmp_path, monkeypatch):
    root = tmp_path / 'pair';root.mkdir()
    for arm in module.ARMS:
        path = root / arm;path.mkdir()
        for name in ('report.json', 'identity.json', 'payment_summary.json', 'teaching_summary.json'):
            (path / name).write_bytes((terminal / name).read_bytes())
    set_negative(module, root / 'stochastic')
    report = module.read(root / 'stochastic/report.json');report['final_env_rng_sha256'] = 'allowed-reset-divergence'
    report['arm'] = 'stochastic'
    replace_json(root / 'stochastic/report.json', report)
    (root / 'CPU_mean').mkdir()
    (root / 'CPU_mean/identity.json').write_bytes((terminal / 'identity.json').read_bytes())
    monkeypatch.setattr(module, 'load_plan', lambda output: ({}, {}))
    for arm in module.ARMS:
        module.write(root / (arm + '.process.json'), dict(pid=-999, process_exit_observed=True,
            readback=module.diagnostic_review(module.bound_driver(), root / arm)))
    result = module.pair_review(root)
    assert result['same_private_sampling_stream'] and result['arms']['stochastic']['status'].endswith('negative')
    (root / 'pair_review.json').unlink()
    report['initial_physical_birth_sha256'] = 'different'
    replace_json(root / 'stochastic/report.json', report)
    row = module.read(root / 'stochastic.process.json');row['readback']['report_sha256'] = module.sha(root / 'stochastic/report.json')
    replace_json(root / 'stochastic.process.json', row)
    with pytest.raises(ValueError, match='initial birth'):
        module.pair_review(root)


def test_other_local_compute_owner_refuses_native_launch(module, monkeypatch):
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='123, other-user-app\n'))
    with pytest.raises(ValueError, match='compute owner'):
        module.local_GPU_empty()


def test_changed_source_checkpoint_admission_or_command_is_rejected(module, tmp_path):
    output = tmp_path / 'pair'
    module.prepare(output, (HERE / 'immutable_REC3076/model.pt').resolve())
    plan = module.read(output / 'invocation.json');plan['commands']['stochastic'][-3] = 'mean'
    replace_json(output / 'invocation.json', plan)
    with pytest.raises(ValueError, match='arguments changed'):
        module.load_plan(output)
