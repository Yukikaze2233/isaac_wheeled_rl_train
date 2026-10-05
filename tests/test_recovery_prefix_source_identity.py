"""CPU-only source authentication tests in disposable complete Git checkouts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import recovery_prefix_support as support


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True).stdout


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / 'complete-source'
    root.mkdir()
    git(root, 'init', '-q')
    names = [*support.CORE_FILES, 'src/wheeled_tasks/chassis/preflight.py', 'scripts/probe_recovery_physical_prefix.py']
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(support.PREPARE_CHANGE['old']+'\n' if name == support.PREPARE_CHANGE['file'] else '# fixed CPU fixture\n')
    git(root, 'add', '.')
    git(root, '-c', 'user.name=CPU Source Test', '-c', 'user.email=cpu-test@example.invalid', 'commit', '-qm', 'harness baseline')
    harness = git(root, 'rev-parse', 'HEAD').decode().strip()
    monkeypatch.setattr(support, 'CORE_REVISION', harness)
    return {'root': root, 'harness': harness, 'names': names, 'manifest': tmp_path / 'external-identity.json'}


def candidate(source, extra=''):
    path = source['root'] / support.PREPARE_CHANGE['file']
    path.write_text(support.PREPARE_CHANGE['new']+'\n'+extra)
    git(source['root'], 'add', '.')
    git(source['root'], '-c', 'user.name=CPU Source Test', '-c', 'user.email=cpu-test@example.invalid', 'commit', '-qm', 'candidate')


def manifest(source, mode):
    root, harness = source['root'], source['harness']
    revision = git(root, 'rev-parse', 'HEAD').decode().strip()
    actual = {name: support.digest(root / name) for name in source['names']}
    base = {name: hashlib.sha256(git(root, 'show', f'{harness}:{name}')).hexdigest() for name in support.CORE_FILES}
    value = {'schema': 'recovery-prefix-source-identity-v1', 'mode': mode, 'source_root': str(root),
             'base_core_revision': harness, 'harness_revision': harness, 'source_revision': revision,
             'base_core_files_sha256': base, 'core_files_sha256': {name: actual[name] for name in support.CORE_FILES},
             'runtime_sources_sha256': actual, 'candidate_change': None if mode == 'baseline' else support.PREPARE_CHANGE,
             'candidate_diff_sha256': hashlib.sha256(git(root, 'diff', '--no-ext-diff', '--no-textconv', harness, revision)).hexdigest(),
             'engineering_expectations_sha256': support.ENGINEERING_EXPECTATIONS_SHA}
    support.write_json(source['manifest'], value)
    return value, actual


def verify(source, actual, identity_sha=None):
    return support.core_sources(source['root'], identity_path=source['manifest'],
                                identity_sha256=identity_sha or support.digest(source['manifest']), actual_sources=actual)


@pytest.mark.parametrize('mode', ['baseline', 'prepare_slew_1_candidate'])
def test_exact_complete_source_identity_is_authenticated(source, mode):
    if mode != 'baseline':
        candidate(source)
    identity, actual = manifest(source, mode)
    receipt = verify(source, actual)
    assert receipt['revision'] == identity['source_revision'] and receipt['mode'] == mode
    assert receipt['base_revision'] == source['harness'] and receipt['runtime_sources_sha256'] == actual
    if mode != 'baseline':
        assert receipt['revision'] != receipt['base_revision']


def test_candidate_without_manifest_is_rejected(source):
    candidate(source)
    with pytest.raises(ValueError, match='differs from b4'):
        support.core_sources(source['root'])


def test_wrong_external_manifest_sha_and_missing_digest_are_rejected(source):
    _, actual = manifest(source, 'baseline')
    with pytest.raises(ValueError, match='manifest SHA mismatch'):
        verify(source, actual, '0'*64)
    with pytest.raises(ValueError, match='supplied together'):
        support.core_sources(source['root'], identity_path=source['manifest'])


def test_additional_core_edit_is_rejected_even_with_resigned_manifest(source):
    candidate(source)
    path = source['root'] / support.CORE_FILES[0]
    path.write_text('# additional core edit\n')
    git(source['root'], 'add', '.')
    git(source['root'], '-c', 'user.name=CPU Source Test', '-c', 'user.email=cpu-test@example.invalid', 'commit', '-qm', 'extra')
    _, actual = manifest(source, 'prepare_slew_1_candidate')
    with pytest.raises(ValueError, match='only the fixed PREPARE'):
        verify(source, actual)


def test_arbitrary_same_file_diff_is_rejected_even_with_resigned_manifest(source):
    candidate(source, extra='# arbitrary overlay\n')
    _, actual = manifest(source, 'prepare_slew_1_candidate')
    with pytest.raises(ValueError, match='beyond approved line'):
        verify(source, actual)


@pytest.mark.parametrize('field,value,message', [
    ('source_revision', '0'*40, 'Actual Git HEAD'),
    ('candidate_diff_sha256', '0'*64, 'Candidate diff SHA'),
    ('engineering_expectations_sha256', '0'*64, 'independent four-birth expectations'),
    ('source_root', '/tmp/not-the-complete-checkout', 'root identity mismatch'),
])
def test_identity_cannot_rebind_head_diff_design_or_root(source, field, value, message):
    identity, actual = manifest(source, 'baseline')
    identity[field] = value
    support.write_json(source['manifest'], identity)
    with pytest.raises(ValueError, match=message):
        verify(source, actual)


def test_dirty_or_untracked_overlay_is_rejected(source):
    _, actual = manifest(source, 'baseline')
    (source['root'] / 'extra_runtime.py').write_text('# untracked overlay\n')
    with pytest.raises(ValueError, match='not clean'):
        verify(source, actual)


def test_runtime_source_sha_mismatch_is_rejected(source):
    _, actual = manifest(source, 'baseline')
    actual[support.CORE_FILES[0]] = '0'*64
    with pytest.raises(ValueError, match='Full actual runtime source SHA'):
        verify(source, actual)


def test_independent_four_birth_expectations_match_the_fixed_receipt(tmp_path):
    support.materialize(ROOT, ROOT / 'contracts/v6_task_campaign_recovery_worker_v8.json', tmp_path)
    assert support.digest(tmp_path / 'engineering_expectations.json') == support.ENGINEERING_EXPECTATIONS_SHA


def test_preflight_final_identity_failure_cannot_return_success(tmp_path, monkeypatch):
    import probe_recovery_physical_prefix as probe
    checkpoint = ROOT / 'models/v6_model_delivery_20261003T2235/recommended_flat/model_final.pt'
    if not checkpoint.exists():
        pytest.skip('Local authenticated flat delivery checkpoint absent')
    original, calls = probe.core_sources, []
    def changing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('Injected final Git identity change; CPU only')
        return original(*args, **kwargs)
    monkeypatch.setattr(probe, 'core_sources', changing)
    output = tmp_path / 'final-identity-change'
    assert probe.main(['--contract', str(ROOT / 'contracts/v6_task_campaign_recovery_worker_v8.json'),
                       '--checkpoint', str(checkpoint), '--output', str(output), '--preflight-only']) == 1
    result = support.read_json(output / 'report.json')
    assert result['status'] == 'failed_source_identity' and result['source_identity_final_verified'] is False
    assert result['checkpoint_unchanged'] and result['physics_steps'] == 0
