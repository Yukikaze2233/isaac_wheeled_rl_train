"""Authenticate actual frozen-source evidence before a named continuation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tarfile

OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent / 'learning_audit' / 'hold_cost_candidate_021'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    meta = read(OUT / 'source_candidate_021.json')
    evidence = {n: read(OUT / n) for n in ('clean_tests_021.json', 'clean_30_roles_021.json',
                'remote_30_roles_021.json', 'role_inheritance_review_021.json',
                'native_teaching_readback_021_002.json')}
    assert sha(Path(meta['archive_local'])) == meta['archive_sha256']
    with tarfile.open(meta['archive_local'], 'r:gz') as archive:
        assert archive.pax_headers['comment'] == meta['source_commit']
    inventory = read(Path(meta['source_inventory_file']))
    assert sha(Path(meta['source_inventory_file'])) == meta['source_inventory_sha256']
    root = Path(meta['clean_root'])
    assert all(sha(root / n) == s for n, s in inventory.items())
    for d in evidence.values():
        assert d['status'] == 'passed' and d['source_commit'] == meta['source_commit']
        assert d['archive_sha256'] == meta['archive_sha256']
        assert d['formal_updates_consumed'] == 0 and d['qualification'] is None
    tests = evidence['clean_tests_021.json']
    assert tests['exitcode'] == 0 and tests['source_files_unchanged'] and tests['exact_bytes_match_archive']
    for n in ('clean_30_roles_021.json', 'remote_30_roles_021.json'):
        d = evidence[n]
        assert d['role_count'] == len(d['receipts']) == 30 and d['all_selected_source_bytes_match_archive']
        assert d['archive_header_commit'] == meta['source_commit']
        assert all(x['status'] == 'static_passed' for x in d['receipts'])
    assert evidence['role_inheritance_review_021.json']['local_remote_all_30_roles_exactly_equal']
    def roles(d):
        return {(x['stage'], x['role']): {k: x[k] for k in
                  ('seed', 'block_index', 'num_envs', 'physics_dt', 'policy_dt', 'decimation', 'runtime_expected')}
                for x in d['receipts']}
    assert roles(evidence['clean_30_roles_021.json']) == roles(evidence['remote_30_roles_021.json'])
    assert len(roles(evidence['clean_30_roles_021.json'])) == 30
    driver = AUDIT / 'validate_native_teaching_021_002.py'
    assert sha(driver) == '4fed2e4f85f4d8e92c2af804aa98975dc13381c0a6bfcb02f7f49f92d35c84e9'
    spec = importlib.util.spec_from_file_location('source021_admission_readback', driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = module.review_output(OUT / 'native_teaching_021_002')
    assert native['all_required_passed'] and native['actual_goal_bridge_exposure_complete']
    readback = evidence['native_teaching_readback_021_002.json']
    assert all(readback[k] == v for k, v in native.items()) and readback['process_exit_code'] == 0
    invocation = read(OUT / 'native_teaching_021_002.invocation.json')
    assert invocation['source_commit'] == meta['source_commit'] and invocation['archive_sha256'] == meta['archive_sha256']
    assert invocation['driver_sha256'] == sha(driver)
    assert invocation['parent_admission_sha256'] == sha(OUT / 'native_engineering_parent_021_002.json')
    review = read(OUT / 'independent_deploy_review_021.json')
    assert review['blocking_findings'] == [] and review['source_commit'] == meta['source_commit']
    assert review['script_sha256_before'] == review['script_sha256_after'] == sha(OUT / 'deploy_recovery_quality_021.py')
    assert review['archive_header_authenticated'] and review['all_existing219_contracts_unchanged']
    files = [OUT / n for n in evidence]
    files += [OUT / 'source_candidate_021.json', Path(meta['source_inventory_file']),
              OUT / 'clean_tests_021.stdout.log', OUT / 'native_observer_failed_021_001.json',
              OUT / 'native_engineering_parent_021_002.json', OUT / 'native_teaching_021_002.invocation.json',
              OUT / 'independent_deploy_review_021.json', driver,
              OUT / 'independent_native_teaching_CPU_021_002.json',
              OUT / 'independent_native_teaching_review_021_002.json',
              OUT / 'independent_stop_emitter_review_021_002.json',
              AUDIT / 'native_observer_CPU_verification_021_002.json',
              AUDIT / 'actual_engineering_review_021_002.json',
              AUDIT / 'solver_independent_native_teaching_review_021.json',
              OUT / 'deploy_recovery_quality_021.py', OUT / 'emit_stop_previous_021.py',
              OUT / 'observe_first_progress_021.py']
    files += [OUT / 'native_teaching_021_002' / n for n in
              ('report.json', 'identity.json', 'payment_summary.json', 'teaching_summary.json')]
    receipt = dict(schema='frozen-source021-deployment-admission-v1', status='passed',
                   source_commit=meta['source_commit'], archive_sha256=meta['archive_sha256'],
                   evidence_sha256={str(p): sha(p) for p in files},
                   native_observation_scope='engineering_flat_prefix_not_formal_learning_parent',
                   formal_learning_parent='latest_stopped_own_recovery_full_seal_at_prepare',
                   resume_critic_warmup_updates=10, critic_updates_charged=True,
                   actor_std_or_Adam_reset=False, refunds=False, qualification=None,
                   formal_updates_consumed=0, solver_iterations=[64, 32],
                   leg_response_bags='unqualified_not_integrated',
                   limits=['Engineering admission does not establish learned behavior or hardware readiness.',
                           'The native probe retained one guard termination and a 4.28467 mm peak closure error; it does not certify every lane below 3 mm.',
                           'Lower solver iterations and leg-response fits remain unqualified.'])
    target = OUT / 'admission_021.json'
    assert not target.exists(), 'Admission already published: inspect'
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], admission_sha256=sha(target))))


if __name__ == '__main__':
    main()
