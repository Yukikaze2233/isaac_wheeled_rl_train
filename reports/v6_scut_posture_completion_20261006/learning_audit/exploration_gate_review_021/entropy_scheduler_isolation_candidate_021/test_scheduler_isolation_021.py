"""CPU-only candidate: claim-first crash recovery and owned publishing boundaries."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_task_campaign as original
from wheeled_tasks.chassis import task_campaign as accounting

spec = importlib.util.spec_from_file_location('isolated_scheduler_candidate021', BASE / 'candidate/scripts/run_task_campaign.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


class Child:
    def __init__(self): self.signals = []
    def poll(self): return None
    def send_signal(self, value): self.signals.append(value)


def make_job(root, module=candidate, version='v2'):
    root.mkdir(parents=True, exist_ok=True)
    planfile = ROOT / f'contracts/v6_task_campaign_{version}.json'
    plan = json.loads(planfile.read_text())
    job = object.__new__(module.TaskCampaign)
    job.root, job.plan = root.resolve(), plan
    job.args = SimpleNamespace(contract=planfile)
    job.claim = root / 'account.json'
    job.records = {}
    job.report = dict(domains={}, evaluations={}, status='running',
                      contract_sha256=module.digest(planfile),
                      **accounting.account_updates(plan, {}))
    job._session_started = None
    job.active = next(r for r in plan['domains'] if r['name'] == 'recovery')
    job.trial_state = job.active_trial = None
    job.child = Child()
    job.assert_no_live_workers = lambda: None
    return job


def trial_payload():
    # The control payload is deliberately opaque: full trial admission is absent.
    return dict(trial_id='controlled-fixture', admission_sha256='a'*64,
                arms={'A': {'charged': 4}}, marker='authoritative_claim')


def write_claim(job, state=None, include=True):
    ledger = dict(campaign_id=accounting.CAMPAIGN_ID, run_directory=str(job.root),
                  contract_sha256=candidate.digest(job.args.contract),
                  attempts=deepcopy(job.records))
    if include: ledger['trial_state'] = deepcopy(state)
    candidate.write_json(job.claim, ledger)


def file_hashes(root, names):
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def own_trial(job):
    directory = job.root / 'research/controlled-fixture/A/recovery/attempt_000/train'
    directory.mkdir(parents=True)
    job.trial_state = trial_payload()
    job.active_trial = dict(trial_id='controlled-fixture', arm_id='A', run_directory=str(directory))
    return directory


def test_normal_persist_remains_byte_exact(tmp_path):
    a, b = make_job(tmp_path, original), make_job(tmp_path)
    a.persist()
    before = file_hashes(tmp_path, ['account.json','consumption.json','campaign_state.json'])
    b.persist()
    assert before == file_hashes(tmp_path, before)
    assert 'trial_state' not in json.loads(b.claim.read_text())


def test_authoritative_claim_overrides_stale_local_report(tmp_path):
    job = make_job(tmp_path); state = trial_payload(); write_claim(job, state)
    stale = deepcopy(job.report); stale['trial_state'] = dict(trial_id='fake', arms={'A': {'charged': 900}})
    candidate.write_json(tmp_path/'campaign_state.json', stale)
    job.open_account()
    assert job.trial_state == state == job.report['trial_state']
    assert json.loads(job.claim.read_text())['trial_state'] == state
    job.report['trial_state']['arms']['A']['charged'] = 777
    assert job.trial_state['arms']['A']['charged'] == 4


def test_local_report_cannot_author_trial_without_claim_field(tmp_path):
    job = make_job(tmp_path); write_claim(job, include=False)
    stale = deepcopy(job.report); stale['trial_state'] = trial_payload()
    candidate.write_json(tmp_path/'campaign_state.json', stale)
    job.open_account()
    assert job.trial_state is None and 'trial_state' not in job.report
    assert 'trial_state' not in json.loads(job.claim.read_text())


def test_claim_write_survives_crash_before_local_consumption(tmp_path, monkeypatch):
    job = make_job(tmp_path); job.trial_state = trial_payload()
    old = deepcopy(job.report); old['trial_state'] = dict(trial_id='stale')
    candidate.write_json(tmp_path/'campaign_state.json', old)
    native_write = candidate.write_json
    def fail_local(path, value):
        if Path(path).name == 'consumption.json': raise OSError('injected claim/report gap')
        native_write(path, value)
    monkeypatch.setattr(candidate, 'write_json', fail_local)
    with pytest.raises(OSError, match='claim/report gap'): job.persist()
    assert json.loads(job.claim.read_text())['trial_state'] == trial_payload()
    monkeypatch.setattr(candidate, 'write_json', native_write)
    reopened = make_job(tmp_path); reopened.open_account()
    assert reopened.trial_state == trial_payload()
    assert reopened.report['trial_state'] == reopened.trial_state


@pytest.mark.parametrize('payload', [[], 'bad', 2, False])
def test_malformed_claim_control_rejected_before_publication(tmp_path, payload):
    job = make_job(tmp_path); write_claim(job, payload)
    before = job.claim.read_bytes()
    with pytest.raises(ValueError, match='trial state'): job.open_account()
    assert job.claim.read_bytes() == before


def test_original_attempt_monotonic_charge_not_replaced_by_trial_payload(tmp_path):
    job = make_job(tmp_path); job.trial_state = trial_payload()
    directory = own_trial(job)
    record = dict(successful_updates=3, attempted_updates=4, actor_updates_in_block=2)
    job.account(directory, record); key = str(directory.relative_to(job.root))
    assert job.records[key] == dict(domain='recovery', consumed_updates=4, actor_updates=2)
    assert job.report['consumed_updates'] == 1004
    assert job.trial_state == trial_payload()
    with pytest.raises(ValueError, match='backwards'):
        job.account(directory, dict(successful_updates=3, attempted_updates=3, actor_updates_in_block=2))
    assert json.loads(job.claim.read_text())['attempts'][key]['consumed_updates'] == 4


def test_research_publish_preserves_main_files_context_tokens_and_signals(tmp_path, monkeypatch):
    job = make_job(tmp_path, version='v9'); directory = own_trial(job)
    main_names = ['startup.json','behavior_metrics.json','torque_monitor.json','live_state.json',
                  'review_available.json','domain_review_request.json','domain_review_response.json']
    for name in main_names: candidate.write_json(tmp_path/name, dict(main=name))
    for name in main_names[:4]: candidate.write_json(directory/name, dict(research=name))
    candidate.write_json(directory/'progress.json', dict(pid=99999999, successful_updates=1,
                         attempted_updates=2, actor_updates_in_block=1, task_actor_update=5))
    job._active_review = dict(existing='main context')
    job._review_request_token = 'existing-token'
    job.report['consumed_review_tokens'] = ['prior-token']
    job.report['domains']['recovery'] = dict(pending_review={'existing':'main pending'})
    before_files = file_hashes(tmp_path, main_names)
    before_domain = deepcopy(job.report['domains'])
    calls=[]
    monkeypatch.setattr(job, 'review_request', lambda *a: calls.append('review'))
    monkeypatch.setattr(job, 'assess_request', lambda *a: calls.append('assessment'))
    assert job.publish(directory)
    assert file_hashes(tmp_path, main_names) == before_files
    assert job.report['domains'] == before_domain and job.report['consumed_review_tokens'] == ['prior-token']
    assert job._active_review == {'existing':'main context'} and job._review_request_token == 'existing-token'
    assert not job.child.signals and not calls
    progress=json.loads((tmp_path/'progress.json').read_text())
    assert progress['active_trial'] == job.active_trial and progress['active_domain'] == 'recovery'
    assert progress['orchestration'].endswith('_research') and progress['successful_updates']==1002
    before_claim=job.claim.read_bytes(); assert job.publish(directory)
    assert job.claim.read_bytes()==before_claim


def test_direct_trial_review_never_consumes_existing_main_request(tmp_path):
    job=make_job(tmp_path,version='v9'); directory=own_trial(job)
    candidate.write_json(tmp_path/'domain_review_request.json', {'should_not_be_read':'old main token'})
    before=(tmp_path/'domain_review_request.json').read_bytes()
    job.review_request(directory);job.review_request()
    assert not job.child.signals and getattr(job,'_active_review',None) is None
    assert not (tmp_path/'domain_review_response.json').exists()
    assert not hasattr(job,'_review_request_token')
    assert (tmp_path/'domain_review_request.json').read_bytes()==before


def test_direct_trial_assessment_is_not_a_main_cooperative_evaluation(tmp_path):
    job=make_job(tmp_path,version='v1');directory=own_trial(job)
    (directory/'assessment_request.json').write_text('invalid JSON must not be read')
    job.assess_request(directory)
    assert not (directory/'assessment_decision.json').exists()


def test_research_path_without_context_fails_closed_before_charging(tmp_path):
    job=make_job(tmp_path);directory=tmp_path/'research/unknown/recovery/attempt_000/train'
    directory.mkdir(parents=True);candidate.write_json(directory/'progress.json',dict(successful_updates=1))
    with pytest.raises(ValueError,match='explicit trial context'):job.publish(directory)
    assert job.records=={} and not job.claim.exists()


@pytest.mark.parametrize('change', ['other_trial','wrong_directory','outside_root','extra_field','missing_field'])
def test_foreign_or_malformed_context_fails_before_any_account_or_review(tmp_path,change):
    job=make_job(tmp_path);directory=own_trial(job)
    if change=='other_trial':job.active_trial['trial_id']='foreign'
    if change=='wrong_directory':directory=directory.parent/'other'
    if change=='outside_root':job.active_trial['run_directory']=str(tmp_path/'recovery/attempt_000/train')
    if change=='extra_field':job.active_trial['extra']=True
    if change=='missing_field':job.active_trial.pop('arm_id')
    with pytest.raises(ValueError,match='publisher'):job.publish(directory)
    assert job.records=={} and not job.claim.exists() and not job.child.signals


def test_main_review_still_consumes_legitimate_request_once_after_context_clear(tmp_path,monkeypatch):
    job=make_job(tmp_path,version='v9');directory=own_trial(job)
    candidate.write_json(tmp_path/'domain_review_request.json', {'main':'valid fixture'})
    monkeypatch.setattr(candidate,'validate_domain_review',lambda *a:dict(domain='recovery',decision='finish_domain'))
    job.review_request(directory);assert not job.child.signals
    job.active_trial=None
    job.review_request();job.review_request()
    assert job.child.signals==[signal.SIGTERM]
    assert job._active_review['domain']=='recovery'
    assert job.report['domains']['recovery']['pending_review']['token']==candidate.digest(tmp_path/'domain_review_request.json')
    assert json.loads((tmp_path/'domain_review_response.json').read_text())['status']=='sealing_latest_learning_state'


def test_normal_publish_is_byte_exact_and_unchanged_receipt_skips_account(tmp_path):
    directory=tmp_path/'recovery/attempt_000/train';directory.mkdir(parents=True)
    for name in ('startup.json','behavior_metrics.json','torque_monitor.json','live_state.json'):
        candidate.write_json(directory/name,{name:True})
    candidate.write_json(directory/'progress.json',dict(pid=99999999,successful_updates=2,
                         attempted_updates=3,actor_updates_in_block=2))
    names=['account.json','consumption.json','campaign_state.json','progress.json',
           'startup.json','behavior_metrics.json','torque_monitor.json','live_state.json']
    before_job=make_job(tmp_path,original);before_job.publish(directory)
    hashes=file_hashes(tmp_path,names)
    after_job=make_job(tmp_path);after_job.publish(directory)
    assert file_hashes(tmp_path,names)==hashes
    assert after_job.publish(directory) and file_hashes(tmp_path,names)==hashes


def test_predecessor_reproduces_claim_control_loss_candidate_preserves_it(tmp_path):
    state=trial_payload()
    before=make_job(tmp_path/'before',original);write_claim(before,state)
    stale=deepcopy(before.report);stale['trial_state']={'trial_id':'obsolete-local'}
    candidate.write_json(before.root/'campaign_state.json',stale)
    before.open_account()
    # Actual predecessor reconstructs the claim without its control state.
    assert 'trial_state' not in json.loads(before.claim.read_text())
    assert before.report['trial_state']=={'trial_id':'obsolete-local'}
    after=make_job(tmp_path/'after');write_claim(after,state)
    stale=deepcopy(after.report);stale['trial_state']={'trial_id':'obsolete-local'}
    candidate.write_json(after.root/'campaign_state.json',stale)
    after.open_account()
    assert json.loads(after.claim.read_text())['trial_state']==state
    assert after.report['trial_state']==state


def test_predecessor_reproduces_main_SIGTERM_on_research_candidate_blocks_it(tmp_path,monkeypatch):
    for module,name,should_signal in [(original,'before',True),(candidate,'after',False)]:
        job=make_job(tmp_path/name,module,version='v9');directory=own_trial(job)
        candidate.write_json(directory/'progress.json',dict(task_actor_update=0))
        candidate.write_json(job.root/'domain_review_request.json',dict(main='old valid request fixture'))
        monkeypatch.setattr(module,'validate_domain_review',lambda *a:dict(domain='recovery',decision='finish_domain'))
        job.review_request(directory)
        assert bool(job.child.signals) is should_signal
        if should_signal:assert job.child.signals==[signal.SIGTERM]
        else:assert not (job.root/'domain_review_response.json').exists()


def test_research_symlink_escape_without_context_rejects_before_main_dispatch(tmp_path):
    job=make_job(tmp_path/'campaign');outside=tmp_path/'outside';outside.mkdir()
    research=job.root/'research';research.mkdir();(research/'escaped').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='explicit trial context'):job.publish(research/'escaped')
    assert job.records=={} and not job.child.signals


def test_active_research_root_symlink_outside_campaign_is_not_owned(tmp_path):
    job=make_job(tmp_path/'campaign');outside=tmp_path/'outside';outside.mkdir()
    (job.root/'research').symlink_to(outside,target_is_directory=True)
    directory=job.root/'research/trial/A/recovery/attempt_000/train';directory.mkdir(parents=True)
    job.trial_state=trial_payload()
    job.active_trial=dict(trial_id='controlled-fixture',arm_id='A',run_directory=str(directory))
    with pytest.raises(ValueError,match='claimed trial directory'):job.publish(directory)
    assert job.records=={} and not job.claim.exists()
