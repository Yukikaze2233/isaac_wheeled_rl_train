"""Independent task workers never mint budget or promote failed domains."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_task_campaign import TaskCampaign, write_json


def plan():
    return dict(contract_id=campaign.REVISION, campaign_id=campaign.CAMPAIGN_ID,
        historical_consumed_updates=1000, historical_actor_consumed_updates=980,
        total_ceiling=121000, target_num_envs=20000, seed=290927,
        evaluation_seeds=[290929, 290930], domains=[dict(name=name, updates=20000,
            parent_domain="A", required_case_names=["stand"], kind="foundation",
            contract="test.json", sha256="expected") for name in campaign.DOMAINS])


def test_frozen_caps_are_per_domain_and_historical_work_is_retained():
    config = plan()
    campaign.validate_plan(config)
    assert campaign.account_updates(config, {})["consumed_updates"] == 1000
    for key, value in (("historical_consumed_updates", 0), ("total_ceiling", 120000),
                       ("historical_actor_consumed_updates", 1000), ("automatic_promotion", True)):
        with pytest.raises(ValueError):
            campaign.validate_plan({**config, key: value})
    changed = deepcopy(config)
    changed["domains"][0]["updates"] = 20001
    with pytest.raises(ValueError):
        campaign.validate_plan(changed)


def test_discarded_attempts_and_update_reservations_are_never_refunded():
    config, records = plan(), {}
    records = campaign.update_attempt(config, records, "flat/0", "flat",
        dict(successful_updates=37, attempted_updates=38, actor_updates_in_block=37))
    # Recovering a snapshot at 30 does not remove the eight already charged updates.
    records = campaign.update_attempt(config, records, "flat/1", "flat",
        dict(successful_updates=10, actor_updates_in_block=10))
    records = campaign.update_attempt(config, records, "speed/0", "speed",
        dict(successful_updates=20, actor_updates_in_block=20))
    total = campaign.account_updates(config, records)
    assert total["consumed_updates"] == 1068
    assert total["actor_updates_consumed"] == 1047
    assert total["domain_consumed_updates"]["flat"] == 48
    with pytest.raises(ValueError, match="backwards"):
        campaign.update_attempt(config, records, "flat/0", "flat",
            dict(successful_updates=30, actor_updates_in_block=30))
    with pytest.raises(ValueError, match="budget"):
        campaign.update_attempt(config, records, "flat/2", "flat",
            dict(successful_updates=20000, actor_updates_in_block=20000))
    for row in records.values():
        assert "stage_actor_update" not in row  # Checkpoint lineage is not the consumption ledger.


@pytest.mark.parametrize("first,second", [(False, False), (False, True), (True, False), (True, True)])
def test_each_domain_needs_both_seeds_without_claiming_stage_or_deployment(first, second):
    results = [dict(cases={"stand": dict(passed=value)}) for value in (first, second)]
    assessment = campaign.assess_domain(results, ["stand"])
    assert assessment["domain_qualified"] == (first and second)
    assert assessment["accepted_stage"] is None
    assert assessment["deployment_checkpoint"] is None
    with pytest.raises(ValueError, match="coverage"):
        campaign.assess_domain(results[:1], ["stand"])
    with pytest.raises(ValueError, match="coverage"):
        campaign.assess_domain(results, ["stand", "missing"])


def minimal_worker():
    return dict(physics_dt=.001, policy_dt=.02, actor_dim=35, critic_dim=81,
        target_num_envs=20000, learning_rate=1e-4, learning_rate_schedule="fixed",
        task_campaign=dict(campaign_id=campaign.CAMPAIGN_ID, domain="flat",
            updates=20000, clock="domain_actor_updates"))


def test_cross_domain_resume_authenticates_source_and_preserves_physics(tmp_path, monkeypatch):
    source = minimal_worker()
    source.pop("task_campaign")
    source_path = tmp_path / "source.json"
    write_json(source_path, source)
    target = minimal_worker()
    target["task_campaign"].update(source_contract_file="source.json",
        source_contract_sha256=campaign.digest(source_path))
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    assert campaign.validate_task_resume(source, target)["task_clock"] == "new_domain_zero"
    for key, value in (("physics_dt", .005), ("critic_dim", 82), ("actuator_response", {"enabled": False})):
        with pytest.raises(ValueError, match="identity"):
            campaign.validate_task_resume(source, {**target, key: value})
    with pytest.raises(ValueError, match="origin"):
        campaign.validate_task_resume({**source, "unknown": 1}, target)
    assert campaign.validate_task_resume(target, target)["task_clock"] == "restored"
    with pytest.raises(ValueError, match="Same-domain"):
        campaign.validate_task_resume(target, {**target, "new_reward": 1})


def task(tmp_path):
    result = object.__new__(TaskCampaign)
    result.root = tmp_path
    result.plan = plan()
    result.plan["source_checkpoint"] = str(tmp_path / "A.pt")
    result.records = {}
    result.report = dict(domains={}, evaluations={}, assessment_seconds=0.,
        **campaign.account_updates(result.plan, {}))
    result.args = SimpleNamespace(device="cpu")
    config = minimal_worker()
    config["evaluation"] = {"cases": [{"name": "stand"}]}
    contract = tmp_path / "test.json"
    write_json(contract, config)
    for row in result.plan["domains"]:
        row["contract"] = str(contract)
    result.deadline = __import__("time").monotonic() + 100
    result.active = result.plan["domains"][0]
    result.stop_requested = False
    return result


def seal(path, consumed=658, task_clock=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"model")
    config = minimal_worker() if task_clock is not None else {"origin": "A"}
    write_json(path.with_suffix(".contract.json"), config)
    write_json(path.with_suffix(".checkpoint.json"), dict(schema="chassis-learning-checkpoint-v1",
        sealed=True, checkpoint_sha256=campaign.digest(path),
        contract_sha256=campaign.digest(path.with_suffix(".contract.json")),
        learning_updates=consumed, learning_lineage_updates=378,
        stage_actor_update=368, actor_updates_consumed=638, task_actor_update=task_clock))


def test_whole_domain_has_one_training_command_and_independent_clock(tmp_path):
    job = task(tmp_path)
    source = tmp_path / "A.pt"
    seal(source)
    command = job.train_command(job.active, tmp_path / "train", source, 20000)
    assert command[command.index("--updates") + 1] == "20000"
    assert command[command.index("--consumed-updates") + 1] == "1000"
    assert command[command.index("--stage-actor-update") + 1] == "368"
    assert command[command.index("--task-actor-update") + 1] == "0"
    seal(source, consumed=1030, task_clock=30)
    command = job.train_command(job.active, tmp_path / "train", source, 19962)
    assert command[command.index("--task-actor-update") + 1] == "30"
    assert command[command.index("--updates") + 1] == "19962"


def test_campaign_account_rejects_new_revision_or_duplicate_root(tmp_path):
    job = task(tmp_path)
    contract = tmp_path / "plan.json"
    write_json(contract, job.plan)
    job.args.contract = contract
    job.claim = tmp_path / "account.json"
    write_json(job.claim, dict(campaign_id=campaign.CAMPAIGN_ID,
        run_directory=str(tmp_path / "different"), contract_sha256=campaign.digest(contract), attempts={}))
    with pytest.raises(ValueError, match="different run/revision"):
        job.open_account()


def test_unqualified_completed_domain_does_not_retrain_or_get_promoted(tmp_path, monkeypatch):
    job = task(tmp_path)
    checkpoint = tmp_path / "final.pt"
    seal(checkpoint, consumed=21000, task_clock=20000)
    job.report["domains"]["flat"] = dict(status="domain_budget_exhausted",
        domain_qualified=False, candidate_checkpoint=str(checkpoint))
    monkeypatch.setattr(job, "execute", lambda *args: pytest.fail("completed task must not restart"))
    job.run_domain(job.active)
    assert not job.report["domains"]["flat"]["domain_qualified"]


def test_assessment_request_is_idempotent_and_preserves_worker(tmp_path, monkeypatch):
    job = task(tmp_path)
    checkpoint = tmp_path / "train/checkpoints/update_00002000/model.pt"
    seal(checkpoint, consumed=2000, task_clock=1000)
    directory = tmp_path / "train"
    job.report["consumed_updates"] = 2000
    request = dict(token="unique", checkpoint=str(checkpoint), checkpoint_sha256=campaign.digest(checkpoint),
                   task_actor_update=1000, consumed_updates=2000)
    write_json(directory / "assessment_request.json", request)
    called = []
    def evaluate(row, source, label):
        called.append(label)
        job.report["evaluations"]["flat"] = {f"{label}/{seed}": dict(file=f"{seed}.json", sha256="hash")
            for seed in job.plan["evaluation_seeds"]}
        return [dict(cases={"stand": dict(passed=False)})] * 2
    monkeypatch.setattr(job, "evaluate_pair", evaluate)
    monkeypatch.setattr(job, "persist", lambda: None)
    job.assess_request(directory)
    job.assess_request(directory)
    assert len(called) == 1
    decision = json.loads((directory / "assessment_decision.json").read_text())
    assert decision["token"] == "unique"
    assert decision["decision"] == "continue"
    assert [e["seed"] for e in decision["evaluations"]] == [290929, 290930]


def test_worker_resume_preserves_lineage_and_rejects_clock_or_budget_reset(tmp_path, monkeypatch):
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
    source = minimal_worker()
    source.pop("task_campaign")
    checkpoint = tmp_path / "A.pt"
    seal(checkpoint)
    write_json(checkpoint_contract_path(checkpoint), source)
    seal_path = checkpoint.with_suffix(".checkpoint.json")
    metadata = json.loads(seal_path.read_text())
    metadata["contract_sha256"] = campaign.digest(checkpoint_contract_path(checkpoint))
    write_json(seal_path, metadata)
    target = minimal_worker()
    target["task_campaign"].update(source_contract_file="A.contract.json",
        source_contract_sha256=metadata["contract_sha256"])
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    args = SimpleNamespace(resume_task_change=True, resume=checkpoint, num_envs=20000,
        seed=290927, learning_rate_scale=1., global_actor_update_offset=0,
        block_index=0, updates=20000, consumed_updates=1000, task_actor_update=0,
        stage_actor_update=368, actor_updates_consumed=980)
    campaign.validate_worker(target, args)
    for key, value in (("task_actor_update", 368), ("stage_actor_update", 0),
                       ("consumed_updates", 999), ("updates", 20001), ("num_envs", 8192)):
        changed = deepcopy(args)
        setattr(changed, key, value)
        with pytest.raises(ValueError):
            campaign.validate_worker(target, changed)


def test_final_gate_failure_records_candidate_and_allows_next_independent_domain(tmp_path, monkeypatch):
    job = task(tmp_path)
    checkpoint = tmp_path / "flat/attempt_000/train/model_final.pt"
    seal(checkpoint, consumed=21000, task_clock=20000)
    job.report["domain_consumed_updates"]["flat"] = 20000
    job.report["consumed_updates"] = 21000
    monkeypatch.setattr(job, "persist", lambda: None)
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: [dict(cases={"stand":dict(passed=False)})]*2)
    job.run_domain(job.active)
    flat = job.report["domains"]["flat"]
    assert flat["status"] == "domain_budget_exhausted"
    assert flat["candidate_checkpoint"] == str(checkpoint)
    assert not flat["domain_qualified"]
    recovery = {**job.plan["domains"][1], "parent_domain": "flat"}
    assert job.learning_source(recovery) == checkpoint
    assert flat["accepted_stage"] is None


def test_task_budget_authenticates_spec_and_rejects_legacy_stage_alias(tmp_path):
    budget = dict(schema="task-domain-budget-v1", campaign_id=campaign.CAMPAIGN_ID,
        historical_consumed_updates=1000, historical_actor_consumed_updates=980,
        total_ceiling=121000, domains=dict.fromkeys(campaign.DOMAINS, 20000),
        rollback_refunds_consumed_updates=False)
    path = tmp_path / "budget.json"
    write_json(path, budget)
    config = dict(budget_spec_file="budget.json", budget_spec_sha256=campaign.digest(path),
        task_campaign=dict(domain="flat"), budget_stage_id="flat", curriculum_stage="flat",
        total_updates=20000, stages=[dict(updates=20000)])
    assert campaign.validate_task_budget(config, tmp_path)["total_ceiling"] == 121000
    for field, value in (("total_updates", 1000), ("budget_stage_id", "S1"), ("stages", [dict(updates=19999)])):
        with pytest.raises(ValueError):
            campaign.validate_task_budget({**config, field: value}, tmp_path)
    write_json(path, {**budget, "historical_consumed_updates": 0})
    with pytest.raises(ValueError, match="SHA"):
        campaign.validate_task_budget(config, tmp_path)


def evaluated_record(checkpoint, seed, *, passed=True):
    return dict(status="evaluated", contract_sha256="expected", settings=dict(seed=seed),
        candidates=[dict(checkpoint_sha256=campaign.digest(checkpoint), unfinished_episodes=0,
                         cases={"stand": dict(passed=passed)})])


def assessment_fixture(job, *, passed=True):
    directory = job.root / "flat/attempt_000/train"
    snapshot = directory / "checkpoints/update_00002000/model.pt"
    final = directory / "model_final.pt"
    for path in (snapshot, final):
        seal(path, consumed=2000, task_clock=1000)
    evidence = []
    for seed in job.plan["evaluation_seeds"]:
        path = job.root / f"{seed}.json"
        write_json(path, evaluated_record(snapshot, seed, passed=passed))
        evidence.append(dict(file=str(path), sha256=campaign.digest(path), seed=seed))
    request = dict(token="unique", checkpoint=str(snapshot), checkpoint_sha256=campaign.digest(snapshot),
                   task_actor_update=1000, consumed_updates=2000)
    decision = dict(token="unique", checkpoint_sha256=campaign.digest(snapshot), task_actor_update=1000,
        domain_qualified=passed, decision="domain_qualified_stop" if passed else "continue", evaluations=evidence)
    write_json(directory / "assessment_request.json", request)
    write_json(directory / "assessment_decision.json", decision)
    job.report["consumed_updates"] = 2000
    return final, snapshot, directory


def test_final_reuses_authenticated_assessed_snapshot_without_repeating_evaluation(tmp_path, monkeypatch):
    job = task(tmp_path)
    final, snapshot, directory = assessment_fixture(job)
    job.report["domain_consumed_updates"]["flat"] = 1000
    monkeypatch.setattr(job, "execute", lambda *args: pytest.fail("qualified snapshot must not start PPO again"))
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: pytest.fail("same actor clock must not be evaluated twice"))
    monkeypatch.setattr(job, "persist", lambda: None)
    job.run_domain(job.active)
    result = job.report["domains"]["flat"]
    assert result["candidate_checkpoint"] == str(snapshot)
    assert result["final_checkpoint"] == str(final)
    assert result["assessment_reused"] and result["domain_qualified"]
    assert job.report["consumed_updates"] == 2000


def test_old_or_tampered_assessment_cannot_replace_final_evaluation(tmp_path):
    job = task(tmp_path)
    final, snapshot, directory = assessment_fixture(job, passed=False)
    seal(final, consumed=2001, task_clock=1001)
    assert job.assessed_candidate(job.active, final) is None
    seal(final, consumed=2000, task_clock=1000)
    path = directory / "assessment_decision.json"
    record = json.loads(path.read_text())
    write_json(path, {**record, "domain_qualified": True})
    with pytest.raises(ValueError, match="differs from fixed evidence"):
        job.assessed_candidate(job.active, final)
    write_json(path, {**record, "token": "stale"})
    with pytest.raises(ValueError, match="authentication"):
        job.assessed_candidate(job.active, final)


def test_failed_evaluation_attempt_is_preserved_and_completed_retry_is_reused(tmp_path):
    job = task(tmp_path)
    checkpoint = tmp_path / "A.pt"
    seal(checkpoint)
    output = tmp_path / "flat/evaluations/test/290929"
    write_json(output / "evaluation.json", {"status": "failed"})
    retry, candidate = job.evaluation_output(job.active, checkpoint, "test", 290929)
    assert retry.name == "290929_retry_1" and candidate is None and not retry.exists()
    write_json(retry / "evaluation.json", evaluated_record(checkpoint, 290929))
    reused, candidate = job.evaluation_output(job.active, checkpoint, "test", 290929)
    assert reused == retry and candidate["cases"]["stand"]["passed"]
    assert json.loads((output / "evaluation.json").read_text())["status"] == "failed"


def test_claim_survives_missing_local_state_without_refunding(tmp_path, monkeypatch):
    job = task(tmp_path)
    contract = tmp_path / "plan.json"
    write_json(contract, job.plan)
    job.args.contract, job.claim = contract, tmp_path / "shared_account.json"
    attempts = campaign.update_attempt(job.plan, {}, "flat/attempt_000/train", "flat",
        dict(successful_updates=8, attempted_updates=9, actor_updates_in_block=8))
    write_json(job.claim, dict(campaign_id=campaign.CAMPAIGN_ID, run_directory=str(tmp_path),
        contract_sha256=campaign.digest(contract), attempts=attempts))
    monkeypatch.setattr(job, "persist", lambda: None)
    job.open_account()
    assert campaign.account_updates(job.plan, job.records)["consumed_updates"] == 1009


def test_unpublished_pending_checkpoint_is_not_a_recovery_candidate(tmp_path):
    job = task(tmp_path)
    good = tmp_path / "flat/attempt_000/train/checkpoints/update_00001001/model.pt"
    pending = tmp_path / "flat/attempt_000/train/checkpoints/update_00001002.pending/model.pt"
    seal(good, consumed=1001, task_clock=1)
    seal(pending, consumed=1002, task_clock=2)
    assert job.latest_checkpoint(job.active) == good


def test_jump_bin_confirmation_needs_canonical_unassisted_double_pass(tmp_path):
    job = task(tmp_path)
    path = Path(job.active["contract"])
    config = json.loads(path.read_text())
    canonical = dict(name="jump", task="jump", jump_assistance_enabled=False,
        skill=dict(kind="jump", semantics=dict(jump_apex_delta_m=.1)))
    config.update(new_asset_curriculum=dict(jump_curriculum=dict(bins_m=[.1, .2, .3])))
    config["evaluation"]["cases"] = [canonical,
        {**canonical, "name": "assisted", "jump_assistance_enabled": True,
         "skill": dict(kind="jump", semantics=dict(jump_apex_delta_m=.2))},
        {**canonical, "name": "airborne", "skill": dict(kind="jump_landing", semantics=dict(jump_apex_delta_m=.3))}]
    write_json(path, config)
    results = [dict(cases={name:dict(passed=True) for name in ("jump", "assisted", "airborne")})]*2
    assert job.confirmed_jump_bins(job.active, results) == [.1]
    assert job.confirmed_jump_bins(job.active, [dict(cases={}), dict(cases={})]) == [.1]
    with pytest.raises(ValueError, match="frozen curriculum"):
        job.confirmed_jump_bins(job.active, results, previous=[.4])


def test_parent_domain_assessment_cannot_qualify_a_new_domain(tmp_path):
    job = task(tmp_path)
    final, snapshot, directory = assessment_fixture(job)
    recovery = {**job.plan["domains"][1], "parent_domain": "flat"}
    assert job.assessed_candidate(recovery, snapshot) is None
