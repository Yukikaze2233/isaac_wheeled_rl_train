"""Continuous task caps preserve accounting and have no in-training score gate."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
from wheeled_tasks.chassis.preflight import ROLES, validate_design_contract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner


def read(path):
    return json.loads((ROOT / path).read_text())


@pytest.fixture
def plans():
    return read("contracts/v6_task_campaign_v1.json"), read("contracts/v6_task_campaign_v2.json")


def test_named_plan_revision_authenticates_all_plans_workers_and_root(plans):
    before, after = plans
    result = campaign.validate_plan_migration(before, after, after["revision_migration"]["run_directory"])
    assert result["task_clock"] == "restored"
    assert result["consumption"] == "preserved_no_refund"
    assert before["total_ceiling"] == after["total_ceiling"] == 121000
    with pytest.raises(ValueError, match="identity"):
        campaign.validate_plan_migration(before, after, "/some/new/run")
    for plan_key, value in (("historical_consumed_updates", 0), ("source_checkpoint", "different.pt")):
        changed = deepcopy(after)
        changed[plan_key] = value
        with pytest.raises(ValueError):
            campaign.validate_plan_migration(before, changed, after["revision_migration"]["run_directory"])


def test_worker_revision_preserves_every_unlisted_field_and_learning_clock(plans):
    before, after = plans
    for old_row, row in zip(before["domains"], after["domains"], strict=True):
        source, target = read(old_row["contract"]), read(row["contract"])
        result = campaign.validate_task_resume(source, target)
        assert result["task_clock"] == "restored"
        assert result["actor_critic_optimizer"] == "preserved"
        assert target["checkpoint_interval"] == 500
        assert target["task_campaign"]["evaluation_interval"] == 0
        for key in ("dense_tracking", "precision_tracking", "landing_tracking", "actuator_response",
                    "v5_control", "skill_specs", "scene_groups", "evaluation"):
            assert source[key] == target[key]
        for key, value in (("physics_dt", .005), ("learning_rate", .0002), ("checkpoint_interval", 100),
                           ("new_reward", {"weight": 1.})):
            changed = deepcopy(target)
            changed[key] = value
            with pytest.raises(ValueError):
                campaign.validate_task_resume(source, changed)


def test_v2_independent_role_designs_and_exact_case_scopes(plans):
    _, plan = plans
    for row in plan["domains"]:
        worker = read(row["contract"])
        design = read(worker["design_preflight"]["file"])
        control = load_chassis_control(ROOT / worker["control_math_source"])
        assert len(worker["evaluation"]["cases"]) == 322
        for role in ROLES:
            effective = deepcopy(worker)
            seed, count = 290927, 20000
            if role != "train":
                seed = 290930 if role == "confirmation" else 290929
                effective["evaluation"]["seed"] = seed
                effective["evaluation"] = select_evaluation_cases(effective["evaluation"], row["required_case_names"])
                effective = fixed_suite_contract(effective)
                count = 4 * len(row["required_case_names"])
            validate_design_contract(effective, design, control, stage=row["name"], role=role,
                                     seed=seed, num_envs=count, block_index=0)


def test_jump_probability_curriculum_uses_task_clock_and_retains_each_bin(plans):
    _, plan = plans
    for row in plan["domains"]:
        setting = read(row["contract"])["new_asset_curriculum"]["jump_curriculum"]
        if row["name"] not in ("jump", "stepjump"):
            assert "iteration_probability_schedule" not in setting
            continue
        assert "confirmed_pass_transfer_fraction" not in setting
        assert setting["clock"] == "domain_actor_updates"
        schedule = setting["iteration_probability_schedule"]
        assert [entry["actor_update"] for entry in schedule] == [0, 2000, 6000]
        for entry in schedule:
            assert len(entry["probabilities"]) == len(setting["bins_m"])
            assert sum(entry["probabilities"]) == pytest.approx(1.)
            assert min(entry["probabilities"]) >= setting["minimum_probability"]


def continuous_job(tmp_path, plan):
    job = object.__new__(runner.TaskCampaign)
    job.plan = deepcopy(plan)
    job.root = tmp_path
    job.report = dict(domains={}, evaluations={}, training_process_seconds=0., assessment_seconds=0.,
                      **campaign.account_updates(plan, {}))
    job.records = {}
    job.args = SimpleNamespace(device="cpu")
    job.deadline = float("inf")
    job.stop_requested = False
    job.persist = lambda: None
    return job


def test_completed_cap_has_no_evaluation_and_next_domain_can_use_unassessed_parent(tmp_path, monkeypatch, plans):
    _, plan = plans
    job = continuous_job(tmp_path, plan)
    row = plan["domains"][0]
    checkpoint = tmp_path / "flat/attempt_000/train/model_final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sealed learning state")
    job.report["domain_consumed_updates"]["flat"] = 20000
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda _: 21000)
    monkeypatch.setattr(job, "learning_source", lambda _: checkpoint)
    monkeypatch.setattr(job, "evaluate_pair", lambda *a: pytest.fail("continuous training cannot evaluate"))
    job.run_domain(row)
    result = job.report["domains"]["flat"]
    assert result["status"] == "domain_budget_exhausted_unassessed"
    assert result["domain_qualified"] is None and result["assessment_status"] == "not_run"
    assert result["candidate_checkpoint"] == str(checkpoint)
    monkeypatch.undo()
    monkeypatch.setattr(job, "latest_checkpoint", lambda _: None)
    assert runner.TaskCampaign.learning_source(job, plan["domains"][1]) == checkpoint


def test_assessment_file_is_not_consumed_in_v2_training(tmp_path, plans):
    job = continuous_job(tmp_path, plans[1])
    (tmp_path / "assessment_request.json").write_text("invalid content should never be read")
    job.assess_request(tmp_path)
    assert not (tmp_path / "assessment_decision.json").exists()


def test_continuation_trains_remaining_cap_once_and_preserves_three_clocks(tmp_path, monkeypatch, plans):
    _, plan = plans
    job = continuous_job(tmp_path, plan)
    source = tmp_path / "flat/attempt_000/train/model_final.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"sealed")
    runner.write_json(source.with_suffix(".contract.json"), read(plan["domains"][0]["contract"]))
    runner.write_json(source.with_suffix(".checkpoint.json"), dict(stage_actor_update=377,
        task_actor_update=9, actor_updates_consumed=989))
    job.records = {"flat/attempt_000/train":dict(domain="flat", consumed_updates=9, actor_updates=9)}
    job.report.update(campaign.account_updates(plan, job.records))
    monkeypatch.setattr(job, "learning_source", lambda _: source)
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda _: 21000)
    monkeypatch.setattr(job, "evaluate_pair", lambda *a: pytest.fail("must not run assessment"))
    calls = []

    def execute(command, log, directory):
        calls.append(command)
        assert command[command.index("--updates") + 1] == "19991"
        assert command[command.index("--consumed-updates") + 1] == "1009"
        assert command[command.index("--actor-updates-consumed") + 1] == "989"
        assert command[command.index("--task-actor-update") + 1] == "9"
        assert command[command.index("--stage-actor-update") + 1] == "377"
        runner.write_json(directory / "completion.json", dict(status="completed", successful_updates=19991,
            attempted_updates=19991, actor_updates_in_block=19991))
        (directory / "model_final.pt").write_bytes(b"trained candidate")
        return 0

    monkeypatch.setattr(job, "execute", execute)
    job.run_domain(plan["domains"][0])
    assert len(calls) == 1
    assert campaign.account_updates(plan, job.records)["consumed_updates"] == 21000
    assert job.report["domains"]["flat"]["domain_qualified"] is None


def test_offline_evaluation_does_not_start_training_or_change_consumption(tmp_path, monkeypatch, plans):
    _, plan = plans
    job = continuous_job(tmp_path, plan)
    row = plan["domains"][0]
    candidate = tmp_path / "model.pt"
    candidate.write_bytes(b"candidate")
    job.report["domains"]["flat"] = dict(candidate_checkpoint=str(candidate), domain_qualified=None)
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda _: 21000)
    monkeypatch.setattr(job, "execute", lambda *a: pytest.fail("offline evaluation cannot train"))
    monkeypatch.setattr(job, "evaluate_pair", lambda *a: [dict(cases={name:dict(passed=False)
        for name in row["required_case_names"]})] * 2)
    before = deepcopy(job.records)
    job.evaluate_finished_domains()
    assert job.records == before
    assert job.report["domains"]["flat"]["assessment_status"] == "completed"
    assert job.report["domains"]["flat"]["domain_qualified"] is False


def test_revision_account_preserves_progress_newer_than_supervisor_and_rejects_repeat_training(tmp_path, monkeypatch, plans):
    before, after = plans
    job = continuous_job(tmp_path, after)
    job.args.contract = ROOT / "contracts/v6_task_campaign_v2.json"
    job.claim = tmp_path / "account.json"
    job.assert_no_live_workers = lambda: None
    runner.write_json(job.claim, dict(campaign_id=campaign.CAMPAIGN_ID, run_directory=str(tmp_path),
        contract_sha256=campaign.digest(ROOT / "contracts/v6_task_campaign_v1.json"),
        attempts={"flat/attempt_000/train":dict(domain="flat", consumed_updates=8, actor_updates=8)}))
    runner.write_json(tmp_path / "flat/attempt_000/train/progress.json",
                      dict(successful_updates=10, attempted_updates=11, actor_updates_in_block=10))
    monkeypatch.setattr(runner, "validate_plan_migration", lambda *a: dict(
        source_plan_sha256=campaign.digest(ROOT / "contracts/v6_task_campaign_v1.json"),
        target_plan_sha256=campaign.digest(job.args.contract)))
    job.open_account()
    account = campaign.account_updates(after, job.records)
    assert account["consumed_updates"] == 1011
    assert account["actor_updates_consumed"] == 990
    assert job.report["revision_migration"]["domain_consumed_updates"]["flat"] == 11
    assert (tmp_path / "revisions/v1_account_before_migration.json").exists()
