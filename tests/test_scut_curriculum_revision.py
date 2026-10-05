"""Exact SCUT v17 recipes and atomic own-state continuation, with no refunds."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis import scut_curriculum_revision as revision
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.locomotion_revision import FLAT_FILE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner


def read(file, root=ROOT):
    return json.loads((root / file).read_text())


def write(file, value):
    file.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def plans():
    return read(revision.SOURCE_PLAN_FILE), read(revision.PLAN_FILE)


@pytest.fixture
def bindings(tmp_path, monkeypatch):
    # Copy only the recipe closure, not the several-GiB historical corpus.
    root = tmp_path / "bindings"
    source, target = plans()
    files = {revision.SOURCE_PLAN_FILE, revision.PLAN_FILE, revision.MANIFEST_FILE, FLAT_FILE}
    files.update(row["contract"] for plan in (source, target) for row in plan["domains"])
    for domain in revision.DOMAINS_TO_REVISE:
        source_file, _, _, design_file, *_ = revision._recipe(domain)
        files.update((source_file, design_file, read(source_file)["design_preflight"]["file"]))
    for file in files:
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / file, path)
    monkeypatch.setattr(campaign, "ROOT", root)
    yield root
    shutil.rmtree(root)


def test_frozen_plan_and_manifest_are_exact_canonical_objects():
    source, target = plans()
    assert campaign.digest(ROOT / revision.SOURCE_PLAN_FILE) == revision.SOURCE_PLAN_SHA
    assert target == revision.revised_plan(ROOT)
    manifest = read(revision.MANIFEST_FILE)
    assert manifest == revision.revised_manifest(ROOT)
    assert manifest["source_plan_file"] == "contracts/v6_task_campaign_v15.json"
    assert manifest["target_plan_sha256"] == campaign.digest(ROOT / revision.PLAN_FILE)
    assert manifest["critic_adaptation_updates"] == 10
    assert len(revision.validate_frozen_workers(ROOT)) == 4
    receipt = campaign.validate_plan_migration(source, target, manifest["run_directory"])
    assert receipt["actor_critic_optimizer"] == "preserved"
    assert receipt["consumption"] == "preserved_no_refund"


def test_only_four_workers_and_two_completed_endpoints_change():
    source, target = plans()
    changed = {row["name"] for old, row in zip(source["domains"], target["domains"], strict=True)
               if old != row}
    assert changed == set(revision.DOMAINS_TO_REVISE)
    assert target["domain_completion"]["schedule"]["end_actor_updates"] == revision.ENDPOINTS
    assert target["target_num_envs"] == 4096
    assert target["total_ceiling"] == 121000
    for name in ("flat", "speed"):
        assert next(row for row in source["domains"] if row["name"] == name) == next(
            row for row in target["domains"] if row["name"] == name)
    assert [(row["name"], row["parent_domain"]) for row in source["domains"]] == [
        (row["name"], row["parent_domain"]) for row in target["domains"]]
    for domain in revision.DOMAINS_TO_REVISE:
        worker = read(revision._recipe(domain)[2])
        assert worker == revision.canonical_worker(domain)
        assert worker["learning_rate"] == 1e-4 and worker["learning_rate_schedule"] == "fixed"
        assert worker["physics_dt"] == .005 and worker["policy_dt"] == .02
        assert worker["task_campaign"]["clock"] == "domain_actor_updates"
        assert worker["actuator_response"]["enabled"] is True


@pytest.mark.parametrize("domain", revision.DOMAINS_TO_REVISE)
@pytest.mark.parametrize("fault", ("physics", "lr", "task_clock", "criteria", "sampling"))
def test_rehashed_target_worker_cannot_authorize_unreviewed_changes(bindings, domain, fault):
    file = revision._recipe(domain)[2]
    worker = read(file, bindings)
    if fault == "physics":
        worker["physics_dt"] = .001
    elif fault == "lr":
        worker["learning_rate"] = .0005
    elif fault == "task_clock":
        worker["task_campaign"]["clock"] = "global_consumed_updates"
    elif fault == "criteria":
        worker["evaluation"]["cases"][0]["name"] += "_changed"
    else:
        worker["scene_groups"][0]["fraction"] += .01
    write(bindings / file, worker)
    with pytest.raises(ValueError, match="exact recipe"):
        revision.validate_frozen_workers(bindings)


@pytest.mark.parametrize("fault", ("endpoint", "parent", "budget", "envs", "new_gate", "scope", "source"))
def test_manifest_reseal_cannot_expand_authorized_plan(bindings, fault):
    source, target = read(revision.SOURCE_PLAN_FILE, bindings), read(revision.PLAN_FILE, bindings)
    manifest = read(revision.MANIFEST_FILE, bindings)
    if fault == "endpoint":
        target["domain_completion"]["schedule"]["end_actor_updates"]["jump"] += 1
    elif fault == "parent":
        target["domains"][3]["parent_domain"] = "speed"
    elif fault == "budget":
        target["domains"][1]["updates"] += 10
    elif fault == "envs":
        target["target_num_envs"] = 20000
    elif fault == "new_gate":
        target["domain_completion"]["schedule"]["advance_requires_evaluation"] = True
    elif fault == "scope":
        manifest["curriculum_scope"] = "anything"
    else:
        manifest["source_plan_file"] = "contracts/v6_task_campaign_v16.json"
    write(bindings / revision.PLAN_FILE, target)
    manifest["target_plan_sha256"] = campaign.digest(bindings / revision.PLAN_FILE)
    write(bindings / revision.MANIFEST_FILE, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        campaign.validate_plan_migration(source, target, manifest["run_directory"])


def checkpoint(root, domain, *, actor=None, count=None, attempt=0, worker_file=None):
    actor = revision.ORIGINS[domain] if actor is None else actor
    count = (16334 if domain == "recovery" else 26388) if count is None else count
    file, sha, *_ = revision._recipe(domain)
    if worker_file is not None:
        file, sha = worker_file, campaign.digest(ROOT / worker_file)
    worker = read(file)
    path = root / domain / f"attempt_{attempt:03}" / "train/model_final.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    infos = dict(contract_sha256=sha, task_actor_update=actor,
        stage_actor_update=13000 + actor, successful_updates_total=count,
        actor_updates_consumed=count - 130, critic_warmup_remaining=0,
        task_campaign=deepcopy(worker["task_campaign"]))
    torch.save(dict(infos=infos, policy=torch.tensor([1.])), path)
    shutil.copyfile(ROOT / file, path.with_suffix(".contract.json"))
    write(path.with_suffix(".checkpoint.json"), dict(schema="chassis-learning-checkpoint-v1",
        sealed=True, checkpoint_sha256=campaign.digest(path), contract_sha256=sha,
        learning_updates=count, learning_lineage_updates=13050 + actor,
        actor_updates_consumed=count - 130, task_actor_update=actor,
        stage_actor_update=infos["stage_actor_update"],
        task_campaign=deepcopy(worker["task_campaign"])))
    return path


@pytest.fixture
def completed(tmp_path):
    source, target = plans()
    paths = {domain: checkpoint(tmp_path, domain) for domain in revision.ORIGINS}
    records = {f"{name}/attempt_000/train": dict(domain=name, consumed_updates=charge,
        actor_updates=actor) for name, charge, actor in (
            ("flat", 13020, 13000), ("recovery", 2314, 2294), ("speed", 5093, 5063),
            ("terrain", 2206, 2186), ("jump", 5377, 5357), ("stepjump", 0, 0))}
    report = dict(contract_sha256=revision.SOURCE_PLAN_SHA, domains={name: dict(
        status="domain_schedule_finished_unassessed", domain_qualified=None,
        candidate_checkpoint="retained-" + name) for name in campaign.DOMAINS},
        evaluations=dict(recovery=dict(old="retained"), terrain=dict(old="retained")),
        consumed_review_tokens=["prior"], training_process_seconds=0., assessment_seconds=0.,
        elapsed_seconds=0., **campaign.account_updates(source, records))
    for domain, path in paths.items():
        report["domains"][domain].update(candidate_checkpoint=str(path),
            candidate_checkpoint_sha256=campaign.digest(path), final_checkpoint=str(path),
            finish_review=dict(token=domain + "-finish"), pending_review=dict(token=domain + "-pending"),
            schedule_completion=dict(actual_actor_update=revision.ORIGINS[domain]),
            completion_history=[dict(plan_revision="v12", record=dict(old="retained"))])
    return SimpleNamespace(root=tmp_path, paths=paths, source=source, target=target,
                           records=records, report=report)


def reopen(completed):
    return revision.reopen_completed_domains(completed.report, completed.records,
        completed.source, completed.target, completed.root)


def test_both_reopenings_preserve_all_consumption_history_review_and_learning_sources(completed):
    original, records = deepcopy(completed.report), deepcopy(completed.records)
    receipt = reopen(completed)
    assert receipt["consumption"] == "preserved_no_refund"
    assert completed.records == records
    assert completed.report["evaluations"] == original["evaluations"]
    for domain, origin in revision.ORIGINS.items():
        current = completed.report["domains"][domain]
        assert current["status"] == "domain_reopened_unassessed"
        assert current["domain_qualified"] is None
        assert current["candidate_checkpoint"] == str(completed.paths[domain])
        assert current["reopening"]["origin_actor_update"] == origin
        assert current["reopening"]["actor_updates_added"] == 15000 - origin
        assert current["reopening"]["critic_adaptation_updates"] == 10
        assert current["completion_history"][-1] == dict(plan_revision="v15", record=original["domains"][domain])
        assert current["completion_history"][:-1] == original["domains"][domain]["completion_history"]
        assert not {"finish_review", "pending_review", "schedule_completion"} & current.keys()
    for name in ("flat", "speed", "jump", "stepjump"):
        assert completed.report["domains"][name] == original["domains"][name]
    assert set(completed.report["consumed_review_tokens"]) == {
        "prior", "recovery-finish", "recovery-pending", "terrain-finish", "terrain-pending"}
    assert completed.report["consumed_updates"] == original["consumed_updates"]
    assert completed.report["actor_updates_consumed"] == original["actor_updates_consumed"]
    once = deepcopy(completed.report)
    assert reopen(completed) == receipt
    assert completed.report == once
    # Completing a supplement must not be undone by same-version startup.
    completed.report["domains"]["recovery"]["status"] = "domain_schedule_finished_unassessed"
    completed.report["domains"]["recovery"]["schedule_completion"] = dict(actual_actor_update=15000)
    finished = deepcopy(completed.report)
    assert reopen(completed) == receipt
    assert completed.report == finished


@pytest.mark.parametrize("fault", ("second_unsealed", "second_qualified", "second_older",
    "wrong_clock", "float_clock", "wrong_campaign", "wrong_source", "outside_branch",
    "hash", "actor_refund", "domain_cap", "missing_domain"))
def test_bad_second_branch_or_ledger_is_atomic_and_never_reopens_only_first(completed, fault):
    domain = "terrain"
    current = completed.report["domains"][domain]
    path = completed.paths[domain]
    seal_path = path.with_suffix(".checkpoint.json")
    seal = json.loads(seal_path.read_text())
    if fault == "second_unsealed":
        seal["sealed"] = False
    elif fault == "second_qualified":
        current["domain_qualified"] = True
    elif fault == "second_older":
        checkpoint(completed.root, domain, actor=2187, count=26389, attempt=1)
    elif fault in ("wrong_clock", "float_clock"):
        seal["task_actor_update"] = 2187 if fault == "wrong_clock" else 2186.0
    elif fault == "wrong_campaign":
        seal["task_campaign"]["campaign_id"] = "other"
    elif fault == "wrong_source":
        current["candidate_checkpoint"] = str(completed.paths["recovery"])
        current["candidate_checkpoint_sha256"] = campaign.digest(completed.paths["recovery"])
    elif fault == "outside_branch":
        other = completed.root / "unrelated/model.pt"
        other.parent.mkdir(); shutil.copyfile(path, other)
        current["candidate_checkpoint"] = str(other)
        current["candidate_checkpoint_sha256"] = campaign.digest(other)
    elif fault == "hash":
        current["candidate_checkpoint_sha256"] = "0" * 64
    elif fault == "actor_refund":
        for record in completed.records.values():
            record["actor_updates"] = 0
    elif fault == "domain_cap":
        completed.records["terrain/attempt_000/train"]["consumed_updates"] = 19000
    else:
        del completed.report["domains"][domain]
    write(seal_path, seal)
    before = deepcopy(completed.report)
    with pytest.raises(ValueError):
        reopen(completed)
    assert completed.report == before


def test_source_plan_and_reopening_history_cannot_be_replaced(completed):
    before = deepcopy(completed.report)
    completed.source["contract_id"] = "v6-task-campaign-v16"
    with pytest.raises(ValueError, match="exact v15 to v17"):
        reopen(completed)
    assert completed.report == before
    completed.source = plans()[0]
    reopen(completed)
    completed.report["domains"]["terrain"]["completion_history"].clear()
    before = deepcopy(completed.report)
    with pytest.raises(ValueError, match="lost its completed"):
        reopen(completed)
    assert completed.report == before


def account_job(completed, monkeypatch):
    job = object.__new__(runner.TaskCampaign)
    job.plan, job.root = deepcopy(completed.target), completed.root
    job.args = SimpleNamespace(contract=ROOT / revision.PLAN_FILE, device="cpu")
    job.claim = completed.root / "shared/account.json"
    job.report, job.records, job._session_started = {}, {}, None
    monkeypatch.setattr(job, "assert_no_live_workers", lambda: None)

    def authenticate(source, target, root):
        assert root == completed.root
        result = campaign.validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
        result["run_directory"] = str(root)
        return result

    monkeypatch.setattr(runner, "validate_plan_migration", authenticate)
    return job


def previous_account(completed, job):
    runner.write_json(completed.root / "campaign_state.json", completed.report)
    runner.write_json(job.claim, dict(campaign_id=campaign.CAMPAIGN_ID,
        run_directory=str(completed.root), contract_sha256=revision.SOURCE_PLAN_SHA,
        attempts=deepcopy(completed.records), **campaign.account_updates(completed.source, completed.records)))


def test_native_account_migration_charges_once_and_same_version_restart_keeps_both_reopenings(completed, monkeypatch):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    job.open_account()
    once = deepcopy(job.report)
    assert job.records == completed.records
    assert job.report["consumed_updates"] == completed.report["consumed_updates"]
    for domain, origin in revision.ORIGINS.items():
        assert job.report["domains"][domain]["reopening"]["origin_actor_update"] == origin
    job.open_account()
    assert job.report == once and job.records == completed.records


def test_native_crash_after_claim_commit_replays_two_authenticated_reopenings(completed, monkeypatch):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    write = runner.write_json

    def power_loss(path, value):
        if Path(path) == completed.root / "campaign_state.json":
            raise RuntimeError("simulated power loss after claim commit")
        return write(path, value)

    monkeypatch.setattr(runner, "write_json", power_loss)
    with pytest.raises(RuntimeError, match="power loss"):
        job.open_account()
    assert json.loads(job.claim.read_text())["contract_sha256"] == campaign.digest(ROOT / revision.PLAN_FILE)
    assert read("campaign_state.json", completed.root)["contract_sha256"] == revision.SOURCE_PLAN_SHA
    monkeypatch.setattr(runner, "write_json", write)
    resumed = account_job(completed, monkeypatch)
    resumed.open_account()
    for domain in revision.ORIGINS:
        assert resumed.report["domains"][domain]["status"] == "domain_reopened_unassessed"
        assert len(resumed.report["domains"][domain]["completion_history"]) == 2
    assert resumed.records == completed.records
    assert resumed.report["consumed_updates"] == completed.report["consumed_updates"]
    once = deepcopy(resumed.report)
    resumed.open_account()
    assert resumed.report == once


@pytest.mark.parametrize("domain", revision.ORIGINS)
def test_native_scheduler_continues_own_actor_clock_and_charges_ten_new_critic_updates(completed, monkeypatch, domain):
    reopen(completed)
    job = object.__new__(runner.TaskCampaign)
    job.plan, job.root = deepcopy(completed.target), completed.root
    job.report, job.records = completed.report, completed.records
    job.args = SimpleNamespace(device="cpu")
    job.deadline, job.stop_requested = float("inf"), False
    job.persist = lambda: job.report.update(campaign.account_updates(job.plan, job.records))
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: pytest.fail("continuous schedule must not evaluate"))
    origin = revision.ORIGINS[domain]
    actor_updates, charge = 15000 - origin, 15000 - origin + 10
    row = next(row for row in job.plan["domains"] if row["name"] == domain)
    before = deepcopy(job.report)
    calls = []

    def execute(command, log, directory):
        calls.append(command)
        assert int(command[command.index("--updates") + 1]) == charge
        assert command[command.index("--resume") + 1] == str(completed.paths[domain])
        assert int(command[command.index("--task-actor-update") + 1]) == origin
        assert int(command[command.index("--consumed-updates") + 1]) == before["consumed_updates"]
        runner.write_json(directory / "completion.json", dict(status="completed", successful_updates=charge,
            attempted_updates=charge, actor_updates_in_block=actor_updates))
        checkpoint(completed.root, domain, actor=15000, count=before["consumed_updates"] + charge,
                   attempt=int(directory.parent.name.rsplit("_", 1)[1]), worker_file=row["contract"])
        return 0

    monkeypatch.setattr(job, "execute", execute)
    job.run_domain(row)
    assert len(calls) == 1
    current = job.report["domains"][domain]
    assert current["status"] == "domain_schedule_finished_unassessed"
    assert current["domain_qualified"] is None
    assert current["schedule_completion"]["actual_actor_update"] == 15000
    assert current["reopening"] == before["domains"][domain]["reopening"]
    assert sum(record["actor_updates"] for record in job.records.values() if record["domain"] == domain) == 15000
    assert job.report["domain_consumed_updates"][domain] - before["domain_consumed_updates"][domain] == charge
    job.run_domain(row)
    assert len(calls) == 1
