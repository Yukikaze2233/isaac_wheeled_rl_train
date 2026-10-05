"""Independent fault review of the v15-to-v17 four-worker campaign revision."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis import scut_curriculum_revision as revision
import run_task_campaign as runner
from chassis_remote_job import sha, validate_source_revision
from train_chassis import resume_critic_warmup


def read(file, root=ROOT):
    return json.loads((root / file).read_text())


SOURCE = read(revision.SOURCE_PLAN_FILE)
TARGET = read(revision.PLAN_FILE)


def checkpoint(root, domain, *, clock=None, count=None, worker=None, relative=None):
    clock = revision.ORIGINS[domain] if clock is None else clock
    count = dict(recovery=16000, terrain=23000)[domain] if count is None else count
    worker = revision._recipe(domain)[0] if worker is None else worker
    relative = f"{domain}/attempt_002/train/model_final.pt" if relative is None else relative
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = read(worker)
    infos = dict(contract_sha256=campaign.digest(ROOT / worker),
        task_campaign=deepcopy(contract["task_campaign"]), task_actor_update=clock,
        stage_actor_update=13000 + clock, successful_updates_total=count,
        actor_updates_consumed=count - 130, critic_warmup_remaining=0)
    torch.save(dict(infos=infos, policy=torch.tensor([1.])), path)
    shutil.copyfile(ROOT / worker, path.with_suffix(".contract.json"))
    seal = dict(schema="chassis-learning-checkpoint-v1", sealed=True,
        checkpoint_sha256=campaign.digest(path), learning_updates=count,
        learning_lineage_updates=13050 + clock, **{k: v for k, v in infos.items()
            if k not in ("successful_updates_total", "critic_warmup_remaining")})
    runner.write_json(path.with_suffix(".checkpoint.json"), seal)
    return path, infos


@pytest.fixture
def completed(tmp_path):
    records = {f"{name}/attempt_000/train": dict(domain=name, consumed_updates=charge,
        actor_updates=actor) for name, charge, actor in (
            ("flat", 13020, 13000), ("recovery", 2314, 2294), ("speed", 5093, 5063),
            ("terrain", 2206, 2186), ("jump", 5310, 5290), ("stepjump", 0, 0))}
    report = dict(contract_sha256=revision.SOURCE_PLAN_SHA, domains={name: dict(
        status="domain_schedule_finished_unassessed", domain_qualified=None,
        candidate_checkpoint="previous-" + name) for name in campaign.DOMAINS},
        evaluations=dict(terrain=dict(old="retained"), recovery=dict(old="retained")),
        consumed_review_tokens=["prior"], training_process_seconds=0.,
        assessment_seconds=0., elapsed_seconds=0., **campaign.account_updates(SOURCE, records))
    report["domains"]["jump"]["status"] = "running"
    report["domains"]["stepjump"] = {}
    paths, infos = {}, {}
    for domain, origin in revision.ORIGINS.items():
        path, info = checkpoint(tmp_path, domain)
        paths[domain], infos[domain] = path, info
        report["domains"][domain].update(candidate_checkpoint=str(path),
            candidate_checkpoint_sha256=campaign.digest(path), final_checkpoint=str(path),
            finish_review=dict(token=domain + "-finish"), pending_review=dict(token=domain + "-pending"),
            schedule_completion=dict(end_actor_update=origin, actual_actor_update=origin),
            completion_history=[dict(plan_revision="v12", record=dict(old="retained"))])
    return SimpleNamespace(root=tmp_path, paths=paths, infos=infos, records=records, report=report)


def test_exact_recipe_all_roles_parent_tree_and_no_v16_predecessor():
    receipt = campaign.validate_plan_migration(SOURCE, TARGET, TARGET["revision_migration"]["run_directory"])
    assert receipt["schema"] == "task-plan-revision-v17"
    assert receipt["consumption"] == "preserved_no_refund"
    assert len(runner.validate_contracts(TARGET)) == 30
    assert read(revision.PLAN_FILE) == revision.revised_plan(ROOT)
    assert read(revision.MANIFEST_FILE) == revision.revised_manifest(ROOT)
    assert TARGET["domain_completion"]["schedule"]["advance_requires_evaluation"] is False
    assert TARGET["total_ceiling"] == SOURCE["total_ceiling"] == 121000
    assert TARGET["target_num_envs"] == SOURCE["target_num_envs"] == 4096
    before = {row["name"]: row for row in SOURCE["domains"]}
    for row in TARGET["domains"]:
        if row["name"] in ("flat", "speed"):
            assert row == before[row["name"]]
        elif row["name"] == "stepjump":
            config = read(row["contract"])
            parent = next(r for r in TARGET["domains"] if r["name"] == "jump")
            assert config["task_campaign"]["source_contract_file"] == parent["contract"]
            assert config["task_campaign"]["source_contract_sha256"] == parent["sha256"]
    with pytest.raises(ValueError, match="named consecutive"):
        campaign.validate_plan_migration(read("contracts/v6_task_campaign_v16.json"), TARGET,
            TARGET["revision_migration"]["run_directory"])


@pytest.fixture
def bound_files(tmp_path, monkeypatch):
    # Link immutable dependencies; copy mutated files before touching their bytes.
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    for source in (ROOT / "contracts").iterdir():
        (contracts / source.name).symlink_to(source)
    for file in (revision.PLAN_FILE, revision.MANIFEST_FILE):
        path = tmp_path / file
        path.unlink()
        shutil.copyfile(ROOT / file, path)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("fault", ["recovery_endpoint", "jump_endpoint", "stepjump_endpoint", "parent",
    "sampling", "reward", "cap", "envs", "origin", "critic", "scope"])
def test_resealed_manifest_cannot_authorize_extra_changes(bound_files, fault):
    root = bound_files
    source, target = read(revision.SOURCE_PLAN_FILE, root), read(revision.PLAN_FILE, root)
    manifest = read(revision.MANIFEST_FILE, root)
    if fault.endswith("endpoint"):
        target["domain_completion"]["schedule"]["end_actor_updates"][fault.split("_")[0]] += 1
    elif fault == "parent":
        target["domains"][3]["parent_domain"] = "speed"
    elif fault == "sampling":
        target["domains"][3]["sampling_pools"][0]["fraction"] += .01
    elif fault == "reward":
        row = target["domains"][3]
        path = root / row["contract"]
        worker = json.loads(path.read_text())
        worker["dense_tracking"]["stationary_width_m_s"] = .8
        path.unlink()
        runner.write_json(path, worker)
        row["sha256"] = campaign.digest(path)
        manifest["workers"][3]["target_sha256"] = row["sha256"]
    elif fault == "cap":
        target["domains"][3]["updates"] += 1
    elif fault == "envs":
        target["target_num_envs"] = 20000
    elif fault == "origin":
        manifest["reopen_actor_origins"]["recovery"] -= 1
    elif fault == "critic":
        manifest["critic_adaptation_updates"] = 0
    else:
        manifest["curriculum_scope"] = "arbitrary_change"
    runner.write_json(root / revision.PLAN_FILE, target)
    manifest["target_plan_sha256"] = campaign.digest(root / revision.PLAN_FILE)
    runner.write_json(root / revision.MANIFEST_FILE, manifest)
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(source, target, manifest["run_directory"])


def test_both_domains_reopen_atomically_idempotently_with_history_and_consumption(completed):
    before = deepcopy(completed.report)
    records = deepcopy(completed.records)
    receipt = revision.reopen_completed_domains(completed.report, records, SOURCE, TARGET, completed.root)
    for domain, origin in revision.ORIGINS.items():
        record = completed.report["domains"][domain]
        assert record["status"] == "domain_reopened_unassessed"
        assert record["completion_history"][-1] == dict(plan_revision="v15", record=before["domains"][domain])
        assert record["reopening"]["origin_actor_update"] == origin
        assert record["reopening"]["actor_updates_added"] == 15000 - origin
        assert record["reopening"]["critic_adaptation_updates"] == 10
        assert not {"finish_review", "pending_review", "schedule_completion"} & record.keys()
    for domain in set(campaign.DOMAINS) - set(revision.ORIGINS):
        assert completed.report["domains"][domain] == before["domains"][domain]
    assert completed.report["evaluations"] == before["evaluations"]
    assert completed.report["consumed_updates"] == before["consumed_updates"]
    assert records == completed.records
    once = deepcopy(completed.report)
    assert revision.reopen_completed_domains(completed.report, records, SOURCE, TARGET, completed.root) == receipt
    assert completed.report == once


@pytest.mark.parametrize("fault", ["second_missing", "second_qualified", "second_hash", "second_unsealed",
    "second_wrong_actor", "second_newer", "first_domain_cap", "second_domain_cap", "campaign_cap"])
def test_invalid_either_branch_never_partially_reopens_first(completed, fault):
    if fault == "second_missing":
        del completed.report["domains"]["terrain"]
    elif fault == "second_qualified":
        completed.report["domains"]["terrain"]["domain_qualified"] = True
    elif fault == "second_hash":
        completed.report["domains"]["terrain"]["candidate_checkpoint_sha256"] = "0" * 64
    elif fault in ("second_unsealed", "second_wrong_actor"):
        path = completed.paths["terrain"].with_suffix(".checkpoint.json")
        seal = json.loads(path.read_text())
        if fault == "second_unsealed":
            seal["sealed"] = False
        else:
            seal["task_actor_update"] -= 1
        runner.write_json(path, seal)
    elif fault == "second_newer":
        checkpoint(completed.root, "terrain", clock=2187, count=23001,
            relative="terrain/attempt_003/train/checkpoints/update_23001/model.pt")
    elif fault in ("first_domain_cap", "second_domain_cap"):
        domain = "recovery" if fault == "first_domain_cap" else "terrain"
        completed.records[f"{domain}/attempt_000/train"]["consumed_updates"] = 19999
    else:
        # All six domain charges remain individually legal; supplements breach total.
        for record in completed.records.values():
            record["consumed_updates"] = 18000
    before = deepcopy(completed.report)
    records = deepcopy(completed.records)
    with pytest.raises(ValueError):
        revision.reopen_completed_domains(completed.report, records, SOURCE, TARGET, completed.root)
    assert completed.report == before
    assert records == completed.records


def account_job(completed, monkeypatch):
    job = object.__new__(runner.TaskCampaign)
    job.plan, job.root = deepcopy(TARGET), completed.root
    job.args = SimpleNamespace(contract=ROOT / revision.PLAN_FILE, device="cpu")
    job.claim = completed.root / "shared/account.json"
    job.report, job.records, job._session_started = {}, {}, None
    monkeypatch.setattr(job, "assert_no_live_workers", lambda: None)

    def authenticate(source, target, root):
        assert root == completed.root
        result = campaign.validate_plan_migration(source, target, TARGET["revision_migration"]["run_directory"])
        result["run_directory"] = str(root)
        return result

    monkeypatch.setattr(runner, "validate_plan_migration", authenticate)
    return job


def previous_account(completed, job):
    runner.write_json(completed.root / "campaign_state.json", completed.report)
    runner.write_json(job.claim, dict(campaign_id=campaign.CAMPAIGN_ID,
        run_directory=str(completed.root), contract_sha256=revision.SOURCE_PLAN_SHA,
        attempts=deepcopy(completed.records), **campaign.account_updates(SOURCE, completed.records)))


def test_claim_then_state_crash_replays_both_domains_once(completed, monkeypatch):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    write = runner.write_json

    def power_loss(path, value):
        if Path(path) == completed.root / "campaign_state.json":
            raise RuntimeError("power loss after claim commit")
        return write(path, value)

    monkeypatch.setattr(runner, "write_json", power_loss)
    with pytest.raises(RuntimeError, match="power loss"):
        job.open_account()
    assert read("shared/account.json", completed.root)["contract_sha256"] == campaign.digest(ROOT / revision.PLAN_FILE)
    assert read("campaign_state.json", completed.root)["contract_sha256"] == revision.SOURCE_PLAN_SHA
    monkeypatch.setattr(runner, "write_json", write)
    resumed = account_job(completed, monkeypatch)
    resumed.open_account()
    for domain in revision.ORIGINS:
        record = resumed.report["domains"][domain]
        assert record["status"] == "domain_reopened_unassessed"
        assert len(record["completion_history"]) == 2
    assert resumed.records == completed.records
    assert resumed.report["consumed_updates"] == completed.report["consumed_updates"]
    once = deepcopy(resumed.report)
    resumed.open_account()
    assert resumed.report == once


@pytest.mark.parametrize("fault", ["source_plan", "recovery_source", "terrain_source", "missing_reopening"])
def test_claim_replay_rejects_receipt_or_partial_state_tampering(completed, monkeypatch, fault):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    job.open_account()
    if fault == "missing_reopening":
        state = read("campaign_state.json", completed.root)
        del state["domains"]["terrain"]["reopening"]
        runner.write_json(completed.root / "campaign_state.json", state)
    else:
        runner.write_json(completed.root / "campaign_state.json", completed.report)
        path = completed.root / "revisions/v15_to_v17_account_migration.json"
        receipt = json.loads(path.read_text())
        if fault == "source_plan":
            receipt["source_plan_sha256"] = "0" * 64
        else:
            domain = fault.removesuffix("_source")
            receipt["domain_reopening"]["reopening"][domain]["source_checkpoint_sha256"] = "0" * 64
        runner.write_json(path, receipt)
    before = read("campaign_state.json", completed.root)
    resumed = account_job(completed, monkeypatch)
    with pytest.raises(ValueError, match="recovery|lost its durable"):
        resumed.open_account()
    assert read("campaign_state.json", completed.root) == before


@pytest.mark.parametrize("domain", ["recovery", "terrain"])
def test_current_v17_restart_reauthenticates_exact_reopening_source(completed, monkeypatch, domain):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    job.open_account()
    state = read("campaign_state.json", completed.root)
    state["domains"][domain]["reopening"]["source_checkpoint_sha256"] = "0" * 64
    runner.write_json(completed.root / "campaign_state.json", state)
    resumed = account_job(completed, monkeypatch)
    with pytest.raises(ValueError, match="authenticates|source|recovery"):
        resumed.open_account()
    assert read("campaign_state.json", completed.root) == state


def test_stale_v16_queue_cannot_launch_after_v17_owns_job_or_study_plan(tmp_path):
    from test_task_campaign_continuation_queue import fixture as queue_fixture
    from queue_task_campaign_continuation import read_json
    fixture = queue_fixture.__wrapped__(tmp_path)
    previous = read_json(fixture.root / "job.json")
    previous["command"][previous["command"].index("--contract") + 1] = str(
        fixture.old / "contracts/v6_task_campaign_v17.json")
    # Queue must reject before creating any v16 intent, receipt or process.
    runner.write_json(fixture.root / "job.json", previous)
    result = fixture.queue().step()
    assert result["status"] == "attention"
    assert "authenticated frozen v15 predecessor" in result["detail"]
    assert fixture.calls == []
    assert not (fixture.root / ".task_campaign_continuation_v16.intent.json").exists()


def test_fault_resume_keeps_newer_learning_and_uncertain_reserved_updates(completed, monkeypatch):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    job.open_account()
    file = revision._recipe("terrain")[2]
    source, infos = checkpoint(completed.root, "terrain", clock=2193,
        count=job.report["consumed_updates"] + 17, worker=file,
        relative="terrain/attempt_003/train/model_final.pt")
    runner.write_json(source.parent / "progress.json",
        dict(successful_updates=17, attempted_updates=18, actor_updates_in_block=7))
    resumed = account_job(completed, monkeypatch)
    resumed.open_account()
    row = next(row for row in TARGET["domains"] if row["name"] == "terrain")
    assert resumed.learning_source(row) == source
    assert resumed.report["consumed_updates"] == completed.report["consumed_updates"] + 18
    assert resumed.report["actor_updates_consumed"] == completed.report["actor_updates_consumed"] + 7
    assert resume_critic_warmup(read(file), infos, contract_sha256=campaign.digest(ROOT / file), task_change=True) == 0
    assert len(resumed.report["domains"]["terrain"]["completion_history"]) == 2


@pytest.mark.parametrize("domain", ["recovery", "terrain"])
def test_scheduler_runs_remaining_actor_plus_ten_critic_without_evaluation_gate(completed, monkeypatch, domain):
    from test_task_campaign_v2 import continuous_job
    revision.reopen_completed_domains(completed.report, completed.records, SOURCE, TARGET, completed.root)
    job = continuous_job(completed.root, TARGET)
    job.report, job.records = completed.report, completed.records
    job.persist = lambda: job.report.update(campaign.account_updates(TARGET, job.records))
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: pytest.fail("continuous PPO cannot run score gate"))
    row = next(row for row in TARGET["domains"] if row["name"] == domain)
    actor = 15000 - revision.ORIGINS[domain]
    updates = actor + 10
    calls = []

    def execute(command, log, directory):
        calls.append(command)
        assert int(command[command.index("--updates") + 1]) == updates
        assert command[command.index("--resume") + 1] == str(completed.paths[domain])
        assert int(command[command.index("--task-actor-update") + 1]) == revision.ORIGINS[domain]
        runner.write_json(directory / "completion.json", dict(status="completed", successful_updates=updates,
            attempted_updates=updates, actor_updates_in_block=actor))
        checkpoint(completed.root, domain, clock=15000, count=job.report["consumed_updates"] + updates,
            worker=row["contract"], relative=str((directory / "model_final.pt").relative_to(completed.root)))
        return 0

    monkeypatch.setattr(job, "execute", execute)
    job.run_domain(row)
    assert len(calls) == 1
    assert job.report["domains"][domain]["status"] == "domain_schedule_finished_unassessed"
    assert job.report["domains"][domain]["domain_qualified"] is None
    assert job.report["domain_consumed_updates"][domain] == completed.records[f"{domain}/attempt_000/train"]["consumed_updates"] + updates
    job.run_domain(row)
    assert len(calls) == 1


def test_remote_v17_receipt_preserves_exact_argv_archive_and_previous_job(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    for directory, file in ((old, revision.SOURCE_PLAN_FILE), (new, revision.PLAN_FILE)):
        (directory / "contracts").mkdir(parents=True)
        shutil.copyfile(ROOT / file, directory / file)
        (directory / "scripts").mkdir()
        (directory / "scripts/run_task_campaign.py").write_text("# Never executed.\n")
    command = [sys.executable, "-B", str(old / "scripts/run_task_campaign.py"), "--contract",
        str(old / revision.SOURCE_PLAN_FILE), "--num-envs", "4096", "--device", "cuda:0", "--research",
        "--max-runtime-seconds", "15552000", "--run-dir", str(tmp_path / "train")]
    previous = dict(pid=2147483647, command=command)
    runner.write_json(tmp_path / "job.json", previous)
    revised = list(command)
    revised[2], revised[4] = str(new / "scripts/run_task_campaign.py"), str(new / revision.PLAN_FILE)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted(new.rglob("*")):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(new)), recursive=False)
    receipt = dict(schema="task-plan-revision-v17", previous_job_sha256=sha(tmp_path / "job.json"),
        command=revised, source_commit="a" * 40, old_source_root=str(old), new_source_root=str(new),
        new_contract_relative_path=revision.PLAN_FILE, old_contract_sha256=sha(old / revision.SOURCE_PLAN_FILE),
        new_contract_sha256=sha(new / revision.PLAN_FILE), source_archive=str(archive), source_archive_sha256=sha(archive))
    path = tmp_path / "revision.json"
    runner.write_json(path, receipt)
    calls = []

    def validate(before, after, root):
        assert before == SOURCE and after == TARGET and root == tmp_path / "train"
        calls.append(True)

    monkeypatch.setattr(campaign, "validate_plan_migration", validate)
    assert validate_source_revision(tmp_path, revised, previous, path) == receipt
    assert calls == [True]
    revised[revised.index("--num-envs") + 1] = "20000"
    receipt["command"] = revised
    runner.write_json(path, receipt)
    with pytest.raises(ValueError, match="only runner and contract"):
        validate_source_revision(tmp_path, revised, previous, path)
