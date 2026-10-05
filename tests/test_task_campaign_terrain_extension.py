"""Terrain v16 reopens one finished branch without rewinding or refunding it."""
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
from wheeled_tasks.chassis import terrain_schedule_revision as revision
from train_chassis import resume_critic_warmup
import run_task_campaign as runner
from chassis_remote_job import sha, validate_source_revision


def read(file, root=ROOT):
    return json.loads((root / file).read_text())


SOURCE = read(revision.SOURCE_PLAN_FILE)
TARGET = read(revision.PLAN_FILE)


def checkpoint(root, *, clock=2186, count=26388, worker=revision.WORKER_FILE,
               relative="terrain/attempt_002/train/model_final.pt"):
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
    path, infos = checkpoint(tmp_path)
    records = {f"{name}/attempt_000/train": dict(domain=name, consumed_updates=charge,
        actor_updates=actor) for name, charge, actor in (
            ("flat", 13020, 13000), ("recovery", 2314, 2294), ("speed", 5093, 5063),
            ("terrain", 2206, 2186), ("jump", 15020, 15000), ("stepjump", 15010, 15000))}
    report = dict(contract_sha256=revision.SOURCE_PLAN_SHA, domains={name: dict(
        status="domain_schedule_finished_unassessed", domain_qualified=None,
        candidate_checkpoint="previous-" + name) for name in campaign.DOMAINS},
        evaluations=dict(terrain=dict(old="retained")), consumed_review_tokens=["prior"],
        training_process_seconds=0., assessment_seconds=0., elapsed_seconds=0.,
        **campaign.account_updates(SOURCE, records))
    report["domains"]["terrain"].update(candidate_checkpoint=str(path),
        candidate_checkpoint_sha256=campaign.digest(path), final_checkpoint=str(path),
        finish_review=dict(token="old-finish"), pending_review=dict(token="old-pending"),
        schedule_completion=dict(end_actor_update=2186, actual_actor_update=2186),
        completion_history=[dict(plan_revision="v12", record=dict(old="retained"))])
    return SimpleNamespace(root=tmp_path, checkpoint=path, infos=infos,
                           records=records, report=report)


def test_exact_plan_preserves_six_worker_bytes_parents_and_remaining_endpoints():
    receipt = campaign.validate_plan_migration(SOURCE, TARGET,
        TARGET["revision_migration"]["run_directory"])
    assert receipt["consumption"] == "preserved_no_refund"
    assert campaign.digest(ROOT / revision.SOURCE_PLAN_FILE) == revision.SOURCE_PLAN_SHA
    assert SOURCE["domains"] == TARGET["domains"]
    assert TARGET["domain_completion"]["schedule"]["end_actor_updates"] == revision.ENDPOINTS
    assert revision.ENDPOINTS["jump"] == revision.ENDPOINTS["stepjump"] == 15000
    assert len(runner.validate_contracts(TARGET)) == 30
    for row in TARGET["domains"]:
        assert campaign.digest(ROOT / row["contract"]) == row["sha256"]
    assert TARGET["total_ceiling"] == SOURCE["total_ceiling"] == 121000
    assert TARGET["target_num_envs"] == 4096


@pytest.fixture
def bound_files(tmp_path, monkeypatch):
    for name in {revision.SOURCE_PLAN_FILE, revision.PLAN_FILE, revision.MANIFEST_FILE,
                 *(row["contract"] for row in TARGET["domains"])}:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("fault", ["terrain_endpoint", "jump_endpoint", "stepjump_endpoint",
    "parent", "sampling", "worker", "cap", "envs", "reopen_speed", "origin",
    "scope", "source_bytes", "source_plan"])
def test_resealing_manifest_does_not_authorize_extra_changes(bound_files, fault):
    root = bound_files
    source, target = read(revision.SOURCE_PLAN_FILE, root), read(revision.PLAN_FILE, root)
    manifest = read(revision.MANIFEST_FILE, root)
    if fault.endswith("endpoint"):
        target["domain_completion"]["schedule"]["end_actor_updates"][fault.split("_")[0]] += 1
    elif fault == "parent":
        target["domains"][3]["parent_domain"] = "speed"
    elif fault == "sampling":
        target["domains"][3]["sampling_pools"][0]["fraction"] += .01
    elif fault == "worker":
        path = root / target["domains"][3]["contract"]
        path.write_bytes(path.read_bytes() + b" ")
        target["domains"][3]["sha256"] = campaign.digest(path)
        manifest["workers"][3]["target_sha256"] = campaign.digest(path)
    elif fault == "cap":
        target["domains"][3]["updates"] += 1
    elif fault == "envs":
        target["target_num_envs"] = 20000
    elif fault == "reopen_speed":
        manifest["reopen_actor_origins"]["speed"] = 5063
    elif fault == "origin":
        manifest["reopen_actor_origins"]["terrain"] = 1186
    elif fault == "scope":
        manifest["schedule_scope"] = "arbitrary_changes"
    elif fault == "source_bytes":
        path = root / revision.SOURCE_PLAN_FILE
        path.write_bytes(path.read_bytes() + b" ")
        manifest["source_plan_sha256"] = target["revision_migration"]["source_plan_sha256"] = campaign.digest(path)
    else:
        source["contract_id"] = campaign.REVISION_V14
    runner.write_json(root / revision.PLAN_FILE, target)
    manifest["target_plan_sha256"] = campaign.digest(root / revision.PLAN_FILE)
    runner.write_json(root / revision.MANIFEST_FILE, manifest)
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(source, target, manifest["run_directory"])


def test_single_terrain_reopening_is_append_only_idempotent_and_does_not_refund(completed):
    before = deepcopy(completed.report)
    attempts = deepcopy(completed.records)
    receipt = revision.reopen_completed_terrain(completed.report, completed.records,
        SOURCE, TARGET, completed.root)
    terrain = completed.report["domains"]["terrain"]
    assert terrain["status"] == "domain_reopened_unassessed"
    assert terrain["reopening"]["source_checkpoint"] == str(completed.checkpoint)
    assert terrain["reopening"]["actor_updates_added"] == 12814
    assert terrain["reopening"]["repeat_critic_adaptation"] is False
    assert terrain["completion_history"][:-1] == before["domains"]["terrain"]["completion_history"]
    assert terrain["completion_history"][-1] == dict(plan_revision="v15", record=before["domains"]["terrain"])
    assert not {"finish_review", "pending_review", "schedule_completion"} & terrain.keys()
    assert completed.report["consumed_review_tokens"] == ["prior", "old-finish", "old-pending"]
    assert completed.report["evaluations"] == before["evaluations"]
    assert completed.records == attempts
    assert campaign.account_updates(SOURCE, attempts) == campaign.account_updates(TARGET, attempts)
    for domain in set(campaign.DOMAINS) - {"terrain"}:
        assert completed.report["domains"][domain] == before["domains"][domain]
    once = deepcopy(completed.report)
    assert revision.reopen_completed_terrain(completed.report, attempts, SOURCE, TARGET,
        completed.root) == receipt
    assert completed.report == once
    config = read(revision.WORKER_FILE)
    assert resume_critic_warmup(config, completed.infos,
        contract_sha256=revision.WORKER_SHA, task_change=True) == 0


@pytest.mark.parametrize("fault", ["old_actor", "wrong_domain", "wrong_worker", "old_snapshot",
    "outside_branch", "candidate_hash", "qualified", "unsealed", "cap", "float_clock",
    "actor_refund", "wrong_campaign"])
def test_reopening_rejects_a_wrong_or_older_learning_source_without_mutating_state(completed, fault):
    terrain = completed.report["domains"]["terrain"]
    if fault == "old_actor":
        path, _ = checkpoint(completed.root, clock=1187)
        terrain["candidate_checkpoint_sha256"] = campaign.digest(path)
    elif fault == "wrong_domain":
        path = completed.checkpoint.with_suffix(".checkpoint.json")
        seal = json.loads(path.read_text())
        seal["task_campaign"]["domain"] = "speed"
        runner.write_json(path, seal)
    elif fault == "wrong_worker":
        path, _ = checkpoint(completed.root, worker="contracts/v6_task_campaign_terrain_worker_v14.json")
        terrain["candidate_checkpoint_sha256"] = campaign.digest(path)
    elif fault == "old_snapshot":
        checkpoint(completed.root, clock=2187, count=26389,
                   relative="terrain/attempt_003/train/checkpoints/update_26389/model.pt")
    elif fault == "outside_branch":
        path, _ = checkpoint(completed.root, relative="jump/attempt_000/train/model_final.pt")
        terrain.update(candidate_checkpoint=str(path), candidate_checkpoint_sha256=campaign.digest(path))
    elif fault == "candidate_hash":
        terrain["candidate_checkpoint_sha256"] = "0" * 64
    elif fault == "qualified":
        terrain["domain_qualified"] = True
    elif fault == "unsealed":
        path = completed.checkpoint.with_suffix(".checkpoint.json")
        seal = json.loads(path.read_text()); seal["sealed"] = False
        runner.write_json(path, seal)
    elif fault in ("float_clock", "wrong_campaign"):
        path = completed.checkpoint.with_suffix(".checkpoint.json")
        seal = json.loads(path.read_text())
        if fault == "float_clock":
            seal["task_actor_update"] = 2186.0
        else:
            seal["task_campaign"]["campaign_id"] = "other"
        runner.write_json(path, seal)
    elif fault == "actor_refund":
        for record in completed.records.values():
            record["actor_updates"] = 0
    else:
        completed.records["terrain/attempt_000/train"]["consumed_updates"] = 19000
    before = deepcopy(completed.report)
    with pytest.raises(ValueError):
        revision.reopen_completed_terrain(completed.report, completed.records, SOURCE, TARGET, completed.root)
    assert completed.report == before


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


def test_migration_and_same_version_restart_preserve_all_consumption_once(completed, monkeypatch):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    job.open_account()
    once = deepcopy(job.report)
    assert job.records == completed.records
    assert job.report["domains"]["terrain"]["reopening"]["origin_actor_update"] == 2186
    assert job.report["consumed_updates"] == completed.report["consumed_updates"]
    job.open_account()
    assert job.report == once and job.records == completed.records


def test_crash_after_claim_commit_replays_reopening_from_authenticated_receipt(completed, monkeypatch):
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
    assert resumed.report["domains"]["terrain"]["status"] == "domain_reopened_unassessed"
    assert resumed.records == completed.records
    assert len(resumed.report["domains"]["terrain"]["completion_history"]) == 2
    assert resumed.report["consumed_updates"] == completed.report["consumed_updates"]
    once = deepcopy(resumed.report)
    resumed.open_account()
    assert resumed.report == once


def test_native_scheduler_adds_12814_actor_updates_with_zero_critic_and_skips_finished_peers(completed, monkeypatch):
    from test_task_campaign_v2 import continuous_job
    report = deepcopy(completed.report)
    revision.reopen_completed_terrain(report, completed.records, SOURCE, TARGET, completed.root)
    job = continuous_job(completed.root, TARGET)
    job.report, job.records = report, completed.records
    job.persist = lambda: job.report.update(campaign.account_updates(TARGET, job.records))
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: pytest.fail("schedule must not evaluate"))
    calls = []

    def execute(command, log, directory):
        calls.append(command)
        assert int(command[command.index("--updates") + 1]) == 12814
        assert command[command.index("--resume") + 1] == str(completed.checkpoint)
        assert int(command[command.index("--task-actor-update") + 1]) == 2186
        assert int(command[command.index("--consumed-updates") + 1]) == report["consumed_updates"]
        runner.write_json(directory / "completion.json", dict(status="completed", successful_updates=12814,
            attempted_updates=12814, actor_updates_in_block=12814))
        final, _ = checkpoint(completed.root, clock=15000, count=report["consumed_updates"] + 12814,
                              relative=str((directory / "model_final.pt").relative_to(completed.root)))
        return 0

    monkeypatch.setattr(job, "execute", execute)
    row = next(row for row in TARGET["domains"] if row["name"] == "terrain")
    job.run_domain(row)
    assert len(calls) == 1
    count = runner.checkpoint_update_count
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda path:
        count(path) if Path(path).is_relative_to(completed.root) else 1)
    for peer in TARGET["domains"]:
        if peer["name"] != "terrain":
            job.run_domain(peer)
    assert len(calls) == 1
    terrain = job.report["domains"]["terrain"]
    assert terrain["status"] == "domain_schedule_finished_unassessed"
    assert terrain["schedule_completion"]["actual_actor_update"] == 15000
    assert terrain["consumed_updates"] == 15020
    assert terrain["domain_qualified"] is None
    assert terrain["completion_history"] == report["domains"]["terrain"]["completion_history"]
    finished = deepcopy(job.report)
    revision.reopen_completed_terrain(job.report, job.records, SOURCE, TARGET, completed.root)
    assert job.report == finished, "repeating the reopening must not undo a finished supplement"
    job.run_domain(row)
    assert len(calls) == 1


def test_native_extension_rejects_a_repeated_critic_warmup(completed, monkeypatch):
    from test_task_campaign_v2 import continuous_job
    revision.reopen_completed_terrain(completed.report, completed.records, SOURCE, TARGET, completed.root)
    job = continuous_job(completed.root, TARGET)
    job.report, job.records = completed.report, completed.records
    changed = dict(completed.infos, critic_warmup_remaining=10)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: dict(infos=changed))
    row = next(row for row in TARGET["domains"] if row["name"] == "terrain")
    with pytest.raises(ValueError, match="repeat critic"):
        job.run_domain(row)


def test_newer_progress_and_uncertain_reservations_remain_charged(completed, monkeypatch):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    progress = completed.root / "terrain/attempt_003/train/progress.json"
    runner.write_json(progress, dict(successful_updates=2, attempted_updates=3, actor_updates_in_block=2))
    job.open_account()
    assert job.records["terrain/attempt_003/train"] == dict(domain="terrain", consumed_updates=3, actor_updates=2)
    assert job.report["consumed_updates"] == completed.report["consumed_updates"] + 3
    assert job.report["actor_updates_consumed"] == completed.report["actor_updates_consumed"] + 2
    assert job.report["domains"]["terrain"]["reopening"]["origin_actor_update"] == 2186


@pytest.mark.parametrize("fault", ["source_plan_sha256", "source_checkpoint_sha256"])
def test_claim_state_recovery_rejects_a_tampered_migration_receipt(completed, monkeypatch, fault):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    before = deepcopy(completed.report)
    job.open_account()
    runner.write_json(completed.root / "campaign_state.json", before)
    path = completed.root / "revisions/v15_to_v16_account_migration.json"
    receipt = json.loads(path.read_text())
    if fault == "source_plan_sha256":
        receipt[fault] = "0" * 64
    else:
        receipt["domain_reopening"]["reopening"][fault] = "0" * 64
    runner.write_json(path, receipt)
    resumed = account_job(completed, monkeypatch)
    with pytest.raises(ValueError, match="recovery"):
        resumed.open_account()
    assert read("campaign_state.json", completed.root) == before


def test_current_claim_cannot_silently_skip_a_missing_terrain_reopening(completed, monkeypatch):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    job.open_account()
    report = read("campaign_state.json", completed.root)
    del report["domains"]["terrain"]["reopening"]
    runner.write_json(completed.root / "campaign_state.json", report)
    resumed = account_job(completed, monkeypatch)
    with pytest.raises(ValueError, match="lost its durable reopening"):
        resumed.open_account()


def test_fault_restart_keeps_newer_terrain_learning_and_reserved_consumption(completed, monkeypatch):
    job = account_job(completed, monkeypatch)
    previous_account(completed, job)
    job.open_account()
    source, infos = checkpoint(completed.root, clock=2193, count=job.report["consumed_updates"] + 7,
        relative="terrain/attempt_003/train/model_final.pt")
    runner.write_json(source.parent / "progress.json",
        dict(successful_updates=7, attempted_updates=8, actor_updates_in_block=7))
    resumed = account_job(completed, monkeypatch)
    resumed.open_account()
    row = next(row for row in TARGET["domains"] if row["name"] == "terrain")
    assert resumed.learning_source(row) == source
    assert resumed.report["domain_consumed_updates"]["terrain"] == 2214
    assert resumed.report["consumed_updates"] == completed.report["consumed_updates"] + 8
    assert resumed.report["actor_updates_consumed"] == completed.report["actor_updates_consumed"] + 7
    assert resume_critic_warmup(read(revision.WORKER_FILE), infos,
        contract_sha256=revision.WORKER_SHA, task_change=True) == 0
    assert len(resumed.report["domains"]["terrain"]["completion_history"]) == 2


def test_v16_source_receipt_keeps_job_archive_and_invocation_authentication(tmp_path, monkeypatch):
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
    receipt = dict(schema="task-plan-revision-v16", previous_job_sha256=sha(tmp_path / "job.json"),
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
