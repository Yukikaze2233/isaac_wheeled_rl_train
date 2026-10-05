"""Authenticate v15→v18 and replay three own branches without new charges."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis import ordinary_posture_revision as revision
from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner
from chassis_remote_job import sha, validate_source_revision
from train_chassis import resume_critic_warmup


def read(file, root=ROOT):
    return json.loads((root / file).read_text())


def write(file, value):
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def plans():
    return read(revision.SOURCE_PLAN_FILE), read(revision.PLAN_FILE)


@pytest.fixture
def bindings(tmp_path, monkeypatch):
    # Follow only the small SHA-bound recipe closure, never copy all contracts.
    root = tmp_path / "bindings"
    pending = {revision.SOURCE_PLAN_FILE, revision.PLAN_FILE, revision.MANIFEST_FILE,
        "contracts/v6_task_campaign_v17.json", "contracts/v6_task_campaign_migration_v17.json"}
    files = set()

    def references(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from references(item)
        elif isinstance(value, list):
            for item in value:
                yield from references(item)
        elif isinstance(value, str) and value.startswith("contracts/") and value.endswith(".json"):
            yield value

    while pending:
        file = pending.pop()
        if file in files:
            continue
        files.add(file)
        pending.update(set(references(read(file))) - files)
    for file in files:
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / file, path)
    monkeypatch.setattr(campaign, "ROOT", root)
    yield root
    shutil.rmtree(root)


def test_plan_authenticates_exact_five_worker_v15_revision_and_all_budgets():
    source, target = plans()
    assert target == revision.revised_plan(ROOT)
    assert read(revision.MANIFEST_FILE) == revision.revised_manifest(ROOT)
    result = campaign.validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
    assert result["schema"] == "task-plan-revision-v18"
    assert result["source_plan_sha256"] == revision.SOURCE_PLAN_SHA
    assert result["consumption"] == "preserved_no_refund"
    assert target["domain_completion"]["schedule"]["end_actor_updates"] == dict(
        flat=13000, recovery=15000, speed=15000, terrain=15000, jump=15000, stepjump=15000)
    assert target["total_ceiling"] == 121000 and target["target_num_envs"] == 4096
    assert all(row["updates"] == 20000 for row in target["domains"])
    changed = {a["name"] for a, b in zip(source["domains"], target["domains"], strict=True) if a != b}
    assert changed == set(revision.DOMAINS_TO_REVISE)
    assert source["domains"][0] == target["domains"][0]


def test_unlaunched_v17_cannot_be_used_as_deployed_predecessor():
    with pytest.raises(ValueError, match="named consecutive|deployed-predecessor"):
        campaign.validate_plan_migration(read("contracts/v6_task_campaign_v17.json"),
            read(revision.PLAN_FILE), read(revision.PLAN_FILE)["revision_migration"]["run_directory"])


@pytest.mark.parametrize("domain", revision.DOMAINS_TO_REVISE)
def test_each_same_domain_full_state_resume_is_canonical(domain):
    file, _ = revision.source_binding(domain, ROOT)
    target = read(revision.worker_file(domain))
    receipt = campaign.validate_task_resume(read(file), target)
    assert receipt["kind"] == "task_domain_recovery" and receipt["task_clock"] == "restored"
    assert target["ordinary_posture_curriculum"]["origin_actor_update"] == revision.TEACHING_ORIGINS[domain]


def test_stepjump_requires_new_jump_parent_while_own_source_remains_old_stepjump():
    target, parent = read(revision.worker_file("stepjump")), read(revision.worker_file("jump"))
    assert campaign.validate_task_resume(parent, target)["task_clock"] == "new_domain_zero"
    old_jump = next(row for row in plans()[0]["domains"] if row["name"] == "jump")["contract"]
    with pytest.raises(ValueError, match="learning origin|parent alias"):
        campaign.validate_task_resume(read(old_jump), target)


@pytest.mark.parametrize("domain", revision.DOMAINS_TO_REVISE)
@pytest.mark.parametrize("fault", ("physics", "lr", "task_clock", "push", "root_birth", "criteria", "binding"))
def test_worker_schema_and_rehash_cannot_authorize_any_unreviewed_field(bindings, domain, fault):
    file, _ = revision.source_binding(domain, bindings)
    target = read(revision.worker_file(domain), bindings)
    if fault == "physics":
        target["physics_dt"] = .001
    elif fault == "lr":
        target["learning_rate"] = .0005
    elif fault == "task_clock":
        target["task_campaign"]["clock"] = "global_consumed_updates"
    elif fault == "push":
        target["ordinary_posture_curriculum"]["push"]["start_speed_m_s"] = 0.
    elif fault == "root_birth":
        target["ordinary_posture_curriculum"]["birth"]["probability"] = 0.
    elif fault == "criteria":
        target["evaluation"]["cases"][0]["name"] += "_changed"
    else:
        target["task_campaign"]["revision_migration"]["accepted_resume_sources"] = []
    write(bindings / revision.worker_file(domain), target)
    with pytest.raises(ValueError, match="complete named|exact deployed"):
        campaign.validate_worker_revision(read(file, bindings), target)


@pytest.mark.parametrize("fault", ("endpoint", "parent", "budget", "envs", "scope", "source", "warmup"))
def test_rehashed_manifest_does_not_expand_plan_authorization(bindings, fault):
    source, target = read(revision.SOURCE_PLAN_FILE, bindings), read(revision.PLAN_FILE, bindings)
    manifest = read(revision.MANIFEST_FILE, bindings)
    if fault == "endpoint":
        target["domain_completion"]["schedule"]["end_actor_updates"]["speed"] += 1
    elif fault == "parent":
        target["domains"][3]["parent_domain"] = "speed"
    elif fault == "budget":
        target["domains"][2]["updates"] += 1
    elif fault == "envs":
        target["target_num_envs"] = 20000
    elif fault == "scope":
        manifest["curriculum_scope"] = "anything"
    elif fault == "warmup":
        manifest["critic_adaptation_updates"] = 0
    else:
        manifest["source_plan_file"] = "contracts/v6_task_campaign_v17.json"
    write(bindings / revision.PLAN_FILE, target)
    manifest["target_plan_sha256"] = campaign.digest(bindings / revision.PLAN_FILE)
    write(bindings / revision.MANIFEST_FILE, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        campaign.validate_plan_migration(source, target, manifest["run_directory"])


def checkpoint(root, domain, *, actor=None, count=None, attempt=0, worker_file=None, warmup=0):
    actor = revision.TEACHING_ORIGINS[domain] if actor is None else actor
    count = dict(recovery=16334, speed=25988, terrain=28195, jump=29205)[domain] if count is None else count
    file, checksum = revision.source_binding(domain, ROOT)
    if worker_file is not None:
        file, checksum = worker_file, campaign.digest(ROOT / worker_file)
    worker = read(file)
    path = root / domain / f"attempt_{attempt:03d}" / "train/model_final.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    infos = dict(contract_sha256=checksum, task_actor_update=actor, stage_actor_update=13000 + actor,
        successful_updates_total=count, actor_updates_consumed=count - 130, critic_warmup_remaining=warmup,
        task_campaign=deepcopy(worker["task_campaign"]))
    torch.save(dict(infos=infos, policy=torch.tensor([1.])), path)
    shutil.copyfile(ROOT / file, path.with_suffix(".contract.json"))
    write(path.with_suffix(".checkpoint.json"), dict(schema="chassis-learning-checkpoint-v1",
        sealed=True, checkpoint_sha256=campaign.digest(path), contract_sha256=checksum,
        learning_updates=count, learning_lineage_updates=13050 + actor, actor_updates_consumed=count - 130,
        task_actor_update=actor, stage_actor_update=infos["stage_actor_update"], task_campaign=deepcopy(worker["task_campaign"])))
    return path, infos


@pytest.fixture
def completed(tmp_path):
    source, target = plans()
    paths = {d: checkpoint(tmp_path, d)[0] for d in revision.ORIGINS}
    paths["jump"] = checkpoint(tmp_path, "jump")[0]
    records = {f"{d}/attempt_000/train": dict(domain=d, consumed_updates=spent, actor_updates=actor)
        for d, spent, actor in (("flat", 13020, 13000), ("recovery", 2314, 2294),
            ("speed", 5093, 5063), ("terrain", 2206, 2186), ("jump", 5572, 5552), ("stepjump", 0, 0))}
    report = dict(contract_sha256=revision.SOURCE_PLAN_SHA, domains={}, evaluations={"old": "retained"},
        consumed_review_tokens=["prior"], training_process_seconds=0., assessment_seconds=0., elapsed_seconds=0.,
        **campaign.account_updates(source, records))
    for d in revision.ORIGINS:
        report["domains"][d] = dict(status="domain_schedule_finished_unassessed", domain_qualified=None,
            candidate_checkpoint=str(paths[d]), candidate_checkpoint_sha256=campaign.digest(paths[d]),
            finish_review=dict(token=d + "-finish"), pending_review=dict(token=d + "-pending"),
            schedule_completion=dict(actual_actor_update=revision.ORIGINS[d]),
            completion_history=[dict(plan_revision="v12", record=dict(old="retained"))])
    report["domains"]["flat"] = dict(status="domain_schedule_finished_unassessed", old="retained")
    return SimpleNamespace(root=tmp_path, paths=paths, records=records, report=report, source=source, target=target)


def reopen(completed):
    return revision.reopen_completed_domains(completed.report, completed.records,
        completed.source, completed.target, completed.root)


def test_three_atomic_reopenings_are_idempotent_and_preserve_all_consumption_and_history(completed):
    original, records = deepcopy(completed.report), deepcopy(completed.records)
    receipt = reopen(completed)
    assert receipt["domains"] == revision.ORIGINS
    assert completed.records == records
    for d, origin in revision.ORIGINS.items():
        current = completed.report["domains"][d]
        assert current["domain_qualified"] is None
        assert current["reopening"]["origin_actor_update"] == origin
        assert current["reopening"]["critic_adaptation_updates"] == 10
        assert current["completion_history"][-1] == dict(plan_revision="v15", record=original["domains"][d])
        assert current["candidate_checkpoint"] == str(completed.paths[d])
    assert completed.report["consumed_updates"] == original["consumed_updates"] == 29205
    assert completed.report["domains"]["flat"] == original["domains"]["flat"]
    assert completed.report["evaluations"] == original["evaluations"]
    once = deepcopy(completed.report)
    assert reopen(completed) == receipt and completed.report == once


@pytest.mark.parametrize("fault", ("unsealed", "qualified", "older", "actor_clock", "float_actor", "foreign_source", "hash", "domain_cap", "missing"))
def test_last_of_three_reopenings_fails_atomically_without_partial_state(completed, fault):
    d, path = "terrain", completed.paths["terrain"]
    seal_file = path.with_suffix(".checkpoint.json");seal = json.loads(seal_file.read_text())
    if fault == "unsealed":
        seal["sealed"] = False
    elif fault == "qualified":
        completed.report["domains"][d]["domain_qualified"] = True
    elif fault == "older":
        checkpoint(completed.root, d, actor=2187, count=28196, attempt=1)
    elif fault in ("actor_clock", "float_actor"):
        seal["task_actor_update"] = 2187 if fault == "actor_clock" else 2186.
    elif fault == "foreign_source":
        seal["task_campaign"]["domain"] = "speed"
    elif fault == "hash":
        completed.report["domains"][d]["candidate_checkpoint_sha256"] = "0" * 64
    elif fault == "domain_cap":
        completed.records[d + "/attempt_000/train"]["consumed_updates"] = 19000
    else:
        del completed.report["domains"][d]
    write(seal_file, seal)
    before = deepcopy(completed.report)
    with pytest.raises(ValueError):
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
    runner.write_json(completed.root / "campaign_state.json", completed.report)
    runner.write_json(job.claim, dict(campaign_id=campaign.CAMPAIGN_ID, run_directory=str(completed.root),
        contract_sha256=revision.SOURCE_PLAN_SHA, attempts=deepcopy(completed.records),
        **campaign.account_updates(completed.source, completed.records)))
    return job


def restart_job(completed, monkeypatch):
    # Do not rewrite durable claim/state when modelling a same-revision restart.
    job = object.__new__(runner.TaskCampaign)
    job.plan, job.root = deepcopy(completed.target), completed.root
    job.args = SimpleNamespace(contract=ROOT / revision.PLAN_FILE, device="cpu")
    job.claim = completed.root / "shared/account.json"
    job.report, job.records, job._session_started = {}, {}, None
    monkeypatch.setattr(job, "assert_no_live_workers", lambda: None)
    return job


def test_native_migration_and_same_revision_restart_do_not_reopen_or_charge_twice(completed, monkeypatch):
    job = account_job(completed, monkeypatch);job.open_account()
    once, records = deepcopy(job.report), deepcopy(job.records)
    restarted = restart_job(completed, monkeypatch);restarted.open_account()
    assert restarted.report == once and restarted.records == records == completed.records
    assert restarted.learning_source(next(r for r in job.plan["domains"] if r["name"] == "jump")) == completed.paths["jump"]


def test_native_crash_after_claim_commit_replays_three_authenticated_reopenings(completed, monkeypatch):
    job = account_job(completed, monkeypatch);write_json = runner.write_json

    def power_loss(path, value):
        if Path(path) == completed.root / "campaign_state.json":
            raise RuntimeError("power loss after claim commit")
        return write_json(path, value)

    monkeypatch.setattr(runner, "write_json", power_loss)
    with pytest.raises(RuntimeError, match="power loss"):
        job.open_account()
    assert read("campaign_state.json", completed.root)["contract_sha256"] == revision.SOURCE_PLAN_SHA
    assert json.loads(job.claim.read_text())["contract_sha256"] == campaign.digest(ROOT / revision.PLAN_FILE)
    monkeypatch.setattr(runner, "write_json", write_json)
    restarted = restart_job(completed, monkeypatch);restarted.open_account()
    once = deepcopy(restarted.report)
    assert all(len(restarted.report["domains"][d]["completion_history"]) == 2 for d in revision.ORIGINS)
    assert restarted.report["consumed_updates"] == 29205 and restarted.records == completed.records
    restarted.open_account();assert restarted.report == once


@pytest.mark.parametrize("fault", ("source_receipt", "reopening_schema", "source_hash", "history"))
def test_same_revision_restart_reauthenticates_sources_not_only_schema(completed, monkeypatch, fault):
    job = account_job(completed, monkeypatch);job.open_account()
    state = read("campaign_state.json", completed.root)
    if fault == "source_receipt":
        file = completed.root / "revisions/v15_to_v18_account_migration.json"
        receipt = json.loads(file.read_text());receipt["source_plan_sha256"] = "0" * 64;write(file, receipt)
    else:
        domain = state["domains"]["speed"]
        if fault == "reopening_schema":
            domain["reopening"]["schema"] = "task-domain-reopen-v17"
        elif fault == "source_hash":
            domain["reopening"]["source_checkpoint_sha256"] = "0" * 64
        else:
            domain["completion_history"].clear()
        write(completed.root / "campaign_state.json", state)
    with pytest.raises(ValueError):
        restart_job(completed, monkeypatch).open_account()


@pytest.mark.parametrize("domain", tuple(revision.ORIGINS) + ("jump",))
def test_scheduler_restores_each_own_actor_and_adds_only_ten_charged_critic_updates(completed, monkeypatch, domain):
    reopen(completed)
    job = object.__new__(runner.TaskCampaign)
    job.plan, job.root, job.report, job.records = deepcopy(completed.target), completed.root, completed.report, completed.records
    job.args = SimpleNamespace(device="cpu");job.deadline, job.stop_requested = float("inf"), False
    job.persist = lambda: job.report.update(campaign.account_updates(job.plan, job.records))
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: pytest.fail("No online evaluation is allowed"))
    row = next(r for r in job.plan["domains"] if r["name"] == domain)
    origin = revision.TEACHING_ORIGINS[domain];actor_count, charge = 15000 - origin, 15000 - origin + 10
    before, calls = deepcopy(job.report), []

    def execute(command, log, directory):
        calls.append(command)
        assert int(command[command.index("--updates") + 1]) == charge
        assert command[command.index("--resume") + 1] == str(completed.paths[domain])
        assert int(command[command.index("--task-actor-update") + 1]) == origin
        assert int(command[command.index("--consumed-updates") + 1]) == before["consumed_updates"]
        write(directory / "completion.json", dict(status="completed", successful_updates=charge,
            attempted_updates=charge, actor_updates_in_block=actor_count))
        checkpoint(completed.root, domain, actor=15000, count=before["consumed_updates"] + charge,
            attempt=int(directory.parent.name.rsplit("_", 1)[1]), worker_file=row["contract"])
        return 0

    monkeypatch.setattr(job, "execute", execute);job.run_domain(row)
    assert len(calls) == 1 and job.report["domains"][domain]["domain_qualified"] is None
    assert job.report["domains"][domain]["schedule_completion"]["actual_actor_update"] == 15000
    assert job.report["domain_consumed_updates"][domain] - before["domain_consumed_updates"][domain] == charge
    job.run_domain(row);assert len(calls) == 1


def test_new_v18_same_contract_restart_resumes_warmup_without_restarting_it(tmp_path):
    path, infos = checkpoint(tmp_path, "speed", actor=5063, count=29210, attempt=1,
        worker_file=revision.worker_file("speed"), warmup=5)
    assert resume_critic_warmup(read(revision.worker_file("speed")), infos,
        contract_sha256=campaign.digest(path.with_suffix(".contract.json")), task_change=True) == 5
    infos["critic_warmup_remaining"] = 0
    assert resume_critic_warmup(read(revision.worker_file("speed")), infos,
        contract_sha256=campaign.digest(path.with_suffix(".contract.json")), task_change=True) == 0


def test_v18_entry_review_seals_latest_without_worker_or_assessment_and_keeps_tokens(completed, monkeypatch):
    job = account_job(completed, monkeypatch);job.open_account()
    final, _ = checkpoint(completed.root, "recovery", actor=2295, count=29216, attempt=1,
        worker_file=revision.worker_file("recovery"))
    reviewed = final.parent / "checkpoints/update_00029216/model.pt"
    reviewed.parent.mkdir(parents=True)
    for suffix in (".pt", ".contract.json", ".checkpoint.json"):
        shutil.copyfile(final.with_suffix(suffix), reviewed.with_suffix(suffix))
    job.records["recovery/attempt_001/train"] = dict(domain="recovery", consumed_updates=11, actor_updates=1)
    job.report.update(campaign.account_updates(job.plan, job.records))
    evidence = []
    for kind in ("curve_review", "behavior_review"):
        file = completed.root / "reviews" / (kind + ".json")
        write(file, dict(checkpoint_sha256=campaign.digest(reviewed),
            observations=dict(comparison="bounded same-checkpoint curve and behavior review")))
        evidence.append(dict(kind=kind, file=str(file), sha256=campaign.digest(file)))
    request = dict(schema="task-domain-review-v1", campaign_id=campaign.CAMPAIGN_ID,
        plan_sha256=campaign.digest(ROOT / revision.PLAN_FILE), domain="recovery", decision="finish_domain",
        reason="seal this own branch without granting skill qualification", checkpoint=str(reviewed),
        checkpoint_sha256=campaign.digest(reviewed), evidence=evidence)
    write(completed.root / "domain_review_request.json", request)
    row = next(r for r in job.plan["domains"] if r["name"] == "recovery")
    job.child = None;job.deadline = float("inf");job.stop_requested = False
    job.persist = lambda: job.report.update(campaign.account_updates(job.plan, job.records))
    monkeypatch.setattr(job, "execute", lambda *args: pytest.fail("Sealed entry review cannot rebuild a worker"))
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: pytest.fail("Entry review cannot block on evaluation"))
    tokens, records = deepcopy(job.report["consumed_review_tokens"]), deepcopy(job.records)
    job.run_domain(row)
    current = job.report["domains"]["recovery"]
    assert current["status"] == "domain_review_finished_unassessed" and current["domain_qualified"] is None
    assert current["candidate_checkpoint"] == str(final) and current["finish_review"]["checkpoint"] == str(reviewed)
    assert current["finish_review"]["token"] == campaign.digest(completed.root / "domain_review_request.json")
    assert current["completion_history"][-1]["plan_revision"] == "v15"
    assert current["reopening"]["origin_actor_update"] == 2294
    assert job.report["consumed_review_tokens"] == tokens and job.records == records
    once = deepcopy(job.report);job.run_domain(row)
    assert job.report == once and job.report["consumed_updates"] == 29216


def test_source_receipt_authenticates_v18_archive_and_unchanged_invocation(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    for directory, file in ((old, revision.SOURCE_PLAN_FILE), (new, revision.PLAN_FILE)):
        (directory / "contracts").mkdir(parents=True)
        shutil.copyfile(ROOT / file, directory / file)
        (directory / "scripts").mkdir();(directory / "scripts/run_task_campaign.py").write_text("# Not executed.\n")
    command = [sys.executable, "-B", str(old / "scripts/run_task_campaign.py"), "--contract",
        str(old / revision.SOURCE_PLAN_FILE), "--num-envs", "4096", "--device", "cuda:0", "--research",
        "--max-runtime-seconds", "15552000", "--run-dir", str(tmp_path / "train")]
    previous = dict(pid=2147483647, command=command);write(tmp_path / "job.json", previous)
    revised = list(command);revised[2], revised[4] = str(new / "scripts/run_task_campaign.py"), str(new / revision.PLAN_FILE)
    archive = tmp_path / "source.tar.gz"
    def pack(commit):
        with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT, pax_headers={} if commit is None else {"comment": commit}) as output:
            for path in sorted(new.rglob("*")):
                if path.is_file():output.add(path, arcname=str(path.relative_to(new)), recursive=False)
    pack("a" * 40)
    receipt = dict(schema="task-plan-revision-v18", previous_job_sha256=sha(tmp_path / "job.json"), command=revised,
        source_commit="a" * 40, old_source_root=str(old), new_source_root=str(new), new_contract_relative_path=revision.PLAN_FILE,
        old_contract_sha256=sha(old / revision.SOURCE_PLAN_FILE), new_contract_sha256=sha(new / revision.PLAN_FILE),
        source_archive=str(archive), source_archive_sha256=sha(archive))
    file = tmp_path / "revision.json";write(file, receipt);calls = []
    def validate(source, target, root):
        assert (source, target) == plans() and root == tmp_path / "train";calls.append(True)
    monkeypatch.setattr(campaign, "validate_plan_migration", validate)
    assert validate_source_revision(tmp_path, revised, previous, file) == receipt and calls == [True]
    pack(None);receipt["source_archive_sha256"] = sha(archive);write(file, receipt)
    with pytest.raises(ValueError, match="archive commit differs"):
        validate_source_revision(tmp_path, revised, previous, file)
    pack("a" * 40);receipt["source_archive_sha256"] = sha(archive)
    revised[revised.index("--num-envs") + 1] = "20000";receipt["command"] = revised;write(file, receipt)
    with pytest.raises(ValueError, match="only runner and contract"):
        validate_source_revision(tmp_path, revised, previous, file)
