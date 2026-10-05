"""External queue admission is CPU-only, natural-exit-only and at-most-once."""
from copy import deepcopy
import fcntl
import inspect
import json
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis import task_campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from queue_task_campaign_continuation import ContinuationQueue, DOMAINS, NATURAL, SCHEMA, read_json, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def archive_source(f):
    with tarfile.open(f.archive, "w:gz", format=tarfile.PAX_FORMAT,
                      pax_headers={"comment": f.manifest["source_commit"]}) as output:
        for path in sorted(f.new.rglob("*")):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(f.new)), recursive=False)
    f.manifest["source_archive_sha256"] = sha(f.archive)


@pytest.fixture
def fixture(tmp_path):
    root, old, new = tmp_path / "run", tmp_path / "source_old", tmp_path / "source_new"
    train, budget, proc = root / "train", tmp_path / "historical_budget", tmp_path / "proc"
    for directory in (root, old / "scripts", new / "scripts", new / "src/wheeled_tasks/chassis", proc):
        directory.mkdir(parents=True)
    for directory in (new / "src/wheeled_tasks", new / "src/wheeled_tasks/chassis"):
        (directory / "__init__.py").write_text("")
    (old / "scripts/run_task_campaign.py").write_text("# Never executed.\n")
    (new / "scripts/run_task_campaign.py").write_text("# Never executed.\n")
    shutil.copyfile(ROOT / "scripts/chassis_remote_job.py", new / "scripts/chassis_remote_job.py")
    # The real frozen remote validator delegates to this isolated CPU fixture,
    # retaining its job, exact argv, archive and extracted-source authentication.
    (new / "src/wheeled_tasks/chassis/task_campaign.py").write_text(
        inspect.getsource(task_campaign.account_updates) + "\n" +
        "def validate_plan_migration(before, after, run_directory):\n"
        "    assert before['contract_id'] == 'v6-task-campaign-v15'\n"
        "    assert after['contract_id'] == 'v6-task-campaign-v16'\n"
        "    assert after['domains'] == before['domains']\n"
        "    return {'source_plan_sha256': 'fixture'}\n")
    rows, records, domains = [], {}, {}
    for name in DOMAINS:
        worker = dict(domain=name, unchanged=True)
        contract = new / f"contracts/{name}.json"
        write(contract, worker)
        checkpoint = train / name / "attempt_000/train/model_final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"sealed learning state: " + name.encode())
        write(checkpoint.with_suffix(".contract.json"), worker)
        checkpoint_sha, contract_sha = sha(checkpoint), sha(contract)
        write(checkpoint.with_suffix(".checkpoint.json"), dict(
            schema="chassis-learning-checkpoint-v1", sealed=True,
            checkpoint_sha256=checkpoint_sha, contract_sha256=contract_sha,
            learning_updates=100, task_actor_update=90, task_campaign=dict(domain=name)))
        rows.append(dict(name=name, updates=20000, sha256=contract_sha))
        records[name + "/attempt_000/train"] = dict(domain=name, consumed_updates=100, actor_updates=90)
        domains[name] = dict(status="domain_schedule_finished_unassessed", domain_qualified=None,
                             candidate_checkpoint=str(checkpoint), candidate_checkpoint_sha256=checkpoint_sha)
    plan = dict(contract_id="v6-task-campaign-v15", campaign_id=task_campaign.CAMPAIGN_ID,
                budget_source_stage=str(budget), historical_consumed_updates=1000,
                historical_actor_consumed_updates=980, total_ceiling=121000, domains=rows)
    old_contract = old / "contracts/v6_task_campaign_v15.json"
    new_contract = new / "contracts/v6_task_campaign_v16.json"
    write(old_contract, plan)
    write(new_contract, dict(plan, contract_id="v6-task-campaign-v16"))
    shutil.copyfile(old_contract, train / "study_plan.json")
    totals = task_campaign.account_updates(plan, records)
    account = dict(campaign_id=task_campaign.CAMPAIGN_ID, run_directory=str(train),
                   contract_sha256=sha(old_contract), rollback_refunds_consumed_updates=False,
                   attempts=records, **totals)
    claim = budget / (task_campaign.CAMPAIGN_ID + ".account.json")
    write(claim, account)
    write(train / "consumption.json", account)
    state = dict(status=NATURAL, contract_sha256=sha(old_contract), domains=domains, **totals)
    write(train / "campaign_state.json", state)
    write(train / "completion.json", state)
    command = [sys.executable, "-B", str(new / "scripts/run_task_campaign.py"),
               "--contract", str(new_contract), "--num-envs", "4096", "--device", "cuda:0",
               "--research", "--max-runtime-seconds", "15552000", "--run-dir", str(train)]
    old_command = list(command)
    old_command[2] = str(old / "scripts/run_task_campaign.py")
    old_command[4] = str(old_contract)
    write(root / "job.json", dict(pid=2147483647, command=old_command, started_at="old"))
    old_revision = root / "source_revision_015.json"
    write(old_revision, dict(schema="task-plan-revision-v15", command=old_command,
                            new_source_root=str(old), source_commit="a" * 40,
                            new_contract_sha256=sha(old_contract)))
    manifest = dict(schema=SCHEMA, run_root=str(root), old_source_root=str(old), new_source_root=str(new),
        old_contract_relative_path="contracts/v6_task_campaign_v15.json", old_contract_sha256=sha(old_contract),
        new_contract_relative_path="contracts/v6_task_campaign_v16.json", new_contract_sha256=sha(new_contract),
        old_source_commit="a" * 40, source_commit="b" * 40, source_archive=str(tmp_path / "source.tar.gz"),
        old_source_revision_receipt=str(old_revision), source_revision_receipt=str(root / "source_revision_016.json"),
        queue_receipt=str(root / "continuation_queue_016.json"), status_path=str(root / "continuation_status_016.json"),
        launch_log=str(root / "continuation_launch_016.log"), command=command)
    f = SimpleNamespace(root=root, old=old, new=new, train=train, claim=claim, proc=proc,
                        old_contract=old_contract, new_contract=new_contract, state=state,
                        manifest=manifest, archive=Path(manifest["source_archive"]), calls=[])
    archive_source(f)

    def launcher(argv, **kwargs):
        f.calls.append((list(argv), kwargs))
        return SimpleNamespace(pid=4242)

    f.launcher = launcher
    f.queue = lambda: ContinuationQueue(f.manifest, proc_root=f.proc, launcher=launcher)
    return f


def set_status(f, status):
    f.state["status"] = status
    write(f.train / "campaign_state.json", f.state)
    write(f.train / "completion.json", f.state)


def process(f, pid, argv):
    path = f.proc / str(pid) / "cmdline"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")


def test_active_v15_waits_without_hashing_source_or_creating_revision(fixture, monkeypatch):
    f = fixture
    set_status(f, "running")
    process(f, 11, [sys.executable, str(f.old / "scripts/run_task_campaign.py"), "--run-dir", str(f.train)])
    queue = f.queue()
    monkeypatch.setattr(queue, "validate_staged_source", lambda: pytest.fail("Active polling must remain lightweight"))
    assert queue.step()["status"] == "waiting"
    assert not queue.source_revision_receipt.exists() and not queue.intent.exists() and not f.calls
    unchanged = queue.status_path.read_bytes()
    queue.step()
    assert queue.status_path.read_bytes() == unchanged


def test_actual_wrapper_entry_is_counted_once_and_evaluator_also_blocks(fixture):
    f = fixture
    process(f, 11, [sys.executable, str(f.old / "scripts/chassis_remote_job.py"), "--run-root", str(f.root),
                    "--", sys.executable, str(f.old / "scripts/run_task_campaign.py"), "--run-dir", str(f.train)])
    process(f, 12, [sys.executable, str(f.old / "scripts/evaluate_chassis.py"), "--run-dir", str(f.train / "eval")])
    result = f.queue().step()
    assert result["status"] == "waiting"
    assert [row["role"] for row in result["processes"]] == ["wrapper", "evaluator"]
    assert not f.calls


@pytest.mark.parametrize("status", ["running", "starting", "stopped", "failed", "offline_assessments_finished"])
def test_ownerless_or_abnormal_exit_never_continues(fixture, status):
    f = fixture
    set_status(f, status)
    f.state["reason"] = "Campaign deadline/signal; budget remains charged"
    write(f.train / "campaign_state.json", f.state)
    assert f.queue().step()["status"] == "attention"
    assert not f.calls and not (f.root / "source_revision_016.json").exists()


def test_all_natural_domains_launch_literal_native_wrapper_once_and_keep_budget(fixture):
    f = fixture
    evidence = {path: path.read_bytes() for path in (f.claim, f.train / "consumption.json", f.train / "study_plan.json")}
    queue = f.queue()
    assert queue.step()["status"] == "launch_submitted"
    assert f.queue().step()["status"] == "launch_submitted"
    assert len(f.calls) == 1
    argv, kwargs = f.calls[0]
    assert argv[2] == str(f.new / "scripts/chassis_remote_job.py")
    assert argv[argv.index("--") + 1:] == f.manifest["command"]
    assert kwargs["shell"] is False and kwargs["start_new_session"] is True
    assert "env" not in kwargs  # Session's simulation environment remains inherited.
    revision, receipt = read_json(queue.source_revision_receipt), read_json(queue.queue_receipt)
    assert revision["schema"] == "task-plan-revision-v16"
    assert revision["previous_job_sha256"] == sha(f.root / "job.json")
    assert set(receipt["predecessor"]["seals"]) == set(DOMAINS)
    assert all(path.read_bytes() == data for path, data in evidence.items())


def test_receipt_binds_actual_job_only_after_natural_completion(fixture):
    f = fixture
    queue = f.queue()
    set_status(f, "running")
    process(f, 11, [sys.executable, str(f.old / "scripts/run_task_campaign.py"), "--run-dir", str(f.train)])
    queue.step()
    assert not queue.source_revision_receipt.exists()
    (f.proc / "11/cmdline").unlink()
    job = read_json(f.root / "job.json")
    job["finished_late"] = True
    write(f.root / "job.json", job)
    set_status(f, NATURAL)
    assert queue.step()["status"] == "launch_submitted"
    assert read_json(queue.source_revision_receipt)["previous_job_sha256"] == sha(f.root / "job.json")


@pytest.mark.parametrize("corruption", ["checkpoint", "seal", "contract", "missing_domain", "claim", "account_totals",
    "completion", "study_plan", "old_revision", "archive", "new_plan"])
def test_terminal_authentication_failure_never_launches(fixture, corruption):
    f = fixture
    checkpoint = Path(f.state["domains"]["terrain"]["candidate_checkpoint"])
    if corruption == "checkpoint":
        checkpoint.write_bytes(b"changed learning state")
    elif corruption == "seal":
        value = read_json(checkpoint.with_suffix(".checkpoint.json"))
        value["sealed"] = False
        write(checkpoint.with_suffix(".checkpoint.json"), value)
    elif corruption == "contract":
        write(checkpoint.with_suffix(".contract.json"), dict(domain="terrain", changed=True))
    elif corruption == "missing_domain":
        del f.state["domains"]["stepjump"]
        set_status(f, NATURAL)
    elif corruption == "claim":
        account = read_json(f.claim)
        account["consumed_updates"] -= 1
        write(f.claim, account)
    elif corruption == "account_totals":
        account = read_json(f.claim)
        account["consumed_updates"] -= 1
        write(f.claim, account)
        write(f.train / "consumption.json", account)
    elif corruption == "completion":
        write(f.train / "completion.json", dict(f.state, status="stopped"))
    elif corruption == "study_plan":
        write(f.train / "study_plan.json", {"contract_id": "different"})
    elif corruption == "old_revision":
        revision = read_json(f.root / "source_revision_015.json")
        revision["source_commit"] = "c" * 40
        write(f.root / "source_revision_015.json", revision)
    elif corruption == "archive":
        f.archive.write_bytes(b"not the frozen archive")
    else:
        write(f.new_contract, {"changed": True})
    queue = f.queue()
    assert queue.step()["status"] == "attention"
    assert not f.calls and not queue.intent.exists()


def test_native_claim_lock_blocks_even_when_proc_snapshot_has_no_owner(fixture):
    f = fixture
    with f.claim.with_suffix(".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert f.queue().step()["reason"] == "native_campaign_account_lock_still_owned"
    assert not f.calls


def test_singleton_queue_lock_does_not_spawn_a_second_observer(fixture):
    f = fixture
    queue = f.queue()
    with queue.lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert queue.run(once=True) == 0
    assert not f.calls and not queue.status_path.exists()


def test_uncertain_launch_failure_persists_intent_and_cannot_retry(fixture):
    f = fixture
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise OSError("Synthetic launch failure")

    queue = ContinuationQueue(f.manifest, proc_root=f.proc, launcher=fail)
    assert queue.step()["status"] == "attention"
    assert queue.intent.exists() and not queue.queue_receipt.exists()
    assert queue.step()["reason"] == "durable_launch_intent_exists_do_not_launch_again"
    assert len(calls) == 1
    other = dict(f.manifest, queue_receipt=str(f.root / "another_receipt.json"),
                 status_path=str(f.root / "another_status.json"))
    assert ContinuationQueue(other, proc_root=f.proc, launcher=f.launcher).step()["status"] == "attention"
    assert not f.calls


def test_fresh_frozen_validation_ignores_cached_old_package(fixture, monkeypatch):
    f = fixture

    def reject(*args):
        pytest.fail("Cached old package cannot authenticate the new v16 revision")

    monkeypatch.setattr(task_campaign, "validate_plan_migration", reject)
    assert f.queue().step()["status"] == "launch_submitted"
    assert len(f.calls) == 1


def test_frozen_native_plan_validator_rejection_is_not_bypassed(fixture):
    f = fixture
    path = f.new / "src/wheeled_tasks/chassis/task_campaign.py"
    path.write_text(path.read_text() + "\ndef validate_plan_migration(*args):\n    raise ValueError('Named migration rejected')\n")
    archive_source(f)
    result = f.queue().step()
    assert result["status"] == "attention" and "Named migration rejected" in result["detail"]
    assert not f.calls


def test_predeployment_manifest_audit_creates_no_revision_or_launch(fixture):
    f = fixture
    set_status(f, "running")
    queue = f.queue()
    assert queue.validate_staged_source()["status"] == "passed"
    assert not queue.source_revision_receipt.exists() and not queue.queue_receipt.exists()
    assert not queue.status_path.exists() and not queue.intent.exists() and not f.calls


@pytest.mark.parametrize("tamper", ["archive", "extracted_python"])
def test_unverified_staged_python_is_rejected_before_any_execution(fixture, tamper):
    f = fixture
    path = f.new / "src/wheeled_tasks/chassis/task_campaign.py"
    path.write_text("raise RuntimeError('Unverified staged Python must never execute')\n")
    if tamper == "archive":
        f.archive.write_bytes(b"unverified archive")
    result = f.queue().step()
    assert result["status"] == "attention"
    assert "Staged" in result["detail"] and "must never execute" not in result["detail"]
    assert not f.calls and not (f.root / "source_revision_016.json").exists()


@pytest.mark.parametrize("change", ["population", "run_directory", "entry", "no_research", "shell_entry"])
def test_manifest_cannot_replace_native_research_entry(fixture, change):
    manifest = deepcopy(fixture.manifest)
    command = manifest["command"]
    if change == "population":
        command[command.index("--num-envs") + 1] = "20000"
    elif change == "run_directory":
        command[command.index("--run-dir") + 1] = "/other/train"
    elif change == "entry":
        command[2] = str(fixture.new / "scripts/train_chassis.py")
    elif change == "no_research":
        command.remove("--research")
    else:
        command[1:1] = ["-c", "dangerous code"]
    with pytest.raises(ValueError, match="native 4096 research"):
        ContinuationQueue(manifest)
