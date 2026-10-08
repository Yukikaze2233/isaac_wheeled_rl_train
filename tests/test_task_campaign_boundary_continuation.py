"""CPU tests: the external REC boundary queue never stops REC or refunds work."""
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
import torch

from wheeled_tasks.chassis.task_campaign import account_updates, update_attempt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import queue_task_campaign_boundary_continuation as boundary
from train_chassis import resume_critic_warmup
from run_task_campaign import TaskCampaign


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def proc(f, pid, argv, *, starttime=100, ppid=0):
    path = f.proc / str(pid)
    path.mkdir(exist_ok=True)
    (path / "cmdline").write_bytes(b"\0".join(v.encode() for v in argv) + b"\0")
    tail = ["S", str(ppid)] + ["0"] * 17 + [str(starttime)] + ["0"] * 5
    (path / "stat").write_text(f"{pid} (fixture worker) " + " ".join(tail))


def remove_owners(f):
    # Native supervisor commits terminal state before its wrapper exits.
    f.state["status"] = "stopped"
    write(f.train / "campaign_state.json", f.state)
    write(f.train / "completion.json", f.state)
    for p in f.proc.iterdir():
        shutil.rmtree(p)


def checkpoint(f, domain, directory, *, count=1, task_clock=1, final=False):
    path = directory / ("model_final.pt" if final else f"checkpoints/update_{count:08d}/model.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"complete-learning-{domain}-{count}-{task_clock}".encode())
    worker = f.workers[domain]
    write(path.with_suffix(".contract.json"), worker)
    seal = dict(schema="chassis-learning-checkpoint-v1", sealed=True, checkpoint_sha256=boundary.sha(path),
        contract_sha256=f.rows[domain]["sha256"], task_campaign=dict(domain=domain),
        learning_updates=count, task_actor_update=task_clock, stage_actor_update=task_clock)
    write(path.with_suffix(".checkpoint.json"), seal)
    if not final:
        write(path.parent / "progress.json", dict(successful_updates=count, actor_updates_in_block=count))
    return path


def worker(f, domain, attempt):
    directory = f.train / domain / attempt / "train"
    directory.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-B", str(f.old / "scripts/train_chassis.py"),
        "--contract", str(f.old / f.rows[domain]["contract"]), "--run-dir", str(directory)]
    parents = [p for p in (f.train / domain).glob("attempt_*/train/**/*.checkpoint.json")
               if not p.is_relative_to(directory)]
    extra = {}
    if parents:
        parent_seal = max(parents, key=lambda p: boundary.read_json(p)["learning_updates"])
        parent = parent_seal.with_suffix("").with_suffix(".pt")
        argv.extend(["--resume", str(parent)])
        extra = dict(source_checkpoint=str(parent), source_checkpoint_sha256=boundary.sha(parent))
    write(directory.parent / "invocation.json", dict(command=argv, **extra))
    proc(f, 202, argv, starttime=1002, ppid=201)
    return directory


def rec_finished(f):
    path = checkpoint(f, "recovery", f.rec_dir, count=100, task_clock=15000, final=True)
    write(f.rec_dir / "completion.json", dict(status="completed", successful_updates=100))
    f.state["domains"]["recovery"] = dict(status="domain_schedule_finished_unassessed",
        domain_qualified=None, candidate_checkpoint=str(path), candidate_checkpoint_sha256=boundary.sha(path),
        schedule_completion=dict(reason="iteration_endpoint", end_actor_update=15000, actual_actor_update=15000))
    write(f.train / "campaign_state.json", f.state)
    return path


def archiver_finished(f):
    f.state["status"] = "stopped"
    write(f.train / "campaign_state.json", f.state)
    write(f.train / "completion.json", f.state)
    write(f.train / "consumption.json", {})
    archive = f.root / "delivery.tar.gz"
    archive.write_bytes(b"actual wrapper final archive fixture")
    files = {}
    for name in ("job.json", "train/consumption.json", "train/completion.json", "train/campaign_state.json"):
        path = f.root / name
        files[name] = dict(size=path.stat().st_size, sha256=boundary.sha(path))
    for path in f.train.rglob("*"):
        if path.is_file() and path.suffix in (".pt", ".json"):
            files[str(path.relative_to(f.root))] = dict(size=path.stat().st_size, sha256=boundary.sha(path))
    write(f.root / "delivery.json", dict(status="ready", archive=archive.name,
        archive_bytes=archive.stat().st_size, archive_sha256=boundary.sha(archive), files=files))
    write(f.root / "archiver_status.json", dict(status="finished", pid=201))


class FakeAdapter:
    """Only side effects are mocked; fixtures use real files/proc and native account helpers."""
    def __init__(self, q, f):
        self.q, self.f = q, f

    def seal(self, path, domain, contract_sha256):
        path = Path(path)
        seal = boundary.read_json(path.with_suffix(".checkpoint.json"))
        if (seal.get("sealed") is not True or seal["checkpoint_sha256"] != boundary.sha(path)
                or seal["contract_sha256"] != contract_sha256):
            raise ValueError("fixture stale/missing native seal")
        return dict(path=str(path), sha256=boundary.sha(path), seal_sha256=boundary.sha(path.with_suffix(".checkpoint.json")),
            contract_sha256=contract_sha256, learning_updates=seal["learning_updates"],
            task_actor_update=seal["task_actor_update"], stage_actor_update=seal["stage_actor_update"],
            critic_warmup_remaining=seal.get("critic_warmup_remaining", 0))

    def preflight(self, job):
        self.f.preflight_calls += 1
        if self.f.preflight_error:
            raise ValueError(self.f.preflight_error)
        return dict(status="passed")

    def signal_once(self, owner, before_signal):
        if self.f.signal_race:
            self.f.signal_race()
        before_signal()
        self.f.signals.append(owner)

    def prepare(self, job):
        self.f.prepare_calls += 1
        records = deepcopy(self.f.records)
        for path in self.f.train.glob("*/attempt_*/train/progress.json"):
            records = update_attempt(self.f.plan, records, str(path.parent.relative_to(self.f.train)),
                path.parents[2].name, boundary.read_json(path))
        totals = account_updates(self.f.plan, records)
        own = {}
        for domain in ("recovery", "speed"):
            paths = list((self.f.train / domain).glob("attempt_*/train/**/*.checkpoint.json"))
            path = max(paths, key=lambda p: boundary.read_json(p)["learning_updates"])
            own[domain] = self.seal(path.with_suffix("").with_suffix(".pt"), domain, self.f.rows[domain]["sha256"])
        if self.f.prepare_race:
            self.f.prepare_race(self.f.prepare_calls)
        infos = dict(contract_sha256=own["speed"]["contract_sha256"], successful_updates_total=own["speed"]["learning_updates"],
            critic_warmup_remaining=own["speed"]["critic_warmup_remaining"])
        warmup = resume_critic_warmup(self.f.workers["speed"], infos,
            contract_sha256=self.f.rows["speed"]["sha256"], task_change=True)
        return dict(totals=totals, latest_own=own, records=records, speed_critic_warmup_updates=warmup)

    def launch(self, wrapper):
        self.f.launches.append(list(wrapper))
        return SimpleNamespace(pid=303)


@pytest.fixture
def f(tmp_path):
    root, old, new = (tmp_path / name for name in ("run", "source_old", "source_new"))
    proc_root, train, budget = tmp_path / "proc", root / "train", tmp_path / "budget"
    for p in (root, proc_root, old / "scripts", new / "scripts", new / "src/wheeled_tasks/chassis", budget):
        p.mkdir(parents=True)
    for name in ("__init__.py", "chassis/__init__.py"):
        (new / "src/wheeled_tasks" / name).write_text("")
    for name in ("run_task_campaign.py", "chassis_remote_job.py", "train_chassis.py"):
        (old / "scripts" / name).write_text("# old fixture, never executed\n")
        shutil.copyfile(ROOT / "scripts" / name, new / "scripts" / name)
    for name in ("task_campaign.py", "full_curriculum.py"):
        shutil.copyfile(ROOT / "src/wheeled_tasks/chassis" / name, new / "src/wheeled_tasks/chassis" / name)
    workers, rows = {}, {}
    for domain in ("flat", "recovery", "speed", "terrain", "jump", "stepjump"):
        workers[domain] = dict(task_campaign=dict(domain=domain), fixed=True)
        relative = f"contracts/{domain}.json"
        write(old / relative, workers[domain])
        write(new / relative, workers[domain])
        rows[domain] = dict(name=domain, contract=relative, sha256=boundary.sha(old / relative), updates=20000)
    plan = dict(contract_id="v6-task-campaign-v21", campaign_id="v6-task-domains-20261001",
        domains=list(rows.values()), budget_source_stage=str(budget), historical_consumed_updates=1000,
        historical_actor_consumed_updates=980, total_ceiling=121000,
        domain_completion=dict(schedule=dict(end_actor_updates=dict(recovery=15000))))
    old_plan = old / "contracts/v6_task_campaign_v21.json"
    new_plan = new / "contracts/v6_task_campaign_v22.json"
    write(old_plan, plan)
    write(new_plan, dict(plan, contract_id="v6-task-campaign-v22"))
    train.mkdir(exist_ok=True)
    shutil.copyfile(old_plan, train / "study_plan.json")
    command = [sys.executable, "-B", str(new / "scripts/run_task_campaign.py"), "--contract", str(new_plan),
        "--run-dir", str(train), "--num-envs", "4096", "--research", "--device", "cuda:0"]
    old_command = list(command)
    old_command[2], old_command[4] = str(old / "scripts/run_task_campaign.py"), str(old_plan)
    job = dict(pid=201, command=old_command)
    write(root / "job.json", job)
    revision = root / "source_revision_023.json"
    write(revision, dict(schema="task-plan-revision-v21", command=old_command, new_source_root=str(old),
        source_commit=boundary.OLD_COMMIT, new_contract_sha256=boundary.sha(old_plan)))
    runtime = tmp_path / "sim60-runtime.sh"
    runtime.write_text("# isolated test runtime\n")
    manifest = dict(schema=boundary.SCHEMA, old_source_commit=boundary.OLD_COMMIT, source_commit="b" * 40,
        run_root=str(root), old_source_root=str(old), new_source_root=str(new),
        old_contract_relative_path="contracts/v6_task_campaign_v21.json", new_contract_relative_path="contracts/v6_task_campaign_v22.json",
        old_contract_sha256=boundary.sha(old_plan), new_contract_sha256=boundary.sha(new_plan),
        old_source_revision_receipt=str(revision), old_source_revision_sha256=boundary.sha(revision),
        old_job_sha256=boundary.sha(root / "job.json"), source_revision_receipt=str(root / "source_revision_024.json"),
        queue_receipt=str(root / "boundary_queue_024.json"), status_path=str(root / "boundary_status_024.json"),
        launch_log=str(root / "boundary_launch_024.log"), runtime_script=str(runtime), runtime_script_sha256=boundary.sha(runtime),
        command=command, old_owners=dict(wrapper=dict(pid=200, starttime=1000), supervisor=dict(pid=201, starttime=1001)))
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT, pax_headers={"comment": manifest["source_commit"]}) as output:
        for p in new.rglob("*"):
            if p.is_file():
                output.add(p, arcname=str(p.relative_to(new)), recursive=False)
    manifest.update(source_archive=str(archive), source_archive_sha256=boundary.sha(archive))
    f = SimpleNamespace(root=root, old=old, new=new, train=train, proc=proc_root, plan=plan,
        manifest=manifest, rows=rows, workers=workers, records={}, signals=[], launches=[], prepare_calls=0,
        signal_race=None, prepare_race=None, preflight_calls=0, preflight_error=None,
        state=dict(status="running", contract_sha256=boundary.sha(old_plan), domains={}))
    write(train / "campaign_state.json", f.state)
    write(budget / (plan["campaign_id"] + ".account.json"), {})
    write(root / "archiver_status.json", dict(status="running", pid=201))
    proc(f, 200, [sys.executable, "-B", str(old / "scripts/chassis_remote_job.py"), "--run-root", str(root), "--", *old_command], starttime=1000)
    proc(f, 201, old_command, starttime=1001, ppid=200)
    old_speed = worker(f, "speed", "attempt_000")
    checkpoint(f, "speed", old_speed, count=99, task_clock=5093)
    f.rec_dir = worker(f, "recovery", "attempt_009")
    f.q = lambda: boundary.BoundaryContinuationQueue(f.manifest, proc_root=f.proc, adapter_factory=lambda q: FakeAdapter(q, f))
    return f


def retarget(f, revision):
    if revision == 22:
        return
    assert revision == 23
    target = f.new / "contracts/v6_task_campaign_v23.json"
    write(target, dict(f.plan, contract_id="v6-task-campaign-v23"))
    f.manifest.update(new_contract_relative_path=str(target.relative_to(f.new)),
        new_contract_sha256=boundary.sha(target),
        source_revision_receipt=str(f.root / "source_revision_025.json"),
        queue_receipt=str(f.root / "boundary_continuation_025.receipt.json"),
        status_path=str(f.root / "boundary_continuation_025.status.json"),
        launch_log=str(f.root / "boundary_continuation_025.launch.log"))
    f.manifest["command"][f.manifest["command"].index("--contract") + 1] = str(target)
    archive = Path(f.manifest["source_archive"])
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT,
                      pax_headers={"comment": f.manifest["source_commit"]}) as output:
        for path in f.new.rglob("*"):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(f.new)), recursive=False)
    f.manifest["source_archive_sha256"] = boundary.sha(archive)


def admitted(f):
    q = f.q()
    assert q.step()["reason"] == "rec_normal_endpoint_not_complete"
    rec_finished(f)
    speed = worker(f, "speed", "attempt_010")
    checkpoint(f, "speed", speed, count=101, task_clock=5094)
    return q, speed


def test_check_only_live_authenticates_without_signal_launch_or_intent(f):
    q = f.q()
    assert q.step(check_only=True)["status"] == "waiting"
    assert not q.signal_intent.exists() and not q.intent.exists() and not q.state_path.exists()
    assert not f.signals and not f.launches


def test_lightweight_waiting_does_not_repeat_archive_scan(f, monkeypatch):
    q = f.q()
    calls, original = [], q.staged
    monkeypatch.setattr(q, "staged", lambda: (calls.append(1), original())[1])
    q.step(); q.step()
    assert calls == [1]


def test_historical_speed_seal_does_not_admit_new_worker(f):
    old_speed = worker(f, "speed", "attempt_000")
    checkpoint(f, "speed", old_speed, count=99, task_clock=5093)
    worker(f, "recovery", "attempt_009")
    q = f.q(); q.step(); rec_finished(f)
    speed = worker(f, "speed", "attempt_010")
    assert q.step()["reason"] == "post_rec_speed_complete_snapshot_not_yet_sealed"
    assert not f.signals
    checkpoint(f, "speed", speed, count=100, task_clock=5094)
    assert q.step()["reason"] == "one_authenticated_supervisor_sigterm_submitted"


@pytest.mark.parametrize("revision", [22, 23])
@pytest.mark.parametrize("fault", ["missing", "not_completed", "below_endpoint", "bad_sha"])
def test_rec_requires_actual_completed_endpoint_and_full_final_seal(f, fault, revision):
    retarget(f, revision)
    q = f.q(); q.step(); path = rec_finished(f)
    if fault == "missing":
        path.with_suffix(".checkpoint.json").unlink()
    elif fault == "not_completed":
        write(path.parent / "completion.json", dict(status="stopped"))
    elif fault == "below_endpoint":
        seal = boundary.read_json(path.with_suffix(".checkpoint.json")); seal["task_actor_update"] = 14999
        write(path.with_suffix(".checkpoint.json"), seal)
    else:
        path.write_bytes(b"changed model")
    assert q.step()["status"] == "attention"
    assert not f.signals and not q.signal_intent.exists()


def test_endpoint_is_taken_from_bound_plan(f):
    plan = deepcopy(f.plan); plan["domain_completion"]["schedule"]["end_actor_updates"]["recovery"] = 16000
    write(f.old / f.manifest["old_contract_relative_path"], plan)
    shutil.copyfile(f.old / f.manifest["old_contract_relative_path"], f.train / "study_plan.json")
    f.manifest["old_contract_sha256"] = boundary.sha(f.train / "study_plan.json")
    revision = boundary.read_json(Path(f.manifest["old_source_revision_receipt"]))
    revision["new_contract_sha256"] = f.manifest["old_contract_sha256"]
    write(Path(f.manifest["old_source_revision_receipt"]), revision)
    f.manifest["old_source_revision_sha256"] = boundary.sha(Path(f.manifest["old_source_revision_receipt"]))
    f.state["contract_sha256"] = f.manifest["old_contract_sha256"]
    write(f.train / "campaign_state.json", f.state)
    q = f.q(); q.step(); rec_finished(f)
    assert q.step()["status"] == "attention"
    assert not f.signals


def test_changed_supervisor_starttime_is_rejected(f):
    q, _ = admitted(f)
    proc(f, 201, boundary.read_json(f.root / "job.json")["command"], starttime=9999, ppid=200)
    assert q.step()["status"] == "attention"
    assert not f.signals


def test_pre_signal_race_rechecks_rec_full_seal(f):
    q, _ = admitted(f)
    f.signal_race = lambda: (f.rec_dir / "model_final.pt").write_bytes(b"changed after admission")
    assert q.step()["status"] == "attention"
    assert not f.signals and not q.signal_intent.exists()


def test_pidfd_adapter_never_uses_numeric_pid_kill(f, monkeypatch):
    q = f.q(); adapter = boundary.NativeAdapter(q)
    owner = boundary.process_identity(201, f.proc)
    events = []
    monkeypatch.setattr(boundary, "open_pidfd", lambda pid: (events.append(("open", pid)), 88)[1])
    monkeypatch.setattr(boundary, "send_pidfd_sigterm", lambda fd: events.append(("signal", fd, boundary.signal.SIGTERM)))
    monkeypatch.setattr(boundary.os, "close", lambda fd: events.append(("close", fd)))
    adapter.signal_once(owner, lambda: events.append(("intent",)))
    assert events == [("open", 201), ("intent",), ("signal", 88, boundary.signal.SIGTERM), ("close", 88)]


def test_pid_reuse_after_pidfd_open_rejected_before_intent(f, monkeypatch):
    q = f.q(); adapter = boundary.NativeAdapter(q); owner = boundary.process_identity(201, f.proc)
    def open_pid(pid):
        proc(f, pid, owner["command"], starttime=9999, ppid=200)
        return 88
    events = []
    monkeypatch.setattr(boundary, "open_pidfd", open_pid)
    monkeypatch.setattr(boundary.os, "close", lambda fd: None)
    monkeypatch.setattr(boundary, "send_pidfd_sigterm", lambda *args: events.append("signal"))
    with pytest.raises(ValueError, match="changed"):
        adapter.signal_once(owner, lambda: events.append("intent"))
    assert not events


@pytest.mark.parametrize("revision", [22, 23])
def test_single_signal_then_archive_claim_release_single_launch(f, revision):
    retarget(f, revision)
    q, _ = admitted(f)
    assert q.step()["status"] == "waiting"
    assert len(f.signals) == 1
    assert f.q().step()["reason"] == "old_owners_or_wrapper_still_active"
    assert len(f.signals) == 1
    remove_owners(f)
    assert q.step()["reason"] == "old_wrapper_final_archive_not_finished"
    archiver_finished(f)
    lock = Path(f.plan["budget_source_stage"]) / (f.plan["campaign_id"] + ".account.lock")
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert q.step()["reason"] == "native_account_lock_still_owned"
    assert q.step()["status"] == "launch_submitted"
    assert f.q().step()["status"] == "launch_submitted"
    assert len(f.signals) == len(f.launches) == 1
    assert f.prepare_calls == 2
    assert "--resume-existing" in f.launches[0]


def test_ambiguous_once_intents_are_not_retried(f):
    q, _ = admitted(f)
    boundary.durable_json(q.signal_intent, dict(manifest_sha256=q.config_sha256), exclusive=True)
    assert q.step()["reason"] == "ambiguous_signal_intent_never_resignal"
    boundary.durable_json(q.intent, dict(manifest_sha256=q.config_sha256), exclusive=True)
    assert q.step()["reason"] == "ambiguous_launch_intent_never_resubmit"
    assert not f.signals and not f.launches


def test_already_consumed_and_unsealed_reserved_updates_are_not_refunded(f):
    q, speed = admitted(f)
    write(speed / "progress.json", dict(successful_updates=102, attempted_updates=103, actor_updates_in_block=92))
    f.records = {"speed/attempt_010/train": dict(domain="speed", consumed_updates=102, actor_updates=92)}
    q.step(); remove_owners(f); archiver_finished(f)
    assert q.step()["status"] == "launch_submitted"
    receipt = boundary.read_json(q.queue_receipt)
    assert receipt["predecessor"]["totals"]["consumed_updates"] == 1103
    assert receipt["predecessor"]["records"]["speed/attempt_010/train"]["consumed_updates"] == 103


def test_native_budget_cap_rejects_overconsumption(f):
    q, speed = admitted(f)
    write(speed / "progress.json", dict(successful_updates=20001, actor_updates_in_block=19990))
    q.step(); remove_owners(f); archiver_finished(f)
    assert q.step()["status"] == "attention"
    assert not f.launches


@pytest.mark.parametrize("revision", [22, 23])
def test_partial_original_speed_warmup_is_preserved_not_restarted(f, revision):
    retarget(f, revision)
    q, speed = admitted(f)
    f.workers["speed"].update(resume_critic_warmup_updates=10, critic_warmup_updates=0)
    path = next(speed.glob("checkpoints/update_*/model.checkpoint.json"))
    seal = boundary.read_json(path); seal["critic_warmup_remaining"] = 7
    write(path, seal)
    q.step(); remove_owners(f); archiver_finished(f)
    assert q.step()["status"] == "launch_submitted"
    receipt = boundary.read_json(q.queue_receipt)
    assert receipt["predecessor"]["speed_critic_warmup_updates"] == 7
    assert len(f.signals) == len(f.launches) == 1


@pytest.mark.parametrize("pins", [None, {}, {"wrapper": {}, "supervisor": {}},
    {"wrapper": dict(pid=True, starttime=1), "supervisor": dict(pid=2, starttime=2)}])
def test_check_only_bad_owner_pins_reports_attention_without_traceback(f, monkeypatch, capsys, pins):
    manifest = deepcopy(f.manifest); manifest["old_owners"] = pins
    path = f.root / "invalid_manifest.json"; write(path, manifest)
    monkeypatch.setattr(sys, "argv", ["boundary", "--manifest", str(path), "--check-only"])
    assert boundary.main() == 2
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "attention" and not output.err
    assert not f.signals and not f.launches


def test_final_account_race_does_not_launch(f):
    q, _ = admitted(f); q.step(); remove_owners(f); archiver_finished(f)
    f.prepare_race = lambda n: f.records.update(extra=dict(domain="flat", consumed_updates=1, actor_updates=1)) if n == 1 else None
    assert q.step()["status"] == "attention"
    assert not f.launches and not q.intent.exists()


def test_check_only_after_boundary_never_signals(f):
    q, _ = admitted(f)
    assert q.step(check_only=True)["reason"] == "check_only_boundary_admitted_no_signal_or_launch"
    assert not f.signals and not q.signal_intent.exists()


@pytest.mark.parametrize("revision", [22, 23])
def test_staged_tampering_is_rejected_before_action(f, revision):
    retarget(f, revision)
    q, _ = admitted(f)
    (f.new / "scripts/run_task_campaign.py").write_text("# tampered\n")
    assert q.step()["status"] == "attention"
    assert not f.signals


def test_native_cpu_seal_loads_full_actor_critic_adam_and_rejects_missing_state(f):
    q = f.q(); adapter = boundary.NativeAdapter(q)
    actor = {f"weight{i}": torch.ones(1) for i in range(8)}
    actor["distribution.std_param"] = torch.ones(6)
    critic = {f"weight{i}": torch.ones(1) for i in range(8)}
    opt = dict(param_groups=[dict(params=list(range(17)))], state={i: dict(
        exp_avg=torch.ones(6 if i == 16 else 1), exp_avg_sq=torch.ones(6 if i == 16 else 1), step=torch.tensor(1.)) for i in range(17)})
    path = f.rec_dir / "model_final.pt"
    infos = dict(contract_sha256=f.rows["recovery"]["sha256"], task_campaign=dict(domain="recovery"),
        successful_updates_total=1, task_actor_update=15000)
    saved = dict(actor_state_dict=actor, critic_state_dict=critic, optimizer_state_dict=opt, infos=infos)
    torch.save(saved, path)
    seal = dict(schema="chassis-learning-checkpoint-v1", sealed=True, checkpoint_sha256=boundary.sha(path),
        contract_sha256=f.rows["recovery"]["sha256"], task_campaign=dict(domain="recovery"),
        learning_updates=1, task_actor_update=15000, stage_actor_update=15000)
    write(path.with_suffix(".checkpoint.json"), seal)
    write(path.with_suffix(".contract.json"), f.workers["recovery"])
    assert adapter.seal(path, "recovery", f.rows["recovery"]["sha256"])["task_actor_update"] == 15000
    saved["optimizer_state_dict"]["state"].pop(16)
    torch.save(saved, path); seal["checkpoint_sha256"] = boundary.sha(path)
    write(path.with_suffix(".checkpoint.json"), seal)
    with pytest.raises(ValueError, match="17-parameter Adam"):
        adapter.seal(path, "recovery", f.rows["recovery"]["sha256"])


def test_pidfd_fallback_opens_current_cpu_test_process_without_signal():
    descriptor = boundary.open_pidfd(boundary.os.getpid())
    boundary.os.close(descriptor)


def test_copied_parent_seal_in_new_speed_attempt_is_not_progress(f):
    q = f.q(); q.step(); rec_finished(f)
    speed = worker(f, "speed", "attempt_010")
    checkpoint(f, "speed", speed, count=99, task_clock=5093)
    assert q.step()["reason"] == "post_rec_speed_complete_snapshot_not_yet_sealed"
    assert not f.signals


def test_native_preflight_incompatibility_refuses_before_signal(f):
    q, _ = admitted(f)
    f.preflight_error = "native v21-to-v22 worker migration incompatible"
    assert q.step()["status"] == "attention"
    assert not f.signals and not q.signal_intent.exists()


@pytest.mark.parametrize("revision", [22, 23])
def test_real_fresh_native_preflight_prepare_partial_warmup_and_charged_account(f, revision):
    """Real wrapper/archive, checkpoint, account, latest-own and warmup helpers.

    Only the absolute production plan/ABI registry is replaced by a small
    explicitly scoped fixture registry; its schema, five unchanged workers and
    same-root binding are still authenticated by the real remote validator.
    """
    retarget(f, revision)
    registry = f.new / "src/wheeled_tasks/chassis/task_campaign.py"
    with registry.open("a") as stream:
        stream.write("\ndef validate_plan_migration(before, after, run_directory):\n"
            "    assert before['contract_id'] == 'v6-task-campaign-v21'\n"
            f"    assert after['contract_id'] == 'v6-task-campaign-v{revision}'\n"
            "    assert before['budget_source_stage'] == after['budget_source_stage']\n"
            "    assert [r for r in before['domains'] if r['name'] != 'terrain'] == [r for r in after['domains'] if r['name'] != 'terrain']\n"
            "    return {'status': 'fixture-plan-passed'}\n"
            "def validate_task_resume(before, after):\n"
            "    assert before['task_campaign']['domain'] == after['task_campaign']['domain']\n"
            "    if after['task_campaign']['domain'] == 'speed': assert before == after\n")
    # Use the actual native methods, without importing unrelated supervisor UI
    # dependencies into this minimal frozen CPU fixture.
    (f.new / "scripts/run_task_campaign.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count\n"
        "from wheeled_tasks.chassis.task_campaign import INTEGRATED_REVISIONS\n"
        "def validate_contracts(plan):\n"
        "    return [dict(status='fixture-design-passed') for r in plan['domains']]\n"
        "class TaskCampaign:\n" + inspect.getsource(TaskCampaign.latest_checkpoint)
        + inspect.getsource(TaskCampaign.learning_source))
    target_terrain = dict(f.workers["terrain"], resume_critic_warmup_updates=10)
    terrain_file = f"contracts/terrain{revision}.json"
    write(f.new / terrain_file, target_terrain)
    plan = deepcopy(f.plan); plan["contract_id"] = f"v6-task-campaign-v{revision}"
    terrain_row = next(r for r in plan["domains"] if r["name"] == "terrain")
    terrain_row.update(contract=terrain_file, sha256=boundary.sha(f.new / terrain_file))
    write(f.new / f.manifest["new_contract_relative_path"], plan)
    f.manifest["new_contract_sha256"] = boundary.sha(f.new / f.manifest["new_contract_relative_path"])
    archive = Path(f.manifest["source_archive"])
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT, pax_headers={"comment": f.manifest["source_commit"]}) as output:
        for path in f.new.rglob("*"):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(f.new)), recursive=False)
    f.manifest["source_archive_sha256"] = boundary.sha(archive)

    def full(path, domain, count, task, remaining=0):
        path.parent.mkdir(parents=True, exist_ok=True)
        actor = {f"weight{i}": torch.ones(1) for i in range(8)}
        actor["distribution.std_param"] = torch.ones(6)
        critic = {f"weight{i}": torch.ones(1) for i in range(8)}
        opt = dict(param_groups=[dict(params=list(range(17)))], state={i: dict(
            exp_avg=torch.ones(6 if i == 16 else 1), exp_avg_sq=torch.ones(6 if i == 16 else 1), step=torch.tensor(1.)) for i in range(17)})
        infos = dict(contract_sha256=f.rows[domain]["sha256"], task_campaign=dict(domain=domain),
            successful_updates_total=count, task_actor_update=task, critic_warmup_remaining=remaining)
        torch.save(dict(actor_state_dict=actor, critic_state_dict=critic, optimizer_state_dict=opt, infos=infos), path)
        write(path.with_suffix(".contract.json"), f.workers[domain])
        write(path.with_suffix(".checkpoint.json"), dict(schema="chassis-learning-checkpoint-v1", sealed=True,
            checkpoint_sha256=boundary.sha(path), contract_sha256=f.rows[domain]["sha256"],
            task_campaign=dict(domain=domain), learning_updates=count, task_actor_update=task, stage_actor_update=task))
    rec = rec_finished(f)
    full(rec, "recovery", 100, 15000)
    f.state["domains"]["recovery"]["candidate_checkpoint_sha256"] = boundary.sha(rec)
    speed = worker(f, "speed", "attempt_010")
    full(speed / "model_final.pt", "speed", 101, 5094, remaining=7)
    historical = f.train / "speed/attempt_000/train/checkpoints/update_00000099/model.pt"
    full(historical, "speed", 99, 5093)
    full(f.train / "terrain/attempt_002/train/model_final.pt", "terrain", 20, 2206)
    records = {"speed/attempt_010/train": dict(domain="speed", consumed_updates=3, actor_updates=0)}
    write(speed / "progress.json", dict(successful_updates=2, attempted_updates=3, actor_updates_in_block=0))
    totals = account_updates(f.plan, records)
    f.state.update(status="stopped", **totals)
    write(f.train / "campaign_state.json", f.state); write(f.train / "completion.json", f.state)
    account = dict(campaign_id=f.plan["campaign_id"], run_directory=str(f.train),
        contract_sha256=f.manifest["old_contract_sha256"], rollback_refunds_consumed_updates=False,
        attempts=records, **totals)
    write(f.train / "consumption.json", account)
    write(Path(f.plan["budget_source_stage"]) / (f.plan["campaign_id"] + ".account.json"), account)
    q = boundary.BoundaryContinuationQueue(f.manifest, proc_root=f.proc)
    job = q.previous_job()
    assert q.adapter.preflight(job) == {"status": "passed", "design_preflight_receipts": 6}
    assert not q.source_revision_receipt.exists()
    boundary.durable_json(q.source_revision_receipt, q.migration_receipt(), exclusive=True)
    result = q.adapter.prepare(job)
    assert result["adaptation"]["speed"]["critic_warmup_updates"] == 7
    assert result["adaptation"]["terrain"]["critic_warmup_updates"] == 10
    assert result["totals"]["consumed_updates"] == 1003
    assert result["latest_own"]["speed"]["path"] == str(speed / "model_final.pt")
    assert boundary.read_json(f.train / "consumption.json") == account


@pytest.mark.parametrize("revision,bad_index", [(22, 25), (23, 24)])
def test_target_revision_requires_its_named_source_receipt(f, revision, bad_index):
    retarget(f, revision)
    f.manifest["source_revision_receipt"] = str(f.root / f"source_revision_{bad_index:03d}.json")
    with pytest.raises(ValueError, match="Only named"):
        f.q()


@pytest.mark.parametrize("key", ["queue_receipt", "status_path", "launch_log"])
def test_v23_requires_fresh_named_output_paths(f, key):
    old_path = f.manifest[key]
    retarget(f, 23)
    f.manifest[key] = old_path
    with pytest.raises(ValueError, match="named source025 output paths"):
        f.q()


def test_v23_preserves_retired_v22_waiting_state_and_uses_shared_lock(f):
    old = f.q()
    assert old.step()["reason"] == "rec_normal_endpoint_not_complete"
    old_state = old.state_path.read_bytes()
    old_manifest = deepcopy(f.manifest)
    retarget(f, 23)
    new = f.q()
    assert new.lock == old.lock
    assert new.state_path != old.state_path
    assert new.signal_intent.name == ".task_campaign_boundary_v23.signal.intent.json"
    assert new.intent.name == ".task_campaign_boundary_v23.launch.intent.json"
    assert new.migration_receipt()["schema"] == "task-plan-revision-v23"
    with old.lock.open("a") as ownership:
        fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert new.run(once=True) == 0
        assert not new.state_path.exists()
    assert new.step()["reason"] == "rec_normal_endpoint_not_complete"
    assert old.state_path.read_bytes() == old_state
    assert boundary.read_json(old.state_path)["manifest_sha256"] == boundary.BoundaryContinuationQueue(
        old_manifest, proc_root=f.proc, adapter_factory=lambda q: FakeAdapter(q, f)).config_sha256
    assert not f.signals and not f.launches


def test_v23_refuses_live_old_observer_before_any_boundary_action(f):
    retarget(f, 23)
    proc(f, 909, [sys.executable, "-B", str(f.root / "boundary_observer_024" / Path(boundary.__file__).name),
        "--manifest", str(f.root / "boundary_observer_024/manifest.json")], starttime=7000)
    q = f.q()
    result = q.step(check_only=True)
    assert result["status"] == "attention" and "retire" in result["detail"]
    assert f.preflight_calls == 0 and not q.state_path.exists()
    assert not q.signal_intent.exists() and not q.intent.exists()
    shutil.rmtree(f.proc / "909")
    assert q.step(check_only=True)["reason"] == "rec_normal_endpoint_not_complete"
    assert not f.signals and not f.launches


@pytest.mark.parametrize("target,other", [(22, 23), (23, 22)])
@pytest.mark.parametrize("artifact", ["signal", "launch", "revision", "receipt"])
def test_other_target_action_artifacts_refuse_supersession(f, target, other, artifact):
    retarget(f, target)
    index = {22: 24, 23: 25}[other]
    name = {"signal": f".task_campaign_boundary_v{other}.signal.intent.json",
            "launch": f".task_campaign_boundary_v{other}.launch.intent.json",
            "revision": f"source_revision_{index:03d}.json",
            "receipt": f"boundary_continuation_{index:03d}.receipt.json"}[artifact]
    write(f.root / name, {"never_replace": True})
    q = f.q()
    result = q.step(check_only=True)
    assert result["status"] == "attention" and "action artifact" in result["detail"]
    assert f.preflight_calls == 0 and not f.signals and not f.launches
    assert boundary.read_json(f.root / name) == {"never_replace": True}


def test_old_target_signal_intent_race_blocks_v23_signal(f):
    retarget(f, 23)
    q, _ = admitted(f)
    f.signal_race = lambda: write(f.root / ".task_campaign_boundary_v22.signal.intent.json", {})
    assert q.step()["status"] == "attention"
    assert not q.signal_intent.exists() and not f.signals


def test_old_target_intent_after_final_account_read_blocks_v23_launch(f):
    retarget(f, 23)
    q, _ = admitted(f)
    q.step(); remove_owners(f); archiver_finished(f)
    f.prepare_race = lambda n: write(f.root / ".task_campaign_boundary_v22.launch.intent.json", {}) if n == 2 else None
    assert q.step()["status"] == "attention"
    assert len(f.signals) == 1 and not f.launches and not q.intent.exists()
