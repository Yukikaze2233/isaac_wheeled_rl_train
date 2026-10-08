"""Real two-prefix, full-state normal boundary; CPU only, no PPO or scene."""
from copy import deepcopy
import ast
import json
import subprocess
from pathlib import Path
import shutil
import tarfile

import pytest
import torch

import run_task_campaign as dispatcher
import chassis_remote_job as wrapper
from test_integrated_early_production import early_frozen
from test_integrated_heading_revision import heading_frozen
from test_integrated_lineage import checkpoint, metadata, write
from test_integrated_production import arguments, frozen, job_for, read, static_bundle
from test_integrated_runtime import seal_clocks
from wheeled_tasks.chassis import integrated_completed_speed_revision as revision
from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis import integrated_runtime as runtime
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.policy_transfer import verify_learning_state_restore

ROOT = Path(__file__).resolve().parents[1]


def archive(source, path, commit):
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": commit}) as output:
        for file in sorted(source.rglob("*")):
            if file.is_file():
                output.add(file, arcname=str(file.relative_to(source)), recursive=False)


@pytest.fixture
def completed_speed(early_frozen, monkeypatch):
    f = early_frozen
    f.old_plan = deepcopy(f.plan)
    # Actual source029 paid speed remainder: 20k - 5093 = 14907,
    # including ten charged critic updates, hence only 14897 actor updates.
    f.records["speed"]["consumed_updates"] += 93
    f.records["speed"]["actor_updates"] += 93
    old_entry = production.recovery_entry_admission(f.binding, f.report)
    speed_meta = metadata(f, "speed", joint_actor=14897, domain_actor=14897, complete=True)
    f.speed = checkpoint(f, "speed", count=49000, task=14897, lineage=38959, final=True,
        meta=speed_meta, actor_consumed=48920)
    first = lineage.select_learning_source(f.binding, "speed", old_entry)
    f.speed = seal_clocks(f.speed, runtime.advance_clock_document(runtime.source_clocks(f.binding, first), 14897))
    f.speed_preterminal = deepcopy(f.report["domains"]["speed"])
    f.speed_record = dict(status="joint_domain_finished_unassessed", domain_qualified=None,
        candidate_checkpoint=f.speed["path"], candidate_checkpoint_sha256=f.speed["sha256"],
        integrated_lineage=speed_meta, schedule_completion=dict(clock="fresh_joint_domain_actor_updates",
            end_actor_update=15000, actual_actor_update=14897, reason="charged_cap", automatic_advance=True, domain_qualified=None))
    f.records["speed/attempt_001/train"] = dict(domain="speed", consumed_updates=14907, actor_updates=14897)
    totals = campaign.account_updates(f.plan, f.records)
    assert totals["consumed_updates"] == 49000 and totals["actor_updates_consumed"] == 48920
    f.report.update(contract_sha256=lineage.digest(f.source / production.artifact_files(26)["plan"]), **totals)
    f.report["domains"]["speed"] = deepcopy(f.speed_record)
    f.report["authorized_REC_early_stop"] = deepcopy(old_entry["record"])
    selected = lineage.select_learning_source(f.binding, "speed", old_entry, own_candidates=[f.speed])
    f.report["joint_learning_sources"] = {"speed": selected}
    f.report["integrated_continuation"] = dict(lineage_id=f.manifest["lineage_id"], plan_sha256=f.binding.sha256,
        origin_checkpoint_sha256=f.rec["sha256"])
    write(f.state, f.report)
    write(f.claim, dict(campaign_id=campaign.CAMPAIGN_ID, run_directory=str(f.run),
        contract_sha256=lineage.digest(f.source / production.artifact_files(26)["plan"]), attempts=f.records,
        rollback_refunds_consumed_updates=False, joint_learning_sources={"speed": selected},
        integrated_continuation=f.report["integrated_continuation"], **totals))
    experiment = f.run.parent
    link = experiment / "train"
    if not link.exists():
        link.symlink_to(f.run, target_is_directory=True)
    assert link.resolve() == f.run.resolve()
    (f.source / "scripts").mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "scripts/run_task_campaign.py", f.source / "scripts/run_task_campaign.py")
    shutil.copyfile(ROOT / "scripts/chassis_remote_job.py", f.source / "scripts/chassis_remote_job.py")
    f.source029 = experiment / "source029"
    if f.source029.exists():
        shutil.rmtree(f.source029)
    shutil.copytree(f.source, f.source029)
    relative = production.artifact_files(26)["plan"]
    prior_command = ["python", "-B", str(f.source / "scripts/run_task_campaign.py"), "--contract",
        str(f.source / relative), "--run-dir", str(f.run), "--num-envs", "4096"]
    old_command = [arg.replace(str(f.source), str(f.source029)) for arg in prior_command]
    write(experiment / "job.previous_024.json", dict(command=prior_command, pid=123))
    old_archive = experiment / "source029.tar"
    archive(f.source029, old_archive, revision.SOURCE029_COMMIT)
    write(experiment / "job.source_revision_024.json", dict(schema="task-source-revision-v1",
        source_commit=revision.SOURCE029_COMMIT, previous_job_sha256=lineage.digest(experiment / "job.previous_024.json"),
        command=old_command, old_source_root=str(f.source), new_source_root=str(f.source029),
        old_contract_sha256=lineage.digest(f.source / relative), new_contract_sha256=lineage.digest(f.source029 / relative),
        source_archive=str(old_archive), source_archive_sha256=lineage.digest(old_archive)))
    write(experiment / "job.json", dict(command=old_command, pid=456,
        previous_job="job.previous_024.json", source_revision_receipt="job.source_revision_024.json"))
    f.old_root = f.source
    f.old_binding = f.binding
    f.source = experiment / "source030"
    if f.source.exists():
        shutil.rmtree(f.source)
    shutil.copytree(f.source029, f.source)
    for file in ("src/wheeled_tasks/chassis/ordinary_large_stair.py",):
        (f.source / file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / file, f.source / file)
    production.write_bundle(f.source, source_root=str(f.source), run_root=str(f.run), version=27)
    f.plan = read(f.source / production.artifact_files(27)["plan"])
    f.binding = production.binding_for(f.plan, f.source, f.run)
    f.manifest = f.binding.read()
    for domain, _ in lineage.CHAIN:
        f.contracts[domain] = f.source / production.worker_file(domain, version=27)
    f.worker = read(f.contracts["terrain"])
    monkeypatch.setattr(dispatcher, "ROOT", f.source)
    monkeypatch.setattr(campaign, "ROOT", f.source)
    f.job = job_for(f, monkeypatch)
    f.job.args.contract = f.source / production.artifact_files(27)["plan"]
    return f


def migrate(f):
    f.job.open_account()
    row = next(r for r in f.plan["domains"] if r["name"] == "terrain")
    source = f.job.learning_source(row)
    receipt = production.make_admission(f.plan, f.source, f.run, f.job._joint_selection, f.job.records,
        account_path=f.claim, state_path=f.state, snapshot_directory=f.run / "terrain/attempt_next/train/admission")
    return row, source, receipt


def test_real_boundary_reentry_dispatch_and_native_full_state(completed_speed, monkeypatch):
    f = completed_speed
    parent_bytes = {p: p.read_bytes() for p in Path(f.speed["path"]).parent.glob("*") if p.is_file()}
    old_rec = deepcopy(f.report["domains"]["recovery"])
    # Exercise the unchanged original native terminal branch under its old
    # binding, not an invented completed record. The paid cap is reached at
    # 14897 actor + 10 critic; all worker-start paths are fail-closed.
    original = subprocess.check_output(["git", "show", revision.SOURCE029_COMMIT + ":scripts/run_task_campaign.py"], cwd=ROOT, text=True)
    def terminal_ast(text):
        tree = ast.parse(text)
        return ast.dump(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_integrated_domain"), include_attributes=False)
    assert terminal_ast(original) == terminal_ast((ROOT / "scripts/run_task_campaign.py").read_text())
    old_job = job_for(f, monkeypatch)
    old_job.plan, old_job.records, old_job.report = deepcopy(f.old_plan), deepcopy(f.records), read(f.state)
    old_job.args.contract = f.old_root / production.artifact_files(26)["plan"]
    old_job.report["domains"]["speed"] = deepcopy(f.speed_preterminal)
    write(f.state, old_job.report)
    with monkeypatch.context() as m:
        m.setattr(dispatcher, "ROOT", f.old_root)
        m.setattr(campaign, "ROOT", f.old_root)
        closed = revision.close_native_completed_speed(old_job, f.old_root)
    assert closed["PPO_updates"] == 0 and not closed["next_domain_dispatched"]
    assert old_job.report["domains"]["speed"]["schedule_completion"]["actual_actor_update"] == 14897
    assert old_job.report["domains"]["speed"]["schedule_completion"]["reason"] == "charged_cap"
    f.speed_record = deepcopy(old_job.report["domains"]["speed"])
    row, source, receipt = migrate(f)
    once = deepcopy(f.job.report), deepcopy(f.job.records), f.claim.read_bytes()
    f.job.open_account()
    assert (f.job.report, f.job.records, f.claim.read_bytes()) == once
    assert source == Path(f.speed["path"])
    assert receipt["selection"]["mode"] == revision.MODE
    assert receipt["selection"]["origin_recovery_actor_update"] == 14052
    assert receipt["selection"]["fresh_joint_actor_updates"] == 14897
    assert receipt["selection"]["fresh_domain_actor_updates"] == 0
    assert receipt["allowance"]["charged_critic_updates"] == 10
    assert receipt["allowance"]["requested_actor_updates"] == 15000
    assert receipt["component_clocks_at_source"]["component_actor_updates"] == dict(recovery=28949, speed=14897, terrain=0)
    assert production.authenticate_invocation(f.worker, arguments(f, receipt), f.source) == receipt
    for domain in ("terrain", "jump", "stepjump"):
        assert campaign.validate_task_budget(read(f.contracts[domain]), f.source)["total_ceiling"] == 121000
    from wheeled_tasks.chassis.integrated_training import critic_warmup_remaining
    assert critic_warmup_remaining(receipt, torch.load(source, weights_only=True)["infos"]) == 10
    parent_config = read(source.with_suffix(".contract.json"))
    assert campaign.validate_task_resume(parent_config, f.worker)["task_clock"] == "new_domain_zero"
    inventory = production.production_source_files(f.worker, receipt=receipt, root=f.source)
    assert "scripts/chassis_remote_job.py" in inventory
    assert production.worker_file("speed", version=26) in inventory
    effective = production.attach_runtime(f.worker, receipt, f.source)
    stairs = [s for s in effective["skill_specs"].values() if s.get("ordinary_large_stair_curriculum")]
    assert len(stairs) == 8
    assert all(s["ordinary_large_stair_curriculum"]["actor_allowance"] == 15000 for s in stairs)
    assert max(s["ordinary_large_stair_curriculum"]["activation_end"] for s in stairs) == 4000
    calls = []
    monkeypatch.setattr(f.job, "run_integrated_domain", lambda r: calls.append(r["name"]))
    for value in f.plan["domains"]:
        f.job.run_domain(value)
    assert calls == ["terrain", "jump", "stepjump"]
    assert f.job.report["domains"]["speed"] == f.speed_record
    assert f.job.report["domains"]["recovery"] == old_rec
    assert parent_bytes == {p: p.read_bytes() for p in parent_bytes}
    saved = torch.load(source, weights_only=True)
    algorithm = lineage._native_learning_template(f.worker)
    algorithm.load(saved, load_cfg=dict(optimizer=True), strict=True)
    verify_learning_state_restore(algorithm, saved)
    assert all(torch.equal(value, algorithm.save()["actor_state_dict"][key])
        for key, value in saved["actor_state_dict"].items())
    assert read(f.claim)["attempts"] == f.records
    assert read(f.claim)["consumed_updates"] == 49000
    assert read(f.claim)["actor_updates_consumed"] == 48920
    with pytest.raises(ValueError, match="REC re-entry"):
        production.prepare_admission(f.plan, f.source, f.run, account_path=f.claim, state_path=f.state)


@pytest.mark.parametrize("fault", ["incomplete", "wrong_parent", "refund", "wrong_source", "started_terrain"])
def test_bad_boundary_refuses_claim_mutation(completed_speed, fault):
    f = completed_speed
    if fault == "incomplete":
        path = Path(f.speed["path"]).parent / "completion.json"
        value = read(path)
        value["status"] = "stopped"
        write(path, value)
    elif fault == "wrong_parent":
        value = read(f.state)
        value["domains"]["speed"]["candidate_checkpoint_sha256"] = "0" * 64
        write(f.state, value)
    elif fault == "refund":
        value = read(f.claim)
        value["consumed_updates"] -= 1
        write(f.claim, value)
    elif fault == "wrong_source":
        path = f.run.parent / "job.source_revision_024.json"
        value = read(path)
        value["source_commit"] = "b" * 40
        write(path, value)
    else:
        value = read(f.claim)
        value["joint_learning_sources"]["terrain"] = {"unexpected": "old own progress"}
        write(f.claim, value)
    before = f.claim.read_bytes(), f.state.read_bytes()
    with pytest.raises((ValueError, KeyError)):
        f.job.open_account()
    assert before == (f.claim.read_bytes(), f.state.read_bytes())
    assert not (f.run / revision.DIRECTORY / "entry.json").exists()
    assert not (f.run / revision.DIRECTORY).exists()


def test_rejected_old_downstream_floor_can_be_restored_then_admitted(completed_speed):
    f = completed_speed
    original = f.claim.read_bytes()
    value = read(f.claim)
    value["joint_learning_sources"]["terrain"] = {"unexpected": "old own progress"}
    write(f.claim, value)
    with pytest.raises(ValueError, match="already started"):
        f.job.open_account()
    assert not (f.run / revision.DIRECTORY).exists()
    f.claim.write_bytes(original)
    _, _, receipt = migrate(f)
    assert receipt["selection"]["mode"] == revision.MODE


@pytest.mark.parametrize("cut", ["claim", "consumption", "state"])
def test_initial_native_three_write_cuts_recover_without_refund(completed_speed, monkeypatch, cut):
    f = completed_speed
    native_write = dispatcher.write_json
    fired = False
    targets = dict(claim=f.claim, consumption=f.run / "consumption.json", state=f.state)
    def fail_after(path, value):
        nonlocal fired
        native_write(path, value)
        if Path(path) == targets[cut] and not fired:
            fired = True
            raise OSError("injected native " + cut + " commit failure")
    monkeypatch.setattr(dispatcher, "write_json", fail_after)
    with pytest.raises(OSError, match="injected"):
        f.job.open_account()
    assert read(f.claim)["contract_sha256"] == lineage.digest(f.job.args.contract)
    if cut != "state":
        assert read(f.state)["contract_sha256"] != lineage.digest(f.job.args.contract)
    f.job.open_account()
    assert read(f.claim)["attempts"] == f.records
    assert read(f.claim)["consumed_updates"] == read(f.state)["consumed_updates"] == 49000
    assert read(f.state)["domains"]["speed"] == f.speed_record


def test_actual_wrapper_new_prefix_then_admission(completed_speed):
    f = completed_speed
    experiment = f.run.parent
    previous = read(experiment / "job.json")
    command = list(previous["command"])
    command[2] = str(f.source / "scripts/run_task_campaign.py")
    command[command.index("--contract")+1] = str(f.source / production.artifact_files(27)["plan"])
    path = experiment / "source030.tar"
    archive(f.source, path, "b" * 40)
    receipt_path = experiment / "source_revision_030.json"
    write(receipt_path, dict(schema="task-plan-revision-v27", source_commit="b" * 40,
        previous_job_sha256=lineage.digest(experiment / "job.json"), command=command,
        old_source_root=str(f.source029), new_source_root=str(f.source),
        old_contract_sha256=lineage.digest(f.source029 / production.artifact_files(26)["plan"]),
        new_contract_sha256=lineage.digest(f.source / production.artifact_files(27)["plan"]),
        new_contract_relative_path=production.artifact_files(27)["plan"], source_archive=str(path),
        source_archive_sha256=lineage.digest(path)))
    log = wrapper.prepare_job_log(experiment, command, resume=True, source_revision_receipt=receipt_path)
    previous_name = wrapper.preserve_previous_job(experiment, log, receipt_path)
    write(experiment / "job.json", dict(command=command, pid=789, previous_job=previous_name,
        source_revision_receipt="job.source_revision_" + log.name.removeprefix("train.resume_").removesuffix(".log") + ".json"))
    _, _, receipt = migrate(f)
    assert production.authenticate_invocation(f.worker, arguments(f, receipt), f.source) == receipt
    evidence = read(f.run / revision.DIRECTORY / "entry.json")
    old_job = read(evidence["before"]["source029_job"]["file"])
    assert old_job == previous and evidence["predecessor"]["source_revision_commit"] == revision.SOURCE029_COMMIT


def test_same_clock_cross_attempt_fork_refuses_normal_boundary(completed_speed):
    f = completed_speed
    original = Path(f.speed["path"])
    fork = f.run / "speed/attempt_fork/train"
    shutil.copytree(original.parent, fork)
    before = f.claim.read_bytes(), f.state.read_bytes()
    with pytest.raises(ValueError, match="fork across native attempts"):
        f.job.open_account()
    assert before == (f.claim.read_bytes(), f.state.read_bytes())
    assert not (f.run / revision.DIRECTORY).exists()


def test_forged_partial_claim_is_never_recovered(completed_speed, monkeypatch):
    f = completed_speed
    native_write = dispatcher.write_json
    def fail_after_claim(path, value):
        native_write(path, value)
        if Path(path) == f.claim:
            raise OSError("injected initial claim cut")
    with monkeypatch.context() as m:
        m.setattr(dispatcher, "write_json", fail_after_claim)
        with pytest.raises(OSError): f.job.open_account()
    value = read(f.claim)
    value["integrated_continuation"]["origin_checkpoint_sha256"] = "0" * 64
    write(f.claim, value)
    before = f.claim.read_bytes(), f.state.read_bytes()
    with pytest.raises(ValueError, match="exact frozen initial"):
        f.job.open_account()
    assert before == (f.claim.read_bytes(), f.state.read_bytes())


def test_new_terrain_partial_critic_resume_retains_origin_and_course(completed_speed):
    f = completed_speed
    for domain in ("terrain", "jump", "stepjump"):
        assert campaign.validate_task_budget(read(f.contracts[domain]), f.source)["total_ceiling"] == 121000
    row, source, receipt = migrate(f)
    old = torch.load(source, weights_only=True)
    source_seal = read(source.with_suffix(".checkpoint.json"))
    own_meta = metadata(f, "terrain", joint_actor=14897, domain_actor=0)
    saved = deepcopy(old)
    sha = lineage.digest(f.contracts["terrain"])
    saved["infos"].update(contract_sha256=sha, successful_updates_total=49003,
        learning_lineage_updates=38962, task_actor_update=0, critic_warmup_remaining=7,
        task_campaign=read(f.contracts["terrain"])["task_campaign"], integrated_lineage=own_meta,
        integrated_runtime_clocks=deepcopy(receipt["component_clocks_at_source"]))
    file = f.run / "terrain/attempt_critic/train/model.pt"
    file.parent.mkdir(parents=True)
    torch.save(saved, file)
    file.with_suffix(".contract.json").write_bytes(f.contracts["terrain"].read_bytes())
    own = dict(path=str(file), sha256=lineage.digest(file))
    seal = dict(source_seal, checkpoint_sha256=own["sha256"], contract_sha256=sha,
        learning_updates=49003, learning_lineage_updates=38962, task_actor_update=0,
        task_campaign=saved["infos"]["task_campaign"], integrated_lineage=own_meta,
        integrated_runtime_clocks=deepcopy(receipt["component_clocks_at_source"]))
    write(file.with_suffix(".checkpoint.json"), seal)
    progress = dict(parent_updates=49000, successful_updates=3, attempted_updates=3, actor_updates_in_block=0)
    write(file.parent / "progress.json", progress)
    f.job.account(file.parent, progress)
    f.job.open_account()
    resumed_source = f.job.learning_source(row)
    resumed = production.make_admission(f.plan, f.source, f.run, f.job._joint_selection, f.job.records,
        account_path=f.claim, state_path=f.state, snapshot_directory=f.run / "terrain/attempt_after_critic/admission")
    assert resumed_source == file and resumed["selection"]["mode"] == "resume_joint_own"
    assert resumed["selection"]["critic_warmup_remaining"] == resumed["allowance"]["charged_critic_updates"] == 7
    assert resumed["component_clocks_at_source"] == receipt["component_clocks_at_source"]
    assert resumed["allowance"]["consumed_updates"] == 49003
    assert resumed["allowance"]["actor_updates_consumed"] == 48920
    effective = production.attach_runtime(f.worker, resumed, f.source)
    assert effective["integrated_runtime_candidate"]["course_introduction"] == dict(
        schema="ordinary-large-stair-introduction-v1", component="terrain", origin_actor_update=0, actor_allowance=15000)
    algorithm = lineage._native_learning_template(f.worker)
    algorithm.load(saved, load_cfg=dict(optimizer=True), strict=True)
    verify_learning_state_restore(algorithm, saved)
