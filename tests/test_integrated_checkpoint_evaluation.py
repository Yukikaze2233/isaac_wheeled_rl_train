"""CPU interface/integrity tests; fabricated child reports prove no physical ability."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from test_integrated_production import frozen, static_bundle
from test_integrated_lineage import checkpoint, metadata
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.evaluation import select_evaluation_cases

import evaluate_integrated_checkpoint as final_eval
import run_task_campaign as dispatcher


@pytest.fixture
def evaluation(frozen, monkeypatch):
    f = frozen
    for name in ("evaluate_chassis.py", "run_task_campaign.py", "train_chassis.py"):
        target = f.source / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts" / name, target)
    for row in f.plan["domains"]:
        config = json.loads((f.source / row["contract"]).read_text())
        for name in final_eval.evaluation_source_files(config, root=ROOT):
            target = f.source / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
    bundle = Path(config["asset_directory"])
    manifest = json.loads((ROOT / bundle / "manifest.json").read_text())
    for name in ("manifest.json", *manifest["files_sha256"]):
        target = f.source / bundle / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / bundle / name, target)
    monkeypatch.setattr(final_eval, "ROOT", f.source)
    output = f.source.parent / "same_checkpoint_assessment"
    shutil.rmtree(output, ignore_errors=True)
    args = SimpleNamespace(contract=f.source / production.PLAN_FILE, campaign_run_dir=f.run,
        checkpoint=Path(f.rec["path"]), checkpoint_sha256=f.rec["sha256"], output=output,
        device="cpu", max_runtime_seconds=60., preflight=False)
    job = final_eval.IntegratedCheckpointEvaluation(args)
    preserved = [f.claim, f.state, f.run / "consumption.json"]
    dispatcher.write_json(preserved[-1], {"old": "consumption", "unchanged": True})
    f.before = {path: path.read_bytes() for path in preserved}
    return f, job


def native_record(command):
    """Expected native report schema, solely for routing/cache negative tests."""
    checkpoint = Path(command[command.index("--checkpoint") + 1])
    contract = Path(command[command.index("--contract") + 1])
    config = json.loads(contract.read_text())
    seed = int(command[command.index("--seed") + 1])
    names = command[command.index("--cases") + 1:]
    settings = select_evaluation_cases(config["evaluation"], names)
    settings["seed"] = seed
    candidate = dict(checkpoint_sha256=campaign.digest(checkpoint), unfinished_episodes=0,
        actor_input_dim=config["actor_dim"], actor_output_dim=config["action_dim"],
        cross_asset_evaluation=False, source_asset_manifest_sha256=config["asset_manifest_sha256"],
        evaluated_asset_manifest_sha256=config["asset_manifest_sha256"],
        cases={name: dict(passed=True) for name in names})
    return dict(status="evaluated", candidates=[candidate], contract_sha256=campaign.digest(contract),
        source_sha256={name: campaign.digest(final_eval.ROOT / name) for name in
                       final_eval.evaluation_source_files(config, root=final_eval.ROOT)},
        device=command[command.index("--device") + 1],
        settings=settings, episodes_per_case=settings["episodes_per_case"],
        evaluated_case_names=[case["name"] for case in settings["cases"]],
        script_sha256=campaign.digest(Path(command[2])), deterministic_actor=True,
        random_pushes=False, case_resampling=False)


def fake_children(monkeypatch, mutate=None, fail_first=False):
    calls = []

    class Child:
        def __init__(self, command, **kwargs):
            calls.append(command)
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            record = native_record(command)
            self.returncode = 1 if fail_first and len(calls) == 1 else 0
            if self.returncode:
                record["status"] = "failed"
            if mutate:
                mutate(record)
            for name in record["source_sha256"]:
                target = output / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(final_eval.ROOT / name, target)
            dispatcher.write_json(output / "evaluation.json", record)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(dispatcher.subprocess, "Popen", Child)
    return calls


def test_cpu_preflight_has_explicit_six_domain_plan_and_no_launch(evaluation, monkeypatch):
    f, job = evaluation
    monkeypatch.setattr(dispatcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("CPU preflight cannot launch native"))
    calls, preflight = [], final_eval.preflight

    def checked_preflight(*args, **kwargs):
        calls.append(args)
        return preflight(*args, **kwargs)

    monkeypatch.setattr(final_eval, "preflight", checked_preflight)
    report = job.prepare()
    job.report.update(status="cpu_preflight_only", native_scene_started=False)
    job.persist()
    assert tuple(row["name"] for row in report["inputs"]["domains"]) == campaign.DOMAINS
    assert report["inputs"]["selected_checkpoint_scope"] == "legacy_research_candidate"
    assert report["inputs"]["checkpoint_sha256"] == f.rec["sha256"]
    assert report["qualification"] is None and report["PPO_updates"] == 0
    assert len(calls) == 1
    source = json.loads(job.checkpoint.with_suffix(".contract.json").read_text())
    bundle = f.source / source["asset_directory"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert {str((bundle / name).resolve()): sha for name, sha in manifest["files_sha256"].items()}.items() <= report["inputs"]["files_sha256"].items()
    assert report["inputs"]["files_sha256"][str(f.source / source["control_math_source"])] == source["control_math_sha256"]
    assert f.before == {path: path.read_bytes() for path in f.before}


def test_all_domains_and_both_seeds_use_one_actor_without_account_writes(evaluation, monkeypatch):
    f, job = evaluation
    calls = fake_children(monkeypatch)
    report = job.run_assessment()
    assert len(calls) == 12
    assert {command[command.index("--checkpoint") + 1] for command in calls} == {f.rec["path"]}
    assert [command[command.index("--role") + 1] for command in calls] == ["quick", "confirmation"] * 6
    assert tuple(report["domains"]) == campaign.DOMAINS
    assert {row["checkpoint_sha256"] for row in report["domains"].values()} == {f.rec["sha256"]}
    assert report["all_required_passed_both_seeds"] is True
    assert report["qualification"] is report["deployment_checkpoint"] is report["accepted_stage"] is None
    assert report["training_account_written"] is False
    assert f.before == {path: path.read_bytes() for path in f.before}
    assert not (f.run / "flat/evaluations").exists()


@pytest.mark.parametrize("change", ["model", "seal", "contract", "missing_seal", "caller_sha", "plan"])
def test_rejects_tampered_or_unsealed_inputs(evaluation, change):
    _, job = evaluation
    if change == "model":
        job.checkpoint.write_bytes(b"changed model")
    elif change == "missing_seal":
        job.checkpoint.with_suffix(".checkpoint.json").unlink()
    elif change == "caller_sha":
        job.expected_sha = "0" * 64
    elif change == "plan":
        job.plan = deepcopy(job.plan)
        job.plan["domains"][2]["parent_domain"] = "flat"
    else:
        path = job.checkpoint.with_suffix(".checkpoint.json" if change == "seal" else ".contract.json")
        value = json.loads(path.read_text())
        value["sealed" if change == "seal" else "actor_layout"] = False
        dispatcher.write_json(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        job.prepare()


def test_live_campaign_lock_forbids_every_child(evaluation, monkeypatch):
    _, job = evaluation
    monkeypatch.setattr(dispatcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("lock must reject before launch"))
    with job.lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            job.run_assessment()
    assert not job.report_path.exists()


def test_live_orphan_worker_forbids_every_child(evaluation, monkeypatch):
    _, job = evaluation
    monkeypatch.setattr(dispatcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("owner must reject before launch"))
    monkeypatch.setattr(final_eval.TaskCampaign, "assert_no_live_workers", lambda _: (_ for _ in ()).throw(ValueError("live PPO")))
    with pytest.raises(ValueError, match="live PPO"):
        job.run_assessment()


@pytest.mark.parametrize("change", ["model_sha", "contract_sha", "seed", "settings", "missing_case", "unfinished", "script_sha", "actor_dim", "source_map"])
def test_rejects_incomplete_or_falsely_labelled_reports(evaluation, monkeypatch, change):
    f, job = evaluation

    def mutate(record):
        candidate = record["candidates"][0]
        if change in ("model_sha", "actor_dim", "unfinished"):
            key = {"model_sha": "checkpoint_sha256", "actor_dim": "actor_input_dim", "unfinished": "unfinished_episodes"}[change]
            candidate[key] = "0" * 64 if change == "model_sha" else 1
        elif change == "missing_case":
            candidate["cases"].pop(next(iter(candidate["cases"])))
        elif change == "settings":
            record["settings"]["episode_seconds"] += 1
        elif change == "seed":
            record["settings"]["seed"] = 3
        elif change == "source_map":
            record["source_sha256"].pop("src/wheeled_tasks/chassis/env.py")
        else:
            record["contract_sha256" if change == "contract_sha" else "script_sha256"] = "0" * 64

    calls = fake_children(monkeypatch, mutate)
    with pytest.raises(ValueError):
        job.run_assessment()
    assert len(calls) == 1 and job.report["status"] == "failed"
    assert f.before == {path: path.read_bytes() for path in f.before}


def test_failed_attempt_retry_preserves_old_evidence_and_completed_cache(evaluation, monkeypatch):
    f, job = evaluation
    calls = fake_children(monkeypatch, fail_first=True)
    with pytest.raises(RuntimeError, match="failed or stopped"):
        job.run_assessment()
    first = next(job.root.glob("flat/evaluations/*/290929/evaluation.json"))
    original = first.read_bytes()
    retry = final_eval.IntegratedCheckpointEvaluation(job.args)
    report = retry.run_assessment()
    assert len(calls) == 13 and report["status"] == "assessed"
    assert first.read_bytes() == original
    assert first.parent.with_name("290929_retry_1").is_dir()
    assert report["attempt_history"][-1]["status"] == "failed"
    cached = final_eval.IntegratedCheckpointEvaluation(job.args)
    cached.run_assessment()
    assert len(calls) == 13
    assert f.before == {path: path.read_bytes() for path in f.before}


def test_changed_cached_report_hash_is_rejected(evaluation, monkeypatch):
    _, job = evaluation
    calls = fake_children(monkeypatch)
    job.run_assessment()
    evidence = next(iter(job.report["evaluations"]["flat"].values()))
    path = Path(evidence["file"])
    record = json.loads(path.read_text())
    record["candidates"][0]["cases"][next(iter(record["candidates"][0]["cases"]))]["passed"] = False
    dispatcher.write_json(path, record)
    with pytest.raises(ValueError, match="recorded evaluation SHA changed"):
        final_eval.IntegratedCheckpointEvaluation(job.args).run_assessment()
    assert len(calls) == 12


def test_output_cannot_contaminate_campaign_or_source(evaluation):
    f, job = evaluation
    for output in (f.run, f.run / "flat/evaluations/new", f.source / "reports/new", f.claim.parent / "new"):
        args = deepcopy(job.args)
        args.output = output
        with pytest.raises(ValueError, match="outside"):
            final_eval.IntegratedCheckpointEvaluation(args)


def test_normally_sealed_joint_endpoint_accepts_static_worker_contract(evaluation):
    f, job = evaluation
    meta = metadata(f, "stepjump", joint_actor=15000, domain_actor=15000, complete=True)
    reference = checkpoint(f, "stepjump", count=55000, task=15000, lineage=45000,
                           final=True, meta=meta)
    job.checkpoint = Path(reference["path"])
    job.expected_sha = reference["sha256"]
    report = job.prepare()
    assert report["inputs"]["source_domain"] == "stepjump"
    assert report["inputs"]["selected_checkpoint_scope"] == "joint_candidate"
    target = next(row for row in f.plan["domains"] if row["name"] == "stepjump")
    assert report["inputs"]["source_contract_sha256"] == target["sha256"]
    assert report["inputs"]["checkpoint_sha256"] == reference["sha256"]
    assert report["qualification"] is None


def test_source_change_cannot_reuse_old_completed_assessment(evaluation, monkeypatch):
    f, job = evaluation
    calls = fake_children(monkeypatch)
    job.run_assessment()
    path = f.source / "src/wheeled_tasks/chassis/env.py"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# Different native evaluation implementation.\n")
        with pytest.raises(ValueError):
            final_eval.IntegratedCheckpointEvaluation(job.args).run_assessment()
    finally:
        path.write_bytes(original)
    assert len(calls) == 12


@pytest.mark.parametrize("change", ["asset", "control"])
def test_physical_dependency_change_forbids_cached_assessment(evaluation, monkeypatch, change):
    f, job = evaluation
    calls = fake_children(monkeypatch)
    job.run_assessment()
    source = json.loads(job.checkpoint.with_suffix(".contract.json").read_text())
    bundle = f.source / source["asset_directory"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    path = (bundle / next(iter(manifest["files_sha256"]))) if change == "asset" else f.source / source["control_math_source"]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\nchanged physical input\n")
        with pytest.raises(ValueError, match="Immutable assessment input changed"):
            job.unchanged()
        with pytest.raises(ValueError):
            final_eval.IntegratedCheckpointEvaluation(job.args).run_assessment()
    finally:
        path.write_bytes(original)
    assert len(calls) == 12
    assert f.before == {path: path.read_bytes() for path in f.before}


def test_retained_source_snapshot_tampering_is_rejected(evaluation, monkeypatch):
    _, job = evaluation
    calls = fake_children(monkeypatch)
    job.run_assessment()
    evidence = next(iter(job.report["evaluations"]["flat"].values()))
    path = Path(evidence["file"]).parent / "source/src/wheeled_tasks/chassis/env.py"
    path.write_bytes(b"replaced snapshot")
    with pytest.raises(ValueError, match="Retained native"):
        final_eval.IntegratedCheckpointEvaluation(job.args).run_assessment()
    assert len(calls) == 12


@pytest.mark.parametrize("reason", ["deadline", "stop"])
def test_owned_child_ignoring_term_is_killed_and_original_lock_released(evaluation, monkeypatch, reason):
    _, job = evaluation
    calls = []
    monkeypatch.setattr(final_eval, "STOP_GRACE_SECONDS", 0.)
    monkeypatch.setattr(dispatcher.time, "sleep", lambda _: None)

    class Child:
        returncode = None

        def __init__(self, command, **kwargs):
            calls.append(self)
            self.terms, self.kills = 0, 0
            if reason == "deadline":
                job.deadline = 0.
            else:
                job.stop_requested = True

        def poll(self):
            return self.returncode

        def send_signal(self, signum):
            self.terms += 1

        def kill(self):
            self.kills += 1
            self.returncode = -9

    monkeypatch.setattr(dispatcher.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="failed or stopped"):
        job.run_assessment()
    assert calls[0].terms >= 1 and calls[0].kills == 1
    assert job.eval_child is None and job.report["status"] == "failed"
    with job.lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_native_source_helper_preserves_order_dedup_and_conditional_evidence(tmp_path):
    evidence = tmp_path / "evidence.json"
    dispatcher.write_json(evidence, dict(files_sha256={"original-actuator-source.json": "sha"}))
    config = dict(design_preflight={"file": "design.json"}, actor_observation_source="scut35",
        task_semantics={}, skill_specs={"group": {}}, reward_profile="named", motion_limits={"enabled": True},
        command_transport={"enabled": True}, usb_transport={"enabled": True}, signal_delay={},
        contact_domain={"enabled": True}, command_reference={"enabled": True}, response_migration="named",
        recovery_training={"enabled": True, "profiles_file": "profiles.json", "height_lookup_file": "height.json"},
        actuator_response={"enabled": True}, actuator_response_evidence={"file": "evidence.json"},
        dynamics_randomization={"enabled": True}, step_assist={"enabled": True}, new_asset_curriculum={"enabled": True})
    files = final_eval.evaluation_source_files(config, root=tmp_path)
    assert files[:3] == ["scripts/evaluate_chassis.py", "src/wheeled_tasks/chassis/env.py", "src/wheeled_tasks/chassis/eval_env.py"]
    assert len(files) == len(set(files))
    assert files.index("evidence.json") < files.index("original-actuator-source.json")
    assert {"design.json", "profiles.json", "height.json", "src/wheeled_tasks/chassis/signal_delay.py",
            "src/wheeled_tasks/chassis/response_migration.py", "src/wheeled_tasks/chassis/step_assist.py"} <= set(files)
    assert config["recovery_training"]["profiles_file"] == "profiles.json"
