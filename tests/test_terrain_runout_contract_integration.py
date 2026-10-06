"""CPU entry/account checks for a future candidate; no native scene or PPO runs."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis import terrain_runout_revision as recipe
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.evaluation import fixed_suite_contract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import chassis_remote_job as remote
import run_task_campaign as runner
import train_chassis as train


def read(root, name):
    return json.loads((root / name).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Test copies may refer to immutable inputs; detach before any mutation.
    if path.is_symlink():
        path.unlink()
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def link_inputs(root, source, *, omit_outputs=False):
    root.mkdir()
    (root / "contracts").mkdir()
    for path in (source / "contracts").iterdir():
        if omit_outputs and str(path.relative_to(source)) in (
                recipe.PLAN_FILE, recipe.WORKER_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE):
            continue
        (root / "contracts" / path.name).symlink_to(path, target_is_directory=path.is_dir())
    for path in source.iterdir():
        if path.name != "contracts":
            (root / path.name).symlink_to(path, target_is_directory=path.is_dir())


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("runout_entries") / "frozen"
    outputs = (recipe.PLAN_FILE, recipe.WORKER_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE)
    before = {name: (ROOT / name).read_bytes() if (ROOT / name).exists() else None
              for name in outputs}
    link_inputs(root, ROOT, omit_outputs=True)
    recipe.freeze_contracts(root)
    assert before == {name: (ROOT / name).read_bytes() if (ROOT / name).exists() else None
                      for name in outputs}
    return root


@pytest.fixture
def frozen(baseline, tmp_path, monkeypatch):
    root = tmp_path / "frozen"
    link_inputs(root, baseline)
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(train, "ROOT", root)
    return root


def test_exact_plan_preserves_five_worker_bytes_schedule_and_parent_tree(frozen):
    source, target = read(frozen, recipe.SOURCE_PLAN_FILE), read(frozen, recipe.PLAN_FILE)
    receipt = campaign.validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
    assert receipt["schema"] == recipe.SCHEMA
    assert receipt["consumption"] == "preserved_no_refund" and "domain_reopening" not in receipt
    assert source["domain_completion"] == target["domain_completion"]
    assert target["domain_completion"]["schedule"]["end_actor_updates"] == campaign.ORDINARY_POSTURE_ENDPOINTS
    assert source["total_ceiling"] == target["total_ceiling"] == 121000
    for old, new in zip(source["domains"], target["domains"], strict=True):
        if old["name"] == "terrain":
            assert {k: v for k, v in old.items() if k not in ("contract", "sha256")} == {
                k: v for k, v in new.items() if k not in ("contract", "sha256")}
        else:
            assert old == new
            worker = read(frozen, old["contract"])
            assert (frozen / old["contract"]).read_bytes() == (ROOT / old["contract"]).read_bytes()
            design = worker["design_preflight"]["file"]
            assert (frozen / design).read_bytes() == (ROOT / design).read_bytes()


@pytest.mark.parametrize("source_file", [recipe.SOURCE_WORKER_FILE, recipe.OWN_SOURCE_WORKER_FILE])
def test_real_own_source_uses_existing_v18_authentication_once(frozen, monkeypatch, source_file):
    old, new = read(frozen, source_file), read(frozen, recipe.WORKER_FILE)
    original, calls = campaign.validate_worker_revision, []

    def traced(source, target):
        calls.append((source, target))
        return original(source, target)

    monkeypatch.setattr(campaign, "validate_worker_revision", traced)
    receipt = campaign.validate_task_resume(old, new)
    assert receipt["task_clock"] == "restored" and receipt["actor_critic_optimizer"] == "preserved"
    if source_file == recipe.OWN_SOURCE_WORKER_FILE:
        assert len(calls) == 2
        assert calls[1] == (old, read(frozen, recipe.SOURCE_WORKER_FILE))
    else:
        assert len(calls) == 1
    assert new["task_campaign"]["revision_migration"] == read(frozen, recipe.SOURCE_WORKER_FILE)["task_campaign"]["revision_migration"]


def test_source_audit_is_complete_before_projection_and_old_path_does_not_activate_addon(frozen, monkeypatch):
    old, new = read(frozen, recipe.SOURCE_WORKER_FILE), read(frozen, recipe.WORKER_FILE)
    inherited = train.scut_curriculum_source_files(old, root=frozen)
    sources = train.scut_curriculum_source_files(new, root=frozen)
    required = {recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE, recipe.SOURCE_DESIGN_FILE,
        recipe.OWN_SOURCE_WORKER_FILE, recipe.PLAN_FILE, recipe.WORKER_FILE, recipe.DESIGN_FILE,
        recipe.MANIFEST_FILE, *["src/wheeled_tasks/chassis/" + name + ".py" for name in (
            "terrain_runout", "terrain_runout_revision", "terrain_runout_runtime", "task", "env",
            "preflight", "evaluation", "task_campaign")],
        "scripts/train_chassis.py", "scripts/run_task_campaign.py", "scripts/chassis_remote_job.py"}
    assert set(inherited) | required == set(sources) and len(sources) == len(set(sources))
    assert all((frozen / name).is_file() for name in sources)
    with pytest.raises(ValueError):
        train.scut_curriculum_source_files(fixed_suite_contract(new), root=frozen)
    monkeypatch.setattr(recipe, "canonical_worker", lambda *a, **k: pytest.fail("Inactive addon was called"))
    assert train.scut_curriculum_source_files(old, root=frozen) == inherited
    assert train.scut_curriculum_source_files({}) == []


@pytest.mark.parametrize("fault", ["null_binding", "missing_binding", "bool_enabled", "other_domain", "warmup", "reward", "primary_binding"])
def test_bad_addon_cannot_fall_back_to_valid_v18_source_audit(frozen, fault):
    target = read(frozen, recipe.WORKER_FILE)
    if fault == "null_binding":
        target["task_campaign"]["terrain_runout_revision_migration"] = None
    elif fault == "missing_binding":
        target["task_campaign"].pop("terrain_runout_revision_migration")
    elif fault == "bool_enabled":
        target["terrain_route_runout"]["enabled"] = 1
    elif fault == "other_domain":
        target["task_campaign"]["domain"] = "speed"
    elif fault == "warmup":
        target["resume_critic_warmup_updates"] = 0
    elif fault == "reward":
        target["unreviewed_reward"] = 1
    else:
        target["task_campaign"]["revision_migration"]["schema"] = "task-worker-revision-v22"
    with pytest.raises(ValueError):
        train.scut_curriculum_source_files(target, root=frozen)
    with pytest.raises(ValueError):
        campaign.validate_worker_revision(read(frozen, recipe.OWN_SOURCE_WORKER_FILE), target)


@pytest.mark.parametrize("file", [recipe.SOURCE_WORKER_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE])
def test_bound_input_or_frozen_output_tampering_is_rejected(frozen, file):
    changed = read(frozen, file)
    changed["unreviewed"] = True
    write(frozen / file, changed)
    with pytest.raises(ValueError):
        train.scut_curriculum_source_files(read(frozen, recipe.WORKER_FILE), root=frozen)


def test_fixed_exam_remains_exact_and_runout_is_explicitly_disabled(frozen):
    old, new = map(fixed_suite_contract, (read(frozen, recipe.SOURCE_WORKER_FILE), read(frozen, recipe.WORKER_FILE)))
    assert new["terrain_route_runout"] == dict(recipe.SETTINGS, enabled=False)
    assert len(new["evaluation"]["cases"]) == len(old["evaluation"]["cases"]) == 322
    restored = deepcopy(new)
    restored.pop("terrain_route_runout")
    restored["task_campaign"].pop("terrain_runout_revision_migration")
    restored["task_campaign"]["worker_revision_id"] = old["task_campaign"]["worker_revision_id"]
    restored["design_preflight"] = old["design_preflight"]
    assert restored == old


def test_complete_thirty_role_design_preflight_uses_canonical_train_worker(frozen):
    assert len(runner.validate_contracts(read(frozen, recipe.PLAN_FILE))) == 30


def account_job(frozen, run, monkeypatch):
    source, target = read(frozen, recipe.SOURCE_PLAN_FILE), read(frozen, recipe.PLAN_FILE)
    job = object.__new__(runner.TaskCampaign)
    job.plan, job.root, job.args = target, run, SimpleNamespace(contract=frozen / recipe.PLAN_FILE, device="cpu")
    job.claim, job.report, job.records, job._session_started = run / "shared/account.json", {}, {}, None
    monkeypatch.setattr(job, "assert_no_live_workers", lambda: None)

    def authenticate(before, after, directory):
        assert directory == run
        result = campaign.validate_plan_migration(before, after, after["revision_migration"]["run_directory"])
        result["run_directory"] = str(run)
        return result

    monkeypatch.setattr(runner, "validate_plan_migration", authenticate)
    records = {"terrain/attempt_002/train": dict(domain="terrain", consumed_updates=2206, actor_updates=2186),
        "recovery/attempt_009/train": dict(domain="recovery", consumed_updates=210, actor_updates=200)}
    # Synthetic durable metadata tests preservation, not real tensor restoration.
    reopening = dict(schema="task-domain-reopen-v18", origin_actor_update=2186, endpoint_actor_update=15000,
        source_contract_sha256=recipe.OWN_SOURCE_WORKER_SHA, source_checkpoint="real-own-v15-seal")
    state = dict(contract_sha256=recipe.SOURCE_PLAN_SHA, domains={"terrain": dict(reopening=reopening,
        completion_history=[dict(plan_revision="v15", status="unassessed")], candidate_checkpoint="real-own-v15-seal")},
        evaluations={"old": "retained"}, consumed_review_tokens=["prior"], **campaign.account_updates(source, records))
    write(run / "campaign_state.json", state)
    write(job.claim, dict(campaign_id=campaign.CAMPAIGN_ID, run_directory=str(run),
        contract_sha256=recipe.SOURCE_PLAN_SHA, attempts=records, **campaign.account_updates(source, records)))
    return job, state, records


@pytest.mark.parametrize("crash", [False, True])
def test_account_migration_replay_retains_consumption_and_existing_reopening(frozen, tmp_path, monkeypatch, crash):
    job, state, records = account_job(frozen, tmp_path / "run", monkeypatch)
    original_write = runner.write_json
    if crash:
        def power_loss(path, value):
            if Path(path) == job.root / "campaign_state.json":
                raise RuntimeError("power loss after claim commit")
            return original_write(path, value)
        monkeypatch.setattr(runner, "write_json", power_loss)
        with pytest.raises(RuntimeError, match="power loss"):
            job.open_account()
        assert read(job.root, "campaign_state.json") == state
        assert json.loads(job.claim.read_text())["contract_sha256"] == recipe.digest(frozen / recipe.PLAN_FILE)
        monkeypatch.setattr(runner, "write_json", original_write)
    else:
        job.open_account()
    job.report, job.records = {}, {}
    job.open_account()
    once = deepcopy(job.report)
    job.open_account()
    assert job.report == once and job.records == records
    assert job.report["domains"] == state["domains"]
    assert job.report["evaluations"] == state["evaluations"]
    assert job.report["consumed_review_tokens"] == state["consumed_review_tokens"]
    assert job.report["consumed_updates"] == state["consumed_updates"]
    assert "domain_reopening" not in job.report["revision_migration"]


def test_one_ten_critic_adaptation_is_charged_without_advancing_actor(frozen):
    plan, worker = read(frozen, recipe.PLAN_FILE), read(frozen, recipe.WORKER_FILE)
    infos = dict(contract_sha256=recipe.OWN_SOURCE_WORKER_SHA, successful_updates_total=2206,
        critic_warmup_remaining=0, task_campaign=read(frozen, recipe.OWN_SOURCE_WORKER_FILE)["task_campaign"])
    checksum = recipe.digest(frozen / recipe.WORKER_FILE)
    assert train.resume_critic_warmup(worker, infos, contract_sha256=checksum, task_change=True) == 10
    assert plan["domain_completion"]["schedule"]["end_actor_updates"]["terrain"] - 2186 + 10 == 12824
    for remaining in (10, 5, 0):
        same = dict(infos, contract_sha256=checksum, critic_warmup_remaining=remaining)
        assert train.resume_critic_warmup(worker, same, contract_sha256=checksum, task_change=True) == remaining
    records = {"terrain/attempt_002/train": dict(domain="terrain", consumed_updates=2206, actor_updates=2186)}
    before = campaign.account_updates(plan, records)
    adapted = campaign.update_attempt(plan, records, "terrain/attempt_future/train", "terrain",
        dict(successful_updates=10, attempted_updates=10, actor_updates_in_block=0))
    after = campaign.account_updates(plan, adapted)
    assert after["domain_consumed_updates"]["terrain"] == before["domain_consumed_updates"]["terrain"] + 10
    assert after["consumed_updates"] == before["consumed_updates"] + 10
    assert after["actor_updates_consumed"] == before["actor_updates_consumed"]


@pytest.mark.parametrize("defect", [None, "missing_header", "wrong_commit", "schema_mismatch", "command"])
def test_real_wrapper_archive_admits_v22_and_rejects_identity_changes(frozen, tmp_path, monkeypatch, defect):
    old, new, run = (tmp_path / name for name in ("old", "new", "wrapper"))
    names = [recipe.SOURCE_PLAN_FILE, recipe.PLAN_FILE, "scripts/run_task_campaign.py"]
    for root in (old, new):
        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((frozen / name).read_bytes() if name.endswith(".json") else b"# never executed\n")
    command = ["/env/python", str(old / names[2]), "--contract", str(old / recipe.SOURCE_PLAN_FILE),
        "--num-envs", "4096", "--run-dir", str(run / "train")]
    previous = dict(command=command)
    write(run / "job.json", previous)
    changed = list(command)
    changed[1], changed[3] = str(new / names[2]), str(new / recipe.PLAN_FILE)
    archive, commit = tmp_path / "source.tar.gz", "a" * 40
    header = {} if defect == "missing_header" else {"comment": "b" * 40 if defect == "wrong_commit" else commit}
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT, pax_headers=header) as bundle:
        for name in names:
            data = (new / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    receipt = dict(schema=recipe.SCHEMA, previous_job_sha256=remote.sha(run / "job.json"), command=changed,
        source_commit=commit, old_source_root=str(old), new_source_root=str(new),
        new_contract_relative_path=recipe.PLAN_FILE, old_contract_sha256=remote.sha(old / recipe.SOURCE_PLAN_FILE),
        new_contract_sha256=remote.sha(new / recipe.PLAN_FILE), source_archive=str(archive),
        source_archive_sha256=remote.sha(archive))
    if defect == "schema_mismatch":
        receipt["schema"] = "task-plan-revision-v21"
    elif defect == "command":
        changed[changed.index("--num-envs") + 1] = "20000"
    path, calls = tmp_path / "revision.json", []
    write(path, receipt)

    def validate(source, target, directory):
        assert source == read(frozen, recipe.SOURCE_PLAN_FILE)
        assert target == read(frozen, recipe.PLAN_FILE) and directory == run / "train"
        calls.append(True)
    monkeypatch.setattr(campaign, "validate_plan_migration", validate)
    if defect is None:
        assert remote.validate_source_revision(run, changed, previous, path) == receipt and calls == [True]
    else:
        with pytest.raises(ValueError):
            remote.validate_source_revision(run, changed, previous, path)
        assert calls == []
