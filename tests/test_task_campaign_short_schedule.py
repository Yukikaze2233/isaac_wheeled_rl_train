"""Named first-pass budgets preserve workers, parentage and charged history."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner
from test_task_campaign_v2 import continuous_job
from test_task_remote_resume_v2 import repack, validate as validate_remote


def read(name):
    return json.loads((ROOT / name).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


PLAN = read("contracts/v6_task_campaign_v9.json")
SOURCE = read("contracts/v6_task_campaign_v8.json")


def test_named_short_schedule_preserves_all_workers_parents_and_30_roles():
    receipt = campaign.validate_plan_migration(SOURCE, PLAN, PLAN["revision_migration"]["run_directory"])
    assert receipt["consumption"] == "preserved_no_refund"
    assert receipt["actor_critic_optimizer"] == "preserved"
    assert SOURCE["domains"] == PLAN["domains"]
    assert PLAN["domain_completion"]["schedule"]["end_actor_updates"] == campaign.SHORT_SCHEDULE_ENDPOINTS
    assert next(row for row in PLAN["domains"] if row["name"] == "speed")["parent_domain"] == "flat"
    assert len(runner.validate_contracts(PLAN)) == 30
    manifest = read(PLAN["revision_migration"]["manifest_file"])
    assert manifest["schedule_scope"] == campaign.SHORT_SCHEDULE_SCOPE


@pytest.mark.parametrize("revision,domain,endpoint", [
    ("v7", "speed", 3563), ("v8", "terrain", 1186),
    ("v9", "speed", 3564), ("v9", "terrain", 1187), ("v9", "flat", 13001),
    ("v9", "recovery", 6000), ("v9", "jump", 6000), ("v9", "stepjump", 6000),
    ("v9", "speed", 3563.0), ("v9", "terrain", True),
])
def test_short_budgets_are_v9_only_and_not_a_generic_lower_bound_exception(revision, domain, endpoint):
    plan = read(f"contracts/v6_task_campaign_{revision}.json")
    plan["domain_completion"]["schedule"]["end_actor_updates"][domain] = endpoint
    with pytest.raises(ValueError):
        campaign.validate_plan(plan)


@pytest.fixture
def short_revision(tmp_path, monkeypatch):
    new = tmp_path / "source_new"
    source, target = deepcopy(SOURCE), deepcopy(PLAN)
    run_dir = str(tmp_path / "train")
    for plan in (source, target):
        plan["revision_migration"]["run_directory"] = run_dir
    for row in SOURCE["domains"]:
        path = new / row["contract"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / row["contract"], path)
    write(new / "contracts/v6_task_campaign_v8.json", source)
    target["revision_migration"]["source_plan_sha256"] = campaign.digest(new / "contracts/v6_task_campaign_v8.json")
    fixture = dict(root=tmp_path, new=new, source=source, target=target)
    reseal(fixture)
    monkeypatch.setattr(campaign, "ROOT", new)
    return fixture


def reseal(fixture):
    new, target = fixture["new"], fixture["target"]
    write(new / "contracts/v6_task_campaign_v9.json", target)
    manifest = deepcopy(read("contracts/v6_task_campaign_migration_v9.json"))
    manifest.update(run_directory=str(fixture["root"] / "train"),
        source_plan_sha256=campaign.digest(new / "contracts/v6_task_campaign_v8.json"),
        target_plan_sha256=campaign.digest(new / "contracts/v6_task_campaign_v9.json"))
    write(new / "contracts/v6_task_campaign_migration_v9.json", manifest)


@pytest.mark.parametrize("mutation", [
    "other_completion_field", "parent_domain", "worker_path", "worker_sha",
    "total_ceiling", "source_checkpoint", "extra_binding", "wrong_manifest", "worker_bytes",
])
def test_resealed_manifest_cannot_authorize_other_changes(short_revision, mutation):
    fixture = short_revision
    target = fixture["target"]
    if mutation == "other_completion_field":
        target["domain_completion"]["review_interval_actor_updates"] = 250
    elif mutation == "parent_domain":
        target["domains"][2]["parent_domain"] = "recovery"
    elif mutation == "worker_path":
        target["domains"][2]["contract"] = target["domains"][0]["contract"]
    elif mutation == "worker_sha":
        target["domains"][2]["sha256"] = "0" * 64
    elif mutation == "total_ceiling":
        target["total_ceiling"] = 120000
    elif mutation == "source_checkpoint":
        target["source_checkpoint"] = "/other/model.pt"
    elif mutation == "extra_binding":
        target["revision_migration"]["refund"] = True
    elif mutation == "wrong_manifest":
        target["revision_migration"]["manifest_file"] = "contracts/alias.json"
    elif mutation == "worker_bytes":
        with (fixture["new"] / target["domains"][2]["contract"]).open("a") as stream:
            stream.write(" ")
    reseal(fixture)
    if mutation == "wrong_manifest":
        shutil.copyfile(fixture["new"] / "contracts/v6_task_campaign_migration_v9.json",
                        fixture["new"] / "contracts/alias.json")
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(fixture["source"], target, fixture["root"] / "train")


def test_v9_requires_its_consecutive_v8_predecessor():
    with pytest.raises(ValueError, match="consecutive"):
        campaign.validate_plan_migration(read("contracts/v6_task_campaign_v7.json"), PLAN,
                                         PLAN["revision_migration"]["run_directory"])


@pytest.mark.parametrize("wrong_env_count", [False, True])
def test_remote_named_v9_receipt_preserves_existing_command_and_logs(short_revision, wrong_env_count):
    fixture = short_revision
    old, new = fixture["root"] / "source_old", fixture["new"]
    (old / "contracts").mkdir(parents=True)
    shutil.copyfile(new / "contracts/v6_task_campaign_v8.json", old / "contracts/v6_task_campaign_v8.json")
    for directory in (old, new):
        (directory / "scripts").mkdir()
        (directory / "scripts/run_task_campaign.py").write_text("# Never executed in this test.\n")
    for name in ("src/wheeled_tasks/__init__.py", "src/wheeled_tasks/chassis/__init__.py",
                 "src/wheeled_tasks/chassis/task_campaign.py"):
        path = new / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    command = [sys.executable, "-B", str(old / "scripts/run_task_campaign.py"),
               "--contract", str(old / "contracts/v6_task_campaign_v8.json"),
               "--run-dir", str(fixture["root"] / "train"), "--num-envs", "4096", "--research"]
    write(fixture["root"] / "job.json", dict(pid=2147483647, command=command))
    (fixture["root"] / "train.log").write_text("old sealed evidence\n")
    revised = [arg.replace(str(old), str(new)) for arg in command]
    revised[revised.index("--contract") + 1] = str(new / "contracts/v6_task_campaign_v9.json")
    if wrong_env_count:
        revised[revised.index("--num-envs") + 1] = "20000"
    fixture.update(old=old, archive=fixture["root"] / "source.tar.gz",
                   receipt_path=fixture["root"] / "revision.json")
    fixture["receipt"] = dict(schema="task-plan-revision-v9",
        previous_job_sha256=campaign.digest(fixture["root"] / "job.json"), command=revised,
        source_commit="a" * 40, old_source_root=str(old), new_source_root=str(new),
        new_contract_relative_path="contracts/v6_task_campaign_v9.json",
        old_contract_sha256=campaign.digest(old / "contracts/v6_task_campaign_v8.json"),
        new_contract_sha256=campaign.digest(new / "contracts/v6_task_campaign_v9.json"),
        source_archive=str(fixture["archive"]))
    repack(fixture)
    result = validate_remote(fixture)
    assert result.returncode == (3 if wrong_env_count else 0), result.stdout + result.stderr
    assert (fixture["root"] / "train.log").read_text() == "old sealed evidence\n"


@pytest.mark.parametrize("domain,endpoint", [("speed", 3563), ("terrain", 1186)])
def test_short_actor_endpoint_charges_adaptation_and_uses_flat_parent(tmp_path, monkeypatch, domain, endpoint):
    job = continuous_job(tmp_path, PLAN)
    row = next(row for row in PLAN["domains"] if row["name"] == domain)
    flat_row = PLAN["domains"][0]
    source = tmp_path / "flat/attempt_007/train/model_final.pt"
    source.parent.mkdir(parents=True)
    infos = dict(task_campaign=dict(domain="flat"), task_actor_update=13000,
        successful_updates_total=14020, stage_actor_update=13368,
        contract_sha256=flat_row["sha256"], actor_updates_consumed=13980)
    torch.save(dict(infos=infos, actor="flat actor", critic="flat critic", optimizer="flat Adam"), source)
    write(source.with_suffix(".contract.json"), read(flat_row["contract"]))
    write(source.with_suffix(".checkpoint.json"), dict(infos, learning_updates=14020))
    job.records = {
        "flat/attempt_007/train": dict(domain="flat", consumed_updates=13020, actor_updates=13000),
        "recovery/attempt_001/train": dict(domain="recovery", consumed_updates=2300, actor_updates=2280),
    }
    job.report.update(campaign.account_updates(PLAN, job.records))
    prior_consumed = job.report["consumed_updates"]
    job.report["domains"]["flat"] = dict(candidate_checkpoint=str(source))
    job.persist = lambda: job.report.update(campaign.account_updates(PLAN, job.records))
    monkeypatch.setattr(runner, "checkpoint_update_count", lambda _: 14020)
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: pytest.fail("no blocking evaluation"))
    calls = []

    def execute(command, log, directory):
        calls.append(command)
        assert command[command.index("--resume") + 1] == str(source)
        assert command[command.index("--task-actor-update") + 1] == "0"
        assert command[command.index("--updates") + 1] == str(endpoint + 10)
        assert command[command.index("--consumed-updates") + 1] == str(prior_consumed)
        write(directory / "completion.json", dict(status="completed", successful_updates=endpoint + 10,
            attempted_updates=endpoint + 10, actor_updates_in_block=endpoint))
        final = directory / "model_final.pt"
        torch.save(dict(infos=infos), final)
        write(final.with_suffix(".checkpoint.json"), dict(task_actor_update=endpoint))
        return 0

    monkeypatch.setattr(job, "execute", execute)
    job.run_domain(row)
    assert len(calls) == 1
    assert job.report["consumed_updates"] == prior_consumed + endpoint + 10
    assert job.report["domain_consumed_updates"]["recovery"] == 2300
    assert job.report["domain_consumed_updates"][domain] == endpoint + 10
    assert job.report["domains"][domain]["status"] == "domain_schedule_finished_unassessed"
    assert job.report["domains"][domain]["domain_qualified"] is None
    job.run_domain(row)
    assert len(calls) == 1, "completed endpoint must not reopen or refund history"
