"""V7 -> v8 binds one recovery patch; all other task and budget state is exact."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType

import pytest

from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_chassis import resume_critic_warmup
from test_task_remote_resume_v2 import repack, validate as validate_remote


# Isolate migration authentication from the separately tested numerical patch.
# A fresh wrapper interpreter imports this same fixture helper from its archive.
PATCH_SOURCE = """def apply_recovery_takeover_revision(config, origin_actor_update):
    config['recovery_training']['complete_started_blend'] = True
    config['test_recovery_revision_origin'] = origin_actor_update
"""


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


def reseal(fixture):
    root = fixture["new"]
    source, target = fixture["source"], fixture["target"]
    for row in target["domains"]:
        row["sha256"] = campaign.digest(root / row["contract"])
    write(root / fixture["new_plan_name"], target)
    manifest = dict(schema="task-plan-revision-v8", source_plan_file=fixture["old_plan_name"],
        source_plan_sha256=campaign.digest(root / fixture["old_plan_name"]),
        target_plan_file=fixture["new_plan_name"], target_plan_sha256=campaign.digest(root / fixture["new_plan_name"]),
        run_directory=str(fixture["root"] / "train"), workers=[dict(source_file=a["contract"],
            source_sha256=a["sha256"], target_file=b["contract"], target_sha256=b["sha256"])
            for a, b in zip(source["domains"], target["domains"], strict=True)])
    write(root / fixture["manifest_name"], manifest)


@pytest.fixture
def revision(tmp_path, monkeypatch):
    new = tmp_path / "source_new"
    old_plan_name, new_plan_name = "contracts/v6_task_campaign_v7.json", "contracts/v6_task_campaign_v8.json"
    manifest_name = "contracts/v6_task_campaign_migration_v8.json"
    source = read(ROOT / old_plan_name)
    files = {old_plan_name, "contracts/v6_task_campaign_budget_v1.json", "contracts/v6_task_campaign_budget_v2.json"}
    for row in source["domains"]:
        files.add(row["contract"])
        config = read(ROOT / row["contract"])
        files.update((config["task_campaign"]["source_contract_file"], config["design_preflight"]["file"]))
    for name in files:
        (new / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, new / name)
    module = ModuleType("wheeled_tasks.chassis.recovery_revision")
    exec(PATCH_SOURCE, module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(campaign, "ROOT", new)
    original = read(new / campaign.RECOVERY_V6_FILE)
    worker = deepcopy(original)
    module.apply_recovery_takeover_revision(worker, 345)
    worker["task_campaign"]["revision_migration"] = dict(schema="task-worker-revision-v8",
        source_contract_file=campaign.RECOVERY_V6_FILE, source_contract_sha256=campaign.RECOVERY_V6_SHA256,
        origin_actor_update=345)
    design = "contracts/test_recovery_expectations_v8.json"
    write(new / design, {"scope": "migration fixture"})
    worker["design_preflight"] = dict(file=design, sha256=campaign.digest(new / design), stage="recovery")
    worker_name = "contracts/v6_task_campaign_recovery_worker_v8.json"
    write(new / worker_name, worker)
    target = deepcopy(source)
    target["contract_id"] = campaign.REVISION_V8
    target["revision_migration"] = dict(manifest_file=manifest_name,
        source_plan_sha256=campaign.digest(new / old_plan_name), run_directory=str(tmp_path / "train"))
    target["domains"][1]["contract"] = worker_name
    fixture = dict(root=tmp_path, new=new, source=source, target=target, worker=worker, original=original,
        worker_name=worker_name, old_plan_name=old_plan_name, new_plan_name=new_plan_name, manifest_name=manifest_name)
    reseal(fixture)
    return fixture


def test_named_v8_preserves_endpoints_parent_tree_full_state_and_all_consumption(revision):
    f = revision
    result = campaign.validate_plan_migration(f["source"], f["target"], f["root"] / "train")
    assert result["schema"] == "task-plan-revision-v8"
    assert result["actor_critic_optimizer"] == "preserved"
    assert result["task_clock"] == "restored" and result["consumption"] == "preserved_no_refund"
    assert f["source"]["domain_completion"] == f["target"]["domain_completion"]
    assert all(a == b for a, b in zip(f["source"]["domains"], f["target"]["domains"])
               if a["name"] != "recovery")
    for source in (f["original"], f["worker"]):
        receipt = campaign.validate_task_resume(source, f["worker"])
        assert receipt["task_clock"] == "restored" and not receipt["hardware_release"]
    assert campaign.validate_task_budget(f["worker"], f["new"])["total_ceiling"] == 121000
    charged = campaign.account_updates(f["target"], {"recovery/old": dict(
        domain="recovery", consumed_updates=370, actor_updates=360)})
    assert charged["consumed_updates"] == 1370 and charged["actor_updates_consumed"] == 1340


@pytest.mark.parametrize("origin", [-1, True, 0.0, "345", None])
def test_origin_must_be_nonnegative_integer(revision, origin):
    worker = deepcopy(revision["worker"])
    worker["task_campaign"]["revision_migration"]["origin_actor_update"] = origin
    with pytest.raises(ValueError, match="nonnegative actor origin"):
        campaign.validate_worker_revision(revision["original"], worker)


@pytest.mark.parametrize("key,value", [("target_num_envs", 20000), ("physics_dt", .001),
    ("actor_dim", 36), ("critic_dim", 82), ("success_event_bonus", 300.), ("learning_rate", .0002),
    ("resume_critic_warmup_updates", 0)])
def test_reconstructed_patch_rejects_undeclared_worker_changes(revision, key, value):
    worker = deepcopy(revision["worker"])
    worker[key] = value
    with pytest.raises(ValueError, match="outside the reviewed patch"):
        campaign.validate_worker_revision(revision["original"], worker)


def test_frozen_v6_source_parent_and_patch_origin_cannot_be_rebound(revision):
    f = revision
    for mutation in ("source_alias", "parent", "origin_effect", "unknown_binding"):
        worker = deepcopy(f["worker"])
        binding = worker["task_campaign"]["revision_migration"]
        if mutation == "source_alias":
            binding["source_contract_file"] = "contracts/alias_v6.json"
            shutil.copyfile(f["new"] / campaign.RECOVERY_V6_FILE, f["new"] / binding["source_contract_file"])
        elif mutation == "parent":
            worker["task_campaign"]["source_contract_file"] = campaign.RECOVERY_V6_FILE
            worker["task_campaign"]["source_contract_sha256"] = campaign.RECOVERY_V6_SHA256
        elif mutation == "origin_effect":
            binding["origin_actor_update"] += 1
        else:
            binding["unreviewed"] = True
        with pytest.raises(ValueError):
            campaign.validate_worker_revision(f["original"], worker)


@pytest.mark.parametrize("mutation", ["endpoint", "parent", "budget", "other_worker", "missing_recovery", "source_version"])
def test_resealed_v8_plan_cannot_expand_migration_scope(revision, mutation):
    f = revision
    if mutation == "endpoint":
        f["target"]["domain_completion"]["schedule"]["end_actor_updates"]["recovery"] = 16000
    elif mutation == "parent":
        f["target"]["domains"][1]["parent_domain"] = "A"
    elif mutation == "budget":
        f["target"]["domains"][1]["updates"] = 21000
    elif mutation == "source_version":
        f["source"]["contract_id"] = campaign.REVISION_V6
    elif mutation == "missing_recovery":
        f["target"]["domains"][1] = deepcopy(f["source"]["domains"][1])
    else:
        row = f["target"]["domains"][0]
        config = read(f["new"] / row["contract"])
        config["success_event_bonus"] = 31.
        row["contract"] = "contracts/unauthorized_flat.json"
        write(f["new"] / row["contract"], config)
    reseal(f)
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(f["source"], f["target"], f["root"] / "train")


def test_v8_adapts_critic_once_and_faults_resume_only_remaining_updates(revision):
    config = revision["worker"]
    infos = dict(contract_sha256=campaign.RECOVERY_V6_SHA256, task_campaign=dict(domain="recovery"),
                 successful_updates_total=14500, task_actor_update=345, critic_warmup_remaining=0)
    assert resume_critic_warmup(config, infos, contract_sha256="v8", task_change=True) == 10
    infos.update(contract_sha256="v8", critic_warmup_remaining=3)
    assert resume_critic_warmup(config, infos, contract_sha256="v8", task_change=True) == 3
    infos["critic_warmup_remaining"] = 0
    assert resume_critic_warmup(config, infos, contract_sha256="v8", task_change=True) == 0
    assert infos["task_actor_update"] == 345


@pytest.mark.parametrize("wrong_count", [False, True])
def test_remote_wrapper_accepts_only_authenticated_v8_without_resize(revision, wrong_count):
    f = revision
    old, new = f["root"] / "source_old", f["new"]
    (old / "contracts").mkdir(parents=True)
    shutil.copyfile(new / f["old_plan_name"], old / f["old_plan_name"])
    for directory in (old, new):
        (directory / "scripts").mkdir()
        (directory / "scripts/run_task_campaign.py").write_text("# Never executed in this test.\n")
    for name in ("src/wheeled_tasks/__init__.py", "src/wheeled_tasks/chassis/__init__.py",
                 "src/wheeled_tasks/chassis/task_campaign.py"):
        (new / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, new / name)
    (new / "src/wheeled_tasks/chassis/recovery_revision.py").write_text(PATCH_SOURCE)
    command = [sys.executable, "-B", str(old / "scripts/run_task_campaign.py"),
               "--contract", str(old / f["old_plan_name"]), "--run-dir", str(f["root"] / "train"),
               "--num-envs", "4096", "--research"]
    write(f["root"] / "job.json", dict(pid=2147483647, command=command))
    (f["root"] / "train.log").write_text("old evidence\n")
    revised = [arg.replace(str(old), str(new)) for arg in command]
    revised[revised.index("--contract") + 1] = str(new / f["new_plan_name"])
    if wrong_count:
        revised[revised.index("--num-envs") + 1] = "20000"
    f.update(old=old, archive=f["root"] / "source.tar.gz", receipt_path=f["root"] / "revision.json")
    f["receipt"] = dict(schema="task-plan-revision-v8", previous_job_sha256=campaign.digest(f["root"] / "job.json"),
        command=revised, source_commit="a" * 40, old_source_root=str(old), new_source_root=str(new),
        new_contract_relative_path=f["new_plan_name"], old_contract_sha256=campaign.digest(old / f["old_plan_name"]),
        new_contract_sha256=campaign.digest(new / f["new_plan_name"]), source_archive=str(f["archive"]))
    repack(f)
    result = validate_remote(f)
    assert result.returncode == (3 if wrong_count else 0), result.stdout + result.stderr
    assert (f["root"] / "train.log").read_text() == "old evidence\n"
