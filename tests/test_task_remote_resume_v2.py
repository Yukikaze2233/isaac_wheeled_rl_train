"""Exercise the frozen-source wrapper with the actual task-plan v2 validator."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from chassis_remote_job import sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def repack(fixture):
    archive = fixture["archive"]
    with tarfile.open(archive, "w:gz", pax_headers={"comment": fixture["receipt"]["source_commit"]}) as target:
        for path in sorted(fixture["new"].rglob("*")):
            if path.is_file():
                target.add(path, arcname=str(path.relative_to(fixture["new"])), recursive=False)
    fixture["receipt"]["source_archive_sha256"] = sha(archive)
    write(fixture["receipt_path"], fixture["receipt"])


@pytest.fixture
def revision(tmp_path):
    old, new = tmp_path / "source_old", tmp_path / "source_new"
    old_plan_name, new_plan_name = "contracts/v6_task_campaign_v1.json", "contracts/v6_task_campaign_v2.json"
    plan = json.loads((ROOT / new_plan_name).read_text())
    manifest_name = plan["revision_migration"]["manifest_file"]
    manifest = json.loads((ROOT / manifest_name).read_text())
    files = {old_plan_name, new_plan_name, manifest_name}
    for pin in manifest["workers"]:
        for key in ("source_file", "target_file"):
            path = pin[key]
            files.add(path)
            worker = json.loads((ROOT / path).read_text())
            files.add(worker["task_campaign"]["source_contract_file"])
            files.add(worker["design_preflight"]["file"])
    files.update(("src/wheeled_tasks/__init__.py", "src/wheeled_tasks/chassis/__init__.py",
                  "src/wheeled_tasks/chassis/task_campaign.py"))
    for name in files:
        target = new / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (old / "contracts").mkdir(parents=True)
    shutil.copyfile(ROOT / old_plan_name, old / old_plan_name)
    for directory in (old, new):
        (directory / "scripts").mkdir()
        (directory / "scripts/run_task_campaign.py").write_text("# Frozen runner placeholder; never executed.\n")
    run_dir = str(tmp_path / "train")
    plan["revision_migration"]["run_directory"] = run_dir
    write(new / new_plan_name, plan)
    manifest.update(run_directory=run_dir, target_plan_sha256=sha(new / new_plan_name))
    write(new / manifest_name, manifest)
    command = [sys.executable, "-B", str(old / "scripts/run_task_campaign.py"),
               "--contract", str(old / old_plan_name), "--run-dir", run_dir,
               "--num-envs", "20000", "--max-runtime-seconds", "15552000", "--research"]
    write(tmp_path / "job.json", {"pid": 2147483647, "command": command})
    (tmp_path / "train.log").write_text("old training evidence\n")
    revised = [arg.replace(str(old), str(new)) for arg in command]
    revised[revised.index("--contract") + 1] = str(new / new_plan_name)
    archive, receipt_path = tmp_path / "source.tar.gz", tmp_path / "revision.json"
    receipt = dict(schema="task-plan-revision-v2", previous_job_sha256=sha(tmp_path / "job.json"),
        command=revised, source_commit="a"*40, old_source_root=str(old), new_source_root=str(new),
        new_contract_relative_path=new_plan_name, old_contract_sha256=sha(old / old_plan_name),
        new_contract_sha256=sha(new / new_plan_name), source_archive=str(archive))
    fixture = dict(root=tmp_path, old=old, new=new, archive=archive, receipt_path=receipt_path,
                   receipt=receipt, manifest_name=manifest_name, new_plan_name=new_plan_name)
    repack(fixture)
    return fixture


def validate(fixture):
    # A fresh interpreter mirrors the wrapper process and proves it imports the
    # validator from the newly authenticated source, not pytest's module cache.
    script = """
import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('wrapper',sys.argv[1]); wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
root, receipt=Path(sys.argv[2]),Path(sys.argv[3])
command=json.loads(receipt.read_text())['command']
try:
 result=wrapper.prepare_job_log(root,command,resume=True,source_revision_receipt=receipt)
 print(json.dumps({'status':'passed','log':str(result)}))
except Exception as e:
 print(json.dumps({'status':'rejected','type':type(e).__name__,'error':str(e)}))
 sys.exit(3)
"""
    return subprocess.run([sys.executable, "-B", "-c", script, str(ROOT / "scripts/chassis_remote_job.py"),
                           str(fixture["root"]), str(fixture["receipt_path"])],
                          text=True, capture_output=True, env={**os.environ, "PYTHONPATH":""}, timeout=30)


def test_actual_v2_validator_accepts_new_relative_plan_and_preserves_old_logs(revision):
    result = validate(revision)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["log"].endswith("train.resume_001.log")
    assert (revision["root"] / "train.log").read_text() == "old training evidence\n"
    assert not (revision["root"] / "train.resume_001.log").exists()
    assert revision["receipt"]["old_contract_sha256"] != revision["receipt"]["new_contract_sha256"]


@pytest.mark.parametrize("change", ["old_sha", "new_sha", "relative_escape", "absolute_relative",
                                    "run_dir", "num_envs", "manifest_root", "unsupported_plan_field"])
def test_v2_revision_rejects_forged_fields_even_when_command_or_archive_is_resealed(revision, change):
    receipt = revision["receipt"]
    if change in ("old_sha", "new_sha"):
        receipt["old_contract_sha256" if change == "old_sha" else "new_contract_sha256"] = "0"*64
    elif change == "relative_escape":
        receipt["new_contract_relative_path"] = "../outside.json"
    elif change == "absolute_relative":
        receipt["new_contract_relative_path"] = str(revision["new"] / revision["new_plan_name"])
    elif change in ("run_dir", "num_envs"):
        flag = "--run-dir" if change == "run_dir" else "--num-envs"
        receipt["command"][receipt["command"].index(flag)+1] = str(revision["root"] / "another") if change == "run_dir" else "8192"
    else:
        path = revision["new"] / revision["manifest_name"]
        manifest = json.loads(path.read_text())
        if change == "manifest_root":
            manifest["run_directory"] = str(revision["root"] / "another")
        else:
            plan_path = revision["new"] / revision["new_plan_name"]
            plan = json.loads(plan_path.read_text())
            plan["unapproved_reward_patch"] = {"stationary_weight": 99}
            write(plan_path, plan)
            manifest["target_plan_sha256"] = sha(plan_path)
            receipt["new_contract_sha256"] = sha(plan_path)
        write(path, manifest)
        repack(revision)
    write(revision["receipt_path"], receipt)
    result = validate(revision)
    assert result.returncode == 3, result.stdout + result.stderr
    assert json.loads(result.stdout)["type"] == "ValueError"
    assert not (revision["root"] / "train.resume_001.log").exists()
