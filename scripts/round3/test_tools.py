"""CPU-only deployment protocol tests; no Isaac, training, SSH server or Git writes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import deploy
import launch
import watch


def fixture_plan(tmp_path, monkeypatch, profile="smoke", **overrides):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    repo = common.code_directory(snapshot)
    (repo / "contracts").mkdir(parents=True)
    (repo / common.CONTRACT).write_text(json.dumps({"contract_id": "r3a-fixture"}))
    runtime = tmp_path / "runtime/bin/sim60-runtime.sh"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("# test runtime only\n")
    interpreter = runtime.parent.parent / "env/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("# not executed\n")
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(launch, "verify_snapshot", lambda _p: {"git_commit": "a" * 40})
    monkeypatch.setattr(launch, "load_contract", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(launch, "contract_digest", lambda _c: "b" * 64)
    args = SimpleNamespace(snapshot=str(snapshot), runtime=str(runtime), run_root=str(runs), profile=profile,
                           label="test", num_envs=None, updates=None, seed=43, smoke_status=None, capacity_status=None)
    vars(args).update(overrides)
    return args


def test_smoke_is_readonly_and_uses_approved_parent(tmp_path, monkeypatch):
    args = fixture_plan(tmp_path, monkeypatch)
    plan = launch.build_plan(args)
    assert (plan["num_envs"], plan["updates"]) == (2, 3)
    assert "--resume" not in plan["command"] and "--finetune" not in plan["command"]
    index = plan["command"].index("--warm-start")
    assert plan["command"][index + 1] == str(Path(args.snapshot) / "parent/model_final.pt")
    assert plan["parent_model_sha256"] == common.PARENT_SHA256
    assert not Path(plan["stage_root"]).exists()
    assert plan["purpose"] == "engineering_not_policy_quality"
    assert Path(plan["repo"]) == Path(args.snapshot).parent / "isaac_wheeled_rl_train"
    assert plan["command"][1] == str(Path(plan["repo"]) / "scripts/train_v40.py")
    assert not Path(plan["repo"]).is_symlink()


def test_formal_cannot_relabel_three_updates(tmp_path, monkeypatch):
    args = fixture_plan(tmp_path, monkeypatch, "train", updates=3)
    with pytest.raises(common.JobError, match="formal target"):
        launch.build_plan(args)


def test_formal_requires_current_smoke_and_matching_capacity(tmp_path, monkeypatch):
    args = fixture_plan(tmp_path, monkeypatch, "train")
    with pytest.raises(common.JobError, match="verified"):
        launch.build_plan(args)
    calls = []
    def gate(path, plan, profiles):
        calls.append(profiles)
        return {"num_envs": 1024}
    monkeypatch.setattr(launch, "checked_gate", gate)
    plan = launch.build_plan(args)
    assert calls == [("smoke",), ("capacity-1024",)]
    assert plan["learning_seconds"] == 48 * 3600
    assert plan["initialization_seconds"] == plan["finalization_seconds"] == 1800
    assert plan["training_hard_seconds"] == 49 * 3600
    assert plan["updates"] == 10000


def test_wrong_gate_identity_rejected(tmp_path):
    path = tmp_path / "worker.status.json"
    path.write_text(json.dumps({"profile": "smoke", "status": "completed", "git_commit": "old"}))
    with pytest.raises(common.JobError, match="another source"):
        launch.checked_gate(path, {"git_commit": "new"}, ("smoke",))


def test_lineage_requires_parent_and_fresh_optimizer():
    lineage = {"mode": "warm_start", "parent_checkpoint_sha256": common.PARENT_SHA256,
               "target_contract_sha256": "target", "optimizer_reset": True, "initial_iteration": 0}
    common.verify_warm_start({"source_provenance": lineage}, "target")
    for key, value in (("parent_checkpoint_sha256", "smoke"), ("optimizer_reset", False), ("initial_iteration", 3)):
        with pytest.raises(common.JobError, match="warm-start lineage"):
            common.verify_warm_start({"source_provenance": {**lineage, key: value}}, "target")


def test_transport_never_falls_back_to_new_auth(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=255, stderr=b"missing master")
    monkeypatch.setattr(common.subprocess, "run", run)
    client = common.ControlSSH("user@host", "/tmp/nonexistent-round3-control")
    with pytest.raises(common.Unavailable, match="no password fallback"):
        client.exec("true")
    assert len(calls) == 1 and "check" in calls[0]
    assert "BatchMode=yes" in calls[0] and "ProxyCommand=false" in calls[0]


def test_remote_unpack_rejects_escape_and_accepts_hashes(tmp_path):
    def execute(root, member_name, payload=b"data"):
        root.mkdir()
        manifest = {"git_commit": "a" * 40, "parent_model_sha256": common.PARENT_SHA256,
                    "code_directory_name": common.CODE_DIRECTORY,
                    "files": {member_name: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}}
        bundle = root / "bundle.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            for name, data in ((member_name, payload), ("snapshot.json", json.dumps(manifest).encode())):
                item = tarfile.TarInfo(name)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
        script = deploy.REMOTE_UNPACK.replace("ROOT", repr(str(root)), 1).replace(
            "DIGEST", repr(hashlib.sha256(bundle.read_bytes()).hexdigest()))
        return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    bad = execute(tmp_path / "bad", "../escape")
    assert bad.returncode != 0 and not (tmp_path / "escape").exists()
    good = execute(tmp_path / "good", "isaac_wheeled_rl_train/example.txt")
    assert good.returncode == 0, good.stderr
    code = tmp_path / "isaac_wheeled_rl_train"
    assert (code / "example.txt").read_bytes() == b"data"
    assert not code.is_symlink() and not (tmp_path / "good/source").exists()
    assert execute(tmp_path / "repeat", "isaac_wheeled_rl_train/example.txt", b"changed").returncode != 0
    assert (code / "example.txt").read_bytes() == b"data"


def test_watcher_resume_rejects_another_run(tmp_path):
    root = tmp_path / "watch"
    watch.ensure_root(root, {"run": "one"})
    watch.ensure_root(root, {"run": "one"})
    with pytest.raises(common.JobError, match="exact run"):
        watch.ensure_root(root, {"run": "two"})
    assert json.loads((root / "watch-owner.json").read_text()) == {"run": "one"}


def test_reused_child_runner_records_real_failure(tmp_path):
    (tmp_path / "audit").mkdir()
    worker = {"stage_root": str(tmp_path), "repo": str(tmp_path), "python": sys.executable,
              "identity": {"stage": "cpu-test"}}
    with pytest.raises(common.JobError, match="exited 7"):
        launch.run_child(worker, "fixture", [sys.executable, "-c", "raise SystemExit(7)"], 5)
    status = json.loads((tmp_path / "audit/fixture.status.json").read_text())
    assert status["exit_code"] == 7 and status["pid"] > 0 and not status["timed_out"]


def test_read_only_health_does_not_mistake_missing_tmux_for_success(tmp_path):
    class LocalReadOnlyClient:
        def exec(self, command):
            return subprocess.check_output(shlex.split(command), text=True, timeout=10)
    result = watch.read_worker(LocalReadOnlyClient(), {"audit_dir": str(tmp_path / "audit"),
        "run_dir": str(tmp_path / "train"), "session": "round3-test-no-such-session"})
    assert not result["completion_present"] and not result["training_pid_present"]
    assert result["worker_status"] is None and not result["tmux_session_present"]
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(sys.platform != "linux", reason="WSL/Linux subreaper lifecycle")
def test_worker_reaps_detached_exporter_descendant():
    code = '''
import subprocess,sys,time
sys.path.insert(0, MODULE)
import launch,psutil
launch.own_descendants()
try:
    parent=subprocess.Popen([sys.executable,'-c',
        "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'],start_new_session=True)"])
    parent.wait(timeout=3)
    assert psutil.Process().children(recursive=True), 'exporter was not adopted'
    assert launch.cleanup_descendants()
    assert not [p for p in psutil.Process().children(recursive=True) if p.is_running()]
finally:
    launch.cleanup_descendants()
'''.replace("MODULE", repr(str(Path(__file__).resolve().parent)))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
