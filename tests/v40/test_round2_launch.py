"""Local CPU tests; all tmux/Isaac children are mocked, exports use synthetic weights."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from wheeled_algo import v40_job as job
from wheeled_algo.v40_export import export_checkpoint
from test_export import make_checkpoint, make_manifest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("round2_launcher", ROOT / "scripts/start_v40_round2.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def options(tmp_path, **overrides):
    return SimpleNamespace(**({"run_root": str(tmp_path / "new round2"), "python": "/opt/isaac env/bin/python",
                              "repo": str(ROOT), "tmux": "/usr/bin/tmux", "stages": ["stand", "locomotion"],
                              "num_envs": 1024, "total_iterations": 20000, "pilot_iterations": 20,
                              "training_runtime_seconds": 86400, "seed_base": 41} | overrides))


def argument(command, flag):
    return command[command.index(flag) + 1]


def prepare_worker(tmp_path, stage="stand"):
    worker = cli.build_plan(options(tmp_path, stages=[stage]))["workers"][0]
    audit = Path(worker["stage_root"]) / "audit"
    audit.mkdir(parents=True)
    path = audit / "plan.json"
    path.write_bytes(job.json_bytes(worker))
    return worker, audit, path


@pytest.fixture(scope="module")
def synthetic_export(tmp_path_factory):
    """A real CPU ONNX checker/ORT export, not a claim that any PPO update ran."""
    root = tmp_path_factory.mktemp("round2-export")
    identity = cli.build_plan(options(root))["workers"][0]["identity"]
    manifest = make_manifest() | identity
    (root / "run_manifest.json").write_bytes(job.json_bytes(manifest))
    checkpoint, _ = make_checkpoint(manifest)
    torch.save(checkpoint, root / "model_final.pt")
    export_checkpoint(root / "model_final.pt", root / "run_manifest.json", root / "policy.onnx")
    for name in job.BASE_ARTIFACTS - {"run_manifest.json"}:
        (root / name).write_bytes(job.json_bytes({"synthetic_fixture": name}))
    return root


def populate_run(worker, phase, synthetic_export):
    run = Path(worker["stage_root"]) / phase["name"]
    run.mkdir()
    for name in job.ALLOWED_ARTIFACTS:
        shutil.copyfile(synthetic_export / name, run / name)
    count = phase["requested_iterations"]
    receipt = {"schema_version": 1, "complete": True, "status": "completed", "completed_updates": count,
               "requested_iterations": count, "requested_iterations_completed": True,
               "stop_reason": "iterations_completed", "policy_quality_verified": False,
               "export_status": "verified", "error": None, "update_failed": False,
               "finished_at": datetime.now(timezone.utc).isoformat(), "max_runtime_seconds": phase["runtime_seconds"],
               "stop_at": None, "learning_elapsed_seconds": 1.0,
               "artifacts": [job.artifact_record(run, name) for name in sorted(job.ALLOWED_ARTIFACTS)]}
    (run / "completion.json").write_bytes(job.json_bytes(job.validate_completion(receipt)))
    return run, receipt


def test_default_plan_no_side_effects_and_v2_cli(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run started a child or created a directory")
    monkeypatch.setattr(cli.subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    args = options(tmp_path)
    assert cli.main(["--run-root", args.run_root, "--python", args.python]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert not Path(args.run_root).exists()
    assert len(plan["workers"]) == 2
    for worker, seed in zip(plan["workers"], (41, 42), strict=True):
        preflight, pilot, train = worker["phases"]
        assert worker["identity"]["seed"] == seed
        assert [pilot["requested_iterations"], train["requested_iterations"]] == [20, 19980]
        assert train["runtime_seconds"] == 86400
        assert train["timeout_seconds"] == 86400 + cli.INITIALIZATION_SECONDS + cli.EXPORT_SECONDS
        assert "--resume" not in pilot["command"] and "--finetune" not in pilot["command"]
        assert argument(train["command"], "--resume") == str(Path(worker["stage_root"]) / "pilot/model_final.pt")
        for phase in (preflight, pilot, train):
            command = phase["command"]
            assert argument(command, "--contract") == str(ROOT / "contracts/own_v40_v2.json")
            assert argument(command, "--seed") == str(seed)
            assert argument(command, "--num-envs") == "1024"
            assert "--stop-at" not in command  # Deferred until actual phase start.
    caches = [argument(w["phases"][1]["command"], "--usd-cache-dir") for w in plan["workers"]]
    assert len(set(caches)) == 2
    for worker, cache in zip(plan["workers"], caches, strict=True):
        assert cache == str(Path(worker["stage_root"]) / "usd_cache")
        assert argument(worker["phases"][2]["command"], "--usd-cache-dir") == cache
    solo = cli.build_plan(options(tmp_path, stages=["locomotion"]))["workers"][0]
    assert solo["identity"]["seed"] == 42


@pytest.mark.parametrize("override", [{"run_root": "relative"}, {"python": "python"},
    {"python": "/opt/python"}, {"tmux": "tmux"}, {"num_envs": 0}, {"pilot_iterations": 0},
    {"total_iterations": 20}, {"seed_base": -1}, {"stages": ["stand", "stand"]}])
def test_invalid_plan_is_rejected(tmp_path, override):
    with pytest.raises(job.JobError):
        cli.build_plan(options(tmp_path, **override))


def test_cli_cannot_select_v1_or_external_resume(tmp_path):
    for flags in (["--contract", str(ROOT / "contracts/own_v40_v1.json")], ["--resume", "/old/model.pt"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(["--run-root", str(tmp_path / "run"), "--python", "/env/bin/python", *flags])
        assert exc.value.code == 2
    assert not (tmp_path / "run").exists()


def test_real_stdlib_only_dry_run(tmp_path):
    result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "scripts/start_v40_round2.py"),
                             "--python", "/nonexistent/env/bin/python", "--run-root", str(tmp_path / "run")],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["dry_run"] is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["preflight", "pilot_exit", "pilot_timeout", "missing_receipt", "partial", "hash", "v1",
                                     "other_stage", "other_seed", "bad_export", "duplicate_json"])
def test_failed_pilot_blocks_main(tmp_path, monkeypatch, synthetic_export, failure):
    worker, audit, path = prepare_worker(tmp_path)
    calls = []

    def child(_worker, name, command, timeout):
        calls.append(name)
        if name == "preflight":
            (audit / "preflight.log").write_bytes(job.json_bytes({
                **worker["identity"], "ready": failure != "preflight", "simulation_started": False}))
            return
        assert name == "pilot", "failed pilot must never start main training"
        if failure == "pilot_exit":
            raise job.JobError("pilot exited 3")
        if failure == "pilot_timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        run, receipt = populate_run(worker, worker["phases"][1], synthetic_export)
        if failure == "missing_receipt":
            (run / "completion.json").unlink()
            return
        if failure == "partial":
            receipt.update(status="stopped", completed_updates=19, requested_iterations_completed=False,
                           stop_reason="max_runtime_seconds")
        elif failure in {"v1", "other_stage", "other_seed"}:
            manifest = job.strict_json((run / "run_manifest.json").read_bytes())
            key, value = {"v1": ("contract_id", "own-v40-jointspace-h5-v1"),
                          "other_stage": ("stage", "locomotion"), "other_seed": ("seed", 42)}[failure]
            manifest[key] = value
            (run / "run_manifest.json").write_bytes(job.json_bytes(manifest))
            receipt["artifacts"] = [job.artifact_record(run, item["path"]) for item in receipt["artifacts"]]
        elif failure == "bad_export":
            sidecar = job.strict_json((run / "policy.onnx.json").read_bytes())
            sidecar["validation"]["passed"] = False
            (run / "policy.onnx.json").write_bytes(job.json_bytes(sidecar))
            receipt["artifacts"] = [job.artifact_record(run, item["path"]) for item in receipt["artifacts"]]
        elif failure == "hash":
            (run / "model_final.pt").write_bytes(b"changed")
        raw = job.json_bytes(receipt) if failure != "duplicate_json" else b'{"status":1,"status":2}'
        (run / "completion.json").write_bytes(raw)

    monkeypatch.setattr(cli, "run_child", child)
    assert cli.run_worker(path) == 2
    assert calls == (["preflight"] if failure == "preflight" else ["preflight", "pilot"])
    assert job.strict_json((audit / "worker.status.json").read_bytes())["status"] == "failed"


@pytest.mark.parametrize("main_stopped", [False, True])
def test_verified_pilot_resumes_own_checkpoint_with_fresh_cutoff(tmp_path, monkeypatch, synthetic_export, main_stopped):
    worker, audit, path = prepare_worker(tmp_path)
    calls = []
    now = datetime(2040, 1, 1, tzinfo=timezone.utc)

    class Clock:
        @staticmethod
        def now(_timezone):
            return now

    def child(_worker, name, command, timeout):
        nonlocal now
        calls.append(command)
        if name == "preflight":
            (audit / "preflight.log").write_bytes(job.json_bytes({
                **worker["identity"], "ready": True, "simulation_started": False}))
        else:
            phase = next(p for p in worker["phases"] if p["name"] == name)
            assert datetime.fromisoformat(argument(command, "--stop-at")) == now + timedelta(
                seconds=cli.INITIALIZATION_SECONDS + phase["runtime_seconds"])
            assert timeout == cli.INITIALIZATION_SECONDS + phase["runtime_seconds"] + cli.EXPORT_SECONDS
            run, receipt = populate_run(worker, phase, synthetic_export)
            if name == "train" and main_stopped:
                receipt.update(status="stopped", completed_updates=100, requested_iterations_completed=False,
                               stop_reason="max_runtime_seconds")
                (run / "completion.json").write_bytes(job.json_bytes(job.validate_completion(receipt)))
        now += timedelta(hours=3)

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(cli, "run_child", child)
    assert cli.run_worker(path) == 0
    assert len(calls) == 3
    assert "--resume" not in calls[1]
    assert argument(calls[2], "--resume") == str(Path(worker["stage_root"]) / "pilot/model_final.pt")
    assert argument(calls[2], "--max-iterations") == "19980"
    status = job.strict_json((audit / "worker.status.json").read_bytes())
    assert status["status"] == ("stopped" if main_stopped else "completed")
    assert sum(status["checks"][p]["completed_updates"] for p in ("pilot", "train")) == (120 if main_stopped else 20000)
    with pytest.raises(FileExistsError):
        cli.run_worker(path)
    assert len(calls) == 3


@pytest.mark.parametrize("eula", [None, "YES"])
def test_actual_tmux_argv_environment_and_audit(tmp_path, monkeypatch, eula):
    plan = cli.build_plan(options(tmp_path))
    calls = []
    monkeypatch.setenv("PYTHONHOME", "/stale/home")
    monkeypatch.setenv("PYTHONPATH", "/stale/modules")
    monkeypatch.setenv("PATH", "/stale/bin")
    monkeypatch.setenv("ENABLE_CAMERAS", "1")
    monkeypatch.setenv("LIVESTREAM", "2")
    if eula is None:
        monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    else:
        monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", eula)

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["start_new_session"] is True and "shell" not in kwargs
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["stdout"].name.endswith("tmux.log")
        env = kwargs["env"]
        assert "PYTHONHOME" not in env and "PYTHONPATH" not in env
        assert "/stale/" not in env["PATH"]
        assert env["ENABLE_CAMERAS"] == env["LIVESTREAM"] == "0"
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env.get("OMNI_KIT_ACCEPT_EULA") == eula
        return SimpleNamespace(pid=7000 + len(calls), wait=lambda timeout: 0, returncode=0)

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.launch(plan) == 0
    assert len(calls) == 2
    for worker, (command, _) in zip(plan["workers"], calls, strict=True):
        assert command[:3] == ["/usr/bin/tmux", "new-session", "-d"]
        assert argument(command, "-s") == worker["session"]
        assert f"OMNI_KIT_ACCEPT_EULA={eula or ''}" in command
        assert "PYTHONHOME=" in command and "PYTHONPATH=" in command
        assert command[-4:] == [worker["python"], str(ROOT / "scripts/start_v40_round2.py"),
                                "--worker", str(Path(worker["stage_root"]) / "audit/plan.json")]
        audit = Path(worker["stage_root"]) / "audit"
        status = job.strict_json((audit / "tmux.status.json").read_bytes())
        assert status["pid"] > 7000 and status["started_at"] and status["exit_code"] == 0
        assert (audit / "tmux.started.json").exists()
        assert not (Path(worker["stage_root"]) / "pilot").exists()
    with pytest.raises(FileExistsError):
        cli.launch(plan)
    with pytest.raises(job.JobError, match="overwrite"):
        cli.build_plan(options(tmp_path))
    assert len(calls) == 2


def test_one_submission_failure_does_not_block_other_worker(tmp_path, monkeypatch):
    plan = cli.build_plan(options(tmp_path))
    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise OSError("synthetic tmux spawn failure")
        return SimpleNamespace(pid=7001, wait=lambda timeout: 0, returncode=0)

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.launch(plan) == 2
    assert len(calls) == 2
    status = job.strict_json((Path(plan["workers"][0]["stage_root"]) / "audit/tmux.status.json").read_bytes())
    assert status["pid"] is None and "synthetic" in status["error"]


@pytest.mark.parametrize("needs_kill", [False, True])
def test_timeout_signals_only_owned_group(tmp_path, monkeypatch, needs_kill):
    worker, audit, _ = prepare_worker(tmp_path)
    waits, kills = [], []

    child = SimpleNamespace(pid=9876, returncode=None)

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1 or (needs_kill and len(waits) == 2):
            raise subprocess.TimeoutExpired(["fake-child"], timeout)
        child.returncode = -9 if needs_kill else -15
        return child.returncode

    def popen(command, **kwargs):
        assert kwargs["start_new_session"] is True
        child.wait = wait
        return child

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        cli.run_child(worker, "pilot", ["fake-child"], 17)
    assert waits == ([17, 20, 20] if needs_kill else [17, 20])
    assert kills == [(9876, signal.SIGTERM)] + ([(9876, signal.SIGKILL)] if needs_kill else [])
    status = job.strict_json((audit / "pilot.status.json").read_bytes())
    assert status["timed_out"] is True and status["pid"] == 9876


def test_cache_option_reaches_importer_and_preserves_legacy_default(tmp_path, monkeypatch):
    train_spec = importlib.util.spec_from_file_location("round2_train", ROOT / "scripts/train_v40.py")
    train = importlib.util.module_from_spec(train_spec)
    train_spec.loader.exec_module(train)
    parser = argparse.ArgumentParser()
    train.add_common_arguments(parser)
    cfg = SimpleNamespace(scene=SimpleNamespace(), sim=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "wheeled_tasks.direct.v40_serial.env_cfg", SimpleNamespace(V40EnvCfg=lambda: cfg))
    monkeypatch.setitem(sys.modules, "wheeled_tasks.direct.v40_serial.env", SimpleNamespace(V40Env=lambda cfg: cfg))
    assert train.make_env(parser.parse_args([])).usd_cache_dir is None
    cache = tmp_path / "private cache"
    assert train.make_env(parser.parse_args(["--usd-cache-dir", str(cache)])).usd_cache_dir == str(cache)
    tree = ast.parse((ROOT / "src/wheeled_world/assets/v40.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_v40_articulation")
    import math
    class Config(SimpleNamespace):
        InitialStateCfg = SimpleNamespace
    cfg_types = SimpleNamespace(UrdfFileCfg=SimpleNamespace, RigidBodyPropertiesCfg=SimpleNamespace,
                               ArticulationRootPropertiesCfg=SimpleNamespace)
    class Drive(SimpleNamespace):
        PDGainsCfg = SimpleNamespace
    ns = {"math": math, "Path": Path, "ArticulationCfg": Config, "IdealPDActuatorCfg": SimpleNamespace,
          "sim_utils": cfg_types, "UrdfConverterCfg": SimpleNamespace(JointDriveCfg=Drive)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "asset_fixture", "exec"), ns)
    kwargs = {"urdf_path": tmp_path / "asset/urdf/robot.urdf", "joint_names": list("abcdef"),
              "nominal_positions": [0] * 6, "effort_limits": [1] * 6, "armatures": [0] * 6,
              "nominal_base_height": .32, "asset_manifest_sha256": "a" * 64}
    assert ns["make_v40_articulation"](**kwargs).spawn.usd_dir == str(tmp_path / "logs/v40_usd_cache" / ("a" * 64))
    assert ns["make_v40_articulation"](**kwargs, usd_cache_dir=cache).spawn.usd_dir == str(cache)
    env_tree = ast.parse((ROOT / "src/wheeled_tasks/direct/v40_serial/env.py").read_text())
    call = next(n for n in ast.walk(env_tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "make_v40_articulation")
    assert any(k.arg == "usd_cache_dir" and ast.unparse(k.value) == "cfg.usd_cache_dir" for k in call.keywords)
    assert "scripts/start_v40_round2.py" in job.SOURCE_FILES
