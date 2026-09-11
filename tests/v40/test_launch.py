"""CPU-only launch/cleanup adapters: no Isaac import, simulation or real tmux."""
from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import os
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = ("train_v40", "check_v40_env", "play_v40", "evaluate_v40")


def load_script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("launch_test_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("uid,headless", [(0, True), (1000, False)])
@pytest.mark.parametrize("inherited", ["1", "invalid"])
def test_launch_overrides_inherited_rendering_and_passes_root_kit_args(monkeypatch, uid, headless, inherited):
    cli = load_script("train_v40", monkeypatch)
    monkeypatch.setenv("ENABLE_CAMERAS", inherited)
    monkeypatch.setenv("LIVESTREAM", inherited)
    monkeypatch.setattr(cli.os, "geteuid", lambda: uid)
    before = dict(os.environ)
    events = []
    launcher = types.SimpleNamespace(app=object())
    args = types.SimpleNamespace(headless=headless, device="cuda:3")

    def app_launcher(config):
        events.append("launch")
        expected = {"headless": headless, "device": "cuda:3", "enable_cameras": False, "livestream": 0}
        if uid == 0:
            expected["kit_args"] = "--allow-root"
        assert config == expected
        assert dict(os.environ) == {**before, "ENABLE_CAMERAS": "0", "LIVESTREAM": "0"}
        return launcher

    original_import = builtins.__import__

    def stub_import(name, *args, **kwargs):
        if name == "isaaclab.app":
            events.append("import")
            assert os.environ["ENABLE_CAMERAS"] == os.environ["LIVESTREAM"] == "0"
            return types.SimpleNamespace(AppLauncher=app_launcher)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_import)
    budget = types.SimpleNamespace(check=lambda: events.append("check"))
    assert cli.launch_app(args, budget=budget) is launcher
    assert events == ["check", "import", "check", "launch"]
    events.clear()
    assert cli.launch_app(args) is launcher
    assert events == ["import", "launch"]


@pytest.mark.parametrize("stop_on_check", [1, 2])
def test_launch_deadline_checked_before_and_after_lazy_import(monkeypatch, stop_on_check):
    cli = load_script("train_v40", monkeypatch)
    from wheeled_algo.v40_job import PlannedStop
    monkeypatch.setenv("ENABLE_CAMERAS", "1")
    monkeypatch.setenv("LIVESTREAM", "2")
    events = []

    def check():
        events.append("check")
        if events.count("check") == stop_on_check:
            raise PlannedStop("stop_at")

    original_import = builtins.__import__

    def stub_import(name, *args, **kwargs):
        if name == "isaaclab.app":
            events.append("import")
            return types.SimpleNamespace(AppLauncher=lambda config: pytest.fail("expired budget must not launch"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_import)
    with pytest.raises(PlannedStop, match="stop_at"):
        cli.launch_app(types.SimpleNamespace(headless=True, device="cuda:0"), budget=types.SimpleNamespace(check=check))
    assert events == (["check"] if stop_on_check == 1 else ["check", "import", "check"])


@pytest.mark.parametrize("script", ENTRYPOINTS)
@pytest.mark.parametrize("mode", ["help", "preflight_ready", "preflight_blocked", "runtime_blocked"])
def test_cpu_paths_never_import_sim_or_change_environment(monkeypatch, script, mode, tmp_path):
    original_import = builtins.__import__

    def no_sim_import(name, *args, **kwargs):
        assert not name.startswith(("isaaclab", "isaacsim", "omni", "pxr", "rsl_rl")), name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sim_import)
    monkeypatch.setenv("ENABLE_CAMERAS", "1")
    monkeypatch.setenv("LIVESTREAM", "2")
    before = dict(os.environ)
    cli = load_script(script, monkeypatch)
    ready = mode == "preflight_ready"
    monkeypatch.setattr(cli, "preflight", lambda args: (
        {"blockers": [] if ready else ["offline blocker"], "ready": ready, "simulation_started": False}, None, None))
    if hasattr(cli, "make_manifest"):
        monkeypatch.setattr(cli, "make_manifest", lambda *args: {})
    if hasattr(cli, "checked_checkpoint"):
        monkeypatch.setattr(cli, "checked_checkpoint", lambda *args: (object(), {"run_manifest": {"stage": "stand"}}))
    monkeypatch.setattr(cli, "launch_app", lambda *args, **kwargs: pytest.fail("CPU path must not launch"))
    if mode == "help":
        with pytest.raises(SystemExit) as exc:
            cli.main(["--help"])
        assert exc.value.code == 0
    else:
        arguments = ["--preflight-only"] if mode.startswith("preflight") else []
        arguments += ["--research"]
        if script in ("check_v40_env", "evaluate_v40"):
            arguments += ["--apply"]
        if script in ("play_v40", "evaluate_v40"):
            arguments += ["--checkpoint", str(tmp_path / "not_loaded.pt")]
        if script == "evaluate_v40":
            arguments += ["--output-dir", str(tmp_path / "not_created")]
        assert cli.main(arguments) == (0 if ready else 2)
    assert dict(os.environ) == before
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("package", ["onnx", "onnxruntime"])
@pytest.mark.parametrize("installed", [None, "1.20.0"])
def test_exporter_metadata_rejects_before_sim(monkeypatch, tmp_path, capsys, package, installed):
    original_import = builtins.__import__

    def metadata_only_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "isaaclab", "isaacsim", "omni", "pxr", "rsl_rl", "torch", "onnx", "onnxruntime"
        }, name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", metadata_only_import)
    cli = load_script("train_v40", monkeypatch)
    assert cli.TARGET_VERSIONS[package] == "1.20.1"
    # Isolate distribution metadata; keep the real approved asset/baseline checks.
    monkeypatch.setattr(cli, "sys", types.SimpleNamespace(version_info=(3, 11, 0)))
    monkeypatch.setattr(cli, "check_isaaclab_source", lambda: {"tag": cli.LAB_TAG, "commit": cli.LAB_COMMIT})
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: cli.TARGET_VERSIONS[name])
    args = types.SimpleNamespace(research=True, stage="stand", num_envs=1, contract=None)
    report, _, _ = cli.preflight(args)
    assert report["ready"] is True and report["blockers"] == []

    def version(name):
        if name == package:
            if installed is None:
                raise cli.importlib.metadata.PackageNotFoundError(name)
            return installed
        return cli.TARGET_VERSIONS[name]

    monkeypatch.setattr(cli.importlib.metadata, "version", version)
    before = dict(os.environ)
    assert cli.main(["--research", "--headless", "--run-dir", str(tmp_path / "not_created")]) == 2
    report = json.loads(capsys.readouterr().out)
    expected = (f"missing target runtime distribution: {package}==1.20.1" if installed is None
                else f"{package}: expected 1.20.1, found {installed}")
    assert report["blockers"] == [expected]
    assert report["versions"][package] == installed
    assert report["ready"] is False and report["simulation_started"] is False
    assert dict(os.environ) == before and not list(tmp_path.iterdir())


@pytest.mark.parametrize("script", [script for script in ENTRYPOINTS if script != "evaluate_v40"])
def test_app_close_survives_env_close_failure(script):
    # Execute only the actual cleanup adapter, not a mock physics environment.
    # Evaluation's publication/cleanup ordering is exercised through main in test_metrics.
    tree = ast.parse((ROOT / "scripts" / (script + ".py")).read_text())
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "launch_app"
               for node in ast.walk(tree))
    cleanup = next(node.finalbody[0] for node in ast.walk(tree) if isinstance(node, ast.Try)
                   and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                           and isinstance(call.func.value, ast.Name) and call.func.value.id == "simulation_app"
                           and call.func.attr == "close"
                           for statement in node.finalbody for call in ast.walk(statement)))
    closed = []

    def close_env():
        closed.append("env")
        raise OSError("offline close failure")

    env = types.SimpleNamespace(close=close_env)
    namespace = {"env": env, "raw_env": env, "wrapped_env": None, "runtime_error": None,
                 "simulation_app": types.SimpleNamespace(close=lambda: closed.append("app"))}
    code = compile(ast.Module(body=[cleanup], type_ignores=[]), "offline_cleanup_adapter", "exec")
    with pytest.raises(OSError, match="offline close failure"):
        exec(code, namespace)
    assert closed == ["env", "app"]
