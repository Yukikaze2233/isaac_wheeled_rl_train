"""Offline adapter/protocol tests only: no Isaac, PPO learning, real SSH or port3080.

The HTTP fixture uses a dynamically bound loopback port and executes only the
read-only probe locally against pytest tmp files. tmux subprocesses are mocked.
Synthetic torch weights exercise the existing strict exporter, NOT policy quality.
"""
from __future__ import annotations

from contextlib import contextmanager
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import builtins
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import types
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo import v40_job as job


def load_script(name):
    spec = importlib.util.spec_from_file_location("offline_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pull = load_script("pull_v40_artifacts")
train = load_script("train_v40")
launcher = load_script("start_v40_tmux")


class Clock:
    def __init__(self):
        self.seconds = 0.0
        self.base = datetime(2040, 1, 1, tzinfo=timezone.utc)
        self.sleeps = []

    def monotonic(self):
        return self.seconds

    def utcnow(self):
        return self.base + timedelta(seconds=self.seconds)

    def advance(self, seconds):
        self.seconds += seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)

    def budget(self, seconds=None, stop_at=None):
        return job.TrainingBudget(seconds, stop_at, monotonic=self.monotonic, utcnow=self.utcnow)


class FakeWrapper:
    """Only a structural adapter stand-in, not an Isaac environment mock."""
    def __init__(self, clock):
        self.clock, self.calls = clock, 0

    def step(self, action):
        self.calls += 1
        self.clock.advance(1)
        return action


class FakeAlgorithm:
    def __init__(self, clock):
        self.clock, self.calls, self.hook = clock, 0, None

    def update(self):
        self.calls += 1
        if self.hook:
            self.hook()
        self.clock.advance(2)
        return {"fake_only": True}


class FakeRunner:
    def __init__(self, clock):
        self.env, self.alg = FakeWrapper(clock), FakeAlgorithm(clock)
        self.current_learning_iteration = 0
        self.saves = []
        self.serializer = None

    def learn(self, num_learning_iterations, init_at_random_ep_len):
        assert init_at_random_ep_len is False
        for _ in range(num_learning_iterations):
            self.env.step("fake action")
            self.alg.update()
            self.current_learning_iteration += 1

    def save(self, path, infos=None):
        self.saves.append((path, infos))
        if self.serializer:
            self.serializer(path, infos)
        else:
            Path(path).write_bytes(job.json_bytes({"infos": infos, "fake_checkpoint": True}))


def dummy_export(checkpoint, manifest, output):
    """Protocol fixture, explicitly not an ONNX verification substitute."""
    assert checkpoint.is_file() and manifest.is_file()
    output.write_bytes(b"fake ONNX fixture only")
    Path(str(output) + ".json").write_bytes(b'{"offline_fixture":true}')


def base_files(path):
    path.mkdir()
    for name in job.BASE_ARTIFACTS:
        (path / name).write_bytes(job.json_bytes({"fixture": name}))
    return path


def stub_run(tmp_path, *, budget_seconds=None, requested=2, hook=None):
    clock = Clock()
    budget = clock.budget(budget_seconds)
    runner = FakeRunner(clock)
    runner.alg.hook = (lambda: hook(budget, runner)) if hook else None
    run = base_files(tmp_path / "run")
    train.bind_checkpoint_metadata(runner, {key: "bound-" + key for key in train.METADATA_KEYS})
    receipt = job.run_training_job(runner, run, budget, requested, exporter=dummy_export)
    return run, receipt, runner, budget, clock


@pytest.mark.parametrize("text", ["2040-01-01", "2040-01-01T01:00:00", "bad", "2040-01-01T01:00+25:00"])
def test_deadline_requires_explicit_timezone(text):
    with pytest.raises(ValueError):
        job.parse_stop_at(text)


def test_timezone_and_fake_clock_boundaries():
    assert job.parse_stop_at("2040-01-01T08:00:00+08:00") == job.parse_stop_at("2040-01-01T00:00:00Z")
    clock = Clock()
    budget = clock.budget(10)
    clock.advance(500)  # startup doesn't spend relative learning time
    budget.check()
    budget.begin_learning()
    clock.advance(9.999)
    budget.check()
    clock.advance(.001)
    with pytest.raises(job.PlannedStop, match="max_runtime_seconds"):
        budget.check()
    absolute = clock.budget(stop_at=clock.utcnow())
    with pytest.raises(job.PlannedStop, match="stop_at"):
        absolute.check()
    with pytest.raises(job.PlannedStop):
        absolute.begin_learning()
    assert absolute.started is None


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_bad_relative_budgets(value):
    with pytest.raises(ValueError):
        job.TrainingBudget(value)


def test_signal_marks_only_and_update_count_after_return():
    clock, runner = Clock(), FakeRunner(Clock())
    budget = clock.budget()
    wrapper = runner.env
    original_step, original_update = wrapper.step, runner.alg.update
    def during_update():
        assert budget.in_update and budget.completed_updates == 0
        budget.mark_signal(signal.SIGTERM, None)
        assert budget.completed_updates == 0  # handler did not save or throw
    runner.alg.hook = during_update
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    with budget.signal_handlers(), budget.bind(runner):
        budget.begin_learning()
        assert runner.env is wrapper
        assert runner.alg.update() == {"fake_only": True}
        assert budget.completed_updates == 1 and not budget.in_update
        with pytest.raises(job.PlannedStop, match="sigterm"):
            wrapper.step("never stepped")
        assert wrapper.calls == 0 and runner.saves == []
    assert wrapper.step == original_step and runner.alg.update == original_update
    assert {s: signal.getsignal(s) for s in before} == before


def test_budget_stop_saves_consistent_final_and_preserves_metadata(tmp_path):
    run, receipt, runner, budget, clock = stub_run(tmp_path, budget_seconds=3, requested=4)
    assert receipt["status"] == "stopped" and receipt["stop_reason"] == "max_runtime_seconds"
    assert receipt["completed_updates"] == 1
    assert receipt["requested_iterations_completed"] is False
    assert receipt["policy_quality_verified"] is False
    assert runner.env.calls == runner.alg.calls == 1
    saved = json.loads((run / "model_final.pt").read_bytes())["infos"]
    assert all(saved[key] == "bound-" + key for key in train.METADATA_KEYS)
    assert saved["completed_updates"] == 1 and saved["policy_quality_verified"] is False
    assert set(item["path"] for item in receipt["artifacts"]) == job.ALLOWED_ARTIFACTS
    assert job.validate_completion(json.loads((run / "completion.json").read_bytes())) == receipt
    assert not list(run.glob("*.partial*"))


@pytest.mark.parametrize("signum,reason", [(signal.SIGINT, "sigint"), (signal.SIGTERM, "sigterm")])
def test_signal_during_update_finishes_then_stops(tmp_path, signum, reason):
    run, receipt, runner, budget, _ = stub_run(tmp_path, requested=5,
        hook=lambda b, r: b.mark_signal(signum, None))
    assert receipt["stop_reason"] == reason and budget.completed_updates == 1
    assert receipt["status"] == "stopped" and (run / "model_final.pt").is_file()


@pytest.mark.parametrize("stop", ["deadline", "soft"])
def test_zero_updates_untrained_no_save_or_export(tmp_path, stop):
    clock = Clock()
    budget = clock.budget(stop_at=clock.utcnow() if stop == "deadline" else None)
    if stop == "soft":
        budget.request_stop()
    runner = FakeRunner(clock)
    run = base_files(tmp_path / "run")
    receipt = job.run_training_job(runner, run, budget, 3,
        exporter=lambda *args: pytest.fail("zero updates must not export"))
    assert receipt["status"] == "untrained" and receipt["completed_updates"] == 0
    assert receipt["requested_iterations_completed"] is False
    assert runner.saves == [] and runner.env.calls == 0
    assert not (run / "model_final.pt").exists() and not (run / "policy.onnx").exists()
    job.validate_completion(receipt)


def test_failed_update_retains_periodic_and_never_saves_half_final(tmp_path):
    clock = Clock()
    runner = FakeRunner(clock)
    budget = clock.budget()
    run = base_files(tmp_path / "run")
    periodic = run / "model_0.pt"
    periodic.write_bytes(b"old consistent checkpoint")
    def failure():
        if runner.alg.calls == 2:
            budget.mark_signal(signal.SIGTERM, None)
            raise RuntimeError("synthetic optimizer failure after partial mutation")
    runner.alg.hook = failure
    with pytest.raises(RuntimeError, match="partial mutation"):
        job.run_training_job(runner, run, budget, 4, exporter=dummy_export)
    receipt = json.loads((run / "completion.json").read_bytes())
    assert receipt["status"] == "training_failed" and receipt["stop_reason"] == "training_exception"
    assert receipt["update_failed"] is True and receipt["completed_updates"] == 1
    assert "partial mutation" in receipt["error"]
    assert not (run / "model_final.pt").exists() and not runner.saves
    assert periodic.read_bytes() == b"old consistent checkpoint"


def test_export_failure_retains_pt_and_receipt_is_honest(tmp_path):
    clock, runner = Clock(), FakeRunner(Clock())
    run = base_files(tmp_path / "run")
    def fail(*args):
        raise ValueError("ORT unavailable fixture")
    receipt = job.run_training_job(runner, run, clock.budget(), 1, exporter=fail)
    assert receipt["status"] == receipt["export_status"] == "export_failed"
    assert receipt["requested_iterations_completed"] is True
    assert (run / "model_final.pt").is_file() and "ORT unavailable" in receipt["error"]
    assert set(x["path"] for x in receipt["artifacts"]) == job.BASE_ARTIFACTS | {"model_final.pt"}


def test_export_time_not_charged_to_learning(tmp_path):
    clock = Clock()
    runner = FakeRunner(clock)
    run = base_files(tmp_path / "run")
    def slow_export(*args):
        clock.advance(900)
        dummy_export(*args)
    receipt = job.run_training_job(runner, run, clock.budget(10), 1, exporter=slow_export)
    assert receipt["learning_elapsed_seconds"] == 3 and receipt["status"] == "completed"


def test_raw_contract_asset_snapshots_and_real_strict_export(tmp_path, monkeypatch):
    import torch
    from wheeled_tasks.v40.contract import load_contract, audit_asset, make_run_manifest
    contract = load_contract()
    asset = audit_asset(contract)  # Read-only audit, NOT gate waiver and NOT training.
    manifest = make_run_manifest(contract, asset)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_bytes(job.json_bytes(manifest))
    (run / "agent_config.json").write_bytes(b'{"offline_synthetic_export_test":true}')
    job.snapshot_provenance(run, repo_root=ROOT, contract_path=None,
                            contract=contract, asset=asset, manifest=manifest)
    assert (run / "contract.json").read_bytes() == (ROOT / "contracts/own_v40_v1.json").read_bytes()
    assert (run / "asset_manifest.json").read_bytes() == Path(asset["manifest_path"]).read_bytes()
    sources = json.loads((run / "source_hashes.json").read_bytes())
    assert sources["full_asset_snapshot"] is False
    assert set(sources["source_files_sha256"]) == set(job.SOURCE_FILES)
    clock = Clock()
    runner = FakeRunner(clock)
    state = {"std": torch.ones(6)}
    generator = torch.Generator().manual_seed(40)
    for prefix, input_dim, output_dim in (("actor", 125, 6), ("critic", 29, 1)):
        dims = [input_dim, *contract["policy"][prefix + "_hidden_dims"], output_dim]
        for i, (left, right) in enumerate(zip(dims, dims[1:])):
            state[f"{prefix}.{2*i}.weight"] = .01 * torch.randn(right, left, generator=generator)
            state[f"{prefix}.{2*i}.bias"] = .01 * torch.randn(right, generator=generator)
    runner.serializer = lambda path, infos: torch.save({"model_state_dict": state,
        "optimizer_state_dict": {}, "iter": runner.current_learning_iteration, "infos": infos}, path)
    train.bind_checkpoint_metadata(runner, manifest)
    clean_env, clean_cwd = os.environ.copy(), os.getcwd()
    monkeypatch.setenv("PYTHONPATH", "/KIT/poisoned/python")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/KIT/poisoned/native")
    monkeypatch.setenv("PYTHONHOME", "/KIT/nonexistent/python")
    original_import = builtins.__import__
    def guard_native_import(name, *args, **kwargs):
        if name.split(".")[0] in {"onnx", "onnxruntime"}:
            pytest.fail("ONNX/ORT must load only in the CPU child, never in the parent")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard_native_import)
    receipt = job.run_training_job(runner, run, clock.budget(3), 5,
        export_environment=clean_env, export_cwd=clean_cwd)
    assert receipt["status"] == "stopped" and receipt["policy_quality_verified"] is False, receipt.get("error")
    assert (run / job.EXPORT_STDOUT_LOG).is_file() and (run / job.EXPORT_STDERR_LOG).is_file()
    sidecar = json.loads((run / "policy.onnx.json").read_bytes())
    assert sidecar["validation"]["passed"] is True and sidecar["validation"]["sample_count"] == 9
    assert sidecar["onnx_sha256"] == job.sha256_file(run / "policy.onnx")
    assert sidecar["checkpoint_sha256"] == job.sha256_file(run / "model_final.pt")
    assert sidecar["run_manifest_sha256"] == job.sha256_file(run / "run_manifest.json")
    for item in receipt["artifacts"]:
        assert item["sha256"] == job.sha256_file(run / item["path"])


def test_original_cad_gate_remains_closed_and_budget_cli_is_cpu_only():
    from wheeled_tasks.v40.contract import load_contract, validate_asset
    # The parent now wires a separately approved research_manifest by default.
    # Check the ORIGINAL CAD manifest explicitly, without changing any file/gate.
    original_contract = load_contract()
    original_contract["asset"]["manifest"] = "manifest.json"
    with pytest.raises(ValueError, match="collision gate BLOCKED"):
        validate_asset(original_contract, allow_research=True)
    # --preflight-only explicitly never calls learn or launches simulation.
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/train_v40.py"),
        "--preflight-only", "--research", "--max-runtime-seconds", "30", "--stop-at", "2000-01-01T00:00:00Z"],
        capture_output=True, text=True, timeout=30)
    report = json.loads(completed.stdout)
    assert completed.returncode == 2 and report["simulation_started"] is False
    assert any("cutoff" in x for x in report["blockers"])
    # Default research approval is not itself a raw-CAD material repair; expired
    # absolute time still rejects startup independently of that approved scope.
    assert train.runtime_version_matches("5.1.0.0", "5.1.0")
    assert not train.runtime_version_matches("2.7.0+cu130", "2.7.0+cu128")


@contextmanager
def fake_http(*, corrupt=None, pending=False, redirect=False):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == pull.EXEC_ROUTE
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert set(body) == {"alias", "command", "timeoutMs"}
            assert body["alias"] == "OFFLINE_FIXTURE_ONLY"
            calls.append(("exec", body))
            argv = shlex.split(body["command"])
            assert argv[:2] == ["python3", "-c"]
            assert argv[2].startswith("\nimport json, os, stat\n")
            if pending:
                stdout = '{"state":"pending"}'
            else:
                # Execute the read-only, confined probe against tmp fixtures, never SSH.
                result = subprocess.run([sys.executable, "-c", argv[2]], capture_output=True, text=True, timeout=10)
                assert result.returncode == 0, result.stderr
                stdout = result.stdout
            raw = job.json_bytes({"result": {"success": True, "exitCode": 0, "timedOut": False,
                                               "stdout": stdout, "stderr": ""}})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            query = urlsplit(self.path)
            assert query.path == pull.DOWNLOAD_ROUTE
            params = parse_qs(query.query)
            assert set(params) == {"alias", "remotePath"} and params["alias"] == ["OFFLINE_FIXTURE_ONLY"]
            path = Path(params["remotePath"][0])
            calls.append(("download", str(path)))
            if redirect:
                self.send_response(302)
                self.send_header("Location", "http://192.0.2.99/never-contact")
                self.end_headers()
                return
            raw = path.read_bytes()
            if path.name == corrupt:
                raw = bytes((raw[0] ^ 1,)) + raw[1:]
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    assert server.server_port != 3080
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = pull.ManagedSSH("OFFLINE_FIXTURE_ONLY", f"http://127.0.0.1:{server.server_port}")
        yield client, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_successful_small_file_transfer_via_exact_managed_routes(tmp_path):
    run, receipt, *_ = stub_run(tmp_path)
    destination = tmp_path / "received"
    with fake_http() as (client, calls):
        local = pull.pull_artifacts(client, str(run), destination)
    assert local["transfer_verified"] is True and local["policy_quality_verified"] is False
    assert len([x for x in calls if x[0] == "download"]) == len(receipt["artifacts"])
    for item in receipt["artifacts"]:
        assert (destination / item["path"]).read_bytes() == (run / item["path"]).read_bytes()
    assert (destination / "completion.json").read_bytes() == (run / "completion.json").read_bytes()
    assert json.loads((destination / "local_receipt.json").read_bytes()) == local
    assert not list(destination.glob("*.partial"))


@pytest.mark.parametrize("name", ["/etc/passwd", "../escape", "x/../model_final.pt", "x/model_final.pt", "C:\\model.pt", ".env", ".ssh/id_rsa", ".git/config", "model_0.pt"])
def test_malicious_completion_paths_rejected_without_download(tmp_path, name):
    run, receipt, *_ = stub_run(tmp_path)
    receipt["artifacts"][0]["path"] = name
    (run / "completion.json").write_bytes(job.json_bytes(receipt))
    with fake_http() as (client, calls), pytest.raises(job.JobError):
        pull.pull_artifacts(client, str(run), tmp_path / "received")
    assert not [x for x in calls if x[0] == "download"]
    assert not (tmp_path / "received").exists()


@pytest.mark.parametrize("change", ["false_quality", "zero_trained", "missing", "duplicate", "bad_sha", "bool_count", "failed_update"])
def test_bad_completion_schema_is_rejected(tmp_path, change):
    _, receipt, *_ = stub_run(tmp_path)
    if change == "false_quality":
        receipt["policy_quality_verified"] = True
    elif change == "zero_trained":
        receipt["completed_updates"] = 0
    elif change == "missing":
        receipt["artifacts"].pop()
    elif change == "duplicate":
        receipt["artifacts"][1] = receipt["artifacts"][0]
    elif change == "bad_sha":
        receipt["artifacts"][0]["sha256"] = "not a hash"
    elif change == "bool_count":
        receipt["completed_updates"] = True
    elif change == "failed_update":
        receipt["update_failed"] = True
    with pytest.raises(job.JobError):
        job.validate_completion(receipt)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}', b'{broken'])
def test_strict_completion_json(raw):
    with pytest.raises(job.JobError):
        job.strict_json(raw)


def test_hash_error_leaves_partial_no_local_success(tmp_path):
    run, *_ = stub_run(tmp_path)
    dest = tmp_path / "received"
    with fake_http(corrupt="model_final.pt") as (client, _), pytest.raises(job.JobError, match="SHA256"):
        pull.pull_artifacts(client, str(run), dest)
    assert (dest / "model_final.pt.partial").is_file()
    assert not (dest / "model_final.pt").exists()
    assert not (dest / "local_receipt.json").exists()
    assert (run / "model_final.pt").is_file()


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "parent_symlink"])
def test_destination_overwrite_and_symlink_guard_before_network(tmp_path, kind):
    dest = tmp_path / "received"
    if kind == "file":
        dest.write_bytes(b"keep me")
    elif kind == "directory":
        dest.mkdir()
        (dest / "unknown.txt").write_bytes(b"keep me")
    elif kind == "symlink":
        dest.symlink_to(tmp_path / "nonexistent")
    else:
        (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
        dest = tmp_path / "linked/received"
    client = types.SimpleNamespace(exec=lambda *a, **k: pytest.fail("no network before destination guard"))
    with pytest.raises((job.JobError, OSError)):
        pull.pull_artifacts(client, "/offline/run", dest)
    if kind == "file":
        assert dest.read_bytes() == b"keep me"
    if kind == "directory":
        assert (dest / "unknown.txt").read_bytes() == b"keep me"


@pytest.mark.parametrize("kind", ["member", "completion", "parent", "hardlink", "missing_member"])
def test_remote_symlinks_and_missing_completed_files_rejected(tmp_path, kind):
    run, *_ = stub_run(tmp_path)
    remote = run
    name = "completion.json" if kind == "completion" else "model_final.pt"
    if kind == "parent":
        remote = tmp_path / "linked_run"
        remote.symlink_to(run, target_is_directory=True)
    elif kind == "hardlink":
        os.link(run / name, tmp_path / "other_link")
    elif kind == "missing_member":
        (run / name).unlink()
    else:
        original = tmp_path / "outside"
        (run / name).rename(original)
        (run / name).symlink_to(original)
    with fake_http() as (client, calls), pytest.raises(job.JobError):
        pull.pull_artifacts(client, str(remote), tmp_path / "received")
    assert not [x for x in calls if x[0] == "download"]


@pytest.mark.parametrize("url", ["http://192.0.2.1:3080", "https://example.com", "http://user:pass@127.0.0.1", "http://127.0.0.1/a", "http://127.0.0.1?x=1", "http://127.0.0.1/#fragment", "file:///tmp/api"])
def test_only_credential_free_loopback_origins(url):
    with pytest.raises(job.JobError):
        pull.ManagedSSH("OFFLINE", url)


def test_http_redirect_never_leaves_loopback(tmp_path):
    run, *_ = stub_run(tmp_path)
    with fake_http(redirect=True) as (client, _), pytest.raises(job.JobError, match="redirect"):
        pull.pull_artifacts(client, str(run), tmp_path / "received")
    assert not (tmp_path / "received/local_receipt.json").exists()


def test_wait_is_bounded_minimum_30_seconds_and_no_fake_success(tmp_path):
    clock = Clock()
    with fake_http(pending=True) as (client, calls), pytest.raises(pull.NotReady, match="budget"):
        pull.pull_artifacts(client, "/offline/not-ready", tmp_path / "received", wait=True,
            poll_interval=30, max_wait_seconds=65, monotonic=clock.monotonic, sleep=clock.sleep)
    assert clock.sleeps == [30, 30] and len(calls) == 3
    assert not (tmp_path / "received").exists()
    with pytest.raises(job.JobError, match="at least 30"):
        pull.pull_artifacts(object(), "/offline/run", tmp_path / "received", poll_interval=29)


def test_platform_off_is_explicit_not_cloud_channel(tmp_path):
    def offline(*args):
        raise pull.Unavailable("platform off; wait for power-on; no cloud-file fallback")
    client = types.SimpleNamespace(exec=offline)
    with pytest.raises(pull.Unavailable, match="power-on"):
        pull.pull_artifacts(client, "/offline/run", tmp_path / "received")
    assert not (tmp_path / "received").exists()


def test_atomic_receipt_and_final_never_overwrite(tmp_path):
    path = tmp_path / "completion.json"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        job.atomic_bytes(path, b"new")
    assert path.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.partial"))


def test_tmux_plan_is_unique_absolute_and_does_not_touch_run(tmp_path):
    kwargs = dict(python="/opt/v40/bin/python", repo="/repo", run_dir="/runs/new", audit_dir="/audits/new",
                  stop_at="2099-01-01T00:00:00Z", research=True, max_runtime_seconds=27000)
    first, command = launcher.launch_plan(**kwargs)
    second, _ = launcher.launch_plan(**kwargs)
    assert first["session"] != second["session"] and first["session"].startswith("v40-")
    args = shlex.split(command)
    assert args[:2] == ["/opt/v40/bin/python", "-c"]
    assert "kill-server" not in args[2] and "new-session" in args[2]
    assert not list(tmp_path.iterdir())
    with pytest.raises(job.JobError):
        launcher.launch_plan(**{**kwargs, "python": "python"})
    with pytest.raises(job.JobError):
        launcher.launch_plan(**{**kwargs, "audit_dir": "/runs/new/audit"})


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("eula", ["", "YES"])
def test_remote_tmux_preflight_gate_and_audit_are_mocked(tmp_path, monkeypatch, ready, eula):
    repo = tmp_path / "repo 'quoted' $literal;"
    repo.mkdir()
    interpreter = tmp_path / "env 'quoted' $literal;/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"fake nonexecutable interpreter")
    run, audit = tmp_path / "run", tmp_path / "audit"
    params, command = launcher.launch_plan(python=str(interpreter), repo=str(repo), run_dir=str(run),
        audit_dir=str(audit), stop_at="2099-01-01T00:00:00Z", research=True)
    stale = {"PYTHONHOME": "/stale/home", "PYTHONPATH": "/stale/path", "PATH": "/stale/bin",
             "ENABLE_CAMERAS": "1", "LIVESTREAM": "2", "PYTHONUNBUFFERED": "0",
             "OMNI_KIT_ACCEPT_EULA": eula}
    for key, value in stale.items():
        monkeypatch.setenv(key, value)
    calls = []
    def fake_subprocess(argv, **kwargs):
        calls.append((argv, kwargs))
        assert Path(kwargs["cwd"]) == repo
        assert kwargs["env"]["PATH"].startswith(str(interpreter.parent) + ":")
        assert "PYTHONHOME" not in kwargs["env"] and "PYTHONPATH" not in kwargs["env"]
        assert {key: kwargs["env"][key] for key in ("ENABLE_CAMERAS", "LIVESTREAM", "PYTHONUNBUFFERED")} == {
            "ENABLE_CAMERAS": "0", "LIVESTREAM": "0", "PYTHONUNBUFFERED": "1"}
        assert kwargs.get("shell", False) is False
        if len(calls) == 1:
            assert argv[0] == str(interpreter) and argv[-1] == "--preflight-only"
            assert argv[1] == str(repo / "scripts/train_v40.py")
            return types.SimpleNamespace(returncode=0 if ready else 2,
                stdout=json.dumps({"ready": ready, "simulation_started": False}), stderr="")
        if len(calls) == 3:
            assert argv == calls[0][0][:-1]  # Same explicit interpreter and args, without preflight.
            assert kwargs["stderr"] == subprocess.STDOUT
            kwargs["stdout"].write("offline child log")
            return types.SimpleNamespace(returncode=7)
        assert argv[:2] == ["/usr/bin/tmux", "new-session"]
        assert argv[4:6] == [params["session"], "-c"]
        assert argv[6] == str(repo)
        assert argv[-3:-1] == [str(interpreter), "-c"]  # Multi argv, NOT one shell string.
        options = argv[7:-3]
        assert options[::2] == ["-e"] * len(stale)
        session_env = dict(value.split("=", 1) for value in options[1::2])
        assert session_env == {key: kwargs["env"].get(key, "") for key in stale}
        return types.SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(subprocess, "run", fake_subprocess)
    # Execute orchestration with ALL subprocesses mocked: no real preflight/tmux/training.
    exec(compile(shlex.split(command)[2], "offline_tmux_fixture", "exec"), {})
    assert len(calls) == (2 if ready else 1)
    assert not run.exists()
    assert (audit / "preflight.stdout.log").is_file()
    result = json.loads((audit / "launch.json").read_bytes())
    assert result["launched"] is ready
    assert {key: os.environ[key] for key in stale} == stale  # Only child environments changed.
    if ready:
        assert result["training_success"] is False
        # Execute the serialized worker under stale server env; child remains fully mocked.
        exec(compile(calls[1][0][-1], "offline_detached_worker_fixture", "exec"), {})
        assert len(calls) == 3 and not run.exists()
        status = json.loads((audit / "exit.json").read_bytes())
        assert status["exit_code"] == 7 and status["training_success"] is False
        assert (audit / "train.log").read_text() == "offline child log"


def test_save_failure_is_not_a_completed_model(tmp_path):
    clock = Clock()
    runner = FakeRunner(clock)
    run = base_files(tmp_path / "run")
    def interrupted_save(path, infos):
        Path(path).write_bytes(b"partial serialization")
        raise OSError("fixture disk failure")
    runner.serializer = interrupted_save
    receipt = job.run_training_job(runner, run, clock.budget(), 1, exporter=dummy_export)
    assert receipt["status"] == "save_failed" and receipt["completed_updates"] == 1
    assert not (run / "model_final.pt").exists() and not (run / "policy.onnx").exists()
    assert "disk failure" in receipt["error"]
    job.validate_completion(receipt)


def test_post_update_exception_still_reports_exact_count_without_final(tmp_path):
    clock = Clock()
    runner = FakeRunner(clock)
    original_learn = runner.learn
    def failed_logging(**kwargs):
        original_learn(**kwargs)
        raise RuntimeError("fixture logging failure after all updates")
    runner.learn = failed_logging
    run = base_files(tmp_path / "run")
    with pytest.raises(RuntimeError, match="logging failure"):
        job.run_training_job(runner, run, clock.budget(), 1, exporter=dummy_export)
    receipt = json.loads((run / "completion.json").read_bytes())
    assert receipt["status"] == "training_failed" and receipt["completed_updates"] == 1
    assert receipt["requested_iterations_completed"] is True
    assert not (run / "model_final.pt").exists()


@pytest.mark.parametrize("field,value", [("finished_at", "2040-01-01"), ("stop_at", "not a date"),
    ("max_runtime_seconds", "30"), ("max_runtime_seconds", True), ("learning_elapsed_seconds", -1),
    ("unknown_field", "not schema v1")])
def test_completion_time_and_field_schema(tmp_path, field, value):
    _, receipt, *_ = stub_run(tmp_path)
    receipt[field] = value
    with pytest.raises(job.JobError):
        job.validate_completion(receipt)


def test_slow_ready_probe_cannot_exceed_wait_budget(tmp_path, monkeypatch):
    run, receipt, *_ = stub_run(tmp_path)
    clock = Clock()
    def slow_ready(*args):
        clock.advance(70)
        return (run / "completion.json").read_bytes(), receipt, {}
    monkeypatch.setattr(pull, "probe", slow_ready)
    client = types.SimpleNamespace(timeout=90)
    with pytest.raises(pull.NotReady, match="budget"):
        pull.pull_artifacts(client, str(run), tmp_path / "received", wait=True,
            max_wait_seconds=60, monotonic=clock.monotonic, sleep=clock.sleep)
    assert client.timeout == 90
    assert not (tmp_path / "received").exists()


def test_export_failed_pt_only_transfer_does_not_claim_onnx(tmp_path):
    clock = Clock()
    runner = FakeRunner(clock)
    run = base_files(tmp_path / "run")
    def fail_export(*args):
        raise ValueError("fixture export failure")
    job.run_training_job(runner, run, clock.budget(), 1, exporter=fail_export)
    with fake_http() as (client, _):
        receipt = pull.pull_artifacts(client, str(run), tmp_path / "received")
    assert receipt["transfer_verified"] is True and receipt["export_status"] == "export_failed"
    assert receipt["policy_quality_verified"] is False
    assert not (tmp_path / "received/policy.onnx").exists()


def fake_child_sidecar(run, *, mutation=None):
    """Stdlib-only output fixture, NOT an ONNX model or a numerical validation."""
    manifest = json.loads((run / "run_manifest.json").read_bytes())
    (run / "policy.onnx").write_bytes(b"fixture child output, not real ONNX")
    sidecar = {
        "schema_version": 1, "run_manifest": manifest, "policy": manifest["policy"],
        "onnx_filename": "policy.onnx", "opset": 17, "dynamo": False,
        "actor_obs_dim": 125, "critic_obs_dim": 29, "action_dim": 6,
        "input": {"name": "obs_history", "dtype": "float32", "shape": [1, 125]},
        "output": {"name": "actions", "dtype": "float32", "shape": [1, 6]},
        **{key: manifest[key] for key in train.METADATA_KEYS},
        "checkpoint_sha256": job.sha256_file(run / "model_final.pt"),
        "run_manifest_sha256": job.sha256_file(run / "run_manifest.json"),
        "onnx_sha256": job.sha256_file(run / "policy.onnx"),
        "validation": {"passed": True, "provider": "CPUExecutionProvider", "sample_count": 9,
            "seed": 40, "atol": 1e-6, "rtol": 1e-5, "max_abs_error": 0.0, "max_relative_error": 0.0,
            "samples": [{"sample": i, "max_abs_error": 0.0, "max_relative_error": 0.0} for i in range(9)]},
    }
    if mutation:
        mutation(sidecar)
    (run / "policy.onnx.json").write_bytes(job.json_bytes(sidecar))


def child_run_files(tmp_path):
    run = base_files(tmp_path / "run")
    manifest = {"schema_version": 1, "contract_id": "own-v40-jointspace-h5-v1",
        "contract_sha256": "a" * 64, "asset_manifest_sha256": "b" * 64,
        "actor_obs_dim": 125, "critic_obs_dim": 29, "action_dim": 6,
        "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64], "activation": "elu", "empirical_normalization": False}}
    (run / "run_manifest.json").write_bytes(job.json_bytes(manifest))
    clock = Clock()
    runner = FakeRunner(clock)
    train.bind_checkpoint_metadata(runner, manifest)
    return run, runner, clock


def test_cpu_child_uses_clean_snapshot_absolute_venv_and_saved_metadata(tmp_path, monkeypatch):
    run, runner, clock = child_run_files(tmp_path)
    clean = {"PYTHONPATH": "src:../.rl_deps_rsl23", "LD_LIBRARY_PATH": "/startup/native",
             "PYTHONUSERBASE": "/startup/user-site", "CUDA_VISIBLE_DEVICES": "0", "PATH": "/startup/bin"}
    monkeypatch.setenv("PYTHONPATH", "/KIT/poison")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/KIT/poison")
    monkeypatch.setenv("KIT_ONLY", "must not leak")
    calls = []
    def child(command, **kwargs):
        calls.append((command, kwargs))
        assert command[:2] == [str(Path(sys.executable).absolute()), str(ROOT / "scripts/export_v40_onnx.py")]
        assert command[2:] == ["--checkpoint", str(run / "model_final.pt"),
            "--run-manifest", str(run / "run_manifest.json"), "--output", str(run / "policy.onnx")]
        assert kwargs["env"] == {**clean, "CUDA_VISIBLE_DEVICES": "-1"}
        assert kwargs["cwd"] == str(tmp_path) and kwargs["start_new_session"] is True
        assert kwargs["check"] is False and kwargs["timeout"] == 900
        saved = json.loads((run / "model_final.pt").read_bytes())["infos"]
        assert saved["completed_updates"] == 1 and saved["contract_sha256"] == "a" * 64
        assert not (run / "completion.json").exists()
        kwargs["stdout"].write(b"fixture child stdout")
        kwargs["stderr"].write(b"fixture child stderr")
        fake_child_sidecar(run)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(job.subprocess, "run", child)
    receipt = job.run_training_job(runner, run, clock.budget(), 1,
        export_environment=clean, export_cwd=str(tmp_path))
    assert receipt["status"] == "completed" and len(calls) == 1
    assert clean["CUDA_VISIBLE_DEVICES"] == "0"  # Caller snapshot wasn't changed.
    assert (run / job.EXPORT_STDERR_LOG).read_bytes() == b"fixture child stderr"
    assert not ({job.EXPORT_STDOUT_LOG, job.EXPORT_STDERR_LOG} & job.ALLOWED_ARTIFACTS)
    assert {entry["path"] for entry in receipt["artifacts"]} == job.ALLOWED_ARTIFACTS


@pytest.mark.parametrize("failure", ["nonzero", "native_exit_after_publish", "missing_pair", "missing_sidecar",
    "wrong_hash", "wrong_checkpoint", "wrong_manifest", "wrong_identity", "ort_failed", "gpu_provider",
    "bad_samples", "bad_json", "symlink", "timeout", "spawn_error"])
def test_child_failures_retain_pt_and_stderr_without_false_completion(tmp_path, monkeypatch, failure):
    run, runner, clock = child_run_files(tmp_path)
    def child(command, **kwargs):
        kwargs["stderr"].write(b"fixture native stderr\n")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 900)
        if failure == "spawn_error":
            raise OSError("fixture cannot start interpreter")
        if failure in {"nonzero", "missing_pair"}:
            return types.SimpleNamespace(returncode=1 if failure == "nonzero" else 0)
        changes = {
            "wrong_hash": lambda s: s.update(onnx_sha256="f" * 64),
            "wrong_checkpoint": lambda s: s.update(checkpoint_sha256="f" * 64),
            "wrong_manifest": lambda s: s.update(run_manifest_sha256="f" * 64),
            "wrong_identity": lambda s: s.update(contract_sha256="f" * 64),
            "ort_failed": lambda s: s["validation"].update(passed=False),
            "gpu_provider": lambda s: s["validation"].update(provider="CUDAExecutionProvider"),
            "bad_samples": lambda s: s["validation"].update(sample_count=1),
        }
        fake_child_sidecar(run, mutation=changes.get(failure))
        if failure == "missing_sidecar":
            (run / "policy.onnx.json").unlink()
        if failure == "bad_json":
            (run / "policy.onnx.json").write_bytes(b'{"broken":')
        if failure == "symlink":
            (run / "policy.onnx.json").rename(tmp_path / "outside.json")
            (run / "policy.onnx.json").symlink_to(tmp_path / "outside.json")
        # A child can publish BOTH valid files, then crash during interpreter teardown.
        return types.SimpleNamespace(returncode=-11 if failure == "native_exit_after_publish" else 0)
    monkeypatch.setattr(job.subprocess, "run", child)
    receipt = job.run_training_job(runner, run, clock.budget(), 1,
        export_environment=os.environ.copy(), export_cwd=str(tmp_path))
    assert receipt["status"] == receipt["export_status"] == "export_failed"
    assert receipt["completed_updates"] == 1 and receipt["policy_quality_verified"] is False
    assert (run / "model_final.pt").is_file()
    assert {entry["path"] for entry in receipt["artifacts"]} == job.BASE_ARTIFACTS | {"model_final.pt"}
    audit = (run / job.EXPORT_STDERR_LOG).read_text()
    assert "fixture native stderr" in audit and "[parent export audit]" in audit
    if failure == "native_exit_after_publish":
        assert "-11" in receipt["error"]
        assert (run / "policy.onnx").is_file() and (run / "policy.onnx.json").is_file()
    job.validate_completion(json.loads((run / "completion.json").read_bytes()))


@pytest.mark.parametrize("name", ["export.stdout.log", "export.stderr.log"])
def test_export_audit_files_never_overwrite_or_follow_symlink(tmp_path, monkeypatch, name):
    run, runner, clock = child_run_files(tmp_path)
    target = tmp_path / "untouched"
    target.write_bytes(b"unknown existing data")
    (run / name).symlink_to(target)
    monkeypatch.setattr(job.subprocess, "run", lambda *a, **kw: pytest.fail("must reject before spawning"))
    receipt = job.run_training_job(runner, run, clock.budget(), 1,
        export_environment=os.environ.copy(), export_cwd=str(tmp_path))
    assert receipt["status"] == "export_failed" and target.read_bytes() == b"unknown existing data"
    assert (run / "model_final.pt").is_file()


def test_clean_context_precedes_kit_and_completion_precedes_close():
    source = ast.parse((ROOT / "scripts/train_v40.py").read_text())
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignments = {target.id: node for node in ast.walk(main) if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    app_launch = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "launch_app")
    assert assignments["clean_export_environment"].lineno < app_launch.lineno
    assert assignments["clean_export_cwd"].lineno < app_launch.lineno
    assert {item.arg: item.value.id for item in app_launch.keywords} == {"budget": "budget"}
    call = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "run_training_job")
    keywords = {item.arg: item.value.id for item in call.keywords if isinstance(item.value, ast.Name)}
    assert keywords == {"export_environment": "clean_export_environment", "export_cwd": "clean_export_cwd"}
    close = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "close"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "simulation_app")
    assert call.lineno < close.lineno
    job_ast = ast.parse((ROOT / "src/wheeled_algo/v40_job.py").read_text())
    for node in ast.walk(job_ast):
        if isinstance(node, ast.Import):
            assert all(item.name.split(".")[0] not in {"torch", "onnx", "onnxruntime"} for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in {"wheeled_algo.v40_export", "torch", "onnx", "onnxruntime"}
    for tree in (source, job_ast):
        assert not any(isinstance(node, ast.Attribute) and node.attr == "_exit" for node in ast.walk(tree))


def test_production_default_requires_pre_kit_context_before_learning(tmp_path):
    run, runner, clock = child_run_files(tmp_path)
    with pytest.raises(job.JobError, match="clean startup"):
        job.run_training_job(runner, run, clock.budget(), 1)
    assert runner.env.calls == runner.alg.calls == 0 and not (run / "model_final.pt").exists()


def test_tmux_cli_defaults_to_zero_network_dry_plan(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "ManagedSSH", lambda *args, **kwargs: pytest.fail("dry-run must not connect"))
    assert launcher.main(["--alias", "OFFLINE_ONLY", "--python", "/opt/v40/bin/python",
        "--remote-repo", "/repo", "--remote-run-dir", "/runs/new", "--audit-dir", "/audits/new",
        "--stop-at", "2099-01-01T00:00:00Z"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["dry_run"] is True and receipt["launched"] is False
