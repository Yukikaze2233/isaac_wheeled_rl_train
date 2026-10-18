"""Real CPU worker lifecycle tests; no simulator or formal PPO allocation."""
import importlib.util
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time

import pytest

from wheeled_algo.training_control import TrainingControl, atomic_json, read_json
from wheeled_algo.v40_job import PlannedStop


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v7ctl", ROOT / "tools/v7ctl.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def wait_for(predicate, seconds=8):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        result = predicate()
        if result:
            return result
        time.sleep(.02)
    raise AssertionError("CPU fixture worker did not reach its expected boundary")


def fixture_worker(run):
    run = Path(run)
    control = TrainingControl(run)
    rng = random.Random(20261018)
    position, momentum = 0., 0.

    def save(update):
        path = run / f"saved_{update}.json"
        atomic_json(path, {"position": position, "momentum": momentum, "update": update,
                           "rng": repr(rng.getstate())})
        return path

    def stop():
        raise PlannedStop("soft_stop")

    try:
        for step in range(1, 81):
            momentum = .9 * momentum + rng.uniform(-1, 1)
            position += momentum
            control.boundary(step, checkpoint=save, flush=lambda: None,
                             check_limits=lambda: None, stop=stop)
            time.sleep(.01)
    except PlannedStop:
        atomic_json(run / "completion.json", {"status": "stopped"})
        control.finish("stopped")
    else:
        atomic_json(run / "result.json", {"position": position, "momentum": momentum,
                                          "rng": repr(rng.getstate())})
        control.finish("completed")


def fixture_controller(tmp_path, run):
    ledger = tmp_path / "ledger.json"
    atomic_json(ledger, {"checkpoints": {}, "attempts": {"fixture": {"identity": {"requested_updates": 80}}},
                         "reservations": [], "campaign_policy": {}})
    config = tmp_path / "config.json"
    atomic_json(config, {"schema": "v7ctl-v1", "run_directory": str(run),
                         "ledger": str(ledger), "attempt_id": "fixture", "blocked_reasons": []})
    return cli.Controller(config)


def test_resident_pause_retains_rng_momentum_and_physical_state(tmp_path):
    run, baseline = tmp_path / "paused", tmp_path / "baseline"
    controller = fixture_controller(tmp_path, run)
    ctx = mp.get_context("spawn")
    worker = ctx.Process(target=fixture_worker, args=(run,))
    worker.start()
    try:
        wait_for(lambda: read_json(run / "control/status.json", {}).get("learning_updates", 0))
        paused = controller.request("pause", timeout=4)
        assert paused["phase"] == "paused"
        checkpoint_bytes = Path(paused["checkpoint"]).read_bytes()
        time.sleep(.25)
        assert read_json(run / "control/status.json")["learning_updates"] == paused["learning_updates"]
        assert Path(paused["checkpoint"]).read_bytes() == checkpoint_bytes
        assert controller.request("pause", timeout=1) == paused
        resumed = controller.start(timeout=4)
        assert resumed["identity"] == paused["identity"] and resumed["phase"] == "running"
        assert controller.start()["message"].startswith("现有进程")
        worker.join(5)
        assert worker.exitcode == 0
        fixture_worker(baseline)
        assert read_json(run / "result.json") == read_json(baseline / "result.json")
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join()


def test_save_exit_from_resident_pause_finishes_without_an_extra_update(tmp_path):
    run = tmp_path / "worker"
    controller = fixture_controller(tmp_path, run)
    worker = mp.get_context("spawn").Process(target=fixture_worker, args=(run,))
    worker.start()
    try:
        wait_for(lambda: read_json(run / "control/status.json", {}).get("learning_updates", 0))
        paused = controller.request("pause", timeout=4)
        stopped = controller.request("stop", timeout=4)
        worker.join(3)
        assert worker.exitcode == 0 and stopped["learning_updates"] == paused["learning_updates"]
        assert read_json(run / "completion.json")["status"] == "stopped"
        assert not (run / "result.json").exists()
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join()


def test_old_boot_pid_cannot_receive_requests_and_expired_request_is_ignored(tmp_path):
    control = TrainingControl(tmp_path / "run")
    old = {**control.identity, "boot_id": "old-boot"}
    atomic_json(control.directory / "request.json", {"id": "stale", "identity": old, "action": "pause"})
    control.boundary(1, checkpoint=lambda _: pytest.fail("stale request saved"),
                     flush=lambda: None, check_limits=lambda: None, stop=lambda: None)
    control.publish(identity=old)
    controller = fixture_controller(tmp_path, tmp_path / "run")
    with pytest.raises(ValueError, match="存活"):
        controller.request("pause", timeout=.01)


def test_pending_request_timeout_never_signals_or_claims_pause(tmp_path):
    control = TrainingControl(tmp_path / "run")
    controller = fixture_controller(tmp_path, tmp_path / "run")
    with pytest.raises(TimeoutError):
        controller.request("pause", timeout=.05)
    assert control.state["phase"] == "starting"
    assert cli.identity_alive(control.identity)
    request = read_json(control.directory / "request.json")
    with pytest.raises(TimeoutError):
        controller.request("pause", timeout=.05)
    assert read_json(control.directory / "request.json") == request


def test_paused_worker_still_obeys_existing_limits(tmp_path):
    control = TrainingControl(tmp_path / "run")
    atomic_json(control.directory / "request.json", {"id": "pause", "identity": control.identity, "action": "pause"})

    def expired():
        raise PlannedStop("max_runtime_seconds")

    with pytest.raises(PlannedStop, match="max_runtime_seconds"):
        control.boundary(20, checkpoint=lambda _: "/saved", flush=lambda: None,
                         check_limits=expired, stop=lambda: None)


def test_corrupt_latest_checkpoint_fails_closed_not_rollback(tmp_path):
    controller = fixture_controller(tmp_path, tmp_path / "run")
    model = tmp_path / "model.pt"
    model.write_bytes(b"checkpoint")
    seal = model.with_suffix(".checkpoint.json")
    metadata = {"attempt_id": "fixture", "paid_updates_attempt": 2}
    atomic_json(seal, {"learning_updates": 102, "checkpoint_sha256": cli.digest(model), "training_budget": metadata})
    ledger = controller.ledger()
    ledger["checkpoints"]["saved"] = {"status": "registered", "path": str(model), "metadata": metadata,
        "checkpoint_sha256": cli.digest(model), "seal_sha256": cli.digest(seal)}
    atomic_json(controller.config["ledger"], ledger)
    assert controller.latest_checkpoint(verify=True)["learning_updates"] == 102
    model.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="不静默回退"):
        controller.latest_checkpoint(verify=True)


def test_dashboard_wide_characters_and_plain_output(tmp_path):
    controller = fixture_controller(tmp_path, tmp_path / "run")
    snapshot = controller.snapshot()
    for columns in (40, 88):
        view = cli.render(snapshot, color=False, columns=columns)
        assert "\033" not in view
        assert all(cli.width(line) == columns for line in view.splitlines())
    assert "启动" in cli.render(snapshot, color=False)


def test_known_physics_block_prevents_launch_without_changing_budget(tmp_path):
    controller = fixture_controller(tmp_path, tmp_path / "run")
    controller.config["blocked_reasons"] = ["已定位的接触异常修复尚未部署"]
    before = Path(controller.config["ledger"]).read_bytes()
    with pytest.raises(ValueError, match="接触异常"):
        controller.start()
    assert Path(controller.config["ledger"]).read_bytes() == before
    assert not controller.session_path.exists()


def cold_controller(tmp_path):
    controller = fixture_controller(tmp_path, tmp_path / "original")
    source = tmp_path / "source"
    (source / "src/wheeled_algo").mkdir(parents=True)
    hook = source / "src/wheeled_algo/training_control.py"
    hook.write_bytes((ROOT / "src/wheeled_algo/training_control.py").read_bytes())
    contract = source / "contract.json"
    atomic_json(contract, {"cpu_fixture": True})
    runtime = source / "runtime.sh"
    runtime.write_text("set -euo pipefail\n")
    worker = source / "worker.py"
    worker.write_text(f"import sys\nsys.path[:0] = {[str(ROOT / 'src'), str(ROOT / 'tests')]!r}\n"
        "from test_training_control import fixture_worker\n"
        "fixture_worker(sys.argv[sys.argv.index('--run-dir') + 1])\n")
    model = tmp_path / "model.pt"
    model.write_bytes(b"CPU fixture only")
    metadata = {"attempt_id": "fixture", "paid_updates_attempt": 2}
    seal = model.with_suffix(".checkpoint.json")
    atomic_json(seal, {"learning_updates": 102, "contract_sha256": cli.digest(contract),
                      "checkpoint_sha256": cli.digest(model), "training_budget": metadata})
    ledger = controller.ledger()
    ledger["checkpoints"]["checkpoint"] = {"path": str(model), "status": "registered",
        "metadata": metadata, "seal_sha256": cli.digest(seal), "checkpoint_sha256": cli.digest(model)}
    ledger["reservations"] = [{"attempt_id": "fixture"}] * 3  # Includes paid work after saving.
    atomic_json(controller.config["ledger"], ledger)
    review = source / "review.json"
    atomic_json(review, {"attempts": {"fixture": ledger["attempts"]["fixture"]["identity"]},
                        "absolute_deadline_utc": "2099-01-01T00:00:00+00:00", "paid_update_cap": None})
    argv = [sys.executable, str(worker), "--run-dir", str(tmp_path / "ignored"),
            "--contract", str(contract), "--resume", str(model), "--updates", "80", "--rng-mode", "inherit",
            "--budget-attempt-id", "fixture", "--budget-ledger", controller.config["ledger"],
            "--budget-review-file", str(review), "--budget-review-sha256", cli.digest(review)]
    controller.config["launch"] = {"cwd": str(source), "runtime": str(runtime), "argv": argv,
        "files_sha256": {str(p.relative_to(source)): cli.digest(p) for p in (worker, runtime, hook, contract)}}
    atomic_json(controller.path, controller.config)
    return controller


def test_cold_start_uses_remaining_paid_budget_and_is_detached(tmp_path):
    controller = cold_controller(tmp_path)
    ledger_before = Path(controller.config["ledger"]).read_bytes()
    plan = controller.start(dry_run=True)
    assert plan["remaining_updates"] == 77  # Not 78 from the older checkpoint.
    assert not controller.session_path.exists()
    started = controller.start()
    run = Path(started["run_directory"])
    try:
        assert started["phase"] == "starting"
        wait_for(lambda: read_json(run / "control/status.json", {}).get("learning_updates", 0))
        assert controller.start()["message"].startswith("现有进程")
        paused = controller.request("pause", timeout=4)
        assert paused["identity"]["pid"] == started["pid"]
        controller.request("stop", timeout=4)
        assert Path(controller.config["ledger"]).read_bytes() == ledger_before
        assert read_json(controller.session_path)["checkpoint"]["learning_updates"] == 102
        assert read_json(run / "completion.json")["status"] == "stopped"
    finally:
        wait_for(lambda: not cli.identity_alive(read_json(run / "control/status.json", {}).get("identity")), seconds=5)


def test_cold_start_refuses_modified_source_or_missing_review_history(tmp_path):
    controller = cold_controller(tmp_path)
    worker = Path(controller.config["launch"]["cwd"]) / "worker.py"
    worker.write_text(worker.read_text() + "\n# Changed after freezing\n")
    with pytest.raises(ValueError, match="启动依赖"):
        controller.plan()


def test_entry_hooks_only_after_logging_and_hashes_control_module():
    # Guard the real entry integration; the process tests above exercise behavior.
    import ast
    tree = ast.parse((ROOT / "scripts/train_chassis.py").read_text())
    log = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "controlled_log")
    assert ast.unparse(log.body[0].value).startswith("control_original_log(")
    assert "training_control.boundary" in ast.unparse(log.body[1])
    source = (ROOT / "scripts/train_chassis.py").read_text()
    assert '"src/wheeled_algo/training_control.py"' in source
