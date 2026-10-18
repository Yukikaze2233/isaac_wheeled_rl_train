"""Invoke the actual CLI subprocesses against an isolated CPU Adam worker."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import subprocess
import sys
import time

import pytest

torch = pytest.importorskip("torch")
from wheeled_algo.training_control import atomic_json, identity_alive, read_json
from wheeled_tasks.chassis.training_budget import BudgetOwner, REVIEW_SCHEMA, digest_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_fixture", ROOT / "tests/fixtures/training_control_cpu_worker.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


def wait_for(predicate, timeout=30.):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        time.sleep(.03)
    raise AssertionError("CLI process did not reach the requested state")


def setup_fixture(tmp_path):
    source = tmp_path / "source"
    paths = ("tests/fixtures/training_control_cpu_worker.py", "src/wheeled_algo/training_control.py",
             "src/wheeled_tasks/chassis/training_budget.py")
    for relative in paths:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    runtime = source / "runtime.sh"
    runtime.write_text('set -euo pipefail\nexport PYTHONPATH="' + str(source / "src") + '"\n')
    contract = source / "contract.json"
    atomic_json(contract, {"fixture": "CPU Adam only, no simulation or PPO"})
    identity = {"contract_sha256": digest_file(contract), "training_source_sha256": "b" * 64,
        "seed": 321, "stage": "foundation", "requested_updates": 200, "parent_checkpoint_sha256": None}
    review = {"schema": REVIEW_SCHEMA, "campaign_id": "cli_command_cpu_fixture", "ledger_id": "cpu_only",
        "ledger_path": str(tmp_path / "ledger.json"), "paid_update_cap": 20000,
        "pilot_paid_update_cap": 0, "absolute_deadline_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "max_attempt_wall_seconds": 3600., "gpu_occupied_seconds_cap": 1.,
        "gpu_meter_owner": {"kind": "test_only_cpu_meter_v1", "id": fixture.Meter.owner_id},
        "review_reference": "Isolated CPU lifecycle test, never formal work",
        "attempts": {"cpu": {**identity, "role": "main"}}}
    review_path = source / "review.json"
    atomic_json(review_path, review)
    book = BudgetOwner.authenticate(review_path, digest_file(review_path), review["ledger_path"], "cpu",
        {**identity, "requested_updates": 1}, gpu_meter=fixture.Meter(), allow_test_meter=True)
    model = tmp_path / "initial.pt"
    torch.manual_seed(321)
    actor, critic = torch.nn.Linear(3, 2), torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=.001)
    receipt = book.prepare_checkpoint(model)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(), "update": 0,
                "infos": {"training_budget": receipt}}, model)
    atomic_json(model.with_suffix(".checkpoint.json"), {"checkpoint_sha256": digest_file(model),
        "contract_sha256": digest_file(contract), "learning_updates": 0, "training_budget": receipt})
    book.register_checkpoint(model, receipt)
    book.close()
    config = tmp_path / "config.json"
    argv = [sys.executable, str(source / paths[0]), "--run-dir", str(tmp_path / "original"),
            "--contract", str(contract), "--resume", str(model), "--updates", "200", "--rng-mode", "inherit",
            "--budget-attempt-id", "cpu", "--budget-ledger", review["ledger_path"],
            "--budget-review-file", str(review_path), "--budget-review-sha256", digest_file(review_path)]
    atomic_json(config, {"schema": "v7ctl-v1", "run_directory": str(tmp_path / "original"),
        "attempt_id": "cpu", "ledger": review["ledger_path"], "blocked_reasons": [],
        "launch": {"cwd": str(source), "runtime": str(runtime), "argv": argv,
            "files_sha256": {str(p.relative_to(source)): digest_file(p) for p in
                [*(source / relative for relative in paths), runtime, contract]}}})
    command = [os.environ.get("V7CTL_TEST_COMMAND", sys.executable)]
    if "V7CTL_TEST_COMMAND" not in os.environ:
        command.append(str(ROOT / "tools/v7ctl.py"))
    return [*command, "--config", str(config)], config


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_installed_cli_commands_and_dashboard_keys(tmp_path):
    command, config = setup_fixture(tmp_path)
    transcript = []

    def run(*args, expect=0):
        result = subprocess.run([*command, *args], capture_output=True, text=True, timeout=40)
        transcript.append({"argv": list(args), "exit_code": result.returncode,
                           "stdout": result.stdout[-1000:], "stderr": result.stderr[-1000:]})
        assert result.returncode == expect, result.stdout + result.stderr
        return result.stdout

    def state():
        session = read_json(tmp_path / "session.json", {})
        path = Path(session.get("run_directory", tmp_path / "original"))
        return read_json(path / "control/status.json", {})

    try:
        assert "V7" in run("--help")
        assert "V7" in run()  # Non-TTY default is one-shot status.
        plan = json.loads(run("--json", "plan"))
        assert plan["remaining_updates"] == 200
        run("pause", expect=2)
        run("stop", expect=2)
        run("start")
        wait_for(lambda: state().get("phase") == "running")
        first_pid = state()["identity"]["pid"]
        run("start")
        assert state()["identity"]["pid"] == first_pid
        run("pause")
        paused = state()
        assert paused["phase"] == "paused"
        paid_before = len(read_json(tmp_path / "ledger.json")["reservations"])
        saved = torch.load(paused["checkpoint"], weights_only=True, map_location="cpu")
        time.sleep(.25)
        run("pause")
        assert len(read_json(tmp_path / "ledger.json")["reservations"]) == paid_before
        assert state()["learning_updates"] == saved["update"]
        status = json.loads(run("--json", "status"))
        assert status["phase"] == "paused" and status["paid_updates_global"] == paid_before
        run("resume")
        wait_for(lambda: state().get("learning_updates", 0) > saved["update"])
        run("stop")
        assert not identity_alive(state()["identity"])
        final_state = state()
        saved = torch.load(final_state["checkpoint"], weights_only=True, map_location="cpu")
        assert saved["optimizer"]["state"]  # Adam moments actually exist.
        assert "CPU_FIXTURE_UPDATE" in run("logs")
        follower = subprocess.Popen([*command, "logs", "-f"], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, start_new_session=True)
        try:
            assert select.select([follower.stdout], [], [], 4.)[0]
            assert b"CPU_FIXTURE_UPDATE" in os.read(follower.stdout.fileno(), 4096)
        finally:
            os.killpg(follower.pid, signal.SIGINT)
            follower.communicate(timeout=5)
        run("start")
        wait_for(lambda: state().get("phase") == "running" and state()["identity"]["pid"] != first_pid)
        resumed_run = Path(read_json(tmp_path / "session.json")["run_directory"])
        loaded = torch.load(resumed_run / "loaded.pt", weights_only=True, map_location="cpu")
        assert_same(loaded, {key: saved[key] for key in loaded})
        master, slave = pty.openpty()
        panel = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave,
                                 env={**os.environ, "TERM": "xterm-256color"}, start_new_session=True)
        os.close(slave)
        captured = bytearray()

        def read_panel():
            if select.select([master], [], [], .1)[0]:
                captured.extend(os.read(master, 65536))
            return b"\x1b[?1049h" in captured

        try:
            wait_for(read_panel)
            os.write(master, b"p")
            wait_for(lambda: state().get("phase") == "paused")
            same_pid = state()["identity"]["pid"]
            os.write(master, b"r")
            wait_for(lambda: state().get("phase") == "running")
            assert state()["identity"]["pid"] == same_pid
            os.write(master, b"s")
            wait_for(lambda: state().get("phase") == "stopped")
            os.write(master, b"q")
            panel.wait(timeout=5)
            assert panel.returncode == 0
        finally:
            if panel.poll() is None:
                panel.terminate()
                panel.wait(timeout=5)
            os.close(master)
        run("status")
        run("unknown-command", expect=2)
        transcript.append({"dashboard_keys": ["p", "r", "s", "q"],
            "cold_resume_actor_critic_adam_rng_exact": True, "formal_ppo_used": 0})
    finally:
        if identity_alive(state().get("identity")):
            subprocess.run([*command, "stop"], capture_output=True, timeout=40)
        output = os.environ.get("V7CTL_TEST_RECEIPT")
        if output:
            atomic_json(output, {"scope": "installed CLI, isolated real CPU Adam worker, not Isaac/PPO",
                                 "commands": transcript, "fixture_directory": str(tmp_path)})
