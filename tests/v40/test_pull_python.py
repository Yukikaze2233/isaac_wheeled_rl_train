"""Remote interpreter regressions: in-memory HTTP and local stdlib probes only."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit

import pytest

from test_job import Clock, job, pull, stub_run


class MemoryHTTP:
    """Exercise ManagedSSH's wire format without opening any socket."""

    def __init__(self, *, pending=False):
        self.pending = pending
        self.commands = []
        self.downloads = []

    def open(self, request, *, timeout):
        url = urlsplit(request.full_url)
        assert url.netloc == "127.0.0.1:3080"
        if url.path == pull.EXEC_ROUTE:
            body = json.loads(request.data)
            assert set(body) == {"alias", "command", "timeoutMs"}
            assert body["alias"] == "OFFLINE_FIXTURE_ONLY"
            self.commands.append(body["command"])
            if self.pending:
                self.pending = False
                stdout = '{"state":"pending"}'
            else:
                result = subprocess.run(
                    ["/bin/sh", "-c", body["command"]],
                    env={**os.environ, "PATH": ""},
                    capture_output=True, text=True, timeout=10,
                )
                assert result.returncode == 0, result.stderr
                stdout = result.stdout
            return io.BytesIO(job.json_bytes({"result": {
                "success": True, "exitCode": 0, "timedOut": False,
                "stdout": stdout, "stderr": "",
            }}))
        assert url.path == pull.DOWNLOAD_ROUTE
        params = parse_qs(url.query)
        assert set(params) == {"alias", "remotePath"}
        assert params["alias"] == ["OFFLINE_FIXTURE_ONLY"]
        path = Path(params["remotePath"][0])
        self.downloads.append(path)
        return io.BytesIO(path.read_bytes())


@pytest.fixture
def remote_python(tmp_path):
    interpreter = tmp_path / "ENV 'quoted' $literal; space" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    return str(interpreter)


def test_probe_default_and_absolute_argv():
    default = shlex.split(pull.probe_command("/offline/run"))
    assert default[:2] == ["python3", "-c"] and len(default) == 3
    for interpreter in (
        "/root/autodl-tmp/scut-isaac51-lab230/py311/bin/python",
        "/ENV 'quoted' $literal; space/bin/python",
    ):
        argv = shlex.split(pull.probe_command("/offline/run", interpreter))
        assert argv == [interpreter, "-c", default[2]]


@pytest.mark.parametrize("interpreter", [
    "", "python", "ENV/bin/python", "python3 -I", "/", "/ENV/../bin/python",
    "/ENV/./bin/python", "//ENV/bin/python", "/ENV//bin/python", "/ENV/bin/python/",
    "/ENV\\bin/python", "/ENV/\npython", "/ENV/\x00python",
])
def test_invalid_interpreter_rejected_before_http(tmp_path, monkeypatch, capsys, interpreter):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid interpreter must not contact HTTP")

    monkeypatch.setattr(pull.ManagedSSH, "open", forbidden)
    destination = tmp_path / "received"
    assert pull.main([
        "--alias", "OFFLINE_FIXTURE_ONLY", "--remote-run-dir", "/offline/run",
        "--destination", str(destination), "--remote-python", interpreter,
    ]) == 2
    assert "normalized absolute POSIX path" in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.parametrize("wait", [False, True])
def test_absolute_interpreter_through_cli_and_wait(tmp_path, remote_python, monkeypatch, capsys, wait):
    run, completion, *_ = stub_run(tmp_path)
    destination = tmp_path / "received"
    transport = MemoryHTTP(pending=wait)
    client = pull.ManagedSSH("OFFLINE_FIXTURE_ONLY")
    client.opener = transport
    if wait:
        clock = Clock()
        receipt = pull.pull_artifacts(
            client, str(run), destination, remote_python=remote_python, wait=True,
            monotonic=clock.monotonic, sleep=clock.sleep,
        )
        assert clock.sleeps == [30]
    else:
        monkeypatch.setattr(pull, "ManagedSSH", lambda alias, management: client)
        assert pull.main([
            "--alias", client.alias, "--remote-run-dir", str(run),
            "--destination", str(destination), "--remote-python", remote_python,
        ]) == 0
        receipt = json.loads(capsys.readouterr().out)
    assert receipt["transfer_verified"] is True
    assert receipt["policy_quality_verified"] is False
    assert len(transport.commands) == len(completion["artifacts"]) + 2 + int(wait)
    assert all(shlex.split(command)[:2] == [remote_python, "-c"] for command in transport.commands)
    assert len(transport.downloads) == len(completion["artifacts"])
    for item in completion["artifacts"]:
        assert (destination / item["path"]).read_bytes() == (run / item["path"]).read_bytes()
    assert (destination / "completion.json").read_bytes() == (run / "completion.json").read_bytes()
    assert json.loads((destination / "local_receipt.json").read_bytes()) == receipt
    assert not list(destination.glob("*.partial"))


@pytest.mark.parametrize("exit_code", [1, 127, None, -1])
def test_command_failure_is_not_connection_unavailable(tmp_path, monkeypatch, exit_code):
    detail = "zsh: command not found: python3" if exit_code in (1, 127) else "connection refused"
    result = {"success": False, "exitCode": exit_code, "timedOut": False,
              "stdout": "", "stderr": detail}
    client = pull.ManagedSSH("OFFLINE_FIXTURE_ONLY")
    calls = []

    def response(route, payload=None):
        assert route == pull.EXEC_ROUTE
        calls.append(payload)
        return io.BytesIO(job.json_bytes({"result": result}))

    monkeypatch.setattr(client, "open", response)
    if exit_code in (1, 127):
        clock = Clock()
        with pytest.raises(job.JobError, match=f"remote command failed .*{exit_code}") as error:
            pull.pull_artifacts(
                client, "/offline/run", tmp_path / "received", wait=True,
                monotonic=clock.monotonic, sleep=clock.sleep,
            )
        assert not isinstance(error.value, pull.Unavailable)
        assert "power-on" not in str(error.value)
        assert clock.sleeps == [] and not (tmp_path / "received").exists()
    else:
        with pytest.raises(pull.Unavailable, match="power-on"):
            client.exec("unused fixture command")
    assert len(calls) == 1
