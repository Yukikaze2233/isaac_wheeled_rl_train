"""Stdlib-only Round3 deployment identity and existing-ControlMaster transport."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from pull_v40_artifacts import Unavailable, absolute_remote_path
from wheeled_algo.v40_job import JobError, strict_json

PARENT_SHA256 = "f94919f8f7ae1cfee95e45b922a9939e21c9f6ad9dff5656ff6f29dc0f4e6360"
CONTRACT = "contracts/own_v40_round3_a.json"
CODE_DIRECTORY = "isaac_wheeled_rl_train"


def code_directory(snapshot: Path) -> Path:
    """Keep source at the fixed sibling directory, separate from deployment metadata."""
    return snapshot.absolute().parent / CODE_DIRECTORY


def verify_warm_start(record, contract_sha256):
    lineage = record.get("source_provenance", {})
    if (lineage.get("mode") != "warm_start" or lineage.get("parent_checkpoint_sha256") != PARENT_SHA256
            or lineage.get("target_contract_sha256") != contract_sha256
            or lineage.get("optimizer_reset") is not True
            or type(lineage.get("initial_iteration")) is not int or lineage["initial_iteration"] != 0):
        raise JobError("warm-start lineage must identify the Round2 parent, reset optimizer and iteration zero")


def file_record(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise JobError(f"regular file required: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def verify_snapshot(root: Path) -> dict:
    root = root.absolute()
    manifest = strict_json((root / "snapshot.json").read_bytes())
    if (manifest.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("git_commit", ""))
            or manifest.get("parent_model_sha256") != PARENT_SHA256
            or manifest.get("code_directory_name") != CODE_DIRECTORY):
        raise JobError("snapshot identity does not match the approved Round2 parent")
    repo = code_directory(root)
    if repo.is_symlink() or not repo.is_dir():
        raise JobError("fixed code directory must be a real directory, not a symlink")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
            raise JobError("snapshot member escapes its root")
        if relative.parts[0] == CODE_DIRECTORY:
            base, path = repo, repo.joinpath(*relative.parts[1:])
        elif relative.parts[0] == "parent":
            base, path = root, root / relative
        else:
            raise JobError("unknown snapshot member root")
        if not path.resolve().is_relative_to(base.resolve()):
            raise JobError("snapshot member escapes its root")
        if file_record(path) != expected:
            raise JobError(f"snapshot hash mismatch: {name}")
    if manifest["files"]["parent/model_final.pt"]["sha256"] != PARENT_SHA256:
        raise JobError("smoke/other checkpoint cannot replace the approved parent")
    return manifest


class ControlSSH:
    """Adapter for pull_v40_artifacts; never falls back to a new authentication."""

    def __init__(self, host, control_path, port=2222, timeout=90):
        if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", host) or host.startswith("-"):
            raise JobError("explicit SSH user@host required")
        self.alias = self.host = host
        self.control_path = str(Path(control_path).absolute())
        self.port, self.timeout = int(port), timeout

    def options(self):
        return ["-o", f"ControlPath={self.control_path}", "-o", "ControlMaster=no",
                "-o", "BatchMode=yes", "-o", "ProxyCommand=false"]

    def check(self):
        try:
            result = subprocess.run(["ssh", *self.options(), "-p", str(self.port), "-O", "check", self.host],
                                    capture_output=True, timeout=min(self.timeout, 10))
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Unavailable("ControlMaster unavailable; restore authentication separately") from error
        if result.returncode:
            raise Unavailable("ControlMaster unavailable; restore authentication separately (no password fallback)")

    def exec(self, command, *, timeout_ms=60000):
        self.check()
        try:
            result = subprocess.run(["ssh", *self.options(), "-p", str(self.port), "-T", self.host, command],
                                    capture_output=True, text=True, timeout=min(self.timeout, timeout_ms / 1000 + 5))
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Unavailable("SSH request unavailable; remote training is not stopped") from error
        if result.returncode == 255:
            raise Unavailable("SSH transport unavailable; restore the ControlMaster")
        if result.returncode:
            raise JobError(f"remote command exited {result.returncode}: {result.stderr[-3000:]}")
        return result.stdout

    def upload(self, local, remote):
        self.check()
        absolute_remote_path(remote)
        result = subprocess.run(["scp", *self.options(), "-P", str(self.port), str(local), f"{self.host}:{remote}"],
                                capture_output=True, text=True, timeout=self.timeout)
        if result.returncode:
            raise Unavailable(f"SCP upload unavailable: {result.stderr[-1000:]}")

    @contextmanager
    def download(self, remote):
        self.check()
        absolute_remote_path(remote)
        with tempfile.TemporaryDirectory(prefix="round3-transfer-") as temporary:
            path = Path(temporary) / "payload"
            try:
                result = subprocess.run(["scp", *self.options(), "-P", str(self.port), f"{self.host}:{remote}", str(path)],
                                        capture_output=True, text=True, timeout=self.timeout)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise Unavailable("SCP download interrupted; a new verified attempt may retry") from error
            if result.returncode:
                raise Unavailable(f"SCP download unavailable: {result.stderr[-1000:]}")
            with path.open("rb") as stream:
                yield stream


def add_ssh_arguments(parser):
    parser.add_argument("--host", required=True)
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--ssh-port", type=int, default=2222)
