#!/usr/bin/env python3
"""Archive committed Round4 source into a new experiment; no parent and no training."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from r4_common import (
    CODE_DIRECTORY, CONTRACT, EXPERIMENTS, REPO, ControlSSH, JobError,
    AUTHORIZATION, add_ssh_arguments, file_record, json_bytes, legacy_identity, strict_json,
)


def committed_files(commit):
    commit = subprocess.check_output(["git", "rev-parse", "--verify", commit + "^{commit}"], cwd=REPO, text=True).strip()
    raw = subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=REPO)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise JobError("committed archive must contain only confined regular files")
            files[CODE_DIRECTORY + "/" + member.name] = archive.extractfile(member).read()
    required = [CONTRACT, "scripts/train_v40.py", "scripts/start_v40_round2.py", "scripts/pull_v40_artifacts.py"]
    required += [AUTHORIZATION]
    required += ["scripts/round4/" + name + ".py" for name in ("r4_common", "deploy", "launch", "watch", "evaluation", "schedule_serial")]
    if any(CODE_DIRECTORY + "/" + name not in files for name in required):
        raise JobError("commit must contain the complete Round4 contract, trainer and wrappers")
    contract = strict_json(files[CODE_DIRECTORY + "/" + CONTRACT])
    if not isinstance(contract.get("round4"), dict) or not contract["round4"]:
        raise JobError("committed contract lacks the Round4 full profile")
    legacy_identity(contract, lambda name: files[CODE_DIRECTORY + "/" + name])
    return commit, files


def make_bundle(commit, files, output):
    contract = strict_json(files[CODE_DIRECTORY + "/" + CONTRACT])
    identity = legacy_identity(contract, lambda name: files[CODE_DIRECTORY + "/" + name])
    manifest = {
        **identity,
        "schema_version": 1, "git_commit": commit, "working_tree_used": False,
        "code_directory_name": CODE_DIRECTORY, "contract": CONTRACT,
        "initialization": "scratch", "parent": None,
        "files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in files.items()},
    }
    with output.open("xb") as target, gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name, data in sorted({**files, "snapshot.json": json_bytes(manifest)}.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode = len(data), 0o644
                archive.addfile(member, io.BytesIO(data))
    return file_record(output)["sha256"]


def unpack(root, digest):
    """Self-contained stdlib remote unpack, also exercised by CPU tests."""
    import hashlib
    import json
    from pathlib import Path
    import tarfile

    root = Path(root)
    if (root / "isaac_wheeled_rl_train").exists() or (root / "snapshot.json").exists():
        raise FileExistsError("experiment already unpacked")
    bundle = root / "bundle.tar.gz"
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != digest:
        raise ValueError("bundle SHA mismatch")
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError("archive must contain unique regular files")
        manifest = json.load(archive.extractfile("snapshot.json"))
        if (set(names) != set(manifest["files"]) | {"snapshot.json"}
                or manifest["initialization"] != "scratch" or manifest["parent"] is not None):
            raise ValueError("archive identity mismatch")
        for name in manifest["files"]:
            p = Path(name)
            if p.is_absolute() or ".." in p.parts or len(p.parts) < 2 or p.parts[0] != "isaac_wheeled_rl_train":
                raise ValueError("unconfined source file")
        for name, expected in manifest["files"].items():
            data = archive.extractfile(name).read()
            if {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} != expected:
                raise ValueError("source file SHA mismatch")
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        with (root / "snapshot.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
    return {"deployed": True, "git_commit": manifest["git_commit"], "experiment": str(root), "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--deploy", action="store_true")
    add_ssh_arguments(parser)
    args = parser.parse_args()
    commit, files = committed_files(args.commit)
    root = EXPERIMENTS / ("round4-full-" + commit)
    if not args.deploy:
        print(json.dumps({"dry_run": True, "git_commit": commit, "experiment": str(root), "initialization": "scratch", "parent": None}))
        return 0
    if args.bundle is None:
        parser.error("--deploy requires a new --bundle path")
    digest = make_bundle(commit, files, args.bundle)
    client = ControlSSH(args.host, args.control_path, args.ssh_port, timeout=600)
    prepare = f"from pathlib import Path; p=Path({str(root)!r}); assert p.parent.is_dir(); p.mkdir()"
    client.exec(shlex.join(["python3", "-c", prepare]))
    client.upload(args.bundle, str(root / "bundle.tar.gz"))
    import inspect
    script = inspect.getsource(unpack) + f"\nimport json\nprint(json.dumps(unpack({str(root)!r}, {digest!r})))"
    print(client.exec(shlex.join(["python3", "-c", script]), timeout_ms=300000))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
