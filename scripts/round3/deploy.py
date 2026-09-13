#!/usr/bin/env python3
"""Deploy committed source and the verified Round2 parent into a NEW snapshot; never launch training."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

from common import CODE_DIRECTORY, CONTRACT, PARENT_SHA256, REPO, ControlSSH, add_ssh_arguments, code_directory, file_record
from wheeled_algo.v40_job import JobError, artifact_record, json_bytes, strict_json, validate_completion

REMOTE_UNPACK = r'''
import hashlib, json, pathlib, tarfile
root = pathlib.Path(ROOT)
code = root.parent / "isaac_wheeled_rl_train"
if code.exists() or code.is_symlink():
    raise FileExistsError("fixed code directory already exists; refusing overwrite or symlink")
bundle = root / "bundle.tar.gz"
if hashlib.sha256(bundle.read_bytes()).hexdigest() != DIGEST:
    raise ValueError("uploaded bundle SHA mismatch")
with tarfile.open(bundle, "r:gz") as archive:
    members = archive.getmembers()
    names = [m.name for m in members]
    if len(names) != len(set(names)) or any(not m.isfile() for m in members):
        raise ValueError("only unique regular files allowed")
    manifest = json.load(archive.extractfile("snapshot.json"))
    if set(names) != set(manifest["files"]) | {"snapshot.json"}:
        raise ValueError("bundle members differ from manifest")
    if manifest.get("code_directory_name") != "isaac_wheeled_rl_train":
        raise ValueError("unexpected code directory name")
    for name in names:
        if name == "snapshot.json":
            continue
        relative = pathlib.PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2 or relative.parts[0] not in ("isaac_wheeled_rl_train", "parent"):
            raise ValueError("unconfined bundle member")
    code.mkdir()
    for member in members:
        if member.name == "snapshot.json":
            continue
        relative = pathlib.PurePosixPath(member.name)
        data = archive.extractfile(member).read()
        if {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} != manifest["files"][member.name]:
            raise ValueError("bundle file hash mismatch")
        target = code.joinpath(*relative.parts[1:]) if relative.parts[0] == "isaac_wheeled_rl_train" else root / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    with (root / "snapshot.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
(root / "deployment.json").write_text(json.dumps({"verified": True, "bundle_sha256": DIGEST,
    "git_commit": manifest["git_commit"], "code_dir": str(code), "parent_model_sha256": manifest["parent_model_sha256"]}, indent=2))
print(json.dumps({"deployed": True, "snapshot": str(root), "code_dir": str(code), "git_commit": manifest["git_commit"],
                  "training_started": False}))
'''


def committed_files(commit):
    sha = subprocess.check_output(["git", "rev-parse", "--verify", commit + "^{commit}"], cwd=REPO, text=True).strip()
    raw = subprocess.check_output(["git", "archive", "--format=tar", sha], cwd=REPO)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise JobError("committed snapshot requires regular, confined files")
            files[CODE_DIRECTORY + "/" + member.name] = archive.extractfile(member).read()
    required = (CONTRACT, "scripts/train_v40.py", "scripts/round3/launch.py", "scripts/round3/common.py")
    if any(CODE_DIRECTORY + "/" + name not in files for name in required):
        raise JobError("commit does not yet include Round3-A contract/launcher; commit the reviewed integration first")
    if b"--warm-start" not in files[CODE_DIRECTORY + "/scripts/train_v40.py"]:
        raise JobError("committed train entry does not yet expose --warm-start")
    return sha, files


def parent_files(parent):
    record = validate_completion(strict_json((parent / "completion.json").read_bytes()))
    if record["status"] != "completed" or record["export_status"] != "verified":
        raise JobError("parent must be the completed, exported Round2 final")
    manifest = strict_json((parent / "run_manifest.json").read_bytes())
    if manifest["contract_id"] != "own-v40-jointspace-h5-v2":
        raise JobError("parent is not the Round2 research contract")
    files = {"parent/completion.json": (parent / "completion.json").read_bytes()}
    for item in record["artifacts"]:
        if artifact_record(parent, item["path"]) != item:
            raise JobError(f"parent artifact mismatch: {item['path']}")
        files["parent/" + item["path"]] = (parent / item["path"]).read_bytes()
    if hashlib.sha256(files["parent/model_final.pt"]).hexdigest() != PARENT_SHA256:
        raise JobError("only the approved Round2 final parent is accepted, never a smoke checkpoint")
    return files


def make_bundle(commit, parent, output):
    sha, files = committed_files(commit)
    files.update(parent_files(parent))
    manifest = {"schema_version": 1, "git_commit": sha, "working_tree_used": False,
                "code_directory_name": CODE_DIRECTORY,
                "contract": CONTRACT, "parent_model_sha256": PARENT_SHA256,
                "files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in files.items()}}
    files["snapshot.json"] = json_bytes(manifest)
    with output.open("xb") as target, gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name, data in sorted(files.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode = len(data), 0o644
                archive.addfile(member, io.BytesIO(data))
    return manifest, file_record(output)["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--deploy", action="store_true", help="Without this flag: validate/print only, no files or network")
    add_ssh_arguments(parser)
    args = parser.parse_args()
    from common import absolute_remote_path
    remote = absolute_remote_path(args.remote_dir)
    if Path(remote).parent != Path("/home/kaiser/robot-rl-sim60") or Path(remote).name == CODE_DIRECTORY:
        raise JobError("deployment metadata must be a separate directory directly under /home/kaiser/robot-rl-sim60")
    sha, _ = committed_files(args.commit)
    parent_files(args.parent)
    if not args.deploy:
        print(json.dumps({"dry_run": True, "git_commit": sha, "remote_dir": remote,
                          "code_dir": str(code_directory(Path(remote))),
                          "parent_sha256": PARENT_SHA256, "working_tree_used": False, "training_started": False}, indent=2))
        return
    manifest, digest = make_bundle(sha, args.parent, args.bundle)
    client = ControlSSH(args.host, args.control_path, args.ssh_port, timeout=600)
    prepare = ("from pathlib import Path; p=Path(" + repr(remote) + "); c=p.parent/'isaac_wheeled_rl_train'; "
               "assert not c.exists() and not c.is_symlink(), 'fixed code directory already exists'; "
               "assert p.parent.is_dir(), 'deployment parent missing'; p.mkdir()")
    client.exec(shlex.join(["python3", "-c", prepare]))
    client.upload(args.bundle, remote + "/bundle.tar.gz")
    script = REMOTE_UNPACK.replace("ROOT", repr(remote), 1).replace("DIGEST", repr(digest))
    print(client.exec(shlex.join(["python3", "-c", script]), timeout_ms=300000))
    print(json.dumps({"bundle": str(args.bundle), "bundle_sha256": digest, "git_commit": manifest["git_commit"]}))


if __name__ == "__main__":
    try:
        main()
    except (JobError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Deployment not confirmed: {error}; inspect any retained snapshot before retrying", file=sys.stderr)
        raise SystemExit(2)
