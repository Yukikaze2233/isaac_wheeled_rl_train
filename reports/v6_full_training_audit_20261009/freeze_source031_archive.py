#!/usr/bin/env python3
"""Bind a clean, committed source031 tree to its git archive; no remote action."""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[2] / ".worktrees/stepjump-landing-goal-20261009"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-commit", required=True)
args = parser.parse_args()
COMMIT = args.source_commit
assert len(COMMIT) == 40 and all(c in "0123456789abcdef" for c in COMMIT)
ARCHIVE = ROOT / f"source031-{COMMIT}-umask0022.tar.gz"
MANIFEST = ROOT / "source031_archive_manifest001.json"
RECEIPT = ROOT / "source031_freeze001.json"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=SOURCE)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path: Path, data: object) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    assert git("rev-parse", "HEAD").decode().strip() == COMMIT
    assert git("rev-parse", f"{COMMIT}^").decode().strip() == "0f4b5f673e5f7e2cb382a8fcbd5766b0e13d8a9b"
    assert not git("status", "--porcelain", "--untracked-files=no")
    assert not any(p.exists() for p in (ARCHIVE, MANIFEST, RECEIPT))
    files = {}
    for row in git("ls-tree", "-r", "-z", "-l", COMMIT).split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        mode, kind, blob, raw_size = metadata.decode().split()
        name = raw_path.decode()
        assert mode in {"100644", "100755"} and kind == "blob"
        path = SOURCE / name
        size = int(raw_size)
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == size
        git_hash = hashlib.sha1(f"blob {size}\0".encode())
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                git_hash.update(block)
                sha.update(block)
        assert git_hash.hexdigest() == blob, name
        files[name] = dict(sha256=sha.hexdigest(), size=size, git_mode=mode)
    with ARCHIVE.open("xb") as stream:
        subprocess.run(["git", "-c", "tar.umask=0022", "archive", "--format=tar.gz", COMMIT], cwd=SOURCE,
                       stdout=stream, check=True)
    seen = set()
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        assert archive.pax_headers.get("comment") == COMMIT
        for member in archive:
            if member.isdir():
                continue
            assert member.isfile() and member.name in files and member.name not in seen
            row = files[member.name]
            assert member.size == row["size"]
            assert member.mode == (int(row["git_mode"], 8) & 0o777)
            with archive.extractfile(member) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
            row["archive_mode"] = member.mode
            seen.add(member.name)
    assert seen == set(files)
    generation = json.loads((ROOT / "source031_contract_generation001.json").read_text())
    for row in generation["files"]:
        assert files[row["path"]]["sha256"] == row["sha256"]
    for name, sha in generation["preserved_old_files"].items():
        assert files[name]["sha256"] == sha
    publish(MANIFEST, dict(schema="source031-git-archive-manifest-v1", commit=COMMIT,
                           archive_sha256=digest(ARCHIVE), files=files))
    publish(RECEIPT, dict(
        schema="source031-static-freeze-v1", status="frozen_archive_verified",
        commit=COMMIT, code_parent=git("rev-parse", f"{COMMIT}^").decode().strip(),
        archive=str(ARCHIVE), archive_bytes=ARCHIVE.stat().st_size,
        archive_sha256=digest(ARCHIVE), manifest=str(MANIFEST), manifest_sha256=digest(MANIFEST),
        file_count=len(files), unpacked_bytes=sum(r["size"] for r in files.values()),
        new_static_contracts=len(generation["files"]), old_contracts_preserved=36,
        tar_umask="0022", complete_file_modes_match_git=True,
        supersedes_candidate_commit="81ad124554c4ab45cc274e2d482d46c76ba125b0",
        supersedes_scope="pre-entry future candidate only; original source029 PPO unchanged",
        qualification=None, actual_admission=None, future_checkpoint=None,
        remote_writes=0, PPO_updates=0, freeze_ready=True,
    ))
    print(json.dumps(dict(status="frozen_archive_verified", files=len(files),
                          archive_bytes=ARCHIVE.stat().st_size)))


if __name__ == "__main__":
    main()
