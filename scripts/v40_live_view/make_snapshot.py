#!/usr/bin/env python3
"""Archive current tracked + untracked source content, including dirty edits, without Git mutation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
names = subprocess.check_output(
    ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
     "src", "scripts", "tools", "contracts", "assets", "tests", "docs", "pyproject.toml", "README.md"], cwd=ROOT,
).decode().split("\0")
hashes = {}
with args.output.open("xb") as output, tarfile.open(fileobj=output, mode="w:gz") as archive:
    for name in sorted(set(names) - {""}):
        path = ROOT / name
        if path.is_symlink():
            raise ValueError(f"snapshot requires regular source files: {name}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        member = tarfile.TarInfo(name)
        member.size, member.mode = len(data), 0o644
        archive.addfile(member, io.BytesIO(data))
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"concurrent edit while archiving: {name}")
        hashes[name] = digest
    identity = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "working_tree_content": True, "files_sha256": hashes}
    data = json.dumps(identity, indent=2).encode()
    member = tarfile.TarInfo("snapshot.json")
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))
print(json.dumps({"output": str(args.output), "files": len(hashes),
                  "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))
