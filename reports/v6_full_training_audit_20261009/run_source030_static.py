#!/usr/bin/env python3
"""Run only the frozen source030 transfer/stage/CPU/check steps, never launch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

REPORT = Path(__file__).resolve().parent
HOST = "kaiser@192.168.64.234"
RUN = Path("/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358")
OLD = RUN / "source_serial_dispatch029_20261009"
NEW = RUN / "source_serial_dispatch030_20261009"
PYTHON = "/home/kaiser/robot-rl-sim60/env/bin/python"
HELPER = REPORT / "source030_deploy/deploy_source030.py"
HELPER_SHA = "612c66e8f7cbdd56d65ed88a1861c0d2add1fcefba33704adfa177664bcea6cb"
REMOTE_HELPER = RUN / "deploy_source030_612c66e8.py"
SSH = ["ssh", "-p", "2222", "-oBatchMode=yes", "-oConnectTimeout=10", HOST]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("transfer", "static-stage", "static-preflight", "controller-check"))
    args = parser.parse_args()
    assert digest(HELPER) == HELPER_SHA
    freeze = json.loads((REPORT / "source030_freeze002.json").read_text())
    archive = Path(freeze["archive"])
    remote_archive = RUN / archive.name
    output = REPORT / f"source030_{args.mode.replace('-', '_')}001"
    assert not any(output.with_suffix(s).exists() for s in (".intent.json", ".execution.json", ".stdout", ".stderr"))
    expected = {archive.name: freeze["archive_sha256"], HELPER.name: HELPER_SHA}
    arguments = ["--run-root", str(RUN), "--old-source-root", str(OLD), "--new-source-root", str(NEW),
                 "--source-archive", str(remote_archive), "--source-commit", freeze["commit"],
                 "--source-archive-sha256", freeze["archive_sha256"]]
    intent = dict(at_utc=datetime.now(timezone.utc).isoformat(), mode=args.mode,
                  freeze=dict(file="source030_freeze002.json", sha256=digest(REPORT / "source030_freeze002.json")),
                  helper_sha256=HELPER_SHA, qualification=None, PPO_updates_by_helper=0)
    output.with_suffix(".intent.json").write_text(json.dumps(intent, indent=2) + "\n")
    started = time.monotonic()
    records = []

    def execute(command: list[str], *, input: str | None = None) -> str:
        result = subprocess.run(command, input=input, text=True, capture_output=True, timeout=900)
        records.append(dict(command=command, exit_code=result.returncode,
                            stdin_sha256=hashlib.sha256(input.encode()).hexdigest() if input else None))
        with output.with_suffix(".stdout").open("a") as stream:
            stream.write(result.stdout)
        with output.with_suffix(".stderr").open("a") as stream:
            stream.write(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{args.mode} returned {result.returncode}; retained output must be inspected")
        return result.stdout

    def remote_script(code: str) -> str:
        return execute([*SSH, shlex.join([PYTHON, "-B", "-"])], input=code)

    try:
        if args.mode == "transfer":
            assert digest(archive) == freeze["archive_sha256"]
            incoming = RUN / "source030_incoming001"
            init = f'''import json, os
from pathlib import Path
root=Path({str(RUN)!r}); incoming=Path({str(incoming)!r})
for p in (incoming, Path({str(NEW)!r}), root/'source030_deploy', Path({str(remote_archive)!r}), Path({str(REMOTE_HELPER)!r})):
    assert not p.exists(), str(p)
free=os.statvfs(root); available=free.f_bavail*free.f_frsize
assert available > 2_000_000_000, available
incoming.mkdir(mode=0o700)
print(json.dumps(dict(status='exclusive_incoming_reserved', available_bytes=available)))
'''
            execute([*SSH, shlex.join([PYTHON, "-B", "-"])], input=init)
            execute(["scp", "-P", "2222", "-oBatchMode=yes", "-oConnectTimeout=10",
                     str(archive), str(HELPER), f"{HOST}:{incoming}/"])
            activate = f'''import hashlib,json,os
from pathlib import Path
incoming=Path({str(incoming)!r}); expected={expected!r}
for name, sha in expected.items():
    with (incoming/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==sha,name
for name, destination in { {archive.name: str(remote_archive), HELPER.name: str(REMOTE_HELPER)}!r}.items():
    os.link(incoming/name,destination)
    (incoming/name).unlink()
incoming.rmdir()
print(json.dumps(dict(status='frozen_inputs_copied',files=2,sha256=expected)))
'''
            text = remote_script(activate)
        elif args.mode == "controller-check":
            command = [PYTHON, "-B", str(NEW / "scripts/hold_completed_speed_boundary.py"),
                       "--source-root", str(OLD), "--run-root", str(RUN / "train"),
                       "--intent", str(RUN / "source030_speed_boundary.json")]
            text = execute([*SSH, shlex.join(["env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", *command])])
        else:
            check = f'''import hashlib
from pathlib import Path
with Path({str(REMOTE_HELPER)!r}).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()=={HELPER_SHA!r}
'''
            remote_script(check)
            command = [PYTHON, "-B", str(REMOTE_HELPER), "--mode", args.mode, *arguments]
            text = execute([*SSH, shlex.join(["env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", *command])])
        result = json.loads(text)
        output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        status = "completed"
    except Exception:
        status = "failed_retained_not_retried"
        raise
    finally:
        output.with_suffix(".execution.json").write_text(json.dumps(dict(
            status=status, wall_seconds=time.monotonic()-started, calls=records,
            PPO_updates_by_helper=0, train_signals=0, new_training_launches=0,
        ), indent=2) + "\n")
    print(json.dumps(dict(mode=args.mode, status=result.get("status"),
                         file_count=len(result.get("files", {})) if isinstance(result.get("files"), dict) else result.get("files"),
                         qualification=None)))


if __name__ == "__main__":
    main()
