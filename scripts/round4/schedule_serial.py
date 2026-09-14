#!/usr/bin/env python3
"""Supersede the exact repaired cron; deploy and submit the authorized legacy serial run once."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import uuid

from r4_common import (
    AUTHORIZATION, CODE_DIRECTORY, EXPERIMENTS, REPO, ControlSSH, JobError,
    add_ssh_arguments, file_record, json_bytes, strict_json, verify_snapshot, write_json,
)
from round4 import deploy, launch

OLD_MARKER = "# robot-rl-schedule:round4-repaired-20260915"
OLD_DIRECTORY = "/home/kaiser/robot-rl-sim60/scheduled-training/round4-repaired-20260915"


def cancel_old(directory, marker, authorization, run=None):
    """Self-contained remote cancellation; preserve every unrelated cron byte and old evidence."""
    from datetime import datetime, timezone
    import fcntl
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess

    run = run or subprocess.run
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError("old schedule directory missing; inspect before cancellation")
    if (marker != "# robot-rl-schedule:round4-repaired-20260915"
            or authorization.get("explicit_user_authorization") is not True
            or authorization.get("repaired_dynamics_used") is not False
            or authorization.get("supersedes_cron_marker") != marker
            or not authorization.get("user_authorization_text")):
        raise ValueError("explicit superseding authorization is required")
    environment = {**os.environ, "LC_ALL": "C"}

    def read_cron():
        reply = run(["crontab", "-l"], capture_output=True, text=True, env=environment, timeout=10)
        if reply.returncode == 0:
            return reply.stdout
        if reply.returncode == 1 and "no crontab" in reply.stderr.lower():
            return ""
        raise RuntimeError("cannot read the current user crontab")

    with (root / "schedule.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = read_cron()
        lines = before.splitlines(keepends=True)
        matches = [line for line in lines if line.rstrip().endswith(marker)]
        if len(matches) > 1 or (matches and str(root) not in matches[0]):
            raise ValueError("ambiguous cron ownership; refusing to remove other entries")
        after = "".join(line for line in lines if line not in matches)
        if matches:
            if read_cron() != before:
                raise RuntimeError("crontab changed concurrently; retry cancellation")
            run(["crontab", "-"], input=after, capture_output=True, text=True,
                env=environment, check=True, timeout=10)
        if read_cron() != after:
            raise RuntimeError("crontab verification differs; cancellation not confirmed")
        cancelled_path = root / "cancelled.json"
        if not matches and cancelled_path.exists():
            previous_cancellation = json.loads(cancelled_path.read_text())
            if previous_cancellation.get("authorization") != authorization:
                raise ValueError("existing cancellation has a different authorization; preserve its evidence")
            return {**previous_cancellation, "already_cancelled": True, "cron_absence_reverified": True}
        state_path = root / "state.json"
        previous = json.loads(state_path.read_text()) if state_path.exists() else None
        record = {
            "schema_version": 1, "status": "superseded", "cancelled": True,
            "cancelled_at": datetime.now(timezone.utc).isoformat(), "marker": marker,
            "removed_lines": matches, "other_cron_preserved": True,
            "crontab_before_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "crontab_after_sha256": hashlib.sha256(after.encode()).hexdigest(),
            "authorization": authorization, "training_started_by_cancellation": False,
            "previous_training_not_stopped": True, "repaired_asset_validation_passed": False,
        }
        backup = root / "state-before-supersession.json"
        if not backup.exists():
            with backup.open("x") as stream:
                json.dump(previous, stream, ensure_ascii=False, indent=2)
        for target in (root / "cancelled.json", state_path):
            temporary = target.with_suffix(".cancel.tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(target)
        return record


def remote_cancel(client, authorization):
    script = inspect.getsource(cancel_old)
    script += f"\nimport json\nprint(json.dumps(cancel_old({OLD_DIRECTORY!r}, {OLD_MARKER!r}, {authorization!r}), ensure_ascii=False))"
    return strict_json(client.exec(shlex.join(["python3", "-c", script])))


def tick(args, now=None, run=None):
    """The original midnight is a lower bound, never roll an elapsed slot to tomorrow."""
    run = run or subprocess.run
    root = args.experiment.absolute()
    snapshot = verify_snapshot(root)
    start = datetime.fromisoformat(snapshot["initial_not_before"])
    now = now or datetime.now(timezone.utc)
    if start.tzinfo is None or now.tzinfo is None:
        raise JobError("not-before and current time must be timezone-aware")
    status = {"initial_not_before": start.isoformat(), "checked_at": now.isoformat(),
              "physics_model": snapshot["physics_model"], "git_commit": snapshot["git_commit"],
              "training_success": False}
    if now < start:
        return {**status, "status": "waiting_initial_not_before"}
    if not args.execute:
        return {**status, "status": "ready_to_launch_now", "plan": launch.build_plan(args)}
    state_path = root / "serial-schedule.json"
    with (root / "serial-schedule.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if state_path.exists():
            previous = strict_json(state_path.read_bytes())
            if previous["status"] in {"launch_claimed", "submitted", "launch_failed"}:
                return previous
        plan = launch.build_plan(args)
        command = [plan["python"], str(Path(plan["repo"]) / "scripts/round4/launch.py"),
                   "--experiment", str(root), "--ground-usd", plan["ground_usd"],
                   "--seed", "44", "--soft-hours", str(args.soft_hours), "--launch"]
        if args.evaluation_entry:
            command += ["--evaluation-entry", args.evaluation_entry]
        write_json(state_path, {**status, "status": "launch_claimed", "command": command})
        try:
            reply = run(command, cwd=plan["repo"], capture_output=True, text=True, timeout=120)
            (root / "serial-launch.stdout.log").write_text(reply.stdout)
            (root / "serial-launch.stderr.log").write_text(reply.stderr)
            submission = strict_json(reply.stdout)
            if reply.returncode or submission.get("submitted") is not True:
                raise JobError("formal launcher did not confirm submission")
            status.update(status="submitted", plan=submission["plan"], submission=submission)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            status.update(status="launch_failed", reason=str(error), inspect_before_retry=True)
        write_json(state_path, status)
        return status


def start(args):
    commit, files = deploy.committed_files(args.commit)
    root = EXPERIMENTS / ("round4-full-" + commit)
    authorization = strict_json(files[CODE_DIRECTORY + "/" + AUTHORIZATION])
    if not args.execute:
        return {"dry_run": True, "git_commit": commit, "experiment": str(root),
                "initial_not_before": authorization["initial_not_before"],
                "action": "cancel exact old cron, deploy, launch immediately if slot elapsed, start local return watcher"}
    if args.destination is None or not args.destination.is_absolute() or not args.destination.parent.is_dir():
        raise JobError("an absolute local --destination with existing parent is required before submission")
    client = ControlSSH(args.host, args.control_path, args.ssh_port, timeout=600)
    cancellation = remote_cancel(client, authorization)
    exists = client.exec(shlex.join(["python3", "-c", f"from pathlib import Path; print(int(Path({str(root)!r}).exists()))"]))
    if exists.strip() != "1":
        with tempfile.TemporaryDirectory(prefix="round4-serial-", dir="/tmp/opencode") as directory:
            bundle = Path(directory) / "bundle.tar.gz"
            digest = deploy.make_bundle(commit, files, bundle)
            prepare = f"from pathlib import Path; p=Path({str(root)!r}); assert p.parent.is_dir(); p.mkdir()"
            client.exec(shlex.join(["python3", "-c", prepare]))
            client.upload(bundle, str(root / "bundle.tar.gz"))
            script = inspect.getsource(deploy.unpack) + f"\nimport json\nprint(json.dumps(unpack({str(root)!r}, {digest!r})))"
            client.exec(shlex.join(["python3", "-c", script]), timeout_ms=300000)
    command = ["/home/kaiser/robot-rl-sim60/env/bin/python", str(root / CODE_DIRECTORY / "scripts/round4/schedule_serial.py"),
               "tick", "--experiment", str(root), "--ground-usd", str(args.ground_usd), "--soft-hours", str(args.soft_hours), "--execute"]
    if args.evaluation_entry:
        command += ["--evaluation-entry", args.evaluation_entry]
    result = strict_json(client.exec(shlex.join(command), timeout_ms=180000))
    result["old_schedule_cancelled"] = cancellation["cancelled"]
    if result["status"] != "submitted":
        return result
    plan_path = args.destination.with_name(args.destination.name + ".plan.json")
    write_json(plan_path, result["plan"])
    wait_hours = 52 if args.soft_hours == 48 else 76
    watcher = [sys.executable, str(REPO / "scripts/round4/watch.py"), "--plan", str(plan_path),
               "--destination", str(args.destination), "--wait-hours", str(wait_hours),
               "--host", args.host, "--control-path", str(args.control_path), "--ssh-port", str(args.ssh_port)]
    log = str(args.destination) + ".watcher.log"
    session = "round4-return-" + uuid.uuid4().hex[:12]
    shell = shlex.join(["timeout", "--signal=TERM", "--kill-after=30s", str(wait_hours * 3600), *watcher])
    reply = subprocess.run(["tmux", "new-session", "-d", "-s", session, shell + " > " + shlex.quote(log) + " 2>&1"],
                           capture_output=True, text=True, timeout=15)
    result.update(local_watcher_submitted=reply.returncode == 0, local_watcher_session=session,
                  local_plan=str(plan_path), local_log=log, local_watcher_stderr=reply.stderr)
    write_json(args.destination.with_name(args.destination.name + ".submission.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cancel-old", "start", "tick"))
    parser.add_argument("--commit", help="main agent's complete committed integration")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--ground-usd", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--evaluation-entry")
    parser.add_argument("--soft-hours", type=int, choices=(48, 72), default=48)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host", default="kaiser@192.168.64.234")
    parser.add_argument("--control-path", default="/tmp/opencode/kaiser-training-control")
    parser.add_argument("--ssh-port", type=int, default=2222)
    args = parser.parse_args()
    if args.mode == "cancel-old":
        authorization = strict_json((REPO / AUTHORIZATION).read_bytes())
        result = (remote_cancel(ControlSSH(args.host, args.control_path, args.ssh_port), authorization)
                  if args.execute else {"dry_run": True, "marker": OLD_MARKER, "authorization": authorization})
    else:
        if args.ground_usd is None:
            parser.error("--ground-usd is required")
        if args.mode == "start":
            if args.commit is None:
                parser.error("--commit from the main agent is required")
            result = start(args)
        else:
            if args.experiment is None:
                parser.error("--experiment is required")
            args.seed = 44
            args.runtime = Path("/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh")
            result = tick(args)
    print(json.dumps(result, ensure_ascii=False))
    return 2 if result.get("status") in {"launch_failed", "launch_claimed"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
