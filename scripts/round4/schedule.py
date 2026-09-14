#!/usr/bin/env python3
"""Cron-driven, not-before readiness check for one repaired-V40 training launch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


OLD_ASSET = "df5ca7693022b7a4c68cb1dc823482291f265e3addc8c4870fb3b0b2ab364886"
CHECKS = ("kinematic_closure", "dynamic_constraints", "mass_properties",
          "actuator_mapping", "height_domain")
TERMINAL = {"launch_claimed", "submitted", "launch_failed", "expired"}


def load(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def stamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("schedule timestamps require an explicit timezone")
    return result


def request_data(path):
    request = load(path)
    if request.get("schema_version") != 1 or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", request["id"]):
        raise ValueError("invalid schedule identity")
    start = stamp(request["not_before"])
    if request["expires_at"] is not None:
        end = stamp(request["expires_at"])
        if not 0 < (end - start).total_seconds() <= 7 * 86400:
            raise ValueError("a bounded readiness wait must be positive and at most seven days")
    if request.get("require_repaired_dynamics") is not True:
        raise ValueError("this schedule requires repaired dynamics")
    if not re.fullmatch(r"[0-9a-f]{64}", request.get("required_geometry_source_sha256", "")):
        raise ValueError("the confirmed geometry source must be pinned by SHA256")
    return request


def inside(root, value):
    root, path = Path(root).resolve(), Path(value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"file must exist inside its declared root: {value}")
    return path


def checked_file(root, value, expected):
    path = inside(root, value)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"file SHA256 mismatch: {path.name}")
    return path


def ready_command(request, ready_path):
    """Require a hash-bound dynamics report and an independently checked deployment."""
    ready = load(ready_path)
    base = Path(request["deployment_root"]).resolve()
    commit = ready["git_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("ready deployment requires a full Git commit")
    experiment = (base / "experiments" / ("round4-full-" + commit)).resolve()
    snapshot_path = checked_file(experiment, experiment / "snapshot.json", ready["snapshot_sha256"])
    snapshot = load(snapshot_path)
    if snapshot["git_commit"] != commit:
        raise ValueError("snapshot commit mismatch")
    repo = experiment / "isaac_wheeled_rl_train"
    for name, record in snapshot["files"].items():
        checked_file(experiment, experiment / name, record["sha256"])
    validation_path = checked_file(experiment, ready["validation_path"], ready["validation_sha256"])
    validation = load(validation_path)
    if (validation.get("scope") != "v40_repaired_dynamics" or validation.get("passed") is not True
            or any(validation.get("checks", {}).get(key) is not True for key in CHECKS)):
        raise ValueError("repaired dynamics validation is incomplete; geometry preview is insufficient")
    geometry_hash = request["required_geometry_source_sha256"]
    checked_file(base, ready["geometry_source_path"], geometry_hash)
    if validation.get("geometry_source_sha256") != geometry_hash:
        raise ValueError("dynamics report is not bound to the user-confirmed chassis geometry")
    contract_path = inside(repo, repo / snapshot["contract"])
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != validation["contract_file_sha256"]:
        raise ValueError("dynamics report belongs to a different contract")
    contract = load(contract_path)
    asset = contract["asset"]
    manifest_path = inside(repo, repo / asset["directory"] / asset["manifest"])
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest == OLD_ASSET or digest != validation["asset_manifest_sha256"]:
        raise ValueError("old or unvalidated asset is forbidden for this schedule")
    launcher = inside(repo, repo / "scripts/round4/launch.py")
    if "isaac_wheeled_rl_train/scripts/round4/launch.py" not in snapshot["files"]:
        raise ValueError("launcher is not bound to the deployment snapshot")
    python = inside(base, base / "env/bin/python")
    ground = checked_file(base, ready["ground_usd"], ready["ground_sha256"])
    return [str(python), str(launcher), "--experiment", str(experiment),
            "--ground-usd", str(ground), "--seed", "44", "--soft-hours", "48"], repo, commit


def tick(request_path, *, now=None, run=subprocess.run):
    path = Path(request_path).resolve()
    request = request_data(path)
    now = now or datetime.now(timezone.utc)
    state_path = path.parent / "state.json"
    with (path.parent / "schedule.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "another_check_in_progress"}
        previous = load(state_path) if state_path.exists() else {}
        if previous.get("status") in TERMINAL:
            return previous

        def save(status, **details):
            state = {"schema_version": 1, "id": request["id"], "status": status,
                     "checked_at": now.isoformat(), "not_before": request["not_before"],
                     "training_success": False, **details}
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
            temporary.replace(state_path)
            return state

        if now < stamp(request["not_before"]):
            return save("scheduled_waiting_time")
        if request["expires_at"] is not None and now >= stamp(request["expires_at"]):
            return save("expired", reason="repaired deployment was not launched within readiness window")
        ready_path = path.parent / "ready.json"
        if not ready_path.is_file():
            return save("waiting_repaired_asset", missing=list(CHECKS),
                        reason="no verified repaired deployment has been registered")
        try:
            command, repo, commit = ready_command(request, ready_path)
            dry = run(command, cwd=repo, capture_output=True, text=True, timeout=90)
            if dry.returncode:
                raise ValueError(f"deployment preflight exited {dry.returncode}: {dry.stderr[-2000:]}")
            answer = json.loads(dry.stdout)
            plan = answer["plan"]
            if (answer.get("dry_run") is not True or plan.get("initialization") != "scratch"
                    or plan.get("parent") is not None or plan.get("num_envs") != 1024
                    or plan.get("updates") != 30000 or plan.get("git_commit") != commit):
                raise ValueError("launcher plan is not the requested fresh 1024-env/30000-update run")
            if any(flag in plan["command"] for flag in ("--resume", "--warm-start", "--finetune", "--stage-transfer")):
                raise ValueError("scheduled initial training cannot load old weights")
        except (KeyError, ValueError, OSError, subprocess.SubprocessError) as error:
            return save("blocked_readiness", reason=str(error))

        # Persist before spawning: an interrupted submission requires inspection,
        # never an automatic second launch on the next cron tick.
        save("launch_claimed", command=command + ["--launch"], git_commit=commit)
        try:
            result = run(command + ["--launch"], cwd=repo, capture_output=True, text=True, timeout=90)
            (path.parent / "launch.stdout.log").write_text(result.stdout)
            (path.parent / "launch.stderr.log").write_text(result.stderr)
            submitted = json.loads(result.stdout)
            if result.returncode or submitted.get("submitted") is not True:
                return save("launch_failed", exit_code=result.returncode, inspect_before_retry=True)
            return save("submitted", submission=submitted, git_commit=commit)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            return save("launch_failed", reason=str(error), inspect_before_retry=True)


def install(request_path, *, run=subprocess.run):
    path = Path(request_path).resolve()
    request = request_data(path)
    script = Path(__file__).resolve()
    command = shlex.join(["/usr/bin/python3", str(script), "tick", "--request", str(path)])
    if "%" in command:
        raise ValueError("cron paths cannot contain percent characters")
    marker = "# robot-rl-schedule:" + request["id"]
    line = "* * * * * " + command + " >> " + shlex.quote(str(path.parent / "cron.log")) + " 2>&1 " + marker
    environment = {**os.environ, "LC_ALL": "C"}
    current = run(["crontab", "-l"], capture_output=True, text=True, env=environment)
    if current.returncode and not (current.returncode == 1 and "no crontab" in current.stderr.lower()):
        raise RuntimeError("cannot read current user crontab")
    old = current.stdout if current.returncode == 0 else ""
    matches = [entry for entry in old.splitlines() if entry.rstrip().endswith(marker)]
    if matches and matches != [line]:
        raise ValueError("a different cron entry already owns this schedule id")
    if not matches:
        latest = run(["crontab", "-l"], capture_output=True, text=True, env=environment)
        if (latest.returncode, latest.stdout, latest.stderr) != (current.returncode, current.stdout, current.stderr):
            raise RuntimeError("crontab changed concurrently; retry installation")
        text = old + ("\n" if old and not old.endswith("\n") else "") + line + "\n"
        run(["crontab", "-"], input=text, text=True, capture_output=True, check=True, env=environment)
    return {"installed": True, "id": request["id"], "cron_entry": line,
            "not_before": request["not_before"], "expires_at": request["expires_at"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tick", "install"))
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    result = install(args.request) if args.mode == "install" else tick(args.request)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
