#!/usr/bin/env python3
"""Build an audited V40 tmux launch plan; remote execution requires --launch.

Only the user-approved narrow research model is authorized; all gates still apply.
Uses the managed HTTP API, a unique session, and an explicit ENV/bin/python.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_v40_artifacts import ManagedSSH, absolute_remote_path, management_url
from wheeled_algo.v40_job import JobError, PlannedStop, TrainingBudget, parse_stop_at, positive_seconds, strict_json

# The detached worker writes only to its separate audit directory. No run mkdir,
# no shell expansion, no source/activate, no tmux server restart or session reuse.
REMOTE_LAUNCH = r'''
import json, os, pathlib, subprocess
p = PARAMS
repo, audit, run = (pathlib.Path(p[k]) for k in ("repo", "audit_dir", "run_dir"))
if os.path.lexists(run):
    raise ValueError("run directory already exists")
if not repo.is_dir() or not pathlib.Path(p["python"]).is_file():
    raise ValueError("explicit repository/interpreter missing")
if os.path.lexists(audit):
    raise ValueError("audit directory already exists")
audit.mkdir(mode=0o700)  # Never create train's run directory (or its parents).
env = dict(os.environ)
env.pop("PYTHONHOME", None)
env.pop("PYTHONPATH", None)
env["PATH"] = str(pathlib.Path(p["python"]).parent) + ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
env.update(ENABLE_CAMERAS="0", LIVESTREAM="0", PYTHONUNBUFFERED="1")
command = [p["python"], str(repo / "scripts/train_v40.py"), "--run-dir", str(run), *p["train_args"]]
preflight = subprocess.run([*command, "--preflight-only"], cwd=repo, env=env,
                           capture_output=True, text=True, timeout=120)
(audit / "preflight.stdout.log").write_text(preflight.stdout)
(audit / "preflight.stderr.log").write_text(preflight.stderr)
try:
    report = json.loads(preflight.stdout)
except ValueError:
    report = {}
if preflight.returncode != 0 or report.get("ready") is not True or report.get("simulation_started") is not False:
    result = {"launched": False, "reason": "preflight_rejected", "preflight_exit_code": preflight.returncode}
else:
    worker = """import json, os, pathlib, subprocess, time
p = WORKER_PARAMS
audit = pathlib.Path(p['audit_dir'])
env = dict(os.environ)
env.pop('PYTHONHOME', None)
env.pop('PYTHONPATH', None)
env['PATH'] = str(pathlib.Path(p['python']).parent) + ':/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
env.update(ENABLE_CAMERAS='0', LIVESTREAM='0', PYTHONUNBUFFERED='1')
started = time.time()
status = {'session': p['session'], 'training_success': False, 'authority': 'inspect completion.json and artifact hashes, not tmux disappearance'}
try:
    with (audit / 'train.log').open('x') as log:
        result = subprocess.run(p['command'], cwd=p['repo'], env=env, stdout=log, stderr=subprocess.STDOUT)
    status['exit_code'] = result.returncode
except BaseException as e:
    status['error'] = type(e).__name__ + ': ' + str(e)
finally:
    status['elapsed_seconds'] = time.time() - started
    temporary = audit / 'exit.json.partial'
    with temporary.open('x') as f:
        json.dump(status, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.link(temporary, audit / 'exit.json')
    temporary.unlink()
""".replace("WORKER_PARAMS", repr({**p, "command": command}), 1)
    # The existing tmux server may have stale env even when its client is clean.
    tmux_env_args = [arg for key in ("PYTHONHOME", "PYTHONPATH", "PATH", "ENABLE_CAMERAS", "LIVESTREAM", "PYTHONUNBUFFERED", "OMNI_KIT_ACCEPT_EULA")
                     for arg in ("-e", key + "=" + env.get(key, ""))]
    tmux = subprocess.run([p["tmux"], "new-session", "-d", "-s", p["session"], "-c", str(repo),
                           *tmux_env_args, p["python"], "-c", worker], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=15)
    result = {"launched": tmux.returncode == 0, "session": p["session"],
              "tmux_exit_code": tmux.returncode, "stderr": tmux.stderr,
              "training_success": False, "completion_required": True}
(audit / "launch.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
'''


def launch_plan(*, python, repo, run_dir, audit_dir, stop_at, max_runtime_seconds=None,
                research=False, stage="stand", num_envs=256, max_iterations=1000,
                seed=40, device="cuda:0", contract=None, resume=None, finetune=None,
                tmux="/usr/bin/tmux"):
    for value in (python, repo, run_dir, audit_dir, tmux):
        absolute_remote_path(value)
    interpreter = PurePosixPath(python)
    if interpreter.parent.name != "bin" or interpreter.name not in {"python", "python3", "python3.11"}:
        raise JobError("python must be explicit absolute ENV/bin/python (or python3/python3.11)")
    if PurePosixPath(run_dir).is_relative_to(audit_dir) or PurePosixPath(audit_dir).is_relative_to(run_dir):
        raise JobError("audit and run directories must be separate, not ancestors of one another")
    if resume and finetune:
        raise JobError("resume and finetune are mutually exclusive")
    if type(max_iterations) is not int or max_iterations < 1 or type(num_envs) is not int or num_envs < 1:
        raise JobError("iteration/environment counts must be positive integers")
    deadline = parse_stop_at(stop_at) if isinstance(stop_at, str) else stop_at
    if deadline is None:
        raise JobError("explicit training stop-at required; obtain actual platform shutdown time first")
    TrainingBudget(max_runtime_seconds, deadline).check()
    args = ["--headless", "--stop-at", deadline.isoformat(), "--stage", stage,
            "--num-envs", str(num_envs), "--max-iterations", str(max_iterations),
            "--seed", str(seed), "--device", device]
    if research:
        args.append("--research")
    if max_runtime_seconds is not None:
        args += ["--max-runtime-seconds", str(positive_seconds(max_runtime_seconds))]
    for key, value in (("--contract", contract), ("--resume", resume), ("--finetune", finetune)):
        if value:
            args += [key, absolute_remote_path(value)]
    session = "v40-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
    params = {"python": python, "repo": repo, "run_dir": run_dir, "audit_dir": audit_dir,
              "tmux": tmux, "session": session, "train_args": args}
    script = REMOTE_LAUNCH.replace("PARAMS", repr(params), 1)
    return params, shlex.join([python, "-c", script])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--python", required=True, help="Absolute target ENV/bin/python, never PATH python")
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--audit-dir", required=True, help="NEW independent directory with existing parent")
    parser.add_argument("--stop-at", required=True, help="Training cutoff WITH timezone, >=30min before actual shutdown")
    parser.add_argument("--max-runtime-seconds", type=positive_seconds)
    parser.add_argument("--tmux", default="/usr/bin/tmux")
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--stage", choices=("stand", "height", "locomotion"), default="stand")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contract")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume")
    source.add_argument("--finetune")
    parser.add_argument("--management-url", default="http://127.0.0.1:3080", type=management_url)
    parser.add_argument("--launch", action="store_true", help="Explicit opt-in to managed remote preflight and tmux launch; absent means NO network")
    args = parser.parse_args(argv)
    try:
        params, command = launch_plan(python=args.python, repo=args.remote_repo, run_dir=args.remote_run_dir,
            audit_dir=args.audit_dir, stop_at=args.stop_at, max_runtime_seconds=args.max_runtime_seconds,
            tmux=args.tmux, research=args.research, stage=args.stage, num_envs=args.num_envs,
            max_iterations=args.max_iterations, seed=args.seed, device=args.device,
            contract=args.contract, resume=args.resume, finetune=args.finetune)
        if not args.launch:
            print(json.dumps({"launched": False, "dry_run": True, "plan": params,
                  "warning": "Approved research scope and all asset/runtime gates remain required; supply actual shutdown time and reserve >=30min for finalization/transfer"}))
            return 0
        result = strict_json(ManagedSSH(args.alias, args.management_url, timeout=180).exec(command, timeout_ms=150000))
        print(json.dumps(result))
        return 0 if result.get("launched") is True else 2
    except (JobError, PlannedStop, ValueError, RuntimeError) as exc:
        print(f"NOT confirmed launched: {exc}; do not retry blindly; inspect audit and completion", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
