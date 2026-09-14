#!/usr/bin/env python3
"""Start a separate, ten-minute native R3A repaired-visual replay in tmux."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    artifact = workspace / "reports/current/kaiser_round3_a_20260914/final-return/artifacts"
    model = workspace / "model/纯底盘"
    sys.path.insert(0, str(repo / "src"))
    from wheeled_tasks.v40.repaired_visuals import POLICY_SHA256, RepairedKinematics
    RepairedKinematics(model)
    if hashlib.sha256((artifact / "policy.onnx").read_bytes()).hexdigest() != POLICY_SHA256:
        raise ValueError("R3A policy SHA256 mismatch")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = "r3a-repaired-" + stamp
    report = repo / "reports" / name
    if not report.parent.is_dir():
        raise FileNotFoundError(report.parent)
    ground = workspace / ("reports/current/kaiser_round3_a_20260914/"
                          "evaluation-recovery-20260914T070110Z/recovered/default_environment.usd")
    command = [str(Path.home() / "isaacsim60-venv/bin/python"), "-u", str(repo / "scripts/play_v40_onnx.py"),
               "--onnx", str(artifact / "policy.onnx"), "--contract", str(artifact / "contract.json"),
               "--report-dir", str(report), "--ground-usd", str(ground), "--repaired-visuals", str(model),
               "--research", "--device", "cpu", "--num-envs", "1", "--keyboard", "--visible-grid",
               "--render-interval", "4", "--command", "0", "0", ".32", "--linear-speed", "2",
               "--angular-speed", "2", "--max-steps", "60000", "--max-wall-seconds", "600"]
    environment = dict(DISPLAY=":1", WAYLAND_DISPLAY="wayland-1",
                       TMPDIR=str(Path.home() / ".cache/kit-tmp"),
                       LD_LIBRARY_PATH=str(Path.home() / ".local/lib/compat"),
                       OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PYTHONUNBUFFERED="1")
    shell = shlex.join(["env", *(f"{key}={value}" for key, value in environment.items()),
                        "timeout", "--signal=INT", "--kill-after=20s", "600s", *command])
    shell += " > " + shlex.quote(str(report) + ".log") + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", name, "-c", str(repo), shell], check=True)
    print(f"session={name}\nreport={report}\nlog={report}.log", flush=True)


if __name__ == "__main__":
    main()
