"""Plot training reward + eval curves for one or more V3.3 training runs.

Usage:
    .venv_mj314/bin/python tools/plot_curves.py RUN_NAME [RUN_NAME ...]

Each RUN_NAME maps to <robot_rl>/runs_v33_<name>/ (train.log) and its periodic
checkpoints are evaluated by eval_curve.sh on the minipc. Saves a comparison
figure to docs/training_curves.png.
"""
import os, re, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # robot_rl/
DOCS = os.path.join(ROOT, "isaac_wheeled_rl_train/docs")
os.makedirs(DOCS, exist_ok=True)
COLORS = {"c": "#f5a623", "d": "#4c8bf5", "e": "#2ecc71", "f": "#9b59b6"}
runs = sys.argv[1:] or ["c"]


def parse_reward(run):
    it, mr = [], []
    log = os.path.join(ROOT, f"runs_v33_{run}", "train.log")
    text = open(log).read() if os.path.exists(log) else ""
    if not text:
        try:
            cmd = f"ssh -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null alliance@192.168.64.216 'cat ~/robot_rl/runs_v33_{run}/train.log'"
            text = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
        except Exception:
            return it, mr
    for line in text.splitlines():
        m = re.search(r"iter (\d+): mean_r=([-+0-9.eE]+)", line)
        if m:
            it.append(int(m.group(1))); mr.append(float(m.group(2)))
    return it, mr


def eval_curve(run):
    out = []
    try:
        cmd = f"ssh -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null alliance@192.168.64.216 '~/robot_rl/eval_curve.sh runs_v33_{run} 0.3 8'"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        for line in res.stdout.splitlines():
            p = line.split()
            if len(p) == 4 and p[0].isdigit():
                out.append((int(p[0]), float(p[1].split("/")[0].rstrip("s")), float(p[2]), float(p[3])))
    except Exception as e:
        print(f"eval_curve({run}) skipped: {e}")
    return sorted(out)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for run in runs:
    c = COLORS.get(run, None)
    it, mr = parse_reward(run)
    if it:
        axes[0].plot(it, mr, color=c, lw=1.3, label=f"run {run}")
    ec = eval_curve(run)
    if ec:
        ck = [x[0] for x in ec]
        axes[1].plot(ck, [x[2] for x in ec], "o-", color=c, lw=1.4, label=f"run {run}")
        axes[2].plot(ck, [x[3] for x in ec], "o-", color=c, lw=1.4, label=f"run {run}")

axes[0].set_title("training reward"); axes[0].set_xlabel("iter"); axes[0].set_ylabel("mean_r"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
axes[1].set_title("eval base height"); axes[1].set_xlabel("checkpoint iter"); axes[1].set_ylabel("base z [m]")
axes[1].axhline(0.48, color="#2ecc71", ls="--", lw=1, label="target 0.48")
axes[1].axhline(0.32, color="#e74c3c", ls="--", lw=1, label="MIN_BASE_Z 0.32")
axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
axes[2].set_title("eval forward speed"); axes[2].set_xlabel("checkpoint iter"); axes[2].set_ylabel("v_fwd [m/s]")
axes[2].axhline(0.3, color="#2ecc71", ls="--", lw=1, label="cmd 0.3")
axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

fig.suptitle("V3.3 training runs comparison", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(DOCS, "training_curves.png")
fig.savefig(out, dpi=130)
print("saved", out)
