import re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOCS = "/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train/docs"
os.makedirs(DOCS, exist_ok=True)

# --- parse training reward ---
it, mr = [], []
for line in open("/tmp/runC_train.log"):
    m = re.search(r"iter (\d+): mean_r=([-+0-9.]+)", line)
    if m:
        it.append(int(m.group(1))); mr.append(float(m.group(2)))

# --- parse eval curve ---
ck, surv, bz, vf = [], [], [], []
for line in open("/tmp/curve.txt"):
    p = line.split()
    if len(p) == 4 and p[0].isdigit():
        ck.append(int(p[0])); surv.append(float(p[1].split("/")[0].rstrip("s"))); bz.append(float(p[2])); vf.append(float(p[3]))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
fig.suptitle("run C (dt0.001 + obs4/act3 + DR + noise + RV2)  —  crouch exploit visible", fontsize=13)

ax = axes[0]
ax.plot(it, mr, color="#4c8bf5", lw=1.2)
ax.set_title("training reward (mean_r)"); ax.set_xlabel("iteration"); ax.set_ylabel("mean reward"); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(ck, bz, "o-", color="#f5a623", lw=1.4)
ax.axhline(0.48, color="#2ecc71", ls="--", lw=1, label="target 0.48")
ax.axhline(0.32, color="#e74c3c", ls="--", lw=1, label="MIN_BASE_Z 0.32")
ax.set_title("eval base height (z)"); ax.set_xlabel("checkpoint iter"); ax.set_ylabel("base z [m]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(ck, vf, "o-", color="#9b59b6", lw=1.4)
ax.axhline(0.3, color="#2ecc71", ls="--", lw=1, label="cmd 0.3")
ax.set_title("eval forward speed (v_fwd)"); ax.set_xlabel("checkpoint iter"); ax.set_ylabel("v_fwd [m/s]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(DOCS, "runC_curves.png")
fig.savefig(out, dpi=130)
print("saved", out)
print("iters:", len(it), "ckpts:", len(ck))
print("base_z min/mean/max:", round(min(bz),3), round(np.mean(bz),3), round(max(bz),3))
print("v_fwd min/mean/max:", round(min(vf),3), round(np.mean(vf),3), round(max(vf),3))
