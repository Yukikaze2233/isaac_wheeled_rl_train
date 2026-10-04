"""Plot captured physics frames; no simulation, training or smoothing."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
REPORT = json.loads((BASE / "report.json").read_text())
font_manager.fontManager.addfont("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
family = font_manager.FontProperties(fname="/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc").get_name()
plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 10})
fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2), constrained_layout=True)
styles = {"flat": ("旧平地", "#28756c"), "old_speed": ("旧高速", "#d17838"),
          "v14": ("v14（新增122次策略更新）", "#487ac0")}

for model, (label, color) in styles.items():
    def rows(case, phase=None):
        captured = [json.loads(x) for x in Path(REPORT["cases"][model][case]["frames_file"]).read_text().splitlines()]
        return [x for x in captured if phase is None or x["phase"] == phase]

    stand = rows("stand_305mm", "stand")
    t = np.array([x["time_s"] - stand[0]["time_s"] for x in stand])
    velocity = np.array([x["velocity_m_s"] for x in stand])
    position = np.array([x["position_m"][:2] for x in stand])
    anchor = np.array(stand[0]["phase_anchor_xy_m"])
    axes[0, 0].plot(t[::4], 100 * np.linalg.norm(velocity[:, :2], axis=1)[::4], color=color, label=label, lw=1.2)
    axes[0, 1].plot(t[::4], 100 * np.linalg.norm(position - anchor, axis=1)[::4], color=color, lw=1.3)
    axes[0, 2].plot(t[::4], [x["omega_rad_s"][2] for x in stand][::4], color=color, lw=1.1)
    moving = rows("forward_0p5")
    rt = np.array([x["time_s"] for x in moving])
    axes[1, 0].plot(rt[::4], [x["velocity_m_s"][0] for x in moving][::4], color=color, lw=1.1)
    axes[1, 1].plot(rt[::4], [x["pitch_deg"] for x in moving][::4], color=color, lw=1.1)
    if model == "flat":
        axes[1, 0].plot(rt[::4], [x["command"][0] for x in moving][::4], color="#252525", ls="--", lw=1.1, label="实际速度命令")
    stop = rows("forward_0p5", "stop")
    tt = np.array([x["time_s"] - stop[0]["time_s"] for x in stop])
    pp = np.array([x["position_m"][:2] for x in stop])
    aa = np.array(stop[0]["phase_anchor_xy_m"])
    axes[1, 2].plot(tt[::4], 100 * np.linalg.norm(pp - aa, axis=1)[::4], color=color, lw=1.3)

titles = ("305mm静站：平面速度", "静站：离静站锚的位移", "静站：yaw角速度",
          "前进0.5m/s与松杆", "前进0.5m/s：pitch", "松杆归零＋2s后：完整10s驻停")
for index, (ax, title, ylabel) in enumerate(zip(axes.flat, titles, ("cm/s", "cm", "rad/s", "m/s", "°", "cm"))):
    ax.set_title(title)
    ax.set_xlabel("仿真时间 / s" if index in (3, 4) else "相位时间 / s")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
axes[0, 2].axhline(0, color="#777777", ls="--", lw=.8)
axes[0, 0].legend(loc="upper right", fontsize=8)
axes[1, 0].legend(fontsize=8)
fig.suptitle("同V6资产、同初态/命令、200Hz反馈PD；确定性ONNX短诊断，非全域验收\n"
             "低速与漂移回弹；零速yaw偏差仍在，新版尚未恢复旧平地品质", fontsize=13)
fig.savefig(BASE / "quality_comparison.png", dpi=160)
