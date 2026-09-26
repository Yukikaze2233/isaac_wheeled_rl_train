#!/usr/bin/env python3
"""V40 checkpoint playback in MuJoCo (contract-parity, correct visuals).

Same control pipeline as training (core.py). Physics = inspection.xml
(与部署 sim2sim 同源的 MuJoCo 模型)。Viewer 可鼠标旋转/缩放；力矩打印到终端。

Examples:
  python scripts/play_v40_mujoco.py --onnx                    # ONNX 推理 + viewer
  python scripts/play_v40_mujoco.py --headless --seconds 10   # 定量自检
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = Path(__file__).resolve().parents[1] / "runs_v40"
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
from wheeled_tasks.v40.contract import load_contract
from wheeled_tasks.v40.core import (
    HistoryStack,
    build_observation,
    compute_torques,
    decode_targets,
)

import mujoco
import mujoco.viewer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(RUNS / "stand2000/model_final.pt"))
parser.add_argument("--contract", default=str(REPO / "contracts/own_v40_v1.json"))
parser.add_argument("--mjcf", default=str(REPO / "assets/urdf_v40/inspection.xml"))
parser.add_argument("--stage", default="stand", choices=["stand", "height", "locomotion"])
parser.add_argument("--onnx", action="store_true", help="inference via policy.onnx (onnxruntime)")
parser.add_argument("--headless", action="store_true", help="no viewer; print metrics only")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--realtime", action="store_true")
parser.add_argument("--push", action="store_true", help="random horizontal pushes every 3 s")
args = parser.parse_args()

contract = load_contract(args.contract)
names = list(contract["joints"]["action_order"])
actuators = contract["actuators"]
leg_ids = set(contract["joints"]["leg_indices"])
limits = [actuators["leg" if i in leg_ids else "wheel"]["effort_limit"] for i in range(6)]

# ---------------- policy ----------------
onnx_sess = None
actor = None
if args.onnx:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(Path(args.checkpoint).parent / "policy.onnx"),
                                providers=["CPUExecutionProvider"])
    onnx_in = sess.get_inputs()[0].name
    print(f"[play] ONNX 推理模式: {Path(args.checkpoint).parent.name}/policy.onnx")
else:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"]
    actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    layers = []
    prev = contract["observations"]["actor_dim"]
    for h in contract["policy"]["actor_hidden_dims"]:
        layers += [torch.nn.Linear(prev, h), torch.nn.ELU()]
        prev = h
    layers += [torch.nn.Linear(prev, contract["actions"]["dimension"])]
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict(actor_sd, strict=True)
    actor.eval()
    print("[play] torch checkpoint 推理")

# ---------------- mujoco ----------------
model = mujoco.MjModel.from_xml_path(args.mjcf)
model.opt.timestep = contract["timing"]["physics_dt"]
data = mujoco.MjData(model)

jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
jadr = {n: model.jnt_qposadr[i] for i, n in enumerate(jnames)}
dadr = {n: model.jnt_dofadr[i] for i, n in enumerate(jnames)}
SIX = ["L_joint1", "L_joint2", "L_joint3", "R_joint1", "R_jonit2", "R_joint3"]
leg_set = set(contract["joints"]["leg_indices"])
for c, n in enumerate(SIX):
    model.dof_armature[dadr[n]] = actuators["leg" if c in leg_ids else "wheel"]["armature"]

mujoco.mj_resetData(model, data)
data.qpos[2] = contract["asset"]["nominal_base_height"]
data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
for c, n in enumerate(SIX):
    data.qpos[jadr[n]] = contract["joints"]["nominal_positions"][c]
mujoco.mj_forward(model, data)

nominal = torch.tensor(contract["joints"]["nominal_positions"], dtype=torch.float32).unsqueeze(0)
stage = contract["commands"]["stages"][args.stage]
commands = torch.tensor([[stage["vx"][0], stage["wz"][0], stage["height"][0]]], dtype=torch.float32)
history = HistoryStack(1, "cpu", contract["observations"]["history_length"], contract["observations"]["single_dim"])
last_actions = torch.zeros(1, 6)
leg_targets = nominal[:, contract["joints"]["leg_indices"]]
wheel_targets = torch.zeros(1, 2)

GRAVITY_W = torch.tensor([0.0, 0.0, -1.0])
min_height = contract["termination"]["min_base_height"]
total_ticks = int(args.seconds / policy_dt)

viewer = None
if not args.headless:
    viewer = mujoco.viewer.launch_passive(model, data)

fell_at = None
heights, speeds = [], []
tick = 0


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


try:
    while tick < total_ticks:
        q = np.array([[data.qpos[jadr[n]] for n in SIX]], dtype=np.float64)
        dq = np.array([[data.qvel[dadr[n]] for n in SIX]], dtype=np.float64)
        quat = data.qpos[3:7].copy()
        R = quat_to_rot(quat)
        gravity_b = torch.tensor((R.T @ np.array([0.0, 0.0, -1.0])), dtype=torch.float32).unsqueeze(0)
        ang_vel_b = torch.tensor([data.qvel[3:6]], dtype=torch.float32)
        height = float(data.qpos[2])

        obs25 = build_observation(ang_vel_b, gravity_b, commands,
                                  torch.tensor(q, dtype=torch.float32),
                                  torch.tensor(dq, dtype=torch.float32),
                                  last_actions, contract)
        policy_obs = history.update(obs25, tick)
        if onnx_sess is not None:
            actions_np = sess.run(None, {onnx_in: policy_obs.cpu().numpy()})[0]
            actions = torch.tensor(actions_np, dtype=torch.float32)
        else:
            with torch.no_grad():
                actions = actor(policy_obs)
        leg_targets, wheel_targets, clipped = decode_targets(actions, torch.tensor(q, dtype=torch.float32), contract)
        last_actions = clipped

        torques = torch.zeros(1, 6)
        for _ in range(decimation):
            qn = np.array([[data.qpos[jadr[n]] for n in SIX]], dtype=np.float64)
            dqn = np.array([[data.qvel[dadr[n]] for n in SIX]], dtype=np.float64)
            torques = compute_torques(torch.tensor(qn, dtype=torch.float32),
                                      torch.tensor(dqn, dtype=torch.float32),
                                      leg_targets, wheel_targets, contract)
            data.qfrc_applied[:] = 0.0
            for c, n in enumerate(SIX):
                data.qfrc_applied[dadr[n]] = float(torques[0, c])
            mujoco.mj_step(model, data)
        last_torques = torques

        height = float(data.qpos[2])
        speeds.append(float(np.linalg.norm(data.qvel[0:2])))
        heights.append(height)
        g_b = R.T @ np.array([0.0, 0.0, -1.0])
        if fell_at is None and (height < min_height or -float(g_b[2]) < 0.8192):
            fell_at = tick * policy_dt
            print(f"[play] TERMINATED at t={fell_at:.2f}s: height={height:.3f}")

        if args.push and tick % 300 == 150:
            ang = float(np.random.uniform(0, 2 * np.pi))
            dv = float(np.random.uniform(0.15, 0.35))
            data.qvel[0] += dv * np.cos(ang)
            data.qvel[1] += dv * np.sin(ang)
            print(f"[push] t={tick * policy_dt:.1f}s 强度={dv:.2f} m/s", flush=True)

        if viewer is not None:
            viewer.sync()
        if args.realtime:
            time.sleep(policy_dt)
        tick += 1
except KeyboardInterrupt:
    pass
finally:
    if viewer is not None:
        viewer.close()

print(f"[play] done: {args.seconds}s ({args.stage}), fell_at={fell_at}")
print(f"[play] height mean={np.mean(heights):.4f} min={np.min(heights):.4f} | speed mean={np.mean(speeds):.4f}")
