"""Evaluate a trained V3.3 policy in the deploy sim2sim loop with metrics.

Usage:
  .venv_mj314/bin/python eval_v33.py <checkpoint.pt> [vx_cmd] [seconds]
Reports survival time, mean base z, mean forward speed, mean wheel target.
"""
import json
import os
import sys

import numpy as np
import mujoco
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # robot_rl/
sys.path.insert(0, ROOT + "/isaac_wheeled_rl_deploy/sim2sim")
import mujoco_sim2sim as S  # noqa: E402

SCENE = ROOT + "/isaac_wheeled_rl_deploy/models/urdf_v3.3_scene.xml"
DEFAULTS = json.load(open(ROOT + "/isaac_wheeled_rl_train/assets/urdf_v33/defaults.json"))
POSE = DEFAULTS["default_joint_pos"]
LEGS = ("L_joint1", "L_joint2", "R_joint1", "R_joint2")
WHEELS = ("L_joint3", "R_joint3")
DEFAULT_POSE = np.array([POSE[n] for n in LEGS])
VX = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
policy_path = sys.argv[1]

mj = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(mj)
d.qpos[2] = 0.48
for n in LEGS:
    d.qpos[mj.jnt_qposadr[mj.joint(n).id]] = POSE[n]
mujoco.mj_forward(mj, d)
leg_ids = [mj.joint(n).id for n in LEGS]
wheel_ids = [mj.joint(n).id for n in WHEELS]
leg_act = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEGS]
wheel_act = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in WHEELS]

sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
i_name = sess.get_inputs()[0].name
cmd = np.array([VX, 0.0, 0.0], np.float32)
last_action = np.zeros(6, np.float32)
action = np.zeros(6, np.float32)
zs, vf, wa = [], [], []
n_ticks = int(DUR * S.CTRL_HZ)
t_fall = None
for tick in range(n_ticks):
    if tick % S.STEPS_PER_POLICY == 0:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(mj, d, mujoco.mjtObj.mjOBJ_BODY, 1, vel, 1)
        ang = vel[0:3]
        R = d.xmat[1].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        leg_pos = np.array([d.qpos[mj.jnt_qposadr[j]] for j in leg_ids]) - DEFAULT_POSE
        leg_vel = np.array([d.qvel[mj.jnt_dofadr[j]] for j in leg_ids])
        wheel_vel = np.array([d.qvel[mj.jnt_dofadr[j]] for j in wheel_ids])
        obs = np.concatenate([
            cmd, [0.48 * 5.0], ang * 0.5, grav,
            leg_pos, np.zeros(2), leg_vel * 0.1, wheel_vel * 0.1,
            last_action, [1.0, 0, 0, 0, 0, 0, 0],
        ]).astype(np.float32)
        obs = np.clip(np.nan_to_num(obs), -100.0, 100.0)
        action = sess.run(None, {i_name: obs[None]})[0][0]
        last_action = action
    leg_t = np.clip(DEFAULT_POSE + 0.5 * action[:4],
                    [mj.jnt_range[j][0] for j in leg_ids],
                    [mj.jnt_range[j][1] for j in leg_ids])
    wheel_t = np.clip(10.0 * action[4:], -150.0, 150.0)
    for _ in range(2):
        for k, j in enumerate(leg_ids):
            q = d.qpos[mj.jnt_qposadr[j]]
            dq = d.qvel[mj.jnt_dofadr[j]]
            d.ctrl[leg_act[k]] = float(np.clip(60.0 * (leg_t[k] - q) - 2.0 * dq, -40, 40))
        for k, j in enumerate(wheel_ids):
            dq = d.qvel[mj.jnt_dofadr[j]]
            d.ctrl[wheel_act[k]] = float(np.clip(1.0 * (wheel_t[k] - dq), -5, 5))
        mujoco.mj_step(mj, d)
    zs.append(d.qpos[2])
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(mj, d, mujoco.mjtObj.mjOBJ_BODY, 1, vel, 1)
    vf.append(-vel[4])  # forward = -y_base
    wa.append(np.mean(wheel_t))
    if d.qpos[2] < 0.15 and t_fall is None:
        t_fall = tick * 0.002
        break
surv = DUR if t_fall is None else t_fall
print(f"policy={policy_path.split('/')[-1]} cmd_vx={VX}")
print(f"survived={surv:.2f}s/{DUR:.0f}s  mean_base_z={np.mean(zs):.3f}  "
      f"mean_v_fwd={np.mean(vf):+.3f} m/s  mean_wheel_target={np.mean(wa):+.1f} rad/s")
