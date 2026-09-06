"""MuJoCo PPO training for the urdf_V3.3 serial-leg wheeled biped (CPU).

Runs the same 35D->6D policy contract as the deploy repo (CONTRACT.md):
  obs  [cmd3, height_cmd*5, ang_vel*0.5, gravity, leg_pos4, wheel_pos2(0),
        leg_vel*0.1, wheel_vel*0.1, prev_action6, flags7]
  act  [L_hip, L_knee, R_hip, R_knee, L_wheel, R_wheel]
       legs: q_target = default + 0.5*a (clamped to joint range), PD 60/2, +-40 Nm
       wheels: w_target = 10*a clamped +-150 rad/s, servo 0.2, +-5 Nm

Physics: MuJoCo dt=0.002 (500 Hz control), policy at 50 Hz, floor plane.
Frame convention: export frame kept (forward = -y_base); v_fwd = -lin_vel_b[1].
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# Works both when run as a script and when imported under python -c
# (the latter has no __file__ global, so derive it from the module object).
_HERE = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))  # tools/
_REPO = os.path.dirname(_HERE)  # isaac_wheeled_rl_train/
MODEL_XML = os.path.join(_REPO, "assets/urdf_v33/urdf_V3.3_rl.xml")
DEFAULTS_JSON = os.path.join(_REPO, "assets/urdf_v33/defaults.json")
LOG_DIR = os.environ.get("V33_LOG_DIR", os.path.join(_REPO, "runs_v33"))

N_ENVS = int(os.environ.get("V33_N_ENVS", "48"))
N_THREADS = int(os.environ.get("V33_N_THREADS", "6"))
STEPS_PER_EPISODE = 300   # 6 s at 50 Hz
ROLLOUT_STEPS = 24
MAX_ITERS = int(os.environ.get("V33_MAX_ITERS", "1000"))
BASE_HEIGHT = 0.48
HEIGHT_RANGE = (0.45, 0.50)  # narrow: policy must learn to stand TALL
MIN_BASE_Z = 0.32            # crouch below this = terminated (kills the crouch basin)
RAISING_FLOOR = os.environ.get("V33_RAISING_FLOOR", "0") == "1"  # run-5 curriculum
ITERATION = [0]  # shared curriculum counter


def current_min_base_z() -> float:
    """Run-5 raising-floor curriculum: learn balance in the easy crouch basin
    first, then force the policy taller as training progresses."""
    if not RAISING_FLOOR:
        return MIN_BASE_Z
    it = ITERATION[0]
    if it < 300:
        return 0.20
    if it < 500:
        return 0.28
    if it < 700:
        return 0.36
    return 0.42

LEGS = ("L_joint1", "L_joint2", "R_joint1", "R_joint2")
WHEELS = ("L_joint3", "R_joint3")
LEG_KP, LEG_KD, LEG_TORQUE = 60.0, 2.0, 40.0
WHEEL_KV = float(os.environ.get("V33_WHEEL_KV", "1.0"))  # velocity servo gain
WHEEL_TORQUE = 5.0
LEG_SCALE, WHEEL_SCALE, MAX_WHEEL_VEL = 0.5, 10.0, 150.0
SIGMA_V, SIGMA_H = 0.3, 0.03
SIGMA_WZ = 0.25

# Deploy-parity pipeline delays (mirror mujoco_sim2sim.py deque semantics):
# OBS_DELAY=4 -> steady-state 3 policy steps (60 ms); ACT_DELAY=3 -> exactly
# 3 policy steps (60 ms). The ROS2 controller uses the same obs 4 / act 3.
OBS_DELAY = int(os.environ.get("V33_OBS_DELAY", "0"))
ACT_DELAY = int(os.environ.get("V33_ACT_DELAY", "0"))
DR = os.environ.get("V33_DR", "0") == "1"       # domain randomization + obs noise
RV2 = os.environ.get("V33_RV2", "0") == "1"     # height-gated speed + stand-still
YAW = os.environ.get("V33_YAW", "0") == "1"     # yaw-rate tracking (turning)

# ---------------------------------------------------------------- model setup
def build_model(rng=None):
    raw = open(MODEL_XML).read()
    # from_xml_string resolves relative mesh paths against the CWD, not the
    # XML location; pin meshdir so the model loads from any directory.
    mesh_dir = os.path.dirname(MODEL_XML)
    raw = raw.replace("<compiler", f'<compiler meshdir="{mesh_dir}"', 1)
    floor = ('  <worldbody>\n'
             '    <geom name="floor" size="0 0 0.05" type="plane" friction="1 0.01 0.01"/>\n'
             '  </worldbody>\n')
    if "floor" not in raw:
        raw = raw.replace("</mujoco>", floor + "</mujoco>")
    mj = mujoco.MjModel.from_xml_string(raw)
    # Physics parity with the deploy sim2sim scene: timestep 0.001 and two
    # substeps per 2 ms control tick. Derive substeps from the model so a
    # timestep change can never silently desync the two sims.
    substeps = max(1, int(round(1.0 / 50.0 / mj.opt.timestep)))
    defaults = json.load(open(DEFAULTS_JSON))["default_joint_pos"]
    leg_ids = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEGS]
    wheel_ids = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n) for n in WHEELS]
    leg_act = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEGS]
    wheel_act = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in WHEELS]
    base_bid = 1  # free-joint root body
    leg_qadr = [mj.jnt_qposadr[i] for i in leg_ids]
    leg_dadr = [mj.jnt_dofadr[i] for i in leg_ids]
    wheel_dadr = [mj.jnt_dofadr[i] for i in wheel_ids]
    leg_range = np.array([mj.jnt_range[i] for i in leg_ids])
    default_pose = np.array([defaults[n] for n in LEGS])
    gs = ts = 1.0  # leg gain / torque-limit scales (per-env under DR)
    kv = WHEEL_KV
    if rng is not None:
        # --- SCUT-style domain randomization (per-env model copy) ---
        mj.body_mass[base_bid] *= rng.uniform(0.9, 1.3)          # base mass
        for b in range(2, mj.nbody):
            mj.body_mass[b] *= rng.uniform(0.9, 1.1)             # leg/wheel links
        mj.body_ipos[base_bid, 0] += rng.uniform(-0.04, 0.04)    # base COM
        mj.body_ipos[base_bid, 1] += rng.uniform(-0.02, 0.02)
        mj.body_ipos[base_bid, 2] += rng.uniform(-0.02, 0.02)
        wheel_bodies = {mj.jnt_bodyid[i] for i in wheel_ids}
        for g in range(mj.ngeom):
            if mj.geom_bodyid[g] in wheel_bodies:
                mj.geom_friction[g, 0] *= rng.uniform(0.4, 1.0)  # wheel grip
        for dadr in leg_dadr:
            mj.dof_frictionloss[dadr] += rng.uniform(0.05, 0.2)  # joint friction
        gs = rng.uniform(0.75, 1.25)      # leg PD gain scale
        kv = WHEEL_KV * rng.uniform(0.9, 1.1)
        ts = rng.uniform(0.8, 1.1)        # leg torque-limit scale
    return mj, dict(leg_ids=leg_ids, wheel_ids=wheel_ids, leg_act=leg_act,
                    wheel_act=wheel_act, base_bid=base_bid, leg_qadr=leg_qadr,
                    leg_dadr=leg_dadr, wheel_dadr=wheel_dadr, leg_range=leg_range,
                    default_pose=default_pose, substeps=substeps, gs=gs, ts=ts, kv=kv)


def euler_to_quat(roll, pitch, yaw):
    """XYZ intrinsic -> MuJoCo quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


class VecEnv:
    def __init__(self):
        self.rng = np.random.default_rng(0)
        self._pool = ThreadPoolExecutor(max_workers=N_THREADS)  # reused across steps
        self.mjs, self.cfgs, self.datas = [], [], []
        for i in range(N_ENVS):
            rng_i = np.random.default_rng(1000 + i) if DR else None
            mj, cfg = build_model(rng_i)
            self.mjs.append(mj)
            self.cfgs.append(cfg)
            self.datas.append(mujoco.MjData(mj))
        self.rngs = [np.random.default_rng(2000 + i) for i in range(N_ENVS)]
        self.obs_delay, self.act_delay = OBS_DELAY, ACT_DELAY
        self._rebuild_rings()
        self.cmds = np.zeros((N_ENVS, 3))
        self.h_cmds = np.full(N_ENVS, BASE_HEIGHT)
        self.prev_actions = np.zeros((N_ENVS, 6))
        self.step_count = np.zeros(N_ENVS, dtype=np.int64)
        self.next_push = np.full(N_ENVS, 5.0)
        self.reset_all()

    def _rebuild_rings(self):
        # Mirrors mujoco_sim2sim.py: obs deque(maxlen=obs_delay) returning buf[0]
        # (steady-state delay = obs_delay-1 steps); action deque(maxlen=act_delay+1)
        # prefilled with zeros (delay = exactly act_delay steps).
        self.obs_ring = [[np.zeros(35, np.float32) for _ in range(max(1, self.obs_delay))]
                         for _ in range(N_ENVS)]
        self.act_ring = [[np.zeros(6, np.float32) for _ in range(max(1, self.act_delay + 1))]
                         for _ in range(N_ENVS)]

    def set_delays(self, od, ad):
        self.obs_delay, self.act_delay = od, ad
        self._rebuild_rings()

    def reset_all(self):
        for i, d in enumerate(self.datas):
            self._reset_one(i, d)

    def _reset_one(self, i, d):
        cfg = self.cfgs[i]
        ri = self.rngs[i]
        yaw = self.rng.uniform(-0.5, 0.5)
        roll = pitch = 0.0
        if DR:
            roll = ri.uniform(-0.15, 0.15)
            pitch = ri.uniform(-0.15, 0.15)
        d.qpos[0:2] = self.rng.uniform(-0.1, 0.1, 2)
        d.qpos[2] = BASE_HEIGHT + (ri.uniform(-0.02, 0.02) if DR else 0.0)
        d.qpos[3:7] = euler_to_quat(roll, pitch, yaw)
        d.qvel[:] = 0
        spread = 0.05 if DR else 0.02
        for k, qadr in enumerate(cfg["leg_qadr"]):
            d.qpos[qadr] = cfg["default_pose"][k] + self.rng.uniform(-spread, spread)
        if DR:
            for dadr in cfg["leg_dadr"]:
                d.qvel[dadr] = ri.uniform(-0.5, 0.5)
            for dadr in cfg["wheel_dadr"]:
                d.qvel[dadr] = ri.uniform(-5.0, 5.0)
        mujoco.mj_forward(self.mjs[i], d)
        for ring in (self.obs_ring[i], self.act_ring[i]):
            for arr in ring:
                arr[:] = 0.0
        self.next_push[i] = 5.0 + ri.uniform(0.0, 5.0)
        # curriculum: long balance-only phase at FULL height (400 iters),
        # then gentle speed, then the full range. Run 3 showed the crouch basin
        # + early speed commands destroy balance; MIN_BASE_Z now forbids crouch.
        it = ITERATION[0]
        if it < 400:
            vx = 0.0
        elif it < 750:
            vx = self.rng.choice([-1, 1]) * self.rng.uniform(0.1, 0.25)
        else:
            if self.rng.random() < 0.25:
                vx = 0.0
            else:
                vx = self.rng.choice([-1, 1]) * self.rng.uniform(0.2, 0.8)
        wz = 0.0
        if YAW and it >= 400 and self.rng.random() < 0.4:
            wz = self.rng.choice([-1, 1]) * self.rng.uniform(0.2, 1.0)  # rad/s
        self.cmds[i] = [vx, 0.0, wz]
        self.h_cmds[i] = self.rng.uniform(*HEIGHT_RANGE)
        self.prev_actions[i] = 0
        self.step_count[i] = 0

    def step(self, actions):
        chunks = np.array_split(np.arange(N_ENVS), N_THREADS)
        results = [None] * N_THREADS
        futs = [self._pool.submit(self._step_chunk, c, actions) for c in chunks]
        for j, f in enumerate(futs):
            results[j] = f.result()
        obs = np.zeros((N_ENVS, 35), np.float32)
        rew = np.zeros(N_ENVS, np.float32)
        dones = np.zeros(N_ENVS, bool)
        for chunk in results:
            for (i, o, r, dn) in chunk:
                obs[i], rew[i], dones[i] = o, r, dn
        for i in np.where(dones)[0]:
            self._reset_one(i, self.datas[i])
        return obs, rew, dones

    def _step_chunk(self, idxs, actions):
        out = []
        for i in idxs:
            obs, rew, dn = self._step_one(i, actions[i])
            out.append((int(i), obs, rew, dn))
        return out

    def _step_one(self, i, a):
        mj = self.mjs[i]
        d = self.datas[i]
        cfg = self.cfgs[i]
        # --- action delay: newest action goes to the back, apply the front ---
        act_ring = self.act_ring[i]
        act_ring.append(a)
        a_app = act_ring.pop(0)
        gs, ts, kv = cfg["gs"], cfg["ts"], cfg["kv"]
        leg_t = np.clip(cfg["default_pose"] + LEG_SCALE * a_app[0:4],
                        cfg["leg_range"][:, 0], cfg["leg_range"][:, 1])
        wheel_t = np.clip(WHEEL_SCALE * a_app[4:6], -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
        for _ in range(cfg["substeps"]):  # 500 Hz control, policy at 50 Hz
            for k in range(4):
                q = d.qpos[cfg["leg_qadr"][k]]
                dq = d.qvel[cfg["leg_dadr"][k]]
                tau = gs * LEG_KP * (leg_t[k] - q) - gs * LEG_KD * dq
                d.ctrl[cfg["leg_act"][k]] = float(np.clip(tau, -LEG_TORQUE * ts, LEG_TORQUE * ts))
            for k in range(2):
                dq = d.qvel[cfg["wheel_dadr"][k]]
                tau = kv * (wheel_t[k] - dq)
                d.ctrl[cfg["wheel_act"][k]] = float(np.clip(tau, -WHEEL_TORQUE, WHEEL_TORQUE))
            mujoco.mj_step(mj, d)
        # --- intermittent push disturbance (SCUT push_robot, every 5-10 s) ---
        if DR and self.step_count[i] * 0.02 >= self.next_push[i]:
            ri = self.rngs[i]
            d.qvel[3] += ri.uniform(-0.25, 0.25)
            d.qvel[4] += ri.uniform(-0.25, 0.25)
            self.next_push[i] = self.step_count[i] * 0.02 + ri.uniform(5.0, 10.0)
        # ---------------- observations (35D contract) ----------------
        bid = cfg["base_bid"]
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(mj, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 1)
        ang_vel = vel[0:3]
        lin_vel = vel[3:6]
        R = d.xmat[bid].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        leg_pos = np.array([d.qpos[qadr] for qadr in cfg["leg_qadr"]]) - cfg["default_pose"]
        leg_vel = np.array([d.qvel[dadr] for dadr in cfg["leg_dadr"]])
        wheel_vel = np.array([d.qvel[dadr] for dadr in cfg["wheel_dadr"]])
        obs = np.concatenate([
            self.cmds[i],                       # 0-2
            [self.h_cmds[i] * 5.0],             # 3
            ang_vel * 0.5,                      # 4-6
            grav,                               # 7-9
            leg_pos,                            # 10-13
            np.zeros(2),                        # 14-15 wheel pos slots
            leg_vel * 0.1,                      # 16-19
            wheel_vel * 0.1,                    # 20-21
            self.prev_actions[i],               # 22-27
            [1.0, 0, 0, 0, 0, 0, 0],            # 28-34
        ]).astype(np.float32)
        obs = np.clip(np.nan_to_num(obs), -100.0, 100.0)
        if DR:  # SCUT self_obs_noise (scaled by the same obs scales)
            ri = self.rngs[i]
            obs[4:7] += ri.uniform(-0.125, 0.125, 3)
            obs[7:10] += ri.uniform(-0.03, 0.03, 3)
            obs[10:14] += ri.uniform(-0.025, 0.025, 4)
            obs[16:20] += ri.uniform(-0.1, 0.1, 4)
            obs[20:22] += ri.uniform(-0.1, 0.1, 2)
            obs = np.clip(obs, -100.0, 100.0)
        # --- observation delay: append newest, return oldest ---
        obs_ring = self.obs_ring[i]
        obs_ring.append(obs)
        obs = obs_ring.pop(0)
        # ---------------- reward ----------------
        v_fwd = -lin_vel[1]  # export frame: forward = -y
        r_v = 0.5 * np.exp(-((self.cmds[i][0] - v_fwd) ** 2) / SIGMA_V ** 2)
        if RV2:
            err_h = abs(self.h_cmds[i] - d.qpos[2])
            r_v *= float(np.clip((0.1 - err_h) / 0.05, 0.0, 1.0))  # height gate
        r_wz = 0.0
        if YAW:
            r_wz = 0.5 * np.exp(-((self.cmds[i][2] - ang_vel[2]) ** 2) / SIGMA_WZ ** 2)
        r_h = 2.0 * np.exp(-((self.h_cmds[i] - d.qpos[2]) ** 2) / SIGMA_H ** 2)
        r_up = float(grav[2] + 1.0)  # 1 upright -> 0 horizontal
        r_act = -0.005 * float(np.sum(a * a))
        r_rate = -0.005 * float(np.sum((a - self.prev_actions[i]) ** 2))
        r_leg = -0.001 * float(np.sum(leg_vel ** 2))
        r_ss = 0.0
        if RV2 and abs(self.cmds[i][0]) < 0.1:
            r_ss = -1.0 * abs(v_fwd)  # stand-still deadzone
        rew = r_v + r_wz + r_h + r_up + r_act + r_rate + r_leg + r_ss + 0.5
        # ---------------- termination ----------------
        tipped = grav[2] > -0.55
        low = d.qpos[2] < current_min_base_z()  # crouch = death (forces tall standing)
        self.step_count[i] += 1
        timeout = self.step_count[i] >= STEPS_PER_EPISODE
        done = bool(tipped or low or timeout)
        if done and (tipped or low):
            rew += -100.0
        elif done:  # survived the full episode
            rew += 50.0
        self.prev_actions[i] = a
        return obs, rew, done


# ---------------------------------------------------------------- PPO
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=35, act_dim=6):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh())
        self.mean = nn.Linear(128, act_dim)
        self.value = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.orthogonal_(self.mean.weight, 0.01)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.trunk(x)
        return self.mean(h), self.log_std.exp(), self.value(h).squeeze(-1)


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    n = len(rewards)
    for t in reversed(range(n)):
        if dones[t]:
            nxt = 0.0
        elif t == n - 1:
            nxt = 0.0  # rollout boundary: truncated bootstrap
        else:
            nxt = values[t + 1]
        delta = rewards[t] + gamma * nxt - values[t]
        adv[t] = delta + gamma * lam * last * (0.0 if dones[t] else 1.0)
        last = adv[t]
    return adv


def bc_controller(obs):
    """Hand-tuned linear balancer from the controllability sweep (Kt=60,
    Ktd=0.5, Kv=0): wheel action = -(60*grav_y + 0.5*ang_vel_x)/10.
    Verified: survives the full 6 s at height 0.48 in the sim."""
    a = np.zeros((obs.shape[0], 6), np.float32)
    th = obs[:, 8]           # grav_y ~= sin(tilt) ~= tilt
    thd = obs[:, 4] / 0.5    # ang_vel_x (obs scale 0.5 undone)
    a[:, 4] = a[:, 5] = -(60.0 * th + 0.5 * thd) / 10.0
    return a


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.manual_seed(0)
    env = VecEnv()
    ac = ActorCritic()
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
    obs_mean = np.zeros(35, np.float32)
    obs_var = np.ones(35, np.float32)
    actions = np.zeros((N_ENVS, 6), np.float32)

    # ---------------- behavior-cloning warm start ----------------
    if os.environ.get("V33_BC", "0") == "1":
        print("BC warm start: collecting hand-balancer data ...", flush=True)
        env.set_delays(0, 0)  # the hand balancer needs undelayed feedback
        buf_o, buf_a = [], []
        for _ in range(2):  # 2 rollout lengths
            o = env.step(actions)
            buf_o.append(o[0])
            buf_a.append(actions)
            with torch.no_grad():
                for t in range(ROLLOUT_STEPS - 1):
                    a_bc = bc_controller(o[0]) + np.random.normal(0, 0.05, o[0].shape[0] * 6).reshape(o[0].shape[0], 6)
                    o = env.step(a_bc)
                    buf_o.append(o[0])
                    buf_a.append(a_bc)
                    actions = a_bc
        bc_o = np.concatenate(buf_o, axis=0)
        bc_a = np.concatenate(buf_a, axis=0)
        obs_mean = bc_o.mean(0)
        obs_var = bc_o.var(0) + 1e-6
        bc_o = (bc_o - obs_mean) / np.sqrt(obs_var)
        bt = torch.from_numpy(bc_o)
        at = torch.from_numpy(bc_a)
        bc_opt = torch.optim.Adam(ac.parameters(), lr=1e-3)
        for step in range(2000):
            ids = torch.randint(0, bt.shape[0], (256,))
            mu, std, _ = ac(bt[ids])
            loss = ((mu - at[ids]) ** 2).mean() - 0.01 * torch.log(std).sum(-1).mean()
            bc_opt.zero_grad()
            loss.backward()
            bc_opt.step()
        with torch.no_grad():
            ac.log_std.fill_(-1.2)  # start PPO with low exploration
        env.set_delays(OBS_DELAY, ACT_DELAY)  # PPO trains with deploy delays
        print("BC warm start done", flush=True)

    print("training start", flush=True)
    t0 = time.time()
    best_reward = -1e9
    for it in range(MAX_ITERS):
        ITERATION[0] = it
        o_buf = np.zeros((ROLLOUT_STEPS, N_ENVS, 35), np.float32)
        a_buf = np.zeros((ROLLOUT_STEPS, N_ENVS, 6), np.float32)
        r_buf = np.zeros((ROLLOUT_STEPS, N_ENVS), np.float32)
        d_buf = np.zeros((ROLLOUT_STEPS, N_ENVS), bool)
        with torch.no_grad():
            for t in range(ROLLOUT_STEPS):
                o = env.step(actions)
                o_n = (o[0] - obs_mean) / np.sqrt(obs_var + 1e-8)
                mu, std, _ = ac(torch.from_numpy(o_n))
                dist = Normal(mu, std)
                actions = dist.sample().numpy()
                o_buf[t] = o[0]
                a_buf[t] = actions
                r_buf[t] = o[1]
                d_buf[t] = o[2]
        obs_mean = 0.995 * obs_mean + 0.005 * o_buf.reshape(-1, 35).mean(0)
        obs_var = 0.995 * obs_var + 0.005 * o_buf.reshape(-1, 35).var(0)
        o_n = (o_buf - obs_mean) / np.sqrt(obs_var + 1e-8)
        o_t = torch.from_numpy(o_n.reshape(-1, 35))
        a_t = torch.from_numpy(a_buf.reshape(-1, 6))
        with torch.no_grad():
            mu, std, v_t = ac(o_t)
        v_np = v_t.numpy()
        adv = gae(r_buf.reshape(-1), v_np, d_buf.reshape(-1))
        adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
        ret_t = torch.from_numpy(adv + v_np)
        idx = torch.randperm(o_t.shape[0])
        batch = o_t.shape[0] // 4
        for _ in range(5):
            for mb in range(4):
                ids = idx[mb * batch:(mb + 1) * batch]
                mu, std, v = ac(o_t[ids])
                dist = Normal(mu, std)
                logp = dist.log_prob(a_t[ids]).sum(-1)
                with torch.no_grad():
                    mu_old, std_old, _ = ac(o_t[ids])
                    logp_old = Normal(mu_old, std_old).log_prob(a_t[ids]).sum(-1)
                ratio = (logp - logp_old).exp()
                surr1 = ratio * adv_t[ids]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[ids]
                v_loss = 0.5 * ((v - ret_t[ids]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = -torch.min(surr1, surr2).mean() + v_loss - 0.001 * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
        if it % 10 == 0 or it == MAX_ITERS - 1:
            mean_r = r_buf.mean()
            print(f"iter {it}: mean_r={mean_r:+.3f} el={time.time()-t0:.1f}s", flush=True)
            if mean_r > best_reward:
                best_reward = mean_r
                torch.save({"ac": ac.state_dict(), "obs_mean": obs_mean, "obs_var": obs_var},
                            os.path.join(LOG_DIR, "v33_policy.pt"))
        if it % 50 == 49:
            torch.save({"ac": ac.state_dict(), "obs_mean": obs_mean, "obs_var": obs_var},
                        os.path.join(LOG_DIR, f"v33_policy_{it+1}.pt"))
    print(f"done in {time.time()-t0:.0f}s, best_mean_r={best_reward:+.3f}", flush=True)
    # quick eval: 6 s, cmd 0.5
    ck = torch.load(os.path.join(LOG_DIR, "v33_policy.pt"), weights_only=False)
    ac.load_state_dict(ck["ac"])
    ac.eval()
    env.cmds[:] = [0.5, 0, 0]
    env.h_cmds[:] = BASE_HEIGHT
    env.reset_all()
    actions = np.zeros((N_ENVS, 6), np.float32)
    for t in range(300):
        o = env.step(actions)
        o_n = (o[0] - ck["obs_mean"]) / np.sqrt(ck["obs_var"] + 1e-8)
        with torch.no_grad():
            mu, std, _ = ac(torch.from_numpy(o_n))
        actions = mu.numpy()
        if t % 50 == 0:
            zs = np.mean([d.qpos[2] for d in env.datas])
            print(f"eval t={t*0.02:.1f}s: mean_base_z={zs:.3f}", flush=True)


if __name__ == "__main__":
    main()
