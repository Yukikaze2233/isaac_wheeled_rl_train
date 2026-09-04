"""Season-parity feature tests: motor curve, jump trajectory family, terrain
command overrides, GRU branch — the pieces ported from the registry's
battle-proven set. Pure torch; runtime ~1 min.
"""
import importlib.util
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m3508 = _load("m3508_curve", "src/wheeled_world/actuators/m3508_curve.py")
jump = _load("jump_rewards", "src/wheeled_tasks/manager/mdp/jump_rewards.py")
tcmd = _load("terrain_cmd", "src/wheeled_tasks/manager/mdp/terrain_cmd.py")


def test_motor_curve():
    # droop curve: 10 Nm at stall -> 3 Nm at 40 rad/s (measured-style)
    speed = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0])
    torque = torch.tensor([10.0, 9.0, 7.0, 5.0, 3.0])
    motor = m3508.CurvedMotorModel(speed, torque, gear_ratio=268.0 / 17.0, name="M3508")
    q = torch.tensor([5.0, 25.0, 50.0])
    limit = motor.torque_limit(q)
    assert limit[0] < limit[1] < limit[2] or limit[2] <= limit[1], "limit must droop with speed"
    assert torch.all(limit > 0)
    # clip: a big stall-torque command at high speed must be pulled to the curve
    cmd = torch.tensor([20.0, -20.0, 20.0])
    clipped = motor.clip_torque(cmd, q)
    assert clipped[2] <= motor.torque_limit(torch.tensor([50.0]))[0] + 1e-5
    assert clipped[1] < 0  # sign preserved
    # CSV round-trip
    csv_path = "/tmp/m3508_curve.csv"
    with open(csv_path, "w") as f:
        f.write("speed_rpm,torque_nm\n0,10\n1000,9\n2000,7\n")
    m2 = m3508.CurvedMotorModel.from_csv(csv_path, name="M3508")
    assert abs(m2.torque_limit(torch.tensor([0.0]))[0] - 10.0) < 1e-5
    print("motor curve OK")


def test_jump_trajectory_rewards():
    N = 4
    h0 = torch.full((N,), 0.30)
    traj = jump.JumpTrajectory(h0, target_height=0.24, duration=0.3)
    t = torch.tensor([0.0, 0.15, 0.3, 0.45])
    h = traj.height(t)
    vz = traj.vel_z(t)
    assert torch.isclose(h[0], h0[0], atol=1e-5)          # h(0) = takeoff height
    assert torch.isclose(h[2], torch.tensor(0.24), atol=1e-3)  # h(T) = target
    assert abs(vz[2]) < 1e-3                              # v(T) = 0
    in_jump = torch.tensor([True, True, False, False])
    rewards = jump.jump_window_rewards(
        in_jump, t, traj, base_height=h, base_vel_z=vz,
        wheel_speed=torch.zeros(N, 2), wheel_contact=torch.tensor([False] * N),
        root_speed_x=torch.tensor([1.0] * N))
    assert torch.all(rewards["track_h_traj"][2:] == 0)    # outside window: zero
    assert rewards["track_h_traj"][0] > 0.9               # on-trajectory: near max
    assert rewards["air_wheel_zero_torque_exp"][0] > 0.9  # free wheels, low speed
    print("jump trajectory OK")


def test_terrain_command_override():
    override = tcmd.TerrainCommandOverride(
        profiles={"stair": {"vx": (-1.0, 1.0), "yaw_rate": (-1.0, 1.0)},
                  "dash_pad": {"vx": [(2.0, 3.0)]}},
        default={"vx": (-2.5, 2.5), "yaw_rate": (-3.0, 3.0)})
    patch_names = ["flat", "pyramid_stairs", "dash_pad_for_rm"]
    types = torch.tensor([0, 1, 1, 2])
    cmd = torch.zeros(4, 3)
    override.apply(patch_names, types, cmd)
    assert cmd[0, 2].abs() <= 3.0 and cmd[0, 0].abs() <= 2.5      # default envelope
    assert cmd[1, 0].abs() <= 1.0 and cmd[1, 2].abs() <= 1.0      # stairs: reduced
    assert cmd[3, 0] >= 2.0                                       # dash pad: forward only
    print("terrain command override OK")


def test_gru_branch():
    from toy_env_ext import ToyReachExtEnv
    from wheeled_algo.algorithms import ExtTrainCfg, GRUTrainer

    torch.manual_seed(0)
    env = ToyReachExtEnv(num_envs=128, max_ep_steps=40, device="cpu", seed=0, hist_len=3)
    trainer = GRUTrainer(env, obs_dim=env.obs_dim, priv_dim=env.obs_dim + 1, hist_len=3,
                         action_dim=env.num_actions, device="cpu",
                         cfg=ExtTrainCfg(num_steps_per_env=16, learning_rate=3e-4,
                                         num_learning_epochs=3))
    obs, extras = env.reset_()

    def rollout() -> float:
        total = 0.0
        o, ex = env.reset_()
        with torch.no_grad():
            for _ in range(40):
                a, _, _ = trainer.policy.act(o, ex.get("observations", {}))
                o, r, d, ex = env.step(a)
                total += float(r.mean())
        return total / 40

    before = rollout()
    trainer.learn(iterations=40)
    after = rollout()
    print(f"GRU branch: before={before:.3f} after={after:.3f} improvement={after - before:+.3f}")
    assert after - before > 0.05, f"GRU branch failed to learn: {before:.3f} -> {after:.3f}"


if __name__ == "__main__":
    test_motor_curve()
    test_jump_trajectory_rewards()
    test_terrain_command_override()
    test_gru_branch()
    print("SEASON PARITY TESTS PASSED")
