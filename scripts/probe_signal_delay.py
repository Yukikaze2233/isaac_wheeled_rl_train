#!/usr/bin/env python3
"""Bounded native-scene delay probe: no actor, optimizer, PPO or skill credit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight

    path = ROOT / "contracts/v6_task_campaign_flat_worker_v6.json"
    config, manifest = preflight(path)
    control = load_chassis_control(ROOT / config["control_math_source"])
    plan = json.loads((ROOT / "contracts/v6_task_campaign_v6.json").read_text())
    count, seed = config["target_num_envs"], plan["seed"]
    receipt = design_preflight(config, control, ROOT, stage="foundation", role="train", seed=seed, num_envs=count)
    os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": "cuda:0"})
    env = None
    try:
        import torch
        torch.set_num_threads(4)
        from wheeled_tasks.chassis.env import ChassisEnv
        env = ChassisEnv(config, manifest, control, ROOT, stage_name="foundation", num_envs=count,
                         device="cuda:0", level=1., seed=seed, block_index=0)
        # Exercise the frozen schedule's endpoint only in this engineering
        # probe. No training clock, contract, checkpoint or budget is changed.
        env.task_actor_update = config["signal_delay"]["schedule"]["end"]
        feedback_checks = 0
        motor_efforts = env.v5.motor_efforts

        def checked_efforts(q, dq, legs, wheels):
            nonlocal feedback_checks
            current_q = env.robot.data.joint_pos.torch[:, env.ids]
            current_dq = env.robot.data.joint_vel.torch[:, env.ids]
            torch.testing.assert_close(q, current_q, rtol=0., atol=0.)
            torch.testing.assert_close(dq, current_dq, rtol=0., atol=0.)
            feedback_checks += 1
            return motor_efforts(q, dq, legs, wheels)

        env.v5.motor_efforts = checked_efforts
        started = time.monotonic()
        for tick in range(24):
            with torch.no_grad():
                actions = torch.full((count, 6), .015 if tick % 2 else -.015, device="cuda:0")
                observations, reward, _, _ = env.step(actions)
                assert observations["policy"].shape == (count, 35)
                assert observations["critic"].shape == (count, 81)
                assert all(torch.isfinite(x).all() for x in (*observations.values(), reward))
        cached, physical_steps = env.signal_delay.feedback.clone(), env.signal_delay.physics_steps
        for _ in range(3):
            env.get_observations()
        torch.testing.assert_close(cached, env.signal_delay.feedback, rtol=0., atol=0.)
        assert env.signal_delay.physics_steps == physical_steps == 96 == feedback_checks
        assert env.signal_delay.lags[:, :4].max() == 16 and env.signal_delay.lags[:, 4:].max() == 12
        assert env.signal_delay.lags[~env.signal_delay.enabled].eq(0).all()
        assert env.actuator_response.enabled and env.startup_report["actuator_response"]["native_properties_verified"]
        torch.cuda.synchronize()
        report = dict(status="engineering_native_delay_passed", scope="no_PPO_no_skill_acceptance",
            formal_updates_consumed=0, forced_engineering_actor_clock=env.task_actor_update,
            contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), design_preflight=receipt,
            num_envs=count, feedback_checks=feedback_checks, physics_steps=physical_steps,
            wall_seconds=time.monotonic() - started, runtime_preflight=env.startup_report["runtime_preflight"],
            control_timing=env.control_audit.last,
            delays={k: float(v) for k, v in env.signal_delay.metrics().items()})
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report))
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
