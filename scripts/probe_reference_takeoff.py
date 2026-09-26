#!/usr/bin/env python3
"""Bounded actuator-only reference probe; no root forces or post-reset pose writes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from train_chassis import preflight, digest
    from compare_v5_spring_load import prepare_trial
    from scipy.optimize import brentq
    from analyze_v5_spring_limits import balanced_standing_pose
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    config, manifest = preflight(args.contract)
    bundle = ROOT / config["asset_directory"]
    spec = json.loads((bundle / "model_spec.json").read_text())
    fit = json.loads((bundle / "fit_10mpa.json").read_text())
    heights = np.array([.23, .24, .245, .26, .28, .305, .32, .34, .36, .40, .43])
    poses, torques, com_heights = [], [], []
    for height in heights:
        angle = brentq(lambda value: balanced_standing_pose(spec, value)["base_frame_height_m"] - height, 40., 110.)
        trial = prepare_trial(spec, manifest, fit, angle, allow_reserve_extrapolation=True)
        com_heights.append(balanced_standing_pose(spec, angle)["whole_robot_com_height_m"])
        poses.append([trial["joint_positions"][manifest["control_joint_names"][i]] for i in (0, 1, 3, 4)])
        torques.append(np.asarray(trial["feedforward_nm"][1])[[0, 1, 3, 4]])
    poses, torques = np.asarray(poses), np.asarray(torques)
    com_heights = np.asarray(com_heights)
    derivative = np.gradient(poses, com_heights, axis=0)
    nominal_com_height = float(np.interp(.305, heights, com_heights))
    config["evaluation"]["cases"] = [deepcopy(case) for case in config["evaluation"]["cases"]
                                    if case["name"] in ("jump_cold_03", "jump_small", "jump_full")]
    config.update(auto_reset=False, flat_half_length_m=12., flat_triangle_mesh=True)
    for name in ("dynamics_randomization", "signal_perturbations", "usb_transport"):
        config[name]["enabled"] = False
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {"status": "starting", "scope": "actuator_reference_probe_not_learned_policy_acceptance",
        "post_reset_state_writes": 0, "external_forces": False,
        "source_contract_sha256": digest(args.contract), "script_sha256": digest(Path(__file__)),
        "reference_height_m": heights.tolist(), "reference_leg_q": poses.tolist()}
    launcher, env = None, None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "device": "cuda:0", "enable_cameras": False})
        import torch
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.scut_observation import POLICY_FROM_CONTROL
        from wheeled_tasks.v40.core import load_contract
        torch.set_num_threads(4)
        count = len(config["evaluation"]["cases"])
        env = FixedCaseEnv(fixed_suite_contract(config), manifest, load_contract(ROOT / config["control_math_source"]), ROOT,
                          stage_name=config["enabled_stages"][0], num_envs=count, device="cuda:0", seed=617, level=1.)
        alive = torch.ones(count, dtype=torch.bool, device=env.device)
        previous_vz = torch.zeros(count, device=env.device)
        rows = []
        for tick in range(400):
            r = env.references
            moving = (r.phase == r.PRELOAD) | (r.phase == r.PUSH)
            requested_com_height = nominal_com_height + r.height_delta.cpu().numpy()
            if np.any(((r.phase == r.PRELOAD) | (r.phase == r.PUSH)).cpu().numpy()
                      & ((requested_com_height < com_heights[0]) | (requested_com_height > com_heights[-1]))):
                raise ValueError("COM reference is outside the closed-chain posture domain")
            base_height = np.interp(requested_com_height, com_heights, heights)
            target_height = torch.tensor(base_height, device=env.device, dtype=torch.float32)
            target_height = torch.where(r.phase == r.FLIGHT, .24, target_height).clamp(.23, .43)
            qref = np.column_stack([np.interp(target_height.cpu(), heights, poses[:, i]) for i in range(4)])
            jac = np.column_stack([np.interp(target_height.cpu(), heights, derivative[:, i]) for i in range(4)])
            ff = np.column_stack([np.interp(target_height.cpu(), heights, torques[:, i]) for i in range(4)])
            qref, jac, ff = [torch.tensor(value, device=env.device, dtype=torch.float32) for value in (qref, jac, ff)]
            acceleration = ((r.vertical_velocity - previous_vz) / env.policy_dt).clamp(-15., 15.) * moving
            previous_vz.copy_(r.vertical_velocity)
            dynamic = env.body_mass.sum(-1)[:, None] * acceleration[:, None] * jac / jac.square().sum(-1, keepdim=True).clamp_min(1e-6)
            desired = qref + (ff + dynamic + 2 * jac * r.vertical_velocity[:, None] * moving[:, None]) / 60.
            action = torch.zeros(count, 6, device=env.device)
            action[:, [0, 1, 3, 4]] = (desired - env.v5.nominal[[0, 1, 3, 4]]) / .25
            _, omega, gravity, _, _, dq, _ = env.state()
            pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
            wheel_torque = (40 * pitch + 4 * omega[:, 1]).clamp(-3.8, 3.8)
            action[:, 2] = (dq[:, 2] + wheel_torque / .2) / 10
            action[:, 5] = (dq[:, 5] - wheel_torque / .2) / 10
            _, _, done, extras = env.step(action[:, POLICY_FROM_CONTROL])
            d = extras["diagnostics"]
            for index in (done & alive).nonzero().flatten().tolist():
                rows.append({"case": env.scene_groups[index], "seconds": (tick + 1) * env.policy_dt, "terminal": True,
                    "clearance_m": float(d["jump_clearance_peak"][index]),
                    "air_time_s": float(d["jump_air_time_peak"][index]),
                    "com_release_speed_m_s": float(d["jump_com_release_speed"][index]),
                    "com_rise_m": float(d["jump_com_rise"][index]), "success": bool(d["success"][index]),
                    "reasons": [name for name, value in d["reasons"].items() if bool(value[index])]})
            alive &= ~done
            if not alive.any():
                break
        for index in alive.nonzero().flatten().tolist():
            rows.append({"case": env.scene_groups[index], "terminal": False,
                "clearance_m": float(env.full_tasks.clearance_peak[index]),
                "air_time_s": float(env.full_tasks.clear_air_time_peak[index]),
                "com_release_speed_m_s": float(env.full_tasks.com_release_speed[index]),
                "com_rise_m": float(env.full_tasks.com_rise[index])})
        report.update(status="measured", results=rows, unfinished=int(alive.sum()), physics=env.summary())
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output / "probe.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    main()
