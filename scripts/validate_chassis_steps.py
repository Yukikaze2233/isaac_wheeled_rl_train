#!/usr/bin/env python3
"""Check actual solid-step collision geometry and the height-command interface."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--policy", type=Path, help="Optional bounded ONNX approach diagnostic, not an acceptance test")
    parser.add_argument("--lookahead-m", type=float, help="Explicit perception-distance ablation")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be in [1,1000]")
    from train_chassis import digest, preflight
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    config, manifest = preflight(args.contract)
    wanted = {"step_up_15", "step_up_20", "step_up_25"}
    cases = [deepcopy(case) for case in config["evaluation"]["cases"] if case["name"] in wanted]
    if len(cases) != 3:
        raise ValueError("All three target step cases are required")
    wall = deepcopy(cases[-1])
    wall.update(name="wall_035", terrain_limits={**wall["terrain_limits"], "step_up_m": .35})
    wall["skill"]["terrain_limits"] = wall["terrain_limits"]
    cases.append(wall)
    config["evaluation"]["cases"] = cases
    config["usb_transport"]["enabled"] = False
    config["dynamics_randomization"]["enabled"] = False
    config["step_assist"]["enabled"] = True
    if args.lookahead_m is not None:
        config["step_assist"]["lookahead_m"] = args.lookahead_m
    config["enable_scene_queries"] = True
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "collision_and_interface_probe_not_climbing_acceptance",
        "source_contract_sha256": digest(args.contract), "contract_sha256": digest(args.output / "contract.json"),
        "source_sha256": {name: digest(ROOT / name) for name in (
            "scripts/validate_chassis_steps.py", "src/wheeled_tasks/chassis/env.py",
            "src/wheeled_tasks/chassis/task.py", "src/wheeled_tasks/chassis/step_assist.py")}}
    launcher, env = None, None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "device": "cuda:0", "enable_cameras": False,
            "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
        import torch
        import omni.physx
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.v40.core import load_contract
        torch.set_num_threads(4)
        env = FixedCaseEnv(fixed_suite_contract(config), manifest,
            load_contract(ROOT / config["control_math_source"]), ROOT,
            stage_name=config["enabled_stages"][0], num_envs=4, device="cuda:0", seed=617, level=1.)
        query = omni.physx.get_physx_scene_query_interface()
        geometry = []
        for i, name in enumerate(env.scene_groups):
            ox, oy, oz = env.origins[i].cpu().tolist()
            expected = next(case["terrain_limits"]["step_up_m"] for case in cases if case["name"] == name)
            hit = query.raycast_closest((ox + 2., oy, oz + 1.), (0., 0., -1.), 2.)
            assert hit["hit"] and "/Terrain/" in hit["collision"], (name, hit)
            top = float(hit["position"][2]) - oz
            assert abs(top - expected) < 1e-5, (name, top, expected)
            face = query.raycast_closest((ox - .2, oy, oz + .025), (1., 0., 0.), .4)
            assert face["hit"] and "/Terrain/" in face["collision"], (name, face)
            assert abs(float(face["position"][0]) - ox) < 1e-4
            geometry.append({"case": name, "top_height_m": top, "riser_solid_at_25mm": True})
        # Inject an explicit sensor sequence, not a robot pose change, to check
        # command assembly before the following real transition and reward.
        ids = torch.arange(4, device=env.device)
        env.raw_height[:] = .305
        env.commands[:, 0] = .4
        samples = torch.zeros(4, 2, device=env.device)
        valid = torch.ones_like(samples, dtype=torch.bool)
        direction = torch.tensor([[1., 0.]], device=env.device).repeat(4, 1)
        eligible = torch.ones(4, dtype=torch.bool, device=env.device)
        env.step_assist.reset(ids)
        env.step_assist.update(env.raw_height, samples, valid, direction, eligible, eligible)
        samples[:] = env.platform_delta[:, None]
        env.commands[:, 2] = env.step_assist.update(env.raw_height, samples, valid, direction, eligible, eligible)
        obs = env.get_observations()
        torch.testing.assert_close(obs["policy"][:, 3], env.commands[:, 2] * 5)
        _, _, _, extras = env.step(torch.zeros(4, 6, device=env.device))
        diagnostic = extras["diagnostics"]
        wall_id = env.scene_groups.index("wall_035")
        step_ids = ids[ids != wall_id]
        torch.testing.assert_close(diagnostic["commands"][step_ids, 2], torch.full((3,), .355, device=env.device))
        expected_error = (diagnostic["height"] - diagnostic["commands"][:, 2]).abs().mean()
        torch.testing.assert_close(extras["log"]["/task/height_error_m"], expected_error)
        assert bool(diagnostic["reasons"]["blocked"][wall_id]) and not bool(diagnostic["success"][wall_id])
        for _ in range(args.steps - 1):
            env.step(torch.zeros(4, 6, device=env.device))
        report.update(status="passed", geometry=geometry, startup=env.startup_report, physics=env.summary(),
            observation_and_reward_use_effective_height=True, wall_not_success=True,
            interface_probe="synthetic_forward_samples_with_real_physics_transition")
        if args.policy:
            import onnxruntime as ort
            session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
            env.cfg["auto_reset"] = False
            observation = env.reset_suite()
            alive = torch.ones(4, dtype=torch.bool, device=env.device)
            events = torch.zeros(4, dtype=torch.long, device=env.device)
            rows = []
            for _ in range(args.steps):
                actions = session.run(None, {"obs": observation["policy"].cpu().numpy()})[0]
                observation, _, done, extras = env.step(torch.as_tensor(actions, device=env.device))
                diagnostic = extras["diagnostics"]
                events += diagnostic["step_detected"] & alive
                for index in (done & alive).nonzero().flatten().tolist():
                    rows.append({"case": env.scene_groups[index], "seconds": float(diagnostic["episode_ticks"][index]) * env.policy_dt,
                        "reasons": [name for name, value in diagnostic["reasons"].items() if bool(value[index])],
                        "contact_bodies_above_5N": [name for j, name in enumerate(env.robot.body_names)
                            if float(env.contact_force[index, j].norm()) > 5.],
                        "base_local_x_m": float(diagnostic["position"][index, 0]),
                        "wheel_local_x_m": (env.wheel_centers()[index, :, 0] - env.origins[index, 0]).cpu().tolist(),
                        "assist_events": int(events[index])})
                alive &= ~done
                if not alive.any():
                    break
            report["approach_diagnostic"] = {"policy_sha256": digest(args.policy), "outcomes": rows,
                                              "unfinished": int(alive.sum()), "assist_events": events.cpu().tolist()}
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
