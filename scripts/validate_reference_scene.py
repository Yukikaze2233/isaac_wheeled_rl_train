#!/usr/bin/env python3
"""Validate command-reference wiring, partial resets and actual terrain dimensions."""
import argparse
from copy import deepcopy
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
    args = parser.parse_args()
    from train_chassis import preflight, digest
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    config, manifest = preflight(args.contract)
    wanted = {"stand", "step_up_15", "step_up_25", "jump_cold_03", "jump_full"}
    cases = [deepcopy(case) for case in config["evaluation"]["cases"] if case["name"] in wanted]
    for name in ("slope_up", "slope_down"):
        cases.append({"name": name, "terrain": name, "task": "traverse", "command": [.4, 0., .305],
            "terrain_limits": {**config["terrain_limits"], "slope_deg": 15.},
            "skill": {"kind": name, "mode": 2 if name == "slope_up" else 3, "command": [.4, 0., .305]}})
    config["evaluation"]["cases"] = cases
    config["signal_perturbations"]["enabled"] = False
    config["usb_transport"]["enabled"] = False
    config["dynamics_randomization"]["enabled"] = False
    config["enable_scene_queries"] = True
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {"status": "starting", "scope": "scene_and_interface_not_policy_acceptance",
              "contract_sha256": digest(args.output / "contract.json"),
              "source_sha256": {name: digest(ROOT / name) for name in (
                  "scripts/validate_reference_scene.py", "src/wheeled_tasks/chassis/env.py",
                  "src/wheeled_tasks/chassis/eval_env.py", "src/wheeled_tasks/chassis/references.py")}}
    launcher, env = None, None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "device": "cuda:0", "enable_cameras": False})
        import torch
        import omni.physx
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.v40.core import load_contract
        torch.set_num_threads(4)
        env = FixedCaseEnv(fixed_suite_contract(config), manifest, load_contract(ROOT / config["control_math_source"]), ROOT,
            stage_name=config["enabled_stages"][0], num_envs=len(cases), device="cuda:0", seed=617, level=1.)
        query = omni.physx.get_physx_scene_query_interface()
        measured = {}
        for i, name in enumerate(env.scene_groups):
            if name not in ("step_up_15", "step_up_25"):
                continue
            ox, oy, oz = env.origins[i].cpu().tolist()
            hit = query.raycast_closest((ox + 2., oy, oz + 1.), (0., 0., -1.), 2.)
            assert hit["hit"] and "/Terrain/" in hit["collision"]
            height = float(hit["position"][2]) - oz
            assert abs(height - (.15 if name.endswith("15") else .25)) < 1e-5
            measured[name] = height
        step_ids = torch.tensor([i for i, name in enumerate(env.scene_groups) if name.startswith("step_up")], device=env.device)
        slope_ids = torch.tensor([i for i, name in enumerate(env.scene_groups) if name.startswith("slope")], device=env.device)
        reset_seen = 0
        for tick in range(150):
            obs, reward, done, extras = env.step(torch.zeros(len(cases), 6, device=env.device))
            assert obs["policy"].shape == (len(cases), 36) and obs["critic"].shape == (len(cases), 114)
            torch.testing.assert_close(env.raw_height[step_ids], torch.full((len(step_ids),), .305, device=env.device))
            torch.testing.assert_close(obs["policy"][:, 3], env.commands[:, 2] * 5.)
            assert torch.isfinite(reward).all()
            reset_seen += int(done.sum())
            if tick < 10:
                assert not bool(extras["diagnostics"]["terminated"][slope_ids].any()), "Slope setup failed before approach"
        assert reset_seen > 0, "Probe must exercise partial resets"
        report.update(status="passed", measured_step_heights_m=measured, partial_resets=reset_seen,
            raw_height_preserved=True, reference_observation_consistent=True,
            slope_first_200ms_valid=True, physics=env.summary())
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
