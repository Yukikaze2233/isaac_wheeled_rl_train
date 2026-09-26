#!/usr/bin/env python3
"""Read back startup variants and verify reset/pose invariants in real PhysX."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "contracts/v5_full_usb_v59.json")
    parser.add_argument("--dynamics", type=Path, default=ROOT / "contracts/rigid_body_startup_v1.json")
    parser.add_argument("--asset-directory", type=Path,
                        help="Relocate the identical manifest-pinned asset without changing its identity")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be in [1,1000]")
    from train_chassis import digest, preflight
    from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(json.loads(args.plan.read_text()), load)
    config = stage_contract(load(plan["base_contract"]), plan, plan["stages"][0], 128)
    if args.asset_directory is not None:
        config["asset_directory"] = str(args.asset_directory)
    config["dynamics_randomization"] = json.loads(args.dynamics.read_text())
    profiles = [("nominal", None)]
    for suffix, base, other in (("light", .9, .9), ("heavy", 1.3, 1.1)):
        for index, xyz in enumerate(itertools.product((-.04, .04), (-.02, .02), (-.02, .02))):
            profiles.append((f"{suffix}_corner_{index}", {
                "base_mass_scale": base, "leg_mass_scale": other,
                "wheel_mass_scale": other, "base_com_offset_m": list(xyz)}))
    config["evaluation"]["cases"] = [{"name": name, "command": [0., 0., .305],
        "skill": {"kind": "stand", "mode": 0, "command": [0., 0., .305]},
        "dynamics_profile": profile, "transport_enabled": False} for name, profile in profiles]
    config.update(flat_half_length_m=4., evaluation_long_corridors=False)
    args.output.mkdir(parents=True, exist_ok=False)
    contract_path = args.output / "contract.json"
    contract_path.write_text(json.dumps(config, indent=2) + "\n")
    config, manifest = preflight(contract_path)
    report = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": digest(contract_path), "device": args.device,
        "scope": "property_and_reset_invariants_not_policy_or_workspace_acceptance",
        "source_sha256": {name: digest(ROOT / name) for name in (
            "scripts/validate_chassis_dynamics.py", "src/wheeled_tasks/chassis/dynamics.py",
            "src/wheeled_tasks/chassis/env.py", "src/wheeled_tasks/chassis/eval_env.py")}}
    launcher, env = None, None
    try:
        os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "device": args.device, "enable_cameras": False,
            "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
        import torch
        import warp as wp
        from wheeled_tasks.chassis.dynamics import RigidBodyRandomization
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.v40.core import load_contract
        torch.set_num_threads(4)
        env = FixedCaseEnv(fixed_suite_contract(config), manifest,
            load_contract(ROOT / config["control_math_source"]), ROOT,
            stage_name=config["enabled_stages"][0], num_envs=len(profiles),
            device=args.device, seed=617, level=1.)
        sample = env.dynamics_randomization
        report["startup"] = env.startup_report
        nominal_id = env.scene_groups.index("nominal")
        nominal = [value[nominal_id:nominal_id + 1].repeat(len(profiles), 1, *([1] if value.ndim == 3 else []))
                   for value in (sample.masses, sample.inertias, sample.coms)]
        restored = RigidBodyRandomization(*nominal, env.robot.body_names,
            {**config["dynamics_randomization"], "enabled_fraction": 0.}, 617)
        poses = env.robot.data.body_link_pose_w.torch.clone()
        joints = env.robot.data.joint_pos.torch.clone()
        checks = {}
        for label, variant in (("nominal_restore", restored), ("variant_reapply", sample)):
            variant.apply(env.robot)
            for key, expected, getter in (("mass", variant.masses, env.robot.root_view.get_masses),
                    ("inertia", variant.inertias, env.robot.root_view.get_inertias),
                    ("com", variant.coms, env.robot.root_view.get_coms)):
                actual = wp.to_torch(getter()).to(args.device)
                if key == "com":
                    actual, expected = actual[:, :, :3], expected[:, :, :3]
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                checks[f"{label}_{key}_readback"] = True
            torch.testing.assert_close(wp.to_torch(env.robot.root_view.get_link_transforms()), poses, atol=0., rtol=0.)
            torch.testing.assert_close(wp.to_torch(env.robot.root_view.get_dof_positions()), joints, atol=0., rtol=0.)
        checks["property_writes_do_not_change_link_or_joint_pose"] = True
        env.body_mass.copy_(sample.masses)
        torch.testing.assert_close(env.robot.data.body_mass.torch, sample.masses)
        torch.testing.assert_close(env.robot.data.body_com_pose_b.torch,
                                   wp.to_torch(env.robot.root_view.get_coms()).to(args.device))
        checks["asset_mass_and_com_caches_match"] = True
        before = sample.summary()["sampled_properties_sha256"]
        env.reset(torch.tensor([0, len(profiles) - 1], device=args.device))
        for _ in range(args.steps):
            env.step(torch.zeros(len(profiles), 6, device=args.device))
        for expected, getter in ((sample.masses, env.robot.root_view.get_masses),
                (sample.inertias, env.robot.root_view.get_inertias), (sample.coms, env.robot.root_view.get_coms)):
            actual = wp.to_torch(getter()).to(args.device)
            if expected.shape[-1] == 7:
                actual, expected = actual[:, :, :3], expected[:, :, :3]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert before == sample.summary()["sampled_properties_sha256"]
        checks["partial_reset_and_steps_preserve_startup_properties"] = True
        report.update(status="passed", checks=checks, physics=env.summary(),
            mass_kg_each=env.body_mass.sum(-1).cpu().tolist(), scene_groups=env.scene_groups)
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
