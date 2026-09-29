#!/usr/bin/env python3
"""Two-object native GPU contact-direction/patch-classification probe.

A 5 kg cuboid drops onto a 20 cm platform at 1 kHz. This verifies the contact
adapter and legal top-support predicate; it is not a robot or climbing trial.
No PPO is instantiated and no formal training updates are consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DT = .001


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def classify_native_sample(task, points, forces, matrix):
    """Only nonzero, physically measured patches can constitute evidence."""
    import torch

    require(points.ndim == 2 and points.shape == forces.shape and points.shape[1] == 3,
            "Native points/forces must be [patch,3]")
    require(torch.isfinite(points).all() and torch.isfinite(forces).all() and torch.isfinite(matrix).all(),
            "Nonfinite contact data")
    vector = forces.sum(0)
    expected = matrix.reshape(-1, 3).sum(0)
    error = (vector - expected).norm()
    require(error <= .01 + .002 * expected.norm(), "force*normal direction differs from native matrix")
    nonzero = forces.norm(dim=-1) > .1
    if not bool(nonzero.any()):
        return {"status": "not_observed", "patches": 0, "upward_force_n": 0.}
    legal = task.legal_body_contacts(points[None], forces[None])[0]
    require(legal[nonzero].all(), "A measured nonzero top-support patch was classified illegal")
    require((forces[nonzero, 2] > 0.).all(), "Expected upward native normal forces on the falling cube")
    return {"status": "observed", "patches": int(nonzero.sum()),
            "upward_force_n": float(forces[nonzero, 2].sum()),
            "force_matrix_error_n": float(error),
            "max_top_height_error_m": float((points[nonzero, 2] - .2).abs().max()),
            "positive_patches_below_20n": int((nonzero & (forces[:, 2] < 20.)).sum()),
            "all_nonzero_patches_legal": True,
            "points_m": points[nonzero].cpu().tolist(), "forces_n": forces[nonzero].cpu().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics-steps", type=int, default=250)
    args = parser.parse_args()
    if not 200 <= args.physics_steps <= 250:
        parser.error("200--250 physics steps required for contact and settling evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "scope": "engineering_native_contact_adapter_only",
              "formal_updates_consumed": 0, "capability_acceptance": "not_assessed",
              "physics_dt_s": DT, "physics_steps": args.physics_steps,
              "cube_mass_kg": 5., "cube_size_m": [.4, .4, .05],
              "cube_initial_center_m": [.35, 0., .3], "platform_top_m": .2,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "step_climb_sha256": hashlib.sha256((ROOT / "src/wheeled_tasks/chassis/step_climb.py").read_bytes()).hexdigest(),
              "samples": []}
    launcher = sim = None
    try:
        os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": "cuda:0",
            "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
        import carb.settings
        import torch
        import warp as wp
        from pxr import PhysxSchema, UsdPhysics
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxCfg, PhysxManager
        from wheeled_tasks.chassis.step_climb import StepClimbTask

        torch.set_num_threads(4)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=DT,
            gravity=(0., 0., -9.81), render_interval=20,
            physics=PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)))
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        material = sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.8, restitution=0.)
        platform = sim_utils.CuboidCfg(size=(.8, .8, .2),
            collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=material)
        platform.func("/World/Platform", platform, translation=(.4, 0., .1))
        cube = sim_utils.CuboidCfg(size=(.4, .4, .05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.),
            collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=material)
        cube.func("/World/Cube", cube, translation=(.35, 0., .3))
        bodies = [prim for prim in sim.stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        require(len(bodies) == 1, "Expected exactly one dynamic rigid body plus the static platform")
        cube_path = str(bodies[0].GetPath())
        PhysxSchema.PhysxContactReportAPI.Apply(bodies[0]).CreateThresholdAttr().Set(0.)
        platform_path = "/World/Platform/geometry/mesh"
        require(sim.stage.GetPrimAtPath(platform_path).HasAPI(UsdPhysics.CollisionAPI), "Platform collider path missing")
        sim.reset()
        scene = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        require(scene.GetAttribute("physxScene:enableGPUDynamics").Get(), "Probe is not using GPU dynamics")
        view = PhysxManager.get_physics_sim_view()
        body_view = view.create_rigid_body_view([cube_path])
        masses = wp.to_torch(body_view.get_masses())
        require(torch.allclose(masses, torch.full_like(masses, 5.)), "Native cube mass differs from 5 kg")
        contact = view.create_rigid_contact_view([cube_path], filter_patterns=[[platform_path]], max_contact_data_count=32)
        require(contact.filter_count == 1, "Unexpected contact filter count")
        task = StepClimbTask(["contact"], {"contact": {"body_climb": {
            "rises_m": [.2], "treads_m": [.8], "body_contact_min_force_n": 20.,
            "support_hold_seconds": .1, "settle_seconds": .35, "landing_margin_m": .06}}}, "cuda:0", DT)
        observed = []
        for tick in range(1, args.physics_steps + 1):
            sim.step(render=False)
            force, point, normal, _, counts, starts = map(wp.to_torch, contact.get_contact_data(DT))
            count, start = int(counts.reshape(-1)[0]), int(starts.reshape(-1)[0])
            require(0 <= count <= 32 and 0 <= start <= len(force)
                    and start + count <= len(force), "Native patch buffer indices out of range")
            points = point[start:start + count]
            forces = normal[start:start + count] * force.reshape(-1, 1)[start:start + count]
            matrix = wp.to_torch(contact.get_contact_force_matrix(dt=DT))
            result = classify_native_sample(task, points, forces, matrix)
            if result["status"] == "observed":
                result["physics_step"] = tick
                observed.append(result)
                if len(observed) == 1 or tick % 10 == 0 or tick == args.physics_steps:
                    report["samples"].append(result)
        require(observed, "No actual base/platform contact was observed")
        settled = [row for row in observed if row["physics_step"] > args.physics_steps - 20]
        require(len(settled) == 20 and all(row["upward_force_n"] > 20. for row in settled),
                "No sustained loaded top support in the final 20 ms")
        require(max(row["max_top_height_error_m"] for row in settled) < .003,
                "Settled native contact points differ from the platform top")
        split = [row for row in settled if row["patches"] >= 2 and row["positive_patches_below_20n"] > 0]
        require(split, "Shared load over individually sub-20 N legal patches was not observed")
        report.update(status="native_contact_verified", gpu_dynamics_enabled=True,
            dynamic_rigid_bodies=1, static_colliders=1, actual_mass_kg=float(masses.sum()),
            observed_contact_ticks=len(observed), first_contact_step=observed[0]["physics_step"],
            settled_upward_force_mean_n=sum(row["upward_force_n"] for row in settled) / len(settled),
            settled_top_error_max_m=max(row["max_top_height_error_m"] for row in settled),
            legal_split_load_ticks=len(split), native_contact_direction="force_times_normal_matches_matrix_and_points_up")
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if sim is not None:
            sim.stop()
        if launcher is not None:
            launcher.app.close()
    print(json.dumps({key: report.get(key) for key in ("status", "formal_updates_consumed", "observed_contact_ticks")}))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
