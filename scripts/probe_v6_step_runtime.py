#!/usr/bin/env python3
"""Build the frozen S5D 20k scene and check 3--5 zero-action policy steps.

Engineering probe only: no actor, optimizer, PPO update, skill acceptance, or
formal budget consumption. Isaac imports stay behind AppLauncher startup.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
COUNT = 20000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def materialize(plan_path, output, expected_sha256=None):
    if expected_sha256 and digest(plan_path) != expected_sha256:
        raise ValueError("Plan differs from the requested frozen SHA256")
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
    from wheeled_tasks.chassis.preflight import design_preflight

    plan = resolve_plan(json.loads(plan_path.read_text()),
                        lambda name: json.loads((ROOT / name).read_text()))
    require(plan.get("curriculum_revision") == "v6_step_curriculum_v1", "Expected the new step curriculum")
    recipe = next(item for item in plan["stages"] if item["name"] == "S5D")
    config = stage_contract({}, plan, recipe, COUNT)
    require(config["record_diagnostics"], "Probe requires the frozen diagnostics path")
    require((config["physics_dt"], config["policy_dt"], config["actor_dim"], config["critic_dim"])
            == (.001, .02, 35, 81), "Frozen clock or observation dimensions changed")
    path = output / "materialized_contract.json"
    path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    config, manifest = preflight(path)
    control = load_chassis_control(ROOT / config["control_math_source"])
    seed = plan["training_seed"]
    receipt = design_preflight(config, control, ROOT, stage=recipe["kind"], role="train",
                               seed=seed, num_envs=COUNT, block_index=0)
    return config, manifest, control, recipe, seed, receipt


def inspect_geometry(env):
    """Check materialized surfaces, including both independently named tasks."""
    rows = {}
    for index, name in enumerate(env.scene_groups):
        spec = env.skills.specs.get(name, {})
        task = spec.get("step_jump", spec.get("body_climb", {}))
        if len(task.get("rises_m", [])) != 2:
            continue
        surfaces = env.surfaces[index]
        expected = [(-4., 0., 0.), (0., .8, .2), (.8, 1.4, .45), (1.4, 4., .45)]
        actual = [(surface.x0, surface.x1, surface.z0) for surface in surfaces]
        require(len(actual) == len(expected) and all(
            abs(left - right) <= 1e-6 for a, b in zip(actual, expected) for left, right in zip(a, b)),
            f"Double-platform geometry differs for {name}: {actual}")
        count = rows.get(name, {}).get("environments_verified", 0) + 1
        rows[name] = {"kind": spec["kind"], "surfaces_x0_x1_z0_m": actual,
                      "environments_verified": count}
    require(any(row["kind"] == "double_step_jump" for row in rows.values()), "Double jump not instantiated")
    require(any(row["kind"] == "body_climb" for row in rows.values()), "Double climb not instantiated")
    return rows


def inspect_native_response(env):
    import torch
    import warp as wp

    config = env.actuator_response.cfg
    require(env.actuator_response.enabled, "Wheel response model is disabled")
    evidence = env.startup_report.get("actuator_response", {})
    require(evidence.get("native_properties_verified") is True, "Native response startup validation missing")
    ids = [env.robot.joint_names.index(name) for name in config["joint_names"]]
    friction = wp.to_torch(env.robot.root_view.get_dof_friction_properties())[:, ids]
    actual = {"armature": wp.to_torch(env.robot.root_view.get_dof_armatures())[:, ids],
              "frictionloss": friction[..., 0], "dynamic_friction": friction[..., 1], "damping": friction[..., 2]}
    readback = {}
    for name, values in actual.items():
        expected = values.new_tensor(config["frictionloss" if name == "dynamic_friction" else name])
        require(values.shape == (COUNT, 6) and torch.isfinite(values).all(), "Invalid native response tensor")
        require(torch.allclose(values, expected.expand_as(values), atol=1e-7, rtol=1e-6),
                f"Native {name} differs from frozen response")
        readback[name] = {"first_environment": values[0].cpu().tolist(),
                          "max_absolute_error": float((values - expected).abs().max())}
    require(all(config["armature"][index] > 0 and config["frictionloss"][index] > 0 for index in (2, 5)),
            "Expected both calibrated wheel response parameters")
    return {"enabled": True, "joint_names": config["joint_names"], "native_readback": readback,
            "time_constant_s": config["time_constant_s"],
            "calibration_scope": "frozen_wheel_only_response_not_global_leg_identification"}


def compare_contact_vectors(reconstructed, matrix, existing, *, threshold_n=.1):
    """Check observed normal-force direction; empty contact is not a pass."""
    import torch

    summed = reconstructed.sum(1)
    matrix_sum = matrix.sum(1)
    observed = torch.maximum(matrix_sum.norm(dim=-1), existing.norm(dim=-1)) > threshold_n
    count = int(observed.sum())
    if not count:
        return {"status": "not_observed", "observed_bodies": 0}
    require(torch.isfinite(reconstructed).all() and torch.isfinite(matrix).all()
            and torch.isfinite(existing).all(), "Native contact sample is nonfinite")
    # The independently accumulated pre-existing all-body view protects against
    # accidentally validating a self-consistent but reversed new contact view.
    error = (summed[observed] - matrix_sum[observed]).norm(dim=-1)
    previous_error = (summed[observed] - existing[observed]).norm(dim=-1)
    tolerance = .01 + .002 * torch.maximum(matrix_sum[observed].norm(dim=-1), existing[observed].norm(dim=-1))
    require((error <= tolerance).all(), "force*normal disagrees with native contact matrix")
    require((previous_error <= tolerance).all(), "Base patch direction disagrees with existing all-body contact view")
    return {"status": "observed_consistent", "observed_bodies": count,
            "matrix_error_max_n": float(error.max()), "existing_view_error_max_n": float(previous_error.max()),
            "upward_samples": int((summed[observed, 2] > threshold_n).sum())}


def inspect_base_contact_direction(env, ended):
    import torch
    import warp as wp

    observations = []
    for ids, view in env.base_support_views:
        force, _, normal, _, counts, starts = map(wp.to_torch, view.get_contact_data(env.dt))
        counts, starts = counts.long(), starts.long()
        require((counts <= 32).all(), "Per-surface native base contact capacity exceeded")
        if not len(force):
            require(not bool(counts.any()), "Nonempty contact counts with empty native force buffer")
            continue
        offsets = torch.arange(32, device=counts.device)
        valid = offsets < counts[..., None]
        indices = (starts[..., None] + offsets).clamp(0, len(force) - 1)
        normal_force = force.reshape(-1, 1)
        vectors = torch.where(valid[..., None], normal[indices] * normal_force[indices], 0.).sum(-2)
        matrix = wp.to_torch(view.get_contact_force_matrix(dt=env.dt))
        require(vectors.shape == matrix.shape, "Native base contact matrix shape differs")
        existing = env.contact_force[ids, env.base_support_id]
        # ChassisEnv may have reset ended lanes after recording diagnostics.
        # Their cleared contact cache is no longer the same physical sample.
        retained = ~ended[ids].bool()
        observations.append(compare_contact_vectors(vectors[retained], matrix[retained], existing[retained]))
    total = sum(row["observed_bodies"] for row in observations)
    return {"status": "observed_consistent" if total else "not_observed", "observed_bodies": total,
            "views": observations, "scope": "native_normal_force_direction_only_not_body_climb_success"}


def finite_observations(observations):
    import torch

    shapes = {"policy": (COUNT, 35), "critic": (COUNT, 81)}
    for name, shape in shapes.items():
        require(observations[name].shape == shape, f"Unexpected {name} dimensions")
        require(torch.isfinite(observations[name]).all(), f"Nonfinite {name} observations")
    return {key: list(value.shape) for key, value in observations.items() if key in shapes}


def step_trace(env, tick, extras, indices):
    import torch

    sensors, task = env.step_sensors, env.step_jump
    beam_finite = torch.isfinite(sensors.range)
    values = []
    for index in indices:
        values.append({"environment": index, "group": env.scene_groups[index],
            "range_m": float(sensors.range[index]) if bool(beam_finite[index]) else None,
            "range_valid": bool(sensors.range_valid[index]), "range_age_s": float(sensors.range_age[index]),
            "conditional_height_m": sensors.height[index].cpu().tolist(),
            "height_valid": sensors.height_valid[index].cpu().tolist(), "support_estimate": bool(sensors.support[index]),
            "requested_count": int(task.request_count[index]), "completed_count": int(task.completed[index].sum())})
    return {"policy_step": tick, "done_environments": int(extras["diagnostics"]["done"].sum()),
            "valid_beam_environments": int(sensors.range_valid.sum()),
            "request_count": int(task.request_count.sum()), "representatives": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "contracts/v6_step_curriculum_training_v1.json")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-steps", type=int, default=5, choices=(3, 4, 5))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "scope": "engineering_20k_scene_probe_no_policy_optimization",
              "formal_updates_consumed": 0, "capability_acceptance": "not_assessed",
              "zero_policy_actions": True, "num_envs": COUNT, "stage": "S5D",
              "source_plan_sha256": digest(args.plan), "script_sha256": digest(Path(__file__)),
              "requested_policy_steps": args.policy_steps}
    launcher, env = None, None
    try:
        config, manifest, control, recipe, seed, receipt = materialize(
            args.plan, args.output, args.expected_plan_sha256)
        report.update(design_preflight=receipt, source_contract_sha256=digest(args.output / "materialized_contract.json"),
                      asset_manifest_sha256=config["asset_manifest_sha256"], seed=seed)
        if args.preflight_only:
            report["status"] = "preflight_only_no_scene"
            return 0
        os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": "cuda:0",
            "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        from wheeled_tasks.chassis.env import ChassisEnv
        started = time.monotonic()
        env = ChassisEnv(config, manifest, control, ROOT, stage_name=recipe["kind"], num_envs=COUNT,
                         device="cuda:0", level=1., seed=seed, block_index=0)
        require(env.num_envs == COUNT, "Scene environment count differs")
        require(env.step_jump is not None and env.step_jump.has_tasks and env.step_climb is not None
                and env.step_climb.has_tasks and env.step_sensors is not None, "Step task modules not active")
        require(env.startup_report["gpu_dynamics_enabled"], "GPU dynamics was not enabled")
        require(env.command_transport is not None
                and env.command_transport.delivery_model == "phase_aware_hold_v1",
                "Phase-aware command transport was not instantiated")
        require(env.dynamics_randomization is not None and env.contact_domain is not None
                and env.perturbations is not None, "S5D robustness modules were not instantiated")
        report.update(status="scene_created", creation_seconds=time.monotonic() - started,
            geometry=inspect_geometry(env), native_response=inspect_native_response(env),
            scene_group_counts=dict(Counter(env.scene_groups)),
            runtime_preflight=env.startup_report.get("runtime_preflight"),
            observation_shapes=finite_observations(env.get_observations()),
            gpu_dynamics_enabled=True, policy_trace=[], base_contact_direction=[])
        representatives = []
        for kind in ("double_step_jump", "body_climb"):
            for index, name in enumerate(env.scene_groups):
                if env.skills.specs.get(name, {}).get("kind") == kind:
                    representatives.append(index)
                    break
        for index, name in enumerate(env.scene_groups):
            if env.skills.specs.get(name, {}).get("step_jump", {}).get("trigger_source") == "laser":
                representatives.append(index)
                break
        write_report(args.output / "report.json", report)
        started = time.monotonic()
        for tick in range(1, args.policy_steps + 1):
            with torch.no_grad():
                observations, reward, _, extras = env.step(torch.zeros(COUNT, 6, device="cuda:0"))
            finite_observations(observations)
            require(torch.isfinite(reward).all(), "Nonfinite probe rewards")
            report["policy_trace"].append(step_trace(env, tick, extras, representatives))
            report["base_contact_direction"].append(inspect_base_contact_direction(env, extras["diagnostics"]["done"]))
        torch.cuda.synchronize()
        timing = env.control_audit.last
        require(timing is not None and timing["policy_steps"] == args.policy_steps
                and timing["physics_steps"] == args.policy_steps * 20
                and timing["pd_updates"] == timing["feedback_refreshes"] == 20
                and timing["target_decodes"] == 1 and timing["physics_dt"] == .001 and timing["policy_dt"] == .02,
                "Actual PD/physics cadence differs from 1 kHz / 50 Hz")
        report.update(status="engineering_runtime_passed", control_timing=timing,
            runtime_seconds=time.monotonic() - started,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            native_contact_direction_status=("observed_consistent" if any(
                row["status"] == "observed_consistent" for row in report["base_contact_direction"]) else "not_observed"))
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_report(args.output / "report.json", report)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    print(json.dumps({key: report.get(key) for key in ("status", "formal_updates_consumed", "native_contact_direction_status")}))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
