#!/usr/bin/env python3
"""Bind successful, model-matched validation receipts into a V6 manifest."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def geometry_functions(path, names):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names}


def seal_cpu_only(bundle, cpu_path, dynamics_path, comparison_path):
    """Freeze a research candidate with genuine CPU receipts and pending PhysX."""
    manifest = load(bundle / "manifest.json")
    if "build_manifest_sha256" in manifest:
        raise ValueError("Refusing to reseal an already frozen candidate")
    if any(manifest.get(key) is not False for key in ("training_ready", "hardware_deployment_ready")):
        raise ValueError("CPU-only sealing requires explicit research-only readiness flags")
    if (any((bundle / name).exists() for name in ("physx_validation.json", "physx_input_manifest.json")) or
            any(word in str(manifest.get("physx_validation", "")).lower()
                for word in ("passed", "completed"))):
        raise ValueError("CPU-only candidate must not inherit a PhysX success receipt")
    for name, digest in manifest["files_sha256"].items():
        if sha(bundle / name) != digest:
            raise ValueError(f"Build input hash mismatch: {name}")
    build_hash = sha(bundle / "manifest.json")
    cpu, dynamics, comparison = map(load, (cpu_path, dynamics_path, comparison_path))
    for receipt, source, binding in (
            (cpu, "validate_v6_asset.py", "bundle_manifest_sha256"),
            (dynamics, "validate_v6_dynamics.py", "bundle_manifest_sha256"),
            (comparison, "compare_stop105_to_r3.py", "candidate_build_manifest_sha256")):
        if (receipt.get("passed") is not True or receipt.get(binding) != build_hash or
                receipt.get("source_sha256") != sha(bundle / "tools" / source)):
            raise ValueError(f"Invalid or stale CPU receipt: {source}")
    if (cpu.get("model_sha256") != sha(bundle / "robot.xml") or
            dynamics.get("model_sha256") != sha(bundle / "robot.xml") or
            comparison.get("parent_manifest_sha256") != manifest.get("parent_manifest_sha256") or
            not manifest.get("parent_manifest_sha256")):
        raise ValueError("CPU receipt model/parent identity mismatch")
    shutil.copyfile(bundle / "manifest.json", bundle / "build_manifest.json")
    for source, name in ((cpu_path, "cpu_validation.json"), (dynamics_path, "dynamics_validation.json"),
                         (comparison_path, "parent_comparison.json")):
        shutil.copyfile(source, bundle / name)
    manifest.update(
        build_manifest_sha256=build_hash,
        candidate_status="cpu_geometry_and_bounded_mujoco_validated_physx_pending",
        cpu_geometry_validation_passed=True,
        dynamics_validation="bounded_MuJoCo_passed_PhysX_pending",
        physx_validation="pending_not_run_for_this_asset",
        training_ready=False, hardware_deployment_ready=False,
        G1_status="pending_hardware_identification_and_effective_stop_zero_separation",
        validation_receipts=["validation.json", "cpu_validation.json", "dynamics_validation.json", "parent_comparison.json"],
        pending_gates=["PhysX_new_limit_and_contact_validation", "G1_hardware_identification",
                       "effective_stop_and_encoder_zero_separation", "full_contact_and_self_collision_review",
                       "motor_spring_thermal_aging_identification", "deployment_and_task_domain_acceptance"],
        files_sha256={str(path.relative_to(bundle)): sha(path)
                      for path in sorted(bundle.rglob("*"))
                      if path.is_file() and "__pycache__" not in path.parts and path != bundle / "manifest.json"})
    save(bundle / "manifest.json", manifest)
    return {"manifest_sha256": sha(bundle / "manifest.json"), "mjcf_sha256": sha(bundle / "robot.xml"),
            "usd_sha256": sha(bundle / "robot.usda"), "model_spec_sha256": sha(bundle / "model_spec.json"),
            "candidate_status": manifest["candidate_status"], "training_ready": False}


def seal(bundle, cpu_path, dynamics_path, physx_path, physx_bundle):
    manifest = load(bundle / "manifest.json")
    if "build_manifest_sha256" in manifest:
        raise ValueError("Refusing to reseal an already frozen candidate")
    for name, digest in manifest["files_sha256"].items():
        if sha(bundle / name) != digest:
            raise ValueError(f"Build input hash mismatch: {name}")
    cpu, dynamics, physx = map(load, (cpu_path, dynamics_path, physx_path))
    assert cpu["passed"] and dynamics["passed"]
    assert cpu["bundle_manifest_sha256"] == sha(bundle / "manifest.json")
    assert cpu["model_sha256"] == dynamics["model_sha256"] == sha(bundle / "robot.xml")
    assert physx["status"] == "completed" and physx["solver"]["loops"] == 6
    assert physx["max_loop_gap_m"] < .001 and physx["max_pin_length_error_m"] < 1e-5
    assert abs(physx["solver"]["mass_kg"] - manifest["total_mass_kg"]) < 1e-5
    assert physx["passive_pose_writes_after_reset"] == physx["kinematic_solver_calls_during_physics"] == 0
    assert physx["asset_manifest_sha256"] == sha(physx_bundle / "manifest.json")
    assert physx["source_sha256"] == sha(bundle / "tools/preview_v6_springs.py")
    # PhysX need not rerun for a MuJoCo-only damping export change. Require exact
    # hashes for every physical input and identical offline FK/IK algorithms.
    relevant = ["robot.usda", "meshes/visuals.usdc", "model_spec.json", "fit_10mpa.json",
                "gas_spring_binding.json", "installation.json", "mechanical_limits.json"]
    for name in relevant:
        assert sha(bundle / name) == sha(physx_bundle / name), name
    for name, functions in (("build_v6_closedchain.py", {"closure_error", "solve_pose"}),
                            ("v6_mechanism.py", {"fk", "point", "transform"})):
        actual = geometry_functions(bundle / "tools" / name, functions)
        assert set(actual) == functions
        assert actual == geometry_functions(physx_bundle / "tools" / name, functions)
    shutil.copyfile(bundle / "manifest.json", bundle / "build_manifest.json")
    shutil.copyfile(physx_bundle / "manifest.json", bundle / "physx_input_manifest.json")
    shutil.copyfile(cpu_path, bundle / "cpu_validation.json")
    shutil.copyfile(dynamics_path, bundle / "dynamics_validation.json")
    physx.update(validated_model_inputs={name: sha(bundle / name) for name in relevant},
                 final_bundle_physical_inputs_identical=True,
                 source_report_sha256=sha(physx_path),
                 validation_scope="single_environment_fixed_base_3s_not_full_contact_or_hardware_acceptance")
    save(bundle / "physx_validation.json", physx)
    script = bundle / "tools" / Path(__file__).name
    if script.resolve() != Path(__file__).resolve():
        shutil.copyfile(__file__, script)
    manifest.update(
        build_manifest_sha256=sha(bundle / "build_manifest.json"),
        cpu_geometry_validation_passed=True,
        dynamics_validation="bounded_MuJoCo_and_fixed_base_PhysX_passed",
        physx_validation="single_environment_fixed_base_3s_passed",
        validation_receipts=["validation.json", "cpu_validation.json", "dynamics_validation.json", "physx_validation.json"],
        pending_gates=["full_contact_and_self_collision_review", "hardware_axis_zero_and_torque_binding",
                       "motor_spring_thermal_aging_identification", "deployment_and_task_domain_acceptance"],
        files_sha256={str(path.relative_to(bundle)): sha(path)
                      for path in sorted(bundle.rglob("*"))
                      if path.is_file() and "__pycache__" not in path.parts and path != bundle / "manifest.json"})
    save(bundle / "manifest.json", manifest)
    return {"manifest_sha256": sha(bundle / "manifest.json"), "mjcf_sha256": sha(bundle / "robot.xml"),
            "usd_sha256": sha(bundle / "robot.usda"), "model_spec_sha256": sha(bundle / "model_spec.json"),
            "training_ready": manifest["training_ready"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "cpu", "dynamics"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    for name in ("physx", "physx-bundle", "comparison"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.cpu_only:
        if args.comparison is None or args.physx is not None or args.physx_bundle is not None:
            parser.error("--cpu-only requires --comparison and excludes --physx/--physx-bundle")
        result = seal_cpu_only(args.bundle, args.cpu, args.dynamics, args.comparison)
    else:
        if args.physx is None or args.physx_bundle is None:
            parser.error("Full sealing requires --physx and --physx-bundle")
        result = seal(args.bundle, args.cpu, args.dynamics, args.physx, args.physx_bundle)
    print(json.dumps(result, indent=2))
