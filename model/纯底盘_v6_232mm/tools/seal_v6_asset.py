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
    for name in ("bundle", "cpu", "dynamics", "physx", "physx-bundle"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.bundle, args.cpu, args.dynamics, args.physx, args.physx_bundle), indent=2))
