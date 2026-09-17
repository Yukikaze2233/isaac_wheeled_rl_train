#!/usr/bin/env python3
"""Publish a new immutable V5 delivery with matching engine evidence and a ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(candidate, physx_report, mujoco_report, destination, archive):
    manifest = json.loads((candidate / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if sha(candidate / name) != expected:
            raise ValueError(f"Candidate dependency mismatch: {name}")
    physics = json.loads(physx_report.read_text())
    dynamics = json.loads(mujoco_report.read_text())
    if (physics["status"] != "completed" or physics["asset_manifest_sha256"] != sha(candidate / "manifest.json")
            or physics["max_loop_gap_m"] > .001 or physics["simulated_seconds"] < 10):
        raise ValueError("PhysX report does not validate this candidate")
    if dynamics["model_sha256"] != sha(candidate / "robot.xml") or not dynamics["passed"]:
        raise ValueError("MuJoCo report does not validate this candidate")
    if destination.exists() or archive.exists():
        raise FileExistsError("Publishing never overwrites an existing model/archive")
    shutil.copytree(candidate, destination)
    shutil.copyfile(ROOT / "scripts/preview_v5_springs.py", destination / "preview_v5_springs.py")
    shutil.copyfile(ROOT / "docs/V5_SPRING_INSTALLATION.md", destination / "README.md")
    shutil.copyfile(ROOT / "reports/v5_spring_reference/scut_usd_springs.json", destination / "scut_spring_reference.json")
    compact = {k: v for k, v in physics.items() if k != "samples"}
    compact["validated_model_inputs"] = {name: sha(candidate / name) for name in
        ("robot.usda", "model_spec.json", "gas_spring_binding.json", "fit_10mpa.json", "meshes/visuals.usdc")}
    compact["sample_ranges"] = {
        "compression_m": [[min(s["springs"][i]["compression_m"] for s in physics["samples"]),
                           max(s["springs"][i]["compression_m"] for s in physics["samples"])] for i in range(2)],
        "force_n": [[min(s["springs"][i]["force_n"] for s in physics["samples"]),
                     max(s["springs"][i]["force_n"] for s in physics["samples"])] for i in range(2)],
    }
    (destination / "physx_validation.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n")
    compact_mj = {k: v for k, v in dynamics.items() if k != "cases"}
    compact_mj["cases"] = {k: {key: value for key, value in v.items() if key != "samples"} for k, v in dynamics["cases"].items()}
    (destination / "dynamics_validation.json").write_text(json.dumps(compact_mj, indent=2, allow_nan=False) + "\n")
    manifest.update(dynamics_validation="bounded_MuJoCo_and_fixed_base_PhysX_passed",
                    original_candidate_manifest_sha256=sha(candidate / "manifest.json"),
                    spring_force_runtime="preview_v5_springs.py_or_tools/validate_v5_dynamics.py",
                    physics_validation_scope="research_mass_inertia_and_CAD_mount_reference_not_hardware_approval")
    manifest["files_sha256"] = {str(p.relative_to(destination)): sha(p) for p in destination.rglob("*")
                                if p.is_file() and p != destination / "manifest.json"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for file in sorted(destination.rglob("*")):
            if file.is_file():
                output.write(file, "v5_closedchain/" + str(file.relative_to(destination)))
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None:
            raise RuntimeError("Archive integrity failed")
        for path in destination.rglob("*"):
            if path.is_file():
                if check.read("v5_closedchain/" + str(path.relative_to(destination))) != path.read_bytes():
                    raise RuntimeError("Archive and folder differ")
    result = {"directory": str(destination), "archive": str(archive), "archive_bytes": archive.stat().st_size,
              "archive_sha256": sha(archive), "manifest_sha256": sha(destination / "manifest.json")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--physx-report", type=Path, required=True)
    parser.add_argument("--mujoco-report", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT / "model/纯底盘_v5/urdf")
    parser.add_argument("--archive", type=Path, default=ROOT / "model/纯底盘_v5/v5_closedchain_20260918.zip")
    args = parser.parse_args()
    publish(args.candidate, args.physx_report, args.mujoco_report, args.destination, args.archive)
