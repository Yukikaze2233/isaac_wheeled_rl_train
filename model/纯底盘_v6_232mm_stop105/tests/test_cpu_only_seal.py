"""Simulator-free acceptance/rejection checks for the stop105 CPU-only seal.

Small synthetic bundles exercise receipt binding and state transitions only.
They do not stand in for the real CPU geometry or MuJoCo validation receipts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SEAL_PATH = TOOLS / "seal_v6_asset.py"
PARENT_SHA = "1" * 64
BAD_SHA = "0" * 64
RECEIPT_SCRIPTS = {
    "cpu": "validate_v6_asset.py",
    "dynamics": "validate_v6_dynamics.py",
    "comparison": "compare_stop105_to_r3.py",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot(bundle):
    return {str(p.relative_to(bundle)): p.read_bytes()
            for p in bundle.rglob("*") if p.is_file()}


@pytest.fixture
def sealer():
    spec = importlib.util.spec_from_file_location("stop105_seal_test", SEAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "tools").mkdir(parents=True)
    for name, text in {
        "robot.xml": '<mujoco model="test_stop105"/>\n',
        "robot.usda": '#usda 1.0\n',
        "model_spec.json": '{}\n',
        "fit_10mpa.json": '{}\n',
        "validation.json": '{"passed": true}\n',
    }.items():
        (bundle / name).write_text(text)
    for name in RECEIPT_SCRIPTS.values():
        (bundle / "tools" / name).write_text(f'# synthetic fixture for {name}\n')
    (bundle / "tools/seal_v6_asset.py").write_bytes(SEAL_PATH.read_bytes())
    manifest = {
        "schema_version": 1,
        "model_kind": "v6_gas_spring_closedchain_research",
        "coordinate_schema": "v6_download_link_frames_v1",
        "asset_variant": "v6_232mm_stop105",
        "parent_manifest_sha256": PARENT_SHA,
        "total_mass_kg": 13.886,
        "knee_inner_limits_deg": [40.0, 105.0],
        "static_validation_passed": True,
        "training_ready": False,
        "hardware_deployment_ready": False,
        "dynamics_validation": "pending",
        "physx_validation": "pending_GPU_not_run",
        "pending_gates": ["CPU_domain_and_virtual_work", "bounded_MuJoCo_dynamics", "PhysX_comparison"],
        "files_sha256": {str(p.relative_to(bundle)): digest(p)
                         for p in sorted(bundle.rglob("*")) if p.is_file()},
    }
    save(bundle / "manifest.json", manifest)
    paths = {}
    for name, script in RECEIPT_SCRIPTS.items():
        receipt = {"passed": True, "source_sha256": digest(bundle / "tools" / script)}
        if name == "comparison":
            receipt.update(candidate_build_manifest_sha256=digest(bundle / "manifest.json"),
                           parent_manifest_sha256=PARENT_SHA)
        else:
            receipt.update(bundle_manifest_sha256=digest(bundle / "manifest.json"),
                           model_sha256=digest(bundle / "robot.xml"))
        if name == "dynamics":
            receipt.update(passive_pose_writes_after_reset=0, per_step_kinematic_solver_calls=0)
        paths[name] = tmp_path / f"{name}.json"
        save(paths[name], receipt)
    return bundle, paths


def seal(sealer, candidate):
    bundle, paths = candidate
    return sealer.seal_cpu_only(bundle, paths["cpu"], paths["dynamics"], paths["comparison"])


def change_manifest(candidate, mutate):
    """Keep receipt identity correct so the intended state check is isolated."""
    bundle, paths = candidate
    manifest = read(bundle / "manifest.json")
    mutate(manifest)
    save(bundle / "manifest.json", manifest)
    for name, path in paths.items():
        receipt = read(path)
        key = "candidate_build_manifest_sha256" if name == "comparison" else "bundle_manifest_sha256"
        receipt[key] = digest(bundle / "manifest.json")
        save(path, receipt)


def reject_unchanged(sealer, candidate, exceptions=ValueError):
    before = snapshot(candidate[0])
    with pytest.raises(exceptions):
        seal(sealer, candidate)
    assert snapshot(candidate[0]) == before, "A rejected seal must not partially freeze the bundle"


def test_cpu_only_success_keeps_physx_pending_and_readiness_false(sealer, candidate):
    bundle, paths = candidate
    original_manifest = (bundle / "manifest.json").read_bytes()
    seal(sealer, candidate)
    manifest = read(bundle / "manifest.json")
    assert manifest["training_ready"] is False
    assert manifest["hardware_deployment_ready"] is False
    assert manifest["cpu_geometry_validation_passed"] is True
    assert "pending" in manifest["physx_validation"].lower()
    assert "mujoco" in manifest["dynamics_validation"].lower()
    assert "physx_passed" not in manifest["dynamics_validation"].lower()
    assert any("physx" in name.lower() for name in manifest["pending_gates"])
    assert not (bundle / "physx_validation.json").exists()
    assert not (bundle / "physx_input_manifest.json").exists()
    assert all("physx" not in name.lower() for name in manifest["validation_receipts"])
    assert (bundle / "build_manifest.json").read_bytes() == original_manifest
    assert manifest["build_manifest_sha256"] == digest(bundle / "build_manifest.json")
    for name, expected in manifest["files_sha256"].items():
        assert digest(bundle / name) == expected, name
    published_digests = set(manifest["files_sha256"].values())
    assert all(digest(path) in published_digests for path in paths.values())
    assert manifest["parent_manifest_sha256"] == PARENT_SHA


@pytest.mark.parametrize("name,field", [
    ("cpu", "bundle_manifest_sha256"), ("cpu", "model_sha256"), ("cpu", "source_sha256"),
    ("dynamics", "bundle_manifest_sha256"), ("dynamics", "model_sha256"), ("dynamics", "source_sha256"),
    ("comparison", "candidate_build_manifest_sha256"),
    ("comparison", "parent_manifest_sha256"), ("comparison", "source_sha256"),
])
@pytest.mark.parametrize("missing", [False, True])
def test_missing_or_wrong_receipt_identity_is_rejected(sealer, candidate, name, field, missing):
    path = candidate[1][name]
    receipt = read(path)
    if missing:
        del receipt[field]
    else:
        receipt[field] = BAD_SHA
    save(path, receipt)
    reject_unchanged(sealer, candidate)


@pytest.mark.parametrize("name", RECEIPT_SCRIPTS)
@pytest.mark.parametrize("status", [False, None, "true", 1])
def test_failed_missing_or_nonboolean_receipt_success_is_rejected(sealer, candidate, name, status):
    path = candidate[1][name]
    receipt = read(path)
    if status is None:
        del receipt["passed"]
    else:
        receipt["passed"] = status
    save(path, receipt)
    reject_unchanged(sealer, candidate)


@pytest.mark.parametrize("name", RECEIPT_SCRIPTS)
def test_missing_receipt_file_is_rejected(sealer, candidate, name):
    candidate[1][name].unlink()
    reject_unchanged(sealer, candidate, (ValueError, FileNotFoundError))


@pytest.mark.parametrize("name", ["robot.xml", "fit_10mpa.json", "model_spec.json"])
def test_changed_build_input_is_rejected(sealer, candidate, name):
    path = candidate[0] / name
    path.write_bytes(path.read_bytes() + b" \n")
    reject_unchanged(sealer, candidate)


@pytest.mark.parametrize("name", ["physx_validation.json", "physx_input_manifest.json"])
def test_stale_physx_artifact_is_rejected(sealer, candidate, name):
    save(candidate[0] / name, {"status": "completed"})
    reject_unchanged(sealer, candidate)


def test_stale_physx_success_state_is_rejected(sealer, candidate):
    change_manifest(candidate, lambda m: m.update(physx_validation="single_environment_fixed_base_3s_passed"))
    reject_unchanged(sealer, candidate)


@pytest.mark.parametrize("key", ["training_ready", "hardware_deployment_ready"])
@pytest.mark.parametrize("state", [True, None, "false", 0])
def test_readiness_must_be_explicit_boolean_false(sealer, candidate, key, state):
    def mutate(manifest):
        if state is None:
            del manifest[key]
        else:
            manifest[key] = state
    change_manifest(candidate, mutate)
    reject_unchanged(sealer, candidate)


def test_already_frozen_candidate_cannot_be_resealed(sealer, candidate):
    seal(sealer, candidate)
    reject_unchanged(sealer, candidate)


def test_identity_rejection_survives_python_optimization(candidate):
    bundle, paths = candidate
    receipt = read(paths["cpu"])
    receipt["model_sha256"] = BAD_SHA
    save(paths["cpu"], receipt)
    before = snapshot(bundle)
    code = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('optimized_seal', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.seal_cpu_only(*(Path(p) for p in sys.argv[2:]))
except ValueError:
    sys.exit(0)
sys.exit(3)
"""
    result = subprocess.run([sys.executable, "-O", "-c", code, str(SEAL_PATH), str(bundle),
                             str(paths["cpu"]), str(paths["dynamics"]), str(paths["comparison"])],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot(bundle) == before
