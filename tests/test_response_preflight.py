"""Synthetic response-design/evidence checks; no real calibration is certified."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.actuator_response import ActuatorResponse, JOINT_NAMES, SCHEMA
from wheeled_tasks.chassis.preflight import (
    design_preflight, validate_design_contract, validate_response_evidence, validate_runtime_state,
)
from test_chassis_preflight import fixture as legacy_fixture

ROOT = Path(__file__).resolve().parents[1]


def response_config():
    # Arbitrary synthetic values, not a leg/wheel calibration candidate.
    return {"enabled": True, "schema": SCHEMA, "version": 1,
            "joint_names": list(JOINT_NAMES), "gain": 1.,
            "armature": [.001, .002, .003, .004, .005, .006],
            "frictionloss": [.01, .02, .03, .04, .05, .06],
            "damping": [.001, .002, .003, .004, .005, .006],
            "time_constant_s": [.002, .003, 0., .004, .005, 0.]}


def fixture():
    config, design, control = legacy_fixture()
    config.update(actuator_response=response_config(),
                  actuator_response_evidence={"file": "synthetic.json", "sha256": "test-only"},
                  response_migration="v6_actuator_response_v1")
    row = design["stages"]["S1"]["roles"]["train"]
    for field in ("actuator_response", "actuator_response_evidence", "response_migration"):
        row["contract"][field] = deepcopy(config[field])
    row["runtime"]["modules"]["actuator_response"] = True
    row["runtime"]["actuator_response"] = deepcopy(config["actuator_response"])
    return config, design, control


def validate(config, design, control):
    return validate_design_contract(config, design, control,
        stage="S1", role="train", seed=617, num_envs=20000)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal_synthetic_evidence(tmp_path, config):
    result_path = tmp_path / "synthetic-replay.json"
    result_path.write_text(json.dumps({"synthetic_test_only": True, "response_error": .01}))
    evidence = {"schema": "actuator-response-evidence-v1", "status": "validated_for_research",
                "hardware_release": False, "scope": "SYNTHETIC_TEST_ONLY_NOT_REAL_CALIBRATION",
                "actuator_response": deepcopy(config["actuator_response"]),
                "files_sha256": {result_path.name: digest(result_path)}}
    evidence_path = tmp_path / "synthetic-evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    config["actuator_response_evidence"] = {"file": evidence_path.name, "sha256": digest(evidence_path)}
    return evidence, evidence_path, result_path


def write_design(tmp_path, config, design):
    row = design["stages"]["S1"]["roles"]["train"]
    row["contract"]["actuator_response_evidence"] = deepcopy(config["actuator_response_evidence"])
    path = tmp_path / "synthetic-design.json"
    path.write_text(json.dumps(design))
    config["design_preflight"] = {"file": path.name, "sha256": digest(path), "stage": "S1"}


def test_matching_synthetic_design_and_authenticated_test_evidence(tmp_path):
    config, design, control = fixture()
    seal_synthetic_evidence(tmp_path, config)
    write_design(tmp_path, config, design)
    result = design_preflight(config, control, tmp_path,
        stage="foundation", role="train", seed=617, num_envs=20000)
    assert result["status"] == "static_passed"
    assert result["actuator_response_evidence"]["status"] == "authenticated"
    assert result["actuator_response_evidence"]["scope"] == "SYNTHETIC_TEST_ONLY_NOT_REAL_CALIBRATION"
    assert result["actuator_response_evidence"]["validated_file_count"] == 1


def test_enabled_response_not_in_independent_design_is_rejected():
    config, design, control = legacy_fixture()
    config["actuator_response"] = response_config()
    with pytest.raises(ValueError, match="explicit response expectation"):
        validate(config, design, control)


def test_enabled_response_cannot_bypass_the_entire_design_by_omitting_binding(tmp_path):
    config, _, control = fixture()
    config.pop("requires_design_preflight", None)
    config.pop("design_preflight", None)
    with pytest.raises(ValueError, match="binding"):
        design_preflight(config, control, tmp_path,
            stage="foundation", role="train", seed=617, num_envs=20000)


@pytest.mark.parametrize("field", ["armature", "frictionloss", "damping", "time_constant_s"])
def test_effective_parameter_drift_rejected_against_frozen_design(field):
    config, design, control = fixture()
    config["actuator_response"][field][2] += .01
    with pytest.raises(ValueError, match=field):
        validate(config, design, control)


@pytest.mark.parametrize("field", ["schema", "version", "joint_names", "gain", "armature",
                                   "frictionloss", "damping", "time_constant_s", "enabled"])
def test_response_fields_never_fall_back_to_runtime_defaults(field):
    config, design, control = fixture()
    del config["actuator_response"][field]
    with pytest.raises(ValueError):
        validate(config, design, control)


@pytest.mark.parametrize("field", ["actuator_response", "actuator_response_evidence", "response_migration"])
def test_independent_design_must_declare_each_response_binding(field):
    config, design, control = fixture()
    del design["stages"]["S1"]["roles"]["train"]["contract"][field]
    with pytest.raises(ValueError, match="explicit response expectation"):
        validate(config, design, control)


@pytest.mark.parametrize("field", ["actuator_response", "modules"])
def test_design_requires_runtime_parameters_and_enablement(field):
    config, design, control = fixture()
    runtime = design["stages"]["S1"]["roles"]["train"]["runtime"]
    if field == "modules":
        del runtime["modules"]["actuator_response"]
    else:
        del runtime[field]
    with pytest.raises(ValueError, match="design.runtime"):
        validate(config, design, control)


@pytest.mark.parametrize("mutation", ["missing", "disabled", "parameter"])
def test_runtime_parameters_cannot_silently_differ(mutation):
    config, design, control = fixture()
    expected = validate(config, design, control)["runtime_expected"]
    actual = deepcopy(expected)
    if mutation == "missing":
        del actual["actuator_response"]
    elif mutation == "disabled":
        actual["modules"]["actuator_response"] = False
    else:
        actual["actuator_response"]["armature"][5] += .001
    with pytest.raises(ValueError, match="actuator_response"):
        validate_runtime_state(expected, actual)


def test_evidence_file_hash_mismatch_rejected(tmp_path):
    config, _, _ = fixture()
    _, path, _ = seal_synthetic_evidence(tmp_path, config)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="evidence SHA mismatch"):
        validate_response_evidence(config, tmp_path)


def test_evidence_parameter_mismatch_rejected_even_after_rehash(tmp_path):
    config, _, _ = fixture()
    evidence, path, _ = seal_synthetic_evidence(tmp_path, config)
    evidence["actuator_response"]["damping"][4] = 0.
    path.write_text(json.dumps(evidence))
    config["actuator_response_evidence"]["sha256"] = digest(path)
    with pytest.raises(ValueError, match="parameters differ"):
        validate_response_evidence(config, tmp_path)


def test_referenced_result_tampering_is_rejected(tmp_path):
    config, _, _ = fixture()
    _, _, result = seal_synthetic_evidence(tmp_path, config)
    result.write_text(json.dumps({"synthetic_test_only": True, "response_error": 10.}))
    with pytest.raises(ValueError, match="result SHA mismatch"):
        validate_response_evidence(config, tmp_path)


@pytest.mark.parametrize("mutation", [{"status": "candidate"}, {"hardware_release": True},
                                      {"schema": "unknown"}, {"files_sha256": {}}])
def test_unreviewed_or_unsealed_evidence_cannot_authenticate(mutation, tmp_path):
    config, _, _ = fixture()
    evidence, path, _ = seal_synthetic_evidence(tmp_path, config)
    evidence.update(mutation)
    path.write_text(json.dumps(evidence))
    config["actuator_response_evidence"]["sha256"] = digest(path)
    with pytest.raises(ValueError):
        validate_response_evidence(config, tmp_path)


def test_legacy_without_response_remains_unchanged(tmp_path):
    config, design, control = legacy_fixture()
    assert validate(config, design, control)["status"] == "static_passed"
    assert design_preflight(config, control, tmp_path,
        stage="foundation", role="train", seed=617, num_envs=20000) is None


def native_readback_fixture():
    """Exercise production readback code with fake native values and stale cache."""
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/env.py").read_text())
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    block = next(n for n in init.body if isinstance(n, ast.If) and any(
        isinstance(child, ast.Assign) and any(isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute) and target.value.attr == "startup_report"
            and isinstance(target.slice, ast.Constant) and target.slice.value == "actuator_response"
            for target in child.targets) for child in ast.walk(n)))
    code = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), "native_response_readback", "exec")
    cfg = response_config()
    ids = [6, 1, 4, 2, 7, 3]  # Solver order deliberately differs from canonical.
    armature, friction = torch.zeros(3, 8), torch.zeros(3, 8, 3)
    armature[:, ids] = torch.tensor(cfg["armature"])
    for channel, field in enumerate(("frictionloss", "frictionloss", "damping")):
        friction[:, ids, channel] = torch.tensor(cfg[field])
    stale = SimpleNamespace(torch=torch.full((3, 8), 123.))
    solver_names = [f"passive_{i}" for i in range(8)]
    for name, index in zip(JOINT_NAMES, ids):
        solver_names[index] = name
    robot = SimpleNamespace(joint_names=solver_names,
        root_view=SimpleNamespace(get_dof_armatures=lambda: armature,
        get_dof_friction_properties=lambda: friction), data=SimpleNamespace(
        joint_friction_coeff=stale, joint_dynamic_friction_coeff=stale, joint_viscous_friction_coeff=stale))
    env = SimpleNamespace(robot=robot, ids=ids, device="cpu", num_envs=3,
                          actuator_response=ActuatorResponse(3, "cpu", .001, cfg), startup_report={})
    scope = {"self": env, "torch": torch, "wp": SimpleNamespace(to_torch=lambda value: value),
             "manifest": {"control_joint_names": list(JOINT_NAMES)}, "num_envs": 3}
    return code, scope, armature, friction, ids


def test_native_readback_checks_actual_solver_values_despite_stale_cache():
    code, scope, _, _, _ = native_readback_fixture()
    exec(code, scope)
    assert scope["self"].startup_report["actuator_response"]["native_properties_verified"]


@pytest.mark.parametrize("field", ["armature", "static", "dynamic", "viscous"])
def test_native_readback_rejects_nonfirst_environment_mismatch(field):
    code, scope, armature, friction, ids = native_readback_fixture()
    if field == "armature":
        armature[2, ids[4]] += .05
    else:
        friction[2, ids[4], {"static": 0, "dynamic": 1, "viscous": 2}[field]] += .05
    with pytest.raises(RuntimeError, match="readback differs"):
        exec(code, scope)


def test_native_readback_rejects_noncanonical_asset_order():
    code, scope, _, _, _ = native_readback_fixture()
    scope["manifest"]["control_joint_names"][0:2] = reversed(scope["manifest"]["control_joint_names"][0:2])
    with pytest.raises(RuntimeError, match="axis order"):
        exec(code, scope)


def test_environment_drive_envelope_clips_retained_response_state():
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/env.py").read_text())
    step = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "step")
    block = next(n for n in ast.walk(step) if isinstance(n, ast.If)
        and isinstance(n.test, ast.Attribute) and n.test.attr == "enabled"
        and isinstance(n.test.value, ast.Attribute) and n.test.value.attr == "actuator_response")
    code = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), "response_effort_hook", "exec")
    response = ActuatorResponse(2, "cpu", .001, response_config())
    bounds = torch.tensor([[1., 2., .5, 3., 4., .8]]).expand(2, -1).clone()
    env = SimpleNamespace(actuator_response=response, torque=torch.full((2, 6), 40.),
                          v5=SimpleNamespace(current_motor_bounds=bounds))
    exec(code, {"self": env})
    assert torch.equal(env.torque, response.state)
    assert torch.all(abs(response.state) <= bounds)
    before = response.state.clone()
    env.torque = torch.zeros(2, 6)
    exec(code, {"self": env})
    torch.testing.assert_close(response.state, before * (1 - response.alpha))
