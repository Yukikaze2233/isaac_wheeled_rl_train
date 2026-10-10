"""Failure-path checks run without Isaac, GPU, or a training launch."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis.preflight import (
    GAS_SPRING_MODEL_KINDS, MODULES, REQUIRED_CONTRACT_PATHS, SubstepControlAudit, control_decimation,
    design_preflight, validate_design_contract, validate_runtime_state,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v6_asset_entry_preserves_identity_and_requires_design_binding(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("test_train_entry", ROOT / "scripts/train_chassis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "version", lambda name: "5.5.1")
    bundle = tmp_path / "asset"
    bundle.mkdir()
    manifest = {"model_kind": "v6_gas_spring_closedchain_research", "files_sha256": {}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    config = {"contract_id": "v6-gas-spring-full-stage-research-v1",
        "task_source": "upper_level_commands", "stages": [{"updates": 1}], "total_updates": 1,
        "asset_directory": "asset", "asset_manifest_sha256": module.digest(bundle / "manifest.json"),
        "requires_design_preflight": True, "design_preflight": {"file": "test-only"}}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(config))
    parsed, actual = module.preflight(path)
    assert actual["model_kind"] == "v6_gas_spring_closedchain_research"
    assert parsed["contract_id"].startswith("v6-")
    del config["requires_design_preflight"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="independent design_preflight"):
        module.preflight(path)
    config["contract_id"] = "v5-gas-spring-full-stage-research-v2"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="multi-body asset"):
        module.preflight(path)


@pytest.mark.parametrize("kind,expected", [("v5_gas_spring_closedchain_research", True),
    ("v6_gas_spring_closedchain_research", True), ("coupled_fourbar_research", False)])
def test_production_environment_gas_spring_dispatch(kind, expected):
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/env.py").read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "is_v5" for target in node.targets))
    expression = compile(ast.Expression(assignment.value), "gas_spring_dispatch", "eval")
    assert eval(expression, {"manifest": {"model_kind": kind},
        "GAS_SPRING_MODEL_KINDS": GAS_SPRING_MODEL_KINDS}) is expected


def fixture():
    # Synthetic test design only: not a released G1/asset/curriculum contract.
    expected = {
        "curriculum_stage": "S1", "enabled_stages": ["foundation"],
        "stages": [{"name": "foundation", "updates": 1000}],
        "evaluation": {"seed": 918, "confirmation_seed": 919},
        "physics_dt": .001, "policy_dt": .02,
        "v5_control": {"leg_kp": 160., "leg_kd": 2.5},
        "usb_transport": {"enabled": False}, "command_transport": {"enabled": False},
        "precision_tracking": {"height_weight": 1., "height_sigma_m": .01},
        "signal_perturbations": {"enabled": False},
        "dynamics_randomization": {"enabled": False}, "contact_domain": {"enabled": False},
        "step_assist": {"enabled": False}, "command_reference": {"mode": "test"},
        "reference_reward": {"height_weight": 1.}, "reward_profile": "test",
        "actor_dim": 35, "critic_dim": 81, "action_dim": 6, "history_length": 1,
        "actor_layout": ["test_actor"], "critic_layout": ["test_critic"],
        "asset_manifest_sha256": "test-only", "target_num_envs": 20000,
        "num_steps_per_env": 24, "num_mini_batches": 10, "num_learning_epochs": 5,
        "total_updates": 1000, "evaluation.seed": 918, "evaluation.confirmation_seed": 919,
    }
    config = {key: deepcopy(value) for key, value in expected.items() if "." not in key}
    config["evaluation"] = {"seed": 918, "confirmation_seed": 919}
    wheel = {"kd": .6, "gear_ratio": 15.8, "effort_limit": 4.5}
    control = {"actuators": {"wheel": deepcopy(wheel)}}
    runtime = {"physics_dt": .001, "policy_dt": .02, "decimation": 20,
        "pc_control_dt": .001, "seed": 617, "modules": dict.fromkeys(MODULES, False),
        "leg_kp": 160., "leg_kd": 2.5, "wheel_velocity_p": .6, "precision_tracking": True}
    row = {"contract": expected, "control": {"actuators.wheel": wheel},
        "invocation": {"num_envs": 20000,
            "seed": {"rule": "root_plus_block_index", "root_seed": 617}},
        "runtime": runtime}
    design = {"schema": "chassis-design-expectations-v1", "status": "frozen",
        "stages": {"S1": {"roles": {"train": row}}}}
    return config, design, control


def validate(config, design, control, **kwargs):
    return validate_design_contract(config, design, control,
        stage="S1", role="train", seed=617, num_envs=20000, **kwargs)


@pytest.mark.parametrize("path", REQUIRED_CONTRACT_PATHS)
def test_missing_materialized_fields_fail_even_when_defaults_would_be_legal(path):
    config, design, control = fixture()
    parent = config
    parts = path.split(".")
    for key in parts[:-1]:
        parent = parent[key]
    del parent[parts[-1]]
    with pytest.raises(ValueError, match="Preflight rejected") as caught:
        validate(config, design, control)
    assert "contract." + path in str(caught.value)


@pytest.mark.parametrize("path", REQUIRED_CONTRACT_PATHS)
def test_design_cannot_omit_mandatory_expectations(path):
    config, design, control = fixture()
    del design["stages"]["S1"]["roles"]["train"]["contract"][path]
    with pytest.raises(ValueError, match="explicit expectation"):
        validate(config, design, control)


def test_matching_design_and_derived_block_seed():
    config, design, control = fixture()
    assert validate(config, design, control)["decimation"] == 20
    result = validate_design_contract(config, design, control,
        stage="S1", role="train", seed=620, num_envs=20000, block_index=3)
    assert result["block_index"] == 3
    row = design["stages"]["S1"]["roles"]["train"]
    row["runtime"]["seed"] = {"source": "effective_seed"}
    result = validate_design_contract(config, design, control,
        stage="S1", role="train", seed=620, num_envs=20000, block_index=3)
    assert result["runtime_expected"]["seed"] == 620
    assert row["runtime"]["seed"] == {"source": "effective_seed"}
    with pytest.raises(ValueError, match="invocation.seed"):
        validate(config, design, control, block_index=3)


def test_nested_range_drift_and_new_unfrozen_field_rejected():
    config, design, control = fixture()
    config["signal_perturbations"]["enabled_fraction"] = .5
    with pytest.raises(ValueError, match="signal_perturbations.enabled_fraction"):
        validate(config, design, control)
    del config["signal_perturbations"]["enabled_fraction"]
    control["actuators"]["wheel"]["kd"] = .2
    with pytest.raises(ValueError, match="wheel.kd"):
        validate(config, design, control)


def test_mutually_exclusive_transports_even_if_table_makes_same_error():
    config, design, control = fixture()
    for module in ("usb_transport", "command_transport"):
        config[module]["enabled"] = True
        design["stages"]["S1"]["roles"]["train"]["contract"][module]["enabled"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate(config, design, control)


def test_stage_and_role_have_no_fallback():
    config, design, control = fixture()
    with pytest.raises(ValueError, match="missing design row"):
        validate_design_contract(config, design, control,
            stage="S1", role="baseline", seed=918, num_envs=20000)


def test_interactive_playback_requires_explicit_unscored_scope():
    config, design, control = fixture()
    design["stages"]["S1"]["roles"]["playback"] = deepcopy(
        design["stages"]["S1"]["roles"]["train"])
    row = design["stages"]["S1"]["roles"]["playback"]
    row["invocation"]["seed"] = {"rule": "fixed", "value": 918}
    row["runtime"]["seed"] = 918
    with pytest.raises(ValueError, match="unsupported run role"):
        validate_design_contract(config, design, control,
            stage="S1", role="playback", seed=918, num_envs=20000)
    design["run_scope"] = "interactive_unscored"
    result = validate_design_contract(config, design, control,
        stage="S1", role="playback", seed=918, num_envs=20000)
    assert result["decimation"] == 20


@pytest.mark.parametrize("changed", ["physics", "wheel_response"])
def test_unscored_playback_still_rejects_physical_contract_drift(changed):
    config, design, control = fixture()
    design["run_scope"] = "interactive_unscored"
    design["stages"]["S1"]["roles"]["playback"] = deepcopy(
        design["stages"]["S1"]["roles"]["train"])
    row = design["stages"]["S1"]["roles"]["playback"]
    row["invocation"]["seed"] = {"rule": "fixed", "value": 918}
    row["runtime"]["seed"] = 918
    if changed == "physics":
        config["physics_dt"] = .005
    else:
        control["actuators"]["wheel"]["kd"] = .4
    with pytest.raises(ValueError, match="physics_dt" if changed == "physics" else "wheel.kd"):
        validate_design_contract(config, design, control,
            stage="S1", role="playback", seed=918, num_envs=20000)


def test_pinned_design_table_and_required_binding(tmp_path):
    config, design, control = fixture()
    config["requires_design_preflight"] = True
    with pytest.raises(ValueError, match="binding is missing"):
        design_preflight(config, control, tmp_path,
            stage="foundation", role="train", seed=617, num_envs=20000)
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design))
    config["design_preflight"] = {"file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "stage": "S1"}
    result = design_preflight(config, control, tmp_path,
        stage="foundation", role="train", seed=617, num_envs=20000)
    assert result["environment_stage"] == "foundation"
    path.write_text(json.dumps({**design, "status": "draft"}))
    with pytest.raises(ValueError, match="SHA mismatch"):
        design_preflight(config, control, tmp_path,
            stage="foundation", role="train", seed=617, num_envs=20000)


@pytest.mark.parametrize("physics,policy", [(0, .02), (.003, .02), (.03, .02),
    (-.001, .02), (float("nan"), .02), (.001, float("inf")), (True, .02)])
def test_invalid_clocks_cannot_round_into_valid_decimation(physics, policy):
    with pytest.raises(ValueError):
        control_decimation(physics, policy)


def test_legacy_clock_is_valid_but_not_a_new_design_reference():
    assert control_decimation(.005, .02) == 4
    config, design, control = fixture()
    config["physics_dt"] = .005
    design["stages"]["S1"]["roles"]["train"]["contract"]["physics_dt"] = .005
    with pytest.raises(ValueError, match="reference.physics_dt"):
        validate(config, design, control)


def test_runtime_module_mismatch_fails_and_does_not_claim_effect_probe():
    _, design, _ = fixture()
    expected = design["stages"]["S1"]["roles"]["train"]["runtime"]
    actual = deepcopy(expected)
    receipt = validate_runtime_state(expected, actual)
    assert receipt["functional_transport_precision_randomization_probes"] == "not_run"
    actual["modules"]["command_transport"] = True
    with pytest.raises(ValueError, match="modules.command_transport"):
        validate_runtime_state(expected, actual)


def test_audit_rejects_repeated_pd_without_new_physics_and_wrong_counts():
    audit = SubstepControlAudit(.001, .02)
    audit.begin_policy()
    audit.targets_decoded()
    audit.feedback_control()
    with pytest.raises(RuntimeError, match="refreshed physical state"):
        audit.feedback_control()
    audit.physics_updated(.001)
    with pytest.raises(RuntimeError, match="declared PD/physics decimation"):
        audit.end_policy()


def test_actual_environment_substep_loop_recomputes_pd_from_updated_feedback():
    np = pytest.importorskip("numpy")
    # Execute the production loop, not a replica of its control ordering. The
    # fake solver changes feedback after each step; no Isaac/GPU is initialized.
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/env.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    loop = next(node for node in step.body if isinstance(node, ast.For))
    code = compile(ast.Module(body=[loop], type_ignores=[]), "production_substep_loop", "exec")
    q = np.zeros((1, 6))
    dq = np.zeros_like(q)
    samples, torques = [], []

    def motor_efforts(position, velocity, legs, wheels):
        samples.append(position.copy())
        result = 160. * (legs - position) - 2.5 * velocity
        torques.append(result.copy())
        return result

    def update(dt):
        assert dt == .001
        q[:] += .0001
        dq[:] = .1

    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=q),
        joint_vel=SimpleNamespace(torch=dq)), update=update,
        set_joint_effort_target_index=lambda **kwargs: None, write_data_to_sim=lambda: None)
    env = SimpleNamespace(decimation=20, dt=.001, ids=list(range(6)), robot=robot,
        usb_transport=None, command_transport=None, signal_delay=None, nonstandard_start=None,
        wheel_force_peak_n=None, is_v5=True, skills=None, jump_assist=None, recovery=None,
        actuator_response=SimpleNamespace(enabled=False),
        v5=SimpleNamespace(motor_efforts=motor_efforts, spring_efforts=lambda position: position),
        motor_strength=1., nominal=np.zeros_like(q), spring_ids=[], spring_strength=1.,
        sim=SimpleNamespace(step=lambda **kwargs: None), torque_monitor=None,
        contact_views=[], references=None, contact_peak=np.zeros(1),
        contact_force=np.zeros((1, 2, 3)), wheel_ids=[0, 1],
        control_audit=SubstepControlAudit(.001, .02))
    class Array(np.ndarray):
        def norm(self, dim):
            return np.linalg.norm(self, axis=dim).view(Array)

        def sum(self, axis=None, **kwargs):
            return super().sum(axis=axis, **kwargs)

    env.contact_force = env.contact_force.view(Array)
    env.control_audit.begin_policy()
    env.control_audit.targets_decoded()
    from wheeled_tasks.chassis.root_kinematics import update_robot_data
    exec(code, {"self": env, "torch": np, "legs": np.full((1, 6), .01), "wheels": None,
                "update_robot_data": update_robot_data})
    receipt = env.control_audit.end_policy()
    assert receipt["pd_updates"] == receipt["feedback_refreshes"] == 20
    assert receipt["target_decodes"] == 1
    assert len(samples) == 20
    assert samples[0][0, 0] == 0.
    assert samples[-1][0, 0] == pytest.approx(.0019)
    assert torques[0][0, 0] == pytest.approx(1.6)
    assert torques[-1][0, 0] == pytest.approx(160. * (.01 - .0019) - 2.5 * .1)
