"""CPU preparation for unscored, one-scene native ONNX playback.

Runtime inputs and expectations are projected separately from the model's
contract and its authenticated, independently frozen full-role design row.
No training contract or acceptance criteria are written by this module.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

SCENARIOS = {
    "flat": "Flat ground / WASD, speed and height",
    "rough": "Rough ground / 20 mm surface variation",
    "slope_up": "Uphill / 15 degrees",
    "slope_down": "Downhill / 15 degrees",
    "cross_slope": "Cross slope / 15 degrees",
    **{f"stairs_{direction}_{mm:02d}mm": f"Three {mm} mm stairs {direction} / ordinary rolling"
       for mm in (5, 10, 20, 40) for direction in ("up", "down")},
    "fallen_left": "Left side 90 degrees / native self-righting then RL",
    "fallen_right": "Right side 90 degrees / native self-righting then RL",
    "fallen_belly": "Prone belly / native self-righting then RL",
    "fallen_back": "Prone back / native self-righting then RL",
    "jump_challenge": "UNTRAINED CHALLENGE / J 6 cm, K 10 cm jump requests",
}
FALLEN_CASES = {
    "fallen_left": "recovery_left_side_90_stand",
    "fallen_right": "recovery_right_side_90_stand",
    "fallen_belly": "recovery_prone_belly_legs_above_stand",
    "fallen_back": "recovery_prone_back_legs_above_stand",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def project_scene(mapping, scenario, seed):
    """Explicit playback changes, independently applied to both source inputs."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown playback scenario: " + scenario)
    result = deepcopy(mapping)
    cases = {case["name"]: case for case in mapping["evaluation"]["cases"]}
    if scenario in FALLEN_CASES:
        # Keep the reviewed geometry-defined birth pose and native FSM.
        case = deepcopy(cases[FALLEN_CASES[scenario]])
    else:
        case = deepcopy(cases["stand_305mm"])
        case.update(name=scenario, reset_seed_key="suite_" + scenario,
                    command=[0., 0., .305], task="survive", episode_seconds=300.,
                    terrain="jump" if scenario == "jump_challenge" else
                    "stairs_down" if scenario.startswith("stairs_down_") else
                    "stairs" if scenario.startswith("stairs_up_") else
                    "flat" if scenario == "flat" else scenario)
        case["skill"] = dict(kind="constant", command=[0., 0., .305], mode=0, sample_amplitude=False)
        case.pop("stationary", None)
        if scenario.startswith("stairs_"):
            rise = int(scenario.rsplit("_", 1)[-1][:-2]) / 1000.
            case["terrain_limits"].update(stair_rise_m=rise,
                stair_heights_m=[rise, 2 * rise, 3 * rise], stair_tread_m=.6)
    name = case["name"]
    evaluation = deepcopy(mapping["evaluation"])
    evaluation.update(protocol_id="chassis_suite_keyboard_unscored", seed=seed,
                      episodes_per_case=1, cases=[case], canonical_case_names=[name])
    for key in ("promotion_case_names", "protected_case_names", "retention_case_names", "quick_case_names"):
        evaluation[key] = []
    # A long flat box floor is the existing native CPU contact-compatible path.
    result.update(evaluation=evaluation, scene_groups=[dict(name=name, fraction=1., terrain=[case["terrain"]])],
        skill_specs={name: deepcopy(case["skill"])}, record_diagnostics=True,
        auto_reset=False, monitor_applied_effort=True, evaluation_exact_cases=True,
        evaluation_long_corridors=True, playback_open_ground=False,
        flat_triangle_mesh=False, flat_half_length_m=44.)
    return result


def prepare_playback(root, contract_path, output, scenario):
    """Authenticate the frozen full role before creating an explicit profile."""
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
    from wheeled_tasks.chassis.preflight import design_preflight

    root, contract_path, output = Path(root), Path(contract_path), Path(output)
    source, manifest = preflight(contract_path)
    if not source.get("requires_design_preflight"):
        raise ValueError("Suite playback requires an independently bound V6 source")
    if tuple(source[key] for key in ("physics_dt", "policy_dt", "actor_dim", "critic_dim", "action_dim", "history_length")) != (.005, .02, 35, 81, 6, 1):
        raise ValueError("Playback requires the frozen 200/50 Hz, 35/81/6, single-frame ABI")
    if (source["position_iterations"], source["velocity_iterations"], source["closure_gap_termination_m"]) != (64, 32, .003):
        raise ValueError("Playback requires 64/32 solver iterations and the 3 mm closure guard")
    binding = source["design_preflight"]
    design_path = root / binding["file"]
    if digest(design_path) != binding["sha256"]:
        raise ValueError("Frozen full-role design identity mismatch")
    design = json.loads(design_path.read_text())
    full_row = design["stages"][binding["stage"]]["roles"]["full"]
    seed_rule = full_row["invocation"]["seed"]
    if seed_rule["rule"] != "fixed":
        raise ValueError("Playback requires the frozen full-role seed")
    seed = seed_rule["value"]
    source_role = deepcopy(source)
    source_role["evaluation"] = select_evaluation_cases(source["evaluation"],
        [case["name"] for case in full_row["contract"]["evaluation"]["cases"]])
    source_role["evaluation"]["seed"] = seed
    source_role = fixed_suite_contract(source_role)
    control = load_chassis_control(root / source["control_math_source"])
    stage = source["enabled_stages"][0]
    original_receipt = design_preflight(source_role, control, root, stage=stage, role="full", seed=seed,
        num_envs=full_row["invocation"]["num_envs"])
    config = project_scene(source_role, scenario, seed)
    expected = deepcopy(full_row)
    expected["contract"] = project_scene(full_row["contract"], scenario, seed)
    expected["invocation"]["num_envs"] = 1
    projection = {key: deepcopy(value) for key, value in design.items() if key != "stages"}
    projection.update(run_scope="interactive_unscored", domain_qualified=None,
        source_contract_sha256=digest(contract_path), source_design_sha256=binding["sha256"],
        source_role="full", scenario=scenario, qualification=None,
        review_basis="Separately projected authenticated runtime and frozen full-role expectations; native scene construction and birth-only resets; no PPO or acceptance credit.",
        allowed_changes=["evaluation", "scene_groups", "skill_specs", "auto_reset", "record_diagnostics",
                         "monitor_applied_effort", "flat_triangle_mesh", "flat_half_length_m", "invocation.num_envs"],
        stages={binding["stage"]: {"roles": {"playback": expected}}})
    output.mkdir(parents=True, exist_ok=False)
    profile_path = output / "playback_expectations.json"
    write_json(profile_path, projection)
    config["design_preflight"] = dict(file=str(profile_path.resolve()), sha256=digest(profile_path), stage=binding["stage"])
    receipt = design_preflight(config, control, root, stage=stage, role="playback", seed=seed, num_envs=1)
    write_json(output / "playback_contract.json", config)
    write_json(output / "CPU_preflight.json", dict(status="CPU_design_preflight_passed_scene_not_run", source_full_role=original_receipt,
        playback=receipt, qualification=None, run_scope="interactive_unscored", scene_run=False))
    return source, config, manifest, control, seed, receipt


def control_ownership(recovery, index=0):
    """Native instantaneous ownership, separated from policy-interval learning mask."""
    from recovery_prefix_support import effective_control_ownership
    from wheeled_tasks.chassis.recovery import RecoveryPhase
    if recovery is None or not bool(recovery.enabled[index]):
        return dict(ownership="ordinary_actor", phase="INACTIVE", motion_released=True, pure_rl=True,
                    policy_learning_mask=True, stable_seconds=0.)
    released = bool(recovery.release_finished[index])
    enabled = released and not bool(recovery.failed[index])
    phase = int(recovery.fsm.phase[index])
    pure = bool(recovery.pure_rl[index])
    hold = bool(recovery.enabled[index] and not recovery.fsm.motion_released[index])
    if not enabled and released:
        owner = "failed_disarmed"
    else:
        owner = effective_control_ownership(phase=phase, enabled=True, release_finished=released,
            motor_enabled=enabled, native_pure_rl=pure, policy_motion_hold=hold)
    return dict(ownership=owner, phase=RecoveryPhase(phase).name,
        motion_released=bool(recovery.fsm.motion_released[index]), pure_rl=pure,
        policy_learning_mask=bool(recovery.learning_mask[index]), stable_seconds=float(recovery.true_stable_peak[index]))
