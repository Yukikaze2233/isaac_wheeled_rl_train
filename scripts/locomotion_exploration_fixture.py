"""CPU authentication/projection for a same-PT clean locomotion diagnostic.

The calling entry supplies root. No runner, optimizer, AppLauncher, simulator,
GPU initialization or network operation is created by authentication/build.
assert_clean_runtime reads an already-created native environment at its birth.
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import sys

PT_SHA = "78dc24240038910b3844cc609a26d259075239b137c008ee038de7a76b4963e7"
CONTRACT_SHA = "f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3"
SEAL_SHA = "854566d4c71f7294efca35d15c00a54d9845a65db5ab8b196b80f61ec51fdfbf"
AGENT_SHA = "cecf04ea62efbfb7cfa2b6837226df736843fa69f4cbe2c233ecb97a1af31f27"
ASSET_SHA = "875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba"
CONTROL_SHA = "8073540d1cec0b32c6216699863b3ef53064a2d385b7110abe291d9da0e9b1ac"
POLICY_ORDER = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3")
ACTIVE_ORDER = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
SEED, NUM_ENVS, REPLICAS, HORIZON_S = 190619, 12, 4, 20.
TERRAIN_LEVEL = 1.
CONSTRUCTOR_OVERRIDES = {"stage": "speed", "num_envs": NUM_ENVS, "terrain_level": TERRAIN_LEVEL, "seed": SEED}
CASE_PROFILES = {"stand_305mm": 0., "forward_0p5": .5, "backward_0p5": -.5}
DELETED_BINDINGS = ("response_migration", "recovery_training", "recovery_takeover", "step_jump_task", "jump_landing")
ABSENT_MODULES = ("usb_transport", "command_transport", "signal_delay", "perturbations",
                  "dynamics_randomization", "contact_domain", "step_assist", "jump_assist",
                  "step_jump", "step_climb", "step_sensors", "recovery", "jump_landing")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, data):
    path = Path(path)
    require(not path.exists(), f"Refusing to replace diagnostic input/output: {path}")
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def native_modules(root):
    root = Path(root).resolve()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    names = ("control_contract", "preflight")
    modules = {}
    for name in names:
        module = importlib.import_module("wheeled_tasks.chassis." + name)
        require(Path(module.__file__).resolve() == root / "src/wheeled_tasks/chassis" / (name + ".py"),
                "Native CPU modules belong to a different root; use a fresh process")
        modules[name] = module
    return modules


def asset_inputs(root, source):
    root = Path(root).resolve()
    directory = (root / source["asset_directory"]).resolve()
    require(directory.is_relative_to(root), "Asset directory must belong to supplied root")
    require(source["asset_manifest_sha256"] == ASSET_SHA and digest(directory / "manifest.json") == ASSET_SHA,
            "Closed-chain asset manifest SHA mismatch")
    manifest = read_json(directory / "manifest.json")
    require(manifest["model_kind"] == "v6_gas_spring_closedchain_research", "Wrong mechanical asset kind")
    for name, expected in manifest["files_sha256"].items():
        target = (directory / name).resolve()
        require(target.is_relative_to(directory) and digest(target) == expected, f"Asset dependency SHA mismatch: {name}")
    control_path = (root / source["control_math_source"]).resolve()
    require(control_path.is_relative_to(root) and source["control_math_sha256"] == CONTROL_SHA
            and digest(control_path) == CONTROL_SHA, "Control math SHA mismatch")
    modules = native_modules(root)
    control = modules["control_contract"].load_chassis_control(control_path)
    return manifest, control, modules


def source_design(root, source, control, modules):
    binding = source["design_preflight"]
    path = Path(root) / binding["file"]
    require(digest(path) == binding["sha256"], "Source independent design SHA mismatch")
    design = read_json(path)
    row = design["stages"][binding["stage"]]["roles"]["train"]
    rule = row["invocation"]["seed"]
    block = binding.get("block_index", 0)
    if rule["rule"] == "root_plus_block_index":
        seed = rule["root_seed"] + block
    elif rule["rule"] == "fixed":
        seed = rule["value"]
    else:
        raise ValueError("Unrecognized independent source seed rule")
    require(source["enabled_stages"] == ["speed"], "Expected one frozen speed worker stage")
    receipt = modules["preflight"].design_preflight(source, control, Path(root), stage="speed", role="train",
        seed=seed, num_envs=row["invocation"]["num_envs"], block_index=block)
    require(receipt["status"] == "static_passed", "Source train design preflight failed")
    return design, row, receipt


def tensor_audit(value, torch):
    count = elements = 0
    def visit(item):
        nonlocal count, elements
        if isinstance(item, torch.Tensor):
            require(item.device.type == "cpu" and bool(torch.isfinite(item).all()), "Checkpoint tensor is nonfinite/not CPU")
            count += 1
            elements += item.numel()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, float):
            require(math.isfinite(item), "Checkpoint scalar is nonfinite")
    visit(value)
    require(count > 0, "Checkpoint state contains no tensors")
    return {"finite": True, "tensor_count": count, "elements": elements}


def authenticate_inputs(root, input_dir):
    """Return source/manifest/control/factory/actor state and a JSON-safe receipt.

    All four file hashes are checked before PT deserialization. The native CPU
    actor/critic strict loads authenticate ABI/std; Adam is only read/audited.
    No optimizer is constructed, restored or stepped.
    """
    from importlib.metadata import version
    root, input_dir = Path(root).resolve(), Path(input_dir).resolve()
    files = {"checkpoint": input_dir / "model.pt", "contract": input_dir / "contract.json",
             "seal": input_dir / "model.checkpoint.json", "agent_config": input_dir / "agent_config.json"}
    pinned = {"checkpoint": PT_SHA, "contract": CONTRACT_SHA, "seal": SEAL_SHA, "agent_config": AGENT_SHA}
    for name, path in files.items():
        require(path.is_file() and digest(path) == pinned[name], f"Pinned {name} SHA mismatch/missing file")
    source, seal, agent = (read_json(files[name]) for name in ("contract", "seal", "agent_config"))
    require(version("rsl-rl-lib") == "5.5.1", "Expected native RSL-RL 5.5.1 factory")
    require((source["physics_dt"], source["policy_dt"], source["actor_dim"], source["critic_dim"], source["action_dim"],
             source["actor_frame_dim"], source["history_length"]) == (.005, .02, 35, 81, 6, 35, 1), "Observation/control ABI mismatch")
    require(tuple(source["policy_action_order"]) == POLICY_ORDER and tuple(source["v5_control"]["active_order"]) == ACTIVE_ORDER,
            "Policy/control axis order mismatch")
    require(source["actor_observation_source"] == "scut35_encoders_imu_commands", "Actor observation source changed")
    require(agent["obs_groups"] == {"actor": ["policy"], "critic": ["critic"]}, "Agent observation groups changed")
    architecture = source["policy_architecture"]
    for role in ("actor", "critic"):
        require(agent[role]["class_name"] == "MLPModel"
                and agent[role]["hidden_dims"] == architecture[role + "_hidden_dims"]
                and agent[role]["activation"] == architecture["activation"]
                and agent[role]["obs_normalization"] is False, f"Native {role} factory ABI mismatch")
    require(agent["actor"]["distribution_cfg"] == {"class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"},
            "Native scalar Gaussian factory changed")
    manifest, control, modules = asset_inputs(root, source)
    _, _, original_preflight = source_design(root, source, control, modules)
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    saved = torch.load(files["checkpoint"], map_location="cpu", weights_only=True)
    require(isinstance(saved, dict) and all(isinstance(saved.get(k), dict) and saved[k]
            for k in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "infos")), "Full PT state is required")
    infos = saved["infos"]
    require(seal["schema"] == "chassis-learning-checkpoint-v1" and seal["sealed"] is True
            and seal["checkpoint_sha256"] == PT_SHA and seal["contract_sha256"] == infos["contract_sha256"] == CONTRACT_SHA,
            "Checkpoint seal/infos identity mismatch")
    clocks = {"learning_updates": "successful_updates_total", "learning_lineage_updates": "learning_lineage_updates",
              "stage_actor_update": "stage_actor_update", "global_actor_update": "global_actor_update",
              "global_actor_update_offset": "global_actor_update_offset", "actor_updates_consumed": "actor_updates_consumed",
              "task_actor_update": "task_actor_update"}
    for seal_key, info_key in clocks.items():
        require(type(seal[seal_key]) is int and seal[seal_key] >= 0 and infos.get(info_key) == seal[seal_key],
                f"Checkpoint seal/infos clock mismatch: {seal_key}")
    require(seal["learning_updates"] == 24500 and seal["task_actor_update"] == 4185
            and seal["task_campaign"] == infos.get("task_campaign") == source["task_campaign"], "Wrong campaign/update PT")
    for key in ("asset_manifest_sha256", "control_math_sha256"):
        require(infos.get(key) == source[key], f"Checkpoint infos ABI mismatch: {key}")
    audits = {key: tensor_audit(saved[key], torch) for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")}
    std = saved["actor_state_dict"]["distribution.std_param"]
    require(std.shape == (6,) and std.is_floating_point() and bool(torch.isfinite(std).all()) and bool((std > 0).all()),
            "Saved scalar-space actor std is invalid")
    obs = TensorDict({"policy": torch.zeros(1, 35), "critic": torch.zeros(1, 81)}, batch_size=[1])
    native = {}
    # Native construction consumes random initializer values; preserve the
    # caller's CPU RNG. devices=[] cannot create a CUDA context.
    with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
        for role, width in (("actor", 6), ("critic", 1)):
            args = deepcopy(agent[role]); args.pop("class_name")
            model = MLPModel(obs, agent["obs_groups"], role, width, **args)
            model.load_state_dict(saved[role + "_state_dict"], strict=True)
            require(all(torch.equal(v, saved[role + "_state_dict"][k]) for k, v in model.state_dict().items()),
                    f"Native strict {role} load changed a tensor")
            native[role + "_strict_exact"] = True
            if role == "actor":
                with torch.inference_mode():
                    mu = model(obs, stochastic_output=False)
                    model.distribution.update(mu)
                    actual_std = model.output_std.detach().clone()
                require(type(model.distribution).__name__ == "GaussianDistribution" and model.distribution.std_type == "scalar"
                        and actual_std.shape[-1] == 6 and bool(torch.isfinite(actual_std).all()) and bool((actual_std > 0).all()),
                        "Native effective actor std is invalid")
                native.update(raw_std_policy_order=std.tolist(), effective_std_policy_order=actual_std.reshape(-1, 6)[0].tolist(),
                              native_std_range=list(model.distribution.std_range))
    require(all(digest(files[k]) == h for k, h in pinned.items()), "Input changed during CPU authentication")
    receipt = {"schema": "locomotion-exploration-input-authentication-v1", "status": "authenticated_CPU_only",
        "files": {k: {"path": str(v), "sha256": pinned[k], "bytes": v.stat().st_size} for k, v in files.items()},
        "clocks": {k: seal[k] for k in clocks}, "native_cpu_factory": native, "tensor_audits": audits,
        "source_design_preflight": original_preflight, "qualification": False, "optimizer_constructed": False,
        "optimizer_steps": 0, "formal_updates_consumed": 0, "PT_exact_restore": False,
        "agent_config_scope": "pinned factory only; historical resume flag is not a training invocation"}
    return {"source": source, "manifest": manifest, "control": control, "agent_config": agent,
            "actor_state_dict": saved["actor_state_dict"], "receipt": receipt}


def cases():
    return [{"name": name, "terrain": "flat", "command": [0., 0., .305], "episode_seconds": HORIZON_S,
             "reset_seed_key": "quality_common_reset", "plateau_vx_m_s": speed,
             "skill": {"kind": "constant", "command": [0., 0., .305], "mode": 0}}
            for name, speed in CASE_PROFILES.items()]


def clean_overrides(mapping):
    """Apply explicit constants separately to source and independent design row."""
    result = deepcopy(mapping)
    for key in DELETED_BINDINGS:
        result.pop(key, None)
    # The existing response validator requires this expectation key even for
    # inference. Explicit None retains the response receipt without claiming a
    # learning-state migration or requiring the recovery/step task modules.
    result["response_migration"] = None
    result.pop("command_curriculum", None)
    result.update(episode_seconds=HORIZON_S, auto_reset=False, record_diagnostics=True, diagnostic_logging=False,
        monitor_applied_effort=True, evaluation_exact_cases=True, evaluation_long_corridors=True,
        playback_open_ground=False, flat_triangle_mesh=False, flat_half_length_m=44., flat_floor_width_m=8.,
        observation_noise_enabled=False, mass_randomization_enabled=False, training_release=False,
        research_training_authorized=False, hardware_deployment_ready=False, command_slew={"vx_m_s2": 1.5, "yaw_rad_s2": 4.})
    result["evaluation"] = {"enabled": True, "mode": "unscored", "protocol_id": "same_pt_clean_exploration_v1",
        "seed": SEED, "confirmation_seed": SEED, "episode_seconds": HORIZON_S, "episodes_per_case": REPLICAS,
        "stable_case_layout": True, "canonical_case_names": list(CASE_PROFILES), "cases": cases(),
        "promotion_case_names": [], "protected_case_names": [], "retention_case_names": [], "quick_case_names": []}
    # The independent row freezes some nested paths separately as well.
    for key in ("evaluation.seed", "evaluation.confirmation_seed"):
        if key in result:
            result[key] = SEED
    result["scene_groups"] = [{"name": name, "fraction": 1/3, "terrain": ["flat"]} for name in CASE_PROFILES]
    result["skill_specs"] = {c["name"]: deepcopy(c["skill"]) for c in cases()}
    for module in ("usb_transport", "command_transport", "signal_perturbations", "signal_delay",
                   "dynamics_randomization", "contact_domain", "step_assist", "jump_assist", "nonstandard_start", "release_stop"):
        if module in result:
            result[module]["enabled"] = False
    result["signal_perturbations"].update(max_delay_steps=0, noise_scale=0., enabled_fraction=0.)
    result["new_asset_curriculum"].update(unilateral_degradation_fraction=0., unilateral_degradation_scale=1., edge_reset_fraction=0.)
    result["new_asset_curriculum"]["request_protocol"] = {}
    result["new_asset_curriculum"].pop("start_stop_start_delay_seconds", None)
    # Exact role bypasses the frozen difficulty rule, which cannot be changed
    # to enabled=false without changing its core validator.
    result["stages"] = [{**s, "push_max": 0., "terrain": ["flat"]} for s in result["stages"]]
    return result


def differences(before, after, prefix=""):
    rows = []
    for key in sorted(before.keys() | after.keys()):
        path = prefix + key
        if key not in before:
            rows.append({"path": path, "operation": "added", "after": deepcopy(after[key])})
        elif key not in after:
            rows.append({"path": path, "operation": "deleted", "before": deepcopy(before[key])})
        elif isinstance(before[key], dict) and isinstance(after[key], dict):
            rows.extend(differences(before[key], after[key], path + "."))
        elif before[key] != after[key]:
            rows.append({"path": path, "operation": "changed", "before": deepcopy(before[key]), "after": deepcopy(after[key])})
    return rows


def build_clean_fixture(root, source, output):
    """Build a full-role CPU-verified diagnostic; source is never mutated.

    Return config/manifest/control/stage/role/seed/num_envs/case_profiles,
    design_preflight, receipt and artifact paths. No source training contracts
    are written. Scene creation/first-cohort accounting belong to the entry.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest, control, modules = asset_inputs(root, source)
    design, original_row, source_receipt = source_design(root, source, control, modules)
    config = clean_overrides(source)
    expected = clean_overrides(original_row["contract"])
    runtime = deepcopy(original_row["runtime"])
    for key in ("recovery_takeover", "recovery_pose_curriculum", "recovery_complete_started_blend", "start_stop_timing"):
        runtime.pop(key, None)
    for key in ("recovery_training", "step_jump_task", "step_climb_task", "jump_landing"):
        runtime["modules"].pop(key, None)
    for key in ("usb_transport", "command_transport", "signal_perturbations", "dynamics_randomization", "contact_domain",
                "step_assist", "jump_assist", "signal_delay", "nonstandard_start", "release_stop"):
        if key in runtime["modules"]:
            runtime["modules"][key] = False
    runtime["signal_delay"] = deepcopy(expected["signal_delay"])
    runtime["command_transport_model"] = None
    from wheeled_tasks.chassis.nonstandard_start import runtime_expectation as nonstandard_runtime
    from wheeled_tasks.chassis.release_stop import release_stop_runtime_expectation
    runtime["nonstandard_start"] = nonstandard_runtime(expected["nonstandard_start"], .02)
    runtime["release_stop"] = release_stop_runtime_expectation(expected["release_stop"], .02)
    runtime["locomotion_difficulty"] = {"enabled": False, "settings": deepcopy(expected["locomotion_difficulty"])}
    projected = {"schema": "chassis-design-expectations-v1", "status": "frozen",
        "scope": "same_pt_clean_locomotion_exploration_diagnostic", "run_scope": "physical_diagnostic_unscored",
        "training_release": False, "hardware_deployment_ready": False, "formal_updates_consumed": 0,
        "source_contract_sha256": CONTRACT_SHA, "inherited_design_sha256": source["design_preflight"]["sha256"],
        "review_basis": "Independent frozen source train row plus explicit clean diagnostic projection constants; no curriculum/qualification credit",
        "stages": {"speed": {"roles": {"full": {"contract": expected, "control": deepcopy(original_row["control"]),
                    "invocation": {"num_envs": NUM_ENVS, "seed": {"rule": "fixed", "value": SEED}}, "runtime": runtime}}}}}
    # CPU compare before creating any output, then bind the independently
    # projected row by SHA and run the shared native hook again.
    static = modules["preflight"].validate_design_contract(config, projected, control, stage="speed", role="full",
                                                          seed=SEED, num_envs=NUM_ENVS)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"design": output/"clean_design_expectations.json", "contract": output/"clean_effective_contract.json",
             "receipt": output/"clean_fixture_receipt.json"}
    require(not any(p.exists() for p in paths.values()), "Fixture output files already exist")
    write_json(paths["design"], projected)
    config["design_preflight"] = {"file": str(paths["design"]), "sha256": digest(paths["design"]), "stage": "speed"}
    receipt = modules["preflight"].design_preflight(config, control, root, stage="speed", role="full", seed=SEED, num_envs=NUM_ENVS)
    write_json(paths["contract"], config)
    audit = {"schema": "same-pt-clean-locomotion-fixture-v1", "status": "CPU_design_preflight_passed_scene_not_run",
        "source_contract_sha256": CONTRACT_SHA, "source_design_preflight": source_receipt,
        "effective_contract_sha256": digest(paths["contract"]), "design_sha256": digest(paths["design"]),
        "overrides": differences(source, config), "role": "full", "seed": SEED, "num_envs": NUM_ENVS,
        "replicas_per_case": REPLICAS, "case_profiles": dict(CASE_PROFILES), "horizon_seconds": HORIZON_S,
        "native_constructor_overrides": dict(CONSTRUCTOR_OVERRIDES),
        "policy_ticks": 1000, "physics_ticks": 4000, "auto_reset": False, "qualification": False,
        "formal_updates_consumed": 0, "optimizer_steps": 0, "runtime_verified": False,
        "actual_nominal_materials": "required_at_native_birth; average combine, actual robot/terrain friction readback",
        "birth_jitter": {"implementation": "native FixedCaseEnv", "reset_seed_key": "quality_common_reset",
                         "roll_range_rad": [-.01,.01], "pitch_range_rad": [-.02,.02], "root_z_jitter_m": [0.,.002]},
        "runtime_assertion_fields": ["module_instances_absent", "schedule_enabled_false", "no_degradation/edge/push",
            "actual_mass/inertia/COM", "actual_robot_and_bound_terrain_materials", "gravity", "gas_spring/closed_chain",
            "native_birth_q18/dq18/root_COM_velocity/jitter", "native_runtime_preflight", "1000_episode_limits"],
        "preflight": receipt}
    write_json(paths["receipt"], audit)
    return {"config": config, "manifest": manifest, "control": control, "stage": "speed", "role": "full",
            "seed": SEED, "num_envs": NUM_ENVS, "level": TERRAIN_LEVEL,
            "case_profiles": dict(CASE_PROFILES), "design_preflight": receipt,
            "receipt": audit, "paths": {k: str(v) for k,v in paths.items()}}


def assert_runtime_config(env, fixture):
    """Check the four explicit native constructor writes against frozen input.

    ChassisEnv.__init__ copies config and updates these four top-level fields;
    FixedCaseEnv.__init__ adds object state, without modifying the config.
    Read the serialized effective contract so shared nested references cannot
    conceal mutation of both env.cfg and fixture['config'].
    """
    contract_path = Path(fixture["paths"]["contract"])
    frozen = read_json(contract_path)
    expected = deepcopy(frozen)
    expected.update(CONSTRUCTOR_OVERRIDES)
    fixture_diff = differences(frozen, fixture["config"])
    runtime_diff = differences(expected, env.cfg)
    invocation = {"stage": fixture["stage"], "num_envs": fixture["num_envs"],
                  "terrain_level": fixture["level"], "seed": fixture["seed"]}
    invocation_diff = differences(CONSTRUCTOR_OVERRIDES, invocation)
    population_matches = type(env.num_envs) is int and env.num_envs == NUM_ENVS
    digest_matches = digest(contract_path) == fixture["receipt"]["effective_contract_sha256"]
    passed = not (fixture_diff or runtime_diff or invocation_diff) and population_matches and digest_matches
    path = contract_path.parent / "runtime_config_validation.json"
    audit = {"schema": "clean-locomotion-native-config-validation-v1", "status": "passed" if passed else "failed",
        "qualification": False, "runtime_verified": False,
        "contract_path": str(contract_path), "effective_contract_sha256": digest(contract_path),
        "effective_contract_sha_matches": digest_matches, "constructor_implementation": {
            "config_writes": "src/wheeled_tasks/chassis/env.py:30-31 ChassisEnv.__init__",
            "fixed_case_config_writes": "none; src/wheeled_tasks/chassis/eval_env.py FixedCaseEnv.__init__"},
        "allowed_constructor_overrides": dict(CONSTRUCTOR_OVERRIDES),
        "legal_constructor_differences": differences(frozen, expected),
        "fixture_invocation_differences": invocation_diff, "fixture_config_differences": fixture_diff,
        "runtime_config_differences": runtime_diff,
        "expected_num_envs": NUM_ENVS, "observed_num_envs": env.num_envs, "population_matches": population_matches}
    write_json(path, audit)
    bad_paths = sorted({row["path"] for row in (*fixture_diff, *runtime_diff, *invocation_diff)})
    require(passed, f"Native effective fixture/population differs; unexpected paths={bad_paths}; "
                   f"population_matches={population_matches}; contract_sha_matches={digest_matches}; receipt={path}")
    return audit


def assert_native_mechanics(startup, num_envs):
    """Check population-wide loop count; the pinned v6 asset has six per robot."""
    expected = {"body_count": 19, "tree_dofs": 18, "loop_constraints": 6 * num_envs}
    require(type(num_envs) is int and num_envs > 0, "Invalid native population")
    require(startup["runtime_preflight"]["status"] == "runtime_configuration_passed"
            and startup["gpu_dynamics_enabled"] is True and startup["gas_spring_force_enabled"] is True
            and all(startup[key] == value for key, value in expected.items()),
            "Native physics/mechanical runtime verification failed")
    return {"num_envs": num_envs, "loop_constraints_per_robot": 6,
            "expected": expected, "observed": {key: startup[key] for key in expected}}


def assert_clean_runtime(env, fixture):
    """Read and verify actual native birth; do not apply control or state writes."""
    config_validation = assert_runtime_config(env, fixture)
    import torch
    import warp as wp
    from pxr import UsdPhysics, UsdShade, PhysxSchema
    require(env.dt == .005 and env.policy_dt == .02 and env.decimation == 4 and env.generator.initial_seed() == SEED,
            "Native control clock/seed differs")
    require(env.max_episode_length == 1000 and bool((env.episode_limits == 1000).all()) and env.cfg["auto_reset"] is False,
            "Native 20s episode/auto-reset differs")
    require(Counter(env.scene_groups) == {k: REPLICAS for k in CASE_PROFILES} and all(k == "flat" for k in env.kinds),
            "Native flat case layout differs")
    require(all(getattr(env, name, None) is None for name in ABSENT_MODULES), "A clean-disabled native module is instantiated")
    require(env.nonstandard_start is None or env.nonstandard_start.enabled is False, "Nonstandard births remain enabled")
    schedule = env.skills.schedule
    require(schedule.enabled is False and env.skills.difficulty_enabled is False
            and env.skills.release_stop.enabled is False, "Training schedule/difficulty/release-stop remains enabled")
    require(not bool(schedule.degraded.any() | schedule.edge_reset.any() | schedule.request_stress.any() | env.push_enabled.any())
            and bool((schedule.motor_scale == 1.).all()) and bool((env.motor_strength == 1.).all())
            and bool((env.spring_strength == 1.).all()), "Push/effort/degradation randomization remains active")
    require(tuple(env.sim.cfg.gravity) == (0.,0.,-9.81), "Gravity differs from nominal")
    startup = env.startup_report
    mechanics = assert_native_mechanics(startup, NUM_ENVS)
    def cpu(value):
        if not isinstance(value, torch.Tensor):
            value = wp.to_torch(value)
        result = value.detach().cpu().clone()
        require(bool(torch.isfinite(result).all()), "Nonfinite native readback")
        return result
    masses = cpu(env.robot.root_view.get_masses())
    inertias = cpu(env.robot.root_view.get_inertias())
    coms = cpu(env.robot.root_view.get_coms())
    require(torch.allclose(masses, cpu(env.nominal_body_mass), atol=1e-5,rtol=0.)
            and torch.allclose(masses.sum(-1), torch.full((NUM_ENVS,),fixture["manifest"]["total_mass_kg"]),atol=1e-5,rtol=0.),
            "Native mass is not nominal")
    robot_material = cpu(env.robot.root_view.get_material_properties())
    require(robot_material.shape[-1] >= 3 and bool((robot_material[..., :2]-.5).abs().max() <= 1e-6)
            and bool(robot_material[...,2].abs().max() <= 1e-6), "Actual robot material is not nominal .5/.5/0")
    terrain_materials = []
    for lane, surfaces in enumerate(env.surfaces):
        for index, surface in enumerate(surfaces):
            path = env.env_paths[lane]+f"/Terrain/surface_{index}/geometry/mesh"
            prim = env.sim.stage.GetPrimAtPath(path)
            require(prim.IsValid(), "Native terrain collision prim missing")
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            require(material.GetPrim().IsValid(), "Native terrain has no bound physics material")
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            values = [api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(), api.GetRestitutionAttr().Get()]
            combine = PhysxSchema.PhysxMaterialAPI(material.GetPrim()).GetFrictionCombineModeAttr().Get()
            require(all(type(v) in (int,float) and math.isfinite(v) for v in values)
                    and all(abs(v-e)<1e-6 for v,e in zip(values,(.5,.5,0.))) and combine == "average"
                    and surface.friction == .5, "Actual bound terrain material/combine differs from nominal")
            terrain_materials.append({"lane":lane,"collision_path":path,"material_path":str(material.GetPath()),
                "static_friction":values[0],"dynamic_friction":values[1],"restitution":values[2],
                "friction_combine":"average","effective_nominal_friction":.5})
    q, dq = cpu(env.robot.data.joint_pos.torch), cpu(env.robot.data.joint_vel.torch)
    pose = cpu(env.robot.data.root_link_pose_w.torch); pose[:, :3] -= cpu(env.origins)
    velocity = torch.cat((cpu(env.robot.data.root_com_lin_vel_w.torch),cpu(env.robot.data.root_com_ang_vel_w.torch)),-1)
    require(torch.allclose(q,cpu(env.nominal),atol=2e-5,rtol=0.) and bool(dq.abs().max()<=2e-5)
            and bool(velocity.abs().max()<=2e-5), "Native birth q/dq/COM velocity differs from normal reset")
    samples = cpu(env.fixed_reset_samples)
    require(samples.shape == (NUM_ENVS,3) and bool(((samples>=0)&(samples<1)).all()), "Missing native seeded reset samples")
    roll,pitch=(samples[:,0]*2-1)*.01,(samples[:,1]*2-1)*.02
    sr,cr,sp,cp=torch.sin(roll/2),torch.cos(roll/2),torch.sin(pitch/2),torch.cos(pitch/2)
    quat=torch.stack((sr*cp,cr*sp,-sr*sp,cr*cp),-1)
    require(torch.allclose(pose[:,3:],quat,atol=2e-5,rtol=0.) and bool(pose[:,:2].abs().max()<=2e-5)
            and torch.allclose(pose[:,2],.324+samples[:,2]*.002,atol=2e-5,rtol=0.), "Native reset jitter differs")
    return {"schema":"clean-locomotion-native-birth-readback-v1","runtime_verified":True,"qualification":False,
        "seed":SEED,"num_envs":NUM_ENVS,"scene_groups":list(env.scene_groups),"clone_indices":list(env.clone_indices),
        "nominal_mass_kg_each":masses.sum(-1).tolist(),"body_masses_kg":masses.tolist(),
        "body_inertias":inertias.tolist(),"body_COMs":coms.tolist(),"robot_material_properties":robot_material.tolist(),
        "bound_terrain_materials":terrain_materials,"gravity_m_s2":[0.,0.,-9.81],
        "birth":{"joint_names":list(env.robot.joint_names),"q18":q.tolist(),"dq18":dq.tolist(),
            "root_link_pose_local_xyzw":pose.tolist(),"root_COM_velocity_world6":velocity.tolist(),"fixed_reset_samples":samples.tolist()},
        "module_instances_absent":list(ABSENT_MODULES),"training_schedule_enabled":False,
        "nonstandard_start_enabled":False,"push_enabled":False,"motor_strength":cpu(env.motor_strength).tolist(),
        "spring_strength":cpu(env.spring_strength).tolist(),"auto_reset":False,"episode_limits_policy_ticks":[1000]*NUM_ENVS,
        "runtime_preflight":deepcopy(startup["runtime_preflight"]),"native_mechanics":mechanics,
        "runtime_config_validation":config_validation,
        "PhysX_exact_snapshot_resume":False}
