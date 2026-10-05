"""CPU-only input authentication and independent clean-role projection checks."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/v6_locomotion_quality_repair_20261005/probe/v14_update24500_input"
sys.path.insert(0, str(ROOT / "scripts"))
import locomotion_exploration_fixture as fixture
from locomotion_diagnostic_suite import DEFAULT_SUITE, HIGH_SPEED_SUITE, get_suite


@pytest.fixture(scope="module")
def authenticated():
    return fixture.authenticate_inputs(ROOT, INPUT)


@pytest.fixture(scope="module")
def saved():
    return torch.load(INPUT / "model.pt", map_location="cpu", weights_only=True)


def test_real_24500_strict_actor_std_no_optimizer_or_cuda(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Authentication must not create Adam")
    monkeypatch.setattr(torch.optim, "Adam", reject)
    before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.is_initialized()
    result = fixture.authenticate_inputs(ROOT, INPUT)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == cuda_before
    receipt = result["receipt"]
    assert receipt["status"] == "authenticated_CPU_only"
    assert receipt["native_cpu_factory"]["actor_strict_exact"]
    assert receipt["native_cpu_factory"]["critic_strict_exact"]
    assert receipt["native_cpu_factory"]["raw_std_policy_order"] == result["actor_state_dict"]["distribution.std_param"].tolist()
    assert receipt["clocks"]["learning_updates"] == 24500
    assert receipt["clocks"]["task_actor_update"] == 4185
    assert receipt["optimizer_constructed"] is False
    assert receipt["formal_updates_consumed"] == receipt["optimizer_steps"] == 0
    assert receipt["qualification"] is False and receipt["PT_exact_restore"] is False
    assert "isaaclab.app" not in sys.modules


@pytest.mark.parametrize("filename,label", [("model.pt","checkpoint"),("contract.json","contract"),
    ("model.checkpoint.json","seal"),("agent_config.json","agent_config")])
def test_hash_rejection_before_deserialization(tmp_path, monkeypatch, filename, label):
    for name in ("model.pt","contract.json","model.checkpoint.json","agent_config.json"):
        shutil.copyfile(INPUT/name, tmp_path/name)
    with (tmp_path/filename).open("ab") as stream:
        stream.write(b"tampered")
    def reject(*args, **kwargs):
        raise AssertionError("Invalid external input must be rejected before torch.load")
    monkeypatch.setattr(torch, "load", reject)
    with pytest.raises(ValueError, match=f"Pinned {label} SHA"):
        fixture.authenticate_inputs(ROOT, tmp_path)


@pytest.mark.parametrize("key,value,message", [("successful_updates_total",24501,"clock mismatch"),
    ("task_actor_update",4186,"clock mismatch"),("asset_manifest_sha256","a"*64,"infos ABI")])
def test_infos_tamper_rejected(saved, monkeypatch, key, value, message):
    altered = deepcopy(saved)
    altered["infos"][key] = value
    monkeypatch.setattr(torch,"load",lambda *args,**kwargs: altered)
    with pytest.raises(ValueError, match=message):
        fixture.authenticate_inputs(ROOT, INPUT)


@pytest.mark.parametrize("invalid", [torch.tensor([0.]*6), torch.ones(5), torch.tensor([float("nan")]*6)])
def test_invalid_saved_std_rejected(saved, monkeypatch, invalid):
    altered = deepcopy(saved)
    altered["actor_state_dict"]["distribution.std_param"] = invalid
    monkeypatch.setattr(torch,"load",lambda *args,**kwargs: altered)
    with pytest.raises(ValueError, match="std is invalid|nonfinite"):
        fixture.authenticate_inputs(ROOT, INPUT)


def test_missing_native_actor_tensor_is_not_transferred(saved, monkeypatch):
    altered = deepcopy(saved)
    key = next(k for k in altered["actor_state_dict"] if k != "distribution.std_param")
    del altered["actor_state_dict"][key]
    monkeypatch.setattr(torch,"load",lambda *args,**kwargs: altered)
    with pytest.raises(RuntimeError, match="Missing key"):
        fixture.authenticate_inputs(ROOT, INPUT)


def test_policy_axis_order_rejected(monkeypatch):
    read = fixture.read_json
    def alter(path):
        value = read(path)
        if Path(path).name == "contract.json":
            value["policy_action_order"] = list(reversed(value["policy_action_order"]))
        return value
    monkeypatch.setattr(fixture,"read_json",alter)
    with pytest.raises(ValueError, match="axis order"):
        fixture.authenticate_inputs(ROOT, INPUT)


def test_clean_fixture_complete_overrides_and_source_unchanged(authenticated, tmp_path):
    source = authenticated["source"]
    before = deepcopy(source)
    result = fixture.build_clean_fixture(ROOT, source, tmp_path)
    config = result["config"]
    assert source == before
    assert (result["stage"],result["role"],result["seed"],result["num_envs"]) == ("speed","full",190619,12)
    assert config["episode_seconds"] == 20. and config["auto_reset"] is False
    assert config["evaluation"]["episodes_per_case"] == 4
    assert result["receipt"]["policy_ticks"] == 1000 and result["receipt"]["physics_ticks"] == 4000
    assert result["case_profiles"] == {"stand_305mm":0.,"forward_0p5":.5,"backward_0p5":-.5}
    assert all(c["terrain"] == "flat" and c["episode_seconds"] == 20. and c["command"] == [0.,0.,.305]
               and c["skill"]["mode"] == 0 and c["reset_seed_key"] == "quality_common_reset" for c in config["evaluation"]["cases"])
    for module in ("usb_transport","command_transport","signal_delay","signal_perturbations","dynamics_randomization",
                   "contact_domain","step_assist","jump_assist","nonstandard_start","release_stop"):
        assert config[module]["enabled"] is False
    for key in ("recovery_training","step_jump_task","recovery_takeover","jump_landing"):
        assert key not in config
    assert config["response_migration"] is None
    assert config["signal_perturbations"]["max_delay_steps"] == 0
    assert config["signal_perturbations"]["noise_scale"] == 0.
    assert config["observation_noise_enabled"] is False
    assert config["new_asset_curriculum"]["unilateral_degradation_fraction"] == 0.
    assert config["new_asset_curriculum"]["edge_reset_fraction"] == 0.
    assert config["new_asset_curriculum"]["request_protocol"] == {}
    assert all(s["push_max"] == 0. and s["terrain"] == ["flat"] for s in config["stages"])
    for key in ("asset_manifest_sha256","control_math_sha256","actor_dim","critic_dim","action_dim","v5_control",
                "position_iterations","velocity_iterations","actuator_response","actuator_response_evidence","motion_limits"):
        assert config[key] == source[key]
    assert result["design_preflight"]["status"] == "static_passed"
    assert result["receipt"]["runtime_verified"] is False
    assert result["receipt"]["qualification"] is False
    overrides = {r["path"]:r for r in result["receipt"]["overrides"]}
    assert overrides["recovery_training"]["operation"] == "deleted"
    assert overrides["command_transport.enabled"]["before"] is True
    assert overrides["command_transport.enabled"]["after"] is False
    assert overrides["episode_seconds"]["before"] == 32.
    assert overrides["episode_seconds"]["after"] == 20.
    assert result["receipt"]["effective_contract_sha256"] == fixture.digest(result["paths"]["contract"])
    assert result["receipt"]["design_sha256"] == fixture.digest(result["paths"]["design"])


def test_source_abi_change_rejected_before_output(authenticated, tmp_path):
    altered = deepcopy(authenticated["source"])
    altered["actor_dim"] = 36
    with pytest.raises(ValueError, match="actor_dim"):
        fixture.build_clean_fixture(ROOT, altered, tmp_path/"must_not_exist")
    assert not (tmp_path/"must_not_exist").exists()


def test_no_overwrite_of_existing_fixture(authenticated, tmp_path):
    result = fixture.build_clean_fixture(ROOT, authenticated["source"], tmp_path)
    before = {name:fixture.digest(path) for name,path in result["paths"].items()}
    with pytest.raises(ValueError, match="already exist"):
        fixture.build_clean_fixture(ROOT, authenticated["source"], tmp_path)
    assert {name:fixture.digest(path) for name,path in result["paths"].items()} == before


@pytest.mark.parametrize("fixture_role", fixture.FIXTURE_ROLES)
@pytest.mark.parametrize("suite_name", (DEFAULT_SUITE, HIGH_SPEED_SUITE))
def test_projection_passes_frozen_b4_preflight_without_dirty_playback(authenticated, tmp_path, monkeypatch, fixture_role, suite_name):
    raw = subprocess.run(["git","show","b4ce80964d47c0127e0b3d9929ad35a620bbf204:src/wheeled_tasks/chassis/preflight.py"],
                         cwd=ROOT,check=True,capture_output=True).stdout
    clean = types.ModuleType("frozen_clean_preflight")
    clean.__package__ = "wheeled_tasks.chassis"
    exec(compile(raw,"<frozen:b4:preflight>","exec"),clean.__dict__)
    original = fixture.native_modules(ROOT)
    monkeypatch.setattr(fixture,"native_modules",lambda root: {**original,"preflight":clean})
    result = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, fixture_role, suite_name)
    assert result["role"] == "full"
    assert result["design_preflight"]["status"] == "static_passed"
    assert result["config"]["auto_reset"] is False


@pytest.mark.parametrize("num_envs", [4, 12, 64])
def test_population_wide_native_loop_constraints(num_envs):
    startup = {"runtime_preflight": {"status": "runtime_configuration_passed"},
               "gpu_dynamics_enabled": True, "gas_spring_force_enabled": True,
               "body_count": 19, "tree_dofs": 18, "loop_constraints": 6 * num_envs}
    receipt = fixture.assert_native_mechanics(startup, num_envs)
    assert receipt["observed"]["loop_constraints"] == 6 * num_envs
    assert receipt["loop_constraints_per_robot"] == 6
    if num_envs == 12:
        startup["loop_constraints"] = 24
        with pytest.raises(ValueError, match="mechanical runtime verification"):
            fixture.assert_native_mechanics(startup, num_envs)


def native_constructor_config(config, num_envs=12):
    """Execute actual native config statements, without importing simulator APIs."""
    parsed = ast.parse((ROOT/"src/wheeled_tasks/chassis/env.py").read_text())
    chassis = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    constructor = next(node for node in chassis.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    obj = types.SimpleNamespace()
    statements = ast.Module(body=constructor.body[:2], type_ignores=[])
    values = {"self": obj, "config": config, "stage_name": "speed", "num_envs": num_envs, "level": 1., "seed": 190619}
    exec(compile(statements, "<native ChassisEnv config statements>", "exec"), values)
    return obj.cfg


def test_exact_native_constructor_projection_preserves_every_other_field(authenticated, tmp_path):
    built = fixture.build_clean_fixture(ROOT, authenticated["source"], tmp_path)
    config_before = deepcopy(built["config"])
    env = types.SimpleNamespace(cfg=native_constructor_config(built["config"]), num_envs=12)
    audit = fixture.assert_runtime_config(env, built)
    assert built["config"] == config_before
    assert env.cfg == {**config_before, "stage": "speed", "num_envs": 12, "terrain_level": 1., "seed": 190619}
    assert audit["status"] == "passed" and audit["runtime_config_differences"] == []
    assert {row["path"] for row in audit["legal_constructor_differences"]} == {"stage", "num_envs", "terrain_level", "seed"}
    assert fixture.read_json(tmp_path/"runtime_config_validation.json") == audit


@pytest.mark.parametrize("change,path", [("seed", "seed"), ("level", "terrain_level"),
    ("missing", "asset_manifest_sha256"), ("extra", "unapproved_runtime_override"),
    ("module", "signal_delay.enabled"), ("shared", "signal_delay.enabled")])
def test_unapproved_native_config_change_has_failure_receipt(authenticated, tmp_path, change, path):
    built = fixture.build_clean_fixture(ROOT, authenticated["source"], tmp_path)
    env = types.SimpleNamespace(cfg=native_constructor_config(built["config"]), num_envs=12)
    if change == "seed":
        env.cfg["seed"] += 1
    elif change == "level":
        env.cfg["terrain_level"] = 0.
    elif change == "missing":
        del env.cfg["asset_manifest_sha256"]
    elif change == "extra":
        env.cfg["unapproved_runtime_override"] = True
    elif change == "module":
        env.cfg = deepcopy(env.cfg)
        env.cfg["signal_delay"]["enabled"] = True
    elif change == "shared":
        # Native config is a shallow copy: changing this nested value changes
        # both in-memory dictionaries. Frozen serialized input must catch it.
        env.cfg["signal_delay"]["enabled"] = True
        assert built["config"]["signal_delay"]["enabled"] is True
    with pytest.raises(ValueError, match=path):
        fixture.assert_runtime_config(env, built)
    audit = fixture.read_json(tmp_path/"runtime_config_validation.json")
    assert audit["status"] == "failed"
    assert path in {row["path"] for row in audit["runtime_config_differences"]}
    if change == "shared":
        assert path in {row["path"] for row in audit["fixture_config_differences"]}


@pytest.mark.parametrize("change", ["population", "contract_sha", "invocation"])
def test_runtime_config_identity_failures_saved(authenticated, tmp_path, change):
    built = fixture.build_clean_fixture(ROOT, authenticated["source"], tmp_path)
    env = types.SimpleNamespace(cfg=native_constructor_config(built["config"]), num_envs=12)
    if change == "population":
        env.num_envs = 4
    elif change == "contract_sha":
        with Path(built["paths"]["contract"]).open("a") as stream:
            stream.write("\n")
    else:
        built["seed"] += 1
    with pytest.raises(ValueError, match="Native effective fixture/population differs"):
        fixture.assert_runtime_config(env, built)
    audit = fixture.read_json(tmp_path/"runtime_config_validation.json")
    assert audit["status"] == "failed"
    assert not audit["population_matches"] if change == "population" else (
        not audit["effective_contract_sha_matches"] if change == "contract_sha" else bool(audit["fixture_invocation_differences"]))


def test_pipeline_only_restores_exact_frozen_delay(authenticated, tmp_path):
    clean = fixture.build_clean_fixture(ROOT, authenticated["source"], tmp_path/"clean")
    pipeline = fixture.build_fixture(ROOT, authenticated["source"], tmp_path/"pipeline", "pipeline_age_only")
    assert pipeline["fixture_role"] == pipeline["receipt"]["fixture_role"] == "pipeline_age_only"
    assert pipeline["task_actor_update"] == 4185
    assert pipeline["config"]["signal_delay"] == authenticated["source"]["signal_delay"]
    assert pipeline["signal_delay_expectation"] == authenticated["source"]["signal_delay"]
    assert pipeline["receipt"]["generator_consumption_equal_across_fixture_roles"] is False
    assert pipeline["design_preflight"]["runtime_expected"]["modules"]["signal_delay"] is True
    assert pipeline["design_preflight"]["runtime_expected"]["signal_delay"] == authenticated["source"]["signal_delay"]
    assert pipeline["design_preflight"]["status"] == "static_passed"
    paths = {row["path"] for row in fixture.differences(clean["config"], pipeline["config"])}
    assert paths == {"signal_delay.enabled", "design_preflight.file", "design_preflight.sha256"}
    for module in ("usb_transport", "command_transport", "signal_perturbations", "dynamics_randomization", "contact_domain",
                   "nonstandard_start", "release_stop", "step_assist", "jump_assist"):
        assert pipeline["config"][module]["enabled"] is False
    assert pipeline["config"]["auto_reset"] is False and pipeline["config"]["episode_seconds"] == 20.


def test_default_clean_and_wrapper_are_identical(authenticated, tmp_path):
    default = fixture.build_fixture(ROOT, authenticated["source"], tmp_path/"default")
    wrapper = fixture.build_clean_fixture(ROOT, authenticated["source"], tmp_path/"wrapper")
    assert default["fixture_role"] == wrapper["fixture_role"] == "clean"
    assert default["task_actor_update"] == wrapper["task_actor_update"] == 0
    assert fixture.differences(default["config"], wrapper["config"])[0]["path"] == "design_preflight.file"
    assert len(fixture.differences(default["config"], wrapper["config"])) == 1
    assert Path(default["paths"]["design"]).read_bytes() == Path(wrapper["paths"]["design"]).read_bytes()


def test_unknown_role_and_clean_wrapper_reject_pipeline(authenticated, tmp_path):
    with pytest.raises(ValueError, match="Unknown locomotion fixture role"):
        fixture.build_fixture(ROOT, authenticated["source"], tmp_path/"absent", "raw_usb")
    assert not (tmp_path/"absent").exists()
    pipeline = fixture.build_fixture(ROOT, authenticated["source"], tmp_path/"pipeline", "pipeline_age_only")
    with pytest.raises(ValueError, match="Clean runtime wrapper requires clean fixture role"):
        fixture.assert_clean_runtime(None, pipeline)


def cpu_native_delay_environment(built):
    from wheeled_tasks.chassis.signal_delay import PhysicalSignalDelay
    generator = torch.Generator(device="cpu").manual_seed(190619)
    num_envs = built["num_envs"]
    native = PhysicalSignalDelay(num_envs, "cpu", built["config"]["signal_delay"], .005, generator)
    native.reset(torch.arange(num_envs), torch.zeros(num_envs, 18))
    return types.SimpleNamespace(cfg=native_constructor_config(built["config"], num_envs), generator=generator,
                                 signal_delay=native, task_actor_update=built["task_actor_update"], tick=0)


def test_native_cpu_pipeline_readback_never_draws_or_advances(authenticated, tmp_path):
    pipeline = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, "pipeline_age_only")
    env = cpu_native_delay_environment(pipeline)
    rng_before = env.generator.get_state().clone()
    birth = fixture.read_signal_delay_runtime(env, pipeline)
    assert torch.equal(rng_before, env.generator.get_state())
    assert birth["task_actor_update"] == 4185 and birth["strength"] == 0.
    assert birth["physics_steps_observed"] == 0 and max(max(row) for row in birth["lags_physics_steps"]) == 0
    assert birth["raw_jitter_draw_coverage"] == "not_covered_native_local_not_persisted"
    assert birth["class_name"] == "PhysicalSignalDelay" and any(birth["enabled"])
    # Exercise actual native delay and native CPU buffers, without simulator,
    # PD, optimizer, or synthetic control. Each actual append advances once.
    env.signal_delay.begin_policy(4185)
    rng_after_native = env.generator.get_state().clone()
    for _ in range(4):
        env.signal_delay.targets(torch.zeros(12, 4), torch.zeros(12, 2))
        env.signal_delay.observe(torch.zeros(12, 18))
    observed = fixture.read_signal_delay_runtime(env, pipeline, policy_tick=1, physics_tick=4)
    assert torch.equal(rng_after_native, env.generator.get_state())
    assert observed["strength"] == 1. and observed["physics_steps_observed"] == 4
    assert max(max(row) for row in observed["lags_physics_steps"]) > 0
    assert env.signal_delay.sensor_buffer._num_pushes.tolist() == [4]*12
    assert env.signal_delay.target_buffer._num_pushes.tolist() == [4]*12


@pytest.mark.parametrize("change,message", [("clock", "task actor delay clock"),
    ("instance", "PhysicalSignalDelay instance"), ("cfg", "PhysicalSignalDelay instance"),
    ("append_clock", "physics append clock"), ("strength", "pipeline strength"),
    ("profile", "samples are invalid"), ("bounds", "profile/base delay"),
    ("nonfinite", "samples are invalid"), ("lag", "jitter/clip/quantization"),
    ("lag_dtype", "sample ABI")])
def test_pipeline_native_readback_rejects_corruption(authenticated, tmp_path, change, message):
    pipeline = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, "pipeline_age_only")
    env = cpu_native_delay_environment(pipeline)
    delay = env.signal_delay
    delay.begin_policy(4185)
    if change == "clock":
        env.task_actor_update = 4184
    elif change == "instance":
        env.signal_delay = None
    elif change == "cfg":
        delay.cfg["jitter_ms"] = 4.
    elif change == "append_clock":
        delay.physics_steps = 1
    elif change == "strength":
        delay.strength = .5
    elif change == "profile":
        delay.profile[0] = 3
    elif change == "bounds":
        delay.bounds_ms[0, 0, 0] = -1.
    elif change == "nonfinite":
        delay.base_ms[0, 0] = float("nan")
    elif change == "lag":
        delay.lags[0, 0] = 999
    else:
        delay.lags = delay.lags.float()
    with pytest.raises(ValueError, match=message):
        fixture.read_signal_delay_runtime(env, pipeline, policy_tick=1, physics_tick=0)


@pytest.mark.parametrize("policy_tick,physics_tick", [(0, 1), (1, 5), (2, 0)])
def test_delay_policy_physics_clock_coupling_rejects(authenticated, tmp_path, policy_tick, physics_tick):
    pipeline = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, "pipeline_age_only")
    env = cpu_native_delay_environment(pipeline)
    with pytest.raises(ValueError, match="coupled 50Hz/200Hz"):
        fixture.read_signal_delay_runtime(env, pipeline, policy_tick=policy_tick, physics_tick=physics_tick)


def test_native_environment_policy_clock_is_observed(authenticated, tmp_path):
    pipeline = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, "pipeline_age_only")
    env = cpu_native_delay_environment(pipeline)
    env.tick = 6
    with pytest.raises(ValueError, match="environment policy clock"):
        fixture.read_signal_delay_runtime(env, pipeline)


@pytest.mark.parametrize("fixture_role", fixture.FIXTURE_ROLES)
def test_high_suite_projection_is_bounded_and_birth_remains_nominal(authenticated, tmp_path, fixture_role):
    source = authenticated["source"]
    before = deepcopy(source)
    suite = get_suite(HIGH_SPEED_SUITE)
    built = fixture.build_fixture(ROOT, source, tmp_path, fixture_role, suite_name=suite.name)
    config, receipt = built["config"], built["receipt"]
    assert source == before
    assert (suite.num_envs, suite.replicas, suite.horizon_s, suite.policy_steps, suite.physics_steps) == (64, 4, 32., 1600, 6400)
    assert len(suite.cases) == 16
    for record in (built, config, receipt, fixture.read_json(built["paths"]["design"])):
        assert record["suite_name"] == HIGH_SPEED_SUITE
        assert record["diagnostic_suite"] == suite.as_dict()
        assert "suite" not in record
    assert config["episode_seconds"] == config["evaluation"]["episode_seconds"] == 32.
    assert config["evaluation"]["episodes_per_case"] == 4
    assert config["evaluation"]["canonical_case_names"] == list(suite.case_names)
    assert built["num_envs"] == receipt["num_envs"] == receipt["native_constructor_overrides"]["num_envs"] == 64
    assert (receipt["policy_ticks"], receipt["physics_ticks"], receipt["horizon_seconds"]) == (1600, 6400, 32.)
    assert built["case_profiles"] == receipt["case_profiles"] == suite.profiles
    assert config["scene_groups"] == [{"name": name, "fraction": 1/16, "terrain": ["flat"]} for name in suite.case_names]
    assert any(case.height_m == .43 for case in suite.cases)
    assert all(case["command"] == case["skill"]["command"] == [0., 0., .305]
               and case["episode_seconds"] == 32. and case["reset_seed_key"] == "quality_common_reset"
               for case in config["evaluation"]["cases"])
    assert all(spec == {"kind": "constant", "command": [0., 0., .305], "mode": 0}
               for spec in config["skill_specs"].values())
    for key in ("physics_dt", "policy_dt", "actor_dim", "critic_dim", "action_dim", "v5_control",
                "position_iterations", "velocity_iterations", "actuator_response", "motion_limits", "control_math_sha256"):
        assert config[key] == source[key]
    assert config["signal_delay"] == (source["signal_delay"] if fixture_role == "pipeline_age_only"
                                      else {**source["signal_delay"], "enabled": False})
    assert built["design_preflight"]["status"] == "static_passed"
    assert built["design_preflight"]["num_envs"] == 64
    assert receipt["qualification"] is False and receipt["formal_updates_consumed"] == 0


def test_default_protocol_still_has_exact_original_case_and_constructor_values(authenticated, tmp_path):
    built = fixture.build_fixture(ROOT, authenticated["source"], tmp_path)
    assert built["suite_name"] == DEFAULT_SUITE
    assert built["diagnostic_suite"] == get_suite().as_dict()
    assert fixture.cases() == [{"name": name, "terrain": "flat", "command": [0.,0.,.305], "episode_seconds": 20.,
            "reset_seed_key": "quality_common_reset", "plateau_vx_m_s": speed,
            "skill": {"kind": "constant", "command": [0.,0.,.305], "mode": 0}}
            for name, speed in {"stand_305mm": 0., "forward_0p5": .5, "backward_0p5": -.5}.items()]
    assert (fixture.NUM_ENVS, fixture.REPLICAS, fixture.HORIZON_S) == (12, 4, 20.)
    assert fixture.CONSTRUCTOR_OVERRIDES == {"stage": "speed", "num_envs": 12, "terrain_level": 1., "seed": 190619}


@pytest.mark.parametrize("fixture_role", fixture.FIXTURE_ROLES)
def test_unknown_suite_fails_before_creating_output(authenticated, tmp_path, fixture_role):
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="Unknown diagnostic suite"):
        fixture.build_fixture(ROOT, authenticated["source"], output, fixture_role, "ad_hoc_80s")
    assert not output.exists()


@pytest.mark.parametrize("field", ["horizon_s", "num_envs", "physics_steps"])
def test_catalog_snapshot_cannot_be_mutated_into_ad_hoc_runtime(authenticated, tmp_path, field):
    built = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, suite_name=HIGH_SPEED_SUITE)
    built["diagnostic_suite"][field] += 1
    built["receipt"]["diagnostic_suite"][field] += 1
    with pytest.raises(ValueError, match="immutable diagnostic catalog"):
        fixture.selected_suite(built)
    assert get_suite(HIGH_SPEED_SUITE).as_dict()[field] in (32., 64, 6400)


def test_native_constructor_produces_high_episode_limits_without_hot_writes(authenticated, tmp_path):
    built = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, suite_name=HIGH_SPEED_SUITE)
    suite = get_suite(HIGH_SPEED_SUITE)
    env = types.SimpleNamespace(cfg=native_constructor_config(built["config"], suite.num_envs), num_envs=suite.num_envs,
                               policy_dt=.02, scene_groups=[case for case in suite.case_names for _ in range(suite.replicas)])
    # Execute only the actual CPU constructor clock statements. This is not a
    # simulator run and does not rewrite limits after an environment is born.
    parsed = ast.parse((ROOT / "src/wheeled_tasks/chassis/env.py").read_text())
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    ctor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name) and target.value.id == "self" and target.attr == name
                for target in node.targets)
    maximum = next(node for node in ctor.body if assigns(node, "max_episode_length"))
    start = next(i for i, node in enumerate(ctor.body) if assigns(node, "episode_length_buf"))
    statements = ast.Module(body=[maximum, *ctor.body[start:start+3]], type_ignores=[])
    exec(compile(statements, "<native constructor episode clock>", "exec"),
         {"self": env, "config": built["config"], "num_envs": suite.num_envs, "device": "cpu", "torch": torch})
    assert env.max_episode_length == 1600 and env.episode_limits.tolist() == [1600] * 64
    audit = fixture.assert_runtime_config(env, built)
    assert audit["population_matches"] and audit["expected_num_envs"] == 64
    assert audit["allowed_constructor_overrides"]["num_envs"] == 64
    assert audit["diagnostic_suite"] == suite.as_dict()
    # The fixture observer itself only reads episode limits / simulator state.
    observer = ast.parse(Path(fixture.__file__).read_text())
    assert not any(isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                   and isinstance(node.value, ast.Name) and node.value.id == "env"
                   for node in ast.walk(observer))


def test_high_native_cpu_delay_sample_population_and_no_observer_draws(authenticated, tmp_path):
    built = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, "pipeline_age_only", HIGH_SPEED_SUITE)
    env = cpu_native_delay_environment(built)
    birth_rng = env.generator.get_state().clone()
    birth = fixture.read_signal_delay_runtime(env, built)
    assert torch.equal(birth_rng, env.generator.get_state())
    assert len(birth["enabled"]) == len(birth["feedback_sensor18"]) == 64
    env.signal_delay.begin_policy(4185)
    rng = env.generator.get_state().clone()
    for _ in range(4):
        env.signal_delay.targets(torch.zeros(64, 4), torch.zeros(64, 2))
        env.signal_delay.observe(torch.zeros(64, 18))
    observed = fixture.read_signal_delay_runtime(env, built, policy_tick=1, physics_tick=4)
    assert torch.equal(rng, env.generator.get_state())
    assert observed["suite_name"] == HIGH_SPEED_SUITE
    assert observed["strength"] == 1. and observed["physics_steps_observed"] == 4
    assert len(observed["base_ms"]) == len(observed["lags_physics_steps"]) == 64
    assert env.signal_delay.sensor_buffer._num_pushes.tolist() == [4] * 64
    assert env.signal_delay.target_buffer._num_pushes.tolist() == [4] * 64


@pytest.mark.parametrize("suite_name", (DEFAULT_SUITE, HIGH_SPEED_SUITE))
def test_named_suite_delay_clocks_allow_exact_horizon_and_reject_beyond(authenticated, tmp_path, suite_name):
    built = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, suite_name=suite_name)
    suite = get_suite(suite_name)
    # Clean role has no delay buffers. This checks boundary validation, not
    # unexecuted physical exposure or a synthetic populated history.
    env = types.SimpleNamespace(cfg=built["config"], tick=suite.policy_steps,
                                task_actor_update=0, signal_delay=None)
    observed = fixture.read_signal_delay_runtime(env, built, policy_tick=suite.policy_steps,
                                                physics_tick=suite.physics_steps)
    assert observed["policy_tick"] == suite.policy_steps and observed["physics_tick"] == suite.physics_steps
    for policy_tick, physics_tick in ((suite.policy_steps + 1, suite.physics_steps),
                                     (suite.policy_steps, suite.physics_steps + 1)):
        with pytest.raises(ValueError, match="bounded diagnostic delay clock"):
            fixture.read_signal_delay_runtime(env, built, policy_tick=policy_tick, physics_tick=physics_tick)


@pytest.mark.parametrize("checkpoint_update,task_actor,pt_sha,seal_sha", [
    (24500, 4185, fixture.PT_SHA, fixture.SEAL_SHA),
    (25000, 4685, fixture.PT_25000_SHA, fixture.SEAL_25000_SHA)])
def test_only_two_explicit_checkpoint_hash_clock_tuples(checkpoint_update, task_actor, pt_sha, seal_sha):
    expected = {"name": f"v14_update{checkpoint_update}", "learning_updates": checkpoint_update,
                "task_actor_update": task_actor, "checkpoint_sha256": pt_sha, "seal_sha256": seal_sha,
                "contract_sha256": fixture.CONTRACT_SHA, "agent_config_sha256": fixture.AGENT_SHA}
    assert fixture.checkpoint_snapshot(checkpoint_update) == expected
    assert fixture.snapshot_from_checkpoint_sha(pt_sha) == expected
    # A caller changing its copy cannot add another tuple to the registry.
    expected["task_actor_update"] = 99999
    assert fixture.checkpoint_snapshot(checkpoint_update)["task_actor_update"] == task_actor


@pytest.mark.parametrize("unknown", [24000, 25001, 25000., True, "25000", None])
def test_ad_hoc_checkpoint_clock_rejected_before_output(authenticated, tmp_path, unknown):
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="Unknown pinned diagnostic checkpoint"):
        fixture.build_fixture(ROOT, authenticated["source"], output, checkpoint_update=unknown)
    assert not output.exists()


@pytest.mark.parametrize("file_label", ["seal", "contract", "agent_config"])
def test_selected_25000_tuple_still_checks_all_files_before_pt_load(tmp_path, monkeypatch, file_label):
    names = {"checkpoint": "model.pt", "seal": "model.checkpoint.json", "contract": "contract.json",
             "agent_config": "agent_config.json"}
    snapshot = fixture.checkpoint_snapshot(25000)
    expected = {"checkpoint": snapshot["checkpoint_sha256"], "seal": snapshot["seal_sha256"],
                "contract": snapshot["contract_sha256"], "agent_config": snapshot["agent_config_sha256"]}
    for name in names.values():
        (tmp_path / name).write_text("fake input deliberately not deserializable")
    observed = {names[label]: value for label, value in expected.items()}
    observed[names[file_label]] = "0" * 64
    monkeypatch.setattr(fixture, "digest", lambda path: observed[Path(path).name])
    def reject_load(*args, **kwargs):
        raise AssertionError("All four selected input hashes must pass before loading PT")
    monkeypatch.setattr(torch, "load", reject_load)
    with pytest.raises(ValueError, match=f"Pinned {file_label} SHA mismatch"):
        fixture.authenticate_inputs(ROOT, tmp_path)


def test_unknown_pt_hash_cannot_select_even_a_known_clock(tmp_path, monkeypatch):
    for name in ("model.pt", "model.checkpoint.json", "contract.json", "agent_config.json"):
        (tmp_path / name).write_text("unrecognized")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Unknown PT must fail before load"))
    with pytest.raises(ValueError, match="Pinned checkpoint SHA mismatch"):
        fixture.authenticate_inputs(ROOT, tmp_path)


@pytest.mark.parametrize("suite_name", (DEFAULT_SUITE, HIGH_SPEED_SUITE))
@pytest.mark.parametrize("fixture_role", fixture.FIXTURE_ROLES)
def test_25000_projection_and_native_delay_clock_come_from_fixed_snapshot(authenticated, tmp_path, fixture_role, suite_name):
    built = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, fixture_role, suite_name, checkpoint_update=25000)
    assert built["checkpoint_update"] == built["receipt"]["checkpoint_update"] == 25000
    assert built["pinned_snapshot"] == built["receipt"]["pinned_snapshot"] == fixture.checkpoint_snapshot(25000)
    expected_actor = 4685 if fixture_role == "pipeline_age_only" else 0
    assert built["task_actor_update"] == built["receipt"]["task_actor_update"] == expected_actor
    suite = get_suite(suite_name)
    if fixture_role == "pipeline_age_only":
        env = cpu_native_delay_environment(built)
        env.signal_delay.begin_policy(4685)
    else:
        env = types.SimpleNamespace(cfg=built["config"], tick=0, task_actor_update=0, signal_delay=None)
    observed = fixture.read_signal_delay_runtime(env, built, policy_tick=1, physics_tick=0)
    assert observed["task_actor_update"] == expected_actor
    assert observed["suite_name"] == suite.name
    env.task_actor_update = 4185
    with pytest.raises(ValueError, match="task actor delay clock"):
        fixture.read_signal_delay_runtime(env, built, policy_tick=1, physics_tick=0)


def test_25000_tuple_rejects_forged_task_actor_metadata(authenticated, tmp_path):
    built = fixture.build_fixture(ROOT, authenticated["source"], tmp_path, "pipeline_age_only", checkpoint_update=25000)
    env = cpu_native_delay_environment(built)
    env.signal_delay.begin_policy(4685)
    built["pinned_snapshot"]["task_actor_update"] = 4185
    with pytest.raises(ValueError, match="pinned checkpoint identity"):
        fixture.read_signal_delay_runtime(env, built, policy_tick=1, physics_tick=0)
