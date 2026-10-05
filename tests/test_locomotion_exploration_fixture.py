"""CPU-only input authentication and independent clean-role projection checks."""
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


def test_projection_passes_frozen_b4_preflight_without_dirty_playback(authenticated, tmp_path, monkeypatch):
    raw = subprocess.run(["git","show","b4ce80964d47c0127e0b3d9929ad35a620bbf204:src/wheeled_tasks/chassis/preflight.py"],
                         cwd=ROOT,check=True,capture_output=True).stdout
    clean = types.ModuleType("frozen_clean_preflight")
    clean.__package__ = "wheeled_tasks.chassis"
    exec(compile(raw,"<frozen:b4:preflight>","exec"),clean.__dict__)
    original = fixture.native_modules(ROOT)
    monkeypatch.setattr(fixture,"native_modules",lambda root: {**original,"preflight":clean})
    result = fixture.build_clean_fixture(ROOT, authenticated["source"], tmp_path)
    assert result["role"] == "full"
    assert result["design_preflight"]["status"] == "static_passed"
    assert result["config"]["auto_reset"] is False


@pytest.mark.parametrize("num_envs", [4, 12])
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
