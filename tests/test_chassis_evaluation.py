"""Outcome accounting and fixed-suite acceptance must not confuse truncation with survival."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
from wheeled_tasks.chassis.evaluation import cross_asset_checkpoint_provenance, fixed_suite_contract, grade_fixed_suite, validate_cross_asset_actor


ROOT = Path(__file__).resolve().parents[1]


def actor_interfaces():
    from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v5_nominal_recovery_v58.json"), load)
    contract = stage_contract(load(plan["base_contract"]), plan, plan["stages"][0], 64)
    manifest = load("model/纯底盘_v5/urdf/manifest.json")
    return contract, manifest


def test_cross_asset_comparison_preserves_actor_interface():
    contract, manifest = actor_interfaces()
    source = deepcopy(manifest)
    source["files_sha256"]["model_spec.json"] = "old-geometry"
    validate_cross_asset_actor(contract, deepcopy(contract), source, manifest)


@pytest.mark.parametrize("key,value", [("policy_dt", .01), ("physics_dt", .001),
                                      ("actor_dim", 46), ("policy_action_order", ["swapped"])])
def test_cross_asset_comparison_rejects_control_abi_changes(key, value):
    contract, manifest = actor_interfaces()
    target = deepcopy(contract)
    target[key] = value
    with pytest.raises(ValueError, match=key):
        validate_cross_asset_actor(contract, target, manifest, manifest)


def test_cross_asset_comparison_rejects_nominal_zero_or_pd_change():
    contract, manifest = actor_interfaces()
    target = deepcopy(manifest)
    target["nominal_joint_pos"]["L_joint1"] += .01
    with pytest.raises(ValueError, match="nominal_joint_pos"):
        validate_cross_asset_actor(contract, contract, manifest, target)
    changed = deepcopy(contract)
    changed["v5_control"]["leg_kp"] *= 2
    with pytest.raises(ValueError, match="v5_control"):
        validate_cross_asset_actor(contract, changed, manifest, manifest)


def test_cross_asset_migration_authenticates_original_sidecars(tmp_path):
    contract, manifest = actor_interfaces()
    original_manifest = deepcopy(manifest)
    original_manifest["files_sha256"]["model_spec.json"] = "original-model"
    source_sha = hashlib.sha256((json.dumps(original_manifest, indent=2) + "\n").encode()).hexdigest()
    (tmp_path / "asset_manifest.json").write_text(json.dumps(original_manifest, indent=2))
    source = {**contract, "asset_manifest_sha256": source_sha}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(source))
    infos = {"asset_manifest_sha256": source_sha, "contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "control_math_sha256": "control-math"}
    target = {**contract, "cross_asset_source_manifest_sha256": source_sha}
    proof = cross_asset_checkpoint_provenance(tmp_path / "model.pt", infos, target, manifest, "control-math")
    assert proof["source_asset_manifest_sha256"] != proof["target_asset_manifest_sha256"]
    with pytest.raises(ValueError, match="not authorized"):
        cross_asset_checkpoint_provenance(tmp_path / "model.pt", infos, contract, manifest, "control-math")
    path.write_text(json.dumps({**source, "policy_dt": .01}))
    with pytest.raises(ValueError, match="provenance"):
        cross_asset_checkpoint_provenance(tmp_path / "model.pt", infos, target, manifest, "control-math")


def sample(count=2):
    return {"velocity": torch.zeros(count, 3), "omega": torch.zeros(count, 3),
            "gravity": torch.tensor([[0., 0., -1.]]).repeat(count, 1),
            "height": torch.full((count,), .3), "position": torch.zeros(count, 3),
            "commands": torch.tensor([[0., 0., .3]]).repeat(count, 1),
            "episode_ticks": torch.ones(count, dtype=torch.long),
            "motor_effort": torch.full((count, 6), 3.), "reward": torch.full((count,), .05),
            "gap": torch.zeros(count), "done": torch.ones(count, dtype=torch.bool),
            "terminated": torch.zeros(count, dtype=torch.bool), "success": torch.zeros(count, dtype=torch.bool),
            "reasons": {"boundary": torch.tensor([True, False]), "fall": torch.tensor([False, True])}}


def test_outcomes_have_disjoint_denominators_and_preserve_boundaries():
    metrics = EpisodeMetrics(["a", "a"], "cpu", .01)
    data = sample()
    data["terminated"][1] = True
    metrics.observe(data)
    report = metrics.report()["groups"]["a"]
    assert report["episodes"] == 2
    assert report["failures"] == 1
    assert report["timeouts"] == report["boundary_truncations"] == 1
    assert report["successes"] == 0
    assert report["episodes"] == report["failures"] + report["timeouts"] + report["successes"]


def test_finished_env_mask_and_warmup_do_not_inflate_samples():
    metrics = EpisodeMetrics(["a", "b"], "cpu", .01, warmup_seconds=.01)
    data = sample()
    metrics.observe(data, torch.tensor([True, False]))
    assert metrics.report()["groups"]["a"]["frames"] == 0
    data["episode_ticks"][:] = 2
    data["done"][:] = False
    metrics.observe(data, torch.tensor([True, False]))
    report = metrics.report()["groups"]
    assert report["a"]["frames"] == 1 and report["b"]["frames"] == 0
    assert report["a"]["rms_motor_torque_nm"] == [3.] * 6
    assert report["b"]["episodes"] == 0


def test_large_accumulators_preserve_each_sample():
    metrics = EpisodeMetrics(["a", "a"], "cpu", .01)
    metrics.frames.fill_(2**29)
    metrics.torque_square.fill_(2**29)
    metrics.observe(sample())
    assert int(metrics.frames[0]) == 2**29 + 1
    assert float(metrics.torque_square[0, 0]) == 2**29 + 9


def test_boundary_exit_is_not_a_passed_fixed_horizon():
    settings = json.loads((ROOT / "contracts/v5_locomotion_v2.json").read_text())["evaluation"]
    settings["cases"] = [{"name": "a", "command": [0., 0., .3]}]
    metrics = EpisodeMetrics(["a", "a"], "cpu", .01)
    data = sample()
    data["reasons"]["fall"][:] = False
    metrics.observe(data)
    result = grade_fixed_suite(metrics.report(), settings, 2)
    assert result["cases"]["a"]["survival_rate"] == .5
    assert not result["passed"]
    assert not result["cases"]["a"]["checks"]["survival"]


def test_fixed_suite_preserves_physical_actuator_grouping():
    original = json.loads((ROOT / "contracts/v5_locomotion_v2.json").read_text())
    config = fixed_suite_contract(original)
    assert config["monitor_applied_effort"] is True
    assert config["v5_control"] == original["v5_control"]
    assert config["episode_seconds"] == 10. and original["episode_seconds"] == 20.
    assert sum(g["fraction"] for g in config["scene_groups"]) == pytest.approx(1.)
    assert {g["name"] for g in original["scene_groups"]} == {"stand", "translate", "rotate"}


def test_spin_translation_uses_mean_reference_velocity_per_episode_instance():
    metrics = EpisodeMetrics(["spin", "spin"], "cpu", .01)
    data = sample()
    data["done"][:] = False
    data["reference_velocity_error_vector"] = torch.tensor([[.3, 0.], [.4, 0.]])
    metrics.observe(data)
    data["episode_ticks"][:] = 2
    data["reference_velocity_error_vector"] = torch.tensor([[-.1, 0.], [0., 0.]])
    metrics.observe(data)
    assert metrics.report()["groups"]["spin"]["reference_velocity_error"] == pytest.approx(.2)


def test_zero_valid_frames_retains_terminal_time_and_observed_exposure():
    metrics = EpisodeMetrics(["a", "b"], "cpu", .02, warmup_seconds=1.)
    data = sample()
    data["done"][:] = False
    for tick in range(1, 31):
        data["episode_ticks"][:] = tick
        data["done"][0] = tick == 30
        data["terminated"][0] = tick == 30
        metrics.observe(data, torch.tensor([True, False]))
    a, b = (metrics.report()["groups"][name] for name in ("a", "b"))
    assert a["frames"] == 0
    assert a["observed_frames"] == 30
    assert a["observed_sim_seconds"] == pytest.approx(.6)
    assert a["terminal_episode_time_min_s"] == pytest.approx(.6)
    assert a["terminal_episode_time_mean_s"] == pytest.approx(.6)
    assert a["episodes_ending_before_or_at_warmup"] == 1
    assert b["observed_frames"] == 0
    assert b["terminal_episode_time_min_s"] is None
    assert b["vx_command_response_slope"] is None


def test_signed_velocity_distinguishes_overshoot_and_wrong_direction():
    metrics = EpisodeMetrics(["overshoot", "reverse"], "cpu", .02)
    data = sample()
    data["commands"][:, 0] = .5
    data["velocity"][:, 0] = torch.tensor([2., -1.])
    metrics.observe(data)
    groups = metrics.report()["groups"]
    assert groups["overshoot"]["vx_mae_m_s"] == groups["reverse"]["vx_mae_m_s"] == 1.5
    assert groups["overshoot"]["vx_bias_m_s"] == 1.5
    assert groups["reverse"]["vx_bias_m_s"] == -1.5
    assert groups["reverse"]["vx_actual_mean_m_s"] == -1.
    assert groups["reverse"]["vx_command_response_slope"] is None


def test_command_response_slope_is_descriptive_and_masks_warmup():
    metrics = EpisodeMetrics(["a", "a"], "cpu", .02, warmup_seconds=.02)
    data = sample()
    data["done"][:] = False
    data["commands"][:, 0] = torch.tensor([-1., 1.])
    data["velocity"][:, 0] = 100.
    metrics.observe(data)
    data["episode_ticks"][:] = 2
    data["velocity"][:, 0] = 2 * data["commands"][:, 0] + .5
    metrics.observe(data)
    group = metrics.report()["groups"]["a"]
    assert group["vx_command_response_slope"] == pytest.approx(2.)
    assert group["vx_actual_mean_m_s"] == pytest.approx(.5)
    assert group["post_warmup_sample_fraction"] == .5


def test_height_coverage_measures_reached_targets_instead_of_command_presence():
    metrics = EpisodeMetrics(["a", "a"], "cpu", .02, height_range_m=[.21, .35])
    data = sample()
    data["commands"][:, 2] = torch.tensor([.21, .35])
    data["height"][:] = torch.tensor([.30, .345])
    metrics.observe(data)
    group = metrics.report()["groups"]["a"]
    assert group["height_low_frames"] == group["height_high_frames"] == 1
    assert group["height_mid_frames"] == 0
    assert group["height_low_within_10mm_fraction"] == 0.
    assert group["height_high_within_10mm_fraction"] == 1.
    assert group["height_mid_mae_m"] is None
    assert group["height_low_mae_m"] == pytest.approx(.09)
