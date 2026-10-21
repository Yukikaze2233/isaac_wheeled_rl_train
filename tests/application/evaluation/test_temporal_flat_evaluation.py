"""Paid-boundary and inference projection checks; no native capability claim."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from tests.support.support_flat_evaluation_runtime import stock_state
from tests.support.support_flat_evaluation_setup import make_env
from tests.support.support_flat_policy_evaluation import frame, begin
from wheeled_tasks.chassis.application.evaluation.flat_evaluation_runtime import BoundEvaluationInputs, canonical, runtime_source_files, sha, validate_overlay
from wheeled_tasks.chassis.application.evaluation.flat_evaluation_setup import FlatEvaluationSetup
from wheeled_tasks.chassis.application.evaluation.flat_policy_evaluation import FlatEpisodeLedger, validate_protocol
from wheeled_tasks.chassis.tasks.rewards.flat_reward import WEIGHTS
from wheeled_tasks.chassis.application.evaluation.temporal_flat_evaluation import (
    FIELD, PHASE_TICKS, TemporalEpisodeMeasurements, configuration, request_at_tick,
)
from wheeled_tasks.chassis.application.configuration.training_profile import profile_identity


ROOT = Path(__file__).resolve().parents[3]


def protocol():
    return json.loads((ROOT / "configs/v7_formal_20261012/evaluation/temporal_retention_mean_v1.json").read_bytes())


def test_real_command_owner_and_paid_endpoint_agree_across_all_three_boundaries():
    p = protocol()
    lanes = tuple(c["case_id"] for c in p["cases"])
    setup = FlatEvaluationSetup(p, lanes, 159, 15000)
    env = make_env(setup)
    rng = env.generator.get_state().clone()
    setup.apply_before_first_reset(env)
    commands = env.flat_episode.commands
    assert not commands.fixed_request_mask.any()
    assert torch.equal(rng, env.generator.get_state())
    ids = torch.arange(len(lanes))
    commands.reset(ids, torch.zeros(len(lanes)), torch.zeros(len(lanes), dtype=torch.bool),
        env.generator, iteration=15000)
    for tick in range(1, 1000):
        commands.begin_policy()
        for _ in range(4):
            commands.advance_physics()
        actual = torch.stack((commands.command3[:, 0], commands.command3[:, 2], commands.height), -1)
        expected = torch.tensor([request_at_tick(c["base_request"], tick) for c in p["cases"]])
        torch.testing.assert_close(actual, expected)
        commands.advance(.02, torch.zeros(len(lanes)), 15000, env.generator)


def test_ledger_accepts_sequence_but_rejects_one_interval_early_switch():
    p = protocol()
    ledger = FlatEpisodeLedger(p, [c["case_id"] for c in p["cases"]], list(WEIGHTS))
    begin(ledger)
    for tick in range(1, 252):
        request = request_at_tick(p["cases"][0]["base_request"], tick)
        if tick == 250:
            with pytest.raises(ValueError, match="base request changed"):
                ledger.observe(0, frame(tick, base_request=[0., 0., .3], effective_command=[0., 0., .3]))
        ledger.observe(0, frame(tick, base_request=request, effective_command=request))
    record = ledger.censor_open_episodes("test_end")[0]
    temporal = record["temporal_measurements"]
    assert not temporal["completed_four_phase_horizon"]
    assert [r["frames"] for r in temporal["phases"]] == [250, 1]


def test_distances_use_actual_boundary_and_held_time_is_confirmed_without_interpolation():
    measurement = TemporalEpisodeMeasurements([3., 0., .3], [0., 0., 0.], 0.)
    position = 0.
    for tick in range(1, 277):
        velocity = 2. if tick <= 250 else .1
        position += velocity * .02
        measurement.observe(frame(tick, root_link_position_world=[position, 0., 0.],
            root_com_velocity_body=[velocity, 0., 0.]))
    stop = measurement.report("censored")["phases"][1]
    assert stop["entry_body_velocity_m_s"][0] == 2.  # Not the3m/s request.
    assert stop["path_xy_m"] == pytest.approx(.052)
    assert stop["endpoint_forward_displacement_m"] == pytest.approx(.052)
    band = stop["zero_speed_bands"]["0.1"]
    assert band["first_held"]["time_s"] == .02
    assert band["first_held"]["confirmed_at_s"] == .52
    assert band["endpoint_continuous_observed_s"] == .5
    assert stop["zero_speed_bands"]["0.05"]["first_held"] is None


def test_invalid_measurement_breaks_hold_and_path_without_forgiving_failed_episode():
    measurement = TemporalEpisodeMeasurements([-3., 0., .3], [0., 0., 0.], 0.)
    for tick in range(1, 277):
        measurement.observe(frame(tick, measurement_valid=tick != 270,
            root_link_position_world=[0., 0., 0.], root_com_velocity_body=[0., 0., 0.]))
    report = measurement.report("terminated")
    stop = report["phases"][1]
    assert not report["qualification"] and not report["completed_four_phase_horizon"]
    assert not stop["path_contiguous"] and stop["invalid_frames"] == 1
    assert stop["zero_speed_bands"]["0.2"]["first_held"] is None
    assert stop["zero_speed_bands"]["0.2"]["endpoint_consecutive_samples"] == 6
    with pytest.raises(ValueError, match="reset"):
        measurement.observe(frame(1))


@pytest.mark.parametrize("key,value", [(FIELD, {"phase_policy_ticks": [250, 250, 250, 250]}),
    ("actor_mode", "sampled"), ("heading_hold", "initial_heading_gain5_limit2pi_v1")])
def test_unreviewed_sequence_or_controller_is_rejected(key, value):
    with pytest.raises(ValueError):
        validate_protocol({**protocol(), key: value})


def test_named_contact_intervention_is_explicit_and_cannot_select_a_different_surface():
    from wheeled_tasks.chassis.simulation.terrain.contact_generation_distance import VERSION, UPPER_LINK_VERSION
    p = protocol()
    overlay = dict(num_envs=8, lane_case_ids=[c["case_id"] for c in p["cases"]],
        record_diagnostics=True, contact_generation_distance=VERSION)
    assert validate_overlay(p, overlay) == overlay
    updated = dict(overlay, contact_generation_distance=UPPER_LINK_VERSION)
    assert validate_overlay(p, updated) == updated
    for change in ({"contact_generation_distance": "unknown"}, {"evaluation_surface": "plane_regression_v1"}):
        with pytest.raises(ValueError):
            validate_overlay(p, {**overlay, **change})
    ordinary = deepcopy(p)
    ordinary.pop(FIELD)
    with pytest.raises(ValueError):
        validate_overlay(ordinary, overlay)


@pytest.mark.parametrize("parent_name,stage,local,global_clock", [
    ("recovery_after10000_056", "foundation", 5000, 14841),
    ("mixed_tracking_repair_063", "terrain", 159, 15000),
])
def test_authentic_parent_clocks_and_physics_survive_inference_only_projection(
        tmp_path, parent_name, stage, local, global_clock):
    config = json.loads((ROOT / f"configs/v7_formal_20261012/{parent_name}.json").read_bytes())
    original = deepcopy(config)
    effective = configuration(config)
    assert config == original
    changed = {k for k in config.keys() | effective.keys() if config.get(k) != effective.get(k)}
    assert not changed & {"asset_directory", "asset_manifest_sha256", "v5_control", "physics_dt",
        "policy_dt", "physx_solver_type", "position_iterations", "velocity_iterations", "actuator_response",
        "round_trip_transport", "wheel_material_priors", "pc_relative_priors", "dynamics_randomization",
        "formal_contact_generation_distance", "gas_pressure_application"}
    assert "recovery_training" not in effective and "formal_small_rough_mix" not in effective
    assert effective["formal_foundation_nominal_birth"] is True
    p = protocol()
    paths = {}
    for name, value in (("contract", config), ("protocol", p),
            ("overlay", dict(num_envs=8, lane_case_ids=[c["case_id"] for c in p["cases"]], record_diagnostics=True))):
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_bytes(canonical(value))
    hashes = {name: sha((ROOT / name).read_bytes()) for name in runtime_source_files(config)}
    infos = {**profile_identity(config, hashes), "profile_source_sha256": hashes,
        "contract_sha256": sha(paths["contract"].read_bytes()),
        "asset_manifest_sha256": config["asset_manifest_sha256"],
        "control_math_sha256": config["control_math_sha256"], "stage": stage,
        "stage_actor_update": local, "global_actor_update": global_clock,
        "global_actor_update_offset": global_clock - local, "actor_updates_consumed": global_clock + 1753}
    paths["checkpoint"] = tmp_path / "cpu_fixture.pt"
    torch.save(dict(infos=infos, actor_state_dict=stock_state()), paths["checkpoint"])
    paths.update(root=ROOT, manifest=ROOT / config["asset_directory"] / "manifest.json",
        control=ROOT / config["control_math_source"])
    bound = BoundEvaluationInputs.capture(**paths)
    assert bound.authenticate()["stage"] == stage
    assert bound.execution_stage == "foundation" and not bound.is_rough
    setup = bound.build_setup()
    assert (setup.stage_actor_update, setup.global_actor_update) == (local, global_clock)
    from wheeled_tasks.chassis.application.configuration.preflight import design_preflight
    design_preflight(bound.evaluation_config(), bound.control, ROOT, stage=bound.execution_stage,
        role=bound.run_role, seed=p["environment_seed"], num_envs=8, evaluation_setup=setup)
