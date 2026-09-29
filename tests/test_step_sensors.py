"""Sensor adapter invariants: one ray, encoder geometry, and an IMU sample."""
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.step_sensors import StepTaskSensors
from wheeled_tasks.chassis.step_jump import StepJumpTask
from wheeled_tasks.chassis.step_trigger import single_beam_range_2d
from wheeled_tasks.chassis.task import terrain_surfaces


ROOT = Path(__file__).resolve().parents[1]
LOOKUP = ROOT / "contracts/v6_height_lookup_v1.json"
NOMINAL = json.loads((ROOT / "model/纯底盘_v6_232mm_stop105/urdf/model_spec.json").read_text())["nominal_joint_pos"]
AXES = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]


def settings(**changes):
    return {"height_lookup_file": str(LOOKUP.relative_to(ROOT)),
            "height_lookup_sha256": hashlib.sha256(LOOKUP.read_bytes()).hexdigest(),
            "range_sample_period_s": .02, "sensor_origin_b_m": [.1, -.2],
            "sensor_pitch_rad": .55, "ground_echo_margin_m": .015,
            "range_max_m": 4., "range_noise_std_m": .003,
            "range_dropout_probability": .002, **changes}


def inputs(count=2):
    surfaces = terrain_surfaces("stairs", {"stair_heights_m": [.2, .45], "stair_tread_m": [.8, .6]}, 1.)
    surface_data = torch.tensor([[s.x0, s.x1, s.z0, s.slope] for s in surfaces])
    local = torch.tensor([[-.24, 0., .32]]).repeat(count, 1)
    return {"time": torch.zeros(count), "q": torch.tensor([[NOMINAL[name] for name in AXES]]).repeat(count, 1),
            "dq": torch.zeros(count, 6), "omega": torch.zeros(count, 3),
            "gravity": torch.tensor([[0., 0., -1.]]).repeat(count, 1),
            "pose": torch.cat((local, torch.tensor([[0., 0., 0., 1.]]).repeat(count, 1)), dim=-1),
            "velocity_world": torch.zeros(count, 3), "local_position": local,
            "surfaces": surface_data[None].repeat(count, 1, 1),
            "surface_valid": torch.ones(count, len(surfaces), dtype=torch.bool)}


def sensor(count=2, seed=19, evaluation=True, **changes):
    return StepTaskSensors(count, "cpu", .02, settings(**changes), ROOT, seed, evaluation=evaluation)


def test_adapter_rejects_wrong_lookup_hash():
    with pytest.raises(ValueError, match="frozen contract"):
        sensor(height_lookup_sha256="0" * 64)


def test_conditional_height_uses_encoders_and_retains_individual_side_validity():
    sample = inputs()
    adapter = sensor()
    adapter.reset(torch.arange(2), sample["velocity_world"])
    adapter.update(**sample)
    assert adapter.support.tolist() == [True, True]
    before = adapter.height.clone()
    # Moving the simulator root without changing encoder/IMU signals must not
    # change the conditional FK height (it does change the separate laser ray).
    sample["local_position"][:, 2] += .2
    sample["pose"][:, 2] += .2
    adapter.update(**sample)
    torch.testing.assert_close(adapter.height, before)
    sample["q"][0, 1] = 20.
    adapter.update(**sample)
    assert adapter.height_valid.tolist() == [[False, True], [True, True]]
    assert adapter.support.tolist() == [False, True]


def test_freefall_is_not_support_and_repeated_time_does_not_forge_static_imu():
    sample, adapter = inputs(), sensor()
    adapter.reset(torch.arange(2), sample["velocity_world"])
    adapter.update(**sample)
    sample["time"][:] = .02
    sample["velocity_world"][:, 2] = -9.81 * .02
    adapter.update(**sample)
    assert adapter.force_norm.max() < 1e-5
    assert not adapter.support.any()
    before = {key: value.clone() for key, value in adapter.diagnostics().items()}
    adapter.update(**sample)
    for key, value in adapter.diagnostics().items():
        torch.testing.assert_close(value, before[key])
    assert not adapter.support.any()


def test_single_beam_lane_heading_and_unseen_riser_are_invalid():
    sample, adapter = inputs(4), sensor(4)
    # The down-tilted beam hits the 20 cm riser before reaching the floor.
    sample["local_position"][1, 1] = 2.
    sample["pose"][2, 5:7] = torch.tensor([math.sin(.3 / 2), math.cos(.3 / 2)])
    sample["local_position"][3, 2] = 1.
    adapter.update(**sample)
    assert adapter.range_valid.tolist() == [True, False, False, False]
    assert adapter.range[0].item() == pytest.approx(.14 / math.cos(.55))
    assert adapter.forward_distance[0].item() == pytest.approx(.24)


def test_sample_period_holds_range_and_exposes_age_until_next_sample():
    sample, adapter = inputs(), sensor(range_sample_period_s=.04)
    adapter.update(**sample)
    original = adapter.range.clone()
    original_forward = adapter.forward_distance.clone()
    sample["time"][:] = .02
    sample["local_position"][:, 0] += .1
    adapter.update(**sample)
    torch.testing.assert_close(adapter.range, original)
    torch.testing.assert_close(adapter.forward_distance, original_forward)
    torch.testing.assert_close(adapter.range_age, torch.full((2,), .02))
    sample["time"][:] = .04
    adapter.update(**sample)
    torch.testing.assert_close(adapter.range, original - .1 / math.cos(.55))
    torch.testing.assert_close(adapter.forward_distance, original_forward - .1)
    assert not adapter.range_age.any()


def test_sensor_randomness_has_its_own_seed_and_evaluation_disables_noise_dropout():
    a, b = sensor(evaluation=False), sensor(evaluation=False)
    other = sensor(seed=20, evaluation=False)
    sample = inputs()
    different = False
    for tick in range(5):
        sample["time"][:] = tick * .02
        a.update(**sample)
        torch.rand(37)
        b.update(**sample)
        other.update(**sample)
        torch.testing.assert_close(a.range, b.range)
        torch.testing.assert_close(a.range_valid, b.range_valid)
        torch.testing.assert_close(a.forward_distance, b.forward_distance)
        different |= not torch.equal(a.range, other.range)
    assert different
    frozen = sensor(evaluation=True, range_noise_std_m=10., range_dropout_probability=1.)
    frozen.update(**inputs())
    assert frozen.range_valid.all()
    torch.testing.assert_close(frozen.range, torch.full((2,), .14 / math.cos(.55)))


def test_partial_reset_reinitializes_only_selected_imu_and_range_clock():
    sample, adapter = inputs(), sensor()
    adapter.update(**sample)
    sample["velocity_world"][:] = torch.tensor([[1., 0., 0.], [2., 0., 0.]])
    previous_other = adapter.imu.previous_velocity[1].clone()
    adapter.reset(torch.tensor([0]), sample["velocity_world"])
    torch.testing.assert_close(adapter.imu.previous_velocity[0], sample["velocity_world"][0])
    torch.testing.assert_close(adapter.imu.previous_velocity[1], previous_other)
    assert not adapter.range_valid[0] and adapter.range_valid[1]
    assert torch.isneginf(adapter.sample_time[0]) and adapter.sample_time[1] == 0.


def test_real_floor_echo_is_filtered_using_encoder_imu_endpoint_height():
    sample, adapter = inputs(), sensor()
    sample["surfaces"] = torch.tensor([[[-4., 4., 0., 0.]]]).repeat(2, 1, 1)
    sample["surface_valid"] = torch.ones(2, 1, dtype=torch.bool)
    distance, hit = single_beam_range_2d(
        sample["local_position"][:, [0, 2]], torch.zeros(2), sample["surfaces"], sample["surface_valid"],
        sensor_origin_b=(.1, -.2), sensor_pitch_rad=.55,
    )
    assert hit.all()  # A real geometric echo exists; this is not a no-hit test.
    assert distance.min() > 0.
    adapter.update(**sample)
    assert adapter.height_valid.all() and adapter.support.all()
    assert not adapter.range_valid.any()


@pytest.mark.parametrize("height", [.03, .06])
def test_down_tilted_single_beam_can_see_3cm_and_6cm_platforms(height):
    sample, adapter = inputs(), sensor()
    surfaces = terrain_surfaces("step_up", {"step_up_m": height}, 1.)
    sample["surfaces"] = torch.tensor([[[s.x0, s.x1, s.z0, s.slope] for s in surfaces]]).repeat(2, 1, 1)
    sample["surface_valid"] = torch.ones(2, len(surfaces), dtype=torch.bool)
    adapter.update(**sample)
    # At x=-.24 the 3 cm echo is on the top and the 6 cm echo on the riser.
    # Both are above the inferred low support plane and pass the same filter.
    expected_slant = max((.32 - .2 - height) / math.sin(.55), .14 / math.cos(.55))
    assert adapter.range_valid.all()
    torch.testing.assert_close(adapter.range, torch.full((2,), expected_slant))
    torch.testing.assert_close(adapter.forward_distance, torch.full((2,), .1 + expected_slant * math.cos(.55)))


def test_endpoint_filter_requires_two_valid_supported_sides():
    sample, adapter = inputs(), sensor()
    sample["q"][0, 1] = 20.
    sample["omega"][1, 0] = 4.
    adapter.update(**sample)
    assert not adapter.range_valid.any()
    assert adapter.height_valid.tolist() == [[False, True], [False, False]]


@pytest.mark.parametrize("height", [.03, .06, 0.])
def test_nominal_fk_forward_approach_triggers_low_steps_but_never_the_floor(height):
    adapter, sample = sensor(1), inputs(1)
    adapter.update(**sample)
    # Use the actual sealed nominal FK stance height, not an independently
    # invented robot-height truth inside the deployable request controller.
    nominal_height = float(adapter.height.max())
    sample["local_position"][:, 2] = sample["pose"][:, 2] = nominal_height
    surfaces = terrain_surfaces("step_up", {"step_up_m": height}, 1.) if height else terrain_surfaces("flat", {}, 1.)
    sample["surfaces"] = torch.tensor([[[s.x0, s.x1, s.z0, s.slope] for s in surfaces]])
    sample["surface_valid"] = torch.ones(1, len(surfaces), dtype=torch.bool)
    settings = {"rises_m": [height or .03], "treads_m": [.8], "launch_distance_m": .25,
                "approach_speed_m_s": .4, "landing_margin_m": .1, "settle_seconds": .35,
                "landing_clearance_m": .08, "trigger_source": "laser"}
    task = StepJumpTask(["step"], {"step": {"step_jump": settings}}, "cpu", .02)
    sample["velocity_world"][:, 0] = .4
    sample["dq"][:, 2], sample["dq"][:, 5] = .4 / .06, -.4 / .06
    adapter.reset(torch.tensor([0]), sample["velocity_world"])
    accepted_x = None
    for tick in range(94):
        sample["time"][:] = tick * .02
        sample["local_position"][:, 0] = -.8 + tick * .008
        sample["pose"][:, 0] = sample["local_position"][:, 0]
        wheel_speed = adapter.update(**sample)
        event = task.control(sample["time"], wheel_speed, adapter.support, adapter.force_norm,
            adapter.range, adapter.range_valid, adapter.range_age,
            beam_forward_distance=adapter.forward_distance)
        if event.item():
            accepted_x = float(sample["local_position"][0, 0])
            assert adapter.range_valid.item()
            assert 0 < adapter.forward_distance.item() <= .25
            break
    if height:
        assert accepted_x is not None and -.25 <= accepted_x < 0.
        assert task.request_count.item() == 1
    else:
        assert accepted_x is None and task.request_count.item() == 0
