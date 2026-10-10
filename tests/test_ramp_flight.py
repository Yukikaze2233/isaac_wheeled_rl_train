"""Gap geometry and failure-credit boundaries for a requested ramp-top jump."""
from dataclasses import asdict
import math

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.ramp_flight import RampFlightEvidence, RampFlightGeometry


@pytest.mark.parametrize("direction", [1, -1])
def test_exact_lip_and_gap_edges_with_no_rotated_box_overhang(direction):
    geometry = RampFlightGeometry(direction=direction)
    floor, approach, ramp, landing = geometry.meshes()
    ramp_points = np.asarray(ramp[0])
    landing_points = np.asarray(landing[0])
    assert np.max(direction * ramp_points[:, 0]) == pytest.approx(0.)
    lip = ramp_points[np.isclose(ramp_points[:, 0], 0.)]
    assert lip[:, 2].max() == pytest.approx(.553)
    assert np.max(np.abs(ramp_points[:, 1])) == pytest.approx(.430)
    assert np.min(direction * landing_points[:, 0]) == pytest.approx(.650123)
    assert np.max(landing_points[:, 2]) == pytest.approx(.200)
    assert math.degrees(math.atan((.553 - .203) / geometry.ramp_run_m)) == pytest.approx(17.)
    # Sample the top surface, not the solid's bounding box or underside.
    query = torch.tensor([[direction * -.1, 0.], [direction * .3, 0.],
                          [direction * .8, 0.], [direction * -.1, .431],
                          [direction * .8, .576]])
    height = geometry.support_height(query)
    assert height[0] > .5
    assert height[1] == 0.
    assert height[2] == pytest.approx(.2)
    assert height[3] == 0. and height[4] == 0.
    for points, faces in (floor, approach, ramp, landing):
        assert len(points) == 8 and len(faces) == 24
        points = np.asarray(points)
        center = points.mean(0)
        for face in np.asarray(faces).reshape(6, 4):
            polygon = points[face]
            normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
            assert np.dot(normal, polygon.mean(0) - center) > 0


def test_only_gap_advances_with_geometry_level_and_two_halves_are_mirrored():
    limits = {"ramp_flight_geometry": dict(asdict(RampFlightGeometry()), gap_start_m=.15)}
    easy = RampFlightGeometry.from_limits(limits, 0.)
    full = RampFlightGeometry.from_limits(limits, 1.)
    assert easy.gap_m == .15 and full.gap_m == .650123
    assert easy.ramp_run_m == full.ramp_run_m and easy.lip_height_m == full.lip_height_m
    positive = np.asarray(full.meshes()[2][0])
    negative = np.asarray(RampFlightGeometry(direction=-1).meshes()[2][0])
    assert {tuple(point) for point in negative} == {
        tuple(point * (-1, 1, 1)) for point in positive}


@pytest.mark.parametrize("changes", [dict(direction=True), dict(gap_m=0.),
    dict(ramp_run_m=float("nan")), dict(lip_height_m=.1),
    dict(landing_height_m=.6), dict(floor_width_m=.2)])
def test_geometry_rejects_invalid_units_or_dimensions(changes):
    with pytest.raises(ValueError):
        RampFlightGeometry(**changes)
    with pytest.raises(ValueError):
        RampFlightGeometry.from_limits({}, 1.)
    with pytest.raises(ValueError):
        RampFlightGeometry.from_limits({"ramp_flight_geometry": {"unknown": 1}}, 1.)


def rig(direction=1, count=1):
    geometry = RampFlightGeometry(direction=direction)
    observer = RampFlightEvidence([geometry] * count, "cpu", .005)
    wheel = torch.tensor([[[direction * -.10, -.22, .583],
                           [direction * -.10, .22, .583]]] * count)
    data = dict(wheel_centers=wheel, contacts=torch.ones(count, 2, dtype=torch.bool),
                surface_contacts=torch.zeros(count, 2, 4, dtype=torch.bool),
                com_position=torch.tensor([[direction * -.1, 0., .75]] * count),
                com_velocity=torch.tensor([[direction * 2., 0., .5]] * count),
                stable=torch.zeros(count, dtype=torch.bool), body_contact=torch.zeros(count, dtype=torch.bool),
                accepted_jump_event=torch.ones(count, dtype=torch.bool))
    data["surface_contacts"][..., 2] = True
    return observer, data


def release(observer, data, ticks=12):
    observer.observe(**data)
    data["accepted_jump_event"].zero_()
    data["contacts"].zero_()
    data["surface_contacts"].zero_()
    for _ in range(ticks):
        observer.observe(**data)


def land(observer, data, direction=1):
    data["wheel_centers"][..., 0] = direction * .8
    data["wheel_centers"][..., 2] = .26
    data["com_position"][:, 0] = direction * .8
    data["com_position"][:, 2] = .505
    data["com_velocity"][:, 2] = -.4
    data["contacts"].fill_(True)
    data["surface_contacts"][..., 3] = True
    data["stable"].fill_(True)
    observer.observe(**data)


@pytest.mark.parametrize("direction", [1, -1])
def test_lip_jump_preserves_first_loss_sample_and_requires_full_stable_second(direction):
    observer, data = rig(direction)
    release(observer, data, 11)
    assert not observer.verified_flight.item()
    data["com_position"][:, 2] = .8
    data["com_velocity"][:, 2] = .1
    observer.observe(**data)
    assert observer.verified_flight.item()
    assert observer.release_position[0, 2] == .75
    assert observer.release_velocity[0, 2] == .5
    assert observer.consume_flight_events().item()
    assert not observer.consume_flight_events().item()
    land(observer, data, direction)
    assert observer.landed.item() and observer.crossed_gap.item()
    assert not observer.success.item()
    for _ in range(198):
        observer.observe(**data)
    assert observer.stable_peak_ticks.item() == 199 and not observer.success.item()
    observer.observe(**data)
    assert observer.success.item() and observer.stable_peak_ticks.item() == 200
    # Forward motion remains valid; this observer never changes its command.
    assert data["com_velocity"][0, 0] == direction * 2.


@pytest.mark.parametrize("fault", ["floor_start", "stationary_release", "short_air",
    "floor_landing", "side_landing", "body_contact", "contact_mismatch"])
def test_non_flying_routes_and_wrong_landings_cannot_get_full_credit(fault):
    observer, data = rig()
    if fault == "floor_start":
        data["surface_contacts"].zero_()
        data["surface_contacts"][..., 0] = True
    elif fault == "stationary_release":
        data["com_velocity"][:, 0] = 0.
    release(observer, data, 11 if fault == "short_air" else 12)
    if fault in ("floor_start", "stationary_release", "short_air"):
        assert not observer.verified_flight.item()
    if fault == "body_contact":
        data["body_contact"].fill_(True)
    land(observer, data)
    if fault in ("floor_landing", "side_landing", "contact_mismatch"):
        # A fresh observer sees the incorrect first contact, not a later edit
        # to an already valid platform landing.
        observer, data = rig()
        release(observer, data)
        data["contacts"].fill_(True)
        data["wheel_centers"][..., 0] = .8
        data["wheel_centers"][..., 2] = .26
        if fault == "floor_landing":
            data["surface_contacts"][..., 0] = True
            data["wheel_centers"][..., 2] = .06
        else:
            data["surface_contacts"][..., 3] = True
            if fault == "side_landing":
                data["wheel_centers"][..., 1] = .8
            else:
                data["contacts"].zero_()
        data["stable"].fill_(True)
    for _ in range(300):
        observer.observe(**data)
    assert not observer.success.item()


def test_short_bounce_cannot_accumulate_air_credit_and_a_single_bad_stable_tick_resets_hold():
    observer, data = rig()
    for _ in range(3):
        release(observer, data, 8)
        data["contacts"].fill_(True)
        data["surface_contacts"][..., 2] = True
        observer.observe(**data)
    assert not observer.verified_flight.item() and observer.air_peak_ticks.item() == 8
    release(observer, data)
    land(observer, data)
    for _ in range(99):
        observer.observe(**data)
    data["stable"].zero_()
    observer.observe(**data)
    assert observer.stable_ticks.item() == 0
    data["stable"].fill_(True)
    for _ in range(199):
        observer.observe(**data)
    assert not observer.success.item()
    observer.observe(**data)
    assert observer.success.item()
    observer.reset(torch.tensor([0]))
    assert not observer.landed.item() and observer.air_peak_ticks.item() == 0
    assert not observer.jump_accepted.item()
    assert not observer.consume_flight_events().item()


def test_disabled_lanes_and_unobserved_touchdown_remain_distinct():
    observer, data = rig(count=2)
    observer = RampFlightEvidence([RampFlightGeometry(), None], "cpu", .005)
    release(observer, data)
    diagnostics = observer.diagnostics()
    assert diagnostics["ramp_flight_verified"].tolist() == [True, False]
    assert diagnostics["ramp_flight_touchdown_observed"].tolist() == [False, False]
    # Zero-filled storage is not a measured zero landing error.
    assert observer.touchdown_position.eq(0.).all()


def test_observer_invalid_state_cannot_receive_credit():
    observer, data = rig()
    data["com_position"][0, 0] = float("nan")
    observer.observe(**data)
    assert observer.failed.item() and observer.invalid_state.item()
    assert not observer.verified_flight.item() and not observer.success.item()


def test_touching_the_vertical_ramp_wall_from_the_floor_is_not_top_support():
    observer, data = rig()
    data["wheel_centers"][..., 2] = .06
    observer.observe(**data)
    assert not observer.previous_ramp_support.item()
    data["contacts"].zero_()
    data["surface_contacts"].zero_()
    data["wheel_centers"][..., 2] = .60
    for _ in range(20):
        observer.observe(**data)
    assert not observer.verified_flight.item() and not observer.pending.item()


def test_unrequested_ballistic_route_cannot_complete_the_lip_jump_task():
    observer, data = rig()
    data["accepted_jump_event"].zero_()
    release(observer, data)
    land(observer, data)
    for _ in range(250):
        observer.observe(**data)
    assert not observer.jump_accepted.item()
    assert not observer.verified_flight.item() and not observer.success.item()


def test_accepted_request_without_physical_release_has_no_flight_credit():
    observer, data = rig()
    observer.observe(**data)
    data["accepted_jump_event"].zero_()
    for _ in range(250):
        observer.observe(**data)
    assert observer.diagnostics()["ramp_flight_jump_accepted"].item()
    assert not observer.verified_flight.item() and not observer.success.item()


def test_an_approach_request_does_not_replace_the_ramp_jump_event():
    observer, data = rig()
    data["surface_contacts"][..., 2] = False
    data["surface_contacts"][..., 1] = True
    observer.observe(**data)
    data["accepted_jump_event"].zero_()
    data["surface_contacts"][..., 1] = False
    data["surface_contacts"][..., 2] = True
    release(observer, data)
    assert not observer.jump_accepted.item() and not observer.verified_flight.item()


@pytest.mark.parametrize("direction", [1, -1])
def test_preload_can_begin_on_the_incline_but_release_must_be_near_the_lip(direction):
    observer, data = rig(direction)
    data["wheel_centers"][..., 0] = direction * -.5
    data["wheel_centers"][..., 2] = .553 - .5 * math.tan(math.radians(17.)) + .06
    observer.observe(**data)
    assert observer.jump_accepted.item() and not observer.previous_ramp_support.item()
    data["accepted_jump_event"].zero_()
    data["wheel_centers"][..., 0] = direction * -.1
    data["wheel_centers"][..., 2] = .583
    release(observer, data)
    assert observer.verified_flight.item()


def test_acceptance_at_first_physics_release_uses_previous_top_support():
    observer, data = rig()
    data["accepted_jump_event"].zero_()
    observer.observe(**data)
    data["contacts"].zero_()
    data["surface_contacts"].zero_()
    data["accepted_jump_event"].fill_(True)
    observer.observe(**data)
    data["accepted_jump_event"].zero_()
    for _ in range(11):
        observer.observe(**data)
    assert observer.jump_accepted.item() and observer.verified_flight.item()


def test_accepted_jump_that_releases_far_from_the_lip_is_not_a_ramp_top_jump():
    observer, data = rig()
    data["wheel_centers"][..., 0] = -.5
    data["wheel_centers"][..., 2] = .553 - .5 * math.tan(math.radians(17.)) + .06
    release(observer, data)
    assert observer.jump_accepted.item() and not observer.verified_flight.item()


def test_accepted_event_requires_a_boolean_pulse_per_lane():
    observer, data = rig()
    data["accepted_jump_event"] = torch.ones(1)
    with pytest.raises(ValueError, match="accepted_jump_event requires boolean"):
        observer.observe(**data)
