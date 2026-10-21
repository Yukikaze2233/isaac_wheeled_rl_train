"""Synthetic pre-reset rows for the optional fixed-heading inference protocol."""
from copy import deepcopy
import math
import struct

import pytest

from wheeled_tasks.chassis.application.evaluation.flat_policy_evaluation import (
    CENTER_REFERENCE, HEADING_HOLD_VERSION, SCHEMA, FlatEpisodeLedger, validate_protocol,
)
from wheeled_tasks.chassis.tasks.rewards.flat_reward import WEIGHTS


def protocol(vx=1., **changes):
    value = dict(schema=SCHEMA, mode="fixed_measurement", actor_mode="mean", episodes_per_lane=1,
        environment_seed=17, actor_seed=31, policy_dt_s=.02, horizon_seconds=20.,
        center_reference=CENTER_REFERENCE, assessment="measurement_only",
        cases=[dict(case_id="translation", base_request=[vx, 0., .3])],
        heading_hold=HEADING_HOLD_VERSION)
    value.update(changes)
    return value


def ledger(vx=1., reset_yaw=0., **changes):
    obj = FlatEpisodeLedger(protocol(vx, **changes), ["translation"], list(WEIGHTS))
    begin(obj, reset_yaw)
    return obj


def begin(obj, yaw, episode_id=0):
    obj.begin_episode(0, episode_id, root_link_position_world=[0., 0., .3], yaw_rad=yaw,
        reset_metadata={"physical_tick": 0, "source": "explicit synthetic returned reset"})


def row(tick=1, vx=1., command_yaw=0., yaw=0., **changes):
    value = dict(episode_id=0, episode_tick=tick, reward=0., payments={name: 0. for name in WEIGHTS},
        done=False, terminated=False, timeout=False, boundary=False, termination_reasons=[],
        learning_mask=True, bootstrap_mask=True, measurement_valid=True, command_override_active=False,
        effective_command=[vx, command_yaw, .3], base_request=[vx, 0., .3],
        root_link_position_world=[tick * .02 * vx, 0., .3], root_com_velocity_body=[vx, 0., 0.],
        angular_velocity_body=[0., 0., .1], yaw_rad=yaw)
    value.update(changes)
    return value


def terminal(**changes):
    return dict(done=True, terminated=True, termination_reasons=["synthetic_contact"], **changes)


def wrapped_command(reference, previous):
    delta = reference - previous
    return max(-2. * math.pi, min(2. * math.pi, 5. * math.atan2(math.sin(delta), math.cos(delta))))


@pytest.mark.parametrize("value", [True, None, "legacy", "initial_heading_gain6_limit2pi_v1"])
def test_only_declared_heading_controller_is_accepted(value):
    with pytest.raises(ValueError, match="fixed initial-heading controller"):
        validate_protocol(protocol(heading_hold=value))


def test_heading_controller_requires_fixed_mode_and_translation_case():
    source = protocol(mode="source_distribution")
    source.pop("cases")
    with pytest.raises(ValueError, match="fixed measurement mode"):
        validate_protocol(source)
    for command in ([0., 0., .3], [0., 2., .3], [1., .1, .3]):
        with pytest.raises(ValueError, match="at least one pure translation"):
            validate_protocol(protocol(cases=[dict(case_id="translation", base_request=command)]))


def test_default_still_rejects_changed_fixed_yaw_without_heading_metadata():
    p = protocol()
    p.pop("heading_hold")
    obj = FlatEpisodeLedger(p, ["translation"], list(WEIGHTS))
    begin(obj, 0.)
    with pytest.raises(ValueError, match="Fixed command differs"):
        obj.observe(0, row(command_yaw=.1))
    record = obj.observe(0, row(**terminal()))
    assert "heading_hold" not in record
    assert record["measurements"]["all"]["body_omega_z_abs_mean_rad_s"] == .1


@pytest.mark.parametrize("vx", [2.7, -2.7])
def test_controller_uses_previous_yaw_and_preserves_original_translation_kind(vx):
    obj = ledger(vx, reset_yaw=1.4)
    obj.observe(0, row(vx=vx, yaw=1.5, angular_velocity_body=[0., 0., -.2]))
    record = obj.observe(0, row(2, vx=vx, yaw=1.49, command_yaw=-.5, **terminal()))
    assert set(record["measurements"]) == {"all", "pure_translation"}
    assert record["base_request_last"] == [vx, 0., .3]
    assert record["effective_command_last"] == [vx, -.5, .3]
    assert record["heading_hold"]["reference_yaw_rad"] == 1.4
    assert "preceding accepted pre-reset yaw" in record["heading_hold"]["feedback_yaw_source"]
    assert record["measurements"]["all"]["body_omega_z_abs_mean_rad_s"] == pytest.approx(.15)
    assert record["measurements"]["all"]["body_omega_z_tracking_rmse_rad_s"] != pytest.approx(.15)
    assert record["command_override_frames"] == 0


@pytest.mark.parametrize("reference,previous", [
    (math.pi - .03, -math.pi + .02), (-math.pi + .03, math.pi - .02),
    (0., -2.), (0., 2.),
])
def test_wrapping_and_symmetric_limit_follow_previous_not_current_yaw(reference, previous):
    obj = ledger(reset_yaw=reference)
    obj.observe(0, row(yaw=previous))
    f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
    record = obj.observe(0, row(2, yaw=reference, command_yaw=f32(wrapped_command(reference, previous)),
        **terminal()))
    assert record["heading_hold"]["reference_yaw_rad"] == reference


@pytest.mark.parametrize("change", [
    {"command_yaw": .1}, {"effective_command": [1.1, 0., .3]},
    {"effective_command": [1., 0., .31]}, {"base_request": [1., .1, .3]},
])
def test_heading_option_does_not_relax_yaw_forward_height_or_original_request(change):
    obj = ledger()
    before = deepcopy(obj.active[0]["tick"])
    with pytest.raises(ValueError, match="Heading hold effective command|Fixed base request"):
        obj.observe(0, row(**change))
    assert obj.active[0]["tick"] == before


def test_source_override_is_separate_and_does_not_reanchor_heading_reference():
    obj = ledger()
    obj.observe(0, row(command_override_active=True, effective_command=[0., .7, .3], yaw=.4))
    record = obj.observe(0, row(2, command_yaw=-2., yaw=.35, **terminal()))
    assert record["heading_hold"]["reference_yaw_rad"] == 0.
    assert record["command_override_frames"] == 1
    assert set(record["measurements"]) == {"all", "capture_override", "pure_translation"}
    assert record["measurements"]["pure_translation"]["valid_frames"] == 1


def test_invalid_preceding_measurement_cannot_be_used_as_controller_feedback():
    obj = ledger()
    obj.observe(0, row(measurement_valid=False, yaw=float("nan")))
    with pytest.raises(ValueError, match="valid preceding pre-reset yaw"):
        obj.observe(0, row(2))
    obj.observe(0, row(2, command_override_active=True, yaw=.2))
    record = obj.observe(0, row(3, command_yaw=-1., yaw=.1, **terminal()))
    assert record["invalid_measurement_frames"] == 1


def test_only_real_episode_reset_reanchors_controller():
    obj = ledger(episodes_per_lane=2)
    obj.observe(0, row(yaw=.4, **terminal()))
    begin(obj, 1.1, episode_id=1)
    record = obj.observe(0, row(episode_id=1, yaw=1.2, **terminal()))
    assert record["heading_hold"]["reference_yaw_rad"] == 1.1


def test_spin_lane_remains_strict_under_protocol_with_translation_lane():
    p = protocol()
    p["cases"].append(dict(case_id="spin", base_request=[0., 2., .3]))
    obj = FlatEpisodeLedger(p, ["translation", "spin"], list(WEIGHTS))
    obj.begin_episode(1, 0, root_link_position_world=[0., 0., .3], yaw_rad=0., reset_metadata={"tick": 0})
    spin = row(vx=0., command_yaw=1., base_request=[0., 2., .3])
    with pytest.raises(ValueError, match="Fixed command differs"):
        obj.observe(1, spin)
    spin["effective_command"][1] = 2.
    spin.update(terminal())
    assert set(obj.observe(1, spin)["measurements"]) == {"all", "pure_spin"}
