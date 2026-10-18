"""CPU pre-reset measurement ownership; no simulator or policy updates."""
import math

import pytest
import torch

from support_flat_policy_evaluation import begin, fixed, frame, terminal
from wheeled_tasks.chassis.flat_evaluation_loop import _CaseTrackingMeasurements
from wheeled_tasks.chassis.flat_policy_evaluation import FlatEpisodeLedger
from wheeled_tasks.chassis.flat_evaluation_runtime import PreResetLedgerAdapter
from wheeled_tasks.chassis.scut_flat_reward import WEIGHTS


def test_case_tracking_terminal_invalid_reset_and_exhausted_quota_frames():
    protocol = fixed()
    protocol["cases"] = [{"case_id": name, "base_request": [1., 0., .3]}
                         for name in ("early", "late")]
    ledger = FlatEpisodeLedger(protocol, ["early", "late"], list(WEIGHTS))
    for lane in range(2):
        begin(ledger, lane)
    measurements = _CaseTrackingMeasurements(ledger.lane_cases, .02, required=True)

    def accept(rows, heights, gaps):
        diagnostics = {"height": torch.tensor(heights), "gap": torch.tensor(gaps),
            "effective_command": torch.tensor([r["effective_command"] for r in rows]),
            "measurement_valid_mask": torch.tensor([r["measurement_valid"] for r in rows]),
            "root_COM_velocity_body": torch.tensor([r["root_com_velocity_body"] for r in rows]),
            "angular_velocity_body": torch.tensor([r["angular_velocity_body"] for r in rows])}
        active = [lane for lane, episode in enumerate(ledger.active) if episode is not None]
        before = measurements.report()
        pending = measurements.prepare(diagnostics, active)
        assert measurements.report() == before  # Preparation does not book an unaccepted frame.
        for lane in active:
            ledger.observe(lane, rows[lane])
        measurements.accept(pending)

    # A valid terminal row counts even when excluded from PPO. The other lane's
    # invalid NaN measurements must not affect any error or gap statistic.
    accept([terminal(learning_mask=False, root_com_velocity_body=[.4, 0., 0.],
                     angular_velocity_body=[0., 0., .4]),
            frame(measurement_valid=False, learning_mask=False)], [.5, math.nan], [.004, math.nan])
    # Native batches continue after early exhausts its quota. Its newborn rows
    # are deliberately enormous, and must never enter the case measurements.
    newborn = frame(episode_id=1, root_com_velocity_body=[99., 0., 0.])
    accept([newborn, frame(2, learning_mask=False, root_com_velocity_body=[1.2, 0., 0.],
                          angular_velocity_body=[0., 0., -.2])], [99., .4], [99., .003])
    accept([newborn, terminal(3, learning_mask=False, root_com_velocity_body=[.6, 0., 0.],
                             angular_velocity_body=[0., 0., .4])], [99., .1], [99., .007])
    report = measurements.report()["by_case"]
    early, late = report["early"], report["late"]
    assert ledger.attempts == [1, 1]
    assert early["accepted_frames"] == early["valid_frames"] == 1
    assert early["root_link_height_mae_m"] == pytest.approx(.2)
    assert early["root_link_height_mean_m"] == pytest.approx(.5)
    assert early["root_link_height_signed_error_mean_m"] == pytest.approx(.2)
    assert early["root_com_body_vx_mean_m_s"] == pytest.approx(.4)
    assert early["root_com_body_vx_signed_error_mean_m_s"] == pytest.approx(-.6)
    assert early["body_omega_z_mean_rad_s"] == pytest.approx(.4)
    assert early["body_omega_z_signed_error_mean_rad_s"] == pytest.approx(.4)
    assert early["closure_gap_max_m"] == pytest.approx(.004)
    assert late["accepted_frames"] == 3 and late["valid_frames"] == 2
    assert late["invalid_measurement_frames"] == 1
    assert late["valid_lane_seconds"] == pytest.approx(.04)
    assert late["root_link_height_mae_m"] == pytest.approx(.15)
    assert late["root_link_height_rmse_m"] == pytest.approx(math.sqrt(.025))
    assert late["root_com_body_vx_tracking_mae_m_s"] == pytest.approx(.3)
    assert late["body_omega_z_tracking_mae_rad_s"] == pytest.approx(.3)
    assert late["root_link_height_mean_m"] == pytest.approx(.25)
    assert late["root_link_height_signed_error_mean_m"] == pytest.approx(-.05)
    assert late["root_com_body_vx_mean_m_s"] == pytest.approx(.9)
    assert late["root_com_body_vx_signed_error_mean_m_s"] == pytest.approx(-.1)
    assert late["body_omega_z_mean_rad_s"] == pytest.approx(.1)
    assert late["body_omega_z_signed_error_mean_rad_s"] == pytest.approx(.1)
    assert late["closure_gap_max_m"] == pytest.approx(.007)

    with pytest.raises(ValueError, match="Formal evaluation tracking diagnostics missing"):
        measurements.prepare({}, [0])
    legacy = _CaseTrackingMeasurements(["case"], .02, required=False)
    legacy.accept(legacy.prepare({}, [0]))
    unavailable = legacy.report()["by_case"]["case"]
    assert unavailable["status"] == "not_collected"
    assert unavailable["unavailable_diagnostic_frames"] == 1
    assert unavailable["root_link_height_mae_m"] is unavailable["closure_gap_max_m"] is None
    assert unavailable["root_link_height_mean_m"] is unavailable["body_omega_z_signed_error_mean_rad_s"] is None


@pytest.mark.parametrize('command,kind', [([1., 0., .3], 'pure_translation'),
    ([0., 2., .3], 'pure_spin'), ([0., 0., .3], 'standing')])
def test_paid_attitude_and_lateral_velocity_include_terminal_but_exclude_invalid(command, kind):
    ledger = FlatEpisodeLedger(fixed(command), ['case'], list(WEIGHTS))
    begin(ledger)
    roll, pitch = .3, -.2
    gravity = [math.sin(pitch), -math.sin(roll) * math.cos(pitch),
               -math.cos(roll) * math.cos(pitch)]
    ledger.observe(0, frame(1, effective_command=command, base_request=command,
        projected_gravity_body=gravity, root_com_velocity_body=[command[0], .2, 0.]))
    ledger.observe(0, frame(2, effective_command=command, base_request=command,
        measurement_valid=False, projected_gravity_body=[math.nan] * 3,
        root_com_velocity_body=[math.nan] * 3))
    result = ledger.observe(0, terminal(3, effective_command=command, base_request=command,
        learning_mask=False, projected_gravity_body=[v * 2 for v in gravity],
        root_com_velocity_body=[command[0], -.4, 0.]))
    for measurement in (result['measurements'][kind], result['measurements']['all']):
        assert measurement['root_com_body_vy_signed_mean_m_s'] == pytest.approx(-.1)
        assert measurement['root_com_body_vy_rms_m_s'] == pytest.approx(math.sqrt(.1))
        assert measurement['body_roll_rms_rad'] == pytest.approx(.3)
        assert measurement['max_abs_body_roll_rad'] == pytest.approx(.3)
        assert measurement['body_pitch_rms_rad'] == pytest.approx(.2)
        assert measurement['max_abs_body_pitch_rad'] == pytest.approx(.2)
        tilt = math.acos(math.cos(roll) * math.cos(pitch))
        assert measurement['body_tilt_rms_rad'] == pytest.approx(tilt)
        assert measurement['max_body_tilt_rad'] == pytest.approx(tilt)
        assert measurement['attitude_measurement']['status'] == 'collected'
        assert measurement['attitude_measurement']['valid_frames'] == 2


@pytest.mark.parametrize('gravity,status', [(None, 'not_recorded'),
    ([0., 0., 0.], 'invalid_gravity'), ([1., 0., 0.], 'partial')])
def test_absent_or_undefined_attitude_is_explicit_not_fabricated_upright(gravity, status):
    ledger = FlatEpisodeLedger(fixed(), ['case'], list(WEIGHTS))
    begin(ledger)
    extra = {} if gravity is None else {'projected_gravity_body': gravity}
    result = ledger.observe(0, terminal(**extra))['measurements']['pure_translation']
    assert result['attitude_measurement']['status'] == status
    assert result['body_roll_rms_rad'] is None
    if status == 'partial':
        assert result['body_pitch_rms_rad'] == pytest.approx(math.pi / 2)
        assert result['body_tilt_rms_rad'] == pytest.approx(math.pi / 2)
        assert result['attitude_measurement']['roll_undefined_frames'] == 1
    else:
        assert result['body_pitch_rms_rad'] is result['body_tilt_rms_rad'] is None


def test_adapter_forwards_the_paid_gravity_instead_of_returned_reset_pose():
    adapter = PreResetLedgerAdapter(fixed(), ['case'])
    adapter.begin_initial({'episode_id': torch.tensor([0]), 'episode_tick': torch.tensor([0])},
        root_position_world=torch.zeros(1, 3), yaw_rad=torch.zeros(1),
        reset_metadata=[{'fixture': 'initial upright'}])
    row = terminal(projected_gravity_body=[0., -.6, -.8])
    rename = {'measurement_valid': 'measurement_valid_mask', 'learning_mask': 'policy_learning_mask',
        'bootstrap_mask': 'policy_bootstrap_mask', 'root_com_velocity_body': 'root_COM_velocity_body',
        'yaw_rad': 'root_yaw_rad', 'projected_gravity_body': 'gravity'}
    diagnostics = {rename.get(key, key): torch.tensor([value]) for key, value in row.items()
                   if key not in ('payments', 'termination_reasons')}
    diagnostics['payments'] = {key: torch.tensor([value]) for key, value in row['payments'].items()}
    diagnostics['reasons'] = {'source_attitude_or_contact': torch.tensor([True])}
    extras = {'diagnostics': diagnostics, 'returned_episode_state': {
        'episode_id': torch.tensor([1]), 'episode_tick': torch.tensor([0]), 'reset_mask': torch.tensor([True])}}
    record, = adapter.observe_paid(extras, diagnostics['reward'], diagnostics['done'],
        returned_root_position_world=torch.zeros(1, 3), returned_yaw_rad=torch.zeros(1),
        reset_metadata=[{'fixture': 'newborn upright must not overwrite terminal posture'}])
    assert record['measurements']['pure_translation']['body_roll_rms_rad'] == pytest.approx(math.atan2(.6, .8))
