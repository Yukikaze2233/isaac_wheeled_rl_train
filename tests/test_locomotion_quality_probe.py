"""The probe must separate braking/stop timing and avoid episode-origin drift."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("quality_probe", Path(__file__).resolve().parents[1] / "scripts/probe_locomotion_quality.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("speed", [.5, -.5, 1., -1., 3., -3.])
def test_braking_uses_zero_request_and_full_physical_slope(speed):
    plan = probe.trajectory(speed)
    assert plan[1][1] == plan[3][1]
    assert plan[3] == ("braking", plan[1][1], 0.)
    assert plan[4] == ("stop_warmup", 100, 0.)
    assert plan[5] == ("stop", 500, 0.)
    assert plan[1][1] * .02 >= abs(speed) / 1.5 - 1e-9


def test_stopping_displacement_anchors_after_motion():
    frames = []
    for index in range(100):
        frames.append(dict(phase="stop", time_s=5. + index * .005,
            position_m=[20. + .01 * index / 99, 0., .305], phase_anchor_xy_m=[20., 0.],
            velocity_m_s=[.02, 0., 0.], omega_rad_s=[0., 0., 0.], command=[0., 0., .305],
            roll_deg=0., pitch_deg=0., tilt_deg=0., wheel_saturated=[False, False], height_m=.305))
    result = probe.summarize_frames(frames)["stop"]
    assert result["peak_displacement_from_phase_anchor_m"] == pytest.approx(.01)
    assert result["final_half_second_all_below_003_m_s"] is True
    assert result["samples_200hz"] == 100


def test_native_terminal_fields_and_success_are_preserved():
    diagnostic = dict(reasons={"fall": [False, True], "boundary": [False, False]}, success=[True, False])
    assert probe.terminal_reasons(diagnostic, 0) == {"task_success": True}
    assert probe.terminal_reasons(diagnostic, 1) == {"fall": True}
