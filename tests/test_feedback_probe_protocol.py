"""A missing fixed observation window is never credited as zero error."""
import json

import torch

from diagnose_feedback_link import observed_metrics
from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics


def test_empty_native_window_keeps_denominator_and_nulls_behavior():
    metrics = EpisodeMetrics(["test", "test"], "cpu", .02)
    group = observed_metrics(metrics)["groups"]["test"]
    assert group["frames"] == group["observed_frames"] == group["failures"] == 0
    assert group["behavior_window_observed"] is False
    assert group["vx_mae_m_s"] is group["yaw_mae_rad_s"] is None
    assert group["planar_speed_mean_m_s"] is group["rms_motor_torque_nm"] is None
    json.dumps(group, allow_nan=False)


def test_early_native_failure_is_counted_without_inventing_post_warmup_precision():
    metrics = EpisodeMetrics(["test"], "cpu", .02, warmup_seconds=2.)
    vector = torch.zeros(1, 3)
    scalar = torch.zeros(1)
    true, false = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    metrics.observe(dict(episode_ticks=torch.ones(1, dtype=torch.long), position=vector,
        commands=vector, velocity=vector, omega=vector, height=scalar, reward=scalar,
        motor_effort=torch.zeros(1, 6), gravity=torch.tensor([[0., 0., -1.]]), gap=scalar,
        done=true, terminated=true, success=false, reasons={"fall": true, "boundary": false}))
    group = observed_metrics(metrics)["groups"]["test"]
    assert group["frames"] == 0 and group["observed_frames"] == 1
    assert group["failures"] == group["episodes_ending_before_or_at_warmup"] == 1
    assert group["termination_reasons"]["fall"] == 1
    assert group["height_mae_m"] is group["tilt_max_deg"] is None
    assert group["terminal_episode_time_mean_s"] == .02
