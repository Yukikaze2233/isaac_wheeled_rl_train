"""A missing fixed observation window is never credited as zero error."""
import json

import pytest
import torch

from diagnose_feedback_link import isolate_feedback_link, observed_metrics
from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
from wheeled_tasks.chassis.pc_communication import apply_pc_communication


class ProbeCommand:
    def apply_torque(self, torque):
        return torque + 1


class ProbeEnvironment:
    def __init__(self):
        self.cfg = apply_pc_communication(dict(physics_dt=.005, policy_dt=.02,
            signal_delay=dict(version="physical_signal_delay_v1", enabled=False),
            usb_transport=dict(enabled=False), signal_perturbations=dict(max_delay_steps=0),
            evaluation=dict(cases=[dict(name="test")]), evaluation_exact_cases=True,
            run_scope="interactive_unscored", evaluation_transport_mode="training_domain_prior",
            auto_reset=False))
        self.feedback_transport = object()
        self.command_transport = ProbeCommand()
        self.usb_transport = self.signal_delay = None
        self.raise_on_observation = False

    @property
    def _pc_feedback_transport(self):
        return self.feedback_transport

    def get_observations(self):
        transport = self._pc_feedback_transport
        if self.raise_on_observation:
            raise RuntimeError("Native observation failure")
        return transport


@pytest.mark.parametrize("selection,policy,control,downlink", [
    ("actor-feedback", True, False, False),
    ("control-feedback", False, True, False),
    ("command-delay", False, False, True)])
def test_isolation_routes_native_consumers_and_preserves_other_instances(selection, policy, control, downlink):
    env, other = ProbeEnvironment(), ProbeEnvironment()
    original = ProbeEnvironment._pc_feedback_transport
    with isolate_feedback_link(env, selection) as receipt:
        assert env.command_transport is not None
        assert (env.get_observations() is env.feedback_transport) is policy
        assert (env._pc_feedback_transport is env.feedback_transport) is control
        assert env.command_transport.apply_torque(4) == (5 if downlink else 4)
        assert other.get_observations() is other.feedback_transport
        assert other.command_transport.apply_torque(4) == 5
        assert receipt["delayed_policy_reads"] == int(policy)
        assert receipt["delayed_control_reads"] == int(control)
        assert receipt["command_bypasses"] == int(not downlink)
        assert receipt["hooks_restored"] is False
    assert receipt["hooks_restored"] is True
    assert ProbeEnvironment._pc_feedback_transport is original
    assert env.get_observations() is env.feedback_transport
    assert env.command_transport.apply_torque(4) == 5
    assert "get_observations" not in env.__dict__
    assert "apply_torque" not in env.command_transport.__dict__


def test_isolation_restores_on_native_exception_and_observation_scope_does_not_leak():
    env = ProbeEnvironment()
    with pytest.raises(RuntimeError, match="Native observation failure"):
        with isolate_feedback_link(env, "actor-feedback") as receipt:
            env.raise_on_observation = True
            try:
                env.get_observations()
            except RuntimeError:
                assert env._pc_feedback_transport is None
                raise
    assert receipt["hooks_restored"] is True
    assert env._pc_feedback_transport is env.feedback_transport
    assert env.command_transport.apply_torque(4) == 5


def test_second_isolation_in_same_process_is_rejected_before_mutation():
    env, other = ProbeEnvironment(), ProbeEnvironment()
    with isolate_feedback_link(env, "control-feedback") as receipt:
        with pytest.raises(RuntimeError, match="cannot nest"):
            with isolate_feedback_link(other, "actor-feedback"):
                pytest.fail("Shared diagnostic owner was admitted")
        assert other.get_observations() is other.feedback_transport
        assert env.get_observations() is None
    assert receipt["hooks_restored"] is True


@pytest.mark.parametrize("field,value", [
    ("run_scope", "train"), ("auto_reset", True), ("qualified", True),
    ("evaluation_transport_mode", "fixed"), ("evaluation_exact_cases", False)])
def test_isolation_rejects_training_scored_and_resetting_environments(field, value):
    env = ProbeEnvironment()
    env.cfg[field] = value
    with pytest.raises(ValueError, match="unscored"):
        with isolate_feedback_link(env, "control-feedback"):
            pytest.fail("Invalid diagnostic scope admitted")
    assert env.get_observations() is env.feedback_transport


def test_full_probe_installs_no_hooks():
    env = ProbeEnvironment()
    original = ProbeEnvironment._pc_feedback_transport
    with isolate_feedback_link(env, "full") as receipt:
        assert env.get_observations() is env.feedback_transport
        assert env.command_transport.apply_torque(4) == 5
        assert receipt["hooks_installed"] is False
    assert receipt["hooks_restored"] is True
    assert ProbeEnvironment._pc_feedback_transport is original


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
