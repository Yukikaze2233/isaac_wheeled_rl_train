"""A missing fixed observation window is never credited as zero error."""
import json

import pytest
import torch

from diagnose_feedback_link import (
    fixed_feedback_link, fixed_profiles, isolate_feedback_link, observed_metrics)
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


class NativeProbeEnvironment(ProbeEnvironment):
    def __init__(self):
        super().__init__()
        from wheeled_tasks.chassis.command_transport import UsbCommandTransport
        from wheeled_tasks.chassis.feedback_transport import FractionalFeedbackTransport

        self.dt, self.time = .005, 0.
        draw = torch.zeros(1)
        self.feedback_transport = FractionalFeedbackTransport(1, "cpu", self.dt,
            self.cfg["feedback_transport"], 17, domain_draw=draw)
        self.command_transport = UsbCommandTransport(1, "cpu", self.dt,
            self.cfg["command_transport"], 17, domain_draw=draw)
        self.feedback_transport.reset(self.feedback_transport.rows, self._transport_sensors())

    def _transport_sensors(self):
        sensors = torch.zeros(1, 18)
        sensors[:, 0] = self.time
        sensors[:, 17] = -1.
        return sensors

    def advance(self, torque):
        result = self.command_transport.apply_torque(torque)
        self.time += self.dt
        self.feedback_transport.observe(self._transport_sensors())
        return result


@pytest.mark.parametrize("feedback,usb,event,outputs", [
    (.04, .04, .5, [.9, 1., 1.]), (2.5, 2.5, 2.5, [.5, 1., 1.]),
    (7.5, 7.5, 7.5, [0., .5, 1.]), (15., 14., 14.5, [0., 0., .1])])
def test_fixed_native_profiles_match_CAN_impulse_and_feedback_time(feedback, usb, event, outputs):
    env = NativeProbeEnvironment()
    original = env.feedback_transport, env.command_transport
    rng = torch.get_rng_state().clone()
    with fixed_feedback_link(env, feedback, usb, .5, 17) as receipt:
        assert receipt["profile"]["expected_can_event_delay_ms"] == event
        assert receipt["canonical_preflight_covers_profile_override"] is False
        assert env.feedback_transport is not original[0]
        assert env.command_transport is not original[1]
        for expected in outputs:
            torch.testing.assert_close(env.advance(torch.ones(1, 6)),
                torch.full((1, 6), expected))
        assert env.feedback_transport.age_s.item() == pytest.approx(feedback * .001, abs=1e-12)
        assert env.command_transport.tick == 15
        assert env.feedback_transport.physics_steps == 3
        assert env.command_transport.lost.eq(0).all()
        assert torch.equal(torch.get_rng_state(), rng)
    assert (env.feedback_transport, env.command_transport) == original
    assert receipt["hooks_restored"] is True


@pytest.mark.parametrize("cold", [False, True])
def test_warm_handoff_keeps_shadow_clocks_and_only_cold_condition_clears_queue(cold):
    env = NativeProbeEnvironment()
    with fixed_feedback_link(env, 7.5, 7.5, .5, 17):
        with isolate_feedback_link(env, "full", activate_at_s=2.,
                cold_queue_at_handoff=cold) as receipt:
            for tick in range(100):
                env._feedback_link_probe_time_s = tick * .02
                assert env.get_observations() is None
                for _ in range(4):
                    assert env.advance(torch.ones(1, 6)).eq(1).all()
            assert env.feedback_transport.physics_steps == 400
            assert env.command_transport.tick == 2000
            # The observation at warm step end is still before the explicit
            # next-policy handoff, even though the physics clock reached 2 s.
            assert env.get_observations() is None
            assert receipt["handoff"] is None
            before = env._transport_sensors().clone()
            cached = env.feedback_transport.feedback.clone()
            env._feedback_link_probe_time_s = 2.
            assert env.get_observations() is env.feedback_transport
            assert torch.equal(env._transport_sensors(), before)
            assert env.feedback_transport.physics_steps == 400
            assert env.command_transport.tick == 2000
            assert env.feedback_transport.age_s.item() == pytest.approx(0. if cold else .0075)
            assert torch.equal(env.feedback_transport.feedback, before if cold else cached)
            output = env.advance(torch.ones(1, 6))
            assert output.eq(0. if cold else 1.).all()
            assert receipt["native_command_calls"] == 401
            assert receipt["command_bypasses"] == 400
            assert receipt["handoff"]["command_clock_s"] == pytest.approx(2.)
        assert not hasattr(env, "_feedback_link_probe_time_s")
        assert receipt["hooks_restored"] is True


def test_cold_queue_rejects_prior_redraw_as_a_queue_only_comparison():
    env = NativeProbeEnvironment()
    with pytest.raises(ValueError, match="fixed timing"):
        with isolate_feedback_link(env, "full", activate_at_s=2., cold_queue_at_handoff=True):
            pytest.fail("Cold queue with a redrawn random prior was admitted")


def test_fixed_profiles_reject_partial_out_of_prior_or_scored_overrides_and_restore_exception():
    for values in ((None, 2.5, .5), (2.5, None, .5), (.01, .04, .5), (15.1, 14., .5), (15., 14.1, .5)):
        with pytest.raises(ValueError, match="engineering prior"):
            fixed_profiles(*values)
    env = NativeProbeEnvironment()
    originals = env.feedback_transport, env.command_transport
    with pytest.raises(RuntimeError, match="probe failure"):
        with fixed_feedback_link(env, 2.5, 2.5, .5, 17) as receipt:
            with pytest.raises(RuntimeError, match="cannot nest"):
                with fixed_feedback_link(env, 2.5, 2.5, .5, 17):
                    pytest.fail("Nested profile admitted")
            raise RuntimeError("probe failure")
    assert receipt["hooks_restored"] is True
    assert (env.feedback_transport, env.command_transport) == originals
    env.cfg["run_scope"] = "train"
    with pytest.raises(ValueError, match="unscored"):
        with fixed_feedback_link(env, 2.5, 2.5, .5, 17):
            pytest.fail("Production profile override admitted")


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
