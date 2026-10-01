"""Timing attribution must not double-count nested checkpoint/diagnostic work."""
import pytest

from wheeled_tasks.chassis.training_timing import TrainingTiming


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_nested_work_is_exclusive_and_iteration_totals_reset():
    clock = Clock()
    timer = TrainingTiming(clock=clock)
    clock.advance(2.)
    with timer.measure("env_step"):
        clock.advance(40.)
    with timer.measure("optimizer"):
        clock.advance(.3)
        with timer.measure("checkpoint"):
            clock.advance(.1)
        clock.advance(.2)
    result = timer.finish_iteration()
    assert result["Timing/iteration_wall_seconds"] == pytest.approx(42.6)
    assert result["Timing/env_step_seconds"] == 40.
    assert result["Timing/optimizer_seconds"] == pytest.approx(.5)
    assert result["Timing/checkpoint_seconds"] == pytest.approx(.1)
    assert result["Timing/unattributed_wall_seconds"] == pytest.approx(2.)
    assert result["Timing/checkpoint_calls"] == 1
    assert result["Timing/logger_calls"] == 0
    clock.advance(1.)
    second = timer.finish_iteration()
    assert second["Timing/iteration_wall_seconds"] == 1.
    assert second["Timing/env_step_seconds"] == 0.
    assert second["Timing/checkpoint_calls"] == 0
    summary = timer.summary()
    assert summary["completed_iterations"] == 2
    assert summary["phase_seconds"]["env_step"] == 40.
    assert summary["phase_calls"]["checkpoint"] == 1
    assert summary["unattributed_wall_seconds"] == pytest.approx(3.)


def test_exception_records_work_and_does_not_hide_failure():
    clock = Clock()
    timer = TrainingTiming(clock=clock)
    with pytest.raises(ValueError, match="original failure"):
        with timer.measure("optimizer"):
            clock.advance(1.)
            with timer.measure("diagnostics"):
                clock.advance(2.)
                raise ValueError("original failure")
    result = timer.finish_iteration()
    assert result["Timing/optimizer_seconds"] == 1.
    assert result["Timing/diagnostics_seconds"] == 2.
    assert result["Timing/unattributed_wall_seconds"] == 0.


def test_nested_identical_names_and_invalid_boundary():
    clock = Clock()
    timer = TrainingTiming(clock=clock)
    with timer.measure("custom"):
        clock.advance(1.)
        with timer.measure("custom"):
            clock.advance(2.)
            with pytest.raises(RuntimeError, match="outside timing scopes"):
                timer.finish_iteration()
        clock.advance(3.)
    result = timer.finish_iteration()
    assert result["Timing/custom_seconds"] == 6.
    assert result["Timing/custom_calls"] == 2
    assert result["Timing/unattributed_wall_seconds"] == 0.
