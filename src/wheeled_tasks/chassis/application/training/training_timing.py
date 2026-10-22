"""Exclusive host wall timings; never synchronize CUDA or affect learning state.

CUDA work is asynchronous, so these values identify CPU/driver blocking and
end-to-end phase cost, not kernel execution time. Use a profiler for GPU kernels.
Wrap an entire policy step rather than individual physics substeps by default.
"""
from contextlib import contextmanager
from time import perf_counter


class TrainingTiming:
    """Attribute nested scopes once and export per-iteration TensorBoard scalars.

    Call ``finish_iteration`` outside scopes at the same point each iteration.
    Timings outside measured scopes remain explicitly unattributed. No learning
    tensors, RNG, optimizer state, or checkpoint state enter this object.
    """

    DEFAULT_PHASES = (
        "env_step", "returns", "optimizer", "diagnostics", "checkpoint",
        "progress_io", "logger",
    )

    def __init__(self, *, clock=perf_counter):
        self._clock = clock
        self._started = self._iteration_started = clock()
        self._stack = []
        self._seconds = dict.fromkeys(self.DEFAULT_PHASES, 0.)
        self._calls = dict.fromkeys(self.DEFAULT_PHASES, 0)
        self._total_seconds = dict(self._seconds)
        self._total_calls = dict(self._calls)
        self._iterations = 0

    def _add_seconds(self, name, seconds):
        self._seconds[name] += seconds
        self._total_seconds[name] += seconds

    @contextmanager
    def measure(self, name):
        if name not in self._seconds:
            if not isinstance(name, str) or not name or "/" in name:
                raise ValueError("Timing phase must be a nonempty name without '/'")
            self._seconds[name] = self._total_seconds[name] = 0.
            self._calls[name] = self._total_calls[name] = 0
        started = self._clock()
        if self._stack:
            parent, previous = self._stack[-1]
            self._add_seconds(parent, started - previous)
        self._stack.append((name, started))
        try:
            yield
        finally:
            finished = self._clock()
            phase, previous = self._stack.pop()
            self._add_seconds(phase, finished - previous)
            self._calls[phase] += 1
            self._total_calls[phase] += 1
            if self._stack:
                self._stack[-1] = (self._stack[-1][0], finished)

    def finish_iteration(self):
        if self._stack:
            raise RuntimeError("Finish iteration outside timing scopes")
        finished = self._clock()
        wall = finished - self._iteration_started
        result = {
            "Timing/iteration_wall_seconds": wall,
            "Timing/unattributed_wall_seconds": max(0., wall - sum(self._seconds.values())),
        }
        for name in self._seconds:
            result[f"Timing/{name}_seconds"] = self._seconds[name]
            result[f"Timing/{name}_calls"] = self._calls[name]
            self._seconds[name] = 0.
            self._calls[name] = 0
        self._iterations += 1
        self._iteration_started = finished
        return result

    def summary(self):
        if self._stack:
            raise RuntimeError("Read summary outside timing scopes")
        wall = self._clock() - self._started
        return {
            "clock": "perf_counter_host_wall_exclusive_no_cuda_synchronize",
            "completed_iterations": self._iterations,
            "wall_seconds": wall,
            "unattributed_wall_seconds": max(0., wall - sum(self._total_seconds.values())),
            "phase_seconds": dict(self._total_seconds),
            "phase_calls": dict(self._total_calls),
        }
