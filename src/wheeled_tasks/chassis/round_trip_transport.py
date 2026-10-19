"""One shared V7 round-trip budget and interval-averaged held-command delay.

The equal uplink/downlink split is an explicit engineering allocation, not a
measurement of either direction. Downlink delay already includes USB/CAN;
there is no additional bus phase or invented 1 kHz feedback. Configuration
helpers can be imported without Torch or the simulation runtime.
"""
from copy import deepcopy
import math


ROUND_TRIP_VERSION = "v7_round_trip_v1"
COMMAND_VERSION = "v7_effective_command_delay_v1"


def round_trip_transport_settings():
    """Return the approved total-delay prior; all budget times are seconds."""
    return dict(version=ROUND_TRIP_VERSION, enabled=True, units="seconds",
        total_delay_range_s=[.0001, .005], uplink_fraction=.5,
        distribution="uniform", resample="episode_reset", enabled_fraction=1.)


def _canonical(settings, expected, name):
    if not isinstance(settings, dict) or set(settings) != set(expected):
        raise ValueError(f"{name} requires complete canonical settings")
    for key, wanted in expected.items():
        actual = settings[key]
        values = actual if isinstance(wanted, list) else [actual]
        targets = wanted if isinstance(wanted, list) else [wanted]
        if isinstance(wanted, list) and (not isinstance(actual, list) or len(actual) != len(wanted)):
            raise ValueError(f"{name}.{key} differs from the canonical prior")
        for value, target in zip(values, targets):
            if type(target) in (int, float):
                if type(value) not in (int, float):
                    raise ValueError(f"{name}.{key} requires finite numbers")
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError(f"{name}.{key} requires finite numbers")
            elif type(value) is not type(target):
                raise ValueError(f"{name}.{key} has an invalid type")
            if value != target:
                raise ValueError(f"{name}.{key} differs from the canonical prior")
    return deepcopy(settings)


def validate_round_trip_transport(settings):
    """Reject disabled, partial, independently sampled or differently split priors."""
    return _canonical(settings, round_trip_transport_settings(), "round_trip_transport")


def command_settings(round_trip_config=None):
    """Derive the complete downlink bound from the single round-trip budget."""
    cfg = validate_round_trip_transport(round_trip_transport_settings()
        if round_trip_config is None else round_trip_config)
    return dict(version=COMMAND_VERSION, enabled=True,
        delay_ms=[value * (1 - cfg["uplink_fraction"]) * 1000 for value in cfg["total_delay_range_s"]],
        scope="complete_host_to_drive_effective_delay_including_CAN",
        delivery_model="continuous_interval_impulse_v1")


def feedback_settings(round_trip_config=None):
    """Derive a valid feedback-history config without a second delay draw.

    The existing history schema is retained. Runtime consumers must borrow
    RoundTripDelayBudget.uplink_s; zero jitter preserves the shared total.
    """
    cfg = validate_round_trip_transport(round_trip_transport_settings()
        if round_trip_config is None else round_trip_config)
    return dict(version="complete_one_way_feedback_v1", enabled=True, units="seconds",
        scope="engineering_prior_complete_sensor_to_host_path_not_measured_rtt",
        ordered_delivery="monotonic_sensor_timestamp", reconstruction="linear_physics_history",
        delay_range_s=[value * cfg["uplink_fraction"] for value in cfg["total_delay_range_s"]],
        enabled_fraction=1., jitter_s=0., jitter_correlation=1.)


class RoundTripDelayBudget:
    """Draw once per selected episode; transport components only borrow arrays.

    Construction does not consume RNG. The environment must call reset(ids)
    before resetting feedback and command components for those same episodes.
    """

    SEED_XOR = 0x525454

    def __init__(self, count, device, config, seed):
        import torch

        self.cfg = validate_round_trip_transport(config)
        self.device = device
        self.rows = torch.arange(count, device=device)
        self.low, self.high = self.cfg["total_delay_range_s"]
        self.uplink_fraction = self.cfg["uplink_fraction"]
        self.generator = torch.Generator(device=device)
        self.reseed(seed)
        self.total_s = torch.zeros(count, device=device, dtype=torch.float64)
        self.uplink_s = torch.zeros_like(self.total_s)
        self.downlink_s = torch.zeros_like(self.total_s)
        self.enabled = torch.ones(count, device=device, dtype=torch.bool)

    def reseed(self, seed):
        """Restart only this budget's random stream."""
        self.generator.manual_seed(seed ^ self.SEED_XOR)

    def reset(self, ids):
        import torch

        if not len(ids):
            return
        unit = torch.rand(len(ids), device=self.device, dtype=torch.float64, generator=self.generator)
        total = self.low + (self.high - self.low) * unit
        self.total_s[ids] = total
        self.uplink_s[ids] = total * self.uplink_fraction
        self.downlink_s[ids] = total - self.uplink_s[ids]

    def metrics(self):
        return {
            "/transport/round_trip_enabled_fraction": self.enabled.float().mean(),
            "/transport/round_trip_delay_mean_s": self.total_s.mean(),
            "/transport/round_trip_delay_min_s": self.total_s.min(),
            "/transport/round_trip_delay_max_s": self.total_s.max(),
            "/transport/allocated_uplink_delay_mean_s": self.uplink_s.mean(),
            "/transport/allocated_downlink_delay_mean_s": self.downlink_s.mean(),
        }


class EffectiveCommandDelay:
    """Exact delayed-ZOH interval impulse for D <= the 5 ms PD interval.

    This is an average applied torque, not a finer simulation of the resulting
    nonlinear robot motion. The old command acts for D seconds and the current
    command for dt-D. Original PD/policy/feedback timestamps receive the same
    duration weights, independently of command magnitude or cached submission.
    """

    def __init__(self, count, device, dt, config, budget):
        import torch

        expected = command_settings(budget.cfg)
        if isinstance(config, dict) and config.get("version") == ROUND_TRIP_VERSION:
            config = command_settings(config)
        self.cfg = _canonical(config, expected, "effective_command_delay")
        if type(dt) not in (int, float) or not math.isfinite(dt) or dt != .005:
            raise ValueError("Effective command delay requires the 200 Hz / 5 ms PD clock")
        requested_device = torch.device(device)
        budget_device = budget.downlink_s.device
        if (budget.downlink_s.shape != (count,) or requested_device.type != budget_device.type
                or (requested_device.index is not None and requested_device.index != budget_device.index)):
            raise ValueError("Command delay requires a shared budget on the same environments and device")
        if expected["delay_ms"][1] * .001 > dt:
            raise ValueError("Effective command delay cannot exceed one PD interval")
        self.dt, self.transport_dt, self.tick = float(dt), float(dt), 0
        self.substeps, self.phase_aware = 1, False
        self.delivery_model = self.cfg["delivery_model"]
        self.device, self.budget = budget_device, budget
        self.rows = torch.arange(count, device=device)
        self.enabled = budget.enabled
        self.delay = self.base_delay = budget.downlink_s
        self.low, self.high = [value * .001 for value in self.cfg["delay_ms"]]
        self.last_command = torch.zeros(count, 6, device=device, dtype=torch.float32)
        self.held = self.last_command
        self.last_origins_s = torch.zeros(count, 3, device=device, dtype=torch.float64)
        self.held_origins = self.last_origins_s
        self.applied_origins_s = torch.zeros_like(self.last_origins_s)
        self.last_target_sample_s = torch.zeros(count, 2, device=device, dtype=torch.float64)
        self.last_target_policy_generated_s = torch.zeros_like(self.last_target_sample_s)
        self.applied_target_sample_s = torch.zeros_like(self.last_target_sample_s)
        self.applied_target_policy_generated_s = torch.zeros_like(self.last_target_sample_s)
        self.applied_at_s = torch.zeros(count, device=device, dtype=torch.float64)
        self.pd_age_s = torch.zeros_like(self.applied_at_s)
        self.age = self.pd_age_s
        self.sent = torch.zeros_like(self.applied_at_s)
        self.reset(self.rows)

    def reset(self, ids):
        """Clear selected episode history, without changing time or the budget."""
        now = self.tick * self.dt
        self.last_command[ids] = 0.
        self.last_origins_s[ids] = now
        self.applied_origins_s[ids] = now
        self.last_target_sample_s[ids] = now
        self.last_target_policy_generated_s[ids] = now
        self.applied_target_sample_s[ids] = now
        self.applied_target_policy_generated_s[ids] = now
        self.applied_at_s[ids] = now
        self.pd_age_s[ids] = 0.
        self.sent[ids] = 0.

    def apply_torque(self, command, *, pd_generated_s=None, policy_generated_s=None,
                     feedback_sample_s=None, actuator_response=None, effort_bounds=None,
                     target_sample_s=None, target_policy_generated_s=None):
        import torch

        if (command.shape != self.last_command.shape or command.device != self.last_command.device
                or command.dtype != self.last_command.dtype):
            raise ValueError("Command delay requires a float32 [environment, 6] torque tensor")
        now = self.tick * self.dt
        columns = []
        for value in (pd_generated_s, policy_generated_s, feedback_sample_s):
            column = torch.as_tensor(now if value is None else value, device=self.device, dtype=torch.float64)
            if column.ndim == 0:
                column = column.expand(len(self.rows))
            if column.shape != (len(self.rows),):
                raise ValueError("Command event timestamps must have one value per environment")
            columns.append(column)
        origins = torch.stack(columns, -1)
        target_clocks = []
        for value, fallback in ((target_sample_s, origins[:, :1]),
                                (target_policy_generated_s, origins[:, 1:2])):
            column = fallback.expand(-1, 2) if value is None else value
            if (not isinstance(column, torch.Tensor) or column.shape != (len(self.rows), 2)
                    or column.device != self.last_command.device or column.dtype != torch.float64):
                raise ValueError("Target origin metadata requires float64 [environment,2] leg/wheel timestamps")
            target_clocks.append(column)
        old_fraction = (self.delay / self.dt)[:, None]
        if actuator_response is None:
            applied = command + old_fraction.to(command.dtype) * (self.last_command - command)
        else:
            # Integrate before replacing the old held command. The actuator
            # endpoint state is separate from this interval's native mean effort.
            applied = actuator_response.apply_delayed_torque(
                self.last_command, command, self.delay, effort_bounds=effort_bounds)
        # These origins describe transported PD requests, before motor dynamics.
        # Difference form avoids unnecessary rounding of large absolute times.
        self.applied_origins_s.copy_(origins + old_fraction * (self.last_origins_s - origins))
        for current, previous, applied_clock in (
                (target_clocks[0], self.last_target_sample_s, self.applied_target_sample_s),
                (target_clocks[1], self.last_target_policy_generated_s, self.applied_target_policy_generated_s)):
            applied_clock.copy_(current + old_fraction * (previous - current))
            previous.copy_(current)
        self.last_command.copy_(command)
        self.last_origins_s.copy_(origins)
        self.tick += 1
        self.applied_at_s.fill_(self.tick * self.dt)
        self.pd_age_s.copy_(self.applied_at_s - self.applied_origins_s[:, 0])
        self.sent += 1
        return applied

    def metrics(self):
        return {
            "/transport/downlink_enabled_fraction": self.enabled.float().mean(),
            "/transport/effective_downlink_delay_mean_s": self.delay.mean(),
            "/transport/effective_downlink_delay_max_s": self.delay.max(),
            "/transport/executed_command_age_ms": (self.pd_age_s * 1000).mean(),
            "/transport/executed_command_age_max_ms": (self.pd_age_s * 1000).max(),
            "/transport/applied_pd_computation_age_ms": (self.pd_age_s * 1000).mean(),
            "/transport/command_loss_fraction": self.delay.new_zeros(()),
        }
