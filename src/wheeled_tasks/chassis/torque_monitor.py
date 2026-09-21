"""Physics-rate applied-effort monitoring; motor Nm and spring N remain separate."""
import torch


class TorqueMonitor:
    def __init__(self, groups, active_names, device, wheel_limit):
        self.names = sorted(set(groups))
        self.active_names = list(active_names)
        self.group_ids = torch.tensor([self.names.index(g) for g in groups], device=device)
        self.group_count = torch.bincount(self.group_ids, minlength=len(self.names))
        self.caps = torch.tensor([40., 40., wheel_limit, 40., 40., wheel_limit], device=device)
        # Float32 counters/sums lose individual samples after long physics-rate runs.
        # Keep exact event counts and accumulate small mechanical moments in float64.
        self.samples = torch.zeros(len(self.names), device=device, dtype=torch.int64)
        # Accumulate per environment at physics rate. Reducing thousands of
        # environments into a few groups every step serializes CUDA atomics.
        # Group reduction is needed only when a report is requested.
        self.square = torch.zeros(len(groups), 6, device=device, dtype=torch.float64)
        self.saturated = torch.zeros(len(groups), 6, device=device, dtype=torch.int64)
        self.peak = torch.zeros(len(groups), 6, device=device)
        self.speed = torch.zeros_like(self.peak)
        self.positive_power = torch.zeros_like(self.square)
        self.negative_power = torch.zeros_like(self.square)
        self.gas_peak = torch.zeros(len(groups), 2, device=device)
        self.gas_speed = torch.zeros_like(self.gas_peak)
        self.compression_min = torch.full_like(self.gas_peak, torch.inf)
        self.compression_max = torch.full_like(self.gas_peak, -torch.inf)

    def observe(self, applied_motor, velocity, gas_force, gas_velocity, compression, requested_motor, current_bounds):
        self.samples += self.group_count
        self.square += applied_motor.double().square()
        saturation = (requested_motor.abs() >= .95 * current_bounds.clamp_min(1e-6)).long()
        self.saturated += saturation
        torch.maximum(self.peak, applied_motor.abs(), out=self.peak)
        torch.maximum(self.speed, velocity.abs(), out=self.speed)
        power = applied_motor.double() * velocity.double()
        self.positive_power += power.clamp_min(0)
        self.negative_power += (-power).clamp_min(0)
        torch.maximum(self.gas_peak, gas_force.abs(), out=self.gas_peak)
        torch.maximum(self.gas_speed, gas_velocity.abs(), out=self.gas_speed)
        torch.minimum(self.compression_min, compression, out=self.compression_min)
        torch.maximum(self.compression_max, compression, out=self.compression_max)

    def _group_reduce(self, values, reduction="sum"):
        result = values.new_zeros((len(self.names), values.shape[1]))
        if reduction == "sum":
            return result.index_add_(0, self.group_ids, values)
        index = self.group_ids[:, None].expand_as(values)
        return result.scatter_reduce_(0, index, values, reduce=reduction, include_self=False)

    def report(self):
        denominator = self.samples.clamp_min(1)[:, None]
        grouped = {
            "rms_motor_torque_nm": (self._group_reduce(self.square) / denominator).sqrt().cpu().tolist(),
            "peak_motor_torque_nm": self._group_reduce(self.peak, "amax").cpu().tolist(),
            "saturation_fraction_95pct": (self._group_reduce(self.saturated).double() / denominator).cpu().tolist(),
            "peak_motor_speed_rad_s": self._group_reduce(self.speed, "amax").cpu().tolist(),
            "mean_positive_mechanical_power_w": (self._group_reduce(self.positive_power) / denominator).cpu().tolist(),
            "mean_negative_mechanical_power_w": (self._group_reduce(self.negative_power) / denominator).cpu().tolist(),
            "peak_gas_force_n": self._group_reduce(self.gas_peak, "amax").cpu().tolist(),
            "peak_gas_speed_m_s": self._group_reduce(self.gas_speed, "amax").cpu().tolist(),
            "gas_compression_min_m": self._group_reduce(self.compression_min, "amin").cpu().tolist(),
            "gas_compression_max_m": self._group_reduce(self.compression_max, "amax").cpu().tolist(),
        }
        samples = self.samples.cpu().tolist()
        result = {"source": "explicit_actuator_applied_torque_after_clipping_each_physics_step",
                  "accumulator_version": 3, "counter_dtype": "int64", "moment_dtype": "float64",
                  "group_reduction": "on_report_after_per_environment_accumulation",
                  "active_joint_order": self.active_names, "simulation_effort_caps_nm": self.caps.cpu().tolist(),
                  "hardware_continuous_ratings_verified": False,
                  "saturation_reference": "preclip_motor_request_vs_instantaneous_torque_speed_bound", "groups": {}}
        for i, name in enumerate(self.names):
            result["groups"][name] = {"physics_samples": samples[i], **{key: value[i] for key, value in grouped.items()}}
            if not samples[i]:
                result["groups"][name].update(gas_compression_min_m=None, gas_compression_max_m=None)
        return result
