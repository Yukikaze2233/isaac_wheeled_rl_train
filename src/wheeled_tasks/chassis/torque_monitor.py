"""Physics-rate applied-effort monitoring; motor Nm and spring N remain separate."""
import torch


class TorqueMonitor:
    def __init__(self, groups, active_names, device, wheel_limit):
        self.names = sorted(set(groups))
        self.active_names = list(active_names)
        self.group_ids = torch.tensor([self.names.index(g) for g in groups], device=device)
        self.group_count = torch.bincount(self.group_ids, minlength=len(self.names))
        self.motor_index = self.group_ids[:, None].expand(-1, 6)
        self.spring_index = self.group_ids[:, None].expand(-1, 2)
        self.caps = torch.tensor([40., 40., wheel_limit, 40., 40., wheel_limit], device=device)
        self.samples = torch.zeros(len(self.names), device=device)
        self.square = torch.zeros(len(self.names), 6, device=device)
        self.saturated = torch.zeros_like(self.square)
        self.peak = torch.zeros_like(self.square)
        self.speed = torch.zeros_like(self.square)
        self.positive_power = torch.zeros_like(self.square)
        self.negative_power = torch.zeros_like(self.square)
        self.gas_peak = torch.zeros(len(self.names), 2, device=device)
        self.gas_speed = torch.zeros_like(self.gas_peak)
        self.compression_min = torch.full_like(self.gas_peak, torch.inf)
        self.compression_max = torch.full_like(self.gas_peak, -torch.inf)

    def observe(self, applied_motor, velocity, gas_force, gas_velocity, compression, requested_motor, current_bounds):
        self.samples += self.group_count
        self.square.index_add_(0, self.group_ids, applied_motor.square())
        saturation = (requested_motor.abs() >= .95 * current_bounds.clamp_min(1e-6)).float()
        self.saturated.index_add_(0, self.group_ids, saturation)
        self.peak.scatter_reduce_(0, self.motor_index, applied_motor.abs(), reduce="amax", include_self=True)
        self.speed.scatter_reduce_(0, self.motor_index, velocity.abs(), reduce="amax", include_self=True)
        power = applied_motor * velocity
        self.positive_power.index_add_(0, self.group_ids, power.clamp_min(0))
        self.negative_power.index_add_(0, self.group_ids, (-power).clamp_min(0))
        self.gas_peak.scatter_reduce_(0, self.spring_index, gas_force.abs(), reduce="amax", include_self=True)
        self.gas_speed.scatter_reduce_(0, self.spring_index, gas_velocity.abs(), reduce="amax", include_self=True)
        self.compression_min.scatter_reduce_(0, self.spring_index, compression, reduce="amin", include_self=True)
        self.compression_max.scatter_reduce_(0, self.spring_index, compression, reduce="amax", include_self=True)

    def report(self):
        denominator = self.samples.clamp_min(1)[:, None]
        result = {"source": "explicit_actuator_applied_torque_after_clipping_each_physics_step",
                  "active_joint_order": self.active_names, "simulation_effort_caps_nm": self.caps.cpu().tolist(),
                  "hardware_continuous_ratings_verified": False,
                  "saturation_reference": "preclip_motor_request_vs_instantaneous_torque_speed_bound", "groups": {}}
        for i, name in enumerate(self.names):
            result["groups"][name] = {
                "physics_samples": int(self.samples[i]),
                "rms_motor_torque_nm": torch.sqrt(self.square[i] / denominator[i]).cpu().tolist(),
                "peak_motor_torque_nm": self.peak[i].cpu().tolist(),
                "saturation_fraction_95pct": (self.saturated[i] / denominator[i]).cpu().tolist(),
                "peak_motor_speed_rad_s": self.speed[i].cpu().tolist(),
                "mean_positive_mechanical_power_w": (self.positive_power[i] / denominator[i]).cpu().tolist(),
                "mean_negative_mechanical_power_w": (self.negative_power[i] / denominator[i]).cpu().tolist(),
                "peak_gas_force_n": self.gas_peak[i].cpu().tolist(),
                "peak_gas_speed_m_s": self.gas_speed[i].cpu().tolist(),
                "gas_compression_min_m": self.compression_min[i].cpu().tolist() if self.samples[i] > 0 else None,
                "gas_compression_max_m": self.compression_max[i].cpu().tolist() if self.samples[i] > 0 else None,
            }
        return result
