"""Per-group behavioral diagnostics, with explicit episode outcome denominators."""
from __future__ import annotations

import torch


class EpisodeMetrics:
    """Accumulate policy-rate observations without losing long-run count precision."""

    ERROR_NAMES = ("vx_mae_m_s", "vx_mse_m2_s2", "yaw_mae_rad_s", "yaw_mse_rad2_s2",
                   "height_mae_m", "height_mse_m2", "reward_per_policy_step")

    def __init__(self, groups, device, policy_dt, warmup_seconds=0.):
        self.groups = list(groups)
        self.names = list(dict.fromkeys(groups))
        self.dt = policy_dt
        self.warmup_ticks = round(warmup_seconds / policy_dt)
        count = len(groups)
        self.frames = torch.zeros(count, dtype=torch.int64, device=device)
        self.episodes = torch.zeros_like(self.frames)
        self.failures = torch.zeros_like(self.frames)
        self.timeouts = torch.zeros_like(self.frames)
        self.boundary_timeouts = torch.zeros_like(self.frames)
        self.successes = torch.zeros_like(self.frames)
        self.sums = torch.zeros(count, len(self.ERROR_NAMES), dtype=torch.float64, device=device)
        self.torque_square = torch.zeros(count, 6, dtype=torch.float64, device=device)
        self.torque_peak = torch.zeros(count, 6, device=device)
        self.origin_xy = torch.zeros(count, 2, device=device)
        self.drift = torch.zeros(count, device=device)
        self.tilt = torch.zeros_like(self.drift)
        self.gap = torch.zeros_like(self.drift)
        self.reasons = {}
        self.task_peaks = {}
        self.reference_error_sum = torch.zeros(count, 2, dtype=torch.float64, device=device)

    def observe(self, data, active=None):
        active = torch.ones_like(self.frames, dtype=torch.bool) if active is None else active
        first = (data["episode_ticks"] == 1) & active
        self.origin_xy[first] = data["position"][first, :2]
        valid = active & (data["episode_ticks"] > self.warmup_ticks)
        self.frames += valid
        vx = data["velocity"][:, 0] - data["commands"][:, 0]
        yaw = data["omega"][:, 2] - data["commands"][:, 1]
        height = data["height"] - data["commands"][:, 2]
        values = torch.stack((vx.abs(), vx.square(), yaw.abs(), yaw.square(), height.abs(), height.square(), data["reward"]), -1)
        self.sums += values.double() * valid[:, None]
        if "reference_velocity_error_vector" in data:
            self.reference_error_sum += data["reference_velocity_error_vector"].double() * valid[:, None]
        torque = data["motor_effort"]
        self.torque_square += torque.double().square() * valid[:, None]
        self.torque_peak = torch.maximum(self.torque_peak, torque.abs() * valid[:, None])
        stand = data["commands"][:, :2].abs().amax(-1) < .01
        drift = (data["position"][:, :2] - self.origin_xy).norm(dim=-1)
        self.drift = torch.maximum(self.drift, drift * active * stand)
        tilt = torch.acos((-data["gravity"][:, 2]).clamp(-1., 1.)) * (180. / torch.pi)
        self.tilt = torch.maximum(self.tilt, tilt * valid)
        self.gap = torch.maximum(self.gap, data["gap"] * active)
        done = data["done"] & active
        failed = data["terminated"] & done
        success = data["success"] & done & ~failed
        self.episodes += done
        self.failures += failed
        self.successes += success
        self.timeouts += done & ~failed & ~success
        self.boundary_timeouts += done & ~failed & ~success & data["reasons"]["boundary"]
        for name, mask in data["reasons"].items():
            if name not in self.reasons:
                self.reasons[name] = torch.zeros_like(self.frames)
            self.reasons[name] += mask & done
        for name in ("jump_clearance_peak", "jump_air_time_peak", "jump_height_peak", "jump_release_velocity",
                     "jump_com_rise", "jump_com_release_speed", "settled_stop_speed"):
            if name in data:
                if name not in self.task_peaks:
                    self.task_peaks[name] = torch.zeros_like(self.drift)
                mask = valid if name == "settled_stop_speed" else active
                self.task_peaks[name] = torch.maximum(self.task_peaks[name], data[name] * mask)

    def report(self):
        result = {"sample_rate_hz": 1. / self.dt, "warmup_seconds": self.warmup_ticks * self.dt,
                  "reason_counts_may_overlap": True, "groups": {}}
        for name in self.names:
            ids = [i for i, group in enumerate(self.groups) if group == name]
            frames = int(self.frames[ids].sum())
            episodes = int(self.episodes[ids].sum())
            denominator = max(frames, 1)
            averages = (self.sums[ids].sum(0) / denominator).cpu().tolist()
            group = dict(zip(self.ERROR_NAMES, averages))
            group.update(frames=frames, episodes=episodes, failures=int(self.failures[ids].sum()),
                         timeouts=int(self.timeouts[ids].sum()), successes=int(self.successes[ids].sum()),
                         boundary_truncations=int(self.boundary_timeouts[ids].sum()),
                         vx_rmse_m_s=averages[1] ** .5, yaw_rmse_rad_s=averages[3] ** .5,
                         height_rmse_m=averages[5] ** .5, reward_per_sim_second=averages[6] / self.dt,
                         stand_drift_max_m=float(self.drift[ids].max()), tilt_max_deg=float(self.tilt[ids].max()),
                         closure_gap_max_m=float(self.gap[ids].max()),
                         rms_motor_torque_nm=(self.torque_square[ids].sum(0) / denominator).sqrt().cpu().tolist(),
                         peak_motor_torque_nm=self.torque_peak[ids].amax(0).cpu().tolist(),
                         termination_reasons={key: int(value[ids].sum()) for key, value in self.reasons.items()})
            result["groups"][name] = group
            reference_error = self.reference_error_sum[ids] / self.frames[ids, None].clamp_min(1)
            group["reference_velocity_error"] = float(reference_error.norm(dim=-1).max())
            group.update({key: float(value[ids].max()) for key, value in self.task_peaks.items()})
        return result
