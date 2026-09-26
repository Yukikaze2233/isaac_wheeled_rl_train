"""Per-group behavioral diagnostics, with explicit episode outcome denominators."""
from __future__ import annotations

import torch


class EpisodeMetrics:
    """Accumulate policy-rate observations without losing long-run count precision."""

    ERROR_NAMES = ("vx_mae_m_s", "vx_mse_m2_s2", "yaw_mae_rad_s", "yaw_mse_rad2_s2",
                   "height_mae_m", "height_mse_m2", "reward_per_policy_step",
                   "vx_command_mean_m_s", "vx_actual_mean_m_s", "vx_bias_m_s")

    def __init__(self, groups, device, policy_dt, warmup_seconds=0., warmup_by_group=None, height_range_m=None):
        self.groups = list(groups)
        self.names = list(dict.fromkeys(groups))
        self.dt = policy_dt
        self.warmup_ticks = round(warmup_seconds / policy_dt)
        self.warmup_per_env = torch.tensor([round((warmup_by_group or {}).get(g, warmup_seconds) / policy_dt)
                                           for g in groups], device=device)
        count = len(groups)
        self.frames = torch.zeros(count, dtype=torch.int64, device=device)
        self.observed_frames = torch.zeros_like(self.frames)
        self.episodes = torch.zeros_like(self.frames)
        self.early_episodes = torch.zeros_like(self.frames)
        self.terminal_ticks_sum = torch.zeros_like(self.frames)
        self.terminal_ticks_min = torch.full_like(self.frames, torch.iinfo(torch.int64).max)
        self.terminal_ticks_max = torch.zeros_like(self.frames)
        self.failures = torch.zeros_like(self.frames)
        self.timeouts = torch.zeros_like(self.frames)
        self.boundary_timeouts = torch.zeros_like(self.frames)
        self.blocked_timeouts = torch.zeros_like(self.frames)
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
        self.height_velocity_sum = torch.zeros(count, dtype=torch.float64, device=device)
        self.has_height_velocity = False
        self.height_min = torch.full((count,), torch.inf, device=device)
        self.height_max = torch.full((count,), -torch.inf, device=device)
        self.velocity_moments = torch.zeros(count, 2, dtype=torch.float64, device=device)
        self.height_edges = None
        self.reference_counts = None
        if height_range_m is not None:
            low, high = height_range_m
            if not (float("-inf") < low < high < float("inf")):
                raise ValueError("Diagnostic height range must be finite and increasing")
            self.height_edges = torch.linspace(low, high, 4, device=device)
            self.height_band_frames = torch.zeros(count, 3, dtype=torch.int64, device=device)
            self.height_band_error = torch.zeros(count, 3, dtype=torch.float64, device=device)
            self.height_band_tracking = torch.zeros_like(self.height_band_frames)

    def observe(self, data, active=None):
        active = torch.ones_like(self.frames, dtype=torch.bool) if active is None else active
        first = (data["episode_ticks"] == 1) & active
        self.origin_xy[first] = data["position"][first, :2]
        valid = active & (data["episode_ticks"] > self.warmup_per_env)
        self.observed_frames += active
        if "reference_phase" in data:
            if self.reference_counts is None:
                self.reference_counts = torch.zeros(len(self.groups), 9, dtype=torch.int64, device=active.device)
                self.reference_errors = torch.zeros(len(self.groups), 9, 2, dtype=torch.float64, device=active.device)
                self.jump_seen = torch.zeros(len(self.groups), 5, dtype=torch.bool, device=active.device)
                self.jump_events = torch.zeros(len(self.groups), 5, dtype=torch.int64, device=active.device)
            self.jump_seen[first] = False
            reference_error = torch.stack((data["reference_height_error"].abs(),
                                            (data["whole_com_vz"] - data["reference_vz"]).abs()), -1).double()
            for phase in range(9):
                chosen = active & (data["reference_phase"] == phase)
                self.reference_counts[:, phase] += chosen
                self.reference_errors[:, phase] += reference_error * chosen[:, None]
            requested = data["jump_requested"]
            events = torch.stack((requested, requested & (data["physical_phase"] == 1),
                requested & (data["jump_air_time_peak"] >= .06),
                requested & (data["jump_com_release_speed"] > .2), requested & data["success"]), -1)
            self.jump_events += events & ~self.jump_seen & active[:, None]
            self.jump_seen |= events & active[:, None]
        self.frames += valid
        vx = data["velocity"][:, 0] - data["commands"][:, 0]
        yaw = data["omega"][:, 2] - data["commands"][:, 1]
        height = data["height"] - data["commands"][:, 2]
        requested, actual = data["commands"][:, 0], data["velocity"][:, 0]
        values = torch.stack((vx.abs(), vx.square(), yaw.abs(), yaw.square(), height.abs(), height.square(),
                              data["reward"], requested, actual, vx), -1)
        self.sums += values.double() * valid[:, None]
        self.velocity_moments += torch.stack((requested.double().square(), requested.double() * actual), -1) * valid[:, None]
        if self.height_edges is not None:
            target = data["commands"][:, 2]
            bands = torch.bucketize(target.contiguous(), self.height_edges[1:-1], right=True)
            for index in range(3):
                selected = valid & (bands == index) & (target >= self.height_edges[0]) & (target <= self.height_edges[-1])
                self.height_band_frames[:, index] += selected
                self.height_band_error[:, index] += height.double().abs() * selected
                self.height_band_tracking[:, index] += selected & (height.abs() <= .01)
        self.height_min = torch.minimum(self.height_min, data["height"].masked_fill(~valid, torch.inf))
        self.height_max = torch.maximum(self.height_max, data["height"].masked_fill(~valid, -torch.inf))
        if "height_velocity_error" in data:
            self.has_height_velocity = True
            self.height_velocity_sum += data["height_velocity_error"].double().abs() * valid
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
        self.early_episodes += done & (data["episode_ticks"] <= self.warmup_per_env)
        self.terminal_ticks_sum += data["episode_ticks"] * done
        self.terminal_ticks_min = torch.minimum(self.terminal_ticks_min,
            data["episode_ticks"].masked_fill(~done, torch.iinfo(torch.int64).max))
        self.terminal_ticks_max = torch.maximum(self.terminal_ticks_max, data["episode_ticks"] * done)
        self.failures += failed
        self.successes += success
        self.timeouts += done & ~failed & ~success
        self.boundary_timeouts += done & ~failed & ~success & data["reasons"]["boundary"]
        self.blocked_timeouts += done & ~failed & ~success & ~data["reasons"]["boundary"] & data["reasons"].get("blocked", False)
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
        if self.height_edges is not None:
            result["height_band_edges_m"] = self.height_edges.cpu().tolist()
        for name in self.names:
            ids = [i for i, group in enumerate(self.groups) if group == name]
            frames = int(self.frames[ids].sum())
            episodes = int(self.episodes[ids].sum())
            denominator = max(frames, 1)
            averages = (self.sums[ids].sum(0) / denominator).cpu().tolist()
            group = dict(zip(self.ERROR_NAMES, averages))
            observed = int(self.observed_frames[ids].sum())
            group.update(observed_frames=observed, observed_sim_seconds=observed * self.dt,
                         post_warmup_sample_fraction=frames / max(observed, 1),
                         episodes_ending_before_or_at_warmup=int(self.early_episodes[ids].sum()),
                         terminal_episode_time_mean_s=float(self.terminal_ticks_sum[ids].sum()) * self.dt / episodes if episodes else None,
                         terminal_episode_time_min_s=int(self.terminal_ticks_min[ids].min()) * self.dt if episodes else None,
                         terminal_episode_time_max_s=int(self.terminal_ticks_max[ids].max()) * self.dt if episodes else None)
            moment = (self.velocity_moments[ids].sum(0) / denominator).cpu().tolist()
            variance = max(0., moment[0] - averages[7]**2)
            group["vx_command_variance_m2_s2"] = variance
            # This is descriptive covariance, not a causal policy sensitivity.
            group["vx_command_response_slope"] = (moment[1] - averages[7] * averages[8]) / variance if frames > 1 and variance > 1e-8 else None
            if self.height_edges is not None:
                for index, band in enumerate(("low", "mid", "high")):
                    n = int(self.height_band_frames[ids, index].sum())
                    group[f"height_{band}_frames"] = n
                    group[f"height_{band}_mae_m"] = float(self.height_band_error[ids, index].sum()) / n if n else None
                    group[f"height_{band}_within_10mm_fraction"] = float(self.height_band_tracking[ids, index].sum()) / n if n else None
            group["height_min_m"] = float(self.height_min[ids].min()) if frames else None
            group["height_max_m"] = float(self.height_max[ids].max()) if frames else None
            if self.has_height_velocity:
                group["height_velocity_mae_m_s"] = float(self.height_velocity_sum[ids].sum()) / denominator
            group.update(frames=frames, episodes=episodes, failures=int(self.failures[ids].sum()),
                         timeouts=int(self.timeouts[ids].sum()), successes=int(self.successes[ids].sum()),
                          boundary_truncations=int(self.boundary_timeouts[ids].sum()),
                          blocked_truncations=int(self.blocked_timeouts[ids].sum()),
                         vx_rmse_m_s=averages[1] ** .5, yaw_rmse_rad_s=averages[3] ** .5,
                         height_rmse_m=averages[5] ** .5, reward_per_sim_second=averages[6] / self.dt,
                         stand_drift_max_m=float(self.drift[ids].max()), tilt_max_deg=float(self.tilt[ids].max()),
                         closure_gap_max_m=float(self.gap[ids].max()),
                         rms_motor_torque_nm=(self.torque_square[ids].sum(0) / denominator).sqrt().cpu().tolist(),
                         peak_motor_torque_nm=self.torque_peak[ids].amax(0).cpu().tolist(),
                         termination_reasons={key: int(value[ids].sum()) for key, value in self.reasons.items()})
            result["groups"][name] = group
            group["warmup_seconds"] = float(self.warmup_per_env[ids].max()) * self.dt
            if self.reference_counts is not None:
                for phase in range(9):
                    n = int(self.reference_counts[ids, phase].sum())
                    group[f"reference_phase_{phase}_frames"] = n
                    if n:
                        values = self.reference_errors[ids, phase].sum(0) / n
                        group[f"reference_phase_{phase}_height_mae_m"] = float(values[0])
                        group[f"reference_phase_{phase}_vz_mae_m_s"] = float(values[1])
                for index, event in enumerate(("requested", "takeoff_entered", "airborne_60ms", "positive_com_release", "succeeded")):
                    group["jump_episodes_" + event] = int(self.jump_events[ids, index].sum())
            reference_error = self.reference_error_sum[ids] / self.frames[ids, None].clamp_min(1)
            group["reference_velocity_error"] = float(reference_error.norm(dim=-1).max())
            group.update({key: float(value[ids].max()) for key, value in self.task_peaks.items()})
        return result
