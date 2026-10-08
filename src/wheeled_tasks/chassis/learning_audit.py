"""Observe native dense rewards and PPO batches without modifying either."""
import torch

from .diagnostic_curriculum import PolicyUpdateProbe
from .recovery_prefix_curriculum import LESSONS
from .task import Phase


class RecoveryPhaseAudit:
    """Rollout event counts by actual birth lesson, using native producer truth.

    Event latches survive rollout boundaries and clear after the producer's
    terminal snapshot. Ended-event counts use that same snapshot, so their
    denominator is the change in existing RecoveryOutcomes/.../ended counters.
    """
    EVENTS = ("script_complete", "pure_rl_started", "stable_hold", "motion_released")

    def __init__(self, count, device, stable_seconds):
        self.stable_seconds = stable_seconds
        self.seen = torch.zeros(count, len(self.EVENTS), dtype=torch.bool, device=device)
        self.previous_done = torch.zeros(count, dtype=torch.bool, device=device)
        self.counts = torch.zeros(3, 2 * len(self.EVENTS) + 2, dtype=torch.long, device=device)

    def observe(self, data, policy_learning_mask):
        enabled, levels = data["recovery_enabled"], data["recovery_prefix_lesson_id"]
        if (policy_learning_mask is None or policy_learning_mask.shape != enabled.shape
                or policy_learning_mask.dtype != torch.bool):
            raise ValueError("Recovery phase audit needs the final native policy-learning mask")
        self.seen &= ~self.previous_done[:, None]
        eligible = enabled & (levels >= 0) & (levels < len(LESSONS))
        events = torch.stack((data["recovery_ever_blended"],
            data["recovery_handover_count"] > 0,
            data["recovery_stable_peak_seconds"] >= self.stable_seconds - 1e-5,
            data["recovery_motion_released"]), -1) & eligible[:, None]
        first = events & ~self.seen
        self.seen |= events
        ended = self.seen & data["done"][:, None]
        values = torch.cat((first, ended, eligible[:, None],
            (eligible & policy_learning_mask)[:, None]), -1).long()
        for level in range(len(LESSONS)):
            self.counts[level] += (values * (eligible & (levels == level))[:, None]).sum(0)
        self.previous_done.copy_(data["done"])

    def metrics(self):
        result = {}
        for level, lesson in enumerate(LESSONS):
            prefix = "RecoveryPhaseAudit/" + lesson + "/"
            for column, event in enumerate(self.EVENTS):
                result[prefix + "first_" + event] = self.counts[level, column]
                result[prefix + "ended_with_" + event] = self.counts[level, column + len(self.EVENTS)]
            observed, owned = self.counts[level, -2:]
            result.update({prefix + "observed_transitions": observed,
                prefix + "policy_owned_transitions": owned,
                prefix + "policy_owned_fraction": owned / observed.clamp_min(1)})
        return result

    def clear_report(self):
        self.counts.zero_()


class StationaryWindowAudit:
    """Completed 10 s quiet windows, separate from lifetime maximum displacement."""
    def __init__(self, count, device, policy_dt):
        self.warmup_ticks, self.window_ticks = round(2. / policy_dt), round(10. / policy_dt)
        self.quiet_ticks = torch.zeros(count, dtype=torch.long, device=device)
        self.ticks = torch.zeros_like(self.quiet_ticks)
        self.active = torch.zeros(count, dtype=torch.bool, device=device)
        self.anchor = torch.zeros(count, 2, device=device)
        self.peak = torch.zeros(count, device=device)
        self.speed = torch.zeros_like(self.peak)
        self.counts = torch.zeros_like(self.quiet_ticks)
        self.completed_sum = torch.zeros(count, 3, device=device)
        self.completed_max = torch.zeros_like(self.peak)

    def observe(self, data):
        quiet = (data["commands"][:, :2].abs() < .001).all(-1) & ~data["done"].bool()
        self.quiet_ticks.copy_(torch.where(quiet, self.quiet_ticks + 1, 0))
        eligible = (quiet & (self.quiet_ticks >= self.warmup_ticks)
                    & ((data["height"] - data["commands"][:, 2]).abs() <= .01))
        entering = eligible & ~self.active
        self.anchor.copy_(torch.where(entering[:, None], data["position"][:, :2], self.anchor))
        self.ticks.copy_(torch.where(eligible & ~entering, self.ticks + 1, 0))
        displacement = (data["position"][:, :2] - self.anchor).norm(dim=-1)
        self.peak.copy_(torch.where(eligible & ~entering, torch.maximum(self.peak, displacement), 0.))
        self.speed.copy_(torch.where(eligible & ~entering, self.speed + data["planar_speed_world"], 0.))
        completed = self.ticks >= self.window_ticks
        self.counts += completed
        values = torch.stack((displacement, self.peak, self.speed / self.window_ticks), -1)
        self.completed_sum += values * completed[:, None]
        self.completed_max.copy_(torch.maximum(self.completed_max, self.peak * completed))
        self.active.copy_(eligible & ~completed)

    def report(self, ids):
        count = self.counts[ids].sum()
        endpoint, peak, speed = self.completed_sum[ids].sum(0) / count.clamp_min(1)
        return dict(samples=count, endpoint_mean_m=endpoint, peak_mean_m=peak,
                    peak_max_m=self.completed_max[ids].max(), mean_speed_m_s=speed)

    def clear_report(self):
        self.counts.zero_()
        self.completed_sum.zero_()
        self.completed_max.zero_()


class LearningAudit:
    def __init__(self, env, algorithm):
        self.env, self.algorithm = env, algorithm
        self.probe = PolicyUpdateProbe(algorithm.actor, algorithm.storage, env.skills.batches)
        self.hold = torch.zeros(env.num_envs, device=env.device)
        self.previous_height = env.commands[:, 2].clone()
        self.rollout_height = self.previous_height.clone()
        self.stable_height = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.sums = None
        self.steps = 0
        self.pending = {}
        self.stop_sums = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self.stop_peak = torch.zeros(env.num_envs, 2, device=env.device)
        self.stationary_windows = StationaryWindowAudit(env.num_envs, env.device, env.policy_dt)
        self.release_groups = {spec["group_name"] for _, spec in env.skills.batches
                               if spec["kind"] == "release_stop"}
        enabled = env.cfg.get("recovery_phase_audit", False)
        if type(enabled) is not bool:
            raise ValueError("recovery_phase_audit must be an explicit boolean")
        self.recovery_phases = None
        if enabled:
            if env.cfg.get("auto_reset", True) is not True:
                raise ValueError("Recovery phase audit requires native episode auto-reset")
            recovery = getattr(env, "recovery", None)
            if not callable(getattr(getattr(recovery, "pose_curriculum", None), "rollout_outcome_metrics", None)):
                raise ValueError("Recovery phase audit requires the native prefix lesson outcomes")
            self.recovery_phases = RecoveryPhaseAudit(env.num_envs, env.device,
                recovery.fsm.config["stable_seconds"])

    def observe(self, data, components, transition_info=None):
        if self.recovery_phases is not None:
            self.recovery_phases.observe(data, (transition_info or {}).get("policy_learning_mask"))
        if "position" in data:
            self.stationary_windows.observe(data)
        if self.steps == 0:
            self.rollout_height.copy_(data["commands"][:, 2])
            self.stable_height.fill_(True)
        self.stable_height &= (data["commands"][:, 2] - self.rollout_height).abs() < 1e-6
        quiet = (data["commands"][:, :2].abs() < .001).all(-1)
        unchanged = (data["commands"][:, 2] - self.previous_height).abs() < 1e-6
        self.hold.copy_(torch.where(quiet & unchanged, self.hold + self.env.policy_dt, 0.))
        self.previous_height.copy_(data["commands"][:, 2])
        self.last_commands = data["commands"]
        values = {"reward_density/" + k: v for k, v in components.items()}
        values.update(total_reward_density=data["reward"] / self.env.policy_dt,
            planar_speed_m_s=data["planar_speed_world"],
            height_error_m=(data["height"] - data["commands"][:, 2]).abs(),
            quiet_nonfall=((data["commands"][:, :2].abs() < .001).all(-1)
                             & (data["reasons"]["fall"] == 0)).float(),
            continuous_zero_10s=(self.hold >= 10.).float(),
            continuous_zero_15s=(self.hold >= 15.).float(),
            done=data["done"].float())
        if self.env.cfg.get("reward_units"):
            vx, yaw = data["commands"][:, 0], data["commands"][:, 1]
            values.update(command_vx_m_s=vx, command_yaw_rad_s=yaw,
                held_low_vx_fraction=((vx.abs() >= .1) & (vx.abs() <= .3) & (yaw.abs() < .01)).float(),
                held_low_yaw_fraction=((yaw.abs() >= .1) & (yaw.abs() <= .3) & (vx.abs() < .01)).float())
            for sx, sy in ((1,1),(1,-1),(-1,1),(-1,-1)):
                values[f"joint_command/vx_{sx}_yaw_{sy}_fraction"] = ((vx * sx > .01) & (yaw * sy > .01)).float()
        if "physical_phase" in data:
            for phase in Phase:
                mask = data["physical_phase"] == phase
                values[f"phase/{phase.name.lower()}_fraction"] = mask.float()
                if "physical_phase_time_s" in data:
                    values[f"phase/{phase.name.lower()}_time_s"] = mask * data["physical_phase_time_s"]
        if "landing_reward_mask" in data:
            values["landing/paid_mask_fraction"] = (data["landing_reward_mask"] > 0).float()
            values["landing/window_weight"] = data["landing_reward_mask"]
        if "landing_window_elapsed_s" in data:
            values["landing/window_elapsed_s"] = data["landing_window_elapsed_s"]
            values["landing/task_eligible_fraction"] = data["landing_task_eligible"].float()
        if "jump_landing_active" in data:
            for name in ("active", "touchdown", "verified_flight", "cycle_accepted", "expired",
                         "outcome_timeout", "entry_height_clipped", "entry_speed_clipped"):
                values["jump_contact/" + name + "_fraction"] = data["jump_landing_" + name].float()
            values["jump_contact/elapsed_s"] = data["jump_landing_elapsed_s"]
            values["jump_contact/released_paid_fraction"] = (
                data["jump_landing_active"] & ~data["jump_requested"]).float()
        if "velocity" in data:
            values["velocity/signed_forward_m_s"] = data["velocity"][:, 0]
            values["velocity/signed_lateral_m_s"] = data["velocity"][:, 1]
        if "start_stop_waiting" in data:
            values["start_stop/waiting_fraction"] = data["start_stop_waiting"].float()
            values["start_stop/profile_time_s"] = data["start_stop_profile_time_s"]
            values["start_stop/start_delay_s"] = data["start_stop_start_delay_s"]
        if "release_requested" in data:
            for name, key in (("requested", "release_requested"),
                              ("waiting", "release_stop_waiting"),
                              ("driving", "release_stop_driving"),
                              ("accelerating", "release_stop_accelerating"),
                              ("cruising", "release_stop_cruising"),
                              ("decelerating", "release_stop_decelerating"),
                              ("braking", "release_stop_braking"),
                              ("settled", "release_stop_settled")):
                values["release_stop/" + name + "_fraction"] = data[key].float()
            for name in ("wait", "drive", "dwell", "brake_budget", "release_at", "end_at"):
                values["release_stop/" + name + "_s"] = data["release_stop_" + name + "_s"]
            pose_values = dict(
                pitch_abs_rad=data["gravity"][:, 0].clamp(-1., 1.).asin().abs(),
                roll_abs_rad=torch.atan2(-data["gravity"][:, 1], -data["gravity"][:, 2]).abs(),
                gyro_xy_rad_s=data["omega"][:, :2].norm(dim=-1),
                speed_m_s=data["planar_speed_world"])
            for phase in ("accelerating", "cruising", "decelerating", "stationary_window"):
                weight = data["release_stop_" + phase].float()
                values[f"release_posture/{phase}/fraction"] = weight
                for key, value in pose_values.items():
                    values[f"release_posture/{phase}/{key}"] = weight * value
        if self.sums is None:
            self.names = tuple(values)
            # observe runs under rollout inference mode, drain under optimization.
            with torch.inference_mode(False):
                self.sums = torch.zeros(self.env.num_envs, len(values), device=self.env.device)
        if tuple(values) != self.names:
            raise ValueError("Reward audit component keys changed within a rollout")
        self.sums += torch.stack(list(values.values()), -1)
        if "braking_window" in data:
            # Same body-plane speed and settle mask as the fixed evaluation metric.
            speed = data["velocity"][:, :2].norm(dim=-1)
            for i, phase in enumerate(("braking_window", "settled_stop_window")):
                mask = data[phase].float()
                self.stop_sums[:, i] += torch.stack((mask, speed * mask, (speed > .03) * mask), -1)
                self.stop_peak[:, i] = torch.maximum(self.stop_peak[:, i], speed * mask)
        self.steps += 1
        self.hold.masked_fill_(data["done"].bool(), 0.)

    def groups(self):
        for ids, spec in self.env.skills.batches:
            yield spec["group_name"], ids
            if spec["kind"] == "stand":
                for h in self.env.cfg["new_asset_curriculum"]["height_sampling_schedule"]["fixed_heights_m"]:
                    selected = ids[((self.last_commands[ids, 2] - h).abs() < 1e-5) & self.stable_height[ids]]
                    if len(selected):
                        yield f"{spec['group_name']}/height_{round(h * 1000)}mm", selected

    def capture(self):
        self.probe.capture()
        storage = self.algorithm.storage
        for name, ids in self.groups():
            prefix = "LearningAudit/" + name + "/"
            advantage = storage.advantages[:, ids]
            residual = storage.returns[:, ids] - storage.values[:, ids]
            self.pending.update({prefix + "advantage_mean": advantage.mean(),
                prefix + "advantage_positive_fraction": (advantage > 0).float().mean(),
                prefix + "return_target_residual_mean": residual.mean(),
                prefix + "rollout_samples": advantage.new_tensor(advantage.numel())})

    def drain(self):
        result = {**self.pending, **self.probe.metrics()}
        if self.recovery_phases is not None:
            result.update(self.recovery_phases.metrics())
        for name, ids in self.groups():
            mean = self.sums[ids].mean(0) / max(1, self.steps)
            result.update({f"LearningAudit/{name}/{key}": value for key, value in zip(self.names, mean)
                           if not key.startswith(("release_posture/", "release_stop/")) or name in self.release_groups})
            if name in self.release_groups:
                for phase in ("accelerating", "cruising", "decelerating", "stationary_window"):
                    prefix = f"LearningAudit/{name}/release_posture/{phase}/"
                    count = result[prefix + "fraction"] * len(ids) * self.steps
                    result[prefix + "samples"] = count
                    for key in ("pitch_abs_rad", "roll_abs_rad", "gyro_xy_rad_s", "speed_m_s"):
                        result[prefix + key] /= result[prefix + "fraction"].clamp_min(1e-12)
            # Phase duration is conditional on actually occupying that phase.
            for phase in Phase:
                prefix = f"LearningAudit/{name}/phase/{phase.name.lower()}"
                if prefix + "_time_s" in result:
                    result[prefix + "_time_s"] /= result[prefix + "_fraction"].clamp_min(1e-12)
            for i, phase in enumerate(("braking", "settled_stop")):
                count, speed, exceed = self.stop_sums[ids, i].sum(0)
                prefix = f"LearningAudit/{name}/{phase}/"
                result.update({prefix + "samples": count,
                    prefix + "mean_speed_m_s": speed / count.clamp_min(1),
                    prefix + "above_3cm_s_fraction": exceed / count.clamp_min(1),
                    prefix + "peak_speed_m_s": self.stop_peak[ids, i].max()})
            result.update({f"LearningAudit/{name}/stationary_10s/{key}": value
                           for key, value in self.stationary_windows.report(ids).items()})
        keys = tuple(result)
        values = torch.stack([result[k].detach() for k in keys]).cpu().tolist()
        self.sums.zero_()
        self.stop_sums.zero_()
        self.stop_peak.zero_()
        self.stationary_windows.clear_report()
        if self.recovery_phases is not None:
            self.recovery_phases.clear_report()
        self.steps = 0
        self.pending.clear()
        return dict(zip(keys, values))
