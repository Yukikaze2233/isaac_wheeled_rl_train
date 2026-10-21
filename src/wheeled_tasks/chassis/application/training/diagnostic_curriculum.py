"""A reproducible paired diagnostic fork, separate from the formal curriculum."""
from copy import deepcopy
import math


PLAN_ID = "v6-p0-diagnostic-plan-v1"
REWARD_TERMS = (
    "track_lin_vel", "velocity_huber", "velocity_precision", "track_height", "height_square",
    "height_precision", "stand_translation", "stationary_speed_precision", "stand_precision",
    "pitch_exp", "pitch_velocity", "roll_exp", "roll_velocity", "no_fork", "no_fork_square",
    "motor_torque", "action_rate",
)


def validate_diagnostic_plan(plan):
    if plan["contract_id"] != PLAN_ID:
        raise ValueError("Unknown diagnostic study")
    stages = plan["stages"]
    if [s["name"] for s in stages] != ["A_control", "B_wide"]:
        raise ValueError("This paired study requires a control and a wide-kernel arm")
    if [s["updates"] for s in stages] != [100, 100] or plan["critic_adaptation_updates"] != 10:
        raise ValueError("Diagnostic budget is100 updates per arm, including10 critic updates")
    if stages[0]["precision_overrides"] or stages[1]["precision_overrides"] != {
            "stationary_width_m_s": .15, "height_width_m": .03}:
        raise ValueError("Only the two declared precision widths may differ between arms")
    if (len(plan["evaluation_seeds"]) != 2 or len(set(plan["evaluation_seeds"])) != 2
            or plan["num_steps_per_env"] != 24 or plan["source_updates"] != 750
            or not 0 < plan["scan_warmup_seconds"] < plan["episode_seconds"]):
        raise ValueError("Invalid diagnostic clocks, seeds or source checkpoint")
    if plan["heights_m"] != [.26, .28, .305, .32]:
        raise ValueError("Diagnostic height sweep must retain all four declared targets")


def diagnostic_contract(source, plan, arm, num_envs):
    """Keep mechanics/control exact; isolate nominal stand/forward training."""
    validate_diagnostic_plan(plan)
    if arm not in {s["name"] for s in plan["stages"]} or not 32 <= num_envs <= 16384:
        raise ValueError("Invalid diagnostic arm or environment count")
    if (source.get("actor_dim"), source.get("critic_dim"), source.get("physics_dt"),
            source.get("policy_dt")) != (35, 81, .005, .02):
        raise ValueError("Diagnostic fork requires the original35/81 200Hz/50Hz controller")
    if source["precision_tracking"]["stationary_width_m_s"] != .03 or source["precision_tracking"]["height_width_m"] != .01:
        raise ValueError("Diagnostic control widths differ from the authenticated experiment")
    config = deepcopy(source)
    recipe = next(s for s in plan["stages"] if s["name"] == arm)
    config.update(total_updates=recipe["updates"], target_num_envs=num_envs,
                  curriculum_stage="p0_diagnostic", critic_warmup_updates=10,
                  learning_rate=plan["learning_rate"],
                  checkpoint_interval=50, checkpoint_first_update=10,
                  episode_seconds=plan["episode_seconds"], diagnostic_logging=True)
    config["stages"][0]["updates"] = recipe["updates"]
    config["precision_tracking"].update(recipe["precision_overrides"])
    config["diagnostic_study"] = {"plan": deepcopy(plan), "arm": arm, "num_envs": num_envs}
    config.pop("protected_rehearsal", None)
    for key in ("dynamics_randomization", "contact_domain", "command_transport"):
        config[key]["enabled_fraction"] = 0.
    if config.get("signal_perturbations"):
        config["signal_perturbations"]["enabled"] = False
    config["observation_noise_enabled"] = False
    config["scene_groups"], config["skill_specs"] = [], {}
    config["behavior_pool_membership"] = {}
    for kind, vx in (("stand", 0.), ("forward", plan["forward_velocity_m_s"])):
        for height in plan["heights_m"]:
            name = f"diag_{kind}_h{round(height * 1000):03d}"
            config["scene_groups"].append({"name": name, "fraction": 1 / 8, "terrain": ["flat"]})
            config["skill_specs"][name] = {
                "kind": kind, "mode": int(vx != 0.), "command": [vx, 0., height],
                "sample_amplitude": False, "sample_yaw_sign": False,
                "episode_seconds": plan["episode_seconds"], "acceleration_m_s2": plan["acceleration_m_s2"],
                "terrain_limits": deepcopy(config["terrain_limits"])}
            config["behavior_pool_membership"][name] = kind
    config["behavior_pool_fractions"] = {"stand": .5, "forward": .5}
    settings = config["evaluation"]
    settings.update(mode="monitor", cumulative_retention=False, regression_patience=None,
                    skip_training_if_initially_accepted=False)
    settings.pop("regression_recovery", None)
    settings["protected_case_names"] = []
    settings["promotion_case_names"] = []
    config["training_reference"].update(
        scope="Paired engineering diagnostic; not formal capability promotion",
        initialization="Exact750 actor/critic/Adam fork; diagnostic counters start at0, source provenance retained",
        schedule="100 new optimization updates per arm, including10 critic-only adaptation updates")
    return config


def height_scan_contract(training_contract):
    config = deepcopy(training_contract)
    plan = config["diagnostic_study"]["plan"]
    cases = [{"name": name, "terrain": "flat", "command": list(spec["command"]),
              "skill": deepcopy(spec), "task": "survive", "anchor": False,
              "episode_seconds": plan["episode_seconds"], "warmup_seconds": plan["scan_warmup_seconds"]}
             for name, spec in config["skill_specs"].items()]
    config["evaluation"].update(protocol_id="v6-height-response-diagnostic-v1", cases=cases,
        canonical_case_names=[c["name"] for c in cases], retention_case_names=[c["name"] for c in cases],
        episode_seconds=plan["episode_seconds"], warmup_seconds=plan["scan_warmup_seconds"],
        episodes_per_case=plan["scan_episodes_per_case"], seed=plan["evaluation_seeds"][0])
    config["diagnostic_trace"] = True
    return config


def validate_diagnostic_resume(source, target):
    study = target.get("diagnostic_study", {})
    if not study:
        raise ValueError("Diagnostic resume requires the explicit paired study")
    expected = diagnostic_contract(source, study["plan"], study["arm"], study["num_envs"])
    if target != expected:
        changes = sorted(k for k in target.keys() | expected.keys() if target.get(k) != expected.get(k))
        raise ValueError(f"Diagnostic fork has undeclared changes: {changes}")
    return {"kind": "explicit_paired_diagnostic_fork", "arm": study["arm"],
            "formal_updates_consumed": study["plan"]["source_updates"],
            "diagnostic_update_ceiling": target["total_updates"],
            "actor_critic_optimizer": "preserved", "formal_acceptance": "not_modified_or_awarded"}


def height_response_fit(candidate, kind):
    points = [(c["command"][2], c["height_actual_mean_m"]) for name, c in candidate["cases"].items()
              if name.startswith(f"diag_{kind}_h") and c.get("frames", 0) > 0
              and c.get("full_horizon_episodes") == c.get("requested_episodes")]
    if len(points) < 3:
        return {"valid_points": len(points), "status": "insufficient_complete_episodes"}
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    variance = sum((p[0] - x)**2 for p in points)
    slope = sum((a - x) * (b - y) for a, b in points) / variance
    intercept = y - slope * x
    return {"valid_points": len(points), "slope": slope, "intercept_m": intercept,
            "fit_rmse_m": math.sqrt(sum((b - slope * a - intercept)**2 for a, b in points) / len(points)),
            "scope": "closed_loop_policy_response_not_a_calibration_correction"}


class PolicyUpdateProbe:
    """Measure update drift on the same stratified rollout observations, without sampling."""

    def __init__(self, actor, storage, batches):
        import torch
        self.actor, self.storage = actor, storage
        pieces, self.groups, offset = [], {}, 0
        for ids, spec in batches:
            selected = ids[:16]
            if not len(selected):
                continue
            pieces.append(selected)
            self.groups[spec["group_name"]] = slice(offset, offset + len(selected))
            offset += len(selected)
        self.ids = torch.cat(pieces)

    def _snapshot(self):
        import torch
        with torch.no_grad():
            self.actor.distribution.update(self.actor.mlp(self.actor.get_latent(self.observations)))
            return (tuple(p.clone() for p in self.actor.output_distribution_params),
                    self.actor.output_mean.clone(), self.actor.output_std.clone())

    def capture(self):
        self.observations = self.storage.observations[-1][self.ids].clone()
        self.before, self.before_mean, _ = self._snapshot()

    def metrics(self):
        import torch
        after, mean, std = self._snapshot()
        with torch.no_grad():
            kl = self.actor.get_kl_divergence(self.before, after)
            shift = (mean - self.before_mean).norm(dim=-1)
            result = {"DiagnosticUpdate/kl_rollout_probe": kl.mean()}
            for name, rows in self.groups.items():
                result[f"DiagnosticUpdate/{name}/kl"] = kl[rows].mean()
                result[f"DiagnosticUpdate/{name}/action_mean_shift_l2"] = shift[rows].mean()
            for axis, value in enumerate(std.mean(0)):
                result[f"DiagnosticUpdate/action_std_{axis}"] = value
            return result


class DiagnosticRewardAccumulator:
    """Accumulate full-rollout diagnostics without per-step scalar synchronization."""

    def __init__(self, count, device, batches, weights):
        import torch
        self.batches, self.names = batches, tuple(weights)
        # Lazy construction happens in rollout inference mode; these persistent
        # buffers must also be writable by the optimizer-side drain operation.
        with torch.inference_mode(False):
            self.weights = torch.tensor(list(weights.values()), device=device)
            self.rewards = torch.zeros(count, len(REWARD_TERMS), device=device)
            self.active = torch.zeros(count, len(weights), device=device)
            self.saturated = torch.zeros_like(self.active)
        self.steps = 0

    def observe(self, components, masks):
        import torch
        self.rewards += torch.stack([components[name] for name in REWARD_TERMS], -1)
        active = torch.stack([masks[name] for name in self.names], -1)
        values = torch.stack([components[name] for name in self.names], -1)
        self.active += active
        self.saturated += (values <= self.weights * math.expm1(-9.)) & active
        self.steps += 1

    def drain(self, policy_dt):
        if not self.steps:
            return {}
        result = {}
        for ids, spec in self.batches:
            if not len(ids):
                continue
            prefix = "/diagnostic/" + spec["group_name"]
            rewards = self.rewards[ids].mean(0) * policy_dt / self.steps
            active = self.active[ids].sum(0)
            saturation = self.saturated[ids].sum(0) / active.clamp_min(1)
            activation = active / (len(ids) * self.steps)
            result.update({prefix + "/reward/" + name: value for name, value in zip(REWARD_TERMS, rewards)})
            for i, name in enumerate(self.names):
                result[prefix + "/saturation/" + name] = saturation[i]
                result[prefix + "/activation/" + name] = activation[i]
        self.rewards.zero_()
        self.active.zero_()
        self.saturated.zero_()
        self.steps = 0
        return result
