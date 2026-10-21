"""Learning configuration and restart decisions without eager Torch imports."""
import math


def resume_critic_warmup(config, infos, *, contract_sha256=None, task_change=False, reward_change=False):
    """Adapt once to a named contract change; a fault resumes the remaining work."""
    if "formal_critic_adaptation" in config:
        from wheeled_tasks.chassis.application.training.critic_adaptation import remaining_updates
        return remaining_updates(config, infos)
    if task_change and contract_sha256 is None:
        raise ValueError("Task adaptation requires the frozen contract SHA")
    revision = config.get("task_campaign", {}).get("revision_migration", {})
    resize_only = (task_change and revision.get("schema") == "task-worker-revision-v5"
        and infos.get("contract_sha256") == revision.get("source_contract_sha256")
        and infos.get("task_campaign", {}).get("domain") == config["task_campaign"]["domain"])
    if reward_change or task_change and infos.get("contract_sha256") != contract_sha256 and not resize_only:
        return config.get("resume_critic_warmup_updates", 0)
    return infos.get("critic_warmup_remaining", max(0,
        config.get("critic_warmup_updates", 0) - infos["successful_updates_total"]))


def configure_learning_rate(algorithm, contract_rate, scale=None):
    """Keep PPO's scalar aligned with restored Adam, or apply an explicit ratio.

    Recovery's decay factor is bounded by the orchestrator. The resulting
    ratio to the contract's initial LR can exceed one after adaptive updates.
    """
    rate = algorithm.optimizer.param_groups[0]["lr"]
    if scale is not None:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Recovery learning-rate scale must be finite and positive")
        rate = contract_rate * scale
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Optimizer learning rate must be finite and positive")
    if scale is not None:
        for group in algorithm.optimizer.param_groups:
            group["lr"] = rate
    algorithm.learning_rate = rate
    return rate


def completion_learning_rate(optimizer):
    """Read a JSON-safe final rate, rejecting empty/inconsistent Adam groups."""
    rates = [group.get("lr") for group in optimizer.param_groups]
    if not rates or any(not isinstance(rate, (int, float)) or isinstance(rate, bool)
            or not math.isfinite(rate) or rate <= 0 for rate in rates) or any(
            rate != rates[0] for rate in rates):
        raise ValueError(f"Invalid optimizer learning rates at completion: {rates!r}")
    return rates[0]


def validate_entropy_coef(value):
    """Accept a contract scalar without changing the native distribution."""
    if type(value) not in (int, float):
        raise ValueError("entropy_coef must be a finite nonnegative scalar")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("entropy_coef must be a finite nonnegative scalar") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError("entropy_coef must be a finite nonnegative scalar")
    return value


def verify_entropy_coef(algorithm, expected):
    """Read back the native loss coefficient; never reset std or Adam."""
    try:
        actual = validate_entropy_coef(algorithm.entropy_coef)
    except ValueError as error:
        raise RuntimeError("Native PPO entropy coefficient is invalid") from error
    if actual != expected:
        raise RuntimeError("Native PPO entropy coefficient differs from its configuration")
    return actual


def configure_recovery_rollout(algorithm, env):
    """Exclude any opted-in interval whose motor output is externally owned."""
    if not (env.cfg.get("recovery_training", {}).get("enabled", False)
            or env.cfg.get("nonstandard_start", {}).get("enabled", False)
            or getattr(env, "flat_episode", None) is not None):
        return None
    from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
    return attach_recovery_rollout_mask(algorithm, num_envs=env.num_envs,
        require_terminal_observation="formal_recovery_episode_window" in env.cfg)
