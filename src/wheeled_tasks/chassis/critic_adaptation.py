"""One explicit critic reset at the retained-policy course boundary."""
from copy import deepcopy


KEY = "formal_critic_adaptation"
SETTINGS = {
    "version": "motion-critic-coordinate-reset-v1",
    "source_contract_sha256": "9ba0b6ee95d48177b74d9c7388c2693fd067912fb7f6072de5eb9728f2d9066f",
    "source_checkpoint_sha256": "725c1cdd03e19912c0a78c299c64104151d4c56fea0cb2727752da753fc60069",
    "source_learning_updates": 14841,
    "initialization": "fresh_critic_preserve_actor_std",
    "critic_only_updates_before_review": 50,
    "after_window": "checkpoint_and_review",
}


def selection(contract):
    if KEY not in contract:
        return None
    value = contract[KEY]
    if not isinstance(value, dict) or set(value) != set(SETTINGS):
        raise ValueError("Critic adaptation requires complete versioned settings")
    for key, expected in SETTINGS.items():
        actual = value[key]
        if key == "critic_only_updates_before_review":
            if type(actual) is not int or actual <= 0:
                raise ValueError("Critic-only readback window requires a positive integer")
        elif type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Unsupported critic adaptation {key}")
    if (contract.get("formal_small_rough_mix", {}).get("version") != "motion-retention-course-v1"
            or contract.get("critic_dim") != 78 or contract.get("actor_dim") != 35
            or contract.get("critic_height_reference") != "root_link_world_z"
            or contract.get("policy_signal_delay", {}).get("version") != "policy-signal-lag-v1"
            or contract.get("formal_stage_optimizer") != "scut_finetune_v1"
            or contract.get("critic_warmup_updates", 0) != 0
            or contract.get("resume_critic_warmup_updates", 0) != 0):
        raise ValueError("Critic adaptation requires the explicit changed-input course and no second warmup owner")
    return deepcopy(value)


def validate_pair(source, target):
    """Reject a claimed coordinate reset when the actual source does not match."""
    cfg = selection(target)
    if cfg is None:
        return None
    if (source.get("critic_dim") != 78
            or source.get("critic_height_reference") not in (None, "root_world_z_minus_environment_origin_z")
            or "policy_signal_delay" in source or source.get("reward_profile") != "scut0719_flat_v1"
            or target.get("reward_profile") != "scut0730_rough_v1"):
        raise ValueError("Critic coordinate reset does not match the authenticated source/target semantics")
    return {"initialization": cfg["initialization"],
        "changed_critic_inputs": ["height_datum", "observation_age", "target_pipeline_age"],
        "value_target": "changed reward profile and exact-command repair",
        "warmup_window_is_not_qualification": True}


def remaining_updates(contract, infos):
    """A parent's completed warmup cannot cancel a new coordinate adaptation."""
    cfg = selection(contract)
    if cfg is None:
        raise ValueError("Critic adaptation was not selected")
    if infos.get("stage") == "foundation":
        if (infos.get("contract_sha256") != cfg["source_contract_sha256"]
                or infos.get("successful_updates_total") != cfg["source_learning_updates"]):
            raise ValueError("Critic adaptation requires its authenticated14841 parent")
        return cfg["critic_only_updates_before_review"]
    saved = infos.get(KEY)
    if infos.get("stage") != "terrain" or not isinstance(saved, dict):
        raise ValueError("Critic adaptation restart requires its saved configuration")
    remaining = infos.get("critic_warmup_remaining")
    old_count = saved.get("critic_only_updates_before_review")
    if (type(old_count) is not int or old_count <= 0 or type(remaining) is not int
            or not 0 <= remaining <= old_count):
        raise ValueError("Critic adaptation restart requires a valid remaining counter")
    if saved == cfg:
        return remaining
    validate_extension({KEY: saved}, contract)
    if infos.get("stage_actor_update") != 0:
        raise ValueError("Critic-only extension is invalid after already completed actor updates")
    return remaining + cfg["critic_only_updates_before_review"] - old_count


def validate_extension(source, target):
    """Extend only a readback window; keep the learned critic and Adam intact."""
    before, after = source.get(KEY), selection(target)
    if not isinstance(before, dict) or after is None:
        raise ValueError("Critic extension requires the previous explicit recipe")
    old_count = before.get("critic_only_updates_before_review")
    if (type(old_count) is not int or old_count <= 0
            or after["critic_only_updates_before_review"] <= old_count
            or before != {**after, "critic_only_updates_before_review": old_count}):
        raise ValueError("Critic extension may only increase the total readback window")


def verify_frozen_actor(algorithm, reference):
    """Check actual policy/std tensors and untouched actor Adam state."""
    import torch

    actual = algorithm.actor.state_dict()
    if actual.keys() != reference.keys() or any(not torch.equal(actual[key], expected)
            for key, expected in reference.items()):
        raise RuntimeError("Critic-only update changed actor or exploration state")
    parameters = list(algorithm.actor.parameters())
    if any(parameter.requires_grad or parameter.grad is not None for parameter in parameters):
        raise RuntimeError("Critic-only update retained an actor gradient")
    if any(parameter in algorithm.optimizer.state for parameter in parameters):
        raise RuntimeError("Critic-only update modified actor optimizer state")
    return {"actor_and_std_exact": True, "actor_optimizer_state_entries": 0}
