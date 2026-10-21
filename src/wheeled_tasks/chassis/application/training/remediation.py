"""Authenticated S1 experiments that charge both forks to one formal budget."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

REVISION = "v6-s1-dense-remediation-v1"
ARMS = ("A_fixed", "B_stationary")
SOURCE_SHA = "f6cc481eee917cbd6a3034db409ecb8c0d7c5bcca41652ccb02bc7731be2c3da"
CHECKPOINT_SHA = "7d80b90dd5d59559beb05c0ae99ec09e92bea7380f9bfbef0256721bfcfb5447"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_changes(arm):
    if arm not in ARMS:
        raise ValueError("Unknown S1 remediation arm")
    return {"learning_rate": 1e-4, "learning_rate_schedule": "fixed",
            "resume_critic_warmup_updates": 10, "checkpoint_interval": 10,
            "checkpoint_first_update": 619 + 40 * ARMS.index(arm),
            "remediation": {"revision": REVISION, "arm": arm,
                "source_contract_sha256": SOURCE_SHA, "source_checkpoint_sha256": CHECKPOINT_SHA,
                "source_consumed_updates": 618, "source_stage_actor_update": 338,
                "arm_consumed_start": 618 + 40 * ARMS.index(arm),
                "arm_actor_consumed_start": 618 + 30 * ARMS.index(arm),
                "updates": 40, "critic_warmup_updates": 10, "learning_audit": True}}


def arm_contract(source, arm, binding):
    result = deepcopy(source)
    result.update(arm_changes(arm))
    if arm == "B_stationary":
        result["dense_tracking"]["stationary"].update(broad_weight=3., fine_weight=1.5)
    result["design_preflight"] = deepcopy(binding)
    return result


def validate_remediation_resume(source, target):
    settings = target.get("remediation", {})
    arm = settings.get("arm")
    if settings != arm_changes(arm)["remediation"]:
        raise ValueError("Remediation metadata differs from the frozen experiment")
    if (source.get("physics_dt"), source.get("policy_dt"), source.get("actor_dim"),
            source.get("critic_dim"), source.get("curriculum_stage")) != (.001, .02, 35, 81, "S1"):
        raise ValueError("Remediation requires the authenticated 1kHz S1 source")
    expected = arm_contract(source, arm, target.get("design_preflight"))
    if expected != target:
        changed = sorted(k for k in expected.keys() | target.keys() if expected.get(k) != target.get(k))
        raise ValueError(f"Remediation changes undeclared semantics: {changed}")
    if target["design_preflight"]["file"] != "contracts/v6_s1_remediation_expectations_v1.json":
        raise ValueError("Remediation requires its independent design table")
    return {"kind": REVISION, "arm": arm, "acceptance": "unchanged",
            "actor_critic_optimizer": "preserved", "learning_rate": 1e-4,
            "schedule": "fixed", "consumption": "shared_across_forks_including_critic_only"}


def validate_worker(config, args):
    settings = config["remediation"]
    if (not args.resume_reward_change or digest(args.resume) != CHECKPOINT_SHA
            or args.consumed_updates != settings["arm_consumed_start"]
            or args.actor_updates_consumed != settings["arm_actor_consumed_start"]
            or args.stage_actor_update != 338 or args.updates != 40
            or args.learning_rate_scale != 1. or args.num_envs != 20000
            or args.seed != 290927 or args.block_index != 0):
        raise ValueError("Remediation worker changes origin, counts, LR or paired sampling")


def charged_updates(completions, *, base=618):
    """Count every observed completed optimization, even on failed/discarded arms."""
    total = base
    for record in completions:
        n = record.get("successful_updates", 0)
        if type(n) is not int or not 0 <= n <= 40:
            raise ValueError("Invalid remediation consumption")
        total += n
    if total > 698:
        raise ValueError("Remediation exceeds its shared 80-update allowance")
    return total


def validate_evaluation(result, names):
    if (result.get("status") != "evaluated" or len(result.get("candidates", [])) != 1
            or set(result["candidates"][0].get("cases", {})) != set(names)
            or result["candidates"][0].get("unfinished_episodes") != 0):
        raise ValueError("invalid_evaluation_coverage")
    return result["candidates"][0]


def compare_arms(baselines, evaluations, names, protected):
    """Report tradeoffs; never auto-promote a diagnostic or erase old protection."""
    summaries = {}
    for arm in ARMS:
        pairs = evaluations[arm]
        lost = sorted(n for n in names if all(b["cases"][n]["passed"] for b in baselines)
                      and not all(e["cases"][n]["passed"] for e in pairs))
        pending = sorted(n for n in protected if not all(e["cases"][n]["passed"] for e in pairs))
        drift_names = [n for n in names if n.startswith("stand_") and "__" not in n]
        summaries[arm] = {"lost_from_origin": lost, "protected_still_pending": pending,
            "passed_both": sorted(n for n in names if all(e["cases"][n]["passed"] for e in pairs)),
            "mean_named_stand_drift_m": sum(e["cases"][n]["stand_drift_max_m"]
                for e in pairs for n in drift_names) / (len(pairs) * len(drift_names)),
            "worst_settled_stop_speed_m_s": max(e["cases"]["start_stop_05"]["settled_stop_speed"] for e in pairs)}
    a, b = (summaries[k] for k in ARMS)
    preferred = ("B_stationary" if not b["lost_from_origin"]
        and b["mean_named_stand_drift_m"] < a["mean_named_stand_drift_m"]
        and b["worst_settled_stop_speed_m_s"] < a["worst_settled_stop_speed_m_s"]
        and set(b["protected_still_pending"]) <= set(a["protected_still_pending"])
        else None)
    return {"arms": summaries, "candidate_for_review": preferred,
            "automatic_promotion": False, "automatic_formal_resume": False}
