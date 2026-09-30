"""A fixed-LR S1 continuation with separate learning lineage and spent budget."""
from copy import deepcopy

from .remediation import digest

REVISION = "v6-s1-fixed-continuation-v1"
SOURCE_SHA = "8c1b5b2b8c5c8352b00d1db8a966a5698df3c4756d0d4bb756c96646d1c1a6fc"
CHECKPOINT_SHA = "f3d83941049030591fcd57acb41df241b904232b9dc0d387c178292625d5270b"
ORIGIN = "reports/v6_s1_continuation_20260930/origin.json"


def settings():
    return dict(revision=REVISION, source_contract_sha256=SOURCE_SHA,
        source_checkpoint_sha256=CHECKPOINT_SHA, source_consumed_updates=698,
        source_actor_consumed_updates=678, source_stage_actor_update=368,
        source_learning_lineage_updates=378, updates=40, learning_audit=True)


def worker_contract(source, binding):
    result = deepcopy(source)
    result.pop("remediation")
    result.update(continuation=settings(), resume_critic_warmup_updates=0,
                  checkpoint_first_update=699, design_preflight=deepcopy(binding))
    return result


def validate_resume(source, target):
    if (source.get("remediation", {}).get("arm") != "A_fixed"
            or source.get("learning_rate") != 1e-4
            or source.get("learning_rate_schedule") != "fixed"):
        raise ValueError("Continuation requires the authenticated fixed-LR A source")
    expected = worker_contract(source, target.get("design_preflight"))
    if expected != target:
        changed = sorted(k for k in expected.keys() | target.keys() if expected.get(k) != target.get(k))
        raise ValueError(f"Continuation changes undeclared contract fields: {changed}")
    return dict(kind=REVISION, reward="unchanged", sampling="unchanged", acceptance="unchanged",
        actor_critic_optimizer="preserved", learning_rate=1e-4, schedule="fixed",
        critic_warmup_updates=0, consumption="includes_discarded_B_updates")


def validate_worker(config, args):
    if (config["continuation"] != settings() or not args.resume_reward_change
            or not args.resume or digest(args.resume) != CHECKPOINT_SHA
            or args.consumed_updates != 698 or args.actor_updates_consumed != 678
            or args.stage_actor_update != 368 or args.updates != 40
            or args.global_actor_update_offset != 0 or args.learning_rate_scale != 1.
            or args.num_envs != 20000 or args.seed != 290927 or args.block_index != 0):
        raise ValueError("Continuation worker changes origin, consumption, clock or fixed LR")


def charged_updates(records):
    records = list(records)
    if len(records) > 1:
        raise ValueError("Only one continuation worker is authorized")
    total = 698
    for record in records:
        count = record.get("successful_updates", 0)
        actor = record.get("actor_updates_in_block", 0)
        if type(count) is not int or not 0 <= count <= 40 or actor != count:
            raise ValueError("Continuation must charge every actor/critic update without warmup")
        total += count
    return total


def compare(baselines, evaluations, protected, targets):
    names = set(baselines[0]["cases"])
    if (len(baselines) != 2 or len(evaluations) != 2
            or any(set(e["cases"]) != names for e in [*baselines, *evaluations])
            or set(protected) - names or set(targets) - names):
        raise ValueError("invalid_evaluation_coverage")
    before = {n for n in names if all(e["cases"][n]["passed"] for e in baselines)}
    passed = {n for n in names if all(e["cases"][n]["passed"] for e in evaluations)}
    return dict(passed_both=sorted(passed), newly_passed=sorted(passed - before),
        lost_from_A=sorted(before - passed), protected_still_pending=sorted(set(protected) - passed),
        target_still_pending=sorted(set(targets) - passed),
        ready_for_full_audit=not (set(protected) | set(targets)) - passed,
        automatic_promotion=False)
