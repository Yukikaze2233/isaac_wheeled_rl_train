"""One exact sampling revision: stationary, low and high commands share a policy."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FLAT_FILE = "contracts/v6_task_campaign_flat_worker_v6.json"
FLAT_SHA = "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7"
SPEED_FILE = "contracts/v6_task_campaign_speed_worker_v6.json"
SPEED_SHA = "e8961567035e9287528e9a689e791b4f12f8966a3abeae36e492f34ecedcb52d"
DESIGN_FILE = "contracts/v6_task_campaign_speed_expectations_v10.json"
WORKER_FILE = "contracts/v6_task_campaign_speed_worker_v10.json"
SCHEMA = "task-worker-revision-v10"
SCOPE = "constant_stationary_low_high_sampling_only_preserve_all_129_groups"
POOL_FRACTIONS = dict(stand_anchors=.12, stand_continuous=.08, height=.05,
    basic_translation=.15, basic_yaw=.10, basic_transitions=.15,
    high_motion=.25, high_transitions=.10)


def read_bound(name, sha, root=ROOT):
    raw = (root / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"Mixed locomotion frozen source SHA differs: {name}")
    return json.loads(raw)


def sampling_patch(root=ROOT):
    """Classify by frozen flat membership, never by a velocity-name heuristic."""
    flat = read_bound(FLAT_FILE, FLAT_SHA, root)
    speed = read_bound(SPEED_FILE, SPEED_SHA, root)
    basic = flat["behavior_pool_membership"]
    aliases = dict(translation="basic_translation", yaw="basic_yaw", transitions="basic_transitions")
    membership = {name: aliases.get(basic[name], basic[name]) if name in basic
        else ("high_transitions" if pool == "transitions" else "high_motion")
        for name, pool in speed["behavior_pool_membership"].items()}
    if (len(membership) != 129 or set(basic) - set(membership)
            or set(membership.values()) != set(POOL_FRACTIONS)):
        raise ValueError("Mixed locomotion requires the complete frozen 41/129 group coverage")
    sizes = Counter(membership.values())
    groups = deepcopy(speed["scene_groups"])
    for group in groups:
        pool = membership[group["name"]]
        group["fraction"] = POOL_FRACTIONS[pool] / sizes[pool]
    if not math.isclose(math.fsum(g["fraction"] for g in groups), 1., abs_tol=1e-12):
        raise ValueError("Mixed locomotion pools must sum to one")
    by_name = {group["name"]: group for group in groups}
    for group in groups:
        name = group["name"]
        for partner in (name + "__height_train", name + "__reverse_train",
                        name.replace("forward_", "backward_", 1)):
            if partner != name and partner in by_name:
                other = by_name[partner]
                if other["fraction"] != group["fraction"]:
                    raise ValueError("Mixed locomotion directional/height allocations are asymmetric")
    return dict(scene_groups=groups, behavior_pool_membership=membership,
        behavior_pool_fractions=deepcopy(POOL_FRACTIONS)), dict(
        schema="mixed-locomotion-sampling-v1", scope=SCOPE,
        policy_scope="one_policy_stationary_low_high", allocation_clock="constant_all_actor_updates",
        flat_membership_file=FLAT_FILE, flat_membership_sha256=FLAT_SHA,
        group_counts=dict(sizes), integer_allocation_population=4096,
        integer_allocation="existing_choose_scene_groups_largest_remainders_minimum_one")


def apply_mixed_locomotion_revision(config, root=ROOT):
    if config.get("task_campaign", {}).get("domain") != "speed":
        raise ValueError("Mixed locomotion retains the existing speed account and actor clock")
    patch, metadata = sampling_patch(root)
    if config.get("evaluation_exact_cases"):
        patch.pop("scene_groups")
    config.update(patch)
    config["task_campaign"].update(theme="mixed_locomotion", sampling_revision=metadata)


def validate_integer_allocation(config):
    """Freeze/test only; pure migration authentication has no torch dependency."""
    from .task import choose_scene_groups
    groups = config["scene_groups"]
    counts = Counter(name for name, _ in choose_scene_groups(groups, 4096))
    names = {group["name"] for group in groups}
    if sum(counts.values()) != 4096 or set(counts) != names or len(names) != 129:
        raise ValueError("Mixed locomotion integer allocation lost an environment or behavior")
    for name in names:
        for partner in (name + "__height_train", name + "__reverse_train",
                        name.replace("forward_", "backward_", 1)):
            if partner != name and partner in names and abs(counts[name] - counts[partner]) > 1:
                raise ValueError("Mixed locomotion integer allocation violates paired floor/ceil quotas")
    membership = config["behavior_pool_membership"]
    return dict(num_envs=4096, groups=len(names), group_environment_counts=dict(counts),
        pool_environment_counts={pool: sum(count for name, count in counts.items()
            if membership[name] == pool) for pool in POOL_FRACTIONS})


def revised_design(root=ROOT):
    """Apply the same declared rule separately to the five independent role inputs."""
    source = read_bound(SPEED_FILE, SPEED_SHA, root)
    design = read_bound(source["design_preflight"]["file"], source["design_preflight"]["sha256"], root)
    design["reviewed_change"] = SCOPE
    design["inherited_mixed_sampling_design_sha256"] = source["design_preflight"]["sha256"]
    binding = dict(schema=SCHEMA, source_contract_file=SPEED_FILE, source_contract_sha256=SPEED_SHA)
    for role in design["stages"]["speed"]["roles"].values():
        apply_mixed_locomotion_revision(role["contract"], root)
        role["contract"]["task_campaign"]["revision_migration"] = deepcopy(binding)
    return design


def plan_sampling_pools(root=ROOT):
    patch, _ = sampling_patch(root)
    return [dict(name=pool, fraction=fraction, skills=[g["name"] for g in patch["scene_groups"]
        if patch["behavior_pool_membership"][g["name"]] == pool], cross_height=False)
        for pool, fraction in POOL_FRACTIONS.items()]
