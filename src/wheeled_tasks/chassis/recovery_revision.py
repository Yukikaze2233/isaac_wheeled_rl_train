"""Exact reviewed takeover patch for the named task-campaign v8 migration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .rewards import RECOVERY_TAKEOVER_V1

ROOT = Path(__file__).resolve().parents[3]
FLAT_FILE = "contracts/v6_task_campaign_flat_worker_v6.json"
FLAT_SHA = "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7"
POSE_FILE = "contracts/v6_recovery_profiles_v2.json"
POSE_SHA = "007363fdfe35804b01ad616a1453bf163bf70c1bb7dd59b6a8f743b26c0b4441"
RECOVERY_FILE = "contracts/v6_task_campaign_recovery_worker_v6.json"
RECOVERY_SHA = "e2a7ecaec1985adac34c280a71b1359a6446fc12c38d9aaa6fb98ac769e4d74c"


def read_bound(name, sha):
    raw = (ROOT / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"Recovery revision frozen source SHA differs: {name}")
    return json.loads(raw)


def pose_curriculum(origin):
    if type(origin) is not int or origin < 0:
        raise ValueError("Recovery teaching origin must retain an absolute domain actor clock")
    return dict(schema="v6-recovery-pose-curriculum-v1", enabled=True,
        clock="domain_actor_updates", origin_actor_update=origin,
        sampling="episode_reset_only", interpolation="linear",
        keyframes=[dict(actor_update=u, probabilities=dict(zip(("easy", "medium", "hard"), p)))
            for u, p in ((0, (.9, .1, 0.)), (1000, (.5, .35, .15)),
                         (3000, (.25, .35, .4)), (6000, (.2, .2, .6)))],
        profile_sets={
            "flat": dict(terrain="flat", required_platform_height_m=0.,
                easy=["upright", "crouched"], medium=["upright_release", "crouched_release",
                    "pitch_forward_45", "pitch_backward_45"]),
            "step200": dict(terrain="step_up", required_platform_height_m=.2,
                easy=["step200_upright_top"], medium=["step200_upright_top"]),
            **{f"stairs_{level}": dict(terrain="stairs", required_platform_height_m=height,
                easy=[f"double_step_{level}_upright_top"],
                medium=[f"double_step_{level}_upright_top"])
                for level, height in (("first", .2), ("second", .45))}})


def apply_recovery_takeover_revision(config, origin_actor_update):
    """Change only recovery sampling, takeover densities and the declared blend fix."""
    read_bound(POSE_FILE, POSE_SHA)
    flat = read_bound(FLAT_FILE, FLAT_SHA)
    original = read_bound(RECOVERY_FILE, RECOVERY_SHA)
    recover = [deepcopy(row) for row in original["scene_groups"]
        if original["skill_specs"][row["name"]].get("recovery_pose")]
    if len(recover) != 54:
        raise ValueError("Recovery revision requires all 54 frozen followup lanes")
    for row in recover:
        row["fraction"] = .65 / len(recover)
    retained = deepcopy(flat["scene_groups"])
    for row in retained:
        row["fraction"] *= .35
    if not config.get("evaluation_exact_cases"):
        config["scene_groups"] = recover + retained
    config["skill_specs"] = {row["name"]: deepcopy(original["skill_specs"][row["name"]]) for row in recover}
    config["skill_specs"].update(deepcopy(flat["skill_specs"]))
    config["behavior_pool_membership"] = {row["name"]: "scripted_recovery" for row in recover}
    config["behavior_pool_membership"].update(deepcopy(flat["behavior_pool_membership"]))
    config["behavior_pool_fractions"] = {"scripted_recovery": .65,
        **{name: .35 * fraction for name, fraction in flat["behavior_pool_fractions"].items()}}
    config["recovery_training"].update(profiles_file=POSE_FILE, profiles_sha256=POSE_SHA,
        complete_started_blend=True, pose_curriculum=pose_curriculum(origin_actor_update))
    config["recovery_takeover"] = deepcopy(RECOVERY_TAKEOVER_V1)
    config["resume_critic_warmup_updates"] = 10
