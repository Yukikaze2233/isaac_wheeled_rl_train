#!/usr/bin/env python3
"""Freeze recovery teaching and independent expectations from reviewed step rows.

This generator never calls the executable stage or evaluation materializers.
All unchanged expectations are inherited from the frozen, reviewed step design;
the new task rows, pool deductions and case suffix are constructed explicitly.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
REVISION = "v6_recovery_curriculum_v1"
CLAIM = "scripted_physical_recovery_then_RL_handoff_separate_acceptance"
PROFILE_FILE = "contracts/v6_recovery_profiles_v1.json"
POSES = (
    "upright", "crouched", "upright_release", "crouched_release",
    "left_side_90", "right_side_90", "pitch_forward_45", "pitch_backward_45",
    "front_down_90", "back_down_90", "front_down_left_forward_right_back_90",
    "front_down_left_back_right_forward_90", "back_down_left_forward_right_back_90",
    "back_down_left_back_right_forward_90", "prone_belly_legs_above", "prone_back_legs_above",
)
STEP_POSES = ("step200_wheels_split", "step200_shell_high_wheels_low", "step200_lip_near_shell",
              "step200_upright_top", "step200_prone_belly_top")
DOUBLE_POSES = ("double_step_first_riser_split", "double_step_second_riser_split",
                "double_step_first_prone_top", "double_step_second_prone_top", "double_step_second_side_top")
STAGES = ("S1", "S2", "S3", "S4", "S5", "S5T", "S5D", "S6")
# Preserve the standing, height, motion and existing push/landing rehearsal pools.
POOL_FRACTIONS = {
    "S3": {"terrain": .30}, "S4": {"steps": .45}, "S5": {"jumps": .45},
    "S5T": {"step_jumps": .30},
    "S5D": {"double_step_jumps": .125, "double_body_climb": .125},
    "S6": {"steps": .15},
}


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def write(name, value):
    path = ROOT / name
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return digest(name)


def recipes(profiles):
    result = []
    for pose in POSES:
        for followup, command in (("stand", [0., 0., .305]), ("forward", [.3, 0., .305])):
            result.append({"name": f"recovery_{pose}_{followup}", "skill": "recovery_handoff",
                "introduced_stage": "S3", "recovery_pose": pose,
                "command": command, "recovery_followup_command": list(command), "episode_seconds": 20.})
    for pose, suffix, yaw in (("upright_release", "turn_left", .6), ("crouched_release", "turn_right", -.6)):
        command = [0., yaw, .305]
        result.append({"name": f"recovery_{pose}_{suffix}", "skill": "recovery_handoff",
            "introduced_stage": "S3", "recovery_pose": pose,
            "command": command, "recovery_followup_command": list(command), "episode_seconds": 20.})
    for poses, stage in ((STEP_POSES, "S4"), (DOUBLE_POSES, "S5D")):
        for pose in poses:
            profile = profiles["profiles"][pose]
            terrain = profile["terrain"]
            limits = ({"step_up_m": .2} if terrain == "step_up" else
                      {"stair_heights_m": list(profile["stair_heights_m"]),
                       "stair_tread_m": list(profile["stair_tread_lengths_m"])})
            for followup, command in (("stand", [0., 0., .305]), ("forward", [.3, 0., .305])):
                result.append({"name": f"recovery_{pose}_{followup}", "skill": "recovery_handoff",
                    "introduced_stage": stage, "recovery_pose": pose, "terrain": terrain, "terrain_limits": limits,
                    "required_platform_height_m": profile["required_platform_height_m"],
                    "min_ending_x": profile["min_ending_x"], "command": command,
                    "recovery_followup_command": list(command), "episode_seconds": 20.})
    return result


def explicit_recovery_row(item, limits):
    """The independently reviewed recipe-to-case mapping, not skill_spec()."""
    bare = {"kind": "recovery_handoff", "command": list(item["command"]), "mode": 0,
        "sample_amplitude": False, "recovery_pose": item["recovery_pose"],
        "recovery_followup_command": list(item["recovery_followup_command"])}
    outcome = {key: item[key] for key in ("required_platform_height_m", "min_ending_x") if key in item}
    bare.update(outcome)
    limits = {**deepcopy(limits), **deepcopy(item.get("terrain_limits", {}))}
    train = {**deepcopy(bare), "terrain_limits": limits, "episode_seconds": 20.,
        "sample_yaw_sign": False, "push_m_s": 0.}
    case = {"name": item["name"], "command": list(item["command"]), "terrain": item.get("terrain", "flat"),
        "task": "survive", "skill": bare, "terrain_limits": deepcopy(limits),
        "terrain_seed": zlib.crc32(item["name"].encode()), "episode_seconds": 20.,
        "height_mae_m_max": .01, "velocity_mae_m_s_max": .1, "yaw_mae_rad_s_max": .15,
        "recovery_case": True, "introduced_stage": item["introduced_stage"], "success_rate_min": .95,
        "survival_rate_min": .95, "anchor": False, "reset_seed_key": item["name"], **outcome}
    return train, case


def build():
    previous = read("contracts/v6_step_curriculum_training_v1.json")
    design = read(previous["design_preflight"]["file"])
    profiles = read(PROFILE_FILE)
    for name in (*POSES, *STEP_POSES, *DOUBLE_POSES):
        profile = profiles["profiles"][name]
        terrain = "flat" if name in POSES else "step_up" if name in STEP_POSES else "stairs"
        if profile.get("enabled") is not True or profile.get("geometry_valid") is not True or profile.get("terrain") != terrain:
            raise ValueError(f"Recovery course needs frozen enabled initial geometry with matching terrain: {name}")
    plan = deepcopy(previous)
    plan.pop("design_preflight")
    plan["curriculum_revision"] = REVISION
    plan["runtime_base"]["response_migration"] = REVISION
    sensor = plan["runtime_base"]["step_jump_task"]
    settings = {"enabled": True, "profiles_file": PROFILE_FILE, "profiles_sha256": digest(PROFILE_FILE),
        "height_lookup_file": sensor["height_lookup_file"], "height_lookup_sha256": sensor["height_lookup_sha256"],
        "reference_dt": .005, "blend_seconds": .2, "stable_seconds": 1., "max_script_seconds": 8.,
        "reroute_tilt_rad": 1.2217304763960306, "reroute_support_lost_seconds": .15, "max_reroutes": 1,
        "learning_mask": "pure_rl_intervals_only", "reset_mode": "initial_state_only",
        "controller": "encoder_imu_recovery_fsm", "action_history": "effective_reference",
        "wheel_balance": {"enabled": True, "pitch_kp_nm_rad": 8., "pitch_kd_nm_s_rad": 1.5,
            "wheel_damping_nm_s_rad": .2, "prepare_tilt_limit_deg": 55., "capture_tilt_limit_deg": 85.}}
    plan["runtime_base"]["recovery_training"] = settings
    plan["release_scope"]["R0"] = CLAIM
    additions = recipes(profiles)
    plan["skill_catalog"].extend(deepcopy(additions))
    for stage in plan["stages"]:
        sid = stage["name"]
        if sid not in POOL_FRACTIONS:
            continue
        for pool in stage["sampling_pools"]:
            if pool["name"] in POOL_FRACTIONS[sid]:
                pool["fraction"] = POOL_FRACTIONS[sid][pool["name"]]
        stage["sampling_pools"].append({"name": "scripted_recovery", "fraction": .05,
            "skills": [item["name"] for item in additions
                       if STAGES.index(item["introduced_stage"]) <= STAGES.index(sid)]})
    budget = read(previous["budget_spec_file"])
    budget["budget_id"] = "v6-recovery-curriculum-budget-v1"
    budget["release_requirements"].append(
        "Require the scripted physical recovery plus pure-RL handoff cases separately; no actor-only self-righting credit. "
        "Script/blend samples are charged physically but excluded from native PPO losses and GAE.")
    plan["budget_spec_file"] = "contracts/v6_recovery_curriculum_budget_v1.json"
    plan["budget_spec_sha256"] = write(plan["budget_spec_file"], budget)
    specs, cases = {}, []
    for item in additions:
        spec, case = explicit_recovery_row(item, plan["runtime_base"]["terrain_limits"])
        specs[item["name"]] = spec
        cases.append(case)
    names = [case["name"] for case in cases]
    for stage in plan["stages"]:
        sid = stage["name"]
        active = sid in POOL_FRACTIONS
        active_names = [item["name"] for item in additions if STAGES.index(item["introduced_stage"]) <= STAGES.index(sid)]
        for role, row in design["stages"][sid]["roles"].items():
            config = row["contract"]
            config.update(response_migration=REVISION, recovery_training=deepcopy(settings),
                budget_spec_file=plan["budget_spec_file"], budget_spec_sha256=plan["budget_spec_sha256"],
                release_scope=deepcopy(plan["release_scope"]))
            row["runtime"]["modules"]["recovery_training"] = True
            if active:
                config["new_asset_curriculum"]["edge_reset_fraction"] = .05
                # Derive only the explicitly changed pool weights from frozen
                # prior rows; preserve exact floats for all unchanged groups.
                for group in config["scene_groups"] if role == "train" else []:
                    pool = config["behavior_pool_membership"][group["name"]]
                    if pool in POOL_FRACTIONS[sid]:
                        # Native materialization divides equal per-skill shares.
                        siblings = sum(config["behavior_pool_membership"][g["name"]] == pool
                                       for g in config["scene_groups"])
                        group["fraction"] = POOL_FRACTIONS[sid][pool] / siblings
                config["skill_specs"].update({name: deepcopy(specs[name]) for name in active_names})
                config["behavior_pool_membership"].update({name: "scripted_recovery" for name in active_names})
                config["behavior_pool_fractions"].update(POOL_FRACTIONS[sid], scripted_recovery=.05)
                if role == "train":
                    terrain_by_name = {item["name"]: item.get("terrain", "flat") for item in additions}
                    config["scene_groups"].extend({"name": name, "fraction": .05 / len(active_names),
                        "terrain": [terrain_by_name[name]]} for name in active_names)
                    config["episode_seconds"] = max(config["episode_seconds"], 20.)
            ev = config["evaluation"]
            ev["cases"].extend(deepcopy(cases))
            for key in ("canonical_case_names", "quick_case_names", "retention_case_names"):
                ev[key].extend(names)
            if active:
                ev["promotion_case_names"].extend(active_names)
            if role != "train":
                count = len(ev["cases"])
                row["invocation"]["num_envs"] = count * ev["episodes_per_case"]
                config["scene_groups"] = [{"name": case["name"], "fraction": 1. / count,
                                          "terrain": [case["terrain"]]} for case in ev["cases"]]
    design["review_basis"] += (
        " Recovery revision: keep all 268 predecessor cases and S1/S2 pools unchanged; append 54 named "
        "physical-script/RL-handoff cases: 34 flat from S3, ten single-platform from S4 and ten double-platform from S5D. "
        "S3+ takes five percentage points from explicit primary task pools; "
        "standing/motion/height/previous push-recovery reserves remain. Edge reset probability independently becomes .05. "
        "The native PPO adapter excludes every script/blend sample and cuts GAE at handoff boundaries.")
    design["reviewed_plan_payload_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    name = "contracts/v6_recovery_curriculum_expectations_v1.json"
    plan["design_preflight"] = {**previous["design_preflight"], "file": name, "sha256": write(name, design)}
    write("contracts/v6_recovery_curriculum_training_v1.json", plan)
    print("Frozen recovery candidate: 54 appended cases, unchanged 8500-update/20000-environment budget. Run all 40 roles.")


if __name__ == "__main__":
    build()
