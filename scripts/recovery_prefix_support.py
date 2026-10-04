"""CPU-only authentication and first-cohort accounting for the prefix probe."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess


WORKER_SHA = "f6aa77c6808c9746dc8dee431e8b3331b6b0f883db701335e57069e7df738ee8"
PLAN_FILE = "contracts/v6_task_campaign_v8.json"
PLAN_SHA = "c77926685265ebc7700820abdb2230bfb8fb4d9ec8c6921c6e77187e29d97abb"
CORE_REVISION = "b4ce80964d47c0127e0b3d9929ad35a620bbf204"
BASELINE_PT_SHA = "7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f"
BASELINE_CONTRACT_SHA = "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7"
CASES = (
    "recovery_prone_belly_legs_above_stand", "recovery_prone_back_legs_above_stand",
    "recovery_left_side_90_stand", "recovery_right_side_90_stand",
)
ABI_KEYS = (
    "asset_manifest_sha256", "control_math_sha256", "physics_dt", "policy_dt",
    "actor_dim", "critic_dim", "action_dim", "actor_observation_source",
    "policy_action_order", "policy_architecture", "v5_control",
)
CORE_FILES = (
    "src/wheeled_tasks/chassis/env.py", "src/wheeled_tasks/chassis/eval_env.py",
    "src/wheeled_tasks/chassis/recovery.py", "src/wheeled_tasks/chassis/recovery_training.py",
    "src/wheeled_tasks/chassis/continuous_encoders.py", "src/wheeled_tasks/chassis/recovery_observer.py",
    "src/wheeled_tasks/chassis/control_contract.py", "src/wheeled_tasks/v40/core.py",
    "src/wheeled_tasks/chassis/v5_control.py", "src/wheeled_tasks/chassis/mechanism_observer.py",
    "src/wheeled_tasks/chassis/actuator_response.py", "src/wheeled_tasks/chassis/command_transport.py",
)
PARTITIONS = ("success", "failure", "timeout", "boundary_truncation", "right_censored_unfinished")
MILESTONES = (
    "birth_complete", "passive_release_finished", "phase_CAPTURE", "phase_PREPARE", "phase_BLEND",
    "blend_entry_pre_effort", "blend_first_applied_post_physics", "blend_last_mixed_post_physics",
    "first_pure_rl_post_physics", "sensor_hold_dwell_complete", "truth_hold_dwell_complete",
    "motion_unlock", "followup_success", "terminal_pre_autoreset",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON literal: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def core_sources(root):
    """Bind unchanged physical/control code to b4; record preflight separately."""
    result = {}
    for name in CORE_FILES:
        frozen = subprocess.run(["git", "show", f"{CORE_REVISION}:{name}"], cwd=root,
                                check=True, capture_output=True).stdout
        expected = hashlib.sha256(frozen).hexdigest()
        require(digest(Path(root) / name) == expected, f"Physical/control source differs from b4: {name}")
        result[name] = expected
    return {"revision": CORE_REVISION, "core_files_sha256": result,
            "actual_preflight_sha256": digest(Path(root) / "src/wheeled_tasks/chassis/preflight.py"),
            "preflight_note": "Actual module is recorded; existing user edits are neither overwritten nor claimed as b4."}


def project_role(source, independent_design, repeats=1):
    """Project authenticated runtime and design inputs independently."""
    from probe_v6_recovery_runtime import project_evaluation, project_scenes
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract

    require(type(repeats) is int and repeats in (1, 4), "Only one or four replicas per case are supported")
    binding = source["design_preflight"]
    stage = binding["stage"]
    row = independent_design["stages"][stage]["roles"]["full"]
    config = fixed_suite_contract(source)
    expected = deepcopy(row)
    for mapping in (config, expected["contract"]):
        mapping["evaluation"] = project_evaluation(mapping["evaluation"], CASES, repeats)
        mapping["scene_groups"] = project_scenes(mapping["scene_groups"], CASES)
    expected["invocation"]["num_envs"] = len(CASES) * repeats
    projected = {k: deepcopy(v) for k, v in independent_design.items() if k != "stages"}
    projected["stages"] = {stage: {"roles": {"full": expected}}}
    projected["engineering_projection"] = {
        "scope": "physical_prefix_diagnostic_no_PPO_no_qualification", "formal_updates_consumed": 0,
        "source_design_file": binding["file"], "source_design_sha256": binding["sha256"],
        "source_role": "full", "selected_case_names": list(CASES), "episodes_per_case": repeats,
        "allowed_changes": ["evaluation.cases", "evaluation.*_case_names", "evaluation.episodes_per_case",
                            "scene_groups", "invocation.num_envs"],
    }
    return config, projected


def materialize(root, contract_path, output, repeats=1):
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
    from wheeled_tasks.chassis.preflight import design_preflight

    root, output = Path(root), Path(output)
    require(digest(contract_path) == WORKER_SHA, "Explicit frozen V8 recovery worker SHA required")
    source, manifest = preflight(Path(contract_path))
    require(tuple(source[k] for k in ("physics_dt", "policy_dt", "actor_dim", "critic_dim", "action_dim"))
            == (.005, .02, 35, 81, 6), "200/50 Hz and 35/81/6 ABI required")
    require(source["recovery_training"]["reset_mode"] == "initial_state_only", "Birth-only recovery required")
    binding = source["design_preflight"]
    path = root / binding["file"]
    require(digest(path) == binding["sha256"], "Independent V8 design SHA changed")
    independent = read_json(path)
    row = independent["stages"][binding["stage"]]["roles"]["full"]
    rule = row["invocation"]["seed"]
    require(rule.get("rule") == "fixed", "Explicit full-role fixed seed required")
    seed = rule["value"]
    stage = source["enabled_stages"][0]
    require(len(source["enabled_stages"]) == 1, "One native environment stage required")
    control = load_chassis_control(root / source["control_math_source"])
    # Same full-role selection as run_task_campaign.validate_contracts, taken
    # from a second authenticated input rather than inferred from the design.
    require(digest(root / PLAN_FILE) == PLAN_SHA, "Frozen V8 campaign SHA changed")
    plan = read_json(root / PLAN_FILE)
    domain = next(r for r in plan["domains"] if r["name"] == "recovery")
    require(domain["sha256"] == WORKER_SHA, "Campaign does not bind the V8 worker")
    names = sorted(set(domain["required_case_names"]) | set(domain.get("monitor_case_names", [])))
    role_source = deepcopy(source)
    role_source["evaluation"]["seed"] = seed
    role_source["evaluation"] = select_evaluation_cases(source["evaluation"], names)
    original = fixed_suite_contract(role_source)
    original_receipt = design_preflight(original, control, root, stage=stage, role="full", seed=seed,
                                       num_envs=row["invocation"]["num_envs"])
    config, projection = project_role(original, independent, repeats)
    projection["engineering_projection"].update(source_campaign_file=PLAN_FILE, source_campaign_sha256=PLAN_SHA)
    by_name = {c["name"]: c for c in config["evaluation"]["cases"]}
    require(set(by_name) == set(CASES), "Four frozen fallen cases required")
    for name in CASES:
        c = by_name[name]
        require(c["terrain"] == "flat" and c["episode_seconds"] == 20.
                and c["command"] == [0., 0., .305], f"Frozen first-cohort case changed: {name}")
    require(config["signal_delay"]["enabled"] is False
            and config["command_transport"]["enabled"] is True, "Full-role delay enablement changed")
    design_path = output / "engineering_expectations.json"
    write_json(design_path, projection)
    config["design_preflight"] = {**binding, "file": str(design_path.resolve()), "sha256": digest(design_path)}
    effective_path = output / "effective_contract.json"
    write_json(effective_path, config)
    checked, checked_manifest = preflight(effective_path)
    require(checked_manifest == manifest, "Projection changed the asset")
    count = len(CASES) * repeats
    receipt = design_preflight(checked, control, root, stage=stage, role="full", seed=seed, num_envs=count)
    receipt["original_full_role_verified"] = {k: original_receipt[k] for k in ("status", "design_sha256", "num_envs")}
    receipt["engineering_projection"] = projection["engineering_projection"]
    return checked, manifest, control, stage, seed, count, receipt


def validate_checkpoint_payload(saved, source, target, source_sha, seal_updates):
    require(isinstance(saved, dict) and all(k in saved for k in
            ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "infos")),
            "Full learning checkpoint required")
    infos = saved["infos"]
    require(isinstance(infos, dict) and infos.get("contract_sha256") == source_sha,
            "PT infos/source contract SHA mismatch")
    for key in ABI_KEYS:
        require(key in source and key in target and source[key] == target[key], f"Actor ABI changes {key}")
    for key in ("asset_manifest_sha256", "control_math_sha256"):
        require(infos.get(key) == target[key], f"PT infos changes {key}")
    require(type(seal_updates) is int and seal_updates >= 0
            and infos.get("successful_updates_total") == seal_updates, "PT infos/seal learning update mismatch")
    for name in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        require(isinstance(saved[name], dict) and bool(saved[name]), f"Empty checkpoint state: {name}")
    return {k: infos.get(k) for k in ("successful_updates_total", "learning_lineage_updates",
            "task_actor_update", "stage_actor_update", "global_actor_update", "actor_updates_consumed")}


class StateWriteGuard:
    """Birth is a transaction; a terminated first cohort is never reopened."""
    def __init__(self, count):
        require(type(count) is int and count > 0, "Positive cohort count required")
        self.count = count
        self.birth_open = True
        self.live = set(range(count))
        self.calls = Counter()

    def write(self, method, ids):
        ids = list(ids)
        require(all(type(i) is int and 0 <= i < self.count for i in ids), "Invalid state-write lane IDs")
        if not self.birth_open and self.live.intersection(ids):
            raise ValueError(f"State write after birth to live cohort: {method} {ids}")
        self.calls[(method, "birth" if self.birth_open else "closed_cohort")] += 1

    def close_birth(self):
        require(self.birth_open, "Birth already closed")
        self.birth_open = False

    def close_terminal(self, ids):
        ids = list(ids)
        require(not self.birth_open, "Terminal closure during birth")
        require(len(ids) == len(set(ids)), "Duplicate terminal lane IDs")
        require(set(ids).issubset(self.live), "Duplicate terminal closure")
        self.live.difference_update(ids)


class CohortLedger:
    def __init__(self, case_names, seed):
        require(case_names and set(case_names) == set(CASES), "Fixed cohort case coverage changed")
        counts = Counter(case_names)
        require(len(set(counts.values())) == 1, "Equal case replicas required")
        replicas = Counter()
        self.rows = []
        for lane, name in enumerate(case_names):
            replica = replicas[name]
            replicas[name] += 1
            self.rows.append({"lane": lane, "case": name, "replica": replica, "seed": seed,
                "birth_id": 1, "events": {}, "partition": None, "terminal": None})

    def event(self, lane, name, time_s):
        require(type(lane) is int and 0 <= lane < len(self.rows), "Invalid cohort lane")
        require(math.isfinite(time_s) and time_s >= 0., "Invalid event clock")
        row = self.rows[lane]
        require(row["partition"] is None, "Event after first cohort closed")
        row["events"].setdefault(name, time_s)

    def finish(self, lane, *, success, terminated, boundary, timeout, time_s, reasons):
        require(type(lane) is int and 0 <= lane < len(self.rows), "Invalid terminal lane")
        require(math.isfinite(time_s) and time_s >= 0., "Invalid terminal clock")
        row = self.rows[lane]
        require(row["partition"] is None, "Duplicate first-cohort terminal")
        # Explicit precedence; overlapping native reason flags remain recorded.
        partition = ("failure" if terminated else "success" if success else
                     "boundary_truncation" if boundary else "timeout" if timeout else None)
        require(partition is not None, "Reset lane was not a native terminal")
        row["partition"] = partition
        row["terminal"] = {"time_s": time_s, "success": bool(success), "terminated": bool(terminated),
                           "boundary": bool(boundary), "timeout": bool(timeout), "reasons": dict(reasons)}

    def summary(self, *, censor_time_s=0.):
        require(math.isfinite(censor_time_s) and censor_time_s >= 0., "Invalid censor clock")
        rows = deepcopy(self.rows)
        for row in rows:
            if row["partition"] is None:
                row["partition"] = "right_censored_unfinished"
                row["censor_time_s"] = censor_time_s
        count = len(rows)
        partitions = {name: sum(r["partition"] == name for r in rows) for name in PARTITIONS}
        require(sum(partitions.values()) == count, "Cohort accounting does not conserve births")
        names = sorted(set(MILESTONES).union(*(r["events"] for r in rows)))
        rates = {name: {"numerator": sum(name in r["events"] for r in rows), "denominator": count}
                 for name in names}
        for value in rates.values():
            value["fraction"] = value["numerator"] / count
        return {"fixed_birth_denominator": count, "partitions": partitions,
                "success_per_seeded_cohort": partitions["success"] / count, "events_per_seeded_cohort": rates,
                "rows": rows, "qualification": False, "physx_internal_state_restore_claim": False}
