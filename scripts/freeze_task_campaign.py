#!/usr/bin/env python3
"""Freeze independent task domains from the reviewed complete V6 skill catalogue.

Workers reuse the complete native stage template. Expectations are amended from
the separately frozen S6 role table, never inferred from a generated worker.
"""
from copy import deepcopy
import math

from freeze_s1_remediation import ROOT, read, sha, write
from wheeled_tasks.chassis.full_curriculum import stage_contract
from wheeled_tasks.chassis.task_campaign import CONTINUOUS_MODE, jump_probability_schedule

CAMPAIGN_ID = "v6-task-domains-20261001"
PLAN_FILE = "contracts/v6_task_campaign_v1.json"
BUDGET_FILE = "contracts/v6_task_campaign_budget_v1.json"
SOURCE_PLAN = "contracts/v6_recovery_curriculum_training_v1.json"
SOURCE_A = "contracts/v6_s1_A_fixed_v1.json"
DOMAIN_ORDER = ("flat", "recovery", "speed", "terrain", "jump", "stepjump")
PARENTS = dict(flat="A", recovery="flat", speed="flat", terrain="flat", jump="flat", stepjump="jump")
KINDS = dict(flat="foundation", recovery="mixed", speed="speed", terrain="terrain", jump="jump", stepjump="mixed")
HEIGHTS = (.23, .25, .28, .305, .32, .35, .38, .4, .43)
STAND = [f"stand_{round(h * 1000)}mm" for h in HEIGHTS]
BASIC = ["stand_305mm", "forward_05", "backward_05", "start_stop_05", "rotate_1", "rotate_1_reverse"]


def pools(catalog):
    """Named terrain/command assignments; a task domain is larger than one case."""
    def names(*kinds):
        return [n for n, item in catalog.items() if item["skill"] in kinds]

    def pool(name, fraction, skills, cross_height=False):
        return dict(name=name, fraction=fraction, skills=skills, cross_height=cross_height)

    flat_motion = ["forward_05", "backward_05", "forward_1", "backward_1",
                   "forward_continuous_low", "backward_continuous_low", "curve_low"]
    flat_stop = ["start_stop_05", "start_stop_1", "velocity_curve_1"]
    flat_yaw = ["rotate_1", "yaw_continuous_low"]
    retention = [pool("stand_anchors", .09, STAND), pool("stand_continuous", .06, ["stand_continuous"]),
                 pool("entry_exit", .05, BASIC[1:], True)]
    # The explicit reverse case has a separate training spec suffix.
    retention[-1]["skills"] = ["forward_05", "backward_05", "start_stop_05", "rotate_1"]
    return {
        "flat": [pool("stand_anchors", .18, STAND), pool("stand_continuous", .07, ["stand_continuous"]),
                 pool("height", .15, ["height_scan"]), pool("translation", .25, flat_motion, True),
                 pool("transitions", .20, flat_stop, True), pool("yaw", .15, flat_yaw, True)],
        "recovery": [pool("scripted_recovery", .65, names("recovery_handoff")),
                     pool("push_air_landing", .15, names("push_recovery", "airborne", "landing")), *retention],
        "speed": [pool("motion", .60, [n for n in names("forward", "backward", "rotate", "curve", "spin_translate")
                                       if n != "material_transfer"], True),
                  pool("transitions", .20, names("start_stop", "velocity_curve"), True),
                  pool("stand_anchors", .09, STAND), pool("stand_continuous", .06, ["stand_continuous"]),
                  pool("height", .05, ["height_scan"])],
        "terrain": [pool("untriggered_steps", .60, names("rolling_up", "rolling_stairs", "body_climb", "step_down", "stairs_down")),
                    pool("surface_traversal", .20, ["slope_up", "slope_down", "cross_slope", "rough", "material_transfer"]), *retention],
        "jump": [pool("flat_jumps", .65, [n for n in names("jump", "running_jump") if not catalog[n].get("request_events")
                                          and not catalog[n].get("command_sequence")]),
                 pool("request_sequences", .15, [n for n in names("jump") if catalog[n].get("request_events")
                                                  or catalog[n].get("command_sequence")]), *retention],
        "stepjump": [pool("single_step_jump", .50, names("step_jump")),
                     pool("double_step_jump", .25, names("double_step_jump")),
                     pool("flat_jump_rehearsal", .05, ["jump_10", "running_jump_10"]), *retention],
    }


def case_owner(case, catalog):
    name = case["name"]
    if name.startswith("heldout_"):
        name = case["reset_seed_key"]
    possible = [n for n in catalog if name == n or name.startswith(n + "_")]
    if not possible:
        raise ValueError(f"Case without a reviewed skill owner: {case['name']}")
    return max(possible, key=len)


def case_scopes(cases, catalog, domain_pools):
    scopes = {}
    for domain in DOMAIN_ORDER:
        main = {n for p in domain_pools[domain] for n in p["skills"]}
        if domain != "flat":
            main.difference_update(STAND)
            main.discard("stand_continuous")
            main.difference_update(BASIC)
        if domain == "speed":
            # The complete motion domain also owns its low-speed anchors.
            main.update(n for n, v in catalog.items() if v["skill"] in
                        ("forward", "backward", "rotate", "curve", "spin_translate", "start_stop", "velocity_curve")
                        and n != "material_transfer")
        required = [c["name"] for c in cases if case_owner(c, catalog) in main or c["name"] in BASIC]
        scopes[domain] = required
    return scopes


def distribution(source_specs, source_groups, domain_pools, height_sampling):
    specs = deepcopy(source_specs)
    terrains = {g["name"]: g["terrain"] for g in source_groups}
    for height, name in zip(HEIGHTS, STAND):
        specs[name] = deepcopy(specs["stand_continuous"])
        specs[name].pop("height_sampling")
        specs[name]["command"][2] = height
        terrains[name] = ["flat"]
    active, groups, membership = {}, [], {}
    for pool in domain_pools:
        selected = []
        for name in pool["skills"]:
            selected.append(name)
            reverse = name + "__reverse_train"
            if reverse in specs:
                selected.append(reverse)
        for name in selected:
            if name in active:
                raise ValueError(f"Duplicate task training allocation: {name}")
            variants = [(name, 1.)]
            if pool["cross_height"]:
                variants = [(name, .5), (name + "__height_train", .5)]
            for variant, share in variants:
                spec = deepcopy(specs[name])
                if "command_resampling_seconds" in spec:
                    spec["command_resampling_seconds"] = [1., 8.]
                if variant.endswith("__height_train"):
                    spec.update(height_sampling=deepcopy(height_sampling), height_transition_seconds=4.)
                active[variant] = spec
                membership[variant] = pool["name"]
                groups.append(dict(name=variant, fraction=pool["fraction"] * share / len(selected),
                                   terrain=deepcopy(terrains[name])))
    if not math.isclose(sum(g["fraction"] for g in groups), 1., abs_tol=1e-10):
        raise ValueError("Task sampling allocations do not sum to one")
    return dict(skill_specs=active, scene_groups=groups, behavior_pool_membership=membership,
                behavior_pool_fractions={p["name"]: p["fraction"] for p in domain_pools},
                episode_seconds=max(s["episode_seconds"] for s in active.values()))


def amend(config, domain, allocations, required, meta, source_specs, source_groups, plan, budget_sha):
    """Explicit reviewed amendments, applied separately to worker and design inputs."""
    config.update(distribution(source_specs, source_groups, allocations, plan["height_combination_sampling"]))
    config.update(curriculum_stage=domain, budget_stage_id=domain, total_updates=20000,
                  budget_spec_file=BUDGET_FILE, budget_spec_sha256=budget_sha,
                  learning_rate=1e-4, learning_rate_schedule="fixed", critic_warmup_updates=0,
                  resume_critic_warmup_updates=0, checkpoint_interval=100, checkpoint_first_update=1001,
                  checkpoint_snapshots=True, save_interval=20000, task_campaign=deepcopy(meta))
    config["enabled_stages"] = [KINDS[domain]]
    config["stages"] = [dict(name=KINDS[domain], updates=20000,
        vx_max=1. if domain == "flat" else 5., yaw_max=1. if domain == "flat" else 6 * math.pi,
        push_max=0., terrain=list(dict.fromkeys(t for g in config["scene_groups"] for t in g["terrain"])))]
    config["landing_tracking"] = dict(height_weight=2., velocity_weight=1., impact_weight=.05,
                                      passive_semantics="finite_task_cost_v1", window_seconds=.6)
    config["jump_assist"]["enabled"] = domain in ("jump", "stepjump")
    schedule = config["new_asset_curriculum"]
    schedule.update(stage=domain, command_resampling_seconds=[1., 8.],
                    long_command_resampling_seconds=[8., 15.], long_command_fraction=.2,
                    start_stop_start_delay_seconds=[0., 8.],
                    edge_reset_fraction=0. if domain == "flat" else .05,
                    unilateral_degradation_fraction=0. if domain == "flat" else .05)
    schedule["request_protocol"]["stress_episode_fraction"] = .2 if domain in ("jump", "stepjump") else 0.
    config["evaluation"].update(promotion_case_names=list(required), quick_case_names=list(required),
        retention_case_names=list(required), protected_case_names=[], episodes_per_case=4,
        minimum_updates=1000, block_updates=1000, cumulative_retention=False,
        regression_recovery={"max_retries": 0, "learning_rate_factor": 1.})
    config.pop("continuation", None)
    config.pop("remediation", None)
    return config


def build():
    original = read(SOURCE_PLAN)
    old_design = read(original["design_preflight"]["file"])
    if sha(original["design_preflight"]["file"]) != original["design_preflight"]["sha256"]:
        raise ValueError("Reviewed independent role table changed")
    template = stage_contract({}, original, original["stages"][-1], num_envs=20000)
    independent_specs = old_design["stages"]["S6"]["roles"]["train"]["contract"]["skill_specs"]
    independent_groups = old_design["stages"]["S6"]["roles"]["train"]["contract"]["scene_groups"]
    if template["skill_specs"] != independent_specs or template["scene_groups"] != independent_groups:
        raise ValueError("Native stage template drifted from the reviewed independent design")
    catalog = {s["name"]: s for s in original["skill_catalog"]}
    allocations = pools(catalog)
    cases = template["evaluation"]["cases"]
    if len(cases) != 322:
        raise ValueError("The exact 322-case catalogue must be retained")
    required = case_scopes(cases, catalog, allocations)
    budget_sha = write(BUDGET_FILE, dict(schema="task-domain-budget-v1", campaign_id=CAMPAIGN_ID,
        historical_consumed_updates=1000, historical_actor_consumed_updates=980,
        rollback_refunds_consumed_updates=False, total_ceiling=121000,
        domains={n: 20000 for n in DOMAIN_ORDER}, update_unit="one_20000x24_PPO_update"))
    previous = read("contracts/v6_s1_persistent_recovery_v1.json")
    if sha(SOURCE_A) != previous["source_contract_sha256"]:
        raise ValueError("A learning source contract changed")
    result = dict(contract_id="v6-task-campaign-v1", campaign_id=CAMPAIGN_ID,
        scope="independent_task_domain_research", target_num_envs=20000, num_steps_per_env=24,
        seed=290927, evaluation_seeds=[290929, 290930], historical_consumed_updates=1000,
        historical_actor_consumed_updates=980, total_ceiling=121000,
        rollback_refunds_consumed_updates=False, automatic_promotion=False,
        source_checkpoint=previous["source_checkpoint"], source_checkpoint_sha256=previous["source_checkpoint_sha256"],
        source_contract_sha256=sha(SOURCE_A), source_contract_file=SOURCE_A,
        source_stage_actor_update=368, source_learning_lineage_updates=378,
        budget_source_stage="/home/kaiser/robot-rl-sim60/experiments/v6-s1-continue-20260930T174327Z-45dd84/train",
        budget_source_files_sha256={}, budget_spec_file=BUDGET_FILE, budget_spec_sha256=budget_sha,
        source_curriculum_plan_sha256=sha(SOURCE_PLAN),
        runtime_estimate=dict(seconds_per_update=42., ppo_upper_bound_seconds_per_domain=840000,
                              basis="observed_prior_20k_1kHz_worker; estimate_not_required_run_length"),
        domain_completion=dict(assessment_interval_actor_updates=1000, cooperative_same_scene=True,
            evaluation_gpu_exclusive=True, stop_if_all_required_pass_both_seeds=True,
            continue_after_failed_assessment=True, cap_result_if_failed="unqualified",
            research_child_may_start_from_unqualified_parent=True, integrated_policy_qualified=False),
        domains=[])
    receipts = "reports/v6_s1_persistent_recovery_20261001/deploy/"
    for remote, local in (("completion.json", "completion.json"), ("consumption.json", "consumption.json"),
                          ("A_persistent_recovery/train/model_final.checkpoint.json", "final_checkpoint_seal.json")):
        result["budget_source_files_sha256"][remote] = sha(receipts + local)
    if read(receipts + "consumption.json")["consumed_updates"] != 1000:
        raise ValueError("Historical budget source has not been sealed at 1000")
    workers = {"A": SOURCE_A}
    for domain in DOMAIN_ORDER:
        source_file = workers[PARENTS[domain]]
        meta = dict(campaign_id=CAMPAIGN_ID, domain=domain, source_contract_file=source_file,
                    source_contract_sha256=sha(source_file), updates=20000, clock="domain_actor_updates",
                    learning_audit=True, evaluation_interval=1000, initialization="full_learning_state",
                    parent_candidate_scope="research_not_accepted")
        worker = amend(deepcopy(template), domain, allocations[domain], required[domain], meta,
                       template["skill_specs"], template["scene_groups"], original, budget_sha)
        design = {k: deepcopy(v) for k, v in old_design.items() if k != "stages"}
        design["inherited_design_sha256"] = original["design_preflight"]["sha256"]
        design["inherited_reviewed_plan_payload_sha256"] = design.pop("reviewed_plan_payload_sha256")
        design["reviewed_change"] = "independent_task_domain_campaign_20261001"
        rows = deepcopy(old_design["stages"]["S6"])
        for role, row in rows["roles"].items():
            expected = amend(row["contract"], domain, allocations[domain], required[domain], meta,
                             independent_specs, independent_groups, original, budget_sha)
            row["runtime"]["finite_passive_landing"] = dict(enabled=True, window_seconds=.6, policy_dt=.02)
            row["runtime"]["start_stop_timing"] = dict(enabled=role == "train",
                start_delay_seconds=[0., 8.] if role == "train" else None, policy_dt=.02)
            row["runtime"]["modules"]["jump_assist"] = role == "train" and domain in ("jump", "stepjump")
            if role != "train":
                expected["jump_assist"]["enabled"] = False
                expected["evaluation"]["cases"] = [c for c in expected["evaluation"]["cases"] if c["name"] in required[domain]]
                selected = expected["evaluation"]["cases"]
                expected["scene_groups"] = [dict(name=c["name"], fraction=1 / len(selected),
                                                terrain=[c.get("terrain", "flat")]) for c in selected]
                expected["episode_seconds"] = max(c["episode_seconds"] for c in selected)
                expected["evaluation"]["episode_seconds"] = expected["episode_seconds"]
                row["invocation"]["num_envs"] = len(selected) * 4
        design["stages"] = {domain: rows}
        design_file = f"contracts/v6_task_campaign_{domain}_expectations_v1.json"
        worker["design_preflight"] = dict(file=design_file, sha256=write(design_file, design), stage=domain)
        worker_file = f"contracts/v6_task_campaign_{domain}_worker_v1.json"
        workers[domain] = worker_file
        item = dict(name=domain, kind=KINDS[domain], parent_domain=PARENTS[domain], updates=20000,
                    contract=worker_file, sha256=write(worker_file, worker),
                    required_case_names=required[domain], monitor_case_names=required[domain],
                    sampling_pools=allocations[domain], source_scope="research_candidate_full_learning_state",
                    required_pass_does_not_imply_hardware_or_integrated_policy_acceptance=True)
        if domain == "recovery":
            item["scene_validation"] = dict(required=True, recovery_cases=54,
                prior_engineering_result="65/108; not qualified", scripted_and_blended_samples_in_ppo=False,
                validation="native_20k_startup_and_exact_pose_kinematics_before_domain_training")
        result["domains"].append(item)
    write(PLAN_FILE, result)
    print({"campaign": CAMPAIGN_ID, "ceiling": 121000,
           "domains": {d["name"]: len(d["required_case_names"]) for d in result["domains"]}})
    return result


def build_v2():
    """Keep the deployed v1 immutable; amend independently frozen worker/roles."""
    old = read(PLAN_FILE)
    result = deepcopy(old)
    new_plan_file = "contracts/v6_task_campaign_v2.json"
    migration_file = "contracts/v6_task_campaign_migration_v2.json"
    run_directory = "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train"
    result["contract_id"] = "v6-task-campaign-v2"
    result["domain_completion"] = dict(execution_mode=CONTINUOUS_MODE,
        assessment_interval_actor_updates=0, cooperative_same_scene=False,
        assessment_mode="explicit_offline_only", stop_if_all_required_pass_both_seeds=False,
        cap_result="unassessed_research_candidate", research_child_may_start_from_unqualified_parent=True,
        integrated_policy_qualified=False)
    result["revision_migration"] = dict(manifest_file=migration_file, source_plan_sha256=sha(PLAN_FILE),
        run_directory=run_directory)
    workers = {"A": SOURCE_A}
    pins = []
    for old_row, row in zip(old["domains"], result["domains"], strict=True):
        domain = row["name"]
        worker = read(old_row["contract"])
        if sha(old_row["contract"]) != old_row["sha256"]:
            raise ValueError("Frozen v1 worker changed")
        binding = worker["design_preflight"]
        if sha(binding["file"]) != binding["sha256"]:
            raise ValueError("Frozen v1 design changed")
        design = read(binding["file"])
        source = workers[row["parent_domain"]]
        meta = deepcopy(worker["task_campaign"])
        meta.update(source_contract_file=source, source_contract_sha256=sha(source),
            execution_mode=CONTINUOUS_MODE, evaluation_interval=0,
            revision_migration=dict(schema="task-worker-revision-v2", source_contract_file=old_row["contract"],
                                    source_contract_sha256=old_row["sha256"]))

        def patch(config):
            config["checkpoint_interval"] = 500
            config["task_campaign"] = deepcopy(meta)
            if domain in ("jump", "stepjump"):
                curriculum = config["new_asset_curriculum"]["jump_curriculum"]
                curriculum.pop("confirmed_pass_transfer_fraction", None)
                curriculum.update(clock="domain_actor_updates", iteration_probability_schedule=jump_probability_schedule())

        patch(worker)
        design["reviewed_change"] = "continuous_task_caps_without_runtime_assessment_v2"
        design["inherited_task_design_sha256"] = binding["sha256"]
        for expected in design["stages"][domain]["roles"].values():
            patch(expected["contract"])
        design_file = f"contracts/v6_task_campaign_{domain}_expectations_v2.json"
        worker["design_preflight"] = dict(file=design_file, sha256=write(design_file, design), stage=domain)
        worker_file = f"contracts/v6_task_campaign_{domain}_worker_v2.json"
        row.update(contract=worker_file, sha256=write(worker_file, worker))
        workers[domain] = worker_file
        pins.append(dict(domain=domain, source_file=old_row["contract"], source_sha256=old_row["sha256"],
                         target_file=worker_file, target_sha256=row["sha256"]))
    new_sha = write(new_plan_file, result)
    write(migration_file, dict(schema="task-plan-revision-v2", campaign_id=CAMPAIGN_ID,
        source_plan_file=PLAN_FILE, source_plan_sha256=sha(PLAN_FILE),
        target_plan_file=new_plan_file, target_plan_sha256=new_sha, run_directory=run_directory,
        consumed_updates_policy="preserve_all_attempts_no_refund", learning_state="actor_critic_Adam_and_task_clock_preserved",
        workers=pins))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    print({"contract_id": (build() if args.revision == 1 else build_v2())["contract_id"]})
