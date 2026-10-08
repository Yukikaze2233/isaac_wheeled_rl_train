#!/usr/bin/env python3
"""Fail-closed v25 authority adapter; existing bounded TFRecord parser is reused.

No SSH, simulator, model, torch, training operation, or event download. The
reviewed binding must name actual adopted immutable metadata, never a plan.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

SOURCE_COMMIT = "59c201678b37cb33a03a878e734d1a263d8ab97a"
ARCHIVE_SHA = "c2d0d2fb023466d1d7f33abdf0e3149250fcb033a5310d6691ddd71f84304faa"
RECIPE_SHA = "69efab87fb9422f19fee7318ca3aab278488202b644e906b1be7c5ff77724902"
KERNEL_SHA = "ffefbaae2e1cbf30b520224fc286a8adbc194708570e4ce0fcb00fa7f86cdb17"
SOURCE_FILES = {
    "scripts/train_chassis.py": "d7808417b1443fb2a36b09dcab4d89ed3999fab1458b8c5501c4676163195a27",
    "src/wheeled_tasks/chassis/integrated_production.py": "fa4a218254f7c9d90da7d4edb4b3f4a84fc95710809db9d06ea5889ab9fcef90",
    "src/wheeled_tasks/chassis/integrated_runtime.py": "8e5d19869b18ad523f36dd57f71e106932673973783b34479cf95d4c34517f62",
    "src/wheeled_tasks/chassis/episode_metrics.py": "26438717a12b3ab37b7279cbcae09dda91c549438c7a6090e7d6876c7a046511",
    "src/wheeled_tasks/chassis/learning_audit.py": "0ac7a3502199c0912bebf6a1689cf3d388ffe03e23fcb3217628a9a70d324715",
    "src/wheeled_tasks/chassis/recovery_prefix_curriculum.py": "e25a2a89db76fde2dff680a3c76af283989b6377e2a64896de1568a050e40e2c",
}
PAID_FIELDS = ("reward_density/dense_velocity_broad", "reward_density/dense_velocity_precision",
               "reward_density/wheel_velocity", "reward_density/wheel_power", "total_reward_density",
               "advantage_mean", "return_target_residual_mean", "rollout_samples")
OUTCOME_FIELDS = ("ended", "successes", "failures", "collection_timeouts")
REQUIRED_FILES = ("revision", "adoption", "invocation", "selection", "admission", "startup",
                  "resume", "design", "worker", "recipe", "agent")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    # Immutable small metadata/source files only; event files never use this.
    path = Path(path)
    require(path.stat().st_size <= 16 * 1024**2, "Metadata/source file exceeds bounded size")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    require(Path(path).stat().st_size <= 16 * 1024**2, "Metadata file exceeds bounded size")
    return json.loads(Path(path).read_text())


def flag(argv, key):
    require(argv.count(key) == 1 and argv.index(key) + 1 < len(argv), "Missing/duplicate flag " + key)
    return argv[argv.index(key) + 1]


def integer(value, label):
    require(type(value) is int and value >= 0, "Invalid native integer: " + label)
    return value


def load_kernel(path):
    require(digest(path) == KERNEL_SHA, "Existing 019 parser bytes differ")
    spec = importlib.util.spec_from_file_location("bounded_native_kernel019", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize(worker, recipe, source_file, admission):
    """Execute only the frozen native pure _expand function, no module imports.

    This is source-authenticated reconstruction, not fresh runtime readback.
    The two attach_runtime additions are compared against saved admission/design.
    """
    tree = ast.parse(Path(source_file).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_expand"]
    require(len(nodes) == 1, "Frozen pure materializer missing/ambiguous")
    namespace = {"deepcopy": deepcopy}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_file), "exec"), namespace)
    config = namespace["_expand"](worker, None, recipe)
    runtime = config["integrated_runtime_candidate"]
    runtime.pop("static_recipe_preflight")
    runtime.update(learning_source=deepcopy(admission["selection"]),
                   clocks_at_source=deepcopy(admission["component_clocks_at_source"]))
    return config


def clock_helpers(source_file):
    """Reuse three frozen pure native clock functions without importing torch."""
    names = {"_clock_document", "_validate_clock_document", "advance_clock_document"}
    nodes = [n for n in ast.parse(Path(source_file).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, "Frozen pure clock helpers missing")
    namespace = dict(deepcopy=deepcopy, _number=integer, STAGES=("speed", "terrain", "jump", "stepjump"),
                     CLOCK_SCHEMA="v6-integrated-component-clocks-v1")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_file), "exec"), namespace)
    return namespace


def read_owner(pid):
    directory = Path("/proc") / str(pid)
    stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, ppid=int(stat[1]), start_ticks=int(stat[19]),
                command=(directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0"))


def campaign_workers(root):
    owners = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            owner = read_owner(int(directory.name))
            argv = owner["command"]
            if any(arg.endswith("/scripts/train_chassis.py") for arg in argv) and "--run-dir" in argv:
                if Path(flag(argv, "--run-dir")).resolve().is_relative_to(root):
                    owners.append(owner)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return owners


def authenticate(binding):
    require(binding.get("schema") == "v25-readonly-actual-authority-v1" and
            binding.get("reviewed_actual_adoption") is True, "Actual adoption has not been reviewed")
    require(binding.get("source_commit") == SOURCE_COMMIT, "Reader only admits frozen source027/v25")
    root, source, run = (Path(binding[key]).resolve() for key in ("campaign_root", "source_root", "run_dir"))
    require(source.is_relative_to(root) and run.is_relative_to(root) and run != root,
            "Source/run must belong to the actual campaign")
    pins = binding["files"]
    require(set(REQUIRED_FILES) <= set(pins), "Actual immutable metadata pins are missing")
    for key, path in {
        "revision": root / "source_revision_027.json", "adoption": root / "boundary_continuation_027.receipt.json",
        "invocation": run.parent / "invocation.json", "admission": run / "integrated_admission.json",
        "startup": run / "startup.json", "resume": run / "resume_verification.json",
        "design": run / "integrated_design.json", "agent": run / "agent_config.json",
    }.items():
        require(Path(pins[key]["path"]).resolve() == path, "Wrong native metadata path: " + key)
    data = {}
    for key in REQUIRED_FILES:
        pin = pins[key]
        path = Path(pin["path"]).resolve()
        require(path.is_relative_to(root) and digest(path) == pin["sha256"], "Immutable pin differs: " + key)
        data[key] = read_json(path)
    revision, adoption = data["revision"], data["adoption"]
    require(revision.get("schema") == "task-plan-revision-v25" and
            revision.get("source_commit") == SOURCE_COMMIT and revision.get("new_source_root") == str(source) and
            revision.get("source_archive_sha256") == ARCHIVE_SHA, "Not the actual frozen v25 revision")
    require(adoption.get("status") == "launch_submitted" and adoption.get("revision") == revision and
            adoption.get("reason") == "rec_boundary_native_continuation_authenticated",
            "Boundary is still staged/waiting or admission differs")
    argv = data["invocation"]["command"]
    require(str(source / "scripts/train_chassis.py") in argv and flag(argv, "--run-dir") == str(run) and
            flag(argv, "--contract") == pins["worker"]["path"] and "--validate-scene" not in argv,
            "Invocation is not this actual PPO worker")
    require(Path(pins["selection"]["path"]) == Path(flag(argv, "--integrated-source-selection-file")) and
            pins["selection"]["sha256"] == flag(argv, "--integrated-source-selection-sha256") ==
            pins["admission"]["sha256"] and data["selection"] == data["admission"], "Native source-selection binding differs")
    admission, startup, resume, design, worker = (data[k] for k in ("admission", "startup", "resume", "design", "worker"))
    selection, allowance = admission["selection"], admission["allowance"]
    domain = selection["domain"]
    require(admission.get("schema") == "v6-integrated-production-admission-v1" and
            selection.get("lineage_id") == "source027-joint-heading-v1" and
            domain in ("speed", "terrain", "jump", "stepjump") and admission.get("qualification") is None and
            allowance.get("admitted") is True, "Missing actual unqualified joint admission")
    require(worker["integrated_production"]["domain"] == domain and
            worker["integrated_production"]["recipe_sha256"] == pins["recipe"]["sha256"] == RECIPE_SHA and
            admission["target_contract_sha256"] == selection["contract_sha256"] == pins["worker"]["sha256"],
            "Contract/recipe/domain binding differs")
    require(startup["contract_sha256"] == pins["worker"]["sha256"] and
            startup["integrated_admission_sha256"] == pins["selection"]["sha256"] and
            startup["args"]["run_dir"] == str(run) and design["source_receipt"] == admission and
            design["selection"]["sha256"] == pins["selection"]["sha256"], "Startup/design belongs to another admission")
    for name, sha in SOURCE_FILES.items():
        require(digest(source / name) == sha and startup["source_sha256"].get(name) == sha,
                "Frozen producer/native source differs: " + name)
    config = materialize(worker, data["recipe"], source / "src/wheeled_tasks/chassis/integrated_production.py", admission)
    clock_file = source / "src/wheeled_tasks/chassis/integrated_runtime.py"
    clocks = clock_helpers(clock_file)["_validate_clock_document"](admission["component_clocks_at_source"], selection)
    native = startup["startup"]
    runtime = native["runtime_preflight"]
    require(runtime.get("status") == "runtime_configuration_passed" and
            runtime.get("scoped_validation") == "v6-integrated-preflight-candidate-v1" and
            "integrated_curriculum_observed" in runtime["observed"], "Actual scoped runtime readback missing")
    require((runtime["observed"]["physics_dt"], runtime["observed"]["policy_dt"],
             runtime["observed"]["decimation"]) == (.005, .02, 4), "Actual native timing readback differs")
    scope = design["runtime_expected"]["integrated_curriculum"]
    population, pools, specs = native["scene_group_counts"], config["behavior_pool_membership"], config["skill_specs"]
    groups = [row["name"] for row in config["scene_groups"]]
    require(len(groups) == len(set(groups)) and set(groups) == set(population) == set(specs) == set(pools) and
            population == scope["row_population"] == config["integrated_sampling_candidate"]["allocation"]["counts"] and
            scope["row_ownership"] == config["integrated_runtime_candidate"]["row_ownership"] and
            scope["clocks_at_source"] == admission["component_clocks_at_source"], "Reconstructed recipe and actual runtime catalog differ")
    actual_clocks = runtime["observed"]["integrated_curriculum_observed"]
    require(actual_clocks.get("schema") == "v6-integrated-preflight-candidate-v1" and
            actual_clocks.get("task_actor_updates") == clocks["domain_actor_updates"] and
            actual_clocks.get("component_actor_updates") == clocks["component_actor_updates"],
            "Actual startup component clocks differ from sealed admission")
    require(all(type(n) is int and n > 0 for n in population.values()) and sum(population.values()) == 4096,
            "Actual lane population differs")
    pool_population = Counter()
    for group, count in population.items():
        pool_population[pools[group]] += count
    require(dict(pool_population) == native["behavior_pool_counts"], "Actual pool/lane allocation differs")
    require(resume.get("parent_checkpoint_sha256") == selection["checkpoint"]["sha256"] and
            all(resume.get(key) is True for key in ("actor_exact", "critic_exact", "optimizer_exact")),
            "Actual complete learning-state resume missing")
    parent = integer(resume["parent_updates"], "consumption parent")
    warmup = integer(selection["critic_warmup_remaining"], "remaining critic adaptation")
    require(parent == allowance["consumed_updates"] == int(flag(argv, "--consumed-updates")) and warmup <= 10 and
            resume["restored_checkpoint_updates"] == selection["checkpoint_consumed_updates"] and
            int(flag(argv, "--task-actor-update")) == selection["fresh_domain_actor_updates"], "Sealed parent/critic/actor clocks differ")
    steps = integer(data["agent"]["num_steps_per_env"], "native rollout length")
    require(steps == 24 and (config["physics_dt"], config["policy_dt"]) == (.005, .02), "Native rollout/time base differs")
    progress = read_json(run / "progress.json")
    owner = read_owner(progress["pid"])
    require(owner == binding["owner"] and owner["command"] == argv and campaign_workers(root) == [owner],
            "Native worker is not the reviewed unique campaign owner")
    authority = dict(identity_verified=True, source_commit=SOURCE_COMMIT, source_root=str(source), run_dir=str(run),
        campaign_root=str(root), owner=owner, groups=groups, group_pools=pools, specs=specs, population=population,
        base_groups=[g for g in groups if pools[g] != "scripted_recovery"], policy_dt=.02,
        steps_per_update=steps, num_envs=4096, clocks=dict(parent_updates=parent, critic_warmup_updates=warmup,
        task_actor_origin=selection["fresh_domain_actor_updates"], stage_actor_origin=int(flag(argv, "--stage-actor-update")),
        actor_consumption_origin=int(flag(argv, "--actor-updates-consumed"))), progress=progress,
        source_clocks=clocks, clock_source_file=str(clock_file),
        immutable_files=pins, source_files_sha256=SOURCE_FILES,
        catalog_scope="frozen native pure _expand plus admission reconstruction, cross-checked with actual startup populations; not new runtime readback")
    check_progress(authority, progress)
    return authority


def check_progress(authority, progress):
    clocks = authority["clocks"]
    complete, actors = (integer(progress[k], k) for k in ("successful_updates", "actor_updates_in_block"))
    require(progress["pid"] == authority["owner"]["pid"] and progress["num_envs"] == authority["num_envs"] and
            progress["parent_updates"] == clocks["parent_updates"] and
            complete - actors == min(complete, clocks["critic_warmup_updates"]) and
            progress["task_actor_update"] == clocks["task_actor_origin"] + actors and
            progress["stage_actor_update"] == clocks["stage_actor_origin"] + actors and
            progress["global_budget_actor_updates"] == clocks["actor_consumption_origin"] + actors,
            "Actual completed/critic/actor clocks differ")
    expected = clock_helpers(authority["clock_source_file"])["advance_clock_document"](authority["source_clocks"], actors)
    expected_lineage = {key: expected[key] for key in ("lineage_id", "plan_sha256", "origin_checkpoint_sha256",
                                                       "domain", "joint_actor_updates", "domain_actor_updates")}
    expected_lineage["domain_complete"] = False  # native live progress is not a terminal seal
    require(progress.get("integrated_runtime_clocks") == expected and
            progress.get("integrated_lineage") == expected_lineage,
            "Actual progress joint/component lineage clocks differ")


def window_spec(authority, end_actor):
    require(type(end_actor) is int and end_actor >= 110, "100 actors need a real earlier cumulative baseline")
    warmup, parent = (authority["clocks"][k] for k in ("critic_warmup_updates", "parent_updates"))
    require((warmup + end_actor) % 10 == 0, "Endpoint must match native ten-completed-update Behavior cadence")
    baseline = end_actor - 100
    start, end = parent + warmup + baseline, parent + warmup + end_actor - 1
    return dict(actor_updates=100, actor_interval=[baseline + 1, end_actor], baseline_actor_update=baseline,
                baseline_step=start - 1, start_step=start, end_step=end,
                behavior_steps=list(range(start - 1, end + 1, 10)), dense_steps=list(range(start, end + 1)),
                task_actor_interval=[authority["clocks"]["task_actor_origin"] + baseline + 1,
                                     authority["clocks"]["task_actor_origin"] + end_actor])


def stop_groups(authority):
    return [g for g in authority["groups"] if authority["specs"][g].get("kind") in
            ("start_stop", "velocity_curve", "release_stop")]


def wanted_tags(kernel, authority):
    # Reuse frozen field names, without allocating unrelated019 diagnostics.
    tags = {f"Behavior/{g}/{f}" for g in authority["groups"]
            for f in (*kernel.COUNTS, *kernel.MEANS, "tilt_max_deg")}
    tags.update(f"LearningAudit/{g}/stationary_10s/{f}" for g in authority["groups"]
                for f in kernel.STATIONARY_FIELDS)
    tags.update("Timing/" + f for f in kernel.TIMING_FIELDS)
    tags.update("RecoveryRollout/" + f for f in ("valid_transitions", "script_or_blend_transitions", "valid_fraction"))
    tags.update(f"LearningAudit/{g}/{phase}/{f}" for g in stop_groups(authority)
                for phase in ("braking", "settled_stop") for f in kernel.STOP_FIELDS)
    tags.update(f"LearningAudit/{g}/{f}" for g in ("forward_05", "backward_05") for f in PAID_FIELDS)
    tags.update(f"RecoveryOutcomes/{lesson}/{f}" for lesson in kernel.LESSONS for f in OUTCOME_FIELDS)
    return tags


def summarize(kernel, series, authority, audit, window):
    """Parameterize catalog/clocks only; scalar/stats/counter invariants reuse019."""
    missing = []
    def covered(tag, steps):
        absent = [s for s in steps if kernel.scalar(series, tag, s) is None]
        if absent:
            missing.append(dict(tag=tag, missing_steps=absent))
        return not absent
    steps, baseline, end = window["dense_steps"], window["baseline_step"], window["end_step"]
    physical = authority["num_envs"] * authority["steps_per_update"]
    behavior = {}
    for group in authority["groups"]:
        prefix = f"Behavior/{group}/"
        fields = (*kernel.COUNTS, *kernel.MEANS, "tilt_max_deg")
        if not all([covered(prefix + f, window["behavior_steps"]) for f in fields]):
            behavior[group] = dict(complete=False)
            continue
        previous = None
        for step in window["behavior_steps"]:
            counts = {f: kernel.scalar(series, prefix + f, step) for f in kernel.COUNTS}
            kernel.validate_counts(counts, group)
            expected = authority["population"][group] * (step - authority["clocks"]["parent_updates"] + 1) * authority["steps_per_update"]
            require(counts["observed_frames"] == expected and
                    (previous is None or all(counts[f] >= previous[f] for f in kernel.COUNTS)), "Worker-local counter denominator/monotonicity differs")
            previous = counts
        old = {f: kernel.scalar(series, prefix + f, baseline) for f in kernel.COUNTS}
        new = {f: kernel.scalar(series, prefix + f, end) for f in kernel.COUNTS}
        row = {f: int(new[f] - old[f]) for f in kernel.COUNTS}
        kernel.validate_counts(row, group + "/window")
        for f in kernel.MEANS:
            row[f] = (kernel.scalar(series, prefix + f, end) * new["frames"] -
                      kernel.scalar(series, prefix + f, baseline) * old["frames"]) / row["frames"] if row["frames"] else None
        row.update(complete=True, population=authority["population"][group],
            success_per_ended_episode=row["successes"] / row["episodes"] if row["episodes"] else None,
            tilt_max_scope="observer-lifetime maximum, cannot recover interval pitch/roll precision",
            tilt_max_deg_since_observer_birth=kernel.scalar(series, prefix + "tilt_max_deg", end))
        behavior[group] = row
    if all(row["complete"] for row in behavior.values()):
        require(sum(row["observed_frames"] for row in behavior.values()) == 100 * physical, "Full runtime exposure denominator differs")
    timing = {f: kernel.stats(series, "Timing/" + f, steps) for f in kernel.TIMING_FIELDS}
    for f in kernel.TIMING_FIELDS:
        covered("Timing/" + f, steps)
    paid = {}
    for group in ("forward_05", "backward_05"):
        paid[group] = dict(command_semantics="continuous distribution under legacy group name; not fixed +/-0.5m/s",
                           physical_rollout_samples=authority["population"].get(group, 0) * authority["steps_per_update"], fields={})
        for field in PAID_FIELDS:
            tag = f"LearningAudit/{group}/{field}"
            covered(tag, steps)
            unit = ("stored_policy_frames_including_masked" if field == "rollout_samples" else
                    "stored_advantage_normalization_and_mask_dependent_not_gradient" if field == "advantage_mean" else
                    "stored_return_units" if field == "return_target_residual_mean" else
                    "reward_per_second_including_folded_events" if field == "total_reward_density" else "reward_per_second")
            paid[group]["fields"][field] = dict(unit=unit, full100=kernel.stats(series, tag, steps),
                first50=kernel.stats(series, tag, steps[:50]), second50=kernel.stats(series, tag, steps[50:]))
            if field == "rollout_samples" and paid[group]["fields"][field]["full100"]["complete"]:
                require(all(kernel.scalar(series, tag, s) == paid[group]["physical_rollout_samples"] for s in steps),
                        "Actual group rollout denominator differs")
    outcomes = {}
    for lesson in kernel.LESSONS:
        prefix = f"RecoveryOutcomes/{lesson}/"
        if not all([covered(prefix + f, [baseline, end]) for f in OUTCOME_FIELDS]):
            outcomes[lesson] = dict(complete=False, ended=None, success_fraction=None)
            continue
        endpoints = []
        for step in (baseline, end):
            counts = {f: kernel.scalar(series, prefix + f, step) for f in OUTCOME_FIELDS}
            require(all(v >= 0 and v == int(v) and v <= 2**24 for v in counts.values()), "Native outcome count is not exactly representable integer")
            require(counts["ended"] == sum(counts[f] for f in OUTCOME_FIELDS[1:]), "Native ended outcome partition differs")
            endpoints.append(counts)
        delta = {f: int(endpoints[1][f] - endpoints[0][f]) for f in OUTCOME_FIELDS}
        require(all(v >= 0 for v in delta.values()) and delta["ended"] == sum(delta[f] for f in OUTCOME_FIELDS[1:]),
                "Same-worker ended outcome counters regressed/differ")
        outcomes[lesson] = dict(complete=True, **delta, baseline=endpoints[0], endpoint=endpoints[1],
            success_fraction=delta["successes"] / delta["ended"] if delta["ended"] else None,
            denominator="same actual worker native ended episodes for this reset-selected lesson")
    stop = {}
    for group in stop_groups(authority):
        stop[group] = {}
        for phase in ("braking", "settled_stop"):
            prefix = f"LearningAudit/{group}/{phase}/"
            ready = all([covered(prefix + f, steps) for f in kernel.STOP_FIELDS])
            row = dict(complete=ready, samples=None, mean_speed_m_s=None, above_3cm_s_fraction=None, peak_speed_m_s=None)
            if ready:
                weights = [kernel.scalar(series, prefix + "samples", s) for s in steps]
                require(all(w >= 0 and w == int(w) for w in weights), "Invalid conditional phase samples")
                row["samples"] = int(sum(weights))
                if row["samples"]:
                    for f in ("mean_speed_m_s", "above_3cm_s_fraction"):
                        row[f] = sum(kernel.scalar(series, prefix + f, s) * w for s, w in zip(steps, weights)) / row["samples"]
                    row["peak_speed_m_s"] = max(kernel.scalar(series, prefix + "peak_speed_m_s", s) for s, w in zip(steps, weights) if w)
            stop[group][phase] = row
    ownership = {}
    tags = ["RecoveryRollout/" + f for f in ("valid_transitions", "script_or_blend_transitions", "valid_fraction")]
    ready = all([covered(tag, steps) for tag in tags])
    ownership["complete"] = ready
    if ready:
        valid_total = excluded_total = 0
        for s in steps:
            valid, excluded, fraction = (kernel.scalar(series, tag, s) for tag in tags)
            require(valid >= 0 and excluded >= 0 and valid == int(valid) and excluded == int(excluded) and
                    valid + excluded == physical and abs(fraction - valid / physical) <= 2e-6, "PPO mask partition differs")
            valid_total += int(valid); excluded_total += int(excluded)
        ownership.update(valid_transitions=valid_total, excluded_transitions=excluded_total, observed_transitions=100 * physical)
    stationary = {}
    for group in authority["groups"]:
        prefix = f"LearningAudit/{group}/stationary_10s/"
        ready = all([covered(prefix + f, steps) for f in kernel.STATIONARY_FIELDS])
        row = dict(complete=ready, samples=None, endpoint_mean_m=None, peak_mean_m=None, peak_max_m=None, mean_speed_m_s=None)
        if ready:
            weights = [kernel.scalar(series, prefix + "samples", s) for s in steps]
            require(all(w >= 0 and w == int(w) for w in weights), "Invalid eligible stationary-window samples")
            row["samples"] = int(sum(weights))
            if row["samples"]:
                for field in ("endpoint_mean_m", "peak_mean_m", "mean_speed_m_s"):
                    row[field] = sum(kernel.scalar(series, prefix + field, s) * w for s, w in zip(steps, weights)) / row["samples"]
                row["peak_max_m"] = max(kernel.scalar(series, prefix + "peak_max_m", s) for s, w in zip(steps, weights) if w)
        stationary[group] = row
    return dict(schema="v25-native-readonly-window-v1", complete=not missing, qualification=None, window=window,
        source_commit=SOURCE_COMMIT, owner=authority["owner"], catalog=dict(groups=len(authority["groups"]),
        pools=len(set(authority["group_pools"].values())), population=authority["population"], scope=authority["catalog_scope"]),
        behavior=behavior, timing=timing, payment=paid, recovery_outcomes=outcomes, conditional_stop=stop,
        stationary_10s=stationary,
        learning_ownership=ownership, missing=missing, event=audit,
        limitations=["Stochastic training window is not deterministic policy or hardware qualification.",
            "Legacy named continuous groups do not identify fixed-command cases; old REC comparisons change distributions.",
            "Paid rewards do not establish per-term policy gradients; stored group frames include masked samples.",
            "Mixed recovery-label success is distinct from original_chain exact ended-episode rate.",
            "Counter differences are only within this attempt/owner; never subtract prior-attempt cumulative values."])


def read_window(binding, kernel_path, end_actor, max_read_bytes=512 * 1024**2):
    started = time.monotonic()
    require(type(max_read_bytes) is int and 1024 <= max_read_bytes <= 1024**3, "Suffix cap must be explicit and <=1GiB")
    authority = authenticate(binding)
    window = window_spec(authority, end_actor)
    require(authority["progress"]["actor_updates_in_block"] >= end_actor, "Requested actor window has not completed; no event read")
    kernel = load_kernel(kernel_path)
    run = Path(authority["run_dir"])
    events = list(run.glob(f"events.out.tfevents.*.{authority['owner']['pid']}.*"))
    require(len(events) == 1, "Native unique worker event missing/ambiguous")
    series, audit = kernel.parse_events(events[0], wanted_tags(kernel, authority), window["end_step"],
        min_step=window["baseline_step"], max_read_bytes=max_read_bytes)
    require(audit["incomplete_live_tail_bytes"] == 0, "Incomplete live TFRecord tail; no successful read is certified")
    result = summarize(kernel, series, authority, audit, window)
    require(read_owner(authority["owner"]["pid"]) == authority["owner"] and
            campaign_workers(Path(authority["campaign_root"])) == [authority["owner"]], "Owner changed during read")
    for pin in authority["immutable_files"].values():
        require(digest(pin["path"]) == pin["sha256"], "Immutable metadata changed during read")
    for name, sha in SOURCE_FILES.items():
        require(digest(Path(authority["source_root"]) / name) == sha, "Frozen source changed during read")
    require(digest(kernel_path) == KERNEL_SHA, "Preserved bounded parser changed during read")
    after = read_json(run / "progress.json")
    check_progress(authority, after)
    require(after["successful_updates"] >= authority["progress"]["successful_updates"], "Completed native clock regressed")
    result.update(reader_wall_seconds=time.monotonic() - started, PPO_updates_by_reader=0,
                  owner_continuity_verified=True, progress_after=after)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--kernel019", required=True, type=Path)
    parser.add_argument("--actor-end", required=True, type=int)
    parser.add_argument("--max-read-bytes", type=int, default=512 * 1024**2)
    args = parser.parse_args()
    try:
        result = read_window(read_json(args.binding), args.kernel019, args.actor_end, args.max_read_bytes)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["complete"] else 3
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(schema="v25-native-readonly-rejection-v1", complete=False, qualification=None,
              error=f"{type(error).__name__}: {error}", PPO_updates_by_reader=0), allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
