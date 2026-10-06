#!/usr/bin/env python3
"""Emit one read-only CPU stdin reader for source023's fixed native TB windows.

Default output is Python source, not a remote execution. This program never
starts SSH, a simulator, GPU work, training, evaluation or a TensorBoard server.
The emitted reader uses the already preserved, SHA-pinned 022/021/020/019 bundle
without writing remote files. Reports go only to stdout. No production gate or
contract is added; receipt checks identify the event being summarized.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "learning_audit"
COMMIT = "6f07ec298e01d2686216787cfba34b394151f1aa"
ROOT = "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358"
DEFAULT_BUNDLE = "/tmp/readonly_source022_stopped2440_20261007_001"
READERS = {
    "read_recovery_window_022.py": (AUDIT / "recovery_window_candidate_022/read_recovery_window_022.py", "a5427d51ce7998c98064f0251a632727aceeb5eae5d1ef66ae4e0ada38a55f56"),
    "read_recovery_window_021.py": (AUDIT / "recovery_window_candidate_021/read_recovery_window_021.py", "5faa75e24b0dd484bdacb0273756642caab8fcc6d6e66f90c034a0bda6ac19a6"),
    "read_recovery_window_020.py": (AUDIT / "read_recovery_window_020.py", "521b10353125272aa0f21ff9dc986c65b958ea7dd225051cfc6cfdd269a215e5"),
    "read_recovery_window_019.py": (AUDIT / "read_recovery_window_019.py", "ffefbaae2e1cbf30b520224fc286a8adbc194708570e4ce0fcb00fa7f86cdb17"),
}
RECEIPTS = {
    "launch_023.json": "4e179761ab2ebfb68fb7b78efcef02002f9ee04ce6fd7c22f7d3a19316411352",
    "first_PPO_verification_023.json": "a1431d9261f080f6d367b89bd2cb3728414fa7cc85c52d35a4dec5f3afb4a2a3",
    "predecessor_023.json": "45b55296d4bcd151c0bfd015c413d9303a5350ce53bdfea4b996e235db920e5d",
    "source_revision_023.json": "b17e79369bdf23c862d619281c70cf93f5e924d2bf738bfbb19e00ae63b5ffd2",
    "frozen_source_inventory_023.json": "f1b20cfce97329447f037a422c546728735fbc190825035ada3227e28832e3d3",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundaries(end_actor):
    require(type(end_actor) is int and end_actor >= 110 and (end_actor - 110) % 100 == 0,
            "Use fixed non-overlapping endpoints 110, 210, 310, ...")
    base = end_actor - 100
    return dict(actor_updates=100, actor_updates_inclusive=[base + 1, end_actor],
                baseline_update=base, end_update=end_actor,
                successful_update_endpoints_in_attempt=[10 + base, 10 + end_actor],
                baseline_step=33630 + base - 1,
                event_steps_inclusive=[33630 + base, 33630 + end_actor - 1],
                task_actor_updates_inclusive=[6425 + base + 1, 6425 + end_actor],
                expected_observed_frames=9830400, policy_seconds_per_environment=48.,
                critic_warmup_excluded=True, new_critic_warmup_updates=10)


def payload(end_actor, bundle, max_read_mib):
    require(1 <= max_read_mib <= 1024, "Read byte limit must be 1..1024 MiB")
    records = {}
    for name, expected in RECEIPTS.items():
        require(digest(HERE / name) == expected, "Actual receipt bytes changed: " + name)
        records[name] = json.loads((HERE / name).read_text())
    for name, (path, expected) in READERS.items():
        require(digest(path) == expected, "Preserved reader bytes changed: " + name)
    launch, first = records["launch_023.json"], records["first_PPO_verification_023.json"]
    require(first["source_commit"] == launch["source_commit"] == COMMIT and
            first["status"] == "first_actual_actor_PPO_verified" and
            first["check_counts"] == dict(passed=10, pending=0, failed=0), "Actual first-PPO identity differs")
    require(first["evidence"]["launch"] == launch, "Actual launch differs from first-PPO receipt")
    seal = launch["parent"]["seal"]
    require((seal["learning_updates"], seal["actor_updates_consumed"], seal["task_actor_update"]) ==
            (33620, 33460, 6425) and launch["expected_critic_warmup"] == 10, "Actual parent clocks differ")
    # Take the existing metric source set from literal declarations, not imports.
    names = set()
    for file, assignment in ((READERS["read_recovery_window_019.py"][0], "SOURCE_HASHES"),
                             (READERS["read_recovery_window_020.py"][0], "SOURCE_UPDATES")):
        for node in ast.parse(file.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == assignment for t in node.targets):
                names.update(ast.literal_eval(node.value))
    names.update({"contracts/v6_task_campaign_v21.json", "contracts/v6_task_campaign_recovery_worker_v21.json",
                  "src/wheeled_tasks/chassis/recovery.py", "src/wheeled_tasks/chassis/recovery_script_revision.py",
                  "src/wheeled_tasks/chassis/recovery_quality_revision.py", "src/wheeled_tasks/chassis/task_campaign.py"})
    inventory = records["frozen_source_inventory_023.json"]
    return dict(source_commit=COMMIT, root=ROOT, bundle=bundle, window=boundaries(end_actor),
                max_read_bytes=max_read_mib * 1024**2, reader_sha256=digest(__file__),
                reader_dependencies_sha256={name: h for name, (_, h) in READERS.items()},
                actual_receipts_sha256=RECEIPTS, launch=launch,
                revision=records["source_revision_023.json"],
                first_PPO=dict(invocation=first["evidence"]["invocation"],
                               resume_verification=first["evidence"]["resume_verification"],
                               actual_recovery_script_revision=first["evidence"]["actual_recovery_script_revision"],
                               native_hold_cost_gate_binding=first["checks"]["actual_native_full_restore_new_modules_origins_and_clocks"]["detail"]["native_hold_cost_gate_binding"]),
                source_files_sha256={name: inventory[name] for name in sorted(names)})


def remote_main(p):
    import hashlib
    import importlib.util
    import json
    import os
    import sys
    from collections import Counter
    from datetime import datetime, timezone
    from pathlib import Path

    sys.dont_write_bytecode = True

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def read(path):
        return json.loads(Path(path).read_text())

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    result = dict(schema="source023-recovery-native-fixed100-window-readonly-v1", complete=False,
                  source_commit=p["source_commit"], qualification=None, window=p["window"],
                  reader_sha256=p["reader_sha256"], actual_receipts_sha256=p["actual_receipts_sha256"],
                  reader_dependencies_sha256=p["reader_dependencies_sha256"], PPO_updates_by_reader=0,
                  scope="Existing native stochastic PPO event; CPU only, no remote writes, no simulator/model/GPU/PPO/evaluation/TB server.",
                  comparison_scope="Sampled training statistics are separate from deterministic mean-policy qualification; different actor ages/curriculum/DR cannot establish a same-difficulty single-variable benefit.")
    try:
        print("READONLY_READER_PID=" + str(os.getpid()), file=sys.stderr, flush=True)
        root = Path(p["root"])
        run = root / "train/recovery/attempt_009/train"
        source = root / "source_6f07ec298e01"
        bundle = Path(p["bundle"])
        require(read(root / "recovery_entropy_launch_023.json") == p["launch"], "Actual launch receipt changed")
        require(read(root / "source_revision_023.json") == p["revision"], "Actual revision receipt changed")
        require(digest(root / "predecessor_023.json") == p["actual_receipts_sha256"]["predecessor_023.json"], "Actual predecessor bytes changed")
        require({name: digest(bundle / name) for name in p["reader_dependencies_sha256"]} == p["reader_dependencies_sha256"], "Preserved CPU reader bytes differ")
        require({name: digest(source / name) for name in p["source_files_sha256"]} == p["source_files_sha256"], "Frozen metric/controller/contract bytes differ")
        spec = importlib.util.spec_from_file_location("frozen022_for_source023_window", bundle / "read_recovery_window_022.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = read(source / "contracts/v6_task_campaign_recovery_worker_v21.json")
        invocation = read(run.parent / "invocation.json")
        argv = invocation["command"]
        require(argv == p["first_PPO"]["invocation"]["command"], "Actual worker invocation changed")
        rv = read(run / "resume_verification.json")
        require(rv == p["first_PPO"]["resume_verification"], "Actual resume evidence changed")
        startup = read(run / "startup.json")
        native = startup["startup"]
        observed = native["runtime_preflight"]["observed"]
        require(native["runtime_preflight"]["status"] == "runtime_configuration_passed" and
                (observed["physics_dt"], observed["policy_dt"], observed["decimation"], observed["seed"]) ==
                (.005, .02, 4, 290927), "Native timing identity differs")
        require(observed["recovery_hold_cost"] == config["recovery_hold_cost"] and
                observed["recovery_script_revision"] == p["first_PPO"]["actual_recovery_script_revision"] and
                observed["recovery_hold_cost_gate_binding"] == p["first_PPO"]["native_hold_cost_gate_binding"], "Native metric bindings differ")
        progress = read(run / "progress.json")
        completed, actors = progress["successful_updates"], progress["actor_updates_in_block"]
        critic = min(completed, 10)
        require(progress["pid"] == 3662149 and progress["num_envs"] == 4096 and
                completed == critic + actors and progress["parent_updates"] == 33620 and
                progress["task_actor_update"] == 6425 + actors and progress["stage_actor_update"] == 19793 + actors and
                progress["global_budget_actor_updates"] == 33460 + actors and
                progress["entropy_coef"] == config["entropy_coef"] == 0., "Native source023 identity/clocks differ")
        context = dict(consumption_origin=33620, task_actor_origin=6425, actor_consumption_origin=33460,
                       critic_warmup_updates=10, critic_completed=critic, actor_completed=actors,
                       total_completed=completed, first_actor_event_step=33630, first_actor_completion=33631,
                       actual_task_actor_update=progress["task_actor_update"], native_first11_observed=True,
                       consumption_and_actor_are_separate=True)
        adapter, reader = module.make_reader(context, bundle / "read_recovery_window_021.py",
                                             bundle / "read_recovery_window_020.py", bundle / "read_recovery_window_019.py")
        reader.COMMIT = p["source_commit"]
        owner = adapter.native_owner(progress["pid"], argv)
        require(owner["ppid"] == 3662142, "Native worker owner differs")
        groups = [row["name"] for row in config["scene_groups"]]
        pools = config["behavior_pool_membership"]
        bases = [g for g in groups if pools[g] != "scripted_recovery"]
        population = native["scene_group_counts"]
        require(len(groups) == len(set(groups)) == 95 and len(bases) == 41 and
                set(population) == set(groups) and sum(population.values()) == 4096 and
                sum(population[g] for g in bases) == 1449 and
                Counter(pools.values()) == Counter(scripted_recovery=54, stand_anchors=9,
                    stand_continuous=1, height=1, translation=16, transitions=6, yaw=8), "Native 95/41 catalog differs")
        events = list(run.glob("events.out.tfevents.*.3662149.*"))
        require(len(events) == 1, "Expected one event from actual source023 worker")
        metadata = {"startup": run / "startup.json", "invocation": run.parent / "invocation.json",
                    "resume_verification": run / "resume_verification.json"}
        metadata_sha = {name: digest(path) for name, path in metadata.items()}
        authority = dict(identity_verified=True, source_commit=p["source_commit"], source_root=str(source),
                         worker_pid=progress["pid"], event=str(events[0]), groups=groups, base_groups=bases,
                         group_pools=pools, population=population, progress=progress, clocks=context,
                         source_files_sha256=p["source_files_sha256"], invocation=invocation, native_owner=owner,
                         ordinary_posture_coverage=dict(available=True, native_eligible_population=1449,
                             base_group_population=1449, nonstandard_start_leg_bank_present=False))
        result.update(identity=authority, selection=dict(chosen_actor_end_update=p["window"]["end_update"],
                      available_actor_updates=actors, additional_actor_updates_needed=max(0, p["window"]["end_update"] - actors),
                      mode="explicit_nonoverlapping_actor_endpoint", native_zero_baseline_fabricated=False))
        if actors < p["window"]["end_update"]:
            result.update(status="pending_requested_actor_window", reason="Requested actor endpoint has not completed; event not parsed.")
        else:
            size_before = events[0].stat().st_size
            series, audit = reader.parse_events(events[0], reader.wanted_tags(authority) | adapter.extra_tags(authority),
                p["window"]["event_steps_inclusive"][1], min_step=reader.BASELINE_STEP, max_read_bytes=p["max_read_bytes"])
            window = adapter.select_actor_window(reader, series, authority, audit,
                100, p["window"]["end_update"], "after_ten_actor_baseline")
            window["reused_metric_schema"] = window["schema"]
            window["schema"] = "source023-recovery-native-actor-window-summary-v1"
            window["limitations"] = [line for line in window["limitations"] if not line.startswith("Source021 changes")]
            window["limitations"].append("Source023 sets recovery entropy coefficient to zero and charges ten critic adaptation updates; this stochastic window is descriptive, not causal or functional qualification.")
            require(window["window"]["event_steps_inclusive"] == p["window"]["event_steps_inclusive"] and
                    window["window"]["task_actor_updates_inclusive"] == p["window"]["task_actor_updates_inclusive"], "Fixed window boundary differs")
            result.update(status="requested_actor_window_complete" if window["complete"] else "pending_missing_native_tags",
                          complete=window["complete"], windows={"100": window}, parser_metric_reuse=reader.adapter_reuse,
                          event_bytes_before=size_before, event_bytes_after=events[0].stat().st_size,
                          read_byte_limit=p["max_read_bytes"])
        require({name: digest(source / name) for name in p["source_files_sha256"]} == p["source_files_sha256"], "Frozen source changed during read")
        require({name: digest(path) for name, path in metadata.items()} == metadata_sha, "Immutable native metadata changed during read")
        adapter.verify_owner_continuity(owner, adapter.native_owner(progress["pid"], argv))
        after = read(run / "progress.json")
        require(after["successful_updates"] >= completed, "Native completed clock went backwards")
        result.update(progress_after=after, owner_continuity_verified=True, source_files_unchanged=True,
                      immutable_metadata_unchanged=True)
    except Exception as error:
        result.update(status="read_failed", complete=False, error=type(error).__name__ + ": " + str(error))
    result["queried_at_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2 if result["status"] == "read_failed" else 0


def emitted_code(p):
    return ("# Run through existing remote python -B stdin; stdout is the only report destination.\n"
            "import json\n" + inspect.getsource(remote_main) +
            "\nraise SystemExit(remote_main(json.loads(" + repr(json.dumps(p, ensure_ascii=False, allow_nan=False)) + ")))\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("emit", "self-check"), default="emit")
    parser.add_argument("--end-actor", type=int, default=110, help="Fixed endpoints 110, 210, 310, ...")
    parser.add_argument("--reader-bundle", default=DEFAULT_BUNDLE, help="Existing remote SHA-pinned bundle; never created here")
    parser.add_argument("--max-read-mib", type=int, default=256)
    args = parser.parse_args(argv)
    p = payload(args.end_actor, args.reader_bundle, args.max_read_mib)
    code = emitted_code(p)
    compile(code, "source023_readonly_stdin", "exec")
    if args.mode == "emit":
        print(code, end="")
        return 0
    ast.parse(Path(__file__).read_text())
    windows = [boundaries(endpoint) for endpoint in (110, 210, 310)]
    require(windows[0]["event_steps_inclusive"] == [33640, 33739] and
            windows[0]["task_actor_updates_inclusive"] == [6436, 6535], "First-window boundary differs")
    require(all(b["event_steps_inclusive"][0] == a["event_steps_inclusive"][1] + 1
                for a, b in zip(windows, windows[1:])), "Fixed windows overlap or leave a gap")
    print(json.dumps(dict(status="local_syntax_and_fixed_boundary_checks_passed", reader_sha256=digest(__file__),
                         remote_code_compiled_not_executed=True, remote_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                         fixed_windows=windows, preserved_readers_sha256=p["reader_dependencies_sha256"],
                         actual_receipts_sha256=RECEIPTS, SSH_actions=0, GPU_actions=0, PPO_actions=0,
                         remote_writes=0, qualification=None), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
