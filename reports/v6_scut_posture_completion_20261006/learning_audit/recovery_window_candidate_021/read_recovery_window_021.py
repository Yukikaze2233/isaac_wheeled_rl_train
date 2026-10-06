#!/usr/bin/env python3
"""Read source021 native TB with the frozen020/019 CRC and metric parser.

No SSH, simulator, model, optimizer or TB server is instantiated. Existing
events remain the curve source. Default windows retain a ten-actor baseline.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD020 = HERE.parent / "read_recovery_window_020.py"
OLD019 = HERE.parent / "read_recovery_window_019.py"
READER020_SHA = "521b10353125272aa0f21ff9dc986c65b958ea7dd225051cfc6cfdd269a215e5"
COMMIT = "d0dce6181df83b03fe166abcbadef3c81de2e57e"
ARCHIVE_SHA = "2c019ba1fd56053af75bf631258397176de8c5b40f36d35ccdf54d32f5b21ee9"
INVENTORY_SHA = "c3f3f17a3747b27e2982793a796fff29fe622d3406b9d2d614a682bbaf34d068"
FIRST_PPO_SHA = "e7af7199d4bd64ef7dd527d2cc56cc83d46dbac9a7e6c0fdc5c0fd5fa56aa30e"
PLAN_SHA = "dbce7bf39105bb284ee9927f06239ff8ce767bcadbab896f2b653897c19649f9"
WORKER = "contracts/v6_task_campaign_recovery_worker_v19.json"
HOLD_TAGS = ("/HoldCost/actor_hold_population",) + tuple(
    f"/HoldCost/{axis}/{name}_mean" for axis in ("leg", "wheel", "gyro")
    for name in ("gate_relative_error", "broad_density", "fine_density"))
GATE_TAGS = tuple(f"/recovery/hold_gate/{gate}_pass_fraction" for gate in
    ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed"))
HOLD_STATE_TAGS = ("/recovery/hold_population", "/recovery/stable_peak_seconds_mean",
                   "/recovery/stable_break_count_mean")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load020(path=OLD020):
    require(digest(path) == READER020_SHA, "Frozen020 reader SHA differs")
    spec = importlib.util.spec_from_file_location("source020_for_021_window", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_extra_tag(tag, original):
    # Original019 already canonicalizes Episode//recovery/. Only the new native
    # HoldCost channel is added; unrelated namespaces are never merged.
    if tag.startswith("Episode/") and tag[len("Episode/"):].lstrip("/").startswith("HoldCost/"):
        return "/" + tag[len("Episode/"):].lstrip("/")
    return original(tag)


def clock_context(launch, first, progress):
    """Authenticate actual native clocks; neither critic rollouts nor debt refund."""
    seal = launch["parent"]["seal"]
    origin, actor_origin = seal["learning_updates"], seal["task_actor_update"]
    warmup = launch["expected_critic_warmup"]
    require(type(warmup) is int and warmup == 10, "Source021 critic adaptation differs")
    require(first["source_commit"] == launch["source_commit"] == COMMIT
        and all(row["status"] == "passed" for row in first["checks"].values()),
        "First-PPO admission incomplete")
    expected = dict(consumption_origin=origin, actor_consumption_origin=seal["actor_updates_consumed"],
        restored_checkpoint_learning_updates=origin, restored_task_actor_origin=actor_origin,
        expected_critic_only_updates=warmup, first_actor_consumption_update=origin + warmup + 1)
    require(first["evidence"]["clocks"] == expected, "First-PPO and parent clocks differ")
    direct = first["evidence"]["direct_native_progress_phases"]
    require(direct["observed_completed_updates"] == list(range(1, 12))
        and direct["observer_status"] == "complete", "Native first11 phase evidence absent")
    successful, actors = progress["successful_updates"], progress["actor_updates_in_block"]
    require(type(successful) is int and type(actors) is int and successful >= 0 and actors >= 0,
        "Invalid native completed update counts")
    critic = min(successful, warmup)
    require(successful == critic + actors and progress["parent_updates"] == origin
        and progress["task_actor_update"] == actor_origin + actors,
        "Critic/actor/consumption partition differs")
    require(progress["global_budget_actor_updates"] == seal["actor_updates_consumed"] + actors,
        "Actor consumption origin differs")
    require(progress["optimizer_phase"] == ("critic_warmup" if successful <= warmup else "actor_and_critic"),
        "Native optimizer phase differs")
    return dict(consumption_origin=origin, task_actor_origin=actor_origin,
        actor_consumption_origin=seal["actor_updates_consumed"], critic_warmup_updates=warmup,
        critic_completed=critic, actor_completed=actors, total_completed=successful,
        first_actor_event_step=origin + warmup, first_actor_completion=origin + warmup + 1,
        actual_task_actor_update=progress["task_actor_update"],
        native_first11_observed=True, consumption_and_actor_are_separate=True)


def make_reader(context, reader020=OLD020, reader019=OLD019):
    adapter = load020(reader020)
    reader = adapter.make_reader(reader019)
    original_canonical = reader.canonical_tag
    reader.canonical_tag = lambda tag: canonical_extra_tag(tag, original_canonical)
    # The old metric checks cumulative observed_frames for *all* collections,
    # including critic warmup. Keep original consumption origin, and use a
    # private successful-count coordinate. No authority/learning clock changes.
    reader.COMMIT = COMMIT
    reader.CONSUMPTION_ORIGIN = context["consumption_origin"]
    reader.ACTOR_ORIGIN = context["task_actor_origin"] - context["critic_warmup_updates"]
    reader.BASELINE_STEP = context["consumption_origin"] + context["critic_warmup_updates"] - 1
    reader.adapter_reuse["source021_projection"] = "private successful-count metric coordinate; public actor/parent remain authoritative"
    return reader


def native_owner(pid, argv, proc_root=Path("/proc")):
    """Read-only native process ownership; no signal or process is started."""
    command=(proc_root/str(pid)/"cmdline").read_bytes().split(b"\0")
    if command and command[-1]==b"":command.pop()
    require([part.decode() for part in command]==argv,"Live native worker cmdline differs")
    stat=(proc_root/str(pid)/"stat").read_text()
    fields=stat[stat.rfind(")")+2:].split()
    require(len(fields)>19 and fields[0] not in {"Z","X"},"Native worker is not alive")
    return dict(pid=pid,ppid=int(fields[1]),command=argv,starttime_ticks=int(fields[19]),state=fields[0])


def verify_owner_continuity(before, after):
    require(all(after[key]==before[key] for key in ("pid","ppid","command","starttime_ticks")),
        "Live worker ownership changed during read")


def authenticate(args, first, inventory, proc_root=Path("/proc")):
    reader020 = load020(args.reader020)
    read = reader020.make_reader(args.reader019).read_json
    launch = read(args.launch)
    require(launch["source_commit"] == COMMIT and launch["source_archive_sha256"] == ARCHIVE_SHA
        and launch["plan_sha256"] == PLAN_SHA, "Source021 launch header differs")
    source, root = Path(launch["source_root"]).resolve(), Path(launch["run_root"]).resolve()
    run = Path(first["evidence"]["actual_worker_directory"]).resolve()
    require(run.parts[-4:] == ("train", "recovery", "attempt_005", "train")
        and run.parents[3] == root, "First-PPO actual attempt/root differs")
    revision, predecessor = read(root / "source_revision_021.json"), read(root / "predecessor_021.json")
    require(revision["source_commit"] == COMMIT and revision["new_source_root"] == str(source)
        and revision["source_archive_sha256"] == ARCHIVE_SHA and revision["new_contract_sha256"] == PLAN_SHA,
        "Actual source021 revision differs")
    require(predecessor["actual_parents"]["recovery"] == launch["parent"]
        and predecessor["expected_critic_warmup"] == launch["expected_critic_warmup"] == 10,
        "Launch/predecessor parent differs")
    require(first["evidence"]["launch"] == launch, "Launch changed since admitted first-PPO")
    invocation = read(run.parent / "invocation.json")
    require(invocation["command"] == first["evidence"]["invocation"]["command"], "Actual worker invocation changed")
    argv = invocation["command"]
    def flag(name):
        require(argv.count(name) == 1, "Missing/duplicate worker flag " + name)
        return argv[argv.index(name) + 1]
    require(flag("--contract") == str(source / WORKER) and flag("--run-dir") == str(run)
        and flag("--resume") == launch["parent"]["path"] and flag("--num-envs") == "4096"
        and flag("--task-actor-update") == str(launch["parent"]["seal"]["task_actor_update"])
        and flag("--consumed-updates") == str(launch["parent"]["seal"]["learning_updates"]),
        "Native invocation/parent differs")
    progress = read(run / "progress.json")
    context = clock_context(launch, first, progress)
    worker = next(row for row in first["checks"]["unique_native_owner_chain_no_parallel_evaluator"]["detail"] if row["kind"] == "worker")
    require(progress["pid"] == worker["pid"] and worker["command"] == argv and progress["num_envs"] == 4096,
        "Native worker PID/population differs")
    owner=native_owner(progress["pid"],argv,proc_root)
    require(owner["ppid"]==worker["ppid"],"Live native worker parent differs")
    source_names = set(reader020.SOURCE_UPDATES) | set(reader020.make_reader(args.reader019).SOURCE_HASHES)
    source_names.update({WORKER, "contracts/v6_task_campaign_v19.json",
        "src/wheeled_tasks/chassis/recovery.py", "src/wheeled_tasks/chassis/recovery_prefix_curriculum.py"})
    expected = {name: inventory[name] for name in sorted(source_names)}
    actual = {name: digest(source / name) for name in expected}
    require(actual == expected, "Frozen source021 metric/controller/contract bytes differ")
    config = read(source / WORKER);groups = [row["name"] for row in config["scene_groups"]];pools = config["behavior_pool_membership"]
    bases = [g for g in groups if pools[g] != "scripted_recovery"]
    require(len(groups) == len(set(groups)) == 95 and len(bases) == 41
        and set(groups) == set(pools) == set(config["skill_specs"])
        and Counter(pools.values()) == Counter(scripted_recovery=54, stand_anchors=9,
            stand_continuous=1, height=1, translation=16, transitions=6, yaw=8), "Native catalog differs")
    require(config["recovery_training"]["prefix_curriculum"]["goal_bridge"] == "native-original-followup-after-stable-gate-v1"
        and config["recovery_hold_cost"]["enabled"] is True, "New teaching fields absent")
    startup = read(run / "startup.json");native = startup["startup"];observed = native["runtime_preflight"]["observed"]
    require(startup.get("pid", progress["pid"]) == progress["pid"]
        and native["runtime_preflight"]["status"] == "runtime_configuration_passed"
        and (observed["physics_dt"], observed["policy_dt"], observed["decimation"], observed["seed"]) == (.005,.02,4,290927)
        and observed["recovery_hold_cost"] == config["recovery_hold_cost"]
        and observed["modules"]["recovery_hold_cost"] is True, "Native timing/new hold runtime differs")
    require(observed["recovery_hold_cost_gate_binding"] == first["checks"]["actual_native_full_restore_new_modules_origins_and_clocks"]["detail"]["native_hold_cost_gate_binding"], "Actual hold gate binding differs")
    population = native["scene_group_counts"]
    require(set(population) == set(groups) and all(type(n) is int and n>0 for n in population.values())
        and sum(population.values()) == 4096 and sum(population[g] for g in bases) == 1449,
        "Actual positive95/41 population differs")
    ordinary = native["ordinary_posture_curriculum"]
    require(ordinary["origin_actor_update"] == config["ordinary_posture_curriculum"]["origin_actor_update"] == 2294
        and ordinary["runtime_observed"]["eligible_count"] == ordinary["runtime_observed"]["birth_eligible_population"] == 1449,
        "Ordinary native origin/eligibility differs")
    require(config["recovery_training"]["prefix_curriculum"]["origin_actor_update"] == 2294, "Recovery origin changed")
    events = list(run.glob("events.out.tfevents.*." + str(progress["pid"]) + ".*"))
    require(len(events) == 1, "Expected one event from authenticated worker")
    return dict(identity_verified=True,source_commit=COMMIT,source_root=str(source),worker_pid=progress["pid"],
        event=str(events[0]),groups=groups,base_groups=bases,group_pools=pools,population=population,
        progress=progress,clocks=context,source_files_sha256=actual,invocation=invocation,native_owner=owner,
        ordinary_posture_coverage=dict(available=True,native_eligible_population=1449,base_group_population=1449,
            nonstandard_start_leg_bank_present=False),
        metadata_sha256={"launch":digest(args.launch),"first_PPO":digest(args.first_ppo),
            "inventory":digest(args.inventory),"startup":digest(run/"startup.json"),
            "revision":digest(root/"source_revision_021.json"),"predecessor":digest(root/"predecessor_021.json"),
            "invocation":digest(run.parent/"invocation.json")})


def metric_view(authority):
    view=deepcopy(authority);c=authority["clocks"]
    view["clocks"]=dict(event_step_origin=c["consumption_origin"],
        task_actor_origin=c["task_actor_origin"]-c["critic_warmup_updates"],critic_warmup_updates=0)
    view["progress"]["actor_updates_in_block"]=c["total_completed"]
    return view


def extra_tags(authority):
    commands=tuple(f"LearningAudit/{group}/{field}" for group in authority["groups"]
        if group not in authority["base_groups"] for field in ("command_vx_m_s","command_yaw_rad_s"))
    return set(HOLD_TAGS+GATE_TAGS+HOLD_STATE_TAGS+commands)


def add_teaching_summary(reader,series,authority,report):
    steps=report["window"]["dense_steps"];missing=[]
    def observed(tag):
        row=reader.stats(series,tag,steps)
        if not row["complete"]:
            missing.append(dict(section="source021_teaching",tag=tag,
                missing_steps=[step for step in steps if reader.scalar(series,tag,step) is None]))
        return row
    hold={tag:observed(tag) for tag in HOLD_TAGS}
    release={tag:observed(tag) for tag in GATE_TAGS+HOLD_STATE_TAGS}
    commands={group:{field:observed(f"LearningAudit/{group}/{field}")
        for field in ("command_vx_m_s","command_yaw_rad_s")} for group in authority["groups"] if group not in authority["base_groups"]}
    report["teaching"]=dict(hold_cost=hold,hold_gates=release,static_group_command_means=commands,
        L0_lesson_success=report["dynamic_lessons"]["levels"]["prefix_stand"],
        original_chain_success=report["dynamic_lessons"]["full_chain_success"],
        release_profile_seconds=report["dynamic_lessons"]["lesson/release_seconds"],
        actual_L0_motion_release=dict(available=False,value=None,reason="Frozen native TB has no per-lesson motion_released tag; hold population is not a release count."),
        actual_L0_nonzero_goal_exposure=dict(available=False,value=None,reason="Static recovery groups mix lesson levels; group command means cannot certify L0-specific exposure."),
        scopes=["HoldCost fields are24-policy-frame averages of per-frame actor-hold conditional means; not exact lane sums or motor current.",
            "Gate-relative errors are normalized by the authenticated .25m/s wheel rim,2rad/s legs,.75rad/s three-axis gyro gates.",
            "Instantaneous gate pass fractions and historical stable peaks are not continuous1s completion probabilities.",
            "release_seconds is configured zero-torque release duration; not the instant of motion release.",
            "L0 lesson success cannot credit original-chain success or hardware qualification."])
    report["missing"].extend(missing);report["missing_required_tag_count"]=len(report["missing"])
    report["complete"]=not report["missing"]
    report["status"]="complete_actor_window" if report["complete"] else "pending_complete_actor_window"
    return report


def select_actor_window(reader,series,authority,audit,width,end_update="latest",mode="after_ten_actor_baseline"):
    c=authority["clocks"];warmup=c["critic_warmup_updates"];available=c["actor_completed"]
    earliest=width+10 if mode=="after_ten_actor_baseline" else width
    require(mode in {"after_ten_actor_baseline","after_critic_baseline"},"Unknown actor window mode")
    latest=available//10*10
    endpoints=(list(range(latest,earliest-1,-10)) or [earliest]) if end_update=="latest" else [end_update]
    chosen=None;tried=[]
    for endpoint in endpoints:
        require(type(endpoint)is int and endpoint%10==0 and endpoint>=earliest,"Invalid actor endpoint")
        report=reader.summarize(series,metric_view(authority),audit,reader.window_spec(width,warmup+endpoint))
        report=add_teaching_summary(reader,series,authority,report);tried.append(endpoint)
        if chosen is None:chosen=report
        if report["complete"]:chosen=report;break
    w=chosen["window"];actual_end=w["end_update"]-warmup;actual_base=w["baseline_update"]-warmup
    w.update(successful_update_endpoints_in_attempt=[w["baseline_update"],w["end_update"]],
        baseline_update=actual_base,end_update=actual_end,critic_warmup_excluded=True,new_critic_warmup_updates=warmup,
        task_actor_updates_inclusive=[c["task_actor_origin"]+actual_base+1,c["task_actor_origin"]+actual_end],
        authoritative_parent_consumption=c["consumption_origin"],post_critic_event_offset=c["first_actor_event_step"],
        actor_baseline_mode=mode,baseline_evidence="actual95-group Behavior at native completed critic10" if actual_base==0 else "actual95-group Behavior after ten or more actor updates")
    chosen.update(schema="source021-recovery-native-actor-window-summary-v1",
        selection=dict(mode="latest_complete" if end_update=="latest" else "explicit_actor_endpoint",
            available_actor_updates=available,considered_actor_end_updates=tried,chosen_actor_end_update=actual_end,
            earliest_certifiable_actor_endpoint=earliest,additional_actor_updates_needed=max(0,actual_end-available),
            native_zero_baseline_fabricated=False),authoritative_clocks=c)
    chosen["limitations"] = [text for text in chosen["limitations"] if not text.startswith("Source019 eligible")]
    chosen["limitations"].extend([
        "Source021 changes only the frozen v19 recovery teaching bridge/costs; this stochastic window is descriptive and does not identify their causal benefit.",
        "Ten critic-only collections remain in native cumulative frame denominators, but the selected dense intervals and reported task actor endpoints exclude those updates.",
        "Static group command means cannot certify L0-specific nonzero goal exposure; actual L0 motion release and exposure are unavailable in this event schema."])
    return chosen


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("launch","first-ppo","inventory","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--reader020",type=Path,default=OLD020);p.add_argument("--reader019",type=Path,default=OLD019)
    p.add_argument("--updates",type=int,choices=(50,100),nargs="+",default=[50,100])
    p.add_argument("--end-update",default="latest",help="Actor updates in this attempt, excluding critic warmup")
    p.add_argument("--window-mode",choices=("after_ten_actor_baseline","after_critic_baseline"),default="after_ten_actor_baseline")
    p.add_argument("--max-read-mib",type=int,default=512);args=p.parse_args(argv)
    require(not args.output.exists(),"Output exists; preserve previous read")
    schema="source021-recovery-native-event-window-set-v1"
    try:
        require(digest(args.first_ppo)==FIRST_PPO_SHA and digest(args.inventory)==INVENTORY_SHA,"First-PPO/inventory admission bytes differ")
        require(1<=args.max_read_mib<=1024 and len(args.updates)==len(set(args.updates)),"Invalid byte cap/duplicate window")
        first=json.loads(args.first_ppo.read_text());inventory=json.loads(args.inventory.read_text())
        authority=authenticate(args,first,inventory);reader=make_reader(authority["clocks"],args.reader020,args.reader019)
        end="latest" if args.end_update=="latest" else int(args.end_update)
        last_step=authority["clocks"]["consumption_origin"]+authority["clocks"]["total_completed"]-1
        wanted=reader.wanted_tags(authority)|extra_tags(authority)
        series,audit=reader.parse_events(Path(authority["event"]),wanted,last_step,
            min_step=reader.BASELINE_STEP,max_read_bytes=args.max_read_mib*1024**2)
        windows={str(width):select_actor_window(reader,series,authority,audit,width,end,args.window_mode) for width in args.updates}
        require({name:digest(Path(authority["source_root"])/name) for name in authority["source_files_sha256"]}==authority["source_files_sha256"],"Frozen source changed during read")
        require(digest(args.reader020)==READER020_SHA and digest(args.reader019)==load020(args.reader020).BASE_READER_SHA256,
            "Frozen parser/metric dependency changed during read")
        paths={"launch":args.launch,"first_PPO":args.first_ppo,"inventory":args.inventory,
            "startup":Path(first["evidence"]["actual_worker_directory"])/"startup.json",
            "revision":Path(authority["source_root"]).parent/"source_revision_021.json",
            "predecessor":Path(authority["source_root"]).parent/"predecessor_021.json",
            "invocation":Path(first["evidence"]["actual_worker_directory"]).parent/"invocation.json"}
        require({name:digest(path) for name,path in paths.items()}==authority["metadata_sha256"],
            "Immutable admission/startup metadata changed during read")
        after_owner=native_owner(authority["worker_pid"],authority["invocation"]["command"])
        verify_owner_continuity(authority["native_owner"],after_owner)
        complete=all(row["complete"] for row in windows.values())
        report=dict(schema=schema,status="requested_actor_windows_complete" if complete else "pending_requested_actor_windows",
            complete=complete,source_commit=COMMIT,qualification=None,windows=windows,identity=authority,
            parser_metric_reuse=reader.adapter_reuse,source_files_unchanged=True,
            phase_scope="Ten critic-only updates consume budget and collect physical frames; selected windows contain actor updates only.",
            observation_limits="Stochastic PPO rollout summaries, not deterministic mean-policy evaluation or qualification. No actor-level advantages or causal benefit inferred.")
    except Exception as error:
        report=dict(schema=schema,status="read_failed",complete=False,source_commit=COMMIT,qualification=None,error=type(error).__name__+": "+str(error))
    report.update(queried_at_utc=datetime.now(timezone.utc).isoformat(),reader_sha256=digest(Path(__file__)),PPO_updates_by_reader=0,
        scope="read_existing_native_event_no_simulator_no_GPU_no_SSH_no_TensorBoard_server")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(report,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"status":report["status"],"complete":report["complete"],"output":str(args.output),"error":report.get("error")}))
    return 2 if report["status"]=="read_failed" else 0


if __name__=="__main__":
    raise SystemExit(main())
