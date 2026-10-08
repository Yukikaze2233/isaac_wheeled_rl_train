#!/usr/bin/env python3
"""Once-only read-only CPU/proc verification. No event parser, simulator or signals.

Missing launch/startup/immutable first-actor snapshots return waiting. Root saves
stdout as original evidence; this program never writes remotely or retries.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

DEFAULT_RUN = "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358"
DEFAULT_SOURCE = DEFAULT_RUN + "/source_serial_dispatch029_20261009"
PREVIOUS_SOURCE = DEFAULT_RUN + "/source_integrated_v26_20261008"
EXPECTED_COMMIT = "e5bd9c1367b875d7f229b9a1930647013d023372"
PARENT_SHA = "e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967"
PLAN_SHA = "db693283b744400ed5b6b586055f1514b56aea9106580c40a3bc3ee882d36d92"
WORKER_SHA = "941fb9e1f7a8ced3aa83fca76e82b7cf3064b594602d1e2900e56e0df0cc6b64"
UTILITY_SOURCE_SHA256 = "afbfa577d941887f4a4044f9719d5f8558d0c6631e0bf92774d18a2edec37b59"

class Waiting(Exception):
    pass

EVIDENCE = {}

def read(path):
    path = Path(path)
    if not path.is_file():
        raise Waiting("missing " + str(path))
    raw = path.read_bytes()
    EVIDENCE[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return json.loads(raw)

def sha(path):
    path = Path(path)
    if not path.is_file():
        raise Waiting("missing " + str(path))
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    EVIDENCE[str(path)] = {"sha256": value, "bytes": path.stat().st_size}
    return value

def flag(argv, name):
    require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "missing/duplicate " + name)
    return argv[argv.index(name) + 1]

def owners(run, source):
    rows = []
    scripts = {"chassis_remote_job.py": "wrapper", "run_task_campaign.py": "supervisor", "train_chassis.py": "worker", "evaluate_chassis.py": "evaluator"}
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            argv = [x.decode() for x in (directory / "cmdline").read_bytes().split(b"\0") if x]
            if not any(x == str(run) or x.startswith(str(run) + "/") for x in argv):
                continue
            entries = [x for x in argv if Path(x).name in scripts]
            if not entries:
                continue
            stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            require(stat[0] != "Z", "native owner is a zombie")
            rows.append(dict(pid=int(directory.name), ppid=int(stat[1]), starttime=int(stat[19]),
                role=scripts[Path(entries[0]).name], command=argv, entry=entries[0]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    counts = {role: sum(row["role"] == role for row in rows) for role in scripts.values()}
    require(all(counts[role] <= 1 for role in ("wrapper", "supervisor", "worker")) and counts["evaluator"] == 0, "duplicate owner or evaluator exists")
    if counts != dict(wrapper=1, supervisor=1, worker=1, evaluator=0):
        raise Waiting("native owner set not yet unique: " + repr(counts))
    require(all(Path(row["entry"]).parent == source / "scripts" for row in rows), "old/other source owner still exists")
    byrole = {row["role"]: row for row in rows}
    require(byrole["worker"]["ppid"] == byrole["supervisor"]["pid"] and
            byrole["supervisor"]["ppid"] == byrole["wrapper"]["pid"], "native owner parent chain differs")
    return byrole

def require(condition, message):
    if not condition:
        raise ValueError(message)

def equal(a, b):
    import torch
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b

def finite(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)

def full_learning_state_check(saved):
    """Require the complete native 9 actor, 8 critic and 17 Adam parameter state."""
    from collections import Counter
    import torch
    actor=saved.get('actor_state_dict', {}); critic=saved.get('critic_state_dict', {})
    optimizer=saved.get('optimizer_state_dict', {})
    require(isinstance(actor,dict) and len(actor)==9, 'Expected all 9 actor/std tensors')
    require(isinstance(critic,dict) and len(critic)==8, 'Expected all 8 critic tensors')
    tensors=list(actor.values())+list(critic.values())
    require(all(torch.is_tensor(x) and x.is_floating_point() and x.numel()>0 and bool(torch.isfinite(x).all()) for x in tensors),
            'Invalid/nonfinite actor or critic tensor')
    stdkeys=[key for key in actor if key == 'distribution.std_param']
    require(len(stdkeys)==1 and actor[stdkeys[0]].shape==(6,) and bool((actor[stdkeys[0]]>0).all()), 'Missing/invalid 6-axis actor std')
    groups=optimizer.get('param_groups', []); states=optimizer.get('state', {})
    require(isinstance(groups,list) and groups and isinstance(states,dict), 'Missing Adam group/state')
    params=[value for group in groups for value in group.get('params', [])]
    require(len(params)==17 and all(type(x) is int and x>=0 for x in params) and len(set(params))==17 and set(states)==set(params),
            'Adam does not cover exactly the unique 17 native parameters')
    require(finite(optimizer), 'Nonfinite Adam state/group')
    moments=[]
    for param in params:
        state=states[param]
        require(isinstance(state,dict) and {'step','exp_avg','exp_avg_sq'} <= set(state), 'Incomplete Adam moments/step')
        mean,square,step=state['exp_avg'],state['exp_avg_sq'],state['step']
        require(torch.is_tensor(mean) and torch.is_tensor(square) and mean.is_floating_point() and
                mean.shape==square.shape and mean.dtype==square.dtype and mean.numel()>0 and
                bool(torch.isfinite(mean).all()) and bool(torch.isfinite(square).all()) and bool((square>=0).all()),
                'Invalid Adam moment tensors')
        require((torch.is_tensor(step) and step.numel()==1 and bool(torch.isfinite(step).all()) and step.item()>=0) or
                (type(step) in (int,float) and math.isfinite(step) and step>=0), 'Invalid Adam step')
        moments.append(mean)
    require(Counter((tuple(x.shape),str(x.dtype)) for x in moments)==Counter((tuple(x.shape),str(x.dtype)) for x in tensors),
            'Adam moment shapes/dtypes do not cover actor and critic parameters')
    return {'actor_tensors':9,'critic_tensors':8,'std_axes':6,'Adam_parameter_states':17,'all_learning_tensors_finite':True}

def bounded_phase_checks(parent, initial, first_critic, first_actor, *, warmup,
                         consumption_origin, actor_consumption_origin):
    """Native checkpoint500 evidence: initial, first critic and first actor only."""
    import torch
    require(type(warmup) is int and 0 <= warmup <= 10, 'Invalid remaining critic warmup')
    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
        require(equal(parent[key], initial[key]), 'Initial full state differs: ' + key)
    require(len(initial['actor_state_dict']) == 9, 'Expected nine actor/std tensors')
    def clocks(saved, index, actors):
        expected = dict(successful_updates_total=consumption_origin + index,
                        actor_updates_consumed=actor_consumption_origin + actors,
                        task_actor_update=parent['infos']['task_actor_update'] + actors,
                        stage_actor_update=parent['infos']['stage_actor_update'] + actors,
                        learning_lineage_updates=parent['infos']['learning_lineage_updates'] + index,
                        critic_warmup_remaining=max(0, warmup - index))
        require(all(saved['infos'].get(k) == v for k, v in expected.items()), 'Clock/remaining mismatch at update ' + str(index))
    clocks(initial, 0, 0)
    if warmup:
        require(first_critic is not None, 'First critic snapshot absent')
        clocks(first_critic, 1, 0)
        require(equal(initial['actor_state_dict'], first_critic['actor_state_dict']), 'Actor/std moved at first critic update')
        require(not equal(initial['critic_state_dict'], first_critic['critic_state_dict']) and
                not equal(initial['optimizer_state_dict'], first_critic['optimizer_state_dict']), 'First critic/Adam update absent')
    clocks(first_actor, warmup + 1, 1)
    require(initial['actor_state_dict'].keys() == first_actor['actor_state_dict'].keys(), 'Actor catalog differs')
    changed = [k for k, value in initial['actor_state_dict'].items() if not torch.equal(value, first_actor['actor_state_dict'][k])]
    require(len(changed) == 9, 'First actor update did not change all nine actor/std tensors')
    baseline = first_critic if warmup else initial
    require(not equal(baseline['critic_state_dict'], first_actor['critic_state_dict']) and
            not equal(baseline['optimizer_state_dict'], first_actor['optimizer_state_dict']), 'First actor prefix critic/Adam did not change')
    return dict(critic_tensor_observed_indices=[1] if warmup else [], first_actor_index=warmup + 1,
                actor_changed=changed, intermediate_actor_tensor_observation='unobserved_not_sealed_by_native_checkpoint500',
                critic_Adam_between_observed_seals='changed; intermediate changes are not individually tensor-observed')

def authenticate_saved_admission(config, receipt, plan, root, train, binding):
    """Authenticate immutable entry snapshots after launch; live charges may grow.

    Production authenticate_receipt deliberately requires zero post-entry live
    charges and is only valid before launching a worker. Do not call it here or
    weaken it: independently verify its immutable identities and allowance, then
    check current consumption monotonically below.
    """
    from wheeled_tasks.chassis import integrated_production as production
    from wheeled_tasks.chassis import integrated_lineage as lineage
    from wheeled_tasks.chassis import integrated_runtime as runtime
    from wheeled_tasks.chassis.task_campaign import account_updates
    require(production.validate_static_worker(config, root) == "speed", "worker is not static integrated speed")
    production.validate_plan(plan, root)
    require(receipt.get("schema") == production.ADMISSION_SCHEMA and receipt.get("qualification") is None and
            receipt.get("freeze_ready") is False and receipt["campaign_plan"] ==
            dict(file="contracts/v6_task_campaign_v26.json", sha256=PLAN_SHA) and
            receipt["lineage_manifest"] == plan["integrated_continuation"]["lineage_manifest"],
            "immutable admission campaign/lineage identity differs")
    selection = runtime._verified_selection(binding, receipt["selection"])
    require(receipt["entry_admission"] == selection["entry_admission"] and
            receipt["target_contract_sha256"] == selection["contract_sha256"] == WORKER_SHA and
            receipt["component_clocks_at_source"] == runtime.source_clocks(binding, selection),
            "immutable admission source/target/component binding differs")
    account_path = Path(plan["budget_source_stage"]) / (plan["campaign_id"] + ".account.json")
    snapshots = {}
    for key, authority in (("account", account_path), ("state", train / "campaign_state.json")):
        pin = receipt[key]
        snapshot = Path(pin["file"]).resolve()
        live_pin = receipt.get("authoritative_" + key, pin)
        require(Path(live_pin["file"]).resolve() == authority.resolve() and snapshot.is_relative_to(train) and
                snapshot != authority.resolve() and sha(snapshot) == pin["sha256"],
                "worker admission requires an immutable " + key + " snapshot bound to its live authority")
        snapshots[key] = read(snapshot)
    account = snapshots["account"]
    require(account.get("campaign_id") == plan["campaign_id"] and account.get("run_directory") == str(train) and
            account.get("contract_sha256") == PLAN_SHA and account.get("rollback_refunds_consumed_updates") is False and
            all(account.get(k) == v for k,v in account_updates(plan, account["attempts"]).items()),
            "admission account snapshot identity/arithmetic differs")
    records = receipt["reconciled_attempts"]
    require(account_updates(plan, records) == receipt["reconciled_account"] == account_updates(plan, account["attempts"]),
            "admission reconciled attempts would refund or duplicate work")
    expected = lineage.remaining_budget(binding, selection, records,
        requested_actor_updates=receipt["allowance"]["requested_actor_updates"])
    endpoint = plan["domain_completion"]["schedule"]["end_actor_updates"]["speed"]
    require(receipt["allowance"] == expected and expected["admitted"] and receipt["actor_endpoint"] == endpoint and
            expected["requested_actor_updates"] == min(endpoint - selection["fresh_domain_actor_updates"], expected["maximum_actor_updates"]),
            "admission historical caps/actor allowance differ")
    state = snapshots["state"]
    require(state.get("authorized_REC_early_stop") == selection["entry_admission"]["record"],
            "immutable admission state does not contain corrected origin")
    live = read(account_path)
    require(live.get("contract_sha256") == PLAN_SHA and live.get("campaign_id") == plan["campaign_id"] and
            live.get("rollback_refunds_consumed_updates") is False and
            all(live.get(k) == v for k,v in account_updates(plan, live["attempts"]).items()),
            "live account identity/arithmetic differs")
    for key, before in records.items():
        after = live["attempts"].get(key)
        require(after is not None and after["domain"] == before["domain"] and
                all(after[field] >= before[field] for field in ("consumed_updates", "actor_updates")),
                "live account lost charged history: " + key)
    return dict(status="immutable_admission_authenticated_after_launch", live_growth_allowed=True,
        admission_totals=receipt["reconciled_account"], observed_live_totals=account_updates(plan, live["attempts"]))


def inspect(args):
    import torch
    run, source = args.run_root.resolve(), args.source_root.resolve()
    train = run / "train"
    sys.path[:0] = [str(source / "src"), str(source / "scripts")]
    from wheeled_tasks.chassis import integrated_lineage as lineage
    from wheeled_tasks.chassis import integrated_runtime as runtime
    from wheeled_tasks.chassis import integrated_production as production
    from wheeled_tasks.chassis.task_campaign import account_updates
    require(len(args.source_commit) == 40 and all(c in "0123456789abcdef" for c in args.source_commit),
            "Pin the actual frozen source029 commit before running this verifier")
    revision = read(run / "source_revision_029.json")
    require(revision.get("schema") == "task-source-revision-v1" and
            revision.get("source_commit") == args.source_commit and
            revision.get("new_source_root") == str(source) and
            revision.get("old_source_root") == PREVIOUS_SOURCE and
            revision.get("old_contract_sha256") == revision.get("new_contract_sha256") == PLAN_SHA,
            "source029 is not the pinned code-only same-v26 revision")
    launch = read(run / "source029_deploy/launch.receipt.json")
    require(launch.get("status") == "launch_submitted", "source029 launch was not submitted")
    launch_intent = read(run / "source029_deploy/launch.intent.json")
    staged = read(run / "source029_deploy/stage.json")
    preflight = read(run / "source029_deploy/preflight.json")
    entry_before_launch = read(run / "source029_deploy/admission_before_launch.json")
    require(launch.get("source_commit") == args.source_commit and
            launch.get("actual_dispatch") == ["speed", "terrain", "jump", "stepjump"] and
            launch_intent.get("command") == launch.get("command") and
            launch_intent.get("source_revision_sha256") == sha(run / "source_revision_029.json") and
            launch_intent.get("admission_sha256") == sha(run / "source029_deploy/admission_before_launch.json"),
            "source029 launch intent/command/source/admission binding differs")
    require(staged.get("schema") == "source029-staging-v1" and staged.get("source_commit") == args.source_commit and
            staged.get("receipt_sha256") == sha(run / "source_revision_029.json") and
            staged.get("source_archive_sha256") == revision.get("source_archive_sha256") and
            staged.get("native_scenes_created") == staged.get("PPO_updates") == 0 and
            preflight.get("status") == "CPU_preflight_passed" and
            preflight.get("actual_dispatch") == ["speed", "terrain", "jump", "stepjump"] and
            preflight.get("source_revision_sha256") == sha(run / "source_revision_029.json") and
            preflight.get("native_scenes_created") == preflight.get("PPO_updates") == 0,
            "source029 code-only staging/CPU dispatch verification differs")
    # Every frozen contract, including control/mechanics, must remain byte-identical.
    previous = Path(PREVIOUS_SOURCE)
    old_contracts = {str(p.relative_to(previous / "contracts")): p for p in (previous / "contracts").rglob("*") if p.is_file()}
    new_contracts = {str(p.relative_to(source / "contracts")): p for p in (source / "contracts").rglob("*") if p.is_file()}
    require(old_contracts.keys() == new_contracts.keys() and old_contracts, "code-only contract catalog differs")
    for name, old_file in old_contracts.items():
        require(sha(old_file) == sha(new_contracts[name]), "code-only contract bytes differ: " + name)
    require(sha(source / "contracts/v6_task_campaign_v26.json") == PLAN_SHA and
            sha(source / "contracts/v6_task_campaign_speed_worker_v26.json") == WORKER_SHA,
            "frozen campaign/worker contract differs")
    # Reuse prior physics engineering evidence honestly: its REC14169 parent is
    # not the new REC14188 learning parent. Actual worker startup is checked below.
    native_dir = run / "boundary_integrated_028_cli_repair001.native"
    native = read(run / "boundary_integrated_028_completion_adoption002.native.receipt.json")
    require(native.get("additional_native_runs") == 0 and native.get("status") == "passed" and
            native.get("exit_code") == 0 and native.get("PPO_updates") == 0,
            "prior unmodified-v26 engineering startup evidence is not passed")
    for relative, expected in native["repair_evidence_pins"].items():
        require(sha(run / relative) == expected, "prior native evidence differs: " + relative)
    scene = read(native_dir / "scene/completion.json")
    require(sha(native_dir / "scene/completion.json") == native["report_sha256"] and
            scene.get("status") == "scene_validated" and scene.get("successful_updates") == 0 and
            scene.get("attempted_updates") == 0, "prior scene report binding differs")
    native_selection = read(native_dir / "source_selection.json")
    require(sha(native_dir / "source_selection.json") == native["selection_sha256"] and
            native_selection["selection"]["checkpoint"]["sha256"] ==
            "ed24fe4e5e62fd8c8e01f81427c0741ae46799a237dee2b0fbd4b288df4e1085",
            "prior native parent is not the preserved REC14169 engineering source")
    correction_dir = train / "integrated_early_origin_v26_dispatch_correction029"
    correction = read(correction_dir / "receipt.json")
    intent = read(correction_dir / "intent.json")
    require(correction.get("status") == "corrected" and correction.get("PPO_updates_by_repair") == 0 and
            correction.get("qualification") is None and correction.get("retained_new_actor_updates") == 19 and
            correction.get("consumed_updates") == 41393 and correction.get("actor_updates_consumed") == 41223 and
            correction.get("REC_actor") == 14188 and correction.get("checkpoint", {}).get("sha256") == PARENT_SHA,
            "origin correction does not preserve exact latest nineteen REC actor updates")
    require(intent.get("schema") == "known-integrated-dispatch-origin-correction029-v1" and
            intent.get("action") == "retain_latest_misdirected_REC_then_joint_chain" and
            intent.get("checkpoint") == correction["checkpoint"] and
            sha(correction_dir / "intent.json") == correction.get("intent_sha256") and
            sha(correction_dir / "admission.json") == correction.get("admission_sha256"),
            "origin correction immutable intent/admission binding differs")
    for key, pin in intent["before"].items():
        require(sha(correction_dir / ("before_" + key + ".json")) == pin["sha256"],
                "origin correction previous snapshot changed: " + key)
    plan = read(source / "contracts/v6_task_campaign_v26.json")
    binding = production.binding_for(plan, source, train)
    current_owners = owners(run, source)
    require(launch["pid"] == current_owners["wrapper"]["pid"], "launch wrapper PID differs")
    worker = current_owners["worker"]
    directory = Path(flag(worker["command"], "--run-dir")).resolve()
    require(directory.is_relative_to(train / "speed") and directory.name == "train", "worker is not new speed attempt")
    invocation = read(directory.parent / "invocation.json")
    require(invocation["command"] == worker["command"], "worker argv differs from native invocation")
    require(flag(worker["command"], "--contract") == str(source / "contracts/v6_task_campaign_speed_worker_v26.json"), "speed worker contract differs")
    source_file = Path(flag(worker["command"], "--integrated-source-selection-file"))
    receipt = read(source_file)
    require(sha(source_file) == flag(worker["command"], "--integrated-source-selection-sha256"), "worker selection file SHA differs")
    config = read(source / "contracts/v6_task_campaign_speed_worker_v26.json")
    saved_admission = authenticate_saved_admission(config, receipt, plan, source, train, binding)
    selection, allowance = receipt["selection"], receipt["allowance"]
    require(selection["mode"] == "rebase_from_authorized_REC" and selection["domain"] == "speed" and
            selection["fresh_joint_actor_updates"] == selection["fresh_domain_actor_updates"] == 0 and
            selection["critic_warmup_remaining"] == 10, "first speed entry was not a fresh REC rebase")
    runtime._verified_selection(binding, selection)
    require(selection["entry_admission"] == correction["entry_admission"] and
            selection == entry_before_launch["selection"] and allowance == entry_before_launch["allowance"] and
            selection["checkpoint"] == correction["checkpoint"], "worker does not use corrected latest REC origin")
    require(selection["origin_recovery_actor_update"] == 14188 and
            allowance["consumed_updates"] == 41393 and allowance["actor_updates_consumed"] == 41223 and
            allowance["requested_total_updates"] == 14907 and allowance["requested_actor_updates"] == 14897,
            "fresh speed clocks/caps differ or old REC work was refunded")
    corrected_record = selection["entry_admission"]["record"]
    require(corrected_record["early_stop_origin_correction"] ==
            dict(file=str(correction_dir / "intent.json"), sha256=sha(correction_dir / "intent.json")),
            "selection lacks immutable origin-correction binding")
    for key, name in (("early_stop_campaign_state", "campaign_state.json"),
                      ("early_stop_campaign_completion", "completion.json")):
        require(corrected_record[key] == dict(file=str(correction_dir / name), sha256=sha(correction_dir / name)),
                "new origin stopped snapshot differs: " + name)
    # Native worker's own assertions replace any claim that the old REC14169
    # engineering probe tested this new REC14188 actor state.
    startup = read(directory / "startup.json")
    startup_args = startup.get("args", {})
    require(startup.get("status") == "running" and startup.get("contract_sha256") == WORKER_SHA and
            startup.get("integrated_admission_sha256") == sha(source_file) and
            startup_args.get("integrated_source_selection_file") == str(source_file) and
            startup_args.get("integrated_source_selection_sha256") == sha(source_file) and
            startup_args.get("num_envs") == 4096 and startup_args.get("task_actor_update") == 0 and
            startup_args.get("validate_scene") == 0,
            "actual integrated worker startup does not bind fresh speed admission")
    require(read(directory / "integrated_admission.json") == receipt, "worker saved admission differs")
    require(startup_args.get("resume_task_change") is True and startup_args.get("resume") == selection["checkpoint"]["path"] and
            startup_args.get("seed") == 290927 and startup_args.get("block_index") == 0 and
            startup_args.get("learning_rate_scale") == 1.0 and startup_args.get("global_actor_update_offset") == 0 and
            startup_args.get("consumed_updates") == 41393 and startup_args.get("actor_updates_consumed") == 41223 and
            startup_args.get("stage_actor_update") == 27556 and startup_args.get("updates") == 14907 and
            startup_args.get("contract") == str(source / "contracts/v6_task_campaign_speed_worker_v26.json") and
            startup_args.get("run_dir") == str(directory),
            "worker startup invocation differs from its admitted full-state clocks/cap")
    for relative, expected in startup["source_sha256"].items():
        require(sha(source / relative) == expected and sha(directory / "source" / relative) == expected,
                "actual worker startup source snapshot differs: " + relative)
    native_runtime = startup["startup"]
    timing = native_runtime["control_timing"]
    require(native_runtime.get("num_envs") == 4096 and native_runtime.get("gpu_dynamics_enabled") is True and
            timing.get("physics_dt") == timing.get("pc_control_dt") == 0.005 and
            timing.get("policy_dt") == 0.02 and timing.get("decimation") == 4,
            "actual worker scene count/control timing differs")
    runtime_preflight = native_runtime["runtime_preflight"]
    require(runtime_preflight.get("status") == "runtime_configuration_passed", "actual worker runtime assertions did not pass")
    design = read(directory / "integrated_design.json")
    require(design["source_receipt"] == receipt, "actual worker design does not use this source")
    effective = production.static_training_config(config, source)
    effective["integrated_runtime_candidate"].pop("static_recipe_preflight")
    effective["integrated_runtime_candidate"].update(learning_source=deepcopy(selection),
        clocks_at_source=deepcopy(receipt["component_clocks_at_source"]))
    from wheeled_tasks.chassis.integrated_preflight import validate_integrated_runtime_state
    rechecked_startup = validate_integrated_runtime_state(design["runtime_expected"],
        runtime_preflight["observed"], config=effective, num_envs=4096, actor_update=0)
    require(rechecked_startup == runtime_preflight, "actual worker runtime assertions do not reproduce from saved state")
    parent_path = Path(selection["checkpoint"]["path"])
    require(invocation["source_checkpoint"] == str(parent_path) and
            invocation["source_checkpoint_sha256"] == selection["checkpoint"]["sha256"] and
            flag(worker["command"], "--resume") == str(parent_path), "worker selected a flat/old speed branch")
    require(parent_path == train / "recovery/attempt_010/train/model_final.pt" and sha(parent_path) == PARENT_SHA,
            "actual parent is not the authenticated latest REC41393 final")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    require(all(parent["infos"].get(k) == v for k, v in dict(successful_updates_total=41393,
        actor_updates_consumed=41223, task_actor_update=14188, stage_actor_update=27556,
        learning_lineage_updates=27646, critic_warmup_remaining=0).items()),
        "latest REC parent clock/full-state identity differs")
    lineage._checkpoint(binding, selection["checkpoint"], "recovery", binding.read()["recovery_entry"]["contract_sha256"])
    parent_architecture = full_learning_state_check(parent)
    origin = parent["infos"]["task_actor_update"]
    require(origin == selection["origin_recovery_actor_update"] and origin < 15000, "early REC origin was fabricated")
    resume = read(directory / "resume_verification.json")
    require(resume.get("parent_checkpoint_sha256") == selection["checkpoint"]["sha256"] and
            all(resume.get(name) is True for name in ("actor_exact", "critic_exact", "optimizer_exact")), "native restore exactness absent")
    initial_path = directory / "initial_state/model_final.pt"
    read(initial_path.parent / "completion.json")
    initial_ref = dict(path=str(initial_path), sha256=sha(initial_path))
    require(production._initial_learning_state(binding, "speed", initial_ref, selection["entry_admission"]), "initial saved learning state not authenticated")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        require(equal(parent[key], initial[key]), "initial full restore differs: " + key)
    def snapshot(index):
        total = allowance["consumed_updates"] + index
        path = directory / "checkpoints" / ("update_%08d" % total) / "model.pt"
        reference = dict(path=str(path), sha256=sha(path))
        checkpoint = lineage._checkpoint(binding, reference, "speed", selection["contract_sha256"])
        saved = torch.load(path, map_location="cpu", weights_only=True)
        full_learning_state_check(saved)
        actors = max(0, index - 10)
        expected = dict(successful_updates_total=total, actor_updates_consumed=allowance["actor_updates_consumed"] + actors,
            stage_actor_update=parent["infos"]["stage_actor_update"] + actors, task_actor_update=actors,
            learning_lineage_updates=parent["infos"]["learning_lineage_updates"] + index,
            critic_warmup_remaining=max(0, 10-index))
        require(all(saved["infos"].get(key) == value for key,value in expected.items()), "immutable prefix clock mismatch at " + str(index))
        doc = saved["infos"]["integrated_runtime_clocks"]
        require(doc["origin_recovery_actor_update"] == origin and
                doc["component_actor_updates"] == dict(recovery=origin + actors, speed=actors), "component origin/actor clock mismatch")
        return saved, dict(path=str(path), sha256=reference["sha256"], index=index, clocks=expected,
                           integrated_runtime_clocks=doc, seal=checkpoint["seal"])
    first_critic, critic_record = snapshot(1)
    require(equal(initial["actor_state_dict"], first_critic["actor_state_dict"]), "actor changed in critic-only prefix")
    require(not equal(initial["critic_state_dict"], first_critic["critic_state_dict"]) and
            not equal(initial["optimizer_state_dict"], first_critic["optimizer_state_dict"]), "first critic/Adam learning absent")
    first_actor, actor_record = snapshot(11)
    fresh_parent = dict(parent, infos=dict(parent["infos"], task_actor_update=0))
    phases = bounded_phase_checks(fresh_parent, initial, first_critic, first_actor, warmup=10,
        consumption_origin=allowance["consumed_updates"], actor_consumption_origin=allowance["actor_updates_consumed"])
    changed = [key for key,value in initial["actor_state_dict"].items() if not equal(value, first_actor["actor_state_dict"][key])]
    require(any(key != "distribution.std_param" for key in changed), "first actor mean parameters did not change")
    require(not equal(first_critic["critic_state_dict"], first_actor["critic_state_dict"]) and
            not equal(first_critic["optimizer_state_dict"], first_actor["optimizer_state_dict"]), "first actor endpoint lacks critic/Adam learning")
    progress = read(directory / "progress.json")
    require(progress.get("successful_updates",0) >= 11 and progress.get("task_actor_update",0) >= 1,
            "current worker progress precedes first actor seal")
    account_path = Path(plan["budget_source_stage"]) / (plan["campaign_id"] + ".account.json")
    account = read(account_path)
    computed = account_updates(plan, account["attempts"])
    require(all(account.get(key) == value for key,value in computed.items()) and
            account.get("rollback_refunds_consumed_updates") is False, "authoritative account arithmetic/refund differs")
    if computed["consumed_updates"] < allowance["consumed_updates"] + 11 or computed["actor_updates_consumed"] < allowance["actor_updates_consumed"] + 1:
        raise Waiting("first actor seal exists; publisher has not yet charged that full prefix")
    require(owners(run, source) == current_owners, "native owners changed during bounded read")
    return dict(status="passed", source_commit=args.source_commit, parent_checkpoint=selection["checkpoint"],
        actual_origin_recovery_actor_update=origin, architecture=parent_architecture, exact_restore=resume,
        source_revision=revision, origin_correction=correction, saved_admission_verification=saved_admission,
        prior_native_engineering_evidence=dict(receipt=native, parent_REC_actor=14169,
            source_checkpoint_sha256=native_selection["selection"]["checkpoint"]["sha256"],
            scope="unchanged_v26_engineering_prior_not_new_REC14188_actor_validation"),
        own_worker_startup=dict(file=str(directory / "startup.json"), sha256=sha(directory / "startup.json"),
            status=startup["status"], runtime_preflight_status=runtime_preflight["status"],
            num_envs=4096, policy_hz=50, physics_feedback_PD_hz=200, contract_catalog_unchanged=len(old_contracts)),
        owners=current_owners, selection=selection, bounded_phase_checks=phases, first_critic=critic_record,
        first_actor=actor_record, actor_changed_tensor_names=changed, charged_critic_updates_before_first_actor=10,
        critic_actor_tensor_observation="initial and update1 exact; updates2-10 have no native immutable snapshots and are not claimed individually observed",
        RNG_scope="complete source RNG authenticated by checkpoint admission; initial state follows declared new-episode reset, so post-reset RNG equality is not asserted",
        current_progress=progress, authoritative_account=computed, metadata_read_scope="bounded separate live progress/account reads; not an atomic runtime snapshot")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,default=Path(DEFAULT_RUN))
    parser.add_argument("--source-root",type=Path,default=Path(DEFAULT_SOURCE))
    parser.add_argument("--source-commit",default=EXPECTED_COMMIT)
    args=parser.parse_args()
    result=dict(schema="source029-code-only-first-speed-actor-readonly-verification-v1",queried_at_utc=datetime.now(timezone.utc).isoformat(),
        PPO_updates_by_reader=0,GPU_calls=0,native_runs=0,SSH_calls=0,signals_sent=0,remote_writes=0,qualification=None,
        reused_source023_utility_SHA256=UTILITY_SOURCE_SHA256)
    try:
        result.update(inspect(args))
    except Waiting as error:
        result.update(status="waiting",reason=str(error))
    except Exception as error:
        result.update(status="attention",reason=type(error).__name__ + ": " + str(error))
    result["raw_evidence_file_pins"]=EVIDENCE
    result["finished_at_utc"]=datetime.now(timezone.utc).isoformat()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 0 if result["status"] in ("passed","waiting") else 1

if __name__ == "__main__":
    raise SystemExit(main())
