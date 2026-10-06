#!/usr/bin/env python3
"""Read existing DiagnosticUpdate scalars through the unchanged source021 CRC reader.

No producer hook, simulator, torch, GPU, SSH, optimizer or TensorBoard server.
This sidecar fixes a previous reader's tag-selection omission, not the training log.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_READER = HERE.parent / "learning_audit/recovery_window_candidate_021/read_recovery_window_021.py"
READER021_SHA = "5faa75e24b0dd484bdacb0273756642caab8fcc6d6e66f90c034a0bda6ac19a6"
SOURCE_FILES = (
    "src/wheeled_tasks/chassis/diagnostic_curriculum.py",
    "src/wheeled_tasks/chassis/scut_observation.py",
    "src/wheeled_tasks/chassis/v5_control.py",
    "src/wheeled_tasks/agents/v40_ppo_cfg.py",
)
EXPECTED_CONTROL_ORDER = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
EXPECTED_POLICY_FROM_CONTROL = (0, 1, 3, 4, 2, 5)
EXPECTED_CONTROL_FROM_POLICY = (0, 1, 4, 2, 3, 5)
STD_TAGS = tuple(f"DiagnosticUpdate/action_std_{i}" for i in range(6))
GLOBAL_KL = "DiagnosticUpdate/kl_rollout_probe"
AGGREGATE_STD = "Policy/mean_std"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reader(path):
    require(digest(path) == READER021_SHA, "Unchanged source021 reader SHA differs")
    spec = importlib.util.spec_from_file_location("source021_std_sidecar_dependency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wanted_tags(groups):
    require(len(groups) == len(set(groups)) == 95, "Expected admitted static95 groups")
    return set(STD_TAGS + (GLOBAL_KL, AGGREGATE_STD)) | {
        f"DiagnosticUpdate/{group}/kl" for group in groups}


def actor_window(clocks):
    """The same actor11..110 window as the complete source021 first100 read."""
    require(clocks["consumption_origin"] == 29996 and clocks["task_actor_origin"] == 3075
        and clocks["critic_warmup_updates"] == 10
        and clocks["first_actor_event_step"] == 30006,
        "Actual parent/critic/event clocks differ from admitted source021 window")
    actor_updates = list(range(11, 111))
    steps = [clocks["first_actor_event_step"] + actor - 1 for actor in actor_updates]
    return dict(attempt_actor_updates_inclusive=[11, 110],
        task_actor_updates_inclusive=[3086, 3185], event_steps_inclusive=[30016, 30115],
        dense_steps=steps, expected_records_per_tag=100, critic_updates_excluded=10,
        authoritative_parent_consumption=29996, authoritative_parent_task_actor=3075,
        early_ten_actor_updates_excluded_as_in_original_first100=True,
        actor_updates_needed=max(0, 110 - clocks["actor_completed"]))


def literal_assignment(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
            for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Source literal absent: " + name)


def action_layout(source_root, config, binding):
    """Bind policy-axis interpretation to admitted bytes and native geometry."""
    mapping = ast.parse((source_root / SOURCE_FILES[1]).read_text())
    control = ast.parse((source_root / SOURCE_FILES[2]).read_text())
    policy_from_control = tuple(literal_assignment(mapping, "POLICY_FROM_CONTROL"))
    control_from_policy = tuple(literal_assignment(mapping, "CONTROL_FROM_POLICY"))
    active = tuple(literal_assignment(control, "ACTIVE"))
    require(policy_from_control == EXPECTED_POLICY_FROM_CONTROL
        and control_from_policy == EXPECTED_CONTROL_FROM_POLICY
        and active == EXPECTED_CONTROL_ORDER
        and tuple(config["v5_control"]["active_order"]) == active, "Native policy/control action ordering differs")
    settings = config["v5_control"]
    require((settings["leg_position_scale"], settings["wheel_velocity_scale"],
        settings["action_clip"], settings["wheel_action_clip"]) == (.25, 10., 3., 9.),
        "Native action scale/clipping differs")
    require(binding["status"] == "authenticated_native_fsm_and_geometry"
        and binding["wheel_radius_m"] == .06, "Native wheel radius binding differs")
    axes = []
    for axis, control_axis in enumerate(policy_from_control):
        is_wheel = control_axis in (2, 5)
        scale = settings["wheel_velocity_scale"] if is_wheel else settings["leg_position_scale"]
        clip = settings["wheel_action_clip"] if is_wheel else settings["action_clip"]
        axes.append(dict(policy_axis=axis, control_axis=control_axis, joint_name=active[control_axis],
            kind="wheel_velocity_target" if is_wheel else "leg_position_target_offset",
            raw_action_unit="dimensionless", target_unit="rad/s" if is_wheel else "rad",
            linear_target_scale=scale, action_clip_absolute=clip,
            clipped_target_absolute_bound=scale * clip,
            wheel_rim_scale_m_s_per_action=scale * binding["wheel_radius_m"] if is_wheel else None))
    return dict(policy_from_control=list(policy_from_control), control_from_policy=list(control_from_policy),
        axes=axes, native_gate_binding=binding,
        interpretation="std*scale is the un-clipped linear target spread; clipping, angle wrapping, delay, transport, PD and linkage alter executed response. It is not joint velocity, body velocity, torque or gyro std.")


def scale_std(value, axis):
    require(value is None or math.isfinite(value) and value > 0, "Native std must be positive finite")
    return dict(raw_action_std=value,
        unclipped_linear_target_std=None if value is None else value * axis["linear_target_scale"],
        unclipped_wheel_rim_target_std_m_s=None if value is None or axis["wheel_rim_scale_m_s_per_action"] is None
            else value * axis["wheel_rim_scale_m_s_per_action"])


def summarize(reader, series, authority, layout, audit):
    window = actor_window(authority["clocks"])
    steps = window["dense_steps"]
    missing = []

    def stats(tag, selected):
        row = reader.stats(series, tag, selected)
        row["missing_steps"] = [step for step in selected if reader.scalar(series, tag, step) is None]
        return row

    metrics = {}
    for tag in sorted(wanted_tags(authority["groups"])):
        full = stats(tag, steps)
        if full["missing_steps"]:
            missing.append(dict(tag=tag, missing_steps=full["missing_steps"]))
        points = [{"event_step": step, "attempt_actor_update": step - 30006 + 1,
            "task_actor_update": step - 30006 + 1 + 3075,
            "value": reader.scalar(series, tag, step),
            "native_wall_time": series.get(tag, {}).get(step, [None, None])[1]} for step in steps]
        metrics[tag] = dict(full100=full, first50=stats(tag, steps[:50]),
            last50=stats(tag, steps[50:]), points=points)
    axes = []
    for axis in layout["axes"]:
        metric = metrics[STD_TAGS[axis["policy_axis"]]]
        axes.append(dict(**axis, tag=STD_TAGS[axis["policy_axis"]],
            full100=scale_std(metric["full100"]["mean"], axis),
            first50=scale_std(metric["first50"]["mean"], axis),
            last50=scale_std(metric["last50"]["mean"], axis)))
        for point in metric["points"]:
            point.update(scale_std(point["value"], axis))
    aggregate_checks = []
    kl_checks = []
    weights = {group: min(16, authority["population"][group]) for group in authority["groups"]}
    for step in steps:
        values = [reader.scalar(series, tag, step) for tag in STD_TAGS]
        aggregate = reader.scalar(series, AGGREGATE_STD, step)
        if aggregate is not None and all(value is not None for value in values):
            expected = sum(values) / 6
            aggregate_checks.append(dict(event_step=step, observed=aggregate, per_axis_mean=expected,
                residual=aggregate - expected, within_float32_roundoff=math.isclose(aggregate, expected, rel_tol=2e-6, abs_tol=1e-7)))
        group_values = {group: reader.scalar(series, f"DiagnosticUpdate/{group}/kl", step)
            for group in authority["groups"]}
        overall = reader.scalar(series, GLOBAL_KL, step)
        if overall is not None and all(value is not None for value in group_values.values()):
            expected = sum(group_values[group] * weight for group, weight in weights.items()) / sum(weights.values())
            kl_checks.append(dict(event_step=step, observed=overall, stratified_probe_weighted_mean=expected,
                residual=overall - expected, within_float32_roundoff=math.isclose(overall, expected, rel_tol=2e-5, abs_tol=1e-7)))
    complete = not missing and window["actor_updates_needed"] == 0
    return dict(schema="source021-existing-native-per-axis-std-KL-sidecar-v1",
        status="complete_native100_per_axis_std_and_KL" if complete else "pending_native100_per_axis_std_and_KL",
        complete=complete, qualification=None, window=window, metrics=metrics, physical_target_scales=axes,
        missing_required_tags=missing, observed_native_actor_updates=authority["clocks"]["actor_completed"],
        diagnostic_probe=dict(groups=95, selected_lanes_per_group=weights, selected_lanes=sum(weights.values()),
            observation="last rollout observation; same stratified observations before and after PPO update; no sampling in probe",
            KL="KL(old||new) summed over six raw action coordinates, then averaged over the probe lanes; not episode success or simulated trajectory divergence"),
        aggregate_std_reconciliation=dict(comparable_steps=len(aggregate_checks), checks=aggregate_checks,
            all_comparable_passed=all(row["within_float32_roundoff"] for row in aggregate_checks) if aggregate_checks else None,
            scope="Supplementary consistency check; Policy/mean_std is not substituted for a missing per-axis tag."),
        global_group_KL_reconciliation=dict(comparable_steps=len(kl_checks), checks=kl_checks,
            all_comparable_passed=all(row["within_float32_roundoff"] for row in kl_checks) if kl_checks else None,
            scope="Group means weighted by the native probe's min(16,population) selected lanes, not full environment population."),
        action_layout=layout, native_CRC_region=audit,
        limitations=["These existing tags were omitted by the old50/100 reader; absence from that report did not establish absence from TensorBoard.",
            "Logged std is post-update policy distribution std on the probe observations; it is not sampled action residual or executed motion noise.",
            "Unclipped target scales cannot be compared directly to wheel/leg/gyro velocity gates; leg targets are positions, and the closed loop, delay, clipping and passive forces change motion.",
            "Per-axis std and KL alone cannot establish why a continuous1s native gate fails or identify entropy causality.",
            "This sidecar retains the exact actor11..110 window and excludes ten critic collections; no parent clock, budget, training or qualification changes."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("launch", "first-ppo", "inventory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reader021", type=Path, default=DEFAULT_READER)
    parser.add_argument("--reader020", type=Path, default=HERE.parent / "learning_audit/read_recovery_window_020.py")
    parser.add_argument("--reader019", type=Path, default=HERE.parent / "learning_audit/read_recovery_window_019.py")
    parser.add_argument("--max-read-mib", type=int, default=512)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Output exists; preserve previous read")
    try:
        helper = load_reader(args.reader021)
        require(digest(args.first_ppo) == helper.FIRST_PPO_SHA and digest(args.inventory) == helper.INVENTORY_SHA,
            "First-PPO/inventory admission bytes differ")
        require(1 <= args.max_read_mib <= 1024, "Invalid read byte cap")
        first, inventory = json.loads(args.first_ppo.read_text()), json.loads(args.inventory.read_text())
        authority = helper.authenticate(args, first, inventory)
        source = Path(authority["source_root"])
        extra_source_hashes = {name: digest(source / name) for name in SOURCE_FILES}
        require(extra_source_hashes == {name: inventory[name] for name in SOURCE_FILES},
            "Frozen source diagnostic producer/action interpretation bytes differ")
        config = json.loads((source / helper.WORKER).read_text())
        binding = first["checks"]["actual_native_full_restore_new_modules_origins_and_clocks"]["detail"]["native_hold_cost_gate_binding"]
        layout = action_layout(source, config, binding)
        reader = helper.make_reader(authority["clocks"], args.reader020, args.reader019)
        window = actor_window(authority["clocks"])
        series, audit = reader.parse_events(Path(authority["event"]), wanted_tags(authority["groups"]),
            window["dense_steps"][-1], min_step=window["dense_steps"][0],
            max_read_bytes=args.max_read_mib * 1024**2)
        report = summarize(reader, series, authority, layout, audit)
        all_source = {**authority["source_files_sha256"], **extra_source_hashes}
        require({name: digest(source / name) for name in all_source} == all_source, "Frozen source changed during read")
        require(digest(args.reader021) == READER021_SHA and digest(args.reader020) == helper.READER020_SHA
            and digest(args.reader019) == helper.load020(args.reader020).BASE_READER_SHA256,
            "Frozen reader dependency changed during read")
        run = Path(first["evidence"]["actual_worker_directory"])
        paths = dict(launch=args.launch, first_PPO=args.first_ppo, inventory=args.inventory,
            startup=run / "startup.json", revision=source.parent / "source_revision_021.json",
            predecessor=source.parent / "predecessor_021.json", invocation=run.parent / "invocation.json")
        require({name: digest(path) for name, path in paths.items()} == authority["metadata_sha256"],
            "Immutable admission metadata changed during read")
        helper.verify_owner_continuity(authority["native_owner"], helper.native_owner(
            authority["worker_pid"], authority["invocation"]["command"]))
        report.update(identity=authority, additional_source_sha256=extra_source_hashes,
            reader_reuse=reader.adapter_reuse, source_files_unchanged=True)
    except Exception as error:
        report = dict(schema="source021-existing-native-per-axis-std-KL-sidecar-v1", status="read_failed",
            complete=False, qualification=None, error=type(error).__name__ + ": " + str(error))
    report.update(queried_at_utc=datetime.now(timezone.utc).isoformat(), sidecar_sha256=digest(Path(__file__)),
        reused_reader021_sha256=READER021_SHA, PPO_updates_by_sidecar=0,
        scope="read_existing_native_event_no_new_producer_no_simulator_no_GPU_no_SSH_no_TensorBoard_server")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({key: report.get(key) for key in ("status", "complete", "error")}))
    return 2 if report["status"] == "read_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
