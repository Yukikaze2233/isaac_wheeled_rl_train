"""Bounded, fresh-process ONNX playback of clean versus shared PC feedback.

The original full role is authenticated before both runtime and independent
expectations are projected. Each invocation creates one scene and one actor;
it never constructs an optimizer or grants training/deployment qualification.
"""
from copy import deepcopy
from contextlib import contextmanager, ExitStack
import argparse
from datetime import datetime, timezone
import faulthandler
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chassis_suite_support import digest, prepare_playback, write_json
from play_chassis_suite import validated_session
from wheeled_tasks.chassis.pc_communication import (
    apply_pc_communication, runtime_expectation, validate_pc_communication)
from wheeled_tasks.chassis.preflight import design_preflight

NAMES = ("link_stand", "link_yaw_positive", "link_yaw_negative")
ISOLATIONS = ("full", "actor-feedback", "control-feedback", "command-delay")


def require_probe_scope(env):
    config = env.cfg
    if (config.get("run_scope") != "interactive_unscored"
            or config.get("evaluation_transport_mode") != "training_domain_prior"
            or config.get("evaluation_exact_cases") is not True
            or config.get("auto_reset") is not False
            or any(config.get(key) is not None for key in
                ("qualified", "domain_qualified", "qualification"))):
        raise ValueError("Override requires a bounded, unscored, first-episode probe")
    validate_pc_communication(config)
    if (env.feedback_transport is None or env.command_transport is None
            or env.usb_transport is not None or env.signal_delay is not None):
        raise ValueError("Override requires the paired new PC path without legacy delays")


def fixed_profiles(feedback_ms, usb_ms, phase_ms):
    """CAN phase is an event offset, not an added serial propagation delay."""
    values = (feedback_ms, usb_ms, phase_ms)
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
            or not .04 <= feedback_ms <= 15 or not .04 <= usb_ms <= 14
            or not 0 <= phase_ms <= 1):
        raise ValueError("Fixed probe timings exceed the declared PC engineering prior")
    event_delay = max(0, math.ceil(usb_ms - phase_ms - 1e-7)) + phase_ms
    return dict(feedback_ms=feedback_ms, usb_ms=usb_ms, can_phase_ms=phase_ms,
        expected_can_event_delay_ms=event_delay, jitter_ms=0., drop_probability=0.,
        command_age_definition="midpoint_held_message_age_not_PD_feedback_loop_age")


@contextmanager
def fixed_feedback_link(env, feedback_ms, usb_ms, phase_ms, seed):
    """Replace only native transport instances in an explicitly unscored probe."""
    if feedback_ms is None and usb_ms is None:
        yield dict(enabled=False, hooks_restored=True)
        return
    profile = fixed_profiles(feedback_ms, usb_ms, phase_ms)
    require_probe_scope(env)
    if getattr(env, "_fixed_feedback_link_active", False):
        raise RuntimeError("Fixed feedback-link profiles cannot nest")
    import torch
    from wheeled_tasks.chassis.feedback_transport import FractionalFeedbackTransport
    from wheeled_tasks.chassis.command_transport import UsbCommandTransport

    previous_feedback, previous_command = env.feedback_transport, env.command_transport
    count, device = len(previous_feedback.enabled), previous_feedback.feedback.device
    rows, draw = torch.arange(count, device=device), torch.zeros(count, device=device)
    settings = deepcopy(previous_feedback.cfg)
    settings.update(delay_range_s=[feedback_ms * .001] * 2,
        enabled_fraction=1., jitter_s=0.)
    feedback = FractionalFeedbackTransport(count, device, env.dt, settings, seed, domain_draw=draw)
    feedback.physics_steps = previous_feedback.physics_steps
    feedback.reset(rows, env._transport_sensors())
    settings = deepcopy(previous_command.cfg)
    settings["enabled_fraction"] = 1.
    profiles = [dict(delay_ms=usb_ms, can_phase_ms=phase_ms, jitter_ms=0.,
        drop_probability=0., burst_every_packets=0, burst_packets=0) for _ in range(count)]
    command = UsbCommandTransport(count, device, env.dt, settings, seed, profiles=profiles, domain_draw=draw)
    command.tick = previous_command.tick
    command.reset(rows)
    receipt = dict(enabled=True, profile=profile, feedback_settings=feedback.cfg,
        command_settings=command.cfg, command_profiles=profiles,
        scope="noncanonical_native_runtime_profile_unscored_only",
        canonical_preflight_covers_profile_override=False, hooks_restored=False)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(env, "_fixed_feedback_link_active", True, create=True))
            stack.enter_context(patch.object(env, "feedback_transport", feedback))
            stack.enter_context(patch.object(env, "command_transport", command))
            yield receipt
    finally:
        receipt["hooks_restored"] = (env.feedback_transport is previous_feedback
            and env.command_transport is previous_command
            and not getattr(env, "_fixed_feedback_link_active", False))
        if not receipt["hooks_restored"]:
            raise RuntimeError("Fixed feedback-link transports were not restored")


@contextmanager
def isolate_feedback_link(env, selection, *, activate_at_s=0., cold_queue_at_handoff=False):
    """Route one unscored environment's feedback without changing its plant.

    Keeping the command module in feedback-only probes preserves the physical
    wheel-limit branch. Its queue advances even when the torque return is
    bypassed, so warm handoff retains real history. No production mode is added.
    """
    if selection not in ISOLATIONS:
        raise ValueError("Unknown feedback-link isolation")
    if (type(activate_at_s) not in (int, float) or not math.isfinite(activate_at_s)
            or not 0 <= activate_at_s <= 5
            or type(cold_queue_at_handoff) is not bool
            or (cold_queue_at_handoff and not activate_at_s)):
        raise ValueError("Invalid bounded feedback-link handoff")
    enabled = env.feedback_transport is not None
    receipt = dict(selection=selection, policy_feedback=enabled,
        control_feedback=enabled, command_delay=env.command_transport is not None,
        scope="interactive_unscored_runtime_override", hooks_installed=False,
        hooks_restored=False, policy_reads=0, control_reads=0,
        delayed_policy_reads=0, delayed_control_reads=0, command_bypasses=0,
        native_command_calls=0, activation_s=activate_at_s,
        cold_queue_at_handoff=cold_queue_at_handoff, handoff=None)
    if selection == "full" and not activate_at_s:
        try:
            yield receipt
        finally:
            receipt["hooks_restored"] = True
        return
    require_probe_scope(env)
    if cold_queue_at_handoff and not getattr(env, "_fixed_feedback_link_active", False):
        raise ValueError("Cold queue comparison requires fixed timing profiles")
    owner = next(cls for cls in type(env).__mro__ if "_pc_feedback_transport" in cls.__dict__)
    if "_feedback_link_probe_owner" in owner.__dict__:
        raise RuntimeError("Feedback-link isolation cannot nest or share a process")
    descriptor = owner.__dict__["_pc_feedback_transport"]
    if not isinstance(descriptor, property):
        raise TypeError("Feedback-link isolation requires the native feedback property")
    receipt.update(policy_feedback=selection in ("full", "actor-feedback"),
        control_feedback=selection in ("full", "control-feedback"),
        command_delay=selection in ("full", "command-delay"))
    get_observations = env.get_observations
    in_policy_read = False
    previous_observation_override = env.__dict__.get("get_observations")
    command = env.command_transport
    previous_command_override = command.__dict__.get("apply_torque")
    apply_torque = command.apply_torque

    def active():
        if not activate_at_s:
            return True
        elapsed = env._feedback_link_probe_time_s
        if elapsed + 1e-9 < activate_at_s:
            return False
        if receipt["handoff"] is None:
            transport = env.feedback_transport
            receipt["handoff"] = dict(elapsed_s=elapsed, feedback_steps=transport.physics_steps,
                command_ticks=command.tick, feedback_clock_s=transport.physics_steps * env.dt,
                command_clock_s=command.tick * command.transport_dt)
            if cold_queue_at_handoff:
                transport.reset(transport.rows, env._transport_sensors())
                command.reset(command.rows)
        return True

    def feedback(instance):
        if instance is not env:
            return descriptor.fget(instance)
        channel = "policy" if in_policy_read else "control"
        receipt[channel + "_reads"] += 1
        if active() and receipt[channel + "_feedback"]:
            receipt["delayed_" + channel + "_reads"] += 1
            return descriptor.fget(instance)
        return None

    def observations(*args, **kwargs):
        nonlocal in_policy_read
        previous = in_policy_read
        in_policy_read = True
        try:
            return get_observations(*args, **kwargs)
        finally:
            in_policy_read = previous

    def routed_command(command_torque):
        delayed = active() and receipt["command_delay"]
        queued = apply_torque(command_torque)
        receipt["native_command_calls"] += 1
        if not delayed:
            receipt["command_bypasses"] += 1
        return queued if delayed else command_torque

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(owner, "_feedback_link_probe_owner", env, create=True))
            stack.enter_context(patch.object(owner, "_pc_feedback_transport", property(feedback)))
            stack.enter_context(patch.object(env, "get_observations", observations))
            stack.enter_context(patch.object(command, "apply_torque", routed_command))
            if activate_at_s:
                stack.enter_context(patch.object(env, "_feedback_link_probe_time_s", 0., create=True))
            receipt["hooks_installed"] = True
            yield receipt
    finally:
        receipt["hooks_restored"] = (
            owner.__dict__["_pc_feedback_transport"] is descriptor
            and "_feedback_link_probe_owner" not in owner.__dict__
            and env.__dict__.get("get_observations") is previous_observation_override
            and command.__dict__.get("apply_torque") is previous_command_override)
        if not receipt["hooks_restored"]:
            raise RuntimeError("Feedback-link diagnostic hooks were not restored")


def observed_metrics(accumulator):
    """Unavailable observation windows carry no numerical behavior credit."""
    report = accumulator.report()
    for group in report["groups"].values():
        if group["frames"]:
            continue
        for key in (*accumulator.ERROR_NAMES, "planar_speed_mean_cm_s", "vx_rmse_m_s",
                "yaw_rmse_rad_s", "height_rmse_m", "reward_per_sim_second",
                "stand_drift_max_m", "tilt_max_deg", "closure_gap_max_m",
                "rms_motor_torque_nm", "peak_motor_torque_nm"):
            group[key] = None
        group["behavior_window_observed"] = False
    return report


def physical_state(env):
    response = env.actuator_response.state
    return dict(sensor_channels=env._transport_sensors().tolist(),
        root_link_pose=env.robot.data.root_link_pose_w.torch.tolist(),
        root_com_linear_velocity_body_m_s=env.robot.data.root_com_lin_vel_b.torch.tolist(),
        response_state=response.tolist() if response is not None else None)


@contextmanager
def trace_physics(env, handoff_s):
    """Pair native PC inputs and physical outputs on the same 5 ms interval."""
    require_probe_scope(env)
    feedback, command, control = env.feedback_transport, env.command_transport, env.v5
    origin_steps = feedback.physics_steps
    observe, apply_torque, decode = feedback.observe, command.apply_torque, control.decode
    traces, pending, decoded = [], None, None

    def enabled():
        elapsed = (feedback.physics_steps - origin_steps) * env.dt
        return elapsed < .5 or (handoff_s > 0 and handoff_s - .05 <= elapsed < handoff_s + .5)

    def targets(actions, q):
        nonlocal decoded
        result = decode(actions, q)
        fresh_q = env.robot.data.joint_pos.torch[:, env.ids]
        fresh = decode(actions, fresh_q)
        decoded = dict(decoded_at_s=(feedback.physics_steps - origin_steps) * env.dt,
            leg_targets_rad=result[0].tolist(), wheel_targets_rad_s=result[1].tolist(),
            leg_target_branch_difference_rad=(result[0] - fresh[0]).tolist())
        return result

    def command_step(torque):
        nonlocal pending
        capture = enabled()
        if capture:
            pending = dict(physical_index=feedback.physics_steps - origin_steps,
                interval_start_s=(feedback.physics_steps - origin_steps) * env.dt,
                before=physical_state(env), decoded=decoded,
                cached_feedback_before=feedback.feedback.tolist(),
                requested_motor_torque_nm=control.requested_motor_effort.tolist(),
                feedback_motor_bounds_nm=control.current_motor_bounds.tolist(),
                post_control_request_nm=torque.tolist(), command_tick_before=command.tick)
        result = apply_torque(torque)
        if capture:
            pending.update(returned_to_plant_before_cap_nm=result.tolist(),
                native_held_nm=command.held.tolist(), command_tick_after=command.tick,
                held_message_generated_s=command.held_generated.tolist(),
                last_usb_arrival_s=command.last_arrival.tolist(),
                held_message_age_s=command.age.tolist())
        return result

    def observe_step(sensors):
        nonlocal pending
        observe(sensors)
        if pending is not None:
            pending.update(interval_end_s=(feedback.physics_steps - origin_steps) * env.dt,
                after=physical_state(env), feedback_age_s=feedback.age_s.tolist(),
                applied_motor_torque_nm=env.robot.data.applied_torque.torch[:, env.ids].tolist(),
                post_transport_motor_bounds_nm=control.current_motor_bounds.tolist(),
                sensor_timestamp_s=feedback.sensor_timestamp_s.tolist(),
                feedback_clock_s=feedback.physics_steps * env.dt,
                command_clock_s=command.tick * command.transport_dt)
            traces.append(pending)
            pending = None

    with ExitStack() as stack:
        stack.enter_context(patch.object(control, "decode", targets))
        stack.enter_context(patch.object(command, "apply_torque", command_step))
        stack.enter_context(patch.object(feedback, "observe", observe_step))
        yield traces


def project(mapping, enabled):
    result = apply_pc_communication(mapping, enabled=enabled)
    original = result["evaluation"]["cases"][0]
    cases = []
    for name in NAMES:
        case = deepcopy(original)
        case.update(name=name, reset_seed_key="feedback_link_common_birth",
                    episode_seconds=30.)
        case.pop("communication_profile", None)
        cases.append(case)
    result["evaluation"].update(cases=cases, canonical_case_names=list(NAMES),
        episodes_per_case=2, protocol_id="shared_pc_feedback_unscored_v1")
    result.update(scene_groups=[dict(name=name, fraction=1 / 3, terrain=["flat"]) for name in NAMES],
        skill_specs={case["name"]: deepcopy(case["skill"]) for case in cases},
        flat_floor_width_m=8., playback_open_ground=False,
        evaluation_exact_cases=True, run_scope="interactive_unscored", diagnostic_trace=True,
        evaluation_transport_mode="training_domain_prior")
    return result


def prepare(policy, output, enabled):
    source, base, manifest, control, seed, playback_receipt = prepare_playback(
        ROOT, Path(str(policy) + ".contract.json"), output / "source_projection", "flat")
    original = json.loads((output / "source_projection/CPU_preflight.json").read_text())["source_full_role"]
    design_path = Path(base["design_preflight"]["file"])
    design = json.loads(design_path.read_text())
    stage = base["design_preflight"]["stage"]
    row = design["stages"][stage]["roles"]["playback"]
    config = project(base, enabled)
    row["contract"] = project(row["contract"], enabled)
    row["invocation"]["num_envs"] = 6
    row["runtime"].update(runtime_expectation(row["contract"]))
    row["runtime"]["modules"].update(feedback_transport=enabled,
        command_transport=enabled, signal_delay=False, usb_transport=False)
    row["runtime"]["signal_delay"] = deepcopy(row["contract"]["signal_delay"])
    row["runtime"]["command_transport_model"] = "phase_aware_hold_v1" if enabled else None
    design.update(scenario="shared_pc_feedback", communication_enabled=enabled,
        allowed_changes=design["allowed_changes"] + ["feedback_transport", "command_transport",
            "signal_delay.enabled", "usb_transport.enabled", "signal_perturbations.max_delay_steps",
            "flat_floor_width_m", "runtime.feedback_consumers", "diagnostic_trace"],
        qualification=None, domain_qualified=None)
    path = output / "probe_expectations.json"
    write_json(path, design)
    config["design_preflight"] = dict(file=str(path.resolve()), sha256=digest(path), stage=stage)
    receipt = design_preflight(config, control, ROOT, stage=config["enabled_stages"][0],
        role="playback", seed=seed, num_envs=6)
    session, metadata = validated_session(policy, Path(str(policy) + ".contract.json"), source)
    write_json(output / "probe_contract.json", config)
    write_json(output / "CPU_preflight.json", dict(source_full_role=original, probe=receipt,
        source_playback_projection=playback_receipt,
        source_policy_sha256=digest(policy), qualification=None, scene_run=False))
    return config, manifest, control, seed, session, metadata


def run(args, config, manifest, control, seed, session, report):
    from isaaclab.app import AppLauncher
    launcher, env, writer = None, None, None
    overrides = ExitStack()
    def stage(name):
        report["initialization_stage"] = name
        write_json(args.output / "report.json", report)
        print("Feedback diagnostic: " + name, flush=True)
    faulthandler.dump_traceback_later(90., repeat=True)
    try:
        stage("launch_application")
        launcher = AppLauncher(dict(headless=True, device="cpu", enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
        from wheeled_tasks.chassis.playback_command import advance_playback_command
        stage("initialize_native_cpu_threads")
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        stage("construct_native_scene")
        env = FixedCaseEnv(config, manifest, control, ROOT,
            stage_name=config["enabled_stages"][0], num_envs=6,
            device="cpu", seed=seed, level=1., run_role="playback")
        writer = SummaryWriter(str(args.output / "events"))
        stage("reset_native_scene")
        observations = env.reset_suite()
        report["fixed_profile_override"] = overrides.enter_context(fixed_feedback_link(env,
            args.fixed_feedback_ms, args.fixed_usb_ms, args.fixed_can_phase_ms, seed))
        observations = env.get_observations()
        previous, alive = env.commands.clone(), torch.ones(6, dtype=torch.bool)
        metrics = EpisodeMetrics(env.scene_groups, "cpu", env.policy_dt,
            warmup_seconds=2. + args.warmup_seconds)
        phase_metrics = {name: EpisodeMetrics(env.scene_groups, "cpu", env.policy_dt)
            for name in ("steady_yaw_3_to_9s", "settled_stop_10_6_to_12s")}
        report.update(status="running", startup_report=env.startup_report,
            scene_groups=env.scene_groups, samples=[], terminal={},
            motor_joint_names=[env.robot.joint_names[j] for j in env.ids],
            initial_state=dict(sensor_channels=env._transport_sensors().tolist(),
                root_link_pose=env.robot.data.root_link_pose_w.torch.tolist(),
                policy_observation=observations["policy"].tolist()),
            initial_feedback_delay_s=env.feedback_transport.delay_s.tolist() if env.feedback_transport else [0.] * 6,
            initial_command_base_delay_s=env.command_transport.base_delay.tolist() if env.command_transport else [0.] * 6,
            initial_command_can_phase_s=env.command_transport.can_phase.tolist() if env.command_transport else [0.] * 6)
        report["diagnostic_override"] = overrides.enter_context(
            isolate_feedback_link(env, args.isolation, activate_at_s=args.warmup_seconds,
                cold_queue_at_handoff=args.cold_queue_at_handoff))
        if env.feedback_transport is not None:
            report["physical_trace"] = overrides.enter_context(trace_physics(env, args.warmup_seconds))
        feedback_origin = env.feedback_transport.physics_steps if env.feedback_transport else None
        command_origin = env.command_transport.tick if env.command_transport else None
        warmup_ticks = round(args.warmup_seconds / env.policy_dt)
        for tick in range(600 + warmup_ticks):
            elapsed = tick * env.policy_dt
            t = elapsed - args.warmup_seconds
            if args.warmup_seconds and tick == warmup_ticks:
                report["handoff_state_before_routing"] = physical_state(env)
                report["handoff_state_before_routing"].update(
                    previous_control_actions=env.actions.tolist(),
                    previous_policy_observation=observations["policy"].tolist())
            if args.warmup_seconds:
                env._feedback_link_probe_time_s = elapsed
            requested = previous.new_tensor([[0., (0. if name == NAMES[0] else
                2 * math.pi * (1 if name == NAMES[1] else -1)) if 1 <= t < 9 else 0., .305]
                for name in env.scene_groups])
            applied = advance_playback_command(previous, requested, env.policy_dt,
                config["command_slew"], config.get("motion_limits"))
            env.command_acceleration.copy_((applied[:, 0] - previous[:, 0]) / env.policy_dt)
            previous.copy_(applied)
            env.commands.copy_(applied)
            env.command_target.copy_(applied[:, :2])
            env.raw_height.copy_(applied[:, 2])
            env.mode.copy_((applied[:, :2].abs().sum(-1) > .01).long())
            env.history.last_tick = None
            env.update_targets()
            env.update_command_reference()
            observations = env.get_observations()
            policy_before = observations["policy"].clone()
            feedback_before = (env.feedback_transport.feedback.clone()
                if env.feedback_transport is not None else None)
            actions = session.run(None, {"obs": policy_before.cpu().numpy().astype(np.float32)})[0]
            if actions.shape != (6, 6) or not np.isfinite(actions).all():
                raise RuntimeError("Invalid deterministic ONNX output")
            active = alive.clone()
            with torch.no_grad():
                observations, _, done, extras = env.step(torch.from_numpy(actions))
            metrics.observe(extras["diagnostics"], active)
            for name, (start, end) in zip(phase_metrics, ((3., 9.), (10.6, 12.))):
                if start <= t < end:
                    phase_metrics[name].observe(extras["diagnostics"], active)
            if tick % 10 == 0 or bool((active & done).any()):
                state = extras["diagnostics"]
                report["samples"].append(dict(tick=tick + 1, time_s=(tick + 1) * env.policy_dt,
                    handoff_relative_interval_start_s=t, interval_start_s=elapsed,
                    last_substep_start_s=(tick + 1) * env.policy_dt - env.dt,
                    active=active.tolist(), command=applied.tolist(),
                    velocity_m_s=state["velocity"].tolist(), omega_rad_s=state["omega"].tolist(),
                    position_m=state["position"].tolist(), height_m=state["height"].tolist(),
                    gravity=state["gravity"].tolist(), motor_effort_nm=state["motor_effort"].tolist(),
                    policy_observation_at_tick_start=policy_before.tolist(),
                    action_at_tick_start=actions.tolist(),
                    feedback_buffer_at_tick_start=feedback_before.tolist() if feedback_before is not None else None,
                    motor_position_at_tick_end_rad=state["motor_position"].tolist(),
                    motor_velocity_at_tick_end_rad_s=state["motor_velocity"].tolist(),
                    last_substep_requested_motor_effort_nm=state["requested_motor_effort"].tolist(),
                    last_substep_motor_effort_bounds_nm=state["motor_effort_bounds"].tolist(),
                    feedback_age_s=env.feedback_transport.age_s.tolist() if env.feedback_transport else [0.] * 6,
                    feedback_enabled=env.feedback_transport.enabled.tolist() if env.feedback_transport else [False] * 6,
                    command_enabled=env.command_transport.enabled.tolist() if env.command_transport else [False] * 6,
                    command_base_delay_s=env.command_transport.base_delay.tolist() if env.command_transport else [0.] * 6,
                    command_can_phase_s=env.command_transport.can_phase.tolist() if env.command_transport else [0.] * 6,
                    command_lost_packets=env.command_transport.lost.tolist() if env.command_transport else [0.] * 6,
                    command_age_s=env.command_transport.age.tolist() if env.command_transport else [0.] * 6))
                for tag, value in extras["log"].items():
                    if tag.startswith("/transport/"):
                        writer.add_scalar(tag, value, tick + 1)
            for lane in (active & done).nonzero(as_tuple=False).flatten().tolist():
                report["terminal"][str(lane)] = dict(time_s=(tick + 1) * .02,
                    handoff_relative_time_s=(tick + 1) * env.policy_dt - args.warmup_seconds,
                    case=env.scene_groups[lane], nonwheel_body_names=[env.robot.body_names[j]
                        for j in env.nonwheel_ids if float(env.contact_force[lane, j].norm()) > 5.],
                    reasons=[name for name, mask in extras["diagnostics"]["reasons"].items() if bool(mask[lane])])
            alive &= ~done
            if tick % 50 == 0:
                write_json(args.output / "report.json", report)
            if not bool(alive.any()):
                break
        report.update(status="completed_bounded_unscored", native_metrics=observed_metrics(metrics),
            phase_metrics={name: observed_metrics(accumulator) for name, accumulator in phase_metrics.items()},
            right_censored=alive.tolist(), policy_ticks=tick + 1, physics_ticks=(tick + 1) * 4)
        report["transport_clock_advances"] = dict(
            feedback_physics_steps=env.feedback_transport.physics_steps - feedback_origin
                if env.feedback_transport else None,
            command_can_ticks=env.command_transport.tick - command_origin if env.command_transport else None)
        if env.feedback_transport is not None:
            if report["transport_clock_advances"] != dict(
                    feedback_physics_steps=(tick + 1) * 4, command_can_ticks=(tick + 1) * 20):
                raise RuntimeError("Shadow or active transport clocks did not advance with native physics")
    except BaseException as error:
        report.update(status="failed_unscored", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        overrides.close()
        write_json(args.output / "report.json", report)
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--communication", choices=("clean", "prior"), required=True)
    parser.add_argument("--isolation", choices=ISOLATIONS, default="full")
    parser.add_argument("--fixed-feedback-ms", type=float)
    parser.add_argument("--fixed-usb-ms", type=float)
    parser.add_argument("--fixed-can-phase-ms", type=float, default=.5)
    parser.add_argument("--warmup-seconds", type=float, default=0.)
    parser.add_argument("--cold-queue-at-handoff", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.communication == "clean" and args.isolation != "full":
        parser.error("Path isolation requires --communication prior")
    if args.fixed_feedback_ms is not None or args.fixed_usb_ms is not None:
        if args.communication != "prior":
            parser.error("Fixed timing profiles require --communication prior")
        try:
            fixed_profiles(args.fixed_feedback_ms, args.fixed_usb_ms, args.fixed_can_phase_ms)
        except ValueError as error:
            parser.error(str(error))
    if (not math.isfinite(args.warmup_seconds) or not 0 <= args.warmup_seconds <= 5
            or abs(args.warmup_seconds / .02 - round(args.warmup_seconds / .02)) > 1e-8
            or (args.cold_queue_at_handoff and not args.warmup_seconds)
            or (args.cold_queue_at_handoff and args.fixed_feedback_ms is None)
            or (args.warmup_seconds and args.communication != "prior")):
        parser.error("Handoff requires prior timing and a policy-aligned warmup in [0, 5] seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    config, manifest, control, seed, session, metadata = prepare(
        args.onnx, args.output, args.communication == "prior")
    report = dict(status="CPU_design_identity_passed_scene_not_run", qualification=None,
        at_utc=datetime.now(timezone.utc).isoformat(), source_policy_sha256=digest(args.onnx),
        source_checkpoint_sha256=metadata.get("source_checkpoint_sha256"),
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_files_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(
            (ROOT / "src/wheeled_tasks/chassis").glob("*.py"))}, script_sha256=digest(Path(__file__)),
        communication=args.communication, isolation=args.isolation,
        fixed_profile_requested=fixed_profiles(args.fixed_feedback_ms, args.fixed_usb_ms,
            args.fixed_can_phase_ms) if args.fixed_feedback_ms is not None else None,
        run_scope="interactive_unscored", policy_hz=50, physics_pd_hz=200,
        formal_updates_consumed=0, optimizer_constructed=False, actor_count=1,
        independent_process_per_model_profile=True, max_sim_seconds=12. + args.warmup_seconds,
        warmup_seconds=args.warmup_seconds, cold_queue_at_handoff=args.cold_queue_at_handoff,
        phase_window_clock="seconds_relative_to_handoff",
        episode_metric_clock="absolute_seconds_since_episode_birth",
        scope="Two replicas per zero-speed and +/-1rps case; no recovery or hardware qualification")
    write_json(args.output / "report.json", report)
    if not args.preflight_only:
        run(args, config, manifest, control, seed, session, report)


if __name__ == "__main__":
    main()
