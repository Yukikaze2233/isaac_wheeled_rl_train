"""Bounded scene and restored-policy validation before training updates.

Validation reads native transitions without inserting them into PPO storage.
Simulator and tensor dependencies stay local so the entry can load before Kit.
"""

import json
from pathlib import Path


def validate_numerical_failure_policy(args, config):
    """Episode resets are explicit formal PPO behavior, never validation behavior."""
    policy = getattr(args, "numerical_failure_policy", "abort-worker")
    if policy not in ("abort-worker", "episode-reset"):
        raise ValueError("Unknown numerical failure policy")
    if policy == "episode-reset" and (
            config.get("formal_v7_flat") is not True
            or not args.capture_numerical_failure
            or args.validate_scene
            or args.validate_restored_policy
            or args.validate_restored_sampled_policy):
        raise ValueError("episode-reset requires formal PPO with numerical capture; excludes scene and restored-policy checks")
    return policy


def validate_restored_policy_request(args, config, root):
    """Explicit inference-only window; never a weights-only or scene-only probe."""
    steps = args.validate_restored_policy
    if type(steps) is not int or not 0 <= steps <= 10000:
        raise ValueError("Restored-policy validation requires 1..10000 steps, or zero to disable")
    if not steps:
        return False
    from wheeled_tasks.chassis.flat_episode_selection import validate_scut_flat_recovery
    from wheeled_tasks.chassis.training_profile import uses_profile
    if (args.resume is None or args.validate_scene or args.preflight_only or not uses_profile(config)
            or config.get("formal_learning_state_resume") is not True
            or not args.capture_numerical_failure or config.get("record_diagnostics") is not True
            or validate_scut_flat_recovery(config, root) is None):
        raise ValueError("Restored-policy validation requires full formal resume, explicit scripted recovery, diagnostics and first-substep numerical capture; excludes scene/preflight checks")
    return True


def validate_restored_sampled_policy_request(args, config):
    """One actual rollout of the changed decoder, after complete state restore."""
    steps = getattr(args, "validate_restored_sampled_policy", 0)
    if (type(steps) is not int or steps not in (0, 24, 48)
            or steps == 48 and config.get("gas_pressure_application") != "main_link_mounts_v1"):
        raise ValueError("Restored sampled-policy validation requires24 steps, or48 for mount pressure")
    if not steps:
        return False
    from wheeled_tasks.chassis.fixed_flat_rehearsal import validate_dynamic_nonsteady_birth
    from wheeled_tasks.chassis.training_profile import uses_profile
    if (args.resume is None or args.validate_scene or args.preflight_only
            or getattr(args, "validate_restored_policy", 0) or args.updates != 1
            or args.num_envs != 4096 or not uses_profile(config)
            or config.get("formal_learning_state_resume") is not True
            or not args.capture_numerical_failure or config.get("record_diagnostics") is not True
            or config.get("v5_control", {}).get("leg_winding_version") != "paired_branch107_v1"
            or not validate_dynamic_nonsteady_birth(config)):
        raise ValueError("Sampled validation requires the explicit branch107 dynamic birth, full4096 resume, diagnostics and zero-optimization entry")
    return True


def bind_sampled_policy_inputs(algorithm, env, directory):
    """Keep the consumed actor input in the existing substep evidence ring."""
    import torch
    original = algorithm.act

    def record(observations, actions, mean, std, *, actor_mode="sampled"):
        env._formal_actor_input = {"policy_input35": observations["policy"].detach().clone(),
                                  "policy_mean_action": mean.detach().clone(),
                                  "policy_std_action": std.detach().clone()}
        if env.physical_tick == 0:
            packet = {"schema": "v7-consumed-first-policy-input-v1", "actor_mode": actor_mode,
                "physical_tick": 0, "policy_tick": env.tick,
                "episode_id": env.flat_episode.episode_id.detach().cpu().clone(),
                "episode_physics_ticks": env.flat_episode.commands.episode_physics_ticks.detach().cpu().clone(),
                "observations": {name: value.detach().cpu().clone() for name, value in observations.items()},
                "raw_policy_action": actions.detach().cpu().clone(),
                **{name: value.cpu() for name, value in env._formal_actor_input.items()},
                "scope": "actual input and declared actor mode before env.step; not a reconstructed proxy"}
            if actor_mode == "sampled":
                packet["raw_sampled_action"] = packet["raw_policy_action"]
            torch.save(packet, Path(directory) / "policy_first_action.pt")

    def act(observations):
        actions = original(observations)
        mean, std = algorithm.transition.distribution_params
        record(observations, actions, mean, std)
        return actions

    algorithm.act = act
    env._record_formal_actor_input = record


def validate_restored_sampled_policy(runner, env, steps, report, *, check_runtime):
    """Sample one native rollout without storage, returns, optimization or saves."""
    import torch
    capture = getattr(env, "_numerical_failure_capture", None)
    allowed_steps = (24, 48) if getattr(env, "cfg", {}).get("gas_pressure_application") == "main_link_mounts_v1" else (24,)
    if (steps not in allowed_steps or env.physical_tick != 0 or env.flat_episode is None
            or capture is None or capture.stop_on_first_event is not True
            or runner.alg.storage.step != 0):
        raise ValueError("Sampled-policy validation requires unstepped full state and first-substep capture")
    clocks = (env.stage_actor_update, env.global_actor_update)
    report.update(status="running", requested_policy_steps=steps, completed_policy_steps=0,
        actor_mode="sampled_training_mode", ppo_updates=0, actor_direct_loss_credit=0,
        checkpoint_written=False, qualification=False, readback_files=[],
        physically_eligible_transitions_not_optimized=0, reset_transitions=0,
        restored_stage_actor_update=clocks[0], restored_global_actor_update=clocks[1],
        scope="bounded physical window after actor/critic/std/Adam/RNG restore; not settling or recovery qualification")
    try:
        runner.alg.train_mode()
        observations = env.get_observations()
        with torch.inference_mode():
            for step in range(1, steps + 1):
                check_runtime()
                if any(not bool(torch.isfinite(value).all()) for value in observations.values()):
                    raise FloatingPointError("Sampled-policy validation non-finite observation")
                try:
                    actions = runner.alg.act(observations)
                    if actions.shape != (env.num_envs, 6) or not bool(torch.isfinite(actions).all()):
                        raise FloatingPointError("Sampled-policy validation non-finite or malformed action")
                    before = env.physical_tick
                    observations, reward, done, info = env.step(actions)
                finally:
                    # act writes only a temporary transition. Never submit it.
                    runner.alg.transition.clear()
                if (env.physical_tick != before + 4 or runner.alg.storage.step != 0
                        or (env.stage_actor_update, env.global_actor_update) != clocks):
                    raise RuntimeError("Sampled-policy validation changed storage, control or learning clocks")
                if (not bool(torch.isfinite(reward).all())
                        or any(not bool(torch.isfinite(value).all()) for value in observations.values())
                        or bool(info["diagnostics"]["reasons"]["numerical_safety"].any()) or capture.recorded):
                    raise FloatingPointError("Sampled-policy validation numerical failure")
                report["physically_eligible_transitions_not_optimized"] += int(info["policy_learning_mask"].sum())
                report["reset_transitions"] += int(done.sum())
                if step in (1, 9, 10, 11, 24, 48):
                    frame = nonsteady_scene_validation_frame(env, observations, done, info, step)
                    frame["scope"] = "restored sampled-policy physical/interface observation; no optimizer or recovery qualification"
                    name = f"sampled_policy_step_{step:04d}.pt"
                    torch.save(frame, Path(env.cfg["formal_physics_trace_directory"]) / name)
                    report["readback_files"].append(name)
                report["completed_policy_steps"] += 1
        report["status"] = "window_completed"
    except BaseException as error:
        report.update(status="failed_or_interrupted", error=repr(error))
        raise
    finally:
        report["physical_steps_completed"] = env.physical_tick
        report["storage_step"] = runner.alg.storage.step
        report["numerical_failure_capture"] = dict(capture.summary)


def validate_restored_policy(runner, env, steps, report, *, check_runtime, behavior_metrics=None):
    """Observe a finite mean-policy window after the entry's complete restore.

    Caller supplies the scene/profile/command allocation. The existing physical
    journal is discarded from learning every interval: no optimizer or PPO credit.
    """
    import torch
    from collections import Counter
    from wheeled_tasks.chassis.recovery_evidence import RecoveryEvidenceJournal
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("Restored-policy validation step bound is invalid")
    recovery = env.recovery
    capture = getattr(env, "_numerical_failure_capture", None)
    if (recovery is None or not recovery.source_flat_handoff or not recovery.has_tasks
            or capture is None or capture.stop_on_first_event is not True or env.physical_tick != 0):
        raise ValueError("Restored-policy validation requires an unstepped script recovery and first-substep abort")
    clocks = (env.stage_actor_update, env.global_actor_update)
    journal = RecoveryEvidenceJournal(num_envs=env.num_envs, steps_per_rollout=1,
        source_identity=recovery.source_identity(), device=env.device)
    report.update(status="running", requested_policy_steps=steps, completed_policy_steps=0,
        actor_mode="mean", ppo_updates=0, actor_direct_loss_credit=0, checkpoint_written=False,
        qualification=False, profile_lane_counts=dict(Counter(
            name if name is not None else "ordinary" for name in recovery.profile_names)),
        restored_stage_actor_update=clocks[0], restored_global_actor_update=clocks[1],
        scope="current finite window; continuous native script/handoff; physical eligible intervals are not optimized samples",
        commands="unchanged caller contract and command owner; no validation command override")
    rough_metrics = None
    rough_adapter, rough_adapter_masks = None, []
    rough_birth_endpoints = []
    rough_adapter_steps = min(steps, 24)
    if getattr(env, "cfg", {}).get("formal_small_rough_mix", {}).get("version") == "motion-retention-course-v1":
        from wheeled_tasks.chassis.rough_course_exposure import RoughCourseExposure
        rough_adapter = RoughCourseExposure(torch.tensor(
            [name == "source_rough" for name in env.scene_groups],
            device=env.device, dtype=torch.bool), env.flat_episode.commands._rough, rough_adapter_steps)
    window_counts = None
    if getattr(recovery, "episode_window", None) is not None:
        window_fields = ("actor_owned_intervals", "masked_intervals", "post_handoff_horizon_episodes",
            "prelude_failure_episodes", "post_handoff_failure_episodes", "boundary_truncated_episodes")
        window_counts = torch.zeros(env.num_envs, len(window_fields), dtype=torch.long, device=env.device)
        report["recovery_episode_window"] = {"settings": dict(recovery.episode_window),
            "count_scope": "physical opportunities, not optimized samples or tracking success",
            "first_terminal_readback_file": None}
    if behavior_metrics is not None:
        runner.logger.init_logging_writer()
        if getattr(env, "cfg", {}).get("formal_small_rough_mix", {}).get("version") == "motion-retention-course-v1":
            from wheeled_tasks.chassis.small_rough_evaluation import RoughBirthWindowMetrics
            rough_metrics = RoughBirthWindowMetrics(env)

    def publish_behavior():
        if behavior_metrics is None:
            return
        measured = behavior_metrics.report()
        measured["scope"] = "mean-policy physical measurements on actor-owned intervals; not PPO exposure or acceptance"
        if "fixed_rehearsal" in measured:
            measured["fixed_rehearsal"]["scope"] = "overlapping fixed-case views of the same mean-policy physical samples"
        report["behavior_metrics"] = measured
        reports = [("behavior", measured)]
        if "fixed_rehearsal" in measured:
            reports.append(("fixed17", measured["fixed_rehearsal"]))
        if rough_metrics is not None:
            report["rough_birth_metrics"] = rough_metrics.report()
            reports.append(("rough_birth", report["rough_birth_metrics"]["measurements"]))
        writer = runner.logger.writer
        if writer is not None:
            for prefix, values in reports:
                for group, items in values["groups"].items():
                    for name, value in items.items():
                        if type(value) in (int, float):
                            writer.add_scalar(f"ParentInference/{prefix}/{group}/{name}", value,
                                report["completed_policy_steps"])
            writer.flush()
    try:
        actor = runner.get_inference_policy()
        observations = env.get_observations()
        with torch.inference_mode():
            for _ in range(steps):
                check_runtime()
                if any(not bool(torch.isfinite(value).all()) for value in observations.values()):
                    raise FloatingPointError("Restored-policy validation non-finite observation")
                actions = actor(observations, stochastic_output=False)
                if actions.shape != (env.num_envs, 6) or not bool(torch.isfinite(actions).all()):
                    raise FloatingPointError("Restored-policy validation non-finite or malformed mean action")
                recorder = getattr(env, "_record_formal_actor_input", None)
                if recorder is not None:
                    std = runner.alg.actor.distribution.std_param.expand_as(actions)
                    recorder(observations, actions, actions, std, actor_mode="mean")
                before = env.physical_tick
                observations, reward, done, extras = env.step(actions)
                if (env.physical_tick != before + 4
                        or (env.stage_actor_update, env.global_actor_update) != clocks):
                    raise RuntimeError("Restored-policy validation changed control or learning clocks")
                if (not bool(torch.isfinite(reward).all())
                        or any(not bool(torch.isfinite(value).all()) for value in observations.values())):
                    raise FloatingPointError("Restored-policy validation non-finite reward or observation")
                data = extras["diagnostics"]
                if rough_adapter is not None and len(rough_adapter_masks) < rough_adapter_steps:
                    rough_adapter.observe(len(rough_adapter_masks), data)
                    rough_adapter_masks.append(extras["policy_learning_mask"].detach().clone())
                    if "formal_rough_nonsteady_birth" in env.cfg:
                        ids = rough_adapter.ids
                        owner = data["source_flat_episode"]
                        ground = owner["ground"][ids]
                        motor_off = ground & ~owner["learning_mask"][ids]
                        effort = data["motor_effort"][ids].abs()
                        endpoint_peak = torch.where(motor_off[:, None], effort, 0.).amax()
                        if bool(endpoint_peak != 0.):
                            raise RuntimeError("Rough motor-off birth has nonzero endpoint actuator effort")
                        rough_birth_endpoints.append(dict(
                            policy_interval_id=env.tick, physical_tick=env.physical_tick,
                            ground_lanes=int(ground.sum()), motor_off_lanes=int(motor_off.sum()),
                            ground_actor_owned_lanes=int((ground & extras["policy_learning_mask"][ids]).sum()),
                            motor_off_endpoint_effort_absmax_nm=float(endpoint_peak)))
                    if len(rough_adapter_masks) == rough_adapter_steps:
                        staged = rough_adapter.prepare(torch.stack(rough_adapter_masks))
                        report["rough_exposure_adapter"] = dict(
                            status="native_inputs_prepared_not_committed",
                            policy_intervals=rough_adapter_steps,
                            actual_rough_lanes=len(rough_adapter.ids),
                            proposed_cell_counts=staged["counts"].cpu().tolist(),
                            proposed_mode_counts=staged["mode_counts"].cpu().tolist(),
                            proposed_totals=staged["totals"].cpu().tolist(),
                            actor_credit=0, optimizer_updates=0,
                            committed_state=rough_adapter.state_dict(),
                            scope="adapter readback on the actual pre-reset fields and ownership mask; no commit, optimization or capability qualification")
                        if rough_birth_endpoints:
                            report["rough_nonsteady_birth_readback"] = dict(
                                actual_membership=env._dynamic_birth_runtime_state(),
                                policy_endpoints=rough_birth_endpoints,
                                effort_scope="native explicit actuator buffer at policy endpoints; not contact/constraint reaction forces or every5ms torque readback",
                                actor_credit=0, qualification=False)
                if bool(data["reasons"]["numerical_safety"].any()) or capture.recorded:
                    raise FloatingPointError("Formal V7 native numerical safety termination")
                payload = extras["recovery_transition"]
                if (not torch.equal(payload["done"], done.bool())
                        or not torch.equal(payload["terminated"], data["terminated"])
                        or not torch.equal(payload["timeouts"], data["timeout"])
                        or not torch.equal(payload["policy_learning_mask"], extras["policy_learning_mask"])
                        or payload["policy_interval_id"] != env.tick):
                    raise ValueError("Recovery validation payload differs from the completed policy interval")
                journal.begin_rollout(actor_enabled=False)
                journal.observe_transition(step_index=0, payload=payload,
                    source_contact_or_attitude=data["reasons"].get("source_attitude_or_contact"))
                journal.discard_rollout()  # Physical measurements only, never loss_population/finish_update.
                if window_counts is not None:
                    started = data["recovery_first_whole_actor_start_s"] >= 0.
                    eligible = recovery.enabled & extras["policy_learning_mask"]
                    boundary = data["reasons"]["boundary"]
                    horizon = recovery.enabled & payload["timeouts"] & ~boundary
                    if bool((horizon & (data["recovery_post_handoff_elapsed_s"] < 20. - 1e-5)).any()):
                        raise ValueError("Recovery horizon truncated the declared post-handoff window")
                    window_counts += torch.stack((eligible, recovery.enabled & ~eligible, horizon,
                        recovery.enabled & data["done"] & ~started & data["reasons"]["recovery_script"],
                        recovery.enabled & data["terminated"] & started,
                        recovery.enabled & payload["timeouts"] & boundary), -1).long()
                    readback = report["recovery_episode_window"]
                    if readback["first_terminal_readback_file"] is None and bool(horizon.any()):
                        name = "recovery_window_terminal_first.pt"
                        torch.save({"scope": "pre-reset terminal critic versus returned reset observation; no PPO",
                            "policy_interval_id": env.tick, "physical_tick": env.physical_tick,
                            "terminal_critic": extras["policy_terminal_critic_observation"].detach().cpu().clone(),
                            "returned_critic": observations["critic"].detach().cpu().clone(),
                            "recovery_horizon": horizon.detach().cpu().clone(),
                            "timeouts": payload["timeouts"].detach().cpu().clone(),
                            "old_episode_ids": payload["episode_ids"].detach().cpu().clone(),
                            "returned_episode_ids": payload["returned_episode_ids"].detach().cpu().clone(),
                            "diagnostics": {key: data[key].detach().cpu().clone() for key in (
                                "root_height_world", "recovery_first_whole_actor_start_s",
                                "recovery_post_handoff_elapsed_s", "recovery_post_handoff_eligible_s",
                                "recovery_post_handoff_full_window", "episode_ticks", "motor_position",
                                "motor_velocity", "commands", "policy_learning_mask") if key in data}},
                            Path(env.cfg["formal_physics_trace_directory"]) / name)
                        readback["first_terminal_readback_file"] = name
                if behavior_metrics is not None:
                    measured = {**data, "measurement_valid_mask": extras["policy_learning_mask"]}
                    behavior_metrics.observe(measured)
                    if rough_metrics is not None:
                        rough_metrics.observe(measured)
                report["completed_policy_steps"] += 1
                if report["completed_policy_steps"] % 100 == 0:
                    publish_behavior()
                    print(json.dumps({"restored_mean_policy_steps": report["completed_policy_steps"],
                        "ppo_updates": 0}), flush=True)
        report["status"] = "window_completed"
    except BaseException as error:
        report.update(status="failed_or_interrupted", error=repr(error))
        raise
    finally:
        report["physical_steps_completed"] = env.physical_tick
        publish_behavior()
        if window_counts is not None:
            report["recovery_episode_window"]["totals"] = dict(zip(window_fields, window_counts.sum(0).cpu().tolist()))
            report["recovery_episode_window"]["profiles"] = {
                name: dict(zip(window_fields, window_counts[torch.tensor(
                    [value == name for value in recovery.profile_names], device=env.device)].sum(0).cpu().tolist()))
                for name in dict.fromkeys(recovery.profile_names) if name is not None}
        report["numerical_failure_capture"] = dict(capture.summary)
        if not journal.collecting and journal.fault is None:
            report["recovery_evidence"] = journal.checkpoint_state()
            report["physical_event_totals"] = journal.metrics()
        else:
            report["recovery_evidence_error"] = journal.fault or "uncommitted invalid physical interval"
        report["scope_note"] = "Milestones are ever-observed legacy producer events, not acceptance; unclosed episodes are censored by this finite window. No rollout is submitted to PPO."


def prepare_rough_scene_validation(env):
    """Move two interior births across 9 m tile seams, only for scene validation."""
    if (env.tick != 0 or env.cfg.get("formal_terrain", {}).get("type") != "rough"
            or env.shared_terrain is None or env.flat_episode is None):
        raise ValueError("Rough scene probe requires an unstepped source rough environment")
    terrain_origins = env.shared_terrain.importer.terrain_origins
    interior = terrain_origins[1:-1, 1:-1, :2].reshape(-1, 2)
    if not len(interior):
        raise ValueError("Rough scene probe requires interior terrain tiles")
    eligible = ((env.origins[:, :2] >= interior.amin(0))
                & (env.origins[:, :2] <= interior.amax(0))).all(-1)
    ids = eligible.nonzero(as_tuple=False).flatten()[:2]
    if len(ids) != 2:
        initial_level = env.cfg.get("formal_rough_initial_level")
        if type(initial_level) is int and initial_level == 0:
            # The declared first-row course has no interior-row births.
            # Keep that course intact while still exercising env.step below.
            return {"scope": "native_geometry_and_consumer_readback_not_policy_capability",
                "seam_probe_status": "skipped",
                "seam_probe_skip_reason": "declared_initial_row0_has_fewer_than_two_interior_births",
                "seam_probe_eligible_births": int(eligible.sum()),
                "seam_probe_env_ids": [], "births_preserved": True,
                "limitations": ["interior seam birth probe not performed",
                    "scene validation continues from the unchanged declared initial births",
                    "short zero-policy nominal-PD observation does not establish rough policy ability"]}
        raise ValueError("Rough scene probe requires two robots on interior tiles")
    offsets = env.origins.new_tensor([[4.6, 0.], [0., 4.6]])
    env.reset_positions[ids, :2] = env.origins[ids, :2] + offsets
    env.reset(ids)
    return {"scope": "native_geometry_and_consumer_readback_not_policy_capability",
        "seam_probe_env_ids": ids.detach().cpu().tolist(),
        "seam_probe_origin_world_m": env.origins[ids].detach().cpu().tolist(),
        "seam_probe_requested_local_xy_m": offsets.detach().cpu().tolist(),
        "seam_probe_reset_root_pose_world": env.robot.data.root_link_pose_w.torch[ids].detach().cpu().tolist(),
        "seam_probe_reset_episode_id": env.flat_episode.episode_id[ids].detach().cpu().tolist(),
        "reset_placement": "existing_whole_shape_mesh_upper_envelope",
        "contact_evidence": "existing_pre_reset_physics_snapshots_contact_and_body_names",
        "coverage_evidence": "startup.formal_terrain_initial_types_and_saved_per_lane_terrain_key_ids",
        "limitations": ["seam births exercise interior tile boundaries, not a driven crossing",
            "short zero-policy nominal-PD observation does not establish rough policy ability",
            "sampled wheel contact is not continuous contact coverage",
            "if enabled, birth_requested.pt is replaced by seam reset; use before_first_step.pt for all lanes",
            "pre-reset diagnostics and returned observations can belong to different episodes"]}


def rough_scene_validation_frame(env, observations, done, info, step_index):
    """Keep terminal measurements separate from the actual returned critic state."""
    from wheeled_tasks.chassis.critic_observation import FIELDS

    def cpu(value):
        return value.detach().cpu().clone()

    diagnostics = info["diagnostics"]
    returned = info["returned_episode_state"]
    fields = ("episode_id", "episode_tick", "root_link_position_world", "root_height_world",
        "reward_ground_height_world", "rough_base_scan_valid_hit_count", "height",
        "boundary", "measurement_valid_mask")
    source_fields = ("terrain_key_ids", "placement_raise_m", "minimum_shape_clearance_m")
    pre = {name: cpu(diagnostics[name]) for name in fields}
    pre["source_flat_episode"] = {
        name: cpu(diagnostics["source_flat_episode"][name]) for name in source_fields}
    pre["reasons"] = {name: cpu(mask) for name, mask in diagnostics["reasons"].items()}
    after = {name: cpu(returned[name]) for name in ("episode_id", "episode_tick", "reset_mask")}
    after["source_flat_episode"] = {
        name: cpu(returned["source_flat_episode"][name]) for name in source_fields}
    height_field = next(field for field in FIELDS if field.name == "root_height")
    after["root_height_world"] = cpu(env.robot.data.root_link_pose_w.torch[:, 2])
    after["critic_height"] = cpu(observations["critic"][:, height_field.start])
    after["expected_critic_height_from_world_z"] = (
        after["root_height_world"].clamp(*height_field.clip) * height_field.scale)
    after["critic_height_minus_expected"] = (
        after["critic_height"] - after["expected_critic_height_from_world_z"])
    observation_readback = {name: {"shape": list(observations[name].shape),
        "finite": bool(cpu(observations[name].isfinite().all()))} for name in ("policy", "critic")}
    payment_means = {name.removeprefix("/reward/"): float(cpu(value))
        for name, value in info["log"].items()
        if name.startswith("/reward/") and name != "/reward/total"}
    total_mean = float(cpu(info["log"]["/reward/total"]))
    reward_readback = {"units": info["native_reward_component_units"],
        "component_count": len(payment_means), "component_means": payment_means,
        "total_mean": total_mean, "component_mean_sum_minus_total_mean": sum(payment_means.values()) - total_mean,
        "per_lane_payment_sum_max_abs_error": float(cpu((
            sum(info["native_reward_components"].values()) - diagnostics["reward"]).abs().max()))}
    return {"schema": "rough-scene-readback-v1", "policy_step": step_index,
        "physical_tick": env.physical_tick,
        "observations": observation_readback, "reward_payments": reward_readback,
        "critic_height_index": height_field.start,
        "critic_height_encoding": {"clip": height_field.clip, "scale": height_field.scale,
            "reference": "root_link_world_z"},
        "pre_reset": pre, "returned": after, "done": cpu(done),
        "same_episode_mask": cpu(~done.bool() & (diagnostics["episode_id"] == returned["episode_id"]))}


def nonsteady_scene_validation_frame(env, observations, done, info, step_index):
    """Read the returned ABI separately from the paid motor-off interval."""
    import torch

    for name, width in (("policy", 35), ("critic", 78)):
        value = observations[name]
        if (value.shape != (env.num_envs, width) or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            raise RuntimeError(f"Nonsteady native {name} observation is invalid")
    paid, returned = info["diagnostics"], info["returned_episode_state"]
    fields = ("episode_id", "episode_physics_ticks", "ground", "holding", "learning_mask")
    pre = {name: paid["source_flat_episode"][name].detach().cpu().clone() for name in fields}
    after = {name: returned["source_flat_episode"][name].detach().cpu().clone() for name in fields}
    learning = info["policy_learning_mask"].detach().cpu().clone()
    bootstrap = info["policy_bootstrap_mask"].detach().cpu().clone()
    terminated = paid["terminated"].detach().cpu().clone()
    expected_learning = pre["learning_mask"].clone()
    expected_bootstrap = ~terminated & ~after["holding"]
    components = {}
    if "scripted_recovery_birth" in paid["source_flat_episode"]:
        for name, source in (("recovery_interval_learning_mask", paid), ("recovery_bootstrap_mask", returned)):
            value = source.get(name)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool
                    or value.shape != (env.num_envs,)):
                raise RuntimeError("Mixed scene readback requires both recovery ownership masks")
            components[name] = value.detach().cpu().clone()
        expected_learning &= components["recovery_interval_learning_mask"]
        expected_bootstrap &= components["recovery_bootstrap_mask"]
    if (not torch.equal(learning, expected_learning)
            or not torch.equal(pre["learning_mask"], ~(pre["ground"] & (pre["episode_physics_ticks"] <= 40)))
            or not torch.equal(bootstrap, expected_bootstrap)):
        raise RuntimeError("Nonsteady native interval and returned-state ownership masks differ")
    noise_readback = {}
    noise = getattr(env, "actor_sensor_noise", None)
    if noise is not None:
        from wheeled_tasks.chassis.scut_observation import build_sensor18
        detail = noise.readback()
        output = detail["sensor_output18"]
        expected = build_sensor18(output[:, 12:15], output[:, 15:18],
            output[:, :6], output[:, 6:12], env.v5.nominal)
        clean = env._actor_sensor_noise_clean_frame
        if (not torch.equal(observations["policy"][:, 4:22], expected)
                or not torch.equal(observations["policy"][:, :4], clean[:, :4])
                or not torch.equal(observations["policy"][:, 22:], clean[:, 22:])
                or not torch.equal(observations["critic"][:, :28], clean[:, :28])
                or not torch.equal(observations["critic"][:, 32:39], clean[:, 28:35])):
            raise RuntimeError("Native physical sensor noise is not isolated to the actor sensor slice")
        detail["clean_actor_frame35"] = clean
        noise_readback["actor_sensor_noise"] = {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in detail.items()}
    policy_delay = getattr(env, "policy_signal_delay", None)
    if policy_delay is not None:
        detail = policy_delay.readback()
        expected_age = (detail["actual_observation_age_s"][:, [2, 3, 0, 1]] / env.dt).float()
        expected_target_age = ((env.critic_applied_at_s[:, None] - env.critic_target_sample_s) / env.dt).float()
        if (noise is None or not torch.equal(noise.input, detail["sensor_output18"])
                or not torch.equal(observations["critic"][:, 57:61], expected_age)
                or not torch.equal(observations["critic"][:, 61:63], expected_target_age)):
            raise RuntimeError("Native policy-delay timestamps differ from actor input or critic ages")
        detail.update(applied_target_sample_s=env.critic_target_sample_s,
            applied_at_s=env.critic_applied_at_s,
            applied_pd_origin_s=env.critic_pd_origin_s,
            applied_target_policy_generated_s=env.command_transport.applied_target_policy_generated_s,
            local_feedback_sensor_timestamp_s=env._pc_feedback_transport.sensor_timestamp_s)
        noise_readback["policy_signal_delay"] = {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in detail.items()}
    return {"schema": "v7-nonsteady-scene-abi-readback-v1", "policy_step": step_index,
            "physical_tick": env.physical_tick,
            "observations": {name: observations[name].detach().cpu().clone() for name in ("policy", "critic")},
            "ownership_components": components,
            "pre_reset_owner": pre, "returned_owner": after,
            "policy_learning_mask": learning, "returned_bootstrap_mask": bootstrap,
            "done": done.detach().cpu().clone(), "terminated": terminated,
            "returned_reset_mask": returned["reset_mask"].detach().cpu().clone(),
            "returned_owner_learning_mask_scope": "last paid interval retained across reset; not returned-state eligibility",
            "bootstrap_scope": "direct environment mask; GAE also uses done independently",
            "scope": "zero-policy scene/interface observation; no optimizer or recovery qualification",
            **noise_readback}
