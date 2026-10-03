#!/usr/bin/env python3
"""Bounded native recovery takeover probe, with no PPO or qualification credit.

The exact train-role worker and independent design table remain authoritative.
--actor-update selects an engineering curriculum clock, then resets all lanes
through the ordinary reset path. It does not restore or consume a training
clock, change a contract, disable randomization, or modify a fixed evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GATES = ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed")
AXES = ("orientation", "angular_velocity", "hold_leg_velocity", "hold_wheel_velocity")
COST_NAMES = tuple(f"recovery_takeover_{axis}_{suffix}" for axis in AXES
                   for suffix in ("broad", "precision"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot_sources():
    paths = [*sorted((ROOT / "src/wheeled_tasks/chassis").glob("*.py")),
             ROOT / "src/wheeled_tasks/agents/v40_ppo_cfg.py", ROOT / "scripts/train_chassis.py",
             Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def authenticate(args):
    """Read the independent train-role invocation; never derive expectations."""
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count
    from wheeled_tasks.chassis.preflight import design_preflight

    config, manifest = preflight(args.contract)
    binding = config["design_preflight"]
    design_path = ROOT / binding["file"]
    require(digest(design_path) == binding["sha256"], "Independent design SHA changed")
    design = json.loads(design_path.read_text())
    invocation = design["stages"][binding["stage"]]["roles"]["train"]["invocation"]
    count = args.num_envs if args.num_envs is not None else invocation["num_envs"]
    block = binding.get("block_index", 0)
    rule = invocation["seed"]
    if isinstance(rule, int):
        seed = rule
    elif rule.get("rule") == "root_plus_block_index":
        seed = rule["root_seed"] + block
    else:
        raise ValueError("Probe requires an explicit independent train-role seed rule")
    require(len(config["enabled_stages"]) == 1, "Probe needs one exact worker stage")
    stage = config["enabled_stages"][0]
    control = load_chassis_control(ROOT / config["control_math_source"])
    receipt = design_preflight(config, control, ROOT, stage=stage, role="train", seed=seed,
                               num_envs=count, block_index=block)
    require(receipt is not None, "Independent train-role preflight is mandatory")
    require((config["physics_dt"], config["policy_dt"], config["actor_dim"], config["critic_dim"],
             config["action_dim"]) == (.005, .02, 35, 81, 6), "Probe requires 200/50 Hz and 35/81/6")
    require(config.get("record_diagnostics") is True, "Frozen worker must enable native diagnostics")
    require(config.get("recovery_takeover", {}).get("enabled") is True,
            "Worker does not enable the requested recovery takeover costs")
    require(config.get("recovery_training", {}).get("pose_curriculum", {}).get("enabled") is True,
            "Worker does not enable the requested recovery pose curriculum")
    require(not config.get("evaluation_exact_cases"), "Probe requires the unchanged train role")
    updates = checkpoint_update_count(args.checkpoint)
    source_path = checkpoint_contract_path(args.checkpoint)
    source = json.loads(source_path.read_text())
    # A frozen flat parent or recovery checkpoint can supply the actor. This is
    # inference provenance, not authorization to migrate an optimizer or budget.
    for key in ("asset_manifest_sha256", "control_math_sha256", "physics_dt", "policy_dt",
                "actor_dim", "critic_dim", "action_dim", "actor_observation_source",
                "policy_action_order", "policy_architecture", "v5_control"):
        require(source.get(key) == config.get(key), f"Probe actor changes physical/observation ABI: {key}")
    provenance = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_contract": str(source_path.resolve()), "checkpoint_contract_sha256": digest(source_path),
        "checkpoint_learning_updates": updates, "actor_only_deterministic_inference": True}
    return config, manifest, control, stage, seed, count, block, receipt, provenance


def load_actor(env, checkpoint_path, config, provenance, seed):
    """Use the same strict native actor loader as the existing physical probe."""
    import torch
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg

    cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
    cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    cfg.seed, cfg.device = seed, env.device
    runner = OnPolicyRunner(env, deepcopy(cfg.to_dict()), log_dir=None, device=env.device)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(all(key in saved for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "infos")),
            "Probe input must be a full learning checkpoint")
    infos = saved["infos"]
    require(infos.get("contract_sha256") == provenance["checkpoint_contract_sha256"],
            "Checkpoint infos do not authenticate the source contract")
    require(infos.get("asset_manifest_sha256") == config["asset_manifest_sha256"]
            and infos.get("control_math_sha256") == config["control_math_sha256"],
            "Checkpoint infos change the target asset or control identity")
    runner.alg.actor.load_state_dict(saved["actor_state_dict"], strict=True)
    actor = runner.alg.actor.as_onnx(verbose=False).to(env.device).eval()
    before = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    require(all(torch.isfinite(value).all() for value in before.values()), "Checkpoint actor has nonfinite values")
    # A runner is constructed solely for its established actor factory. Its
    # learn/update/compute_returns methods are never called; no rollout is stored.
    del saved, runner
    return actor, before


class IntervalAudit:
    """Observe actual substep control ownership without changing any output."""
    def __init__(self, recovery):
        import torch
        self.recovery = recovery
        self.expected = torch.ones_like(recovery.enabled)
        self.script_seen = torch.zeros_like(self.expected)
        self.blend_seen = torch.zeros_like(self.expected)
        self.hold = torch.zeros_like(self.expected)
        self.torque_calls = 0
        begin, torque = recovery.begin_policy, recovery.torque

        def observe_begin():
            begin()
            self.expected.copy_(recovery.pure_rl)
            self.script_seen.copy_(recovery.enabled & (recovery.fsm.scripted | ~recovery.release_finished))
            self.blend_seen.copy_(recovery.enabled & recovery.fsm.blending)
            self.hold.copy_(recovery.policy_motion_hold)
            self.torque_calls = 0

        def observe_torque(q, dq, rl_torque):
            self.script_seen |= recovery.enabled & (recovery.fsm.scripted | ~recovery.release_finished)
            self.blend_seen |= recovery.enabled & recovery.fsm.blending
            self.expected &= ~recovery.enabled | (recovery.release_finished
                              & (recovery.fsm.pure_rl | recovery.fsm.failed))
            self.torque_calls += 1
            return torque(q, dq, rl_torque)

        recovery.begin_policy, recovery.torque = observe_begin, observe_torque


class Recorder:
    """Record all resets and terminals, plus bounded representative step traces."""
    def __init__(self, env, output, actor_update):
        import torch
        from wheeled_tasks.chassis.recovery_curriculum import pose_probabilities
        self.env, self.output, self.actor_update = env, output, actor_update
        self.curriculum = env.recovery.pose_curriculum
        require(self.curriculum is not None, "Runtime pose curriculum is absent")
        self.table = json.loads((ROOT / env.cfg["recovery_training"]["profiles_file"]).read_text())
        self.probabilities = pose_probabilities(self.curriculum.settings, actor_update)
        self.episodes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.handed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.released = torch.zeros_like(self.handed)
        self.counts, self.profile_counts, self.level_counts = Counter(), Counter(), Counter()
        self.gates = {name: {"passed": 0, "denominator": 0} for name in (*GATES, "all")}
        self.reward_ranges = {name: {"minimum": None, "maximum": None, "samples": 0} for name in COST_NAMES}
        self.max_stable, self.max_true_stable, self.max_breaks = 0., 0., 0
        self.initial = (output / "initial_states.jsonl").open("w")
        self.terminal = (output / "terminals.jsonl").open("w")
        self.trace = (output / "trace.jsonl").open("w")
        self.population = (output / "population.jsonl").open("w")
        representatives = {}
        for index, group in enumerate(env.scene_groups):
            if env.skills.specs.get(group, {}).get("recovery_pose"):
                representatives.setdefault(group, index)
        self.representatives = list(representatives.values())

    @staticmethod
    def append(stream, rows):
        stream.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        stream.flush()

    def reset_rows(self, ids, tick):
        """Read back the current full pose, not merely the requested profile."""
        import torch
        env, course = self.env, self.curriculum
        ids = ids[env.recovery.enabled[ids]]
        if not len(ids):
            return
        pids, lids = course.profile_ids[ids], course.level_ids[ids]
        require(((pids >= 0) & (lids >= 0) & (lids < 3)).all(), "Missing sampled recovery profile/level")
        require((course.candidates[ids, lids] == pids[:, None]).any(-1).all(),
                "Runtime sampled a profile outside its terrain-compatible level pool")
        actual_root = env.robot.data.root_link_pose_w.torch[ids].clone()
        actual_joint = env.robot.data.joint_pos.torch[ids].clone()
        expected_root = course.root_pose_bank[pids].clone()
        expected_root[:, :3] += env.origins[ids]
        expected_joint = course.joint_pose_bank[pids]
        require(torch.allclose(actual_root, expected_root, atol=2e-5, rtol=0.),
                "Actual reset root pose differs from selected frozen recovery geometry")
        require(torch.allclose(actual_joint, expected_joint, atol=2e-5, rtol=0.),
                "Actual reset full joint pose differs from selected frozen recovery geometry")
        arrays = {"root_pose_world_xyzw": actual_root,
            "root_pose_local_xyzw": torch.cat((actual_root[:, :3] - env.origins[ids], actual_root[:, 3:]), -1),
            "full_joint_positions_rad": actual_joint,
            "full_joint_velocities_rad_s": env.robot.data.joint_vel.torch[ids],
            "root_linear_velocity_world_m_s": env.robot.data.root_link_lin_vel_w.torch[ids],
            "root_angular_velocity_body_rad_s": env.robot.data.root_com_ang_vel_b.torch[ids],
            "expected_root_pose_world_xyzw": expected_root, "expected_full_joint_positions_rad": expected_joint}
        require(all(torch.isfinite(value).all() for value in arrays.values()), "Nonfinite full recovery reset state")
        arrays = {name: tensor.cpu().tolist() for name, tensor in arrays.items()}
        lanes, profiles, levels = ids.tolist(), pids.tolist(), lids.tolist()
        self.episodes[ids] += 1
        episodes = self.episodes[ids].tolist()
        rows = []
        for index, (lane, profile_id, level_id, episode) in enumerate(zip(lanes, profiles, levels, episodes)):
            name = course.profile_names[profile_id]
            profile = self.table["profiles"][name]
            require(profile.get("terrain", "flat") == env.kinds[lane], "Sampled initial pose changes lane terrain")
            level = ("easy", "medium", "hard")[level_id]
            self.profile_counts[name] += 1
            self.level_counts[level] += 1
            rows.append({"policy_step": tick, "lane": lane, "episode": episode,
                "group": env.scene_groups[lane], "terrain": env.kinds[lane], "profile": name,
                "profile_id": profile_id, "level": level, "level_id": level_id,
                "forced_engineering_actor_update": self.actor_update,
                "level_probabilities": dict(zip(("easy", "medium", "hard"), self.probabilities)),
                "required_platform_height_m": profile.get("required_platform_height_m", 0.),
                "initial_release_s": profile.get("initial_release_s", 0.),
                "terrain_compatible": True, "asset_manifest_sha256": env.cfg["asset_manifest_sha256"],
                **{key: values[index] for key, values in arrays.items()}})
        self.counts["recovery_episodes_started"] += len(rows)
        self.append(self.initial, rows)
        self.handed[ids] = self.released[ids] = False

    def observe(self, tick, reward, done, extras, interval):
        import torch
        env, data = self.env, extras["diagnostics"]
        learning, bootstrap = extras["policy_learning_mask"], extras["policy_bootstrap_mask"]
        enabled = data["recovery_enabled"]
        require(learning.dtype == bootstrap.dtype == torch.bool
                and tuple(learning.shape) == tuple(bootstrap.shape) == (env.num_envs,),
                "Missing distinct policy ownership/bootstrap masks")
        require(torch.equal(learning, interval.expected), "Actual substep ownership differs from PPO eligibility")
        require(not (learning & (interval.script_seen | interval.blend_seen)).any(),
                "Script/blend interval was credited to PPO")
        require(not (bootstrap & data["terminated"]).any(), "Physical terminal incorrectly bootstraps")
        require(interval.torque_calls == env.decimation, "Recovery torque ownership was not checked every substep")
        owned = enabled & learning
        held = owned & interval.hold
        native = extras.get("native_reward_components", {})
        require(set(COST_NAMES) <= set(native), "Native takeover reward component evidence is missing")
        for name in COST_NAMES:
            value = native[name]
            require(value.shape == reward.shape and torch.isfinite(value).all() and (value <= 1e-8).all(),
                    f"Takeover cost is nonfinite, positive, or wrongly shaped: {name}")
            require(value[~owned].eq(0.).all(), f"Takeover reward leaks into nonactor/nonrecovery intervals: {name}")
            row = self.reward_ranges[name]
            low, high = float(value.min()), float(value.max())
            row.update(minimum=low if row["minimum"] is None else min(row["minimum"], low),
                       maximum=high if row["maximum"] is None else max(row["maximum"], high),
                       samples=row["samples"] + value.numel())
        for name in ("track_lin_vel", "track_yaw", "track_height", "pitch_exp", "roll_exp",
                     "dense_height_broad", "dense_height_precision", "dense_stationary_broad", "dense_stationary_precision"):
            require(torch.isfinite(native[name][owned]).all() and (native[name][owned] <= 1e-8).all(),
                    f"Owned takeover retains a positive tracking baseline: {name}")
        handed = enabled & (data["recovery_handover_count"] > 0)
        released = enabled & data["recovery_motion_released"]
        self.counts["episodes_handed_over"] += int((handed & ~self.handed).sum())
        self.counts["episodes_motion_released"] += int((released & ~self.released).sum())
        self.handed |= handed
        self.released |= released
        for name in (*GATES, "all"):
            key = "recovery_stable_pass" if name == "all" else f"recovery_stable_{name}_pass"
            require(key in data, f"Missing actual hold-gate measurement: {key}")
            self.gates[name]["passed"] += int((held & data[key]).sum())
            self.gates[name]["denominator"] += int(held.sum())
        self.max_stable = max(self.max_stable, float(data["recovery_stable_peak_seconds"].max()))
        self.max_true_stable = max(self.max_true_stable, float(data["recovery_rl_stable_s"].max()))
        self.max_breaks = max(self.max_breaks, int(data["recovery_stable_break_count"].max()))
        values = {"all_intervals": env.num_envs, "recovery_intervals": int(enabled.sum()),
            "script_intervals": int(interval.script_seen.sum()), "blend_intervals": int(interval.blend_seen.sum()),
            "actor_owned_recovery_intervals": int(owned.sum()), "actor_hold_intervals": int(held.sum()),
            "actor_terminal_eligible": int((owned & data["terminated"]).sum()),
            "excluded_terminal_intervals": int((enabled & ~learning & data["terminated"]).sum())}
        self.counts.update(values)
        self.append(self.population, [{"policy_step": tick, **values,
            "stable_peak_seconds": self.max_stable, "true_stable_peak_seconds": self.max_true_stable}])
        terminal = enabled & done.bool()
        failed = terminal & data["terminated"]
        self.counts["recovery_episodes_ended"] += int(terminal.sum())
        self.counts["full_chain_success"] += int((terminal & data["recovery_true_success"]).sum())
        self.counts["failed_before_handover_script_or_blend"] += int((failed & ~self.handed).sum())
        self.counts["handed_over_failed"] += int((failed & self.handed).sum())
        self.counts["hold_failed"] += int((failed & self.handed & ~self.released).sum())
        self.counts["released_then_failed"] += int((failed & self.handed & self.released).sum())
        self.counts["timed_out_or_truncated"] += int((terminal & ~data["terminated"] & ~data["success"]).sum())
        ids = terminal.nonzero().flatten()
        if len(ids):
            self.append(self.terminal, self.trace_rows(ids, tick, reward, extras, interval))
        if tick == 1 or tick % 10 == 0:
            ids = torch.tensor(self.representatives, device=env.device, dtype=torch.long)
            self.append(self.trace, self.trace_rows(ids, tick, reward, extras, interval))
        # The native env already autoreset ended lanes. Read that actual new
        # state only after recording all old-episode diagnostics and attribution.
        self.reset_rows(ids=terminal.nonzero().flatten(), tick=tick)

    def trace_rows(self, ids, tick, reward, extras, interval):
        import torch
        data = extras["diagnostics"]
        fields = {name: value for name, value in data.items() if name.startswith("recovery_")
                  and isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.env.num_envs}
        fields.update({name: data[name] for name in ("gravity", "omega", "height", "velocity", "commands",
                       "motor_position", "motor_velocity", "motor_effort", "done", "terminated", "success")})
        fields.update(reward=reward, policy_learning_mask=extras["policy_learning_mask"],
            policy_bootstrap_mask=extras["policy_bootstrap_mask"], interval_motion_hold=interval.hold,
            interval_script_seen=interval.script_seen, interval_blend_seen=interval.blend_seen,
            episode=self.episodes, episode_ever_handed_over=self.handed,
            episode_ever_motion_released=self.released)
        arrays = {name: value[ids].detach().cpu().tolist() for name, value in fields.items()}
        reasons = {name: value[ids].cpu().tolist() for name, value in data["reasons"].items()}
        return [{"policy_step": tick, "time_s": tick * self.env.policy_dt, "lane": lane,
                 "group": self.env.scene_groups[lane], **{name: values[i] for name, values in arrays.items()},
                 "termination_reasons": [name for name, values in reasons.items() if values[i]]}
                for i, lane in enumerate(ids.tolist())]

    def summary(self):
        counts = dict(self.counts)
        def ratio(numerator, denominator):
            n, d = self.counts[numerator], self.counts[denominator]
            return {"numerator": n, "denominator": d, "fraction": n / d if d else None}
        return {"counts": counts, "ratios": {
            "handover_per_started_episode": ratio("episodes_handed_over", "recovery_episodes_started"),
            "full_chain_per_completed_episode": ratio("full_chain_success", "recovery_episodes_ended"),
            "full_chain_per_handed_over_episode": ratio("full_chain_success", "episodes_handed_over"),
            "hold_failure_per_handed_over_episode": ratio("hold_failed", "episodes_handed_over"),
            "failure_after_handover_per_handed_over_episode": ratio("handed_over_failed", "episodes_handed_over")},
            "gate_denominator": "policy intervals wholly actor-owned and holding at interval start",
            "hold_gates": {name: dict(value, fraction=value["passed"] / value["denominator"]
                                      if value["denominator"] else None) for name, value in self.gates.items()},
            "profile_reset_counts": dict(self.profile_counts), "level_reset_counts": dict(self.level_counts),
            "level_probabilities": dict(zip(("easy", "medium", "hard"), self.probabilities)),
            "takeover_cost_densities": self.reward_ranges, "max_stable_peak_seconds": self.max_stable,
            "max_true_stable_peak_seconds": self.max_true_stable, "max_stable_break_count": self.max_breaks,
            "terminal_actor_ownership_observed": self.counts["actor_terminal_eligible"] > 0,
            "unobserved_event_is_not_a_pass": True}

    def close(self):
        for stream in (self.initial, self.terminal, self.trace, self.population):
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--actor-update", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=24.)
    parser.add_argument("--num-envs", type=int, help="Must match the frozen independent train-role count")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.actor_update < 0 or not math.isfinite(args.seconds) or not 0 < args.seconds <= 120.:
        parser.error("actor-update must be nonnegative and seconds must be in (0,120]")
    if args.num_envs is not None and not 1 <= args.num_envs <= 20000:
        parser.error("num-envs must be in [1,20000] and match the independent design")
    args.output.mkdir(parents=True, exist_ok=False)
    source = snapshot_sources()
    report = {"schema": "recovery-takeover-native-probe-v1", "status": "starting",
        "scope": "engineering_only_no_PPO_no_qualification", "formal_updates_consumed": 0,
        "optimizer_steps": 0, "qualification": None, "acceptance": "not_assessed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_files_sha256": source,
        "target_contract": str(args.contract.resolve()), "target_contract_sha256": digest(args.contract),
        "forced_engineering_actor_update": args.actor_update,
        "requested_physics_seconds": args.seconds, "randomization_or_delay_overrides": []}
    path = args.output / "report.json"
    write_report(path, report)
    launcher = env = recorder = None
    try:
        cfg, manifest, control, stage, seed, count, block, receipt, provenance = authenticate(args)
        (args.output / "contract.json").write_bytes(args.contract.read_bytes())
        report.update(design_preflight=receipt, checkpoint=provenance, num_envs=count,
                      stage=stage, run_role="train", seed=seed, block_index=block)
        if args.preflight_only:
            report["status"] = "preflight_passed_scene_not_run"
            return 0
        write_report(path, report)
        os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": "cuda:0",
            "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
        import torch
        from wheeled_tasks.chassis.env import ChassisEnv
        require(snapshot_sources() == source, "Probe source changed before scene construction")
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        env = ChassisEnv(cfg, manifest, control, ROOT, stage_name=stage, num_envs=count,
            device="cuda:0", level=1., seed=seed, run_role="train", block_index=block)
        require(env.recovery is not None and env.recovery.has_tasks, "No recovery lanes instantiated")
        require(env.startup_report.get("gpu_dynamics_enabled") is True, "Probe requires native GPU dynamics")
        require(env.startup_report.get("runtime_preflight", {}).get("status") == "runtime_configuration_passed",
                "Runtime independent design comparison did not pass")
        actor, before = load_actor(env, args.checkpoint, cfg, provenance, seed)
        env.task_actor_update = env.stage_actor_update = env.global_actor_update = args.actor_update
        env.reset(torch.arange(count, device=env.device))
        observations = env.get_observations()
        interval = IntervalAudit(env.recovery)
        recorder = Recorder(env, args.output, args.actor_update)
        recorder.reset_rows(torch.arange(count, device=env.device), 0)
        report.update(status="running", startup=deepcopy(env.startup_report),
            full_joint_names=env.robot.joint_names, active_joint_names=manifest["control_joint_names"],
            initial_state_recording="every recovery episode reset, actual native full state plus chosen bank geometry",
            curriculum_settings=deepcopy(env.recovery.pose_curriculum.settings),
            source_profiles_sha256=cfg["recovery_training"]["profiles_sha256"])
        write_report(path, report)
        started = time.monotonic()
        steps = max(1, math.ceil(args.seconds / env.policy_dt - 1e-9))
        for tick in range(1, steps + 1):
            require(launcher.app.is_running(), "Simulator exited before bounded probe completion")
            with torch.inference_mode():
                action = actor(observations["policy"])
                require(tuple(action.shape) == (count, 6) and torch.isfinite(action).all(), "Invalid fixed actor output")
                observations, reward, done, extras = env.step(action)
            require(tuple(observations["policy"].shape) == (count, 35)
                    and tuple(observations["critic"].shape) == (count, 81), "35/81 observation ABI changed")
            require(all(torch.isfinite(value).all() for value in (*observations.values(), reward)),
                    "Nonfinite native observations or reward")
            recorder.observe(tick, reward, done, extras, interval)
            timing = env.control_audit.last
            require(timing["pd_updates"] == timing["feedback_refreshes"] == 4
                    and timing["target_decodes"] == 1 and timing["physics_steps"] == tick * 4,
                    "Native feedback/PD clock differs from 200 Hz physics / 50 Hz policy")
            if tick == 1:
                report["first_step"] = {"control_timing": dict(timing), "actor_shape": list(action.shape),
                    "observation_shapes": {key: list(observations[key].shape) for key in ("policy", "critic")},
                    "runtime_substep_verification": env.startup_report["control_timing"]["runtime_substep_verification"]}
            if tick == 1 or tick % 50 == 0:
                report.update(policy_steps=tick, elapsed_physics_s=tick * env.policy_dt,
                    wall_seconds=time.monotonic() - started, summary=recorder.summary())
                write_report(path, report)
        require(all(torch.equal(value.detach().cpu(), before[name]) for name, value in actor.state_dict().items()),
                "Probe changed actor tensors")
        require(digest(args.checkpoint) == provenance["checkpoint_sha256"], "Source checkpoint changed during inference")
        require(env.task_actor_update == env.stage_actor_update == env.global_actor_update == args.actor_update,
                "Engineering curriculum clock advanced during a zero-update probe")
        torch.cuda.synchronize()
        report.update(status="engineering_probe_completed", policy_steps=steps,
            elapsed_physics_s=steps * env.policy_dt, wall_seconds=time.monotonic() - started,
            actor_tensors_unchanged=True, checkpoint_unchanged=True, control_timing=env.control_audit.last,
            summary=recorder.summary(), gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        after = snapshot_sources()
        changed = [name for name in sorted(source.keys() | after.keys()) if source.get(name) != after.get(name)]
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), source_files_changed=changed)
        if changed:
            report["status"] = "failed_source_changed"
        if recorder is not None:
            report["summary"] = recorder.summary()
            recorder.close()
        write_report(path, report)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    print(json.dumps({"status": report["status"], "report": str(path), "formal_updates_consumed": 0}))
    return 0 if report["status"] in ("engineering_probe_completed", "preflight_passed_scene_not_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
