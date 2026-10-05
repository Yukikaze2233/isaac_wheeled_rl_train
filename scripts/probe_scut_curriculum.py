#!/usr/bin/env python3
"""Frozen v17/v18 train-role native engineering probe, zero PPO.

The explicit per-domain engineering actor clock is reset before the native
birth transaction. It is never consumption, curriculum progress or success
credit. Run each domain in a separate app from a clean frozen source archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from probe_recovery_takeover import IntervalAudit, digest, load_actor, require, write_report
from recovery_prefix_support import BASELINE_PT_SHA, BASELINE_CONTRACT_SHA, validate_checkpoint_payload

ORIGINS = {"recovery": 2294, "terrain": 2186, "jump": 5290, "stepjump": 0}
POSTURE_ORIGINS = {"recovery": 2294, "speed": 5063, "terrain": 2186, "jump": 5552, "stepjump": 0}
POLICY_STEPS, ENVIRONMENTS = 240, 4096


def source_snapshot(config):
    from train_chassis import scut_curriculum_source_files
    paths = {*sorted((ROOT / "src/wheeled_tasks/chassis").glob("*.py")),
        ROOT / "src/wheeled_tasks/v40/core.py", ROOT / "src/wheeled_tasks/agents/v40_ppo_cfg.py",
        ROOT / "scripts/train_chassis.py", ROOT / "scripts/probe_recovery_takeover.py",
        ROOT / "scripts/recovery_prefix_support.py", Path(__file__).resolve(),
        ROOT / config["design_preflight"]["file"],
        *(ROOT / name for name in scut_curriculum_source_files(config))}
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def invocation(design, config):
    """Use the frozen independent train-role count and seed, without override."""
    binding = config["design_preflight"]
    row = design["stages"][binding["stage"]]["roles"]["train"]["invocation"]
    require(row["num_envs"] == ENVIRONMENTS, "Probe requires exact frozen 4096 train population")
    block = binding.get("block_index", 0)
    rule = row["seed"]
    if type(rule) is int:
        seed = rule
    elif isinstance(rule, dict) and rule.get("rule") == "root_plus_block_index":
        seed = rule["root_seed"] + block
    else:
        raise ValueError("Independent train-role seed rule is unsupported")
    require(len(config["enabled_stages"]) == 1, "One native worker stage required")
    return config["enabled_stages"][0], seed, block


def authenticate(args):
    import torch
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis import scut_curriculum_revision, ordinary_posture_revision

    config, manifest = preflight(args.contract)
    posture = "ordinary_posture_curriculum" in config
    recipe = ordinary_posture_revision if posture else scut_curriculum_revision
    require(config == recipe.canonical_worker(args.domain, ROOT), "Probe worker differs from exact frozen recipe")
    require(not posture or args.policy_steps >= 600, "Native posture probe needs a full 12s independent push window")
    require(config["task_campaign"]["domain"] == args.domain, "Domain does not match worker identity")
    require(not config.get("evaluation_exact_cases"), "An evaluation projection cannot certify train runtime")
    require(config.get("record_diagnostics") is True, "Exact worker diagnostics must be enabled")
    require((config["physics_dt"], config["policy_dt"], config["actor_dim"],
        config["critic_dim"], config["action_dim"]) == (.005, .02, 35, 81, 6),
        "Probe requires unchanged 200/50 Hz, 35/81/6 ABI")
    binding = config["design_preflight"]
    design_path = ROOT / binding["file"]
    require(digest(design_path) == binding["sha256"], "Independent design SHA changed")
    stage, seed, block = invocation(json.loads(design_path.read_text()), config)
    control = load_chassis_control(ROOT / config["control_math_source"])
    receipt = design_preflight(config, control, ROOT, stage=stage, role="train", seed=seed,
        num_envs=ENVIRONMENTS, block_index=block)
    require(receipt is not None, "Independent train-role design receipt required")
    source_path = checkpoint_contract_path(args.checkpoint)
    require(digest(args.checkpoint) == BASELINE_PT_SHA and digest(source_path) == BASELINE_CONTRACT_SHA,
        "Probe requires the immutable recommended_flat full checkpoint")
    updates = checkpoint_update_count(args.checkpoint)
    require(args.checkpoint.with_suffix(".checkpoint.json").is_file(), "Per-file immutable checkpoint seal required")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    clocks = validate_checkpoint_payload(saved, json.loads(source_path.read_text()), config, digest(source_path), updates)
    provenance = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
        checkpoint_contract=str(source_path.resolve()), checkpoint_contract_sha256=digest(source_path),
        checkpoint_learning_updates=updates, checkpoint_clocks=clocks,
        actor_only_deterministic_inference=True, seal_and_infos_authenticated=True)
    return config, manifest, control, stage, seed, block, receipt, provenance


class BirthIntervalAudit:
    """Observe the actual passive-release gate before each native substep."""
    def __init__(self, birth):
        import torch
        self.expected = torch.ones_like(birth.learning_mask)
        self.held = torch.zeros_like(self.expected)
        begin, torque = birth.begin_policy, birth.apply_torque

        def observe_begin():
            begin()
            self.expected.copy_(~birth.holding)
            self.held.zero_()

        def observe_torque(effort):
            self.held |= birth.holding
            self.expected &= ~birth.holding
            return torque(effort)

        birth.begin_policy, birth.apply_torque = observe_begin, observe_torque


class StructuralAudit:
    """Read-only runtime instrumentation; no artificial request or trajectory."""
    def __init__(self, env, domain):
        self.env, self.domain = env, domain
        self.counts, self.accepted_by_group = Counter(), Counter()
        self.max_gap = 0.
        self.recovery = (IntervalAudit(env.recovery)
            if env.recovery is not None and env.recovery.has_tasks else None)
        self.birth = BirthIntervalAudit(env.nonstandard_start) if env.nonstandard_start is not None else None
        self.latest_lessons = {}
        self.posture_initial = None
        if getattr(env, "ordinary_posture", None) is not None:
            import warp as wp
            import torch
            self.posture_initial = env.ordinary_posture.observed(env.task_actor_update)
            require(self.posture_initial["birth_selected_count"] > 0
                and self.posture_initial["birth_abs_angle_max_rad"] > 0,
                "Ordinary root-tilt birth did not physically execute")
            require(self.posture_initial["push_clock_std_s"] > 0
                and self.posture_initial["push_speed_cap_m_s"] > 0,
                "Ordinary push clock is synchronized or has zero amplitude")
            selected = env.ordinary_posture.selected
            actual = wp.to_torch(env.robot.root_view.get_root_transforms())[selected, 3:7]
            roll, pitch = (env.ordinary_posture.angles[selected] * .5).unbind(-1)
            expected = torch.stack((roll.sin() * pitch.cos(), roll.cos() * pitch.sin(),
                -roll.sin() * pitch.sin(), roll.cos() * pitch.cos()), -1)
            require(actual.numel() and torch.allclose(actual, expected, atol=2e-6, rtol=0.),
                "Ordinary root tilt differs from native simulator readback")
            self.posture_initial["native_root_quaternion_readback_passed"] = True
            apply_push = env.apply_push

            def observe_push(ids):
                from wheeled_tasks.chassis.task import Phase
                ordinary = ids[env.ordinary_posture.eligible[ids]]
                require((env.phase.phase[ordinary] == Phase.GROUND).all()
                    and (env.mode[ordinary] != 4).all(), "Native push contaminated a flight/request lane")
                if env.nonstandard_start is not None:
                    require(not env.nonstandard_start.holding[ordinary].any(), "Native push contaminated passive zero torque")
                if env.recovery is not None:
                    require(env.recovery.pure_rl[ordinary].all(), "Native push contaminated SCRIPT/BLEND")
                apply_push(ids)

            env.apply_push = observe_push
        self._accepted_cycle_observer()
        self.check_birth_exclusion()

    def finish_posture(self):
        if getattr(self.env, "ordinary_posture", None) is None:
            return None
        observed = self.env.ordinary_posture.observed(self.env.task_actor_update)
        require(observed["push_events"] > 0 and observed["push_velocity_delta_max_m_s"] > 0,
            "Native gentle pushes were not actually applied and read back")
        return dict(initial=self.posture_initial, final=observed,
            actual_native_velocity_delta_observed=True, actual_root_rotation_observed=True)

    def check_birth_exclusion(self):
        import torch
        env = self.env
        if env.nonstandard_start is None or self.domain not in ("jump", "stepjump"):
            return
        pools = {"stand_anchors", "stand_continuous", "height", "basic_translation", "basic_transitions", "basic_yaw"}
        ordinary = torch.tensor([env.cfg["behavior_pool_membership"][g] in pools
            for g in env.scene_groups], device=env.device)
        require(not (env.nonstandard_start_eligible & ~ordinary).any(), "Request pool eligible for passive birth")
        require(not (env.nonstandard_start.selected & ~ordinary).any(), "Request lane actually selected for passive birth")
        require(not (env.nonstandard_start_eligible & (env.mode == 4)).any(), "Active jump request eligible for birth")
        self.counts["birth_eligible_request_lanes"] = 0
        self.counts["ordinary_birth_eligible_lanes"] = int(env.nonstandard_start_eligible.sum())

    def _accepted_cycle_observer(self):
        env = self.env
        if not env.jump_cycle_reset_on_accepted:
            return
        original = env._reset_jump_evidence

        def observe(ids, position):
            original(ids, position)
            full = env.full_tasks
            for name in ("clearance_peak", "height_peak", "clear_air_time", "clear_air_time_peak",
                "release_velocity", "release_recorded", "com_release_height", "com_release_speed",
                "com_rise", "com_released", "com_release_pending", "com_start_height", "com_start_valid",
                "com_displacement"):
                require(getattr(full, name)[ids].eq(0).all(), "Accepted request retained old evidence: " + name)
            if env.jump_landing is not None:
                observer = env.jump_landing
                for name in ("elapsed", "request_elapsed", "verified_flight", "active", "touchdown",
                    "armed", "expired", "outcome_active", "outcome_timeout", "wheel_contact_time", "cycle_accepted"):
                    require(getattr(observer, name)[ids].eq(0).all(), "Accepted landing cycle retained state: " + name)
            self.counts["accepted_cycle_calls"] += 1
            self.counts["accepted_cycle_lanes_cleared"] += len(ids)
            self.accepted_by_group.update(env.scene_groups[i] for i in ids.detach().cpu().tolist())

        env._reset_jump_evidence = observe

    def observe(self, tick, reward, done, extras):
        import torch
        env, data = self.env, extras["diagnostics"]
        gap = data["gap"]
        require(torch.isfinite(gap).all(), "Closed-chain pin gap is nonfinite")
        self.max_gap = max(self.max_gap, float(gap.max()))
        bootstrap = extras.get("policy_bootstrap_mask")
        closure_limit = env.cfg.get("closure_gap_termination_m")
        if closure_limit:
            # A finite physical safety termination is a valid engineering
            # transition, not evidence that this untrained actor passed stairs.
            # Inspect the pre-auto-reset per-lane producer data, never the
            # historical peak across already-ended episodes.
            overflow = gap > closure_limit
            require((~overflow | (data["terminated"] & done.bool())).all(),
                "Closure overflow did not produce a native physical terminal")
            require(bootstrap is not None or not overflow.any(),
                "Closure terminal lacks its native bootstrap mask")
            if bootstrap is not None:
                require(not (bootstrap & overflow).any(), "Closure terminal incorrectly bootstraps")
            self.counts["closure_guard_terminal_lanes"] += int(overflow.sum())
        else:
            require(self.max_gap <= .003, "Unguarded native closure safety bound exceeded")
        timing = env.control_audit.last
        require(timing["pd_updates"] == timing["feedback_refreshes"] == 4
            and timing["target_decodes"] == 1 and timing["physics_steps"] == tick * 4,
            "Native feedback PD did not execute four fresh 5ms substeps")
        for name in ("motor_position", "motor_velocity", "motor_effort", "velocity", "omega", "height"):
            require(torch.isfinite(data[name]).all(), "Nonfinite native physical diagnostic: " + name)
        learning = extras.get("policy_learning_mask", torch.ones_like(done, dtype=torch.bool))
        expected = torch.ones_like(learning)
        if self.recovery is not None:
            expected &= self.recovery.expected
            excluded = self.recovery.script_seen | self.recovery.blend_seen
            require(not (learning & excluded).any(), "SCRIPT or BLEND credited to PPO")
            require(self.recovery.torque_calls == 4, "Missing substep recovery ownership observation")
            enabled = data["recovery_enabled"]
            self.counts["recovery_script_intervals"] += int(self.recovery.script_seen.sum())
            self.counts["recovery_blend_intervals"] += int(self.recovery.blend_seen.sum())
            self.counts["recovery_actor_owned_intervals"] += int((enabled & learning).sum())
            self.counts["recovery_pure_rl_observed_intervals"] += int((enabled & data["recovery_pure_rl"]).sum())
            original = data["recovery_original_chain"]
            require(int(extras["log"]["/recovery/full_chain_population"]) == int(original.sum()),
                "Full-chain denominator includes prefix lessons")
            truth = data["recovery_true_success"]
            fraction = float((original & truth).sum() / original.sum().clamp_min(1))
            require(abs(float(extras["log"]["/recovery/full_chain_success"]) - fraction) < 1e-7,
                "Easy lesson success leaked into L3 full-chain statistic")
            self.latest_lessons = {name: float(value) for name, value in extras["log"].items()
                if name.startswith("/recovery/lesson/") or name == "/recovery/full_chain_population"}
        if self.birth is not None:
            expected &= self.birth.expected
            require(not (learning & self.birth.held).any(), "Zero-active-torque birth credited to PPO")
            self.counts["passive_birth_excluded_intervals"] += int(self.birth.held.sum())
        require(learning.dtype == torch.bool and learning.shape == (env.num_envs,)
            and torch.equal(learning, expected), "Native full-interval policy ownership differs from actual substeps")
        if bootstrap is not None:
            require(not (bootstrap & data["terminated"]).any(), "Physical terminal incorrectly bootstraps")
        self.counts["all_policy_intervals"] += env.num_envs
        self.counts["native_terminal_lanes"] += int(done.sum())
        self.check_birth_exclusion()

    def summary(self):
        coverage = {"structural": "observed", "functional_qualification": False,
            "task_outcome": "not_assessed_old_flat_actor"}
        if self.domain == "recovery":
            coverage["blend_observed"] = self.counts["recovery_blend_intervals"] > 0
            coverage["pure_rl_observed"] = self.counts["recovery_pure_rl_observed_intervals"] > 0
            coverage["continuous_handoff_observed"] = coverage["blend_observed"] and coverage["pure_rl_observed"]
            if not coverage["continuous_handoff_observed"]:
                coverage["limitation"] = "bounded probe did not observe both BLEND and pure RL; handoff not functionally qualified"
        if self.domain in ("jump", "stepjump"):
            coverage["accepted_cycle_clear_observed"] = self.counts["accepted_cycle_calls"] > 0
            if not coverage["accepted_cycle_clear_observed"]:
                coverage["limitation"] = "native accepted-cycle path not exercised in bounded actor inference"
        return dict(counts=dict(self.counts), accepted_by_group=dict(self.accepted_by_group),
            max_closure_gap_m=self.max_gap, closure_guard_peak_gap_m=self.max_gap,
            latest_lesson_counters=self.latest_lessons, coverage=coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=POSTURE_ORIGINS, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--policy-steps", type=int, default=POLICY_STEPS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="scut-curriculum-native-probe-v17", status="starting", domain=args.domain,
        scope="engineering_only_no_PPO_no_qualification", formal_updates_consumed=0, optimizer_steps=0,
        qualification=None, acceptance="not_assessed", started_at_utc=datetime.now(timezone.utc).isoformat(),
        forced_engineering_actor_update=None, policy_steps_requested=args.policy_steps,
        physics_steps_requested=4 * args.policy_steps, physics_seconds_requested=.02 * args.policy_steps,
        target_contract_sha256=digest(args.contract), randomization_or_delay_overrides=[])
    path, config, source = args.output / "report.json", None, None
    launcher = env = audit = None
    try:
        config, manifest, control, stage, seed, block, receipt, provenance = authenticate(args)
        posture = "ordinary_posture_curriculum" in config
        origin = (POSTURE_ORIGINS if posture else ORIGINS)[args.domain]
        report.update(schema="scut-curriculum-native-probe-v18" if posture else report["schema"],
            forced_engineering_actor_update=origin)
        source = source_snapshot(config)
        (args.output / "contract.json").write_bytes(args.contract.read_bytes())
        report.update(design_preflight=receipt, checkpoint=provenance, num_envs=ENVIRONMENTS,
            stage=stage, run_role="train", seed=seed, block_index=block, source_files_sha256=source)
        if args.preflight_only:
            report["status"] = "preflight_passed_scene_not_run"
        else:
            write_report(path, report)
            os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
            from isaaclab.app import AppLauncher
            launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": "cuda:0",
                "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
            import torch
            from wheeled_tasks.chassis.env import ChassisEnv
            require(source_snapshot(config) == source, "Frozen probe sources changed before scene construction")
            torch.set_num_threads(4)
            torch.manual_seed(seed)
            env = ChassisEnv(config, manifest, control, ROOT, stage_name=stage, num_envs=ENVIRONMENTS,
                device="cuda:0", level=1., seed=seed, run_role="train", block_index=block)
            require(env.startup_report.get("gpu_dynamics_enabled") is True, "Native GPU dynamics required")
            require(env.startup_report.get("runtime_preflight", {}).get("status") == "runtime_configuration_passed",
                "Exact independent runtime expectation did not pass")
            actor, before = load_actor(env, args.checkpoint, config, provenance, seed)
            env.task_actor_update = env.stage_actor_update = env.global_actor_update = origin
            env.reset(torch.arange(ENVIRONMENTS, device=env.device))
            audit = StructuralAudit(env, args.domain)
            observations = env.get_observations()
            report.update(startup=deepcopy(env.startup_report), initial_summary=audit.summary(),
                engineering_clock_reset_before_birth=True)
            start = time.monotonic()
            for tick in range(1, args.policy_steps + 1):
                require(launcher.app.is_running(), "Simulator exited before bounded probe completion")
                with torch.inference_mode():
                    action = actor(observations["policy"])
                    require(action.shape == (ENVIRONMENTS, 6) and torch.isfinite(action).all(), "Invalid actor output")
                    observations, reward, done, extras = env.step(action)
                require(observations["policy"].shape == (ENVIRONMENTS, 35)
                    and observations["critic"].shape == (ENVIRONMENTS, 81), "Native observation ABI changed")
                require(all(torch.isfinite(value).all() for value in (*observations.values(), reward)), "Nonfinite observation or reward")
                audit.observe(tick, reward, done, extras)
            require(all(torch.equal(value.detach().cpu(), before[name]) for name, value in actor.state_dict().items()),
                "Engineering inference changed actor tensors")
            require(digest(args.checkpoint) == provenance["checkpoint_sha256"], "Immutable source checkpoint changed")
            require(env.task_actor_update == env.stage_actor_update == env.global_actor_update == origin,
                "Zero-update engineering actor clock advanced")
            torch.cuda.synchronize()
            posture_proof = audit.finish_posture()
            report.update(status="engineering_probe_completed", policy_steps=args.policy_steps,
                physics_steps=4 * args.policy_steps, elapsed_physics_s=.02 * args.policy_steps, wall_seconds=time.monotonic() - start,
                actor_tensors_unchanged=True, checkpoint_unchanged=True,
                ordinary_posture_native_proof=posture_proof,
                control_timing=dict(env.control_audit.last), summary=audit.summary(),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        if audit is not None:
            report["summary"] = audit.summary()
        traceback.print_exc()
    finally:
        if source is not None:
            after = source_snapshot(config)
            changed = [name for name in sorted(source.keys() | after.keys()) if source.get(name) != after.get(name)]
            report.update(source_files_changed=changed, source_files_unchanged=not changed,
                target_contract_unchanged=digest(args.contract) == report["target_contract_sha256"])
            if changed or not report["target_contract_unchanged"]:
                report["status"] = "failed_source_changed"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(path, report)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    print(json.dumps(dict(status=report["status"], report=str(path), formal_updates_consumed=0)))
    return 0 if report["status"] in ("engineering_probe_completed", "preflight_passed_scene_not_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
