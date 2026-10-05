#!/usr/bin/env python3
"""Same sealed native actor, clean physics, mean versus its saved Gaussian.

One fresh process runs one arm. This is inference only: the native environment
keeps its terminal predicates, and only the first N4 cohorts receive metrics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from recovery_prefix_support import CORE_FILES, CORE_REVISION, digest, require, write_json
from probe_locomotion_quality import trajectory

POLICY_STEPS, PHYSICS_DT, POLICY_DT = 1000, .005, .02
SAMPLING_SEED = 93001
OWNED_SCRIPTS = ("scripts/probe_locomotion_exploration.py", "scripts/locomotion_exploration_sampling.py",
                 "scripts/locomotion_exploration_fixture.py", "scripts/probe_locomotion_quality.py",
                 "scripts/recovery_prefix_support.py")


def source_files(root=ROOT):
    root = Path(root)
    paths = [*sorted((root / "src/wheeled_tasks/chassis").glob("*.py")),
             root / "src/wheeled_tasks/v40/core.py", root / "src/wheeled_tasks/agents/v40_ppo_cfg.py",
             *(root / name for name in OWNED_SCRIPTS), root / "scripts/train_chassis.py"]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def authenticate_source(root, revision):
    """Require one complete clean checkout and every recorded file at H."""
    root = Path(root).resolve()
    require(isinstance(revision, str) and len(revision) == 40
            and all(c in "0123456789abcdef" for c in revision), "Full lowercase source revision required")

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout

    require(git("rev-parse", "--show-toplevel").decode().strip() == str(root), "Complete independent checkout required")
    require(git("rev-parse", "HEAD").decode().strip() == revision, "Actual Git HEAD differs from frozen H")
    require(not git("status", "--porcelain", "--untracked-files=all").strip(), "Frozen source checkout must be clean")
    actual = source_files(root)
    for name, value in actual.items():
        require(hashlib.sha256(git("show", f"{revision}:{name}")).hexdigest() == value,
                f"Actual source is not from H: {name}")
    core = {}
    for name in CORE_FILES:
        core[name] = hashlib.sha256(git("show", f"{CORE_REVISION}:{name}")).hexdigest()
        require(actual.get(name) == core[name], f"Physical/control core differs from b4: {name}")
    return {"source_revision": revision, "base_core_revision": CORE_REVISION,
            "core_files_sha256": core, "actual_source_files_sha256": actual,
            "complete_clean_checkout": True, "source_root": str(root)}


def schedule(speed):
    """Reuse the native quality trajectory, then retain zero-command hold."""
    require(speed in (0., .5, -.5), "Only the three fixed profiles are authorized")
    phases = trajectory(speed, POLICY_DT, 1.5)
    result = [(phase, vx) for phase, count, vx in phases for _ in range(count)]
    require(len(result) <= POLICY_STEPS, "Trajectory exceeds 20 seconds")
    return result + [("hold", 0.)] * (POLICY_STEPS - len(result))


def json_value(value):
    if hasattr(value, "detach"):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite diagnostic JSON")
    return value


class PhaseMetrics:
    """Count every eligible first-cohort physical frame; no invented gate."""
    def __init__(self, case_names, profiles):
        self.case_names = tuple(case_names)
        self.plans = {name: schedule(profiles[name]) for name in profiles}
        self.expected = {name: {phase: 4 * sum(p == phase for p, _ in plan)
                               for phase in dict.fromkeys(p for p, _ in plan)}
                         for name, plan in self.plans.items()}
        self.rows = defaultdict(lambda: {"samples": 0, "first_tick": None, "last_tick": None,
            "sums": defaultdict(float), "peak_displacement_m": 0., "anchor_xy_m": None})

    def observe(self, row):
        lane, phase, tick = row["lane"], row["phase"], row["physics_tick"]
        require(type(lane) is int and 0 <= lane < len(self.case_names)
                and type(tick) is int and 1 <= tick <= POLICY_STEPS * 4,
                "Physical frame lies outside bounded fixture")
        require(self.plans[self.case_names[lane]][(tick - 1) // 4][0] == phase,
                "Physical phase differs from native trajectory clock")
        value = self.rows[lane, phase]
        require(value["last_tick"] is None or tick == value["last_tick"] + 1,
                "First-cohort phase lost a physical frame")
        if not value["samples"]:
            value["first_tick"] = tick
            value["anchor_xy_m"] = row["phase_anchor_xy_m"]
        value["last_tick"] = tick
        value["samples"] += 1
        velocity, command = row["root_link_velocity_body_m_s"], row["command"]
        point = row["root_position_local_m"][:2]
        displacement = math.dist(point, value["anchor_xy_m"])
        speed = math.hypot(*row["root_link_velocity_world_m_s"][:2])
        for name, sample in {"planar_speed_m_s": speed, "vx_mae_m_s": abs(velocity[0] - command[0]),
                "yaw_mae_rad_s": abs(row["root_angular_velocity_body_rad_s"][2] - command[1]),
                "height_mae_m": abs(row["height_m"] - command[2]),
                "pitch_abs_deg": abs(row["pitch_deg"]), "roll_abs_deg": abs(row["roll_deg"])}.items():
            value["sums"][name] += sample
        value["peak_displacement_m"] = max(value["peak_displacement_m"], displacement)
        value["endpoint_displacement_m"] = displacement

    def finish(self, ledger):
        result = []
        for lane, case in enumerate(self.case_names):
            phases = {}
            for phase, expected in self.expected[case].items():
                value = self.rows.get((lane, phase))
                count = value["samples"] if value else 0
                complete = count == expected
                metrics = ({name: total / count for name, total in value["sums"].items()}
                           if count else None)
                phases[phase] = {"observed_physics_frames": count, "expected_physics_frames": expected,
                    "observed_seconds": count * PHYSICS_DT, "complete_exposure": complete,
                    "observed_partial_means": metrics, "completed_means": metrics if complete else None,
                    "peak_displacement_m": value["peak_displacement_m"] if count else None,
                    "endpoint_displacement_m": value["endpoint_displacement_m"] if count else None,
                    "anchor_xy_m": value["anchor_xy_m"] if count else None}
                if phase in ("stand", "steady", "stop"):
                    ledger.record_window(lane, phase, metrics["planar_speed_m_s"] if count else None,
                                         complete=complete, observed_seconds=count * PHYSICS_DT)
            result.append({"lane": lane, "case": case, "phases": phases})
        return result


class PhysicalRecorder:
    """Observe original methods once, retaining immediate immutable copies."""
    def __init__(self, env, ledger, profiles, stream, delay_stream=None):
        self.env, self.ledger, self.stream, self.delay_stream = env, ledger, stream, delay_stream
        self.metrics = PhaseMetrics(env.scene_groups, profiles)
        self.in_policy, self.pending_physics = False, False
        self.physics_tick = self.policy_tick = 0
        self.executed_physics_steps = 0
        self.completed_policy_steps = 0
        self.cache, self.phase, self.anchors = {}, {}, {}
        self.counts = defaultdict(int)
        self.originals = []
        self.delay = getattr(env, "signal_delay", None)
        self.delay_totals = defaultdict(int)
        self.target_delay_rows = [None] * len(env.scene_groups)
        self.sensor_delay_rows = [{"event": "native_reset_feedback", "append_physics_tick": 0,
            "source_physics_tick": [0]*4, "actual_age_steps": [0]*4, "actual_age_ms": [0.]*4,
            "raw_jitter_ms": None, "raw_jitter_coverage": "not_covered_native_draw_is_not_retained"}
            for _ in env.scene_groups]
        require(self.delay is None or self.delay_stream is not None, "Native signal delay needs its observed journal")
        if self.delay is not None:
            for row,state in zip(self.sensor_delay_rows,self.delay_state_rows(),strict=True):
                row.update(state)
        self.install()

    def wrap(self, obj, name, factory):
        original = getattr(obj, name)
        self.originals.append((obj, name, original))
        setattr(obj, name, factory(original))

    def install(self):
        env = self.env

        def decode(original):
            def call(actions, q):
                result = original(actions, q)
                if self.in_policy:
                    self.counts["decode"] += 1
                    self.cache.update(decoded_leg_targets=result[0].clone(), decoded_wheel_targets=result[1].clone(),
                                      clipped_control_actions=result[2].clone(), decode_q=q.clone())
                return result
            return call
        self.wrap(env.v5, "decode", decode)

        def effort(original):
            def call(q, dq, legs, wheels):
                if self.in_policy and self.delay is not None:
                    fresh = ((q == env.robot.data.joint_pos.torch[:, env.ids]).all(-1)
                        & (dq == env.robot.data.joint_vel.torch[:, env.ids]).all(-1))
                    require(bool(fresh.all()), "Pipeline age must not delay physical PD feedback")
                    self.cache["pd_feedback_matches_fresh_physics"] = fresh.clone()
                result = original(q, dq, legs, wheels)
                if self.in_policy:
                    self.counts["motor_efforts"] += 1
                    self.cache.update(pd_q=q.clone(), pd_dq=dq.clone(), pd_leg_targets=legs.clone(),
                        pd_wheel_targets=wheels.clone(), pd_capped_torque=result.clone(),
                        requested_motor_effort=env.v5.requested_motor_effort.clone(),
                        current_motor_bounds=env.v5.current_motor_bounds.clone())
                return result
            return call
        self.wrap(env.v5, "motor_efforts", effort)

        def response(original):
            def call(torque):
                before = torque.clone() if self.in_policy else None
                result = original(torque)
                if self.in_policy:
                    self.counts["response"] += 1
                    self.cache.update(response_input=before, response_output_before_reclip=result.clone())
                return result
            return call
        self.wrap(env.actuator_response, "apply_torque", response)

        def write_effort(original):
            def call(*args, **kwargs):
                target = kwargs.get("target", args[0] if args else None)
                if self.in_policy:
                    require(target is not None, "Native effort write has no target")
                    self.cache["submitted_active_torque"] = target[:, env.ids].clone()
                    self.counts["effort_write"] += 1
                return original(*args, **kwargs)
            return call
        self.wrap(env.robot, "set_joint_effort_target_index", write_effort)

        def step(original):
            def call(*args, **kwargs):
                result = original(*args, **kwargs)
                if self.in_policy:
                    self.executed_physics_steps += 1
                    require(not self.pending_physics, "Native physics step missing feedback update")
                    self.pending_physics = True
                return result
            return call
        self.wrap(env.sim, "step", step)

        def update(original):
            def call(*args, **kwargs):
                result = original(*args, **kwargs)
                if self.in_policy and self.pending_physics:
                    self.pending_physics = False
                    self.physics_tick += 1
                    self.counts["physics"] += 1
                    self.observe()
                return result
            return call
        self.wrap(env.robot, "update", update)
        if self.delay is not None:
            self.install_delay()

    def delay_state_rows(self):
        """Read stored state only; the native random jitter draw is not replayed."""
        delay = self.delay
        values = json_value({"enabled": delay.enabled.clone(), "profile": delay.profile.clone(),
            "base_ms": delay.base_ms.clone(), "bounds_ms": delay.bounds_ms.clone(),
            "requested_lag_steps": delay.lags.clone()})
        rows = []
        for lane in range(len(self.env.scene_groups)):
            row = {key: value[lane] for key, value in values.items()}
            row.update(strength=float(delay.strength), raw_jitter_ms=None,
                raw_jitter_coverage="not_covered_native_draw_is_not_retained",
                jitter_metric_scope="stored quantized lag minus scaled base; includes clipping/rounding/enabled masking")
            row["requested_lag_ms"] = [lag*delay.dt_ms for lag in row["requested_lag_steps"]]
            row["scaled_enabled_base_ms"] = [base*delay.strength if row["enabled"] else 0. for base in row["base_ms"]]
            row["effective_quantized_lag_delta_from_scaled_base_ms"] = [
                lag*delay.dt_ms - base for lag,base in zip(row["requested_lag_steps"],row["scaled_enabled_base_ms"],strict=True)]
            rows.append(row)
        return rows

    def journal_delay(self, event, rows):
        for lane, row in enumerate(rows):
            row.update(event=event, lane=lane, case=self.env.scene_groups[lane],
                policy_tick=self.policy_tick, physics_tick=self.physics_tick,
                first_cohort=self.ledger.live[lane], native_call_once=True)
            self.delay_stream.write(json.dumps(row, allow_nan=False)+"\n")
        return rows

    def delay_provenance(self, buffer, columns, append_tick):
        """Native read_groups clips stored lag to existing history; no second read."""
        pushes = json_value(buffer._num_pushes.clone())
        rows = self.delay_state_rows()
        for lane,row in enumerate(rows):
            valid = [max(0,min(row["requested_lag_steps"][column],pushes[lane]-1)) for column in columns]
            row.update(append_physics_tick=append_tick, history_samples=pushes[lane],
                actual_age_steps=valid, actual_age_ms=[lag*self.delay.dt_ms for lag in valid],
                source_physics_tick=[append_tick-lag for lag in valid],
                age_scope="native stored lag clipped by actual append history; first append primes cleared history")
        return rows

    def install_delay(self):
        delay = self.delay
        def begin(original):
            def call(actor_update):
                result = original(actor_update)
                if self.in_policy:
                    require(actor_update == 4185, "Native delay must read authenticated fixed actor clock 4185")
                    self.counts["delay_begin_policy"] += 1
                    self.delay_totals["begin_policy"] += 1
                    rows = self.delay_state_rows()
                    for row in rows:
                        row["actor_update"] = actor_update
                    self.journal_delay("policy_lag_sample",rows)
                return result
            return call
        def targets(original):
            def call(legs,wheels):
                result = original(legs,wheels)
                if self.in_policy:
                    self.counts["delay_targets"] += 1
                    self.delay_totals["targets"] += 1
                    rows = self.delay_provenance(delay.target_buffer,(4,5),self.physics_tick)
                    inputs = json_value([legs.clone(),wheels.clone()])
                    outputs = json_value([result[0].clone(),result[1].clone()])
                    for lane,row in enumerate(rows):
                        row.update(applied_physics_tick=self.physics_tick+1,
                            input_targets6=inputs[0][lane]+inputs[1][lane], delayed_targets6=outputs[0][lane]+outputs[1][lane])
                    self.target_delay_rows = self.journal_delay("target_applied",rows)
                return result
            return call
        def observe(original):
            def call(sensors):
                result = original(sensors)
                if self.in_policy:
                    self.counts["delay_observe"] += 1
                    self.delay_totals["observe"] += 1
                    rows = self.delay_provenance(delay.sensor_buffer,(0,1,2,3),self.physics_tick)
                    require(delay.physics_steps == self.physics_tick, "Native delay sensor append clock differs from refreshed physics")
                    inputs,feedback = json_value(sensors.clone()),json_value(delay.feedback.clone())
                    for lane,row in enumerate(rows):
                        row.update(input_sensors18=inputs[lane],feedback18=feedback[lane])
                    self.sensor_delay_rows = self.journal_delay("observation_refresh",rows)
                return result
            return call
        self.wrap(delay,"begin_policy",begin)
        self.wrap(delay,"targets",targets)
        self.wrap(delay,"observe",observe)

    def actor_delay_snapshot(self):
        if self.delay is None:
            return None
        from copy import deepcopy
        rows = deepcopy(self.sensor_delay_rows)
        feedback = json_value(self.delay.feedback.clone())
        for lane,row in enumerate(rows):
            row.update(consumed_at_physics_tick=self.physics_tick, feedback18=feedback[lane],
                actor_observation_columns_affected=[4,22], pd_feedback_scope="fresh_physics_feedback")
        return rows

    def begin(self, tick, phase_by_lane):
        env = self.env
        positions = (env.robot.data.root_link_pose_w.torch[:, :2] - env.origins[:, :2]).cpu().tolist()
        for lane, phase in enumerate(phase_by_lane):
            if phase != self.phase.get(lane):
                self.anchors[lane] = positions[lane]
                self.phase[lane] = phase
        self.policy_tick, self.in_policy = tick, True
        self.cache = {"command": env.commands.clone()}
        self.counts.clear()

    def observe(self):
        from wheeled_tasks.chassis.playback_command import attitude_from_gravity
        env, data = self.env, self.env.robot.data
        values = json_value({"q6": data.joint_pos.torch[:, env.ids].clone(),
            "dq6": data.joint_vel.torch[:, env.ids].clone(), "root_pose_world_xyzw": data.root_link_pose_w.torch.clone(),
            "root_position_local_m": data.root_link_pose_w.torch[:, :3] - env.origins,
            "root_link_velocity_world_m_s": data.root_link_lin_vel_w.torch.clone(),
            "root_link_velocity_body_m_s": data.root_link_lin_vel_b.torch.clone(),
            "root_com_velocity_world_m_s": data.root_com_lin_vel_w.torch.clone(),
            "root_com_velocity_body_m_s": data.root_com_lin_vel_b.torch.clone(),
            "root_angular_velocity_body_rad_s": data.root_com_ang_vel_b.torch.clone(),
            "projected_gravity_b": data.projected_gravity_b.torch.clone(),
            "applied_torque6": data.applied_torque.torch[:, env.ids].clone(), **self.cache})
        for lane, live in enumerate(self.ledger.live):
            row = {key: value[lane] for key, value in values.items()}
            roll, pitch = attitude_from_gravity(row["projected_gravity_b"])
            row.update(lane=lane, case=env.scene_groups[lane], first_cohort=live,
                after_first_terminal=not live,
                physics_tick=self.physics_tick, policy_tick=self.policy_tick, time_s=self.physics_tick * PHYSICS_DT,
                phase=self.phase[lane], phase_anchor_xy_m=self.anchors[lane],
                roll_deg=math.degrees(roll), pitch_deg=math.degrees(pitch))
            # The flat fixture has ground z=0; preserve the native relative height.
            row["height_m"] = row["root_position_local_m"][2]
            if self.delay is not None:
                row["signal_delay"] = {"target": self.target_delay_rows[lane],
                    "observation_feedback_before_native_refresh": self.sensor_delay_rows[lane],
                    "native_observation_refresh_follows_this_row": True}
            if live:
                self.metrics.observe(row)
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")

    def end(self, done, extras):
        self.in_policy = False
        require(not self.pending_physics, "Last physics tick lacks refreshed feedback")
        require(self.counts["decode"] == 1 and self.counts["motor_efforts"] == 4
                and self.counts["effort_write"] == 4 and self.counts["physics"] == 4,
                "Native decode/PD/write/physics clocks differ")
        require(self.counts["response"] == (4 if self.env.actuator_response.enabled else 0),
                "Native actuator response clock differs")
        if self.delay is not None:
            require((self.counts["delay_begin_policy"], self.counts["delay_targets"], self.counts["delay_observe"]) == (1,4,4),
                    "Native signal delay policy/physical call clocks differ")
        diagnostic = extras["diagnostics"]
        self.ledger.observe(done=done.bool(), terminated=diagnostic["terminated"].bool(),
            timeouts=extras["time_outs"].bool(), success=diagnostic["success"].bool(),
            reasons=diagnostic["reasons"], physical_tick=self.physics_tick)
        self.completed_policy_steps += 1

    def close(self):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)


def apply_command(env, previous, requested):
    from wheeled_tasks.chassis.playback_command import advance_playback_command
    command = advance_playback_command(previous, requested, POLICY_DT, env.cfg["command_slew"], env.cfg.get("motion_limits"))
    env.commands.copy_(command)
    env.command_target.copy_(command[:, :2])
    env.mode.copy_((command[:, :2].abs().amax(-1) > .01).long())
    # The command was updated before inference at the same native observation tick.
    env.history.last_tick = None
    env.update_targets()
    return command


def save_actual_execution(report, output, recorder, ledger, actor, before_actor, sampler):
    """Retain actual partial exposure and actor identity on every exit path."""
    if recorder is not None:
        report.update(policy_steps=recorder.completed_policy_steps, policy_steps_started=recorder.policy_tick,
            physics_steps=recorder.physics_tick, observed_feedback_steps=recorder.physics_tick,
            physics_executed_steps=recorder.executed_physics_steps,
            pending_physics_step_without_feedback=recorder.pending_physics,
            partial_policy_physics_steps=recorder.physics_tick % 4,
            signal_delay_native_method_counts=dict(recorder.delay_totals))
    if ledger is not None and recorder is not None:
        try:
            phases = recorder.metrics.finish(ledger)
            cohort = ledger.summary(physical_tick=recorder.physics_tick)
            write_json(Path(output) / "first_cohort.json", json_value({**cohort, "lane_phase_metrics": phases}))
            report["first_cohort"] = cohort
            report["first_cohort_accounting_saved"] = True
        except Exception:
            report.update(status="failed_first_cohort_accounting", execution_status="failed",
                first_cohort_accounting_error=traceback.format_exc())
    if actor is not None and before_actor is not None:
        try:
            import torch
            after_actor = actor.state_dict()
            unchanged = set(after_actor) == set(before_actor) and all(
                value.dtype == before_actor[key].dtype and torch.equal(value.detach().cpu(), before_actor[key])
                for key, value in after_actor.items())
            report["actor_tensors_unchanged"] = unchanged
            require(unchanged, "Inference altered the saved actor tensors")
        except Exception:
            report.update(status="failed_actor_identity", execution_status="failed", actor_identity_error=traceback.format_exc())
    if sampler is not None:
        report.update(draw_counter=sampler.draw_counter, sample_element_count=sampler.sample_element_count)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("mean", "stochastic"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--fixture-role", choices=("clean", "pipeline_age_only"), default="clean")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    require(args.device == "cuda:0", "This bounded physical probe requires cuda:0")
    require(not args.output.resolve().is_relative_to(ROOT.resolve()), "Outputs must be outside the frozen source checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "native-locomotion-exploration-probe-v1", "status": "starting", "arm": args.arm,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "formal_updates_consumed": 0, "fixture_role": args.fixture_role,
        "qualification": None, "requested_policy_steps": POLICY_STEPS, "requested_physics_steps": 4000,
        "requested_seconds": 20., "source_revision": args.source_revision,
        "scope": "same sealed native actor; inference only; fixed N4 first cohorts; no qualification"}
    path = args.output / "report.json"
    env = launcher = recorder = inputs = actor = source = None
    ledger = sampler = before_actor = None
    write_json(path, report)
    try:
        source = authenticate_source(ROOT, args.source_revision)
        from locomotion_exploration_fixture import authenticate_inputs, build_fixture, assert_fixture_runtime
        from locomotion_exploration_sampling import load_native_actor, NativeActionSampler, FirstCohortLedger
        inputs = authenticate_inputs(ROOT, args.input_dir)
        fixture = build_fixture(ROOT, inputs["source"], args.output, args.fixture_role)
        require(fixture["fixture_role"] == args.fixture_role, "Independent fixture role differs from requested role")
        require(fixture["task_actor_update"] == (4185 if args.fixture_role == "pipeline_age_only" else 0),
                "Diagnostic actor clock differs from frozen role")
        if args.fixture_role == "pipeline_age_only":
            require(inputs["receipt"]["clocks"]["task_actor_update"] == fixture["task_actor_update"],
                    "Pipeline age actor clock must come from authenticated source seal")
        report.update(source_authentication=source, input_authentication=inputs["receipt"], fixture=fixture["receipt"],
                      design_preflight=fixture["design_preflight"], expected_task_actor_update=fixture["task_actor_update"])
        write_json(args.output / "preflight.json", json_value(report))
        if args.preflight_only:
            report["status"] = "preflight_passed_scene_not_run"
        else:
            os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
            from isaaclab.app import AppLauncher
            launcher = AppLauncher({"headless": True, "enable_cameras": False, "device": args.device,
                "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
            import torch
            from wheeled_tasks.chassis.eval_env import FixedCaseEnv
            torch.set_num_threads(4)
            torch.manual_seed(fixture["seed"])
            require(authenticate_source(ROOT, args.source_revision) == source, "Source changed before scene construction")
            env = FixedCaseEnv(fixture["config"], fixture["manifest"], fixture["control"], ROOT,
                stage_name=fixture["stage"], run_role=fixture["role"], seed=fixture["seed"],
                num_envs=fixture["num_envs"], device=args.device, level=1.)
            require(env.cfg.get("auto_reset") is False, "Native fixed no-autoreset fixture required")
            env.task_actor_update = fixture["task_actor_update"]
            from wheeled_tasks.chassis.training_schedule import stage_actor_updates
            require(stage_actor_updates(env) == fixture["task_actor_update"], "Native task clock differs from inference context")
            observations = env.reset_suite()
            runtime = assert_fixture_runtime(env, fixture)
            write_json(args.output / "runtime.json", json_value(runtime))
            write_json(args.output / "birth.json", json_value({"scene_groups": env.scene_groups,
                "root_pose_world_xyzw": env.robot.data.root_link_pose_w.torch.clone(),
                "joint_position": env.robot.data.joint_pos.torch.clone(),
                "joint_velocity": env.robot.data.joint_vel.torch.clone(), "commands": env.commands.clone(),
                "policy_observations": observations["policy"].clone(), "episode_limits": env.episode_limits.clone(),
                "control_order": env.v5.ACTIVE, "policy_order": inputs["source"]["policy_action_order"],
                "v5_nominal6": env.v5.nominal.clone(), "fixture_role": args.fixture_role,
                "task_actor_update": stage_actor_updates(env)}))
            actor = load_native_actor({"actor_state_dict": inputs["actor_state_dict"]}, inputs["agent_config"], inputs["source"], args.device)
            before_actor = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
            sampler = NativeActionSampler(actor, SAMPLING_SEED, args.device)
            ledger = FirstCohortLedger(env.scene_groups)
            plans = {name: schedule(speed) for name, speed in fixture["case_profiles"].items()}
            previous = env.commands.clone()
            started = time.monotonic()
            report.update(status="running", runtime=runtime, native_policy_action_order=inputs["source"]["policy_action_order"],
                          native_control_order=list(env.v5.ACTIVE), sampling_seed=SAMPLING_SEED,
                          task_actor_update=stage_actor_updates(env),
                          actor_clock_scope="fixed sealed inference context; no PPO or clock advancement",
                          raw_signal_jitter_coverage="not_covered_native_draw_is_not_retained")
            write_json(path, json_value(report))
            with ExitStack() as streams:
                actions_file = streams.enter_context((args.output / "actions.jsonl").open("w"))
                physics_file = streams.enter_context((args.output / "physical_frames.jsonl").open("w"))
                delay_file = (streams.enter_context((args.output / "signal_delay_frames.jsonl").open("w"))
                              if env.signal_delay is not None else None)
                recorder = PhysicalRecorder(env, ledger, fixture["case_profiles"], physics_file, delay_file)
                for tick in range(1, POLICY_STEPS + 1):
                    require(launcher.app.is_running(), "Simulator stopped before bounded run completion")
                    phases = [plans[name][tick - 1][0] for name in env.scene_groups]
                    requested = previous.new_tensor([[plans[name][tick - 1][1], 0., .305] for name in env.scene_groups])
                    previous = apply_command(env, previous, requested)
                    observations = env.get_observations()
                    actor_sensor_delay = recorder.actor_delay_snapshot()
                    draw = sampler.draw(observations)
                    action = draw["mu"] if args.arm == "mean" else draw["sample"]
                    actions_file.write(json.dumps(json_value({"policy_tick": tick, "time_s_before": (tick - 1) * POLICY_DT,
                        "arm": args.arm, "first_cohort_active": ledger.live, "phase": phases,
                        "requested_command": requested, "visible_command": previous, "policy_input": observations["policy"].clone(),
                        "actor_sensor_delay": actor_sensor_delay,
                        **draw, "sent_policy_action": action.clone()}), allow_nan=False) + "\n")
                    recorder.begin(tick, phases)
                    with torch.inference_mode():
                        observations, reward, done, extras = env.step(action)
                    recorder.end(done, extras)
                    require(stage_actor_updates(env) == fixture["task_actor_update"], "Inference advanced the frozen actor clock")
                    timing = env.control_audit.last
                    require(timing["pd_updates"] == timing["feedback_refreshes"] == 4 and timing["target_decodes"] == 1
                            and timing["physics_steps"] == recorder.physics_tick == tick * 4,
                            "Native 50/200 Hz timing changed")
                    if tick % 250 == 0:
                        report.update(policy_steps=tick, physics_steps=recorder.physics_tick, wall_seconds=time.monotonic() - started)
                        write_json(path, json_value(report))
                        print(json.dumps({"arm": args.arm, "policy_steps": tick, "first_cohort_live": sum(ledger.live)}), flush=True)
            report.update(status="native_probe_completed", execution_status="completed", policy_steps=POLICY_STEPS,
                physics_steps=recorder.physics_tick, wall_seconds=time.monotonic() - started,
                draw_counter=sampler.draw_counter,
                sample_element_count=sampler.sample_element_count, control_timing=env.control_audit.last)
    except Exception:
        report.update(status="failed", execution_status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if recorder is not None:
            recorder.close()
        save_actual_execution(report, args.output, recorder, ledger, actor, before_actor, sampler)
        if source is not None:
            try:
                require(authenticate_source(ROOT, args.source_revision) == source, "Source identity changed during probe")
                report["source_identity_final_verified"] = True
            except Exception:
                report.update(status="failed_source_identity", execution_status="failed", source_identity_final_verified=False,
                              source_identity_error=traceback.format_exc())
        if inputs is not None:
            try:
                final_inputs = authenticate_inputs(ROOT, args.input_dir)
                require(final_inputs["receipt"] == inputs["receipt"], "Sealed input identity changed during probe")
                report["sealed_inputs_final_verified"] = True
            except Exception:
                report.update(status="failed_input_identity", execution_status="failed", sealed_inputs_final_verified=False,
                              input_identity_error=traceback.format_exc())
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, json_value(report))
        for resource, close in ((env, lambda: env.close()), (launcher, lambda: launcher.app.close())):
            if resource is not None:
                try:
                    close()
                except Exception:
                    report.update(status="failed_cleanup", execution_status="failed", cleanup_error=traceback.format_exc())
        write_json(path, json_value(report))
    print(json.dumps({"arm": args.arm, "status": report["status"], "report": str(path.resolve())}), flush=True)
    return 0 if report["status"] in ("preflight_passed_scene_not_run", "native_probe_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
