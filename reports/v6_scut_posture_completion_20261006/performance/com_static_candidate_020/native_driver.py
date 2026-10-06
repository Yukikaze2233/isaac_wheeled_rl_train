#!/usr/bin/env python3
"""Isolated optional native engineering; never invoked by the CPU authoring task.

semantic: compare cache/vendor each getter and IMU, deliberately expensive.
throughput: no vendor comparison; unchanged native probe, separate ABBA arms.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import weakref


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)



SEMANTIC_PATHS = ["final_native_com_snapshot", "world_getter", "body_getter", "native_data_updates",
                  "same_epoch_write_cached_world", "same_epoch_write_cached_body", "partial_reset_calls",
                  "imu_specific_force", "imu_rng_state", "native_push_calls", "dr_enabled_environments"]


def coverage_status(report, mode, variant, counts):
    """Pure terminal acceptance; process exit code is deliberately irrelevant."""
    checks = {"terminal_engineering_status": report.get("status") == "engineering_probe_completed",
        "4096_environments": report.get("num_envs") == 4096,
        "600_policy_steps": report.get("policy_steps") == 600,
        "2400_physics_steps": report.get("physics_steps") == 2400,
        "actor_unchanged": report.get("actor_tensors_unchanged") is True,
        "checkpoint_unchanged": report.get("checkpoint_unchanged") is True,
        "source_unchanged": report.get("source_files_unchanged") is True,
        "contract_unchanged": report.get("target_contract_unchanged") is True,
        "zero_PPO": report.get("formal_updates_consumed") == report.get("optimizer_steps") == 0,
        "unqualified": report.get("qualification") is None}
    if variant == "candidate":
        checks["adapter_in_source_snapshot"] = ("src/wheeled_tasks/chassis/root_kinematics.py"
            in report.get("source_files_sha256", {}))
    if mode == "semantic":
        checks.update({"observed_"+name: counts.get(name, 0) > 0 for name in SEMANTIC_PATHS})
        checks["at_least_2400_native_updates"] = counts.get("native_data_updates", 0) >= 2400
    return all(checks.values()), [name for name, passed in checks.items() if not passed]


def write_postclose_receipt(output, code, driver_error):
    """Keep the before-close evidence intact if Kit returns to Python."""
    output = Path(output)
    if not output.is_dir():
        return
    preclose = output/'root_com_candidate_receipt.json'
    evidence = dict(schema='com-static-postclose-v1', native_probe_exit_code=code,
        driver_error=driver_error, preclose_receipt_present=preclose.is_file(),
        preclose_receipt_sha256=digest(preclose) if preclose.is_file() else None,
        preclose_receipt_preserved=True, acceptance_must_use_preclose_receipt=True)
    temporary = output/'root_com_candidate_postclose_receipt.json.tmp'
    temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    temporary.replace(output/'root_com_candidate_postclose_receipt.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['semantic', 'throughput'], required=True)
    parser.add_argument('--variant', choices=['baseline', 'candidate'], required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('patch_manifest.json'))
    args, probe_args = parser.parse_known_args()
    root = args.source_root.resolve()
    output_index = probe_args.index('--output') + 1
    output = Path(probe_args[output_index]).resolve()
    steps = int(probe_args[probe_args.index('--policy-steps')+1]) if '--policy-steps' in probe_args else 240
    require(steps == 600, 'This comparison protocol requires 4096 x 600 policy steps')
    require('--preflight-only' not in probe_args, 'Native driver cannot treat CPU preflight as native evidence')
    require(args.mode != 'semantic' or args.variant == 'candidate', 'Semantic mode requires candidate cache')
    manifest = json.loads(args.manifest.read_text())
    expected_files = manifest['core_files'] | manifest['probe_files']
    for name, hashes in expected_files.items():
        expected = hashes[args.variant+'_sha256'] if args.variant == 'candidate' else hashes['base_sha256']
        if expected is None:
            require(not (root/name).exists(), 'Baseline unexpectedly contains the candidate adapter')
        else:
            require(digest(root/name) == expected, 'Source bytes differ from comparison manifest: '+name)
    for name, expected in manifest.get('unchanged_source_files', {}).items():
        require(digest(root/name) == expected, 'Unrelated code/contract changed: '+name)
    sys.path[:0] = [str(root/'src'), str(root/'scripts')]
    import probe_scut_curriculum as probe
    require(probe.ROOT.resolve() == root, 'Probe imports came from a different source root')
    counts = Counter()
    step_seconds = []
    adapters = []
    original_structural_init = probe.StructuralAudit.__init__
    original_imu_sample = None
    started = time.monotonic()
    code = None
    driver_error = None

    def write_sidecar(source_receipt, *, before_close):
        complete, missing = coverage_status(source_receipt, args.mode, args.variant, dict(counts))
        warmed = step_seconds[50:]
        evidence = dict(schema='com-static-native-candidate-v1', mode=args.mode, variant=args.variant,
            native_probe_exit_code=code, native_probe_status=source_receipt.get('status'),
            mode_complete=complete, missing_acceptance=missing, driver_error=driver_error,
            written_before_app_close=before_close, source_root=str(root), driver_sha256=digest(__file__),
            patch_manifest_sha256=digest(args.manifest), source_base_commit=manifest['base_commit'],
            source_files_sha256=source_receipt.get('source_files_sha256'),
            root_kinematics_source_snapshot_covered=(
                'src/wheeled_tasks/chassis/root_kinematics.py' in source_receipt.get('source_files_sha256', {})),
            semantic_counts=dict(counts), separate_process_wall_seconds=time.monotonic()-started,
            existing_probe_loop_wall_seconds=source_receipt.get('wall_seconds'),
            policy_steps=source_receipt.get('policy_steps'), physics_steps=source_receipt.get('physics_steps'),
            measured_env_steps=len(step_seconds), warmup_steps_excluded=50,
            warmed_env_step_median_s=statistics.median(warmed) if warmed else None,
            warmed_env_step_p95_s=sorted(warmed)[int((len(warmed)-1)*.95)] if warmed else None,
            throughput_contains_vendor_comparison=args.mode == 'semantic',
            formal_updates_consumed=0, optimizer_steps=0, qualification=None, ppo_speedup_claim=None)
        if output.is_dir():
            temporary = output/'root_com_candidate_receipt.json.tmp'
            temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
            temporary.replace(output/'root_com_candidate_receipt.json')

    native_write_report = probe.write_report
    def write_report_before_close(path, report):
        result = native_write_report(path, report)
        if Path(path).resolve() == output/'report.json':
            # The terminal write is in probe.finally before env.close/app.close.
            # Pending/failed receipts are never accepted even if Kit exits 0.
            write_sidecar(report, before_close=True)
        return result
    probe.write_report = write_report_before_close

    if args.mode == 'semantic':
        import torch
        import warp as wp
        from wheeled_tasks.chassis import root_kinematics as kin
        from wheeled_tasks.chassis.recovery_observer import SimulatedImu
        native_kinematics_init = kin.StaticRootLinkKinematics.__init__
        imu_shadows = weakref.WeakKeyDictionary()

        def exact(actual, expected, label):
            require(actual.shape == expected.shape and torch.equal(actual, expected),
                'Non-equivalent native '+label+': '+str(float((actual-expected).abs().max())))
            counts[label] += 1

        def instrument_cache(cache, robot, count, device, **kwargs):
            native_kinematics_init(cache, robot, count, device, **kwargs)
            adapters.append(cache)
            native = wp.to_torch(robot.root_view.get_coms())[:, :1].to(cache.local_com.device)
            exact(cache.local_com, native, 'final_native_com_snapshot')
            original_world, original_body = cache.linear_velocity_world, cache.linear_velocity_body
            original_update = cache.update_robot_data

            def world():
                actual = original_world()
                exact(actual, robot.data.root_link_lin_vel_w.torch, 'world_getter')
                return actual

            def body():
                actual = original_body()
                exact(actual, robot.data.root_link_lin_vel_b.torch, 'body_getter')
                return actual

            def update(dt):
                before = cache.read_epoch
                original_update(dt)
                require(cache.read_epoch == before + (dt > 0), 'Native data/read epoch advancement differs')
                counts['native_data_updates'] += 1

            cache.linear_velocity_world, cache.linear_velocity_body, cache.update_robot_data = world, body, update
            native_write = robot.write_root_com_velocity_to_sim_index

            def write_velocity(*positional, **keywords):
                world_valid = cache.world_epoch == cache.read_epoch
                body_valid = cache.body_epoch == cache.read_epoch
                old_world = original_world().clone() if world_valid else None
                old_body = original_body().clone() if body_valid else None
                result = native_write(*positional, **keywords)
                if world_valid:
                    exact(world(), old_world, 'same_epoch_write_cached_world')
                if body_valid:
                    exact(body(), old_body, 'same_epoch_write_cached_body')
                counts['native_com_velocity_writes'] += 1
                return result

            robot.write_root_com_velocity_to_sim_index = write_velocity
            native_reset = robot.reset

            def reset(*positional, **keywords):
                ids = keywords.get('env_ids', positional[0] if positional else None)
                if ids is not None and len(ids) < count:
                    counts['partial_reset_calls'] += 1
                counts['native_reset_calls'] += 1
                return native_reset(*positional, **keywords)

            robot.reset = reset

        kin.StaticRootLinkKinematics.__init__ = instrument_cache
        original_imu_sample = SimulatedImu.sample

        def sample_imu(imu, velocity_world, orientation):
            require(adapters, 'IMU sample predates final native COM snapshot')
            robot = adapters[-1].robot
            native_velocity = robot.data.root_link_lin_vel_w.torch
            exact(velocity_world, native_velocity, 'imu_input_link_velocity')
            shadow = imu_shadows.get(imu)
            if shadow is None:
                shadow = SimulatedImu(velocity_world.device, len(velocity_world), imu.dt, noise=imu.noise)
                imu_shadows[imu] = shadow
            shadow.previous_velocity = imu.previous_velocity.clone()
            shadow.generator.set_state(imu.generator.get_state())
            expected = original_imu_sample(shadow, native_velocity, orientation)
            actual = original_imu_sample(imu, velocity_world, orientation)
            exact(actual, expected, 'imu_specific_force')
            exact(imu.previous_velocity, shadow.previous_velocity, 'imu_previous_velocity')
            exact(imu.generator.get_state(), shadow.generator.get_state(), 'imu_rng_state')
            return actual

        SimulatedImu.sample = sample_imu

    def structural_init(audit, env, domain):
        original_structural_init(audit, env, domain)
        if args.mode == 'semantic':
            require(env.dynamics_randomization is not None, 'Native test did not exercise startup dynamics DR')
            counts['dr_enabled_environments'] += env.num_envs
            native_push = env.apply_push
            def push(ids):
                before = counts['native_com_velocity_writes']
                result = native_push(ids)
                require(counts['native_com_velocity_writes'] > before, 'Native push did not write COM velocity')
                counts['native_push_calls'] += 1
                return result
            env.apply_push = push
        else:
            # No vendor comparison, native COM readback, extra synchronization or
            # stream/event instrumentation in the throughput arm.
            native_step = env.step
            def step(action):
                started = time.perf_counter()
                result = native_step(action)
                step_seconds.append(time.perf_counter()-started)
                return result
            env.step = step

    probe.StructuralAudit.__init__ = structural_init
    old_argv = sys.argv
    sys.argv = [str(root/'scripts/probe_scut_curriculum.py'), *probe_args]
    try:
        code = probe.main()
        return code
    except Exception as error:
        code = 1
        driver_error = str(error)
        raise
    finally:
        sys.argv = old_argv
        # SimulationApp.close may exit the process directly. If it instead
        # returns, record that fact separately; never replace pre-close proof.
        write_postclose_receipt(output, code, driver_error)


if __name__ == '__main__':
    raise SystemExit(main())
