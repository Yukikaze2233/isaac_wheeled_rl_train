"""Bounded native geometry readback from a clean frozen source; no policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    from train_chassis import preflight, scut_curriculum_source_files
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.terrain_runout_revision import WORKER_FILE, SOURCE_PLAN_FILE

    report = dict(status="starting", scope="native_geometry_and_one_physics_interval_no_policy_or_PPO",
                  source_directory=str(root), source_commit="b0d2800a338fa11099b355ece48c0baaccbb1b02",
                  PPO_updates=0, optimizer_steps=0, qualification=None,
                  requested_policy_intervals=1, requested_physics_steps_per_env=4,
                  contract_file=WORKER_FILE, contract_sha256=digest(root / WORKER_FILE))
    path = args.output / "report.json"
    launcher = env = None
    before = {}
    started = time.monotonic()
    try:
        config, manifest = preflight(root / WORKER_FILE)
        control = load_chassis_control(root / config["control_math_source"])
        sources = scut_curriculum_source_files(config, root=root)
        before = {name: digest(root / name) for name in sources}
        report["source_files_sha256"] = before
        plan = json.loads((root / SOURCE_PLAN_FILE).read_text())
        seed, count = plan["seed"], plan["target_num_envs"]
        assert (seed, count) == (290927, 4096)
        report["design_preflight"] = design_preflight(config, control, root,
            stage="terrain", role="train", seed=seed, num_envs=count, block_index=0)
        report["num_envs"] = count
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, enable_cameras=False, device="cuda:0",
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        from wheeled_tasks.chassis.env import ChassisEnv
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        env = ChassisEnv(config, manifest, control, root, stage_name="terrain",
                         num_envs=count, device="cuda:0", level=1., seed=seed,
                         run_role="train", block_index=0)
        report["startup"] = env.startup_report
        assert env.startup_report["gpu_dynamics_enabled"] is True
        assert env.startup_report["runtime_preflight"]["status"] == "runtime_configuration_passed"
        assert env.startup_report["terrain_route_runout"]["selected_environment_count"] == 1724
        observed = env.startup_report["terrain_route_runout"]["runtime_observed"]
        assert observed["collision_count"] == 18370
        report["construction_seconds"] = time.monotonic() - started
        tick = time.monotonic()
        with torch.inference_mode():
            observations, reward, done, extras = env.step(torch.zeros(count, 6, device=env.device))
        torch.cuda.synchronize()
        assert observations["policy"].shape == (count, 35)
        assert observations["critic"].shape == (count, 81)
        assert torch.isfinite(reward).all()
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(observations["critic"]).all()
        assert all(digest(root / name) == sha for name, sha in before.items())
        report.update(status="native_readback_and_one_interval_passed",
                      physics_interval_wall_seconds=time.monotonic() - tick,
                      actual_policy_intervals=1, actual_physics_steps_per_env=env.decimation,
                      observation_shapes={name: list(value.shape) for name, value in observations.items()},
                      physical_summary=env.summary(), boundary_episode_outcomes_verified=False,
                      real_learning_restore_verified=False, production_changes=0)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds_before_close"] = time.monotonic() - started
        report["sources_unchanged"] = bool(before) and all(digest(root / name) == sha for name, sha in before.items())
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] == "native_readback_and_one_interval_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
