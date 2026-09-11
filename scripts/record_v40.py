#!/usr/bin/env python3
"""Bounded V40 mean-policy MP4 replay; default is stdlib-only preflight, never Sim."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

POLICY_DT = .01
FPS = 25
FRAME_STRIDE = 4
RESOLUTION = (960, 720)
EYE = (1.0, -1.4, .8)
LOOKAT = (0., 0., .3)
SOURCE_LABELS = {
    "source-final": "Original source-final replay: old hip geometry and physical wheel limits +/-pi",
    "fixed-repo": "Fixed-repository physics replay; NOT footage of the original training run",
}


def load_train(repo_root):
    """Resolve the helper AND task packages from one tree, including copied server trees."""
    sys.dont_write_bytecode = True  # Keep selected immutable source trees read-only.
    src = repo_root / "src"
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("wheeled_tasks", "wheeled_algo", "wheeled_world")):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(src):
                raise RuntimeError(f"mixed repository imports: {name}; use a fresh Python process")
    sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("_v40_record_train", repo_root / "scripts/train_v40.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.REPO_ROOT.resolve() != repo_root:
        raise RuntimeError("selected train_v40 resolved a different repository")
    return module


def parse_arguments(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    selected, _ = bootstrap.parse_known_args(argv)
    train = load_train(selected.repo_root.resolve())
    parser = argparse.ArgumentParser(description=__doc__, parents=[bootstrap])
    train.add_common_arguments(parser, num_envs=1)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint beside its original run_manifest.json")
    parser.add_argument("--output-dir", type=Path, help="New directory only")
    parser.add_argument("--source-case", choices=SOURCE_LABELS,
                        help="Required on apply: operator-declared physics variant, recorded with source hashes")
    parser.add_argument("--duration-s", type=float, default=10., help="(0,10] seconds, exact .01s ticks")
    parser.add_argument("--command", nargs=3, type=float, default=(0., 0., .32),
                        metavar=("VX_M_S", "WZ_RAD_S", "HEIGHT_M"))
    args = parser.parse_args(argv)
    if (not math.isfinite(args.duration_s) or not 0 < args.duration_s <= 10
            or not math.isclose(args.duration_s / POLICY_DT, round(args.duration_s / POLICY_DT),
                                rel_tol=0., abs_tol=1e-7)):
        parser.error("duration-s must be (0,10] and an exact .01s multiple (<=1000 steps)")
    if args.num_envs != 1:
        parser.error("recording requires exactly one environment")
    if args.apply and not args.research:
        parser.error("simulation requires BOTH --apply and --research")
    if args.apply and not args.preflight_only and any(
        value is None for value in (args.checkpoint, args.output_dir, args.source_case)
    ):
        parser.error("apply requires --checkpoint, new --output-dir and --source-case")
    args.repo_root = args.repo_root.resolve()
    args.preflight_only = args.preflight_only or not args.apply
    return args, train


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity(root):
    # HEAD alone is insufficient: deployed V40 files can be untracked or locally fixed.
    files = sorted((root / "src").rglob("*.py")) + [root / "scripts/train_v40.py"]
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in files}
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=10)
    return {"repo_root": str(root), "repo_head": result.stdout.strip() if result.returncode == 0 else None,
            "source_files_sha256": hashes,
            "source_tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "record_entrypoint_sha256": sha256_file(Path(__file__).resolve())}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def video_backend():
    """Use only an existing imageio FFmpeg executable; never download or install."""
    import imageio.v2 as imageio
    import imageio_ffmpeg

    try:
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run([executable, "-version"], capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise RuntimeError(result.stderr)
    except Exception as exc:
        raise RuntimeError(f"existing FFmpeg unavailable; no installation attempted: {exc}") from exc

    def writer_factory(path):
        return imageio.get_writer(str(path), format="FFMPEG", mode="I", fps=FPS,
                                  codec="libx264", pixelformat="yuv420p", macro_block_size=1)

    return writer_factory, imageio.imwrite, {"executable": executable, "version": result.stdout.splitlines()[0]}


def launch_recording_app(args):
    os.environ.update(ENABLE_CAMERAS="1", LIVESTREAM="0")
    from isaaclab.app import AppLauncher
    config = {"headless": True, "enable_cameras": True, "livestream": 0, "device": args.device}
    if os.geteuid() == 0:
        config["kit_args"] = "--allow-root"
    return AppLauncher(config)


def make_recording_env(args):
    # Same fields as sibling train.make_env; only viewer/render configuration is added.
    from wheeled_tasks.direct.v40_serial.env_cfg import V40EnvCfg
    from wheeled_tasks.direct.v40_serial.env import V40Env
    cfg = V40EnvCfg()
    cfg.contract_path = str(args.contract.resolve()) if args.contract else None
    cfg.usd_cache_dir = str(args.usd_cache_dir.resolve()) if args.usd_cache_dir else None
    cfg.allow_research = args.research
    cfg.stage = args.stage
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.viewer.resolution = RESOLUTION
    cfg.viewer.eye = EYE
    cfg.viewer.lookat = LOOKAT
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.rerender_on_reset = False
    return V40Env(cfg=cfg, render_mode="rgb_array")


def warm_renderer(raw_env):
    import carb
    # AppLauncher enables offscreen rendering, but leaves rtx_sensors=False.
    # The viewport RGB product is our RTX sensor. Register it, then enable the
    # stock step() render at the final physics substep BEFORE done/auto-reset.
    raw_env.render()
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)
    if not raw_env.sim.has_rtx_sensors():
        raise RuntimeError("RTX sensor render scheduling was not enabled")
    for _ in range(20):
        raw_env.sim.render()  # Renderer warmup only; no physics ticks.
        frame = raw_env.render(recompute=False)
        if frame is not None and frame.size and frame.any():
            return
    raise RuntimeError("RGB renderer stayed empty/black after 20 warmup renders")


def require_finite(value, name):
    # TensorDict has detach() too: traverse its fields before converting tensors.
    if hasattr(value, "items"):
        for key, child in value.items():
            require_finite(child, f"{name}.{key}")
        return
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        for child in value:
            require_finite(child, name)
    elif isinstance(value, (float, int)) and not math.isfinite(value):
        raise ValueError(f"nonfinite {name}")


def collect_video(raw_env, wrapped_env, mean_policy, steps, is_running, writer, save_image, output_dir, summary):
    """O(one RGB frame) collector; writer is injected for offline lifecycle tests."""
    from wheeled_algo.v40_metrics import snapshot_to_record, validate_record
    if not 1 <= steps <= 1000:
        raise ValueError("recording step bound must be 1..1000")
    initial = snapshot_to_record(raw_env.capture_evaluation_initial_snapshot())
    validate_record(initial)
    obs = wrapped_env.get_observations()  # Lab 2.3 returns TensorDict, not (obs, extras).
    frame = None
    last_frame_tick = None

    def append_frame(tick):
        nonlocal frame, last_frame_tick
        # With RTX enabled and rerender_on_reset=False this reads PRE-reset RGB.
        frame = raw_env.render(recompute=False)
        if (frame is None or frame.shape != (RESOLUTION[1], RESOLUTION[0], 3)
                or str(frame.dtype) != "uint8"):
            raise RuntimeError("expected 720x960 uint8 RGB frame")
        writer.append_data(frame)
        summary["frames"] += 1
        last_frame_tick = tick
        summary["frame_policy_ticks"].append(tick)

    try:
        append_frame(0)
        for tick in range(1, steps + 1):
            if not is_running():
                raise RuntimeError("application stopped before recording horizon")
            require_finite(obs, "observation")
            action = mean_policy(obs["policy"])
            require_finite(action, "mean action")
            obs, reward, dones, _ = wrapped_env.step(action)
            summary["policy_steps"] = tick
            summary["sim_duration_s"] = tick * POLICY_DT
            record = snapshot_to_record(raw_env.get_evaluation_snapshot())
            # Save terminal RGB even off the 25Hz sample grid, before validating bad state.
            terminal = bool(dones[0])
            if tick % FRAME_STRIDE == 0 or terminal or tick == steps:
                append_frame(tick)
            summary["terminated"] = bool(record["terminated"])
            summary["timeout"] = bool(record["timeout"])
            summary["termination_reasons"] = [key for key, flag in record["termination_flags"].items() if flag]
            validate_record(record)
            require_finite(obs, "observation")
            require_finite(reward, "reward")
            if record["termination_flags"].get("nonfinite"):
                raise ValueError("nonfinite terminal state/action")
            if (record["policy_tick"] - initial["policy_tick"] != tick
                    or record["physics_steps"] - initial["physics_steps"] != tick * 2
                    or record["policy_dt_s"] != POLICY_DT or record["physics_dt_s"] != .005):
                raise ValueError("snapshot timing differs from fixed V40 contract")
            if terminal != (record["terminated"] or record["timeout"]):
                raise ValueError("wrapper done differs from pre-reset snapshot")
            if terminal:
                summary["stop_reason"] = "terminated" if record["terminated"] else "timeout"
                break
        else:
            summary["stop_reason"] = "duration_limit"
    finally:
        # On numeric failure between sample ticks, retain the actual final cached image too.
        if frame is not None:
            if last_frame_tick != summary["policy_steps"]:
                append_frame(summary["policy_steps"])
            save_image(output_dir / "final_frame.png", frame)


def record(args, train, contract, asset, report, identity):
    summary = {"schema_version": 1, **identity, "source_case": args.source_case,
               "display_label": SOURCE_LABELS[args.source_case],
               "physics_variant_evidence": "operator-declared label; exact loaded source/asset hashes recorded",
               "scope": "New deterministic checkpoint replay in selected repository; not training-time footage or policy acceptance",
               "stage": args.stage, "seed": args.seed, "command": list(args.command),
               "checkpoint_path": str(args.checkpoint.resolve()), "preflight": report,
               "requested_duration_s": args.duration_s, "policy_dt_s": POLICY_DT,
               "physics_dt_s": contract["timing"]["physics_dt"], "decimation": contract["timing"]["decimation"],
               "fps": FPS, "sample_every_policy_ticks": FRAME_STRIDE, "resolution": RESOLUTION,
               "eye": EYE, "lookat": LOOKAT, "policy": "checked mean actor; no exploration",
               "frames": 0, "frame_policy_ticks": [], "policy_steps": 0, "sim_duration_s": 0.,
               "terminated": False, "timeout": False, "termination_reasons": [],
               "stop_reason": "not_started", "recording_ok": False, "runtime_error": None,
               "application_cleanup": "not_observed; summary published before app.close()"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    app = raw_env = wrapped_env = writer = None
    try:
        # Heavy checkpoint/backend imports are APPLY-only; all mandatory gates already passed.
        manifest = train.make_manifest(contract, asset, args)
        actor, provenance = train.checked_checkpoint(args.checkpoint, manifest)
        from wheeled_algo.v40_metrics import require_trained_stage
        require_trained_stage(provenance["run_manifest"], args.stage)
        summary.update(provenance)
        summary.update({key: manifest[key] for key in train.METADATA_KEYS})
        writer_factory, save_image, summary["ffmpeg"] = video_backend()
        writer = writer_factory(args.output_dir / "replay.mp4")
        app = launch_recording_app(args).app
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        raw_env = make_recording_env(args)
        if (raw_env.contract_sha256 != manifest["contract_sha256"]
                or raw_env.asset_manifest_sha256 != manifest["asset_manifest_sha256"]):
            raise RuntimeError("contract/assets changed during recording startup")
        actor, current_provenance = train.checked_checkpoint(args.checkpoint, manifest)
        if current_provenance != provenance or source_identity(args.repo_root) != identity:
            raise RuntimeError("checkpoint/source changed during recording startup")
        actor = actor.to(args.device).eval()
        raw_env.set_evaluation_command(tuple(args.command))
        wrapped_env = RslRlVecEnvWrapper(raw_env)  # Constructor resets with fixed command installed.
        wrapped_env.reset()  # Public 2.3 reset returns (TensorDict, extras).
        warm_renderer(raw_env)
        with torch.inference_mode():
            collect_video(raw_env, wrapped_env, actor, round(args.duration_s / POLICY_DT),
                          app.is_running, writer, save_image, args.output_dir, summary)
        summary["recording_ok"] = True
    except Exception as exc:
        summary.update(recording_ok=False, stop_reason="invalid_or_interrupted",
                       runtime_error=f"{type(exc).__name__}: {exc}")
    finally:
        # Each cleanup is independent; failure evidence and encoder trailer precede fast shutdown.
        try:
            for resource in (writer, wrapped_env if wrapped_env is not None else raw_env):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        summary.update(recording_ok=False, runtime_error=f"cleanup failed: {exc}; previous={summary['runtime_error']}")
            video = args.output_dir / "replay.mp4"
            final_image = args.output_dir / "final_frame.png"
            if (summary["frames"] < 2 or not video.is_file() or video.stat().st_size == 0
                    or not final_image.is_file() or final_image.stat().st_size == 0):
                summary.update(recording_ok=False, runtime_error=f"video/final frame missing or empty; previous={summary['runtime_error']}")
            else:
                summary.update(video="replay.mp4", final_frame="final_frame.png", video_sha256=sha256_file(video))
            summary["encoded_duration_s"] = summary["frames"] / FPS
            write_json(args.output_dir / "summary.json", summary)
            print(json.dumps({"recording_ok": summary["recording_ok"], "summary": str(args.output_dir / "summary.json")}), flush=True)
        finally:
            if app is not None:
                app.close()  # May exit the interpreter; never put required artifacts after this.
    return 0 if summary["recording_ok"] else 3


def main(argv=None):
    args, train = parse_arguments(argv)
    report, contract, asset = train.preflight(args)
    report["checkpoint_status"] = "not_loaded_in_stdlib_preflight; checked_checkpoint required before Sim on apply"
    identity = None
    try:
        for package in ("imageio", "imageio-ffmpeg", "numpy", "Pillow"):
            try:
                report.setdefault("recording_dependencies", {})[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                raise RuntimeError(f"missing recording dependency: {package}; no installation attempted")
        if args.output_dir and (args.output_dir.exists() or args.output_dir.is_symlink()):
            raise FileExistsError("output directory already exists; overwriting is forbidden")
        if args.checkpoint and (not args.checkpoint.is_file() or not (args.checkpoint.parent / "run_manifest.json").is_file()):
            raise ValueError("checkpoint and sibling run_manifest.json must exist")
        if contract is not None:
            from wheeled_algo.v40_metrics import validate_command
            args.command = validate_command(contract, args.stage, args.command)
        identity = source_identity(args.repo_root)
    except Exception as exc:
        report["blockers"].append(f"recording rejected: {exc}")
    report.update(ready=not report["blockers"], recording_source=identity,
                  source_case=args.source_case, simulation_started=False)
    train.print_preflight(report)
    if args.preflight_only or not report["ready"]:
        return 0 if report["ready"] else 2
    return record(args, train, contract, asset, report, identity)


if __name__ == "__main__":
    raise SystemExit(main())
