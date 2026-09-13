#!/usr/bin/env python3
"""Plan or run a separate native V40 checkpoint replay; never attach to a trainer.

Prepare an immutable experiment copy of the frozen repository and copy a stable
periodic checkpoint plus its run_manifest/contract/source_hashes beforehand.
Default preflight is CPU/stdlib-only. --launch is an explicit GPU/Kit opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True
TITLE = "TRAINING CHECKPOINT REPLAY"
FROZEN_COMMIT = "6ed83408d1d1cb02ea687e77fa80e5c20932f36c"
LAB_COMMIT = "ffff603eafc6b74264a5261cc0183d6a65390d78"
ARCHIVE_SHA = "7430847bb4a94124ddd0f25793e70adeba5c4846721821fea152672f09fc5850"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preflight(args):
    if not re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256):
        raise ValueError("recorded checkpoint SHA-256 required")
    if not 1 <= args.seconds <= 300:
        raise ValueError("short replay duration must be 1..300 seconds")
    if args.output.exists():
        raise FileExistsError("replay output must be new")
    if args.repo.name != "isaac_wheeled_rl_train":
        raise ValueError("the experiment code directory must retain the repository name")
    if not args.repo.resolve().is_relative_to((args.native_root / "experiments").resolve()):
        raise ValueError("use a new Windows experiment copy, not the frozen WSL training repository")
    if digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("copied checkpoint differs from the recorded stable source hash")
    inputs = args.checkpoint.parent
    source = json.loads((inputs / "source_hashes.json").read_text())
    for name, expected in source["source_files_sha256"].items():
        path = (args.repo / name).resolve()
        if not path.is_relative_to(args.repo.resolve()) or digest(path) != expected:
            raise ValueError(f"frozen training source mismatch: {name}")
    sys.path.insert(0, str(args.repo / "src"))
    from wheeled_tasks.v40.contract import contract_digest, load_contract, validate_asset
    contract = load_contract(inputs / "contract.json")
    asset = validate_asset(contract, allow_research=True)
    manifest = json.loads((inputs / "run_manifest.json").read_text())
    if (contract_digest(contract) != manifest["contract_sha256"]
            or asset["asset_manifest_sha256"] != manifest["asset_manifest_sha256"]):
        raise ValueError("checkpoint input contract/asset identity mismatch")
    if manifest["runtime"]["checkpoint_format"] != "rsl_rl_5_split_mlp":
        raise ValueError("this entry expects the current RSL 5 split checkpoint")
    for value, key in zip(args.command, ("vx", "wz", "height")):
        low, high = contract["commands"]["stages"][manifest["stage"]][key]
        if not low <= value <= high:
            raise ValueError(f"replay {key} command outside the saved contract")
    archive = json.loads((args.native_root / "downloads/isaac-sim-standalone-6.0.0-windows-x86_64.zip.json").read_text())
    lab = json.loads((args.native_root / "audits/lab-source.json").read_text(encoding="utf-8-sig"))
    if archive.get("sha256") != ARCHIVE_SHA or archive.get("status") != "verified" or lab.get("commit") != LAB_COMMIT:
        raise ValueError("native binary/source installation receipt mismatch")
    versions = {name: metadata.version(name) for name in ("isaaclab", "isaaclab_physx", "torch", "torchvision")}
    expected = {"isaaclab": "6.1.14", "isaaclab_physx": "1.1.3", "torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"}
    if versions != expected or os.name != "nt" or sys.version_info[:2] != (3, 12):
        raise ValueError(f"requires the isolated native Windows runtime: {versions}")
    return {"title": TITLE, "ready_for_controlled_runtime_test": True, "runtime_tested": False,
            "source_commit_label": FROZEN_COMMIT, "source_files_verified": len(source["source_files_sha256"]),
            "checkpoint": args.checkpoint.name, "checkpoint_sha256": args.checkpoint_sha256,
            "contract_sha256": manifest["contract_sha256"], "asset_manifest_sha256": manifest["asset_manifest_sha256"],
            "versions": versions, "source_run": args.source_run, "command": args.command,
            "semantics": "new native Windows PhysX inference rollout; NOT current trainer poses; no learning"}, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=Path("D:/isaac60-native"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", nargs=3, type=float, default=(0.0, 0.0, 0.30))
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--seed", type=int, default=143)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    report, manifest = preflight(args)
    print(json.dumps(report, indent=2), flush=True)
    if not args.launch:
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "replay-input.json").write_text(json.dumps(report, indent=2))
    app = env = None
    code = 1
    try:
        # Reuse the frozen repository's weights-only loader and complete actor/critic validation.
        import torch
        from wheeled_algo.v40_export import load_actor_checkpoint
        actor, provenance = load_actor_checkpoint(args.checkpoint, args.checkpoint.parent / "run_manifest.json")
        if provenance["checkpoint_sha256"] != args.checkpoint_sha256:
            raise ValueError("checkpoint changed after preflight")
        from isaaclab.app import AppLauncher
        os.environ.update(PUBLIC_IP="127.0.0.1", LIVESTREAM="1", ENABLE_CAMERAS="0", OMNI_KIT_ACCEPT_EULA="YES")
        launcher = AppLauncher({"headless": True, "livestream": 1, "device": "cuda:0", "visualizer": ["kit"],
            "kit_args": "--/physics/fabricUpdateTransformations=true --/app/window/hideUi=false "
                        "--/renderer/multiGpu/enabled=false --/exts/omni.services.livestream.session/quitOnSessionEnded=false "
                        "--/log/file=" + str(args.output / "kit.log")})
        app = launcher.app
        import carb
        import omni.ui as ui
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_visualizers.kit import KitVisualizerCfg
        from wheeled_tasks.direct.v40_serial.env_cfg import V40EnvCfg
        from wheeled_tasks.direct.v40_serial.env import V40Env
        carb.settings.get_settings().set_string("/app/window/title", TITLE)
        cfg = V40EnvCfg()
        cfg.contract_path = str(args.checkpoint.parent / "contract.json")
        cfg.usd_cache_dir = str(args.output / "usd-cache")
        cfg.allow_research, cfg.stage, cfg.seed = True, manifest["stage"], args.seed
        cfg.scene.num_envs = 1
        cfg.sim.device, cfg.sim.physics = "cuda:0", PhysxCfg()
        cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=False)]
        env = V40Env(cfg=cfg)
        env.render_enabled = True
        assert len(env.robot.body_names) == 7 and len(env.robot.joint_names) == 6
        assert env.sim.physics_sim_view.is_valid and env.sim.physics_sim_view.device_ordinal == 0
        assert env.sim.physics_sim_view.cuda_context
        env.set_evaluation_command(tuple(args.command))
        observations, _ = env.reset()
        origin = env.scene.env_origins[0].cpu().tolist()
        env.sim.set_camera_view([origin[0] + 1.2, origin[1] + 1.2, origin[2] + .9],
                                [origin[0], origin[1], origin[2] + .25])
        actor = actor.to(env.device)
        panel = ui.Window(TITLE, width=440, height=240)
        with panel.frame:
            with ui.VStack():
                ui.Label(TITLE)
                ui.Label(f"{args.checkpoint.name}\nSHA256 {args.checkpoint_sha256[:20]}...")
                ui.Label("INDEPENDENT NATIVE WINDOWS PHYSX REPLAY\nNOT the running WSL training state; no optimization")
                status = ui.Label("Starting copied policy inference")
        started, steps, last_write = time.monotonic(), 0, 0.0
        while app.is_running() and time.monotonic() - started < args.seconds:
            with torch.inference_mode():
                actions = actor(observations["policy"])
            if not torch.isfinite(actions).all():
                raise ValueError("nonfinite replay action")
            observations, _, _, _, _ = env.step(actions)
            steps += 1
            elapsed = time.monotonic() - started
            if elapsed - last_write >= .2:
                status.text = f"Replay step {steps} | sim {steps * env.step_dt:.2f}s | wall {elapsed:.2f}s\ncommand {tuple(args.command)}"
                state = {**report, "runtime_tested": True, "pid": os.getpid(), "step": steps,
                         "wall_seconds": elapsed, "body_names": env.robot.body_names,
                         "poses_xyzw": env.robot.data.body_link_pose_w.torch[0].detach().cpu().tolist()}
                (args.output / "replay-state.json").write_text(json.dumps(state, indent=2))
                last_write = elapsed
            # Pacing only this separate replay, never the training process.
            time.sleep(max(0.0, started + steps * env.step_dt - time.monotonic()))
        (args.output / "replay-result.json").write_text(json.dumps({**report, "runtime_tested": True,
            "steps": steps, "body_names": env.robot.body_names, "joint_names": env.robot.joint_names,
            "learning_updates": 0, "policy_quality_verified": False}, indent=2))
        code = 0
    except BaseException as error:
        traceback.print_exc()
        (args.output / "replay-error.json").write_text(json.dumps({"error": repr(error), "title": TITLE}))
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close(exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
