#!/usr/bin/env python3
"""Fail-closed V40 entry point. Help/preflight do not import Isaac or start Sim."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import re
import os
import subprocess
from urllib.parse import unquote, urlparse
from pathlib import Path
import sys
import types
import traceback
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
LAB_TAG = "v3.0.0-beta2.patch1"
LAB_COMMIT = "ffff603eafc6b74264a5261cc0183d6a65390d78"
# Repository release != Python extension package versions. Pin the installed
# Sim 6 / Lab 3 stack whose APIs this branch actually exercises.
TARGET_VERSIONS = {"isaaclab": "6.1.14", "isaaclab_assets": "0.3.4",
                   "isaaclab_tasks": "1.10.9", "isaaclab_rl": "0.5.5",
                   "isaacsim": "6.0.0.1", "rsl-rl-lib": "5.5.1",
                   "torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128",
                   "onnx": "1.23.0rc1", "onnxruntime": "1.30.0"}


def runtime_version_matches(installed: str, expected: str) -> bool:
    """Stable numeric releases only; accept equivalent trailing zero metadata.

    Isaac 5.1.0 may publish metadata5.1.0.0. A required CUDA local build tag
    must match exactly; pre/dev releases and a different CUDA wheel are rejected.
    """
    pattern = r"^(\d+(?:\.\d+)*)(?:\+([A-Za-z0-9.]+))?$"
    a, b = re.fullmatch(pattern, installed), re.fullmatch(pattern, expected)
    if a is None or b is None:
        return False
    def release(match):
        parts = list(map(int, match.group(1).split('.')))
        while len(parts) > 1 and parts[-1] == 0:
            parts.pop()
        return parts
    return release(a) == release(b) and (a.group(2) or '') == (b.group(2) or '')


METADATA_KEYS = ("contract_id", "contract_sha256", "asset_manifest_sha256")


def add_common_arguments(parser: argparse.ArgumentParser, *, num_envs: int = 1) -> None:
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--usd-cache-dir", type=Path, default=None,
                        help="Optional private importer cache for concurrent jobs")
    parser.add_argument("--preflight-only", action="store_true", help="CPU checks only; nonzero exit on blockers")
    parser.add_argument("--research", action="store_true", help="Accept declared research motor priors, NOT failed collisions")
    parser.add_argument("--num_envs", "--num-envs", type=int, default=num_envs)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--stage", choices=("stand", "height", "locomotion"), default="stand")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")


def check_training_baseline(contract: dict) -> None:
    from wheeled_tasks.v40.contract import is_round2, validate_contract
    validate_contract(contract)
    expected = {"class_name": "ActorCritic", "actor_hidden_dims": [256, 128, 64],
                "critic_hidden_dims": [256, 128, 64], "activation": "elu",
                "empirical_normalization": False}
    if any(contract["policy"].get(key) != value for key, value in expected.items()):
        raise ValueError("contract policy does not match the fixed V40 stock PPO baseline")
    if any(contract["policy"].get(key, False) is not False
           for key in ("actor_obs_normalization", "critic_obs_normalization")):
        raise ValueError("V40 normalization must remain disabled")
    if contract["timing"] != {"physics_dt": 0.005, "decimation": 2, "policy_dt": 0.01}:
        raise ValueError("this baseline requires physics_dt=.005, decimation=2, policy_dt=.01")
    for name in ("stand", "height", "locomotion"):
        stage = contract["commands"]["stages"][name]
        if not is_round2(contract) and (any(abs(v) > .5 for v in stage["vx"]) or any(abs(v) > 1.0 for v in stage["wz"])):
            raise ValueError("initial V40 command range must not exceed vx +/- .5, wz +/- 1")
        if not .28 <= stage["height"][0] <= stage["height"][1] <= .32:
            raise ValueError("initial V40 height range must stay within .28..32m")
        if name in ("stand", "height") and (stage["vx"] != [0.0, 0.0] or stage["wz"] != [0.0, 0.0]):
            raise ValueError("stand/height stages require zero velocity commands")
    if contract["commands"]["stages"]["stand"]["height"] != [.32, .32]:
        raise ValueError("stand stage requires .32m height")


def check_isaaclab_source() -> dict:
    """Verify local official checkout identity without importing Kit or networking."""
    paths = {}
    for package in ('isaaclab', 'isaaclab_assets', 'isaaclab_tasks', 'isaaclab_rl'):
        text = importlib.metadata.distribution(package).read_text('direct_url.json')
        if not text:
            raise ValueError(f'{package}: editable source provenance missing')
        direct = json.loads(text)
        url = urlparse(direct.get('url', ''))
        if url.scheme != 'file' or url.netloc not in ('', 'localhost') or direct.get('dir_info', {}).get('editable') is not True:
            raise ValueError(f'{package}: expected audited editable source')
        paths[package] = Path(unquote(url.path)).resolve()
    root = paths['isaaclab'].parent.parent
    if any(path != root / 'source' / package for package, path in paths.items()):
        raise ValueError('Isaac Lab packages come from different source trees')
    git_env = os.environ.copy()
    git_env.update(GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0')
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, timeout=10, env=git_env)
        if result.returncode != 0:
            raise ValueError('Isaac Lab source check failed: ' + ' '.join(args))
        return result.stdout.strip()
    head = git('rev-parse', 'HEAD')
    tag = git('rev-parse', f'refs/tags/{LAB_TAG}^{{commit}}')
    if head != LAB_COMMIT or tag != LAB_COMMIT:
        raise ValueError('Isaac Lab checkout does not match required release commit')
    if git('remote', 'get-url', 'origin') != 'https://github.com/isaac-sim/IsaacLab.git':
        raise ValueError('Isaac Lab origin is not the audited official URL')
    git('diff', '--exit-code', 'HEAD', '--')
    return {'tag': LAB_TAG, 'commit': head, 'root': str(root), 'tracked_tree_clean': True}


def preflight(args: argparse.Namespace) -> tuple[dict, dict | None, dict | None]:
    """Read-only validation. Missing evidence is a blocker, never an implicit opt-in."""
    report = {"ready": False, "mode": "research" if args.research else "research_not_acknowledged", "stage": args.stage,
              "blockers": [], "versions": {}, "simulation_started": False}
    contract = asset = None
    if os.environ.get("V40_USD_SEED"):
        report["blockers"].append(
            "V40_USD_SEED is not bound to the audited asset manifest; use canonical URDF import"
        )
    if args.num_envs < 1:
        report["blockers"].append("num_envs must be positive")
    try:
        # The contract module is deliberately stdlib-only; core re-exports this API.
        from wheeled_tasks.v40.contract import load_contract, validate_asset, contract_digest
        contract = load_contract(args.contract)
        report["contract_id"] = contract["contract_id"]
        report["contract_sha256"] = contract_digest(contract)
        asset = validate_asset(contract, allow_research=args.research)
        # Redundant, intentional independent launch gate; --research cannot waive it.
        if asset["manifest"].get("collision_validation", {}).get("passed") is not True:
            raise ValueError("static collision_validation.passed must be exactly true")
        report["asset_manifest_sha256"] = asset["asset_manifest_sha256"]
        check_training_baseline(contract)
    except Exception as exc:
        report["blockers"].append(f"contract/asset rejected: {exc}")
    report["python"] = '.'.join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] != (3, 12):
        report["blockers"].append(f"target Python3.12 required; found {report['python']}")
    for package, expected in TARGET_VERSIONS.items():
        try:
            installed = importlib.metadata.version(package)
            report["versions"][package] = installed
            # Explicit prerelease pins (ONNX) are allowed only by exact equality.
            if installed != expected and not runtime_version_matches(installed, expected):
                report["blockers"].append(f"{package}: expected {expected}, found {installed}")
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = None
            report["blockers"].append(f"missing target runtime distribution: {package}=={expected}")
    try:
        report['isaaclab_source'] = check_isaaclab_source()
    except Exception as exc:
        report['isaaclab_source'] = None
        report['blockers'].append(f'Isaac Lab source identity not verified: {exc}')
    report["ready"] = not report["blockers"]
    return report, contract, asset


def print_preflight(report: dict) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


def launch_app(args: argparse.Namespace, *, budget=None):
    """Runtime only: call after all preflight gates, never from help/preflight."""
    if budget is not None:
        budget.check()
    # Lab v2.3.0 treats enable_cameras=False as permission to inherit the env.
    os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
    from isaaclab.app import AppLauncher
    if budget is not None:
        budget.check()  # Import time can exhaust the absolute startup budget.
    config = {"headless": args.headless, "enable_cameras": False,
              "livestream": 0, "device": args.device}
    if not args.headless:
        # Lab 3 defaults to headless unless a visualizer is explicitly selected.
        config["visualizer"] = ["kit"]
    if os.geteuid() == 0:
        config["kit_args"] = "--allow-root"
    launcher = AppLauncher(config)
    # Lab 3.0.0-beta2 leaves /physics/fabricUpdateTransformations=False even for
    # GUI runs, so viewports render bodies frozen at their imported rest pose
    # while PhysX evolves correctly underneath. Force the sync back on.
    if not args.headless:
        import carb.settings as _carb_settings
        _carb_settings.get_settings_interface().set_bool(
            "/physics/fabricUpdateTransformations", True)
    return launcher


def checked_checkpoint(checkpoint: Path, expected_manifest: dict):
    """Reuse the exporter's weights-only validation, including full actor/critic state."""
    from wheeled_algo.v40_export import load_actor_checkpoint
    actor, provenance = load_actor_checkpoint(checkpoint, checkpoint.parent / "run_manifest.json")
    recorded = provenance["run_manifest"]
    for key in (*METADATA_KEYS, "actor_obs_dim", "critic_obs_dim", "action_dim", "policy"):
        if recorded[key] != expected_manifest[key]:
            raise ValueError(f"checkpoint metadata mismatch with current contract/assets: {key}")
    expected_format = expected_manifest.get("runtime", {}).get("checkpoint_format")
    if expected_format and recorded.get("runtime", {}).get("checkpoint_format") != expected_format:
        raise ValueError("checkpoint runtime format does not match this training branch")
    return actor, provenance


def bind_checkpoint_metadata(runner, manifest: dict) -> None:
    """Wrap stock save only (not PPO): automatic and explicit saves share this hook."""
    original_save = runner.save
    fixed = {key: manifest[key] for key in METADATA_KEYS}

    def require_primitive(value):
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("checkpoint infos keys must be strings")
            for child in value.values():
                require_primitive(child)
        elif type(value) is list:
            for child in value:
                require_primitive(child)
        elif value is not None and type(value) not in (str, bool, int, float):
            raise ValueError("checkpoint infos must contain only primitive JSON values")

    def save_with_metadata(self, path, infos=None):
        if infos is not None and type(infos) is not dict:
            raise ValueError("checkpoint infos must be a primitive JSON dict")
        merged = dict(infos or {})
        for key, value in fixed.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"cannot replace checkpoint identity: {key}")
            merged[key] = value
        require_primitive(merged)
        json.dumps(merged, allow_nan=False)
        return original_save(path, infos=merged)

    runner.save = types.MethodType(save_with_metadata, runner)


def restore_checkpoint(runner, checkpoint: Path, *, resume: bool, expected_manifest: dict) -> None:
    """Safe stock state restoration; fine-tune deliberately drops optimizer/iteration."""
    import torch
    saved = torch.load(checkpoint, map_location=runner.device, weights_only=True)
    if not isinstance(saved.get("infos"), dict) or any(
        saved["infos"].get(key) != expected_manifest[key] for key in METADATA_KEYS
    ):
        raise ValueError("loaded checkpoint identity changed after validation")
    if expected_manifest.get("runtime", {}).get("checkpoint_format") == "rsl_rl_5_split_mlp":
        runner.alg.load(saved, load_cfg={"actor": True, "critic": True, "optimizer": resume,
                                       "iteration": resume}, strict=True)
    else:
        runner.alg.policy.load_state_dict(saved["model_state_dict"], strict=True)
    if resume:
        if type(saved.get("iter")) is not int or saved["iter"] < 0:
            raise ValueError("resume requires a nonnegative integer iter")
        runner.alg.optimizer.load_state_dict(saved["optimizer_state_dict"])
        runner.current_learning_iteration = saved["iter"]
    else:
        runner.current_learning_iteration = 0


def make_manifest(contract: dict, asset: dict, args: argparse.Namespace) -> dict:
    from wheeled_tasks.v40.contract import make_run_manifest
    from wheeled_algo.v40_export import validate_manifest
    # The actual API consumes the validated audit result (including raw-file SHA),
    # not the inner JSON manifest whose serialization would change that SHA.
    manifest = make_run_manifest(contract, asset)
    if manifest["asset_manifest_sha256"] != asset["asset_manifest_sha256"]:
        raise ValueError("asset changed between validation and run manifest creation")
    manifest.update({"stage": args.stage, "seed": args.seed,
                     "research": args.research, "target_versions": TARGET_VERSIONS})
    validate_manifest(manifest)
    return manifest


def make_env(args: argparse.Namespace):
    """Call only AFTER AppLauncher. Each construction revalidates the asset gate."""
    from wheeled_tasks.direct.v40_serial.env_cfg import V40EnvCfg
    from wheeled_tasks.direct.v40_serial.env import V40Env
    cfg = V40EnvCfg()
    cfg.contract_path = str(args.contract.resolve()) if args.contract else None
    cfg.usd_cache_dir = str(args.usd_cache_dir.resolve()) if args.usd_cache_dir else None
    # Retain the experimental setting for provenance checks. Unbound external
    # seeds are rejected by preflight and independently by the asset factory.
    cfg.usd_seed = os.environ.get("V40_USD_SEED") or None
    cfg.allow_research = args.research
    cfg.stage = args.stage
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    if not args.headless:
        from omni.appwindow import get_default_app_window
        from isaaclab.app.settings_manager import get_settings_manager
        window = get_default_app_window()
        if window is None or window.get_window() is None or min(window.get_width(), window.get_height()) <= 0:
            raise RuntimeError("GUI requested but Kit did not create a native application window")
        # This Lab beta reads this flag but AppLauncher does not populate it.
        # Derive it from the actual Kit window before SimulationContext caches it.
        get_settings_manager().set_bool("/isaaclab/has_gui", True)
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (1.4, 1.4, 1.0)
        cfg.viewer.lookat = (0.0, 0.0, 0.25)
    env = V40Env(cfg=cfg)
    if args.headless:
        env.render_enabled = False
    elif env.viewport_camera_controller is not None:
        env.viewport_camera_controller.set_view_env_index(0)
    return env


def main(argv: list[str] | None = None) -> int:
    # Capture before ANY AppLauncher/Kit imports mutate Python/linker search paths.
    # Keep original PYTHONPATH/user-site and cwd for the isolated CPU exporter.
    clean_export_environment = os.environ.copy()
    clean_export_cwd = os.getcwd()
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, num_envs=256)
    from wheeled_tasks.v40.live_view import add_arguments, validate_options
    add_arguments(parser)
    parser.add_argument("--max_iterations", "--max-iterations", type=int, default=20000,
                        help="number of learning iterations in THIS invocation; additional iterations when resuming")
    from wheeled_algo.v40_job import (TrainingBudget, PlannedStop, positive_seconds, parse_stop_at,
                                      snapshot_provenance, run_training_job)
    parser.add_argument("--max-runtime-seconds", type=positive_seconds, default=None,
                        help="Soft TRAINING-stage budget only; save/export/transfer need separate time")
    parser.add_argument("--stop-at", type=parse_stop_at, default=None,
                        help="Absolute training cutoff: ISO8601 WITH timezone; does not change platform shutdown")
    parser.add_argument("--run-dir", type=Path, help="New directory only, including for resume")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path, help="Restore stock model, optimizer and iteration into a NEW run")
    source.add_argument("--finetune", type=Path, help="Model weights only; fresh optimizer and iteration zero")
    source.add_argument("--warm-start", type=Path,
                        help="Audited Round2 final -> RSL5 actor/critic/std; fresh optimizer, iteration zero")
    args = parser.parse_args(argv)
    if args.max_iterations < 1:
        parser.error("max_iterations must be positive")
    budget = TrainingBudget(args.max_runtime_seconds, args.stop_at)
    report, contract, asset = preflight(args)
    try:
        validate_options(args)
    except Exception as exc:
        report["blockers"].append(f"live view rejected: {exc}")
    try:
        budget.check()
    except PlannedStop as exc:
        report["blockers"].append(f"training cutoff already reached: {exc.reason}")
    if args.run_dir and (args.run_dir.exists() or args.run_dir.is_symlink()):
        report["blockers"].append("run directory already exists; overwriting is forbidden")
    manifest = None
    if not report["blockers"]:
        try:
            manifest = make_manifest(contract, asset, args)
            manifest["runtime"] = {
                "python": report["python"], "versions": report["versions"],
                "isaaclab_source": report["isaaclab_source"],
                "algorithm": "rsl_rl.algorithms.PPO", "actor_class": "MLPModel",
                "critic_class": "MLPModel", "physics_backend": "physx",
                "checkpoint_format": "rsl_rl_5_split_mlp",
            }
            if args.resume or args.finetune:
                checked_checkpoint(args.resume or args.finetune, manifest)
            if args.warm_start:
                from wheeled_algo.v40_warm_start import prepare_warm_start
                _, lineage = prepare_warm_start(args.warm_start, contract, manifest)
                manifest["source_provenance"] = lineage
                report["source_provenance"] = lineage
        except Exception as exc:
            report["blockers"].append(f"run/checkpoint rejected: {exc}")
    report["ready"] = not report["blockers"]
    print_preflight(report)
    if args.preflight_only or not report["ready"]:
        return 0 if report["ready"] else 2

    # SIGINT/SIGTERM stay flag-only through startup, learn, export and cleanup.
    with budget.signal_handlers():
        # Recheck after potentially expensive validation, before any app startup.
        try:
            # Official launch order: no env/sim/asset imports before app creation.
            launcher = launch_app(args, budget=budget)
        except PlannedStop as exc:
            print(f"No simulation started; untrained: {exc.reason}", file=sys.stderr)
            return 2
        simulation_app = launcher.app
        # AppLauncher may install its own handlers; keep our safe handler through cleanup.
        with budget.signal_handlers():
            env = None
            live_view = None
            exit_code = 1
            try:
                from rsl_rl.runners import OnPolicyRunner
                from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
                from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg

                agent_cfg = V40PPORunnerCfg()
                agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, report["versions"]["rsl-rl-lib"])
                agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
                agent_cfg.seed = args.seed
                agent_cfg.device = args.device
                agent_cfg.max_iterations = args.max_iterations
                run_dir = args.run_dir or (REPO_ROOT / "logs" / "v40_serial" /
                          (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]))
                run_dir.mkdir(parents=True, exist_ok=False)
                with (run_dir / "run_manifest.json").open("x", encoding="utf-8") as stream:
                    json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
                with (run_dir / "agent_config.json").open("x", encoding="utf-8") as stream:
                    json.dump(agent_cfg.to_dict(), stream, indent=2, allow_nan=False)
                snapshot_provenance(run_dir, repo_root=REPO_ROOT, contract_path=args.contract,
                                    contract=contract, asset=asset, manifest=manifest)
                env = make_env(args)
                # Catch edits during startup before constructing the runner or learning.
                if env.contract_sha256 != manifest["contract_sha256"] or env.asset_manifest_sha256 != manifest["asset_manifest_sha256"]:
                    raise RuntimeError("contract/asset changed during startup")
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir), device=args.device)
                bind_checkpoint_metadata(runner, manifest)
                if args.warm_start:
                    from wheeled_algo.v40_warm_start import apply_warm_start, prepare_warm_start
                    split, lineage = prepare_warm_start(args.warm_start, contract, manifest)
                    if lineage != manifest["source_provenance"]:
                        raise ValueError("warm-start parent changed during startup")
                    apply_warm_start(runner, split)
                if args.resume or args.finetune:
                    checked_checkpoint(args.resume or args.finetune, manifest)
                    restore_checkpoint(runner, args.resume or args.finetune, resume=args.resume is not None,
                                       expected_manifest=manifest)
                if args.live_view:
                    from wheeled_tasks.v40.live_view import LiveViewSession
                    live_view = LiveViewSession(args, env.unwrapped, env, asset, run_dir,
                                                clean_export_environment)
                    live_view.start(budget=budget)
                receipt = run_training_job(runner, run_dir, budget, args.max_iterations,
                                           export_environment=clean_export_environment, export_cwd=clean_export_cwd)
                print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
                exit_code = 0 if receipt["status"] in {"completed", "stopped"} else 3
            except BaseException:
                # Preserve the original failure even if Kit exits during cleanup.
                traceback.print_exc()
                raise
            finally:
                try:
                    try:
                        if live_view is not None:
                            live_view.close()
                    finally:
                        if env is not None:
                            env.close()
                except BaseException:
                    exit_code = 1
                    traceback.print_exc()
                    raise
                finally:
                    simulation_app.close(exit_code=exit_code)
            return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
