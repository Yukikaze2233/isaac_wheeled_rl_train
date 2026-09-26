#!/usr/bin/env python3
"""Materialize a small, explicitly non-formal PPO/restore probe from a stage."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "contracts/v6_emergency_v1.json")
    parser.add_argument("--stage", default="terrain_motion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    args = parser.parse_args()
    if not 32 <= args.num_envs <= 256:
        parser.error("Probe environment count must be in [32,256]")
    import torch
    from train_chassis import digest
    from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(json.loads(args.plan.read_text()), load)
    recipe = next(stage for stage in plan["stages"] if stage["name"] == args.stage)
    config = stage_contract(load(plan["base_contract"]), plan, recipe, 16384)
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if parent["infos"]["contract_sha256"] != digest(args.source_contract):
        raise ValueError("Parent checkpoint does not match the supplied source contract")
    if parent["infos"]["asset_manifest_sha256"] != config["asset_manifest_sha256"]:
        raise ValueError("Probe parent must use the same mechanical asset")
    groups = [name for name in ("stand", "forward_05", "height_full", "height_pulse_stand",
              "step_up_15", "step_up_20", "step_up_25", "jump_small") if name in config["skill_specs"]]
    scenes = {scene["name"]: scene for scene in config["scene_groups"]}
    config["scene_groups"] = [{**deepcopy(scenes[name]), "fraction": 1 / len(groups)} for name in groups]
    config["skill_specs"] = {name: config["skill_specs"][name] for name in groups}
    if config.get("performance_curriculum"):
        config["performance_curriculum"]["retained_groups"] = [
            name for name in config["performance_curriculum"].get("retained_groups", []) if name in groups]
    config["evaluation"]["cases"] = [case for case in config["evaluation"]["cases"]
        if case["name"] in ("stand", "forward_05", "step_up_15", "step_up_20", "step_up_25")]
    config["evaluation"].pop("continuation_selection", None)
    config["evaluation"].pop("retention_case_names", None)
    config["evaluation"].pop("promotion_case_names", None)
    config.update(target_num_envs=args.num_envs, total_updates=6, num_mini_batches=2,
        critic_warmup_updates=1, checkpoint_interval=2, checkpoint_first_update=2,
        save_interval=2, flat_half_length_m=12., flat_triangle_mesh=True,
        evaluation_long_corridors=True,
        probe_scope="subset_pipeline_validation_not_formal_budget_or_skill_acceptance")
    config["stages"][0]["updates"] = 6
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(args.checkpoint, args.output / "parent.pt")
    shutil.copyfile(args.source_contract, args.output / "parent.contract.json")
    (args.output / "preparation.json").write_text(json.dumps({"parent_checkpoint_sha256": digest(args.checkpoint),
        "source_contract_sha256": digest(args.source_contract), "probe_contract_sha256": digest(args.output / "contract.json"),
        "parent_role": "local_test_snapshot_only_formal_parent_selected_at_authorized_deployment",
        "groups": groups, "num_envs": args.num_envs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
