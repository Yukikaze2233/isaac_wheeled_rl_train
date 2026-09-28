#!/usr/bin/env python3
"""Validate every frozen V6 stage/role and executable CPU probe before PPO."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def validate(plan_path, output):
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    from wheeled_tasks.chassis.full_curriculum import stage_contract
    from wheeled_tasks.chassis.preflight import design_preflight, ROLES
    from validate_v6_functional_probes import run_probes
    from calibrate_v6_rewards import calibration_suite
    from train_chassis import preflight

    plan = json.loads(plan_path.read_text())
    if plan["contract_id"] != "v6-new-asset-curriculum-plan-v1":
        raise ValueError("This preflight is only for the independent new-asset campaign")
    output.mkdir(parents=True, exist_ok=False)
    rows, contracts = [], []
    for recipe in plan["stages"]:
        config = stage_contract({}, plan, recipe, 20000)
        path = output / (config["curriculum_stage"] + ".json")
        path.write_text(json.dumps(config, indent=2) + "\n")
        preflight(path)  # includes the complete sealed asset file manifest
        contracts.append({"stage": config["curriculum_stage"], "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "updates": config["total_updates"]})
        control = load_chassis_control(ROOT / config["control_math_source"])
        for role in ROLES:
            effective = deepcopy(config)
            if role == "train":
                seed, count = plan["training_seed"], 20000
            else:
                seed = config["evaluation"]["confirmation_seed" if role == "confirmation" else "seed"]
                effective["evaluation"]["seed"] = seed
                effective = fixed_suite_contract(effective)
                count = len(effective["evaluation"]["cases"]) * effective["evaluation"]["episodes_per_case"]
            rows.append(design_preflight(effective, control, ROOT, stage=recipe["kind"],
                role=role, seed=seed, num_envs=count, block_index=0))
    probes = run_probes(plan["training_seed"])
    reward, _ = calibration_suite(plan["runtime_base"]["dense_tracking"])
    if not probes["passed"] or not reward["all_checks_passed"]:
        raise RuntimeError("Executable probe or reward counterexample failed")
    receipt = {"status": "passed", "scope": "static_all_roles_and_CPU_functions_not_GPU_or_policy_acceptance",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "formal_actor_updates_consumed": 0, "stage_contracts": contracts,
        "design_preflight": rows, "functional_probes": probes, "reward_counterexamples": reward}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(args.plan, args.output)
    print(json.dumps({"status": receipt["status"], "stage_roles": len(receipt["design_preflight"]),
                      "receipt": str(args.output / "receipt.json")}))


if __name__ == "__main__":
    main()
