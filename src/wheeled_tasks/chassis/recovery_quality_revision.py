"""Exact recovery-only v18 -> v19 recipe; continue the existing actor account.

This authors contracts, never an episode reset, reopening, rollback or optimizer.
The inherited10-step critic adaptation is charged without advancing the actor.
Safety/hold/followup criteria and all origins retain authenticated v18 values.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .nonstandard_start_revision import ROOT, read_bound

SOURCE_PLAN_FILE = "contracts/v6_task_campaign_v18.json"
SOURCE_PLAN_SHA = "027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022"
SOURCE_WORKER_FILE = "contracts/v6_task_campaign_recovery_worker_v18.json"
SOURCE_WORKER_SHA = "7b5ecc7acc6871e072c2ca131d2ad2fa169dfd321b8a6c875cf2df643c5fd013"
SOURCE_DESIGN_FILE = "contracts/v6_task_campaign_recovery_expectations_v18.json"
SOURCE_DESIGN_SHA = "26477f3bc5102203bc5323299601c8107c535246732c97f5086ba3a6a6a4bd23"
PLAN_FILE = "contracts/v6_task_campaign_v19.json"
WORKER_FILE = "contracts/v6_task_campaign_recovery_worker_v19.json"
DESIGN_FILE = "contracts/v6_task_campaign_recovery_expectations_v19.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v19.json"
SCHEMA, WORKER_SCHEMA = "task-plan-revision-v19", "task-worker-revision-v19"
SCOPE = "recovery_native_original_goal_bridge_and_actor_owned_gate_relative_hold_cost_only"
GOAL_BRIDGE = "native-original-followup-after-stable-gate-v1"
CRITIC_ADAPTATION_UPDATES = 10
HOLD_SETTINGS = {
    "enabled": True, "version": "gate-relative-native-hold-cost-v1",
    "scope": "whole-actor-owned-native-motion-hold",
    "state_source": "physical-end-state-same-as-existing-motion-taxes",
    "wheel_radius_m": .06, "wheel_rim_gate_m_s": .25,
    "leg_gate_rad_s": 2., "gyro_gate_rad_s": .75,
    "leg": {"broad_scale": 1., "broad_weight": .25, "fine_scale": .25, "fine_weight": .1},
    "wheel": {"broad_scale": 1., "broad_weight": .25, "fine_scale": .4, "fine_weight": .1},
    "gyro": {"broad_scale": 1., "broad_weight": 1., "fine_scale": .2 / .75, "fine_weight": .35},
}
HOLD_GATE_BINDING = {
    "status": "authenticated_native_fsm_and_geometry",
    "source": "RecoveryFSM.constants+native_fsm_config+ClosedChainHeightModel.wheel_radius_m",
    "wheel_radius_m": .06, "wheel_rim_gate_m_s": .25,
    "leg_gate_rad_s": 2., "gyro_gate_rad_s": .75,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision_binding():
    return dict(schema=WORKER_SCHEMA, source_contract_file=SOURCE_WORKER_FILE,
                source_contract_sha256=SOURCE_WORKER_SHA)


def apply_revision(config):
    meta = config.get("task_campaign", {})
    if meta.get("domain") != "recovery" or meta.get("clock") != "domain_actor_updates":
        raise ValueError("Recovery quality changes only its existing independent recovery account")
    fixed = bool(config.get("evaluation_exact_cases"))
    if not fixed:
        prefix = config["recovery_training"]["prefix_curriculum"]
        if prefix.get("origin_actor_update") != 2294 or prefix.get("stable_seconds") != 1.:
            raise ValueError("Recovery quality preserves the original prefix origin and1s safety gate")
        prefix["goal_bridge"] = GOAL_BRIDGE
    settings = deepcopy(HOLD_SETTINGS)
    settings["enabled"] = not fixed
    config["recovery_hold_cost"] = settings
    meta.update(worker_revision_id="v6-task-campaign-recovery-worker-v19",
                revision_migration=revision_binding())


def revised_design(root=ROOT):
    """Apply the named patch to independent roles, preserving all fixed exams."""
    from .recovery_prefix_curriculum import runtime_expectation

    design = read_bound(SOURCE_DESIGN_FILE, SOURCE_DESIGN_SHA, root)
    design["reviewed_change"] = SCOPE
    design["inherited_recovery_quality_design_sha256"] = SOURCE_DESIGN_SHA
    for row in design["stages"]["recovery"]["roles"].values():
        config, runtime = row["contract"], row["runtime"]
        apply_revision(config)
        runtime["modules"]["recovery_hold_cost"] = config["recovery_hold_cost"]["enabled"]
        runtime["recovery_hold_cost"] = deepcopy(config["recovery_hold_cost"])
        if not config.get("evaluation_exact_cases"):
            runtime["recovery_hold_cost_gate_binding"] = deepcopy(HOLD_GATE_BINDING)
            prefix = config["recovery_training"]["prefix_curriculum"]
            runtime["recovery_prefix_curriculum"] = runtime_expectation(prefix, enabled=True)
            # The producer exposes this complete prefix configuration under
            # its inherited pose-curriculum audit field as well.
            runtime["recovery_pose_curriculum"] = deepcopy(prefix)
    return design


def canonical_worker(domain="recovery", root=ROOT):
    if domain != "recovery":
        raise ValueError("Recovery quality cannot change any of the other five workers")
    root = Path(root)
    if json.loads((root / DESIGN_FILE).read_text()) != revised_design(root):
        raise ValueError("Recovery quality independent design differs")
    worker = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, root)
    apply_revision(worker)
    worker["design_preflight"] = dict(file=DESIGN_FILE, sha256=digest(root / DESIGN_FILE), stage="recovery")
    return worker


def validate_frozen_workers(root=ROOT):
    root = Path(root)
    source = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    for row in source["domains"]:
        read_bound(row["contract"], row["sha256"], root)
    if json.loads((root / WORKER_FILE).read_text()) != canonical_worker(root=root):
        raise ValueError("Recovery quality worker differs from its complete exact recipe")
    return dict(domain="recovery", source_file=SOURCE_WORKER_FILE, source_sha256=SOURCE_WORKER_SHA,
        target_file=WORKER_FILE, target_sha256=digest(root / WORKER_FILE),
        design_file=DESIGN_FILE, design_sha256=digest(root / DESIGN_FILE))


def revised_plan(root=ROOT):
    root = Path(root)
    plan = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    plan["contract_id"] = "v6-task-campaign-v19"
    row = next(row for row in plan["domains"] if row["name"] == "recovery")
    if (row["contract"], row["sha256"]) != (SOURCE_WORKER_FILE, SOURCE_WORKER_SHA):
        raise ValueError("Recovery quality plan requires its actual deployed v18 worker")
    row.update(contract=WORKER_FILE, sha256=digest(root / WORKER_FILE))
    plan["revision_migration"].update(manifest_file=MANIFEST_FILE, source_plan_sha256=SOURCE_PLAN_SHA)
    return plan


def revised_manifest(root=ROOT):
    root = Path(root)
    source, target = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root), revised_plan(root)
    raw = (json.dumps(target, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    return dict(schema=SCHEMA, source_plan_file=SOURCE_PLAN_FILE, source_plan_sha256=SOURCE_PLAN_SHA,
        target_plan_file=PLAN_FILE, target_plan_sha256=hashlib.sha256(raw).hexdigest(),
        run_directory=target["revision_migration"]["run_directory"], quality_scope=SCOPE,
        critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
        workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
            target_file=b["contract"], target_sha256=b["sha256"])
            for a, b in zip(source["domains"], target["domains"], strict=True)],
        designs=[dict(domain="recovery", file=DESIGN_FILE, sha256=digest(root / DESIGN_FILE))])


def freeze_contracts(root=ROOT):
    """Explicit authoring only; output four new files, preserving all old bytes."""
    root = Path(root)
    for file in (DESIGN_FILE, WORKER_FILE, PLAN_FILE, MANIFEST_FILE):
        if (root / file).exists():
            raise ValueError("Recovery quality authoring cannot overwrite an existing frozen file: " + file)
    def write(file, value):
        with (root / file).open("x") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write(DESIGN_FILE, revised_design(root))
    write(WORKER_FILE, canonical_worker(root=root))
    write(PLAN_FILE, revised_plan(root))
    write(MANIFEST_FILE, revised_manifest(root))
    validate_frozen_workers(root)
    return dict(schema=SCHEMA, source_plan_sha256=SOURCE_PLAN_SHA,
        output_sha256={file: digest(root / file) for file in (DESIGN_FILE, WORKER_FILE, PLAN_FILE, MANIFEST_FILE)},
        critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES, reopening=None, qualified=None)
