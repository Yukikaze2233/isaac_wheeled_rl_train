"""Exact recovery-only v19 -> v20 recipe; continue the existing actor account.

This authors contracts, never an episode reset, reopening, rollback or optimizer.
The inherited10-step critic adaptation is charged without advancing the actor.
Safety/hold/followup criteria and all origins retain authenticated v19 values.
Only SIDE speed and the SHA-bound CAPTURE gyro predicate are revised.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .nonstandard_start_revision import ROOT, read_bound

SOURCE_PLAN_FILE = 'contracts/v6_task_campaign_v19.json'
SOURCE_PLAN_SHA = 'dbce7bf39105bb284ee9927f06239ff8ce767bcadbab896f2b653897c19649f9'
SOURCE_WORKER_FILE = 'contracts/v6_task_campaign_recovery_worker_v19.json'
SOURCE_WORKER_SHA = '7918e7850e16b96edfa54a8854d3f3571e8101dfa6527cad78a06b4e15baea95'
SOURCE_DESIGN_FILE = 'contracts/v6_task_campaign_recovery_expectations_v19.json'
SOURCE_DESIGN_SHA = 'f84a73d47f90f941a4520472cd62576b8e55b784ccc0b0643f328b7b7b84d423'
PLAN_FILE = "contracts/v6_task_campaign_v20.json"
WORKER_FILE = "contracts/v6_task_campaign_recovery_worker_v20.json"
DESIGN_FILE = "contracts/v6_task_campaign_recovery_expectations_v20.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v20.json"
SCHEMA, WORKER_SCHEMA = "task-plan-revision-v20", "task-worker-revision-v20"
SCOPE = 'recovery_script_SIDE1p5_and_current_tick_CAPTURE_gyro_only'
CRITIC_ADAPTATION_UPDATES = 10
SOURCE_RECOVERY_FILE = "src/wheeled_tasks/chassis/recovery.py"
SOURCE_RECOVERY_SHA = "04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6"
CAPTURE_GYRO_GATE = "native-current-tick-norm-below-existing-handover-gyro-max"
SIDE_SPEED_RAD_S = 1.5
HANDOVER_GYRO_MAX_RAD_S = .75


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision_binding():
    return dict(schema=WORKER_SCHEMA, source_contract_file=SOURCE_WORKER_FILE,
                source_contract_sha256=SOURCE_WORKER_SHA,
                source_recovery_file=SOURCE_RECOVERY_FILE,
                source_recovery_sha256=SOURCE_RECOVERY_SHA)


def apply_revision(config):
    meta = config.get("task_campaign", {})
    if meta.get("domain") != "recovery" or meta.get("clock") != "domain_actor_updates":
        raise ValueError("Recovery script changes only its existing independent recovery account")
    if not config.get("evaluation_exact_cases"):
        config["recovery_training"]["side_speed_rad_s"] = SIDE_SPEED_RAD_S
    meta.update(worker_revision_id="v6-task-campaign-recovery-worker-v20",
                revision_migration=revision_binding())


def runtime_expectation(config):
    """Bind the actual script configuration and the exact native predicate source."""
    if (config.get("task_campaign", {}).get("domain") != "recovery"
            or config.get("task_campaign", {}).get("revision_migration") != revision_binding()
            or config.get("recovery_training", {}).get("enabled") is not True):
        raise ValueError("Recovery script runtime requires its exact v20 recovery binding")
    settings = config["recovery_training"]
    side = settings.get("side_speed_rad_s", 6.)
    expected_side = 6. if config.get("evaluation_exact_cases") else SIDE_SPEED_RAD_S
    if side != expected_side or settings.get("handover_gyro_max", .75) != HANDOVER_GYRO_MAX_RAD_S:
        raise ValueError("Recovery script side speed or existing gyro gate differs")
    return dict(source_recovery_sha256=SOURCE_RECOVERY_SHA, side_speed_rad_s=side,
        capture_prepare_gyro_gate=CAPTURE_GYRO_GATE,
        handover_gyro_max_rad_s=HANDOVER_GYRO_MAX_RAD_S)


def native_runtime_binding(fsm, config, root=ROOT):
    """Authenticate the real class definition, not any instance observer wrapper."""
    import inspect
    from .recovery import RecoveryFSM

    expected = runtime_expectation(config)
    definition = Path(inspect.getfile(RecoveryFSM.update)).resolve()
    path = (Path(root) / SOURCE_RECOVERY_FILE).resolve()
    if type(fsm) is not RecoveryFSM or definition != path or digest(definition) != SOURCE_RECOVERY_SHA:
        raise ValueError("Recovery script native class/source differs from its frozen predicate")
    actual = dict(expected, side_speed_rad_s=fsm.config["side_speed_rad_s"],
                  handover_gyro_max_rad_s=fsm.config["handover_gyro_max"])
    if actual != expected:
        raise ValueError("Recovery script native FSM speed or gyro gate differs")
    return actual


def revised_design(root=ROOT):
    """Keep the SHA-bound predecessor exam and all inherited teaching settings."""
    design = read_bound(SOURCE_DESIGN_FILE, SOURCE_DESIGN_SHA, root)
    design["reviewed_change"] = SCOPE
    design["inherited_recovery_script_design_sha256"] = SOURCE_DESIGN_SHA
    for row in design["stages"]["recovery"]["roles"].values():
        apply_revision(row["contract"])
        row["runtime"]["recovery_script_revision"] = runtime_expectation(row["contract"])
    return design


def canonical_worker(domain="recovery", root=ROOT):
    if domain != "recovery":
        raise ValueError("Recovery script cannot change any of the other five workers")
    root = Path(root)
    if digest(root / SOURCE_RECOVERY_FILE) != SOURCE_RECOVERY_SHA:
        raise ValueError("Recovery script source differs from its exact frozen native predicate")
    if json.loads((root / DESIGN_FILE).read_text()) != revised_design(root):
        raise ValueError("Recovery script independent design differs")
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
        raise ValueError("Recovery script worker differs from its complete exact recipe")
    return dict(domain="recovery", source_file=SOURCE_WORKER_FILE, source_sha256=SOURCE_WORKER_SHA,
        target_file=WORKER_FILE, target_sha256=digest(root / WORKER_FILE),
        design_file=DESIGN_FILE, design_sha256=digest(root / DESIGN_FILE))


def revised_plan(root=ROOT):
    root = Path(root)
    plan = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    plan["contract_id"] = "v6-task-campaign-v20"
    row = next(row for row in plan["domains"] if row["name"] == "recovery")
    if (row["contract"], row["sha256"]) != (SOURCE_WORKER_FILE, SOURCE_WORKER_SHA):
        raise ValueError("Recovery script plan requires its actual deployed v19 worker")
    row.update(contract=WORKER_FILE, sha256=digest(root / WORKER_FILE))
    plan["revision_migration"].update(manifest_file=MANIFEST_FILE, source_plan_sha256=SOURCE_PLAN_SHA)
    return plan


def revised_manifest(root=ROOT):
    root = Path(root)
    source, target = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root), revised_plan(root)
    raw = (json.dumps(target, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    return dict(schema=SCHEMA, source_plan_file=SOURCE_PLAN_FILE, source_plan_sha256=SOURCE_PLAN_SHA,
        target_plan_file=PLAN_FILE, target_plan_sha256=hashlib.sha256(raw).hexdigest(),
        run_directory=target["revision_migration"]["run_directory"], script_scope=SCOPE,
        critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
        script_source=dict(file=SOURCE_RECOVERY_FILE, sha256=SOURCE_RECOVERY_SHA),
        workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
            target_file=b["contract"], target_sha256=b["sha256"])
            for a, b in zip(source["domains"], target["domains"], strict=True)],
        designs=[dict(domain="recovery", file=DESIGN_FILE, sha256=digest(root / DESIGN_FILE))])


def freeze_contracts(root=ROOT):
    """Explicit authoring only; output four new files, preserving all old bytes."""
    root = Path(root)
    for file in (DESIGN_FILE, WORKER_FILE, PLAN_FILE, MANIFEST_FILE):
        if (root / file).exists():
            raise ValueError("Recovery script authoring cannot overwrite an existing frozen file: " + file)
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
