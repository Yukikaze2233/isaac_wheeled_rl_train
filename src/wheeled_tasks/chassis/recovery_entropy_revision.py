"""Freeze a recovery-only entropy loss change with all learning state preserved."""
import hashlib
import json
from pathlib import Path

from .nonstandard_start_revision import ROOT, read_bound

SOURCE_PLAN_FILE = "contracts/v6_task_campaign_v20.json"
SOURCE_PLAN_SHA = "5d94e35e3fe1bcfc5ae8488e4ce1b1f5779b8ce9598ee1b5642ceb879f47dfac"
SOURCE_WORKER_FILE = "contracts/v6_task_campaign_recovery_worker_v20.json"
SOURCE_WORKER_SHA = "a64eb6c03231e56d8efd8d83b87f70ee88c198de55fee0c2809aa766c7e19f62"
SOURCE_DESIGN_FILE = "contracts/v6_task_campaign_recovery_expectations_v20.json"
SOURCE_DESIGN_SHA = "a78141cbafa46abbc2b5d6b93462f00406bff5f98f27ed92b18c924600d8aa46"
PLAN_FILE = "contracts/v6_task_campaign_v21.json"
WORKER_FILE = "contracts/v6_task_campaign_recovery_worker_v21.json"
DESIGN_FILE = "contracts/v6_task_campaign_recovery_expectations_v21.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v21.json"
SCHEMA = "task-plan-revision-v21"
WORKER_SCHEMA = "task-worker-entropy-revision-v21"
SCOPE = "recovery_native_entropy_coefficient_zero_only"
ENTROPY_COEF = 0.
CRITIC_ADAPTATION_UPDATES = 10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision_binding():
    return dict(schema=WORKER_SCHEMA, source_contract_file=SOURCE_WORKER_FILE,
                source_contract_sha256=SOURCE_WORKER_SHA, entropy_coef=ENTROPY_COEF)


def apply_revision(config):
    metadata = config.get("task_campaign", {})
    if metadata.get("domain") != "recovery" or metadata.get("clock") != "domain_actor_updates":
        raise ValueError("Entropy revision requires the existing recovery actor account")
    config["entropy_coef"] = ENTROPY_COEF
    # Preserve the independent v20 mechanical-script source binding verbatim.
    metadata.update(worker_revision_id="v6-task-campaign-recovery-worker-v21",
                    entropy_revision_migration=revision_binding())


def revised_design(root=ROOT):
    design = read_bound(SOURCE_DESIGN_FILE, SOURCE_DESIGN_SHA, root)
    design["reviewed_change"] = SCOPE
    design["inherited_recovery_entropy_design_sha256"] = SOURCE_DESIGN_SHA
    for row in design["stages"]["recovery"]["roles"].values():
        apply_revision(row["contract"])
    return design


def canonical_worker(domain="recovery", root=ROOT):
    if domain != "recovery":
        raise ValueError("Entropy revision cannot change another task worker")
    root = Path(root)
    if json.loads((root / DESIGN_FILE).read_text()) != revised_design(root):
        raise ValueError("Entropy revision design differs from its full inherited recipe")
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
        raise ValueError("Entropy revision worker changed undeclared fields")
    return dict(domain="recovery", source_file=SOURCE_WORKER_FILE, source_sha256=SOURCE_WORKER_SHA,
                target_file=WORKER_FILE, target_sha256=digest(root / WORKER_FILE),
                design_file=DESIGN_FILE, design_sha256=digest(root / DESIGN_FILE))


def revised_plan(root=ROOT):
    root = Path(root)
    plan = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root)
    plan["contract_id"] = "v6-task-campaign-v21"
    row = next(row for row in plan["domains"] if row["name"] == "recovery")
    if (row["contract"], row["sha256"]) != (SOURCE_WORKER_FILE, SOURCE_WORKER_SHA):
        raise ValueError("Entropy revision requires the deployed v20 recovery worker")
    row.update(contract=WORKER_FILE, sha256=digest(root / WORKER_FILE))
    plan["revision_migration"].update(manifest_file=MANIFEST_FILE, source_plan_sha256=SOURCE_PLAN_SHA)
    return plan


def revised_manifest(root=ROOT):
    root = Path(root)
    source, target = read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, root), revised_plan(root)
    raw = (json.dumps(target, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    return dict(schema=SCHEMA, source_plan_file=SOURCE_PLAN_FILE, source_plan_sha256=SOURCE_PLAN_SHA,
                target_plan_file=PLAN_FILE, target_plan_sha256=hashlib.sha256(raw).hexdigest(),
                run_directory=target["revision_migration"]["run_directory"], loss_scope=SCOPE,
                entropy_coef=ENTROPY_COEF, critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES,
                learning_state="actor_critic_Adam_std_preserved", sampling="native_Gaussian_unchanged",
                workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
                              target_file=b["contract"], target_sha256=b["sha256"])
                         for a, b in zip(source["domains"], target["domains"], strict=True)],
                designs=[dict(domain="recovery", file=DESIGN_FILE, sha256=digest(root / DESIGN_FILE))])


def freeze_contracts(root=ROOT):
    """Write four new immutable contracts; no restart, budget refund or qualification."""
    root = Path(root)
    paths = (DESIGN_FILE, WORKER_FILE, PLAN_FILE, MANIFEST_FILE)
    if any((root / name).exists() for name in paths):
        raise ValueError("Entropy revision cannot replace existing frozen contracts")
    for name, make in ((DESIGN_FILE, revised_design), (WORKER_FILE, canonical_worker),
                       (PLAN_FILE, revised_plan), (MANIFEST_FILE, revised_manifest)):
        value = make(root=root)
        with (root / name).open("x") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    validate_frozen_workers(root)
    return dict(schema=SCHEMA, output_sha256={name: digest(root / name) for name in paths},
                critic_adaptation_updates=CRITIC_ADAPTATION_UPDATES, qualification=None)
