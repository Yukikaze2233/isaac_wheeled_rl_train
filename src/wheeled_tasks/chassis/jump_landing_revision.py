"""Exact jump contact-cost revision; the deployed predecessor is plan v10."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILE = "contracts/v6_task_campaign_jump_worker_v6.json"
SOURCE_SHA = "9c5e14a1aa6da1301ccc6c04b890c858e6104999aa7f5bea5bd0def3af1afa93"
SOURCE_PLAN_FILE = "contracts/v6_task_campaign_v10.json"
SOURCE_PLAN_SHA = "22f76bcd5676dae800207040ce446e0ec7de1a9842380d4b2e4d5ca2e90f3587"
WORKER_FILE = "contracts/v6_task_campaign_jump_worker_v12.json"
DESIGN_FILE = "contracts/v6_task_campaign_jump_expectations_v12.json"
PLAN_FILE = "contracts/v6_task_campaign_v12.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v12.json"
STEPJUMP_FILE = "contracts/v6_task_campaign_stepjump_worker_v6.json"
STEPJUMP_SHA = "47cd1e6369ef7a0398fd44fb2792eaab6381c87359c67de352d062643e2371ac"
SCHEMA = "task-worker-revision-v12"
SCOPE = "verified_physical_touchdown_finite_zero_centered_jump_costs_only"


def read_bound(name, sha, root=ROOT):
    raw = (Path(root) / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"Jump landing frozen source SHA differs: {name}")
    return json.loads(raw)


def revision_binding():
    return dict(schema=SCHEMA, source_contract_file=SOURCE_FILE, source_contract_sha256=SOURCE_SHA)


def apply_landing_revision(config):
    """Only add contact costs and bind their exact same-domain predecessor."""
    from .jump_landing import JUMP_LANDING_V1, validate_jump_landing
    if (config.get("task_campaign", {}).get("domain") != "jump"
            or config["task_campaign"]["clock"] != "domain_actor_updates"):
        raise ValueError("Jump landing revision must retain the jump domain and actor clock")
    settings = deepcopy(JUMP_LANDING_V1)
    validate_jump_landing(settings)
    config["jump_landing"] = settings
    config["task_campaign"]["revision_migration"] = revision_binding()


def revised_design(root=ROOT):
    """Patch the five existing independent rows, preserving evaluation definitions."""
    source = read_bound(SOURCE_FILE, SOURCE_SHA, root)
    binding = source["design_preflight"]
    design = read_bound(binding["file"], binding["sha256"], root)
    design["reviewed_change"] = SCOPE
    design["inherited_jump_landing_design_sha256"] = binding["sha256"]
    for row in design["stages"]["jump"]["roles"].values():
        apply_landing_revision(row["contract"])
        row["runtime"]["modules"]["jump_landing"] = True
        row["runtime"]["jump_landing"] = runtime_expectation(row["contract"])
    return design


def runtime_expectation(config):
    """Independent instance/clock/settings expectations, including fine costs."""
    settings = config["jump_landing"]
    return dict(enabled=settings["enabled"], version=settings["version"],
        window_seconds=settings["window_seconds"], absorption_seconds=settings["absorption_seconds"],
        min_clear_air_seconds=settings["min_clear_air_seconds"], policy_dt=config["policy_dt"],
        clock_semantics="first_verified_physical_contact_nonrenewable", zero_tracking_baselines=True,
        settings=deepcopy(settings))


def validate_landing_parent_alias(source, target, root=ROOT):
    """Authenticate v12 jump as the sole alias of stepjump's unchanged v6 parent.

    Changing the jump worker must not rewrite the downstream frozen contract.
    The alias is checked against both exact contracts, not a domain-name match.
    """
    if (source.get("task_campaign", {}).get("revision_migration", {}).get("schema") != SCHEMA
            or source.get("task_campaign", {}).get("domain") != "jump"
            or target.get("task_campaign", {}).get("domain") != "stepjump"):
        return False
    if target != read_bound(STEPJUMP_FILE, STEPJUMP_SHA, root):
        raise ValueError("Jump landing parent alias requires the unchanged frozen stepjump worker")
    before = read_bound(SOURCE_FILE, SOURCE_SHA, root)
    expected = deepcopy(before)
    apply_landing_revision(expected)
    design = Path(root) / DESIGN_FILE
    expected["design_preflight"] = dict(file=DESIGN_FILE,
        sha256=hashlib.sha256(design.read_bytes()).hexdigest(), stage="jump")
    if json.loads(design.read_text()) != revised_design(root) or source != expected:
        raise ValueError("Jump landing parent alias requires the exact authenticated jump revision")
    return True
