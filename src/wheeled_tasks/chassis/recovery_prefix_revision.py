"""Named v17 recovery lessons; source state, physics and truth gates stay native."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .recovery_prefix_curriculum import prefix_settings, runtime_expectation
from .scut_base_tracking import apply_training_tracking, update_runtime_expectation


ROOT = Path(__file__).resolve().parents[3]
SOURCE_WORKER_FILE = "contracts/v6_task_campaign_recovery_worker_v8.json"
SOURCE_WORKER_SHA = "f6aa77c6808c9746dc8dee431e8b3331b6b0f883db701335e57069e7df738ee8"
WORKER_FILE = "contracts/v6_task_campaign_recovery_worker_v17.json"
DESIGN_FILE = "contracts/v6_task_campaign_recovery_expectations_v17.json"
SCHEMA = "task-worker-revision-v17"
SCOPE = "native_prefix_takeover_stand_release_original_chain_global_visible_tracking_no_hidden_peak_subtraction"
ORIGIN_ACTOR_UPDATE = 2294


def read_bound(name, sha, root=ROOT):
    raw = (Path(root) / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Recovery prefix frozen source SHA differs: " + name)
    return json.loads(raw)


def revision_binding():
    return dict(schema=SCHEMA, source_contract_file=SOURCE_WORKER_FILE,
        source_contract_sha256=SOURCE_WORKER_SHA, origin_actor_update=ORIGIN_ACTOR_UPDATE)


def apply_prefix_revision(config, root=ROOT):
    """No reward, command or initial-state hot edits; this is a new contract."""
    original = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, root)
    meta = config.get("task_campaign", {})
    if meta.get("domain") != "recovery" or meta.get("clock") != "domain_actor_updates":
        raise ValueError("Recovery prefix requires its unchanged independent domain actor account")
    if not config.get("evaluation_exact_cases"):
        # Same 54 original scenarios and complete flat 41 population. The actual
        # lesson is reset-local and separately labelled, never an altered case.
        for field in ("scene_groups", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions"):
            config[field] = deepcopy(original[field])
        config["recovery_training"].pop("pose_curriculum", None)
        config["recovery_training"]["prefix_curriculum"] = prefix_settings(ORIGIN_ACTOR_UPDATE)
        # Remove both legacy hidden peak subtraction and identity-gated costs.
        # The same visible stationary/tracking rewards remain on ordinary and
        # recovery lanes. Do not add a second quiet tax in this revision.
        config.pop("recovery_takeover", None)
        apply_training_tracking(config)
    meta["revision_migration"] = revision_binding()


def revised_design(root=ROOT):
    """Transform independently bound roles, preserving the fixed-case exams."""
    source = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, root)
    binding = source["design_preflight"]
    design = read_bound(binding["file"], binding["sha256"], root)
    design["reviewed_change"] = SCOPE
    design["inherited_recovery_prefix_design_sha256"] = binding["sha256"]
    for row in design["stages"]["recovery"]["roles"].values():
        config, runtime = row["contract"], row["runtime"]
        apply_prefix_revision(config, root)
        update_runtime_expectation(config, runtime)
        if not config.get("evaluation_exact_cases"):
            settings = config["recovery_training"]["prefix_curriculum"]
            runtime.pop("recovery_takeover", None)
            runtime["recovery_pose_curriculum"] = deepcopy(settings)
            runtime["modules"]["recovery_prefix_curriculum"] = True
            runtime["recovery_prefix_curriculum"] = runtime_expectation(settings, enabled=True)
    return design


def plan_sampling_pools(root=ROOT):
    source = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, root)
    return [dict(name=pool, fraction=fraction, skills=[row["name"] for row in source["scene_groups"]
        if source["behavior_pool_membership"][row["name"]] == pool], cross_height=False)
        for pool, fraction in source["behavior_pool_fractions"].items()]


def freeze_worker(root=ROOT):
    """Explicit authoring helper; never called by a running worker."""
    root = Path(root)
    design = revised_design(root)
    raw = (json.dumps(design, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    (root / DESIGN_FILE).write_bytes(raw)
    worker = read_bound(SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, root)
    apply_prefix_revision(worker, root)
    worker["design_preflight"] = dict(file=DESIGN_FILE,
        sha256=hashlib.sha256(raw).hexdigest(), stage="recovery")
    (root / WORKER_FILE).write_text(json.dumps(worker, indent=2, ensure_ascii=False,
        allow_nan=False) + "\n")
    return dict(worker_sha256=hashlib.sha256((root / WORKER_FILE).read_bytes()).hexdigest(),
        design_sha256=hashlib.sha256(raw).hexdigest(), source_sha256=SOURCE_WORKER_SHA,
        qualification=False, native_prefix_validation="pending")
