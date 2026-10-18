"""Select an explicitly budgeted historical81 diagnostic, without granting authority.

This is a CPU-only selection/provenance adapter. It does not open a ledger,
authenticate a resource owner, deserialize learning state, or permit native work.
Only the existing750-parent paired study has a migration validator. A marker
cannot turn another REC parent or recipe into an implemented migration.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .diagnostic_curriculum import validate_diagnostic_plan, validate_diagnostic_resume
from .training_profile import PROFILE, CRITIC_VERSION


MARKER = "v7_diagnostic_budget"
MARKER_SCHEMA = "v7-diagnostic81-budget-v1"
MARKER_SCOPE = "historical81_learning_diagnostic"
LEGACY_SCOPE = "legacy_unmanaged"
NAMED78_SCOPE = "named78"
DIAGNOSTIC_SCOPE = "v7_diagnostic81"


class DiagnosticBudgetError(ValueError):
    pass


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise DiagnosticBudgetError(f"Invalid {name} SHA-256")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticBudgetError(f"Missing explicit {name}")
    return value


def _count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise DiagnosticBudgetError(f"{name} requires integer >= {minimum}")
    return value


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticBudgetError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise DiagnosticBudgetError(f"Nonfinite JSON constant: {value}")


def _json_bytes(path, name):
    """The parsed object and digest always describe the same captured bytes."""
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys,
                           parse_constant=_reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticBudgetError(f"Invalid {name} JSON") from exc
    if not isinstance(value, dict):
        raise DiagnosticBudgetError(f"{name} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan(contract):
    study = contract.get("diagnostic_study")
    if not isinstance(study, Mapping) or set(study) != {"plan", "arm", "num_envs"}:
        raise DiagnosticBudgetError("Budgeted81 requires the existing explicit paired diagnostic study")
    plan = study["plan"]
    if not isinstance(plan, Mapping):
        raise DiagnosticBudgetError("Diagnostic plan must be an object")
    try:
        validate_diagnostic_plan(plan)
        # The original validator uses numeric equality; reject bool/float clocks
        # for a new identity without changing historical V6 behavior.
        for key in ("source_updates", "critic_adaptation_updates", "num_steps_per_env"):
            _count(plan[key], key, 1)
        if study["arm"] not in {stage["name"] for stage in plan["stages"]}:
            raise DiagnosticBudgetError("Unknown diagnostic arm")
        for stage in plan["stages"]:
            _count(stage["updates"], "diagnostic stage updates", 1)
        _count(study["num_envs"], "diagnostic environment count", 32)
        if study["num_envs"] > 16384:
            raise DiagnosticBudgetError("Diagnostic environment count exceeds existing recipe")
        for key in ("source_contract_sha256", "source_checkpoint_sha256"):
            _sha(plan[key], key)
    except (KeyError, TypeError, ValueError) as exc:
        raise DiagnosticBudgetError(f"Only the existing750 diagnostic recipe is supported: {exc}") from exc
    return plan


def select_budget_scope(contract):
    """Return a scope only; this result is never native or budget admission."""
    if not isinstance(contract, Mapping):
        raise DiagnosticBudgetError("Training contract must be an object")
    name = contract.get("training_profile")
    if name is not None and name != PROFILE:
        raise DiagnosticBudgetError("Unknown next-generation training profile")
    if MARKER not in contract:
        if name == PROFILE:
            return NAMED78_SCOPE
        if contract.get("critic_dim") == 78 or contract.get("critic_observation_version") == CRITIC_VERSION:
            raise DiagnosticBudgetError("78D critic requires the named training profile")
        return LEGACY_SCOPE
    marker = contract[MARKER]
    if (not isinstance(marker, Mapping) or set(marker) != {"schema", "scope"}
            or marker.get("schema") != MARKER_SCHEMA or marker.get("scope") != MARKER_SCOPE):
        raise DiagnosticBudgetError("Invalid explicit V7 diagnostic budget marker")
    if name is not None:
        raise DiagnosticBudgetError("Historical81 budget marker cannot mix with a training profile")
    for key, expected in {"actor_dim": 35, "actor_frame_dim": 35, "critic_dim": 81,
                          "action_dim": 6, "history_length": 1}.items():
        if type(contract.get(key)) is not int or contract[key] != expected:
            raise DiagnosticBudgetError(f"Historical81 interface mismatch: {key}")
    if "critic_observation_version" in contract:
        raise DiagnosticBudgetError("Historical81 cannot select a new critic field schema")
    for key in ("remediation", "continuation", "task_campaign", "integrated_production"):
        if key in contract:
            raise DiagnosticBudgetError(f"Diagnostic budget cannot mix another experiment owner: {key}")
    _plan(contract)
    return DIAGNOSTIC_SCOPE


def validate_budgeted_diagnostic_resume(source, target):
    """Strip only the marker, then run the actual historical migration validator.

    This validates a recipe, not the checkpoint's actor/critic/Adam provenance.
    The train entry must still authenticate parent infos and restored state.
    """
    if select_budget_scope(target) != DIAGNOSTIC_SCOPE:
        raise DiagnosticBudgetError("Expected an explicit budgeted81 diagnostic")
    if MARKER in source or select_budget_scope(source) != LEGACY_SCOPE:
        raise DiagnosticBudgetError("First diagnostic fork requires its historical V6 source contract")
    unmarked = deepcopy(target)
    del unmarked[MARKER]
    return validate_diagnostic_resume(source, unmarked)


def diagnostic_source_files(contract):
    """Required additions to the complete existing worker source registry."""
    if select_budget_scope(contract) != DIAGNOSTIC_SCOPE:
        return []
    return ["scripts/train_chassis.py", *(
        f"src/wheeled_tasks/chassis/{name}.py" for name in (
            "diagnostic_budget_selection", "diagnostic_curriculum", "training_budget",
            "training_profile", "policy_transfer", "full_curriculum"))]


def diagnostic_source_identity(contract, source_hashes):
    """Hash the full captured registry with the same encoding as profile_identity."""
    if select_budget_scope(contract) != DIAGNOSTIC_SCOPE:
        return {}
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise DiagnosticBudgetError("Diagnostic requires complete SHA-256 source provenance")
    if set(diagnostic_source_files(contract)) - set(source_hashes):
        raise DiagnosticBudgetError("Diagnostic required source provenance is incomplete")
    for name, digest in source_hashes.items():
        _text(name, "source path")
        _sha(digest, name)
    raw = json.dumps(dict(source_hashes), sort_keys=True, separators=(",", ":")).encode()
    return {"diagnostic_budget_schema": MARKER_SCHEMA,
            "training_source_sha256": hashlib.sha256(raw).hexdigest()}


def reject_legacy_evaluation(contract):
    """Inference has no PPO cost but still needs a V7 allocation owner."""
    if select_budget_scope(contract) == DIAGNOSTIC_SCOPE:
        raise DiagnosticBudgetError(
            "V7 diagnostic81 cannot use legacy evaluation: its GPU allocation owner is not integrated")


def bind_diagnostic_budget(contract, *, review_file, review_sha256, ledger_path,
                           attempt_id, actual_identity, resume, resume_diagnostic,
                           source_contract_file=None, consumed_updates=None,
                           transfer=None, lr_scale=None, other_migrations=()):
    """Bind selection to an existing review; BudgetOwner must authenticate next.

    ``resume_diagnostic=True`` is an explicit first external-parent fork. False
    means a same-attempt registered V7 restart; the original V6 file need not
    remain on disk. No reset/rebase of the common ledger occurs in either case.
    Returned booleans make remaining checks explicit, never award authority.
    """
    from .training_budget import IDENTITY_KEYS, PAID_CAP, REVIEW_SCHEMA

    if select_budget_scope(contract) != DIAGNOSTIC_SCOPE:
        raise DiagnosticBudgetError("Expected an explicit budgeted81 diagnostic")
    if type(resume_diagnostic) is not bool:
        raise DiagnosticBudgetError("resume_diagnostic requires an explicit boolean")
    if resume is None:
        raise DiagnosticBudgetError("Budgeted81 requires an explicit parent or registered resume checkpoint")
    if consumed_updates is not None or transfer is not None or lr_scale is not None:
        raise DiagnosticBudgetError("Budgeted81 forbids consumed-update rebasing, transfer and LR scaling")
    if not isinstance(other_migrations, (tuple, list)) or any(type(flag) is not bool for flag in other_migrations):
        raise DiagnosticBudgetError("Other migration flags must be explicit booleans")
    if any(other_migrations):
        raise DiagnosticBudgetError("Budgeted81 cannot mix other migrations")
    if any(value is None for value in (review_file, review_sha256, ledger_path, attempt_id)):
        raise DiagnosticBudgetError("Budgeted81 requires the existing hash-pinned common20000 budget arguments")
    _sha(review_sha256, "budget review")
    _text(attempt_id, "attempt identity")
    ledger = Path(ledger_path)
    if not ledger.is_absolute() or ledger.resolve() != ledger:
        raise DiagnosticBudgetError("Shared ledger path must be absolute and canonical")
    review, observed_review_sha = _json_bytes(review_file, "budget review")
    if observed_review_sha != review_sha256:
        raise DiagnosticBudgetError("Budget review bytes differ from their pinned SHA-256")
    if (review.get("schema") != REVIEW_SCHEMA or type(review.get("paid_update_cap")) is not int
            or review["paid_update_cap"] != PAID_CAP or review.get("ledger_path") != str(ledger)):
        raise DiagnosticBudgetError("Diagnostic must share the existing common20000 review and ledger")
    attempts = review.get("attempts")
    if not isinstance(attempts, dict) or attempt_id not in attempts:
        raise DiagnosticBudgetError("Diagnostic attempt is not present in the pinned review")
    attempt = attempts[attempt_id]
    if not isinstance(attempt, dict) or set(attempt) != IDENTITY_KEYS | {"role"}:
        raise DiagnosticBudgetError("Diagnostic review attempt requires the exact identity fields")
    if attempt["role"] != "diagnostic":
        raise DiagnosticBudgetError("Budgeted81 requires a reviewed diagnostic attempt role")
    if not isinstance(actual_identity, Mapping) or set(actual_identity) != IDENTITY_KEYS:
        raise DiagnosticBudgetError("Diagnostic actual identity requires the exact fields")
    for identity in (attempt, actual_identity):
        for key in ("contract_sha256", "training_source_sha256", "parent_checkpoint_sha256"):
            _sha(identity[key], key)
        _count(identity["seed"], "seed")
        _text(identity["stage"], "stage")
        _count(identity["requested_updates"], "requested updates", 1)
    if (any(actual_identity[key] != attempt[key] for key in IDENTITY_KEYS - {"requested_updates"})
            or actual_identity["requested_updates"] > attempt["requested_updates"]
            or attempt["requested_updates"] > PAID_CAP):
        raise DiagnosticBudgetError("Diagnostic actual identity differs from the pinned attempt")
    plan = _plan(contract)
    if actual_identity["parent_checkpoint_sha256"] != plan["source_checkpoint_sha256"]:
        raise DiagnosticBudgetError("Attempt must retain the original reviewed V6 parent SHA-256")
    _count(contract.get("total_updates"), "diagnostic total updates", 1)
    if actual_identity["requested_updates"] > contract["total_updates"]:
        raise DiagnosticBudgetError("Requested updates exceed the existing diagnostic recipe")
    checkpoint = Path(resume).resolve(strict=True)
    if not checkpoint.is_file():
        raise DiagnosticBudgetError("Resume checkpoint must be a file")
    recipe_receipt = None
    if resume_diagnostic:
        if source_contract_file is None:
            raise DiagnosticBudgetError("First parent fork requires the authenticated V6 source contract")
        source, source_sha = _json_bytes(source_contract_file, "V6 source contract")
        if source_sha != plan["source_contract_sha256"]:
            raise DiagnosticBudgetError("V6 source contract differs from the diagnostic plan")
        if _checkpoint_digest(checkpoint) != plan["source_checkpoint_sha256"]:
            raise DiagnosticBudgetError("First parent checkpoint differs from the diagnostic plan")
        recipe_receipt = validate_budgeted_diagnostic_resume(source, contract)
    elif source_contract_file is not None:
        raise DiagnosticBudgetError("Registered restart must not re-infer the original V6 source contract")
    return {
        "schema": "v7-diagnostic81-budget-selection-receipt-v1",
        "budget_scope": DIAGNOSTIC_SCOPE,
        "mode": "first_parent_fork" if resume_diagnostic else "attempt_restart",
        "budget_resume_checkpoint": None if resume_diagnostic else str(checkpoint),
        "original_parent_checkpoint_sha256": plan["source_checkpoint_sha256"],
        "original_source_contract_sha256": plan["source_contract_sha256"],
        "budget_review_sha256": review_sha256, "budget_attempt_id": attempt_id,
        "budget_ledger_path": str(ledger), "budget_role": "diagnostic",
        "paid_update_cap": PAID_CAP, "parent_recipe_validation": recipe_receipt,
        "requires_registered_checkpoint_validation": not resume_diagnostic,
        "requires_parent_learning_state_validation": resume_diagnostic,
        "requires_same_contract_learning_state_validation": not resume_diagnostic,
        "budget_owner_authenticated": False, "human_authorization_proven": False,
        "native_admission": False,
        "scope": "selection_only_requires_budget_owner_and_existing_native_admission",
    }
