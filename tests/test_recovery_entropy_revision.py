"""Protect complete physical, reward, curriculum and budget inheritance."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis import recovery_entropy_revision as recipe
from wheeled_tasks.chassis import recovery_script_revision as script_recipe
from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def test_only_entropy_and_provenance_change_entire_worker_and_roles():
    source, target = read(recipe.SOURCE_WORKER_FILE), read(recipe.WORKER_FILE)
    restored = deepcopy(target)
    assert restored.pop("entropy_coef") == 0.
    assert restored["task_campaign"].pop("entropy_revision_migration") == recipe.revision_binding()
    restored["task_campaign"]["worker_revision_id"] = source["task_campaign"]["worker_revision_id"]
    restored["design_preflight"] = source["design_preflight"]
    assert restored == source
    old_roles = read(recipe.SOURCE_DESIGN_FILE)["stages"]["recovery"]["roles"]
    new_roles = read(recipe.DESIGN_FILE)["stages"]["recovery"]["roles"]
    assert old_roles.keys() == new_roles.keys()
    for role, original in old_roles.items():
        after = deepcopy(new_roles[role])
        assert after["contract"].pop("entropy_coef") == 0.
        after["contract"]["task_campaign"].pop("entropy_revision_migration")
        after["contract"]["task_campaign"]["worker_revision_id"] = original["contract"]["task_campaign"]["worker_revision_id"]
        assert after == original


def test_original_script_class_binding_survives_in_all_roles():
    for row in read(recipe.DESIGN_FILE)["stages"]["recovery"]["roles"].values():
        assert script_recipe.runtime_expectation(row["contract"]) == row["runtime"]["recovery_script_revision"]


def test_exact_plan_and_worker_restore_without_reopening_or_refund():
    old, new = read(recipe.SOURCE_PLAN_FILE), read(recipe.PLAN_FILE)
    receipt = campaign.validate_plan_migration(old, new, new["revision_migration"]["run_directory"])
    assert receipt["task_clock"] == "restored"
    assert receipt["consumption"] == "preserved_no_refund"
    assert "domain_reopening" not in receipt
    assert old["domain_completion"] == new["domain_completion"]
    assert old["total_ceiling"] == new["total_ceiling"] == 121000
    for before, after in zip(old["domains"], new["domains"], strict=True):
        if before["name"] != "recovery":
            assert after == before
    source, target = read(recipe.SOURCE_WORKER_FILE), read(recipe.WORKER_FILE)
    result = campaign.validate_task_resume(source, target)
    assert result["actor_critic_optimizer"] == "preserved"
    assert result["accepted_stage"] is None and result["hardware_release"] is False
    assert target["resume_critic_warmup_updates"] == 10


@pytest.mark.parametrize("path,value", [
    (("entropy_coef",), .005),
    (("entropy_coef",), False),
    (("task_campaign", "entropy_revision_migration", "entropy_coef"), False),
    (("learning_rate",), .0002),
    (("initial_noise_std",), .1),
    (("resume_critic_warmup_updates",), 0),
    (("recovery_training", "stable_seconds"), .5),
    (("recovery_training", "side_speed_rad_s"), 6.),
    (("task_campaign", "entropy_revision_migration", "source_contract_sha256"), "0" * 64),
])
def test_undeclared_learning_physics_or_hold_change_rejected(path, value):
    original, changed = read(recipe.SOURCE_WORKER_FILE), read(recipe.WORKER_FILE)
    cursor = changed
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        campaign.validate_worker_revision(original, changed)


def test_cannot_rewrite_frozen_contracts_or_other_workers():
    with pytest.raises(ValueError, match="replace"):
        recipe.freeze_contracts(ROOT)
    with pytest.raises(ValueError, match="another"):
        recipe.canonical_worker("speed", ROOT)
