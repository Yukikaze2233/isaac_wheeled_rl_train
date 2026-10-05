"""v17 teaches new lessons without changing the predecessor's fixed exams."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("terrain", "jump", "stepjump", "recovery")
RESTORED_FIELDS = ("skill_specs", "behavior_pool_membership", "behavior_pool_fractions",
                   "new_asset_curriculum", "dense_tracking", "recovery_training", "recovery_takeover")


def read(path):
    return json.loads((ROOT / path).read_text())


@pytest.mark.parametrize("domain", DOMAINS)
def test_all_four_revised_workers_restore_exact_sha_bound_fixed_sampling_and_rewards(domain):
    worker = read(f"contracts/v6_task_campaign_{domain}_worker_v17.json")
    unchanged = deepcopy(worker)
    before = read(worker["task_campaign"]["revision_migration"]["source_contract_file"])
    source_fixed, actual = fixed_suite_contract(before), fixed_suite_contract(worker)
    for key in RESTORED_FIELDS:
        assert actual.get(key) == source_fixed.get(key), key
    assert actual["evaluation"] == worker["evaluation"]
    assert actual["task_campaign"] == worker["task_campaign"]
    assert actual["evaluation_exact_cases"] is True
    assert actual["jump_assist"]["enabled"] is False
    assert worker == unchanged
    assert "prefix_curriculum" not in actual.get("recovery_training", {})
    if "nonstandard_start" in actual:
        assert actual["nonstandard_start"]["enabled"] is False
    if domain in ("jump", "stepjump"):
        assert actual["jump_cycle_reset_on_accepted"] is True


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("role", ("baseline", "quick", "full", "confirmation"))
def test_revised_worker_fixed_projection_matches_every_independent_role(domain, role):
    worker = read(f"contracts/v6_task_campaign_{domain}_worker_v17.json")
    design = read(worker["design_preflight"]["file"])
    expected = design["stages"][domain]["roles"][role]["contract"]
    worker["evaluation"] = deepcopy(expected["evaluation"])
    actual = fixed_suite_contract(worker)
    for key in RESTORED_FIELDS:
        assert actual.get(key) == expected.get(key), key
    for key in ("scene_groups", "episode_seconds", "evaluation", "nonstandard_start",
                "jump_cycle_reset_on_accepted"):
        assert actual.get(key) == expected.get(key), key


@pytest.mark.parametrize("fault", ("source_sha", "source_file", "cross_domain", "case_threshold", "case_pose", "cycle_reset"))
def test_revised_fixed_evaluation_rejects_unbound_or_weakened_exams(fault):
    worker = read("contracts/v6_task_campaign_jump_worker_v17.json")
    binding = worker["task_campaign"]["revision_migration"]
    if fault == "source_sha":
        binding["source_contract_sha256"] = "0" * 64
    elif fault == "source_file":
        binding["source_contract_file"] = "contracts/v6_task_campaign_jump_worker_v6.json"
    elif fault == "cross_domain":
        worker["task_campaign"]["domain"] = "terrain"
    elif fault == "case_threshold":
        worker["evaluation"]["cases"][0]["height_mae_m_max"] = 1.
    elif fault == "case_pose":
        worker["evaluation"]["cases"][0]["command"][2] = .3
    else:
        worker["jump_cycle_reset_on_accepted"] = False
    with pytest.raises(ValueError):
        fixed_suite_contract(worker)


@pytest.mark.parametrize("domain", DOMAINS)
def test_fixed_case_subset_keeps_actual_definition_and_canonical_placement(domain):
    worker = read(f"contracts/v6_task_campaign_{domain}_worker_v17.json")
    names = [case["name"] for case in worker["evaluation"]["cases"][:2]]
    worker["evaluation"] = select_evaluation_cases(worker["evaluation"], names)
    actual = fixed_suite_contract(worker)
    assert [row["name"] for row in actual["scene_groups"]] == names
    assert actual["evaluation"] == worker["evaluation"]
    assert [case["name"] for case in actual["evaluation"]["cases"]] == names
