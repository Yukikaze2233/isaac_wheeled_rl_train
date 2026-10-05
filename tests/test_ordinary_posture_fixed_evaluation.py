"""v18 teaches posture without changing the actual v15 predecessor's exams."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
from wheeled_tasks.chassis import ordinary_posture_revision as revision

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = revision.DOMAINS_TO_REVISE
RESTORED_FIELDS = ("skill_specs", "behavior_pool_membership", "behavior_pool_fractions",
                   "new_asset_curriculum", "dense_tracking", "recovery_training", "recovery_takeover")
spec = importlib.util.spec_from_file_location("v18_train_source_entry", ROOT / "scripts/train_chassis.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def read(path):
    return json.loads((ROOT / path).read_text())


def worker(domain):
    return read(revision.worker_file(domain))


@pytest.mark.parametrize("domain", DOMAINS)
def test_v18_fixed_sampling_rewards_and_births_use_actual_v15_source(domain):
    cfg = worker(domain)
    unchanged = deepcopy(cfg)
    source_file, _ = revision.source_binding(domain)
    source_fixed = fixed_suite_contract(read(source_file))
    actual = fixed_suite_contract(cfg)
    for key in RESTORED_FIELDS:
        assert actual.get(key) == source_fixed.get(key), key
    assert actual["ordinary_posture_curriculum"]["enabled"] is False
    assert actual["evaluation"] == cfg["evaluation"]
    assert actual["task_campaign"] == cfg["task_campaign"]
    assert cfg == unchanged
    assert not actual.get("recovery_training", {}).get("prefix_curriculum")
    if "nonstandard_start" in actual:
        assert actual["nonstandard_start"]["enabled"] is False
    if domain in ("jump", "stepjump"):
        assert actual["jump_cycle_reset_on_accepted"] is True


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("role", ("baseline", "quick", "full", "confirmation"))
def test_v18_fixed_projection_matches_all_independent_role_inputs(domain, role):
    cfg = worker(domain)
    design = read(cfg["design_preflight"]["file"])
    expected = design["stages"][domain]["roles"][role]["contract"]
    cfg["evaluation"] = deepcopy(expected["evaluation"])
    actual = fixed_suite_contract(cfg)
    for key in RESTORED_FIELDS + ("scene_groups", "episode_seconds", "evaluation",
            "nonstandard_start", "ordinary_posture_curriculum", "jump_cycle_reset_on_accepted"):
        assert actual.get(key) == expected.get(key), key


@pytest.mark.parametrize("domain", DOMAINS)
def test_v18_fixed_subset_retains_case_definition_without_training_posture(domain):
    cfg = worker(domain)
    names = [case["name"] for case in cfg["evaluation"]["cases"][:2]]
    cfg["evaluation"] = select_evaluation_cases(cfg["evaluation"], names)
    actual = fixed_suite_contract(cfg)
    assert actual["evaluation"] == cfg["evaluation"]
    assert [row["name"] for row in actual["scene_groups"]] == names
    assert actual["ordinary_posture_curriculum"]["enabled"] is False


@pytest.mark.parametrize("fault", ("source_sha", "unlaunched_source", "cross_domain", "unknown_domain",
                                   "case_threshold", "cycle_reset"))
def test_v18_fixed_evaluation_rejects_unbound_or_weakened_exams(fault):
    cfg = worker("jump")
    binding = cfg["task_campaign"]["revision_migration"]
    if fault == "source_sha":
        binding["source_contract_sha256"] = "0" * 64
    elif fault == "unlaunched_source":
        binding["source_contract_file"] = "contracts/v6_task_campaign_jump_worker_v17.json"
    elif fault == "cross_domain":
        cfg["task_campaign"]["domain"] = "terrain"
    elif fault == "unknown_domain":
        cfg["task_campaign"]["domain"] = "unknown"
    elif fault == "case_threshold":
        cfg["evaluation"]["cases"][0]["height_mae_m_max"] = 1.
    else:
        cfg["jump_cycle_reset_on_accepted"] = False
    with pytest.raises(ValueError):
        fixed_suite_contract(cfg)


@pytest.mark.parametrize("domain", DOMAINS)
def test_v18_source_audit_binds_new_native_module_revision_and_independent_design(domain):
    cfg = worker(domain)
    files = entry.scut_curriculum_source_files(cfg)
    assert len(files) == len(set(files))
    assert all((ROOT / file).is_file() for file in files)
    for name in ("ordinary_posture_curriculum", "ordinary_posture_revision", "scut_curriculum_revision"):
        assert f"src/wheeled_tasks/chassis/{name}.py" in files
    assert cfg["design_preflight"]["file"] in files
    assert cfg["task_campaign"]["revision_migration"]["source_contract_file"] in files
    if domain == "speed":
        assert "src/wheeled_tasks/chassis/locomotion_quality_revision.py" in files
        assert "src/wheeled_tasks/chassis/scut_jump_revision.py" not in files
        assert "src/wheeled_tasks/chassis/jump_landing_revision.py" not in files
    cfg["evaluation_exact_cases"] = True
    cfg["ordinary_posture_curriculum"]["enabled"] = False
    if "nonstandard_start" in cfg:
        cfg["nonstandard_start"]["enabled"] = False
    assert entry.scut_curriculum_source_files(cfg) == files


@pytest.mark.parametrize("fault", ("source_sha", "source_file", "design_sha", "design_file"))
def test_v18_source_audit_rejects_changed_source_or_design_binding(fault):
    cfg = worker("speed")
    if fault == "source_sha":
        cfg["task_campaign"]["revision_migration"]["source_contract_sha256"] = "0" * 64
    elif fault == "source_file":
        cfg["task_campaign"]["revision_migration"]["source_contract_file"] = revision.worker_file("jump")
    elif fault == "design_sha":
        cfg["design_preflight"]["sha256"] = "0" * 64
    else:
        cfg["design_preflight"]["file"] = revision.design_file("jump")
    with pytest.raises(ValueError, match="source audit"):
        entry.scut_curriculum_source_files(cfg)
