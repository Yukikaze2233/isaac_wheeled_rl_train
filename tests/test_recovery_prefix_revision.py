"""The new training recipe does not rewrite archived contracts or fixed exams."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis import recovery_prefix_revision as revision
from wheeled_tasks.chassis.locomotion_quality_revision import POSTURE, STATIONARY
from wheeled_tasks.chassis.response_migration import validate_recovery_configuration


ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def test_frozen_v8_source_and_independent_v17_recipe_match_files():
    source = revision.read_bound(revision.SOURCE_WORKER_FILE, revision.SOURCE_WORKER_SHA)
    worker = read(revision.WORKER_FILE)
    revision.apply_prefix_revision(source)
    source["design_preflight"] = worker["design_preflight"]
    assert source == worker
    assert read(revision.DESIGN_FILE) == revision.revised_design()
    assert hashlib.sha256((ROOT / revision.DESIGN_FILE).read_bytes()).hexdigest() == worker["design_preflight"]["sha256"]
    assert worker["task_campaign"]["revision_migration"] == revision.revision_binding()
    assert worker["recovery_training"]["prefix_curriculum"]["origin_actor_update"] == 2294
    assert revision.revision_binding()["schema"] == "task-worker-revision-v17"


def test_all_54_original_recovery_cases_and_41_flat_groups_remain_online():
    before, after = read(revision.SOURCE_WORKER_FILE), read(revision.WORKER_FILE)
    for field in ("scene_groups", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions"):
        assert before[field] == after[field]
    counts, masses = Counter(), Counter()
    for row in after["scene_groups"]:
        pool = after["behavior_pool_membership"][row["name"]]
        counts[pool] += 1
        masses[pool] += row["fraction"]
    assert sum(counts.values()) == 95
    assert masses == pytest.approx(after["behavior_pool_fractions"], abs=1e-12)
    assert counts.pop("scripted_recovery") == 54
    assert sum(counts.values()) == 41
    assert masses.pop("scripted_recovery") == pytest.approx(.65)
    assert sum(masses.values()) == pytest.approx(.35)
    plan = revision.plan_sampling_pools()
    assert sum(len(row["skills"]) for row in plan) == 95
    assert all(row["cross_height"] is False for row in plan)


def test_same_native_control_truth_gates_clocks_and_unqualified_leg_status():
    before, after = read(revision.SOURCE_WORKER_FILE), read(revision.WORKER_FILE)
    changed = {"recovery_takeover", "recovery_training", "dense_tracking", "task_campaign", "design_preflight"}
    for key in before:
        if key not in changed:
            assert before[key] == after[key], key
    old_recovery, new_recovery = deepcopy(before["recovery_training"]), deepcopy(after["recovery_training"])
    old_recovery.pop("pose_curriculum")
    new_recovery.pop("prefix_curriculum")
    assert old_recovery == new_recovery
    assert after["physics_dt"] == .005 and after["policy_dt"] == .02
    assert after["v5_control"] == before["v5_control"]
    validate_recovery_configuration(after["recovery_training"], ROOT)
    old_meta, new_meta = deepcopy(before["task_campaign"]), deepcopy(after["task_campaign"])
    old_meta.pop("revision_migration")
    new_meta.pop("revision_migration")
    assert old_meta == new_meta


def test_no_legacy_hidden_peak_subtraction_and_same_visible_native_costs():
    before, after = read(revision.SOURCE_WORKER_FILE), read(revision.WORKER_FILE)
    assert "recovery_takeover" not in after
    assert "pose_curriculum" not in after["recovery_training"]
    dense = deepcopy(after["dense_tracking"])
    assert dense.pop("stand_still_lin_vel") == STATIONARY
    assert dense.pop("locomotion_posture") == POSTURE
    assert dense == before["dense_tracking"]
    assert STATIONARY["weight"] == -2.


def test_all_independent_roles_preserve_fixed_case_definitions_and_truth():
    worker = read(revision.SOURCE_WORKER_FILE)
    before = read(worker["design_preflight"]["file"])
    after = read(revision.DESIGN_FILE)
    roles = after["stages"]["recovery"]["roles"]
    assert roles.keys() == before["stages"]["recovery"]["roles"].keys()
    for role, row in roles.items():
        old = before["stages"]["recovery"]["roles"][role]
        assert row["control"] == old["control"]
        assert row["invocation"] == old["invocation"]
        assert row["contract"]["evaluation"] == old["contract"]["evaluation"]
        if role == "train":
            assert "recovery_takeover" not in row["contract"]
            assert "recovery_takeover" not in row["runtime"]
            assert row["runtime"]["recovery_prefix_curriculum"]["enabled"]
            assert row["runtime"]["recovery_prefix_curriculum"]["settings"] == row["contract"]["recovery_training"]["prefix_curriculum"]
            assert row["runtime"]["modules"]["recovery_prefix_curriculum"]
        else:
            restored = deepcopy(row["contract"])
            restored["task_campaign"]["revision_migration"] = old["contract"]["task_campaign"]["revision_migration"]
            assert restored == old["contract"]
            assert row["runtime"] == old["runtime"]
            assert "prefix_curriculum" not in row["contract"]["recovery_training"]


def test_two_samplers_or_tampered_frozen_source_are_refused(tmp_path):
    settings = deepcopy(read(revision.WORKER_FILE)["recovery_training"])
    settings["pose_curriculum"] = read(revision.SOURCE_WORKER_FILE)["recovery_training"]["pose_curriculum"]
    with pytest.raises(ValueError, match="one reset curriculum"):
        validate_recovery_configuration(settings)
    (tmp_path / "contracts").mkdir()
    raw = (ROOT / revision.SOURCE_WORKER_FILE).read_bytes() + b" "
    (tmp_path / revision.SOURCE_WORKER_FILE).write_bytes(raw)
    with pytest.raises(ValueError, match="frozen source SHA"):
        revision.read_bound(revision.SOURCE_WORKER_FILE, revision.SOURCE_WORKER_SHA, tmp_path)
