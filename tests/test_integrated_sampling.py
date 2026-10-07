"""Native source coverage and explicit conflict boundaries for joint sampling."""
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path

import pytest

from wheeled_tasks.chassis import integrated_sampling as recipe
from wheeled_tasks.chassis.ordinary_posture_curriculum import ordinary_eligibility
from wheeled_tasks.chassis.task import choose_scene_groups


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sources():
    return recipe.load_sources(ROOT)


@pytest.mark.parametrize("domain,groups", [("speed", 189), ("terrain", 233),
                                          ("jump", 248), ("stepjump", 262)])
def test_native_coverage_allocation_and_frozen_semantics(sources, domain, groups):
    before = deepcopy(sources[domain])
    candidate = recipe.apply(sources[domain], ROOT)
    audit = candidate["integrated_sampling_candidate"]
    assert len(candidate["scene_groups"]) == len(candidate["skill_specs"]) == groups
    assert math.fsum(row["fraction"] for row in candidate["scene_groups"]) == pytest.approx(1., abs=1e-12)
    allocation = Counter(name for name, _ in choose_scene_groups(candidate["scene_groups"], 4096))
    assert sum(allocation.values()) == 4096 and len(allocation) == groups
    assert min(allocation.values()) >= 1 and dict(allocation) == audit["allocation"]["counts"]
    assert sum(spec["kind"] == "recovery_handoff" for spec in candidate["skill_specs"].values()) == 54
    assert candidate["evaluation"] == before["evaluation"]
    for key in recipe.RUNTIME_FIELDS:
        assert candidate.get(key) == before.get(key)
    restored = deepcopy(candidate)
    restored.pop("integrated_sampling_candidate")
    for key in recipe.SAMPLING_FIELDS:
        restored[key] = before[key]
    assert restored == before and sources[domain] == before
    assert audit["runtime_ready"] is False and audit["qualification"] is None
    assert audit["PPO_updates"] == 0
    assert "recovery_training" in audit["pending_runtime_configuration"]
    for key in ("entropy_coef", "recovery_hold_cost", "locomotion_difficulty"):
        assert key in audit["pending_runtime_configuration"]
    if domain != "speed":
        assert "terrain_route_runout" in audit["pending_runtime_configuration"]
    for row in candidate["scene_groups"]:
        name = row["name"]
        receipt = audit["row_provenance"][name]
        source, old_name = receipt["component"], receipt["source_name"]
        native_row = next(item for item in sources[source]["scene_groups"] if item["name"] == old_name)
        assert candidate["skill_specs"][name] == sources[source]["skill_specs"][old_name]
        assert {k: v for k, v in row.items() if k not in {"name", "fraction"}} == {
            k: v for k, v in native_row.items() if k not in {"name", "fraction"}}
        assert candidate["behavior_pool_membership"][name] == sources[source]["behavior_pool_membership"][old_name]
    for component in audit["components"]:
        source = component["source"]
        # Both original ordinary sources and recovery's 41 duplicate rows remain covered.
        assert set(audit["source_coverage"][source]) == set(sources[source]["skill_specs"])


@pytest.mark.parametrize("role", ["fixed_eval", "eval", "playback", "native_review", "export"])
def test_every_non_training_role_is_exact_noop_without_source_access(sources, role):
    assert recipe.apply(sources["terrain"], ROOT / "does-not-exist", role=role) == sources["terrain"]


def test_six_reviewed_index_renames_preserve_both_release_trajectories(sources):
    candidate = recipe.apply(sources["terrain"], ROOT)
    for old, new in recipe.TERRAIN_RENAMES.items():
        assert candidate["skill_specs"][old] == sources["speed"]["skill_specs"][old]
        assert candidate["skill_specs"][new] == sources["terrain"]["skill_specs"][old]
        assert candidate["skill_specs"][old] != candidate["skill_specs"][new]
        assert candidate["skill_specs"][old]["kind"] == candidate["skill_specs"][new]["kind"] == "release_stop"
        assert candidate["integrated_sampling_candidate"]["row_provenance"][new]["source_name"] == old


@pytest.mark.parametrize("change", ["spec", "terrain"])
def test_unreviewed_same_name_conflict_is_rejected_before_exclusion(sources, change):
    changed = deepcopy(sources)
    if change == "spec":
        changed["jump"]["skill_specs"]["stand_305mm"]["command"][2] = .31
    else:
        next(row for row in changed["jump"]["scene_groups"] if row["name"] == "stand_305mm")["terrain"] = ["rough"]
    with pytest.raises(ValueError, match="Native spec/terrain conflict: stand_305mm"):
        recipe._compose(changed, "speed")


def test_authentication_rejects_native_source_bytes_before_composition(tmp_path):
    (tmp_path / "contracts").mkdir()
    file = recipe.SOURCE_BINDINGS["flat"][0]
    (tmp_path / file).write_bytes((ROOT / file).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Source SHA mismatch: flat"):
        recipe.load_sources(tmp_path)


def test_sampling_cannot_hide_a_changed_closed_chain_bound(sources):
    changed = deepcopy(sources)
    changed["speed"]["closure_gap_termination_m"] = .03
    with pytest.raises(ValueError, match="Native ABI incompatible: speed"):
        recipe._compose(changed, "speed")


@pytest.mark.parametrize("domain", list(recipe.COMPONENTS))
def test_stationary_floor_native_pool_names_and_actual_posture_eligibility(sources, domain):
    candidate = recipe.apply(sources[domain], ROOT)
    rows = candidate["scene_groups"]
    pools = Counter()
    for row in rows:
        pools[candidate["behavior_pool_membership"][row["name"]]] += row["fraction"]
    assert dict(pools) == pytest.approx(candidate["behavior_pool_fractions"])
    stationary = pools["stand_anchors"] + pools["stand_continuous"]
    assert stationary >= .1 - 1e-12
    floor = candidate["integrated_sampling_candidate"]["stationary_floor"]
    if floor["applied"]:
        assert stationary == pytest.approx(.1)
        assert pools["stand_anchors"] / pools["stand_continuous"] == pytest.approx(.18 / .07)
    groups = [row["name"] for row in rows]
    terrains = [row["terrain"][0] for row in rows]
    eligible = ordinary_eligibility(candidate, groups, terrains, "cpu")
    for name in ("stand_305mm", "forward_05", "rotate_1", "forward_5", "rotate_1rps"):
        assert name in groups and bool(eligible[groups.index(name)])
    recoveries = [i for i, name in enumerate(groups) if candidate["skill_specs"][name]["kind"] == "recovery_handoff"]
    assert not eligible[recoveries].any()


def test_capacity_rejection_and_unsupported_target_are_explicit(sources):
    with pytest.raises(ValueError, match="at least one environment"):
        recipe.apply(sources["stepjump"], ROOT, count=261)
    with pytest.raises(ValueError, match="Unsupported integrated target"):
        recipe.apply(sources["recovery"], ROOT)
