"""CPU contract inheritance checks; future contracts are created only in tmp."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

from wheeled_tasks.chassis import terrain_runout_revision as recipe
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis.task import choose_scene_groups
from wheeled_tasks.chassis.terrain_runout import route_runout_selection, validate_runout


ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path("/tmp/scut-entropy-frozen-023-av6rrek4")
OUTPUTS = (recipe.DESIGN_FILE, recipe.WORKER_FILE, recipe.PLAN_FILE, recipe.MANIFEST_FILE)


def read(name, root=ROOT):
    return json.loads((root / name).read_text())


def write(name, value, root):
    (root / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                       allow_nan=False) + "\n")


def inputs():
    names = {recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE,
             recipe.SOURCE_DESIGN_FILE, recipe.OWN_SOURCE_WORKER_FILE}
    for row in read(recipe.SOURCE_PLAN_FILE)["domains"]:
        names.add(row["contract"])
        names.add(read(row["contract"])["design_preflight"]["file"])
    return names


def copy_inputs(root):
    (root / "contracts").mkdir()
    for name in inputs():
        shutil.copyfile(ROOT / name, root / name)


@pytest.fixture(scope="module")
def frozen_candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("terrain_runout_contracts")
    copy_inputs(root)
    before = {name: (ROOT / name).read_bytes() if (ROOT / name).exists() else None
              for name in OUTPUTS}
    receipt = recipe.freeze_contracts(root)
    assert before == {name: (ROOT / name).read_bytes() if (ROOT / name).exists() else None
                      for name in OUTPUTS}
    return root, receipt


def test_real_frozen_inputs_and_exclusive_tmp_freeze_keep_all_old_bytes(frozen_candidate):
    root, receipt = frozen_candidate
    for name, expected in (
            (recipe.SOURCE_PLAN_FILE, recipe.SOURCE_PLAN_SHA),
            (recipe.SOURCE_WORKER_FILE, recipe.SOURCE_WORKER_SHA),
            (recipe.SOURCE_DESIGN_FILE, recipe.SOURCE_DESIGN_SHA),
            (recipe.OWN_SOURCE_WORKER_FILE, recipe.OWN_SOURCE_WORKER_SHA)):
        # A missing original freeze is a failure, never a skipped validation.
        assert (ROOT / name).read_bytes() == (FROZEN / name).read_bytes()
        assert recipe.digest(ROOT / name) == expected
    for name in inputs():
        assert (root / name).read_bytes() == (ROOT / name).read_bytes()
    assert set(receipt["output_sha256"]) == set(OUTPUTS)
    assert all(receipt["output_sha256"][name] == recipe.digest(root / name) for name in OUTPUTS)
    assert receipt["critic_adaptation_updates"] == 10
    assert receipt["qualification"] is None
    assert receipt["deployment"] is receipt["native_runtime_verified"] is False
    with pytest.raises(ValueError, match="replace"):
        recipe.freeze_contracts(root)


def test_whole_worker_has_only_geometry_and_binding_identity_changes(frozen_candidate):
    root, _ = frozen_candidate
    old = read(recipe.SOURCE_WORKER_FILE)
    target = read(recipe.WORKER_FILE, root)
    after = deepcopy(target)
    assert after.pop("terrain_route_runout") == {
        "enabled": True, "support_end_x_m": 6.2, "positive_boundary_x_m": 5.6}
    assert after["task_campaign"].pop("terrain_runout_revision_migration") == recipe.revision_binding()
    after["task_campaign"]["worker_revision_id"] = old["task_campaign"]["worker_revision_id"]
    after["design_preflight"] = old["design_preflight"]
    assert after == old
    assert target["task_campaign"]["revision_migration"] == old["task_campaign"]["revision_migration"]
    assert target == recipe.canonical_worker(root=root)
    assert target["resume_critic_warmup_updates"] == 10
    assert target["critic_warmup_updates"] == 0


def test_all_five_design_roles_preserve_every_existing_expectation(frozen_candidate):
    root, _ = frozen_candidate
    old, new = read(recipe.SOURCE_DESIGN_FILE), read(recipe.DESIGN_FILE, root)
    original = deepcopy(new)
    assert original.pop("inherited_terrain_runout_design_sha256") == recipe.SOURCE_DESIGN_SHA
    original["reviewed_change"] = old["reviewed_change"]
    assert set(old["stages"]["terrain"]["roles"]) == {"train", "baseline", "quick", "full", "confirmation"}
    for role, row in original["stages"]["terrain"]["roles"].items():
        baseline = old["stages"]["terrain"]["roles"][role]
        settings = row["contract"].pop("terrain_route_runout")
        assert settings == dict(recipe.SETTINGS, enabled=role == "train")
        assert row["contract"]["task_campaign"].pop("terrain_runout_revision_migration") == recipe.revision_binding()
        row["contract"]["task_campaign"]["worker_revision_id"] = baseline["contract"]["task_campaign"]["worker_revision_id"]
        assert row["runtime"]["modules"].pop("terrain_route_runout") is (role == "train")
        runtime = row["runtime"].pop("terrain_route_runout")
        if role == "train":
            assert runtime["selected_group_count"] == 54
            assert runtime["selected_environment_count"] == 1724
            assert runtime["settings"] == recipe.SETTINGS
            assert runtime["negative_boundary_x_m"] == -3.6
            assert runtime["lateral_boundary_y_m"] == 1.7
            assert runtime["corridor_spacing_m"] == 332.
            assert runtime["corridor_y_spacing_m"] == 9.
            assert sum(runtime["selected_group_counts"].values()) == 1724
        else:
            assert runtime == {"enabled": False}
        assert row == baseline
    assert original == old


def test_runtime_population_matches_bound_129_native_groups(frozen_candidate):
    root, _ = frozen_candidate
    target = read(recipe.WORKER_FILE, root)
    expected = read(recipe.DESIGN_FILE, root)["stages"]["terrain"]["roles"]["train"]["runtime"]["terrain_route_runout"]
    route_groups = sorted(row["name"] for row in target["scene_groups"]
        if target["skill_specs"][row["name"]].get("mode") in (2, 3))
    assert len(target["scene_groups"]) == 129
    assert expected["selected_groups"] == route_groups
    native = choose_scene_groups(read(recipe.SOURCE_WORKER_FILE)["scene_groups"], 4096)
    actual_counts = {group: sum(name == group for name, kind in native) for group in route_groups}
    assert expected["selected_group_counts"] == actual_counts
    assert sum(actual_counts.values()) == 1724
    for row in target["scene_groups"]:
        if "body_climb" in target["skill_specs"][row["name"]]:
            assert row["name"] not in route_groups


def test_fixed_exam_keeps_actual_predecessor_geometry_catalogue_and_criteria(frozen_candidate):
    root, _ = frozen_candidate
    old = fixed_suite_contract(read(recipe.SOURCE_WORKER_FILE))
    candidate = fixed_suite_contract(read(recipe.WORKER_FILE, root))
    assert validate_runout(candidate) is None
    assert len(candidate["evaluation"]["cases"]) == 322
    assert all(not route_runout_selection(candidate, row["name"], row["terrain"][0])
               for row in candidate["scene_groups"])
    candidate.pop("terrain_route_runout")
    candidate["task_campaign"].pop("terrain_runout_revision_migration")
    candidate["task_campaign"]["worker_revision_id"] = old["task_campaign"]["worker_revision_id"]
    candidate["design_preflight"] = old["design_preflight"]
    assert candidate == old


def test_exact_plan_change_preserves_other_five_workers_schedule_parents_and_caps(frozen_candidate):
    root, _ = frozen_candidate
    old, new = read(recipe.SOURCE_PLAN_FILE), read(recipe.PLAN_FILE, root)
    restored = deepcopy(new)
    restored["contract_id"] = old["contract_id"]
    restored["revision_migration"] = old["revision_migration"]
    changes = []
    for a, b in zip(old["domains"], restored["domains"], strict=True):
        if a != b:
            changes.append(a["name"])
            assert a["name"] == "terrain"
            b["contract"], b["sha256"] = a["contract"], a["sha256"]
    assert changes == ["terrain"]
    assert restored == old
    assert new["total_ceiling"] == 121000
    assert all(row["updates"] == 20000 for row in new["domains"])
    assert new["domain_completion"]["schedule"]["end_actor_updates"] == {
        "flat": 13000, "recovery": 15000, "speed": 15000,
        "terrain": 15000, "jump": 15000, "stepjump": 15000}
    assert new == recipe.revised_plan(root)
    receipt = recipe.validate_frozen_workers(root)
    assert receipt["domain"] == "terrain"
    for row in old["domains"]:
        assert recipe.digest(root / row["contract"]) == row["sha256"]


def test_manifest_binds_all_workers_and_one_adaptation_without_reopening(frozen_candidate):
    root, _ = frozen_candidate
    manifest = read(recipe.MANIFEST_FILE, root)
    assert manifest == recipe.revised_manifest(root)
    assert manifest["target_plan_sha256"] == recipe.digest(root / recipe.PLAN_FILE)
    assert manifest["critic_adaptation_updates"] == 10
    assert manifest["task_clock"] == "restored"
    assert manifest["consumption"] == "preserved_no_refund"
    assert manifest["domain_reopening"] == "preserve_existing_no_new_reopening"
    assert len(manifest["workers"]) == 6
    changed = [row for row in manifest["workers"] if row["source_sha256"] != row["target_sha256"]]
    assert len(changed) == 1
    assert changed[0]["source_file"] == recipe.SOURCE_WORKER_FILE
    assert changed[0]["target_file"] == recipe.WORKER_FILE
    assert manifest["designs"] == [{"domain": "terrain", "file": recipe.DESIGN_FILE,
        "sha256": recipe.digest(root / recipe.DESIGN_FILE)}]
    assert "reopen_actor_origins" not in manifest and "reopen_actor_endpoints" not in manifest


@pytest.mark.parametrize("source_file,source_sha,inheritance", [
    (recipe.SOURCE_WORKER_FILE, recipe.SOURCE_WORKER_SHA, [recipe.WORKER_SCHEMA]),
    (recipe.OWN_SOURCE_WORKER_FILE, recipe.OWN_SOURCE_WORKER_SHA,
        ["task-worker-revision-v18", recipe.WORKER_SCHEMA]),
])
def test_exact_real_own_domain_sources_are_supported(frozen_candidate, source_file, source_sha, inheritance):
    root, _ = frozen_candidate
    target = read(recipe.WORKER_FILE, root)
    source = read(source_file)
    before = deepcopy(source)
    result = recipe.validate_worker_revision(source, target, root)
    assert result["source_contract_file"] == source_file
    assert result["source_contract_sha256"] == source_sha
    assert result["inheritance_path"] == inheritance
    assert result["critic_adaptation_updates"] == 10
    assert source == before
    # The registry must also use this existing full recipe authentication for
    # v15 inheritance; the path label above alone is not that validation.
    if source_file == recipe.OWN_SOURCE_WORKER_FILE:
        campaign.validate_worker_revision(source, read(recipe.SOURCE_WORKER_FILE))


@pytest.mark.parametrize("path,value", [
    (("terrain_route_runout", "support_end_x_m"), 6.),
    (("terrain_route_runout", "positive_boundary_x_m"), 5.7),
    (("terrain_route_runout", "enabled"), 1),
    (("terrain_route_runout", "unexpected"), True),
    (("learning_rate",), .0002),
    (("initial_noise_std",), .1),
    (("resume_critic_warmup_updates",), 0),
    (("physics_dt",), .001),
    (("reward_profile",), "other"),
    (("success_hold_seconds",), .1),
    (("task_campaign", "clock"), "reset"),
    (("task_campaign", "revision_migration", "source_contract_sha256"), "0" * 64),
    (("task_campaign", "terrain_runout_revision_migration", "schema"), "unknown"),
    (("task_campaign", "terrain_runout_revision_migration", "terrain_route_runout", "enabled"), 1),
    (("design_preflight", "sha256"), "0" * 64),
])
def test_entire_target_rejects_undeclared_geometry_learning_physics_or_identity_change(frozen_candidate, path, value):
    root, _ = frozen_candidate
    target = read(recipe.WORKER_FILE, root)
    cursor = target
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError, match="exact own-domain recipe"):
        recipe.validate_worker_revision(read(recipe.OWN_SOURCE_WORKER_FILE), target, root)


@pytest.mark.parametrize("source_file", [
    "contracts/v6_task_campaign_flat_worker_v6.json",
    "contracts/v6_task_campaign_speed_worker_v18.json",
    "contracts/v6_task_campaign_recovery_worker_v21.json",
])
def test_other_domain_or_modified_source_cannot_be_an_alias(frozen_candidate, source_file):
    root, _ = frozen_candidate
    with pytest.raises(ValueError, match="own-domain source"):
        recipe.validate_worker_revision(read(source_file), read(recipe.WORKER_FILE, root), root)
    source = read(recipe.OWN_SOURCE_WORKER_FILE)
    source["learning_rate"] = .0002
    with pytest.raises(ValueError, match="own-domain source"):
        recipe.own_domain_source_binding(source, root)


@pytest.mark.parametrize("domain", ["flat", "recovery", "speed", "jump", "stepjump"])
def test_canonical_worker_cannot_revise_another_domain(domain, frozen_candidate):
    root, _ = frozen_candidate
    with pytest.raises(ValueError, match="another"):
        recipe.canonical_worker(domain, root)


@pytest.mark.parametrize("name", OUTPUTS)
def test_any_preexisting_output_prevents_partial_or_replacement_freeze(tmp_path, name):
    copy_inputs(tmp_path)
    existing = tmp_path / name
    existing.write_bytes(b"occupied\n")
    before = {n: (tmp_path / n).read_bytes() for n in inputs()}
    with pytest.raises(ValueError, match="replace"):
        recipe.freeze_contracts(tmp_path)
    assert existing.read_bytes() == b"occupied\n"
    assert all(not (tmp_path / n).exists() for n in OUTPUTS if n != name)
    assert before == {n: (tmp_path / n).read_bytes() for n in inputs()}


@pytest.mark.parametrize("name", [recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE,
    recipe.SOURCE_DESIGN_FILE, recipe.OWN_SOURCE_WORKER_FILE,
    "contracts/v6_task_campaign_recovery_worker_v21.json"])
def test_changed_immutable_input_rejects_before_any_new_contract_write(tmp_path, name):
    copy_inputs(tmp_path)
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        recipe.freeze_contracts(tmp_path)
    assert all(not (tmp_path / n).exists() for n in OUTPUTS)


def test_design_tampering_cannot_generate_a_canonical_worker(tmp_path, frozen_candidate):
    root, _ = frozen_candidate
    copy_inputs(tmp_path)
    design = read(recipe.DESIGN_FILE, root)
    design["stages"]["terrain"]["roles"]["train"]["runtime"]["terrain_route_runout"]["selected_environment_count"] = 1723
    write(recipe.DESIGN_FILE, design, tmp_path)
    with pytest.raises(ValueError, match="full inherited recipe"):
        recipe.canonical_worker(root=tmp_path)


def test_native_adaptation_rule_charges_one_change_and_resumes_remaining(frozen_candidate):
    root, _ = frozen_candidate
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_chassis import resume_critic_warmup

    target = read(recipe.WORKER_FILE, root)
    target_sha = recipe.digest(root / recipe.WORKER_FILE)
    infos = {"contract_sha256": recipe.OWN_SOURCE_WORKER_SHA,
             "successful_updates_total": 2206, "critic_warmup_remaining": 0,
             "task_actor_update": 2186, "stage_actor_update": 15554}
    before = deepcopy(infos)
    assert resume_critic_warmup(target, infos, contract_sha256=target_sha, task_change=True) == 10
    assert infos == before
    infos["contract_sha256"] = recipe.SOURCE_WORKER_SHA
    assert resume_critic_warmup(target, infos, contract_sha256=target_sha, task_change=True) == 10
    infos.update(contract_sha256=target_sha, critic_warmup_remaining=7)
    assert resume_critic_warmup(target, infos, contract_sha256=target_sha, task_change=True) == 7
    infos["critic_warmup_remaining"] = 0
    assert resume_critic_warmup(target, infos, contract_sha256=target_sha, task_change=True) == 0
    assert 2206 + (15000 - 2186) + 10 == 15030 < 20000


def test_repository_outputs_are_absent_or_a_complete_canonical_freeze():
    present = [(ROOT / name).exists() for name in OUTPUTS]
    assert not any(present) or all(present)
    if all(present):
        recipe.validate_frozen_workers(ROOT)
        assert read(recipe.DESIGN_FILE) == recipe.revised_design(ROOT)
        assert read(recipe.PLAN_FILE) == recipe.revised_plan(ROOT)
        assert read(recipe.MANIFEST_FILE) == recipe.revised_manifest(ROOT)
