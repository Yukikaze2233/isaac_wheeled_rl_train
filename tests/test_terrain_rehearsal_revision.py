"""Terrain v15 changes named sampling, never learning provenance or physics."""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wheeled_tasks.chassis import locomotion_quality_revision as quality
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis import terrain_rehearsal_revision as revision
from wheeled_tasks.chassis.task import choose_scene_groups

from train_chassis import resume_critic_warmup
import run_task_campaign as runner


def read(name, root=ROOT):
    return json.loads((root / name).read_text())


def write(name, value, root):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return campaign.digest(path)


@pytest.fixture
def before():
    return read(revision.SOURCE_WORKER_FILE)


@pytest.fixture
def target():
    return read(revision.WORKER_FILE)


def transformed(before):
    result = deepcopy(before)
    revision.apply_rehearsal_revision(result, root=ROOT)
    return result


EXPECTED_POOLS = dict(untriggered_steps=.42, surface_traversal=.10,
    surface_height_joint=.04, stand_anchors=.09, stand_continuous=.06, height=.025,
    basic_translation=.06, basic_yaw=.045, basic_transitions=.06,
    high_translation=.05, release_stop=.05)
HIGH_NAMES = {f"{direction}_{speed}{height}"
    for direction in ("forward", "backward") for speed in (2, 3, 4, 5)
    for height in ("", "__height_train")}
SURFACE_NAMES = {"slope_up", "slope_down", "cross_slope", "rough", "material_transfer"}
SAMPLING_KEYS = {"scene_groups", "skill_specs", "behavior_pool_membership",
    "behavior_pool_fractions"}


def test_exact_source_registry_is_small_and_bound_to_three_frozen_domains():
    expected = {
        revision.SOURCE_WORKER_FILE: "35014b99f4c8ac7423df386bdddf809699c10fc75be56e6a73002f8f532be1d0",
        revision.LEGACY_WORKER_FILE: "4337842f9689ec2a4ecd0994dfb39ef821c95a9868a91f445de52f2a0afc456e",
        revision.FLAT_FILE: "666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7",
        revision.SPEED_FILE: "f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3",
    }
    for file, sha in expected.items():
        assert campaign.digest(ROOT / file) == sha
    binding = revision.revision_binding()
    assert binding == dict(schema="task-worker-revision-v15",
        source_contract_file=revision.SOURCE_WORKER_FILE,
        source_contract_sha256=expected[revision.SOURCE_WORKER_FILE],
        accepted_resume_sources=[
            dict(source_contract_file=revision.SOURCE_WORKER_FILE,
                source_contract_sha256=expected[revision.SOURCE_WORKER_FILE]),
            dict(source_contract_file=revision.LEGACY_WORKER_FILE,
                source_contract_sha256=expected[revision.LEGACY_WORKER_FILE])])
    assert "origin_actor_update" not in binding


def test_sampling_keeps_all_43_original_specs_and_all_41_flat_behaviors(before):
    patch = revision.sampling_patch(root=ROOT)
    assert set(patch) == SAMPLING_KEYS
    names = [row["name"] for row in patch["scene_groups"]]
    assert len(names) == len(set(names)) == len(patch["skill_specs"]) == 85
    flat = read(revision.FLAT_FILE)
    assert len(before["skill_specs"]) == 43 and len(flat["skill_specs"]) == 41
    for source in (before, flat):
        for name, spec in source["skill_specs"].items():
            assert patch["skill_specs"][name] == spec
    # Coverage includes transition tracking, not just stationary anchor groups.
    assert "height_scan" in names
    old_scenes = {row["name"]: row for row in before["scene_groups"]}
    scenes = {row["name"]: row for row in patch["scene_groups"]}
    for name, row in old_scenes.items():
        assert {k: v for k, v in scenes[name].items() if k != "fraction"} == {
            k: v for k, v in row.items() if k != "fraction"}


def test_high_translation_is_exactly_16_sha_bound_flat_only_groups():
    patch, speed = revision.sampling_patch(root=ROOT), read(revision.SPEED_FILE)
    membership = patch["behavior_pool_membership"]
    assert {name for name, pool in membership.items() if pool == "high_translation"} == HIGH_NAMES
    scenes = {row["name"]: row for row in patch["scene_groups"]}
    for name in HIGH_NAMES:
        assert scenes[name]["terrain"] == ["flat"]
        assert patch["skill_specs"][name] == speed["skill_specs"][name]
    assert not any(name.startswith(("spin_translate", "rotate_")) for name in HIGH_NAMES)


def test_five_surface_height_variants_bind_bounded_height_and_command_domains(before):
    patch = revision.sampling_patch(root=ROOT)
    membership = patch["behavior_pool_membership"]
    names = {name for name, pool in membership.items() if pool == "surface_height_joint"}
    assert names == {name + "__height_train" for name in SURFACE_NAMES}
    scenes = {row["name"]: row for row in patch["scene_groups"]}
    old_scenes = {row["name"]: row for row in before["scene_groups"]}
    for original in SURFACE_NAMES:
        name = original + "__height_train"
        spec = patch["skill_specs"][name]
        assert scenes[name]["terrain"] == old_scenes[original]["terrain"]
        assert spec["planar_command_sampling"] == dict(vx_range_m_s=[.2, .8],
            yaw_range_rad_s=[-.1, .1], nominal_fraction=.25)
        assert spec["height_sampling"] == dict(range_m=[.26, .36], nominal_fraction=0.,
            endpoint_fraction=.2, transition_seconds=4., start_at_target_fraction=.5,
            boundary_fraction_of_range=.1, boundary_weight=2.)
        assert spec["height_transition_seconds"] == 4.
        for key, value in before["skill_specs"][original].items():
            if key not in ("planar_command_sampling", "height_sampling", "height_transition_seconds"):
                assert spec[key] == value


def test_named_pool_ratios_fill_every_one_of_4096_native_environment_slots():
    patch = revision.sampling_patch(root=ROOT)
    assert patch["behavior_pool_fractions"] == EXPECTED_POOLS
    membership = patch["behavior_pool_membership"]
    pools = Counter()
    for row in patch["scene_groups"]:
        pools[membership[row["name"]]] += row["fraction"]
    assert pools == pytest.approx(EXPECTED_POOLS, abs=1e-12)
    assert sum(pools[k] for k in ("untriggered_steps", "surface_traversal", "surface_height_joint")) == pytest.approx(.56)
    assert sum(pools[k] for k in ("stand_anchors", "stand_continuous", "height",
        "basic_translation", "basic_yaw", "basic_transitions")) == pytest.approx(.34)
    assert pools["high_translation"] == pytest.approx(.05)
    assert pools["release_stop"] == pytest.approx(.05)
    counts = Counter(name for name, _ in choose_scene_groups(patch["scene_groups"], 4096))
    assert sum(counts.values()) == 4096 and len(counts) == 85
    assert set(counts) == set(patch["skill_specs"]) and min(counts.values()) > 0
    for row in patch["scene_groups"]:
        ideal = 4096 * row["fraction"]
        assert counts[row["name"]] in (math.floor(ideal), math.ceil(ideal))
    actual_scene = {name: terrain for name, terrain in choose_scene_groups(patch["scene_groups"], 4096)}
    assert all(actual_scene[name] == "flat" for name in HIGH_NAMES)
    plan = revision.plan_sampling_pools(root=ROOT)
    assert len(plan) == len(EXPECTED_POOLS)
    assert {p["name"]: p["fraction"] for p in plan} == EXPECTED_POOLS
    assert Counter(name for p in plan for name in p["skills"]) == Counter({name: 1 for name in membership})
    assert all(p["cross_height"] is False for p in plan)


def test_transform_preserves_every_non_sampling_field_including_birth_origin(before):
    actual = transformed(before)
    for key, value in before.items():
        if key not in SAMPLING_KEYS | {"task_campaign"}:
            assert actual[key] == value, key
    old_meta, new_meta = deepcopy(before["task_campaign"]), deepcopy(actual["task_campaign"])
    old_meta.pop("revision_migration")
    assert new_meta.pop("revision_migration") == revision.revision_binding()
    assert old_meta == new_meta
    assert actual["nonstandard_start"]["origin_actor_update"] == 1186
    assert (actual["physics_dt"], actual["policy_dt"], actual["actor_dim"],
        actual["critic_dim"], actual["action_dim"], actual["target_num_envs"]) == (.005, .02, 35, 81, 6, 4096)
    assert actual["rollback_refunds_consumed_updates"] is False


def test_all_five_roles_keep_fixed_cases_runtime_and_evaluation_exact_scene(before):
    design = revision.revised_design(root=ROOT)
    source = read(before["design_preflight"]["file"])
    old_roles, roles = source["stages"]["terrain"]["roles"], design["stages"]["terrain"]["roles"]
    assert set(roles) == {"train", "baseline", "quick", "full", "confirmation"}
    for role, row in roles.items():
        previous = old_roles[role]
        config = row["contract"]
        assert row["runtime"] == previous["runtime"]
        assert config["evaluation"] == previous["contract"]["evaluation"]
        assert config["evaluation_exact_cases"] is (role != "train")
        for key, value in previous["contract"].items():
            if key not in SAMPLING_KEYS | {"task_campaign"}:
                assert config[key] == value, (role, key)
        if role == "train":
            assert len(config["scene_groups"]) == 85
        else:
            for key in SAMPLING_KEYS:
                assert config[key] == previous["contract"][key], (role, key)
            assert len(config["scene_groups"]) == len(config["evaluation"]["cases"]) == 28
        assert config["task_campaign"]["revision_migration"] == revision.revision_binding()


@pytest.mark.parametrize("role", ["train", "baseline", "quick", "full", "confirmation"])
def test_native_role_materialization_matches_independent_design_and_pinned_cases(target, role):
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
    from wheeled_tasks.chassis.preflight import validate_design_contract
    plan = read(revision.PLAN_FILE)
    row = next(row for row in plan["domains"] if row["name"] == "terrain")
    design = read(revision.DESIGN_FILE)
    effective = deepcopy(target)
    seed, count = 290927, 4096
    if role != "train":
        seed = 290930 if role == "confirmation" else 290929
        names = sorted(set(row["required_case_names"]) | set(row.get("monitor_case_names", [])))
        effective["evaluation"]["seed"] = seed
        effective["evaluation"] = select_evaluation_cases(effective["evaluation"], names)
        effective = fixed_suite_contract(effective)
        count = 4 * len(names)
        old_design = read(read(revision.SOURCE_WORKER_FILE)["design_preflight"]["file"])
        previous = old_design["stages"]["terrain"]["roles"][role]["contract"]
        assert effective["evaluation"] == previous["evaluation"]
        assert effective["scene_groups"] == previous["scene_groups"]
    expected = design["stages"]["terrain"]["roles"][role]["contract"]
    for key in SAMPLING_KEYS:
        assert effective[key] == expected[key], (role, key)
    validate_design_contract(effective, design, load_chassis_control(ROOT / effective["control_math_source"]),
        stage="terrain", role=role, seed=seed, num_envs=count, block_index=0)


@pytest.mark.parametrize("domain,clock", [("speed", "domain_actor_updates"),
    ("terrain", "consumed_updates"), ("terrain", "episode_successes")])
def test_pure_patch_does_not_accept_another_domain_or_advancement_clock(before, domain, clock):
    config = deepcopy(before)
    config["task_campaign"].update(domain=domain, clock=clock)
    with pytest.raises(ValueError):
        revision.apply_rehearsal_revision(config, root=ROOT)


@pytest.mark.parametrize("source_file", [revision.SOURCE_WORKER_FILE, revision.LEGACY_WORKER_FILE])
def test_two_exact_same_domain_resume_sources_are_accepted(target, source_file):
    source = read(source_file)
    campaign.validate_worker_revision(source, target)
    result = campaign.validate_task_resume(source, target)
    assert result["task_clock"] == "restored"
    assert result["actor_critic_optimizer"] == "preserved"
    assert result["hardware_release"] is False and result["accepted_stage"] is None
    if source_file == revision.LEGACY_WORKER_FILE:
        expected = deepcopy(source)
        quality.apply_quality_revision(expected)
        revision.apply_rehearsal_revision(expected, root=ROOT)
        expected["design_preflight"] = target["design_preflight"]
        assert expected == target


@pytest.mark.parametrize("fault", ["height_scan", "high_on_rough", "fraction", "DR", "quality_cost",
    "birth_origin", "source_is_speed", "extra_predecessor", "alias_sha", "alias_order", "refund"])
def test_frozen_worker_rejects_hidden_changes_and_extra_resume_aliases(target, fault):
    if fault == "height_scan":
        target["skill_specs"].pop("height_scan")
        target["scene_groups"] = [r for r in target["scene_groups"] if r["name"] != "height_scan"]
    elif fault == "high_on_rough":
        next(r for r in target["scene_groups"] if r["name"] == "forward_3")["terrain"] = ["rough"]
    elif fault == "fraction":
        target["scene_groups"][0]["fraction"] *= 2
    elif fault == "DR":
        target["signal_delay"]["enabled"] = False
    elif fault == "quality_cost":
        target["dense_tracking"]["stand_still_lin_vel"]["weight"] = -4.
    elif fault == "birth_origin":
        target["nonstandard_start"]["origin_actor_update"] += 1
    elif fault == "source_is_speed":
        target["task_campaign"]["revision_migration"].update(source_contract_file=revision.SPEED_FILE,
            source_contract_sha256=revision.SPEED_SHA)
    elif fault == "extra_predecessor":
        target["task_campaign"]["revision_migration"]["accepted_resume_sources"].append(
            dict(source_contract_file=revision.SPEED_FILE, source_contract_sha256=revision.SPEED_SHA))
    elif fault == "alias_sha":
        target["task_campaign"]["revision_migration"]["accepted_resume_sources"][1]["source_contract_sha256"] = "0" * 64
    elif fault == "alias_order":
        target["task_campaign"]["revision_migration"]["accepted_resume_sources"].reverse()
    else:
        target["rollback_refunds_consumed_updates"] = True
    for source_file in (revision.SOURCE_WORKER_FILE, revision.LEGACY_WORKER_FILE):
        with pytest.raises(ValueError):
            campaign.validate_worker_revision(read(source_file), target)


@pytest.mark.parametrize("source_file", [revision.SPEED_FILE,
    "contracts/v6_task_campaign_terrain_worker_v13.json"])
def test_no_borrowed_speed_or_unlisted_intermediate_same_domain_learning_state(target, source_file):
    with pytest.raises(ValueError):
        campaign.validate_task_resume(read(source_file), target)


def test_original_flat_parent_remains_legal_only_as_the_native_new_domain_parent(target, tmp_path):
    flat = read(revision.FLAT_FILE)
    with pytest.raises(ValueError):
        campaign.validate_worker_revision(flat, target)
    # The unchanged cross-domain parent is not a new same-domain resume alias.
    assert campaign.validate_task_resume(flat, target)["task_clock"] == "new_domain_zero"
    checkpoint, _, _ = sealed_checkpoint(tmp_path, revision.LEGACY_WORKER_FILE, 1186)
    job = object.__new__(runner.TaskCampaign)
    job.root = tmp_path
    job.plan = read(revision.PLAN_FILE)
    parent = tmp_path / "flat/not-selected.pt"
    job.report = dict(domains=dict(flat=dict(candidate_checkpoint=str(parent))))
    row = next(row for row in job.plan["domains"] if row["name"] == "terrain")
    assert job.learning_source(row) == checkpoint


def sealed_checkpoint(tmp_path, source_file, actor_clock):
    # Authentication reads only these bytes and clocks; no actor or simulator executes.
    path = tmp_path / "terrain/attempt_002/train/model.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"CPU-only sealed learning-state fixture")
    contract = path.with_suffix(".contract.json")
    shutil.copyfile(ROOT / source_file, contract)
    seal = dict(schema="chassis-learning-checkpoint-v1", sealed=True,
        checkpoint_sha256=campaign.digest(path), contract_sha256=campaign.digest(contract),
        learning_updates=26012, task_actor_update=actor_clock,
        stage_actor_update=17855, actor_updates_consumed=25902)
    path.with_suffix(".checkpoint.json").write_text(json.dumps(seal))
    args = SimpleNamespace(resume_task_change=True, resume=path, num_envs=4096, seed=290927,
        learning_rate_scale=1., global_actor_update_offset=0, block_index=0, updates=20,
        consumed_updates=26012, task_actor_update=actor_clock,
        stage_actor_update=seal["stage_actor_update"], actor_updates_consumed=seal["actor_updates_consumed"])
    return path, args, seal


@pytest.mark.parametrize("source_file,actor_clock", [
    (revision.LEGACY_WORKER_FILE, 1186), (revision.SOURCE_WORKER_FILE, 1257)])
def test_restored_actor_clock_is_actual_seal_u_not_the_fixed_birth_origin(target, tmp_path, source_file, actor_clock):
    path, args, seal = sealed_checkpoint(tmp_path, source_file, actor_clock)
    campaign.validate_worker(target, args)
    assert args.task_actor_update == seal["task_actor_update"] == actor_clock
    assert target["nonstandard_start"]["origin_actor_update"] == 1186
    source_sha = campaign.digest(path.with_suffix(".contract.json"))
    infos = dict(contract_sha256=source_sha, task_campaign=dict(domain="terrain"),
        task_actor_update=actor_clock, critic_warmup_remaining=0, successful_updates_total=26012)
    target_sha = campaign.digest(ROOT / revision.WORKER_FILE)
    assert resume_critic_warmup(target, infos, contract_sha256=target_sha, task_change=True) == 10
    assert infos["task_actor_update"] == actor_clock
    infos.update(contract_sha256=target_sha, critic_warmup_remaining=4)
    assert resume_critic_warmup(target, infos, contract_sha256=target_sha, task_change=True) == 4
    infos["critic_warmup_remaining"] = 0
    assert resume_critic_warmup(target, infos, contract_sha256=target_sha, task_change=True) == 0


@pytest.mark.parametrize("fault", ["rewind_to_birth_origin", "consumption_refund", "wrong_lineage", "speed_PT"])
def test_native_invocation_cannot_rewind_or_replace_the_sealed_learning_source(target, tmp_path, fault):
    source = revision.SPEED_FILE if fault == "speed_PT" else revision.SOURCE_WORKER_FILE
    _, args, _ = sealed_checkpoint(tmp_path, source, 1257)
    if fault == "rewind_to_birth_origin":
        args.task_actor_update = 1186
    elif fault == "consumption_refund":
        args.actor_updates_consumed -= 1
    elif fault == "wrong_lineage":
        args.stage_actor_update -= 1
    with pytest.raises(ValueError):
        campaign.validate_worker(target, args)


def test_frozen_plan_changes_only_terrain_sampling_not_parent_budget_or_endpoints():
    source, target = read(revision.SOURCE_PLAN_FILE), read(revision.PLAN_FILE)
    receipt = campaign.validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
    assert receipt["consumption"] == "preserved_no_refund" and receipt["task_clock"] == "restored"
    assert source["domain_completion"] == target["domain_completion"]
    assert source["total_ceiling"] == target["total_ceiling"] == 121000
    assert source["target_num_envs"] == target["target_num_envs"] == 4096
    for old, new in zip(source["domains"], target["domains"], strict=True):
        assert old["parent_domain"] == new["parent_domain"]
        assert old["updates"] == new["updates"] == 20000
        if old["name"] != "terrain":
            assert old == new
        else:
            assert new["sampling_pools"] == revision.plan_sampling_pools(root=ROOT)
    attempts = {"terrain/attempt_000/train": dict(domain="terrain", consumed_updates=1196,
        actor_updates=1186), "speed/attempt_003/train": dict(domain="speed",
        consumed_updates=5063, actor_updates=5033)}
    assert campaign.account_updates(source, attempts) == campaign.account_updates(target, attempts)


@pytest.fixture
def copied_contracts(tmp_path, monkeypatch):
    # This migration reads a small explicit registry, not all 220 MB of history.
    names = {revision.SOURCE_PLAN_FILE, revision.PLAN_FILE, revision.MANIFEST_FILE,
        revision.SOURCE_WORKER_FILE, revision.LEGACY_WORKER_FILE, revision.FLAT_FILE,
        revision.SPEED_FILE, revision.DESIGN_FILE}
    for file in (revision.SOURCE_PLAN_FILE, revision.PLAN_FILE):
        plan = read(file)
        names.update(row["contract"] for row in plan["domains"])
        names.add(plan["budget_spec_file"])
    names.add(read(revision.SOURCE_WORKER_FILE)["design_preflight"]["file"])
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("fault", ["endpoint", "parent", "budget", "plan_pool", "other_worker",
    "extra_reopening", "missing_eval_role", "eval_case", "eval_scene", "runtime_DR",
    "resealed_high_on_rough", "resealed_reward", "source_bytes"])
def test_resealing_candidate_and_manifest_cannot_authorize_unreviewed_changes(copied_contracts, fault):
    root = copied_contracts
    source, target = read(revision.SOURCE_PLAN_FILE, root), read(revision.PLAN_FILE, root)
    worker, design = read(revision.WORKER_FILE, root), read(revision.DESIGN_FILE, root)
    manifest = read(revision.MANIFEST_FILE, root)
    terrain = next(row for row in target["domains"] if row["name"] == "terrain")
    if fault == "endpoint":
        target["domain_completion"]["schedule"]["end_actor_updates"]["terrain"] += 1
    elif fault == "parent":
        terrain["parent_domain"] = "speed"
    elif fault == "budget":
        target["total_ceiling"] += 100
    elif fault == "plan_pool":
        terrain["sampling_pools"][0]["fraction"] += .01
    elif fault == "other_worker":
        next(row for row in target["domains"] if row["name"] == "speed")["sha256"] = "0" * 64
    elif fault == "extra_reopening":
        manifest["reopen_actor_origins"] = dict(terrain=1186)
    elif fault == "missing_eval_role":
        del design["stages"]["terrain"]["roles"]["confirmation"]
    elif fault == "eval_case":
        design["stages"]["terrain"]["roles"]["quick"]["contract"]["evaluation"]["cases"][0]["episode_seconds"] += 1
    elif fault == "eval_scene":
        design["stages"]["terrain"]["roles"]["full"]["contract"]["scene_groups"][0]["terrain"] = ["rough"]
    elif fault == "runtime_DR":
        design["stages"]["terrain"]["roles"]["train"]["runtime"]["signal_delay"]["enabled"] = False
    elif fault == "resealed_high_on_rough":
        next(row for row in worker["scene_groups"] if row["name"] == "forward_3")["terrain"] = ["rough"]
    elif fault == "resealed_reward":
        worker["dense_tracking"]["stand_still_lin_vel"]["weight"] = -4.
    else:
        path = root / revision.SOURCE_WORKER_FILE
        path.write_bytes(path.read_bytes() + b" ")
    worker["design_preflight"]["sha256"] = write(revision.DESIGN_FILE, design, root)
    terrain["sha256"] = write(revision.WORKER_FILE, worker, root)
    manifest["target_plan_sha256"] = write(revision.PLAN_FILE, target, root)
    for pin, row in zip(manifest["workers"], target["domains"], strict=True):
        pin["target_sha256"] = row["sha256"]
    write(revision.MANIFEST_FILE, manifest, root)
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(source, target, manifest["run_directory"])


def test_durable_migration_keeps_v14_history_consumption_and_reopening_exactly_once(tmp_path, monkeypatch):
    from wheeled_tasks.chassis import nonstandard_start_revision
    source, target = read(revision.SOURCE_PLAN_FILE), read(revision.PLAN_FILE)
    job = object.__new__(runner.TaskCampaign)
    job.plan, job.root = target, tmp_path
    job.args = SimpleNamespace(contract=ROOT / revision.PLAN_FILE, device="cpu")
    job.claim = tmp_path / "shared_budget/account.json"
    job._session_started = None
    attempts = {"terrain/attempt_000/train": dict(domain="terrain", consumed_updates=1196,
        actor_updates=1186), "speed/attempt_004/train": dict(domain="speed",
        consumed_updates=5063, actor_updates=5033)}
    previous_domains = dict(terrain=dict(status="domain_reopened_unassessed", domain_qualified=None,
        candidate_checkpoint="original-terrain-1186.pt", reopening=dict(schema="task-domain-reopen-v14"),
        completion_history=[dict(plan_revision="v12", record=dict(status="domain_schedule_finished_unassessed"))]),
        speed=dict(status="domain_schedule_finished_unassessed", candidate_checkpoint="speed-final.pt"))
    previous = dict(domains=deepcopy(previous_domains), evaluations=dict(terrain=dict(old="retained")),
        consumed_review_tokens=["old-v14-review"], **campaign.account_updates(source, attempts))
    job.report = {}
    write("campaign_state.json", previous, tmp_path)
    ledger = dict(campaign_id=campaign.CAMPAIGN_ID, run_directory=str(tmp_path),
        contract_sha256=revision.SOURCE_PLAN_SHA, attempts=deepcopy(attempts),
        **campaign.account_updates(source, attempts))
    write("shared_budget/account.json", ledger, tmp_path)

    def authenticated_migration(before, after, root):
        assert before == source and after == target and root == tmp_path
        return dict(schema="task-plan-revision-v15", source_plan_sha256=revision.SOURCE_PLAN_SHA,
            target_plan_sha256=campaign.digest(ROOT / revision.PLAN_FILE), consumption="preserved_no_refund")

    monkeypatch.setattr(runner, "validate_plan_migration", authenticated_migration)
    monkeypatch.setattr(job, "assert_no_live_workers", lambda: None)
    monkeypatch.setattr(nonstandard_start_revision, "reopen_completed_domains",
        lambda *args: pytest.fail("Sampling-only v15 must not reopen any domain"))
    job.open_account()
    assert job.records == attempts
    assert job.report["domains"] == previous_domains
    assert job.report["evaluations"] == previous["evaluations"]
    assert job.report["consumed_review_tokens"] == previous["consumed_review_tokens"]
    assert "domain_reopening" not in job.report["revision_migration"]
    migrated = read("shared_budget/account.json", tmp_path)
    assert migrated["attempts"] == ledger["attempts"]
    for key in ("consumed_updates", "actor_updates_consumed", "domain_consumed_updates"):
        assert migrated[key] == ledger[key]
    once = deepcopy(job.report)
    job.open_account()
    assert job.report == once and job.records == attempts
