"""Portable CPU regression for the recovery-only named v18 -> v19 revision."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from wheeled_tasks.chassis import recovery_quality_revision as recipe
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from train_chassis import resume_critic_warmup, scut_curriculum_source_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plans():
    return (json.loads((ROOT / recipe.SOURCE_PLAN_FILE).read_text()),
            json.loads((ROOT / recipe.PLAN_FILE).read_text()))


@pytest.fixture(scope="module")
def workers():
    return (json.loads((ROOT / recipe.SOURCE_WORKER_FILE).read_text()),
            json.loads((ROOT / recipe.WORKER_FILE).read_text()))


def test_only_recovery_worker_changes_all_others_and_endpoints_remain_exact(plans, workers):
    source, target = plans
    for old, new in zip(source["domains"], target["domains"], strict=True):
        if old["name"] == "recovery":
            assert {k: v for k, v in old.items() if k not in {"contract", "sha256"}} == {
                k: v for k, v in new.items() if k not in {"contract", "sha256"}}
        else:
            assert new == old
            assert recipe.digest(ROOT / old["contract"]) == old["sha256"]
    assert source["domain_completion"] == target["domain_completion"]
    assert source["budget_spec_sha256"] == target["budget_spec_sha256"]
    assert source["total_ceiling"] == target["total_ceiling"] == 121000
    old, new = workers
    assert old["recovery_training"]["prefix_curriculum"]["origin_actor_update"] == new["recovery_training"]["prefix_curriculum"]["origin_actor_update"] == 2294
    assert old["ordinary_posture_curriculum"] == new["ordinary_posture_curriculum"]
    assert old["recovery_training"]["stable_seconds"] == new["recovery_training"]["stable_seconds"] == 1.
    assert old["evaluation"] == new["evaluation"]
    assert old["resume_critic_warmup_updates"] == new["resume_critic_warmup_updates"] == 10


def test_exact_named_plan_worker_and_domain_resume_are_admitted(plans, workers):
    source, target = plans
    receipt = campaign.validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
    assert receipt["schema"] == "task-plan-revision-v19"
    assert receipt["consumption"] == "preserved_no_refund" and receipt["task_clock"] == "restored"
    assert "domain_reopening" not in receipt
    campaign.validate_worker_revision(*workers)
    result = campaign.validate_task_resume(*workers)
    assert result["task_clock"] == "restored" and result["actor_critic_optimizer"] == "preserved"


@pytest.mark.parametrize("section,path,value", [
    ("worker", ("recovery_training", "prefix_curriculum", "origin_actor_update"), 0),
    ("worker", ("recovery_training", "prefix_curriculum", "stable_seconds"), .5),
    ("worker", ("recovery_training", "prefix_curriculum", "goal_bridge"), "unchecked-release"),
    ("worker", ("resume_critic_warmup_updates",), 0),
    ("worker", ("learning_rate",), .0002),
    ("worker", ("recovery_hold_cost", "leg", "broad_weight"), .5),
    ("worker", ("task_campaign", "revision_migration", "source_contract_sha256"), "0" * 64),
    ("plan", ("domain_completion", "schedule", "end_actor_updates", "recovery"), 20000),
    ("plan", ("total_ceiling",), 131000),
])
def test_unreviewed_threshold_origin_optimizer_reward_or_budget_change_rejected(plans, workers, section, path, value):
    pair = deepcopy(workers if section == "worker" else plans)
    target = pair[1]; cursor = target
    for key in path[:-1]: cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        if section == "worker": campaign.validate_worker_revision(*pair)
        else: campaign.validate_plan_migration(*pair, pair[1]["revision_migration"]["run_directory"])


def test_wrong_predecessor_and_run_identity_cannot_migrate(plans, workers):
    old = json.loads((ROOT / "contracts/v6_task_campaign_v15.json").read_text())
    target = deepcopy(plans[1])
    with pytest.raises(ValueError, match="named consecutive"):
        campaign.validate_plan_migration(old, target, target["revision_migration"]["run_directory"])
    with pytest.raises(ValueError, match="run identity"):
        campaign.validate_plan_migration(*plans, "/tmp/a-new-campaign")
    old, new = deepcopy(workers); old["dense_tracking"]["height"]["broad_weight"] += .1
    with pytest.raises(ValueError, match="exact deployed"):
        campaign.validate_worker_revision(old, new)


def test_new_reward_uses_existing_ten_critic_adaptations_then_resume_remaining(workers):
    old, new = workers
    infos = dict(contract_sha256=recipe.SOURCE_WORKER_SHA, successful_updates_total=30000,
        critic_warmup_remaining=0, task_campaign=deepcopy(old["task_campaign"]))
    target_sha = recipe.digest(ROOT / recipe.WORKER_FILE)
    assert resume_critic_warmup(new, infos, contract_sha256=target_sha, task_change=True) == 10
    for remaining in (10, 7, 1, 0):
        same = dict(infos, contract_sha256=target_sha, critic_warmup_remaining=remaining)
        assert resume_critic_warmup(new, same, contract_sha256=target_sha, task_change=True) == remaining
    assert resume_critic_warmup(old, infos, contract_sha256=recipe.SOURCE_WORKER_SHA, task_change=True) == 0


def test_ten_critic_only_updates_are_charged_without_advancing_actor_then_actor_is_charged(plans):
    _, plan = plans
    records = {"flat/attempt_old/train": dict(domain="flat", consumed_updates=14000, actor_updates=13000),
               "recovery/attempt_old/train": dict(domain="recovery", consumed_updates=2858, actor_updates=2828)}
    before = campaign.account_updates(plan, records)
    critic = campaign.update_attempt(plan, records, "recovery/attempt_next/train", "recovery",
        dict(successful_updates=10, attempted_updates=10, actor_updates_in_block=0))
    adapted = campaign.account_updates(plan, critic)
    assert adapted["consumed_updates"] == before["consumed_updates"] + 10
    assert adapted["actor_updates_consumed"] == before["actor_updates_consumed"]
    first_actor = campaign.update_attempt(plan, critic, "recovery/attempt_next/train", "recovery",
        dict(successful_updates=11, attempted_updates=11, actor_updates_in_block=1))
    after = campaign.account_updates(plan, first_actor)
    assert after["consumed_updates"] == before["consumed_updates"] + 11
    assert after["actor_updates_consumed"] == before["actor_updates_consumed"] + 1
    with pytest.raises(ValueError, match="backwards"):
        campaign.update_attempt(plan, first_actor, "recovery/attempt_next/train", "recovery",
            dict(successful_updates=1, actor_updates_in_block=1))


def test_fixed_exam_recurses_to_exact_v18_criteria_and_retains_disabled_hold_object(workers):
    old, new = map(fixed_suite_contract, workers)
    for key in ("evaluation", "scene_groups", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions",
                "new_asset_curriculum", "dense_tracking", "recovery_training", "recovery_takeover",
                "ordinary_posture_curriculum"):
        assert old.get(key) == new.get(key), key
    assert new["recovery_hold_cost"] == dict(recipe.HOLD_SETTINGS, enabled=False)
    assert "prefix_curriculum" not in new["recovery_training"]
    assert new["evaluation_exact_cases"] is True


def test_source_audit_includes_new_recipe_real_gates_rewards_and_both_independent_designs(workers):
    files = scut_curriculum_source_files(workers[1], ROOT)
    assert len(files) == len(set(files))
    for file in [recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE, recipe.SOURCE_DESIGN_FILE, recipe.DESIGN_FILE,
                 "src/wheeled_tasks/chassis/recovery_quality_revision.py", "src/wheeled_tasks/chassis/rewards.py",
                 "src/wheeled_tasks/chassis/env.py", "src/wheeled_tasks/chassis/recovery.py"]:
        assert file in files and (ROOT / file).is_file()


def test_actual_runtime_setting_omission_or_gate_binding_mismatch_rejected(workers):
    from wheeled_tasks.chassis.preflight import validate_runtime_state
    design = json.loads((ROOT / recipe.DESIGN_FILE).read_text())
    # Use the disabled fixed role to avoid manufacturing native ordinary birth
    # observations. All unchanged runtime fields still execute the real validator.
    expected = deepcopy(design["stages"]["recovery"]["roles"]["quick"]["runtime"])
    if expected.get("seed") == {"source": "effective_seed"}: expected["seed"] = 290929
    actual = deepcopy(expected)
    assert validate_runtime_state(expected, actual)["status"] == "runtime_configuration_passed"
    del actual["recovery_hold_cost"]
    with pytest.raises(ValueError, match="hold"):
        validate_runtime_state(expected, actual)
    actual = deepcopy(expected); actual["modules"]["recovery_hold_cost"] = True
    with pytest.raises(ValueError, match="hold"):
        validate_runtime_state(expected, actual)


def test_all_six_domains_and_five_roles_have_real_static_preflight(plans):
    from run_task_campaign import validate_contracts
    receipts = validate_contracts(plans[1])
    assert len(receipts) == 30 and all(r["status"] == "static_passed" for r in receipts)


@pytest.mark.parametrize("key", ["wheel_radius_m", "wheel_rim_gate_m_s", "leg_gate_rad_s", "gyro_gate_rad_s"])
def test_runtime_native_gate_binding_is_complete_and_rejects_real_gate_width_changes(key):
    from wheeled_tasks.chassis.preflight import validate_runtime_state
    design = json.loads((ROOT / recipe.DESIGN_FILE).read_text())
    expected = deepcopy(design["stages"]["recovery"]["roles"]["train"]["runtime"])
    # This isolates the observed hold mapping. Genuine ordinary-birth runtime
    # observations remain independently mandatory in the full native preflight.
    expected.pop("ordinary_posture_curriculum")
    expected["modules"].pop("ordinary_posture_curriculum")
    if expected.get("seed") == {"source": "effective_seed"}: expected["seed"] = 290927
    actual = deepcopy(expected)
    assert validate_runtime_state(expected, actual)["status"] == "runtime_configuration_passed"
    actual["recovery_hold_cost_gate_binding"][key] *= 1.1
    with pytest.raises(ValueError, match="gate_binding"):
        validate_runtime_state(expected, actual)
    actual = deepcopy(expected); del actual["recovery_hold_cost_gate_binding"]
    with pytest.raises(ValueError, match="gate_binding"):
        validate_runtime_state(expected, actual)


@pytest.mark.parametrize("report_behind_claim", [False, True])
def test_v19_account_restart_and_claim_before_report_crash_preserve_all_records_no_reopen(
        tmp_path, monkeypatch, plans, report_behind_claim):
    from types import SimpleNamespace
    import run_task_campaign as runner
    old, new = plans
    live = runner.TaskCampaign.__new__(runner.TaskCampaign)
    live.root = tmp_path / "train"; live.root.mkdir()
    live.claim = tmp_path / "account.json"
    live.args = SimpleNamespace(contract=ROOT / recipe.PLAN_FILE)
    live.plan = deepcopy(new)
    records = {"flat/attempt_old/train": dict(domain="flat", consumed_updates=14000, actor_updates=13000),
        "recovery/attempt_old/train": dict(domain="recovery", consumed_updates=2858, actor_updates=2828)}
    account = dict(campaign_id=campaign.CAMPAIGN_ID, run_directory=str(live.root),
        contract_sha256=recipe.digest(ROOT / recipe.PLAN_FILE), attempts=deepcopy(records))
    live.claim.write_text(json.dumps(account))
    domains = {row["name"]: dict(status="domain_training", inherited_actor_origin=2294 if row["name"] == "recovery" else 0)
               for row in old["domains"]}
    state = dict(contract_sha256=recipe.SOURCE_PLAN_SHA if report_behind_claim else account["contract_sha256"],
                 domains=domains, consumed_updates=17858)
    (live.root / "campaign_state.json").write_text(json.dumps(state)); live.report = {}
    # Run-root identity is tested by the real migration fixture. Here a scratch
    # account replaces only that external path, to exercise durable replay.
    real_migration = campaign.validate_plan_migration(old, new, old["revision_migration"]["run_directory"])
    expected = dict(real_migration, run_directory=str(live.root))
    monkeypatch.setattr(runner, "validate_plan_migration", lambda source, target, root: deepcopy(expected)
                        if source == old and target == new and root == live.root else (_ for _ in ()).throw(ValueError("pair")))
    revision = live.root / "revisions/v18_to_v19_account_migration.json"
    revision.parent.mkdir(); revision.write_text(json.dumps(expected))
    ownership_checks = []
    live.assert_no_live_workers = lambda: ownership_checks.append(True)
    live.persist = lambda: None
    live.open_account()
    assert live.records == records and live.report["domains"] == domains and ownership_checks == [True]
    assert live.report["contract_sha256"] == account["contract_sha256"]
    assert all("reopening" not in row for row in live.report["domains"].values())
    if report_behind_claim:
        replay = json.loads(revision.read_text())
        assert replay["attempts"] == records
        assert replay["consumed_updates"] == campaign.account_updates(new, records)["consumed_updates"]
        assert replay["actor_updates_consumed"] == campaign.account_updates(new, records)["actor_updates_consumed"]


def test_freeze_authoring_cannot_overwrite_published_contracts():
    before = {file: recipe.digest(ROOT / file) for file in (recipe.DESIGN_FILE, recipe.WORKER_FILE, recipe.PLAN_FILE, recipe.MANIFEST_FILE)}
    with pytest.raises(ValueError, match="cannot overwrite"):
        recipe.freeze_contracts(ROOT)
    assert before == {file: recipe.digest(ROOT / file) for file in before}


@pytest.mark.parametrize("wrong_header", [False, True])
def test_remote_owner_accepts_v19_only_with_archive_commit_and_unchanged_command(
        tmp_path, monkeypatch, plans, wrong_header):
    import chassis_remote_job
    old, new = tmp_path / "old", tmp_path / "new"
    run = tmp_path / "run"; run.mkdir()
    paths = [Path(recipe.SOURCE_PLAN_FILE), Path(recipe.PLAN_FILE), Path("scripts/run_task_campaign.py")]
    for root, role in ((old, "old"), (new, "new")):
        for path in paths:
            f = root / path; f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes((ROOT / path).read_bytes() if path.suffix == ".json" else b"# frozen runner\n")
    original = ["/env/python", str(old / "scripts/run_task_campaign.py"), "--contract", str(old / recipe.SOURCE_PLAN_FILE),
        "--num-envs", "4096", "--run-dir", str(run / "train")]
    command = list(original); command[1] = str(new / "scripts/run_task_campaign.py"); command[3] = str(new / recipe.PLAN_FILE)
    previous = {"command": original}; (run / "job.json").write_text(json.dumps(previous))
    archive = tmp_path / "source.tar.gz"; commit = "1" * 40
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT,
                      pax_headers={} if wrong_header else {"comment": commit}) as stream:
        for path in paths:
            data = (new / path).read_bytes(); member = tarfile.TarInfo(str(path)); member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    receipt = dict(schema="task-plan-revision-v19", previous_job_sha256=chassis_remote_job.sha(run / "job.json"),
        command=command, source_commit=commit, old_source_root=str(old), new_source_root=str(new),
        new_contract_relative_path=recipe.PLAN_FILE, old_contract_sha256=chassis_remote_job.sha(old / recipe.SOURCE_PLAN_FILE),
        new_contract_sha256=chassis_remote_job.sha(new / recipe.PLAN_FILE), source_archive=str(archive),
        source_archive_sha256=chassis_remote_job.sha(archive))
    receipt_path = tmp_path / "revision.json"; receipt_path.write_text(json.dumps(receipt))
    calls = []
    def already_independently_checked(source, target, directory):
        assert (source, target) == plans
        calls.append(directory)
    # This fixture isolates wrapper/archive ownership. The preceding real
    # migration test independently enforces the one absolute campaign binding.
    monkeypatch.setattr(campaign, "validate_plan_migration", already_independently_checked)
    if wrong_header:
        with pytest.raises(ValueError, match="archive commit"):
            chassis_remote_job.validate_source_revision(run, command, previous, receipt_path)
        assert calls == []
    else:
        assert chassis_remote_job.validate_source_revision(run, command, previous, receipt_path) == receipt
        assert calls == [run / "train"]
        changed = list(command); changed[changed.index("--num-envs") + 1] = "20000"
        receipt["command"] = changed; receipt_path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="only runner and contract"):
            chassis_remote_job.validate_source_revision(run, changed, previous, receipt_path)
