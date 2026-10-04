"""Authenticate the bounded v12→v14 repair and preserve charged learning state."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis import locomotion_quality_revision as revision
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.locomotion_difficulty import difficulty_settings
from wheeled_tasks.chassis.nonstandard_start_revision import reopen_completed_domains
from wheeled_tasks.chassis.preflight import validate_design_contract, validate_runtime_state

from test_task_campaign_nonstandard_start_revision import (
    checkpoint, reopening as reopening_v13,
)
import run_task_campaign as runner
from chassis_remote_job import sha, validate_source_revision
from train_chassis import resume_critic_warmup

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


SOURCE, TARGET = read(revision.SOURCE_PLAN_FILE), read(revision.PLAN_FILE)


def worker(domain):
    return read(revision.worker_file(domain))


@pytest.fixture
def reopening(reopening_v13):
    reopening_v13.plan = TARGET
    return reopening_v13


def test_frozen_repair_preserves_four_workers_parent_tree_budgets_interfaces_and_fixed_cases():
    receipt = campaign.validate_plan_migration(SOURCE, TARGET, TARGET["revision_migration"]["run_directory"])
    assert receipt["consumption"] == "preserved_no_refund"
    assert TARGET["domain_completion"]["schedule"]["end_actor_updates"] == revision.ENDPOINTS
    assert TARGET["total_ceiling"] == 121000 and TARGET["target_num_envs"] == 4096
    for before, after in zip(SOURCE["domains"], TARGET["domains"], strict=True):
        assert after["parent_domain"] == before["parent_domain"] and after["updates"] == 20000
        if before["name"] not in revision.ORIGINS:
            assert after == before
    for domain in revision.ORIGINS:
        before, after = read(revision.SOURCES[domain][0]), worker(domain)
        for key in ("evaluation", "v5_control", "actuator_response", "signal_delay", "command_transport",
                    "termination_event_cost", "command_slew", "learning_rate", "learning_rate_schedule"):
            assert after[key] == before[key]
        assert (after["physics_dt"], after["policy_dt"], after["actor_dim"], after["critic_dim"]) == (.005, .02, 35, 81)
        assert campaign.validate_task_resume(before, after)["task_clock"] == "restored"
        added = after["scene_groups"][len(before["scene_groups"]):]
        assert len(added) == 6 and sum(g["fraction"] for g in added) == pytest.approx(.05)
        assert after["behavior_pool_fractions"]["stand_anchors"] == before["behavior_pool_fractions"]["stand_anchors"] * .95
        for name, spec in before["skill_specs"].items():
            assert after["skill_specs"][name] == spec
    assert len(runner.validate_contracts(TARGET)) == 30


def test_speed_restores_actor_clock_difficulty_and_full_domain_release_with_finite_dwell():
    speed, terrain = worker("speed"), worker("terrain")
    assert speed["locomotion_difficulty"] == difficulty_settings(3563)
    assert "locomotion_difficulty" not in terrain
    for suffix in ("", "__height_train"):
        for axis in ("translate", "combined"):
            spec = speed["skill_specs"]["release_stop_" + axis + suffix]
            assert spec["command"][0] == 5.0
            assert spec["release_stop_magnitude_ranges"]["vx_m_s"] == [.5, 5.0]
            assert spec["release_stop_magnitude_ranges"]["yaw_rad_s"] == [1.0, 6.0]
        yaw = speed["skill_specs"]["release_stop_yaw" + suffix]
        assert yaw["command"][:2] == [0.0, 6.0]
        assert yaw["release_stop_magnitude_ranges"]["vx_m_s"] == [.5, 3.0]
    release = speed["release_stop"]
    longest = release["initial_wait_seconds"][1] + release["drive_seconds"][1] + max(
        release["zero_request_dwell_seconds"][1], 5.0 / 1.5 + release["stationary_warmup_seconds"] + release["stationary_window_seconds"])
    assert longest <= 32.0
    assert release["request_semantics"] == "abrupt_zero_target_existing_command_slew"


@pytest.mark.parametrize("domain", ["speed", "terrain"])
def test_each_independent_role_keeps_new_rewards_and_training_teachers_are_bypassed_in_fixed_eval(domain):
    config, design = worker(domain), read(revision.design_file(domain))
    assert design == revision.revised_design(domain)
    for role, row in design["stages"][domain]["roles"].items():
        actual, runtime = row["contract"], row["runtime"]
        assert actual["dense_tracking"]["stand_still_lin_vel"] == revision.STATIONARY
        assert actual["dense_tracking"]["locomotion_posture"] == revision.POSTURE
        assert runtime["modules"]["stand_still_lin_vel"] is True
        assert runtime["modules"]["locomotion_posture_floor"] is True
        assert runtime["locomotion_quality_rewards"] == revision.quality_runtime_expectation(actual)
        assert actual["nonstandard_start"]["enabled"] is (role == "train")
        assert actual["release_stop"]["enabled"] is (role == "train")
        if domain == "speed":
            assert runtime["locomotion_difficulty"] == dict(enabled=role == "train", settings=difficulty_settings(3563))
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    exact = fixed_suite_contract(config)
    assert exact["dense_tracking"] == config["dense_tracking"]
    assert exact["nonstandard_start"]["enabled"] is False and exact["release_stop"]["enabled"] is False
    if domain == "speed":
        assert exact["locomotion_difficulty"] == config["locomotion_difficulty"]
    # A configuration receipt does not certify real numerical reward payments.
    expected = design["stages"][domain]["roles"]["train"]["runtime"]
    expected["seed"] = 290927
    assert validate_runtime_state(expected, deepcopy(expected))["status"] == "runtime_configuration_passed"


@pytest.mark.parametrize("domain", ["speed", "terrain"])
@pytest.mark.parametrize("fault", ["missing_l1", "missing_posture", "missing_runtime", "runtime_disabled", "missing_grade", "wrong_origin", "wrong_source", "changed_delay"])
def test_worker_revision_rejects_missing_repair_fields_and_unreviewed_source_changes(domain, fault):
    source, target = read(revision.SOURCES[domain][0]), worker(domain)
    if fault == "missing_l1":
        target["dense_tracking"].pop("stand_still_lin_vel")
    elif fault == "missing_posture":
        target["dense_tracking"].pop("locomotion_posture")
    elif fault == "missing_runtime" or fault == "runtime_disabled":
        design = read(revision.design_file(domain))
        row = design["stages"][domain]["roles"]["train"]
        if fault == "missing_runtime":
            row["runtime"].pop("locomotion_quality_rewards")
        else:
            row["runtime"]["modules"]["stand_still_lin_vel"] = False
        control = read(target["control_math_source"])
        with pytest.raises(ValueError):
            validate_design_contract(target, design, control, stage=domain, role="train", seed=290927, num_envs=4096)
        return
    elif fault == "missing_grade":
        if domain == "terrain":
            target["locomotion_difficulty"] = difficulty_settings(1186)
        else:
            target.pop("locomotion_difficulty")
    elif fault == "wrong_origin":
        target["task_campaign"]["revision_migration"]["origin_actor_update"] += 1
    elif fault == "wrong_source":
        target["task_campaign"]["revision_migration"]["source_contract_sha256"] = "0" * 64
    else:
        target["signal_delay"]["enabled"] = False
    with pytest.raises(ValueError):
        campaign.validate_worker_revision(source, target)


@pytest.mark.parametrize("fault", ["missing_grade_runtime", "wrong_grade_enabled", "missing_grade_design"])
def test_speed_grade_design_is_an_independent_required_runtime_input(fault):
    config, design = worker("speed"), read(revision.design_file("speed"))
    row = design["stages"]["speed"]["roles"]["train"]
    if fault == "missing_grade_runtime":
        row["runtime"].pop("locomotion_difficulty")
    elif fault == "wrong_grade_enabled":
        row["runtime"]["locomotion_difficulty"]["enabled"] = False
    else:
        row["contract"].pop("locomotion_difficulty")
    with pytest.raises(ValueError):
        validate_design_contract(config, design, read(config["control_math_source"]),
            stage="speed", role="train", seed=290927, num_envs=4096)


def test_reopening_is_idempotent_preserves_history_and_charges_adaptation_once(reopening):
    job = reopening
    before, records = deepcopy(job.report), deepcopy(job.records)
    result = reopen_completed_domains(job.report, job.records, SOURCE, TARGET, job.root)
    assert result["schema"] == "task-domain-reopening-v14"
    assert job.records == records
    assert campaign.account_updates(SOURCE, records) == campaign.account_updates(TARGET, records)
    for domain in revision.ORIGINS:
        record = job.report["domains"][domain]
        assert record["completion_history"] == [dict(plan_revision="v12", record=before["domains"][domain])]
        assert record["reopening"]["schema"] == "task-domain-reopen-v14"
        assert job.learning_source(next(r for r in TARGET["domains"] if r["name"] == domain)) == Path(record["candidate_checkpoint"])
        target = worker(domain)
        infos = dict(contract_sha256=revision.SOURCES[domain][1], task_campaign=dict(domain=domain),
            task_actor_update=revision.ORIGINS[domain], critic_warmup_remaining=0, successful_updates_total=23868)
        target_sha = campaign.digest(ROOT / revision.worker_file(domain))
        assert resume_critic_warmup(target, infos, contract_sha256=target_sha, task_change=True) == 10
        infos.update(contract_sha256=target_sha, critic_warmup_remaining=4)
        assert resume_critic_warmup(target, infos, contract_sha256=target_sha, task_change=True) == 4
        infos["critic_warmup_remaining"] = 0
        assert resume_critic_warmup(target, infos, contract_sha256=target_sha, task_change=True) == 0
    once = deepcopy(job.report)
    reopen_completed_domains(job.report, job.records, SOURCE, TARGET, job.root)
    assert job.report == once and job.report["domains"]["jump"] == before["domains"]["jump"]


def test_durable_account_migration_reopens_v14_once_without_consumption_refund(reopening, monkeypatch):
    job = reopening
    job.args = SimpleNamespace(contract=ROOT / revision.PLAN_FILE, device="cpu")
    job.claim = job.root / "shared_budget/account.json"
    job._session_started = None
    old_ledger = dict(campaign_id=campaign.CAMPAIGN_ID, run_directory=str(job.root),
        contract_sha256=revision.SOURCE_PLAN_SHA, attempts=deepcopy(job.records),
        **campaign.account_updates(SOURCE, job.records))
    runner.write_json(job.claim, old_ledger)
    runner.write_json(job.root / "campaign_state.json", job.report)
    def validate(before, after, root):
        assert before == SOURCE and after == TARGET and root == job.root
        return dict(schema="task-plan-revision-v14", source_plan_sha256=revision.SOURCE_PLAN_SHA,
            target_plan_sha256=campaign.digest(job.args.contract), consumption="preserved_no_refund")
    monkeypatch.setattr(runner, "validate_plan_migration", validate)
    monkeypatch.setattr(job, "assert_no_live_workers", lambda: None)
    job.persist = runner.TaskCampaign.persist.__get__(job)
    job.open_account()
    ledger = json.loads(job.claim.read_text())
    assert ledger["attempts"] == old_ledger["attempts"] and ledger["consumed_updates"] == 23868
    assert ledger["contract_sha256"] == campaign.digest(job.args.contract)
    assert job.report["revision_migration"]["domain_reopening"]["schema"] == "task-domain-reopening-v14"
    once = deepcopy(job.report)
    job.open_account()
    assert job.report["domains"] == once["domains"]
    assert job.records == old_ledger["attempts"]


def test_normal_runner_continues_two_own_domain_sources_without_assessment_or_new_budget(reopening, monkeypatch):
    import torch
    job = reopening
    reopen_completed_domains(job.report, job.records, SOURCE, TARGET, job.root)
    job.persist = lambda: job.report.update(campaign.account_updates(TARGET, job.records))
    requested, reviewed = [], []
    monkeypatch.setattr(job, "review_request", lambda *args: reviewed.append(job.active["name"]))
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: pytest.fail("Teaching cannot wait for fixed assessment"))
    def execute(command, log, directory):
        domain = directory.parents[1].name
        row = next(row for row in TARGET["domains"] if row["name"] == domain)
        source = Path(command[command.index("--resume") + 1])
        count = int(command[command.index("--updates") + 1])
        assert source.parent.parent.parent.name == domain
        assert count == revision.ENDPOINTS[domain] - revision.ORIGINS[domain] + 10
        assert int(command[command.index("--task-actor-update") + 1]) == revision.ORIGINS[domain]
        requested.append((domain, count))
        directory.mkdir()
        final = directory / "model_final.pt"
        shutil.copyfile(ROOT / row["contract"], final.with_suffix(".contract.json"))
        torch.save(dict(infos=dict(task_actor_update=revision.ENDPOINTS[domain])), final)
        runner.write_json(final.with_suffix(".checkpoint.json"), dict(schema="chassis-learning-checkpoint-v1",
            sealed=True, checkpoint_sha256=campaign.digest(final), contract_sha256=row["sha256"],
            learning_updates=job.report["consumed_updates"] + count, task_actor_update=revision.ENDPOINTS[domain]))
        runner.write_json(directory / "completion.json", dict(status="completed", successful_updates=count,
            actor_updates_in_block=count - 10))
        return 0
    monkeypatch.setattr(job, "execute", execute)
    for row in TARGET["domains"]:
        if row["name"] in revision.ORIGINS:
            job.run_domain(row)
    assert requested == [("speed", 1510), ("terrain", 1010)]
    assert reviewed == ["speed", "terrain"]
    assert job.report["consumed_updates"] == 23868 + 2520
    assert job.report["domains"]["jump"]["candidate_checkpoint"].endswith("jump/attempt_000/train/model_final.pt")
    for domain in revision.ORIGINS:
        assert job.report["domains"][domain]["status"] == "domain_schedule_finished_unassessed"
        assert job.report["domains"][domain]["domain_qualified"] is None


@pytest.mark.parametrize("domain", ["speed", "terrain"])
def test_native_worker_rejects_a_predecessor_with_the_wrong_actor_origin(tmp_path, domain):
    config = worker(domain)
    actor = revision.ORIGINS[domain] - 1
    path = checkpoint(tmp_path, domain, actor, revision.SOURCES[domain][0], 23868)
    seal = json.loads(path.with_suffix(".checkpoint.json").read_text())
    args = SimpleNamespace(resume_task_change=True, resume=path, num_envs=4096, seed=290927,
        learning_rate_scale=1., global_actor_update_offset=0, block_index=0, updates=100,
        consumed_updates=23868, task_actor_update=actor, stage_actor_update=seal["stage_actor_update"],
        actor_updates_consumed=seal["actor_updates_consumed"])
    with pytest.raises(ValueError, match="sealed predecessor actor clock"):
        campaign.validate_worker(config, args)


@pytest.mark.parametrize("fault", ["endpoint", "parent", "reward", "quota", "origin", "missing_grade", "l1", "posture", "parking_range", "other_worker", "budget"])
def test_resealed_candidate_does_not_authorize_extra_changes(tmp_path, monkeypatch, fault):
    shutil.copytree(ROOT / "contracts", tmp_path / "contracts")
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    target, config = deepcopy(TARGET), worker("speed")
    design, manifest = read(revision.design_file("speed")), read(revision.MANIFEST_FILE)
    if fault == "endpoint":
        target["domain_completion"]["schedule"]["end_actor_updates"]["speed"] += 1
    elif fault == "parent":
        target["domains"][2]["parent_domain"] = "recovery"
    elif fault == "reward":
        config["termination_event_cost"] += 1
    elif fault == "quota":
        config["scene_groups"][0]["fraction"] *= 2
    elif fault == "origin":
        config["task_campaign"]["revision_migration"]["origin_actor_update"] += 1
    elif fault == "missing_grade":
        config.pop("locomotion_difficulty")
    elif fault == "l1":
        config["dense_tracking"]["stand_still_lin_vel"].pop("exclude_world_reference_motion")
    elif fault == "posture":
        config["dense_tracking"]["locomotion_posture"]["roll_scale_floor"] = 3.0
    elif fault == "parking_range":
        config["skill_specs"]["release_stop_translate"]["release_stop_magnitude_ranges"]["vx_m_s"] = [.5, 6.]
    elif fault == "budget":
        target["total_ceiling"] += 1
    else:
        target["domains"][4]["sha256"] = "0" * 64
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value, indent=2) + "\n")
        return campaign.digest(tmp_path / name)
    config["design_preflight"]["sha256"] = write(revision.design_file("speed"), design)
    target["domains"][2]["sha256"] = write(revision.worker_file("speed"), config)
    manifest["target_plan_sha256"] = write(revision.PLAN_FILE, target)
    for pin, row in zip(manifest["workers"], target["domains"], strict=True):
        pin["target_sha256"] = row["sha256"]
    write(revision.MANIFEST_FILE, manifest)
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(SOURCE, target, manifest["run_directory"])


def test_v14_remote_receipt_authenticates_deployed_v12_and_rejects_environment_resize(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    for directory, file, plan in ((old, revision.SOURCE_PLAN_FILE, SOURCE), (new, revision.PLAN_FILE, TARGET)):
        (directory / "contracts").mkdir(parents=True)
        (directory / file).write_text(json.dumps(plan))
        (directory / "scripts").mkdir()
        (directory / "scripts/run_task_campaign.py").write_text("# Not executed.\n")
    command = [sys.executable, "-B", str(old / "scripts/run_task_campaign.py"), "--contract",
        str(old / revision.SOURCE_PLAN_FILE), "--num-envs", "4096", "--device", "cuda:0", "--research",
        "--max-runtime-seconds", "15552000", "--run-dir", str(tmp_path / "train")]
    previous = dict(pid=2147483647, command=command)
    (tmp_path / "job.json").write_text(json.dumps(previous))
    revised = list(command)
    revised[2], revised[4] = str(new / "scripts/run_task_campaign.py"), str(new / revision.PLAN_FILE)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted(new.rglob("*")):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(new)), recursive=False)
    receipt = dict(schema="task-plan-revision-v14", previous_job_sha256=sha(tmp_path / "job.json"),
        command=revised, source_commit="a" * 40, old_source_root=str(old), new_source_root=str(new),
        new_contract_relative_path=revision.PLAN_FILE, old_contract_sha256=sha(old / revision.SOURCE_PLAN_FILE),
        new_contract_sha256=sha(new / revision.PLAN_FILE), source_archive=str(archive), source_archive_sha256=sha(archive))
    path = tmp_path / "revision.json"
    path.write_text(json.dumps(receipt))
    calls = []
    def validate(before, after, root):
        assert before == SOURCE and after == TARGET and root == tmp_path / "train"
        calls.append(True)
    monkeypatch.setattr(campaign, "validate_plan_migration", validate)
    assert validate_source_revision(tmp_path, revised, previous, path) == receipt and calls == [True]
    revised[revised.index("--num-envs") + 1] = "20000"
    receipt["command"] = revised
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="only runner and contract"):
        validate_source_revision(tmp_path, revised, previous, path)
