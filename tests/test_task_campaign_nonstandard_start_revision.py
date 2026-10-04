"""A strict ordinary restart curriculum must really reopen charged branches."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis import nonstandard_start_revision as revision
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner
from chassis_remote_job import sha, validate_source_revision
from train_chassis import configure_recovery_rollout, nonstandard_start_source_files, resume_critic_warmup
from test_task_campaign_v2 import continuous_job


def read(name):
    return json.loads((ROOT / name).read_text())


SOURCE = read(revision.SOURCE_PLAN_FILE)
TARGET = read(revision.PLAN_FILE)


def worker(domain):
    return read(revision.worker_file(domain))


def checkpoint(root, domain, actor, config_file, global_updates):
    path = root / domain / "attempt_000/train/model_final.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = path.with_suffix(".contract.json")
    shutil.copyfile(ROOT / config_file, contract)
    config = json.loads(contract.read_text())
    infos = dict(contract_sha256=campaign.digest(contract), task_campaign=config["task_campaign"],
        task_actor_update=actor, stage_actor_update=13000 + actor, successful_updates_total=global_updates,
        actor_updates_consumed=global_updates - 100, critic_warmup_remaining=0)
    torch.save(dict(infos=infos), path)
    seal = dict(schema="chassis-learning-checkpoint-v1", sealed=True,
        checkpoint_sha256=campaign.digest(path), contract_sha256=campaign.digest(contract),
        learning_updates=global_updates, stage_actor_update=infos["stage_actor_update"],
        actor_updates_consumed=infos["actor_updates_consumed"], task_actor_update=actor)
    runner.write_json(path.with_suffix(".checkpoint.json"), seal)
    assert checkpoint_update_count(path) == global_updates
    return path


@pytest.fixture
def reopening(tmp_path):
    job = continuous_job(tmp_path, TARGET)
    job.child = None
    job.records = {f"{name}/attempt_000/train": dict(domain=name, consumed_updates=count, actor_updates=actor)
        for name, count, actor in (("flat", 13020, 13000), ("recovery", 2314, 2294),
            ("speed", 3583, 3563), ("terrain", 1196, 1186), ("jump", 2755, 2735))}
    job.report.update(campaign.account_updates(TARGET, job.records))
    for domain, origin in revision.ORIGINS.items():
        path = checkpoint(tmp_path, domain, origin, revision.SOURCES[domain][0],
            19837 if domain == "speed" else 21033)
        job.report["domains"][domain] = dict(status="domain_schedule_finished_unassessed", domain_qualified=None,
            candidate_checkpoint=str(path), final_checkpoint=str(path), candidate_checkpoint_sha256=campaign.digest(path),
            parent_domain="flat", consumed_updates=job.report["domain_consumed_updates"][domain],
            schedule_completion=dict(actual_actor_update=origin, end_actor_update=origin),
            finish_review=dict(token=f"old-{domain}-review", reason="previous completed observation"),
            assessment_status="not_run")
    jump = checkpoint(tmp_path, "jump", 2735, SOURCE["domains"][4]["contract"], 23868)
    job.report["domains"]["jump"] = dict(status="running", candidate_checkpoint=str(jump),
        candidate_checkpoint_sha256=campaign.digest(jump), domain_qualified=None)
    job.report["evaluations"] = dict(speed={"old_review": dict(file="immutable_old_evidence.json", sha256="a" * 64)})
    return job


def test_exact_v12_to_v13_patch_keeps_four_workers_parent_tree_rewards_and_fixed_cases():
    receipt = campaign.validate_plan_migration(SOURCE, TARGET, TARGET["revision_migration"]["run_directory"])
    assert receipt["consumption"] == "preserved_no_refund"
    assert TARGET["domain_completion"]["schedule"]["end_actor_updates"] == revision.ENDPOINTS
    assert TARGET["total_ceiling"] == 121000
    for before, after in zip(SOURCE["domains"], TARGET["domains"], strict=True):
        assert after["parent_domain"] == before["parent_domain"] and after["updates"] == 20000
        if after["name"] not in revision.ORIGINS:
            assert before == after
    for domain in revision.ORIGINS:
        before = read(revision.SOURCES[domain][0])
        after = worker(domain)
        expected = deepcopy(before)
        revision.apply_start_revision(expected)
        expected["design_preflight"] = deepcopy(after["design_preflight"])
        assert after == expected
        assert after["evaluation"] == before["evaluation"]
        for old, revised in zip(before["scene_groups"], after["scene_groups"], strict=False):
            assert {k: v for k, v in old.items() if k != "fraction"} == {k: v for k, v in revised.items() if k != "fraction"}
            assert revised["fraction"] == old["fraction"] * .95 > 0
        added = after["scene_groups"][len(before["scene_groups"]):]
        assert len(added) == 6 and sum(row["fraction"] for row in added) == pytest.approx(.05)
        assert all(after["behavior_pool_membership"][row["name"]] == "release_stop" for row in added)
        assert after["command_slew"] == before["command_slew"]
        assert after["v5_control"] == before["v5_control"]
        assert after["actuator_response"] == before["actuator_response"]
        assert after["physics_dt"] == .005 and after["policy_dt"] == .02
        assert (after["actor_dim"], after["critic_dim"]) == (35, 81)
        assert campaign.validate_task_resume(before, after)["task_clock"] == "restored"
    assert len(runner.validate_contracts(TARGET)) == 30


@pytest.mark.parametrize("domain,ranges,terrains", [
    ("speed", dict(vx_m_s=[.5, 3.], yaw_rad_s=[1., 6.]), ["flat"]),
    ("terrain", dict(vx_m_s=[.3, 1.5], yaw_rad_s=[.5, 2.]), ["flat", "rough", "material", "slope"]),
])
def test_release_stop_pool_preserves_old_anchors_and_covers_axes_heights_with_authorized_ranges(domain, ranges, terrains):
    source_row = next(row for row in SOURCE["domains"] if row["name"] == domain)
    target_row = next(row for row in TARGET["domains"] if row["name"] == domain)
    config = worker(domain)
    assert target_row["sampling_pools"] == revision.plan_sampling_pools(domain)
    for old, revised in zip(source_row["sampling_pools"], target_row["sampling_pools"], strict=False):
        expected = deepcopy(old)
        expected["fraction"] *= .95
        assert revised == expected and revised["fraction"] > 0
    pool = target_row["sampling_pools"][-1]
    assert pool["name"] == "release_stop" and pool["fraction"] == .05
    assert len(pool["skills"]) == 6
    assert set(pool["skills"]) == {"release_stop_" + axis + suffix
        for axis in ("translate", "yaw", "combined") for suffix in ("", "__height_train")}
    groups = {row["name"]: row for row in config["scene_groups"]}
    for name in pool["skills"]:
        spec = config["skill_specs"][name]
        assert spec["release_stop_magnitude_ranges"] == ranges
        assert spec["sample_vx_sign"] and spec["sample_yaw_sign"]
        assert groups[name]["terrain"] == terrains and groups[name]["fraction"] == pytest.approx(.05 / 6)
        assert ("height_sampling" in spec) is name.endswith("__height_train")
        if domain == "terrain":
            assert spec["terrain_limits"]["slope_deg"] <= 8.
            assert spec["terrain_limits"]["roughness_m"] <= .02
    assert config["release_stop"]["stationary_window_seconds"] == 10.
    assert config["release_stop"]["stationary_warmup_seconds"] == 2.
    assert config["release_stop"]["unchanged_slew_m_s2_rad_s2"] == [1.5, 4.]


@pytest.mark.parametrize("domain", ["speed", "terrain"])
@pytest.mark.parametrize("fault,reason", [
    ("missing_settings", "independent full settings"),
    ("incomplete_settings", "Release-stop settings"),
    ("missing_runtime", "design.runtime.release_stop"),
    ("missing_module", "design.runtime.modules.release_stop"),
    ("runtime_1khz", "design.runtime.release_stop.policy_dt"),
    ("physics_1khz", "reference.physics_dt"),
    ("vx_slew", "1.5/4"),
    ("yaw_slew", "1.5/4"),
])
def test_independent_release_preflight_rejects_missing_design_or_changed_physics_and_acceleration(domain, fault, reason):
    from wheeled_tasks.chassis.preflight import validate_design_contract
    config, design = worker(domain), read(revision.design_file(domain))
    row = design["stages"][domain]["roles"]["train"]
    control = read(config["control_math_source"])
    invocation = dict(stage=domain, role="train", num_envs=row["invocation"]["num_envs"],
        seed=row["invocation"]["seed"]["root_seed"])
    validate_design_contract(config, design, control, **invocation)
    if fault == "missing_settings":
        row["contract"].pop("release_stop")
    elif fault == "incomplete_settings":
        config["release_stop"].pop("zero_request_dwell_seconds")
        row["contract"]["release_stop"] = deepcopy(config["release_stop"])
    elif fault == "missing_runtime":
        row["runtime"].pop("release_stop")
    elif fault == "missing_module":
        row["runtime"]["modules"].pop("release_stop")
    elif fault == "runtime_1khz":
        row["runtime"]["release_stop"]["policy_dt"] = .001
    elif fault == "physics_1khz":
        config["physics_dt"] = row["contract"]["physics_dt"] = .001
    else:
        axis = "vx_m_s2" if fault == "vx_slew" else "yaw_rad_s2"
        config["command_slew"][axis] += .1
        row["contract"]["command_slew"] = deepcopy(config["command_slew"])
    with pytest.raises(ValueError, match=reason):
        validate_design_contract(config, design, control, **invocation)


@pytest.mark.parametrize("domain", ["speed", "terrain"])
def test_fixed_evaluator_restores_exact_prior_pool_metadata_and_disables_teachers(domain):
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    original = read(revision.SOURCES[domain][0])
    config = worker(domain)
    snapshot = deepcopy(config)
    actual, expected = fixed_suite_contract(config), fixed_suite_contract(original)
    assert config == snapshot
    assert actual["release_stop"]["enabled"] is False
    assert actual["nonstandard_start"]["enabled"] is False
    assert actual["behavior_pool_fractions"] == original["behavior_pool_fractions"]
    assert "release_stop" not in actual["behavior_pool_fractions"]
    for key in ("evaluation", "scene_groups", "episode_seconds", "evaluation_exact_cases"):
        assert actual[key] == expected[key]
    assert all(not row["name"].startswith("release_stop_") for row in actual["scene_groups"])


@pytest.mark.parametrize("domain", ["speed", "terrain"])
def test_startup_source_audit_pins_pose_bank_and_rollout_dependency(tmp_path, domain):
    config = worker(domain)
    sources = nonstandard_start_source_files(config)
    assert set(sources) == {"src/wheeled_tasks/chassis/nonstandard_start.py",
        "src/wheeled_tasks/chassis/nonstandard_start_revision.py", "src/wheeled_algo/recovery_rollout.py",
        "src/wheeled_tasks/chassis/release_stop.py",
        config["nonstandard_start"]["profiles_file"]}
    assert all((ROOT / path).is_file() for path in sources)
    assert nonstandard_start_source_files(read(revision.SOURCES[domain][0])) == []
    path = tmp_path / config["nonstandard_start"]["profiles_file"]
    path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / config["nonstandard_start"]["profiles_file"], path)
    with path.open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="source audit"):
        nonstandard_start_source_files(config, tmp_path)


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_startup_source_audit_rejects_missing_or_changed_release_teacher(fault):
    config = worker("speed")
    if fault == "missing":
        config.pop("release_stop")
    else:
        config["release_stop"]["unchanged_slew_m_s2_rad_s2"] = [2., 4.]
    with pytest.raises(ValueError, match="Release-stop settings"):
        nonstandard_start_source_files(config)


@pytest.mark.parametrize("recovery_enabled", [False, True])
def test_nonstandard_hold_attaches_actual_PPO_mask_without_recovery_dependency(recovery_enabled):
    from test_recovery_rollout_mask import algorithm, collect
    alg = algorithm()
    env = SimpleNamespace(cfg=dict(recovery_training=dict(enabled=recovery_enabled),
        nonstandard_start=dict(enabled=True)), num_envs=2)
    adapter = configure_recovery_rollout(alg, env)
    assert alg.update.__self__ is adapter
    mask = torch.ones(6, 2, dtype=torch.bool)
    mask[:2, 0] = False
    obs = collect(alg, mask)
    alg.compute_returns(obs)
    alg.update()
    assert adapter.metrics["valid_transitions"] == 10


@pytest.mark.parametrize("domain", ["speed", "terrain"])
@pytest.mark.parametrize("mode", ["predecessor", "wrong_origin", "same_contract_fault"])
def test_native_worker_origin_matches_completed_seal_and_fault_restores_later_clock(tmp_path, domain, mode):
    config = worker(domain)
    same = mode == "same_contract_fault"
    actor = revision.ORIGINS[domain] + (13 if same else -int(mode == "wrong_origin"))
    file = revision.worker_file(domain) if same else revision.SOURCES[domain][0]
    path = checkpoint(tmp_path, domain, actor, file, 23724)
    seal = json.loads(path.with_suffix(".checkpoint.json").read_text())
    args = SimpleNamespace(resume_task_change=True, resume=path, num_envs=4096, seed=290927,
        learning_rate_scale=1., global_actor_update_offset=0, block_index=0, updates=100,
        consumed_updates=23724, task_actor_update=actor, stage_actor_update=seal["stage_actor_update"],
        actor_updates_consumed=seal["actor_updates_consumed"])
    if mode == "wrong_origin":
        with pytest.raises(ValueError, match="sealed predecessor actor clock"):
            campaign.validate_worker(config, args)
    else:
        campaign.validate_worker(config, args)


@pytest.mark.parametrize("domain", ["speed", "terrain"])
def test_independent_roles_enable_training_only_and_charge_adaptation_once(domain):
    config = worker(domain)
    original = read(read(revision.SOURCES[domain][0])["design_preflight"]["file"])
    design = read(revision.design_file(domain))
    assert design == revision.revised_design(domain)
    for role, row in design["stages"][domain]["roles"].items():
        assert row["contract"]["evaluation"] == original["stages"][domain]["roles"][role]["contract"]["evaluation"]
        assert row["contract"]["nonstandard_start"]["enabled"] is (role == "train")
        assert row["runtime"]["modules"]["nonstandard_start"] is (role == "train")
        assert row["runtime"]["nonstandard_start"] == revision.runtime_expectation(row["contract"])
        assert row["contract"]["release_stop"]["enabled"] is (role == "train")
        assert row["runtime"]["modules"]["release_stop"] is (role == "train")
        assert row["runtime"]["release_stop"] == revision.release_stop_runtime_expectation(row["contract"])
        if role != "train":
            old_contract = original["stages"][domain]["roles"][role]["contract"]
            assert row["contract"]["scene_groups"] == old_contract["scene_groups"]
            assert row["contract"]["behavior_pool_fractions"] == old_contract["behavior_pool_fractions"]
    infos = dict(contract_sha256=revision.SOURCES[domain][1], task_campaign=dict(domain=domain),
        task_actor_update=revision.ORIGINS[domain], critic_warmup_remaining=0, successful_updates_total=23724)
    target_sha = campaign.digest(ROOT / revision.worker_file(domain))
    assert resume_critic_warmup(config, infos, contract_sha256=target_sha, task_change=True) == 10
    infos.update(contract_sha256=target_sha, critic_warmup_remaining=4)
    assert resume_critic_warmup(config, infos, contract_sha256=target_sha, task_change=True) == 4
    infos["critic_warmup_remaining"] = 0
    assert resume_critic_warmup(config, infos, contract_sha256=target_sha, task_change=True) == 0


def test_reopening_preserves_all_consumption_results_and_history_and_is_idempotent(reopening):
    job = reopening
    before, records = deepcopy(job.report), deepcopy(job.records)
    result = revision.reopen_completed_domains(job.report, job.records, SOURCE, TARGET, job.root)
    assert result["consumption"] == "preserved_no_refund"
    assert job.records == records
    assert campaign.account_updates(TARGET, records) == campaign.account_updates(SOURCE, records)
    assert job.report["evaluations"] == before["evaluations"]
    assert job.report["domains"]["jump"] == before["domains"]["jump"]
    for domain in revision.ORIGINS:
        row = job.report["domains"][domain]
        assert row["status"] == "domain_reopened_unassessed"
        assert row["completion_history"] == [dict(plan_revision="v12", record=before["domains"][domain])]
        assert row["reopening"]["origin_actor_update"] == revision.ORIGINS[domain]
        assert row["reopening"]["end_actor_update"] == revision.ENDPOINTS[domain]
        assert f"old-{domain}-review" in job.report["consumed_review_tokens"]
        assert job.learning_source(next(r for r in TARGET["domains"] if r["name"] == domain)) == Path(before["domains"][domain]["candidate_checkpoint"])
    once = deepcopy(job.report)
    revision.reopen_completed_domains(job.report, job.records, SOURCE, TARGET, job.root)
    assert job.report == once


def test_account_migration_reopens_once_and_fault_resume_keeps_durable_history(reopening, monkeypatch):
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
        return dict(schema="task-plan-revision-v13", source_plan_sha256=revision.SOURCE_PLAN_SHA,
            target_plan_sha256=campaign.digest(job.args.contract), consumption="preserved_no_refund")
    monkeypatch.setattr(runner, "validate_plan_migration", validate)
    monkeypatch.setattr(job, "assert_no_live_workers", lambda: None)
    job.persist = runner.TaskCampaign.persist.__get__(job)
    job.open_account()
    ledger = json.loads(job.claim.read_text())
    assert ledger["attempts"] == old_ledger["attempts"]
    assert ledger["consumed_updates"] == old_ledger["consumed_updates"]
    assert ledger["contract_sha256"] == campaign.digest(job.args.contract)
    assert json.loads((job.root / "revisions/v12_account_before_migration.json").read_text()) == old_ledger
    assert job.report["revision_migration"]["domain_reopening"]["consumption"] == "preserved_no_refund"
    histories = {domain: deepcopy(job.report["domains"][domain]["completion_history"]) for domain in revision.ORIGINS}
    job.open_account()
    assert {domain: job.report["domains"][domain]["completion_history"] for domain in revision.ORIGINS} == histories


@pytest.mark.parametrize("fault", ["unfinished", "qualified", "clock", "SHA", "foreign_branch", "contract", "budget", "lost_history"])
def test_reopening_rejects_wrong_branch_clocks_accounting_and_missing_history(reopening, fault):
    job = reopening
    row = job.report["domains"]["speed"]
    if fault == "unfinished":
        row["status"] = "running"
    elif fault == "qualified":
        row["domain_qualified"] = True
    elif fault == "clock":
        seal = Path(row["candidate_checkpoint"]).with_suffix(".checkpoint.json")
        data = json.loads(seal.read_text())
        data["task_actor_update"] -= 1
        runner.write_json(seal, data)
    elif fault == "SHA":
        row["candidate_checkpoint_sha256"] = "0" * 64
    elif fault == "foreign_branch":
        row["candidate_checkpoint"] = job.report["domains"]["jump"]["candidate_checkpoint"]
    elif fault == "contract":
        seal = Path(row["candidate_checkpoint"]).with_suffix(".checkpoint.json")
        data = json.loads(seal.read_text())
        data["contract_sha256"] = "0" * 64
        runner.write_json(seal, data)
    elif fault == "budget":
        job.records["speed/attempt_000/train"]["consumed_updates"] = 19900
    else:
        revision.reopen_completed_domains(job.report, job.records, SOURCE, TARGET, job.root)
        del job.report["domains"]["speed"]["completion_history"]
    before = deepcopy(job.report)
    with pytest.raises(ValueError):
        revision.reopen_completed_domains(job.report, job.records, SOURCE, TARGET, job.root)
    assert job.report == before


def test_old_review_token_is_consumed_without_erasing_its_request_file(reopening, monkeypatch):
    job = reopening
    path = job.root / "domain_review_request.json"
    runner.write_json(path, dict(old_finish="ordinary speed was already completed"))
    token = campaign.digest(path)
    job.report["domains"]["speed"]["finish_review"]["token"] = token
    revision.reopen_completed_domains(job.report, job.records, SOURCE, TARGET, job.root)
    job.active = next(row for row in TARGET["domains"] if row["name"] == "speed")
    job._active_review = None
    monkeypatch.setattr(runner, "validate_domain_review", lambda *args: pytest.fail("Old finish review must be consumed"))
    job.review_request()
    assert job._active_review is None and path.exists() and campaign.digest(path) == token
    assert job.report["domains"]["speed"]["completion_history"][0]["record"]["finish_review"]["token"] == token


def test_normal_runner_reopens_speed_then_terrain_and_later_resumes_unchanged_jump(reopening, monkeypatch):
    job = reopening
    revision.reopen_completed_domains(job.report, job.records, SOURCE, TARGET, job.root)
    job.persist = lambda: job.report.update(campaign.account_updates(TARGET, job.records))
    calls = []
    monkeypatch.setattr(job, "review_request", lambda *args: None)
    monkeypatch.setattr(job, "evaluate_pair", lambda *args: pytest.fail("Teaching cannot wait for fixed evaluation"))
    def execute(command, log, directory):
        domain = directory.parents[1].name
        row = next(row for row in TARGET["domains"] if row["name"] == domain)
        source = Path(command[command.index("--resume") + 1])
        assert source.parent.parent.parent.name == domain
        requested = int(command[command.index("--updates") + 1])
        actor = int(command[command.index("--task-actor-update") + 1])
        expected = revision.ENDPOINTS[domain] - revision.ORIGINS[domain] + 10 if domain in revision.ORIGINS else 15000 - 2735
        assert requested == expected
        assert actor == (revision.ORIGINS[domain] if domain in revision.ORIGINS else 2735)
        calls.append((domain, requested, source))
        directory.mkdir()
        final = directory / "model_final.pt"
        shutil.copyfile(ROOT / row["contract"], final.with_suffix(".contract.json"))
        torch.save(dict(infos=dict(task_actor_update=revision.ENDPOINTS[domain])), final)
        total = job.report["consumed_updates"] + requested
        runner.write_json(final.with_suffix(".checkpoint.json"), dict(schema="chassis-learning-checkpoint-v1",
            sealed=True, checkpoint_sha256=campaign.digest(final), contract_sha256=row["sha256"],
            learning_updates=total, task_actor_update=revision.ENDPOINTS[domain]))
        runner.write_json(directory / "completion.json", dict(status="completed", successful_updates=requested,
            actor_updates_in_block=requested - (10 if domain in revision.ORIGINS else 0)))
        return 0
    monkeypatch.setattr(job, "execute", execute)
    for row in TARGET["domains"]:
        if row["name"] in ("speed", "terrain", "jump"):
            job.run_domain(row)
    assert [domain for domain, _, _ in calls] == ["speed", "terrain", "jump"]
    assert job.report["consumed_updates"] == 23868 + 1510 + 1010 + 12265
    for domain in revision.ORIGINS:
        result = job.report["domains"][domain]
        assert result["status"] == "domain_schedule_finished_unassessed"
        assert result["domain_qualified"] is None
        assert len(result["completion_history"]) == 1 and result["reopening"]["origin_actor_update"] == revision.ORIGINS[domain]


@pytest.mark.parametrize("fault", ["endpoint", "parent", "reward", "quota", "origin", "hold", "disabled_train", "enabled_eval", "other_worker", "release_disabled_train", "release_enabled_eval", "release_dwell", "release_pool", "release_slew"])
def test_resealed_manifest_and_design_do_not_authorize_extra_changes(tmp_path, monkeypatch, fault):
    shutil.copytree(ROOT / "contracts", tmp_path / "contracts")
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    target = deepcopy(TARGET)
    config = worker("speed")
    design = read(revision.design_file("speed"))
    manifest = read(revision.MANIFEST_FILE)
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
    elif fault == "hold":
        config["nonstandard_start"]["hold_seconds"] = .3
    elif fault == "disabled_train":
        design["stages"]["speed"]["roles"]["train"]["runtime"]["modules"]["nonstandard_start"] = False
    elif fault == "enabled_eval":
        design["stages"]["speed"]["roles"]["quick"]["contract"]["nonstandard_start"]["enabled"] = True
    elif fault == "release_disabled_train":
        design["stages"]["speed"]["roles"]["train"]["runtime"]["modules"]["release_stop"] = False
    elif fault == "release_enabled_eval":
        design["stages"]["speed"]["roles"]["quick"]["contract"]["release_stop"]["enabled"] = True
    elif fault == "release_dwell":
        config["release_stop"]["stationary_window_seconds"] = 9.
    elif fault == "release_pool":
        target["domains"][2]["sampling_pools"][-1]["fraction"] = .06
    elif fault == "release_slew":
        config["command_slew"]["vx_m_s2"] = 2.
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


def test_v13_remote_receipt_uses_v12_and_cannot_change_environment_count(tmp_path, monkeypatch):
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
    revised[2] = str(new / "scripts/run_task_campaign.py")
    revised[4] = str(new / revision.PLAN_FILE)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted(new.rglob("*")):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(new)), recursive=False)
    receipt = dict(schema="task-plan-revision-v13", previous_job_sha256=sha(tmp_path / "job.json"),
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
