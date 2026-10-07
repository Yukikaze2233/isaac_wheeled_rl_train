"""Original production entry admission with native CPU seals, without learning."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import run_task_campaign as dispatcher
from test_integrated_lineage import checkpoint, metadata, write
from test_integrated_runtime import seal_clocks
from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis import integrated_runtime as runtime
from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture(scope="module")
def static_bundle(tmp_path_factory, request):
    # Generate the expensive canonical static graph once; validators remain real.
    tmp_path = tmp_path_factory.mktemp("integrated_production")
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    source, run = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    run.mkdir()
    shutil.copytree(ROOT / "contracts", source / "contracts")
    shutil.copytree(ROOT / production.SCUT_DIRECTORY, source / production.SCUT_DIRECTORY)
    sources = runtime.sampling.load_sources(ROOT)
    for file in [*runtime._references(ROOT, sources), runtime.AUDIT[0]]:
        destination = source / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / file, destination)
    baseline = read(ROOT / production.SOURCE_PLAN_FILE)
    baseline["revision_migration"]["run_directory"] = str(run)
    baseline["budget_source_stage"] = str(tmp_path / "budget")
    baseline_file = "contracts/test_integrated_baseline.json"
    write(source / baseline_file, baseline)
    monkeypatch.setattr(production, "SOURCE_PLAN_FILE", baseline_file)
    monkeypatch.setattr(production, "SOURCE_PLAN_SHA", lineage.digest(source / baseline_file))
    monkeypatch.setattr(campaign, "ROOT", source)
    monkeypatch.setattr(dispatcher, "ROOT", source)
    bundle = production.write_bundle(source, source_root=str(source), run_root=str(run))
    plan = read(source / production.PLAN_FILE)
    binding = production.binding_for(plan, source, run)
    manifest = binding.read()
    contracts = {d: source / (runtime.sampling.SOURCE_BINDINGS[d][0] if d in ("flat", "recovery")
                             else production.worker_file(d)) for d in campaign.DOMAINS}
    f = SimpleNamespace(source=source, run=run, baseline=baseline, bundle=bundle,
                        plan=plan, binding=binding, manifest=manifest, contracts=contracts)
    yield f


@pytest.fixture
def frozen(static_bundle):
    f = deepcopy(static_bundle)
    source, run, plan, binding, contracts = f.source, f.run, f.plan, f.binding, f.contracts
    shutil.rmtree(run)
    run.mkdir()
    budget = Path(plan["budget_source_stage"])
    if budget.exists():
        shutil.rmtree(budget)
    f.records = {d: dict(domain=d, consumed_updates=n, actor_updates=n - 10)
                 for d, n in (("flat", 10000), ("recovery", 15000), ("speed", 5000),
                              ("terrain", 2000), ("jump", 1000))}
    f.rec = checkpoint(f, "recovery", count=26000, task=15000, lineage=25000, final=True)
    f.rec_record = dict(status="domain_schedule_finished_unassessed", domain_qualified=None,
                        candidate_checkpoint=f.rec["path"], candidate_checkpoint_sha256=f.rec["sha256"],
                        schedule_completion=dict(reason="iteration_endpoint", end_actor_update=15000,
                                                 actual_actor_update=15000))
    f.entry = lineage.admit_entry(binding, f.rec, f.rec_record)
    f.claim = Path(plan["budget_source_stage"]) / (campaign.CAMPAIGN_ID + ".account.json")
    f.state = run / "campaign_state.json"
    totals = campaign.account_updates(plan, f.records)
    f.report = dict(contract_sha256=production.SOURCE_PLAN_SHA, domains={"recovery": f.rec_record},
                    evaluations={"old": "retained"}, consumed_review_tokens=["prior"],
                    training_process_seconds=0., assessment_seconds=0., elapsed_seconds=0., **totals)
    for d, _ in lineage.CHAIN:
        f.report["domains"][d] = dict(status="domain_schedule_finished_unassessed",
            candidate_checkpoint="retained-independent-" + d, finish_review={"token": d},
            completion_history=[{"old": "retained"}])
    write(f.state, f.report)
    write(f.claim, dict(campaign_id=campaign.CAMPAIGN_ID, run_directory=str(run),
        contract_sha256=production.SOURCE_PLAN_SHA, attempts=f.records,
        rollback_refunds_consumed_updates=False, **totals))
    f.worker = read(contracts["speed"])
    return f


def admitted(f):
    return production.prepare_admission(f.plan, f.source, f.run, account_path=f.claim, state_path=f.state)


def arguments(f, receipt):
    path = f.run / "source_selection.json"
    write(path, receipt)
    seal = read(Path(receipt["selection"]["checkpoint"]["path"]).with_suffix(".checkpoint.json"))
    return SimpleNamespace(integrated_source_selection_file=path,
        integrated_source_selection_sha256=lineage.digest(path), resume_task_change=True,
        resume=Path(receipt["selection"]["checkpoint"]["path"]), num_envs=4096, seed=290927,
        block_index=0, learning_rate_scale=1., global_actor_update_offset=0,
        consumed_updates=receipt["allowance"]["consumed_updates"],
        actor_updates_consumed=receipt["allowance"]["actor_updates_consumed"],
        task_actor_update=receipt["selection"]["fresh_domain_actor_updates"],
        stage_actor_update=seal["stage_actor_update"], updates=receipt["allowance"]["requested_total_updates"],
        validate_scene=0)


def job_for(f, monkeypatch):
    job = object.__new__(dispatcher.TaskCampaign)
    job.plan, job.root, job.claim = deepcopy(f.plan), f.run, f.claim
    job.args = SimpleNamespace(contract=f.source / production.PLAN_FILE, device="cpu")
    job.report, job.records, job._session_started = {}, {}, None
    job.deadline, job.stop_requested = float("inf"), False
    monkeypatch.setattr(job, "assert_no_live_workers", lambda: None)
    return job


def native_resume_initial(f, reference):
    """Restore the actual native CPU model/Adam, then save zero-update startup."""
    import torch
    from wheeled_tasks.chassis.policy_transfer import verify_learning_state_restore
    source = Path(reference["path"])
    saved = torch.load(source, map_location="cpu", weights_only=True)
    algorithm = lineage._native_learning_template(f.worker)
    algorithm.load(saved, load_cfg=dict(optimizer=True), strict=True)
    verify_learning_state_restore(algorithm, saved)
    initial = algorithm.save()
    initial["infos"] = deepcopy(saved["infos"])
    # New episode startup is allowed to consume environment RNG while no PPO
    # or actor update has been committed in this resumed attempt.
    generator = torch.Generator()
    generator.set_state(initial["infos"]["env_rng_state"])
    torch.rand(1, generator=generator)
    initial["infos"]["env_rng_state"] = generator.get_state()
    path = f.run / "speed/attempt_resumed/train/initial_state/model_final.pt"
    path.parent.mkdir(parents=True)
    torch.save(initial, path)
    path.with_suffix(".contract.json").write_bytes(source.with_suffix(".contract.json").read_bytes())
    seal = read(source.with_suffix(".checkpoint.json"))
    seal["checkpoint_sha256"] = lineage.digest(path)
    write(path.with_suffix(".checkpoint.json"), seal)
    write(path.parent / "completion.json", dict(status="completed", scope="initial_learning_state",
        parent_updates=seal["learning_updates"], successful_updates=0,
        checkpoint_sha256=seal["checkpoint_sha256"]))
    return dict(path=str(path), sha256=seal["checkpoint_sha256"])


def test_static_pin_graph_and_train_projection_are_canonical(frozen):
    f = frozen
    assert f.manifest["campaign_plan"]["file"] == production.SOURCE_PLAN_FILE
    assert production.validate_plan(f.plan, f.source) == f.manifest
    assert campaign.validate_plan(f.plan) is None
    recipe = read(f.source / production.RECIPE_FILE)
    assert recipe["qualification"] is None and recipe["train_only"] is True
    for d, parent in lineage.CHAIN:
        worker = read(f.contracts[d])
        assert production.validate_static_worker(worker, f.source) == d
        assert "lineage_manifest" not in worker["integrated_production"]
        assert "learning_source" not in recipe["domains"][d]["runtime"]
        assert "clocks_at_source" not in recipe["domains"][d]["runtime"]
        effective = production.static_training_config(worker, f.source)
        original = runtime.sampling.load_sources(f.source)[d]
        assert effective["evaluation"] == original["evaluation"]
        assert effective["integrated_runtime_candidate"]["static_recipe_preflight"] is True
        assert worker["task_campaign"]["source_contract_file"] == (
            runtime.sampling.SOURCE_BINDINGS["recovery"][0] if parent == "recovery" else production.worker_file(parent))


@pytest.mark.parametrize("field,value", [("learning_rate", 2e-4), ("position_iterations", 48),
                                        ("closure_gap_termination_m", .004)])
def test_static_worker_rejects_canonical_tampering(frozen, field, value):
    changed = deepcopy(frozen.worker)
    changed[field] = value
    with pytest.raises(ValueError):
        production.validate_static_worker(changed, frozen.source)


def test_prepare_admission_reads_real_normal_rec_and_never_rewrites_account(frozen):
    f = frozen
    before = {p: p.read_bytes() for p in (f.claim, f.state)}
    receipt = admitted(f)
    assert receipt["selection"]["checkpoint"] == f.rec
    assert receipt["selection"]["fresh_domain_actor_updates"] == 0
    assert receipt["allowance"]["consumed_updates"] == 34000
    assert receipt["allowance"]["historical_and_joint_domain_consumed_updates"] == 5000
    assert receipt["allowance"]["charged_critic_updates"] == 10
    assert receipt["allowance"]["maximum_actor_updates"] == 14990
    assert receipt["component_clocks_at_source"]["component_actor_updates"] == {"recovery": 15000, "speed": 0}
    assert receipt["freeze_ready"] is False and receipt["qualification"] is None
    assert {p: p.read_bytes() for p in before} == before


def test_progress_reservation_is_retained_without_mutating_durable_claim(frozen):
    f = frozen
    path = f.run / "speed/attempt_old/train/progress.json"
    write(path, dict(successful_updates=3, attempted_updates=4, actor_updates_in_block=3))
    before = f.claim.read_bytes()
    receipt = admitted(f)
    assert receipt["allowance"]["consumed_updates"] == 34004
    assert receipt["allowance"]["historical_and_joint_domain_consumed_updates"] == 5004
    assert receipt["allowance"]["maximum_actor_updates"] == 14986
    assert f.claim.read_bytes() == before


def test_early_rec_cannot_admit_or_archive_a_joint_segment(frozen):
    f = frozen
    state = read(f.state)
    state["domains"]["recovery"]["schedule_completion"]["actual_actor_update"] = 14999
    write(f.state, state)
    before = f.claim.read_bytes()
    with pytest.raises(ValueError):
        admitted(f)
    assert f.claim.read_bytes() == before


def test_archive_is_idempotent_and_old_terminal_status_cannot_skip_joint(frozen):
    f = frozen
    before = deepcopy(f.report)
    result = production.archive_legacy_domains(f.report, f.binding, f.entry)
    assert f.report == before
    for d, _ in lineage.CHAIN:
        assert result["domains"][d]["status"] == "joint_domain_pending"
        assert result["domains"][d]["legacy_completion_history"] == [before["domains"][d]]
        assert "candidate_checkpoint" not in result["domains"][d]
    assert result["domains"]["recovery"] == before["domains"]["recovery"]
    assert result["evaluations"] == before["evaluations"]
    assert result["consumed_updates"] == before["consumed_updates"]
    assert production.archive_legacy_domains(result, f.binding, f.entry) == result


def test_original_account_migrates_once_and_learning_source_rebases_rec(frozen, monkeypatch):
    f = frozen
    job = job_for(f, monkeypatch)
    job.open_integrated_account()
    assert job.records == f.records
    assert job.report["consumed_updates"] == 34000
    assert read(f.claim)["contract_sha256"] == lineage.digest(f.source / production.PLAN_FILE)
    for d, _ in lineage.CHAIN:
        assert job.report["domains"][d]["status"] == "joint_domain_pending"
    assert (f.run / "revisions/v21_account_before_migration.json").exists()
    once = deepcopy(job.report)
    job.open_integrated_account()
    assert job.report == once and job.records == f.records
    monkeypatch.setattr(job, "latest_checkpoint", lambda *_: pytest.fail("Legacy own must not be selected"))
    row = next(r for r in job.plan["domains"] if r["name"] == "speed")
    assert job.learning_source(row) == Path(f.rec["path"])
    assert job._joint_selection["fresh_joint_actor_updates"] == 0


def test_same_joint_own_partial_critic_restores_component_clocks(frozen):
    f = frozen
    initial = admitted(f)
    legacy = checkpoint(f, "speed", count=39000, task=5000, lineage=29000, suffix="legacy")
    own = checkpoint(f, "speed", count=34004, task=0, lineage=25004,
        meta=metadata(f, "speed", joint_actor=0, domain_actor=0), warmup=6, suffix="joint",
        actor_consumed=campaign.account_updates(f.plan, f.records)["actor_updates_consumed"])
    own = seal_clocks(own, initial["component_clocks_at_source"])
    report = production.archive_legacy_domains(f.report, f.binding, f.entry)
    report["active_joint_domain"] = "speed"
    binding, selection = production.select_source(f.plan, f.source, f.run, report)
    assert selection["checkpoint"] == own and selection["checkpoint"] != legacy
    assert selection["critic_warmup_remaining"] == 6
    assert selection["fresh_joint_actor_updates"] == selection["fresh_domain_actor_updates"] == 0
    records = deepcopy(f.records)
    records["joint"] = dict(domain="speed", consumed_updates=4, actor_updates=0)
    write(f.claim, dict(campaign_id=campaign.CAMPAIGN_ID, run_directory=str(f.run),
        contract_sha256=production.SOURCE_PLAN_SHA, attempts=records, rollback_refunds_consumed_updates=False,
        **campaign.account_updates(f.plan, records)))
    receipt = production.make_admission(f.plan, f.source, f.run, selection, records, account_path=f.claim, state_path=f.state)
    assert production.authenticate_receipt(f.worker, receipt, f.source) == receipt
    assert receipt["allowance"]["charged_critic_updates"] == 6
    assert receipt["component_clocks_at_source"] == initial["component_clocks_at_source"]


def test_resume_skips_authenticated_zero_update_initial_and_keeps_cross_attempt_fork_guard(frozen):
    f = frozen
    initial = admitted(f)
    for domain, _ in lineage.CHAIN:
        assert read(f.contracts[domain])["evaluation"]["regression_recovery"]
    own = checkpoint(f, "speed", count=34004, task=0, lineage=25004,
        meta=metadata(f, "speed", joint_actor=0, domain_actor=0), warmup=6, suffix="joint",
        actor_consumed=campaign.account_updates(f.plan, f.records)["actor_updates_consumed"])
    own = seal_clocks(own, initial["component_clocks_at_source"])
    report = production.archive_legacy_domains(f.report, f.binding, f.entry)
    report["active_joint_domain"] = "speed"
    _, previous = production.select_source(f.plan, f.source, f.run, report)
    startup = native_resume_initial(f, own)
    _, resumed = production.select_source(f.plan, f.source, f.run, report, previous_selection=previous)
    assert resumed == previous and resumed["checkpoint"] == own
    assert resumed["critic_warmup_remaining"] == 6
    assert runtime.source_clocks(f.binding, resumed) == initial["component_clocks_at_source"]
    # The same real state outside the exact startup location is an endpoint,
    # and the cross-attempt clock/RNG fork protection must still reject it.
    startup_path = Path(startup["path"])
    endpoint = startup_path.parent.parent / "model.pt"
    for suffix in (".pt", ".contract.json", ".checkpoint.json"):
        shutil.copyfile(startup_path.with_suffix(suffix), endpoint.with_suffix(suffix))
    with pytest.raises(ValueError, match="fork across native attempts"):
        production.select_source(f.plan, f.source, f.run, report, previous_selection=previous)


@pytest.mark.parametrize("field,value", [("scope", "learning_state"), ("successful_updates", 1),
                                        ("checkpoint_sha256", "0" * 64)])
def test_startup_state_requires_authenticated_zero_update_marker(frozen, field, value):
    f = frozen
    initial = admitted(f)
    own = checkpoint(f, "speed", count=34004, task=0, lineage=25004,
        meta=metadata(f, "speed", joint_actor=0, domain_actor=0), warmup=6, suffix="joint",
        actor_consumed=campaign.account_updates(f.plan, f.records)["actor_updates_consumed"])
    own = seal_clocks(own, initial["component_clocks_at_source"])
    startup = native_resume_initial(f, own)
    completion_path = Path(startup["path"]).parent / "completion.json"
    completion = read(completion_path)
    completion[field] = value
    write(completion_path, completion)
    report = production.archive_legacy_domains(f.report, f.binding, f.entry)
    report["active_joint_domain"] = "speed"
    with pytest.raises(ValueError, match="authenticated zero-update completion"):
        production.select_source(f.plan, f.source, f.run, report)


@pytest.mark.parametrize("field,value", [("num_envs", 4095), ("task_actor_update", 1),
                                        ("consumed_updates", 33999), ("updates", 1)])
def test_original_worker_invocation_rejects_clock_population_and_allowance_tampering(frozen, field, value):
    f = frozen
    receipt = admitted(f)
    args = arguments(f, receipt)
    assert production.authenticate_invocation(f.worker, args, f.source) == receipt
    setattr(args, field, value)
    with pytest.raises(ValueError):
        production.authenticate_invocation(f.worker, args, f.source)


def test_scene_only_requires_one_update_and_remains_separate_from_learning(frozen):
    f = frozen
    args = arguments(f, admitted(f))
    args.validate_scene, args.updates = 1, 1
    assert production.authenticate_invocation(f.worker, args, f.source)["qualification"] is None
    args.updates = 2
    with pytest.raises(ValueError):
        production.authenticate_invocation(f.worker, args, f.source)


def test_metadata_uses_only_actual_admitted_actor_work(frozen):
    receipt = admitted(frozen)
    before = deepcopy(receipt)
    critic = production.checkpoint_metadata(receipt, task_actor_update=0)
    assert critic["integrated_lineage"]["joint_actor_updates"] == 0
    after = production.checkpoint_metadata(receipt, task_actor_update=7)
    assert after["integrated_lineage"]["domain_actor_updates"] == 7
    assert after["integrated_runtime_clocks"]["component_actor_updates"] == {"recovery": 15007, "speed": 7}
    with pytest.raises(ValueError):
        production.checkpoint_metadata(receipt, task_actor_update=7, domain_complete=True)
    with pytest.raises(ValueError):
        production.checkpoint_metadata(receipt, task_actor_update=receipt["allowance"]["requested_actor_updates"] + 1)
    assert receipt == before


def test_absent_opt_in_preserves_native_defaults_without_reading_sources():
    native = dict(task_campaign={"domain": "speed"}, arbitrary="native")
    args = SimpleNamespace(integrated_source_selection_file=None, integrated_source_selection_sha256=None)
    assert production.authenticate_invocation(native, args, Path("/missing")) is None
    assert production.attach_runtime(native, None, Path("/missing")) == native
    assert production.static_training_config(native, Path("/missing")) == native
    assert production.production_source_files(native, root=Path("/missing")) == []
    args.integrated_source_selection_file = Path("/foreign-admission.json")
    with pytest.raises(ValueError):
        production.authenticate_invocation(native, args, Path("/missing"))


def test_formal_admission_snapshots_survive_zero_progress_publication(frozen, monkeypatch):
    f = frozen
    job = job_for(f, monkeypatch)
    job.open_integrated_account()
    row = next(r for r in f.plan["domains"] if r["name"] == "speed")
    source = job.learning_source(row)
    directory = f.run / "speed/attempt_000/train"
    directory.parent.mkdir(parents=True)
    allowance = lineage.remaining_budget(f.binding, job._joint_selection, job.records)
    command = job.train_command(row, directory, source, allowance["requested_total_updates"])
    path = Path(command[command.index("--integrated-source-selection-file") + 1])
    receipt = read(path)
    snapshots = {name: Path(receipt[name]["file"]).read_bytes() for name in ("account", "state")}
    assert receipt["account"]["file"] != str(f.claim)
    # Original supervisor publishes an initial zero-update progress row. It
    # changes live JSON identity without changing either consumption clock.
    job.records["speed/attempt_000/train"] = dict(domain="speed", consumed_updates=0, actor_updates=0)
    job.report["status"] = "running"
    job.persist()
    assert production.authenticate_receipt(f.worker, receipt, f.source) == receipt
    assert all(Path(receipt[name]["file"]).read_bytes() == value for name, value in snapshots.items())
    job.records["speed/attempt_000/train"]["consumed_updates"] = 1
    job.persist()
    with pytest.raises(ValueError, match="consumption changed"):
        production.authenticate_receipt(f.worker, receipt, f.source)


def test_charged_cap_final_reconstructs_joint_completion_after_report_loss(frozen, monkeypatch):
    import torch
    f = frozen
    job = job_for(f, monkeypatch)
    job.open_integrated_account()
    row = next(r for r in f.plan["domains"] if r["name"] == "speed")
    job.learning_source(row)
    receipt = production.make_admission(f.plan, f.source, f.run, job._joint_selection, job.records,
                                        account_path=f.claim, state_path=f.state)
    allowance = receipt["allowance"]
    actor = allowance["requested_actor_updates"]
    assert actor < receipt["actor_endpoint"]  # Old branch consumption still costs budget.
    saved = production.checkpoint_metadata(receipt, task_actor_update=actor, domain_complete=True)
    own = checkpoint(f, "speed", count=34000 + allowance["requested_total_updates"], task=actor,
        lineage=25000 + allowance["requested_total_updates"], final=True, suffix="joint",
        meta=saved["integrated_lineage"], actor_consumed=33930 + actor)
    own = seal_clocks(own, saved["integrated_runtime_clocks"])
    completion_path = Path(own["path"]).parent / "completion.json"
    completion = read(completion_path)
    completion.update(parent_updates=34000, successful_updates=allowance["requested_total_updates"],
                      actor_updates_in_block=actor)
    write(completion_path, completion)
    # Native final and immutable post-update snapshot legitimately share the
    # learning clock while the final carries the terminal metadata transition.
    final_path = Path(own["path"])
    snapshot_path = final_path.parent / "checkpoints/update_00049000/model.pt"
    snapshot_path.parent.mkdir(parents=True)
    snapshot = torch.load(final_path, map_location="cpu", weights_only=True)
    snapshot["infos"]["integrated_lineage"]["domain_complete"] = False
    torch.save(snapshot, snapshot_path)
    snapshot_path.with_suffix(".contract.json").write_bytes(final_path.with_suffix(".contract.json").read_bytes())
    snapshot_seal = read(final_path.with_suffix(".checkpoint.json"))
    snapshot_seal["checkpoint_sha256"] = lineage.digest(snapshot_path)
    snapshot_seal["integrated_lineage"]["domain_complete"] = False
    write(snapshot_path.with_suffix(".checkpoint.json"), snapshot_seal)
    snapshot_ref = dict(path=str(snapshot_path), sha256=lineage.digest(snapshot_path))
    job.report["joint_learning_sources"]["speed"] = lineage.select_learning_source(
        f.binding, "speed", f.entry, own_candidates=[snapshot_ref])
    job.records["speed/attempt_joint/train"] = dict(domain="speed",
        consumed_updates=allowance["requested_total_updates"], actor_updates=actor)
    job.persist()
    f.state.unlink()  # The claim and immutable migration receipt survived.
    restored = job_for(f, monkeypatch)
    restored.open_integrated_account()
    monkeypatch.setattr(restored, "execute", lambda *a: pytest.fail("A sealed completed own must not retrain"))
    restored.run_integrated_domain(row)
    result = restored.report["domains"]["speed"]
    assert result["candidate_checkpoint"] == own["path"]
    assert result["status"] == "joint_domain_finished_unassessed"
    assert result["schedule_completion"]["reason"] == "charged_cap"
    assert result["integrated_lineage"]["domain_actor_updates"] == actor
    assert restored.report["domain_consumed_updates"]["speed"] == 20000
    assert result["legacy_completion_history"] == [f.report["domains"]["speed"]]
    # Equal model and Adam values cannot excuse a fork of future sampling RNG.
    snapshot["infos"]["env_rng_state"][0] ^= 1
    torch.save(snapshot, snapshot_path)
    snapshot_seal["checkpoint_sha256"] = lineage.digest(snapshot_path)
    write(snapshot_path.with_suffix(".checkpoint.json"), snapshot_seal)
    with pytest.raises(ValueError, match="RNG"):
        restored.learning_source(row)


@pytest.mark.parametrize("seed", [290929, 290930])
def test_fixed_exam_subset_preserves_native_inputs_and_rejects_changed_criteria(frozen, seed):
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
    f = frozen
    config = deepcopy(f.worker)
    names = config["evaluation"]["promotion_case_names"]
    assert len(names) < len(config["evaluation"]["cases"])
    config["evaluation"] = select_evaluation_cases(config["evaluation"], names)
    config["evaluation"]["seed"] = seed
    actual = production.fixed_evaluation_config(config, f.source)
    native = runtime.sampling.load_sources(f.source)["speed"]
    native["evaluation"] = deepcopy(config["evaluation"])
    expected = fixed_suite_contract(native)
    for field in ("evaluation", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions",
                  "scene_groups", "dense_tracking", "new_asset_curriculum", "jump_assist"):
        assert actual[field] == expected[field]
    assert "integrated_runtime_candidate" not in actual
    altered = deepcopy(config)
    altered["evaluation"]["cases"][0]["command"][0] += .01
    with pytest.raises(ValueError, match="original cases"):
        production.fixed_evaluation_config(altered, f.source)
    altered = deepcopy(config)
    altered["evaluation"]["seed"] = 290927
    with pytest.raises(ValueError, match="original cases"):
        production.fixed_evaluation_config(altered, f.source)
