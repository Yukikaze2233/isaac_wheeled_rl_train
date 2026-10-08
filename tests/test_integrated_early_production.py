"""The new static graph admits only an authenticated real stopped REC origin."""
from copy import deepcopy
from pathlib import Path

import pytest

from test_integrated_heading_revision import heading_frozen
from test_integrated_lineage import checkpoint, write
from test_integrated_production import admitted, arguments, frozen, read, static_bundle
from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis import task_campaign as campaign


@pytest.fixture
def early_frozen(heading_frozen):
    f = deepcopy(heading_frozen)
    prior_files = [*production.artifact_files(24).values(), *f.bundle["files"]]
    prior_bytes = {file: (f.source / file).read_bytes() for file in prior_files
                   if (f.source / file).is_file()}
    authorization_path = f.source / production.EARLY_AUTHORIZATION_FILE
    authorization = read(authorization_path)
    authorization.update(run_root=str(f.run), source_plan_sha256=production.SOURCE_PLAN_SHA)
    write(authorization_path, authorization)
    f.bundle = production.write_bundle(f.source, source_root=str(f.source), run_root=str(f.run), version=26)
    assert prior_bytes == {file: (f.source / file).read_bytes() for file in prior_bytes}
    f.plan = read(f.source / production.artifact_files(26)["plan"])
    f.binding = production.binding_for(f.plan, f.source, f.run)
    f.manifest = f.binding.read()
    for domain, _ in lineage.CHAIN:
        f.contracts[domain] = f.source / production.worker_file(domain, version=26)
    f.worker = read(f.contracts["speed"])
    f.rec = checkpoint(f, "recovery", count=25052, task=14052, lineage=24052, final=True)
    completion_path = Path(f.rec["path"]).parent / "completion.json"
    completion = read(completion_path)
    completion.update(status="stopped", reason="signal", checkpoint_sha256=f.rec["sha256"])
    write(completion_path, completion)
    f.report["status"] = "stopped"
    write(f.state, f.report)
    write(f.run / "completion.json", f.report)
    for name in ("campaign_state.json", "completion.json"):
        snapshot = f.run / production.EARLY_ORIGIN_DIRECTORY / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((f.run / name).read_bytes())
    return f


def test_v26_reuses_exact_v25_domains_and_complete_source_inventory(early_frozen):
    f = early_frozen
    files = production.artifact_files(26)
    recipe = read(f.source / files["recipe"])
    previous = production.artifact_files(25)["recipe"]
    assert recipe["domains"] == read(f.source / previous)["domains"]
    assert recipe["predecessor_recipe"] == dict(file=previous, sha256=lineage.digest(f.source / previous))
    assert f.manifest["recovery_entry"]["early_stop"]["authorization_sha256"] == lineage.digest(
        f.source / production.EARLY_AUTHORIZATION_FILE)
    assert campaign.validate_plan(f.plan) is None
    for domain, _ in lineage.CHAIN:
        worker = read(f.contracts[domain])
        assert production.validate_static_worker(worker, f.source) == domain
        effective = production.static_training_config(worker, f.source)
        assert effective["evaluation"] == production.sampling.load_sources(f.source)[domain]["evaluation"]
        inventory = production.production_source_files(worker, root=f.source)
        assert production.EARLY_AUTHORIZATION_FILE in inventory
        assert previous in inventory
        assert "src/wheeled_tasks/chassis/heading_command.py" in inventory


def test_prepare_admission_preserves_real_origin_and_account_without_writes(early_frozen):
    f = early_frozen
    paths = [f.claim, f.state, f.run / "completion.json", *
             (f.run / production.EARLY_ORIGIN_DIRECTORY).glob("*.json")]
    before = {p: p.read_bytes() for p in paths}
    receipt = admitted(f)
    assert receipt["selection"]["checkpoint"] == f.rec
    assert receipt["selection"]["mode"] == "rebase_from_authorized_REC"
    assert receipt["selection"]["origin_recovery_actor_update"] == 14052
    assert receipt["component_clocks_at_source"]["component_actor_updates"] == dict(recovery=14052, speed=0)
    assert receipt["allowance"]["consumed_updates"] == 34000
    assert receipt["allowance"]["charged_critic_updates"] == 10
    assert receipt["entry_admission"]["record"]["domain_qualified"] is None
    assert production.authenticate_invocation(f.worker, arguments(f, receipt), f.source) == receipt
    assert before == {p: p.read_bytes() for p in paths}


def test_archive_keeps_old_rec_record_and_replays_immutable_origin(early_frozen):
    f = early_frozen
    receipt = admitted(f)
    archived = production.archive_legacy_domains(f.report, f.binding, receipt["entry_admission"])
    assert archived["domains"]["recovery"] == f.report["domains"]["recovery"]
    assert archived["authorized_REC_early_stop"] == receipt["entry_admission"]["record"]
    assert production.archive_legacy_domains(archived, f.binding, receipt["entry_admission"]) == archived
    archived["status"] = "running"
    write(f.state, archived)
    write(f.run / "completion.json", dict(status="running"))
    replay = admitted(f)
    assert replay["entry_admission"] == receipt["entry_admission"]
    assert replay["selection"] == receipt["selection"]


@pytest.mark.parametrize("change", ["running", "missing_snapshot", "changed_snapshot", "worker_completed"])
def test_first_admission_rejects_unsealed_or_nonstopped_evidence(early_frozen, change):
    f = early_frozen
    if change == "running":
        report = read(f.state)
        report["status"] = "running"
        write(f.state, report)
    elif change == "missing_snapshot":
        (f.run / production.EARLY_ORIGIN_DIRECTORY / "campaign_state.json").unlink()
    elif change == "changed_snapshot":
        write(f.run / production.EARLY_ORIGIN_DIRECTORY / "completion.json", dict(status="stopped"))
    else:
        path = Path(f.rec["path"]).parent / "completion.json"
        completion = read(path)
        completion["status"] = "completed"
        write(path, completion)
    with pytest.raises(ValueError):
        admitted(f)


def test_actual_latest_final_is_selected_without_domain_record_substitution(early_frozen):
    f = early_frozen
    latest = checkpoint(f, "recovery", count=25053, task=14053, lineage=24053, final=True, suffix="002")
    path = Path(latest["path"]).parent / "completion.json"
    completion = read(path)
    completion.update(status="stopped", checkpoint_sha256=latest["sha256"])
    write(path, completion)
    receipt = admitted(f)
    assert receipt["selection"]["checkpoint"] == latest
    assert receipt["selection"]["origin_recovery_actor_update"] == 14053
    assert read(f.state)["domains"]["recovery"] == f.report["domains"]["recovery"]


def test_newer_nonfinal_seal_cannot_be_hidden_by_older_final(early_frozen):
    checkpoint(early_frozen, "recovery", count=25053, task=14053, lineage=24053, suffix="002")
    with pytest.raises(ValueError, match="unique final"):
        admitted(early_frozen)


def test_v26_cached_recipe_still_rechecks_v25_bytes(early_frozen):
    f = early_frozen
    assert production.revised_recipe(f.source, version=26)
    path = f.source / production.artifact_files(25)["recipe"]
    original = path.read_bytes()
    changed = read(path)
    changed["domains"]["speed"]["patch"]["learning_rate"] = .5
    try:
        write(path, changed)
        with pytest.raises(ValueError, match="unchanged frozen v25"):
            production.revised_recipe(f.source, version=26)
    finally:
        path.write_bytes(original)
