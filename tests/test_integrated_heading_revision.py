"""Two frozen joint graphs coexist; only the named new train role uses heading."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_integrated_production import admitted, arguments, frozen, read, static_bundle
from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.heading_command import configure_heading


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def heading_frozen(frozen):
    f = deepcopy(frozen)
    before = {file: (f.source / file).read_bytes() for file in f.bundle["files"]}
    bundle = production.write_bundle(f.source, source_root=str(f.source), run_root=str(f.run), version=25)
    assert before == {file: (f.source / file).read_bytes() for file in before}
    f.bundle = bundle
    f.plan = read(f.source / production.artifact_files(25)["plan"])
    f.binding = production.binding_for(f.plan, f.source, f.run)
    f.manifest = f.binding.read()
    for domain, _ in lineage.CHAIN:
        f.contracts[domain] = f.source / production.worker_file(domain, version=25)
    f.worker = read(f.contracts["speed"])
    f.entry = lineage.admit_entry(f.binding, f.rec, f.rec_record)
    return f


def test_existing_frozen_v24_recipe_remains_exact():
    assert production.revised_recipe(ROOT) == read(ROOT / production.RECIPE_FILE)
    assert "scut_heading_command" not in production.revised_recipe(ROOT)["domains"]["speed"]["patch"]


def test_original_v24_inventory_excludes_new_heading_module(frozen):
    assert "src/wheeled_tasks/chassis/heading_command.py" not in production.production_source_files(
        frozen.worker, root=frozen.source)


def test_new_named_graph_enables_only_training_heading(heading_frozen):
    f = heading_frozen
    assert campaign.validate_plan(f.plan) is None
    production.validate_migration(f.baseline, f.plan, f.run, f.source)
    for domain, _ in lineage.CHAIN:
        worker = read(f.contracts[domain])
        effective = production.static_training_config(worker, f.source)
        heading = configure_heading(effective, f.source)
        assert len(heading["ordinary_names"]) == 24
        assert effective["scut_heading_command"]["enabled"] is True
        assert effective["evaluation"] == production.sampling.load_sources(f.source)[domain]["evaluation"]
        inventory = production.production_source_files(worker, root=f.source)
        assert production.artifact_files(25)["plan"] in inventory
        assert production.PLAN_FILE not in inventory
        assert "src/wheeled_tasks/chassis/heading_command.py" in inventory


def test_v25_full_state_source_admission_and_original_account(heading_frozen):
    f = heading_frozen
    before = {p: p.read_bytes() for p in (f.claim, f.state)}
    receipt = admitted(f)
    assert receipt["selection"]["checkpoint"] == f.rec
    assert receipt["allowance"]["consumed_updates"] == 34000
    assert receipt["allowance"]["charged_critic_updates"] == 10
    assert production.authenticate_invocation(f.worker, arguments(f, receipt), f.source) == receipt
    archived = production.archive_legacy_domains(f.report, f.binding, f.entry)
    assert archived["integrated_continuation"]["lineage_id"] == "source027-joint-heading-v1"
    assert production.archive_legacy_domains(archived, f.binding, f.entry) == archived
    assert before == {p: p.read_bytes() for p in before}


def test_receipt_cannot_select_another_version(heading_frozen):
    f = heading_frozen
    receipt = admitted(f)
    old_worker = read(f.source / production.worker_file("speed"))
    with pytest.raises(ValueError, match="campaign file differs"):
        production.authenticate_receipt(old_worker, receipt, f.source)
    wrong = deepcopy(receipt)
    wrong["campaign_plan"]["file"] = production.PLAN_FILE
    with pytest.raises(ValueError, match="campaign file differs"):
        production.authenticate_receipt(f.worker, wrong, f.source)
    conflict = deepcopy(f.plan)
    conflict["integrated_continuation"]["lineage_id"] = production.LINEAGE_ID
    with pytest.raises(ValueError, match="conflicting"):
        production.validate_plan(conflict, f.source)


def test_new_version_requires_its_own_explicit_source_root(static_bundle):
    with pytest.raises(ValueError, match="own explicit frozen source"):
        production.write_bundle(static_bundle.source, version=25)
    for alias in (production.DEPLOY_SOURCE_ROOT + "/", production.DEPLOY_SOURCE_ROOT + "/../source_integrated_v24_20261008"):
        with pytest.raises(ValueError, match="own explicit frozen source"):
            production.write_bundle(static_bundle.source, source_root=alias, version=25)
    for invalid in (True, 27, "25"):
        with pytest.raises(ValueError):
            production.artifact_files(invalid)
