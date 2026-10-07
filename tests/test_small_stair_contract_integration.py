"""Exercise native entry and budget paths for the terrain-only v23 revision."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from wheeled_tasks.chassis import small_stair_revision as recipe
from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner
import train_chassis as train


def read(root, name):
    return json.loads((root / name).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "contracts").mkdir()
    outputs = (recipe.PLAN_FILE, recipe.WORKER_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE)
    for path in (ROOT / "contracts").iterdir():
        if str(path.relative_to(ROOT)) not in outputs:
            (root / "contracts" / path.name).symlink_to(path, target_is_directory=path.is_dir())
    for path in ROOT.iterdir():
        if path.name != "contracts":
            (root / path.name).symlink_to(path, target_is_directory=path.is_dir())
    recipe.freeze_contracts(root)
    for module in (campaign, runner, train):
        monkeypatch.setattr(module, "ROOT", root)
    return root


def test_all_native_entries_accept_exact_v21_to_v23_and_reject_other_worker_change(frozen):
    source, target = read(frozen, recipe.SOURCE_PLAN_FILE), read(frozen, recipe.PLAN_FILE)
    receipt = campaign.validate_plan_migration(source, target, target["revision_migration"]["run_directory"])
    assert receipt["schema"] == "task-plan-revision-v23"
    assert receipt["consumption"] == "preserved_no_refund"
    assert source["domain_completion"] == target["domain_completion"]
    assert len(runner.validate_contracts(target)) == 30
    for old, new in zip(source["domains"], target["domains"], strict=True):
        if old["name"] != "terrain":
            assert old == new
    changed = deepcopy(target)
    changed["domains"][2]["sampling_pools"][0]["fraction"] += .01
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(source, changed, changed["revision_migration"]["run_directory"])


@pytest.mark.parametrize("source_file", [recipe.SOURCE_WORKER_FILE, recipe.OWN_SOURCE_WORKER_FILE])
def test_real_own_learning_source_and_complete_source_audit(frozen, source_file):
    source, target = read(frozen, source_file), read(frozen, recipe.WORKER_FILE)
    binding = campaign.validate_task_resume(source, target)
    assert binding["actor_critic_optimizer"] == "preserved"
    assert binding["task_clock"] == "restored"
    checksum = recipe.digest(frozen / recipe.WORKER_FILE)
    infos = dict(contract_sha256=recipe.digest(frozen / source_file),
                 successful_updates_total=2206, critic_warmup_remaining=0,
                 task_campaign=source["task_campaign"])
    assert train.resume_critic_warmup(target, infos, contract_sha256=checksum, task_change=True) == 10
    for remaining in (10, 5, 0):
        resumed = dict(infos, contract_sha256=checksum, critic_warmup_remaining=remaining)
        assert train.resume_critic_warmup(target, resumed,
                                        contract_sha256=checksum, task_change=True) == remaining
    names = train.scut_curriculum_source_files(target, root=frozen)
    assert len(names) == len(set(names))
    assert {"src/wheeled_tasks/chassis/small_stair_revision.py",
            recipe.PLAN_FILE, recipe.WORKER_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE} <= set(names)
    assert all((frozen / name).is_file() for name in names)
    bad = deepcopy(target)
    bad["task_campaign"].pop("small_stair_revision_migration")
    with pytest.raises(ValueError):
        train.scut_curriculum_source_files(bad, root=frozen)


@pytest.mark.parametrize("crash_after_claim", [False, True])
def test_existing_account_and_reopening_survive_migration_and_crash_replay(frozen, tmp_path, monkeypatch, crash_after_claim):
    from test_terrain_runout_contract_integration import account_job
    import test_terrain_runout_contract_integration as inherited_tests

    monkeypatch.setattr(inherited_tests, "recipe", recipe)
    job, before, records = account_job(frozen, tmp_path / "run", monkeypatch)
    original_write = runner.write_json
    if crash_after_claim:
        def interrupted(path, value):
            if Path(path) == job.root / "campaign_state.json":
                raise RuntimeError("simulated power loss")
            original_write(path, value)
        monkeypatch.setattr(runner, "write_json", interrupted)
        with pytest.raises(RuntimeError, match="power loss"):
            job.open_account()
        monkeypatch.setattr(runner, "write_json", original_write)
        job.report, job.records = {}, {}
    job.open_account()
    account = json.loads(job.claim.read_text())
    state = json.loads((job.root / "campaign_state.json").read_text())
    assert account["attempts"] == records
    assert account["consumed_updates"] == before["consumed_updates"]
    assert account["actor_updates_consumed"] == before["actor_updates_consumed"]
    assert state["domains"]["terrain"] == before["domains"]["terrain"]
    assert state["evaluations"] == before["evaluations"]
    assert state["contract_sha256"] == recipe.digest(frozen / recipe.PLAN_FILE)
