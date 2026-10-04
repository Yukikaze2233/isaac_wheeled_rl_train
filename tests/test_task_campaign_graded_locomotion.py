"""Authenticate the command-only revision and reject hidden budget/physics changes."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis import graded_locomotion_revision as graded
from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner
from train_chassis import resume_critic_warmup


def read(name):
    return json.loads((ROOT / name).read_text())


SOURCE = read("contracts/v6_task_campaign_v10.json")
TARGET = read("contracts/v6_task_campaign_v11.json")
BEFORE = read(graded.SOURCE_FILE)
WORKER = read(graded.WORKER_FILE)


def test_exact_frozen_patch_preserves_population_rewards_clocks_and_all_other_workers():
    result = campaign.validate_plan_migration(SOURCE, TARGET, TARGET["revision_migration"]["run_directory"])
    assert result["consumption"] == "preserved_no_refund" and result["task_clock"] == "restored"
    assert SOURCE["domain_completion"] == TARGET["domain_completion"]
    for a, b in zip(SOURCE["domains"], TARGET["domains"], strict=True):
        if a["name"] != "speed":
            assert a == b
        else:
            assert a["sampling_pools"] == b["sampling_pools"] and a["parent_domain"] == b["parent_domain"]
    for key in ("scene_groups", "skill_specs", "behavior_pool_fractions", "behavior_pool_membership",
                "new_asset_curriculum", "motion_limits", "dense_tracking", "precision_tracking",
                "signal_delay", "v5_control", "actuator_response", "policy_architecture"):
        assert WORKER[key] == BEFORE[key]
    assert len(runner.validate_contracts(TARGET)) == 30
    assert campaign.validate_task_resume(BEFORE, WORKER)["task_clock"] == "restored"
    assert campaign.validate_task_budget(WORKER, ROOT)["total_ceiling"] == 121000


def test_changed_sampler_charges_critic_once_and_preserves_actor_clock_on_fault_resume():
    infos = dict(contract_sha256=graded.SOURCE_SHA, task_campaign=dict(domain="speed"),
        task_actor_update=2600, critic_warmup_remaining=0, successful_updates_total=18954)
    sha = campaign.digest(ROOT / graded.WORKER_FILE)
    assert resume_critic_warmup(WORKER, infos, contract_sha256=sha, task_change=True) == 10
    infos.update(contract_sha256=sha, critic_warmup_remaining=4)
    assert resume_critic_warmup(WORKER, infos, contract_sha256=sha, task_change=True) == 4
    infos["critic_warmup_remaining"] = 0
    assert resume_critic_warmup(WORKER, infos, contract_sha256=sha, task_change=True) == 0
    assert infos["task_actor_update"] == 2600


@pytest.mark.parametrize("mode", ["matching_predecessor", "wrong_origin", "same_v11_fault_resume"])
def test_native_worker_binds_first_revision_to_sealed_actor_origin_then_restores_later_clocks(tmp_path, mode):
    origin = WORKER["locomotion_difficulty"]["origin_actor_update"]
    source = WORKER if mode == "same_v11_fault_resume" else BEFORE
    clock = origin + 13 if mode == "same_v11_fault_resume" else origin - int(mode == "wrong_origin")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"Fixture learning state; no tensor loading in the clock validator")
    sidecar = checkpoint.with_suffix(".contract.json")
    sidecar.write_text(json.dumps(source, indent=2) + "\n")
    seal = dict(schema="chassis-learning-checkpoint-v1", sealed=True,
        checkpoint_sha256=campaign.digest(checkpoint), contract_sha256=campaign.digest(sidecar),
        learning_updates=18954, task_actor_update=clock, stage_actor_update=15900,
        actor_updates_consumed=18874)
    checkpoint.with_suffix(".checkpoint.json").write_text(json.dumps(seal))
    args = SimpleNamespace(resume_task_change=True, resume=checkpoint, num_envs=4096,
        seed=290927, learning_rate_scale=1., global_actor_update_offset=0, block_index=0,
        updates=100, consumed_updates=18954, task_actor_update=clock,
        stage_actor_update=15900, actor_updates_consumed=18874)
    if mode == "wrong_origin":
        with pytest.raises(ValueError, match="sealed predecessor actor clock"):
            campaign.validate_worker(WORKER, args)
    else:
        campaign.validate_worker(WORKER, args)


@pytest.mark.parametrize("fault", ["reward", "physics", "entropy", "missing_difficulty", "disabled",
    "height_range", "quota", "clock", "origin_mismatch", "skipped_predecessor"])
def test_worker_revision_rejects_undeclared_changes(fault):
    worker = deepcopy(WORKER)
    if fault == "reward":
        worker["dense_tracking"]["stationary"]["broad_weight"] = 123
    elif fault == "physics":
        worker["physics_dt"] = .001
    elif fault == "entropy":
        worker["entropy_coef"] = 0.
    elif fault == "missing_difficulty":
        del worker["locomotion_difficulty"]
    elif fault == "disabled":
        worker["locomotion_difficulty"]["enabled"] = False
    elif fault == "height_range":
        worker["locomotion_difficulty"]["keyframes"][0]["motion_height_range_m"] = [.23, .43]
    elif fault == "quota":
        worker["scene_groups"][0]["fraction"] *= 2
    elif fault == "clock":
        worker["locomotion_difficulty"]["clock"] = "consumed_updates"
    elif fault == "origin_mismatch":
        worker["locomotion_difficulty"]["origin_actor_update"] += 1
    else:
        worker["task_campaign"]["revision_migration"]["source_contract_file"] = "contracts/v6_task_campaign_speed_worker_v6.json"
    with pytest.raises(ValueError):
        campaign.validate_worker_revision(BEFORE, worker)


@pytest.mark.parametrize("fault", ["endpoint", "parent", "sampling", "manifest_origin", "missing_role", "runtime_enabled", "role_clock"])
def test_resealed_plan_or_independent_design_cannot_widen_revision(tmp_path, monkeypatch, fault):
    shutil.copytree(ROOT / "contracts", tmp_path / "contracts")
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    source, target = deepcopy(SOURCE), deepcopy(TARGET)
    design = read(graded.DESIGN_FILE)
    worker = deepcopy(WORKER)
    manifest = read("contracts/v6_task_campaign_migration_v11.json")
    if fault == "endpoint":
        target["domain_completion"]["schedule"]["end_actor_updates"]["speed"] += 1
    elif fault == "parent":
        target["domains"][2]["parent_domain"] = "recovery"
    elif fault == "sampling":
        target["domains"][2]["sampling_pools"][0]["fraction"] += .01
    elif fault == "manifest_origin":
        manifest["origin_actor_update"] += 1
    elif fault == "missing_role":
        del design["stages"]["speed"]["roles"]["confirmation"]
    elif fault == "runtime_enabled":
        design["stages"]["speed"]["roles"]["train"]["runtime"]["locomotion_difficulty"]["enabled"] = False
    else:
        design["stages"]["speed"]["roles"]["quick"]["contract"]["locomotion_difficulty"]["clock"] = "consumed_updates"
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value, indent=2) + "\n")
        return campaign.digest(tmp_path / name)
    worker["design_preflight"]["sha256"] = write(graded.DESIGN_FILE, design)
    target["domains"][2]["sha256"] = write(graded.WORKER_FILE, worker)
    manifest["target_plan_sha256"] = write("contracts/v6_task_campaign_v11.json", target)
    for pin, row in zip(manifest["workers"], target["domains"], strict=True):
        pin["target_sha256"] = row["sha256"]
    write("contracts/v6_task_campaign_migration_v11.json", manifest)
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(source, target, manifest["run_directory"])
