"""Mixed locomotion changes coverage only, keeping clocks and charged learning."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

from wheeled_tasks.chassis import locomotion_revision as mixed
from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner
from train_chassis import resume_critic_warmup


def read(name):
    return json.loads((ROOT / name).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


SOURCE = read("contracts/v6_task_campaign_v9.json")
TARGET = read("contracts/v6_task_campaign_v10.json")
ORIGINAL = read(mixed.SPEED_FILE)
WORKER = read(mixed.WORKER_FILE)


def test_constant_pool_floor_keeps_all_129_groups_and_native_integer_symmetry():
    assert WORKER["skill_specs"] == ORIGINAL["skill_specs"]
    assert len(WORKER["scene_groups"]) == 129
    assert [g["name"] for g in WORKER["scene_groups"]] == [g["name"] for g in ORIGINAL["scene_groups"]]
    assert WORKER["behavior_pool_fractions"] == mixed.POOL_FRACTIONS
    flat = read(mixed.FLAT_FILE)
    assert set(flat["behavior_pool_membership"]) <= set(WORKER["behavior_pool_membership"])
    counts = mixed.validate_integer_allocation(WORKER)
    assert counts["pool_environment_counts"] == dict(stand_anchors=495, stand_continuous=328,
        height=205, basic_translation=624, basic_yaw=408, basic_transitions=612,
        high_motion=1008, high_transitions=416)
    assert WORKER["task_campaign"]["sampling_revision"]["allocation_clock"] == "constant_all_actor_updates"


def test_frozen_roles_worker_parents_endpoints_and_consumption_are_preserved():
    receipt = campaign.validate_plan_migration(SOURCE, TARGET, TARGET["revision_migration"]["run_directory"])
    assert receipt["actor_critic_optimizer"] == "preserved"
    assert receipt["task_clock"] == "restored" and receipt["consumption"] == "preserved_no_refund"
    assert SOURCE["domain_completion"] == TARGET["domain_completion"]
    for before, after in zip(SOURCE["domains"], TARGET["domains"], strict=True):
        assert before["parent_domain"] == after["parent_domain"]
        if before["name"] != "speed":
            assert before == after
            assert campaign.digest(ROOT / after["contract"]) == after["sha256"]
    assert len(runner.validate_contracts(TARGET)) == 30
    assert campaign.validate_task_resume(ORIGINAL, WORKER)["task_clock"] == "restored"
    assert campaign.validate_task_resume(WORKER, WORKER)["task_clock"] == "restored"
    result = campaign.account_updates(TARGET, {"speed/old": dict(domain="speed",
        consumed_updates=1560, actor_updates=1550)})
    assert result["consumed_updates"] == 2560 and result["actor_updates_consumed"] == 2530
    assert campaign.validate_task_budget(WORKER, ROOT)["total_ceiling"] == 121000


@pytest.mark.parametrize("mutation", ["reward", "physics", "clock", "exploration", "delay", "parent",
    "missing_group", "asymmetric_fraction", "wrong_membership", "sampling_theme", "new_binding"])
def test_exact_worker_patch_rejects_hidden_changes(mutation):
    worker = deepcopy(WORKER)
    if mutation == "reward":
        worker["precision_tracking"]["stationary_width_m_s"] = .25
    elif mutation == "physics":
        worker["physics_dt"] = .001
    elif mutation == "clock":
        worker["task_campaign"]["clock"] = "consumed_updates"
    elif mutation == "exploration":
        worker["initial_noise_std"] = .01
    elif mutation == "delay":
        worker["signal_delay"]["enabled"] = False
    elif mutation == "parent":
        worker["task_campaign"]["source_contract_sha256"] = "0" * 64
    elif mutation == "missing_group":
        worker["scene_groups"].pop()
    elif mutation == "asymmetric_fraction":
        worker["scene_groups"][0]["fraction"] *= 2
    elif mutation == "wrong_membership":
        worker["behavior_pool_membership"]["forward_05"] = "high_motion"
    elif mutation == "sampling_theme":
        worker["task_campaign"]["theme"] = "high_speed_only"
    else:
        worker["task_campaign"]["revision_migration"]["refund"] = True
    with pytest.raises(ValueError):
        campaign.validate_worker_revision(ORIGINAL, worker)


@pytest.fixture
def revision(tmp_path, monkeypatch):
    for row in SOURCE["domains"]:
        path = tmp_path / row["contract"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / row["contract"], path)
    for name in (mixed.WORKER_FILE, mixed.DESIGN_FILE, "contracts/v6_task_campaign_speed_expectations_v6.json",
                 "contracts/v6_task_campaign_budget_v1.json", "contracts/v6_task_campaign_budget_v2.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    source, target = deepcopy(SOURCE), deepcopy(TARGET)
    for plan in (source, target):
        plan["revision_migration"]["run_directory"] = str(tmp_path / "train")
    write(tmp_path / "contracts/v6_task_campaign_v9.json", source)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    return dict(root=tmp_path, source=source, target=target)


def reseal(fixture):
    root, target = fixture["root"], fixture["target"]
    target["revision_migration"]["source_plan_sha256"] = campaign.digest(root / "contracts/v6_task_campaign_v9.json")
    for row in target["domains"]:
        row["sha256"] = campaign.digest(root / row["contract"])
    write(root / "contracts/v6_task_campaign_v10.json", target)
    manifest = read("contracts/v6_task_campaign_migration_v10.json")
    manifest.update(source_plan_sha256=campaign.digest(root / "contracts/v6_task_campaign_v9.json"),
        target_plan_sha256=campaign.digest(root / "contracts/v6_task_campaign_v10.json"),
        run_directory=str(root / "train"), workers=[dict(source_file=a["contract"], source_sha256=a["sha256"],
            target_file=b["contract"], target_sha256=b["sha256"])
            for a, b in zip(fixture["source"]["domains"], target["domains"], strict=True)])
    write(root / target["revision_migration"]["manifest_file"], manifest)


@pytest.mark.parametrize("mutation", ["endpoint", "total_budget", "parent", "sampling_pool", "other_worker",
    "worker_reward", "design_reward", "design_role_missing"])
def test_resealed_plan_or_design_cannot_widen_sampling_authorization(revision, mutation):
    root, target = revision["root"], revision["target"]
    if mutation == "endpoint":
        target["domain_completion"]["schedule"]["end_actor_updates"]["speed"] = 3564
    elif mutation == "total_budget":
        target["total_ceiling"] += 10
    elif mutation == "parent":
        target["domains"][2]["parent_domain"] = "recovery"
    elif mutation == "sampling_pool":
        target["domains"][2]["sampling_pools"][0]["fraction"] = .13
    elif mutation == "other_worker":
        path = root / target["domains"][0]["contract"]
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = root / (mixed.WORKER_FILE if mutation == "worker_reward" else mixed.DESIGN_FILE)
        value = json.loads(path.read_text())
        if mutation == "worker_reward":
            value["height_l1_weight"] = 123
        elif mutation == "design_reward":
            value["stages"]["speed"]["roles"]["train"]["contract"]["height_l1_weight"] = 123
        else:
            del value["stages"]["speed"]["roles"]["confirmation"]
        write(path, value)
        if mutation != "worker_reward":
            worker_path = root / mixed.WORKER_FILE
            worker = json.loads(worker_path.read_text())
            worker["design_preflight"]["sha256"] = campaign.digest(path)
            write(worker_path, worker)
    reseal(revision)
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(revision["source"], target, root / "train")


def test_distribution_change_charges_critic_once_fault_resume_does_not_repeat_it():
    old_infos = dict(contract_sha256=mixed.SPEED_SHA, task_campaign=dict(domain="speed"),
        successful_updates_total=17896, task_actor_update=1552, critic_warmup_remaining=0)
    new_sha = campaign.digest(ROOT / mixed.WORKER_FILE)
    assert resume_critic_warmup(WORKER, old_infos, contract_sha256=new_sha, task_change=True) == 10
    same = dict(old_infos, contract_sha256=new_sha, critic_warmup_remaining=4)
    assert resume_critic_warmup(WORKER, same, contract_sha256=new_sha, task_change=True) == 4
    assert resume_critic_warmup(WORKER, dict(same, critic_warmup_remaining=0),
        contract_sha256=new_sha, task_change=True) == 0


def test_v10_cannot_skip_its_v9_predecessor():
    with pytest.raises(ValueError, match="consecutive"):
        campaign.validate_plan_migration(read("contracts/v6_task_campaign_v8.json"), TARGET,
            TARGET["revision_migration"]["run_directory"])
