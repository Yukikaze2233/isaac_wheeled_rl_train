"""Authenticate the enabled jump repair without changing the frozen campaign."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tarfile

import pytest

from wheeled_tasks.chassis import jump_landing_revision as landing
from wheeled_tasks.chassis import task_campaign as campaign
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight, validate_design_contract, validate_runtime_state

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_task_campaign as runner
from chassis_remote_job import sha, validate_source_revision
from train_chassis import resume_critic_warmup


def read(name):
    return json.loads((ROOT / name).read_text())


SOURCE = read(landing.SOURCE_PLAN_FILE)
TARGET = read(landing.PLAN_FILE)
BEFORE = read(landing.SOURCE_FILE)
WORKER = read(landing.WORKER_FILE)
STEPJUMP = read(landing.STEPJUMP_FILE)
DESIGN = read(landing.DESIGN_FILE)


def test_exact_v10_to_v12_patch_preserves_other_workers_parent_tree_budgets_and_endpoints():
    receipt = campaign.validate_plan_migration(SOURCE, TARGET, TARGET["revision_migration"]["run_directory"])
    assert receipt["consumption"] == "preserved_no_refund"
    assert receipt["task_clock"] == "restored" and receipt["actor_critic_optimizer"] == "preserved"
    assert SOURCE["domain_completion"] == TARGET["domain_completion"]
    assert TARGET["target_num_envs"] == 4096 and TARGET["total_ceiling"] == 121000
    for before, after in zip(SOURCE["domains"], TARGET["domains"], strict=True):
        assert before["parent_domain"] == after["parent_domain"] and after["updates"] == 20000
        if after["name"] != "jump":
            assert before == after
    expected = deepcopy(BEFORE)
    landing.apply_landing_revision(expected)
    expected["design_preflight"] = deepcopy(WORKER["design_preflight"])
    assert expected == WORKER
    assert campaign.validate_task_resume(BEFORE, WORKER)["task_clock"] == "restored"
    assert campaign.validate_task_budget(WORKER, ROOT)["total_ceiling"] == 121000
    assert (WORKER["physics_dt"], WORKER["policy_dt"], WORKER["actor_dim"], WORKER["critic_dim"]) == (.005, .02, 35, 81)
    assert WORKER["actuator_response"] == BEFORE["actuator_response"]
    assert WORKER["control_model_status"] == BEFORE["control_model_status"]
    assert len(runner.validate_contracts(TARGET)) == 30


def test_five_independent_roles_enable_real_broad_and_precision_costs_preserving_fixed_evaluation():
    original = read(BEFORE["design_preflight"]["file"])
    assert DESIGN == landing.revised_design(ROOT)
    for role, row in DESIGN["stages"]["jump"]["roles"].items():
        old = original["stages"]["jump"]["roles"][role]
        assert row["contract"]["evaluation"] == old["contract"]["evaluation"]
        assert row["contract"]["evaluation_exact_cases"] == old["contract"]["evaluation_exact_cases"]
        assert row["contract"]["scene_groups"] == old["contract"]["scene_groups"]
        assert row["runtime"]["precision_tracking"] is False  # inherited legacy path
        assert row["runtime"]["modules"]["jump_landing"] is True
        assert row["runtime"]["jump_landing"] == landing.runtime_expectation(row["contract"])
        settings = row["runtime"]["jump_landing"]["settings"]
        assert settings["enabled"] and settings["zero_tracking_baselines"]
        assert settings["window_seconds"] == 3. and settings["absorption_seconds"] == .3
        for axis in ("height", "velocity", "lateral", "yaw", "orientation", "angular_velocity",
                     "vertical_velocity", "down_speed", "impact", "precontact_wheel_velocity"):
            assert settings[axis]["broad_weight"] > 0 and settings[axis]["fine_weight"] > 0
            assert 0 < settings[axis]["fine_scale"] <= settings[axis]["broad_scale"]
        assert settings["wheel_radius_m"] == .06
        assert settings["precontact_wheel_velocity"]["fine_scale"] == .5
        assert settings["impact_bodyweight_threshold"] == 2.


def test_changed_reward_charges_critic_once_and_fault_resume_keeps_remaining_actor_clock():
    infos = dict(contract_sha256=landing.SOURCE_SHA, task_campaign=dict(domain="jump"),
        task_actor_update=2340, critic_warmup_remaining=0, successful_updates_total=22417)
    target_sha = campaign.digest(ROOT / landing.WORKER_FILE)
    assert resume_critic_warmup(WORKER, infos, contract_sha256=target_sha, task_change=True) == 10
    infos.update(contract_sha256=target_sha, critic_warmup_remaining=4)
    assert resume_critic_warmup(WORKER, infos, contract_sha256=target_sha, task_change=True) == 4
    infos["critic_warmup_remaining"] = 0
    assert resume_critic_warmup(WORKER, infos, contract_sha256=target_sha, task_change=True) == 0
    assert infos["task_actor_update"] == 2340


def test_unchanged_stepjump_accepts_exact_revised_jump_parent_without_rebinding_contract():
    assert landing.validate_landing_parent_alias(WORKER, STEPJUMP, ROOT)
    result = campaign.validate_task_resume(WORKER, STEPJUMP)
    assert result["task_clock"] == "new_domain_zero" and result["actor_critic_optimizer"] == "preserved"
    assert STEPJUMP["task_campaign"]["source_contract_sha256"] == landing.SOURCE_SHA
    assert campaign.digest(ROOT / landing.STEPJUMP_FILE) == landing.STEPJUMP_SHA


@pytest.mark.parametrize("fault", ["source_reward", "source_precision", "target_reward", "target_parent"])
def test_stepjump_parent_alias_rejects_any_undeclared_source_or_child_change(fault):
    source, target = deepcopy(WORKER), deepcopy(STEPJUMP)
    if fault == "source_reward":
        source["termination_event_cost"] += 1
    elif fault == "source_precision":
        source["jump_landing"]["height"]["fine_weight"] = 0
    elif fault == "target_reward":
        target["termination_event_cost"] += 1
    else:
        target["task_campaign"]["source_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        campaign.validate_task_resume(source, target)


@pytest.mark.parametrize("fault", ["physics", "entropy", "flat_reward", "sampling", "release_criterion",
    "missing_feature", "disabled", "zero_precision", "window", "SCUT_speed_gate", "wrong_predecessor"])
def test_worker_revision_rejects_changes_outside_the_frozen_landing_patch(fault):
    worker = deepcopy(WORKER)
    if fault == "physics":
        worker["physics_dt"] = .001
    elif fault == "entropy":
        worker["initial_noise_std"] = .01
    elif fault == "flat_reward":
        worker["dense_tracking"]["stationary"]["broad_weight"] += 1
    elif fault == "sampling":
        worker["scene_groups"][0]["fraction"] *= 2
    elif fault == "release_criterion":
        worker["task_semantics"]["success_hold_seconds"] = .3
    elif fault == "missing_feature":
        del worker["jump_landing"]
    elif fault == "disabled":
        worker["jump_landing"]["enabled"] = False
    elif fault == "zero_precision":
        worker["jump_landing"]["height"]["fine_weight"] = 0
    elif fault == "window":
        worker["jump_landing"]["window_seconds"] = 30
    elif fault == "SCUT_speed_gate":
        worker["jump_landing"]["precontact_min_speed_m_s"] = 1.
    else:
        worker["task_campaign"]["revision_migration"]["source_contract_file"] = "contracts/v6_task_campaign_speed_worker_v11.json"
    with pytest.raises(ValueError):
        campaign.validate_worker_revision(BEFORE, worker)


@pytest.mark.parametrize("fault", ["endpoint", "parent", "budget", "v11_predecessor", "missing_role",
    "design_precision", "runtime_disabled", "runtime_settings"])
def test_resealed_plan_or_design_cannot_widen_the_named_revision(tmp_path, monkeypatch, fault):
    shutil.copytree(ROOT / "contracts", tmp_path / "contracts")
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    source, target, worker, design = map(deepcopy, (SOURCE, TARGET, WORKER, DESIGN))
    manifest = read(landing.MANIFEST_FILE)
    if fault == "endpoint":
        target["domain_completion"]["schedule"]["end_actor_updates"]["jump"] += 1
    elif fault == "parent":
        target["domains"][4]["parent_domain"] = "terrain"
    elif fault == "budget":
        target["total_ceiling"] += 10
    elif fault == "v11_predecessor":
        source = read("contracts/v6_task_campaign_v11.json")
        manifest.update(source_plan_file="contracts/v6_task_campaign_v11.json",
            source_plan_sha256=campaign.digest(tmp_path / "contracts/v6_task_campaign_v11.json"))
        target["revision_migration"]["source_plan_sha256"] = manifest["source_plan_sha256"]
    elif fault == "missing_role":
        del design["stages"]["jump"]["roles"]["confirmation"]
    elif fault == "design_precision":
        design["stages"]["jump"]["roles"]["train"]["contract"]["jump_landing"]["height"]["fine_weight"] = 0
    elif fault == "runtime_disabled":
        design["stages"]["jump"]["roles"]["train"]["runtime"]["modules"]["jump_landing"] = False
    else:
        design["stages"]["jump"]["roles"]["train"]["runtime"]["jump_landing"]["settings"]["height"]["fine_weight"] = 0
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value, indent=2) + "\n")
        return campaign.digest(tmp_path / name)
    worker["design_preflight"]["sha256"] = write(landing.DESIGN_FILE, design)
    target["domains"][4]["sha256"] = write(landing.WORKER_FILE, worker)
    manifest["target_plan_sha256"] = write(landing.PLAN_FILE, target)
    for pin, row in zip(manifest["workers"], target["domains"], strict=True):
        pin["target_sha256"] = row["sha256"]
    write(landing.MANIFEST_FILE, manifest)
    with pytest.raises(ValueError):
        campaign.validate_plan_migration(source, target, manifest["run_directory"])


@pytest.mark.parametrize("fault", ["disabled", "zero_precision", "missing_runtime", "runtime_settings"])
def test_independent_static_and_observed_runtime_preflight_assert_actual_precision(fault):
    config, design = deepcopy(WORKER), deepcopy(DESIGN)
    control = load_chassis_control(ROOT / config["control_math_source"])
    receipt = design_preflight(config, control, ROOT, stage="jump", role="train", seed=290927, num_envs=4096)
    actual = deepcopy(receipt["runtime_expected"])
    if fault == "disabled":
        config["jump_landing"]["enabled"] = False
    elif fault == "zero_precision":
        config["jump_landing"]["height"]["fine_weight"] = 0
    elif fault == "missing_runtime":
        del actual["jump_landing"]
    else:
        actual["jump_landing"]["settings"]["height"]["fine_weight"] = 0
    with pytest.raises(ValueError):
        if fault in ("disabled", "zero_precision"):
            validate_design_contract(config, design, control, stage="jump", role="train", seed=290927, num_envs=4096)
        else:
            validate_runtime_state(receipt["runtime_expected"], actual)


def test_remote_v12_receipt_delegates_exact_v10_predecessor_and_keeps_all_command_options(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    for directory, name, plan in ((old, landing.SOURCE_PLAN_FILE, SOURCE), (new, landing.PLAN_FILE, TARGET)):
        (directory / "contracts").mkdir(parents=True)
        (directory / name).write_text(json.dumps(plan))
        (directory / "scripts").mkdir()
        (directory / "scripts/run_task_campaign.py").write_text("# Never executed by authentication test.\n")
    old_cmd = [sys.executable, "-B", str(old / "scripts/run_task_campaign.py"), "--contract",
        str(old / landing.SOURCE_PLAN_FILE), "--num-envs", "4096", "--device", "cuda:0", "--research",
        "--max-runtime-seconds", "15552000", "--run-dir", str(tmp_path / "train")]
    previous = dict(pid=2147483647, command=old_cmd)
    (tmp_path / "job.json").write_text(json.dumps(previous))
    command = list(old_cmd)
    command[2] = str(new / "scripts/run_task_campaign.py")
    command[4] = str(new / landing.PLAN_FILE)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for path in sorted(new.rglob("*")):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(new)), recursive=False)
    receipt = dict(schema="task-plan-revision-v12", previous_job_sha256=sha(tmp_path / "job.json"),
        command=command, source_commit="a" * 40, old_source_root=str(old), new_source_root=str(new),
        new_contract_relative_path=landing.PLAN_FILE, old_contract_sha256=sha(old / landing.SOURCE_PLAN_FILE),
        new_contract_sha256=sha(new / landing.PLAN_FILE), source_archive=str(archive), source_archive_sha256=sha(archive))
    path = tmp_path / "revision.json"
    path.write_text(json.dumps(receipt))
    calls = []
    def validate(before, after, run_directory):
        assert before == SOURCE and after == TARGET and run_directory == tmp_path / "train"
        calls.append(True)
    monkeypatch.setattr(campaign, "validate_plan_migration", validate)
    assert validate_source_revision(tmp_path, command, previous, path) == receipt
    assert calls == [True]
    altered = list(command)
    altered[altered.index("--num-envs") + 1] = "20000"
    receipt["command"] = altered
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="only runner and contract"):
        validate_source_revision(tmp_path, altered, previous, path)
