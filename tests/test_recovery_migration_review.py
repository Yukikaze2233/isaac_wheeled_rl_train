"""Independent failure injection for the complete recovery migration boundary."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO

from test_new_asset_blocks import argument, seal
from test_new_asset_preflight import load_full_entry
from wheeled_tasks.chassis.full_curriculum import stage_contract
from wheeled_tasks.chassis.policy_transfer import resume_budget, transfer_actor_state, verify_learning_state_restore
from wheeled_tasks.chassis.preflight import design_preflight
from wheeled_tasks.chassis.response_migration import (
    RECOVERY_CURRICULUM_MIGRATION, assess_response_retention, response_checkpoint_provenance,
    validate_response_resume,
)

ROOT = Path(__file__).resolve().parents[1]


def pair():
    configs = []
    for name in ("step", "recovery"):
        plan = json.loads((ROOT / f"contracts/v6_{name}_curriculum_training_v1.json").read_text())
        configs.append(stage_contract({}, plan, plan["stages"][0], 20000))
    return configs


def reports():
    source, target = pair()
    result = []
    for config, seed in ((source, 290929), (source, 290930), (target, 290929), (target, 290930)):
        settings = deepcopy(config["evaluation"])
        settings["seed"] = seed
        cases = {}
        for case in settings["cases"]:
            recovery = bool(case.get("recovery_case"))
            checks = {"all_requested_episodes_accounted": True, "has_post_warmup_samples": True, "mechanics": True}
            if recovery:
                checks.update(recovery_episode_evidence=True, recovery_no_failure=False,
                              recovery_blend_and_single_handover=False, recovery_physical_success=False)
            else:
                checks.update(height=True, velocity=True, survival=True)
            cases[case["name"]] = {"checks": checks, "passed": all(checks.values()), "requested_episodes": 4}
        result.append({"status": "evaluated", "deterministic_actor": True, "random_pushes": False,
            "case_resampling": False, "episodes_per_case": 4, "settings": settings,
            "contract_sha256": "b" * 64 if config is source else "c" * 64,
            "coverage": "full_manifest", "evaluated_case_names": list(cases),
            "candidates": [{"checkpoint_sha256": "a" * 64, "cases": cases}]})
    return result


def assess(rows):
    return assess_response_retention(*rows, required_cases=pair()[1]["evaluation"]["promotion_case_names"],
                                     migration_kind=RECOVERY_CURRICULUM_MIGRATION)


def test_failed_recovery_chain_is_not_mislabelled_as_precision_error():
    result = assess(reports())
    assert len(result["added_unconfirmed_case_names"]) == 54
    assert not result["extra_rehearsal_required"]  # future skills do not block unchanged S1
    for name in result["added_unconfirmed_case_names"]:
        assert name in result["basic_or_mechanical_check_failures"]
        assert name not in result["precision_not_yet_met"]
        assert name in result["recovery_chain_failures"]


@pytest.mark.parametrize("target_report", (2, 3))
def test_old_302_coverage_cannot_satisfy_final_322_contract(target_report):
    rows = reports()
    report = rows[target_report]
    removed = report["evaluated_case_names"][-20:]
    report["evaluated_case_names"] = report["evaluated_case_names"][:-20]
    for name in removed:
        del report["candidates"][0]["cases"][name]
    with pytest.raises(ValueError, match="coverage|manifest"):
        assess(rows)


@pytest.mark.parametrize("corruption", ("different_checkpoint", "same_seed", "missing_primary", "changed_check"))
def test_four_way_review_rejects_nonpaired_or_inconsistent_reports(corruption):
    rows = reports()
    if corruption == "different_checkpoint":
        rows[3]["candidates"][0]["checkpoint_sha256"] = "z" * 64
    elif corruption == "same_seed":
        rows[1]["settings"]["seed"] = rows[0]["settings"]["seed"]
    elif corruption == "missing_primary":
        rows[2]["status"] = "running"
    else:
        rows[2]["candidates"][0]["cases"]["forward_05"]["checks"]["invented"] = True
    with pytest.raises(ValueError):
        assess(rows)


@pytest.mark.parametrize("mutation", ("drop_case", "reorder", "old_definition", "early_promotion", "refund_budget", "change_pd"))
def test_s1_resume_fails_closed_on_case_abi_or_budget_changes(mutation):
    source, target = pair()
    settings = target["evaluation"]
    if mutation == "drop_case":
        settings["cases"].pop(0)
    elif mutation == "reorder":
        settings["cases"][0], settings["cases"][1] = settings["cases"][1], settings["cases"][0]
    elif mutation == "old_definition":
        settings["cases"][0]["height_mae_m_max"] += .1
    elif mutation == "early_promotion":
        settings["promotion_case_names"].append(settings["cases"][-1]["name"])
    elif mutation == "refund_budget":
        target["total_updates"] = 1100
    else:
        target["v5_control"]["leg_kp"] = 80.
    with pytest.raises(ValueError):
        validate_response_resume(source, target)


@pytest.mark.parametrize("field,value", [("enabled", False), ("learning_mask", "all_actor_samples"),
    ("reference_dt", .001), ("wheel_balance", {"enabled": False}), ("profiles_sha256", "0" * 64)])
def test_changed_recovery_enablement_clock_mask_balance_or_geometry_fails_preflight(field, value):
    _, target = pair()
    target["recovery_training"][field] = value
    control = json.loads((ROOT / target["control_math_source"]).read_text())
    with pytest.raises(ValueError):
        design_preflight(target, control, ROOT, stage="foundation", role="train", seed=290927, num_envs=20000)


def test_source_seal_actor_abi_adam_and_consumption_survive_named_migration(tmp_path):
    source, target = pair()
    original = tmp_path / "source.json"
    original.write_text(json.dumps(source))
    checkpoint = tmp_path / "source/model_final.pt"
    seal(checkpoint, original, consumed=338, course=218, offset=0)
    infos = {"successful_updates_total": 338, "learning_lineage_updates": 218,
             "contract_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
             "control_math_sha256": source["control_math_sha256"],
             "asset_manifest_sha256": source["asset_manifest_sha256"],
             "training_transitions": 338 * 480000, "batch_transitions": 480000}
    _, controls, receipt = response_checkpoint_provenance(checkpoint, infos, target, ROOT)
    assert receipt["kind"] == "explicit_recovery_curriculum_resume"
    state = {"mlp.0.weight": torch.ones(256, 35)}
    actor, proof = transfer_actor_state(state, source, target, response_controls=controls)
    assert actor is state and proof["actor_parameters"] == "unchanged"
    budget = resume_budget(infos, 338, 480000)
    assert budget["parent_updates"] == 338 and budget["learning_lineage_updates"] == 218
    assert budget["parent_training_transitions"] == 162240000
    with pytest.raises(ValueError, match="Consumed updates"):
        resume_budget(infos, 218, 480000)
    with pytest.raises(ValueError, match="update count"):
        response_checkpoint_provenance(checkpoint, {**infos, "successful_updates_total": 218}, target, ROOT)

    original_alg = PPO(torch.nn.Linear(2, 1), torch.nn.Linear(2, 1), None, learning_rate=.00037)
    sum(p.square().sum() for module in (original_alg.actor, original_alg.critic) for p in module.parameters()).backward()
    original_alg.optimizer.step()
    saved = deepcopy(original_alg.save())
    restored = PPO(torch.nn.Linear(2, 1), torch.nn.Linear(2, 1), None)
    restored.load(saved, load_cfg=None, strict=True)
    assert verify_learning_state_restore(restored, saved) == {
        "actor_exact": True, "critic_exact": True, "optimizer_exact": True}
    assert restored.learning_rate == .00037


def test_actual_recovery_plan_reaches_full_entry_without_revision_branch_drop(tmp_path, monkeypatch):
    module = load_full_entry()
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    args = SimpleNamespace(contract=ROOT / "contracts/v6_recovery_curriculum_training_v1.json",
        run_dir=tmp_path / "full", transfer=None, resume=tmp_path / "sealed.pt", start_stage="S1",
        resume_response_change=True, consumed_updates=338, global_actor_update_offset=0,
        resume_learning_rate_scale=1.9509221155311695,
        max_runtime_seconds=100., updates=None, num_envs=20000, seed=290927, device="cpu", publish_state=False)
    runner = module.FullCurriculum(args)
    calls = []

    def execute(command, log, directory):
        calls.append(command)
        config = json.loads(Path(argument(command, "--contract")).read_text())
        assert config["response_migration"] == RECOVERY_CURRICULUM_MIGRATION
        assert len(config["evaluation"]["cases"]) == 322
        directory.mkdir()
        (directory / "completion.json").write_text(json.dumps({"status": "stopped",
            "successful_updates": 338, "actor_updates_consumed": 338}))
        return 0

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.run() == 0
    assert len(calls) == 1 and "--resume-response-change" in calls[0]
    assert argument(calls[0], "--consumed-updates") == "338"
    assert float(argument(calls[0], "--resume-learning-rate-scale")) == 1.9509221155311695
    assert argument(calls[0], "--updates") == "1000"
    assert argument(calls[0], "--num-envs") == "20000"
    assert runner.report["actor_updates_consumed"] == 338
