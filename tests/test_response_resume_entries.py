"""Exercise real block entry decisions with simulated Isaac subprocess outputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_new_asset_blocks import Campaign, argument, contract, seal
from wheeled_tasks.chassis.response_migration import RESPONSE_MIGRATION, response_checkpoint_provenance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResponseCampaign(Campaign):
    def __init__(self, tmp_path, monkeypatch, *, spent=3, ledger_spent=3):
        old_control = tmp_path / "old_control.json"
        old_control.write_text(json.dumps({"actuators": {
            "leg": {"armature": 0., "kp": 160., "kd": 2.5, "effort_limit": 40.},
            "wheel": {"armature": 0., "kd": .6, "gear_ratio": 15.8, "effort_limit": 4.5}}}))
        new_control = tmp_path / "new_control.json"
        content = json.loads(old_control.read_text())
        content["actuators"]["leg"]["armature"] = .02
        new_control.write_text(json.dumps(content))
        self.source_config = {**contract(), "manual_context35": True,
            "actor_dim": 35, "critic_dim": 81, "action_dim": 6, "physics_dt": .001, "policy_dt": .02,
            "critic_warmup_updates": 0, "total_updates": 1000, "asset_manifest_sha256": "d" * 64,
            "control_math_source": str(old_control), "control_math_sha256": digest(old_control)}
        target = deepcopy(self.source_config)
        target.update(response_migration=RESPONSE_MIGRATION, actuator_response={"enabled": True, "tau_s": .005},
                      control_math_source=str(new_control), control_math_sha256=digest(new_control))
        frozen = tmp_path / "contract.json"
        frozen.write_text(json.dumps(target, indent=2))
        original = tmp_path / "original" / "contract.json"
        original.parent.mkdir()
        original.write_text(json.dumps(self.source_config))
        source = original.parent / "block_000/model_final.pt"
        seal(source, original, consumed=2, course=2)
        (original.parent / "retention_state.json").write_text(json.dumps({
            "contract_sha256": digest(original), "protected_case_names": ["stand", "jump_03"],
            "confirmed_jump_bins_m": [.03], "actor_updates_consumed": ledger_spent}))
        super().__init__(tmp_path, monkeypatch, [{"stand"}, {"stand", "jump_03", "jump_06"}],
                         updates=7, resume=source)
        self.args.resume_response_change = True
        self.args.consumed_updates = spent

    def execute(self, command, log_path, training_directory=None):
        if training_directory is not None:
            return super().execute(command, log_path, training_directory)
        directory = Path(argument(command, "--output"))
        directory.mkdir()
        checkpoint = Path(argument(command, "--checkpoint"))
        frozen_path = Path(argument(command, "--contract"))
        settings = deepcopy(json.loads(frozen_path.read_text())["evaluation"])
        seed = int(argument(command, "--seed")) if "--seed" in command else settings["seed"]
        settings["seed"] = seed
        self.order.append(directory.name)
        self.evaluations.append((directory.name, checkpoint, argument(command, "--role"), seed))
        if directory.name.startswith("response_source_"):
            passed_names = {"stand", "jump_03"}
        elif directory.name.startswith("baseline"):
            passed_names = {"stand"}
        else:
            passed_names = self.outcomes[int(directory.name.split("_")[1])]
        cases = {case["name"]: {"passed": case["name"] in passed_names,
            "requested_episodes": 4, "height_mae_m": .002 if case["name"] in passed_names else .009,
            "checks": {"mechanics": True, "height": case["name"] in passed_names}, "anchor": False}
            for case in settings["cases"]}
        passed = all(case["passed"] for case in cases.values())
        candidate = {"passed": passed, "anchor_passed": False, "cases": cases,
            "rank_lower_is_better": [int(not passed), len(cases) - len(passed_names), 1.],
            "checkpoint_sha256": digest(checkpoint)}
        report = {"status": "evaluated", "deterministic_actor": True, "random_pushes": False,
            "case_resampling": False, "episodes_per_case": 4, "settings": settings,
            "contract_sha256": digest(frozen_path), "candidates": [candidate]}
        (directory / "evaluation.json").write_text(json.dumps(report))
        return 0


def test_response_entry_retests_same_checkpoint_then_preserves_clocks_and_no_refund(tmp_path, monkeypatch):
    run = ResponseCampaign(tmp_path, monkeypatch)
    old_contract_bytes = run.args.resume.with_suffix(".contract.json").read_bytes()
    code, report = run.run()
    assert code == 0 and report["status"] == "stage_accepted"
    assert run.order[:5] == ["response_source_primary", "response_source_confirmation",
        "baseline_evaluation", "baseline_confirmation", "train_0"]
    assert {checkpoint for _, checkpoint, _, _ in run.evaluations[:4]} == {run.args.resume}
    assert report["response_retention_comparison"]["new_confirmed_case_names"] == ["stand"]
    assert report["response_recovery_case_names"] == ["jump_03"]
    first, second = run.calls
    assert "--resume-response-change" in first and "--resume-response-change" not in second
    assert argument(first, "--consumed-updates") == "3"
    assert argument(first, "--stage-actor-update") == "2"
    assert argument(first, "--actor-updates-consumed") == "3"
    assert argument(first, "--confirmed-jump-bins-json") == "[]"
    assert argument(second, "--stage-actor-update") == "4"
    assert report["successful_updates"] == report["actor_updates_consumed"] == 7
    assert report["global_budget_actor_updates"] == 107
    assert report["latest_response_review"]["remaining_recovery_case_names"] == []
    assert report["latest_response_review"]["updates_consumed_since_model_change"] == 4
    assert run.args.resume.with_suffix(".contract.json").read_bytes() == old_contract_bytes
    assert report["retention_learning_checkpoint"].endswith("block_001/model_final.pt")


@pytest.mark.parametrize("spent,ledger_spent,error", [
    (None, 3, "explicit consumed-update"), (2, 3, "precede the authenticated retention ledger"),
])
def test_response_entry_refuses_missing_or_refunded_consumption_before_evaluation(
        tmp_path, monkeypatch, spent, ledger_spent, error):
    run = ResponseCampaign(tmp_path, monkeypatch, spent=spent, ledger_spent=ledger_spent)
    code, report = run.run()
    assert code == 1 and error in report["error"]
    assert run.calls == [] and run.evaluations == []


def test_response_worker_provenance_rejects_control_tamper_and_update_mismatch(tmp_path, monkeypatch):
    run = ResponseCampaign(tmp_path, monkeypatch)
    infos = {"successful_updates_total": 2, "contract_sha256": digest(run.args.resume.with_suffix(".contract.json")),
             "control_math_sha256": run.source_config["control_math_sha256"],
             "asset_manifest_sha256": run.source_config["asset_manifest_sha256"]}
    _, controls, receipt = response_checkpoint_provenance(run.args.resume, infos, run.blocks.contract, tmp_path)
    assert receipt["changed_control_fields"] == ["actuators.leg.armature"]
    assert controls[1]["actuators"]["leg"]["kp"] == 160.
    with pytest.raises(ValueError, match="update count"):
        response_checkpoint_provenance(run.args.resume, {**infos, "successful_updates_total": 3}, run.blocks.contract, tmp_path)
    path = Path(run.source_config["control_math_source"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="source control file hash"):
        response_checkpoint_provenance(run.args.resume, infos, run.blocks.contract, tmp_path)


def test_frozen_evidence_file_is_authenticated_without_claiming_calibration_pass(tmp_path, monkeypatch):
    run = ResponseCampaign(tmp_path, monkeypatch)
    infos = {"successful_updates_total": 2, "contract_sha256": digest(run.args.resume.with_suffix(".contract.json")),
             "control_math_sha256": run.source_config["control_math_sha256"],
             "asset_manifest_sha256": run.source_config["asset_manifest_sha256"]}
    evidence = tmp_path / "calibration.json"
    evidence.write_text(json.dumps({"passed": False, "status": "candidate_not_qualified"}))
    target = deepcopy(run.blocks.contract)
    target["actuator_response_evidence"] = {"file": evidence.name, "sha256": digest(evidence)}
    _, _, receipt = response_checkpoint_provenance(run.args.resume, infos, target, tmp_path)
    assert receipt["evidence_file_identity_verified"] is True
    assert receipt["plant_qualification_evaluated"] is False
    assert receipt["plant_qualification"] == "requires_separate_bag_and_Isaac_replay_receipt"
    evidence.write_text(json.dumps({"passed": True}))
    with pytest.raises(ValueError, match="evidence file hash mismatch"):
        response_checkpoint_provenance(run.args.resume, infos, target, tmp_path)


def test_response_recovery_cannot_load_a_checkpoint_from_the_old_plant(tmp_path, monkeypatch):
    run = ResponseCampaign(tmp_path, monkeypatch)
    run.args.run_dir.mkdir()
    run.blocks.report.update(successful_updates=3, retention_learning_checkpoint=str(run.args.resume))
    with pytest.raises(ValueError, match="differs from the current stage"):
        run.blocks.recover_regression(run.blocks.evaluation_settings, {"lost_parent_passes": ["stand"]})


def test_full_entry_passes_response_flag_and_consumption_to_selected_stage(tmp_path, monkeypatch):
    from test_new_asset_preflight import load_full_entry
    module = load_full_entry()
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    args = SimpleNamespace(contract=module.ROOT / "contracts/v6_new_asset_training_v1.json",
        run_dir=tmp_path / "full", transfer=None, resume=tmp_path / "sealed.pt", start_stage="S1",
        resume_response_change=True, consumed_updates=312, global_actor_update_offset=0,
        max_runtime_seconds=100., updates=None, num_envs=20000, seed=290927, device="cpu",
        publish_state=False)
    runner = module.FullCurriculum(args)
    calls = []

    def execute(command, log, directory):
        calls.append(command)
        directory.mkdir()
        (directory / "completion.json").write_text(json.dumps({"status": "stopped",
            "successful_updates": 312, "actor_updates_consumed": 312}))
        return 0

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.run() == 0
    assert len(calls) == 1 and "--resume-response-change" in calls[0]
    assert argument(calls[0], "--consumed-updates") == "312"
    assert argument(calls[0], "--global-actor-update-offset") == "0"
    assert runner.report["actor_updates_consumed"] == 312


def test_actor_evaluation_requires_response_control_diff_before_changed_control_source(tmp_path, monkeypatch):
    import torch
    from wheeled_tasks.chassis.policy_transfer import transfer_actor_state
    run = ResponseCampaign(tmp_path, monkeypatch)
    state = {"mlp.0.weight": torch.ones(256, 35)}
    with pytest.raises(ValueError, match="control_math_source"):
        transfer_actor_state(state, run.source_config, run.blocks.contract)
    controls = [json.loads(Path(config["control_math_source"]).read_text())
                for config in (run.source_config, run.blocks.contract)]
    actor, receipt = transfer_actor_state(state, run.source_config, run.blocks.contract, response_controls=controls)
    assert actor is state and receipt["actor_parameters"] == "unchanged"
    controls[1]["actuators"]["leg"]["kp"] = 80.
    with pytest.raises(ValueError, match="control semantics"):
        transfer_actor_state(state, run.source_config, run.blocks.contract, response_controls=controls)
