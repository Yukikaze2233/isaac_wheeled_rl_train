"""Mock the new campaign's real checkpoint/dual-seed orchestration boundary."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(monkeypatch):
    spec = importlib.util.spec_from_file_location("new_asset_blocks", ROOT / "scripts/run_chassis_blocks.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    monkeypatch.setattr(value.signal, "signal", lambda *_: None)
    return value


def contract():
    def jump(name, apex, kind="jump", assisted=False):
        return {"name": name, "task": "jump", "jump_assistance_enabled": assisted,
                "skill": {"kind": kind, "semantics": {"jump_apex_delta_m": apex}}}
    cases = [{"name": "stand", "task": "survive"}, jump("jump_03", .03), jump("jump_06", .06),
             jump("running_jump_03", .03, "running_jump"), jump("assisted_10", .10, assisted=True)]
    names = [case["name"] for case in cases]
    return {"contract_id": "new-campaign-test", "curriculum_stage": "S5", "learning_rate": 1e-4,
        "num_steps_per_env": 24, "new_asset_curriculum": {"enabled": True,
            "jump_curriculum": {"bins_m": [.03, .06, .10]}},
        "evaluation": {"seed": 290929, "confirmation_seed": 290930, "block_updates": 2,
            "cases": cases, "retention_case_names": names, "promotion_case_names": ["stand", "jump_06"],
            "protected_case_names": [], "cumulative_retention": True, "mode": "gate",
            "minimum_updates": 2, "consecutive_passes_required": 1,
            "confirmation_on_candidate_or_regression": True, "regression_patience": 2,
            "regression_recovery": {"max_retries": 1, "learning_rate_factor": .5}}}


def seal(path, frozen_contract, *, consumed, course, offset=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"actor-course-{course}-consumed-{consumed}")
    path.with_suffix(".contract.json").write_bytes(frozen_contract.read_bytes())
    path.with_suffix(".checkpoint.json").write_text(json.dumps({
        "schema": "chassis-learning-checkpoint-v1", "sealed": True,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(frozen_contract.read_bytes()).hexdigest(),
        "learning_updates": consumed, "stage_actor_update": course,
        "actor_updates_consumed": consumed, "global_actor_update_offset": offset}))


class Campaign:
    def __init__(self, tmp_path, monkeypatch, outcomes, *, updates=6, resume=None, run_name="run",
                 baseline_names=("stand",), fail_block=None, confirmed="[]", protected="[]"):
        self.module = module(monkeypatch)
        self.path = tmp_path / "contract.json"
        if not self.path.exists():
            self.path.write_text(json.dumps(contract(), indent=2))
        self.frozen = self.path.read_bytes()
        self.args = SimpleNamespace(contract=self.path, run_dir=tmp_path / run_name, transfer=None,
            resume=resume, max_runtime_seconds=100., updates=updates, stage="jump", num_envs=8,
            seed=290927, device="cpu", publish_state=False, global_actor_update_offset=100,
            confirmed_jump_bins_json=confirmed, protected_cases_json=protected)
        self.blocks = self.module.TrainingBlocks(self.args)
        self.outcomes, self.baseline_names, self.fail_block = outcomes, baseline_names, fail_block
        self.calls, self.evaluations, self.order = [], [], []
        monkeypatch.setattr(self.blocks, "execute", self.execute)

    def execute(self, command, log_path, training_directory=None):
        arg = lambda flag, default=None: command[command.index(flag) + 1] if flag in command else default
        log_path.write_text("mock simulator\n")
        if training_directory is not None:
            index = len(self.calls)
            self.calls.append(command)
            self.order.append(f"train_{index}")
            training_directory.mkdir()
            before = int(arg("--consumed-updates", "0"))
            course = int(arg("--stage-actor-update"))
            offset = int(arg("--global-actor-update-offset"))
            successful = 1 if self.fail_block == index else int(arg("--updates"))
            initial = training_directory / "initial_state/model_final.pt"
            seal(initial, self.path, consumed=before, course=course, offset=offset)
            final = training_directory / "model_final.pt"
            seal(final, self.path, consumed=before + successful, course=course + successful, offset=offset)
            for name in ("policy.onnx", "policy.onnx.json", "agent_config.json"):
                (training_directory / name).write_text("mock artifact")
            scale = float(arg("--learning-rate-scale", "1"))
            result = {"status": "failed" if self.fail_block == index else "completed",
                "parent_updates": before, "successful_updates": successful,
                "actor_updates_in_block": successful, "learning_rate": 1e-4 * scale,
                "checkpoint_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
                "export": {"verified": True}, "error": "bounded worker failure"}
            (training_directory / "completion.json").write_text(json.dumps(result))
            return 1 if self.fail_block == index else 0
        directory = Path(arg("--output"))
        directory.mkdir()
        checkpoint = Path(arg("--checkpoint"))
        role = arg("--role")
        self.order.append(directory.name)
        self.evaluations.append((directory.name, checkpoint, role, arg("--seed")))
        if directory.name.startswith("baseline"):
            passed_names = set(self.baseline_names)
        else:
            stage_index = int(directory.name.split("_")[1])
            result = self.outcomes[stage_index]
            passed_names = set(result[1 if role == "confirmation" else 0] if isinstance(result, tuple) else result)
        cases = {case["name"]: {"passed": case["name"] in passed_names,
            "checks": {"mechanics": True}, "anchor": False} for case in self.blocks.evaluation_settings["cases"]}
        passed = all(case["passed"] for case in cases.values())
        candidate = {"passed": passed, "anchor_passed": False, "cases": cases,
            "rank_lower_is_better": [int(not passed), len(cases) - len(passed_names), 1.]}
        (directory / "evaluation.json").write_text(json.dumps({"candidates": [candidate]}))
        return 0

    def run(self):
        code = self.blocks.run()
        assert self.path.read_bytes() == self.frozen
        assert self.blocks.contract == json.loads(self.frozen)
        return code, self.blocks.report


def argument(command, flag):
    return command[command.index(flag) + 1]


def test_scratch_evaluates_authentic_initial_state_before_first_candidate(tmp_path, monkeypatch):
    run = Campaign(tmp_path, monkeypatch, [{"stand", "jump_06"}], updates=2)
    code, report = run.run()
    assert code == 0 and report["status"] == "stage_accepted"
    assert run.order == ["train_0", "baseline_evaluation", "baseline_confirmation",
                         "evaluation_000", "evaluation_000_confirmation"]
    initial = run.args.run_dir / "block_000/initial_state/model_final.pt"
    assert all(row[1] == initial for row in run.evaluations[:2])
    assert run.evaluations[2][1] != initial
    assert run.evaluations[0][2] == "baseline" and run.evaluations[1][2] == "confirmation"
    assert report["baseline_learning_updates"] == 0
    assert report["actor_updates_consumed"] == 2
    assert report["global_budget_actor_updates"] == 102
    assert json.loads((run.args.run_dir / "retention_state.json").read_text())["protected_case_names"] == ["jump_06", "stand"]


def test_jump_bins_need_both_seeds_and_only_unassisted_cold_requests(tmp_path, monkeypatch):
    run = Campaign(tmp_path, monkeypatch, [
        ({"stand", "jump_03", "assisted_10", "running_jump_03"}, {"stand", "assisted_10", "running_jump_03"}),
        {"stand", "jump_03", "assisted_10", "running_jump_03"}, {"stand", "jump_03"}])
    code, report = run.run()
    assert code == 0
    assert json.loads(argument(run.calls[1], "--confirmed-jump-bins-json")) == []
    assert json.loads(argument(run.calls[2], "--confirmed-jump-bins-json")) == [.03]
    assert report["confirmed_jump_bins_m"] == [.03]
    assert run.blocks.contract["evaluation"]["protected_case_names"] == []


def test_rollback_restores_course_clock_without_refunding_consumption_or_confirmation(tmp_path, monkeypatch):
    run = Campaign(tmp_path, monkeypatch,
        [{"stand", "jump_03"}, set(), set(), {"stand", "jump_03", "jump_06"}], updates=8)
    code, report = run.run()
    assert code == 0 and report["status"] == "stage_accepted"
    retry = run.calls[3]
    assert argument(retry, "--resume").endswith("block_000/model_final.pt")
    assert argument(retry, "--consumed-updates") == "6"
    assert argument(retry, "--actor-updates-consumed") == "6"
    assert argument(retry, "--stage-actor-update") == "2"
    assert argument(retry, "--global-actor-update-offset") == "100"
    assert json.loads(argument(retry, "--confirmed-jump-bins-json")) == [.03]
    assert report["successful_updates"] == report["actor_updates_consumed"] == 8
    assert report["confirmed_jump_bins_m"] == [.03, .06]


def test_resume_restores_external_ledger_and_does_not_relabel_candidate_as_baseline(tmp_path, monkeypatch):
    first = Campaign(tmp_path, monkeypatch, [{"stand", "jump_03"}], updates=2)
    assert first.run()[0] == 0
    checkpoint = first.args.run_dir / "block_000/model_final.pt"
    second = Campaign(tmp_path, monkeypatch, [{"stand", "jump_03", "jump_06"}], updates=4,
                      resume=checkpoint, run_name="resumed")
    code, report = second.run()
    assert code == 0 and report["status"] == "stage_accepted"
    assert all(not name.startswith("baseline") for name, *_ in second.evaluations)
    assert argument(second.calls[0], "--stage-actor-update") == "2"
    assert argument(second.calls[0], "--actor-updates-consumed") == "2"
    assert json.loads(argument(second.calls[0], "--confirmed-jump-bins-json")) == [.03]
    assert report["actor_updates_consumed"] == 4


def test_failed_partial_block_charges_updates_and_resume_reads_ledger_budget(tmp_path, monkeypatch):
    first = Campaign(tmp_path, monkeypatch, [{"stand", "jump_03"}], updates=4, fail_block=1)
    code, report = first.run()
    assert code == 1 and "bounded worker failure" in report["error"]
    assert report["successful_updates"] == report["actor_updates_consumed"] == 3
    second = Campaign(tmp_path, monkeypatch, [{"stand", "jump_03", "jump_06"}], updates=5,
        resume=first.args.run_dir / "block_000/model_final.pt", run_name="resumed")
    code, report = second.run()
    assert code == 0
    assert argument(second.calls[0], "--consumed-updates") == "3"
    assert argument(second.calls[0], "--actor-updates-consumed") == "3"
    assert argument(second.calls[0], "--stage-actor-update") == "2"
    assert report["actor_updates_consumed"] == 5


def test_nonzero_checkpoint_cannot_masquerade_as_scratch_baseline(tmp_path, monkeypatch):
    run = Campaign(tmp_path, monkeypatch, [])
    run.args.run_dir.mkdir()
    candidate = tmp_path / "candidate.pt"
    seal(candidate, run.path, consumed=40, course=40)
    with pytest.raises(ValueError, match="update-zero"):
        run.blocks.establish_scratch_baseline(candidate)
    assert not run.evaluations


@pytest.mark.parametrize("protected,confirmed", [("[\"unknown\"]", "[]"), ("[]", "[0.17]")])
def test_runtime_metadata_cannot_invent_cases_or_jump_bins(tmp_path, monkeypatch, protected, confirmed):
    with pytest.raises(ValueError):
        Campaign(tmp_path, monkeypatch, [], protected=protected, confirmed=confirmed)


@pytest.mark.parametrize("requested,flew,peak,status", [
    (4, 0, 0., "zero_unassisted_airtime"),
    (4, 1, .08, "unassisted_airtime_observed"),
    (4, 0, .04, "unassisted_airtime_observed"),
    (4, 1, .05999999865889549, "unassisted_airtime_observed"),
    (0, 0, 0., "no_valid_jump_requests"),
    (4, None, .08, "diagnostic_data_missing"),
    (4, 0, float("nan"), "diagnostic_data_missing"),
    (4, 0, .08, "diagnostic_data_missing"),
])
def test_80_consumed_update_diagnostic_uses_observed_unassisted_airtime_once(
        tmp_path, monkeypatch, requested, flew, peak, status):
    run = Campaign(tmp_path, monkeypatch, [])
    run.args.run_dir.mkdir()
    blocks = run.blocks
    blocks.evaluation_settings["cases"] = [case for case in blocks.evaluation_settings["cases"]
                                            if case["name"] == "jump_03"]
    blocks.report.update(actor_updates_consumed=79, baseline_confirmation={"cases": {}},
                         successful_updates=79)
    checkpoint = tmp_path / "candidate.pt"
    seal(checkpoint, run.path, consumed=80, course=60)
    candidate = {"cases": {"jump_03": {"jump_episodes_requested": requested,
        "jump_episodes_airborne_60ms": flew, "jump_air_time_peak": peak, "observed_frames": 400}}}
    assert blocks.inspect_zero_airtime(candidate, checkpoint, tmp_path / "evaluation") is None
    blocks.report["actor_updates_consumed"] = blocks.report["successful_updates"] = 80
    receipt = blocks.inspect_zero_airtime(candidate, checkpoint, tmp_path / "evaluation")
    assert receipt["status"] == status
    assert receipt["stage_actor_update"] == 60 and receipt["actor_updates_consumed"] == 80
    assert receipt["automatic_stop"] is False and receipt["additional_updates_granted"] == 0
    assert blocks.report["successful_updates"] == 80 and blocks.stop_requested is False
    assert blocks.inspect_zero_airtime(candidate, checkpoint, tmp_path / "evaluation") is None
    ledger = json.loads((run.args.run_dir / "retention_state.json").read_text())
    assert ledger["zero_airtime_checkpoint"]["status"] == status
    assert json.loads((run.args.run_dir / "zero_airtime_diagnostic.json").read_text())["status"] == status
