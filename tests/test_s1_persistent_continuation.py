"""Continuous S1 learning consumes the remaining budget without weakening gates."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis import continuation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_chassis_continuation import Continuation, validate_contracts, validate_plan
from run_chassis_remediation import RemediationStudy


def source():
    return json.loads((ROOT / "contracts/v6_s1_A_fixed_v1.json").read_text())


def test_frozen_plan_preserves_protection_and_all_runtime_entry_roles():
    plan = json.loads((ROOT / "contracts/v6_s1_persistent_recovery_v1.json").read_text())
    assert len(validate_contracts(plan)) == 5
    assert len(plan["protected_case_names"]) == 39
    for field, value in (("source_consumed_updates", 738), ("study_update_ceiling", 223),
                         ("inherited_recovery_attempts", []), ("automatic_promotion", True),
                         ("protected_case_names", plan["protected_case_names"][1:])):
        changed = {**plan, field: value}
        with pytest.raises(ValueError):
            validate_plan(changed)


def test_persistent_worker_keeps_A_state_but_charges_all_discarded_work(monkeypatch):
    revision = continuation.PERSISTENT_REVISION
    config = continuation.worker_contract(source(), {}, revision)
    receipt = continuation.validate_resume(source(), config)
    assert receipt["actor_critic_optimizer"] == "preserved"
    assert receipt["sampling"] == "physical_start_stop_delay_0_to_8_seconds"
    assert config["checkpoint_first_update"] == 779
    assert config["checkpoint_interval"] == 10
    assert config["landing_tracking"]["passive_semantics"] == "finite_task_cost_v1"
    assert config["new_asset_curriculum"]["start_stop_start_delay_seconds"] == [0., 8.]
    args = SimpleNamespace(resume_reward_change=True, resume="A.pt", consumed_updates=778,
        actor_updates_consumed=758, stage_actor_update=368, updates=222,
        global_actor_update_offset=0, learning_rate_scale=1., num_envs=20000, seed=290927, block_index=0)
    monkeypatch.setattr(continuation, "digest", lambda _: continuation.CHECKPOINT_SHA)
    continuation.validate_worker(config, args)
    for key, value in (("consumed_updates", 738), ("actor_updates_consumed", 718),
                       ("stage_actor_update", 408), ("updates", 223), ("updates", 40)):
        invalid = deepcopy(args)
        setattr(invalid, key, value)
        with pytest.raises(ValueError):
            continuation.validate_worker(config, invalid)
    for count in (0, 40, 80, 221, 222):
        assert continuation.charged_updates([dict(successful_updates=count, actor_updates_in_block=count)],
                                            revision) == 778 + count
    for records in ([dict(successful_updates=223, actor_updates_in_block=223)],
                    [dict(successful_updates=40, actor_updates_in_block=40)] * 2):
        with pytest.raises(ValueError):
            continuation.charged_updates(records, revision)


def test_persistent_resume_rejects_undeclared_contract_changes():
    target = continuation.worker_contract(source(), {}, continuation.PERSISTENT_REVISION)
    for key, value in (("start_stop_start_delay_seconds", [0., 20.]), ("joint_sampling", True)):
        changed = deepcopy(target)
        changed["new_asset_curriculum"][key] = value
        with pytest.raises(ValueError, match="undeclared contract fields"):
            continuation.validate_resume(source(), changed)
    for field in ("dense_tracking", "evaluation", "actuator_response", "usb_transport"):
        changed = deepcopy(target)
        changed[field]["unexpected"] = True
        with pytest.raises(ValueError, match="undeclared contract fields"):
            continuation.validate_resume(source(), changed)


def task(tmp_path):
    result = object.__new__(Continuation)
    result.root = tmp_path
    result.stop_requested = False
    result.deadline = time.monotonic() + 10
    result.plan = dict(contract_id=continuation.PERSISTENT_REVISION, stage_ceiling=1000,
        source_consumed_updates=778, source_actor_consumed_updates=758,
        source_stage_actor_update=368, source_learning_lineage_updates=378,
        source_checkpoint="A.pt", evaluation_seeds=[290929, 290930],
        protected_case_names=["stand_230mm"], stages=[dict(name="A_persistent_recovery", updates=222,
        contract="contracts/v6_s1_A_fixed_v1.json")])
    result.args = SimpleNamespace(device="cpu")
    result.report = dict(successful_updates=1000)
    return result


def test_single_command_runs_whole_remaining_budget(tmp_path):
    job = task(tmp_path)
    command = job.train_command(job.plan["stages"][0])
    assert command[command.index("--updates") + 1] == "222"
    assert command[command.index("--consumed-updates") + 1] == "778"
    assert command[command.index("--stage-actor-update") + 1] == "368"


@pytest.mark.parametrize("short_pass,full_pass", [(False, False), (False, True), (True, False), (True, True)])
def test_budget_exit_always_full_audits_and_never_accepts_a_failed_gate(tmp_path, monkeypatch,
                                                                    short_pass, full_pass):
    job = task(tmp_path)
    names = {case["name"] for case in source()["evaluation"]["cases"]}
    baseline = dict(cases={name: dict(passed=True) for name in names})
    short, full = deepcopy(baseline), deepcopy(baseline)
    short["cases"]["stand_230mm"]["passed"] = short_pass
    full["cases"]["stand_230mm"]["passed"] = full_pass
    evaluated, copied = [], []
    def evaluate(recipe, seed, *, full=False):
        evaluated.append((seed, full))
        return deepcopy(full_result)
    full_result = full
    monkeypatch.setattr(job, "evaluate", evaluate)
    monkeypatch.setattr(job, "copy_atomic", lambda path, name: copied.append(name))
    monkeypatch.setattr("run_chassis_continuation.digest", lambda _: "checked-sha")
    job.finish([baseline, baseline], {"A_persistent_recovery": [short, short]})
    assert evaluated == [(290929, True), (290930, True)]
    accepted = short_pass and full_pass
    assert job.report["status"] == ("stage_accepted" if accepted else "budget_exhausted_gate_pending")
    assert job.report["remaining_S1_updates"] == 0
    assert copied == (["accepted_policy.pt"] if accepted else [])
    selection = job.artifact_selection()
    assert selection["accepted_stage"] == ("S1" if accepted else None)
    assert selection["deployment_checkpoint"] is None
    assert not selection["automatic_promotion"]


def test_final_gate_and_observer_share_exclusive_evaluation_lock(tmp_path, monkeypatch):
    first, second = task(tmp_path), task(tmp_path)
    active = []
    def evaluate(self, recipe, seed, *, full=False):
        assert not active, "Two evaluations allocated their scene together"
        active.append(seed)
        time.sleep(.025)
        active.clear()
        return seed
    monkeypatch.setattr(RemediationStudy, "evaluate", evaluate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: item[0].evaluate({}, item[1]),
                               [(first, 290929), (second, 290930)]))
    assert results == [290929, 290930]
    first.stop_requested = True
    with pytest.raises(InterruptedError, match="lock wait"):
        first.evaluate({}, 290929)


def test_latest_budget_authentication_uses_explicit_checkpoint_provenance(tmp_path):
    job = task(tmp_path)
    checkpoint = tmp_path / "A_landing_repair/train/model_final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sealed spent-budget state")
    contract = checkpoint.with_suffix(".contract.json")
    contract.write_text("{}")
    sha = continuation.digest(checkpoint)
    seal = dict(schema="chassis-learning-checkpoint-v1", sealed=True, checkpoint_sha256=sha,
        contract_sha256=continuation.digest(contract), learning_updates=778, actor_updates_consumed=758,
        learning_lineage_updates=418, stage_actor_update=408)
    checkpoint.with_suffix(".checkpoint.json").write_text(json.dumps(seal))
    ledger = dict(consumed_updates=778, actor_updates_consumed=758)
    completion = dict(status="continuation_review_required", successful_updates=778,
        actor_updates_consumed=758, inherited_recovery_attempts=[dict(consumed_updates=498)],
        protected_case_names=["stand_230mm"], arms={"A_landing_repair": dict(checkpoint_sha256=sha)})
    for filename, value in (("completion.json", completion), ("consumption.json", ledger)):
        (tmp_path / filename).write_text(json.dumps(value))
    job.plan.update(budget_source_stage=str(tmp_path),
        budget_source_files_sha256={name: continuation.digest(tmp_path / name)
            for name in ("completion.json", "consumption.json")},
        inherited_recovery_attempts=completion["inherited_recovery_attempts"])
    origin = dict(completion_status=completion["status"], consumption=ledger,
        inherited_recovery_attempts=completion["inherited_recovery_attempts"],
        protected_case_names=completion["protected_case_names"],
        checkpoint_relative_path="A_landing_repair/train/model_final.pt", checkpoint_sha256=sha,
        checkpoint_learning_lineage_updates=418, checkpoint_stage_actor_update=408)
    job.authenticate_budget_source(origin)
    for key, value in (("checkpoint_relative_path", "../A_landing_repair/train/model_final.pt"),
                       ("checkpoint_sha256", "stale"), ("checkpoint_stage_actor_update", 368)):
        changed = {**origin, key: value}
        with pytest.raises(ValueError):
            job.authenticate_budget_source(changed)
