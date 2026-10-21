"""A later zero-consumption evaluation must not break an active worker."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.support.support_training_budget import NOW, TestMeter, make_review, publish
from wheeled_tasks.chassis.application.runtime.training_budget import BudgetError, BudgetOwner, digest_file


def open_owner(path, review, attempt):
    identity = {key: value for key, value in review["attempts"][attempt].items() if key != "role"}
    return BudgetOwner.authenticate(
        path, digest_file(path), review["ledger_path"], attempt, identity,
        gpu_meter=TestMeter(), allow_test_meter=True, now=lambda: NOW,
    )


def append_evaluation(tmp_path, review):
    extended = deepcopy(review)
    extended["attempts"]["evaluation"] = {
        **review["attempts"]["main"],
        "requested_updates": 1,
        "training_source_sha256": "c" * 64,
    }
    path = tmp_path / "evaluation-review.json"
    path.write_text(json.dumps(extended))
    return path, extended


def test_old_worker_can_update_save_and_close_after_new_evaluation(tmp_path):
    path, review = make_review(tmp_path, roles={"main": "main"})
    worker = open_owner(path, review, "main")
    worker.complete_update(worker.reserve_update(phase="actor"), actor_optimized=True)
    newer_path, newer_review = append_evaluation(tmp_path, review)
    evaluation = open_owner(newer_path, newer_review, "evaluation")
    try:
        assert evaluation.summary()["paid_updates_attempt"] == 0
        assert worker.summary()["paid_updates_global"] == 1
        worker.complete_update(worker.reserve_update(phase="actor"), actor_optimized=True)
        checkpoint = tmp_path / "saved.pt"
        publish(worker, checkpoint)
        assert worker.verify_resume(checkpoint)["attempt_id"] == "main"
        assert evaluation.summary()["paid_updates_global"] == 2
    finally:
        evaluation.close()
        worker.close()
    state = json.loads(Path(review["ledger_path"]).read_text())
    assert len(state["reservations"]) == 2
    assert all(lease["closed"] for attempt in state["attempts"].values() for lease in attempt["leases"])
    assert "evaluation" not in json.loads(path.read_text())["attempts"]


def test_unknown_paid_attempt_still_requires_updated_review(tmp_path):
    path, review = make_review(tmp_path, roles={"main": "main"})
    worker = open_owner(path, review, "main")
    newer_path, newer_review = append_evaluation(tmp_path, review)
    evaluation = open_owner(newer_path, newer_review, "evaluation")
    evaluation.complete_update(evaluation.reserve_update(phase="actor"), actor_optimized=True)
    evaluation.close()
    before = Path(review["ledger_path"]).read_bytes()
    try:
        with pytest.raises(BudgetError, match="Paid attempt is absent"):
            worker.reserve_update(phase="actor")
        assert Path(review["ledger_path"]).read_bytes() == before
    finally:
        # The intentionally incompatible review cannot close through a transaction.
        worker._lease.close()


@pytest.mark.parametrize("attempt,field,value", [
    ("main", "seed", 99),
    ("evaluation", "role", "unreviewed"),
    ("evaluation", "training_source_sha256", "not-a-digest"),
    ("evaluation", "requested_updates", 0),
])
def test_zero_consumption_does_not_bypass_identity_checks(tmp_path, attempt, field, value):
    path, review = make_review(tmp_path, roles={"main": "main"})
    worker = open_owner(path, review, "main")
    newer_path, newer_review = append_evaluation(tmp_path, review)
    evaluation = open_owner(newer_path, newer_review, "evaluation")
    evaluation.close()
    ledger = Path(review["ledger_path"])
    original = ledger.read_bytes()
    state = json.loads(original)
    state["attempts"][attempt]["identity"][field] = value
    ledger.write_text(json.dumps(state))
    try:
        with pytest.raises(BudgetError):
            worker.summary()
    finally:
        ledger.write_bytes(original)
        worker.close()


def test_new_attempt_still_needs_its_own_pinned_review(tmp_path):
    path, review = make_review(tmp_path, roles={"main": "main"})
    identity = {key: value for key, value in review["attempts"]["main"].items() if key != "role"}
    with pytest.raises(BudgetError, match="Attempt is not in"):
        BudgetOwner.authenticate(
            path, digest_file(path), review["ledger_path"], "evaluation", identity,
            gpu_meter=TestMeter(), allow_test_meter=True, now=lambda: NOW,
        )


def test_zero_paid_diagnostic_does_not_require_any_pilot_allocation(tmp_path):
    path, review = make_review(tmp_path, pilot_cap=0, roles={"evaluation": "diagnostic"})
    identity = {key: value for key, value in review["attempts"]["evaluation"].items() if key != "role"}
    evaluation = BudgetOwner.authenticate(path, digest_file(path), review["ledger_path"], "evaluation", identity,
        evaluation_only=True, gpu_meter=TestMeter(), allow_test_meter=True, now=lambda: NOW)
    try:
        evaluation.check_runtime()
        with pytest.raises(BudgetError, match="forbids optimizer"):
            evaluation.reserve_update(phase="actor")
        with pytest.raises(BudgetError, match="forbids optimizer"):
            evaluation.reserve_update(phase="critic")
        assert evaluation.summary()["paid_updates_global"] == 0
    finally:
        evaluation.close()
    with pytest.raises(BudgetError, match="pilot work"):
        open_owner(path, review, "evaluation")


def test_evaluation_flag_cannot_disguise_a_mainline_or_a_spent_diagnostic(tmp_path):
    path, review = make_review(tmp_path, pilot_cap=10, roles={"main": "main", "evaluation": "diagnostic"})
    identity = {key: value for key, value in review["attempts"]["main"].items() if key != "role"}
    with pytest.raises(BudgetError, match="diagnostic attempt"):
        BudgetOwner.authenticate(path, digest_file(path), review["ledger_path"], "main", identity,
            evaluation_only=True, gpu_meter=TestMeter(), allow_test_meter=True, now=lambda: NOW)
    diagnostic = open_owner(path, review, "evaluation")
    diagnostic.complete_update(diagnostic.reserve_update(phase="actor"), actor_optimized=True)
    checkpoint = tmp_path / "diagnostic.pt"
    publish(diagnostic, checkpoint)
    diagnostic.close()
    identity = {key: value for key, value in review["attempts"]["evaluation"].items() if key != "role"}
    with pytest.raises(BudgetError, match="zero paid reservations"):
        BudgetOwner.authenticate(path, digest_file(path), review["ledger_path"], "evaluation", identity,
            resume_checkpoint=checkpoint, evaluation_only=True,
            gpu_meter=TestMeter(), allow_test_meter=True, now=lambda: NOW)
