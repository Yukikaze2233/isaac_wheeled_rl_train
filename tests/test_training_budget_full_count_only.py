"""Remove budget limits without refunding history or weakening worker identity."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path

import pytest

from test_training_budget_count_only import NOW, Meter, amend, history, owner, publish, write_review
from wheeled_tasks.chassis import training_budget as budget_module
from wheeled_tasks.chassis.training_budget import (
    COUNT_ONLY_FIELDS, BudgetError, BudgetLimit, BudgetOwner, digest_file,
)


def candidate(tmp_path, count=3):
    _, _, partial_path, partial_review, parent, _ = history(tmp_path, count)
    amend(partial_path, partial_review)
    review = deepcopy(partial_review)
    review.update(dict.fromkeys(COUNT_ONLY_FIELDS))
    identity = {**review["attempts"]["recovery"], "requested_updates": None}
    review["attempts"]["continuous"] = identity
    review["attempts"]["smoke"] = {**identity, "role": "diagnostic", "requested_updates": 2}
    review["attempts"]["screening"] = {**identity, "role": "pilot", "requested_updates": 2}
    path = write_review(tmp_path / "full-review.json", review)
    previous = json.loads(Path(review["ledger_path"]).read_text())["campaign_policy_sha256"]
    return path, review, previous, parent, partial_path, partial_review


def migrate(path, review, previous):
    return BudgetOwner.amend_limits_to_count_only(path, digest_file(path), review["ledger_path"],
        expected_previous_policy_sha256=previous, approval_reference="CPU fixture: no budget limits",
        now=lambda: NOW + timedelta(days=30))


def test_full_migration_preserves_all_history_and_old_checkpoint_seals(tmp_path):
    path, review, previous, parent, _, _ = candidate(tmp_path)
    ledger = Path(review["ledger_path"])
    marker = ledger.with_name(ledger.name + ".identity.json")
    before, marker_bytes = json.loads(ledger.read_text()), marker.read_bytes()
    receipt = migrate(path, review, previous)
    after = json.loads(ledger.read_text())
    for key in before.keys() - {"campaign_policy", "campaign_policy_sha256"}:
        assert after[key] == before[key]
    assert marker.read_bytes() == marker_bytes
    assert receipt["previous_policy"] == before["campaign_policy"]
    assert receipt["paid_updates_at_amendment"] == 3
    assert migrate(path, review, previous) == receipt
    book = owner(path, review, "continuous", None, checkpoint=parent,
                 clock=[NOW + timedelta(days=30)], meter=Meter(1_000_000.))
    try:
        assert book.verify_resume(parent)["paid_updates_global"] == 3
        book.check_runtime()
        book.complete_update(book.reserve_update(phase="critic"), actor_optimized=False)
        checkpoint = tmp_path / "continued.pt"
        publish(book, checkpoint)
        book.fail_update(book.reserve_update(phase="actor"), "interrupted work remains paid")
        assert book.summary()["paid_updates_global"] == 5
    finally:
        book.close()
    restarted = owner(path, review, "continuous", None, checkpoint=checkpoint,
                      clock=[NOW + timedelta(days=31)], meter=Meter(1_000_001.))
    try:
        assert restarted.summary()["paid_updates_attempt"] == 2
        token = restarted.reserve_update(phase="actor")
        assert token["ordinal"] == 3 and token["sequence"] == 6
        restarted.complete_update(token, actor_optimized=True)
    finally:
        restarted.close()


def test_diagnostic_and_pilot_use_real_roles_and_keep_their_technical_windows(tmp_path):
    path, review, previous, parent, _, _ = candidate(tmp_path)
    migrate(path, review, previous)
    for attempt in ("smoke", "screening"):
        book = owner(path, review, attempt, 2, checkpoint=parent, clock=[NOW + timedelta(days=30)])
        try:
            for _ in range(2):
                book.complete_update(book.reserve_update(phase="actor"), actor_optimized=True)
            with pytest.raises(BudgetLimit, match="Attempt paid-update"):
                book.reserve_update(phase="actor")
        finally:
            book.close()
    state = json.loads(Path(review["ledger_path"]).read_text())
    assert len(state["reservations"]) == 7
    assert state["attempts"]["smoke"]["identity"]["role"] == "diagnostic"
    assert state["attempts"]["screening"]["identity"]["role"] == "pilot"


@pytest.mark.parametrize("change", ["partial_limit", "meter", "old_attempt", "missing_attempt", "open_pilot"])
def test_migration_rejects_unrelated_changes_without_writing(tmp_path, change):
    path, review, previous, _, _, _ = candidate(tmp_path)
    ledger = Path(review["ledger_path"])
    before = ledger.read_bytes()
    if change == "partial_limit":
        review["max_attempt_wall_seconds"] = 999999.
    elif change == "meter":
        review["gpu_meter_owner"]["id"] = "different"
    elif change == "old_attempt":
        review["attempts"]["parent"]["requested_updates"] = None
    elif change == "missing_attempt":
        del review["attempts"]["parent"]
    else:
        review["attempts"]["screening"]["requested_updates"] = None
    write_review(path, review)
    with pytest.raises(BudgetError):
        migrate(path, review, previous)
    assert ledger.read_bytes() == before


def test_full_migration_requires_actual_idle_lease_even_with_wrong_closed_flag(tmp_path):
    path, review, previous, parent, partial_path, partial_review = candidate(tmp_path)
    live = owner(partial_path, partial_review, "recovery", 1, checkpoint=parent)
    try:
        with live._transaction() as state:
            state["attempts"]["recovery"]["leases"][-1]["closed"] = True
        before = Path(review["ledger_path"]).read_bytes()
        with pytest.raises(BudgetError, match="actual attempt lease"):
            migrate(path, review, previous)
        assert Path(review["ledger_path"]).read_bytes() == before
    finally:
        live.close()
    migrate(path, review, previous)


def test_full_migration_atomic_write_failure_is_retryable(tmp_path, monkeypatch):
    path, review, previous, _, _, _ = candidate(tmp_path)
    ledger = Path(review["ledger_path"])
    before = ledger.read_bytes()
    original = budget_module._write
    def fail(destination, value):
        if destination == ledger:
            raise OSError("fixture migration write failure")
        original(destination, value)
    with monkeypatch.context() as context:
        context.setattr(budget_module, "_write", fail)
        with pytest.raises(OSError, match="migration write"):
            migrate(path, review, previous)
    assert ledger.read_bytes() == before
    migrate(path, review, previous)


def test_null_limits_cannot_bypass_receipt_and_marker_chain(tmp_path):
    path, review, previous, parent, _, _ = candidate(tmp_path)
    with pytest.raises(BudgetError):
        owner(path, review, "continuous", None, checkpoint=parent)
    migrate(path, review, previous)
    ledger = Path(review["ledger_path"])
    state = json.loads(ledger.read_text())
    state["count_only_limits_amendment"]["paid_updates_at_amendment"] = 0
    ledger.write_text(json.dumps(state))
    with pytest.raises(BudgetError, match="count differs"):
        owner(path, review, "continuous", None, checkpoint=parent)


def test_explicit_count_only_does_not_disable_meter_integrity(tmp_path):
    path, review, previous, parent, _, _ = candidate(tmp_path)
    migrate(path, review, previous)
    meter = Meter(1000.)
    book = owner(path, review, "continuous", None, checkpoint=parent, meter=meter)
    try:
        meter.value = 999.
        with pytest.raises(BudgetError, match="moves backwards"):
            book.check_runtime()
        meter.value = float("nan")
        with pytest.raises(BudgetError, match="invalid"):
            book.reserve_update(phase="actor")
    finally:
        book.close()


def test_count_only_main_removes_worker_time_cap_and_retains_manual_stop(tmp_path):
    from wheeled_algo.v40_job import PlannedStop, TrainingBudget
    path, review, previous, parent, _, _ = candidate(tmp_path)
    migrate(path, review, previous)
    book = owner(path, review, "continuous", None, checkpoint=parent)
    try:
        clock = [0.]
        control = TrainingBudget(book.execution_wall_limit(1.), monotonic=lambda: clock[0])
        control.begin_learning()
        clock[0] = 1_000_000.
        control.check()
        control.request_stop()
        with pytest.raises(PlannedStop, match="soft_stop"):
            control.check()
    finally:
        book.close()


def test_count_only_smoke_keeps_requested_worker_window(tmp_path):
    path, review, previous, parent, _, _ = candidate(tmp_path)
    migrate(path, review, previous)
    book = owner(path, review, "smoke", 2, checkpoint=parent)
    try:
        assert book.execution_wall_limit(600.) == 600.
    finally:
        book.close()
