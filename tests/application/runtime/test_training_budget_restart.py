"""Full-state restarts keep the original attempt parent and durable spending."""
from pathlib import Path

import pytest

from wheeled_tasks.chassis.application.runtime.training_budget import BudgetError, BudgetOwner, digest_file
from tests.application.runtime.test_training_budget_count_only import NOW, Meter, actual, amend, history, owner, publish


def test_restart_checkpoint_does_not_redefine_original_parent(tmp_path):
    _, _, path, review, parent, _ = history(tmp_path, 3)
    amend(path, review)
    book = owner(path, review, "recovery", 2, checkpoint=parent)
    book.complete_update(book.reserve_update(phase="actor"), actor_optimized=True)
    checkpoint = tmp_path / "saved.pt"
    publish(book, checkpoint)
    book.close()
    identity = actual(review, "recovery", 1)
    identity["parent_checkpoint_sha256"] = digest_file(checkpoint)
    restarted = BudgetOwner.authenticate(path, digest_file(path), review["ledger_path"], "recovery", identity,
        resume_checkpoint=checkpoint, gpu_meter=Meter(), allow_test_meter=True, now=lambda: NOW)
    try:
        assert restarted.identity["parent_checkpoint_sha256"] == digest_file(parent)
        assert restarted.summary()["paid_updates_global"] == 4
        restarted.complete_update(restarted.reserve_update(phase="actor"), actor_optimized=True)
        assert restarted.summary()["paid_updates_global"] == 5
    finally:
        restarted.close()
    identity["contract_sha256"] = "e" * 64
    with pytest.raises(BudgetError, match="differ"):
        BudgetOwner.authenticate(path, digest_file(path), review["ledger_path"], "recovery", identity,
            resume_checkpoint=checkpoint, gpu_meter=Meter(), allow_test_meter=True, now=lambda: NOW)


def test_restart_parent_cannot_borrow_foreign_or_tampered_checkpoint(tmp_path):
    _, _, path, review, checkpoint, _ = history(tmp_path, 3)
    amend(path, review)
    identity = actual(review, "pilot", 1)
    identity["parent_checkpoint_sha256"] = digest_file(checkpoint)
    before = Path(review["ledger_path"]).read_bytes()
    with pytest.raises(BudgetError, match="same-attempt"):
        BudgetOwner.authenticate(path, digest_file(path), review["ledger_path"], "pilot", identity,
            resume_checkpoint=checkpoint, gpu_meter=Meter(), allow_test_meter=True, now=lambda: NOW)
    assert Path(review["ledger_path"]).read_bytes() == before
    identity = actual(review, "parent", 1)
    identity["parent_checkpoint_sha256"] = digest_file(checkpoint)
    checkpoint.with_suffix(".checkpoint.json").write_text("{}")
    with pytest.raises(BudgetError, match="same-attempt"):
        BudgetOwner.authenticate(path, digest_file(path), review["ledger_path"], "parent", identity,
            resume_checkpoint=checkpoint, gpu_meter=Meter(), allow_test_meter=True, now=lambda: NOW)
