"""Self-contained filesystem fixtures; no native worker, external helper or PPO."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis import training_budget as budget_module
from wheeled_tasks.chassis.training_budget import (
    BudgetError, BudgetLimit, BudgetOwner, PAID_CAP, REVIEW_SCHEMA, digest_file,
)


NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)
APPROVAL = "CPU fixture only: user approved count-only; caller verified completed055 and closed leases"


class Meter:
    owner_id = "count_only_cpu_fixture"
    def __init__(self, value=2.):
        self.value = value
    def read_total_gpu_seconds(self):
        return self.value


def write_review(path, review):
    path.write_text(json.dumps(review))
    return path


def actual(review, attempt, requested):
    return {**{key: value for key, value in review["attempts"][attempt].items() if key != "role"},
            "requested_updates": requested}


def owner(path, review, attempt="parent", requested=1, *, checkpoint=None, meter=None, clock=None):
    return BudgetOwner.authenticate(path, digest_file(path), review["ledger_path"], attempt,
        actual(review, attempt, requested), resume_checkpoint=checkpoint,
        gpu_meter=meter or Meter(), allow_test_meter=True, now=lambda: (clock or [NOW])[0])


def publish(book, path):
    metadata = book.prepare_checkpoint(path)
    path.write_bytes(b"CPU sealed checkpoint fixture; not a Torch model")
    path.with_suffix(".checkpoint.json").write_text(json.dumps({
        "checkpoint_sha256": digest_file(path), "contract_sha256": book.identity["contract_sha256"],
        "training_budget": metadata}))
    book.register_checkpoint(path, metadata)
    return metadata


def history(tmp_path, count=0):
    identity = {"contract_sha256": "a" * 64, "training_source_sha256": "b" * 64,
        "seed": 290929, "stage": "foundation", "requested_updates": PAID_CAP,
        "parent_checkpoint_sha256": None}
    review = {"schema": REVIEW_SCHEMA, "campaign_id": "count_only_fixture", "ledger_id": "same_ledger",
        "ledger_path": str(tmp_path / "paid.json"), "paid_update_cap": PAID_CAP,
        "pilot_paid_update_cap": 1, "absolute_deadline_utc": (NOW + timedelta(hours=1)).isoformat(),
        "max_attempt_wall_seconds": 300., "gpu_occupied_seconds_cap": 100.,
        "gpu_meter_owner": {"kind": "test_only_cpu_meter_v1", "id": Meter.owner_id},
        "review_reference": "CPU fixture; not native authorization",
        "attempts": {"parent": {**identity, "role": "main"},
                     "pilot": {**identity, "requested_updates": 10, "role": "pilot"}}}
    path = write_review(tmp_path / "old-review.json", review)
    book = owner(path, review)
    if count:
        token = book.reserve_update(phase="actor")
        book.complete_update(token, actor_optimized=True)
        # Install a valid large historical prefix once, avoiding20000 fsyncs.
        with book._transaction() as state:
            row = state["reservations"][0]
            state["reservations"] = [{**row, "sequence": n, "ordinal": n} for n in range(1, count + 1)]
    checkpoint = tmp_path / "parent.pt"
    receipt = publish(book, checkpoint)
    book.close()
    new = deepcopy(review)
    new["paid_update_cap"] = None
    new["attempts"]["recovery"] = {**identity, "requested_updates": 5000,
        "parent_checkpoint_sha256": digest_file(checkpoint), "role": "main"}
    new_path = write_review(tmp_path / "new-review.json", new)
    return path, review, new_path, new, checkpoint, receipt


def amend(path, review, *, previous=None, approval=APPROVAL):
    ledger = Path(review["ledger_path"])
    marker = json.loads(ledger.with_name(ledger.name + ".identity.json").read_text())
    return BudgetOwner.amend_paid_cap_to_count_only(path, digest_file(path), ledger,
        expected_previous_policy_sha256=previous or marker["campaign_policy_sha256"],
        approval_reference=approval, now=lambda: NOW)


def test_count_only_crosses20000_but_5000_attempt_counts_failed_and_critic_work(tmp_path):
    _, _, path, review, checkpoint, _ = history(tmp_path, PAID_CAP)
    amend(path, review)
    book = owner(path, review, "recovery", 5000, checkpoint=checkpoint)
    try:
        token = book.reserve_update(phase="actor")
        book.fail_update(token, "failed optimizer remains paid")
        assert book.summary()["paid_updates_global"] == PAID_CAP + 1
        with pytest.raises(BudgetError, match="completion credit"):
            book.complete_update(token, actor_optimized=True)
        with book._transaction() as state:
            template = state["reservations"][-1]
            state["reservations"] = state["reservations"][:PAID_CAP] + [
                {**template, "sequence": PAID_CAP + n, "ordinal": n} for n in range(1, 5000)]
        last = book.reserve_update(phase="critic")
        book.complete_update(last, actor_optimized=False)
        summary = book.summary()
        assert summary["paid_update_cap"] is None
        assert summary["paid_updates_global"] == 25000 and summary["paid_updates_attempt"] == 5000
        assert summary["critic_completed_attempt"] == 1 and summary["actor_completed_attempt"] == 0
        with pytest.raises(BudgetLimit, match="Attempt paid-update"):
            book.reserve_update(phase="actor")
    finally:
        book.close()


def test_atomic_amendment_preserves_marker_history_and_old_checkpoint_receipt(tmp_path):
    old_path, old_review, path, review, checkpoint, receipt = history(tmp_path, 2)
    ledger = Path(review["ledger_path"])
    marker = ledger.with_name(ledger.name + ".identity.json")
    marker_before, state_before = marker.read_bytes(), json.loads(ledger.read_text())
    with pytest.raises(BudgetError):
        owner(path, review, "recovery", 5000, checkpoint=checkpoint)
    assert json.loads(ledger.read_text()) == state_before
    amendment = amend(path, review)
    state = json.loads(ledger.read_text())
    assert marker.read_bytes() == marker_before
    assert state["reservations"] == state_before["reservations"]
    assert state["checkpoints"] == state_before["checkpoints"]
    assert state["attempts"] == state_before["attempts"]
    assert state["paid_cap_amendment"] == amendment
    assert amend(path, review) == amendment  # No second migration or refund.
    with pytest.raises(BudgetError):
        owner(old_path, old_review, checkpoint=checkpoint)
    book = owner(path, review, "recovery", 5000, checkpoint=checkpoint)
    try:
        assert book.verify_resume(checkpoint, receipt) == receipt
        assert receipt["paid_update_cap"] == PAID_CAP
        assert receipt["campaign_policy_sha256"] == amendment["previous_policy_sha256"]
        book.complete_update(book.reserve_update(phase="actor"), actor_optimized=True)
        new_receipt = publish(book, tmp_path / "new.pt")
        assert new_receipt["paid_update_cap"] is None
        assert new_receipt["campaign_policy_sha256"] == amendment["current_policy_sha256"]
    finally:
        book.close()


def test_none_cannot_initialize_a_new_ledger_or_skip_amendment(tmp_path):
    _, _, path, review, checkpoint, _ = history(tmp_path)
    fresh = deepcopy(review)
    fresh["ledger_id"] = "cannot_replace_history"
    fresh["ledger_path"] = str(tmp_path / "fresh.json")
    fresh_path = write_review(tmp_path / "fresh-review.json", fresh)
    with pytest.raises(BudgetError, match="Count-only cannot initialize"):
        owner(fresh_path, fresh, "recovery", 5000, checkpoint=checkpoint)
    assert not Path(fresh["ledger_path"]).exists()
    assert not Path(fresh["ledger_path"] + ".identity.json").exists()


@pytest.mark.parametrize("change", ["wall", "deadline", "gpu", "meter", "pilot", "old_attempt", "root", "approval", "cap"])
def test_amendment_rejects_every_unapproved_policy_or_history_difference(tmp_path, change):
    _, _, path, review, _, _ = history(tmp_path, 1)
    ledger = Path(review["ledger_path"])
    before = ledger.read_bytes()
    previous, approval = None, APPROVAL
    if change == "wall": review["max_attempt_wall_seconds"] += 1
    elif change == "deadline": review["absolute_deadline_utc"] = (NOW + timedelta(days=1)).isoformat()
    elif change == "gpu": review["gpu_occupied_seconds_cap"] += 1
    elif change == "meter": review["gpu_meter_owner"]["id"] = "different"
    elif change == "pilot": review["pilot_paid_update_cap"] += 1
    elif change == "old_attempt": review["attempts"]["parent"]["requested_updates"] -= 1
    elif change == "root": previous = "0" * 64
    elif change == "approval": approval = ""
    else: review["paid_update_cap"] = PAID_CAP
    write_review(path, review)
    with pytest.raises(BudgetError):
        amend(path, review, previous=previous, approval=approval)
    assert ledger.read_bytes() == before


def test_amendment_rechecks_actual_flock_even_if_closed_flag_claims_idle(tmp_path):
    old_path, old_review, path, review, checkpoint, _ = history(tmp_path)
    live = owner(old_path, old_review, checkpoint=checkpoint)
    try:
        with live._transaction() as state:
            for lease in state["attempts"]["parent"]["leases"]:
                lease["closed"] = True  # Wrong metadata must not defeat the kernel lock.
        before = Path(review["ledger_path"]).read_bytes()
        with pytest.raises(BudgetError, match="actual attempt lease"):
            amend(path, review)
        assert Path(review["ledger_path"]).read_bytes() == before
    finally:
        live.close()


def test_failed_atomic_write_leaves_old_root_and_history_retryable(tmp_path, monkeypatch):
    _, _, path, review, _, _ = history(tmp_path, 1)
    ledger = Path(review["ledger_path"])
    marker = ledger.with_name(ledger.name + ".identity.json")
    original, marker_before = ledger.read_bytes(), marker.read_bytes()
    real = budget_module._write
    def fail_write(path_arg, value):
        if path_arg == ledger:
            raise OSError("fixture publication failure")
        return real(path_arg, value)
    with monkeypatch.context() as context:
        context.setattr(budget_module, "_write", fail_write)
        with pytest.raises(OSError, match="publication failure"):
            amend(path, review)
    assert ledger.read_bytes() == original and marker.read_bytes() == marker_before
    amend(path, review)
    assert len(json.loads(ledger.read_text())["reservations"]) == 1


@pytest.mark.parametrize("cause", ["wall", "deadline", "gpu", "backwards"])
def test_count_only_keeps_every_existing_time_and_meter_limit(tmp_path, cause):
    _, _, path, review, checkpoint, _ = history(tmp_path)
    amend(path, review)
    meter, clock = Meter(), [NOW]
    book = owner(path, review, "recovery", 5000, checkpoint=checkpoint, meter=meter, clock=clock)
    try:
        if cause == "wall": clock[0] += timedelta(seconds=300)
        elif cause == "deadline": clock[0] += timedelta(hours=1)
        elif cause == "gpu": meter.value = 100.
        else: meter.value = 1.
        with pytest.raises(BudgetError):
            book.reserve_update(phase="actor")
        assert book.summary()["paid_updates_global"] == 0
    finally:
        book.close()


def test_count_only_retains_finite_shared_pilot_cap(tmp_path):
    _, _, path, review, _, _ = history(tmp_path)
    amend(path, review)
    pilot = owner(path, review, "pilot")
    try:
        pilot.fail_update(pilot.reserve_update(phase="actor"), "still paid")
        with pytest.raises(BudgetLimit, match="Pilot/diagnostic"):
            pilot.reserve_update(phase="actor")
        assert pilot.summary()["paid_updates_global"] == 1
    finally:
        pilot.close()
