"""Elapsed accounting survives host UTC corrections without refunding work."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.application.runtime import training_budget as budget_module
from wheeled_tasks.chassis.application.runtime.training_budget import (
    BudgetError, BudgetLimit, BudgetOwner, PAID_CAP, REVIEW_SCHEMA, digest_file,
)


NOW = datetime(2026, 10, 16, tzinfo=timezone.utc)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    wall, boot = [NOW], [1000.0]
    monkeypatch.setattr(budget_module, "clock_gettime", lambda clock: boot[0])
    identity = {"contract_sha256": "a" * 64, "training_source_sha256": "b" * 64,
                "seed": 1, "stage": "foundation", "requested_updates": 10,
                "parent_checkpoint_sha256": None}
    review = {"schema": REVIEW_SCHEMA, "campaign_id": "elapsed-clock-fixture",
              "ledger_id": "elapsed-clock-fixture", "ledger_path": str(tmp_path / "paid.json"),
              "paid_update_cap": PAID_CAP, "pilot_paid_update_cap": 0,
              "absolute_deadline_utc": (NOW + timedelta(hours=1)).isoformat(),
              "max_attempt_wall_seconds": 300, "gpu_occupied_seconds_cap": 1000,
              "gpu_meter_owner": {"kind": "campaign_elapsed_wall_v1", "id": "elapsed",
                                  "origin_utc": (NOW - timedelta(seconds=100)).isoformat()},
              "review_reference": "CPU clock fixture only",
              "attempts": {"main": {**identity, "role": "main"}}}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))

    def open_owner():
        return BudgetOwner.authenticate(path, digest_file(path), review["ledger_path"],
                                        "main", identity, now=lambda: wall[0])

    owner = open_owner()
    yield owner, wall, boot, open_owner
    if not owner.closed:
        owner.close()


def test_backward_utc_keeps_charging_and_does_not_scan_ledger(campaign, monkeypatch):
    owner, wall, boot, _ = campaign
    before = owner.path.read_bytes()

    def forbidden_transaction(*args, **kwargs):
        raise AssertionError("Runtime check must remain O(1)")

    with monkeypatch.context() as patch:
        patch.setattr(owner, "_transaction", forbidden_transaction)
        wall[0] -= timedelta(seconds=60)
        boot[0] += 10
        owner.check_runtime()
        assert owner._last_gpu_meter_seconds == 110
        boot[0] += 10
        owner.check_runtime()
        assert owner._last_gpu_meter_seconds == 120
    assert owner.path.read_bytes() == before


def test_forward_then_backward_correction_keeps_the_higher_clock(campaign):
    owner, wall, boot, _ = campaign
    wall[0] += timedelta(seconds=50)
    boot[0] += 1
    owner.check_runtime()
    assert owner._last_gpu_meter_seconds == 150
    wall[0] -= timedelta(seconds=60)
    boot[0] += 2
    owner.check_runtime()
    assert owner._last_gpu_meter_seconds == 152


@pytest.mark.parametrize("limit", ["deadline", "attempt", "campaign"])
def test_clock_correction_cannot_extend_any_time_limit(campaign, limit):
    owner, wall, boot, _ = campaign
    if limit == "deadline":
        owner.deadline = NOW + timedelta(seconds=5)
    elif limit == "attempt":
        owner.review["max_attempt_wall_seconds"] = 5
    else:
        owner.review["gpu_occupied_seconds_cap"] = 105
    wall[0] -= timedelta(hours=1)
    boot[0] += 5
    with pytest.raises(BudgetLimit):
        owner.check_runtime()


def test_restart_preserves_durable_elapsed_highwater(campaign):
    owner, wall, boot, reopen = campaign
    wall[0] += timedelta(seconds=20)
    boot[0] += 20
    owner.check()
    owner.close()
    wall[0] -= timedelta(seconds=30)
    replacement = reopen()
    try:
        assert replacement._last_gpu_meter_seconds == 120
        boot[0] += 3
        replacement.check()
        state = json.loads(Path(replacement.path).read_text())
        assert state["gpu_meter_reading_seconds"] == 123
        assert not state["reservations"]
        assert state["attempts"]["main"]["first_started_at"] == NOW.isoformat()
    finally:
        replacement.close()


@pytest.mark.parametrize("value", [999.0, float("nan")])
def test_invalid_boot_clock_remains_an_error(campaign, value):
    owner, _, boot, _ = campaign
    boot[0] = value
    with pytest.raises(BudgetError, match="elapsed clock"):
        owner.check_runtime()
