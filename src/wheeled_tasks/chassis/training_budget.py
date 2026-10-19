"""Fail-closed shared paid-update accounting for the named 78D worker.

A hash-pinned review is a reproducible input, NOT proof of human authorization
or native admission. Existing admission checks remain mandatory. No production
GPU occupancy provider is admitted here: a real GPU-cap contract therefore
refuses construction. Explicit CPU fixtures include an identity-bound socket
client to the candidate allocation guardian; neither is exposed to the CLI.

Reservations are durable consumption, including failed/unknown optimization.
Checkpoint publication is two-phase: an unregistered file is never resumable.
No simulator/torch import, capability clock, or episode restart is owned here.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import CLOCK_BOOTTIME, clock_gettime
import uuid


REVIEW_SCHEMA = "v7-shared-training-budget-review-v1"
STATE_SCHEMA = "v7-shared-training-budget-state-v1"
RECEIPT_SCHEMA = "v7-shared-training-budget-checkpoint-v1"
PAID_CAP = 20000
COUNT_ONLY_FIELDS = ("paid_update_cap", "pilot_paid_update_cap", "absolute_deadline_utc",
                     "max_attempt_wall_seconds", "gpu_occupied_seconds_cap")
IDENTITY_KEYS = {"contract_sha256", "training_source_sha256", "seed", "stage",
                 "requested_updates", "parent_checkpoint_sha256"}


class BudgetError(RuntimeError):
    pass


class BudgetLimit(BudgetError):
    pass


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise BudgetError(f"{name} requires an integer >= {minimum}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise BudgetError(f"Missing explicit {name}")
    return value


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise BudgetError(f"Invalid {name} SHA-256")
    return value


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise BudgetError(f"Missing explicit positive finite {name}")
    return float(value)


def _count_only(policy):
    return all(key in policy and policy[key] is None for key in COUNT_ONLY_FIELDS)


def _deadline(value):
    if value is None:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise BudgetError("Budget deadline must be UTC or explicitly None") from exc
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise BudgetError("Budget deadline must include UTC timezone")
    return result


def _validate_limits(policy):
    """Only the original bounded policy or the complete count-only policy."""
    cap = policy["paid_update_cap"]
    if cap is not None and (type(cap) is not int or cap != PAID_CAP):
        raise BudgetError("Shared paid cap must be20000 or reviewed count-only None")
    if not _count_only(policy):
        pilot = _count(policy["pilot_paid_update_cap"], "pilot paid cap", 0)
        if cap is not None and pilot > cap:
            raise BudgetError("Pilot cap exceeds the common ceiling")
        if policy["absolute_deadline_utc"] is None:
            raise BudgetError("Partial removal of time limits requires the complete count-only migration")
        _positive(policy["max_attempt_wall_seconds"], "attempt wall cap")
        _positive(policy["gpu_occupied_seconds_cap"], "campaign accounting cap")
    return _deadline(policy["absolute_deadline_utc"])


def _absolute(path):
    value = Path(_text(path, "absolute ledger path"))
    if not value.is_absolute() or value.resolve() != value:
        raise BudgetError("Ledger path must be absolute and canonical; aliases are forbidden")
    return value


def _write(path, value):
    """File and containing directory durability; caller holds the stable lock."""
    raw = _encoded(value) + b"\n"
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def require_budget_arguments(contract, *, review_file, review_sha256, ledger_path, attempt_id,
                             resume=None, consumed_updates=None):
    """Early worker selection check; never upgrades native admission."""
    from .diagnostic_budget_selection import select_budget_scope
    scope = select_budget_scope(contract)
    selected = scope != "legacy_unmanaged"
    values = (review_file, review_sha256, ledger_path, attempt_id)
    if selected:
        if not all(value is not None for value in values):
            raise BudgetError("Selected V7 worker requires a hash-pinned shared review, ledger and attempt identity")
        _sha(review_sha256, "budget review")
        _text(attempt_id, "attempt identity")
        if consumed_updates is not None:
            raise BudgetError("V7 total consumption comes only from the shared ledger, not CLI rebasing")
        if (scope == "named78" and resume is not None and "flat_episode_runtime" in contract
                and not (contract.get("formal_v7_flat") is True
                         and contract.get("formal_learning_state_resume") is True)):
            raise BudgetError("Sourceflat exact episode restart is not implemented; budget state cannot substitute for it")
    elif any(value is not None for value in values):
        raise BudgetError("Shared V7 budget arguments cannot alter a legacy worker")
    return selected


def _authenticate_meter(review, gpu_meter, allow_test_meter):
    """Bind CPU evidence to its owner without opening production admission.

    分配时长并非活跃计算时长，CPU socket 测试也不是 GPU 计量资格。
    Allocation duration is not active-compute time; CPU RPC is not GPU qualification.
    """
    meter = review["gpu_meter_owner"]
    if not isinstance(meter, dict) or "kind" not in meter or "id" not in meter:
        raise BudgetError("GPU cap requires an explicit meter owner")
    _text(meter["id"], "GPU meter owner")
    if meter["kind"] == "campaign_elapsed_wall_v1":
        if set(meter) != {"kind", "id", "origin_utc"} or gpu_meter is not None:
            raise BudgetError("Elapsed campaign accounting requires an explicit fixed UTC origin")
        origin = datetime.fromisoformat(meter["origin_utc"].replace("Z", "+00:00"))
        if origin.tzinfo is None or origin.utcoffset().total_seconds() != 0:
            raise BudgetError("Campaign accounting origin must be UTC")
        return "conservative_campaign_elapsed_wall_including_startup_eval_and_downtime_not_gpu_utilization"
    if meter["kind"] == "test_only_cpu_meter_v1":
        if (set(meter) != {"kind", "id"} or allow_test_meter is not True
                or gpu_meter is None or getattr(gpu_meter, "owner_id", None) != meter["id"]):
            raise BudgetError("Authenticated GPU occupancy owner is unavailable; wall time is not GPU time")
        return "test_only_cpu_fixture_not_native_gpu_measurement"
    if meter["kind"] != "test_only_owned_allocation_rpc_v1" or allow_test_meter is not True:
        # Native admission cannot be manufactured by a reviewed JSON or client attribute.
        raise BudgetError("Authenticated GPU occupancy owner is unavailable; native allocation admission is closed")
    if set(meter) != {"kind", "id", "allocation_review_sha256", "allocation_ledger_id"}:
        raise BudgetError("Allocation RPC meter requires its exact owner/review/ledger binding")
    _sha(meter["allocation_review_sha256"], "allocation review")
    _text(meter["allocation_ledger_id"], "allocation ledger identity")
    from .allocation_meter_client import AllocationMeterClient
    if type(gpu_meter) is not AllocationMeterClient:
        raise BudgetError("Allocation RPC requires the actual guardian socket client")
    receipt = gpu_meter.identity_receipt()
    expected = {"owner_id": meter["id"], "campaign_id": review["campaign_id"],
        "allocation_review_sha256": meter["allocation_review_sha256"],
        "allocation_ledger_id": meter["allocation_ledger_id"],
        "allocation_cap_seconds": review["gpu_occupied_seconds_cap"],
        "metric": "owned_gpu_allocation_seconds_v1", "probe_kind": "cpu_fixture_contexts_v1",
        "scope": "test_only_owned_allocation_rpc_v1", "native_admission": False}
    if any(receipt.get(key) != value or (key == "native_admission" and receipt.get(key) is not False)
            for key, value in expected.items()):
        raise BudgetError("Allocation RPC campaign/owner/ledger/review/cap/scope differs from the budget review")
    # Must finish a live guardian check before a ledger or identity marker is created.
    gpu_meter.check_alive_and_deadline()
    return "test_only_owned_allocation_rpc_not_native_gpu_measurement"


class BudgetOwner:
    """One attempt lease and a cross-process campaign transaction lock.

    A pending update remains charged after an OS/process loss. Acquire a new
    lease to continue an attempt; ordinal allocation uses durable reservations,
    never checkpoint-local successful counts. A missing initialized ledger is
    corruption, not permission to start again at zero.
    """
    @classmethod
    def authenticate(cls, review_file, expected_review_sha256, ledger_path, attempt_id,
                     actual_identity, *, resume_checkpoint=None, gpu_meter=None, allow_test_meter=False,
                     evaluation_only=False, now=None):
        if type(evaluation_only) is not bool:
            raise BudgetError("Evaluation-only admission must be an explicit boolean")
        _sha(expected_review_sha256, "budget review")
        review_bytes = Path(review_file).read_bytes()
        if hashlib.sha256(review_bytes).hexdigest() != expected_review_sha256:
            raise BudgetError("Budget review bytes differ from the expected SHA-256")
        review = json.loads(review_bytes)
        required = {"schema", "campaign_id", "ledger_id", "ledger_path", "paid_update_cap",
                    "pilot_paid_update_cap", "absolute_deadline_utc", "max_attempt_wall_seconds",
                    "gpu_occupied_seconds_cap", "gpu_meter_owner", "review_reference", "attempts"}
        if not isinstance(review, dict) or set(review) != required or review["schema"] != REVIEW_SCHEMA:
            raise BudgetError("Unknown or incomplete shared budget review")
        for key in ("campaign_id", "ledger_id", "review_reference"):
            _text(review[key], key)
        cap = review["paid_update_cap"]
        deadline = _validate_limits(review)
        path = _absolute(review["ledger_path"])
        if _absolute(str(ledger_path)) != path or not path.parent.is_dir():
            raise BudgetError("Worker ledger path differs from the reviewed shared path")
        if path.is_symlink():
            raise BudgetError("Ledger symlinks are forbidden")
        if not isinstance(review["attempts"], dict) or attempt_id not in review["attempts"]:
            raise BudgetError("Attempt is not in the hash-pinned budget review")
        for name, item in review["attempts"].items():
            cls._validate_attempt_identity(name, item, cap, open_main=_count_only(review))
        if evaluation_only and review["attempts"][attempt_id]["role"] != "diagnostic":
            raise BudgetError("Zero-paid evaluation requires its own diagnostic attempt")
        actual_identity = deepcopy(actual_identity)
        restart_parent = None
        expected_parent = review["attempts"][attempt_id]["parent_checkpoint_sha256"]
        if (resume_checkpoint is not None and actual_identity.get("parent_checkpoint_sha256") != expected_parent
                and actual_identity.get("parent_checkpoint_sha256") == digest_file(resume_checkpoint)):
            # Restarting a saved worker must not redefine its original branch
            # parent. Authenticate this exact same-attempt receipt under lock.
            restart_parent = (str(Path(resume_checkpoint).resolve()), digest_file(resume_checkpoint))
            actual_identity["parent_checkpoint_sha256"] = expected_parent
        if set(actual_identity) != IDENTITY_KEYS or any(
                review["attempts"][attempt_id][key] != actual_identity[key]
                for key in IDENTITY_KEYS - {"requested_updates"}):
            raise BudgetError("Actual source/contract/seed/stage/parent/updates differ from reviewed attempt")
        allocation = review["attempts"][attempt_id]["requested_updates"]
        requested = actual_identity["requested_updates"]
        if requested is None:
            if allocation is not None:
                raise BudgetError("An open worker requires an explicitly open main attempt")
        else:
            _count(requested, "worker requested updates", 1)
        if allocation is not None and requested > allocation:
            raise BudgetError("Worker request exceeds the reviewed attempt allocation")
        meter_scope = _authenticate_meter(review, gpu_meter, allow_test_meter)
        obj = cls()
        obj.review, obj.review_sha256, obj.path = review, expected_review_sha256, path
        obj.attempt_id, obj.identity = attempt_id, deepcopy(actual_identity)
        obj.evaluation_only = evaluation_only
        obj.gpu_meter, obj.now = gpu_meter, now or (lambda: datetime.now(timezone.utc))
        obj._gpu_meter_scope = meter_scope
        obj._last_gpu_meter_seconds = 0.
        obj._elapsed_clock_utc = None
        obj._elapsed_clock_boottime = None
        obj.deadline, obj.closed = deadline, False
        # New mainline attempts may be appended after a repair or stage change.
        # Existing attempt identities are still immutable in _validate_state;
        # adding one never changes the campaign cap or refunds reservations.
        obj.policy = {k: deepcopy(v) for k, v in review.items() if k not in ("review_reference", "attempts")}
        obj.policy_sha256 = _digest(obj.policy)
        obj.lease_id = uuid.uuid4().hex
        obj.lease_started = obj.now()
        obj._lease = (path.parent / (path.name + ".attempt-" + _digest(attempt_id) + ".lock")).open("a+")
        try:
            fcntl.flock(obj._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with obj._transaction(initialize=True) as state:
                if restart_parent is not None:
                    matches = [row for row in state["checkpoints"].values()
                        if row["status"] == "registered" and row["path"] == restart_parent[0]
                        and row["checkpoint_sha256"] == restart_parent[1]
                        and row["metadata"]["attempt_id"] == attempt_id]
                    seal = Path(restart_parent[0]).with_suffix(".checkpoint.json")
                    if len(matches) != 1 or not seal.is_file() or digest_file(seal) != matches[0]["seal_sha256"]:
                        raise BudgetError("Restart parent must be an intact registered same-attempt checkpoint")
                attempt = state["attempts"].get(attempt_id)
                if attempt is None:
                    attempt = {"identity": deepcopy(review["attempts"][attempt_id]), "leases": [],
                               "first_started_at": obj.lease_started.isoformat()}
                    state["attempts"][attempt_id] = attempt
                elif attempt["identity"] != review["attempts"][attempt_id]:
                    raise BudgetError("Existing attempt identity cannot be changed")
                obj.attempt_started = datetime.fromisoformat(attempt["first_started_at"])
                obj._check_limits(state)
                summary = obj._summary(state)
                if summary["paid_updates_attempt"] and resume_checkpoint is None:
                    raise BudgetError("A spent attempt requires a registered resume checkpoint; fresh branches need their own identity")
                if evaluation_only and summary["paid_updates_attempt"]:
                    raise BudgetError("Evaluation-only attempts must contain zero paid reservations")
                if not evaluation_only and requested is not None and (
                        cap is not None and requested > cap - summary["paid_updates_global"]
                        or allocation is not None and requested > allocation - summary["paid_updates_attempt"]):
                    raise BudgetLimit("Requested work exceeds durable remaining consumption")
                if (not evaluation_only and attempt["identity"]["role"] in ("pilot", "diagnostic")
                        and review["pilot_paid_update_cap"] is not None
                        and requested > review["pilot_paid_update_cap"] - obj._pilot_count(state)):
                    raise BudgetLimit("Requested pilot work exceeds shared remaining consumption")
                attempt["leases"].append({"id": obj.lease_id, "review_sha256": expected_review_sha256,
                    "pid": os.getpid(), "started_at": obj.lease_started.isoformat(), "closed": False})
            if resume_checkpoint is not None:
                obj.verify_resume(resume_checkpoint)
            return obj
        except BaseException:
            obj._lease.close()
            raise

    @classmethod
    def amend_paid_cap_to_count_only(cls, review_file, expected_review_sha256, ledger_path, *,
            expected_previous_policy_sha256, approval_reference, now=None):
        """Apply the sole20000-to-None change while every actual owner is idle.

        The caller authenticates the completed parent and its closed leases
        before calling. This operation owns accounting only: no training
        lineage, capability clock, new attempt, counter reset or time extension.
        The original identity marker remains the immutable policy root.
        """
        _sha(expected_review_sha256, "count-only review")
        _sha(expected_previous_policy_sha256, "previous campaign policy")
        _text(approval_reference, "count-only approval reference")
        raw_review = Path(review_file).read_bytes()
        if hashlib.sha256(raw_review).hexdigest() != expected_review_sha256:
            raise BudgetError("Count-only review differs from its pinned SHA-256")
        review = json.loads(raw_review)
        if (not isinstance(review, dict) or not isinstance(review.get("attempts"), dict)
                or "paid_update_cap" not in review or review["paid_update_cap"] is not None):
            raise BudgetError("Count-only amendment requires an explicit None review")
        _text(review.get("review_reference"), "review reference")
        path = _absolute(str(ledger_path))
        if review.get("ledger_path") != str(path) or path.is_symlink() or not path.is_file():
            raise BudgetError("Count-only amendment requires the existing canonical ledger")
        policy = {key: deepcopy(value) for key, value in review.items()
                  if key not in ("review_reference", "attempts")}
        previous = {**deepcopy(policy), "paid_update_cap": PAID_CAP}
        if _digest(previous) != expected_previous_policy_sha256:
            raise BudgetError("Count-only amendment cannot change any other campaign policy field")
        current_sha = _digest(policy)
        moment = (now or (lambda: datetime.now(timezone.utc)))()
        if moment.tzinfo is None or moment.utcoffset().total_seconds() != 0:
            raise BudgetError("Count-only amendment timestamp must be UTC")
        marker = path.with_name(path.name + ".identity.json")
        lease_files = []
        with path.with_name(path.name + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (not marker.is_file() or json.loads(marker.read_text()) != {
                    "ledger_id": review["ledger_id"],
                    "campaign_policy_sha256": expected_previous_policy_sha256}):
                raise BudgetError("Count-only amendment cannot replace the original ledger identity root")
            state = json.loads(path.read_text())
            obj = cls()
            obj.review = review
            obj.policy, obj.policy_sha256 = previous, expected_previous_policy_sha256
            if "paid_cap_amendment" in state:
                obj.policy, obj.policy_sha256 = policy, current_sha
            obj._validate_state(state)
            try:
                # Never wait on an attempt lock while holding the ledger lock:
                # workers acquire those locks in the opposite order.
                for attempt in sorted(state["attempts"]):
                    stream = (path.parent / (path.name + ".attempt-" + _digest(attempt) + ".lock")).open("a+")
                    lease_files.append(stream)
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise BudgetError("Count-only amendment requires every actual attempt lease to be idle") from error
                if "paid_cap_amendment" in state:
                    receipt = state["paid_cap_amendment"]
                    if (receipt["review_sha256"] != expected_review_sha256
                            or receipt["approval_reference"] != approval_reference):
                        raise BudgetError("Count-only amendment was already applied with a different reviewed receipt")
                    return deepcopy(receipt)
                amendment = {"previous_policy_sha256": expected_previous_policy_sha256,
                    "current_policy_sha256": current_sha, "review_sha256": expected_review_sha256,
                    "approval_reference": approval_reference, "amended_at_utc": moment.isoformat(),
                    "paid_updates_at_amendment": len(state["reservations"])}
                state.update(campaign_policy=policy, campaign_policy_sha256=current_sha,
                             paid_cap_amendment=amendment)
                obj.policy, obj.policy_sha256 = policy, current_sha
                obj._validate_state(state)
                # One durable rename publishes policy and receipt together.
                # Reservations, historical checkpoints and marker are untouched.
                _write(path, state)
                return deepcopy(amendment)
            finally:
                for stream in lease_files:
                    stream.close()

    @classmethod
    def _prepare_count_only_limits(cls, state, review, review_sha256, previous_sha256,
                                   approval_reference, moment):
        """Validate a captured ledger without I/O; publication still needs locks."""
        _sha(review_sha256, "count-only review")
        _sha(previous_sha256, "previous campaign policy")
        _text(approval_reference, "count-only approval reference")
        _text(review.get("review_reference"), "review reference")
        if moment.tzinfo is None or moment.utcoffset().total_seconds() != 0:
            raise BudgetError("Count-only amendment timestamp must be UTC")
        policy = {key: deepcopy(value) for key, value in review.items()
                  if key not in ("review_reference", "attempts")}
        if not _count_only(policy) or not isinstance(review.get("attempts"), dict):
            raise BudgetError("Count-only migration must explicitly remove every budget limit")
        _validate_limits(policy)
        previous = (state["count_only_limits_amendment"]["previous_policy"]
                    if "count_only_limits_amendment" in state else state["campaign_policy"])
        if (previous.get("paid_update_cap") is not None or _count_only(previous)
                or _digest(previous) != previous_sha256
                or policy != {**previous, **dict.fromkeys(COUNT_ONLY_FIELDS)}):
            raise BudgetError("Count-only migration can only remove the remaining limits of the existing ledger")
        for name, attempt in state["attempts"].items():
            if review["attempts"].get(name) != attempt["identity"]:
                raise BudgetError("Count-only migration must retain every historical attempt identity")
        for name, item in review["attempts"].items():
            cls._validate_attempt_identity(name, item, None, open_main=True)
        obj = cls()
        obj.policy = state["campaign_policy"]
        obj.policy_sha256 = _digest(obj.policy)
        obj.review = {**obj.policy, "review_reference": review["review_reference"],
                      "attempts": review["attempts"]}
        obj._validate_state(state)
        root = obj._policy_anchor(state)
        if "count_only_limits_amendment" in state:
            receipt = state["count_only_limits_amendment"]
            if receipt["review_sha256"] != review_sha256 or receipt["approval_reference"] != approval_reference:
                raise BudgetError("Count-only limits were already migrated with a different receipt")
            return deepcopy(state), root
        receipt = {"previous_policy": deepcopy(previous), "previous_policy_sha256": previous_sha256,
            "current_policy_sha256": _digest(policy), "review_sha256": review_sha256,
            "approval_reference": approval_reference, "amended_at_utc": moment.isoformat(),
            "paid_updates_at_amendment": len(state["reservations"])}
        candidate = {**deepcopy(state), "campaign_policy": policy,
                     "campaign_policy_sha256": _digest(policy), "count_only_limits_amendment": receipt}
        obj.policy, obj.policy_sha256, obj.review = policy, _digest(policy), review
        obj._validate_state(candidate)
        return candidate, root

    @classmethod
    def amend_limits_to_count_only(cls, review_file, expected_review_sha256, ledger_path, *,
                                  expected_previous_policy_sha256, approval_reference, now=None):
        """Remove the remaining budget limits once, preserving all old evidence."""
        raw = Path(review_file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_review_sha256:
            raise BudgetError("Count-only review differs from its pinned SHA-256")
        review = json.loads(raw)
        path = _absolute(str(ledger_path))
        if review.get("ledger_path") != str(path) or path.is_symlink() or not path.is_file():
            raise BudgetError("Count-only migration requires the existing canonical ledger")
        leases = []
        with path.with_name(path.name + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(path.read_text())
            candidate, root = cls._prepare_count_only_limits(state, review,
                expected_review_sha256, expected_previous_policy_sha256, approval_reference,
                (now or (lambda: datetime.now(timezone.utc)))())
            marker = path.with_name(path.name + ".identity.json")
            if not marker.is_file() or json.loads(marker.read_text()) != {
                    "ledger_id": review["ledger_id"], "campaign_policy_sha256": root}:
                raise BudgetError("Count-only migration cannot replace the original identity root")
            try:
                for aid in sorted(state["attempts"]):
                    stream = (path.parent / (path.name + ".attempt-" + _digest(aid) + ".lock")).open("a+")
                    leases.append(stream)
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise BudgetError("Count-only migration requires every actual attempt lease to be idle") from error
                if candidate != state:
                    _write(path, candidate)
                return deepcopy(candidate["count_only_limits_amendment"])
            finally:
                for stream in leases:
                    stream.close()

    def _policy_anchor(self, state):
        """Connect the two bounded migrations to the immutable original root."""
        if "count_only_limits_amendment" in state:
            receipt = state["count_only_limits_amendment"]
            keys = {"previous_policy", "previous_policy_sha256", "current_policy_sha256", "review_sha256",
                    "approval_reference", "amended_at_utc", "paid_updates_at_amendment"}
            if not isinstance(receipt, dict) or set(receipt) != keys or not _count_only(self.policy):
                raise BudgetError("Invalid complete count-only amendment")
            previous = receipt["previous_policy"]
            if (not isinstance(previous, dict) or previous.get("paid_update_cap") is not None
                    or _count_only(previous) or self.policy != {**previous, **dict.fromkeys(COUNT_ONLY_FIELDS)}
                    or _digest(previous) != receipt["previous_policy_sha256"]
                    or receipt["current_policy_sha256"] != self.policy_sha256):
                raise BudgetError("Count-only amendment does not connect the previous and current policy")
            _validate_limits(previous)
            _sha(receipt["review_sha256"], "count-only review")
            _text(receipt["approval_reference"], "count-only approval reference")
            if _deadline(receipt["amended_at_utc"]) is None:
                raise BudgetError("Count-only amendment requires a UTC timestamp")
            paid = _count(receipt["paid_updates_at_amendment"], "count-only migration paid count")
            if paid > len(state["reservations"]) or paid < state.get("paid_cap_amendment", {}).get("paid_updates_at_amendment", 0):
                raise BudgetError("Count-only migration count differs from its history")
            old = BudgetOwner()
            old.policy, old.policy_sha256 = previous, receipt["previous_policy_sha256"]
            return old._policy_anchor({key: value for key, value in state.items()
                                       if key != "count_only_limits_amendment"})
        if _count_only(self.policy):
            raise BudgetError("Complete count-only policy requires the explicit limits amendment")
        if "paid_cap_amendment" not in state:
            if self.policy["paid_update_cap"] is None:
                raise BudgetError("Count-only policy requires the reviewed existing-ledger amendment")
            return self.policy_sha256
        receipt = state["paid_cap_amendment"]
        keys = {"previous_policy_sha256", "current_policy_sha256", "review_sha256",
                "approval_reference", "amended_at_utc", "paid_updates_at_amendment"}
        if (self.policy["paid_update_cap"] is not None or not isinstance(receipt, dict)
                or set(receipt) != keys):
            raise BudgetError("Only the single20000-to-None paid-cap amendment is supported")
        old = {**deepcopy(self.policy), "paid_update_cap": PAID_CAP}
        root = _digest(old)
        if receipt["previous_policy_sha256"] != root or receipt["current_policy_sha256"] != self.policy_sha256:
            raise BudgetError("Paid-cap amendment does not connect the original and current policy")
        _sha(receipt["review_sha256"], "amendment review")
        _text(receipt["approval_reference"], "amendment approval reference")
        paid = _count(receipt["paid_updates_at_amendment"], "amendment paid count")
        if paid > len(state["reservations"]) or paid > PAID_CAP:
            raise BudgetError("Paid-cap amendment count exceeds its original or current history")
        try:
            moment = datetime.fromisoformat(receipt["amended_at_utc"])
        except (ValueError, TypeError) as error:
            raise BudgetError("Invalid paid-cap amendment timestamp") from error
        if moment.tzinfo is None or moment.utcoffset().total_seconds() != 0:
            raise BudgetError("Paid-cap amendment timestamp must be UTC")
        return root

    def _empty(self):
        return {"schema": STATE_SCHEMA, "campaign_policy": self.policy,
                "campaign_policy_sha256": self.policy_sha256, "attempts": {}, "reservations": [],
                "checkpoints": {}, "gpu_meter_reading_seconds": 0.}

    @contextmanager
    def _transaction(self, *, initialize=False):
        if self.closed:
            raise BudgetError("Attempt lease is closed")
        lock = self.path.with_name(self.path.name + ".lock")
        with lock.open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            marker = self.path.with_name(self.path.name + ".identity.json")
            identity = {"ledger_id": self.review["ledger_id"], "campaign_policy_sha256": self.policy_sha256}
            if self.path.exists():
                state = json.loads(self.path.read_text())
                identity["campaign_policy_sha256"] = self._policy_anchor(state)
                if not marker.is_file() or json.loads(marker.read_text()) != identity:
                    raise BudgetError("Shared ledger identity marker is missing or changed")
            elif initialize and not marker.exists():
                if self.policy["paid_update_cap"] is None:
                    raise BudgetError("Count-only cannot initialize or reset a ledger; amend the existing history")
                _write(marker, identity)
                state = self._empty()
            else:
                raise BudgetError("Initialized shared ledger is missing; refusing a consumption reset")
            self._validate_state(state)
            before = _encoded(state)
            yield state
            self._validate_state(state)
            if before != _encoded(state) or not self.path.exists():
                _write(self.path, state)

    @staticmethod
    def _validate_attempt_identity(name, item, cap, *, open_main=False):
        _text(name, "attempt identity")
        if not isinstance(item, dict) or set(item) != IDENTITY_KEYS | {"role"}:
            raise BudgetError("Incomplete approved attempt identity")
        if item["role"] not in ("pilot", "diagnostic", "main"):
            raise BudgetError("Unknown paid attempt role")
        for key in ("contract_sha256", "training_source_sha256"):
            _sha(item[key], key)
        if item["parent_checkpoint_sha256"] is not None:
            _sha(item["parent_checkpoint_sha256"], "parent checkpoint")
        _count(item["seed"], "training seed")
        _text(item["stage"], "stage")
        allocation = item["requested_updates"]
        if allocation is None:
            if not open_main or item["role"] != "main" or cap is not None:
                raise BudgetError("Only complete count-only main attempts may have open update allocation")
            return
        _count(allocation, "requested paid updates", 1)
        if cap is not None and allocation > cap:
            raise BudgetError("Attempt request exceeds campaign ceiling")

    def _validate_state(self, state):
        expected_keys = set(self._empty()) | (set(state) & {"paid_cap_amendment", "count_only_limits_amendment"})
        if (set(state) != expected_keys or state["schema"] != STATE_SCHEMA
                or state["campaign_policy"] != self.policy
                or state["campaign_policy_sha256"] != self.policy_sha256
                or not isinstance(state["reservations"], list)):
            raise BudgetError("Shared ledger identity/schema mismatch")
        self._policy_anchor(state)
        ordinals = {}
        if not isinstance(state["attempts"], dict) or not isinstance(state["checkpoints"], dict):
            raise BudgetError("Invalid ledger attempt/checkpoint registry")
        for aid, attempt in state["attempts"].items():
            if (not isinstance(attempt, dict)
                    or set(attempt) != {"identity", "leases", "first_started_at"}
                    or not isinstance(attempt["leases"], list)):
                raise BudgetError("Stored attempt differs from the pinned review")
            self._validate_attempt_identity(aid, attempt["identity"], self.policy["paid_update_cap"],
                                            open_main=_count_only(self.policy))
            # Later zero-paid evaluations still validate their stored identity;
            # every paid attempt must also appear in this worker's review.
            if aid in self.review["attempts"]:
                if attempt["identity"] != self.review["attempts"][aid]:
                    raise BudgetError("Stored attempt differs from the pinned review")
            try:
                start = datetime.fromisoformat(attempt["first_started_at"])
            except (ValueError, TypeError) as exc:
                raise BudgetError("Invalid persistent attempt wall origin") from exc
            if start.tzinfo is None or start.utcoffset().total_seconds() != 0:
                raise BudgetError("Attempt wall origin must be UTC")
        for seq, row in enumerate(state["reservations"], 1):
            aid = row["attempt_id"]
            if aid not in self.review["attempts"]:
                raise BudgetError("Paid attempt is absent from the pinned review")
            ordinals[aid] = ordinals.get(aid, 0) + 1
            if (type(row["sequence"]) is not int or row["sequence"] != seq
                    or type(row["ordinal"]) is not int or row["ordinal"] != ordinals[aid]
                    or aid not in state["attempts"] or row["phase"] not in ("actor", "critic")
                    or row["status"] not in ("reserved", "completed", "failed", "postcheck_failed")
                    or type(row["optimizer_completed"]) is not bool
                    or row["optimizer_completed"] != (row["status"] in ("completed", "postcheck_failed"))):
                raise BudgetError("Invalid or nonmonotonic paid reservation")
        cap = self.policy["paid_update_cap"]
        pilot_cap = self.policy["pilot_paid_update_cap"]
        if ((cap is not None and len(state["reservations"]) > cap)
                or pilot_cap is not None and self._pilot_count(state) > pilot_cap):
            raise BudgetError("Stored consumption exceeds its reviewed cap")
        if any(state["attempts"][aid]["identity"]["requested_updates"] is not None
               and count > state["attempts"][aid]["identity"]["requested_updates"] for aid, count in ordinals.items()):
            raise BudgetError("Stored attempt consumption exceeds its reviewed allocation")

    def _pilot_count(self, state):
        return sum(state["attempts"][r["attempt_id"]]["identity"]["role"] in ("pilot", "diagnostic")
                   for r in state["reservations"])

    def _check_limits(self, state):
        # Durable transactions recheck the persistent meter high-water mark.
        # The socket fixture asks its actual guardian, never a JSON counter.
        # Production occupancy admission remains a separate, closed gate.
        first_start = datetime.fromisoformat(state["attempts"][self.attempt_id]["first_started_at"])
        state["gpu_meter_reading_seconds"] = self._runtime_limits(
            first_start, state["gpu_meter_reading_seconds"])

    def _campaign_elapsed_now(self, now, origin, minimum_seconds):
        """Keep charging through host clock corrections, including suspend.

        UTC still admits forward jumps. BOOTTIME prevents a backward wall
        correction from refunding elapsed time or extending a deadline. The
        durable high-water mark also survives process replacement.
        """
        tick = clock_gettime(CLOCK_BOOTTIME)
        previous_tick = self._elapsed_clock_boottime
        if (not math.isfinite(tick) or tick < 0
                or (previous_tick is not None and tick < previous_tick)):
            raise BudgetError("Campaign elapsed clock is invalid or moves backwards")
        if previous_tick is not None:
            now = max(now, self._elapsed_clock_utc + timedelta(seconds=tick - previous_tick))
        if minimum_seconds > 0:
            now = max(now, origin + timedelta(seconds=minimum_seconds))
        self._elapsed_clock_utc, self._elapsed_clock_boottime = now, tick
        return now

    def _runtime_limits(self, first_start, minimum_gpu_seconds):
        now = self.now()
        meter = self.review["gpu_meter_owner"]
        if now.tzinfo is not None and meter["kind"] == "campaign_elapsed_wall_v1":
            origin = datetime.fromisoformat(meter["origin_utc"].replace("Z", "+00:00"))
            now = self._campaign_elapsed_now(
                now, origin, max(minimum_gpu_seconds, self._last_gpu_meter_seconds))
        if now.tzinfo is None:
            raise BudgetError("Attempt clock requires an explicit timezone")
        if self.deadline is not None and now >= self.deadline:
            raise BudgetLimit("Campaign absolute wall deadline reached")
        # A restart does not refresh this cap. Conservatively include downtime
        # since the first durable attempt start; this is not GPU occupancy.
        elapsed = (now - first_start).total_seconds()
        if elapsed < 0:
            raise BudgetError("Attempt wall clock moves before its durable origin")
        if self.review["max_attempt_wall_seconds"] is not None and elapsed >= self.review["max_attempt_wall_seconds"]:
            raise BudgetLimit("Attempt wall cap reached")
        if meter["kind"] == "campaign_elapsed_wall_v1":
            seconds = (now - origin).total_seconds()
        else:
            seconds = self.gpu_meter.read_total_gpu_seconds()
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or seconds < max(minimum_gpu_seconds, self._last_gpu_meter_seconds)):
            raise BudgetError("GPU meter reading is invalid or moves backwards")
        if self.review["gpu_occupied_seconds_cap"] is not None and seconds >= self.review["gpu_occupied_seconds_cap"]:
            raise BudgetLimit("Campaign GPU occupancy cap reached")
        self._last_gpu_meter_seconds = float(seconds)
        return float(seconds)

    def check(self):
        with self._transaction() as state:
            self._check_limits(state)

    def check_runtime(self):
        """O(1) wall/meter check for rollout steps, not a paid reservation.

        Read a live owner measurement on every call. The authenticated wall
        origin includes downtime. Global paid ceilings are checked atomically
        by reserve_update before every optimizer call, never from this cache.
        """
        if self.closed:
            raise BudgetError("Attempt lease is closed")
        self._runtime_limits(self.attempt_started, self._last_gpu_meter_seconds)

    def execution_wall_limit(self, requested_seconds):
        """A count-only mainline must not inherit the worker's old time cap."""
        if _count_only(self.policy) and self.review["attempts"][self.attempt_id]["role"] == "main":
            return None
        return _positive(requested_seconds, "worker execution wall cap")

    def _summary(self, state):
        own = [r for r in state["reservations"] if r["attempt_id"] == self.attempt_id]
        return {"campaign_id": self.review["campaign_id"], "ledger_id": self.review["ledger_id"],
                "campaign_policy_sha256": self.policy_sha256, "attempt_id": self.attempt_id,
                "paid_updates_global": len(state["reservations"]), "paid_updates_attempt": len(own),
                "pilot_paid_updates": self._pilot_count(state), "paid_update_cap": self.policy["paid_update_cap"],
                "optimizer_completed_attempt": sum(r["optimizer_completed"] for r in own),
                "actor_completed_attempt": sum(r["optimizer_completed"] and r["phase"] == "actor" for r in own),
                "critic_completed_attempt": sum(r["optimizer_completed"] and r["phase"] == "critic" for r in own),
                "attempt_wall_origin_utc": state["attempts"][self.attempt_id]["first_started_at"],
                "attempt_wall_scope": "elapsed_since_first_durable_start_including_downtime",
                "gpu_meter_scope": self._gpu_meter_scope,
                "gpu_meter_owner_binding": deepcopy(self.review["gpu_meter_owner"])}

    def summary(self):
        with self._transaction() as state:
            return self._summary(state)

    def reserve_update(self, *, phase, ordinal=None):
        if self.evaluation_only:
            raise BudgetError("Evaluation-only admission forbids optimizer reservations")
        if phase not in ("actor", "critic"):
            raise BudgetError("Paid update must declare actor or critic phase")
        with self._transaction() as state:
            self._check_limits(state)
            summary = self._summary(state)
            cap = self.policy["paid_update_cap"]
            if cap is not None and summary["paid_updates_global"] >= cap:
                raise BudgetLimit("Common20000 paid consumption exhausted")
            own = state["attempts"][self.attempt_id]["identity"]
            if (own["role"] in ("pilot", "diagnostic") and self.review["pilot_paid_update_cap"] is not None
                    and self._pilot_count(state) >= self.review["pilot_paid_update_cap"]):
                raise BudgetLimit("Pilot/diagnostic shared paid cap exhausted")
            next_ordinal = summary["paid_updates_attempt"] + 1
            if own["requested_updates"] is not None and next_ordinal > own["requested_updates"]:
                raise BudgetLimit("Attempt paid-update request exhausted")
            if ordinal is not None and (type(ordinal) is not int or ordinal != next_ordinal):
                raise BudgetError("A paid ordinal cannot be replayed or skipped")
            row = {"sequence": len(state["reservations"]) + 1, "attempt_id": self.attempt_id,
                   "ordinal": next_ordinal, "lease_id": self.lease_id, "phase": phase,
                   "status": "reserved", "optimizer_completed": False, "error": None}
            state["reservations"].append(row)
            return {"ledger_id": self.review["ledger_id"], **{k: row[k] for k in
                    ("sequence", "attempt_id", "ordinal", "lease_id", "phase")}}

    def _row(self, state, token):
        if not isinstance(token, dict) or set(token) != {"ledger_id", "sequence", "attempt_id", "ordinal", "lease_id", "phase"}:
            raise BudgetError("Invalid paid update token")
        seq = _count(token["sequence"], "reservation sequence", 1)
        _count(token["ordinal"], "reservation ordinal", 1)
        if (seq > len(state["reservations"]) or token["ledger_id"] != self.review["ledger_id"]
                or token["attempt_id"] != self.attempt_id or token["lease_id"] != self.lease_id):
            raise BudgetError("Reservation belongs to another ledger/attempt/lease")
        row = state["reservations"][seq - 1]
        if any(row[k] != token[k] for k in token if k != "ledger_id"):
            raise BudgetError("Reservation identity changed")
        return row

    def complete_update(self, token, *, actor_optimized):
        if type(actor_optimized) is not bool or actor_optimized != (token["phase"] == "actor"):
            raise BudgetError("Completion phase differs from the paid reservation")
        with self._transaction() as state:
            row = self._row(state, token)
            if row["status"] not in ("reserved", "completed"):
                raise BudgetError("Failed/unknown update cannot acquire completion credit")
            row.update(status="completed", optimizer_completed=True)

    def verify_update(self, token, *, allow_previous_lease=False):
        """Read a real durable reservation without acquiring spending credit.

        Old-lease reads are only for same-attempt checkpoint reconciliation.
        They cannot complete, refund or replay the reservation in this lease.
        """
        if type(allow_previous_lease) is not bool:
            raise BudgetError("Previous-lease verification must be explicit")
        with self._transaction() as state:
            if not allow_previous_lease:
                row = self._row(state, token)
            else:
                keys = {"ledger_id", "sequence", "attempt_id", "ordinal", "lease_id", "phase"}
                if not isinstance(token, dict) or set(token) != keys:
                    raise BudgetError("Invalid paid update token")
                sequence = _count(token["sequence"], "reservation sequence", 1)
                _count(token["ordinal"], "reservation ordinal", 1)
                if (sequence > len(state["reservations"])
                        or token["ledger_id"] != self.review["ledger_id"]
                        or token["attempt_id"] != self.attempt_id):
                    raise BudgetError("Reservation belongs to another ledger/attempt")
                row = state["reservations"][sequence - 1]
                if any(row[key] != token[key] for key in keys - {"ledger_id"}):
                    raise BudgetError("Reservation identity changed")
            return deepcopy(dict(token=token, row=row, identity=self.identity,
                                 review_sha256=self.review_sha256, summary=self._summary(state)))

    def fail_update(self, token, error):
        with self._transaction() as state:
            row = self._row(state, token)
            row.update(status="postcheck_failed" if row["optimizer_completed"] else "failed",
                       error=str(error)[:2000])

    def prepare_checkpoint(self, path):
        path = Path(path).resolve()
        with self._transaction() as state:
            own = [r for r in state["reservations"] if r["attempt_id"] == self.attempt_id]
            if any(r["lease_id"] == self.lease_id and r["status"] != "completed" for r in own):
                raise BudgetError("Current lease has unverified work; checkpoint cannot be sealed")
            identifier = uuid.uuid4().hex
            metadata = {"schema": RECEIPT_SCHEMA, **self._summary(state),
                        "review_sha256": self.review_sha256, "checkpoint_id": identifier}
            state["checkpoints"][identifier] = {"status": "pending", "path": str(path),
                "metadata": metadata, "checkpoint_sha256": None, "seal_sha256": None}
            return deepcopy(metadata)

    def register_checkpoint(self, path, metadata):
        path = Path(path).resolve()
        seal = path.with_suffix(".checkpoint.json")
        native_sha = digest_file(path)
        sidecar = json.loads(seal.read_text())
        if (sidecar.get("checkpoint_sha256") != native_sha or sidecar.get("training_budget") != metadata
                or sidecar.get("contract_sha256") != self.identity["contract_sha256"]):
            raise BudgetError("Checkpoint file/seal does not match its prepared budget receipt")
        for saved in (path, seal):
            with saved.open("rb") as stream:
                os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        with self._transaction() as state:
            row = state["checkpoints"].get(metadata["checkpoint_id"])
            if (row is None or row["status"] != "pending" or row["metadata"] != metadata
                    or row["path"] != str(path) or metadata["attempt_id"] != self.attempt_id):
                raise BudgetError("Checkpoint publication was not prepared by this attempt")
            now = self._summary(state)
            for key in ("paid_updates_attempt", "optimizer_completed_attempt", "actor_completed_attempt", "critic_completed_attempt"):
                if metadata[key] != now[key]:
                    raise BudgetError("Optimization advanced during checkpoint publication")
            row.update(status="registered", checkpoint_sha256=native_sha, seal_sha256=digest_file(seal))

    def verify_resume(self, path, metadata=None):
        path = Path(path).resolve()
        native_sha = digest_file(path)
        seal = path.with_suffix(".checkpoint.json")
        with self._transaction() as state:
            matches = [r for r in state["checkpoints"].values() if r["status"] == "registered"
                       and r["path"] == str(path) and r["checkpoint_sha256"] == native_sha]
            if len(matches) != 1 or not seal.is_file() or digest_file(seal) != matches[0]["seal_sha256"]:
                raise BudgetError("Resume requires an intact registered checkpoint, not a pending/unsealed file")
            expected = matches[0]["metadata"]
            if metadata is not None and metadata != expected:
                raise BudgetError("Checkpoint infos misstate shared budget identity/counters")
            if (expected["attempt_id"] != self.attempt_id
                    and self.identity["parent_checkpoint_sha256"] != native_sha):
                raise BudgetError("Exact restart cannot switch the registered attempt")
            return deepcopy(expected)

    def verify_checkpoint_receipt(self, metadata):
        """Authenticate a registered same-attempt receipt and its actual files."""
        if not isinstance(metadata, dict) or not isinstance(metadata.get("checkpoint_id"), str):
            raise BudgetError("Registered checkpoint receipt required")
        with self._transaction() as state:
            row = state["checkpoints"].get(metadata["checkpoint_id"])
            if (row is None or row["status"] != "registered" or row["metadata"] != metadata
                    or metadata.get("attempt_id") != self.attempt_id):
                raise BudgetError("Pending/foreign/mismatched checkpoint receipt cannot restore progress")
            path = row["path"]
        return self.verify_resume(path, metadata)

    def close(self):
        if self.closed:
            return
        try:
            with self._transaction() as state:
                leases = state["attempts"][self.attempt_id]["leases"]
                next(row for row in leases if row["id"] == self.lease_id)["closed"] = True
        finally:
            self.closed = True
            self._lease.close()
