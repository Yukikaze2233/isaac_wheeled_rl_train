"""Candidate accounting for guardian-owned GPU *allocation* intervals.

This is not active-compute metering, exclusive GPU scheduling, containment, or
native admission. It sends no signals and starts no workloads. A future guardian
must reserve before GO, honor the returned cleanup/start deadlines, and close
only after its actual owned subtree is reaped and NVML contexts are absent.
Unknown owner loss never releases a reservation. No training entry uses this
module yet. All limits and device identities must come from a pinned review.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import ctypes
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid


METRIC = "owned_gpu_allocation_seconds_v1"
REVIEW_SCHEMA = "guardian-gpu-allocation-review-v1"
STATE_SCHEMA = "guardian-gpu-allocation-state-v1"


class AllocationError(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def argv_sha256(argv):
    if not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(v, str) for v in argv):
        raise AllocationError("Workload argv must be an explicit string sequence")
    return hashlib.sha256(_json_bytes(list(argv))).hexdigest()


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise AllocationError("Invalid SHA-256 identity")
    return value


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise AllocationError(f"{name} must be explicitly positive and finite")
    return value


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise AllocationError(f"{name} must be a nonnegative integer")
    return value


def _duration_ns(value, name):
    result = round(_positive(value, name) * 1e9)
    if result <= 0:
        raise AllocationError(f"{name} is below the supported nanosecond clock resolution")
    return result


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise AllocationError("Missing explicit identity")
    return value


def _absolute(value):
    path = Path(_text(value))
    if not path.is_absolute() or path.resolve() != path:
        raise AllocationError("Paths must be canonical absolute paths")
    return path


def _atomic(path, value):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def process_identity(pid):
    """Host PID identity only; this does not prove ownership."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": int(fields[19]), "ppid": int(fields[1]), "state": fields[0]}


def host_identity():
    return {"machine_id_sha256": sha256("/etc/machine-id"),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def interval_union_ns(intervals):
    """Exact integer union, separately per physical UUID; no utilization sum."""
    by_device = {}
    for devices, start, end in intervals:
        _integer(start, "interval start")
        _integer(end, "interval end")
        if end < start or len(set(devices)) != len(devices):
            raise AllocationError("Invalid allocation interval")
        for device in devices:
            by_device.setdefault(device, []).append((start, end))
    total = 0
    for values in by_device.values():
        left = right = None
        for start, end in sorted(values):
            if left is None:
                left, right = start, end
            elif start <= right:
                right = max(right, end)
            else:
                total += right - left
                left, right = start, end
        if left is not None:
            total += right - left
    return total


class CPUFixtureProbe:
    """Explicitly non-native test data; never accepted as an NVML owner."""
    kind = "cpu_fixture_contexts_v1"

    def __init__(self):
        self.rows = {}
        self.failure = None

    def snapshot(self, devices):
        if self.failure:
            raise AllocationError(self.failure)
        return {d: {"contexts": deepcopy(self.rows.get(d, [])), "device_busy_percent_sample": None}
                for d in devices}


class NvmlContextProbe:
    """Read-only NVML context inventory, including graphics (Kit can use it).

    PID/context snapshots cannot reconstruct exact residency between polls and
    the busy sample is device-wide. They are evidence, not the allocation clock.
    MIG and MPS attribution are not implemented and therefore rejected.
    """
    kind = "nvml_context_snapshot_v1"

    def snapshot(self, devices):
        class Process(ctypes.Structure):
            _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                        ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]
        class Utilization(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        lib.nvmlErrorString.argtypes = [ctypes.c_int]
        lib.nvmlErrorString.restype = ctypes.c_char_p
        def check(code):
            if code != 0:
                raise AllocationError("NVML read failed: " + lib.nvmlErrorString(code).decode())
        check(lib.nvmlInit_v2())
        try:
            count = ctypes.c_uint()
            lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
            check(lib.nvmlDeviceGetCount_v2(ctypes.byref(count)))
            inventory = []
            for index in range(count.value):
                handle = ctypes.c_void_p()
                lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
                check(lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)))
                value = ctypes.create_string_buffer(128)
                lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
                check(lib.nvmlDeviceGetUUID(handle, value, len(value)))
                inventory.append(value.value.decode())
            if set(inventory) != set(devices) or len(inventory) != len(devices):
                raise AllocationError("Actual physical GPU inventory differs from pinned review")
            result = {}
            for device in devices:
                handle = ctypes.c_void_p()
                lib.nvmlDeviceGetHandleByUUID.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
                check(lib.nvmlDeviceGetHandleByUUID(device.encode(), ctypes.byref(handle)))
                mode, pending = ctypes.c_uint(), ctypes.c_uint()
                lib.nvmlDeviceGetMigMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
                code = lib.nvmlDeviceGetMigMode(handle, ctypes.byref(mode), ctypes.byref(pending))
                # NVML_ERROR_NOT_SUPPORTED=3 means a non-MIG device.
                if code not in (0, 3):
                    check(code)
                if code == 0 and (mode.value or pending.value):
                    raise AllocationError("MIG ownership is not implemented")
                contexts = {}
                for name, context_kind in (("nvmlDeviceGetComputeRunningProcesses_v3", "compute"),
                                           ("nvmlDeviceGetGraphicsRunningProcesses_v3", "graphics"),
                                           ("nvmlDeviceGetMPSComputeRunningProcesses_v3", "mps")):
                    fn = getattr(lib, name)
                    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(Process)]
                    count = ctypes.c_uint(0)
                    code = fn(handle, ctypes.byref(count), None)
                    if code not in (0, 7):  # NVML_ERROR_INSUFFICIENT_SIZE
                        check(code)
                    if count.value:
                        array = (Process * count.value)()
                        # A raced size change is deliberately a failed snapshot,
                        # never permission to treat the device as empty.
                        check(fn(handle, ctypes.byref(count), array))
                        if context_kind == "mps":
                            raise AllocationError("MPS ownership is not implemented")
                        for info in array[:count.value]:
                            identity = process_identity(info.pid)
                            row = contexts.setdefault((info.pid, identity["start_ticks"]),
                                {"pid": info.pid, "start_ticks": identity["start_ticks"], "kinds": []})
                            row["kinds"].append(context_kind)
                utilization = Utilization()
                lib.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(Utilization)]
                check(lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization)))
                result[device] = {"contexts": list(contexts.values()),
                                  "device_busy_percent_sample": utilization.gpu}
            return result
        finally:
            lib.nvmlShutdown()


class AllocationOwner:
    """Durable candidate called by the actual guardian, never by arbitrary PIDs.

    File locks serialize campaign changes. The child registry must be the
    hash-pinned containment implementation, and the current process must be a
    Linux subreaper. This is local trusted-source provenance, not protection
    against malicious same-UID code or an administrator editing state.
    """
    def __init__(self, review_file, expected_sha256, ownership, probe, *, allow_cpu_fixture=False, clock_ns=None):
        raw = Path(review_file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != _hash(expected_sha256):
            raise AllocationError("Allocation review bytes changed")
        review = json.loads(raw)
        keys = {"schema", "metric", "campaign_id", "ledger_id", "owner_id", "ledger_path", "host",
                "source_sha256", "guardian_source", "gpu_uuids", "allocation_cap_seconds", "probe_kind", "segments",
                "evidence_max_age_seconds"}
        if not isinstance(review, dict) or set(review) != keys or review["schema"] != REVIEW_SCHEMA or review["metric"] != METRIC:
            raise AllocationError("Invalid allocation review schema/metric")
        for key in ("campaign_id", "ledger_id", "owner_id"):
            _text(review[key])
        if review["host"] != host_identity():
            raise AllocationError("Host/boot identity differs; restart requires reconciliation")
        if _hash(review["source_sha256"]) != sha256(__file__):
            raise AllocationError("Allocation source differs from reviewed bytes")
        if not isinstance(review["gpu_uuids"], list) or not review["gpu_uuids"] or len(set(review["gpu_uuids"])) != len(review["gpu_uuids"]):
            raise AllocationError("Physical GPU UUIDs must be explicit and unique")
        if any(not isinstance(d, str) or not d.startswith("GPU-") for d in review["gpu_uuids"]):
            raise AllocationError("Only physical GPU UUID identities are implemented")
        _positive(review["allocation_cap_seconds"], "allocation cap")
        _duration_ns(review["evidence_max_age_seconds"], "evidence maximum age")
        if not isinstance(review["segments"], dict) or not review["segments"]:
            raise AllocationError("Explicit reviewed allocation segments are required")
        for key, spec in review["segments"].items():
            _text(key)
            if set(spec) != {"attempt_id", "contract_sha256", "training_source_sha256", "workload_argv_sha256",
                            "gpu_uuids", "lease_seconds", "cleanup_seconds"}:
                raise AllocationError("Incomplete reviewed segment")
            _text(spec["attempt_id"])
            for name in ("contract_sha256", "training_source_sha256", "workload_argv_sha256"):
                _hash(spec[name])
            if (not spec["gpu_uuids"] or len(set(spec["gpu_uuids"])) != len(spec["gpu_uuids"])
                    or any(d not in review["gpu_uuids"] for d in spec["gpu_uuids"])):
                raise AllocationError("Segment devices differ from the reviewed inventory")
            if _duration_ns(spec["cleanup_seconds"], "cleanup reserve") >= _duration_ns(spec["lease_seconds"], "lease duration"):
                raise AllocationError("Cleanup reserve leaves no workload time")
        self.review, self.review_sha256 = review, expected_sha256
        self.path = _absolute(review["ledger_path"])
        if not self.path.parent.is_dir():
            raise AllocationError("Reviewed ledger parent must already exist")
        if type(probe) is CPUFixtureProbe:
            if allow_cpu_fixture is not True or review["probe_kind"] != probe.kind:
                raise AllocationError("CPU probe cannot impersonate a native meter")
        elif type(probe) is not NvmlContextProbe or review["probe_kind"] != probe.kind or clock_ns is not None:
            raise AllocationError("Native owner requires its explicit NVML probe and host monotonic clock")
        self.probe, self.clock_ns = probe, clock_ns or time.monotonic_ns
        self.ownership = ownership
        source = review["guardian_source"]
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise AllocationError("Guardian source identity missing")
        source_path = _absolute(source["path"])
        if _hash(source["sha256"]) != sha256(source_path):
            raise AllocationError("Guardian source bytes changed")
        if (type(ownership).__name__ != "OwnedChildren"
                or Path(inspect.getfile(type(ownership))).resolve() != source_path
                or type(ownership.api).__name__ != "LinuxProcessAPI"
                or Path(inspect.getfile(type(ownership.api))).resolve() != source_path):
            raise AllocationError("Must bind the actual pinned guardian OwnedChildren registry")
        value = ctypes.c_int()
        if ctypes.CDLL(None).prctl(37, ctypes.byref(value), 0, 0, 0) != 0 or value.value != 1:
            raise AllocationError("Current guardian is not an actual Linux child subreaper")
        if ownership.identities or ownership.pending or ownership.reaped:
            raise AllocationError("Allocation owner must attach before the guardian has owned any workload")
        self.guardian = process_identity(os.getpid())
        self.owner_id = review["owner_id"]
        self.policy_sha256 = hashlib.sha256(_json_bytes(review)).hexdigest()
        self.allocations = set()
        self._lease = self.path.with_name(self.path.name + f".owner-{os.getpid()}-{self.guardian['start_ticks']}.lock").open("a+")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self._transaction(initialize=True):
                pass
        except BaseException:
            self._lease.close()
            raise

    @contextmanager
    def _transaction(self, initialize=False):
        if self._lease.closed:
            raise AllocationError("Owner handle is closed")
        if host_identity() != self.review["host"]:
            raise AllocationError("Host boot changed; allocation cannot be reset")
        with self.path.with_name(self.path.name + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marker = self.path.with_name(self.path.name + ".identity.json")
            identity = {"ledger_id": self.review["ledger_id"], "policy_sha256": self.policy_sha256}
            if self.path.exists():
                if not marker.exists() or json.loads(marker.read_bytes()) != identity:
                    raise AllocationError("Allocation ledger identity marker changed")
                state = json.loads(self.path.read_bytes())
            elif initialize and not marker.exists():
                _atomic(marker, identity)
                state = {"schema": STATE_SCHEMA, "policy_sha256": self.policy_sha256, "segments": {},
                         "last_clock_ns": 0, "last_measured_elapsed_ns": 0}
            else:
                raise AllocationError("Initialized allocation ledger missing; refusing reset")
            if (set(state) != {"schema", "policy_sha256", "segments", "last_clock_ns", "last_measured_elapsed_ns"}
                    or state["schema"] != STATE_SCHEMA or state["policy_sha256"] != self.policy_sha256):
                raise AllocationError("Allocation ledger schema/source mismatch")
            _integer(state["last_clock_ns"], "durable monotonic clock")
            _integer(state["last_measured_elapsed_ns"], "measured elapsed")
            self._validate_segments(state)
            before = _json_bytes(state)
            yield state
            self._validate_segments(state)
            if _json_bytes(state) != before or not self.path.exists():
                _atomic(self.path, state)

    def _validate_segments(self, state):
        if not isinstance(state["segments"], dict):
            raise AllocationError("Invalid allocation segment registry")
        required = {"status", "started_ns", "lease_end_ns", "closed_ns", "guardian", "token",
                    "workload", "owned_identities", "last_observation", "observation_count"}
        for name, row in state["segments"].items():
            if (name not in self.review["segments"] or not isinstance(row, dict)
                    or not required <= set(row) or set(row) - required - {"cleanup_reaped", "evidence_error"}
                    or row["status"] not in ("reserved", "running", "closed", "unresolved")):
                raise AllocationError("Invalid stored allocation state")
            start = _integer(row["started_ns"], "allocation start")
            end = _integer(row["lease_end_ns"], "allocation end")
            if end - start != _duration_ns(self.review["segments"][name]["lease_seconds"], "lease duration"):
                raise AllocationError("Stored lease duration differs from review")
            if row["status"] == "closed":
                if not start <= _integer(row["closed_ns"], "closed allocation end") <= end:
                    raise AllocationError("Invalid closed allocation interval")
            elif row["closed_ns"] is not None:
                raise AllocationError("An unresolved/live allocation cannot claim a closed end")
            if not isinstance(row["guardian"], dict) or set(row["guardian"]) != {"pid", "start_ticks", "ppid", "state"}:
                raise AllocationError("Stored guardian identity is invalid")
            for key in ("pid", "start_ticks", "ppid"):
                _integer(row["guardian"][key], "guardian " + key)
            _integer(row["observation_count"], "observation count")
            _text(row["token"])

    def _now(self, state):
        now = _integer(self.clock_ns(), "host monotonic clock")
        if now < state["last_clock_ns"]:
            # Persist the fault while the transaction lock is still held.
            # Returning the clock forward must not silently reopen the owner.
            for row in state["segments"].values():
                if row["status"] != "closed":
                    row.update(status="unresolved", evidence_error="monotonic_clock_rollback")
            _atomic(self.path, state)
            raise AllocationError("Monotonic clock moved backwards")
        state["last_clock_ns"] = now
        return now

    def _usage(self, state, now):
        elapsed, committed = [], []
        errors = []
        for segment, row in state["segments"].items():
            spec = self.review["segments"][segment]
            if row["status"] != "closed":
                try:
                    current = process_identity(row["guardian"]["pid"])
                    alive = current["start_ticks"] == row["guardian"]["start_ticks"] and current["state"] != "Z"
                except (FileNotFoundError, ProcessLookupError):
                    alive = False
                observation = row["last_observation"]
                stale = (observation is None or now < observation["monotonic_ns"]
                    or now - observation["monotonic_ns"] > _duration_ns(self.review["evidence_max_age_seconds"], "evidence age"))
                if not alive or stale or now > row["lease_end_ns"] or row["status"] == "unresolved":
                    row["status"] = "unresolved"
                    errors.append(segment)
            end = row["closed_ns"] if row["status"] == "closed" else now
            elapsed.append((spec["gpu_uuids"], row["started_ns"], end))
            committed.append((spec["gpu_uuids"], row["started_ns"],
                              row["closed_ns"] if row["status"] == "closed" else row["lease_end_ns"]))
        measured = interval_union_ns(elapsed)
        if measured < state["last_measured_elapsed_ns"]:
            raise AllocationError("Measured allocation elapsed cannot decrease")
        if not errors:
            state["last_measured_elapsed_ns"] = measured
        return {"metric": METRIC, "owner_id": self.owner_id, "campaign_id": self.review["campaign_id"],
                "ledger_id": self.review["ledger_id"], "review_sha256": self.review_sha256,
                "measured_allocation_seconds": None if errors else measured / 1e9,
                "committed_allocation_seconds": interval_union_ns(committed) / 1e9,
                "unresolved_segments": errors, "exact_active_compute_seconds": None,
                "native_admission": False, "probe_kind": self.probe.kind}

    def _owned(self):
        self.ownership.refresh()
        return deepcopy(self.ownership.identities), deepcopy(self.ownership.reaped)

    def _snapshot(self, devices):
        began = _integer(self.clock_ns(), "snapshot begin")
        snapshot = self.probe.snapshot(devices)
        end = _integer(self.clock_ns(), "snapshot end")
        if end < began or end - began > _duration_ns(self.review["evidence_max_age_seconds"], "evidence age"):
            raise AllocationError("GPU context query clock/freshness is invalid")
        if not isinstance(snapshot, dict) or set(snapshot) != set(devices):
            raise AllocationError("Missing GPU context snapshot")
        for data in snapshot.values():
            if set(data) != {"contexts", "device_busy_percent_sample"} or not isinstance(data["contexts"], list):
                raise AllocationError("Invalid GPU context snapshot")
            for context in data["contexts"]:
                if set(context) != {"pid", "start_ticks", "kinds"}:
                    raise AllocationError("Incomplete context process identity")
                _integer(context["pid"], "context PID")
                _integer(context["start_ticks"], "context start ticks")
                if not context["kinds"] or any(k not in ("compute", "graphics") for k in context["kinds"]):
                    raise AllocationError("Unsupported context kind")
        return {"monotonic_ns": began, "snapshot": snapshot}

    def reserve_before_go(self, segment):
        if self.allocations:
            raise AllocationError("Each guardian handle owns one reviewed allocation segment")
        if segment not in self.review["segments"]:
            raise AllocationError("Segment was not reviewed")
        self._owned()
        if self.ownership.pending:
            raise AllocationError("GO reservation requires an empty preexisting child subtree")
        initial = self._snapshot(self.review["gpu_uuids"])
        initial["owned_pid_start_ticks"] = []
        with self._transaction() as state:
            now = self._now(state)
            usage = self._usage(state, now)
            if usage["unresolved_segments"]:
                refusal = "Unknown prior allocation requires explicit reconciliation"
            elif segment in state["segments"]:
                refusal = "Allocation segment cannot be replayed or reset"
            else:
                spec = self.review["segments"][segment]
                end = now + _duration_ns(spec["lease_seconds"], "lease duration")
                cleanup = _duration_ns(spec["cleanup_seconds"], "cleanup duration")
                trial = deepcopy(state)
                trial["segments"][segment] = {"status": "reserved", "started_ns": now, "lease_end_ns": end,
                    "closed_ns": None, "guardian": self.guardian, "token": uuid.uuid4().hex,
                    "workload": None, "owned_identities": [], "last_observation": initial, "observation_count": 1}
                trial_usage = self._usage(trial, now)
                committed = trial_usage["committed_allocation_seconds"]
                if trial_usage["unresolved_segments"]:
                    refusal = "Initial GPU evidence expired while waiting for the reservation lock; GO refused"
                elif committed > self.review["allocation_cap_seconds"]:
                    refusal = "Campaign allocation cap cannot cover this GO reservation"
                else:
                    state["segments"] = trial["segments"]
                    self.allocations.add(segment)
                    return {"segment": segment, "token": trial["segments"][segment]["token"],
                            "begin_cleanup_at_monotonic_ns": end - cleanup, "must_finish_by_monotonic_ns": end}
        raise AllocationError(refusal)

    def _row(self, state, token):
        if not isinstance(token, dict) or set(token) != {"segment", "token", "begin_cleanup_at_monotonic_ns", "must_finish_by_monotonic_ns"}:
            raise AllocationError("Invalid allocation token")
        row = state["segments"].get(token["segment"])
        if (row is None or row["guardian"] != self.guardian or row["token"] != token["token"]
                or token["segment"] not in self.allocations or row["status"] not in ("reserved", "running")):
            raise AllocationError("Allocation is not owned by this live guardian/token")
        spec = self.review["segments"][token["segment"]]
        if (token["must_finish_by_monotonic_ns"] != row["lease_end_ns"]
                or token["begin_cleanup_at_monotonic_ns"] != row["lease_end_ns"] - round(spec["cleanup_seconds"] * 1e9)):
            raise AllocationError("Allocation deadlines changed")
        return row

    def bind_started_workload(self, token, workload):
        if type(workload) is not subprocess.Popen:
            raise AllocationError("Workload must be the actual guardian Popen handle, not a PID list")
        identities, _ = self._owned()
        spec = self.review["segments"][token["segment"]]
        actual = next((r for r in identities if r["pid"] == workload.pid), None)
        if actual is None or argv_sha256(workload.args) != spec["workload_argv_sha256"]:
            raise AllocationError("Workload command or owned child identity differs from review")
        with self._transaction() as state:
            now = self._now(state)
            row = self._row(state, token)
            usage = self._usage(state, now)
            if usage["unresolved_segments"]:
                refusal = "Workload binding evidence is unresolved; allocation remains committed"
            elif row["status"] != "reserved" or now >= token["begin_cleanup_at_monotonic_ns"]:
                refusal = "Allocation cannot begin/replay work at its cleanup boundary"
            else:
                row.update(status="running", workload=actual, owned_identities=identities)
                refusal = None
        # Raise only after the transaction persists any newly unresolved row.
        if refusal:
            raise AllocationError(refusal)

    def observe(self, token):
        try:
            identities, _ = self._owned()
            observation = self._snapshot(self.review["gpu_uuids"])
            owned_pids = {r["pid"] for r in identities}
            assigned = self.review["segments"][token["segment"]]["gpu_uuids"]
            if any(row["pid"] in owned_pids for device, data in observation["snapshot"].items()
                   if device not in assigned for row in data["contexts"]):
                raise AllocationError("Owned workload has a context on an unallocated GPU")
        except BaseException as error:
            with self._transaction() as state:
                row = self._row(state, token)
                row.update(status="unresolved", evidence_error=f"{type(error).__name__}: {error}")
            raise AllocationError("GPU/context evidence failed; allocation remains committed") from error
        with self._transaction() as state:
            now = self._now(state)
            usage = self._usage(state, now)
            if not usage["unresolved_segments"]:
                row = self._row(state, token)
                row["owned_identities"] = identities
                observation["owned_pid_start_ticks"] = [[r["pid"], r["start_ticks"]] for r in identities]
                # Keep the latest evidence, not an unbounded hot-path history.
                # A fresh sample cannot erase an earlier unobserved gap.
                row["last_observation"] = observation
                row["observation_count"] += 1
        if usage["unresolved_segments"]:
            raise AllocationError("Owner/clock allocation interval is unresolved")
        return {**usage, "observation": observation,
                "cleanup_due": now >= token["begin_cleanup_at_monotonic_ns"]}

    def close_after_cleanup(self, token):
        observed = self.observe(token)
        identities, reaped = self._owned()
        if self.ownership.pending or len(reaped) != len(identities):
            raise AllocationError("Owned descendants have not all been reaped")
        owned_pids = {r["pid"] for r in identities}
        contexts = observed["observation"]["snapshot"]
        # NVML gives numeric PIDs, without kernel birth identity. After reaping,
        # a reused PID is ambiguous; it cannot prove that the old context left.
        if any(r["pid"] in owned_pids for d in contexts.values() for r in d["contexts"]):
            raise AllocationError("Owned GPU context is still present; cannot release allocation")
        with self._transaction() as state:
            now = self._now(state)
            row = self._row(state, token)
            # Recheck while holding the final lock. A successful earlier probe
            # must not be allowed to expire while cleanup waits for this lock,
            # then disappear from validation merely by marking the row closed.
            receipt = self._usage(state, now)
            if receipt["unresolved_segments"]:
                failed = True
            else:
                row.update(status="closed", closed_ns=now, cleanup_reaped=reaped)
                failed = False
                receipt = self._usage(state, now)
        if failed:
            raise AllocationError("Cleanup evidence/clock interval is unresolved; allocation remains committed")
        return receipt

    def usage(self):
        with self._transaction() as state:
            result = self._usage(state, self._now(state))
        return result

    def read_total_gpu_seconds(self):
        result = self.usage()
        if result["unresolved_segments"]:
            raise AllocationError("Unknown allocation cannot be reported as measured GPU seconds")
        return result["measured_allocation_seconds"]

    def close_handle(self):
        """Drop only the local lock. Pending allocations are never released."""
        self._lease.close()
