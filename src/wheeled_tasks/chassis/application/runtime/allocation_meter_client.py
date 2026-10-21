"""Private CPU-proven guardian RPC protocol; not native admission.

Only an inherited socket handshake can create a client. Allocation elapsed
seconds are not active GPU compute seconds. Public native execution remains
blocked; the current authenticated scope is explicitly a CPU fixture.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

from wheeled_tasks.chassis.application.runtime.gpu_allocation_owner import METRIC, CPUFixtureProbe, host_identity, process_identity


JOB_SCHEMA = "native_allocation_guardian_job_v1"
SCOPE = "test_only_owned_allocation_rpc_v1"
ENV_FD = "V7_ALLOCATION_RPC_FD"
ENV_JOB = "V7_ALLOCATION_JOB_PATH"
ENV_SHA = "V7_ALLOCATION_JOB_SHA256"
MAX_PACKET = 16384


class MeterProtocolError(RuntimeError):
    pass


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_bound(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise MeterProtocolError("Bound protocol bytes changed")
    return json.loads(raw)


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise MeterProtocolError(f"Explicit finite positive {label} required")
    return value


def load_bound_job(path, expected_sha256):
    """Read each bound document once; all current jobs are CPU fixtures only."""
    job = _read_bound(path, expected_sha256)
    required = {"schema", "scope", "allocation_review", "segment", "guardian_source_sha256",
                "client_source_sha256", "worker_source", "argv", "cwd", "env", "stdout", "stderr",
                "receipt", "term_grace_s", "kill_wait_s", "poll_interval_s", "handshake_timeout_s",
                "cpu_probe_mode"}
    if not isinstance(job, dict) or set(job) != required or job["schema"] != JOB_SCHEMA or job["scope"] != SCOPE:
        raise MeterProtocolError("Only an explicit CPU fixture job is implemented; native admission unavailable")
    if job["client_source_sha256"] != _digest(__file__):
        raise MeterProtocolError("RPC client source changed")
    guardian_source = Path(__file__).resolve().parents[5] / "scripts/native_allocation_guardian.py"
    if job["guardian_source_sha256"] != _digest(guardian_source):
        raise MeterProtocolError("Allocation guardian source changed")
    for name in ("allocation_review", "worker_source"):
        if not isinstance(job[name], dict) or set(job[name]) != {"path", "sha256"}:
            raise MeterProtocolError("Missing bound source/review")
        p = Path(job[name]["path"])
        if not p.is_absolute() or p.resolve() != p:
            raise MeterProtocolError("Bound paths must be canonical absolute paths")
    review = _read_bound(job["allocation_review"]["path"], job["allocation_review"]["sha256"])
    if (review.get("metric") != METRIC or review.get("probe_kind") != CPUFixtureProbe.kind
            or review.get("host") != host_identity() or job["segment"] not in review.get("segments", {})):
        raise MeterProtocolError("Allocation review scope/host/segment is incompatible")
    if _digest(job["worker_source"]["path"]) != job["worker_source"]["sha256"]:
        raise MeterProtocolError("Worker source changed before handshake")
    argv = job["argv"]
    if (not isinstance(argv, list) or len(argv) < 4 or argv[1:3] != ["-I", "-B"]
            or argv[3] != job["worker_source"]["path"]
            or not all(isinstance(v, str) and "\0" not in v for v in argv)):
        raise MeterProtocolError("Worker argv must bind its isolated Python source entry")
    argv_hash = hashlib.sha256(json.dumps(argv, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    spec = review["segments"][job["segment"]]
    if spec["training_source_sha256"] != job["worker_source"]["sha256"]:
        raise MeterProtocolError("Allocation segment worker source differs from actual bound worker")
    if argv_hash != spec["workload_argv_sha256"]:
        raise MeterProtocolError("Reviewed workload argv changed")
    for name in ("cwd", "stdout", "stderr", "receipt"):
        p = Path(job[name])
        if not p.is_absolute() or p.resolve() != p:
            raise MeterProtocolError("Job paths must be canonical absolute paths")
    if (not isinstance(job["env"], dict) or not all(isinstance(k, str) and isinstance(v, str)
            for k, v in job["env"].items()) or job["env"].get("CUDA_VISIBLE_DEVICES") != ""
            or {ENV_FD, ENV_JOB, ENV_SHA} & set(job["env"])):
        raise MeterProtocolError("CPU environment must hide CUDA and cannot supply guardian protocol descriptors")
    for name in ("term_grace_s", "kill_wait_s", "poll_interval_s", "handshake_timeout_s"):
        _positive(job[name], name)
    if (job["term_grace_s"] + job["kill_wait_s"] >= spec["cleanup_seconds"]
            or job["poll_interval_s"] >= review["evidence_max_age_seconds"]
            or job["handshake_timeout_s"] >= spec["lease_seconds"] - spec["cleanup_seconds"]):
        raise MeterProtocolError("Reviewed cleanup/evidence/handshake timing cannot cover this job")
    if job["cpu_probe_mode"] not in ("empty", "initial_error", "error_after_go", "residual_owned_context"):
        raise MeterProtocolError("Unknown explicit CPU probe fixture")
    return job, review


def send_packet(sock, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_PACKET or sock.send(raw) != len(raw):
        raise MeterProtocolError("RPC packet overflow or incomplete send")


def receive_packet(sock):
    raw, _ancillary, flags, _address = sock.recvmsg(MAX_PACKET)
    if not raw or flags & socket.MSG_TRUNC:
        raise MeterProtocolError("Guardian RPC disconnected or packet truncated")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise MeterProtocolError("RPC packet must be an object")
    return value


def identity_receipt(job, review, job_sha256, guardian, worker):
    """Protocol identity builder, not a client constructor or admission token."""
    return {"metric": METRIC, "scope": SCOPE, "native_admission": False,
        "owner_id": review["owner_id"], "campaign_id": review["campaign_id"],
        "allocation_ledger_id": review["ledger_id"],
        "allocation_review_sha256": job["allocation_review"]["sha256"],
        "allocation_cap_seconds": review["allocation_cap_seconds"], "probe_kind": review["probe_kind"],
        "job_sha256": job_sha256, "source_sha256": job["client_source_sha256"],
        "allocation_source_sha256": review["source_sha256"],
        "allocation_guardian_source_sha256": job["guardian_source_sha256"],
        "guardian_pid_start_ticks": [guardian["pid"], guardian["start_ticks"]],
        "worker_pid_start_ticks": [worker["pid"], worker["start_ticks"]], "host_boot": review["host"]}


class AllocationMeterClient:
    """An authenticated live socket owned by this exact worker PID/birth."""
    def __init__(self, *args, **kwargs):
        raise MeterProtocolError("Construct only through await_worker_go; dictionaries do not confer ownership")

    @property
    def owner_id(self):
        return self._identity["owner_id"]

    def identity_receipt(self):
        return deepcopy(self._identity)

    def _check_process_and_deadline(self):
        current = process_identity(os.getpid())
        if [current["pid"], current["start_ticks"]] != self._identity["worker_pid_start_ticks"]:
            raise MeterProtocolError("RPC client cannot be inherited by another process")
        guardian = process_identity(self._identity["guardian_pid_start_ticks"][0])
        if ([guardian["pid"], guardian["start_ticks"]] != self._identity["guardian_pid_start_ticks"]
                or guardian["state"] in ("Z", "X") or host_identity() != self._identity["host_boot"]):
            raise MeterProtocolError("Authenticated guardian/host was lost")
        now = time.monotonic_ns()
        if now < self._last_now or now >= self._cleanup_ns:
            raise MeterProtocolError("Worker allocation clock/deadline prohibits another operation")
        self._last_now = now
        self._socket.settimeout(min(self._max_age_s, (self._cleanup_ns - now) / 1e9))
        return now

    def read_total_gpu_seconds(self):
        self._check_process_and_deadline()
        self._seq += 1
        try:
            send_packet(self._socket, {"type": "measure", "seq": self._seq, "nonce": self._nonce})
            row = receive_packet(self._socket)
            now = self._check_process_and_deadline()
            if (set(row) != {"type", "seq", "nonce", "identity", "measured_allocation_seconds", "monotonic_ns"}
                    or row["type"] != "measurement" or type(row["seq"]) is not int or row["seq"] != self._seq
                    or row["nonce"] != self._nonce or row["identity"] != self._identity
                    or type(row["monotonic_ns"]) is not int or row["monotonic_ns"] > now
                    or now - row["monotonic_ns"] > round(self._max_age_s * 1e9)):
                raise MeterProtocolError("RPC response identity/sequence/freshness differs")
            seconds = row["measured_allocation_seconds"]
            if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                    or seconds < self._last_seconds or seconds >= self._identity["allocation_cap_seconds"]):
                raise MeterProtocolError("Allocation measurement regressed or reached its cap")
            self._last_seconds = float(seconds)
            return self._last_seconds
        except BaseException:
            self._socket.close()
            raise

    def check_alive_and_deadline(self):
        return self.read_total_gpu_seconds()

    def close(self):
        """Close this RPC channel only; never settle or release allocation."""
        if self._socket.fileno() < 0:
            return
        try:
            self._seq += 1
            send_packet(self._socket, {"type": "client_closed", "seq": self._seq, "nonce": self._nonce})
        except (OSError, MeterProtocolError):
            pass
        self._socket.close()


def await_worker_go(*, fd, job_path, job_sha256):
    """First worker action, before Kit/CUDA: validate and wait for private GO.

    Resource ownership is not native admission. Current jobs/clients are CPU
    fixtures; native entry must continue to reject them independently.
    """
    if type(fd) is not int or fd < 0:
        raise MeterProtocolError("An explicit inherited socket descriptor is required")
    job, review = load_bound_job(job_path, job_sha256)
    sock = socket.socket(fileno=fd)
    try:
        if sock.family != socket.AF_UNIX or sock.type != socket.SOCK_SEQPACKET:
            raise MeterProtocolError("Expected private Unix sequenced-packet socket")
        os.set_inheritable(fd, False)
        sock.settimeout(job["handshake_timeout_s"])
        worker = process_identity(os.getpid())
        send_packet(sock, {"type": "hello", "job_sha256": job_sha256,
                           "pid": worker["pid"], "start_ticks": worker["start_ticks"]})
        go = receive_packet(sock)
        guardian = process_identity(os.getppid())
        expected = identity_receipt(job, review, job_sha256, guardian, worker)
        now = time.monotonic_ns()
        if (set(go) != {"type", "identity", "nonce", "cleanup_ns", "must_finish_ns", "monotonic_ns"}
                or go["type"] != "go" or go["identity"] != expected
                or not isinstance(go["nonce"], str) or len(go["nonce"]) != 64
                or any(c not in "0123456789abcdef" for c in go["nonce"])
                or type(go["cleanup_ns"]) is not int or type(go["must_finish_ns"]) is not int
                or not now < go["cleanup_ns"] < go["must_finish_ns"]
                or go["must_finish_ns"] - go["cleanup_ns"] != round(review["segments"][job["segment"]]["cleanup_seconds"] * 1e9)
                or type(go["monotonic_ns"]) is not int or not go["monotonic_ns"] <= now
                or now - go["monotonic_ns"] > round(review["evidence_max_age_seconds"] * 1e9)):
            raise MeterProtocolError("Worker GO identity/clock is invalid")
        obj = object.__new__(AllocationMeterClient)
        obj._socket, obj._identity, obj._nonce = sock, expected, go["nonce"]
        obj._cleanup_ns, obj._max_age_s = go["cleanup_ns"], review["evidence_max_age_seconds"]
        obj._last_now, obj._last_seconds, obj._seq = now, 0., 0
        return obj
    except BaseException:
        sock.close()
        raise
