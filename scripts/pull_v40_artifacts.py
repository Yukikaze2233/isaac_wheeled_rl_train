#!/usr/bin/env python3
"""Explicit, read-only remote V40 artifact pull through DSH's managed SSH HTTP API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.v40_job import (ALLOWED_ARTIFACTS, JobError, MAX_JSON_BYTES,
    json_bytes, positive_seconds, strict_json, validate_completion)

# Read-only audited API: @linxin666/dsh-ssh/lib/index.js lines 2520..2535 / 2755..2787.
# POST /exec: {alias,command,timeoutMs} -> {result:{success,exitCode,timedOut,stdout,stderr}}.
# GET /download?alias=...&remotePath=... -> binary stream, NOT a localPath JSON request.
EXEC_ROUTE = "/api/dsh-ssh/exec"
DOWNLOAD_ROUTE = "/api/dsh-ssh/download"


class Unavailable(JobError):
    pass


class NotReady(JobError):
    pass


def management_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise JobError("management URL must be a loopback HTTP(S) origin without credentials/path/query")
    host = parsed.hostname
    if host == "localhost":
        host = "127.0.0.1"  # Never trust DNS/proxy configuration for a privileged local API.
    try:
        address = ipaddress.ip_address(host or "")
        port = parsed.port
    except ValueError as exc:
        raise JobError("management URL requires a literal loopback address or localhost") from exc
    if not address.is_loopback:
        raise JobError("management URL is not loopback")
    netloc = f"[{host}]" if address.version == 6 else host
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise JobError("management HTTP redirects are forbidden")


class ManagedSSH:
    def __init__(self, alias, management="http://127.0.0.1:3080", *, timeout=90):
        if not isinstance(alias, str) or not alias.strip() or any(ord(c) < 32 for c in alias):
            raise JobError("an explicit nonempty alias is required")
        self.alias, self.origin = alias, management_url(management)
        self.timeout = positive_seconds(timeout)
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def open(self, route, payload=None):
        data = None if payload is None else json_bytes(payload)
        request = Request(self.origin + route, data=data,
                          headers={} if data is None else {"Content-Type": "application/json"})
        try:
            return self.opener.open(request, timeout=self.timeout)
        except (HTTPError, URLError, OSError) as exc:
            raise Unavailable(f"managed SSH unavailable: {exc}; platform may be off, wait for power-on; no cloud-file fallback") from exc

    def exec(self, command, *, timeout_ms=60000):
        with self.open(EXEC_ROUTE, {"alias": self.alias, "command": command, "timeoutMs": timeout_ms}) as response:
            raw = response.read(MAX_JSON_BYTES + 1)
        envelope = strict_json(raw)
        result = envelope.get("result") if isinstance(envelope, dict) else None
        if not isinstance(result, dict) or type(result.get("stdout")) is not str:
            raise JobError("invalid managed exec response")
        if result.get("success") is not True or result.get("exitCode") != 0 or result.get("timedOut") is not False:
            exit_code = result.get("exitCode")
            detail = result.get("error") or result.get("stderr") or exit_code
            if type(exit_code) is int and exit_code >= 0 and result.get("timedOut") is False:
                raise JobError(f"remote command failed (exit code {exit_code}): {detail}")
            raise Unavailable(f"managed SSH exec unavailable: {detail}; wait for power-on if platform is off")
        return result["stdout"]

    def download(self, remote_path):
        return self.open(DOWNLOAD_ROUTE + "?" + urlencode({"alias": self.alias, "remotePath": remote_path}))


def absolute_remote_path(value):
    if (not isinstance(value, str) or not value.startswith("/") or value == "/"
            or "\\" in value or any(ord(c) < 32 for c in value)
            or any(part in {"", ".", ".."} for part in value.split("/")[1:])):
        raise JobError("remote path must be explicit normalized absolute POSIX path, without traversal")
    return value


# This command ONLY reads. Opening every ancestor/member with O_NOFOLLOW rejects
# symlinks; regular-file checks reject FIFOs/devices/directories/hardlinks. The
# managed SFTP API has no nofollow transaction: before/after inode/ctime checks
# plus downloaded SHA256 detect changes, but do not provide a malicious-server sandbox.
REMOTE_PROBE = r'''
import json, os, stat
p = PARAMS
fd = None
receipt_seen = False
def fingerprint(s):
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
def open_file(name):
    f = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    s = os.fstat(f)
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        os.close(f)
        raise ValueError("nonregular/linked file: " + name)
    return f, s
try:
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    for part in p["root"].split("/")[1:]:
        nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        os.close(fd)
        fd = nxt
    f, s = open_file("completion.json")
    receipt_seen = True
    with os.fdopen(f, "rb") as stream:
        raw = stream.read(p["json_limit"] + 1)
        if len(raw) > p["json_limit"] or fingerprint(s) != fingerprint(os.fstat(stream.fileno())):
            raise ValueError("oversize/changing completion")
    receipt = json.loads(raw)
    fingerprints = {"completion.json": fingerprint(s)}
    for item in receipt["artifacts"]:
        name = item["path"]
        if name not in p["allowed"] or name in fingerprints:
            raise ValueError("unknown/escaping/duplicate artifact")
        f, s = open_file(name)
        os.close(f)
        if s.st_size != item["size"]:
            raise ValueError("artifact size differs from completion")
        fingerprints[name] = fingerprint(s)
    print(json.dumps({"state": "ready", "completion": raw.decode("utf-8"), "fingerprints": fingerprints}))
except FileNotFoundError:
    print(json.dumps({"state": "rejected" if receipt_seen else "pending",
                      "detail": "completed member missing" if receipt_seen else "run/completion not yet present"}))
except Exception as e:
    print(json.dumps({"state": "rejected", "detail": str(e)}))
finally:
    if fd is not None:
        os.close(fd)
'''


def probe_command(remote_run_dir, remote_python="python3"):
    if remote_python != "python3":
        absolute_remote_path(remote_python)
    params = {"root": absolute_remote_path(remote_run_dir), "allowed": sorted(ALLOWED_ARTIFACTS),
              "json_limit": MAX_JSON_BYTES}
    script = REMOTE_PROBE.replace("PARAMS", repr(params), 1)
    # Only a stdlib read-only probe; using the training ENV does not start training.
    return shlex.join([remote_python, "-c", script])


def probe(client, remote_run_dir, remote_python="python3"):
    response = strict_json(client.exec(probe_command(remote_run_dir, remote_python)))
    if not isinstance(response, dict):
        raise JobError("invalid probe response")
    if response.get("state") == "pending":
        raise NotReady("completion not ready; no transfer performed")
    if response.get("state") != "ready":
        raise JobError(f"remote artifact confinement rejected: {response.get('detail')}")
    raw = response.get("completion")
    if type(raw) is not str:
        raise JobError("invalid completion body")
    record = validate_completion(strict_json(raw))
    if record["status"] not in {"completed", "stopped", "export_failed"}:
        raise JobError(f"no complete model to transfer: {record['status']}; {record.get('error')}")
    names = {item["path"] for item in record["artifacts"]} | {"completion.json"}
    fingerprints = response.get("fingerprints")
    if (type(fingerprints) is not dict or set(fingerprints) != names
            or any(type(v) is not list or len(v) != 5 or any(type(n) is not int for n in v)
                   for v in fingerprints.values())):
        raise JobError("invalid remote file fingerprints")
    return raw.encode("utf-8"), record, fingerprints


def open_local_parent(destination):
    path = Path(os.path.abspath(destination))
    if ".." in Path(destination).parts or path == Path("/"):
        raise JobError("destination traversal/root is forbidden")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parent.parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        try:
            os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return path, fd
        raise JobError("destination already exists; refusing to overwrite any existing/unknown data")
    except BaseException:
        os.close(fd)
        raise


def publish_partial(fd, name, blocks, expected):
    partial = name + ".partial"
    handle = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    digest, size = hashlib.sha256(), 0
    with os.fdopen(handle, "wb") as stream:
        for block in blocks:
            size += len(block)
            if size > expected["size"]:
                raise JobError(f"download exceeded declared size: {name}")
            digest.update(block)
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise JobError(f"size/SHA256 verification failed: {name}; partial retained, not published")
    os.link(partial, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
    os.unlink(partial, dir_fd=fd)
    os.fsync(fd)


def pull_artifacts(client, remote_run_dir, destination, *, remote_python="python3", wait=False, poll_interval=30,
                   max_wait_seconds=3600, monotonic=time.monotonic, sleep=time.sleep):
    remote_run_dir = absolute_remote_path(remote_run_dir)
    interval = positive_seconds(poll_interval)
    if interval < 30:
        raise JobError("poll interval must be at least 30 seconds")
    deadline = monotonic() + positive_seconds(max_wait_seconds)
    _, parent = open_local_parent(destination)  # Reject unsafe existing targets before network calls.
    os.close(parent)
    while True:
        try:
            remaining = deadline - monotonic()
            if wait and remaining <= 0:
                raise NotReady("maximum wait budget exhausted")
            previous_timeout = getattr(client, "timeout", None)
            try:
                if wait and previous_timeout is not None:
                    client.timeout = min(previous_timeout, remaining)
                raw, record, fingerprints = probe(client, remote_run_dir, remote_python)
            finally:
                if previous_timeout is not None:
                    client.timeout = previous_timeout
            if wait and monotonic() >= deadline:
                raise NotReady("maximum wait budget exhausted during readiness check")
            break
        except (NotReady, Unavailable) as exc:
            if not wait:
                raise
            remaining = deadline - monotonic()
            if remaining < interval:
                raise NotReady(f"maximum wait budget exhausted: {exc}; not transferred, wait for power-on if needed") from exc
            print(str(exc), file=sys.stderr)
            sleep(interval)
            if monotonic() >= deadline:
                raise NotReady("maximum wait budget exhausted; not transferred")
    path, parent = open_local_parent(destination)
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=parent)
        fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        for item in record["artifacts"]:
            # Recheck the complete set before each SFTP stream; never download a supplied arbitrary path.
            if probe(client, remote_run_dir, remote_python) != (raw, record, fingerprints):
                raise JobError("remote completion/artifacts changed before transfer")
            with client.download(str(PurePosixPath(remote_run_dir) / item["path"])) as response:
                blocks = iter(lambda: response.read(1024 * 1024), b"")
                publish_partial(fd, item["path"], blocks, item)
        if probe(client, remote_run_dir, remote_python) != (raw, record, fingerprints):
            raise JobError("remote completion/artifacts changed during transfer")
        completion_hash = hashlib.sha256(raw).hexdigest()
        publish_partial(fd, "completion.json", [raw], {"size": len(raw), "sha256": completion_hash})
        local = {"schema_version": 1, "transfer_verified": True, "alias": client.alias,
                 "remote_run_dir": remote_run_dir, "completion_sha256": completion_hash,
                 "artifacts": record["artifacts"], "run_status": record["status"],
                 "export_status": record["export_status"], "policy_quality_verified": False,
                 "received_at": datetime.now(timezone.utc).isoformat()}
        local_raw = json_bytes(local)
        publish_partial(fd, "local_receipt.json", [local_raw],
                        {"size": len(local_raw), "sha256": hashlib.sha256(local_raw).hexdigest()})
        return local
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default="python3",
                        help="Stdlib probe interpreter: python3 (default) or a normalized absolute POSIX path")
    parser.add_argument("--destination", required=True, type=Path, help="NEW directory; parent must exist and have no symlinks")
    parser.add_argument("--management-url", default="http://127.0.0.1:3080", type=management_url)
    parser.add_argument("--wait", action="store_true", help="Poll only pending/unavailable runs; never retry invalid completion")
    parser.add_argument("--poll-interval", type=positive_seconds, default=30)
    parser.add_argument("--max-wait-seconds", type=positive_seconds, default=3600)
    args = parser.parse_args(argv)
    try:
        receipt = pull_artifacts(ManagedSSH(args.alias, args.management_url), args.remote_run_dir, args.destination,
            remote_python=args.remote_python, wait=args.wait, poll_interval=args.poll_interval,
            max_wait_seconds=args.max_wait_seconds)
    except (JobError, OSError) as exc:
        print(f"NOT transferred: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
