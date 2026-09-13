#!/usr/bin/env python3
"""Resume a versioned official artifact and verify size/MD5 before publication."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bytes", type=int, required=True)
parser.add_argument("--md5", required=True)
parser.add_argument("--seconds", type=int, default=7200)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
partial = args.output.with_suffix(args.output.suffix + ".partial")
started = time.time()
receipt = {"url": args.url, "started": started, "expected_bytes": args.bytes,
           "expected_md5": args.md5, "status": "started"}
record = args.output.with_suffix(args.output.suffix + ".json")
record.write_text(json.dumps(receipt, indent=2))
try:
    path = args.output if args.output.exists() else partial
    if not args.output.exists():
        result = subprocess.run([
            "curl", "--fail", "--location", "--continue-at", "-", "--retry", "8",
            "--retry-delay", "5", "--connect-timeout", "30", "--max-time", str(args.seconds),
            "--speed-limit", "1024", "--speed-time", "120", "--output", str(partial), args.url,
        ], timeout=args.seconds + 60)
        if result.returncode:
            raise RuntimeError(f"curl exit {result.returncode}; partial retained for resume")
    if path.stat().st_size != args.bytes:
        raise ValueError(f"artifact size mismatch: {path.stat().st_size}")
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            md5.update(block)
            sha256.update(block)
    if md5.hexdigest() != args.md5:
        raise ValueError("official MD5 mismatch; refusing extraction")
    if path == partial:
        partial.replace(args.output)
    receipt.update(status="verified", md5=md5.hexdigest(), sha256=sha256.hexdigest(),
                   bytes=args.bytes, elapsed_seconds=time.time() - started)
finally:
    receipt["finished"] = time.time()
    record.write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
