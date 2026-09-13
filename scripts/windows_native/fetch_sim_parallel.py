#!/usr/bin/env python3
"""Resumable, bounded HTTP ranges for the large official Windows ZIP.

Completed 128 MiB chunks are reusable. The final official checksum remains the
authority; range sizes or successful requests alone do not approve extraction.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

directory = Path(sys.argv[1])
size = 10668464877
expected_md5 = "4b49a4258792f09300ece31be1b6cfd9"
name = "isaac-sim-standalone-6.0.0-windows-x86_64.zip"
url = "https://downloads.isaacsim.nvidia.com/" + name
final = directory / name
parts = directory / "sim-zip-parts"
parts.mkdir(exist_ok=True)
chunk_size = 128 * 1024 * 1024
chunks = [(index, start, min(start + chunk_size, size) - 1)
          for index, start in enumerate(range(0, size, chunk_size))]
started = time.time()
seed = directory / (name + ".partial")
if seed.exists():
    # Reuse the completed prefix of the previous single-connection attempt.
    prefix = seed.stat().st_size
    with seed.open("rb") as stream:
        for index, start, end in chunks:
            if end >= prefix:
                break
            target = parts / f"{index:04d}.part"
            if not target.exists():
                stream.seek(start)
                target.write_bytes(stream.read(end - start + 1))


def fetch(chunk):
    index, start, end = chunk
    target = parts / f"{index:04d}.part"
    if target.exists() and target.stat().st_size == end - start + 1:
        return index, "reused"
    temporary = target.with_suffix(".downloading")
    result = subprocess.run([
        "curl", "--silent", "--show-error", "--fail", "--location", "--retry", "4", "--retry-delay", "3",
        "--connect-timeout", "20", "--max-time", "600", "--max-filesize", str(end - start + 1),
        "--range", f"{start}-{end}", "--write-out", "%{http_code}", "--output", str(temporary), url,
    ], capture_output=True, text=True, timeout=700)
    if result.returncode or result.stdout != "206" or temporary.stat().st_size != end - start + 1:
        raise RuntimeError(f"chunk {index}: curl={result.returncode}, status={result.stdout}, {result.stderr}")
    temporary.replace(target)
    return index, "downloaded"


with ThreadPoolExecutor(max_workers=6) as executor:
    for future in as_completed([executor.submit(fetch, chunk) for chunk in chunks]):
        print(future.result(), flush=True)
assembled = directory / (name + ".assembled")
md5, sha256 = hashlib.md5(), hashlib.sha256()
with assembled.open("wb") as output:
    for index, start, end in chunks:
        with (parts / f"{index:04d}.part").open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                output.write(block)
                md5.update(block)
                sha256.update(block)
if assembled.stat().st_size != size or md5.hexdigest() != expected_md5:
    raise ValueError("assembled official archive checksum mismatch; refusing extraction")
if final.exists():
    raise FileExistsError("refusing to overwrite a published archive")
assembled.replace(final)
record = {"url": url, "bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest(),
          "status": "verified", "range_count": len(chunks), "elapsed_seconds": time.time() - started}
final.with_suffix(final.suffix + ".json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
