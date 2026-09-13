#!/usr/bin/env python3
"""Validate every extracted ZIP member by size and CRC before publishing a new native installation."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import zlib

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--publish", type=Path, required=True)
parser.add_argument("--receipt", type=Path, required=True)
args = parser.parse_args()
started = time.time()
if args.publish.exists():
    raise FileExistsError(args.publish)
count = total = 0
with zipfile.ZipFile(args.archive) as archive:
    for item in archive.infolist():
        path = args.directory / item.filename
        if not path.resolve().is_relative_to(args.directory.resolve()):
            raise ValueError("ZIP member escapes installation root")
        if item.is_dir():
            assert path.is_dir(), path
            continue
        assert path.stat().st_size == item.file_size, path
        crc = 0
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                crc = zlib.crc32(block, crc)
        assert crc == item.CRC, path
        count += 1
        total += item.file_size
        if count % 10000 == 0:
            print(f"Verified {count} extracted files", flush=True)
args.directory.rename(args.publish)
record = {"status": "verified_and_published", "file_count": count, "bytes": total,
          "elapsed_seconds": time.time() - started, "destination": str(args.publish),
          "verification": "every ZIP file size and CRC; archive official MD5 verified separately"}
args.receipt.write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
