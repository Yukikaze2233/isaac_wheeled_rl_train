#!/usr/bin/env python3
"""Publish immutable, individually verifiable completed training/evaluation batches."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import tarfile


SUFFIXES = {".json", ".jsonl", ".pt", ".onnx", ".py", ".log", ".npz", ".csv"}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def writer_alive(directory):
    for path in (directory / "progress.json", directory.parent / "progress.json"):
        if not path.exists():
            continue
        try:
            progress = json.loads(path.read_text())
        except json.JSONDecodeError:
            return True
        for key in ("pid", "worker_pid"):
            pid = progress.get(key)
            if not pid:
                continue
            command = Path("/proc") / str(pid) / "cmdline"
            try:
                arguments = command.read_bytes().split(b"\0")
            except FileNotFoundError:
                continue
            if str(directory).encode() in arguments:
                return True
    return False


def export_ready(root):
    root = root.resolve(strict=True)
    train = root / "train"
    output = root / "batch_delivery"
    output.mkdir(exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        units = []
        for marker in train.glob("stage_*/block_*/completion.json"):
            units.append((marker.parent, marker, "training_block", True))
        for marker in train.glob("stage_*/*/evaluation.json"):
            units.append((marker.parent, marker, "evaluation", True))
        for marker in train.glob("stage_*/completion.json"):
            units.append((marker.parent, marker, "stage_summary", False))
        if (train / "completion.json").exists():
            units.append((train, train / "completion.json", "run_summary", False))
        for directory, marker, kind, recursive in units:
            unit = str(directory.relative_to(root))
            key = hashlib.sha256((kind + ":" + unit).encode()).hexdigest()[:20]
            receipt = output / (key + ".json")
            if receipt.exists() or writer_alive(directory):
                continue
            result = json.loads(marker.read_text())
            candidates = directory.rglob("*") if recursive else directory.iterdir()
            selected = [p for p in candidates if p.is_file() and not p.is_symlink()
                        and (p.suffix in SUFFIXES or p.name.startswith("events.out.tfevents."))]
            log = directory.with_suffix(".log")
            if recursive and log.is_file():
                selected.append(log)
            if not recursive:
                selected = [p for p in selected if not p.name.startswith(("block_", "evaluation_", "baseline_evaluation", "baseline_confirmation"))]
            files = {str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": digest(p)} for p in selected}
            archive = output / (key + ".tar.gz")
            temporary = output / (key + ".tar.gz.tmp")
            with tarfile.open(temporary, "w:gz", compresslevel=1) as bundle:
                for path in selected:
                    bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
            # A writer must not have modified the sealed data during packing.
            if any(digest(root / name) != meta["sha256"] for name, meta in files.items()):
                raise RuntimeError(f"Batch changed while packaging: {unit}")
            temporary.replace(archive)
            write_json(receipt, {"batch_id": key, "unit": unit, "kind": kind,
                "status": result.get("status", "evaluated"), "archive": str(archive.relative_to(root)),
                "archive_bytes": archive.stat().st_size, "archive_sha256": digest(archive),
                "files": files, "created_at": datetime.now(timezone.utc).isoformat(),
                "model_role": "candidate_requires_evaluation" if kind == "training_block" else "evaluation_or_selection_evidence"})
        batches = [json.loads(p.read_text()) for p in sorted(output.glob("*.json")) if p.name != "index.json"]
        finished = (train / "completion.json").exists() and not writer_alive(train)
        finished &= all(not writer_alive(p.parent) for p in train.glob("stage_*/*/progress.json"))
        index = {"remote_root": str(root), "run_finished": finished, "batches": batches}
        write_json(output / "index.json", index)
        return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    print(json.dumps(export_ready(parser.parse_args().run_root)))
