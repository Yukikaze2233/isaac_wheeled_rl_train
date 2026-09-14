"""B1 experiment identity and stable A-periodic capture; no A launcher changes."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import shlex

from common import CODE_DIRECTORY, ControlSSH, file_record
from wheeled_algo.v40_job import JobError, strict_json

A_SHA = "e31ba1538be4f26d5a5c6452ca8b06e1d09f41db999118957909d13f7248e73e"
B_CONTRACT = "contracts/own_v40_round3_b1.json"


def capture_parent(run, checkpoint=None, settle_seconds=1.0):
    """Read a periodic snapshot twice. This function is also sent as stdlib-only SSH code."""
    import hashlib
    import json
    import os
    from pathlib import Path
    import re
    import stat
    import time

    root = Path(run)
    if checkpoint is None:
        candidates = [p for p in root.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", p.name)
                      and time.time() - p.lstat().st_mtime >= 5]
        if not candidates:
            raise ValueError("no periodic checkpoint old enough for stable capture")
        checkpoint = max(candidates, key=lambda p: int(p.stem.split("_")[1])).name
    if not re.fullmatch(r"model_\d+\.pt", checkpoint):
        raise ValueError("pilot source must retain a periodic model_<index>.pt name")
    names = [checkpoint, "run_manifest.json", "contract.json", "source_hashes.json", "asset_manifest.json", "agent_config.json"]

    def read(name):
        fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= 64 * 1024 * 1024:
                raise ValueError("parent member is not a bounded regular nonlinked file: " + name)
            data = stream.read()
            after = os.fstat(stream.fileno())
        def identity(s):
            return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
        if identity(before) != identity(after) or identity(after) != identity((root / name).lstat()):
            raise ValueError("parent member changed while reading: " + name)
        return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "fingerprint": identity(after)}, data

    first = {name: read(name) for name in names}
    time.sleep(settle_seconds)
    second = {name: read(name) for name in names}
    if first != second:
        raise ValueError("parent snapshot is still changing")
    manifest = json.loads(first["run_manifest.json"][1])
    contract = json.loads(first["contract.json"][1])
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (manifest["contract_sha256"] != contract_sha or contract.get("round3", {}).get("stage") != "A"
            or manifest.get("runtime", {}).get("checkpoint_format") != "rsl_rl_5_split_mlp"
            or manifest["runtime"]["versions"]["rsl-rl-lib"] != "5.5.1"):
        raise ValueError("source is not the current A / RSL5 task")
    return {"source_run": str(root), "checkpoint": checkpoint, "source_contract_sha256": contract_sha,
            "source_checkpoint_sha256": first[checkpoint][0]["sha256"],
            "files": {name: item[0] for name, item in first.items()}, "completion_required": False}


def remote_capture(client, run, checkpoint=None):
    script = inspect.getsource(capture_parent) + "\nimport json\nprint(json.dumps(capture_parent(" + repr(run) + "," + repr(checkpoint) + ")))"
    record = strict_json(client.exec(shlex.join(["python3", "-c", script])))
    if record["source_contract_sha256"] != A_SHA:
        raise JobError("parent is not the frozen A contract")
    return record


def verify_experiment(root):
    root = Path(root).absolute()
    manifest = strict_json((root / "pipeline.json").read_bytes())
    if manifest.get("classification") != "experiment_not_promoted" or manifest["parent"]["source_contract_sha256"] != A_SHA:
        raise JobError("invalid B1 experiment identity")
    for name, expected in manifest["files"].items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
            raise JobError("experiment member escapes root")
        if file_record(path) != expected:
            raise JobError("experiment content changed: " + name)
    return manifest


def verify_transfer(record, plan):
    lineage = record.get("source_provenance", {})
    expected = {"mode": "stage_transfer", "source_stage": "A", "target_stage": "B1",
                "parent_checkpoint_sha256": plan["source_checkpoint_sha256"],
                "parent_contract_sha256": A_SHA, "target_contract_sha256": plan["contract_sha256"],
                "source_filename": plan["source_filename"], "optimizer_reset": True,
                "std_preserved": True, "initial_iteration": 0}
    if any(lineage.get(key) != value or type(lineage.get(key)) is not type(value) for key, value in expected.items()):
        raise JobError("B1 stage-transfer lineage differs from this experiment")
