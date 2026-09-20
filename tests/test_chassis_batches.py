"""Batch sealing and recovery must be immutable and reject unsafe archives."""
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from chassis_batch_export import digest, export_ready
from sync_chassis_batches import recover_batch


def test_completed_block_is_recoverable_before_the_run_finishes(tmp_path):
    run = tmp_path / "remote"
    block = run / "train/stage_00_stand/block_000"
    block.mkdir(parents=True)
    (block / "model_final.pt").write_bytes(b"checkpoint with optimizer state")
    (block / "contract.json").write_text("{}")
    (block / "completion.json").write_text('{"status":"completed"}')
    index = export_ready(run)
    assert not index["run_finished"]
    assert len(index["batches"]) == 1
    receipt = index["batches"][0]
    output = tmp_path / "local/artifacts"
    output.parent.mkdir()
    recover_batch(run / receipt["archive"], receipt, output)
    assert (output / "train/stage_00_stand/block_000/model_final.pt").read_bytes() == b"checkpoint with optimizer state"
    assert export_ready(run)["batches"] == index["batches"]


def test_recovery_rejects_escaping_paths(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        item = tarfile.TarInfo("../escape")
        item.size = 0
        stream.addfile(item)
    receipt = {"batch_id": "bad", "archive_sha256": digest(archive),
               "files": {"../escape": {"size": 0, "sha256": "invalid"}}}
    with pytest.raises(ValueError, match="Invalid batch"):
        recover_batch(archive, receipt, tmp_path / "artifacts")
    assert not (tmp_path / "escape").exists()
