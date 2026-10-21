"""Publish immutable recovery units without pausing the training process."""

import os
from wheeled_tasks.chassis.application.monitoring.training_metrics import timed_call
import json
from pathlib import Path
import shutil


def seal_checkpoint(run_directory, update, save, contract_path, progress, *, budget_owner=None):
    root = Path(run_directory) / "checkpoints"
    root.mkdir(exist_ok=True)
    destination = root / f"update_{update:08d}"
    temporary = root / (destination.name + ".pending")
    if destination.exists():
        raise FileExistsError(f"Checkpoint already sealed: {destination}")
    temporary.mkdir(exist_ok=False)
    if budget_owner is None:
        save(str(temporary / "model.pt"))
    else:
        # Bind the receipt to its immutable published path. The files remain
        # private until the complete recovery directory is atomically renamed.
        receipt = save(str(temporary / "model.pt"),
                       budget_publication_path=destination / "model.pt")
    shutil.copyfile(contract_path, temporary / "contract.json")
    (temporary / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    (temporary / "completion.json").write_text(json.dumps({
        "status": "checkpoint_sealed", "successful_updates": update,
        "model_role": "candidate_requires_evaluation",
        "resume": "model and optimizer restored; physical episodes restart"}, indent=2) + "\n")
    temporary.rename(destination)
    if budget_owner is not None:
        budget_owner.register_checkpoint(destination / "model.pt", receipt)
    return destination


def write_progress(directory, progress, timer=None):
    def write():
        temporary = directory / "progress.tmp"
        with temporary.open("w") as stream:
            json.dump(progress, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(directory / "progress.json")
    timed_call(timer, "progress_io", write)
