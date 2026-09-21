"""Publish immutable recovery units without pausing the training process."""
import json
from pathlib import Path
import shutil


def seal_checkpoint(run_directory, update, save, contract_path, progress):
    root = Path(run_directory) / "checkpoints"
    root.mkdir(exist_ok=True)
    destination = root / f"update_{update:08d}"
    temporary = root / (destination.name + ".pending")
    if destination.exists():
        raise FileExistsError(f"Checkpoint already sealed: {destination}")
    temporary.mkdir(exist_ok=False)
    save(str(temporary / "model.pt"))
    shutil.copyfile(contract_path, temporary / "contract.json")
    (temporary / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    (temporary / "completion.json").write_text(json.dumps({
        "status": "checkpoint_sealed", "successful_updates": update,
        "model_role": "candidate_requires_evaluation",
        "resume": "model and optimizer restored; physical episodes restart"}, indent=2) + "\n")
    temporary.rename(destination)
    return destination
