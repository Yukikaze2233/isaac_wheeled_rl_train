"""Audited Round3-A -> B1 material-only transfer and B1 resume identity."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import stat

from wheeled_algo.v40_export import (
    _read_checkpoint, _validate_state, load_actor_checkpoint, validate_manifest,
)
from wheeled_algo.v40_job import (
    BASE_ARTIFACTS, MAX_ARTIFACT_BYTES, artifact_record, sha256_file,
    strict_json, validate_completion, verify_export_sidecar,
)
from wheeled_tasks.v40.contract import contract_digest, validate_contract


SOURCE_CONTRACT_SHA256 = "e31ba1538be4f26d5a5c6452ca8b06e1d09f41db999118957909d13f7248e73e"
ASSET_SHA256 = "df5ca7693022b7a4c68cb1dc823482291f265e3addc8c4870fb3b0b2ab364886"


def _snapshot_record(path: Path) -> dict:
    """Hash a copied snapshot file, including periodic names outside final protocol."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"stage-transfer snapshot must be a regular nonlinked file: {path.name}")
    if not 0 < before.st_size <= MAX_ARTIFACT_BYTES:
        raise ValueError(f"stage-transfer invalid snapshot size: {path.name}")
    digest = sha256_file(path)
    after = path.lstat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"stage-transfer snapshot changed while hashing: {path.name}")
    return {"path": path.name, "size": before.st_size, "sha256": digest}


def prepare_stage_transfer(checkpoint: Path, target_contract: dict, target_manifest: dict,
                           *, source_checkpoint_sha256: str, expected_provenance: dict | None = None):
    """Verify a stable A snapshot, returning only actor/critic/std and bound lineage.

    Periodic snapshots need no completion receipt. A final must have a successful
    receipt; any receipt present is verified in full, even for a periodic source.
    Call again with preflight lineage immediately before loading the fresh PPO.
    """
    checkpoint = Path(checkpoint)
    if not isinstance(source_checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_checkpoint_sha256):
        raise ValueError("stage-transfer requires an explicit lowercase checkpoint SHA256")
    is_final = checkpoint.name == "model_final.pt"
    if not is_final and not re.fullmatch(r"model_\d+\.pt", checkpoint.name):
        raise ValueError("stage-transfer requires the original model_<iteration>.pt or model_final.pt filename")
    directory = checkpoint.parent
    names = set(BASE_ARTIFACTS) | {checkpoint.name}
    receipt_path = directory / "completion.json"
    has_receipt = receipt_path.exists() or receipt_path.is_symlink()
    if is_final and not has_receipt:
        raise ValueError("stage-transfer final requires a successful completion receipt")
    if has_receipt:
        names.add("completion.json")
    records = {name: _snapshot_record(directory / name) for name in sorted(names)}
    if records[checkpoint.name]["sha256"] != source_checkpoint_sha256:
        raise ValueError("stage-transfer checkpoint SHA256 mismatch")
    # The existing loader checks ALL actor/critic tensors, finite/std and infos.
    _, source = load_actor_checkpoint(checkpoint, directory / "run_manifest.json")
    if (source["checkpoint_sha256"] != source_checkpoint_sha256
            or source["run_manifest_sha256"] != records["run_manifest.json"]["sha256"]):
        raise ValueError("stage-transfer parent changed during validation")
    receipt = None
    if has_receipt:
        receipt = validate_completion(strict_json(receipt_path.read_bytes()))
        if receipt["status"] not in {"completed", "stopped"}:
            raise ValueError("stage-transfer parent completion is not successful")
        for record in receipt["artifacts"]:
            if artifact_record(directory, record["path"]) != record:
                raise ValueError(f"stage-transfer completion artifact mismatch: {record['path']}")
            if record["path"] in records and records[record["path"]] != record:
                raise ValueError("stage-transfer completion disagrees with the loaded snapshot")
            records[record["path"]] = record
        verify_export_sidecar(directory)

    parent = source["run_manifest"]
    source_contract = strict_json((directory / "contract.json").read_bytes())
    validate_contract(source_contract)
    validate_contract(target_contract)
    validate_manifest(target_manifest)
    if (contract_digest(source_contract) != SOURCE_CONTRACT_SHA256
            or parent["contract_sha256"] != SOURCE_CONTRACT_SHA256
            or source_contract.get("round3", {}).get("stage") != "A"):
        raise ValueError("stage-transfer requires the pinned Round3-A contract")
    if (target_contract.get("round3", {}).get("stage") != "B1"
            or type(target_contract["round3"].get("material_randomization")) is not dict):
        raise ValueError("stage-transfer target must be a validated B1 material task")
    normalized = deepcopy(target_contract)
    normalized["round3"]["stage"] = "A"
    del normalized["round3"]["material_randomization"]
    if json.dumps(normalized, sort_keys=True) != json.dumps(source_contract, sort_keys=True):
        raise ValueError("stage-transfer permits only round3.stage and round3.material_randomization differences")
    if contract_digest(target_contract) != target_manifest["contract_sha256"]:
        raise ValueError("stage-transfer target contract digest mismatch")
    for key in ("contract_id", "asset_manifest_sha256", "actor_obs_dim", "critic_obs_dim", "action_dim", "policy"):
        if parent[key] != target_manifest[key]:
            raise ValueError(f"stage-transfer fixed manifest mismatch: {key}")
    if (parent["asset_manifest_sha256"] != ASSET_SHA256
            or records["asset_manifest.json"]["sha256"] != ASSET_SHA256):
        raise ValueError("stage-transfer requires the audited original asset")
    if parent["policy"] != source_contract["policy"] or target_manifest["policy"] != target_contract["policy"]:
        raise ValueError("stage-transfer policy/contract mismatch")
    for manifest in (parent, target_manifest):
        runtime = manifest.get("runtime", {})
        if (runtime.get("checkpoint_format") != "rsl_rl_5_split_mlp"
                or runtime.get("versions", {}).get("rsl-rl-lib") != "5.5.1"
                or runtime.get("actor_class") != "MLPModel" or runtime.get("critic_class") != "MLPModel"):
            raise ValueError("stage-transfer requires the audited RSL 5.5.1 split MLP runtime")
    hashes = strict_json((directory / "source_hashes.json").read_bytes())
    if (type(hashes) is not dict
            or hashes.get("contract_snapshot_sha256") != records["contract.json"]["sha256"]
            or hashes.get("asset_manifest_sha256") != ASSET_SHA256
            or type(hashes.get("source_files_sha256")) is not dict
            or not hashes["source_files_sha256"]
            or any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in hashes["source_files_sha256"].values())):
        raise ValueError("stage-transfer source snapshot identity mismatch")
    saved, digest = _read_checkpoint(checkpoint)
    if digest != source_checkpoint_sha256:
        raise ValueError("stage-transfer checkpoint changed after validation")
    _validate_state(saved, parent)
    if type(saved.get("iter")) is not int or saved["iter"] < 0:
        raise ValueError("stage-transfer requires a nonnegative saved iteration")
    if any(_snapshot_record(directory / name) != record for name, record in records.items()):
        raise ValueError("stage-transfer source snapshot changed during validation")
    if (receipt_path.exists() or receipt_path.is_symlink()) != has_receipt:
        raise ValueError("stage-transfer completion changed during validation")
    provenance = {
        "mode": "stage_transfer", "source_stage": "A", "target_stage": "B1",
        "optimizer_reset": True, "initial_iteration": 0, "std_preserved": True,
        "source_is_final": is_final, "source_filename": checkpoint.name,
        "saved_iter": saved["iter"], "parent_completion_verified": has_receipt,
        "parent_completion_status": receipt["status"] if receipt else None,
        "parent_completed_updates": receipt["completed_updates"] if receipt else None,
        "parent_checkpoint_sha256": digest,
        "parent_contract_sha256": SOURCE_CONTRACT_SHA256,
        "parent_runtime": parent["runtime"],
        "target_contract_sha256": target_manifest["contract_sha256"],
        "asset_manifest_sha256": ASSET_SHA256,
        "changed_contract_fields": ["round3.stage", "round3.material_randomization"],
        "parent_snapshot_files": records,
    }
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError("stage-transfer parent snapshot changed since preflight")
    split = {f"{role}_state_dict": {key: value.clone() for key, value in saved[f"{role}_state_dict"].items()}
             for role in ("actor", "critic")}
    return split, provenance


def bind_material_report(manifest: dict, contract: dict, env) -> None:
    """Capture the env-owned startup/readback evidence before publishing the manifest."""
    if contract.get("round3", {}).get("stage") != "B1":
        return
    report = getattr(env, "round3_material_report", None)
    if type(report) is not dict or not report:
        raise ValueError("B1 requires the environment startup material report")
    if type(env.num_envs) is not int or env.num_envs < 1:
        raise ValueError("B1 requires a positive environment count")
    updated = deepcopy(manifest)
    updated.update(round3_stage="B1", num_envs=env.num_envs,
                   domain_randomization_report=deepcopy(report))
    validate_manifest(updated)  # Includes strict JSON primitives and finite numbers.
    manifest.update(updated)


def require_same_material_mapping(saved_manifest: dict, current_manifest: dict) -> None:
    """Exact B1 resume includes the startup bucket assignment and material readback."""
    if current_manifest.get("round3_stage") != "B1":
        return
    if saved_manifest.get("num_envs") != current_manifest.get("num_envs"):
        raise ValueError("B1 exact resume requires identical num_envs; resampling is not exact resume")
    old = saved_manifest.get("domain_randomization_report")
    current = current_manifest.get("domain_randomization_report")
    if (saved_manifest.get("round3_stage") != "B1" or not old or not current
            or json.dumps(old, sort_keys=True, allow_nan=False) != json.dumps(current, sort_keys=True, allow_nan=False)):
        raise ValueError("B1 exact resume requires identical domain_randomization_report")
