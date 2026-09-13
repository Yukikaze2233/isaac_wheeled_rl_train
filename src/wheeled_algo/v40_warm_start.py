"""One audited Round2-final -> Round3 RSL 5 migration, never a resume bypass."""
from __future__ import annotations

from pathlib import Path

from wheeled_algo.v40_export import (
    _read_checkpoint, _validate_state, load_actor_checkpoint, sha256_file, validate_manifest,
)
from wheeled_algo.v40_job import (
    artifact_record, strict_json, validate_completion, verify_export_sidecar,
)
from wheeled_tasks.v40.contract import contract_digest, validate_contract


PARENT_SHA256 = "f94919f8f7ae1cfee95e45b922a9939e21c9f6ad9dff5656ff6f29dc0f4e6360"
PARENT_CONTRACT_SHA256 = "667865de8352e67cff6720af9a142d00ff9b980132b27e4993a25c8e09d3d22e"
ALLOWED_CONTRACT_FIELDS = frozenset({"commands", "rewards", "round3"})


def prepare_warm_start(checkpoint: Path, target_contract: dict, target_manifest: dict):
    """Validate the complete parent bundle and return in-memory split weights + lineage.

    The pinned SHA is intentional: this is not a general cross-contract converter.
    Read twice with byte-identity checks so the migrated critic is covered by the
    same exporter validation as the actor. No loaded artifact is ever rewritten.
    """
    checkpoint = Path(checkpoint)
    _, source = load_actor_checkpoint(checkpoint, checkpoint.parent / "run_manifest.json")
    if checkpoint.name != "model_final.pt" or source["checkpoint_sha256"] != PARENT_SHA256:
        raise ValueError("warm-start requires the pinned Round2 final checkpoint")
    parent = source["run_manifest"]
    directory = checkpoint.parent
    completion = validate_completion(strict_json((directory / "completion.json").read_bytes()))
    if completion["status"] not in {"completed", "stopped"}:
        raise ValueError("warm-start parent must have a successful complete receipt")
    for record in completion["artifacts"]:
        if artifact_record(directory, record["path"]) != record:
            raise ValueError(f"warm-start parent receipt hash mismatch: {record['path']}")
    sidecar = verify_export_sidecar(directory)
    if (sidecar["checkpoint_sha256"] != source["checkpoint_sha256"]
            or sidecar["run_manifest_sha256"] != source["run_manifest_sha256"]):
        raise ValueError("warm-start parent changed during validation")
    source_contract = strict_json((directory / "contract.json").read_bytes())
    validate_contract(source_contract)
    validate_contract(target_contract)
    validate_manifest(target_manifest)
    if contract_digest(source_contract) != parent["contract_sha256"] or parent["contract_sha256"] != PARENT_CONTRACT_SHA256:
        raise ValueError("warm-start parent contract digest mismatch")
    if contract_digest(target_contract) != target_manifest["contract_sha256"]:
        raise ValueError("warm-start target contract digest mismatch")
    changed = sorted(key for key in source_contract.keys() | target_contract.keys()
                     if key not in source_contract or key not in target_contract
                     or source_contract[key] != target_contract[key])
    forbidden = set(changed) - ALLOWED_CONTRACT_FIELDS
    if forbidden:
        raise ValueError(f"warm-start forbidden contract differences: {sorted(forbidden)}")
    if "round3" in source_contract or ("round3" in target_contract and type(target_contract["round3"]) is not dict):
        raise ValueError("warm-start round3 must be a new task description object")
    for key in ("contract_id", "asset_manifest_sha256", "actor_obs_dim", "critic_obs_dim", "action_dim", "policy"):
        if parent[key] != target_manifest[key]:
            raise ValueError(f"warm-start fixed manifest field mismatch: {key}")
    if target_manifest["policy"] != target_contract["policy"] or parent["policy"] != source_contract["policy"]:
        raise ValueError("warm-start manifest policy differs from contract")
    if artifact_record(directory, "asset_manifest.json")["sha256"] != parent["asset_manifest_sha256"]:
        raise ValueError("warm-start parent asset digest mismatch")
    if parent.get("target_versions", {}).get("rsl-rl-lib") != "3.0.1" or parent.get("runtime"):
        raise ValueError("warm-start requires the audited RSL 3.0.1 parent runtime")
    runtime = target_manifest.get("runtime", {})
    if (runtime.get("checkpoint_format") != "rsl_rl_5_split_mlp"
            or runtime.get("versions", {}).get("rsl-rl-lib") != "5.5.1"
            or runtime.get("actor_class") != "MLPModel" or runtime.get("critic_class") != "MLPModel"):
        raise ValueError("warm-start target must be RSL 5.5.1 split MLPModel")
    saved, digest = _read_checkpoint(checkpoint)
    if digest != source["checkpoint_sha256"]:
        raise ValueError("warm-start checkpoint changed after validation")
    state = _validate_state(saved, parent)
    if "std" not in state:
        raise ValueError("audited parent requires scalar-space six-dimensional std")
    split = {}
    for role in ("actor", "critic"):
        split[f"{role}_state_dict"] = {
            "mlp." + key.removeprefix(role + "."): value.clone()
            for key, value in state.items() if key.startswith(role + ".")
        }
    split["actor_state_dict"]["distribution.std_param"] = state["std"].clone()
    provenance = {
        "mode": "warm_start", "optimizer_reset": True, "initial_iteration": 0,
        "parent_checkpoint_sha256": digest,
        "parent_contract_sha256": parent["contract_sha256"],
        "parent_run_manifest_sha256": source["run_manifest_sha256"],
        "parent_completion_sha256": sha256_file(directory / "completion.json"),
        "parent_source_hashes_sha256": artifact_record(directory, "source_hashes.json")["sha256"],
        "parent_runtime": {"versions": parent["target_versions"],
                           "checkpoint_format": "rsl_rl_3_actor_critic"},
        "target_contract_sha256": target_manifest["contract_sha256"],
        "asset_manifest_sha256": parent["asset_manifest_sha256"],
        "allowed_contract_fields": sorted(ALLOWED_CONTRACT_FIELDS),
        "changed_contract_fields": changed,
        "std_preserved": True,
    }
    return split, provenance


def apply_warm_start(runner, split: dict) -> None:
    """Load both networks through official PPO; require a genuinely fresh runner."""
    import importlib.metadata
    import torch
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.modules import GaussianDistribution

    if importlib.metadata.version("rsl-rl-lib") != "5.5.1" or type(runner.alg) is not PPO:
        raise ValueError("warm-start requires official RSL 5.5.1 PPO")
    if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
        raise ValueError("warm-start requires fresh optimizer and iteration zero")
    for role in ("actor", "critic"):
        model = getattr(runner.alg, role)
        if type(model) is not MLPModel or model.obs_normalization:
            raise ValueError("warm-start requires unnormalized stock MLPModel")
        # Sequential iteration preserves repeated references to the shared ELU.
        layers = list(model.mlp)
        if [type(layer) for layer in layers] != [torch.nn.Linear, torch.nn.ELU] * 3 + [torch.nn.Linear]:
            raise ValueError("warm-start requires the fixed ELU MLP")
        actual, expected = model.state_dict(), split[f"{role}_state_dict"]
        if actual.keys() != expected.keys() or any(
            actual[key].shape != value.shape or actual[key].dtype != value.dtype
            for key, value in expected.items()
        ):
            raise ValueError(f"warm-start target {role} state signature mismatch")
    distribution = runner.alg.actor.distribution
    std = split["actor_state_dict"]["distribution.std_param"]
    if (type(distribution) is not GaussianDistribution or distribution.std_type != "scalar"
            or not distribution.std_param.requires_grad
            or not bool(((std >= distribution.std_range[0]) & (std <= distribution.std_range[1])).all())
            or runner.alg.critic.distribution is not None):
        raise ValueError("warm-start target would change source std semantics")
    runner.alg.load(split, load_cfg={"actor": True, "critic": True,
                                    "optimizer": False, "iteration": False}, strict=True)
    runner.current_learning_iteration = 0
