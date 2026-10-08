"""Read-only admission for a proposed single learning lineage.

This candidate is deliberately not imported by the production dispatcher or
worker. It selects a full-state parent and computes an allowance. Its CPU
native model/Adam template only restores and compares state; it never collects
rollouts, executes an optimizer update, mutates an account, or grants qualification.

The frozen manifest has ``schema``, ``lineage_id``, absolute ``source_root`` and
``run_root``, ``campaign_plan`` (file/sha256), ``recovery_entry`` (contract_file,
contract_sha256, actor_endpoint), and four ordered ``domains``. Each domain has
name, parent_domain, contract_file, contract_sha256 and critic_warmup_updates.
The REC checkpoint does not exist until the normal boundary: ``admit_entry``
then binds its exact bytes to this frozen manifest without rewriting the plan.

Joint checkpoints carry the same ``integrated_lineage`` object in infos and
their seal: lineage_id, plan_sha256, origin_checkpoint_sha256, domain,
joint_actor_updates, domain_actor_updates, domain_complete. Both actor clocks
count ONLY actor updates in this new lineage. Historical branch work is charged
by the existing campaign account and is never credited to these clocks.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .task_campaign import CAMPAIGN_ID, DOMAINS, account_updates
from .full_curriculum import checkpoint_contract_path, checkpoint_update_count

SCHEMA = "v6-integrated-lineage-plan-v1"
EARLY_SCHEMA = "v6-authorized-REC-early-stop-v1"
EARLY_PROTOCOL = "authenticated_supervisor_planned_stop"
CHAIN = (("speed", "recovery"), ("terrain", "speed"),
         ("jump", "terrain"), ("stepjump", "jump"))
PHYSICAL_KEYS = ("asset_manifest_sha256", "control_math_sha256", "physics_dt", "policy_dt",
                 "position_iterations", "velocity_iterations", "closure_gap_termination_m",
                 "actor_dim", "critic_dim", "action_dim", "history_length", "policy_architecture",
                 "v5_control", "actuator_response", "actuator_response_evidence", "policy_action_order",
                 "actor_layout", "critic_layout", "actor_observation_source")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {name}")
    return value


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("A full lowercase SHA-256 is required")
    return value


def _within(path, root):
    root = Path(root).resolve(strict=True)
    path = Path(path)
    path = (root / path if not path.is_absolute() else path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Bound file is outside its declared root")
    return path


def _json(path, expected_sha, root):
    path = _within(path, root)
    if digest(path) != _sha(expected_sha):
        raise ValueError("Frozen file SHA-256 differs")
    return path, json.loads(path.read_text())


@dataclass(frozen=True)
class FrozenPlan:
    path: Path
    sha256: str
    source_root: Path
    run_root: Path

    def read(self):
        return load_plan(self.path, self.sha256, self.source_root, self.run_root)[1]


def load_plan(path, expected_sha256, source_root, run_root):
    """Authenticate immutable recipes and identical physical/neural interfaces."""
    path, plan = _json(path, expected_sha256, source_root)
    source_root, run_root = Path(source_root).resolve(strict=True), Path(run_root).resolve(strict=True)
    if (plan.get("schema") != SCHEMA or not isinstance(plan.get("lineage_id"), str)
            or re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", plan["lineage_id"]) is None
            or plan.get("source_root") != str(source_root) or plan.get("run_root") != str(run_root)):
        raise ValueError("Joint lineage identity or root binding differs")
    pin = plan["campaign_plan"]
    _, campaign = _json(pin["file"], pin["sha256"], source_root)
    if (campaign.get("campaign_id") != CAMPAIGN_ID
            or campaign.get("total_ceiling") != 121000
            or campaign.get("historical_consumed_updates") != 1000
            or campaign.get("historical_actor_consumed_updates") != 980
            or campaign.get("rollback_refunds_consumed_updates") is not False
            or campaign.get("automatic_promotion") is not False
            or tuple(r["name"] for r in campaign["domains"]) != DOMAINS
            or any(r["updates"] != 20000 for r in campaign["domains"])):
        raise ValueError("Joint admission cannot change the existing six caps or account")
    entry = plan["recovery_entry"]
    recovery = next(r for r in campaign["domains"] if r["name"] == "recovery")
    if (entry.get("actor_endpoint") != 15000
            or entry.get("contract_file") != recovery["contract"]
            or entry.get("contract_sha256") != recovery["sha256"]
            or campaign["domain_completion"]["schedule"]["end_actor_updates"]["recovery"] != 15000):
        raise ValueError("The entry must be the frozen normal REC 15000 endpoint")
    _, parent = _json(entry["contract_file"], entry["contract_sha256"], source_root)
    _early_authorization(plan, source_root, run_root)
    if tuple((r["name"], r["parent_domain"]) for r in plan["domains"]) != CHAIN:
        raise ValueError("Joint domains must form recovery→speed→terrain→jump→stepjump")
    for row in plan["domains"]:
        if row.get("critic_warmup_updates") != 10:
            raise ValueError("Each changed worker requires ten charged critic updates")
        _, worker = _json(row["contract_file"], row["contract_sha256"], source_root)
        meta = worker.get("task_campaign", {})
        if (meta.get("domain") != row["name"] or meta.get("campaign_id") != CAMPAIGN_ID
                or meta.get("updates") != 20000 or meta.get("clock") != "domain_actor_updates"
                or worker.get("resume_critic_warmup_updates") != 10
                or worker.get("target_num_envs") != 4096
                or worker.get("learning_rate") != 1e-4 or worker.get("learning_rate_schedule") != "fixed"
                or any(worker.get(k) != parent.get(k) for k in PHYSICAL_KEYS)):
            raise ValueError("Joint worker changes the frozen account or physical/policy identity")
        parent = worker
    return FrozenPlan(path, _sha(expected_sha256), source_root, run_root), deepcopy(plan)


def _early_authorization(plan, source_root, run_root):
    """An opt-in stop authorization cannot replace the old plan or its ceiling."""
    pin = plan["recovery_entry"].get("early_stop")
    if pin is None:
        return None
    if (not isinstance(pin, dict) or pin.get("schema") != EARLY_SCHEMA
            or set(pin) != {"schema", "authorization_file", "authorization_sha256"}):
        raise ValueError("Unknown REC early-stop authorization binding")
    _, authorization = _json(pin["authorization_file"], pin["authorization_sha256"], source_root)
    entry = plan["recovery_entry"]
    minimum = authorization.get("minimum_actor_update")
    if (authorization.get("schema") != EARLY_SCHEMA
            or authorization.get("lineage_id") != plan["lineage_id"]
            or authorization.get("decision") != "seal_current_REC_then_same_actor_serial"
            or authorization.get("run_root") != str(Path(run_root).resolve(strict=True))
            or authorization.get("source_commit") != "6f07ec298e01d2686216787cfba34b394151f1aa"
            or authorization.get("source_plan_sha256") != plan["campaign_plan"]["sha256"]
            or authorization.get("recovery_contract_sha256") != entry["contract_sha256"]
            or authorization.get("planned_actor_endpoint") != entry["actor_endpoint"]
            or type(minimum) is not int or not 0 < minimum < entry["actor_endpoint"]
            or "qualification" not in authorization or authorization["qualification"] is not None):
        raise ValueError("REC early-stop authorization differs from its frozen lineage/source")
    return authorization


def _native_learning_template(contract):
    """Use the worker's native factory for the frozen, unchanged 35/81 MLP ABI."""
    from contextlib import redirect_stdout
    from io import StringIO
    from types import SimpleNamespace
    import torch
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO

    architecture = contract.get("policy_architecture")
    if (not isinstance(architecture, dict)
            or architecture.get("actor_hidden_dims") != [256, 128, 64]
            or architecture.get("critic_hidden_dims") != [256, 128, 64]
            or architecture.get("activation") != "elu"
            or tuple(contract.get(k) for k in ("actor_dim", "critic_dim", "action_dim", "history_length"))
            != (35, 81, 6, 1)):
        raise ValueError("Frozen contract does not describe the native 35/81, 256/128/64 MLP")
    model = dict(class_name="rsl_rl.models.MLPModel", activation=architecture["activation"],
                 obs_normalization=False)
    cfg = dict(actor=dict(model, hidden_dims=architecture["actor_hidden_dims"],
                          distribution_cfg=dict(class_name="rsl_rl.modules.GaussianDistribution",
                                                init_std=contract.get("initial_noise_std", .2),
                                                std_type="scalar")),
               critic=dict(model, hidden_dims=architecture["critic_hidden_dims"]),
               algorithm=dict(class_name="rsl_rl.algorithms.PPO", optimizer="adam",
                              learning_rate=contract["learning_rate"], schedule="fixed"),
               obs_groups={"actor": ["policy"], "critic": ["critic"]},
               num_steps_per_env=1, multi_gpu=None, torch_compile_mode=None)
    obs = TensorDict({"policy": torch.zeros(1, contract["actor_dim"]),
                      "critic": torch.zeros(1, contract["critic_dim"])}, batch_size=[1])
    # Factory construction initializes random weights, so isolate its CPU RNG.
    # One empty storage slot is allocated; act/update/optimizer.step are never called.
    with torch.random.fork_rng(devices=[]), redirect_stdout(StringIO()):
        return PPO.construct_algorithm(obs, SimpleNamespace(num_envs=1, num_actions=contract["action_dim"]),
                                       cfg, "cpu")


def _validate_native_learning_state(contract, saved):
    """Strict native model restore, complete finite Adam moments, exact re-save."""
    import torch
    from .policy_transfer import verify_learning_state_restore

    algorithm = _native_learning_template(contract)
    for name in ("actor_state_dict", "critic_state_dict"):
        state = saved.get(name)
        if (not isinstance(state, dict)
                or any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all().item()
                       for t in state.values())):
            raise ValueError("Source lacks complete finite actor and critic state")
    std = saved["actor_state_dict"].get("distribution.std_param")
    if not isinstance(std, torch.Tensor) or not (std > 0).all().item():
        raise ValueError("Source lacks a positive native actor standard deviation")
    # The native loader owns key/shape validation; dtype must also survive exactly.
    expected = algorithm.save()
    for name in ("actor_state_dict", "critic_state_dict"):
        for key in saved[name].keys() & expected[name].keys():
            if saved[name][key].dtype != expected[name][key].dtype:
                raise ValueError("Source native model tensor dtype differs")
    try:
        algorithm.load(saved, load_cfg=dict(actor=True, critic=True, optimizer=False), strict=True)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ValueError("Source differs from the strict native model keys/shapes") from exc
    opt = saved.get("optimizer_state_dict")
    if not isinstance(opt, dict) or not isinstance(opt.get("param_groups"), list) or not isinstance(opt.get("state"), dict):
        raise ValueError("Source lacks complete native Adam state")
    native_groups = algorithm.optimizer.param_groups
    if (len(opt["param_groups"]) != len(native_groups)
            or any(len(group.get("params", [])) != len(native["params"])
                   for group, native in zip(opt["param_groups"], native_groups, strict=True))):
        raise ValueError("Source Adam parameter groups differ from the native model")
    ids = [p for group in opt["param_groups"] for p in group["params"]]
    parameters = [p for group in native_groups for p in group["params"]]
    if len(set(ids)) != len(parameters) or set(ids) != opt["state"].keys():
        raise ValueError("Source lacks all native Adam states")
    for index, parameter in zip(ids, parameters, strict=True):
        state = opt["state"][index]
        if not isinstance(state, dict):
            raise ValueError("Source Adam state is incomplete")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state.get(key)
            if (not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype or not torch.isfinite(moment).all().item()
                    or key == "exp_avg_sq" and not (moment >= 0).all().item()):
                raise ValueError("Source Adam moments are incomplete or nonfinite")
        step = state.get("step")
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all().item() or step.item() < 0:
            raise ValueError("Source Adam step is invalid")
    try:
        algorithm.load(saved, load_cfg=dict(optimizer=True), strict=True)
        verify_learning_state_restore(algorithm, saved)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ValueError("Source native Adam/full-state restore failed") from exc


def _checkpoint(binding, reference, domain, contract_sha256):
    """Authenticate real state using CPU templates, without learning or an environment."""
    import torch

    path = _within(reference["path"], binding.run_root)
    if not path.is_relative_to(binding.run_root / domain) or any(p.endswith(".pending") for p in path.parts):
        raise ValueError("Checkpoint is outside its domain or remains pending")
    if digest(path) != _sha(reference["sha256"]):
        raise ValueError("Bound source checkpoint SHA-256 differs")
    seal_path = _within(path.with_suffix(".checkpoint.json"), binding.run_root)
    seal = json.loads(seal_path.read_text())
    contract_path = checkpoint_contract_path(path)
    _, contract = _json(contract_path, contract_sha256, binding.run_root)
    # Use the production per-file seal validator after confining both sidecars.
    # Native factory/strict restoration below also verifies the real model ABI.
    count = checkpoint_update_count(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    infos = saved["infos"]
    if (infos.get("contract_sha256") != contract_sha256
            or infos.get("successful_updates_total") != count
            or contract.get("task_campaign", {}).get("domain") != domain
            or infos.get("task_campaign", {}).get("domain") != domain
            or seal.get("task_campaign", {}).get("domain") != domain):
        raise ValueError("Checkpoint seal, infos or domain identity differs")
    for key in ("asset_manifest_sha256", "control_math_sha256"):
        if infos.get(key) != contract.get(key):
            raise ValueError("Checkpoint physical/control provenance differs")
    for key in ("rng_state", "env_rng_state"):
        state = infos.get(key)
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1 or not state.numel():
            raise ValueError("Source lacks its complete saved RNG state")
    for name, info_name in (("learning_updates", "successful_updates_total"),
                            ("learning_lineage_updates", "learning_lineage_updates"),
                            ("stage_actor_update", "stage_actor_update"),
                            ("task_actor_update", "task_actor_update"),
                            ("actor_updates_consumed", "actor_updates_consumed")):
        _integer(seal.get(name), name)
        if seal[name] != infos.get(info_name):
            raise ValueError("Checkpoint seal and full-state clocks differ")
    if (seal["learning_lineage_updates"] > seal["learning_updates"]
            or seal["actor_updates_consumed"] > seal["learning_updates"]):
        raise ValueError("Checkpoint learning/actor clocks exceed charged work")
    _validate_native_learning_state(contract, saved)
    return dict(path=str(path), sha256=reference["sha256"], seal=seal, infos=infos)


def _recovery_origin_directory(binding, checkpoint, record):
    correction = record.get("early_stop_origin_correction")
    if correction is None:
        return binding.run_root / "integrated_early_origin_v26"
    directory = binding.run_root / "integrated_early_origin_v26_dispatch_correction029"
    file, intent = _json(correction["file"], correction["sha256"], binding.run_root)
    if (file != (directory / "intent.json").resolve(strict=True)
            or intent.get("schema") != "known-integrated-dispatch-origin-correction029-v1"
            or intent.get("action") != "retain_latest_misdirected_REC_then_joint_chain"
            or intent.get("lineage_id") != binding.read()["lineage_id"]
            or intent.get("lineage_manifest_sha256") != binding.sha256
            or intent.get("checkpoint") != {"path": checkpoint["path"], "sha256": checkpoint["sha256"]}
            or intent.get("PPO_updates_by_repair") != 0 or intent.get("qualification") is not None):
        raise ValueError("REC origin correction is not the explicit once-only dispatch repair")
    before = {}
    for key in ("migration", "state", "account"):
        evidence = intent["before"][key]
        file, value = _json(evidence["file"], evidence["sha256"], binding.run_root)
        if file != (directory / ("before_" + key + ".json")).resolve(strict=True):
            raise ValueError("REC correction predecessor is not its immutable snapshot")
        before[key] = value
    expected = intent["expected"]
    if (before["migration"].get("entry_admission") != intent["previous_entry"]
            or before["state"].get("authorized_REC_early_stop") != intent["previous_entry"]["record"]
            or before["state"].get("status") != "stopped"
            or before["account"].get("rollback_refunds_consumed_updates") is not False
            or before["account"].get("joint_learning_sources") or before["state"].get("joint_learning_sources")
            or any(before["state"]["domains"][domain].get("status") != "joint_domain_pending" for domain, _ in CHAIN)
            or expected["consumed_updates"] != checkpoint["seal"]["learning_updates"]
            or expected["REC_actor"] != checkpoint["seal"]["task_actor_update"]
            or expected["consumed_updates"] - expected["previous_consumed_updates"] != 19
            or expected["REC_actor"] - expected["previous_REC_actor"] != 19):
        raise ValueError("REC correction changed its charged predecessor or joint origin")
    for key, name in (("early_stop_campaign_state", "campaign_state.json"),
                      ("early_stop_campaign_completion", "completion.json")):
        original = intent["previous_entry"]["record"][key]
        old_file, _ = _json(original["file"], original["sha256"], binding.run_root)
        if old_file != (binding.run_root / "integrated_early_origin_v26" / name).resolve(strict=True):
            raise ValueError("REC correction must preserve its original immutable origin")
    return directory


def _finished(binding, checkpoint, record, *, recovery=False):
    """A parent must have a final seal and its normal terminal completion."""
    path = Path(checkpoint["path"])
    completion = json.loads(_within(path.parent / "completion.json", binding.run_root).read_text())
    plan = binding.read() if recovery else None
    authorization = _early_authorization(plan, binding.source_root, binding.run_root) if recovery else None
    expected_status = "stopped" if authorization is not None else "completed"
    if (path.name != "model_final.pt" or completion.get("status") != expected_status
            or record.get("candidate_checkpoint") != str(path)
            or record.get("candidate_checkpoint_sha256") != checkpoint["sha256"]
            or completion.get("parent_updates", 0) + completion.get("successful_updates", 0)
            != checkpoint["seal"]["learning_updates"]):
        raise ValueError("Parent has not normally terminated with this final checkpoint")
    if authorization is not None:
        pin = plan["recovery_entry"]["early_stop"]
        origin_directory = _recovery_origin_directory(binding, checkpoint, record)
        stopped = []
        for key, name in (("early_stop_campaign_state", "campaign_state.json"),
                          ("early_stop_campaign_completion", "completion.json")):
            evidence = record.get(key, {})
            file, value = _json(evidence["file"], evidence["sha256"], binding.run_root)
            if file != (origin_directory / name).resolve(strict=True):
                raise ValueError("REC early-stop evidence is not its immutable origin snapshot")
            stopped.append(value)
        state, final = stopped
        actual = checkpoint["seal"]["task_actor_update"]
        if (state != final or state.get("status") != "stopped"
                or record.get("status") != "authorized_REC_early_stop_unassessed"
                or "domain_qualified" not in record or record["domain_qualified"] is not None
                or record.get("early_stop_protocol") != EARLY_PROTOCOL
                or record.get("early_stop_authorization_sha256") != pin["authorization_sha256"]
                or completion.get("checkpoint_sha256") != checkpoint["sha256"]
                or completion.get("task_actor_update") != actual
                or not authorization["minimum_actor_update"] <= actual < authorization["planned_actor_endpoint"]):
            raise ValueError("REC early entry lacks authenticated planned-stop final evidence")
        # A stopped campaign cannot authorize an older branch snapshot as the new origin.
        for other in (binding.run_root / "recovery").glob("attempt_*/train/model_final.pt"):
            if checkpoint_update_count(other) > checkpoint["seal"]["learning_updates"]:
                raise ValueError("REC early entry does not select its latest final learning state")
    elif recovery:
        schedule = record.get("schedule_completion", {})
        if (record.get("status") != "domain_schedule_finished_unassessed"
                or schedule.get("reason") != "iteration_endpoint"
                or schedule.get("end_actor_update") != 15000
                or schedule.get("actual_actor_update") != 15000
                or checkpoint["seal"]["task_actor_update"] != 15000
                or completion.get("task_actor_update") != 15000):
            raise ValueError("REC was not sealed at its frozen normal endpoint")
    elif (record.get("status") != "joint_domain_finished_unassessed"
          or checkpoint["infos"]["integrated_lineage"]["domain_complete"] is not True
          or completion.get("integrated_lineage") != checkpoint["infos"]["integrated_lineage"]):
        raise ValueError("Joint parent lacks authenticated normal domain completion")


def admit_entry(binding, reference, record):
    """Bind the future REC endpoint bytes to an already frozen candidate plan."""
    plan = binding.read()
    checkpoint = _checkpoint(binding, reference, "recovery", plan["recovery_entry"]["contract_sha256"])
    _finished(binding, checkpoint, record, recovery=True)
    return {"schema": "v6-integrated-lineage-entry-v1", "plan_sha256": binding.sha256,
            "lineage_id": plan["lineage_id"], "checkpoint": deepcopy(reference), "record": deepcopy(record)}


def _joint(binding, plan, checkpoint, domain, origin):
    meta = checkpoint["infos"].get("integrated_lineage")
    if (not isinstance(meta, dict) or checkpoint["seal"].get("integrated_lineage") != meta
            or meta.get("lineage_id") != plan["lineage_id"] or meta.get("plan_sha256") != binding.sha256
            or meta.get("origin_checkpoint_sha256") != origin["sha256"] or meta.get("domain") != domain
            or type(meta.get("domain_complete")) is not bool):
        raise ValueError("Checkpoint belongs to another or incomplete joint lineage")
    _integer(meta.get("joint_actor_updates"), "joint actor clock")
    _integer(meta.get("domain_actor_updates"), "joint domain actor clock")
    if (meta["domain_actor_updates"] > meta["joint_actor_updates"]
            or meta["domain_actor_updates"] != checkpoint["infos"]["task_actor_update"]):
        raise ValueError("Joint actor clocks are inconsistent")
    learning_delta = checkpoint["seal"]["learning_lineage_updates"] - origin["seal"]["learning_lineage_updates"]
    actor_delta = checkpoint["seal"]["actor_updates_consumed"] - origin["seal"]["actor_updates_consumed"]
    if learning_delta < meta["joint_actor_updates"] or actor_delta < meta["joint_actor_updates"]:
        raise ValueError("Fresh joint actor clock exceeds learning/actor increments from its sealed REC origin")
    _integer(checkpoint["infos"].get("critic_warmup_remaining"), "remaining critic adaptation")
    if checkpoint["infos"]["critic_warmup_remaining"] > 10:
        raise ValueError("Joint resume cannot add critic adaptation")
    return meta


def select_learning_source(binding, domain, entry, *, own_candidates=(), parent=None,
                           parent_record=None, previous_selection=None):
    """Explicitly rebase once; subsequent faults resume the latest joint own seal.

    ``own_candidates`` are explicit SHA-bound references to this lineage's
    checkpoints, not a glob containing historical branch candidates. A durable
    ``previous_selection`` supplies the monotonic floor after a prior admission.
    """
    plan = binding.read()
    if (entry.get("schema") != "v6-integrated-lineage-entry-v1"
            or entry.get("plan_sha256") != binding.sha256 or entry.get("lineage_id") != plan["lineage_id"]):
        raise ValueError("REC entry admission belongs to another frozen lineage")
    admit_entry(binding, entry["checkpoint"], entry["record"])
    # Read the origin's actual pinned seal again; entry metadata cannot supply clocks.
    origin = _checkpoint(binding, entry["checkpoint"], "recovery", plan["recovery_entry"]["contract_sha256"])
    row = next((r for r in plan["domains"] if r["name"] == domain), None)
    if row is None:
        raise ValueError("Unknown joint domain")
    candidates = []
    for reference in own_candidates:
        checkpoint = _checkpoint(binding, reference, domain, row["contract_sha256"])
        meta = _joint(binding, plan, checkpoint, domain, origin)
        candidates.append((checkpoint["seal"]["learning_updates"], checkpoint, meta))
    candidates.sort(key=lambda item: item[0])
    for before, after in zip(candidates, candidates[1:]):
        if (after[0] == before[0] and after[1]["sha256"] != before[1]["sha256"]
                or after[2]["joint_actor_updates"] < before[2]["joint_actor_updates"]
                or after[2]["domain_actor_updates"] < before[2]["domain_actor_updates"]
                or after[1]["seal"]["learning_lineage_updates"] < before[1]["seal"]["learning_lineage_updates"]):
            raise ValueError("Joint own checkpoints fork or move their learning clocks backwards")
    if candidates:
        _, selected, meta = candidates[-1]
        mode, warmup = "resume_joint_own", selected["infos"]["critic_warmup_remaining"]
        joint_clock, domain_clock = meta["joint_actor_updates"], meta["domain_actor_updates"]
        complete = meta["domain_complete"]
    elif domain == "speed":
        selected = origin
        mode = "rebase_from_authorized_REC" if plan["recovery_entry"].get("early_stop") else "rebase_from_normal_REC"
        warmup, joint_clock, domain_clock = 10, 0, 0
        complete = False
    else:
        if parent is None or parent_record is None:
            raise ValueError("Joint domain requires its normally completed joint parent")
        parent_row = next(r for r in plan["domains"] if r["name"] == row["parent_domain"])
        selected = _checkpoint(binding, parent, parent_row["name"], parent_row["contract_sha256"])
        meta = _joint(binding, plan, selected, parent_row["name"], origin)
        _finished(binding, selected, parent_record)
        mode, warmup = "inherit_completed_joint_parent", 10
        joint_clock, domain_clock = meta["joint_actor_updates"], 0
        complete = False
    result = {"schema": "v6-integrated-lineage-selection-v1", "lineage_id": plan["lineage_id"],
              "plan_sha256": binding.sha256, "domain": domain, "mode": mode,
              "checkpoint": {"path": selected["path"], "sha256": selected["sha256"]},
              "origin_checkpoint_sha256": origin["sha256"], "contract_sha256": row["contract_sha256"],
              "fresh_joint_actor_updates": joint_clock, "fresh_domain_actor_updates": domain_clock,
              "learning_lineage_updates": selected["seal"]["learning_lineage_updates"],
              "checkpoint_consumed_updates": selected["seal"]["learning_updates"],
              "checkpoint_actor_updates_consumed": selected["seal"]["actor_updates_consumed"],
              "domain_complete": complete,
              "critic_warmup_remaining": warmup, "actor_critic_optimizer": "preserved",
              "environment_state": "new_episodes", "qualification": None,
              "entry_admission": deepcopy(entry),
              "parent_record": deepcopy(parent_record) if mode == "inherit_completed_joint_parent" else None}
    if plan["recovery_entry"].get("early_stop"):
        result["origin_recovery_actor_update"] = origin["seal"]["task_actor_update"]
    if previous_selection is not None:
        if previous_selection.get("origin_recovery_actor_update") != result.get("origin_recovery_actor_update"):
            raise ValueError("Previous selection changed its authenticated REC actor origin")
        for key in ("schema", "lineage_id", "plan_sha256", "domain", "origin_checkpoint_sha256", "contract_sha256"):
            if previous_selection.get(key) != result[key]:
                raise ValueError("Previous selection belongs to another lineage/domain")
        for key in ("fresh_joint_actor_updates", "fresh_domain_actor_updates", "learning_lineage_updates", "checkpoint_consumed_updates"):
            _integer(previous_selection.get(key), key)
            if result[key] < previous_selection[key]:
                raise ValueError("Latest joint checkpoint or actor clock regressed")
        if (result["checkpoint_consumed_updates"] == previous_selection["checkpoint_consumed_updates"]
                and result["checkpoint"]["sha256"] != previous_selection["checkpoint"]["sha256"]):
            raise ValueError("A checkpoint at the same consumption clock was replaced")
    return result


def remaining_budget(binding, selection, attempts, *, requested_actor_updates=None):
    """Charge legacy branches and adaptation; fresh learning clocks get no refund."""
    plan = binding.read()
    if (selection.get("schema") != "v6-integrated-lineage-selection-v1"
            or selection.get("plan_sha256") != binding.sha256
            or selection.get("lineage_id") != plan["lineage_id"]):
        raise ValueError("Budget selection differs from the frozen lineage")
    domain = selection["domain"]
    if domain not in dict(CHAIN):
        raise ValueError("Unknown joint allowance domain")
    mode = selection.get("mode")
    kwargs = ({"own_candidates": [selection["checkpoint"]]} if mode == "resume_joint_own" else
              {"parent": selection["checkpoint"], "parent_record": selection["parent_record"]}
              if mode == "inherit_completed_joint_parent" else {})
    if mode not in ("resume_joint_own", "inherit_completed_joint_parent", "rebase_from_normal_REC", "rebase_from_authorized_REC"):
        raise ValueError("Unknown joint source selection mode")
    verified = select_learning_source(binding, domain, selection["entry_admission"], **kwargs)
    if verified != selection:
        raise ValueError("Joint allowance selection differs from authenticated source clocks")
    _, campaign = _json(plan["campaign_plan"]["file"], plan["campaign_plan"]["sha256"], binding.source_root)
    totals = account_updates(campaign, deepcopy(attempts))
    spent = totals["domain_consumed_updates"][domain]
    available = min(20000 - spent, 121000 - totals["consumed_updates"])
    if (totals["consumed_updates"] < selection["checkpoint_consumed_updates"]
            or totals["actor_updates_consumed"] < selection["checkpoint_actor_updates_consumed"]):
        raise ValueError("Consumption account precedes the selected learning state")
    warmup = _integer(selection["critic_warmup_remaining"], "remaining critic adaptation")
    if warmup > 10:
        raise ValueError("Joint allowance changed critic adaptation")
    maximum_actor = 0 if selection["domain_complete"] else max(0, available - warmup)
    requested = maximum_actor if requested_actor_updates is None else _integer(requested_actor_updates, "requested actor work")
    if requested > maximum_actor:
        raise ValueError("Joint work exceeds the remaining charged domain/global caps")
    admitted = requested > 0 and available >= warmup + requested
    return {"schema": "v6-integrated-lineage-allowance-v1", "domain": domain,
            "lineage_id": plan["lineage_id"], "plan_sha256": binding.sha256,
            "consumed_updates": totals["consumed_updates"],
            "actor_updates_consumed": totals["actor_updates_consumed"],
            "historical_and_joint_domain_consumed_updates": spent,
            "domain_remaining_charged_updates": 20000 - spent,
            "global_remaining_charged_updates": 121000 - totals["consumed_updates"],
            "maximum_actor_updates": maximum_actor, "requested_actor_updates": requested,
            "charged_critic_updates": warmup if admitted else 0,
            "requested_total_updates": warmup + requested if admitted else 0,
            "fresh_joint_actor_updates": selection["fresh_joint_actor_updates"],
            "fresh_domain_actor_updates": selection["fresh_domain_actor_updates"],
            "admitted": admitted, "rollback_refunds_consumed_updates": False,
            "qualification": None}
