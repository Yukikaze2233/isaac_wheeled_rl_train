"""Small opt-in bridges into the existing trainer's restore/update/save hooks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .integrated_preflight import with_scoped_expectation

DESIGN_SCHEMA = "v6-integrated-worker-design-v1"


def _read(path, sha):
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Integrated worker design file SHA differs")
    return json.loads(raw)


def make_design_hook(native_preflight, config, receipt, args):
    """Resolve declared scopes before any native instance; static bytes stay intact."""
    invocation = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    return dict(schema=DESIGN_SCHEMA, static_preflight=deepcopy(native_preflight),
        runtime_expected=with_scoped_expectation(native_preflight["runtime_expected"], config,
            num_envs=args.num_envs), source_receipt=deepcopy(receipt), invocation=invocation,
        contract=dict(file=str(args.contract), sha256=receipt["target_contract_sha256"]),
        selection=dict(file=str(args.integrated_source_selection_file),
            sha256=args.integrated_source_selection_sha256))


def validate_design_hook(config, root, native_preflight, hook, *, stage, num_envs, seed, block_index):
    """Re-authenticate the complete admission and independently resolved design."""
    from . import integrated_production as production
    if (not isinstance(hook, dict) or hook.get("schema") != DESIGN_SCHEMA
            or config.get("evaluation_exact_cases") or not config.get("integrated_production")
            or not config.get("integrated_runtime_candidate")):
        raise ValueError("Integrated worker requires its explicit training design admission")
    if hook.get("static_preflight") != native_preflight:
        raise ValueError("Integrated worker independent static preflight differs")
    invocation = deepcopy(hook["invocation"])
    for name in ("contract", "resume", "transfer", "run_dir", "integrated_source_selection_file"):
        if invocation.get(name) is not None:
            invocation[name] = Path(invocation[name])
    args = SimpleNamespace(**invocation)
    if (args.stage, args.num_envs, args.seed, args.block_index) != (stage, num_envs, seed, block_index):
        raise ValueError("Integrated worker design invocation differs from native construction")
    static = _read(hook["contract"]["file"], hook["contract"]["sha256"])
    _read(hook["selection"]["file"], hook["selection"]["sha256"])
    if (Path(hook["contract"]["file"]) != args.contract
            or Path(hook["selection"]["file"]) != args.integrated_source_selection_file
            or hook["selection"]["sha256"] != args.integrated_source_selection_sha256):
        raise ValueError("Integrated worker design source/contract binding differs")
    receipt = production.authenticate_invocation(static, args, root)
    effective = production.attach_runtime(static, receipt, root)
    if receipt != hook["source_receipt"] or effective != config:
        raise ValueError("Integrated worker full admission/runtime differs from its authenticated source")
    audit = config["integrated_runtime_candidate"]
    if (receipt["selection"] != audit["learning_source"]
            or receipt["component_clocks_at_source"] != audit["clocks_at_source"]
            or receipt["selection"]["domain"] != config["task_campaign"]["domain"]):
        raise ValueError("Integrated worker source selection/component clocks differ")
    resolved = with_scoped_expectation(native_preflight["runtime_expected"], config, num_envs=num_envs)
    if hook.get("runtime_expected") != resolved:
        raise ValueError("Integrated worker independent scoped design differs")
    return dict(deepcopy(native_preflight), runtime_expected=resolved)


def critic_warmup_remaining(receipt, infos):
    """Cross-domain adaptation is once; own resume consumes only its saved remainder."""
    selection = receipt["selection"]
    remaining = selection["critic_warmup_remaining"]
    if type(remaining) is not int or not 0 <= remaining <= 10:
        raise ValueError("Integrated worker critic adaptation must be a bounded integer")
    if selection["mode"] == "resume_joint_own":
        if infos.get("critic_warmup_remaining") != remaining:
            raise ValueError("Integrated own resume changed its sealed critic remainder")
    elif selection["mode"] not in ("rebase_from_normal_REC", "rebase_from_authorized_REC", "inherit_completed_joint_parent") or remaining != 10:
        raise ValueError("Integrated first-domain entry requires exactly ten critic updates")
    return remaining


def learning_metadata(receipt, task_actor_update, *, completed=False, successful_updates=None):
    """One shared value for infos, per-file seal, progress and completion."""
    from .integrated_production import checkpoint_metadata
    allowance = receipt["allowance"]
    delta = task_actor_update - receipt["selection"]["fresh_domain_actor_updates"]
    complete = bool(completed and successful_updates == allowance["requested_total_updates"]
        and delta == allowance["requested_actor_updates"]
        and (task_actor_update == receipt["actor_endpoint"]
             or allowance["requested_actor_updates"] == allowance["maximum_actor_updates"]))
    return checkpoint_metadata(receipt, task_actor_update=task_actor_update, domain_complete=complete)
