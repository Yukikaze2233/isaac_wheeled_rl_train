"""Explicit joint continuation through the original campaign account and worker.

The static graph is native sources -> recipe/designs -> workers -> lineage ->
campaign -> migration receipt. Checkpoint selections are separate run artifacts;
no static file hashes itself or invents a future REC endpoint. Importing this
module does not change a process, an account, a checkpoint or a native default.
"""
from copy import deepcopy
import json
from pathlib import Path

from . import integrated_lineage as lineage
from . import integrated_runtime as runtime
from . import integrated_sampling as sampling
from . import scut_reuse_curriculum as reuse
from .task_campaign import CAMPAIGN_ID, DOMAINS, account_updates, update_attempt

REVISION = "v6-task-campaign-v24"
WORKER_SCHEMA = "v6-integrated-static-worker-v1"
RECIPE_SCHEMA = "v6-integrated-static-recipe-v1"
CAMPAIGN_SCHEMA = "v6-integrated-campaign-v1"
ADMISSION_SCHEMA = "v6-integrated-production-admission-v1"
LINEAGE_ID = "source026-joint-v1"
SOURCE_PLAN_FILE = "contracts/v6_task_campaign_v21.json"
SOURCE_PLAN_SHA = "119c77c21da31238ac1af285d0e8df1c627816b0d9645fa099899a6496f11755"
PLAN_FILE = "contracts/v6_task_campaign_v24.json"
RECIPE_FILE = "contracts/v6_integrated_runtime_recipe_v24.json"
LINEAGE_FILE = "contracts/v6_integrated_lineage_v24.json"
MANIFEST_FILE = "contracts/v6_task_campaign_migration_v24.json"
SCUT_DIRECTORY = "references/scut_curriculum_b8ff79f3"
DEPLOY_SOURCE_ROOT = "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_integrated_v24_20261008"
SCUT_MANIFEST_SHA = "d7b280bbb024474efc8df11114a5816e1dcf328531d9049bdea8d8146a28c5e0"
_STATIC_RECIPE_CACHE = {}


def worker_file(domain):
    if domain not in dict(lineage.CHAIN):
        raise ValueError("Unknown integrated worker")
    return f"contracts/v6_task_campaign_{domain}_worker_v24.json"


def design_file(domain):
    worker_file(domain)
    return f"contracts/v6_task_campaign_{domain}_expectations_v24.json"


def _read(root, file, sha):
    return lineage._json(file, sha, root)[1]


def _bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _put(root, file, value):
    path = Path(root) / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value))


def _native_meta(source, domain, parent_file, parent_sha):
    # Old revision markers authorize only old recipes. Retain ordinary worker
    # invocation semantics, never carry those markers into the joint segment.
    keys = ("campaign_id", "domain", "updates", "clock", "learning_audit",
            "evaluation_interval", "initialization", "parent_candidate_scope", "execution_mode")
    meta = {k: deepcopy(source["task_campaign"][k]) for k in keys}
    meta.update(source_contract_file=parent_file, source_contract_sha256=parent_sha,
                worker_revision_id=f"v6-task-campaign-{domain}-worker-v24",
                revision_migration=dict(schema="task-worker-integrated-revision-v24"))
    return meta


def _training_recipe(domain, root, sources):
    """Reviewed source composition, with no proposed learning-state counters."""
    components = ["recovery", *runtime.STAGES[:runtime.STAGES.index(domain) + 1]]
    terrain_bindings = ({name: dict(override="low_speed_stair_for_rm", vx_interval=0)
                         for name in runtime.SMALL_STAIR_NAMES} if "terrain" in components else {})
    candidate = reuse.apply(sampling.apply(sources[domain], root, count=4096),
                            Path(root) / SCUT_DIRECTORY, terrain_bindings=terrain_bindings)
    # These zeros validate static schemas only; no clock document is produced.
    settings = runtime._component_runtime(sources, dict(component_actor_updates={c: 0 for c in components}))
    recovery = deepcopy(sources["recovery"]["recovery_training"])
    prefix = recovery["prefix_curriculum"]
    prefix.update(origin_actor_update=0,
        keyframes=[dict(actor_update=0, probabilities=deepcopy(prefix["keyframes"][-1]["probabilities"]))],
        release_schedule=[dict(actor_update=0, seconds=prefix["release_schedule"][-1]["seconds"])])
    from .recovery_prefix_curriculum import validate_prefix_curriculum
    from .ordinary_posture_curriculum import posture_settings
    from .nonstandard_start import nonstandard_start_settings
    from .locomotion_difficulty import difficulty_settings
    validate_prefix_curriculum(prefix)
    candidate.update(recovery_training=recovery,
        recovery_hold_cost=deepcopy(sources["recovery"]["recovery_hold_cost"]),
        entropy_coef=sources["recovery"]["entropy_coef"],
        ordinary_posture_curriculum=posture_settings(0), nonstandard_start=nonstandard_start_settings(0),
        locomotion_difficulty=difficulty_settings(0), release_stop=deepcopy(sources["speed"]["release_stop"]),
        episode_seconds=max(sources[c]["episode_seconds"] for c in components))
    if "terrain" in components:
        candidate["terrain_route_runout"] = deepcopy(sources["terrain"]["terrain_route_runout"])
    if "jump" in components:
        candidate["jump_landing"] = deepcopy(sources["jump"]["jump_landing"])
        candidate["jump_cycle_reset_on_accepted"] = sources["jump"]["jump_cycle_reset_on_accepted"]
    audit = dict(schema=runtime.SCHEMA, domain=domain, role="train", runtime_ready=False,
        qualification=None, PPO_updates=0, static_recipe_preflight=True, component_runtime=settings,
        row_ownership=runtime._row_ownership(candidate, sources),
        source_bindings=deepcopy(candidate["integrated_sampling_candidate"]["source_bindings"]),
        scut_source_bindings={p: deepcopy(v["files"]) for p, v in reuse.read_sources(Path(root) / SCUT_DIRECTORY).items()},
        scut_terrain_bindings=terrain_bindings, referenced_files=runtime._references(root, sources),
        production_integration_audit=runtime._pin(root, *runtime.AUDIT),
        recovery_prefix_strategy="legal_single_keyframe_frozen_mature_source_endpoint",
        shared_signal_delay_clock="fresh_joint_actor_updates")
    patch = {k: deepcopy(v) for k, v in candidate.items() if sources[domain].get(k) != v}
    # Parent/worker metadata, exams and physical/control/FSM fields stay outside
    # the training patch and retain the explicitly frozen worker baseline.
    forbidden = set(lineage.PHYSICAL_KEYS) | {"task_campaign", "design_preflight", "evaluation",
        "stages", "step_jump_task", "command_transport", "asset_directory", "enabled_stages"}
    if forbidden & patch.keys():
        raise ValueError("Integrated training patch changes a native baseline or fixed examination")
    return dict(patch=patch, runtime=audit)


def revised_recipe(root):
    sources = sampling.load_sources(root)
    references = runtime._references(root, sources)
    archive = _read(root, str(Path(SCUT_DIRECTORY) / "source_manifest.json"), SCUT_MANIFEST_SHA)
    archive_pins = {}
    for file, expected in archive["files"].items():
        relative = str(Path(SCUT_DIRECTORY) / file)
        pin = runtime._pin(root, relative, expected["sha256"])
        if (Path(root) / relative).stat().st_size != expected["bytes"]:
            raise ValueError("Vendored SCUT source/notices byte count differs")
        archive_pins[relative] = dict(pin, bytes=expected["bytes"])
    runtime._pin(root, *runtime.AUDIT)
    _read(root, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA)
    # Recheck every pinned byte input on each admission. Cache only the pure
    # expensive composition; changed source bytes never reuse a cached result.
    fingerprint = (str(Path(root).resolve()), SOURCE_PLAN_FILE, SOURCE_PLAN_SHA,
                   tuple((file, pin["sha256"]) for file, pin in references.items()),
                   tuple((file, pin["sha256"]) for file, pin in archive_pins.items()))
    if fingerprint in _STATIC_RECIPE_CACHE:
        return deepcopy(_STATIC_RECIPE_CACHE[fingerprint])
    result = dict(schema=RECIPE_SCHEMA, lineage_id=LINEAGE_ID, train_only=True,
        source_plan=dict(file=SOURCE_PLAN_FILE, sha256=SOURCE_PLAN_SHA),
        scut_directory=SCUT_DIRECTORY,
        scut_archive_manifest=dict(file=str(Path(SCUT_DIRECTORY) / "source_manifest.json"), sha256=SCUT_MANIFEST_SHA),
        scut_archive_sources=archive_pins,
        domains={d: _training_recipe(d, root, sources) for d, _ in lineage.CHAIN},
        freeze_ready=False, qualification=None,
        pending=["actual normal REC final admission", "native startup on admitted source",
                 "complete source inventory and normal-boundary owner reconciliation"])
    _STATIC_RECIPE_CACHE[fingerprint] = deepcopy(result)
    return result


def _recipe(root):
    path = Path(root) / RECIPE_FILE
    value = json.loads(path.read_text())
    if value != revised_recipe(root):
        raise ValueError("Static joint recipe differs from its exact native/SCUT source composition")
    return value


def _base_worker(domain, root):
    sources = sampling.load_sources(root)
    source = sources[domain]
    parent_domain = dict(lineage.CHAIN)[domain]
    parent_file = sampling.SOURCE_BINDINGS["recovery"][0] if parent_domain == "recovery" else worker_file(parent_domain)
    result = deepcopy(source)
    result["task_campaign"] = _native_meta(source, domain, parent_file, lineage.digest(Path(root) / parent_file))
    result["integrated_production"] = dict(schema=WORKER_SCHEMA, lineage_id=LINEAGE_ID, domain=domain,
        recipe_file=RECIPE_FILE, recipe_sha256=lineage.digest(Path(root) / RECIPE_FILE),
        source_worker_file=sampling.SOURCE_BINDINGS[domain][0], source_worker_sha256=sampling.SOURCE_BINDINGS[domain][1])
    return result


def _expand(config, root, recipe):
    domain = config["integrated_production"]["domain"]
    effective = deepcopy(config)
    effective.update(deepcopy(recipe["domains"][domain]["patch"]))
    effective["integrated_runtime_candidate"] = deepcopy(recipe["domains"][domain]["runtime"])
    return effective


def revised_design(domain, root):
    """Freeze expected source-derived values; never use observed native state."""
    sources = sampling.load_sources(root)
    source = sources[domain]
    binding = source["design_preflight"]
    design = _read(root, binding["file"], binding["sha256"])
    base = _base_worker(domain, root)
    effective = _expand(base, root, _recipe(root))
    for role, role_row in design["stages"][binding["stage"]]["roles"].items():
        if role != "train" and "task_campaign" in role_row["contract"]:
            # Examinations retain their exact original inputs and criteria.
            # Their provenance names the new static worker's parent identity.
            role_row["contract"]["task_campaign"] = deepcopy(base["task_campaign"])
    row = design["stages"][binding["stage"]]["roles"]["train"]
    from .preflight import _at
    for key in row["contract"]:
        row["contract"][key] = deepcopy(_at(effective, key))
    for key in ("recovery_hold_cost", "release_stop", "locomotion_difficulty", "terrain_route_runout",
                "jump_landing", "jump_cycle_reset_on_accepted", "entropy_coef"):
        if key in effective:
            row["contract"][key] = deepcopy(effective[key])
    expected = row["runtime"]
    from .recovery_prefix_curriculum import runtime_expectation as prefix_expectation
    from .ordinary_posture_curriculum import runtime_expectation as posture_expectation
    from .nonstandard_start import runtime_expectation as birth_expectation
    from .release_stop import release_stop_runtime_expectation
    from .recovery_quality_revision import HOLD_GATE_BINDING
    expected.update(recovery_prefix_curriculum=prefix_expectation(effective["recovery_training"]["prefix_curriculum"], enabled=True),
        ordinary_posture_curriculum=posture_expectation(effective["ordinary_posture_curriculum"]),
        nonstandard_start=birth_expectation(effective["nonstandard_start"], effective["policy_dt"]),
        release_stop=release_stop_runtime_expectation(effective["release_stop"], effective["policy_dt"]),
        locomotion_difficulty=dict(enabled=True, settings=deepcopy(effective["locomotion_difficulty"])),
        recovery_hold_cost=deepcopy(effective["recovery_hold_cost"]),
        recovery_hold_cost_gate_binding=deepcopy(HOLD_GATE_BINDING))
    expected["modules"].update(recovery_training=True, recovery_prefix_curriculum=True,
        recovery_hold_cost=True, ordinary_posture_curriculum=True, nonstandard_start=True,
        release_stop=True)
    if "terrain_route_runout" in effective:
        from .terrain_runout_runtime import runtime_expectation as terrain_expectation
        expected["terrain_route_runout"] = terrain_expectation(effective, num_envs=4096)
        expected["modules"]["terrain_route_runout"] = expected["terrain_route_runout"]["enabled"]
    if "jump_landing" in effective:
        from .jump_landing_revision import runtime_expectation as jump_expectation
        expected["jump_landing"] = jump_expectation(effective)
        expected["modules"]["jump_landing"] = effective["jump_landing"]["enabled"]
    if "jump_cycle_reset_on_accepted" in effective:
        expected["jump_cycle_reset_on_accepted"] = effective["jump_cycle_reset_on_accepted"]
    design["revision_migration"] = dict(schema="task-worker-integrated-revision-v24", train_only=True,
        source_design_file=binding["file"], source_design_sha256=binding["sha256"],
        runtime_clocks="separate_authenticated_source_receipt")
    return design


def canonical_worker(domain, root):
    design = json.loads((Path(root) / design_file(domain)).read_text())
    if design != revised_design(domain, root):
        raise ValueError("Joint worker design differs from the source-derived train-only declaration")
    result = _base_worker(domain, root)
    result["design_preflight"] = dict(file=design_file(domain), sha256=lineage.digest(Path(root) / design_file(domain)),
                                    stage=sampling.load_sources(root)[domain]["design_preflight"]["stage"])
    return result


def validate_static_worker(config, root):
    meta = config.get("integrated_production", {})
    if (meta.get("schema") != WORKER_SCHEMA or meta.get("lineage_id") != LINEAGE_ID
            or meta.get("domain") not in dict(lineage.CHAIN)):
        raise ValueError("Joint worker requires its explicit frozen static identity")
    _read(root, meta["recipe_file"], meta["recipe_sha256"])
    if config != canonical_worker(meta["domain"], root):
        raise ValueError("Joint worker changes fields outside its canonical static recipe")
    return meta["domain"]


def static_training_config(config, root):
    """Expand the train role before design checks; no actual clocks are invented."""
    if not config.get("integrated_production"):
        return deepcopy(config)
    validate_static_worker(config, root)
    return _expand(config, root, _recipe(root))


def fixed_evaluation_config(config, root):
    """Project the exact old native exam, including its predecessor catalogue."""
    domain = config.get("integrated_production", {}).get("domain")
    if domain not in dict(lineage.CHAIN):
        raise ValueError("Fixed joint examination lacks its static worker identity")
    canonical = canonical_worker(domain, root)
    baseline = deepcopy(config)
    baseline["evaluation"] = deepcopy(canonical["evaluation"])
    validate_static_worker(baseline, root)
    from .evaluation import select_evaluation_cases
    names = [case["name"] for case in config["evaluation"]["cases"]]
    projection = select_evaluation_cases(canonical["evaluation"], names)
    projection["seed"] = config["evaluation"]["seed"]
    if projection != config["evaluation"] or projection["seed"] not in (290929, 290930):
        raise ValueError("Fixed joint examination changed original cases, criteria or seed")
    sources = sampling.load_sources(root)
    source = deepcopy(sources[domain])
    source["evaluation"] = deepcopy(config["evaluation"])
    from .evaluation import fixed_suite_contract
    # The native source keeps its original revision marker, so its existing
    # fixed-case projection remains authoritative. No training recipe is read
    # as an exam definition and no observed behaviour becomes a criterion.
    result = fixed_suite_contract(source)
    result["task_campaign"] = deepcopy(config["task_campaign"])
    result["integrated_production"] = deepcopy(config["integrated_production"])
    result["design_preflight"] = deepcopy(config["design_preflight"])
    return result


def revised_plan(root):
    source = _read(root, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA)
    plan = deepcopy(source)
    plan.update(contract_id=REVISION, scope="single_policy_joint_continuation_from_normal_REC")
    for row in plan["domains"]:
        domain = row["name"]
        if domain not in dict(lineage.CHAIN):
            continue
        config = canonical_worker(domain, root)
        row.update(contract=worker_file(domain), sha256=lineage.digest(Path(root) / worker_file(domain)),
                   parent_domain=dict(lineage.CHAIN)[domain])
        effective = _expand(config, root, _recipe(root))
        row["sampling_pools"] = [dict(name=p, fraction=f, skills=[n for n, v in effective["behavior_pool_membership"].items() if v == p])
                                 for p, f in effective["behavior_pool_fractions"].items()]
    plan["integrated_continuation"] = dict(schema=CAMPAIGN_SCHEMA, lineage_id=LINEAGE_ID,
        lineage_manifest=dict(file=LINEAGE_FILE, sha256=lineage.digest(Path(root) / LINEAGE_FILE)),
        recipe=dict(file=RECIPE_FILE, sha256=lineage.digest(Path(root) / RECIPE_FILE)),
        actor_endpoint_clock="fresh_joint_domain_actor_updates", legacy_completion="archive_not_joint_progress")
    plan["revision_migration"] = dict(manifest_file=MANIFEST_FILE, source_plan_sha256=SOURCE_PLAN_SHA,
                                     run_directory=source["revision_migration"]["run_directory"])
    return plan


def revised_manifest(root):
    plan = revised_plan(root)
    return dict(schema="task-plan-revision-v24", source_plan_file=SOURCE_PLAN_FILE,
        source_plan_sha256=SOURCE_PLAN_SHA, target_plan_file=PLAN_FILE,
        target_plan_sha256=lineage.digest(Path(root) / PLAN_FILE),
        run_directory=plan["revision_migration"]["run_directory"],
        lineage_manifest=deepcopy(plan["integrated_continuation"]["lineage_manifest"]),
        consumption="preserved_no_refund", legacy_completion="archive_not_joint_progress",
        learning_chain=["recovery_normal_15000", "speed", "terrain", "jump", "stepjump"],
        automatic_promotion=False, qualification=None)


def write_bundle(root, *, source_root=DEPLOY_SOURCE_ROOT, run_root=None):
    """Generate named static artifacts locally; this never freezes or deploys them."""
    root = Path(root)
    source = _read(root, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA)
    run_root = run_root or source["revision_migration"]["run_directory"]
    _put(root, RECIPE_FILE, revised_recipe(root))
    for domain, _ in lineage.CHAIN:
        _put(root, design_file(domain), revised_design(domain, root))
        _put(root, worker_file(domain), canonical_worker(domain, root))
    # Bind the unchanged old accounting plan here, rather than the new plan:
    # new-plan -> lineage -> new-plan would be a SHA cycle.
    manifest = dict(schema=lineage.SCHEMA, lineage_id=LINEAGE_ID,
        source_root=str(source_root), run_root=str(run_root),
        campaign_plan=dict(file=SOURCE_PLAN_FILE, sha256=SOURCE_PLAN_SHA),
        recovery_entry=dict(contract_file=sampling.SOURCE_BINDINGS["recovery"][0],
                            contract_sha256=sampling.SOURCE_BINDINGS["recovery"][1], actor_endpoint=15000),
        domains=[dict(name=d, parent_domain=p, contract_file=worker_file(d),
                      contract_sha256=lineage.digest(root / worker_file(d)), critic_warmup_updates=10) for d, p in lineage.CHAIN])
    _put(root, LINEAGE_FILE, manifest)
    _put(root, PLAN_FILE, revised_plan(root))
    _put(root, MANIFEST_FILE, revised_manifest(root))
    return dict(freeze_ready=False, qualification=None, files=[RECIPE_FILE,
        *[f for d, _ in lineage.CHAIN for f in (design_file(d), worker_file(d))], LINEAGE_FILE, PLAN_FILE, MANIFEST_FILE])


def validate_plan(plan, root):
    if plan.get("contract_id") != REVISION or plan != revised_plan(root):
        raise ValueError("Joint campaign differs from its exact six-domain non-refundable plan")
    pin = plan["integrated_continuation"]["lineage_manifest"]
    frozen = _read(root, pin["file"], pin["sha256"])
    if (frozen.get("schema") != lineage.SCHEMA or frozen.get("lineage_id") != LINEAGE_ID
            or frozen.get("campaign_plan") != dict(file=SOURCE_PLAN_FILE, sha256=SOURCE_PLAN_SHA)
            or tuple((r["name"], r["parent_domain"]) for r in frozen["domains"]) != lineage.CHAIN
            or frozen.get("run_root") != plan["revision_migration"]["run_directory"]):
        raise ValueError("Joint static lineage manifest differs")
    for row in frozen["domains"]:
        if (row["contract_file"] != worker_file(row["name"])
                or row["contract_sha256"] != lineage.digest(Path(root) / row["contract_file"])
                or row["critic_warmup_updates"] != 10):
            raise ValueError("Joint static parent worker binding differs")
    return frozen


def validate_migration(source, target, run_root, root):
    validate_plan(target, root)
    manifest = json.loads((Path(root) / MANIFEST_FILE).read_text())
    if (source != _read(root, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA)
            or manifest != revised_manifest(root)
            or str(Path(run_root).resolve()) != manifest["run_directory"]):
        raise ValueError("Joint migration requires its exact old v21 account and run")
    return deepcopy(manifest)


def binding_for(plan, root, run_root):
    validate_plan(plan, root)
    pin = plan["integrated_continuation"]["lineage_manifest"]
    return lineage.load_plan(Path(root) / pin["file"], pin["sha256"], root, run_root)[0]


def reconcile_records(plan, run_root, records):
    """Read every old/joint reservation without rewriting the authoritative claim."""
    result = deepcopy(records)
    for path in sorted(Path(run_root).glob("*/attempt_*/train/progress.json")):
        result = update_attempt(plan, result, str(path.parent.relative_to(run_root)), path.parents[2].name,
                                json.loads(path.read_text()))
    return result


def prepare_admission(plan, source_root, run_root, *, account_path=None, state_path=None):
    """Read-only normal-boundary admission; run_root is the experiment/train dir."""
    source_root, run_root = Path(source_root), Path(run_root)
    binding = binding_for(plan, source_root, run_root)
    account_path = Path(account_path) if account_path else Path(plan["budget_source_stage"]) / (CAMPAIGN_ID + ".account.json")
    state_path = Path(state_path) if state_path else run_root / "campaign_state.json"
    account = json.loads(account_path.read_text())
    if (account.get("campaign_id") != CAMPAIGN_ID or account.get("run_directory") != str(run_root.resolve())
            or account.get("contract_sha256") not in (SOURCE_PLAN_SHA, lineage.digest(source_root / PLAN_FILE))
            or account.get("rollback_refunds_consumed_updates") is not False):
        raise ValueError("Normal boundary account is not the frozen old or current joint account")
    state = json.loads(state_path.read_text())
    records = reconcile_records(plan, run_root, account["attempts"])
    totals = account_updates(plan, records)
    original = account_updates(plan, account["attempts"])
    if any(account.get(k) != original[k] for k in original):
        raise ValueError("Authoritative account counters differ from its durable attempts")
    rec = state.get("domains", {}).get("recovery", {})
    reference = dict(path=rec.get("candidate_checkpoint"), sha256=rec.get("candidate_checkpoint_sha256"))
    entry = lineage.admit_entry(binding, reference, rec)
    selection = lineage.select_learning_source(binding, "speed", entry)
    endpoint = plan["domain_completion"]["schedule"]["end_actor_updates"]["speed"]
    allowance = lineage.remaining_budget(binding, selection, records)
    allowance = lineage.remaining_budget(binding, selection, records,
        requested_actor_updates=min(endpoint, allowance["maximum_actor_updates"]))
    if not allowance["admitted"]:
        raise ValueError("Old charged speed/global cap leaves no joint actor allowance")
    return dict(schema=ADMISSION_SCHEMA, lineage_manifest=deepcopy(plan["integrated_continuation"]["lineage_manifest"]),
        campaign_plan=dict(file=PLAN_FILE, sha256=lineage.digest(source_root / PLAN_FILE)),
        entry_admission=entry, selection=selection, allowance=allowance,
        component_clocks_at_source=runtime.source_clocks(binding, selection), actor_endpoint=endpoint,
        target_contract_sha256=selection["contract_sha256"],
        account=dict(file=str(account_path.resolve()), sha256=lineage.digest(account_path)),
        state=dict(file=str(state_path.resolve()), sha256=lineage.digest(state_path)),
        reconciled_attempts=records, reconciled_account=totals,
        freeze_ready=False, qualification=None)


def archive_legacy_domains(report, binding, entry):
    """Archive old completion once, including its eval references; never refund it."""
    result = deepcopy(report)
    identity = dict(lineage_id=LINEAGE_ID, plan_sha256=binding.sha256,
                    origin_checkpoint_sha256=entry["checkpoint"]["sha256"])
    existing = result.get("integrated_continuation")
    if existing is not None and existing != identity:
        raise ValueError("Campaign report already belongs to another joint origin")
    for domain, _ in lineage.CHAIN:
        record = result.setdefault("domains", {}).get(domain, {})
        if record.get("integrated_lineage"):
            if record["integrated_lineage"].get("lineage_id") != LINEAGE_ID:
                raise ValueError("Domain report belongs to another joint lineage")
            continue
        history = deepcopy(record.get("legacy_completion_history", []))
        if record and record.get("status") != "joint_domain_pending":
            history.append(deepcopy(record))
        result["domains"][domain] = dict(status="joint_domain_pending", legacy_completion_history=history,
            domain_qualified=None, accepted_stage=None, deployment_checkpoint=None)
    result["integrated_continuation"] = identity
    return result


def _same_saved(left, right):
    """Exact complete native restore state, including all saved RNG families."""
    import torch
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_saved(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_same_saved(a, b) for a, b in zip(left, right, strict=True))
    if type(left).__module__.startswith("numpy"):
        import numpy as np
        return left.dtype == right.dtype and np.array_equal(left, right)
    return left == right


def _normalise_native_save(saved):
    result = deepcopy(saved)
    cursor = result.pop("iter", None)  # RSL OnPolicyRunner's serialization cursor.
    if cursor is not None and (type(cursor) is not int or cursor < 0):
        raise ValueError("Native save cursor is not a nonnegative integer")
    result["infos"]["integrated_lineage"].pop("domain_complete")
    return result, cursor


def _native_duplicates(binding, domain, references, previous_selection):
    """Collapse only proven copies saved by one native attempt, without mtime."""
    import torch
    plan = binding.read()
    row = next(r for r in plan["domains"] if r["name"] == domain)
    groups = {}
    for reference in references:
        checkpoint = lineage._checkpoint(binding, reference, domain, row["contract_sha256"])
        groups.setdefault(checkpoint["seal"]["learning_updates"], []).append((reference, checkpoint))
    unique, aliases = [], {}
    for count, copies in sorted(groups.items()):
        if len(copies) == 1:
            unique.append(copies[0][0])
            continue
        def owner(path):
            return next((p for p in Path(path).parents if p.name == "train"), None)
        if len({owner(cp["path"]) for _, cp in copies}) != 1 or owner(copies[0][1]["path"]) is None:
            raise ValueError("Same-clock joint checkpoints fork across native attempts")
        states = [_normalise_native_save(torch.load(cp["path"], map_location="cpu", weights_only=True)) for _, cp in copies]
        base, cursor = states[0]
        if any(not _same_saved(base, state) or (cursor is None) != (other is None)
               or cursor is not None and abs(cursor-other) > 1 for state, other in states[1:]):
            raise ValueError("Same-clock native saves fork model/Adam/RNG/consumption or actor clocks")
        finals = [(ref, cp) for ref, cp in copies if Path(cp["path"]).name == "model_final.pt"]
        if finals:
            if len(finals) != 1:
                raise ValueError("Same native attempt has multiple final learning states")
            chosen, cp = finals[0]
            if cp["infos"]["integrated_lineage"]["domain_complete"]:
                record = dict(status="joint_domain_finished_unassessed", candidate_checkpoint=cp["path"],
                              candidate_checkpoint_sha256=cp["sha256"])
                lineage._finished(binding, cp, record)
        else:
            if any(cp["infos"]["integrated_lineage"]["domain_complete"] for _, cp in copies):
                raise ValueError("A non-final native snapshot cannot claim terminal completion")
            # The explicit immutable snapshot is preferred to RSL's periodic
            # top-level copy; path order resolves equal native save kinds.
            chosen = min(copies, key=lambda pair: ("checkpoints" not in Path(pair[1]["path"]).parts, pair[1]["path"]))[0]
        unique.append(chosen)
        aliases.update({ref["sha256"]: chosen for ref, _ in copies})
    floor = deepcopy(previous_selection)
    if floor is not None:
        replacement = aliases.get(floor["checkpoint"]["sha256"])
        if replacement is not None and replacement != floor["checkpoint"]:
            # Re-authenticate the durable old floor too. Only the proven native
            # duplicate representation is substituted; no learning clock moves.
            runtime._verified_selection(binding, floor)
            floor["checkpoint"] = deepcopy(replacement)
    return unique, floor


def _initial_learning_state(binding, domain, reference, entry):
    """Authenticate the native zero-work rollback save before excluding it."""
    path = lineage._within(reference["path"], binding.run_root)
    if path.parent.name != "initial_state":
        return False
    if (path.name != "model_final.pt" or path.parent.parent.name != "train"
            or not path.parent.parent.parent.name.startswith("attempt_")
            or path.parent.parent.parent.parent != binding.run_root / domain):
        raise ValueError("Initial learning state is outside its native startup location")
    completion = json.loads(lineage._within(path.parent / "completion.json", binding.run_root).read_text())
    if (completion.get("scope") != "initial_learning_state" or completion.get("status") != "completed"
            or type(completion.get("successful_updates")) is not int or completion["successful_updates"] != 0
            or completion.get("checkpoint_sha256") != reference["sha256"]):
        raise ValueError("Initial learning state lacks its authenticated zero-update completion")
    # This uses the normal native model/Adam/RNG/seal and REC-origin checks.
    # The startup save can have new episode RNG, but cannot claim learned work
    # or replace a committed endpoint from this or another native attempt.
    selection = lineage.select_learning_source(binding, domain, entry, own_candidates=[reference])
    if (type(completion.get("parent_updates")) is not int
            or completion["parent_updates"] != selection["checkpoint_consumed_updates"]
            or selection["domain_complete"]):
        raise ValueError("Initial learning state claims committed or terminal work")
    runtime.source_clocks(binding, selection)
    return True


def select_source(plan, root, run_root, report, *, previous_selection=None):
    """Use explicit same-lineage own seals; leave every legacy checkpoint in place."""
    binding = binding_for(plan, root, run_root)
    rec = report["domains"]["recovery"]
    entry = lineage.admit_entry(binding, dict(path=rec["candidate_checkpoint"], sha256=rec["candidate_checkpoint_sha256"]), rec)
    domain = previous_selection["domain"] if previous_selection else report.get("active_joint_domain")
    if domain not in dict(lineage.CHAIN):
        raise ValueError("Joint source selection requires an explicit active domain")
    references = []
    for path in sorted((Path(run_root) / domain).glob("attempt_*/train/**/*.checkpoint.json")):
        if any(part.endswith(".pending") for part in path.parts):
            continue
        seal = json.loads(path.read_text())
        meta = seal.get("integrated_lineage")
        if meta is None:
            continue  # Retained old branch evidence is never a learning parent.
        if meta.get("lineage_id") != LINEAGE_ID or meta.get("plan_sha256") != binding.sha256:
            raise ValueError("Own checkpoint carries an unexpected joint lineage")
        cp = path.with_suffix("").with_suffix(".pt")
        reference = dict(path=str(cp), sha256=seal["checkpoint_sha256"])
        if _initial_learning_state(binding, domain, reference, entry):
            continue
        references.append(reference)
    references, floor = _native_duplicates(binding, domain, references, previous_selection)
    kwargs = dict(own_candidates=references, previous_selection=floor)
    parent_domain = dict(lineage.CHAIN)[domain]
    if domain != "speed":
        record = report["domains"].get(parent_domain, {})
        kwargs.update(parent=dict(path=record.get("candidate_checkpoint"), sha256=record.get("candidate_checkpoint_sha256")),
                      parent_record=record)
    selection = lineage.select_learning_source(binding, domain, entry, **kwargs)
    return binding, selection


def make_admission(plan, root, run_root, selection, records, *, account_path, state_path, snapshot_directory=None):
    binding = binding_for(plan, root, run_root)
    endpoint = plan["domain_completion"]["schedule"]["end_actor_updates"][selection["domain"]]
    allowance = lineage.remaining_budget(binding, selection, records)
    allowance = lineage.remaining_budget(binding, selection, records,
        requested_actor_updates=min(max(0, endpoint-selection["fresh_domain_actor_updates"]), allowance["maximum_actor_updates"]))
    account_pin = dict(file=str(Path(account_path).resolve()), sha256=lineage.digest(account_path))
    state_pin = dict(file=str(Path(state_path).resolve()), sha256=lineage.digest(state_path))
    snapshot_account, snapshot_state = deepcopy(account_pin), deepcopy(state_pin)
    if snapshot_directory is not None:
        directory = Path(snapshot_directory).resolve()
        if not directory.is_relative_to(Path(run_root).resolve()):
            raise ValueError("Admission snapshot must stay inside this owned campaign run")
        _put(directory, "account_at_admission.json", json.loads(Path(account_path).read_text()))
        _put(directory, "state_at_admission.json", json.loads(Path(state_path).read_text()))
        snapshot_account = dict(file=str(directory / "account_at_admission.json"), sha256=lineage.digest(directory / "account_at_admission.json"))
        snapshot_state = dict(file=str(directory / "state_at_admission.json"), sha256=lineage.digest(directory / "state_at_admission.json"))
    return dict(schema=ADMISSION_SCHEMA, lineage_manifest=deepcopy(plan["integrated_continuation"]["lineage_manifest"]),
        campaign_plan=dict(file=PLAN_FILE, sha256=lineage.digest(Path(root) / PLAN_FILE)),
        entry_admission=deepcopy(selection["entry_admission"]), selection=deepcopy(selection), allowance=allowance,
        component_clocks_at_source=runtime.source_clocks(binding, selection), actor_endpoint=endpoint,
        target_contract_sha256=selection["contract_sha256"],
        account=snapshot_account, state=snapshot_state,
        authoritative_account=account_pin, authoritative_state=state_pin,
        reconciled_attempts=deepcopy(records), reconciled_account=account_updates(plan, records),
        freeze_ready=False, qualification=None)


def authenticate_receipt(config, receipt, root):
    domain = validate_static_worker(config, root)
    if (receipt.get("schema") != ADMISSION_SCHEMA or receipt.get("qualification") is not None
            or receipt.get("freeze_ready") is not False or receipt.get("selection", {}).get("domain") != domain):
        raise ValueError("Joint invocation requires the exact unqualified admission schema")
    pin = receipt["campaign_plan"]
    if pin["file"] != PLAN_FILE:
        raise ValueError("Joint admission campaign file differs")
    plan = _read(root, pin["file"], pin["sha256"])
    manifest = validate_plan(plan, root)
    if receipt["lineage_manifest"] != plan["integrated_continuation"]["lineage_manifest"]:
        raise ValueError("Joint admission lineage manifest differs")
    run_root = Path(manifest["run_root"])
    binding = binding_for(plan, root, run_root)
    selection = runtime._verified_selection(binding, receipt["selection"])
    if (receipt["entry_admission"] != selection["entry_admission"]
            or receipt["target_contract_sha256"] != lineage.digest(Path(root) / worker_file(domain))
            or receipt["target_contract_sha256"] != selection["contract_sha256"]
            or receipt["component_clocks_at_source"] != runtime.source_clocks(binding, selection)):
        raise ValueError("Joint admission source/target/component bindings differ")
    account_pin = receipt["account"]
    expected_claim = Path(plan["budget_source_stage"]) / (CAMPAIGN_ID + ".account.json")
    account_path = Path(account_pin["file"])
    authority = receipt.get("authoritative_account", account_pin)
    if Path(authority["file"]).resolve() != expected_claim.resolve():
        raise ValueError("Joint admission authoritative account path differs")
    if (account_path.resolve() != expected_claim.resolve()
            and not account_path.resolve().is_relative_to(run_root.resolve())):
        raise ValueError("Joint admission snapshot escapes its campaign run")
    if lineage.digest(account_path) != account_pin["sha256"]:
        raise ValueError("Joint admission account snapshot SHA differs")
    ledger = json.loads(account_path.read_text())
    for value in (ledger, json.loads(expected_claim.read_text())):
        if (value.get("campaign_id") != CAMPAIGN_ID or value.get("run_directory") != str(run_root.resolve())
                or value.get("contract_sha256") not in (SOURCE_PLAN_SHA, lineage.digest(Path(root) / PLAN_FILE))
                or value.get("rollback_refunds_consumed_updates") is not False
                or any(value.get(k) != v for k, v in account_updates(plan, value["attempts"]).items())):
            raise ValueError("Joint admission account identity/counters differ")
    records = receipt["reconciled_attempts"]
    if account_updates(plan, records) != receipt["reconciled_account"]:
        raise ValueError("Joint admission would change or refund durable consumption")
    live = reconcile_records(plan, run_root, json.loads(expected_claim.read_text())["attempts"])
    for key, before in records.items():
        after = live.get(key)
        if after is None or after["domain"] != before["domain"] or any(after[k] < before[k] for k in ("consumed_updates", "actor_updates")):
            raise ValueError("Joint live claim lost charged historical work")
    live_totals = account_updates(plan, live)
    if live_totals != receipt["reconciled_account"]:
        raise ValueError("Joint live consumption changed after this allowance was admitted")
    expected = lineage.remaining_budget(binding, selection, records,
        requested_actor_updates=receipt["allowance"]["requested_actor_updates"])
    endpoint = plan["domain_completion"]["schedule"]["end_actor_updates"][domain]
    if (receipt["allowance"] != expected or not expected["admitted"]
            or receipt["actor_endpoint"] != endpoint
            or expected["requested_actor_updates"] != min(max(0, endpoint-selection["fresh_domain_actor_updates"]), expected["maximum_actor_updates"])):
        raise ValueError("Joint invocation allowance or fresh actor endpoint differs")
    state_path = Path(receipt["state"]["file"])
    authority_state = receipt.get("authoritative_state", receipt["state"])
    if (Path(authority_state["file"]).resolve() != (run_root / "campaign_state.json").resolve()
            or not state_path.resolve().is_relative_to(run_root.resolve())
            or lineage.digest(state_path) != receipt["state"]["sha256"]):
        raise ValueError("Joint admission state source differs")
    return deepcopy(receipt)


def authenticate_invocation(config, args, root):
    file, sha = getattr(args, "integrated_source_selection_file", None), getattr(args, "integrated_source_selection_sha256", None)
    if not config.get("integrated_production"):
        if file is not None or sha is not None:
            raise ValueError("Native worker cannot accept an unbound joint admission")
        return None
    if file is None or sha is None:
        raise ValueError("Joint worker requires the separate source selection file/SHA")
    path = Path(file)
    if lineage.digest(path) != lineage._sha(sha):
        raise ValueError("Joint source selection file SHA differs")
    receipt = authenticate_receipt(config, json.loads(path.read_text()), root)
    selection, allowance = receipt["selection"], receipt["allowance"]
    if (not args.resume_task_change or not args.resume or Path(args.resume).resolve() != Path(selection["checkpoint"]["path"]).resolve()
            or args.num_envs != 4096 or args.seed != 290927 or args.block_index != 0
            or args.learning_rate_scale != 1. or args.global_actor_update_offset != 0
            or args.consumed_updates != allowance["consumed_updates"]
            or args.actor_updates_consumed != allowance["actor_updates_consumed"]
            or args.task_actor_update != selection["fresh_domain_actor_updates"]):
        raise ValueError("Joint invocation changes its frozen parent, task clock or account")
    seal = json.loads(Path(args.resume).with_suffix(".checkpoint.json").read_text())
    scene_only = type(getattr(args, "validate_scene", 0)) is int and getattr(args, "validate_scene", 0) > 0
    if (args.stage_actor_update != seal["stage_actor_update"]
            or not scene_only and args.updates != allowance["requested_total_updates"]
            or scene_only and args.updates != 1):
        raise ValueError("Joint invocation changes its stage clock or charged update allowance")
    return receipt


def attach_runtime(config, receipt, root):
    if receipt is None and not config.get("integrated_production"):
        return deepcopy(config)
    receipt = authenticate_receipt(config, receipt, root)
    effective = static_training_config(config, root)
    effective["integrated_runtime_candidate"].pop("static_recipe_preflight")
    effective["integrated_runtime_candidate"].update(learning_source=deepcopy(receipt["selection"]),
        clocks_at_source=deepcopy(receipt["component_clocks_at_source"]))
    return effective


def checkpoint_metadata(receipt, *, task_actor_update, domain_complete=False):
    source = receipt["selection"]
    delta = lineage._integer(task_actor_update, "actual joint domain actor clock") - source["fresh_domain_actor_updates"]
    if delta < 0 or delta > receipt["allowance"]["requested_actor_updates"] or type(domain_complete) is not bool:
        raise ValueError("Joint save metadata exceeds actual admitted actor work")
    if domain_complete and task_actor_update < receipt["actor_endpoint"] and delta != receipt["allowance"]["requested_actor_updates"]:
        raise ValueError("Joint terminal metadata is before its endpoint or charged cap")
    meta = dict(lineage_id=source["lineage_id"], plan_sha256=source["plan_sha256"],
        origin_checkpoint_sha256=source["origin_checkpoint_sha256"], domain=source["domain"],
        joint_actor_updates=source["fresh_joint_actor_updates"] + delta,
        domain_actor_updates=task_actor_update, domain_complete=domain_complete)
    return dict(integrated_lineage=meta,
                integrated_runtime_clocks=runtime.advance_clock_document(receipt["component_clocks_at_source"], delta))


def production_source_files(config, receipt=None, root=None):
    if not config.get("integrated_production"):
        return []
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    domain = validate_static_worker(config, root)
    sources = sampling.load_sources(root)
    references = runtime._references(root, sources)
    return list(dict.fromkeys(["src/wheeled_tasks/chassis/integrated_production.py",
        "src/wheeled_tasks/chassis/integrated_lineage.py", "src/wheeled_tasks/chassis/integrated_runtime.py",
        "src/wheeled_tasks/chassis/integrated_sampling.py", "src/wheeled_tasks/chassis/scut_reuse_curriculum.py",
        "src/wheeled_tasks/chassis/integrated_preflight.py", "src/wheeled_tasks/chassis/task_campaign.py",
        "scripts/run_task_campaign.py", "scripts/train_chassis.py", PLAN_FILE, MANIFEST_FILE, LINEAGE_FILE, RECIPE_FILE,
        *[f for d, _ in lineage.CHAIN for f in (worker_file(d), design_file(d))],
        *[p[0] for p in sampling.SOURCE_BINDINGS.values()], *references,
        *[str(Path(SCUT_DIRECTORY) / name) for name in json.loads((root / SCUT_DIRECTORY / "source_manifest.json").read_text())["files"]],
        str(Path(SCUT_DIRECTORY) / "source_manifest.json"), runtime.AUDIT[0]]))
