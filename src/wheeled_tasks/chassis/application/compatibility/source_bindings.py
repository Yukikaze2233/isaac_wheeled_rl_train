"""Source bindings retained for historical course and checkpoint readers."""
import hashlib
import json
from pathlib import Path

from wheeled_tasks.chassis.application.training.learning_configuration import validate_entropy_coef
from wheeled_tasks.chassis.source_layout import REPOSITORY_ROOT as ROOT, relative_module_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonstandard_start_source_files(config, root=ROOT, *, require_release_stop=True):
    """Audit the exact pose bank and the ordinary birth/release teaching modules."""
    if "nonstandard_start" not in config:
        return []
    from wheeled_tasks.chassis.tasks.recovery.nonstandard_start import validate_nonstandard_start
    from wheeled_tasks.chassis.tasks.recovery.release_stop import validate_release_stop
    settings = config["nonstandard_start"]
    validate_nonstandard_start(settings)
    if require_release_stop or "release_stop" in config:
        validate_release_stop(config.get("release_stop"))
    if (settings["asset_manifest_sha256"] != config["asset_manifest_sha256"]
            or digest(Path(root) / settings["profiles_file"]) != settings["profiles_sha256"]):
        raise ValueError("Nonstandard birth source audit changed its frozen asset or pose bank")
    return ["src/wheeled_tasks/chassis/tasks/recovery/nonstandard_start.py",
            "src/wheeled_tasks/chassis/application/compatibility/nonstandard_start_revision.py",
            "src/wheeled_tasks/chassis/tasks/recovery/release_stop.py",
            "src/wheeled_algo/recovery_rollout.py", settings["profiles_file"]]


def scut_curriculum_source_files(config, root=ROOT):
    """Audit exact teaching recipes and their separately bound optional revisions."""
    meta = config.get("task_campaign", {})
    if ("small_stair_revision_migration" in meta
            or "terrain_runout_revision_migration" in meta or "terrain_route_runout" in config):
        if "small_stair_revision_migration" in meta:
            from wheeled_tasks.chassis.application.compatibility import small_stair_revision as recipe
        else:
            from wheeled_tasks.chassis.application.compatibility import terrain_runout_revision as recipe

        source = recipe.read_bound(recipe.SOURCE_WORKER_FILE, recipe.SOURCE_WORKER_SHA, root)
        recipe.validate_worker_revision(source, config, root=root)
        recipe.validate_frozen_workers(root=root)
        exact = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
        if (exact(json.loads((Path(root) / recipe.PLAN_FILE).read_text())) != exact(recipe.revised_plan(root=root))
                or exact(json.loads((Path(root) / recipe.MANIFEST_FILE).read_text())) != exact(recipe.revised_manifest(root=root))):
            raise ValueError("Terrain source audit differs from its frozen plan or manifest")
        inherited = scut_curriculum_source_files(source, root)
        return list(dict.fromkeys(inherited + [
            "src/wheeled_tasks/chassis/application/compatibility/terrain_runout.py",
            "src/wheeled_tasks/chassis/application/compatibility/terrain_runout_revision.py",
            *(["src/wheeled_tasks/chassis/application/compatibility/small_stair_revision.py"]
               if "small_stair_revision_migration" in meta else []),
            "src/wheeled_tasks/chassis/application/compatibility/terrain_runout_runtime.py",
            "src/wheeled_tasks/chassis/tasks/episodes.py", "src/wheeled_tasks/chassis/application/environment.py",
            "src/wheeled_tasks/chassis/application/configuration/preflight.py", "src/wheeled_tasks/chassis/application/evaluation/evaluation.py",
            "src/wheeled_tasks/chassis/application/compatibility/task_campaign.py", "src/wheeled_tasks/chassis/application/training/train.py",
            "scripts/run_task_campaign.py", "src/wheeled_tasks/chassis/application/runtime/remote_job.py",
            recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE, recipe.SOURCE_DESIGN_FILE,
            recipe.OWN_SOURCE_WORKER_FILE,
            recipe.PLAN_FILE, recipe.WORKER_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE]))
    if "entropy_revision_migration" in meta:
        from wheeled_tasks.chassis.application.compatibility import recovery_entropy_revision as recipe
        if (meta.get("domain") != "recovery"
                or meta["entropy_revision_migration"] != recipe.revision_binding()):
            raise ValueError("Recovery entropy source audit differs from its exact revision binding")
        validate_entropy_coef(meta["entropy_revision_migration"]["entropy_coef"])
        validate_entropy_coef(config.get("entropy_coef"))
        canonical = recipe.canonical_worker(root=root)
        if config != canonical:
            raise ValueError("Recovery entropy source audit differs from its full canonical worker")
        recipe.validate_frozen_workers(root=root)
        if (json.loads((Path(root) / recipe.PLAN_FILE).read_text()) != recipe.revised_plan(root=root)
                or json.loads((Path(root) / recipe.MANIFEST_FILE).read_text()) != recipe.revised_manifest(root=root)):
            raise ValueError("Recovery entropy source audit differs from its frozen plan or manifest")
        source = recipe.read_bound(recipe.SOURCE_WORKER_FILE, recipe.SOURCE_WORKER_SHA, root)
        inherited = scut_curriculum_source_files(source, root)
        return list(dict.fromkeys(inherited + [
            "src/wheeled_tasks/chassis/application/compatibility/recovery_entropy_revision.py",
            recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE, recipe.SOURCE_DESIGN_FILE,
            recipe.PLAN_FILE, recipe.WORKER_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE]))
    binding = meta.get("revision_migration", {})
    schema = binding.get("schema")
    if schema not in ("task-worker-revision-v17", "task-worker-revision-v18", "task-worker-revision-v19", "task-worker-revision-v20"):
        return []
    if schema == "task-worker-revision-v20":
        from wheeled_tasks.chassis.application.compatibility import recovery_script_revision as recipe
        canonical = recipe.canonical_worker(root=root)
        if (meta.get("domain") != "recovery"
                or binding != recipe.revision_binding()
                or config.get("design_preflight") != canonical["design_preflight"]):
            raise ValueError("Recovery script source audit differs from its exact revision/design binding")
        source = json.loads((Path(root) / recipe.SOURCE_WORKER_FILE).read_text())
        inherited = scut_curriculum_source_files(source, root)
        return list(dict.fromkeys(inherited + [
            "src/wheeled_tasks/chassis/application/compatibility/recovery_script_revision.py",
            "src/wheeled_tasks/chassis/tasks/rewards/rewards.py", "src/wheeled_tasks/chassis/application/environment.py",
            "src/wheeled_tasks/chassis/application/runtime/recovery_training.py", "src/wheeled_tasks/chassis/tasks/recovery/recovery.py",
            "src/wheeled_tasks/chassis/application/configuration/preflight.py",
            recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE,
            recipe.SOURCE_DESIGN_FILE, recipe.DESIGN_FILE]))
    if schema == "task-worker-revision-v19":
        from wheeled_tasks.chassis.application.compatibility import recovery_quality_revision as recipe
        canonical = recipe.canonical_worker(root=root)
        if (meta.get("domain") != "recovery"
                or binding != recipe.revision_binding()
                or config.get("design_preflight") != canonical["design_preflight"]):
            raise ValueError("Recovery quality source audit differs from its exact revision/design binding")
        source = json.loads((Path(root) / recipe.SOURCE_WORKER_FILE).read_text())
        inherited = scut_curriculum_source_files(source, root)
        return list(dict.fromkeys(inherited + [
            "src/wheeled_tasks/chassis/application/compatibility/recovery_quality_revision.py",
            "src/wheeled_tasks/chassis/tasks/rewards/rewards.py", "src/wheeled_tasks/chassis/application/environment.py",
            "src/wheeled_tasks/chassis/application/runtime/recovery_training.py", "src/wheeled_tasks/chassis/tasks/recovery/recovery.py",
            "src/wheeled_tasks/chassis/application/configuration/preflight.py",
            recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE,
            recipe.SOURCE_DESIGN_FILE, recipe.DESIGN_FILE]))
    from wheeled_tasks.chassis.application.compatibility import curriculum_revision as scut_recipe
    if schema == "task-worker-revision-v18":
        from wheeled_tasks.chassis.application.compatibility import ordinary_posture_revision as recipe
    else:
        recipe = scut_recipe
    domain = meta.get("domain")
    canonical = recipe.canonical_worker(domain, root)
    if binding != canonical["task_campaign"]["revision_migration"]:
        raise ValueError("reference teaching source audit differs from its exact revision binding")
    sources = ["src/wheeled_tasks/chassis/application/compatibility/curriculum_revision.py",
        "src/wheeled_tasks/chassis/application/compatibility/base_tracking.py",
        "src/wheeled_tasks/chassis/application/compatibility/locomotion_quality_revision.py",
        "src/wheeled_tasks/chassis/application/compatibility/locomotion_revision.py",
        "src/wheeled_tasks/chassis/application/compatibility/nonstandard_start_revision.py",
        binding["source_contract_file"]]
    source = json.loads((Path(root) / binding["source_contract_file"]).read_text())
    sources.append(source["design_preflight"]["file"])
    if schema == "task-worker-revision-v18":
        if config.get("design_preflight") != canonical["design_preflight"]:
            raise ValueError("Ordinary posture source audit differs from its independent design binding")
        sources.extend(["src/wheeled_tasks/chassis/tasks/terrain/ordinary_posture_curriculum.py",
            "src/wheeled_tasks/chassis/application/compatibility/ordinary_posture_revision.py",
            recipe.SOURCE_PLAN_FILE, canonical["design_preflight"]["file"]])
        if domain in scut_recipe.DOMAINS_TO_REVISE:
            _, _, inherited_worker, inherited_design, *_ = scut_recipe._recipe(domain)
            sources.extend((inherited_worker, inherited_design))
    if domain == "recovery":
        sources.extend("src/wheeled_tasks/chassis/" + name + ".py" for name in
            ("recovery_prefix_revision", "recovery_prefix_curriculum", "recovery_curriculum"))
    elif domain == "terrain":
        sources.extend("src/wheeled_tasks/chassis/" + name + ".py" for name in
            ("scut_terrain_revision", "terrain_rehearsal_revision"))
    elif domain in ("jump", "stepjump"):
        sources.extend("src/wheeled_tasks/chassis/" + name + ".py" for name in
            ("scut_jump_revision", "jump_landing_revision", "jump_landing"))
    elif domain != "speed":
        raise ValueError("reference teaching source audit requires a named revised task domain")
    sources.extend(nonstandard_start_source_files(config, root,
        require_release_stop=domain in ("speed", "terrain")))
    return list(dict.fromkeys(sources))
