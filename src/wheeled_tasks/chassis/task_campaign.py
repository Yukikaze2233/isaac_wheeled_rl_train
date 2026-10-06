"""Frozen task-domain research with one non-refundable campaign account."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

REVISION = "v6-task-campaign-v1"
REVISION_V2 = "v6-task-campaign-v2"
REVISION_V3 = "v6-task-campaign-v3"
REVISION_V4 = "v6-task-campaign-v4"
REVISION_V5 = "v6-task-campaign-v5"
REVISION_V6 = "v6-task-campaign-v6"
REVISION_V7 = "v6-task-campaign-v7"
REVISION_V8 = "v6-task-campaign-v8"
REVISION_V9 = "v6-task-campaign-v9"
REVISION_V10 = "v6-task-campaign-v10"
REVISION_V11 = "v6-task-campaign-v11"
REVISION_V12 = "v6-task-campaign-v12"
REVISION_V13 = "v6-task-campaign-v13"
REVISION_V14 = "v6-task-campaign-v14"
REVISION_V15 = "v6-task-campaign-v15"
REVISION_V16 = "v6-task-campaign-v16"
REVISION_V17 = "v6-task-campaign-v17"
REVISION_V18 = "v6-task-campaign-v18"
REVISION_V19 = "v6-task-campaign-v19"
REVISION_V20 = "v6-task-campaign-v20"
SCHEDULE_REVISIONS = (REVISION_V7, REVISION_V8, REVISION_V9, REVISION_V10, REVISION_V11, REVISION_V12, REVISION_V13, REVISION_V14, REVISION_V15, REVISION_V16, REVISION_V17, REVISION_V18, REVISION_V19, REVISION_V20)
SHORT_SCHEDULE_ENDPOINTS = dict(
    flat=13000, recovery=15000, speed=3563, terrain=1186, jump=15000, stepjump=15000)
SHORT_SCHEDULE_SCOPE = "user_requested_first_pass_actor_budgets_not_SCUT_qualification_iterations"
NONSTANDARD_START_ENDPOINTS = dict(SHORT_SCHEDULE_ENDPOINTS, speed=5063, terrain=2186)
TERRAIN_EXTENSION_ENDPOINTS = dict(NONSTANDARD_START_ENDPOINTS, terrain=15000)
ORDINARY_POSTURE_ENDPOINTS = dict(TERRAIN_EXTENSION_ENDPOINTS, speed=15000)
REVIEW_REVISIONS = (REVISION_V5, REVISION_V6, *SCHEDULE_REVISIONS)
CONTINUOUS_REVISIONS = (REVISION_V2, REVISION_V3, REVISION_V4, *REVIEW_REVISIONS)
CONTINUOUS_MODE = "uninterrupted_cap_then_next"
REVIEW_MODE = "continuous_review_or_cap_then_next"
CAMPAIGN_ID = "v6-task-domains-20261001"
DOMAINS = ("flat", "recovery", "speed", "terrain", "jump", "stepjump")
ROOT = Path(__file__).resolve().parents[3]
RECOVERY_V6_FILE = "contracts/v6_task_campaign_recovery_worker_v6.json"
RECOVERY_V6_SHA256 = "e2a7ecaec1985adac34c280a71b1359a6446fc12c38d9aaa6fb98ac769e4d74c"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(plan):
    resized = plan.get("contract_id") in REVIEW_REVISIONS
    if (plan.get("contract_id") not in (REVISION, *CONTINUOUS_REVISIONS) or plan.get("campaign_id") != CAMPAIGN_ID
            or plan.get("historical_consumed_updates") != 1000
            or plan.get("historical_actor_consumed_updates") != 980
            or plan.get("total_ceiling") != 121000 or plan.get("target_num_envs") != (4096 if resized else 20000)
            or plan.get("seed") != 290927 or plan.get("evaluation_seeds") != [290929, 290930]
            or plan.get("rollback_refunds_consumed_updates", False) is not False
            or plan.get("automatic_promotion", False) is not False):
        raise ValueError("Task campaign changes its frozen budget, identity or research boundary")
    domains = plan.get("domains", [])
    if tuple(row["name"] for row in domains) != DOMAINS:
        raise ValueError("Task campaign domain order differs")
    seen = {"A"}
    for row in domains:
        if (row["updates"] != 20000 or row["parent_domain"] not in seen
                or not row["required_case_names"]
                or len(set(row["required_case_names"])) != len(row["required_case_names"])):
            raise ValueError("Each domain needs a 20000-update cap, earlier parent and explicit unique gates")
        seen.add(row["name"])
    if plan["historical_consumed_updates"] + sum(r["updates"] for r in domains) != plan["total_ceiling"]:
        raise ValueError("Task caps disagree with global consumption ceiling")
    if plan["contract_id"] in CONTINUOUS_REVISIONS:
        completion = plan.get("domain_completion", {})
        if (completion.get("execution_mode") != (REVIEW_MODE if resized else CONTINUOUS_MODE)
                or completion.get("assessment_interval_actor_updates") != 0
                or completion.get("assessment_mode") != "explicit_offline_only"
                or completion.get("stop_if_all_required_pass_both_seeds") is not False
                or completion.get("cap_result") != "unassessed_research_candidate"):
            raise ValueError("Task v2 must train continuously without an assessment gate")
        early_finish = ("iteration_schedule_or_explicit_evidence_review"
                        if plan["contract_id"] in SCHEDULE_REVISIONS else "explicit_evidence_review")
        if resized and (completion.get("review_interval_actor_updates") != 500
                or completion.get("early_finish") != early_finish
                or completion.get("review_blocks_training") is not False):
            raise ValueError("Task v5 needs a nonblocking review; 20000 is a ceiling")
        if plan["contract_id"] in SCHEDULE_REVISIONS:
            schedule = completion.get("schedule", {})
            endpoints = schedule.get("end_actor_updates", {})
            fixed_endpoints = (ORDINARY_POSTURE_ENDPOINTS if plan["contract_id"] in (REVISION_V18, REVISION_V19, REVISION_V20) else
                               TERRAIN_EXTENSION_ENDPOINTS if plan["contract_id"] in (REVISION_V16, REVISION_V17) else
                               NONSTANDARD_START_ENDPOINTS if plan["contract_id"] in (REVISION_V13, REVISION_V14, REVISION_V15)
                               else SHORT_SCHEDULE_ENDPOINTS)
            if (set(schedule) != {"clock", "end_actor_updates", "advance_requires_evaluation", "result"}
                    or schedule.get("clock") != "domain_actor_updates"
                    or schedule.get("advance_requires_evaluation") is not False
                    or schedule.get("result") != "unassessed_research_candidate"
                    or set(endpoints) != set(DOMAINS)
                    or any(type(endpoints.get(row["name"])) is not int for row in domains)
                    or (endpoints != fixed_endpoints if plan["contract_id"] in (REVISION_V9, REVISION_V10, REVISION_V11, REVISION_V12, REVISION_V13, REVISION_V14, REVISION_V15, REVISION_V16, REVISION_V17, REVISION_V18, REVISION_V19, REVISION_V20)
                        else any(not 6000 <= endpoints[row["name"]] <= row["updates"] - 10
                                 for row in domains))):
                raise ValueError("Automatic task schedule needs bounded actor clocks, not evaluation promotion")


def validate_plan_migration(source, target, run_directory):
    """Authenticate the one named revision without trusting mutable run receipts."""
    validate_plan(source)
    validate_plan(target)
    if (source["contract_id"], target["contract_id"]) not in ((REVISION, REVISION_V2), (REVISION_V2, REVISION_V3), (REVISION_V3, REVISION_V4), (REVISION_V4, REVISION_V5), (REVISION_V5, REVISION_V6), (REVISION_V6, REVISION_V7), (REVISION_V7, REVISION_V8), (REVISION_V8, REVISION_V9), (REVISION_V9, REVISION_V10), (REVISION_V10, REVISION_V11), (REVISION_V10, REVISION_V12), (REVISION_V12, REVISION_V13), (REVISION_V12, REVISION_V14), (REVISION_V14, REVISION_V15), (REVISION_V15, REVISION_V16), (REVISION_V15, REVISION_V17), (REVISION_V15, REVISION_V18), (REVISION_V18, REVISION_V19), (REVISION_V19, REVISION_V20)):
        raise ValueError("Only a named consecutive task revision or explicit deployed-predecessor revision is authorized")
    revision = target["contract_id"].rsplit("-", 1)[1]
    binding = target["revision_migration"]
    manifest = json.loads((ROOT / binding["manifest_file"]).read_text())
    for role, value in (("source", source), ("target", target)):
        path = ROOT / manifest[f"{role}_plan_file"]
        if digest(path) != manifest[f"{role}_plan_sha256"] or json.loads(path.read_text()) != value:
            raise ValueError("Task revision frozen plan SHA/content differs")
    if (str(Path(run_directory).resolve()) != manifest["run_directory"]
            or binding["run_directory"] != manifest["run_directory"]
            or binding["source_plan_sha256"] != manifest["source_plan_sha256"]
            or manifest.get("schema") != "task-plan-revision-" + revision):
        raise ValueError("Task revision run identity differs")
    if target["contract_id"] == REVISION_V17:
        from .scut_curriculum_revision import (
            MANIFEST_FILE, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, read_bound,
            revised_plan, revised_manifest, validate_frozen_workers,
        )
        if (source != read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, ROOT)
                or target != revised_plan(ROOT)
                or binding["manifest_file"] != MANIFEST_FILE
                or manifest != revised_manifest(ROOT)):
            raise ValueError("SCUT curriculum requires its exact four-worker v15 to v17 revision")
        validate_frozen_workers(ROOT)
    if target["contract_id"] == REVISION_V20:
        from .recovery_script_revision import (
            MANIFEST_FILE, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, read_bound,
            revised_plan, revised_manifest, validate_frozen_workers,
        )
        if (source != read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, ROOT)
                or target != revised_plan(ROOT)
                or binding["manifest_file"] != MANIFEST_FILE
                or manifest != revised_manifest(ROOT)):
            raise ValueError("Recovery script requires its exact recovery-only v19 to v20 revision")
        validate_frozen_workers(ROOT)
    if target["contract_id"] == REVISION_V19:
        from .recovery_quality_revision import (
            MANIFEST_FILE, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, read_bound,
            revised_plan, revised_manifest, validate_frozen_workers,
        )
        if (source != read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, ROOT)
                or target != revised_plan(ROOT)
                or binding["manifest_file"] != MANIFEST_FILE
                or manifest != revised_manifest(ROOT)):
            raise ValueError("Recovery quality requires its exact recovery-only v18 to v19 revision")
        validate_frozen_workers(ROOT)
    if target["contract_id"] == REVISION_V18:
        from .ordinary_posture_revision import (
            MANIFEST_FILE, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, read_bound,
            revised_plan, revised_manifest, validate_frozen_workers,
        )
        if (source != read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, ROOT)
                or target != revised_plan(ROOT)
                or binding["manifest_file"] != MANIFEST_FILE
                or manifest != revised_manifest(ROOT)):
            raise ValueError("Ordinary posture requires its exact five-worker v15 to v18 revision")
        validate_frozen_workers(ROOT)
    if target["contract_id"] == REVISION_V16:
        from .terrain_schedule_revision import (
            MANIFEST_FILE, ORIGINS, PLAN_FILE, SCOPE, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA,
            revised_plan, read_bound,
        )
        if (source != read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, ROOT)
                or target != revised_plan(ROOT)
                or binding["manifest_file"] != MANIFEST_FILE
                or manifest["source_plan_file"] != SOURCE_PLAN_FILE
                or manifest["source_plan_sha256"] != SOURCE_PLAN_SHA
                or manifest["target_plan_file"] != PLAN_FILE
                or manifest.get("schedule_scope") != SCOPE
                or manifest.get("reopen_actor_origins") != ORIGINS
                or manifest.get("reopen_actor_endpoints") != dict(terrain=15000)
                or set(manifest) != {"schema", "source_plan_file", "source_plan_sha256",
                    "target_plan_file", "target_plan_sha256", "run_directory", "schedule_scope",
                    "reopen_actor_origins", "reopen_actor_endpoints", "workers"}):
            raise ValueError("Terrain extension changes only its endpoint and single completed reopening")
    if target["contract_id"] == REVISION_V15:
        from .terrain_rehearsal_revision import (
            MANIFEST_FILE, PLAN_FILE, SCOPE, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA,
            SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, WORKER_FILE, plan_sampling_pools, read_bound,
        )
        if source != read_bound(SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, ROOT):
            raise ValueError("Terrain rehearsal requires its exact frozen v14 plan predecessor")
        expected = deepcopy(source)
        expected["contract_id"] = REVISION_V15
        row = next(row for row in expected["domains"] if row["name"] == "terrain")
        if (row["contract"], row["sha256"]) != (SOURCE_WORKER_FILE, SOURCE_WORKER_SHA):
            raise ValueError("Terrain rehearsal plan predecessor worker differs")
        row.update(contract=WORKER_FILE, sha256=digest(ROOT / WORKER_FILE),
                   sampling_pools=plan_sampling_pools(ROOT))
        expected["revision_migration"].update(manifest_file=MANIFEST_FILE,
            source_plan_sha256=SOURCE_PLAN_SHA)
        if (target != expected or manifest["source_plan_file"] != SOURCE_PLAN_FILE
                or manifest["source_plan_sha256"] != SOURCE_PLAN_SHA
                or manifest["target_plan_file"] != PLAN_FILE or manifest.get("sampling_scope") != SCOPE
                or set(manifest) != {"schema", "source_plan_file", "source_plan_sha256", "target_plan_file",
                    "target_plan_sha256", "run_directory", "sampling_scope", "workers"}):
            raise ValueError("Terrain rehearsal changes only terrain sampling; clocks, parents and reopening stay unchanged")
    if target["contract_id"] in (REVISION_V13, REVISION_V14):
        if target["contract_id"] == REVISION_V14:
            from .locomotion_quality_revision import (
                ENDPOINTS, MANIFEST_FILE, ORIGINS, PLAN_FILE, SCOPE, SOURCE_PLAN_FILE,
                SOURCE_PLAN_SHA, plan_sampling_pools, worker_file,
            )
            scope_key = "quality_scope"
        else:
            from .nonstandard_start_revision import (
                ENDPOINTS, MANIFEST_FILE, ORIGINS, PLAN_FILE, SCOPE, SOURCE_PLAN_FILE,
                SOURCE_PLAN_SHA, plan_sampling_pools, worker_file,
            )
            scope_key = "start_scope"
        expected = deepcopy(source)
        expected["contract_id"] = target["contract_id"]
        expected["domain_completion"]["schedule"]["end_actor_updates"] = deepcopy(ENDPOINTS)
        for row in expected["domains"]:
            if row["name"] in ORIGINS:
                file = worker_file(row["name"])
                row.update(contract=file, sha256=digest(ROOT / file), sampling_pools=plan_sampling_pools(row["name"], ROOT))
        expected["revision_migration"].update(manifest_file=MANIFEST_FILE,
            source_plan_sha256=SOURCE_PLAN_SHA)
        if (target != expected or manifest["source_plan_file"] != SOURCE_PLAN_FILE
                or manifest["source_plan_sha256"] != SOURCE_PLAN_SHA or manifest["target_plan_file"] != PLAN_FILE
                or manifest.get(scope_key) != SCOPE or manifest.get("reopen_actor_origins") != ORIGINS
                or manifest.get("reopen_actor_endpoints") != {name: ENDPOINTS[name] for name in ORIGINS}
                or set(manifest) != {"schema", "source_plan_file", "source_plan_sha256", "target_plan_file",
                    "target_plan_sha256", "run_directory", scope_key, "reopen_actor_origins",
                    "reopen_actor_endpoints", "workers"}):
            raise ValueError("Ordinary restart changes only its two named workers and bounded reopening windows")
    if target["contract_id"] == REVISION_V12:
        from .jump_landing_revision import (
            MANIFEST_FILE, PLAN_FILE, SCOPE, SOURCE_PLAN_FILE, SOURCE_PLAN_SHA, WORKER_FILE,
        )
        expected = deepcopy(source)
        expected["contract_id"] = REVISION_V12
        row = next(row for row in expected["domains"] if row["name"] == "jump")
        row.update(contract=WORKER_FILE, sha256=digest(ROOT / WORKER_FILE))
        expected["revision_migration"].update(manifest_file=MANIFEST_FILE,
            source_plan_sha256=manifest["source_plan_sha256"])
        if (target != expected or manifest["source_plan_file"] != SOURCE_PLAN_FILE
                or manifest["source_plan_sha256"] != SOURCE_PLAN_SHA
                or manifest["target_plan_file"] != PLAN_FILE or manifest.get("landing_scope") != SCOPE
                or set(manifest) != {"schema", "source_plan_file", "source_plan_sha256",
                    "target_plan_file", "target_plan_sha256", "run_directory", "landing_scope", "workers"}):
            raise ValueError("Task v12 changes only the named jump landing worker and revision binding")
    if target["contract_id"] == REVISION_V11:
        from .graded_locomotion_revision import SCOPE, WORKER_FILE
        origin = manifest.get("origin_actor_update")
        if type(origin) is not int or not 0 <= origin <= SHORT_SCHEDULE_ENDPOINTS["speed"] - 460:
            raise ValueError("Graded locomotion needs a frozen actor origin and a full-target training window")
        expected = deepcopy(source)
        expected["contract_id"] = REVISION_V11
        row = next(row for row in expected["domains"] if row["name"] == "speed")
        row.update(contract=WORKER_FILE, sha256=digest(ROOT / WORKER_FILE))
        expected["revision_migration"].update(
            manifest_file="contracts/v6_task_campaign_migration_v11.json",
            source_plan_sha256=manifest["source_plan_sha256"])
        worker = json.loads((ROOT / WORKER_FILE).read_text())
        if (target != expected or manifest["source_plan_file"] != "contracts/v6_task_campaign_v10.json"
                or manifest["target_plan_file"] != "contracts/v6_task_campaign_v11.json"
                or manifest.get("difficulty_scope") != SCOPE
                or worker["task_campaign"]["revision_migration"].get("origin_actor_update") != origin):
            raise ValueError("Task v11 changes only the named command difficulty and revision binding")
    if target["contract_id"] == REVISION_V10:
        from .locomotion_revision import SCOPE, WORKER_FILE, plan_sampling_pools
        expected = deepcopy(source)
        expected["contract_id"] = REVISION_V10
        row = next(row for row in expected["domains"] if row["name"] == "speed")
        row.update(contract=WORKER_FILE, sha256=digest(ROOT / WORKER_FILE),
                   sampling_pools=plan_sampling_pools(ROOT))
        expected["revision_migration"].update(
            manifest_file="contracts/v6_task_campaign_migration_v10.json",
            source_plan_sha256=manifest["source_plan_sha256"])
        if (target != expected or manifest["source_plan_file"] != "contracts/v6_task_campaign_v9.json"
                or manifest["target_plan_file"] != "contracts/v6_task_campaign_v10.json"
                or manifest.get("sampling_scope") != SCOPE):
            raise ValueError("Task v10 changes only the named mixed locomotion sampling and revision binding")
    if target["contract_id"] == REVISION_V9:
        expected = deepcopy(source)
        expected["contract_id"] = REVISION_V9
        expected["domain_completion"]["schedule"]["end_actor_updates"] = deepcopy(SHORT_SCHEDULE_ENDPOINTS)
        expected["revision_migration"].update(
            manifest_file="contracts/v6_task_campaign_migration_v9.json",
            source_plan_sha256=manifest["source_plan_sha256"])
        if (target != expected or manifest["source_plan_file"] != "contracts/v6_task_campaign_v8.json"
                or manifest["target_plan_file"] != "contracts/v6_task_campaign_v9.json"
                or manifest.get("schedule_scope") != SHORT_SCHEDULE_SCOPE):
            raise ValueError("Task v9 changes only its named first-pass endpoints and revision binding")
    allowed = {"contract_id", "domain_completion", "domains", "revision_migration"}
    if target["contract_id"] in (REVISION_V8, REVISION_V10, REVISION_V11, REVISION_V12, REVISION_V15):
        allowed.remove("domain_completion")
    if target["contract_id"] == REVISION_V5:
        allowed.update(("target_num_envs", "budget_spec_file", "budget_spec_sha256", "runtime_estimate"))
        validate_resized_budget(target)
        if target["runtime_estimate"] != dict(seconds_per_update=None,
                basis="4096x24_requires_native_wall_measurement; caps_not_required_run_length"):
            raise ValueError("Task v5 cannot assume an unmeasured wall-clock acceleration")
    if {k:v for k,v in source.items() if k not in allowed} != {k:v for k,v in target.items() if k not in allowed}:
        raise ValueError("Task revision changes budget, learning origin or campaign identity")
    for before, after, pin in zip(source["domains"], target["domains"], manifest["workers"], strict=True):
        changed = (("contract", "sha256", "sampling_pools")
                   if (target["contract_id"] == REVISION_V10 and before["name"] == "speed"
                       or target["contract_id"] in (REVISION_V13, REVISION_V14) and before["name"] in ("speed", "terrain")
                       or target["contract_id"] == REVISION_V15 and before["name"] == "terrain"
                       or target["contract_id"] == REVISION_V17 and before["name"] in ("recovery", "terrain", "jump", "stepjump")
                       or target["contract_id"] == REVISION_V18 and before["name"] in ("recovery", "speed", "terrain", "jump", "stepjump"))
                   else ("contract", "sha256"))
        if ({k:v for k,v in before.items() if k not in changed} !=
                {k:v for k,v in after.items() if k not in changed}):
            raise ValueError("Task revision changes domain scope or budget")
        for role, row in (("source", before), ("target", after)):
            path = ROOT / row["contract"]
            if (pin[f"{role}_file"] != row["contract"] or pin[f"{role}_sha256"] != row["sha256"]
                    or digest(path) != row["sha256"]):
                raise ValueError("Task revision worker SHA differs")
        if target["contract_id"] in (REVISION_V7, REVISION_V9, REVISION_V16):
            if before != after:
                raise ValueError("Automatic scheduling cannot change any worker contract or task scope")
        elif target["contract_id"] == REVISION_V8 and before["name"] != "recovery":
            if before != after:
                raise ValueError("Recovery v8 cannot change any of the other five workers")
        elif target["contract_id"] in (REVISION_V10, REVISION_V11) and before["name"] != "speed":
            if before != after:
                raise ValueError("Locomotion revision cannot change any of the other five workers")
        elif target["contract_id"] == REVISION_V12 and before["name"] != "jump":
            if before != after:
                raise ValueError("Jump landing revision cannot change any of the other five workers")
        elif target["contract_id"] in (REVISION_V13, REVISION_V14) and before["name"] not in ("speed", "terrain"):
            if before != after:
                raise ValueError("Nonstandard start revision cannot change any of the other four workers")
        elif target["contract_id"] == REVISION_V15 and before["name"] != "terrain":
            if before != after:
                raise ValueError("Terrain rehearsal cannot change any of the other five workers")
        elif target["contract_id"] == REVISION_V17 and before["name"] not in ("recovery", "terrain", "jump", "stepjump"):
            if before != after:
                raise ValueError("SCUT curriculum cannot change the completed flat and speed workers")
        elif target["contract_id"] == REVISION_V20 and before["name"] != "recovery":
            if before != after:
                raise ValueError("Recovery script cannot change any of the other five workers")
        elif target["contract_id"] == REVISION_V19 and before["name"] != "recovery":
            if before != after:
                raise ValueError("Recovery quality cannot change any of the other five workers")
        elif target["contract_id"] == REVISION_V18 and before["name"] == "flat":
            if before != after:
                raise ValueError("Ordinary posture cannot change the completed flat worker")
        else:
            worker = json.loads((ROOT / after["contract"]).read_text())
            if (target["contract_id"] == REVISION_V8 and
                    worker.get("task_campaign", {}).get("revision_migration", {}).get("schema") != "task-worker-revision-v8"):
                raise ValueError("Recovery v8 requires its named worker revision")
            if (target["contract_id"] == REVISION_V10 and
                    worker.get("task_campaign", {}).get("revision_migration", {}).get("schema") != "task-worker-revision-v10"):
                raise ValueError("Mixed locomotion v10 requires its named worker revision")
            if (target["contract_id"] == REVISION_V11 and
                    worker.get("task_campaign", {}).get("revision_migration", {}).get("schema") != "task-worker-revision-v11"):
                raise ValueError("Graded locomotion v11 requires its named worker revision")
            if (target["contract_id"] == REVISION_V12 and
                    worker.get("task_campaign", {}).get("revision_migration", {}).get("schema") != "task-worker-revision-v12"):
                raise ValueError("Jump landing v12 requires its named worker revision")
            if (target["contract_id"] in (REVISION_V13, REVISION_V14) and
                    worker.get("task_campaign", {}).get("revision_migration", {}).get("schema") != "task-worker-revision-" + revision):
                raise ValueError("Ordinary restart requires both named worker revisions")
            if (target["contract_id"] == REVISION_V15 and
                    worker.get("task_campaign", {}).get("revision_migration", {}).get("schema") != "task-worker-revision-v15"):
                raise ValueError("Terrain rehearsal requires its named terrain worker revision")
            if (target["contract_id"] == REVISION_V17 and
                    worker.get("task_campaign", {}).get("revision_migration", {}).get("schema") != "task-worker-revision-v17"):
                raise ValueError("SCUT curriculum requires its four named worker revisions")
            if (target["contract_id"] == REVISION_V18 and
                    worker.get("task_campaign", {}).get("revision_migration", {}).get("schema") != "task-worker-revision-v18"):
                raise ValueError("Ordinary posture requires its five named worker revisions")
            if (target["contract_id"] == REVISION_V20 and
                    worker.get("task_campaign", {}).get("revision_migration", {}).get("schema") != "task-worker-revision-v20"):
                raise ValueError("Recovery script requires its named recovery worker revision")
            if (target["contract_id"] == REVISION_V19 and
                    worker.get("task_campaign", {}).get("revision_migration", {}).get("schema") != "task-worker-revision-v19"):
                raise ValueError("Recovery quality requires its named recovery worker revision")
            validate_worker_revision(json.loads((ROOT / before["contract"]).read_text()), worker)
    return dict(schema="task-plan-revision-" + revision, campaign_id=CAMPAIGN_ID,
                source_plan_sha256=manifest["source_plan_sha256"],
                target_plan_sha256=manifest["target_plan_sha256"], run_directory=manifest["run_directory"],
                actor_critic_optimizer="preserved", task_clock="restored", consumption="preserved_no_refund")


def validate_worker_revision(source, target):
    binding = target.get("task_campaign", {}).get("revision_migration", {})
    if binding.get("schema") not in ("task-worker-revision-v2", "task-worker-revision-v3", "task-worker-revision-v4", "task-worker-revision-v5", "task-worker-revision-v6", "task-worker-revision-v8", "task-worker-revision-v10", "task-worker-revision-v11", "task-worker-revision-v12", "task-worker-revision-v13", "task-worker-revision-v14", "task-worker-revision-v15", "task-worker-revision-v17", "task-worker-revision-v18", "task-worker-revision-v19", "task-worker-revision-v20"):
        raise ValueError("Unknown named task worker revision")
    if binding.get("schema") == "task-worker-revision-v20":
        from .recovery_script_revision import (
            SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, canonical_worker, revision_binding,
        )
        if (target.get("task_campaign", {}).get("domain") != "recovery"
                or binding != revision_binding()
                or digest(ROOT / SOURCE_WORKER_FILE) != SOURCE_WORKER_SHA
                or json.loads((ROOT / SOURCE_WORKER_FILE).read_text()) != source):
            raise ValueError("Recovery script worker requires its exact deployed v19 own-domain source")
        if target != canonical_worker(root=ROOT):
            raise ValueError("Recovery script worker changes fields outside its exact named recipe")
        return
    if binding.get("schema") == "task-worker-revision-v19":
        from .recovery_quality_revision import (
            SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, canonical_worker, revision_binding,
        )
        if (target.get("task_campaign", {}).get("domain") != "recovery"
                or binding != revision_binding()
                or digest(ROOT / SOURCE_WORKER_FILE) != SOURCE_WORKER_SHA
                or json.loads((ROOT / SOURCE_WORKER_FILE).read_text()) != source):
            raise ValueError("Recovery quality worker requires its exact deployed v18 own-domain source")
        if target != canonical_worker(root=ROOT):
            raise ValueError("Recovery quality worker changes fields outside its exact named recipe")
        return
    if binding.get("schema") == "task-worker-revision-v18":
        from .ordinary_posture_revision import DOMAINS_TO_REVISE, source_binding, canonical_worker
        domain = target.get("task_campaign", {}).get("domain")
        if domain not in DOMAINS_TO_REVISE:
            raise ValueError("Ordinary posture changes only its five named domains")
        source_file, source_sha = source_binding(domain, ROOT)
        if (binding != dict(schema="task-worker-revision-v18",
                source_contract_file=source_file, source_contract_sha256=source_sha)
                or digest(ROOT / source_file) != source_sha
                or json.loads((ROOT / source_file).read_text()) != source):
            raise ValueError("Ordinary posture worker requires its exact deployed v15 own-domain source")
        if target != canonical_worker(domain, ROOT):
            raise ValueError("Ordinary posture worker changes fields outside its complete named curriculum recipe")
        return
    if binding.get("schema") == "task-worker-revision-v17":
        domain = target.get("task_campaign", {}).get("domain")
        if domain == "recovery":
            from .recovery_prefix_revision import (
                SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, WORKER_FILE, DESIGN_FILE,
                apply_prefix_revision as apply, revised_design,
            )
            source_file, source_sha, file = SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, DESIGN_FILE
            design = revised_design(ROOT)
        elif domain == "terrain":
            from .scut_terrain_revision import (
                SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, WORKER_FILE, DESIGN_FILE,
                apply_revision as apply, revised_design,
            )
            source_file, source_sha, file = SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, DESIGN_FILE
            design = revised_design(ROOT)
        elif domain in ("jump", "stepjump"):
            from .scut_jump_revision import SOURCES, apply_revision as apply, design_file, revised_design
            source_file, source_sha = SOURCES[domain]
            file, design = design_file(domain), revised_design(domain, ROOT)
        else:
            raise ValueError("SCUT curriculum changes only its four named domains")
        if (digest(ROOT / source_file) != source_sha
                or json.loads((ROOT / source_file).read_text()) != source):
            raise ValueError("SCUT worker cannot rewind or replace its frozen own-domain source")
        expected = deepcopy(source)
        apply(expected, ROOT)
        expected["design_preflight"] = dict(file=file, sha256=digest(ROOT / file), stage=domain)
        if json.loads((ROOT / file).read_text()) != design or target != expected:
            raise ValueError("SCUT worker changes fields outside its complete named curriculum recipe")
        return
    if binding.get("schema") == "task-worker-revision-v15":
        from .terrain_rehearsal_revision import (
            DESIGN_FILE, LEGACY_WORKER_FILE, LEGACY_WORKER_SHA, SCHEMA,
            SOURCE_WORKER_FILE, SOURCE_WORKER_SHA, apply_rehearsal_revision, read_bound, revised_design,
        )
        accepted = [dict(source_contract_file=file, source_contract_sha256=sha)
            for file, sha in ((SOURCE_WORKER_FILE, SOURCE_WORKER_SHA),
                              (LEGACY_WORKER_FILE, LEGACY_WORKER_SHA))]
        expected_binding = dict(schema=SCHEMA, source_contract_file=SOURCE_WORKER_FILE,
            source_contract_sha256=SOURCE_WORKER_SHA, accepted_resume_sources=accepted)
        if target.get("task_campaign", {}).get("domain") != "terrain" or binding != expected_binding:
            raise ValueError("Terrain rehearsal requires its two exact ordered same-domain predecessors")
        predecessors = [read_bound(row["source_contract_file"], row["source_contract_sha256"], ROOT)
                        for row in accepted]
        if source not in predecessors:
            raise ValueError("Terrain rehearsal learning source differs from frozen v14/v6 content")
        expected = deepcopy(source)
        if source == predecessors[1]:
            # The campaign declared v14 before this domain had trained it.
            # Authenticate the real v6 seal and compose the exact named repair.
            from .locomotion_quality_revision import apply_quality_revision
            apply_quality_revision(expected)
        apply_rehearsal_revision(expected, ROOT)
        design_path = ROOT / DESIGN_FILE
        expected["design_preflight"] = dict(file=DESIGN_FILE, sha256=digest(design_path), stage="terrain")
        if json.loads(design_path.read_text()) != revised_design(ROOT) or target != expected:
            raise ValueError("Terrain rehearsal changed fields outside its named sampling and design revision")
        return
    path = ROOT / binding.get("source_contract_file", "")
    if not path.is_file() or digest(path) != binding.get("source_contract_sha256") or json.loads(path.read_text()) != source:
        raise ValueError("Task worker revision source is not its frozen predecessor")
    expected = deepcopy(source)
    if binding.get("schema") == "task-worker-revision-v14":
        from .locomotion_quality_revision import apply_quality_revision, design_file, revised_design, revision_binding
        domain = target.get("task_campaign", {}).get("domain")
        if domain not in ("speed", "terrain") or binding != revision_binding(domain):
            raise ValueError("Locomotion quality v14 requires its exact predecessor and completed actor origin")
        apply_quality_revision(expected)
        file = design_file(domain)
        path = ROOT / file
        expected["design_preflight"] = dict(file=file, sha256=digest(path), stage=domain)
        if json.loads(path.read_text()) != revised_design(domain, ROOT) or target != expected:
            raise ValueError("Locomotion quality changed fields outside its named reward and command repair")
        return
    if binding.get("schema") == "task-worker-revision-v13":
        from .nonstandard_start_revision import apply_start_revision, design_file, revised_design, revision_binding
        domain = target.get("task_campaign", {}).get("domain")
        if domain not in ("speed", "terrain") or binding != revision_binding(domain):
            raise ValueError("Nonstandard start v13 requires its exact domain predecessor and completed actor origin")
        apply_start_revision(expected)
        file = design_file(domain)
        path = ROOT / file
        expected["design_preflight"] = dict(file=file, sha256=digest(path), stage=domain)
        if json.loads(path.read_text()) != revised_design(domain, ROOT) or target != expected:
            raise ValueError("Nonstandard start revision changed fields outside its ordinary reset/release patch")
        return
    if binding.get("schema") == "task-worker-revision-v12":
        from .jump_landing_revision import DESIGN_FILE, apply_landing_revision, revised_design, revision_binding
        if binding != revision_binding():
            raise ValueError("Jump landing v12 requires its frozen jump v6 predecessor")
        apply_landing_revision(expected)
        design_path = ROOT / DESIGN_FILE
        expected["design_preflight"] = dict(file=DESIGN_FILE, sha256=digest(design_path), stage="jump")
        if json.loads(design_path.read_text()) != revised_design(ROOT) or target != expected:
            raise ValueError("Jump landing revision changed fields outside the finite contact-cost patch")
        return
    if binding.get("schema") == "task-worker-revision-v11":
        from .graded_locomotion_revision import (
            DESIGN_FILE, apply_graded_revision, revised_design, revision_binding,
        )
        origin = binding.get("origin_actor_update")
        if binding != revision_binding(origin):
            raise ValueError("Graded locomotion v11 requires its frozen v10 predecessor and actor origin")
        apply_graded_revision(expected, origin)
        design_path = ROOT / DESIGN_FILE
        expected["design_preflight"] = dict(file=DESIGN_FILE, sha256=digest(design_path), stage="speed")
        if json.loads(design_path.read_text()) != revised_design(origin, ROOT) or target != expected:
            raise ValueError("Graded locomotion revision changed fields outside the command difficulty patch")
        return
    if binding.get("schema") == "task-worker-revision-v10":
        from .locomotion_revision import (
            DESIGN_FILE, SCHEMA, SPEED_FILE, SPEED_SHA, apply_mixed_locomotion_revision, revised_design,
        )
        if binding != dict(schema=SCHEMA, source_contract_file=SPEED_FILE, source_contract_sha256=SPEED_SHA):
            raise ValueError("Mixed locomotion v10 requires its frozen speed v6 predecessor")
        apply_mixed_locomotion_revision(expected, ROOT)
        expected["task_campaign"]["revision_migration"] = deepcopy(binding)
        design = revised_design(ROOT)
        design_path = ROOT / DESIGN_FILE
        expected["design_preflight"] = dict(file=DESIGN_FILE, sha256=digest(design_path), stage="speed")
        if json.loads(design_path.read_text()) != design or target != expected:
            raise ValueError("Mixed locomotion revision changed fields outside the sampling patch")
        return
    if binding.get("schema") in ("task-worker-revision-v4", "task-worker-revision-v5", "task-worker-revision-v6", "task-worker-revision-v8"):
        if binding["schema"] == "task-worker-revision-v8":
            if (set(binding) != {"schema", "source_contract_file", "source_contract_sha256", "origin_actor_update"}
                    or binding["source_contract_file"] != RECOVERY_V6_FILE
                    or binding["source_contract_sha256"] != RECOVERY_V6_SHA256
                    or type(binding.get("origin_actor_update")) is not int or binding["origin_actor_update"] < 0
                    or source.get("task_campaign", {}).get("domain") != "recovery"
                    or any(target["task_campaign"].get(key) != source["task_campaign"].get(key)
                           for key in ("source_contract_file", "source_contract_sha256"))):
                raise ValueError("Recovery v8 requires the frozen v6 worker, original parent and nonnegative actor origin")
            from .recovery_revision import apply_recovery_takeover_revision
            apply_recovery_takeover_revision(expected, binding["origin_actor_update"])
        elif binding["schema"] == "task-worker-revision-v6":
            add_signal_delay(expected)
        elif binding["schema"] == "task-worker-revision-v5":
            resize_reviewed_worker(expected)
        else:
            repair_coverage_and_events(expected)
        for key in ("source_contract_file", "source_contract_sha256", "revision_migration"):
            expected["task_campaign"][key] = deepcopy(target["task_campaign"][key])
        expected["design_preflight"] = deepcopy(target["design_preflight"])
        for field, hash_field in (("source_contract_file", "source_contract_sha256"),):
            if digest(ROOT / target["task_campaign"][field]) != target["task_campaign"][hash_field]:
                raise ValueError("Reward/coverage revision parent SHA differs")
        design = target["design_preflight"]
        if digest(ROOT / design["file"]) != design["sha256"] or expected != target:
            raise ValueError("Reward/coverage revision changed fields outside the reviewed patch")
        return
    if binding.get("schema") == "task-worker-revision-v3":
        validate_coarse_control_change(source, target)
        for key in ("physics_dt", "control_math_source", "control_math_sha256", "design_preflight"):
            expected[key] = deepcopy(target[key])
        for key in ("source_contract_file", "source_contract_sha256", "revision_migration"):
            expected["task_campaign"][key] = deepcopy(target["task_campaign"][key])
        for field, hash_field in (("source_contract_file", "source_contract_sha256"),):
            if digest(ROOT / target["task_campaign"][field]) != target["task_campaign"][hash_field]:
                raise ValueError("Coarse revision parent SHA differs")
        design = target["design_preflight"]
        if digest(ROOT / design["file"]) != design["sha256"] or expected != target:
            raise ValueError("Coarse revision changed fields outside timing/control identity")
        return
    # Every permitted revision field has an explicit value. Everything else,
    # including reward, mechanics, case thresholds and domain sampling, is exact.
    expected["checkpoint_interval"] = 500
    old_meta = expected["task_campaign"]
    new_meta = target["task_campaign"]
    for key in ("source_contract_file", "source_contract_sha256"):
        old_meta[key] = new_meta[key]
    old_meta.update(execution_mode=CONTINUOUS_MODE, evaluation_interval=0,
                    revision_migration=deepcopy(binding))
    parent = ROOT / new_meta["source_contract_file"]
    if digest(parent) != new_meta["source_contract_sha256"]:
        raise ValueError("Task revision parent SHA differs")
    expected["design_preflight"] = deepcopy(target["design_preflight"])
    design = ROOT / expected["design_preflight"]["file"]
    if digest(design) != expected["design_preflight"]["sha256"]:
        raise ValueError("Task revision design SHA differs")
    if old_meta["domain"] in ("jump", "stepjump"):
        curriculum = expected["new_asset_curriculum"]["jump_curriculum"]
        curriculum.pop("confirmed_pass_transfer_fraction", None)
        curriculum.update(clock="domain_actor_updates", iteration_probability_schedule=jump_probability_schedule())
    if expected != target:
        raise ValueError("Task worker revision changed fields outside the frozen scheduling patch")


def jump_probability_schedule():
    return [dict(actor_update=0, probabilities=[.30,.25,.20,.10,.05,.05,.05]),
            dict(actor_update=2000, probabilities=[.10,.15,.20,.20,.15,.10,.10]),
            dict(actor_update=6000, probabilities=[.05,.05,.10,.15,.20,.20,.25])]


def repair_coverage_and_events(config):
    """Reviewed v4 patch; pool/height/terrain quotas and tracking weights are exact."""
    for spec in config["skill_specs"].values():
        if spec.get("sample_amplitude"):
            spec["amplitude_range"] = [.05, 1.]
        if spec.get("kind") == "curve" and all(spec["command"][:2]):
            vx, yaw = map(abs, spec["command"][:2])
            spec["planar_command_sampling"] = dict(vx_range_m_s=[-vx, vx],
                yaw_range_rad_s=[-yaw, yaw], nominal_fraction=.5)
    # Retain the failure cost: copying SCUT's dt-scaled -4 permits early failure
    # under our unbounded broad costs. Completion must beat collecting its tail.
    config.update(success_event_bonus=30., resume_critic_warmup_updates=10,
        reward_units=dict(densities="per_second_times_policy_dt",
                          termination_event_cost="once_per_true_failure",
                          success_event_bonus="once_per_success"))


def validate_resized_budget(config):
    path = ROOT / config["budget_spec_file"]
    previous = json.loads((ROOT / "contracts/v6_task_campaign_budget_v1.json").read_text())
    expected = dict(previous, schema="task-domain-budget-v2", update_unit="one_PPO_optimizer_update",
        transition_accounting="sum_actual_num_envs_times_rollout_steps_per_attempt_no_refund",
        rollout_steps_per_env=24, previous_num_envs=20000, current_num_envs=4096)
    if (config["budget_spec_file"] != "contracts/v6_task_campaign_budget_v2.json"
            or digest(path) != config["budget_spec_sha256"] or json.loads(path.read_text()) != expected):
        raise ValueError("4096 revision changes the budget beyond its explicit mixed-batch units")


def resize_reviewed_worker(config):
    """V5 changes sampling work per update; control, reward and task clocks stay exact."""
    path = ROOT / "contracts/v6_task_campaign_budget_v2.json"
    config.update(target_num_envs=4096, num_mini_batches=4, ppo_minibatch_samples=24576,
        budget_spec_file=str(path.relative_to(ROOT)),
        budget_spec_sha256=digest(path))
    config["task_campaign"]["execution_mode"] = REVIEW_MODE
    validate_resized_budget(config)


def signal_delay_settings(domain, *, enabled=True):
    return dict(version="physical_signal_delay_v1", enabled=enabled,
        clock="physics_steps", jitter_clock="policy_steps", enabled_fraction=.75, jitter_ms=5.,
        feedback_pd="fresh_physics_feedback", scope="engineering_pipeline_age_not_measured_USB_latency",
        profiles=[dict(probability=p, observation_ms=o, action_ms=a) for p, o, a in
            ((.60, [0.,20.], [0.,15.]), (.35, [20.,40.], [15.,30.]), (.05, [40.,80.], [30.,60.]))],
        schedule=dict(clock="domain_actor_updates", start=1500 if domain == "flat" else 0,
                      end=2500 if domain == "flat" else 1000))


def add_signal_delay(config):
    """V6 adds sensor/setpoint age only; reward, physics and all task quotas stay exact."""
    config["signal_delay"] = signal_delay_settings(config["task_campaign"]["domain"])
    config["evaluation"]["frozen_module_enablement"]["signal_delay"] = False


def validate_domain_review(plan, root, row, request, consumed):
    """Authenticate an operator's early finish; review never grants skill qualification."""
    from .full_curriculum import checkpoint_update_count
    if (plan["contract_id"] not in REVIEW_REVISIONS or request.get("schema") != "task-domain-review-v1"
            or request.get("campaign_id") != CAMPAIGN_ID
            or request.get("plan_sha256") != digest(ROOT / ("contracts/" + plan["contract_id"].replace("-", "_") + ".json"))
            or request.get("domain") != row["name"] or request.get("decision") != "finish_domain"
            or not isinstance(request.get("reason"), str) or not request["reason"].strip()):
        raise ValueError("Review does not identify the active frozen task/domain/decision")
    root = Path(root).resolve()
    checkpoint = Path(request["checkpoint"]).resolve(strict=True)
    if (not checkpoint.is_relative_to(root / row["name"]) or "checkpoints" not in checkpoint.relative_to(root).parts
            or digest(checkpoint) != request["checkpoint_sha256"]):
        raise ValueError("Review must name an immutable checkpoint from the active domain")
    count = checkpoint_update_count(checkpoint)
    seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
    if (count > consumed or seal.get("contract_sha256") != row["sha256"]
            or seal.get("task_campaign", {}).get("domain") != row["name"]):
        raise ValueError("Review checkpoint identity/consumption differs from the active task")
    evidence = request.get("evidence", [])
    if len(evidence) != 2 or {item.get("kind") for item in evidence} != {"curve_review", "behavior_review"}:
        raise ValueError("Early finish requires curve and behavior evidence, not total reward alone")
    for item in evidence:
        path = Path(item["file"]).resolve(strict=True)
        if (not path.is_relative_to(root / "reviews") or path.suffix != ".json"
                or digest(path) != item["sha256"]):
            raise ValueError("Review evidence is outside the immutable review receipt directory or SHA differs")
        data = json.loads(path.read_text())
        if (data.get("checkpoint_sha256") != request["checkpoint_sha256"]
                or not isinstance(data.get("observations"), dict) or not data["observations"]):
            raise ValueError("Review evidence must describe this checkpoint's observed curves/behavior")
    return {**request, "consumed_updates_at_review": consumed, "domain_qualified": None,
            "continuation": "seal_latest_full_learning_state; preserve_all_consumption"}


def validate_coarse_control_change(source, target):
    """Authenticate the explicit approximation while retaining every motor parameter."""
    from .control_contract import load_chassis_control, COARSE_SCOPE, COARSE_TIMING
    if (source.get("physics_dt"), target.get("physics_dt")) != (.001, .005):
        raise ValueError("Coarse revision must change 1 ms to 5 ms")
    if target.get("task_campaign", {}).get("revision_migration", {}).get("schema") not in ("task-worker-revision-v3", "task-worker-revision-v4", "task-worker-revision-v5", "task-worker-revision-v6", "task-worker-revision-v8", "task-worker-revision-v10", "task-worker-revision-v11", "task-worker-revision-v12", "task-worker-revision-v13", "task-worker-revision-v14", "task-worker-revision-v15", "task-worker-revision-v17", "task-worker-revision-v18", "task-worker-revision-v19", "task-worker-revision-v20"):
        raise ValueError("Coarse physics identity requires a named v3 revision")
    controls = []
    for config in (source, target):
        path = ROOT / config["control_math_source"]
        if digest(path) != config["control_math_sha256"]:
            raise ValueError("Coarse control SHA differs")
        controls.append(load_chassis_control(path))
    expected = deepcopy(controls[0])
    expected.update(control_id="v6-wheel-response-control-200hz-v1", timing=COARSE_TIMING,
                    simulation_timing_approximation=COARSE_SCOPE)
    if controls[1] != expected:
        raise ValueError("Coarse migration changes motor parameters or undeclared timing")


def validate_task_resume(source, target):
    meta = target.get("task_campaign", {})
    if (meta.get("campaign_id") != CAMPAIGN_ID or meta.get("domain") not in DOMAINS
            or meta.get("updates") != 20000 or meta.get("clock") != "domain_actor_updates"):
        raise ValueError("Missing frozen task migration identity")
    same_domain = source.get("task_campaign", {}).get("domain") == meta["domain"]
    if same_domain:
        if source != target:
            if not meta.get("revision_migration"):
                raise ValueError("Same-domain recovery cannot change its contract")
            validate_worker_revision(source, target)
    else:
        path = ROOT / meta["source_contract_file"]
        if digest(path) != meta["source_contract_sha256"] or json.loads(path.read_text()) != source:
            from .jump_landing_revision import validate_landing_parent_alias
            if not validate_landing_parent_alias(source, target, ROOT):
                raise ValueError("Task learning origin differs from its frozen source contract")
    # Curriculum/reward domains may differ. The physical, actuator and neural
    # interface remains identical; accepted capability is never inferred here.
    coarse_change = source.get("physics_dt") != target.get("physics_dt")
    if coarse_change:
        validate_coarse_control_change(source, target)
    for key in ("asset_manifest_sha256", "control_math_sha256", "physics_dt", "policy_dt",
                "actor_dim", "critic_dim", "action_dim", "history_length", "policy_architecture",
                "v5_control", "actuator_response", "actuator_response_evidence", "policy_action_order",
                "actor_layout", "critic_layout", "actor_observation_source"):
        if coarse_change and key in ("physics_dt", "control_math_sha256"):
            continue
        if source.get(key) != target.get(key):
            raise ValueError(f"Task migration changes physical/policy identity: {key}")
    revision = target.get("task_campaign", {}).get("revision_migration", {}).get("schema")
    coarse = revision in ("task-worker-revision-v3", "task-worker-revision-v4", "task-worker-revision-v5", "task-worker-revision-v6", "task-worker-revision-v8", "task-worker-revision-v10", "task-worker-revision-v11", "task-worker-revision-v12", "task-worker-revision-v13", "task-worker-revision-v14", "task-worker-revision-v15", "task-worker-revision-v17", "task-worker-revision-v18", "task-worker-revision-v19", "task-worker-revision-v20")
    resized = revision in ("task-worker-revision-v5", "task-worker-revision-v6", "task-worker-revision-v8", "task-worker-revision-v10", "task-worker-revision-v11", "task-worker-revision-v12", "task-worker-revision-v13", "task-worker-revision-v14", "task-worker-revision-v15", "task-worker-revision-v17", "task-worker-revision-v18", "task-worker-revision-v19", "task-worker-revision-v20")
    if (target.get("physics_dt"), target.get("policy_dt"), target.get("actor_dim"),
            target.get("critic_dim"), target.get("target_num_envs")) != (.005 if coarse else .001, .02, 35, 81, 4096 if resized else 20000):
        raise ValueError("Task campaign changes its frozen environment count, network or control clocks")
    if resized:
        validate_resized_budget(target)
        if (target.get("num_steps_per_env"), target.get("num_mini_batches"),
                target.get("ppo_minibatch_samples"), target.get("num_learning_epochs")) != (24, 4, 24576, 5):
            raise ValueError("Task v5 requires 4096x24, four minibatches and five epochs")
    if target.get("learning_rate") != 1e-4 or target.get("learning_rate_schedule") != "fixed":
        raise ValueError("Task campaign fixed learning rate changed")
    return dict(kind="task_domain_recovery" if same_domain else "task_domain_migration",
                actor_critic_optimizer="preserved", environment_state="new_episodes",
                task_clock="restored" if same_domain else "new_domain_zero",
                accepted_stage=None, hardware_release=False)


def validate_worker(config, args):
    from .full_curriculum import checkpoint_contract_path, checkpoint_update_count
    meta = config["task_campaign"]
    if (not args.resume_task_change or not args.resume or args.num_envs != config["target_num_envs"]
            or args.seed != 290927 or args.learning_rate_scale != 1.
            or args.global_actor_update_offset != 0 or args.block_index != 0
            or type(args.updates) is not int or not 0 < args.updates <= 20000
            or type(args.consumed_updates) is not int or args.consumed_updates < 1000
            or args.consumed_updates + args.updates > 121000):
        raise ValueError("Task worker changes its invocation or campaign budget")
    checkpoint_update_count(args.resume)
    source = json.loads(checkpoint_contract_path(args.resume).read_text())
    validate_task_resume(source, config)
    seal = json.loads(Path(args.resume).with_suffix(".checkpoint.json").read_text())
    same = source.get("task_campaign", {}).get("domain") == meta["domain"]
    task_clock = seal.get("task_actor_update") if same else 0
    if (same and source != config
            and meta.get("revision_migration", {}).get("schema") in ("task-worker-revision-v11", "task-worker-revision-v13", "task-worker-revision-v14")
            and meta["revision_migration"]["origin_actor_update"] != task_clock):
        raise ValueError("Named curriculum origin must equal the sealed predecessor actor clock")
    if (type(task_clock) is not int or task_clock < 0 or args.task_actor_update != task_clock
            or args.stage_actor_update != seal["stage_actor_update"]
            or args.actor_updates_consumed < seal["actor_updates_consumed"]
            or task_clock + args.updates > 20000):
        raise ValueError("Task worker changes its task/lineage clock or consumption")


def account_updates(plan, attempts):
    """Every attempt remains charged, including updates absent from a snapshot."""
    caps = {row["name"]: row["updates"] for row in plan["domains"]}
    by_domain = dict.fromkeys(caps, 0)
    actor = plan["historical_actor_consumed_updates"]
    for record in attempts.values():
        count, actor_count = record["consumed_updates"], record.get("actor_updates", 0)
        domain = record["domain"]
        if (domain not in caps or type(count) is not int or type(actor_count) is not int
                or count < 0 or not 0 <= actor_count <= count):
            raise ValueError("Invalid charged task attempt")
        by_domain[domain] += count
        actor += actor_count
    if any(by_domain[name] > caps[name] for name in caps):
        raise ValueError("Task budget exhausted")
    total = plan["historical_consumed_updates"] + sum(by_domain.values())
    if total > plan["total_ceiling"]:
        raise ValueError("Campaign budget exhausted")
    return dict(consumed_updates=total, actor_updates_consumed=actor,
                domain_consumed_updates=by_domain, remaining_updates=plan["total_ceiling"] - total)


def update_attempt(plan, attempts, name, domain, record):
    result = deepcopy(attempts)
    old = result.get(name, dict(domain=domain, consumed_updates=0, actor_updates=0))
    completed = int(record.get("successful_updates", 0))
    # A worker may publish an update reservation before optimization. Retain
    # such work after an abrupt exit rather than refunding an uncertain update.
    count = max(completed, int(record.get("attempted_updates", completed)))
    actors = int(record.get("actor_updates_in_block", 0))
    if old["domain"] != domain or count < old["consumed_updates"] or actors < old["actor_updates"]:
        raise ValueError("Task consumption cannot move backwards")
    result[name] = dict(domain=domain, consumed_updates=count, actor_updates=actors)
    account_updates(plan, result)
    return result


def assess_domain(results, required):
    if len(results) != 2 or any(set(required) - set(r["cases"]) for r in results):
        raise ValueError("invalid_evaluation_coverage")
    failed = sorted(n for n in required if not all(r["cases"][n]["passed"] for r in results))
    return dict(domain_qualified=not failed, required_failed=failed,
                accepted_stage=None, deployment_checkpoint=None,
                scope="individual_simulation_task_domain_not_integrated_policy")


def validate_task_budget(config, root):
    path = Path(root) / config["budget_spec_file"]
    if digest(path) != config["budget_spec_sha256"]:
        raise ValueError("Task budget specification SHA differs")
    budget = json.loads(path.read_text())
    domain = config["task_campaign"]["domain"]
    expected = dict.fromkeys(DOMAINS, 20000)
    resized = config.get("task_campaign", {}).get("revision_migration", {}).get("schema") in ("task-worker-revision-v5", "task-worker-revision-v6", "task-worker-revision-v8", "task-worker-revision-v10", "task-worker-revision-v11", "task-worker-revision-v12", "task-worker-revision-v13", "task-worker-revision-v14", "task-worker-revision-v15", "task-worker-revision-v17", "task-worker-revision-v18", "task-worker-revision-v19", "task-worker-revision-v20")
    if resized:
        validate_resized_budget(config)
    if (budget.get("schema") != ("task-domain-budget-v2" if resized else "task-domain-budget-v1")
            or budget.get("campaign_id") != CAMPAIGN_ID
            or budget.get("historical_consumed_updates") != 1000
            or budget.get("historical_actor_consumed_updates") != 980
            or budget.get("total_ceiling") != 121000 or budget.get("domains") != expected
            or budget.get("rollback_refunds_consumed_updates") is not False
            or domain not in expected or config.get("budget_stage_id") != domain
            or config.get("curriculum_stage") != domain or config.get("total_updates") != 20000
            or sum(stage["updates"] for stage in config["stages"]) != 20000):
        raise ValueError("Task-domain budget must preserve 1000 historical plus six 20000-update caps")
    return dict(schema=budget["schema"], campaign_id=CAMPAIGN_ID, domain=domain,
                domain_update_ceiling=20000, historical_consumed_updates=1000,
                total_ceiling=121000, budget_spec_sha256=digest(path), rollback_refunds_consumed_updates=False)
