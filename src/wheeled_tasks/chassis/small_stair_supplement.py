"""Authenticated evaluation-only derivatives of the unchanged v25 stair exam."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import integrated_production as production
from .control_contract import load_chassis_control
from .preflight import design_preflight


SOURCE_COMMIT = "16ed8f63c832924762778ca6b2221c250b006924"
MANIFEST_FILE = "reports/v6_scut_posture_completion_20261006/source_repair_027/small_stairs_supplement001/case_manifest001.json"
MANIFEST_SHA = "6be052cdc0ebac571f90ba2faa6ebf51d6b90d8431588fda79b9696f6953bd6a"
DESIGN_FILE = "contracts/v6_small_stair_supplement_expectations_v1.json"
DOMAIN = "small_stairs_supplement"
ROLES = {290929: "quick", 290930: "confirmation"}
CHANGED_FIELDS = ["name", "terrain_limits.stair_rise_m"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound(root, file, sha):
    path = (Path(root) / file).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve()) or digest(path) != sha:
        raise ValueError("Supplement input escaped the source root or changed its bound SHA")
    return json.loads(path.read_text())


def validate_cases(manifest, parent):
    """Exact two-field changes; paired reset keys, criteria and commands are immutable."""
    parents = {case["name"]: case for case in parent["evaluation"]["cases"]}
    expected = []
    for mm in (5, 10, 20):
        for original, name in (("rolling_stairs_3x04", f"rolling_stairs_3x{mm:03d}mm"),
                               ("stairs_down", f"stairs_down_3x{mm:03d}mm")):
            case = deepcopy(parents[original])
            case["name"] = name
            case["terrain_limits"]["stair_rise_m"] = mm / 1000
            expected.append(dict(parent_case=original, changed_fields=CHANGED_FIELDS, case=case))
    if (manifest.get("schema") != "prepared-six-small-stair-assessment-case-manifest-v1"
            or manifest.get("source_commit") != SOURCE_COMMIT
            or manifest.get("case_count") != 6 or manifest.get("cases") != expected
            or manifest.get("evaluation_seeds") != list(ROLES)
            or manifest.get("episodes_per_case") != parent["evaluation"]["episodes_per_case"]):
        raise ValueError("Supplement differs from its exact six paired two-field derivatives")
    return deepcopy([row["case"] for row in expected])


def supplement_settings(settings, manifest):
    result = deepcopy(settings)
    result["cases"] = deepcopy([row["case"] for row in manifest["cases"]])
    names = [case["name"] for case in result["cases"]]
    result["canonical_case_names"] = names
    for key in ("quick_case_names", "promotion_case_names", "retention_case_names", "protected_case_names"):
        if key in result:
            parents = set(settings[key])
            result[key] = [row["case"]["name"] for row in manifest["cases"] if row["parent_case"] in parents]
    return result


def derive_design(base_design, manifest, manifest_file, manifest_sha, base_binding):
    """Derive only from the original independent table, never from caller runtime values."""
    rows = {}
    for role in ROLES.values():
        row = deepcopy(base_design["stages"]["terrain"]["roles"][role])
        expected = row["contract"]
        expected["evaluation"] = supplement_settings(expected["evaluation"], manifest)
        expected["scene_groups"] = [dict(name=case["name"], fraction=1 / 6, terrain=[case["terrain"]])
                                    for case in expected["evaluation"]["cases"]]
        row["invocation"]["num_envs"] = 6 * manifest["episodes_per_case"]
        rows[role] = row
    return dict(schema="chassis-design-expectations-v1", status="frozen",
        scope="six_paired_small_stair_cases_only_no_training_or_qualification",
        base_source_commit=SOURCE_COMMIT, base_design_binding=deepcopy(base_binding),
        case_manifest=dict(file=manifest_file, sha256=manifest_sha),
        derivation=dict(case_changed_fields=CHANGED_FIELDS, old418_changed=False,
                        reset_seed_key="original_parent_preserved", training_role_present=False),
        stages=dict(terrain=dict(roles=rows)))


def authenticate_manifest(root, manifest_file, manifest_sha):
    # The previously frozen manifest is a fixed input, not a caller-selected catalogue.
    if manifest_file != MANIFEST_FILE or manifest_sha != MANIFEST_SHA:
        raise ValueError("Only the explicitly frozen six-case supplement manifest is supported")
    manifest = read_bound(root, manifest_file, manifest_sha)
    plan = read_bound(root, manifest["base_plan_file"], manifest["base_plan_sha256"])
    if plan.get("contract_id") != "v6-task-campaign-v25":
        raise ValueError("Supplement requires the unchanged v25 base plan")
    production.validate_plan(plan, Path(root))
    expected = [dict(domain=row["name"], contract=row["contract"], sha256=row["sha256"],
                     selected_case_definitions=len(set(row["required_case_names"]) |
                                                   set(row.get("monitor_case_names", []))))
                for row in plan["domains"]]
    if manifest.get("base_domains") != expected:
        raise ValueError("Supplement changes the unchanged six-domain base identities or old418")
    terrain = next(row for row in plan["domains"] if row["name"] == "terrain")
    if (manifest["parent_terrain_contract"], manifest["parent_terrain_contract_sha256"]) != (
            terrain["contract"], terrain["sha256"]):
        raise ValueError("Supplement parent terrain contract differs from v25")
    parent = read_bound(root, terrain["contract"], terrain["sha256"])
    validate_cases(manifest, parent)
    base_design = read_bound(root, parent["design_preflight"]["file"], parent["design_preflight"]["sha256"])
    return manifest, parent, base_design


def prepare_suite(root, *, manifest_file, manifest_sha, design_file, design_sha, role, seed):
    """Validate the entire original exam, then the separately frozen supplement role."""
    if ROLES.get(seed) != role:
        raise ValueError("Supplement only permits the original paired quick/confirmation seeds")
    manifest, parent, base_design = authenticate_manifest(root, manifest_file, manifest_sha)
    design = read_bound(root, design_file, design_sha)
    expected_design = derive_design(base_design, manifest, manifest_file, manifest_sha, parent["design_preflight"])
    if design != expected_design:
        raise ValueError("Supplement independent design differs from its exact frozen derivation")
    original = deepcopy(parent)
    # The worker retains the full historical catalogue; the independent role
    # table freezes the actual 28-case terrain examination within old418.
    original["evaluation"] = deepcopy(base_design["stages"]["terrain"]["roles"][role]["contract"]["evaluation"])
    fixed = production.fixed_evaluation_config(original, Path(root))
    control = load_chassis_control(Path(root) / parent["control_math_source"])
    base_receipt = design_preflight(fixed, control, Path(root), stage="terrain", role=role, seed=seed,
        num_envs=len(fixed["evaluation"]["cases"]) * fixed["evaluation"]["episodes_per_case"])
    result = deepcopy(fixed)
    result["evaluation"] = supplement_settings(fixed["evaluation"], manifest)
    result["scene_groups"] = deepcopy(design["stages"]["terrain"]["roles"][role]["contract"]["scene_groups"])
    result["design_preflight"] = dict(file=design_file, sha256=design_sha, stage="terrain")
    new_receipt = design_preflight(result, control, Path(root), stage="terrain", role=role, seed=seed,
        num_envs=6 * manifest["episodes_per_case"])
    return result, dict(base_design_preflight=base_receipt, supplement_design_preflight=new_receipt,
        manifest_file=manifest_file, manifest_sha256=manifest_sha, design_file=design_file,
        design_sha256=design_sha, parent_contract_file=manifest["parent_terrain_contract"],
        parent_contract_sha256=manifest["parent_terrain_contract_sha256"],
        scope="paired_rise_comparison_not_training_or_deployment_qualification",
        down_case_scope="original_stairs_down_manual_STEP_candidate_preserved")


def source_files(bindings):
    return ["src/wheeled_tasks/chassis/small_stair_supplement.py",
            bindings["manifest_file"], bindings["design_file"]]
