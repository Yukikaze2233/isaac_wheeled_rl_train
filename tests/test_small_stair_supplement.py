"""CPU integrity/routing only; fake child reports never prove stair-climbing ability."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from test_integrated_checkpoint_evaluation import evaluation
from test_integrated_production import frozen, static_bundle
import evaluate_integrated_checkpoint as final_eval
import evaluate_small_stair_checkpoint as stair_eval
import run_task_campaign as dispatcher
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis import small_stair_supplement as supplement
from wheeled_tasks.chassis.evaluation import canonical_evaluation_layout
from wheeled_tasks.chassis.task import terrain_surfaces


@pytest.fixture(scope="module")
def definitions():
    manifest = json.loads((ROOT / supplement.MANIFEST_FILE).read_text())
    parent = json.loads((ROOT / manifest["parent_terrain_contract"]).read_text())
    return manifest, parent


@pytest.mark.parametrize("fault", ("extra_case", "threshold", "command", "mode", "reset_key", "terrain_seed", "top_array", "rise", "fields"))
def test_case_derivation_rejects_every_change_outside_the_two_frozen_fields(definitions, fault):
    manifest, parent = deepcopy(definitions)
    case = manifest["cases"][0]["case"]
    if fault == "extra_case":
        manifest["cases"].append(deepcopy(manifest["cases"][0]))
    elif fault == "threshold":
        case["height_mae_m_max"] = 1.
    elif fault == "command":
        case["command"][0] = .75
    elif fault == "mode":
        case["skill"]["mode"] = 4
    elif fault == "reset_key":
        case["reset_seed_key"] = case["name"]
    elif fault == "terrain_seed":
        case["terrain_seed"] += 1
    elif fault == "top_array":
        case["terrain_limits"]["stair_heights_m"] = [.005, .01, .015]
    elif fault == "rise":
        case["terrain_limits"]["stair_rise_m"] = .04
    else:
        manifest["cases"][0]["changed_fields"].append("command")
    with pytest.raises(ValueError, match="two-field"):
        supplement.validate_cases(manifest, parent)


def test_geometry_uses_original_colliders_and_preserves_down_manual_candidate(definitions):
    manifest, parent = definitions
    cases = supplement.validate_cases(manifest, parent)
    originals = {case["name"]: case for case in parent["evaluation"]["cases"]}
    for row, case in zip(manifest["cases"], cases):
        r = case["terrain_limits"]["stair_rise_m"]
        surfaces = terrain_surfaces(case["terrain"], case["terrain_limits"], 1.)
        assert [(s.x0, s.x1) for s in surfaces] == [(-4., 0.), (0., .6), (.6, 1.2), (1.2, 4.)]
        tops = [s.z0 for s in surfaces]
        assert tops == ([0., r, 2*r, 3*r] if case["terrain"] == "stairs" else [3*r, 2*r, r, 0.])
        assert case["reset_seed_key"] == originals[row["parent_case"]]["reset_seed_key"]
        assert case["terrain_seed"] == originals[row["parent_case"]]["terrain_seed"]
        if case["terrain"] == "stairs_down":
            assert case["skill"]["kind"] == "stairs_down" and case["skill"]["mode"] == 3
            assert not case.get("untriggered_up_case")
    settings = supplement.supplement_settings(parent["evaluation"], manifest)
    assert canonical_evaluation_layout(settings, 24) == (list(range(24)), 24)


@pytest.fixture
def small_evaluation(evaluation, monkeypatch):
    # Reuse native CPU Adam/RNG seals and canonical generated production fixtures.
    # v25 fixture roots differ from production, so ONLY this fixture's manifest
    # hash is substituted. Original case templates and validators remain real.
    f, original_job = evaluation
    production.write_bundle(f.source, source_root=str(f.source), run_root=str(f.run), version=25)
    plan_file = production.artifact_files(25)["plan"]
    plan = json.loads((f.source / plan_file).read_text())
    manifest = json.loads((ROOT / supplement.MANIFEST_FILE).read_text())
    manifest["base_plan_file"] = plan_file
    manifest["base_plan_sha256"] = supplement.digest(f.source / plan_file)
    manifest["base_domains"] = [dict(domain=row["name"], contract=row["contract"], sha256=row["sha256"],
        selected_case_definitions=len(set(row["required_case_names"]) | set(row.get("monitor_case_names", []))))
        for row in plan["domains"]]
    terrain = next(row for row in plan["domains"] if row["name"] == "terrain")
    manifest["parent_terrain_contract"] = terrain["contract"]
    manifest["parent_terrain_contract_sha256"] = terrain["sha256"]
    path = f.source / supplement.MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    dispatcher.write_json(path, manifest)
    monkeypatch.setattr(supplement, "MANIFEST_SHA", supplement.digest(path))
    parent = json.loads((f.source / terrain["contract"]).read_text())
    base_design = json.loads((f.source / parent["design_preflight"]["file"]).read_text())
    design = supplement.derive_design(base_design, manifest, supplement.MANIFEST_FILE, supplement.MANIFEST_SHA,
                                     parent["design_preflight"])
    dispatcher.write_json(f.source / supplement.DESIGN_FILE, design)
    helper = "src/wheeled_tasks/chassis/small_stair_supplement.py"
    shutil.copyfile(ROOT / helper, f.source / helper)
    # The fixture source snapshot never copies the pre-existing dirty interactive
    # preflight edit. Its retained native file is the exact committed16ed8f6 input.
    (f.source / "src/wheeled_tasks/chassis/preflight.py").write_bytes(subprocess.check_output(
        ["git", "show", supplement.SOURCE_COMMIT + ":src/wheeled_tasks/chassis/preflight.py"], cwd=ROOT))
    monkeypatch.setattr(stair_eval, "ROOT", f.source)
    args = deepcopy(original_job.args)
    args.contract = f.source / plan_file
    args.output = f.source.parent / "small_stairs_assessment"
    shutil.rmtree(args.output, ignore_errors=True)
    args.supplement_manifest = path
    args.supplement_manifest_sha256 = supplement.MANIFEST_SHA
    args.supplement_design = f.source / supplement.DESIGN_FILE
    args.supplement_design_sha256 = supplement.digest(args.supplement_design)
    return f, stair_eval.SmallStairCheckpointEvaluation(args)


def fake_children(job, monkeypatch, fault=None):
    calls = []

    class Child:
        def __init__(self, command, **kwargs):
            calls.append(command)
            seed = int(command[command.index("--seed") + 1])
            settings = job.evaluation_settings(job.supplement_row, seed)
            names = [case["name"] for case in settings["cases"]]
            config = job.suites[seed]
            candidate = dict(checkpoint_sha256=job.expected_sha, unfinished_episodes=0,
                actor_input_dim=35, actor_output_dim=6, cross_asset_evaluation=False,
                source_asset_manifest_sha256=config["asset_manifest_sha256"],
                evaluated_asset_manifest_sha256=config["asset_manifest_sha256"],
                cases={name: dict(passed=True) for name in names})
            record = dict(status="evaluated", candidates=[candidate], settings=settings,
                contract_sha256=job.supplement_row["sha256"], device="cpu",
                episodes_per_case=4, evaluated_case_names=names,
                script_sha256=supplement.digest(stair_eval.ROOT / "scripts/evaluate_chassis.py"),
                source_sha256=deepcopy(job.inputs["native_sources_sha256"][supplement.DOMAIN]),
                deterministic_actor=True, random_pushes=False, case_resampling=False,
                supplement=deepcopy(job.proofs[seed]), old418_changed=False, PPO_updates=0, qualification=None)
            if fault == "proof":
                record["supplement"]["base_design_preflight"]["status"] = "unverified"
            elif fault == "reset_key":
                record["settings"]["cases"][0]["reset_seed_key"] = names[0]
            elif fault == "source":
                record["source_sha256"].pop("src/wheeled_tasks/chassis/small_stair_supplement.py")
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            for name in record["source_sha256"]:
                target = output / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stair_eval.ROOT / name, target)
            dispatcher.write_json(output / "evaluation.json", record)
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(dispatcher.subprocess, "Popen", Child)
    return calls


def test_cpu_preflight_authenticates_both_independent_roles_without_native_or_accounts(small_evaluation, monkeypatch):
    f, job = small_evaluation
    monkeypatch.setattr(dispatcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("CPU preflight must not launch"))
    report = job.prepare()
    job.persist()
    assert report["native_case_definitions"] == 6 and report["old418_changed"] is False
    assert report["qualification"] is None and report["PPO_updates"] == 0
    for seed, role in supplement.ROLES.items():
        assert job.proofs[seed]["base_design_preflight"]["status"] == "static_passed"
        assert job.proofs[seed]["supplement_design_preflight"]["status"] == "static_passed"
        assert job.suites[seed]["design_preflight"]["sha256"] == job.args.supplement_design_sha256
        assert job.proofs[seed]["supplement_design_preflight"]["num_envs"] == 24
    assert f.before == {path: path.read_bytes() for path in f.before}


def test_native_routing_uses_two_seeds_one_actor_and_keeps_completed_cache(small_evaluation, monkeypatch):
    f, job = small_evaluation
    calls = fake_children(job, monkeypatch)
    report = job.run_assessment()
    assert len(calls) == 2 and list(report["domains"]) == [supplement.DOMAIN]
    assert {c[c.index("--checkpoint")+1] for c in calls} == {str(job.checkpoint)}
    assert [c[c.index("--seed")+1] for c in calls] == ["290929", "290930"]
    assert all(c[c.index("--supplement-manifest-sha256")+1] == job.args.supplement_manifest_sha256 for c in calls)
    assert f.before == {path: path.read_bytes() for path in f.before}
    retry = stair_eval.SmallStairCheckpointEvaluation(job.args)
    retry.run_assessment()
    assert len(calls) == 2
    assert report["qualification"] is report["deployment_checkpoint"] is None


@pytest.mark.parametrize("fault", ("proof", "reset_key", "source"))
def test_supplement_report_cannot_forge_design_seed_or_source_evidence(small_evaluation, monkeypatch, fault):
    _, job = small_evaluation
    calls = fake_children(job, monkeypatch, fault)
    with pytest.raises(ValueError):
        job.run_assessment()
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ("base", "design", "asset", "model"))
def test_changed_frozen_input_rejects_before_any_child(small_evaluation, monkeypatch, fault):
    f, job = small_evaluation
    monkeypatch.setattr(dispatcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("tamper must reject before native"))
    if fault == "base":
        path = f.source / "contracts/v6_task_campaign_terrain_worker_v25.json"
    elif fault == "design":
        path = job.args.supplement_design
    elif fault == "model":
        path = job.checkpoint
    else:
        source = json.loads(job.checkpoint.with_suffix(".contract.json").read_text())
        bundle = f.source / source["asset_directory"]
        manifest = json.loads((bundle / "manifest.json").read_text())
        path = bundle / next(iter(manifest["files_sha256"]))
    before = path.read_bytes()
    try:
        path.write_bytes(before + b"\nchanged input\n")
        with pytest.raises((ValueError, json.JSONDecodeError)):
            job.prepare()
    finally:
        path.write_bytes(before)


def test_production_lock_and_live_worker_forbid_supplement_children(small_evaluation, monkeypatch):
    _, job = small_evaluation
    monkeypatch.setattr(dispatcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("busy PPO must reject"))
    with job.lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            job.run_assessment()
    monkeypatch.setattr(final_eval.TaskCampaign, "assert_no_live_workers",
                        lambda _: (_ for _ in ()).throw(ValueError("live PPO")))
    with pytest.raises(ValueError, match="live PPO"):
        job.run_assessment()


def test_original_exact_evaluation_still_rejects_supplement_injection(small_evaluation):
    f, job = small_evaluation
    manifest = json.loads(job.args.supplement_manifest.read_text())
    parent = json.loads((f.source / manifest["parent_terrain_contract"]).read_text())
    parent["evaluation"]["cases"] = [row["case"] for row in manifest["cases"]]
    with pytest.raises(ValueError):
        production.fixed_evaluation_config(parent, f.source)


def test_rehashing_a_weakened_independent_design_still_rejects(small_evaluation):
    _, job = small_evaluation
    path = job.args.supplement_design
    before = path.read_bytes()
    try:
        design = json.loads(before)
        design["stages"]["terrain"]["roles"]["quick"]["contract"]["evaluation"]["cases"][0]["height_mae_m_max"] = 1.
        dispatcher.write_json(path, design)
        job.bindings["design_sha256"] = supplement.digest(path)
        with pytest.raises(ValueError, match="exact frozen derivation"):
            job.prepare()
    finally:
        path.write_bytes(before)


def test_default_native_command_and_six_domain_hooks_preserve_the_old_entry(small_evaluation):
    f, job = small_evaluation
    row = job.plan["domains"][0]
    names = sorted(set(row["required_case_names"]) | set(row.get("monitor_case_names", [])))
    output = job.root / "default_command_only"
    expected = [sys.executable, "-B", str(f.source / "scripts/evaluate_chassis.py"),
        "--contract", str(f.source / row["contract"]), "--checkpoint", str(job.checkpoint),
        "--output", str(output), "--device", "cpu", "--seed", "290929",
        "--role", "quick", "--no-traces", "--cases", *names]
    assert dispatcher.TaskCampaign.evaluation_command(job, row, job.checkpoint, output, 290929, names) == expected
    assert final_eval.IntegratedCheckpointEvaluation.assessment_rows(job) == job.plan["domains"]


@pytest.mark.parametrize("seed,role", ((290929,"train"), (290930,"quick"), (1,"quick")))
def test_no_training_role_or_nonpaired_seed(seed, role):
    with pytest.raises(ValueError, match="paired"):
        supplement.prepare_suite(ROOT, manifest_file=supplement.MANIFEST_FILE, manifest_sha=supplement.MANIFEST_SHA,
            design_file=supplement.DESIGN_FILE, design_sha=supplement.digest(ROOT / supplement.DESIGN_FILE),
            role=role, seed=seed)
