"""Regenerate only the pre-entry superseding v27 graph in an isolated tree.

No checkpoint, account, remote write, freeze, launch, simulation or PPO is made.
The candidate code must already be committed and its inherited contracts clean.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

BASE = "81ad124554c4ab45cc274e2d482d46c76ba125b0"
CODE = "0f4b5f673e5f7e2cb382a8fcbd5766b0e13d8a9b"
TREE = Path("/home/yukikaze/Documents/workspace/robot_rl/.worktrees/stepjump-landing-goal-20261009")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-root", "target-source-root", "target-run-root", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    root = args.candidate_root.resolve(strict=True)
    for value in (args.target_source_root, args.target_run_root, args.report):
        if not value.is_absolute() or ".." in value.parts or value.is_symlink():
            parser.error("Explicit absolute non-symlink paths required")
    if (not re.fullmatch(r"[0-9a-f]{40}", args.code_commit)
            or args.code_commit != CODE or root != TREE.resolve(strict=True) or args.report.exists()
            or args.report.resolve().is_relative_to(root)
            or args.target_source_root.name != "source_serial_dispatch031_20261009"
            or args.target_source_root.parent != args.target_run_root.parent):
        parser.error("New isolated frozen code/source/report identity required")

    def git(*parts):
        return subprocess.check_output(["git", "-C", str(root), *parts], text=True).strip()

    if git("rev-parse", "HEAD") != args.code_commit or git("status", "--porcelain"):
        raise ValueError("Candidate code is not the explicit clean committed tree")
    subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", BASE,
                    args.code_commit], check=True)
    if git("diff", "--name-only", BASE, args.code_commit, "--", "contracts"):
        raise ValueError("Candidate must inherit all 81ad contracts before regeneration")
    sys.path.insert(0, str(root / "src"))
    from wheeled_tasks.chassis import integrated_production as production
    from wheeled_tasks.chassis.integrated_lineage import PHYSICAL_KEYS

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def read(name):
        return json.loads((root / name).read_text())

    old_all = {str(p.relative_to(root)): digest(p)
               for p in (root / "contracts").rglob("*") if p.is_file()}
    old_36 = {}
    for version in (24, 25, 26):
        files = production.artifact_files(version)
        names = [files[k] for k in ("recipe", "lineage", "plan", "manifest")]
        names += [f for domain in ("speed", "terrain", "jump", "stepjump") for f in (
            production.worker_file(domain, version=version), production.design_file(domain, version=version))]
        old_36.update({name: old_all[name] for name in names})
    assert len(old_36) == 36
    files = production.artifact_files(27)
    superseded = {k: dict(file=files[k], sha256=old_all[files[k]])
                  for k in ("recipe", "lineage", "plan", "manifest")}
    before_recipe = read(files["recipe"])
    before_workers = {d: read(production.worker_file(d, version=27))
                      for d in ("speed", "terrain", "jump", "stepjump")}
    old_manifest = read(files["lineage"])
    if (str(args.target_run_root) != old_manifest["run_root"]
            or str(args.target_source_root) == old_manifest["source_root"]):
        raise ValueError("Superseding source must retain the exact run and use a new prefix")

    result = production.write_bundle(root, source_root=str(args.target_source_root),
                                    run_root=str(args.target_run_root), version=27)
    changed_set = set(result["files"])
    if len(changed_set) != 12:
        raise ValueError("Expected exactly the existing 12 v27 artifact paths")
    if any(digest(root / name) != sha for name, sha in old_all.items() if name not in changed_set):
        raise ValueError("Generation changed a legacy contract")
    if {str(p.relative_to(root)) for p in (root / "contracts").rglob("*") if p.is_file()} != set(old_all):
        raise ValueError("Generation added or removed an unplanned contract")
    plan = read(files["plan"])
    manifest = production.validate_plan(plan, root)
    assert plan["total_ceiling"] == 121000 and plan["target_num_envs"] == 4096
    assert manifest["source_root"] == str(args.target_source_root)
    recipe = read(files["recipe"])
    assert recipe["domains"]["speed"] == read(production.artifact_files(26)["recipe"])["domains"]["speed"]
    compatible = deepcopy(recipe)
    for domain in ("jump", "stepjump"):
        assert compatible["domains"][domain]["patch"].pop("step_jump_landing_stop_target") is True
    assert compatible == before_recipe, "Changes exceed the explicit landing-stop patch"
    checks = {}
    for domain in ("speed", "terrain", "jump", "stepjump"):
        worker = read(production.worker_file(domain, version=27))
        assert production.validate_static_worker(worker, root) == domain
        assert {k: worker.get(k) for k in PHYSICAL_KEYS} == {k: before_workers[domain].get(k) for k in PHYSICAL_KEYS}
        assert (worker["actor_dim"], worker["critic_dim"], worker["action_dim"]) == (35, 81, 6)
        assert (worker["policy_dt"], worker["physics_dt"]) == (.02, .005)
        effective = production.static_training_config(worker, root)
        design = read(production.design_file(domain, version=27))
        row = design["stages"][domain]["roles"]["train"]
        on = domain in ("jump", "stepjump")
        assert effective.get("step_jump_landing_stop_target", False) is on
        assert row["contract"].get("step_jump_landing_stop_target", False) is on
        if on:
            from wheeled_tasks.chassis.jump_landing_revision import runtime_expectation
            assert row["runtime"]["jump_landing"] == runtime_expectation(effective)
            assert row["runtime"]["jump_landing"]["step_jump_landing_stop_target"]["enabled"] is True
        inventory = production.production_source_files(worker, root=root)
        assert {"src/wheeled_tasks/chassis/jump_landing.py", "src/wheeled_tasks/chassis/jump_landing_revision.py"} <= set(inventory)
        checks[domain] = dict(step_jump_landing_stop_target=on,
            recovery_phase_audit=effective.get("recovery_phase_audit", False), source_inventory_verified=True)
    assert set(git("diff", "--name-only").splitlines()) <= changed_set
    report = dict(schema="source031-v27-pre-entry-static-contract-generation-v1",
        status="static_canonical_contracts_generated", source_code_commit=args.code_commit,
        target_source=str(args.target_source_root), target_run=str(args.target_run_root),
        files=[dict(path=name, sha256=digest(root / name)) for name in result["files"]],
        effective_checks=checks, preserved_old_files=old_36,
        legacy_contracts_preserved=len(old_all) - len(changed_set),
        supersedes_before_entry=dict(source_commit=BASE, static_graph=superseded,
            eligibility="must_verify_no_81ad_entry_claim_or_downstream_work_before_observer_replacement"),
        remote_writes=0, signals=0, PPO_updates=0, actual_admission=None,
        future_checkpoint=None, freeze_ready=False, qualification=None)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(status=report["status"], artifacts=len(changed_set),
                         old_preserved=len(old_36), legacy_contracts_preserved=report["legacy_contracts_preserved"])))


if __name__ == "__main__":
    main()
