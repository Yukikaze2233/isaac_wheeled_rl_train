"""Resume accounting/dispatch tests; no simulator, GPU or training child."""
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4_common as common
import resume
import evaluation
from wheeled_algo import v40_job as job


def receipt(count, requested, status):
    budget = job.TrainingBudget()
    budget.completed_updates = count
    result = job._receipt(budget, requested, "iterations_completed" if status == "completed" else "sigterm", status, "verified")
    result["artifacts"] = [{"path": name, "size": 1, "sha256": "a" * 64} for name in sorted(job.ALLOWED_ARTIFACTS)]
    return result


@pytest.fixture
def pair():
    parent_completion = receipt(7735, 30000, "stopped")
    identity = {"contract_id": "fixture", "contract_sha256": "c" * 64, "seed": 44, "stage": "locomotion"}
    initial = {"mode": "scratch"}
    parent = {"checkpoint": "/parent/model_final.pt", "checkpoint_sha256": "a" * 64,
              "contract_sha256": identity["contract_sha256"], "completed_updates": 7735,
              "completion": parent_completion, "completion_sha256": hashlib.sha256(job.json_bytes(parent_completion)).hexdigest()}
    plan = {"initialization": "resume", "parent": parent, "updates": 22265, "contract_sha256": identity["contract_sha256"],
            "identity": identity, "expected_initialization": initial,
            "command": ["python", "train_v40.py", "--max-iterations", "22265", "--resume", parent["checkpoint"]]}
    manifest = {**identity, "asset_manifest_sha256": common.LEGACY_ASSET_SHA, "num_envs": 1024,
                "training_profile": "round4_full", "initialization": initial, "source_provenance": initial,
                "domain_randomization_report": {"passed": True},
                "training_curriculum": {"timebase": "successful_ppo_updates", "completed_updates": 30000, "state": {}},
                "resume_provenance": {"checkpoint_sha256": "a" * 64, "completed_updates": 7735,
                                      "semantics": "optimizer_and_curriculum_progress_not_bitwise_trajectory"}}
    return plan, manifest


def test_child_completion_is_honest_and_cumulative_is_separate(pair):
    plan, manifest = pair
    child = receipt(22265, 22265, "completed")
    original = copy.deepcopy(child)
    result = common.training_progress(plan, child, manifest)
    assert child == original
    assert result == {"prior_completed_updates": 7735, "invocation_completed_updates": 22265,
                      "cumulative_completed_updates": 30000, "formal_training_target_completed": True}


def test_partial_child_never_authorizes_final_evaluation(pair):
    plan, manifest = pair
    manifest["training_curriculum"]["completed_updates"] = 8735
    result = common.training_progress(plan, receipt(1000, 22265, "stopped"), manifest)
    assert result["cumulative_completed_updates"] == 8735 and not result["formal_training_target_completed"]


@pytest.mark.parametrize("bad", ["old_target", "parent_count", "parent_sha", "clock", "missing_resume", "finetune"])
def test_resume_evidence_cannot_bypass_existing_identity(pair, bad):
    plan, manifest = pair
    child = receipt(22265, 22265, "completed")
    if bad == "old_target":
        child = receipt(22265, 30000, "stopped")
    elif bad == "parent_count":
        plan["parent"]["completed_updates"] = 7734
    elif bad == "parent_sha":
        plan["parent"]["checkpoint_sha256"] = "b" * 64
    elif bad == "clock":
        manifest["training_curriculum"]["completed_updates"] = 22265
    elif bad == "missing_resume":
        del manifest["resume_provenance"]
    else:
        plan["command"] += ["--finetune", "/old.pt"]
    with pytest.raises(ValueError):
        common.training_progress(plan, child, manifest)


def test_original_scratch_gate_does_not_accept_resume(pair):
    plan, manifest = pair
    with pytest.raises(ValueError):
        common.verify_scratch(manifest, plan)
    plan.update(initialization="scratch", parent=None, updates=30000, command=["python", "train_v40.py"])
    del manifest["resume_provenance"]
    assert common.training_progress(plan, receipt(30000, 30000, "completed"), manifest)["formal_training_target_completed"]


def test_plan_uses_new_directory_and_remaining_updates(pair, monkeypatch, tmp_path):
    plan, _ = pair
    template = {**plan, "experiment": str(tmp_path), "command": ["python", "train", "--max-iterations", "30000",
                "--run-dir", "/old", "--usd-cache-dir", "/old-cache"], "learning_seconds": 48*3600}
    monkeypatch.setattr(resume, "scratch_plan", lambda _: copy.deepcopy(template))
    monkeypatch.setattr(resume, "check_parent", lambda *args: plan["parent"])
    args = SimpleNamespace(parent_checkpoint=Path("/parent/model_final.pt"), parent_sha256="a"*64)
    result = resume.build_plan(args)
    assert result["updates"] == 22265 and result["learning_seconds"] == 172800
    assert result["command"][result["command"].index("--max-iterations")+1] == "22265"
    assert result["run_dir"] != "/old" and Path(result["run_dir"]).name == "train"
    assert "--stop-at" not in result["command"]  # Fresh worker deadline, never the expired parent deadline.
    assert result["command"][-2:] == ["--resume", "/parent/model_final.pt"]


def test_dispatch_blocks_partial_then_binds_cumulative_final(pair, monkeypatch, tmp_path):
    plan, _ = pair
    plan.update(stage_root=str(tmp_path), run_dir=str(tmp_path / "train"), audit_dir=str(tmp_path / "audit"),
                ground_usd="/ground.usd", ground_record={}, requested_profile={"push": {"schedule": [[0, .5]]}},
                physics_model=common.PHYSICS_MODEL, physics_asset_manifest_sha256=common.LEGACY_ASSET_SHA,
                repaired_dynamics_used=False, explicit_user_authorization=True)
    monkeypatch.setattr(evaluation, "verify_training_run", lambda _: {"formal_training_target_completed": False})
    with pytest.raises(ValueError, match="cumulative"):
        evaluation.prepare(plan, [.5, 0])
    assert not (tmp_path / "evaluation").exists()
    monkeypatch.setattr(evaluation, "verify_training_run", lambda _: {"formal_training_target_completed": True,
        "prior_completed_updates": 7735, "invocation_completed_updates": 22265, "cumulative_completed_updates": 30000})
    monkeypatch.setattr(evaluation, "file_record", lambda _: {"size": 1, "sha256": "f"*64})
    _, _, request = evaluation.prepare(plan, [.5, 0])
    assert len(request["cases"]) == 12
    assert request["training_progress"]["invocation_completed_updates"] == 22265
    assert request["resume_parent_checkpoint_sha256"] == "a"*64
