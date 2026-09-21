"""Integrated budgets, immutable checkpoints and curriculum schedule invariants."""
import json
from pathlib import Path
import sys

import pytest

from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count, resolve_plan, stage_contract
from wheeled_tasks.chassis.skill_commands import scheduled_command

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from chassis_checkpoints import seal_checkpoint
from chassis_batch_export import export_ready
from sync_chassis_batches import recover_batch


def test_integrated_plan_keeps_all_skills_and_low_speed_regressions():
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v5_scut35_integrated_v1.json"), load)
    assert sum(s["updates"] for s in plan["stages"]) == 52000
    assert len(plan["stages"]) == 7
    seen = set()
    for recipe in plan["stages"]:
        config = stage_contract(load(plan["base_contract"]), plan, recipe, 4096)
        assert config["actor_dim"] == 35 and config["policy_dt"] == .02 and config["physics_dt"] == .005
        assert config["total_updates"] == recipe["updates"]
        assert sum(g["fraction"] for g in config["scene_groups"]) == pytest.approx(1.)
        assert seen <= set(config["skill_specs"])
        seen.update(recipe["new_skills"])
        cases = config["evaluation"]["cases"]
        assert len({c["name"] for c in cases}) == len(cases)
        assert {"height_hold_low", "height_hold_high", "forward_05", "backward_05"} <= {c["name"] for c in cases}
        assert max(c["episode_seconds"] for c in cases) <= config["evaluation"]["episode_seconds"]
    assert {r["name"] for r in plan["skill_catalog"] if "skill" in r} == seen


def test_schedule_uses_reference_samples_and_reaches_exact_target():
    schedule = {"start": 2000, "ramp": 1000, "initial": [.5, 0., .305]}
    assert scheduled_command([3., 0., .305], schedule, 1999)[0] == .5
    assert scheduled_command([3., 0., .305], schedule, 2500)[0] == 1.75
    assert scheduled_command([3., 0., .305], schedule, 4000)[0] == 3.


def test_live_checkpoint_recovery_and_incomplete_checkpoint_exclusion(tmp_path):
    root = tmp_path / "run"
    block = root / "train/stage_00_flat/block_000"
    block.mkdir(parents=True)
    contract = block / "contract.json"
    contract.write_text('{"actor_dim":35}')
    snapshot = seal_checkpoint(block, 100, lambda path: Path(path).write_bytes(b"model-and-optimizer"),
                               contract, {"successful_updates": 100})
    assert checkpoint_update_count(snapshot / "model.pt") == 100
    partial = snapshot.parent / "update_00000200.pending"
    partial.mkdir()
    (partial / "completion.json").write_text('{"status":"checkpoint_sealed"}')
    index = export_ready(root)
    assert len(index["batches"]) == 1 and not index["run_finished"]
    receipt = index["batches"][0]
    assert receipt["kind"] == "checkpoint_snapshot"
    assert receipt["model_role"] == "candidate_requires_evaluation"
    recover_batch(root / receipt["archive"], receipt, tmp_path / "recovered")
    assert (tmp_path / "recovered" / snapshot.relative_to(root) / "model.pt").read_bytes() == b"model-and-optimizer"
    with pytest.raises(FileExistsError):
        seal_checkpoint(block, 100, lambda _: None, contract, {})


def test_resume_reads_finalized_block_count_not_live_progress(tmp_path):
    (tmp_path / "completion.json").write_text(json.dumps({"status": "stopped", "parent_updates": 500,
                                                        "successful_updates": 27}))
    assert checkpoint_update_count(tmp_path / "model_final.pt") == 527
    with pytest.raises(ValueError, match="immutable"):
        checkpoint_update_count(tmp_path / "model_501.pt")
