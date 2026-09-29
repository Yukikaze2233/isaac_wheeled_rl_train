"""Future recovery evidence stays visible without protecting untaught skills."""
import json
from pathlib import Path

import pytest

from test_new_asset_blocks import Campaign, contract

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = {"recovery_flat": "S3", "recovery_step": "S4", "recovery_double": "S5D"}


@pytest.mark.parametrize("stage,eligible", [("S1", set()), ("S3", {"recovery_flat"}),
    ("S4", {"recovery_flat", "recovery_step"}), ("S5D", set(RECOVERY))])
@pytest.mark.parametrize("entry", ["baseline", "gained"])
def test_baseline_and_new_confirmed_passes_respect_recovery_introduction(
        tmp_path, monkeypatch, stage, eligible, entry):
    cfg = contract()
    cfg["curriculum_stage"] = stage
    cfg["budget_spec_file"] = "contracts/v6_recovery_curriculum_budget_v1.json"
    cases = cfg["evaluation"]["cases"]
    cases.extend({"name": name, "task": "survive", "recovery_case": True,
                  "introduced_stage": introduced, "skill": {"kind": "recovery_handoff"}}
                 for name, introduced in RECOVERY.items())
    cfg["evaluation"]["retention_case_names"].extend(RECOVERY)
    (tmp_path / "contract.json").write_text(json.dumps(cfg, indent=2))
    # Legacy skills retain their existing automatic protection behavior,
    # including jump_03, irrespective of recovery-specific introduction rules.
    legacy = {"stand", "jump_03"}
    baseline = legacy | (set(RECOVERY) if entry == "baseline" else set())
    run = Campaign(tmp_path, monkeypatch, [legacy | set(RECOVERY)], updates=2,
                   baseline_names=baseline)
    code, report = run.run()
    assert code == 0
    protected = set(report["protected_case_names"])
    assert protected == legacy | eligible
    initial = set(run.blocks.evaluation_settings["initial_passed_case_names"])
    assert initial == legacy | (eligible if entry == "baseline" else set())
    recorded = json.loads((run.args.run_dir / "evaluation_000/evaluation.json").read_text())["candidates"][0]
    assert set(RECOVERY) <= recorded["cases"].keys()
    assert all(recorded["cases"][name]["passed"] for name in RECOVERY)
    assert report["blocks"][0]["capability_gate"]["lost_parent_passes"] == []
    assert cfg["evaluation"]["promotion_case_names"] == ["stand", "jump_06"]
