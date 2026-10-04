"""Complete-capture stdout salvage never grants completed or qualified status."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).with_name("summarize_comparison.py")
SPEC = importlib.util.spec_from_file_location("paired_comparison", PATH)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)

ERROR = ('Traceback (most recent call last):\n  File "/home/kaiser/run/source_122d8db63f37/scripts/evaluate_chassis.py", '
         'line 273, in main\n    print("V5_FIXED_EVAL", json.dumps(result, allow_nan=False), flush=True)\n'
         'BrokenPipeError: [Errno 32] Broken pipe\n')


@pytest.fixture
def captured(tmp_path, monkeypatch):
    cases = [{"name": f"case_{i}", "command": [0., 0., .305]} for i in range(3)]
    contract = {"actor_dim": 35, "action_dim": 6, "policy_dt": .02,
                "evaluation": {"episodes_per_case": 4, "cases": cases}}
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    sources = {}
    for name in ("scripts/evaluate_chassis.py", "src/wheeled_tasks/chassis/env.py",
                 "src/wheeled_tasks/chassis/eval_env.py", "src/wheeled_tasks/chassis/evaluation.py",
                 "src/wheeled_tasks/chassis/rewards.py", "src/wheeled_tasks/chassis/preflight.py"):
        path = tmp_path / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        sources[name] = comparison.sha(path)
    report = {"status": "failed", "error": ERROR, "candidates": [],
              "contract_sha256": comparison.sha(tmp_path / "contract.json"),
              "asset_manifest_sha256": "original_asset", "script_sha256": sources["scripts/evaluate_chassis.py"],
              "settings": deepcopy(contract["evaluation"]), "episodes_per_case": 4,
              "evaluated_case_names": [case["name"] for case in cases], "source_sha256": sources}
    for index in range(2):
        candidate = {"checkpoint_sha256": comparison.CHECKPOINT_SHA256[index],
                     "unfinished_episodes": 0, "policy_ticks": 1600, "actor_input_dim": 35,
                     "actor_output_dim": 6, "evaluated_asset_manifest_sha256": "original_asset",
                     "cases": {case["name"]: {"episodes": 4, "requested_episodes": 4, "passed": True}
                               for case in cases}}
        report["candidates"].append(candidate)
        (tmp_path / f"candidate_{index:02d}.json").write_text(json.dumps(candidate))
    monkeypatch.setattr(comparison, "SALVAGE_SCRIPT_SHA256", report["script_sha256"])
    monkeypatch.setattr(comparison, "SALVAGE_CONTRACT_SHA256", report["contract_sha256"])
    monkeypatch.setattr(comparison, "SALVAGE_CASE_ORDER_SHA256", __import__("hashlib").sha256(
        json.dumps(report["evaluated_case_names"], separators=(",", ":")).encode()).hexdigest())
    return tmp_path, report, contract


def test_only_explicit_stdout_salvage_is_diagnostic_and_preserves_original(captured):
    directory, report, contract = captured
    original = deepcopy(report)
    with pytest.raises(ValueError, match="explicit"):
        comparison.validate_capture(directory, report, contract)
    validation = comparison.validate_capture(directory, report, contract, salvage=True)
    assert report == original and report["status"] == "failed" and report["error"] == ERROR
    assert validation["status"] == "complete_diagnostic_capture_after_final_stdout_failure"
    assert validation["original_evaluation_status"] == "failed"
    assert validation["qualification_awarded"] is False


@pytest.mark.parametrize("error", [ERROR.replace("Errno 32", "Errno 5"),
    ERROR.replace("line 273", "line 270"), ERROR.replace('print("V5_FIXED_EVAL"', 'np.save("V5_FIXED_EVAL"'),
    ERROR + "ValueError: incomplete\n", "BrokenPipeError: [Errno 32] Broken pipe\n"])
def test_other_failures_are_not_salvaged(captured, error):
    directory, report, contract = captured
    report["error"] = error
    with pytest.raises(ValueError, match="stdout"):
        comparison.validate_capture(directory, report, contract, salvage=True)


@pytest.mark.parametrize("change", ["single_candidate", "unfinished", "wrong_ticks", "wrong_identity",
    "missing_case", "partial_replicas", "changed_candidate_json", "changed_source",
    "changed_contract", "changed_case_order", "changed_settings", "nonfinite"])
def test_partial_or_inconsistent_capture_is_rejected(captured, change):
    directory, report, contract = captured
    if change == "single_candidate": report["candidates"].pop()
    elif change == "unfinished": report["candidates"][1]["unfinished_episodes"] = 1
    elif change == "wrong_ticks": report["candidates"][1]["policy_ticks"] = 1599
    elif change == "wrong_identity": report["candidates"][1]["checkpoint_sha256"] = "wrong"
    elif change == "missing_case": report["candidates"][1]["cases"].pop("case_2")
    elif change == "partial_replicas": report["candidates"][1]["cases"]["case_1"]["episodes"] = 3
    elif change == "changed_candidate_json": (directory / "candidate_01.json").write_text("{}")
    elif change == "changed_source": (directory / "source/scripts/evaluate_chassis.py").write_text("changed")
    elif change == "changed_contract": (directory / "contract.json").write_text("{}")
    elif change == "changed_case_order": report["evaluated_case_names"].reverse()
    elif change == "changed_settings": report["settings"]["episodes_per_case"] = 3
    elif change == "nonfinite": report["candidates"][1]["cases"]["case_0"]["metric"] = float("nan")
    if change in {"unfinished", "wrong_ticks", "wrong_identity", "missing_case", "partial_replicas"}:
        (directory / "candidate_01.json").write_text(json.dumps(report["candidates"][1]))
    with pytest.raises(ValueError):
        comparison.validate_capture(directory, report, contract, salvage=True)


def arrays():
    result = {key: np.zeros((8, 3, *tail), dtype=float) for key, tail in comparison.CAPTURE_SHAPES.items()}
    for key in ("active", "done", "terminated"):
        result[key] = np.zeros((8, 3), dtype=bool)
    result["active"][:] = True
    result["done"][-1] = True
    result["episode_ticks"] = np.tile(np.arange(1, 9, dtype=np.int64)[:, None], (1, 3))
    result["body_poses"] = np.zeros((2, 3, 2, 7))
    result["body_poses"][..., 6] = 1.
    return result


def test_all_capture_arrays_have_finite_complete_and_terminal_alignment():
    comparison.validate_trace_arrays(arrays(), 8, 3, 2, salvaged=True)


@pytest.mark.parametrize("change", ["missing", "short", "extra_pose", "nonfinite_inactive",
    "no_terminal", "reopened", "clock_gap", "zero_quaternion", "numeric_mask"])
def test_partial_nonfinite_or_misaligned_arrays_are_rejected(change):
    captured = arrays()
    if change == "missing": captured.pop("reference_ax")
    elif change == "short": captured["values"] = captured["values"][:-1]
    elif change == "extra_pose": captured["body_poses"] = np.zeros((3, 3, 2, 7))
    elif change == "nonfinite_inactive": captured["values"][-1, 2, 7] = float("nan")
    elif change == "no_terminal": captured["done"][-1, 0] = False
    elif change == "reopened": captured["active"][2, 0] = False
    elif change == "clock_gap": captured["episode_ticks"][2, 0] += 1
    elif change == "zero_quaternion": captured["body_poses"][0, 0, 0] = 0.
    elif change == "numeric_mask": captured["active"] = captured["active"].astype(float)
    with pytest.raises(ValueError):
        comparison.validate_trace_arrays(captured, 8, 3, 2, salvaged=True)


def test_no_second_stop_success_is_inferred_from_boundary_zero():
    summary = {"capture_validation": {"original_evaluation_status": "failed", "status": "diagnostic"},
               "case_order": ["start_stop_1"], "limitations": [], "candidates": {}}
    for label in comparison.LABELS:
        summary["candidates"][label] = {"all_replicas": {"start_stop_1": {
            "settled_stop_speed": 0., "settled_stop_metric_valid": False, "passed": False}},
            "representatives": {"start_stop_1": {"phases": {phase: {"policy_samples": 0}
                for phase in comparison.PHASES}}}}
    rendered = comparison.markdown(summary)
    assert "| — | False |" in rendered
    assert "原运行状态：`failed`" in rendered
