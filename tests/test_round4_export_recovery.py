"""Recovery gates must preserve failed training evidence and the evaluation matrix."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recovery_test", ROOT / "scripts/round4/export_recovery.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def evidence():
    return {"arithmetic": {
        "internal_precision": "float64-internal", "input_output_precision": "float32",
        "reference": "native PyTorch FP64 ELU, FP32 output", "checkpoint_parameters_changed": False,
        "deployment_provider": "CPUExecutionProvider"}, "validation": {
            "passed": True, "sample_count": 4105, "seed": 40, "atol": 1e-6, "rtol": 1e-5,
            "provider": "CPUExecutionProvider", "max_abs_error": 0., "max_relative_error": 0.,
            "samples": [{"sample": i, "max_abs_error": 0., "max_relative_error": 0.} for i in range(4105)],
            "original_float32_comparison": {"sample_count": 4105, "failed_sample_count": 97,
                "first_nine_failed_samples": [3], "max_abs_error": .000244140625}}}


def test_fp32_mismatch_is_disclosed_not_mislabelled_as_fp64_failure():
    recovery.validate_precision_evidence(evidence())


@pytest.mark.parametrize("field,value", [
    ("passed", False), ("sample_count", 9), ("seed", 41), ("atol", 1e-4),
    ("rtol", 1e-3), ("provider", "CUDAExecutionProvider"), ("samples", []),
    ("max_abs_error", float("nan")), ("original_float32_comparison", {}),
])
def test_precision_evidence_rejects_weakened_or_missing_validation(field, value):
    report = evidence()
    report["validation"][field] = value
    with pytest.raises(ValueError):
        recovery.validate_precision_evidence(report)


def test_precision_evidence_rejects_changed_parameters():
    report = evidence()
    report["arithmetic"]["checkpoint_parameters_changed"] = True
    with pytest.raises(ValueError):
        recovery.validate_precision_evidence(report)


@pytest.mark.parametrize("status,reason,fulfilled,failed", [
    ("training_failed", "training_exception", True, True),
    ("export_failed", "max_runtime", False, False),
    ("completed", "iterations_completed", True, False),
])
def test_recovery_rejects_non_export_only_completion(tmp_path, monkeypatch, status, reason, fulfilled, failed):
    plan = {"run_dir": str(tmp_path / "train"), "stage_root": str(tmp_path),
            "experiment": str(tmp_path), "repo": str(tmp_path / "code"), "git_commit": "a" * 40}
    (tmp_path / "train").mkdir()
    (tmp_path / "plan.json").write_bytes(recovery.json_bytes(plan))
    completion = {"status": status, "update_failed": failed, "stop_reason": reason,
                  "requested_iterations_completed": fulfilled}
    (tmp_path / "train/completion.json").write_bytes(recovery.json_bytes(completion))
    before = (tmp_path / "train/completion.json").read_bytes()
    monkeypatch.setattr(recovery, "verify_snapshot", lambda _: {"code_directory_name": "code", "git_commit": "a" * 40})
    monkeypatch.setattr(recovery, "validate_completion", lambda value: value)
    with pytest.raises(ValueError, match="export-only failure"):
        recovery.verify_failed_training(tmp_path / "plan.json")
    assert (tmp_path / "train/completion.json").read_bytes() == before


def test_injected_verifier_failure_is_reported_before_simulation(tmp_path):
    import evaluate_policy
    request = tmp_path / "request.json"
    request.write_text("{}")
    calls = []

    def reject(path):
        calls.append(path)
        raise ValueError("original checkpoint identity mismatch")

    output = tmp_path / "reports"
    assert evaluate_policy.main(["--request", str(request), "--output", str(output)], input_verifier=reject) == 2
    assert calls == [request]
    result = recovery.strict_json((output / "result.json").read_bytes())
    assert not result["evaluation_success"] and not result["policy_quality_verified"]
    assert "checkpoint identity mismatch" in result["error"]
