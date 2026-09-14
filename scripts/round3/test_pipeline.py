"""CPU-only pipeline isolation, periodic consistency, resource guards and frozen handoff tests."""
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as common
import pipeline_handoff as handoff
import pipeline_pilot as pilot
from evaluate_finished import FLAGS


def parent_fixture(root):
    contract = {"round3": {"stage": "A"}}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"contract_sha256": digest, "runtime": {"checkpoint_format": "rsl_rl_5_split_mlp", "versions": {"rsl-rl-lib": "5.5.1"}}}
    for name, value in (("contract.json", contract), ("run_manifest.json", manifest),
                        ("source_hashes.json", {}), ("asset_manifest.json", {}), ("agent_config.json", {})):
        (root / name).write_text(json.dumps(value))
    for index in (100, 200):
        path = root / f"model_{index}.pt"
        path.write_bytes(f"fixture checkpoint {index}".encode())
        os.utime(path, (time.time() - 30, time.time() - 30))


def test_periodic_capture_retains_filename_and_does_not_need_completion(tmp_path):
    parent_fixture(tmp_path)
    record = common.capture_parent(tmp_path, settle_seconds=0)
    assert record["checkpoint"] == "model_200.pt"
    assert len(record["files"]) == 6 and "agent_config.json" in record["files"]
    assert "completion.json" not in record["files"] and record["completion_required"] is False
    assert record["source_checkpoint_sha256"] == hashlib.sha256((tmp_path / "model_200.pt").read_bytes()).hexdigest()


def test_capture_rejects_checkpoint_changed_between_reads(tmp_path, monkeypatch):
    parent_fixture(tmp_path)
    monkeypatch.setattr(time, "sleep", lambda _: (tmp_path / "model_200.pt").write_bytes(b"still writing"))
    with pytest.raises(ValueError, match="still changing"):
        common.capture_parent(tmp_path, "model_200.pt")


def test_capture_rejects_linked_parent(tmp_path):
    parent_fixture(tmp_path)
    os.link(tmp_path / "model_200.pt", tmp_path / "other.pt")
    with pytest.raises(ValueError, match="nonlinked"):
        common.capture_parent(tmp_path, "model_200.pt", 0)


def test_resource_admission_and_reserve_are_separate():
    sample = {"mem_available_bytes": 6 * pilot.GiB, "gpu_free_mib": 4096}
    assert pilot.admission(sample) and pilot.pressure(sample) is None
    assert not pilot.admission({**sample, "mem_available_bytes": 5 * pilot.GiB})
    assert pilot.pressure({**sample, "mem_available_bytes": pilot.GiB}) == "host_memory_reserve"
    assert pilot.pressure({**sample, "gpu_free_mib": 512}) == "gpu_memory_reserve"
    assert not pilot.admission({**sample, "gpu_free_mib": None})


def test_resource_stop_signals_only_owned_pilot(tmp_path):
    calls = []
    child = SimpleNamespace(pid=321, send_signal=lambda sig: calls.append((321, sig)))
    untouched_a = SimpleNamespace(pid=446292, send_signal=lambda _: pytest.fail("A must never be signalled"))
    pilot.request_pilot_stop(child, tmp_path, "host_memory_reserve", {"mem_available_bytes": 1})
    assert calls == [(321, signal.SIGTERM)] and untouched_a.pid == 446292
    assert json.loads((tmp_path / "soft-stop.json").read_text())["pid"] == 321


def test_stage_transfer_identity_rejects_other_parent_or_optimizer():
    plan = {"source_checkpoint_sha256": "checkpoint", "contract_sha256": "target", "source_filename": "model_200.pt"}
    lineage = {"mode": "stage_transfer", "source_stage": "A", "target_stage": "B1", "parent_checkpoint_sha256": "checkpoint",
               "parent_contract_sha256": common.A_SHA, "target_contract_sha256": "target", "source_filename": "model_200.pt",
               "optimizer_reset": True, "std_preserved": True, "initial_iteration": 0}
    common.verify_transfer({"source_provenance": lineage}, plan)
    for key, value in (("parent_checkpoint_sha256", "other"), ("optimizer_reset", False), ("std_preserved", False)):
        with pytest.raises(ValueError):
            common.verify_transfer({"source_provenance": {**lineage, key: value}}, plan)


def test_handoff_never_promotes_from_pilot_reward(tmp_path, monkeypatch):
    audit = tmp_path / "audit"; audit.mkdir()
    run = tmp_path / "train"; run.mkdir()
    (run / "run_manifest.json").write_text("{}")
    (audit / "worker.status.json").write_text(json.dumps({"status": "completed", "resource_stop": None, "reward": 999999}))
    a = {"run_dir": "/A/train", "contract_sha256": common.A_SHA}
    b = {"run_dir": str(run), "audit_dir": str(audit), "source_run": a["run_dir"]}
    a_path, b_path = tmp_path / "a.json", tmp_path / "b.json"
    a_path.write_text(json.dumps(a)); b_path.write_text(json.dumps(b))
    monkeypatch.setattr(handoff, "verify_run", lambda *_: {"completed_updates": 500})
    monkeypatch.setattr(handoff, "verify_transfer", lambda *_: None)
    monkeypatch.setattr(handoff, "training_alive", lambda _: True)
    assert handoff.decide(a_path, tmp_path, b_path)["next_action"] == "waiting_A_final"
    monkeypatch.setattr(handoff, "training_alive", lambda _: False)
    monkeypatch.setattr(handoff, "verified_final", lambda _: {"artifacts": [{"path": "model_final.pt", "sha256": "A-final"}]})
    monkeypatch.setattr(handoff, "verified_evaluation", lambda *_: {"all_heights_pass": False})
    assert handoff.decide(a_path, tmp_path, b_path)["next_action"] == "continue_A_optimization"
    monkeypatch.setattr(handoff, "verified_evaluation", lambda *_: {"all_heights_pass": True})
    result = handoff.decide(a_path, tmp_path, b_path)
    assert result["next_action"] == "prepare_B1_main_from_A_final" and result["B_main_started"] is False
    assert result["a_final_checkpoint_sha256"] == "A-final"


@pytest.fixture
def evaluation_report(tmp_path):
    """Use the final evaluator's report schema and full 60-second replay clock."""
    run = tmp_path / "a-train"
    run.mkdir()
    (run / "policy.onnx").write_bytes(b"fixture final policy")
    plan = {"run_dir": str(run), "contract_sha256": common.A_SHA}
    plan_path = tmp_path / "a-plan.json"
    plan_path.write_text(json.dumps(plan))
    directory = tmp_path / "final-evaluation"
    directory.mkdir()
    policy_sha = common.file_record(run / "policy.onnx")["sha256"]
    plan_sha = common.file_record(plan_path)["sha256"]
    result = {
        "training_plan_sha256": plan_sha,
        "contract_sha256": common.A_SHA,
        "policy_sha256": policy_sha,
        "criteria": handoff.CRITERIA,
        "classification": "final_policy_research_evaluation",
        "state": "evaluation_completed",
        "cases": [],
    }
    for height in (.29, .30, .31, .32):
        case = directory / f"h{round(height * 100):03d}"
        case.mkdir()
        rows = []
        for step in range(1, 6001):
            tick = (step - 1) % 2000 + 1
            row = {
                "step": step, "policy_tick": step, "sample_kind": "pre_reset",
                "episode_step": tick, "episode_time_s": tick / 100,
                "x": 0, "y": 0, "z": height, "vx": 0, "vy": 0,
                "non_wheel_net_force_max_n": 0, "terminated": False,
                "timeout": tick == 2000, "action_cmd_vx": 0, "action_cmd_wz": 0,
                "reward_cmd_vx": 0, "reward_cmd_wz": 0,
                "action_cmd_height": height, "reward_cmd_height": height,
            }
            row.update({"diagnostic_" + key: False for key in FLAGS})
            rows.append(row)
        with (case / "telemetry.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "onnx_sha256": policy_sha, "contract_sha256": common.A_SHA,
            "status": "completed", "stop_reason": "step_budget", "policy_steps": 6000,
            "diagnostic_flags_version": 1, "termination_resets": 0, "timeout_resets": 3,
            "termination_flags": {}, "diagnostic_frames": {key: 0 for key in FLAGS},
        }
        (case / "summary.json").write_text(json.dumps(summary))
        result["cases"].append({"height_m": height, "status": "completed", "exit_code": 0})
    (directory / "evaluation_result.json").write_text(json.dumps(result))
    (directory / "evaluation_report.md").write_text("Fixture final evaluation\n")
    (directory / "training_completion.json").write_text("{}")
    return directory, plan_path, plan


@pytest.mark.parametrize("scenario", ["pass", "height_error", "wrong_policy", "missing_hash", "changed_csv"])
def test_handoff_checks_real_four_height_evidence(evaluation_report, scenario):
    directory, plan_path, plan = evaluation_report
    csv_path = directory / "h029/telemetry.csv"
    if scenario == "height_error":
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["z"] = .31
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif scenario == "wrong_policy":
        path = directory / "h029/summary.json"
        summary = json.loads(path.read_text())
        summary["onnx_sha256"] = "0" * 64
        path.write_text(json.dumps(summary))
    files = [
        {"path": name, **common.file_record(directory / name)}
        for name in sorted(handoff.ALLOWED)
        if not (scenario == "missing_hash" and name == "h029/telemetry.csv")
    ]
    manifest = {
        "schema_version": 1, "evaluation_id": directory.name,
        "training_plan_sha256": common.file_record(plan_path)["sha256"],
        "state": "evaluation_completed", "files": files,
    }
    (directory / "report_manifest.json").write_text(json.dumps(manifest))
    if scenario == "changed_csv":
        with csv_path.open("a") as stream:
            stream.write("changed after manifest publication\n")
    errors = {
        "wrong_policy": "case policy/contract mismatch",
        "missing_hash": "cover all four-height evidence",
        "changed_csv": "report hash mismatch",
    }
    if scenario in errors:
        with pytest.raises(ValueError, match=errors[scenario]):
            handoff.verified_evaluation(directory, plan_path, plan)
    else:
        report = handoff.verified_evaluation(directory, plan_path, plan)
        assert report["all_heights_pass"] is (scenario == "pass")
        assert len(report["cases"]) == 4
        assert all(case["frames"] == 6000 for case in report["cases"])
