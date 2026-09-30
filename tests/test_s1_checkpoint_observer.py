"""The side observer authenticates evidence and yields resources to training."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import watch_s1_checkpoints as observer


@pytest.mark.parametrize("resources,allowed", [
    ({"gpu_free_mib": 7168, "host_available_mib": 4096}, True),
    ({"gpu_free_mib": 7167, "host_available_mib": 8192}, False),
    ({"gpu_free_mib": 8192, "host_available_mib": 4095}, False),
    ({"error": "no GPU readback"}, False),
    ({"gpu_free_mib": float("nan"), "host_available_mib": 8192}, False),
])
def test_resource_gate(resources, allowed):
    assert observer.sufficient_resources(resources) is allowed


def test_active_evaluator_low_headroom_is_not_treated_as_start_capacity():
    resources = {"gpu_free_mib": 2048, "host_available_mib": 2048}
    assert observer.sufficient_resources(resources, running=True)
    assert not observer.sufficient_resources(resources)
    assert not observer.sufficient_resources({"gpu_free_mib": 1023, "host_available_mib": 2048}, running=True)


def test_evaluation_lock_is_exclusive_and_released(tmp_path):
    with observer.evaluation_lock(tmp_path) as acquired:
        assert acquired
        with observer.evaluation_lock(tmp_path) as second:
            assert not second
    with observer.evaluation_lock(tmp_path) as acquired:
        assert acquired


def checkpoint_fixture(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"sealed model")
    contract = checkpoint.with_suffix(".contract.json")
    contract.write_text("{}")
    seal = {"schema": "chassis-learning-checkpoint-v1", "sealed": True,
            "checkpoint_sha256": observer.digest(checkpoint), "contract_sha256": observer.digest(contract),
            "learning_updates": 860}
    checkpoint.with_suffix(".checkpoint.json").write_text(json.dumps(seal))
    return checkpoint, observer.digest(contract)


def test_checkpoint_hash_clock_and_contract_are_authenticated(tmp_path):
    checkpoint, contract_sha = checkpoint_fixture(tmp_path)
    assert observer.authenticate_checkpoint(checkpoint, 860, contract_sha) == observer.digest(checkpoint)
    for update, sha in [(940, contract_sha), (860, "wrong")]:
        with pytest.raises(ValueError):
            observer.authenticate_checkpoint(checkpoint, update, sha)
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="stale"):
        observer.authenticate_checkpoint(checkpoint, 860, contract_sha)


def evaluation_fixture():
    sources = {name: observer.digest(ROOT / name) for name in
               ("scripts/evaluate_chassis.py", "src/wheeled_tasks/chassis/env.py")}
    return {"status": "evaluated", "candidates": [{"cases": {"a": {"passed": True}},
              "unfinished_episodes": 0, "checkpoint_sha256": "checkpoint"}],
            "contract_sha256": "contract", "settings": {"seed": 290929}, "episodes_per_case": 4,
            "script_sha256": sources["scripts/evaluate_chassis.py"], "source_sha256": sources}


def test_evaluation_seed_coverage_and_source_identity_are_required():
    result = evaluation_fixture()
    def validate(value):
        return observer.authenticate_result(value, ["a"], 290929, "contract", "checkpoint", result["script_sha256"])
    assert validate(result)["cases"]["a"]["passed"]
    for key, value in [("settings", {"seed": 290930}), ("episodes_per_case", 1),
                       ("contract_sha256", "other"), ("source_sha256", {}),
                       ("source_sha256", {**result["source_sha256"], "../outside": "unknown"})]:
        changed = deepcopy(result)
        changed[key] = value
        with pytest.raises(ValueError):
            validate(changed)
    changed = deepcopy(result)
    changed["candidates"][0]["unfinished_episodes"] = 1
    with pytest.raises(ValueError):
        validate(changed)


def test_termination_reaps_only_observer_child():
    task = object.__new__(observer.Observer)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    task.child = child
    try:
        task.stop_child()
        assert child.poll() is not None
        assert task.child is None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_completed_worker_never_launches_or_marks_evaluation_success(tmp_path, monkeypatch):
    task = object.__new__(observer.Observer)
    task.train_root = tmp_path
    task.root = tmp_path / "observations"
    task.root.mkdir()
    task.report = {"observations": {}}
    task.child = None
    monkeypatch.setattr(task, "stopping_reason", lambda: "training_finished")
    monkeypatch.setattr(task, "evaluate", lambda *args: pytest.fail("must not evaluate a finished worker"))
    task.worker = tmp_path / "worker"
    monkeypatch.setattr(observer.signal, "signal", lambda *_: None)
    task.run()
    assert all(row == {"status": "skipped", "reason": "training_finished"}
               for row in task.report["observations"].values())
    assert len(task.report["observations"]) == 2
