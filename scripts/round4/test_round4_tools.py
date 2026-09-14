"""CPU-only scratch deployment, bounded ownership and evidence regressions."""
import copy
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import r4_common as common
from round4 import deploy, evaluation, launch, watch
import start_v40_round2 as child_tools


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    commit = "a" * 40
    root = tmp_path / ("round4-full-" + commit)
    root.mkdir()
    contract = json.loads((common.REPO / common.CONTRACT).read_text())
    identity = {"physics_model": common.PHYSICS_MODEL, "physics_asset_manifest_sha256": common.LEGACY_ASSET_SHA,
                "repaired_dynamics_used": False, "explicit_user_authorization": True,
                "new_fifteen_body_model_role": "preview_only_not_training_physics",
                "authorization_sha256": "fixture-authorization", "initial_not_before": "2026-09-15T00:00:00+08:00"}
    monkeypatch.setattr(common, "legacy_identity", lambda *_: identity)
    monkeypatch.setattr(deploy, "legacy_identity", lambda *_: identity)
    files = {common.CODE_DIRECTORY + "/" + common.CONTRACT: json.dumps(contract).encode(),
             common.CODE_DIRECTORY + "/scripts/train_v40.py": b"# fixture source\n"}
    digest = deploy.make_bundle(commit, files, root / "bundle.tar.gz")
    deploy.unpack(root, digest)
    runtime = tmp_path / "bin/runtime.sh"
    runtime.parent.mkdir()
    runtime.write_text("# fixture runtime\n")
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("fixture interpreter")
    ground = tmp_path / "ground.usd"
    ground.write_bytes(b"fixture ground")
    monkeypatch.setattr(launch, "EXPERIMENTS", tmp_path)
    args = SimpleNamespace(experiment=root, runtime=runtime, ground_usd=ground, seed=44, soft_hours=48)
    return root, args


def test_deployment_preserves_leaf_and_rejects_source_changes(experiment):
    root, _ = experiment
    snapshot = common.verify_snapshot(root)
    assert snapshot["initialization"] == "scratch" and snapshot["parent"] is None
    assert not (root / "parent").exists()
    assert all(name.startswith("isaac_wheeled_rl_train/") for name in snapshot["files"])
    (root / common.CODE_DIRECTORY / "scripts/train_v40.py").write_text("changed")
    with pytest.raises(common.JobError, match="changed"):
        common.verify_snapshot(root)


def test_unpack_rejects_path_escape_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "legacy_identity", lambda *_: {})
    files = {"isaac_wheeled_rl_train/../../escape": b"bad", common.CODE_DIRECTORY + "/" + common.CONTRACT: b"{}"}
    digest = deploy.make_bundle("b" * 40, files, tmp_path / "bundle.tar.gz")
    with pytest.raises(ValueError, match="unconfined"):
        deploy.unpack(tmp_path, digest)
    assert not (tmp_path / "snapshot.json").exists()


def test_scratch_plan_has_only_one_full_training_command(experiment):
    root, args = experiment
    plan = launch.build_plan(args)
    argv = plan["command"]
    assert argv[argv.index("--num-envs") + 1] == "1024"
    assert argv[argv.index("--max-iterations") + 1] == "30000"
    assert argv[argv.index("--ground-usd") + 1] == str(args.ground_usd)
    assert not common.SOURCE_OPTIONS.intersection(argv)
    assert "--preflight-only" not in argv and "phases" not in plan
    assert plan["learning_seconds"] == 48 * 3600
    assert plan["training_hard_seconds"] == 48 * 3600 + 1800
    assert plan["worker_hard_seconds"] == 49 * 3600
    assert not (root / "formal").exists()
    args.soft_hours = 72
    assert launch.build_plan(args)["worker_hard_seconds"] == 73 * 3600


def manifest_fixture(plan):
    return {**plan["identity"], "num_envs": 1024, "training_profile": "round4_full",
            "asset_manifest_sha256": common.LEGACY_ASSET_SHA,
            "initialization": plan["expected_initialization"], "source_provenance": plan["expected_initialization"],
            "domain_randomization_report": {"passed": True},
            "training_curriculum": {"timebase": "successful_ppo_updates", "completed_updates": 30000,
                                    "state": {"max_delta_v_m_s": .5}}}


@pytest.mark.parametrize("bad", ["parent", "resume", "readback", "curriculum"])
def test_scratch_identity_rejects_invalid_evidence(experiment, bad):
    _, args = experiment
    plan = launch.build_plan(args)
    manifest = copy.deepcopy(manifest_fixture(plan))
    common.verify_scratch(manifest, plan)
    if bad == "parent":
        manifest["source_provenance"]["parent_checkpoint_sha256"] = "old-checkpoint"
    elif bad == "resume":
        manifest["resume_provenance"] = {"path": "old-checkpoint"}
    elif bad == "readback":
        manifest["domain_randomization_report"]["passed"] = False
    else:
        manifest["training_curriculum"]["timebase"] = "wall_time"
    with pytest.raises(common.JobError):
        common.verify_scratch(manifest, plan)


def test_worker_runs_formal_once_without_gpu_probe(experiment, monkeypatch):
    _, args = experiment
    plan = launch.build_plan(args)
    audit = Path(plan["audit_dir"])
    audit.mkdir(parents=True)
    run = Path(plan["run_dir"])
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps(manifest_fixture(plan)))
    path = audit / "plan.json"
    path.write_text(json.dumps(plan))
    calls = []
    monkeypatch.setattr(launch, "own_descendants", lambda: None)
    monkeypatch.setattr(launch, "cleanup_descendants", lambda: True)
    monkeypatch.setattr(launch.signal, "signal", lambda *_: None)
    monkeypatch.setattr(launch, "system_snapshot", lambda **_: {"fixture": True})
    monkeypatch.setattr(launch, "run_child", lambda *argv: calls.append(argv))
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, '{"state":"pending"}', ""))
    monkeypatch.setattr(launch, "verify_run", lambda *_: {"status": "completed", "completed_updates": 30000,
                                                         "artifact_hashes_verified": True, "export_verified": True})
    assert launch.run_worker(path) == 0
    assert len(calls) == 1 and calls[0][1] == "train"
    assert "--stop-at" in calls[0][2] and "--preflight-only" not in calls[0][2]
    status = json.loads((audit / "worker.status.json").read_text())
    assert status["formal_training_target_completed"] is True
    with pytest.raises(FileExistsError):
        launch.run_worker(path)
    assert len(calls) == 1


def test_owned_child_timeout_never_signals_another_pid(tmp_path, monkeypatch):
    (tmp_path / "audit").mkdir()
    signals = []

    class Child:
        pid = 4321
        returncode = None

        def wait(self, timeout):
            if not signals:
                raise subprocess.TimeoutExpired("fixture train", timeout)
            self.returncode = -signal.SIGTERM
            return self.returncode

    monkeypatch.setattr(child_tools.subprocess, "Popen", lambda *_, **__: Child())
    monkeypatch.setattr(child_tools.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    plan = {"stage_root": str(tmp_path), "repo": str(tmp_path), "python": "/env/bin/python",
            "identity": {"stage": "locomotion"}}
    with pytest.raises(subprocess.TimeoutExpired):
        child_tools.run_child(plan, "train", ["fixture"], 1)
    assert signals == [(4321, signal.SIGTERM)]
    assert json.loads((tmp_path / "audit/train.status.json").read_text())["timed_out"] is True


@pytest.mark.parametrize("available,exit_code", [(False, 0), (True, 2), (True, 0)])
def test_watcher_inherits_evaluator_and_waits_for_server_dispatch(available, exit_code):
    plan = {"python": "/runtime/python", "repo": "/experiment/isaac_wheeled_rl_train",
            "audit_dir": "/experiment/formal/audit", "evaluation_entry": "scripts/round4/evaluate_policy.py"}
    client = SimpleNamespace(
        exec=lambda *args: "1" if available else "0",
        download=lambda *args: io.BytesIO(json.dumps({"exit_code": exit_code}).encode()),
    )
    if not available:
        with pytest.raises(watch.NotReady, match="server evaluation hook"):
            watch.evaluation_command_after_remote_hook(client, plan)
    elif exit_code:
        with pytest.raises(common.JobError, match="dispatch failed"):
            watch.evaluation_command_after_remote_hook(client, plan)
    else:
        command = watch.evaluation_command_after_remote_hook(client, plan)
        assert command[-3:] == ["--evaluator", plan["evaluation_entry"], "--launch"]


def test_evaluation_plan_keeps_recovery_denominator_and_hold(experiment):
    _, args = experiment
    request = evaluation.evaluation_request(launch.build_plan(args), [.5, 0])
    assert request["schema_version"] == 2 and len(request["cases"]) == 12
    tracking = [case for case in request["cases"] if case["kind"] == "tracking"]
    assert [case["command"] for case in tracking] == [[3, 0, .32], [-3, 0, .32], [0, 6, .32], [0, -6, .32]]
    assert all(case["episodes"] == 2 and case["episode_seconds"] == 20 and case["push"] is None for case in tracking)
    assert sum(case["episodes"] * case["episode_seconds"] for case in request["cases"]) == 2000
    assert request["random_push_enabled"] is False and request["reward_threshold"] is None
    recovery = request["recovery"]
    assert recovery["deadline_s"] == 2 and recovery["steady_band_hold_s"] == 1
    assert recovery["height_error_band_m"] == .01 and recovery["planar_speed_band_m_s"] == .05
    assert recovery["tilt_band_deg"] == 10
    assert recovery["hold_must_finish_within_deadline"] is False
    assert "failed_disturbed_episodes" in recovery["required_counts"]
    assert request["policy_quality_verified"] is False


@pytest.mark.parametrize("failure", ["empty_result", "changed_bytes", "wrong_request"])
def test_report_return_rejects_unverified_or_empty_success(tmp_path, failure):
    payloads = {
        "result.json": json.dumps({"status": "success", "cases": []}).encode(),
        "telemetry.csv": b"fixture,not_simulation\n1,0\n",
    }
    marker = {"schema_version": 1, "request_sha256": "request", "files": [
        {"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in payloads.items()
    ]}
    if failure == "wrong_request":
        marker["request_sha256"] = "other"
    elif failure == "changed_bytes":
        payloads["telemetry.csv"] = b"changed"
    payloads["report_manifest.json"] = json.dumps(marker).encode()

    class Client:
        def exec(self, _):
            return "1"

        @contextmanager
        def download(self, path):
            yield io.BytesIO(payloads[Path(path).name])

    with pytest.raises(common.JobError):
        watch.pull_reports(Client(), "/remote", tmp_path, "request")
    assert not (tmp_path / "evaluation-reports").exists()


def test_actual_legacy_asset_and_authorization_are_hash_bound():
    contract = json.loads((common.REPO / common.CONTRACT).read_text())
    identity = common.legacy_identity(contract, lambda name: (common.REPO / name).read_bytes())
    assert identity["physics_asset_manifest_sha256"] == common.LEGACY_ASSET_SHA
    assert identity["explicit_user_authorization"] is True and identity["repaired_dynamics_used"] is False
    manifest_path = str(Path(contract["asset"]["directory"]) / contract["asset"]["manifest"])

    def changed(name):
        return b"changed asset" if name == manifest_path else (common.REPO / name).read_bytes()

    with pytest.raises(common.JobError, match="authorized legacy"):
        common.legacy_identity(contract, changed)
