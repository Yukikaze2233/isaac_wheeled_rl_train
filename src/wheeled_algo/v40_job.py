"""Stdlib-only V40 job boundaries and closed artifact protocol; never start training.

PPO is not replaced: the official wrapper instance and update implementation stay
in place. Only wrapper.step checks stops; update is counted after a normal return.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import types
import uuid

BASE_ARTIFACTS = frozenset({"run_manifest.json", "agent_config.json", "contract.json",
                            "asset_manifest.json", "source_hashes.json"})
MODEL_ARTIFACTS = frozenset({"model_final.pt", "policy.onnx", "policy.onnx.json"})
ALLOWED_ARTIFACTS = BASE_ARTIFACTS | MODEL_ARTIFACTS
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_BYTES = 4 * 1024**3
SOURCE_FILES = (
    "scripts/train_v40.py", "scripts/pull_v40_artifacts.py", "scripts/start_v40_tmux.py",
    "scripts/export_v40_onnx.py", "scripts/start_v40_round2.py",
    "src/wheeled_algo/v40_job.py", "src/wheeled_algo/v40_export.py",
    "src/wheeled_algo/v40_warm_start.py", "src/wheeled_tasks/v40/round3.py",
    "src/wheeled_algo/v40_stage_transfer.py",
    "src/wheeled_algo/v40_ground.py", "src/wheeled_algo/v40_round4_launch.py",
    "src/wheeled_tasks/v40/round4.py",
    "src/wheeled_tasks/v40/contract.py", "src/wheeled_tasks/v40/core.py",
    "src/wheeled_tasks/direct/v40_serial/env.py",
    "src/wheeled_tasks/direct/v40_serial/env_cfg.py",
    "src/wheeled_tasks/direct/v40_serial/contact_sensor.py",
    "src/wheeled_tasks/agents/v40_ppo_cfg.py", "src/wheeled_world/assets/v40.py",
    "tools/prepare_v40_assets.py",
)
PLANNED_REASONS = frozenset({"iterations_completed", "max_runtime_seconds", "stop_at",
                              "sigint", "sigterm", "soft_stop"})


class JobError(ValueError):
    pass


class PlannedStop(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("seconds must be finite and positive")
    return number


def parse_stop_at(value: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:[Zz]|[+-]\d{2}:\d{2})", value
    ):
        raise ValueError("stop-at requires ISO8601 date/time with explicit timezone (Z or +/-HH:MM)")
    parsed = datetime.fromisoformat(value.upper().replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("stop-at must include timezone")
    return parsed.astimezone(timezone.utc)


class TrainingBudget:
    def __init__(self, max_runtime_seconds=None, stop_at=None, *, monotonic=time.monotonic,
                 utcnow=lambda: datetime.now(timezone.utc)):
        self.max_runtime_seconds = (None if max_runtime_seconds is None
                                    else positive_seconds(max_runtime_seconds))
        if stop_at is not None and (not isinstance(stop_at, datetime) or stop_at.tzinfo is None
                                    or stop_at.utcoffset() is None):
            raise ValueError("stop_at must be timezone-aware")
        self.stop_at = stop_at
        self.monotonic, self.utcnow = monotonic, utcnow
        self.started = None
        self.learning_finished = None
        self.soft_stop = None
        self.completed_updates = 0
        self.in_update = False
        self.update_failed = False

    def request_stop(self, reason="soft_stop"):
        if reason not in {"sigint", "sigterm", "soft_stop"}:
            raise ValueError("invalid soft stop reason")
        if self.soft_stop is None:
            self.soft_stop = reason

    def mark_signal(self, signum, _frame):
        # No raises, saves, logging, CUDA calls, or process termination here.
        if self.soft_stop is None:
            self.soft_stop = "sigint" if signum == signal.SIGINT else "sigterm"

    @contextmanager
    def signal_handlers(self):
        previous = {}
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, self.mark_signal)
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def check(self):
        # Called before app startup, before learn, and BEFORE official env.step.
        if self.soft_stop:
            raise PlannedStop(self.soft_stop)
        if self.stop_at is not None and self.utcnow() >= self.stop_at:
            raise PlannedStop("stop_at")
        if (self.started is not None and self.max_runtime_seconds is not None
                and self.monotonic() - self.started >= self.max_runtime_seconds):
            raise PlannedStop("max_runtime_seconds")

    def begin_learning(self):
        self.check()  # Startup time never consumes the relative learning budget.
        if self.started is not None:
            raise JobError("budget already started; each invocation needs a fresh budget")
        self.started = self.monotonic()

    @contextmanager
    def bind(self, runner, on_update=None):
        env, alg = runner.env, runner.alg
        original_step, original_update = env.step, alg.update
        missing = object()
        step_attr = vars(env).get("step", missing)
        update_attr = vars(alg).get("update", missing)

        def checked_step(_self, *args, **kwargs):
            self.check()
            return original_step(*args, **kwargs)

        def counted_update(_self, *args, **kwargs):
            self.in_update = True
            try:
                result = original_update(*args, **kwargs)
            except BaseException:
                self.update_failed = True
                raise
            else:
                self.completed_updates += 1
                if on_update is not None:
                    try:
                        on_update(self.completed_updates)
                    except BaseException:
                        # A failed curriculum callback cannot publish a usable final.
                        self.update_failed = True
                        raise
                return result
            finally:
                self.in_update = False

        try:
            env.step = types.MethodType(checked_step, env)
            alg.update = types.MethodType(counted_update, alg)
            yield
        finally:
            for obj, name, old in ((env, "step", step_attr), (alg, "update", update_attr)):
                if old is missing:
                    if name in vars(obj):
                        delattr(obj, name)
                else:
                    setattr(obj, name, old)


def strict_json(raw):
    if len(raw) > MAX_JSON_BYTES:
        raise JobError("JSON too large")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise JobError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise JobError(f"nonfinite JSON value: {value}")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
        def finite(child):
            if type(child) is float and not math.isfinite(child):
                raise JobError("nonfinite JSON number")
            if type(child) is dict:
                for nested in child.values():
                    finite(nested)
            elif type(child) is list:
                for nested in child:
                    finite(nested)
        finite(value)
        return value
    except (ValueError, UnicodeError) as exc:
        raise JobError(f"invalid JSON: {exc}") from exc


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path, raw):
    """Publish once: fsync a private partial, hard-link without overwrite, fsync dir."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_provenance(run_dir, *, repo_root, contract_path, contract, asset, manifest):
    """Raw JSON snapshots + hashes of fixed code paths; no recursive user-file copy."""
    from wheeled_tasks.v40.contract import DEFAULT_CONTRACT, contract_digest
    run_dir, repo_root = Path(run_dir), Path(repo_root)
    raw = Path(contract_path or DEFAULT_CONTRACT).read_bytes()
    if strict_json(raw) != contract or contract_digest(contract) != manifest["contract_sha256"]:
        raise JobError("contract changed before snapshot")
    asset_raw = Path(asset["manifest_path"]).read_bytes()
    if (hashlib.sha256(asset_raw).hexdigest() != manifest["asset_manifest_sha256"]
            or strict_json(asset_raw) != asset["manifest"]):
        raise JobError("asset manifest changed before snapshot")
    atomic_bytes(run_dir / "contract.json", raw)
    atomic_bytes(run_dir / "asset_manifest.json", asset_raw)
    sources = {}
    for name in SOURCE_FILES:
        path = repo_root / name
        if path.is_symlink() or not path.resolve().is_relative_to(repo_root.resolve()):
            raise JobError(f"source is not confined: {name}")
        sources[name] = sha256_file(path)
    atomic_bytes(run_dir / "source_hashes.json", json_bytes({
        "schema_version": 1, "source_files_sha256": sources,
        "asset_files_sha256": asset["manifest"]["files_sha256"],
        "contract_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "asset_manifest_sha256": hashlib.sha256(asset_raw).hexdigest(),
        "full_asset_snapshot": False,
        "limitation": "JSON identity and source hashes only; archive the matching repository/assets separately",
    }))


def artifact_record(run_dir, name):
    if name not in ALLOWED_ARTIFACTS:
        raise JobError(f"unknown artifact: {name}")
    path = Path(run_dir) / name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise JobError(f"artifact must be a regular nonlinked file: {name}")
    if not 0 < info.st_size <= MAX_ARTIFACT_BYTES:
        raise JobError(f"invalid artifact size: {name}")
    digest = sha256_file(path)
    after = path.lstat()
    if (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise JobError(f"artifact changed while hashing: {name}")
    return {"path": name, "size": info.st_size, "sha256": digest}


def validate_completion(record):
    """Accept only terminal protocol-v1 receipts and a closed flat file allowlist."""
    if type(record) is not dict or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise JobError("unsupported completion schema")
    required = {"schema_version", "complete", "status", "completed_updates", "requested_iterations",
                "requested_iterations_completed", "stop_reason", "policy_quality_verified",
                "export_status", "error", "update_failed", "finished_at", "max_runtime_seconds",
                "stop_at", "learning_elapsed_seconds", "artifacts"}
    if set(record) != required:
        raise JobError("completion fields do not match schema v1")
    try:
        parse_stop_at(record["finished_at"])
        if record["stop_at"] is not None:
            parse_stop_at(record["stop_at"])
        if record["max_runtime_seconds"] is not None:
            if type(record["max_runtime_seconds"]) not in {int, float}:
                raise ValueError("runtime budget must be numeric")
            positive_seconds(record["max_runtime_seconds"])
    except ValueError as exc:
        raise JobError(f"invalid completion time/budget: {exc}") from exc
    elapsed = record["learning_elapsed_seconds"]
    if elapsed is not None and (type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0):
        raise JobError("invalid learning elapsed time")
    if record.get("complete") is not True or record.get("policy_quality_verified") is not False:
        raise JobError("completion must be terminal and must not claim policy quality")
    count, requested = record.get("completed_updates"), record.get("requested_iterations")
    if type(count) is not int or type(requested) is not int or not 0 <= count <= requested or requested < 1:
        raise JobError("invalid completed/requested update count")
    status, reason = record.get("status"), record.get("stop_reason")
    if status not in {"completed", "stopped", "untrained", "training_failed", "save_failed", "export_failed"}:
        raise JobError("invalid terminal status")
    if status in {"completed", "stopped", "untrained"}:
        if record["error"] is not None:
            raise JobError("planned success/untrained cannot contain an error")
    elif type(record["error"]) is not str or not record["error"]:
        raise JobError("failure receipt requires an error")
    planned = reason in PLANNED_REASONS
    if not planned and reason != "training_exception":
        raise JobError("invalid stop_reason")
    fulfilled = planned and count == requested
    if type(record.get("requested_iterations_completed")) is not bool or record["requested_iterations_completed"] != (count == requested):
        raise JobError("requested_iterations_completed contradicts counts/reason")
    if planned and ((reason == "iterations_completed") != (count == requested)):
        raise JobError("iteration stop reason contradicts counts")
    if status == "completed" and (not fulfilled or record.get("export_status") != "verified"):
        raise JobError("completed requires all updates and verified export")
    if status == "stopped" and (not planned or count == 0 or fulfilled or record.get("export_status") != "verified"):
        raise JobError("stopped requires partial updates and verified export")
    if status == "untrained" and (count != 0 or not planned):
        raise JobError("untrained requires zero updates and a planned stop")
    if status == "training_failed" and reason != "training_exception":
        raise JobError("training_failed requires training_exception")
    if status in {"save_failed", "export_failed"} and (not planned or count == 0):
        raise JobError("save/export failure requires complete updates before planned finalization")
    if type(record.get("update_failed")) is not bool or (record["update_failed"] and status != "training_failed"):
        raise JobError("failed update cannot be treated as a planned completion")
    expected_export = ("verified" if status in {"completed", "stopped"} else
                       "export_failed" if status == "export_failed" else "not_attempted")
    if record.get("export_status") != expected_export:
        raise JobError("export status contradicts terminal status")
    items = record.get("artifacts")
    if type(items) is not list or len(items) > len(ALLOWED_ARTIFACTS):
        raise JobError("invalid artifact list")
    names = set()
    for item in items:
        if type(item) is not dict or set(item) != {"path", "size", "sha256"}:
            raise JobError("invalid artifact descriptor")
        name = item["path"]
        if type(name) is not str or name not in ALLOWED_ARTIFACTS or name in names:
            raise JobError("unknown, duplicate or escaping artifact path")
        names.add(name)
        if type(item["size"]) is not int or not 0 < item["size"] <= MAX_ARTIFACT_BYTES:
            raise JobError("invalid artifact size")
        if type(item["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise JobError("invalid artifact SHA256")
    expected = BASE_ARTIFACTS
    if status in {"completed", "stopped"}:
        expected |= MODEL_ARTIFACTS
    elif status == "export_failed":
        expected |= {"model_final.pt"}
    if names != expected:
        raise JobError("artifact set contradicts terminal status")
    return record


def _receipt(budget, requested, reason, status, export_status, error=None):
    return {
        "schema_version": 1, "complete": True, "status": status,
        "completed_updates": budget.completed_updates, "requested_iterations": requested,
        "requested_iterations_completed": budget.completed_updates == requested,
        "stop_reason": reason, "policy_quality_verified": False,
        "export_status": export_status, "error": error,
        "update_failed": budget.update_failed,
        "finished_at": budget.utcnow().isoformat(),
        "max_runtime_seconds": budget.max_runtime_seconds,
        "stop_at": budget.stop_at.isoformat() if budget.stop_at else None,
        "learning_elapsed_seconds": (None if budget.started is None else
            (budget.learning_finished if budget.learning_finished is not None else budget.monotonic()) - budget.started),
        "artifacts": [],
    }


def _publish_receipt(run_dir, record, names):
    record["artifacts"] = [artifact_record(run_dir, name) for name in sorted(names)]
    validate_completion(record)
    atomic_bytes(Path(run_dir) / "completion.json", json_bytes(record))
    return record


# These run-local audit logs are intentionally NOT transferable artifacts. Keeping
# the closed allowlist unchanged avoids widening the pull protocol to arbitrary logs.
EXPORT_STDOUT_LOG = "export.stdout.log"
EXPORT_STDERR_LOG = "export.stderr.log"
EXPORT_TIMEOUT_SECONDS = 900


def verify_export_sidecar(run_dir):
    """Stdlib-only parent verification after a SUCCESSFUL child process exit.

    ONNX checker and ORT parity run only in export_v40_onnx.py's child. Do not
    import v40_export/torch/onnx/onnxruntime here into a living Kit process.
    """
    run_dir = Path(run_dir)
    records = {name: artifact_record(run_dir, name) for name in
               ("model_final.pt", "run_manifest.json", "policy.onnx", "policy.onnx.json")}
    manifest = strict_json((run_dir / "run_manifest.json").read_bytes())
    sidecar = strict_json((run_dir / "policy.onnx.json").read_bytes())
    if type(manifest) is not dict or type(sidecar) is not dict:
        raise JobError("export manifest/sidecar must be JSON objects")
    if sidecar.get("schema_version") != 1 or type(sidecar.get("schema_version")) is not int:
        raise JobError("unsupported export sidecar schema")
    if (sidecar.get("run_manifest") != manifest
            or sidecar.get("policy") != manifest.get("policy")
            or sidecar.get("onnx_filename") != "policy.onnx"):
        raise JobError("export sidecar manifest/policy/filename mismatch")
    for key in ("contract_id", "contract_sha256", "asset_manifest_sha256"):
        if key not in manifest or sidecar.get(key) != manifest[key]:
            raise JobError(f"export sidecar identity mismatch: {key}")
    for key, name in (("checkpoint_sha256", "model_final.pt"),
                      ("run_manifest_sha256", "run_manifest.json"), ("onnx_sha256", "policy.onnx")):
        if sidecar.get(key) != records[name]["sha256"]:
            raise JobError(f"export sidecar SHA256 mismatch: {key}")
    for key, value in (("opset", 17), ("actor_obs_dim", 125), ("critic_obs_dim", 29), ("action_dim", 6)):
        if type(sidecar.get(key)) is not int or sidecar[key] != value:
            raise JobError(f"export sidecar signature mismatch: {key}")
    if (sidecar.get("dynamo") is not False
            or sidecar.get("input") != {"name": "obs_history", "dtype": "float32", "shape": [1, 125]}
            or sidecar.get("output") != {"name": "actions", "dtype": "float32", "shape": [1, 6]}):
        raise JobError("export sidecar input/output mismatch")
    validation = sidecar.get("validation")
    if (type(validation) is not dict or validation.get("passed") is not True
            or validation.get("provider") != "CPUExecutionProvider"
            or type(validation.get("sample_count")) is not int or validation["sample_count"] != 9
            or type(validation.get("seed")) is not int or validation["seed"] != 40
            or validation.get("atol") != 1e-6 or validation.get("rtol") != 1e-5):
        raise JobError("export sidecar lacks the required successful CPU ORT validation")
    samples = validation.get("samples")
    if type(samples) is not list or len(samples) != 9:
        raise JobError("export sidecar validation samples missing")
    for index, item in enumerate(samples):
        if type(item) is not dict or type(item.get("sample")) is not int or item["sample"] != index:
            raise JobError("export sidecar sample index mismatch")
    for item in [validation, *samples]:
        for key in ("max_abs_error", "max_relative_error"):
            value = item.get(key)
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise JobError(f"export sidecar invalid numerical error: {key}")
    # Reject an observable edit while the parent was validating the JSON.
    if any(artifact_record(run_dir, name) != record for name, record in records.items()):
        raise JobError("export artifacts changed during parent verification")
    return sidecar


def export_checkpoint_subprocess(run_dir, *, environment, cwd,
                                 timeout_seconds=EXPORT_TIMEOUT_SECONDS):
    """Run the unchanged exporter CLI in a separate CPU process, never in Kit.

    environment/cwd MUST be snapshots captured before AppLauncher. Preserve the
    startup PYTHONPATH, user-site and linker settings; only force CPU visibility.
    Never log environment contents. Native faults/nonzero exit cannot be hidden
    by an already-written ONNX pair. Logs are exclusive, fsynced, run-local only.
    """
    run_dir = Path(run_dir).absolute()
    if type(environment) is not dict or any(type(k) is not str or type(v) is not str
                                           for k, v in environment.items()):
        raise JobError("clean startup export environment must be a string dictionary")
    if cwd is None or not Path(cwd).is_absolute():
        raise JobError("clean startup export cwd must be an absolute path")
    child_env = dict(environment)
    child_env["CUDA_VISIBLE_DEVICES"] = "-1"
    command = [str(Path(sys.executable).absolute()),
               str(Path(__file__).resolve().parents[2] / "scripts/export_v40_onnx.py"),
               "--checkpoint", str(run_dir / "model_final.pt"),
               "--run-manifest", str(run_dir / "run_manifest.json"),
               "--output", str(run_dir / "policy.onnx")]
    # Do not resolve the interpreter symlink: doing so would lose its venv identity.
    for name in ("policy.onnx", "policy.onnx.json", EXPORT_STDOUT_LOG, EXPORT_STDERR_LOG):
        if os.path.lexists(run_dir / name):
            raise JobError(f"refusing to overwrite export output/audit: {name}")
    with (run_dir / EXPORT_STDOUT_LOG).open("xb") as stdout, (run_dir / EXPORT_STDERR_LOG).open("xb") as stderr:
        try:
            result = subprocess.run(command, env=child_env, cwd=str(cwd), stdout=stdout, stderr=stderr,
                                    check=False, timeout=positive_seconds(timeout_seconds), start_new_session=True)
            if type(result.returncode) is not int or result.returncode != 0:
                raise JobError(f"CPU export exited {result.returncode}; inspect {EXPORT_STDERR_LOG}")
            sidecar = verify_export_sidecar(run_dir)
        except BaseException as exc:
            stderr.write((f"\n[parent export audit] {type(exc).__name__}: {exc}\n").encode("utf-8", errors="replace"))
            raise
        finally:
            for stream in (stdout, stderr):
                stream.flush()
                os.fsync(stream.fileno())
    return sidecar


def run_training_job(runner, run_dir, budget, requested_iterations, *, exporter=None,
                     export_environment=None, export_cwd=None, on_update=None):
    """Run stock learn once; errors never serialize a potentially partial-update final.

    Existing periodic checkpoints remain untouched. All final saves call runner.save
    (the caller's existing metadata hook); only complete final files are published.
    The receipt certifies file completion/numerical export, NOT a useful policy.
    Production export requires pre-AppLauncher environment/cwd snapshots and uses
    a separate CPU interpreter. exporter is solely an explicit offline test seam.
    """
    if exporter is None:
        if type(export_environment) is not dict or export_cwd is None:
            raise JobError("production export requires clean startup environment/cwd snapshots")
        export_environment = dict(export_environment)
    if type(requested_iterations) is not int or requested_iterations < 1:
        raise JobError("requested_iterations must be positive")
    run_dir = Path(run_dir)
    for name in ("completion.json", *MODEL_ARTIFACTS):
        if os.path.lexists(run_dir / name):
            raise JobError(f"refusing to overwrite {name}")
    with budget.signal_handlers():
        reason = "iterations_completed"
        try:
            with budget.bind(runner, on_update=on_update):
                budget.begin_learning()  # Recheck the absolute deadline immediately before learn.
                runner.learn(num_learning_iterations=requested_iterations, init_at_random_ep_len=False)
            if budget.completed_updates != requested_iterations:
                raise JobError("stock learn returned without the requested complete updates")
        except PlannedStop as exc:
            if budget.update_failed:
                # Only the step boundary may initiate a planned stop, never inside an update.
                failure = _receipt(budget, requested_iterations, "training_exception", "training_failed", "not_attempted", str(exc))
                _publish_receipt(run_dir, failure, BASE_ARTIFACTS)
                raise
            reason = exc.reason
        except BaseException as exc:
            failure = _receipt(budget, requested_iterations, "training_exception", "training_failed", "not_attempted",
                               f"{type(exc).__name__}: {exc}")
            try:
                _publish_receipt(run_dir, failure, BASE_ARTIFACTS)
            except Exception as receipt_error:
                exc.add_note(f"Could not publish failure receipt: {receipt_error}")
            raise
        budget.learning_finished = budget.monotonic()
        if budget.completed_updates == 0:
            return _publish_receipt(run_dir, _receipt(budget, requested_iterations, reason, "untrained", "not_attempted"), BASE_ARTIFACTS)
        temporary = run_dir / f".model_final.{uuid.uuid4().hex}.partial.pt"
        try:
            runner.save(str(temporary), infos={"completed_updates": budget.completed_updates,
                "requested_iterations": requested_iterations, "stop_reason": reason,
                "policy_quality_verified": False})
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, run_dir / "model_final.pt")
        except Exception as exc:
            return _publish_receipt(run_dir, _receipt(budget, requested_iterations, reason, "save_failed", "not_attempted",
                                    f"{type(exc).__name__}: {exc}"), BASE_ARTIFACTS)
        finally:
            temporary.unlink(missing_ok=True)
        try:
            if exporter is None:
                export_checkpoint_subprocess(run_dir, environment=export_environment, cwd=export_cwd)
            else:
                exporter(run_dir / "model_final.pt", run_dir / "run_manifest.json", run_dir / "policy.onnx")
        except Exception as exc:
            return _publish_receipt(run_dir, _receipt(budget, requested_iterations, reason, "export_failed", "export_failed",
                                    f"{type(exc).__name__}: {exc}"), BASE_ARTIFACTS | {"model_final.pt"})
        status = "completed" if reason == "iterations_completed" else "stopped"
        return _publish_receipt(run_dir, _receipt(budget, requested_iterations, reason, status, "verified"), ALLOWED_ARTIFACTS)
