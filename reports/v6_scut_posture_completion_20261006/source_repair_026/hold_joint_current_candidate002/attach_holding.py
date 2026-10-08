"""Passive attachment to the existing source023 same-seal observer factory.

There is no driver, checkpoint selection, App, simulator query or native launch
here. The caller keeps the original driver's admission and execution lifecycle.
"""
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import traceback
import types


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_bound(path, expected, name):
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Attachment source bytes differ: " + str(path))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def authenticate(path, expected):
    path = Path(path)
    if not path.is_absolute() or sha(path) != expected:
        raise ValueError("Attachment manifest identity differs")
    manifest = json.loads(path.read_text())
    if manifest["schema"] != "source023-selected-hold-attachment-candidate-v1":
        raise ValueError("Attachment schema differs")
    files = manifest["files_sha256"]
    for source, digest in files.items():
        if not Path(source).is_absolute() or sha(source) != digest:
            raise ValueError("Bound input bytes differ: " + source)
    required = (str(Path(__file__).resolve()), *(manifest[key] for key in
                ("driver", "holding_trace", "mechanical_hooks")))
    if any(source not in files for source in required):
        raise ValueError("Attachment executable binding incomplete")
    return manifest, dict(files, **{str(path): expected})


def observer_class(parent, report, binding, *, lanes, expected_rows, journal, hooks):
    class ContextTrace(journal.HoldingTrace):
        def _fsm_state(self):
            result = super()._fsm_state()
            # Original device tensors, including the exact pre/post reset
            # lesson/profile. None is unobserved, never an invented lesson.
            curriculum = getattr(self.recovery, "pose_curriculum", None)
            result["lesson_id"] = self._sample(getattr(curriculum, "level_ids", None))
            result["profile_id"] = self._sample(getattr(curriculum, "profile_ids", None))
            return result

    class SelectedHoldingObserver(parent):
        def __init__(self, env, output):
            self.hold_trace = self.hold_stream = None
            self.hold_output = Path(output)
            self.hold_written = 0
            self.hold_cleanup_done = False
            super().__init__(env, output)
            try:
                self.hold_rows = [{"lane": lane, "group": env.scene_groups[lane],
                                   "terrain_kind": env.kinds[lane]} for lane in lanes]
                if self.hold_rows != expected_rows:
                    raise ValueError("Selected original native scene rows differ")
                self.hold_trace = ContextTrace(env, hooks.Hooks(), lanes, max_policy_records=1)
                self.hold_stream = gzip.open(self.hold_output / "holding_intervals.jsonl.gz", "xt")
                report["selected_hold_attachment_binding"] = binding
            except BaseException:
                self._before_close()
                raise

        def observe(self, tick, extras):
            try:
                self.hold_trace.after_policy(tick, extras)
                row = self.hold_trace.records.popleft()
                row["selected_rows"] = self.hold_rows
                self.hold_stream.write(json.dumps(row, allow_nan=False) + "\n")
                self.hold_written += 1
            except BaseException:
                self._invalid("selected_hold_observe", traceback.format_exc())
                raise
            return super().observe(tick, extras)

        def _close_hold(self):
            if not self.hold_cleanup_done:
                self.hold_cleanup_done = True
                try:
                    if self.hold_trace is not None:
                        self.hold_trace.close()
                except BaseException:
                    self._invalid("selected_hold_restore", traceback.format_exc())
                finally:
                    try:
                        if self.hold_stream is not None:
                            self.hold_stream.close()
                    except BaseException:
                        self._invalid("selected_hold_stream_close", traceback.format_exc())
        def _before_close(self):
            self._close_hold()
            # Idempotent original guarded cleanup retains its own ownership.
            return super()._before_close()

        def finish(self, output):
            try:
                # Remove our outer hooks first; leave original parent finish's
                # own cleanup and evidence publication order unchanged.
                self._close_hold()
                summary = self.hold_trace.summary()
                summary.update(interval_records_written=self.hold_written,
                    journal_sha256=sha(self.hold_output / "holding_intervals.jsonl.gz"),
                    serialized_at="50Hz after original env.step; device-only snapshots at 5ms",
                    selected_rows=self.hold_rows, current_checkpoint_admitted_by_attachment=False)
                hooks.json_write(self.hold_output / "holding_summary.json", summary)
                result = super().finish(output)
                result["selected_hold_summary_sha256"] = sha(self.hold_output / "holding_summary.json")
                return result
            except BaseException:
                self._invalid("selected_hold_finish", traceback.format_exc())
                self._before_close()
                raise
    return SelectedHoldingObserver


@contextmanager
def attached(driver, manifest_path, manifest_sha):
    """Nest outside original driver.main; it alone authenticates the current seal."""
    manifest, binding = authenticate(manifest_path, manifest_sha)
    if Path(driver.__file__).resolve() != Path(manifest["driver"]):
        raise ValueError("Existing source023 driver location differs")
    journal = load_bound(manifest["holding_trace"], binding[manifest["holding_trace"]], "bound_selected_hold_journal")
    hooks = load_bound(manifest["mechanical_hooks"], binding[manifest["mechanical_hooks"]], "bound_mechanical_hooks_only")
    previous = driver.source023_runtime_observer

    def factory(parent, identity, report):
        identity["observer_artifact_sha256"].update(binding)
        original = previous(parent, identity, report)
        return observer_class(original, report, binding, lanes=tuple(manifest["lanes"]),
            expected_rows=manifest["selected_rows"], journal=journal, hooks=hooks)

    driver.source023_runtime_observer = factory
    try:
        yield driver
    finally:
        if driver.source023_runtime_observer is not factory:
            raise ValueError("Attachment factory ownership changed; refusing alien overwrite")
        driver.source023_runtime_observer = previous
        authenticate(manifest_path, manifest_sha)
