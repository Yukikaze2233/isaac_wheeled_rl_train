#!/usr/bin/env python3
"""Read source020 attempt_004 with the frozen019 CRC/window/metric implementation.

Only the authenticated attempt identity is adapted. No simulator, model, SSH,
TensorBoard server, training, or evaluation is started. This is an evidence
summary of the existing native event, not another curve source.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
BASE_READER_SHA256 = "ffefbaae2e1cbf30b520224fc286a8adbc194708570e4ce0fcb00fa7f86cdb17"
COMMIT = "7f315f30741be01d70046a11848443038c14b240"
ARCHIVE_SHA256 = "ff0c57ca2255fb0e336b28be57e0a66e7923597dd5398d0bb18c4e34f91c1b87"
PLAN_SHA256 = "027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022"
EXPECTED_PID = 3494096
CONSUMPTION_ORIGIN, ACTOR_ORIGIN = 29679, 2758
BASELINE_STEP = CONSUMPTION_ORIGIN + 9
DEFAULT_RUN = Path("/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_004/train")
SOURCE_UPDATES = {
    "scripts/train_chassis.py": "e7e6f711e238b72ed258ca6793f8cb7c88977a29572c4f9d26877743167d1773",
    "src/wheeled_tasks/chassis/env.py": "d8a8bbaa91f7e7f8055031d714ce97657e950ae93a949b70be5c4cd2ec1d183b",
    "src/wheeled_tasks/chassis/recovery_training.py": "b475b0166e4cfc0c9a079c06c65bdd20fb744de30d00b92f0965febd5e3d24c5",
    "src/wheeled_tasks/chassis/root_kinematics.py": "0c63be06fc026b0687aa4e9d9fd7dabb009f7c1ab24446244b672928368f610b",
    "src/wheeled_tasks/chassis/eval_env.py": "6ce02c158f1cf0ed76f030067e6d9b6d36376c99ba15c23d1e382b87f784625a",
    "src/wheeled_tasks/chassis/rewards.py": "371899baec0fc9aac75ee05ff5f24e6d9fb9eb40e529ffaf8eee4567eda4fec3",
}
REUSED_FUNCTIONS = (
    "wanted_tags", "canonical_tag", "aligned_tail_start", "parse_events", "scalar", "stats",
    "validate_counts", "window_spec", "summarize", "select_window", "load_reference", "compare_reference",
)
# Only these literals inside load_authority are replaced. The frozen parser and
# every metric/window function execute from the original verified bytes.
IDENTITY_REPLACEMENTS = {
    "attempt_003": ("attempt_004", 1),
    "source_revision_019.json": ("source_revision_020.json", 2),
    "2458": ("2758", 1),
    "29379": ("29679", 1),
    "Source019 same-contract continuation must have zero new critic warmup":
        ("Source020 same-contract continuation must have zero new critic warmup", 1),
    "Source019 extends ordinary tilt/push to872 existing motion-review lanes; no nonstandard_start leg-bank hold is introduced.":
        ("Source020 preserves source019 ordinary tilt/push eligibility on872 existing motion-review lanes; no nonstandard_start leg-bank hold is introduced.", 1),
}
CONFOUNDS = [
    "Source020 changes COM invariant caching and the disabled fixed-evaluation catalog boundary; the v18 plan/worker contracts, reward, curriculum, clocks and ordinary eligibility remain unchanged.",
    "COM caching was checked for engineering equivalence; source-only equivalence is not proof of equal learned behavior across different actor updates, sampled resets, exploration, DR or runtime scheduling.",
    "Local engineering ABBA timing (180.13 to168.16s,6.648%) is not a paired Kaiser PPO speed result. Shared GPU load and startup effects remain in native Timing/* samples.",
    "These event windows use stochastic PPO collection. Deterministic fixed mean-policy evaluation and historical source712/seal2579 native mean/stochastic probes are separate evidence.",
    "50 and100 windows may overlap; do not add their frames or episodes. All comparisons are descriptive, not a causal reward/entropy/COM finding or skill qualification.",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_reader(path=HERE / "read_recovery_window_019.py"):
    """Load exact old bytes into isolated globals; adapt only authority literals."""
    path = Path(path).resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BASE_READER_SHA256:
        raise ValueError("Frozen019 reader SHA differs; parser/metrics cannot be substituted")
    reader = types.ModuleType("frozen019_reader_for_source020")
    reader.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), reader.__dict__)
    originals = {name: getattr(reader, name) for name in REUSED_FUNCTIONS}
    tree = ast.parse(payload, filename=str(path))
    authority_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "load_authority"]
    reader.require(len(authority_nodes) == 1, "Frozen authority function differs")
    counts = Counter()

    class IdentityOnly(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is str and node.value in IDENTITY_REPLACEMENTS:
                replacement, _ = IDENTITY_REPLACEMENTS[node.value]
                counts[node.value] += 1
                return ast.copy_location(ast.Constant(value=replacement), node)
            return node

    node = IdentityOnly().visit(authority_nodes[0])
    reader.require(counts == Counter({key: count for key, (_, count) in IDENTITY_REPLACEMENTS.items()}),
                   "Frozen identity literal counts differ")
    identity_tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(identity_tree, str(path) + "::source020_identity_only", "exec"), reader.__dict__)
    reader.COMMIT = COMMIT
    reader.EXPECTED_PID = EXPECTED_PID
    reader.CONSUMPTION_ORIGIN = CONSUMPTION_ORIGIN
    reader.ACTOR_ORIGIN = ACTOR_ORIGIN
    reader.BASELINE_STEP = BASELINE_STEP
    reader.DEFAULT_RUN = DEFAULT_RUN
    reader.SOURCE_HASHES = {**reader.SOURCE_HASHES, **SOURCE_UPDATES}
    reader.require(all(getattr(reader, name) is function for name, function in originals.items()),
                   "Adapter replaced a frozen parser/metric function")
    reader.adapter_reuse = dict(reader_sha256=BASE_READER_SHA256, reader_path=str(path),
        function_objects_unchanged=list(REUSED_FUNCTIONS), authority_only_literal_replacements={
            key: dict(replacement=value, occurrences=count) for key, (value, count) in IDENTITY_REPLACEMENTS.items()},
        isolated_globals=True, frozen019_file_modified=False)
    return reader


def load_authority(reader, run_dir, source):
    """Retain every019 admission check and additionally bind020 revision header."""
    run_dir, source = Path(run_dir).resolve(), Path(source).resolve()
    authority = reader.load_authority(run_dir, source)
    root = run_dir.parents[3]
    revision = reader.read_json(root / "source_revision_020.json")
    reader.require(revision.get("schema") == "task-source-revision-v1"
        and revision.get("source_archive_sha256") == ARCHIVE_SHA256
        and revision.get("old_contract_sha256") == revision.get("new_contract_sha256") == PLAN_SHA256,
        "Source020 revision archive/unchanged plan header differs")
    reader.require(revision.get("old_source_root") == str(root / "source_712d701473f3"),
                   "Wrong source019 predecessor path")
    authority["source_revision"] = dict(source_archive_sha256=ARCHIVE_SHA256,
        plan_sha256=PLAN_SHA256, parent_task_actor=ACTOR_ORIGIN,
        parent_global_consumption=CONSUMPTION_ORIGIN, ordinary_origin_actor_update=2294,
        archive_sha_scope="Authenticated revision header; selected frozen files are rehashed before/after. Archive itself and parent PT are not opened.")
    return authority


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-reader", type=Path, default=HERE / "read_recovery_window_019.py")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--updates", type=int, choices=(50, 100), nargs="+", default=[50, 100])
    parser.add_argument("--end-update", default="latest")
    parser.add_argument("--reference", type=Path, help="Exact sealed recovery_first100_018.json; descriptive only")
    parser.add_argument("--output", type=Path, required=True, help="New evidence filename; existing outputs are preserved")
    parser.add_argument("--max-read-mib", type=int, default=512)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output already exists; use a unique evidence filename")
    schema = "source020-recovery-native-event-window-set-v1"
    try:
        reader = make_reader(args.base_reader)
        reader.require(1 <= args.max_read_mib <= 1024 and len(args.updates) == len(set(args.updates)),
                       "Invalid byte limit/duplicate windows")
        source = args.source_root or args.run_dir.resolve().parents[3] / "source_7f315f30741b"
        endpoint = "latest" if args.end_update == "latest" else int(args.end_update)
        authority = load_authority(reader, args.run_dir, source)
        last_step = CONSUMPTION_ORIGIN + authority["progress"]["actor_updates_in_block"] - 1
        # The original parse_events default was captured at import with019's
        # origin. Always pass020's real baseline explicitly.
        series, audit = reader.parse_events(Path(authority["event"]), reader.wanted_tags(authority),
            last_step, min_step=BASELINE_STEP, max_read_bytes=args.max_read_mib * 1024**2)
        reference = reader.load_reference(args.reference) if args.reference else None
        windows = {str(width): reader.select_window(series, authority, audit, width, endpoint)
                   for width in args.updates}
        for window in windows.values():
            window["reused_metric_schema"] = window["schema"]
            window["schema"] = "source020-recovery-native-event-window-summary-v1"
            window["source020_context"] = CONFOUNDS
            window["comparison_to_v18_first100"] = reader.compare_reference(window, reference) if reference else dict(
                available=False, reason="Pass exact sealed recovery_first100_018.json; no reference observations are invented")
        after = {name: reader.digest(Path(source) / name) for name in reader.SOURCE_HASHES}
        reader.require(after == authority["source_files_sha256"], "Frozen metric source changed during read")
        reader.require(digest(args.base_reader) == BASE_READER_SHA256, "Frozen019 reader changed during read")
        complete = all(window["complete"] for window in windows.values())
        report = dict(schema=schema, complete=complete,
            status="requested_actor_windows_complete" if complete else "pending_requested_actor_windows",
            source_commit=COMMIT, qualification=None, windows=windows, source_files_unchanged=True,
            parser_metric_reuse=reader.adapter_reuse, source020_context=CONFOUNDS,
            reference_file=str(args.reference) if args.reference else None,
            identity={key: authority[key] for key in ("source_root", "worker_pid", "event", "base_groups", "population",
                "group_pools", "clocks", "source_files_sha256", "metadata_sha256", "invocation", "progress", "source_revision")})
        exitcode = 0
    except Exception as error:
        report = dict(schema=schema, status="read_failed", complete=False, source_commit=COMMIT,
                      qualification=None, error=type(error).__name__ + ": " + str(error))
        exitcode = 2
    report.update(queried_at_utc=datetime.now(timezone.utc).isoformat(), reader_sha256=digest(Path(__file__)),
        base_reader_expected_sha256=BASE_READER_SHA256,
        scope="Read_only_existing_native_event_no_GPU_no_SSH_no_model_no_TensorBoard_server")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], complete=report["complete"], output=str(args.output),
        windows={key: dict(complete=value["complete"], actor_interval=value["window"]["task_actor_updates_inclusive"],
            missing_required_tag_count=value["missing_required_tag_count"]) for key, value in report.get("windows", {}).items()},
        error=report.get("error"))))
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
