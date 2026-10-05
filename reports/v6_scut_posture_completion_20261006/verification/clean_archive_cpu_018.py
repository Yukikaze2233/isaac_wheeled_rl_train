#!/usr/bin/env python3
"""Run the unchanged approved v17 CPU scope plus four v18 suites from an archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time


PREDECESSOR_SCOPE_SHA256 = "88ef653248642a7faadbe3f83632d384984fde1b323e3c90fa5e16f1cfb801c9"
NEW_TEST_FILES = (
    "tests/test_ordinary_posture_curriculum.py",
    "tests/test_ordinary_posture_preflight.py",
    "tests/test_ordinary_posture_fixed_evaluation.py",
    "tests/test_task_campaign_posture_revision.py",
)
NEW_SOURCE_FILES = (
    "scripts/freeze_task_campaign_ordinary_posture.py",
    "src/wheeled_tasks/chassis/ordinary_posture_curriculum.py",
    "src/wheeled_tasks/chassis/ordinary_posture_revision.py",
    "contracts/v6_task_campaign_v18.json",
    "contracts/v6_task_campaign_migration_v18.json",
    *tuple(f"contracts/v6_task_campaign_{domain}_{kind}_v18.json"
           for domain in ("recovery", "speed", "terrain", "jump", "stepjump")
           for kind in ("worker", "expectations")),
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scope_digest(previous):
    scope = {key: previous[key] for key in ("selected_test_files", "excluded_nodes_with_reasons")}
    return hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--previous-receipt", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--python", default="/home/yukikaze/isaacsim60-venv/bin/python")
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    metadata, previous = read(args.metadata), read(args.previous_receipt)
    if metadata["source_commit"] != args.source_commit:
        raise ValueError("Candidate metadata source commit differs")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("Frozen commit must be a full Git SHA")
    clean, output = Path(metadata["clean_root"]).resolve(strict=True), args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_files = previous["selected_test_files"]
    exclusions = previous["excluded_nodes_with_reasons"]
    if len(base_files) != 64 or len(exclusions) != 32 or scope_digest(previous) != PREDECESSOR_SCOPE_SHA256:
        raise ValueError("The approved 64 test files or exact 32 exclusions/reasons changed")
    files = [*base_files, *NEW_TEST_FILES]
    if len(set(files)) != 68 or any(row["node"].split("::", 1)[0] in NEW_TEST_FILES for row in exclusions):
        raise ValueError("All four v18 suites must be selected without deselection")
    source_names = sorted(set(previous["exactsource_files"]) | set(NEW_SOURCE_FILES) | set(NEW_TEST_FILES))
    hashes_before = {name: digest(clean / name) for name in source_names}
    archive = Path(metadata.get("archive_local") or metadata["archive"]).resolve(strict=True)
    archive_sha = digest(archive)
    if archive_sha != metadata["archive_sha256"]:
        raise ValueError("Archive SHA differs from candidate metadata")
    matched = {}
    with tarfile.open(archive, "r:gz") as bundle:
        commit = bundle.pax_headers.get("comment")
        if commit != args.source_commit:
            raise ValueError("Archive commit header differs")
        for member in bundle:
            if member.name in hashes_before:
                if not member.isfile():
                    raise ValueError("Expected source file is not an archive file")
                matched[member.name] = hashlib.sha256(bundle.extractfile(member).read()).hexdigest()
    if matched != hashes_before:
        raise ValueError("Selected clean source bytes differ from frozen archive")
    identity_path = output / "clean_archive_identity.json"
    write(identity_path, dict(status="passed", source_commit=args.source_commit,
        archive_sha256=archive_sha, archive_header_commit=commit,
        exactsource_file_count=len(hashes_before), all_selected_source_bytes_match_archive=True,
        exactsource_files=hashes_before))
    temporary = Path(tempfile.mkdtemp(prefix="scut-v18-clean-cpu-", dir="/tmp"))
    command = [args.python, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "--basetemp=" + str(temporary), *["--deselect=" + row["node"] for row in exclusions], *files]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH=os.pathsep.join(
            (str(clean / "src"), str(clean / "scripts"))), PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    log = output / "clean_archive_tests.stdout.log"
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=clean, env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
            print(json.dumps(dict(pid=process.pid, source_commit=args.source_commit,
                files=len(files), exact_deselections=len(exclusions), basetemp=str(temporary),
                archive_identity=str(len(hashes_before)) + " files byte-matched")), flush=True)
            code = process.wait()
        text = log.read_text()
        counts = {kind: int(number) for number, kind in re.findall(
            r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected)",
            text.split("short test summary info")[-1])}
        hashes_after = {name: digest(clean / name) for name in source_names}
        unchanged = hashes_before == hashes_after
        report = dict(status="passed" if code == 0 and unchanged else "failed",
            source_commit=args.source_commit, archive_sha256=archive_sha,
            clean_root=str(clean), command=command,
            environment={key: environment[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
            exitcode=code, counts=counts, duration_seconds=time.monotonic() - started,
            selected_test_files=files, excluded_nodes_with_reasons=exclusions,
            predecessor_scope_sha256=PREDECESSOR_SCOPE_SHA256,
            predecessor_scope_unchanged=True, new_test_files=list(NEW_TEST_FILES),
            all_new_suites_selected_without_deselection=True,
            outputtail=text[-12000:], output_file=str(log), exactsource_files=hashes_before,
            exactsource_files_before=hashes_before, exactsource_files_after=hashes_after,
            source_files_unchanged=unchanged,
            predecessor_first_run_true_failure_receipt=previous.get("first_run_true_failure_receipt"),
            predecessor_first_run_counts=previous.get("first_run_counts"),
            previous_qualified_source_commit=previous["source_commit"],
            previous_qualified_receipt=str(args.previous_receipt.resolve()),
            limitations=[
                "All 64 approved v17 files and their exact 32 named fixture exclusions/reasons are retained.",
                "Four complete v18 test files are added without deselection; failures are retained in this receipt/log.",
                "CPU tests and static role checks do not establish native rollout or hardware skill qualification."],
            archive_identity_receipt=str(identity_path),
            all_selected_source_bytes_match_archive=True, archive_header_commit=commit,
            verification_runner_sha256=digest(__file__))
        write(output / "clean_archive_tests.json", report)
    finally:
        if temporary.parent != Path("/tmp") or not temporary.name.startswith("scut-v18-clean-cpu-"):
            raise ValueError("Unexpected test temporary path; refusing cleanup")
        shutil.rmtree(temporary)
    report["owned_basetemp_removed_after_report"] = str(temporary)
    write(output / "clean_archive_tests.json", report)
    print(json.dumps({key: report[key] for key in (
        "status", "exitcode", "counts", "duration_seconds", "source_files_unchanged")}), flush=True)
    return code if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
