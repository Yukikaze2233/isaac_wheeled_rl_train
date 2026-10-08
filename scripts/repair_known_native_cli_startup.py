#!/usr/bin/env python3
"""One explicitly diagnosed CLI-only repair; preserve the failed admission."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
from queue_task_campaign_boundary_continuation import BoundaryContinuationQueue, NativeAdapter
from queue_task_campaign_continuation import durable_json, read_json, sha

FAILED_PINS = {
    "invocation.intent.json": "de1c59f10790e97d3a59c89a7929d41ae0c64a7998c662ad270d79964b84b8a1",
    "receipt.json": "51e7cd31639b327de490d6ff9c2f053d1bb2745211332db782c62d59b388f28b",
    "source_selection.json": "a50715201c99d4de9eb41505525c214a03536cc9c931e1479ae7c879e9ea2978",
    "stdout.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "stderr.log": "ed37a8e94f18f0e1c0773497bd808965a2c2109a7ce2fa989eedfa9937a119d7",
}
MANIFEST_SHA = "5950cb69a4ddff8f360e6d40e73741f243563052d9dd2ce58aa8589ffb5a7a7e"


def authenticate_failure(q):
    if q.target_revision != 26 or q.config_sha256 != MANIFEST_SHA:
        raise ValueError("Repair must use the original frozen source028 manifest")
    old = q.run_root / "boundary_integrated_028.native"
    for name, digest in FAILED_PINS.items():
        if sha(old / name) != digest:
            raise ValueError("Original CLI failure bytes changed: " + name)
    failed = read_json(old / "receipt.json")
    if (failed.get("status") != "failed" or failed.get("exit_code") != 1
            or failed.get("PPO_updates") != 0 or failed.get("report_sha256") is not None
            or failed.get("manifest_sha256") != q.config_sha256
            or (old / "scene").exists() or q.intent.exists() or q.queue_receipt.exists()
            or q.owners()):
        raise ValueError("Repair requires the proven pre-Simulator failure and no PPO owner/launch")
    command = failed["command"]
    if "--learning-rate-scale" in command or "--validate-scene" not in command:
        raise ValueError("Failure does not match the diagnosed missing explicit LR scale")
    return failed


class DiagnosedCLIAdapter(NativeAdapter):
    def runtime_directory(self):
        return self.q.run_root / "boundary_integrated_028_cli_repair001.native"

    def validate_runtime(self, evidence):
        authenticate_failure(self.q)
        if evidence["integrated_admission"] != read_json(self.q.run_root / "boundary_integrated_028.native/source_selection.json"):
            raise ValueError("Named repair changed the actual source selection/account")
        return super().validate_runtime(evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    code_manifest = read_json(Path(__file__).parent / "code_manifest.json")
    if (code_manifest.get("schema") != "frozen-known-native-cli-repair-code-v1"
            or set(code_manifest.get("files", {})) != {"repair_known_native_cli_startup.py",
                "queue_task_campaign_boundary_continuation.py", "queue_task_campaign_continuation.py"}):
        raise ValueError("External repair must pin its exact three code files")
    for name, digest in code_manifest["files"].items():
        if sha(Path(__file__).parent / name) != digest:
            raise ValueError("Frozen external repair code differs: " + name)
    q = BoundaryContinuationQueue(read_json(args.manifest), adapter_factory=DiagnosedCLIAdapter)
    intent = q.run_root / "boundary_integrated_028_cli_repair001.intent.json"
    receipt = q.run_root / "boundary_integrated_028_cli_repair001.receipt.json"
    with q.lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q.authenticate_observer_exclusivity()
        authenticate_failure(q)
        if intent.exists() or receipt.exists() or q.adapter.runtime_directory().exists():
            raise ValueError("Named repair already has an artifact; no automatic retry")
        q.staged()
        job = q.previous_job()
        if not q.archive_ready(job):
            raise ValueError("Original stopped wrapper archive is not complete")
        durable_json(intent, dict(schema="diagnosed-native-CLI-repair-v1", manifest_sha256=q.config_sha256,
            failure_pins=FAILED_PINS, repair_code=code_manifest, change="explicit learning-rate-scale=1.0",
            requested_action="continue the previously authorized same-actor serial curriculum",
            native_check_PPO_updates=0, qualification=None), exclusive=True)
        result = q.step()
        durable_json(receipt, dict(schema="diagnosed-native-CLI-repair-result-v1", result=result,
            failure_pins=FAILED_PINS, repair_code=code_manifest, qualification=None), exclusive=True)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "launch_submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
