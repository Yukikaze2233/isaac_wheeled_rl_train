#!/usr/bin/env python3
"""Once-only adoption of source028's already completed, zero-PPO native check."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from queue_task_campaign_boundary_continuation import BoundaryContinuationQueue, NativeAdapter
from queue_task_campaign_continuation import durable_json, flag, read_json, sha
from repair_known_native_cli_startup import authenticate_failure


ADOPTION = "boundary_integrated_028_completion_adoption002"
NATIVE = "boundary_integrated_028_cli_repair001.native"
REPAIR_PINS = {
    "boundary_integrated_028_cli_repair001.intent.json": "83ce00789cc8270721595d9a076ea0c86ebd96b273cba896ff8ff451fd8489ca",
    "boundary_integrated_028_cli_repair001.receipt.json": "10fb8da4506e040a786661eb75d154d750787d0196ada3e8e259c9b97aafba35",
    NATIVE + "/invocation.intent.json": "d9429aa9862a7f8e9afe711470038acc633506c731fc48af8c96510cfc04dc91",
    NATIVE + "/receipt.json": "1b536519d00b13249750592ae0d980fcf7dfe7c6e1ece31a0b53d594ef81f181",
    NATIVE + "/source_selection.json": "a50715201c99d4de9eb41505525c214a03536cc9c931e1479ae7c879e9ea2978",
    NATIVE + "/scene/completion.json": "09102d256f400212503b23e37bbd632585038d52a1b1c8216344ca4c8ba0d2f1",
    NATIVE + "/scene/startup.json": "d305f7d360fbdfcea7f2bb66dd572a934137335703e760408a01769d5677f4ab",
}
CODE_FILES = {"adopt_known_native_cli_completion.py", "repair_known_native_cli_startup.py",
              "queue_task_campaign_boundary_continuation.py", "queue_task_campaign_continuation.py"}


def authenticate_existing_native(q, evidence=None):
    """Authenticate this known filename failure, never rerun its native command."""
    authenticate_failure(q)
    if q.launch_log.exists():
        raise ValueError("Launch log already exists; never infer a retry is safe")
    for relative, digest in REPAIR_PINS.items():
        if sha(q.run_root / relative) != digest:
            raise ValueError("Original native completion evidence changed: " + relative)
    repair_intent = read_json(q.run_root / "boundary_integrated_028_cli_repair001.intent.json")
    repair_result = read_json(q.run_root / "boundary_integrated_028_cli_repair001.receipt.json")
    directory = q.run_root / NATIVE
    invocation, failed = read_json(directory / "invocation.intent.json"), read_json(directory / "receipt.json")
    admission = read_json(directory / "source_selection.json")
    completion, startup = read_json(directory / "scene/completion.json"), read_json(directory / "scene/startup.json")
    runtime, capacity = completion.get("startup", {}).get("runtime_preflight", {}), completion.get("scene_capacity", {})
    result = repair_result.get("result", {})
    if (repair_intent.get("schema") != "diagnosed-native-CLI-repair-v1"
            or repair_intent.get("manifest_sha256") != q.config_sha256
            or repair_intent.get("native_check_PPO_updates") != 0
            or repair_intent.get("qualification") is not None
            or result.get("status") != "attention"
            or result.get("detail") != "Actual joint native startup failed; retained receipt, no PPO launch"
            or result.get("manifest_sha256") != q.config_sha256
            or failed.get("status") != "failed" or failed.get("exit_code") != 0
            or failed.get("PPO_updates") != 0 or failed.get("report_sha256") is not None
            or failed.get("manifest_sha256") != q.config_sha256
            or invocation.get("command") != failed.get("command")
            or invocation.get("PPO_updates") != 0 or invocation.get("qualification") is not None
            or (directory / "scene/report.json").exists()):
        raise ValueError("Evidence does not match the proven completion/report filename mismatch")
    for option, value in (("--learning-rate-scale", "1.0"), ("--global-actor-update-offset", "0"),
                          ("--block-index", "0"), ("--validate-scene", "1"), ("--num-envs", "4096")):
        if flag(failed["command"], option) != value:
            raise ValueError("Known native invocation differs: " + option)
    selection_sha = REPAIR_PINS[NATIVE + "/source_selection.json"]
    if (admission.get("schema") != "v6-integrated-production-admission-v1"
            or admission.get("qualification") is not None
            or admission.get("selection", {}).get("mode") != "rebase_from_authorized_REC"
            or completion.get("status") != "scene_validated"
            or completion.get("successful_updates") != 0 or completion.get("attempted_updates") != 0
            or completion.get("contract_sha256") != admission.get("target_contract_sha256")
            or completion.get("integrated_admission_sha256") != selection_sha
            or failed.get("selection_sha256") != selection_sha
            or invocation.get("selection_sha256") != selection_sha
            or completion.get("startup", {}).get("num_envs") != 4096
            or runtime.get("status") != "runtime_configuration_passed"
            or runtime.get("scoped_validation") != "v6-integrated-preflight-candidate-v1"
            or capacity.get("policy_steps") != 1 or capacity.get("physics_steps_per_env") != 4
            or startup.get("status") != "running"
            or startup.get("successful_updates") != 0 or startup.get("attempted_updates") != 0
            or startup.get("contract_sha256") != completion.get("contract_sha256")
            or startup.get("integrated_admission_sha256") != selection_sha
            or startup.get("startup", {}).get("num_envs") != 4096):
        raise ValueError("Existing native completion does not satisfy all original passed predicates")
    if evidence is not None and evidence["integrated_admission"] != admission:
        raise ValueError("Adoption changed the actual source selection or account")
    return dict(schema="diagnosed-native-completion-adoption-v1", status="passed",
        manifest_sha256=q.config_sha256, selection_sha256=selection_sha,
        command=failed["command"], exit_code=0, PPO_updates=0, qualification=None,
        report_file=str(directory / "scene/completion.json"),
        report_sha256=REPAIR_PINS[NATIVE + "/scene/completion.json"],
        startup_sha256=REPAIR_PINS[NATIVE + "/scene/startup.json"],
        original_native_receipt_sha256=REPAIR_PINS[NATIVE + "/receipt.json"],
        repair_evidence_pins=REPAIR_PINS, wall_seconds=failed["wall_seconds"],
        additional_native_runs=0, scope="existing 4096 native startup, one policy/four physics steps; zero PPO, no functional qualification")


class CompletionAdoptionAdapter(NativeAdapter):
    def validate_runtime(self, evidence):
        result = authenticate_existing_native(self.q, evidence)
        durable_json(self.q.run_root / (ADOPTION + ".native.receipt.json"), result, exclusive=True)
        return result


def authenticate_code(directory):
    manifest = read_json(directory / "code_manifest.json")
    if (manifest.get("schema") != "frozen-known-native-completion-adoption-code-v1"
            or set(manifest.get("files", {})) != CODE_FILES):
        raise ValueError("External adoption must pin its exact four code files")
    for name, digest in manifest["files"].items():
        if sha(directory / name) != digest:
            raise ValueError("Frozen external adoption code differs: " + name)
    return manifest


def run_adoption(q, code_manifest):
    intent = q.run_root / (ADOPTION + ".intent.json")
    receipt = q.run_root / (ADOPTION + ".receipt.json")
    native_receipt = q.run_root / (ADOPTION + ".native.receipt.json")
    with q.lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q.authenticate_observer_exclusivity()
        if any(path.exists() for path in (intent, receipt, native_receipt)):
            raise ValueError("Named adoption already has an artifact; no automatic retry")
        native = authenticate_existing_native(q)
        q.staged()
        if not q.archive_ready(q.previous_job()):
            raise ValueError("Original stopped wrapper archive is not complete")
        durable_json(intent, dict(schema="diagnosed-native-completion-adoption-intent-v1",
            manifest_sha256=q.config_sha256, repair_evidence_pins=REPAIR_PINS,
            adoption_code=code_manifest, adopted_native=native, additional_native_runs=0,
            requested_action="continue the previously authorized same-actor serial curriculum",
            qualification=None), exclusive=True)
        result = q.step()
        durable_json(receipt, dict(schema="diagnosed-native-completion-adoption-result-v1", result=result,
            repair_evidence_pins=REPAIR_PINS, adoption_code=code_manifest,
            additional_native_runs=0, qualification=None), exclusive=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    code_manifest = authenticate_code(Path(__file__).parent)
    q = BoundaryContinuationQueue(read_json(args.manifest), adapter_factory=CompletionAdoptionAdapter)
    result = run_adoption(q, code_manifest)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "launch_submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
