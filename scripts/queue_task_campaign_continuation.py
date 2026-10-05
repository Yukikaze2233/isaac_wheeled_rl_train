#!/usr/bin/env python3
"""Queue one frozen v16 native continuation after v15 finishes naturally.

Run outside both frozen source directories, with the simulation runtime already
inherited from its tmux/session environment. This observer never signals an
owner, changes a frozen file, reads TensorBoard, or imports a GPU runtime.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
from types import ModuleType

SCHEMA = "task-campaign-continuation-queue-v1"
NATURAL = "task_domains_finished_unassessed"
CAMPAIGN = "v6-task-domains-20261001"
DOMAINS = ("flat", "recovery", "speed", "terrain", "jump", "stepjump")
FINISHED = {"domain_review_finished_unassessed", "domain_schedule_finished_unassessed",
            "domain_budget_exhausted_unassessed"}
ROLES = {"chassis_remote_job.py": "wrapper", "run_task_campaign.py": "supervisor",
         "train_chassis.py": "worker", "evaluate_chassis.py": "evaluator"}
VALIDATION_CODE = """
import json, pathlib, sys
source, root, receipt = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(source / 'src'))
sys.path.insert(0, str(source / 'scripts'))
from chassis_remote_job import validate_source_revision
data = json.loads(sys.stdin.read())
result = validate_source_revision(root, data['command'], data['previous'], receipt)
print(json.dumps({'status': 'passed', 'receipt': result}))
"""


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def durable_json(path, value, *, exclusive=False):
    """Publish atomically; exclusive intent publication admits at most one launch."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def flag(command, name):
    if command.count(name) != 1 or command.index(name) + 1 >= len(command):
        raise ValueError(f"Native command needs exactly one {name}")
    return command[command.index(name) + 1]


def actual_entry(command):
    for value in command[1:]:
        if value in ("-c", "-m"):
            return None
        if value.endswith(".py"):
            return Path(value)
    return None


def frozen_module(name, path):
    # Compile the sealed source directly: a normal SourceFileLoader can write a
    # __pycache__ overlay when the external observer was not started with -B.
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def owned_processes(root, proc_root=Path("/proc")):
    """Classify the actual entry, avoiding the wrapper's embedded child argv."""
    rows = []
    for path in Path(proc_root).glob("[0-9]*/cmdline"):
        try:
            command = [os.fsdecode(value) for value in path.read_bytes().split(b"\0") if value]
        except FileNotFoundError:
            continue
        entry = actual_entry(command)
        if entry is None or entry.name not in ROLES:
            continue
        if any(Path(value).is_absolute() and Path(value).is_relative_to(root)
               for value in command):
            rows.append(dict(pid=int(path.parent.name), role=ROLES[entry.name], entry=str(entry)))
    return sorted(rows, key=lambda row: row["pid"])


@contextmanager
def account_ownership(path):
    """Probe the native durable claim lock without stealing an active owner."""
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


class ContinuationQueue:
    def __init__(self, manifest, *, proc_root=Path("/proc"), launcher=subprocess.Popen):
        self.config = dict(manifest)
        self.proc_root, self.launcher = Path(proc_root), launcher
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported continuation manifest schema")
        for key in ("old_contract_sha256", "new_contract_sha256", "source_archive_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", manifest.get(key, "")):
                raise ValueError(f"Invalid {key}")
        for key in ("old_source_commit", "source_commit"):
            if not re.fullmatch(r"[0-9a-f]{40}", manifest.get(key, "")):
                raise ValueError(f"Invalid {key}")
        for key in ("run_root", "old_source_root", "new_source_root", "source_archive",
                    "old_source_revision_receipt", "source_revision_receipt", "queue_receipt",
                    "status_path", "launch_log"):
            value = manifest.get(key)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"{key} must be an absolute path")
            setattr(self, key, Path(value).resolve())
        self.train = self.run_root / "train"
        if (self.new_source_root == self.old_source_root
                or self.new_source_root.is_relative_to(self.old_source_root)
                or self.old_source_root.is_relative_to(self.new_source_root)):
            raise ValueError("Continuation requires a separate frozen source")
        for key, expected in (("old_contract_relative_path", "contracts/v6_task_campaign_v15.json"),
                              ("new_contract_relative_path", "contracts/v6_task_campaign_v16.json")):
            if manifest.get(key) != expected:
                raise ValueError("Queue supports only the named v15 to v16 continuation")
        self.old_contract = self.old_source_root / manifest["old_contract_relative_path"]
        self.new_contract = self.new_source_root / manifest["new_contract_relative_path"]
        self.lock = self.run_root / ".task_campaign_continuation.lock"
        self.intent = self.run_root / ".task_campaign_continuation_v16.intent.json"
        outputs = [self.source_revision_receipt, self.queue_receipt, self.status_path, self.launch_log]
        reserved = {"job.json", "delivery.json", "archiver_status.json", "archiver_error.json",
                    self.lock.name, self.intent.name, self.old_source_revision_receipt.name}
        if (len(set(outputs)) != len(outputs)
                or any(path.parent != self.run_root or path.name in reserved for path in outputs)):
            raise ValueError("Queue output paths must be distinct, external run-root files")
        if self.source_revision_receipt.name != "source_revision_016.json":
            raise ValueError("Use the named immutable source_revision_016.json receipt")
        command = manifest.get("command")
        if (not isinstance(command, list) or not command
                or any(not isinstance(value, str) or not value or "\0" in value for value in command)
                or not Path(command[0]).is_absolute()
                or actual_entry(command) != self.new_source_root / "scripts/run_task_campaign.py"
                or flag(command, "--contract") != str(self.new_contract)
                or flag(command, "--run-dir") != str(self.train)
                or flag(command, "--num-envs") != "4096"
                or command.count("--research") != 1
                or any(value in command for value in ("--preflight-only", "--evaluate-only"))):
            raise ValueError("Queue requires the exact native 4096 research training command")
        self.command = command
        self.config_sha256 = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    def publish(self, status, reason, **details):
        summary = dict(schema=SCHEMA, status=status, reason=reason,
                       manifest_sha256=self.config_sha256, run_root=str(self.run_root), **details)
        previous = read_json(self.status_path) if self.status_path.exists() else {}
        if {key: value for key, value in previous.items() if key != "changed_at_utc"} != summary:
            summary["changed_at_utc"] = datetime.now(timezone.utc).isoformat()
            durable_json(self.status_path, summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
        return summary

    def previous_job(self):
        job = read_json(self.run_root / "job.json")
        expected = list(self.command)
        expected[expected.index(str(self.new_source_root / "scripts/run_task_campaign.py"))] = str(
            self.old_source_root / "scripts/run_task_campaign.py")
        expected[expected.index("--contract") + 1] = str(self.old_contract)
        revision = read_json(self.old_source_revision_receipt)
        if (job.get("command") != expected or sha(self.old_contract) != self.config["old_contract_sha256"]
                or revision.get("schema") != "task-plan-revision-v15"
                or revision.get("command") != expected
                or revision.get("new_source_root") != str(self.old_source_root)
                or revision.get("source_commit") != self.config["old_source_commit"]
                or revision.get("new_contract_sha256") != self.config["old_contract_sha256"]
                or sha(self.train / "study_plan.json") != self.config["old_contract_sha256"]):
            raise ValueError("Current job/study plan is not the authenticated frozen v15 predecessor")
        return job

    def completed_evidence(self, plan, state):
        completion = read_json(self.train / "completion.json")
        account = read_json(self.train / "consumption.json")
        claim_path = Path(plan["budget_source_stage"]) / (CAMPAIGN + ".account.json")
        claim = read_json(claim_path)
        if (state != completion or claim != account
                or account.get("campaign_id") != CAMPAIGN
                or account.get("run_directory") != str(self.train)
                or account.get("contract_sha256") != self.config["old_contract_sha256"]
                or account.get("rollback_refunds_consumed_updates") is not False):
            raise ValueError("Terminal completion, consumption and durable v15 claim disagree")
        # Recompute the authoritative account without loading any learning model.
        campaign = frozen_module("continuation_frozen_account",
                                 self.new_source_root / "src/wheeled_tasks/chassis/task_campaign.py")
        totals = campaign.account_updates(plan, account["attempts"])
        if any(account.get(key) != value or state.get(key) != value for key, value in totals.items()):
            raise ValueError("Terminal v15 accounting does not match every charged attempt")
        seals = {}
        if set(state.get("domains", {})) != set(DOMAINS):
            raise ValueError("Natural completion must include all six domains")
        for row in plan["domains"]:
            name = row["name"]
            record = state["domains"][name]
            checkpoint = Path(record.get("candidate_checkpoint", ""))
            if (record.get("status") not in FINISHED or not checkpoint.is_absolute()
                    or not checkpoint.resolve().is_relative_to(self.train / name)
                    or record.get("domain_qualified") is not None):
                raise ValueError(f"Domain {name} lacks its finished unassessed own-domain candidate")
            seal_path = checkpoint.with_suffix(".checkpoint.json")
            seal = read_json(seal_path)
            checkpoint_sha = sha(checkpoint)
            sidecar = checkpoint.with_suffix(".contract.json")
            contract_sha = sha(sidecar if sidecar.exists() else checkpoint.parent / "contract.json")
            if (seal.get("schema") != "chassis-learning-checkpoint-v1" or seal.get("sealed") is not True
                    or seal.get("checkpoint_sha256") != checkpoint_sha
                    or record.get("candidate_checkpoint_sha256") != checkpoint_sha
                    or seal.get("contract_sha256") != contract_sha or contract_sha != row["sha256"]
                    or seal.get("task_campaign", {}).get("domain") != name
                    or type(seal.get("learning_updates")) is not int or seal["learning_updates"] < 0
                    or type(seal.get("task_actor_update")) is not int or seal["task_actor_update"] < 0):
                raise ValueError(f"Domain {name} immutable checkpoint seal differs")
            seals[name] = dict(path=str(checkpoint), sha256=checkpoint_sha,
                               seal_sha256=sha(seal_path), contract_sha256=contract_sha,
                               task_actor_update=seal["task_actor_update"])
        return dict(completion_sha256=sha(self.train / "completion.json"),
                    consumption_sha256=sha(self.train / "consumption.json"),
                    claim_path=str(claim_path), claim_sha256=sha(claim_path),
                    consumed_updates=account["consumed_updates"], seals=seals)

    def migration_receipt(self, job):
        return dict(schema="task-plan-revision-v16", previous_job_sha256=sha(self.run_root / "job.json"),
            command=self.command, source_commit=self.config["source_commit"],
            old_source_root=str(self.old_source_root), new_source_root=str(self.new_source_root),
            new_contract_relative_path=self.config["new_contract_relative_path"],
            old_contract_sha256=self.config["old_contract_sha256"],
            new_contract_sha256=self.config["new_contract_sha256"],
            source_archive=str(self.source_archive), source_archive_sha256=self.config["source_archive_sha256"])

    def validate_staged_source(self):
        """CPU-only predeployment audit without creating a migration/launch receipt."""
        if (sha(self.new_contract) != self.config["new_contract_sha256"]
                or sha(self.source_archive) != self.config["source_archive_sha256"]):
            raise ValueError("Staged frozen v16 plan/archive SHA differs")
        names = set()
        with tarfile.open(self.source_archive, "r:*") as archive:
            if archive.pax_headers.get("comment") != self.config["source_commit"]:
                raise ValueError("Staged archive does not bind the expected frozen source commit")
            for member in archive:
                relative = Path(member.name)
                target = self.new_source_root / relative
                if (relative.is_absolute() or ".." in relative.parts
                        or not target.resolve().is_relative_to(self.new_source_root)
                        or not (member.isfile() or member.isdir())):
                    raise ValueError("Staged source archive contains an unsupported path or link")
                if member.isdir():
                    continue
                if member.name in names:
                    raise ValueError("Staged source archive repeats a file")
                names.add(member.name)
                with archive.extractfile(member) as content:
                    checksum = hashlib.sha256()
                    for block in iter(lambda: content.read(1024 * 1024), b""):
                        checksum.update(block)
                if not target.is_file() or sha(target) != checksum.hexdigest():
                    raise ValueError(f"Staged frozen source differs: {member.name}")
        required = {"scripts/run_task_campaign.py", "scripts/chassis_remote_job.py",
                    "src/wheeled_tasks/chassis/task_campaign.py", self.config["new_contract_relative_path"]}
        if not required <= names:
            raise ValueError("Staged source archive is missing native continuation files")
        return dict(status="passed", manifest_sha256=self.config_sha256,
                    source_commit=self.config["source_commit"], source_archive_sha256=self.config["source_archive_sha256"])

    def validate_migration(self, job):
        self.validate_staged_source()
        receipt = self.migration_receipt(job)
        if self.source_revision_receipt.exists():
            if read_json(self.source_revision_receipt) != receipt:
                raise ValueError("Immutable v16 source revision receipt already differs")
        else:
            durable_json(self.source_revision_receipt, receipt, exclusive=True)
        # A fresh interpreter avoids an old source's cached wheeled_tasks package
        # silently supplying the plan validator after a prefix change.
        result = subprocess.run([self.command[0], "-B", "-c", VALIDATION_CODE,
            str(self.new_source_root), str(self.run_root), str(self.source_revision_receipt)],
            input=json.dumps(dict(command=self.command, previous=job)),
            text=True, capture_output=True, timeout=60, shell=False)
        if result.returncode:
            raise ValueError("Frozen v16 source validation failed: " + result.stderr[-3000:])
        if read_json(self.source_revision_receipt) != receipt or json.loads(result.stdout).get("status") != "passed":
            raise ValueError("Frozen v16 source validation returned an unexpected result")
        return receipt

    def step(self):
        """One lightweight observation; expensive authentication only at natural exit."""
        if self.queue_receipt.exists():
            receipt = read_json(self.queue_receipt)
            if receipt.get("manifest_sha256") != self.config_sha256:
                return self.publish("attention", "existing_queue_receipt_has_different_manifest")
            return self.publish(receipt["status"], receipt["reason"],
                                queue_receipt=str(self.queue_receipt), pid=receipt.get("pid"))
        if self.intent.exists():
            return self.publish("attention", "durable_launch_intent_exists_do_not_launch_again",
                                intent=str(self.intent))
        try:
            job = self.previous_job()
            state = read_json(self.train / "campaign_state.json")
            if state.get("contract_sha256") != self.config["old_contract_sha256"]:
                raise ValueError("Campaign state is not the named old v15 plan")
            if state.get("status") in ("stopped", "failed"):
                return self.publish("attention", "v15_stopped_or_failed_requires_review",
                                    campaign_status=state["status"], detail=state.get("reason") or state.get("error"))
            processes = owned_processes(self.run_root, self.proc_root)
            if processes:
                return self.publish("waiting", "v15_owned_processes_still_active",
                                    campaign_status=state.get("status"), processes=processes)
            if state.get("status") != NATURAL:
                return self.publish("attention", "no_owner_without_natural_v15_completion",
                                    campaign_status=state.get("status"))
            plan = read_json(self.old_contract)
            claim = Path(plan["budget_source_stage"]) / (CAMPAIGN + ".account.json")
            if not claim.exists():
                raise ValueError("The existing durable campaign budget claim is missing")
            with account_ownership(claim.with_suffix(".lock")) as acquired:
                if not acquired:
                    return self.publish("waiting", "native_campaign_account_lock_still_owned")
                # Authenticate every staged byte before executing even the CPU
                # accounting helper from the new frozen source.
                self.validate_staged_source()
                evidence = self.completed_evidence(plan, state)
                revision = self.validate_migration(job)
                # Recheck the live owner and immutable job after terminal hashing.
                if owned_processes(self.run_root, self.proc_root):
                    return self.publish("waiting", "owner_appeared_during_terminal_authentication")
                if sha(self.run_root / "job.json") != revision["previous_job_sha256"]:
                    raise ValueError("Previous job changed during terminal authentication")
                wrapper = [self.command[0], "-B", str(self.new_source_root / "scripts/chassis_remote_job.py"),
                    "--run-root", str(self.run_root), "--resume-existing", "--source-revision-receipt",
                    str(self.source_revision_receipt), "--", *self.command]
                intent = dict(schema=SCHEMA, status="launch_intent", manifest_sha256=self.config_sha256,
                    created_at_utc=datetime.now(timezone.utc).isoformat(), revision=revision,
                    predecessor=evidence, wrapper_command=wrapper)
                durable_json(self.intent, intent, exclusive=True)
            # Native supervisor needs the account lock itself. Launch with inherited
            # runtime, literal argv and no shell; the persistent intent forbids retries.
            with self.launch_log.open("xb") as log:
                process = self.launcher(wrapper, cwd=self.new_source_root,
                    stdout=log, stderr=subprocess.STDOUT, shell=False, start_new_session=True)
            receipt = dict(intent, status="launch_submitted", reason="natural_v15_completion_authenticated",
                           pid=process.pid, launch_log=str(self.launch_log), qualification=None)
            durable_json(self.queue_receipt, receipt, exclusive=True)
            return self.publish(receipt["status"], receipt["reason"],
                                queue_receipt=str(self.queue_receipt), pid=process.pid)
        except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as error:
            return self.publish("attention", "continuation_authentication_or_launch_failed", detail=str(error))

    def run(self, *, once=False, poll_seconds=60):
        with self.lock.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps(dict(status="queue_already_owned", run_root=str(self.run_root))), flush=True)
                return 0
            while True:
                result = self.step()
                if result["status"] == "attention":
                    return 2
                if once or result["status"] == "launch_submitted":
                    return 0
                time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--once", action="store_true", help="Observe once, launching only if admitted")
    parser.add_argument("--validate-manifest", action="store_true",
                        help="Audit staged frozen source only; create no revision or launch receipt")
    args = parser.parse_args()
    if not 60 <= args.poll_seconds <= 3600:
        parser.error("Polling must remain lightweight: 60 to 3600 seconds")
    queue = ContinuationQueue(read_json(args.manifest))
    if args.validate_manifest:
        print(json.dumps(queue.validate_staged_source(), sort_keys=True))
        return
    raise SystemExit(queue.run(once=args.once, poll_seconds=args.poll_seconds))


if __name__ == "__main__":
    main()
