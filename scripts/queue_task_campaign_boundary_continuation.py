#!/usr/bin/env python3
"""Adopt a named frozen revision once after REC's real v21 endpoint.

This external observer has no supervisor boundary hook. It waits for a new
speed attempt's first complete native snapshot before signalling its pinned
supervisor. All consumption, latest-own selection and migration use native
helpers in a fresh CPU interpreter. Run --check-only for read-only admission.
An ambiguous durable signal/launch intent is never retried automatically.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time

from queue_task_campaign_continuation import (
    ContinuationQueue, account_ownership, actual_entry, durable_json, flag, read_json, sha,
)

SCHEMA = "task-campaign-boundary-continuation-v1"
OLD_COMMIT = "6f07ec298e01d2686216787cfba34b394151f1aa"
TARGET_REVISIONS = {
    "contracts/v6_task_campaign_v22.json": (22, 24),
    "contracts/v6_task_campaign_v23.json": (23, 25),
    "contracts/v6_task_campaign_v24.json": (24, 26),
}
ROLES = {"chassis_remote_job.py": "wrapper", "run_task_campaign.py": "supervisor",
         "train_chassis.py": "worker", "evaluate_chassis.py": "evaluator"}

# No Simulator/runner is instantiated. Native helpers own the budget calculation,
# worker compatibility, latest-own checkpoint choice and adaptation rule.
NATIVE_CPU = r'''
import json, pathlib, sys
source, root = map(pathlib.Path, sys.argv[1:3])
sys.path[:0] = [str(source/'src'), str(source/'scripts')]
data = json.loads(sys.stdin.read())
from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count, checkpoint_contract_path
from wheeled_tasks.chassis.task_campaign import account_updates, update_attempt, validate_task_resume
from train_chassis import resume_critic_warmup
import torch
from collections import Counter
def checked(path, domain, contract_sha):
    path = pathlib.Path(path)
    seal = json.loads(path.with_suffix('.checkpoint.json').read_text())
    count = checkpoint_update_count(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    actor, critic, opt = (saved[k] for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict'))
    if len(actor) != 9 or len(critic) != 8 or actor['distribution.std_param'].shape != (6,):
        raise ValueError('Incomplete 9 actor/std or 8 critic state')
    parameters = [*actor.values(), *critic.values()]
    if (not all(torch.is_tensor(v) and v.is_floating_point() and v.numel() > 0
                and torch.isfinite(v).all().item() for v in parameters)
            or not (actor['distribution.std_param'] > 0).all().item()):
        raise ValueError('Nonfinite actor/critic state')
    params = [p for g in opt['param_groups'] for p in g['params']]
    if len(params) != 17 or len(set(params)) != 17 or set(params) != set(opt['state']):
        raise ValueError('Incomplete unique 17-parameter Adam state')
    moments = []
    for value in opt['state'].values():
        if not {'exp_avg','exp_avg_sq','step'} <= value.keys() or not all(
            torch.isfinite(torch.as_tensor(value[k])).all().item() for k in ('exp_avg','exp_avg_sq','step')):
            raise ValueError('Incomplete/nonfinite Adam moments')
        if value['exp_avg'].shape != value['exp_avg_sq'].shape or not (value['exp_avg_sq'] >= 0).all().item():
            raise ValueError('Adam moment shapes/variance differ')
        moments.append(tuple(value['exp_avg'].shape))
    if Counter(moments) != Counter(tuple(v.shape) for v in parameters):
        raise ValueError('Adam moments do not match all native parameter shapes')
    infos = saved['infos']
    if (seal.get('contract_sha256') != contract_sha or infos.get('contract_sha256') != contract_sha
        or seal.get('task_campaign',{}).get('domain') != domain
        or infos.get('task_campaign',{}).get('domain') != domain
        or infos.get('successful_updates_total') != count
        or infos.get('task_actor_update') != seal.get('task_actor_update')):
        raise ValueError('Full learning state differs from native seal/domain')
    from wheeled_tasks.chassis.task_campaign import digest
    return dict(path=str(path), sha256=digest(path), seal_sha256=digest(path.with_suffix('.checkpoint.json')),
        contract_sha256=contract_sha, learning_updates=count,
        learning_lineage_updates=seal.get('learning_lineage_updates'),
        task_actor_update=seal['task_actor_update'], stage_actor_update=seal['stage_actor_update'],
        critic_warmup_remaining=infos.get('critic_warmup_remaining',0)), infos
if data['op'] == 'preflight':
    # Temporary observer receipt only; no job, claim or migration report writes.
    import tempfile
    from chassis_remote_job import validate_source_revision
    with tempfile.TemporaryDirectory(prefix='boundary-native-preflight-') as temp:
        receipt = pathlib.Path(temp)/'revision.json'
        receipt.write_text(json.dumps(data['revision']))
        validate_source_revision(root, data['command'], data['previous'], receipt)
    from run_task_campaign import validate_contracts
    plan = json.loads((source/data['revision']['new_contract_relative_path']).read_text())
    roles = validate_contracts(plan)
    result = dict(status='passed', design_preflight_receipts=len(roles))
elif data['op'] == 'seal':
    result, _ = checked(data['path'], data['domain'], data['contract_sha256'])
else:
    from chassis_remote_job import validate_source_revision
    from run_task_campaign import TaskCampaign
    old, new = (json.loads(pathlib.Path(data[k]).read_text()) for k in ('old_contract','new_contract'))
    validate_source_revision(root, data['command'], data['previous'], pathlib.Path(data['receipt']))
    train = root/'train'
    state, completion, account = (json.loads((train/name).read_text()) for name in
        ('campaign_state.json','completion.json','consumption.json'))
    claim = pathlib.Path(old['budget_source_stage'])/(old['campaign_id']+'.account.json')
    if state != completion or account != json.loads(claim.read_text()) or account.get('rollback_refunds_consumed_updates') is not False:
        raise ValueError('Stopped state/completion/durable account disagree')
    if account.get('run_directory') != str(train) or account.get('contract_sha256') != data['old_contract_sha256']:
        raise ValueError('Account belongs to another run or plan')
    records = account['attempts']
    for p in sorted(train.glob('*/attempt_*/train/progress.json')):
        records = update_attempt(old, records, str(p.parent.relative_to(train)), p.parents[2].name, json.loads(p.read_text()))
    totals = account_updates(old, records)
    if records != account['attempts'] or any(account.get(k) != v or state.get(k) != v for k,v in totals.items()):
        raise ValueError('Latest progress not completely charged in final native account')
    if account_updates(new, records) != totals:
        raise ValueError('Migration changed charged consumption')
    campaign = TaskCampaign.__new__(TaskCampaign)
    campaign.root, campaign.plan, campaign.report = train, (old if new['contract_id'] == 'v6-task-campaign-v24' else new), state
    latest = {}
    for row in new['domains']:
        path = campaign.latest_checkpoint(row)
        if path is not None:
            seal = json.loads(path.with_suffix('.checkpoint.json').read_text())
            latest[row['name']], _ = checked(path, row['name'], seal['contract_sha256'])
    integrated_admission = None
    if new['contract_id'] == 'v6-task-campaign-v24':
        from wheeled_tasks.chassis.integrated_production import prepare_admission
        integrated_admission = prepare_admission(new, source, train)
    adaptations = {}
    for name in (() if integrated_admission is not None else ('speed','terrain')):
        row = next(r for r in new['domains'] if r['name'] == name)
        path = campaign.learning_source(row)
        src_contract = json.loads(checkpoint_contract_path(path).read_text())
        target = json.loads((source/row['contract']).read_text())
        validate_task_resume(src_contract, target)
        seal, infos = checked(path, name, latest[name]['contract_sha256'])
        warmup = resume_critic_warmup(target, infos, contract_sha256=row['sha256'], task_change=True)
        if name == 'speed' and (src_contract != target or warmup != infos.get('critic_warmup_remaining',0)):
            raise ValueError('Same speed contract would repeat adaptation')
        if name == 'terrain' and warmup != 10:
            raise ValueError('Terrain native own-domain revision adaptation must be exactly 10')
        adaptations[name] = dict(source=seal, critic_warmup_updates=warmup)
    from wheeled_tasks.chassis.task_campaign import digest
    result = dict(totals=totals, latest_own=latest, adaptation=adaptations,
        claim_sha256=digest(claim), consumption_sha256=digest(train/'consumption.json'),
        state_sha256=digest(train/'campaign_state.json'), completion_sha256=digest(train/'completion.json'))
    if integrated_admission is not None:
        result['integrated_admission'] = integrated_admission
print(json.dumps(result, allow_nan=False))
'''


def process_identity(pid, proc_root=Path("/proc")):
    path = Path(proc_root) / str(pid)
    try:
        command = [os.fsdecode(p) for p in (path / "cmdline").read_bytes().split(b"\0") if p]
        tail = (path / "stat").read_text().rpartition(")")[2].split()
    except FileNotFoundError:
        return None
    if not command or tail[0] == "Z":
        return None
    entry = actual_entry(command)
    return dict(pid=int(pid), starttime=int(tail[19]), ppid=int(tail[1]),
                entry=str(entry) if entry else None, command=command)


def _linux_pidfd_syscall(number, *args):
    # The Isaac bundled interpreter can omit Python's pidfd functions. These
    # two syscall numbers are Linux x86_64/aarch64 ABI (434 open, 424 signal).
    # Unsupported platforms refuse; there is never an os.kill fallback.
    if sys_platform() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise ValueError("Linux x86_64/aarch64 pidfd support is required")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def sys_platform():
    return platform.system()


def open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    return _linux_pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def send_pidfd_sigterm(descriptor):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(descriptor, signal.SIGTERM, None, 0)
    return _linux_pidfd_syscall(424, ctypes.c_int(descriptor), ctypes.c_int(signal.SIGTERM),
                              ctypes.c_void_p(), ctypes.c_uint(0))


class NativeAdapter:
    def __init__(self, queue):
        self.q = queue

    def call(self, data):
        q = self.q
        result = subprocess.run([q.command[0], "-B", "-c", NATIVE_CPU,
            str(q.new_source_root), str(q.run_root)], input=json.dumps(data),
            text=True, capture_output=True, timeout=120, shell=False)
        if result.returncode:
            raise ValueError("Native CPU admission failed: " + result.stderr[-2500:])
        return json.loads(result.stdout)

    def seal(self, path, domain, contract_sha256):
        return self.call(dict(op="seal", path=str(path), domain=domain, contract_sha256=contract_sha256))

    def prepare(self, job):
        q = self.q
        return self.call(dict(op="prepare", old_contract=str(q.old_contract), new_contract=str(q.new_contract),
            old_contract_sha256=q.config["old_contract_sha256"], command=q.command,
            previous=job, receipt=str(q.source_revision_receipt)))

    def validate_runtime(self, evidence):
        """One isolated native scene check, after the old owners have exited.

        The real admitted REC source supplies all clocks. This does not collect
        a rollout, construct PPO, or write the campaign consumption account.
        A partial invocation is retained and never submitted a second time.
        """
        q = self.q
        admission = evidence["integrated_admission"]
        if (q.target_revision != 24 or admission.get("schema") != "v6-integrated-production-admission-v1"
                or admission.get("qualification") is not None):
            raise ValueError("Native startup requires its named actual joint admission")
        selection = admission["selection"]
        if selection.get("mode") != "rebase_from_normal_REC":
            raise ValueError("First joint native startup must use the real completed REC parent")
        directory = q.run_root / "boundary_integrated_026.native"
        directory.mkdir(exist_ok=True)
        source_file = directory / "source_selection.json"
        selection_sha = __import__("hashlib").sha256(
            (json.dumps(admission, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
        native_intent = directory / "invocation.intent.json"
        native_receipt = directory / "receipt.json"
        if native_receipt.exists():
            result = read_json(native_receipt)
            report_path = directory / "scene" / "report.json"
            if (result.get("status") != "passed" or result.get("manifest_sha256") != q.config_sha256
                    or result.get("selection_sha256") != selection_sha
                    or result.get("report_sha256") != sha(report_path)
                    or not source_file.is_file() or sha(source_file) != selection_sha):
                raise ValueError("Existing native startup receipt does not authenticate this admission")
            return result
        if native_intent.exists():
            raise ValueError("Partial native startup intent exists; never resubmit automatically")
        durable_json(source_file, admission, exclusive=True)
        plan = read_json(q.new_contract)
        row = next(row for row in plan["domains"] if row["name"] == "speed")
        checkpoint = selection["checkpoint"]["path"]
        seal = read_json(Path(checkpoint).with_suffix(".checkpoint.json"))
        argv = [q.command[0], "-B", str(q.new_source_root / "scripts/train_chassis.py"),
            "--contract", str(q.new_source_root / row["contract"]), "--stage", row["kind"],
            "--research", "--run-dir", str(directory / "scene"), "--device", "cuda:0",
            "--num-envs", "4096", "--updates", "1", "--validate-scene", "1",
            "--level", str(row.get("level", 1.)), "--seed", str(plan["seed"]),
            "--resume", str(checkpoint), "--resume-task-change",
            "--consumed-updates", str(evidence["totals"]["consumed_updates"]),
            "--actor-updates-consumed", str(evidence["totals"]["actor_updates_consumed"]),
            "--stage-actor-update", str(seal["stage_actor_update"]),
            "--task-actor-update", str(selection["fresh_domain_actor_updates"]),
            "--integrated-source-selection-file", str(source_file),
            "--integrated-source-selection-sha256", selection_sha,
            "--max-runtime-seconds", "600"]
        invocation = dict(manifest_sha256=q.config_sha256, selection_sha256=selection_sha,
                          command=argv, PPO_updates=0, qualification=None)
        durable_json(native_intent, invocation, exclusive=True)
        wrapped = ["/bin/bash", "-c", 'set -e; source "$1"; shift; exec "$@"',
                   "integrated-native-startup", str(q.runtime_script), *argv]
        started = time.monotonic()
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            process = subprocess.run(wrapped, cwd=q.new_source_root, stdout=stdout, stderr=stderr,
                                     shell=False, timeout=720)
        report_path = directory / "scene" / "report.json"
        report = read_json(report_path) if report_path.is_file() else {}
        startup = report.get("startup", {})
        runtime = startup.get("runtime_preflight", {})
        capacity = report.get("scene_capacity", {})
        passed = (process.returncode == 0 and report.get("status") == "scene_validated"
            and report.get("successful_updates") == 0 and report.get("attempted_updates") == 0
            and report.get("contract_sha256") == admission["target_contract_sha256"]
            and report.get("integrated_admission_sha256") == selection_sha
            and startup.get("num_envs") == 4096
            and runtime.get("status") == "runtime_configuration_passed"
            and runtime.get("scoped_validation") == "v6-integrated-preflight-candidate-v1"
            and capacity.get("policy_steps") == 1 and capacity.get("physics_steps_per_env") == 4)
        result = dict(invocation, status="passed" if passed else "failed", exit_code=process.returncode,
            wall_seconds=time.monotonic()-started, report_sha256=sha(report_path) if report_path.is_file() else None,
            scope="4096 native startup plus one policy/four physics steps; no PPO or functional qualification")
        durable_json(native_receipt, result, exclusive=True)
        if not passed:
            raise ValueError("Actual joint native startup failed; retained receipt, no PPO launch")
        return result

    def preflight(self, job):
        q = self.q
        return self.call(dict(op="preflight", command=q.command, previous=job, revision=q.migration_receipt()))

    def signal_once(self, owner, before_signal):
        """PID reuse cannot redirect this signal: a pidfd owns the target process."""
        descriptor = open_pidfd(owner["pid"])
        try:
            if process_identity(owner["pid"], self.q.proc_root) != owner:
                raise ValueError("Supervisor changed while opening pidfd")
            before_signal()
            send_pidfd_sigterm(descriptor)
        finally:
            os.close(descriptor)

    def launch(self, wrapper):
        q = self.q
        # Literal positional parameters, no interpolation/eval; source runtime
        # again even if this observer was launched without the simulation env.
        argv = ["/bin/bash", "-c", 'set -e; source "$1"; shift; exec "$@"',
                "boundary-native-runtime", str(q.runtime_script), *wrapper]
        with q.launch_log.open("xb") as log:
            return subprocess.Popen(argv, cwd=q.new_source_root, stdout=log,
                stderr=subprocess.STDOUT, shell=False, start_new_session=True)


class BoundaryContinuationQueue:
    def __init__(self, manifest, *, proc_root=Path("/proc"), adapter_factory=NativeAdapter):
        self.config = deepcopy(manifest)
        self.proc_root = Path(proc_root)
        if manifest.get("schema") != SCHEMA or manifest.get("old_source_commit") != OLD_COMMIT:
            raise ValueError("Queue requires the named source023/v21 predecessor")
        for key in ("old_contract_sha256", "new_contract_sha256", "source_archive_sha256",
                    "old_job_sha256", "old_source_revision_sha256", "runtime_script_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", manifest.get(key, "")):
                raise ValueError(f"Invalid {key}")
        if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_commit", "")):
            raise ValueError("Invalid frozen new source commit")
        pins = manifest.get("old_owners")
        if (not isinstance(pins, dict) or set(pins) != {"wrapper", "supervisor"}
                or any(not isinstance(pin, dict) or any(type(pin.get(k)) is not int or pin[k] <= 0
                    for k in ("pid", "starttime")) for pin in pins.values())):
            raise ValueError("Positive integer old wrapper/supervisor PID and starttime pins are required")
        for key in ("run_root", "old_source_root", "new_source_root", "source_archive",
                    "old_source_revision_receipt", "source_revision_receipt", "queue_receipt",
                    "status_path", "launch_log", "runtime_script"):
            value = manifest.get(key)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"{key} must be absolute")
            setattr(self, key, Path(value).resolve())
        self.train = self.run_root / "train"
        if (self.old_source_root == self.new_source_root or self.old_source_root.is_relative_to(self.new_source_root)
                or self.new_source_root.is_relative_to(self.old_source_root)):
            raise ValueError("A separate frozen source is required")
        target = TARGET_REVISIONS.get(manifest.get("new_contract_relative_path"))
        if (manifest.get("old_contract_relative_path") != "contracts/v6_task_campaign_v21.json"
                or target is None
                or self.source_revision_receipt.name != f"source_revision_{target[1]:03d}.json"):
            raise ValueError("Only named v21-to-v22/source024, v23/source025 or v24/source026 continuation is supported")
        self.target_revision, self.source_revision_index = target
        if self.target_revision in (23, 24):
            names = dict(queue_receipt=f"boundary_continuation_{self.source_revision_index:03d}.receipt.json",
                         status_path=f"boundary_continuation_{self.source_revision_index:03d}.status.json",
                         launch_log=f"boundary_continuation_{self.source_revision_index:03d}.launch.log")
            if any(getattr(self, key).name != name for key, name in names.items()):
                raise ValueError(f"New boundary target requires distinct named source{self.source_revision_index:03d} output paths")
        self.old_contract = self.old_source_root / manifest["old_contract_relative_path"]
        self.new_contract = self.new_source_root / manifest["new_contract_relative_path"]
        self.command = manifest.get("command")
        if (not isinstance(self.command, list) or not self.command
                or any(not isinstance(v, str) or not v or "\0" in v for v in self.command)
                or not Path(self.command[0]).is_absolute()
                or actual_entry(self.command) != self.new_source_root / "scripts/run_task_campaign.py"
                or flag(self.command, "--contract") != str(self.new_contract)
                or flag(self.command, "--run-dir") != str(self.train)
                or flag(self.command, "--num-envs") != "4096" or self.command.count("--research") != 1
                or any(v in self.command for v in ("--preflight-only", "--evaluate-only"))):
            raise ValueError("Exact native same-root 4096 research command is required")
        # Keep the deployed v22 observer's lock as the shared ownership lock.
        # A separate v23 lock would allow the already deployed v22 to race it.
        self.lock = self.run_root / ".task_campaign_boundary_v22.lock"
        prefix = f".task_campaign_boundary_v{self.target_revision}"
        self.state_path = self.run_root / (prefix + ".state.json")
        self.signal_intent = self.run_root / (prefix + ".signal.intent.json")
        self.intent = self.run_root / (prefix + ".launch.intent.json")
        outputs = [self.source_revision_receipt, self.queue_receipt, self.status_path, self.launch_log,
                   self.state_path, self.signal_intent, self.intent, self.lock]
        reserved = {"job.json", "delivery.json", "archiver_status.json", "completion.json",
                    "train.log", self.old_source_revision_receipt.name}
        if len(set(outputs)) != len(outputs) or any(p.parent != self.run_root or p.name in reserved for p in outputs):
            raise ValueError("Queue outputs must be distinct external run-root files")
        self.config_sha256 = __import__("hashlib").sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        self.adapter = adapter_factory(self)

    def publish(self, status, reason, **details):
        result = dict(schema=SCHEMA, manifest_sha256=self.config_sha256, status=status, reason=reason, **details)
        if not self.status_path.exists() or read_json(self.status_path) != result:
            durable_json(self.status_path, result)
            print(json.dumps(result, sort_keys=True), flush=True)
        return result

    def save_state(self, state):
        durable_json(self.state_path, dict(state, manifest_sha256=self.config_sha256))

    def authenticate_observer_exclusivity(self):
        """A new target cannot supersede an active or already acting observer."""
        for revision, index in TARGET_REVISIONS.values():
            if revision == self.target_revision:
                continue
            paths = (f".task_campaign_boundary_v{revision}.signal.intent.json",
                     f".task_campaign_boundary_v{revision}.launch.intent.json",
                     f"source_revision_{index:03d}.json",
                     f"boundary_continuation_{index:03d}.receipt.json")
            if any((self.run_root / name).exists() for name in paths):
                raise ValueError("Another boundary target has an action artifact; never supersede or retry")
        for path in self.proc_root.glob("[0-9]*/cmdline"):
            owner = process_identity(int(path.parent.name), self.proc_root)
            if (not owner or owner["pid"] == os.getpid() or not owner["entry"]
                    or Path(owner["entry"]).name != Path(__file__).name):
                continue
            manifest = flag(owner["command"], "--manifest")
            if manifest and Path(manifest).is_absolute() and Path(manifest).is_relative_to(self.run_root):
                raise ValueError("Another same-root boundary observer is active; retire the waiting observer first")

    def previous_job(self):
        job = read_json(self.run_root / "job.json")
        expected = list(self.command)
        expected[expected.index(str(self.new_source_root / "scripts/run_task_campaign.py"))] = str(self.old_source_root / "scripts/run_task_campaign.py")
        expected[expected.index("--contract") + 1] = str(self.old_contract)
        revision = read_json(self.old_source_revision_receipt)
        if (sha(self.run_root / "job.json") != self.config["old_job_sha256"] or job.get("command") != expected
                or sha(self.old_source_revision_receipt) != self.config["old_source_revision_sha256"]
                or revision.get("schema") != "task-plan-revision-v21" or revision.get("command") != expected
                or revision.get("new_source_root") != str(self.old_source_root)
                or revision.get("source_commit") != OLD_COMMIT
                or revision.get("new_contract_sha256") != self.config["old_contract_sha256"]
                or sha(self.old_contract) != self.config["old_contract_sha256"]
                or sha(self.train / "study_plan.json") != self.config["old_contract_sha256"]
                or sha(self.runtime_script) != self.config["runtime_script_sha256"]):
            raise ValueError("Old job/source/plan/runtime identity differs")
        return job

    def owners(self):
        owners = []
        for path in self.proc_root.glob("[0-9]*/cmdline"):
            value = process_identity(int(path.parent.name), self.proc_root)
            if not value or not value["entry"] or Path(value["entry"]).name not in ROLES:
                continue
            if any(Path(v).is_absolute() and Path(v).is_relative_to(self.run_root) for v in value["command"]):
                owners.append(dict(value, role=ROLES[Path(value["entry"]).name]))
        return sorted(owners, key=lambda v: v["pid"])

    def authenticate_owners(self, job, owners):
        for role in ("supervisor", "wrapper"):
            pin = self.config["old_owners"][role]
            rows = [p for p in owners if p["role"] == role]
            if len(rows) != 1 or any(rows[0].get(k) != pin[k] for k in ("pid", "starttime")):
                raise ValueError(f"Pinned old {role} ownership changed")
            if rows[0]["entry"] != str(self.old_source_root / "scripts" / ("run_task_campaign.py" if role == "supervisor" else "chassis_remote_job.py")):
                raise ValueError(f"Old {role} source differs")
        supervisor = next(p for p in owners if p["role"] == "supervisor")
        wrapper = next(p for p in owners if p["role"] == "wrapper")
        if supervisor["pid"] != job["pid"] or supervisor["command"] != job["command"] or supervisor["ppid"] != wrapper["pid"]:
            raise ValueError("Supervisor argv/job/parent differs")
        if (flag(wrapper["command"], "--run-root") != str(self.run_root)
                or wrapper["command"].count("--") != 1
                or wrapper["command"][wrapper["command"].index("--") + 1:] != job["command"]):
            raise ValueError("Wrapper run root/embedded supervisor argv differs")
        rows = {r["name"]: r for r in read_json(self.old_contract)["domains"]}
        for p in owners:
            if p["role"] in ("worker", "evaluator"):
                if p["ppid"] != supervisor["pid"] or Path(p["entry"]).parent != self.old_source_root / "scripts":
                    raise ValueError("Child owner is not this source/supervisor")
                directory = Path(flag(p["command"], "--run-dir"))
                if not directory.is_relative_to(self.train):
                    raise ValueError("Child run root differs")
                relative = directory.relative_to(self.train)
                if (len(relative.parts) != 3 or relative.parts[0] not in rows or relative.parts[2] != "train"
                        or flag(p["command"], "--contract") != str(self.old_source_root / rows[relative.parts[0]]["contract"])
                        or read_json(directory.parent / "invocation.json").get("command") != p["command"]):
                    raise ValueError("Child invocation/domain/contract differs")
        if sum(p["role"] == "worker" for p in owners) > 1 or any(p["role"] == "evaluator" for p in owners):
            raise ValueError("Unexpected duplicate worker or evaluator at training boundary")
        return {k: v for k, v in supervisor.items() if k != "role"}

    def staged(self):
        return ContinuationQueue.validate_staged_source(self)

    def rec_evidence(self, state):
        plan = read_json(self.old_contract)
        endpoint = plan["domain_completion"]["schedule"]["end_actor_updates"]["recovery"]
        if type(endpoint) is not int or endpoint <= 0:
            raise ValueError("Invalid frozen REC actor endpoint")
        record = state.get("domains", {}).get("recovery", {})
        if record.get("status") != "domain_schedule_finished_unassessed":
            return None
        schedule = record.get("schedule_completion", {})
        if (schedule.get("reason") != "iteration_endpoint" or schedule.get("end_actor_update") != endpoint
                or schedule.get("actual_actor_update") != endpoint or record.get("domain_qualified") is not None):
            raise ValueError("REC was not completed at its real frozen iteration endpoint")
        path = Path(record["candidate_checkpoint"])
        if path.name != "model_final.pt" or not path.resolve().is_relative_to(self.train / "recovery"):
            raise ValueError("REC candidate is not its own finalized state")
        completion = read_json(path.parent / "completion.json")
        row = next(r for r in plan["domains"] if r["name"] == "recovery")
        seal = self.adapter.seal(path, "recovery", row["sha256"])
        if (completion.get("status") != "completed" or seal["task_actor_update"] != endpoint
                or record.get("candidate_checkpoint_sha256") != seal["sha256"]):
            raise ValueError("REC final/full seal does not prove normal endpoint completion")
        return dict(seal, endpoint=endpoint)

    def speed_evidence(self, baseline, owners):
        directories = [Path(flag(p["command"], "--run-dir")) for p in owners if p["role"] == "worker"
                       and Path(flag(p["command"], "--run-dir")).is_relative_to(self.train / "speed")]
        directories.extend(p for p in (self.train / "speed").glob("attempt_*/train") if str(p) not in baseline)
        row = next(r for r in read_json(self.old_contract)["domains"] if r["name"] == "speed")
        candidates = []
        for directory in set(directories):
            invocation = read_json(directory.parent / "invocation.json")
            if (Path(flag(invocation["command"], "--run-dir")) != directory
                    or actual_entry(invocation["command"]) != self.old_source_root / "scripts/train_chassis.py"
                    or flag(invocation["command"], "--contract") != str(self.old_source_root / row["contract"])):
                raise ValueError("Speed attempt invocation does not bind its own directory")
            parent = Path(invocation["source_checkpoint"])
            if (flag(invocation["command"], "--resume") != str(parent)
                    or invocation.get("source_checkpoint_sha256") != sha(parent)):
                raise ValueError("Speed attempt parent/argv/hash differs")
            parent_seal = read_json(parent.with_suffix(".checkpoint.json"))
            if parent_seal.get("checkpoint_sha256") != sha(parent) or parent_seal.get("sealed") is not True:
                raise ValueError("Speed parent lacks its authentic full seal")
            for p in directory.glob("checkpoints/update_*/model.checkpoint.json"):
                if any(part.endswith(".pending") for part in p.parts):
                    continue
                progress = read_json(p.parent / "progress.json")
                if progress.get("successful_updates", 0) <= 0:
                    continue
                count = read_json(p)["learning_updates"]
                if count <= parent_seal["learning_updates"]:
                    continue
                candidates.append((count, p.with_suffix("").with_suffix(".pt")))
        if not candidates:
            return None
        return self.adapter.seal(max(candidates)[1], "speed", row["sha256"])

    def archive_ready(self, job):
        archiver = read_json(self.run_root / "archiver_status.json")
        if archiver.get("status") != "finished" or archiver.get("pid") != job["pid"]:
            return False
        delivery = read_json(self.run_root / "delivery.json")
        path = self.run_root / "delivery.tar.gz"
        if (delivery.get("status") != "ready" or delivery.get("archive") != path.name
                or delivery.get("archive_bytes") != path.stat().st_size
                or delivery.get("archive_sha256") != sha(path)):
            raise ValueError("Old wrapper delivery archive does not authenticate")
        for name in ("job.json", "train/consumption.json", "train/completion.json", "train/campaign_state.json"):
            item = delivery.get("files", {}).get(name, {})
            path = self.run_root / name
            if item.get("sha256") != sha(path) or item.get("size") != path.stat().st_size:
                raise ValueError("Old delivery final file differs: " + name)
        return True

    def migration_receipt(self):
        return dict(schema=f"task-plan-revision-v{self.target_revision}", previous_job_sha256=self.config["old_job_sha256"],
            command=self.command, source_commit=self.config["source_commit"],
            old_source_root=str(self.old_source_root), new_source_root=str(self.new_source_root),
            new_contract_relative_path=self.config["new_contract_relative_path"],
            old_contract_sha256=self.config["old_contract_sha256"], new_contract_sha256=self.config["new_contract_sha256"],
            source_archive=str(self.source_archive), source_archive_sha256=self.config["source_archive_sha256"])

    def step(self, *, check_only=False):
        try:
            if self.queue_receipt.exists():
                receipt = read_json(self.queue_receipt)
                if receipt.get("manifest_sha256") != self.config_sha256:
                    raise ValueError("Existing queue receipt has another manifest")
                return self.publish(receipt["status"], receipt["reason"], pid=receipt.get("pid"))
            if self.intent.exists():
                return self.publish("attention", "ambiguous_launch_intent_never_resubmit")
            self.authenticate_observer_exclusivity()
            job = self.previous_job()
            campaign = read_json(self.train / "campaign_state.json")
            if campaign.get("contract_sha256") != self.config["old_contract_sha256"] or campaign.get("status") == "failed":
                raise ValueError("Predecessor campaign identity/status differs")
            state = read_json(self.state_path) if self.state_path.exists() else {}
            if state and state.get("manifest_sha256") != self.config_sha256:
                raise ValueError("Existing state has another manifest")
            if not state.get("staged") or check_only:
                self.staged()
                self.adapter.preflight(job)
                if not check_only:
                    state["staged"] = True
            owners = self.owners()
            if not self.signal_intent.exists():
                supervisor = self.authenticate_owners(job, owners)
                if "speed_attempts_before_rec" not in state:
                    state["speed_attempts_before_rec"] = [str(p) for p in (self.train / "speed").glob("attempt_*/train")]
                rec = self.rec_evidence(campaign)
                if rec is None:
                    if campaign.get("status") == "stopped":
                        raise ValueError("REC stopped before normal endpoint; never continue automatically")
                    if not check_only:
                        self.save_state(state)
                    return self.publish("waiting", "rec_normal_endpoint_not_complete")
                speed = self.speed_evidence(state["speed_attempts_before_rec"], owners)
                if speed is None:
                    if not check_only:
                        self.save_state(state)
                    return self.publish("waiting", "post_rec_speed_complete_snapshot_not_yet_sealed", rec=rec)
                if check_only:
                    return self.publish("waiting", "check_only_boundary_admitted_no_signal_or_launch", rec=rec, speed=speed)

                self.staged()  # Full staged-byte authentication immediately before action.
                self.adapter.preflight(job)

                def before_signal():
                    # Last checks occur after pidfd opening and immediately before
                    # durable once intent; no cached endpoint or speed hash.
                    self.authenticate_observer_exclusivity()
                    fresh_job = self.previous_job()
                    fresh_owners = self.owners()
                    if self.authenticate_owners(fresh_job, fresh_owners) != supervisor:
                        raise ValueError("Owner changed before signal")
                    fresh_rec = self.rec_evidence(read_json(self.train / "campaign_state.json"))
                    fresh_speed = self.speed_evidence(state["speed_attempts_before_rec"], fresh_owners)
                    if fresh_rec != rec or fresh_speed is None:
                        raise ValueError("REC/speed full seal changed before signal")
                    durable_json(self.signal_intent, dict(schema=SCHEMA, manifest_sha256=self.config_sha256,
                        supervisor=supervisor, rec=fresh_rec, speed=fresh_speed), exclusive=True)

                self.adapter.signal_once(supervisor, before_signal)
                state.update(signal_submitted=True, rec=rec, speed=speed)
                self.save_state(state)
                return self.publish("waiting", "one_authenticated_supervisor_sigterm_submitted")
            intent = read_json(self.signal_intent)
            if intent.get("manifest_sha256") != self.config_sha256:
                raise ValueError("Signal intent has another manifest")
            if owners:
                if not state.get("signal_submitted"):
                    return self.publish("attention", "ambiguous_signal_intent_never_resignal", owners=owners)
                # Pinned wrapper may legitimately outlive supervisor while hashing.
                for p in owners:
                    if p["role"] in ("supervisor", "wrapper"):
                        pin = self.config["old_owners"][p["role"]]
                        if any(p[k] != pin[k] for k in ("pid", "starttime")):
                            raise ValueError("Owner replaced during old shutdown")
                return self.publish("waiting", "old_owners_or_wrapper_still_active", owners=owners)
            if campaign.get("status") not in ("stopped", "task_domains_finished_unassessed"):
                raise ValueError("No old owners without native terminal campaign state")
            if not self.archive_ready(job):
                return self.publish("waiting", "old_wrapper_final_archive_not_finished")
            plan = read_json(self.old_contract)
            claim = Path(plan["budget_source_stage"]) / (plan["campaign_id"] + ".account.json")
            if not claim.exists():
                raise ValueError("Durable campaign account is missing; never delete or replace claim")
            with account_ownership(claim.with_suffix(".lock")) as acquired:
                if not acquired:
                    return self.publish("waiting", "native_account_lock_still_owned")
                if check_only:
                    return self.publish("waiting", "check_only_shutdown_admitted_no_launch")
                revision = self.migration_receipt()
                if self.source_revision_receipt.exists():
                    if read_json(self.source_revision_receipt) != revision:
                        raise ValueError("Immutable source revision receipt differs")
                else:
                    durable_json(self.source_revision_receipt, revision, exclusive=True)
                evidence = self.adapter.prepare(job)
                if (evidence["latest_own"]["recovery"]["sha256"] != intent["rec"]["sha256"]
                        or evidence["latest_own"]["speed"]["learning_updates"] < intent["speed"]["learning_updates"]):
                    raise ValueError("Fresh latest own state rolled back REC or speed")
                delivery_files = read_json(self.run_root / "delivery.json")["files"]
                for own in evidence["latest_own"].values():
                    checkpoint = Path(own["path"])
                    for path in (checkpoint, checkpoint.with_suffix(".checkpoint.json")):
                        item = delivery_files.get(str(path.relative_to(self.run_root)), {})
                        if item.get("sha256") != sha(path) or item.get("size") != path.stat().st_size:
                            raise ValueError("Final own learning state is missing from old delivery archive")
                if self.owners() or self.previous_job() != job or not self.archive_ready(job):
                    raise ValueError("Owner/job/archive changed during final native admission")
                # Re-read every native account and latest seal after any expensive
                # archive authentication; never launch from an earlier clock.
                if self.adapter.prepare(job) != evidence:
                    raise ValueError("Latest account/seals changed before launch")
                native = None
                if self.target_revision == 24:
                    native = self.adapter.validate_runtime(evidence)
                    if (self.owners() or self.previous_job() != job or not self.archive_ready(job)
                            or self.adapter.prepare(job) != evidence):
                        raise ValueError("Account/source/owner changed during native startup")
                self.authenticate_observer_exclusivity()
                wrapper = [self.command[0], "-B", str(self.new_source_root / "scripts/chassis_remote_job.py"),
                    "--run-root", str(self.run_root), "--resume-existing", "--source-revision-receipt",
                    str(self.source_revision_receipt), "--", *self.command]
                launch_intent = dict(schema=SCHEMA, manifest_sha256=self.config_sha256,
                    predecessor=evidence, revision=revision, wrapper_command=wrapper, qualification=None)
                if native is not None:
                    launch_intent["native_startup"] = native
                durable_json(self.intent, launch_intent, exclusive=True)
            process = self.adapter.launch(wrapper)
            receipt = dict(launch_intent, status="launch_submitted", reason="rec_boundary_native_continuation_authenticated", pid=process.pid)
            durable_json(self.queue_receipt, receipt, exclusive=True)
            return self.publish(receipt["status"], receipt["reason"], pid=process.pid)
        except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as error:
            return self.publish("attention", "boundary_authentication_or_submission_failed", detail=str(error))

    def run(self, *, once=False, check_only=False, poll_seconds=20):
        with self.lock.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0
            while True:
                result = self.step(check_only=check_only)
                if once or check_only or result["status"] in ("attention", "launch_submitted"):
                    return int(result["status"] == "attention") * 2
                time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Verify staged bytes/plan and observe; never signal or launch")
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("Poll interval must be between 1 and 60 seconds")
    try:
        queue = BoundaryContinuationQueue(read_json(args.manifest))
        return queue.run(once=args.once, check_only=args.check_only, poll_seconds=args.poll_seconds)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(schema=SCHEMA, status="attention", reason="invalid_boundary_manifest",
                              detail=str(error))), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
