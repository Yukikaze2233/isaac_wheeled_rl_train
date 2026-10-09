#!/usr/bin/env python3
"""Transport the frozen once coordinator, validate read-only, then enqueue once.

This launcher never signals training and never waits for the speed boundary.
The detached SHA-pinned coordinator owns the original finite native workflow.
Each transport action has separate immutable local output and remote intent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

REPORT = Path(__file__).resolve().parent
HOST = "kaiser@192.168.64.234"
RUN = Path("/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358")
PYTHON = "/home/kaiser/robot-rl-sim60/env/bin/python"
SCRIPT = REPORT / "source031_queue/coordinate_source031.py"
SCRIPT_SHA = "91f059ca0f28c8f7636620ab8d801e95eca0b24e267475811862ce86f7a8f481"
REMOTE_SCRIPT = RUN / "coordinate_source031_91f059ca.py"
PEER = REPORT / "source031_queue_transition_independent_review001.json"
PEER_SHA = "425439226a2f14581a231c9f76a096c7fa027219ddfa8df304a6685b233d5d80"
RUNTIME = "/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh"
RUNTIME_SHA = "c76cc88f7273a400430ded6251ebc32d984a36fed366733a27ed76e2c94dcff7"
STAGE_SHA = "b7dae15f3878aa92039580387bc98e526078fc4f617064a35c18da8994f93828"
SSH = ["ssh", "-p", "2222", "-oBatchMode=yes", "-oConnectTimeout=10", HOST]

# Read-only child environment construction. Do not print environment values.
# Authenticate the sourced file before bash executes it; the frozen coordinator
# independently validates it again. Waiting processes use one CPU thread, while
# the native new-training wrapper later reloads the original OMP8 runtime.
REMOTE_ENVIRONMENT = f'''runtime=Path({RUNTIME!r})
if runtime.is_symlink() or not runtime.is_file() or runtime.stat().st_mode & 0o7777 != 0o644 or runtime.stat().st_size != 776:
    raise ValueError('Original sourced runtime file/mode/size differs')
with runtime.open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()!={RUNTIME_SHA!r}: raise ValueError('Original runtime SHA differs')
inherited=subprocess.run(['/bin/bash','-c','source "$1"; exec env -0','source031-queue-runtime',str(runtime)],check=True,capture_output=True).stdout
env={{key.decode():value.decode() for key,value in (item.split(b'=',1) for item in inherited.split(b'\\0') if item)}}
env.pop('PYTHONPATH',None)
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
'''


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("transfer", "dryrun", "enqueue"))
    args = parser.parse_args()
    if digest(SCRIPT) != SCRIPT_SHA or digest(PEER) != PEER_SHA:
        raise ValueError("Frozen coordinator or independent review has changed")
    prefix = REPORT / ("source031_queue_" + args.mode + "001")
    if any(prefix.with_suffix(s).exists() for s in (".intent.json", ".json", ".stdout", ".stderr", ".execution.json")):
        raise ValueError("Action already has evidence; inspect it instead of retrying")
    freeze = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(Path(__file__).resolve())],
                            cwd=REPORT, check=True, text=True, capture_output=True).stdout.strip()
    frozen_blob = subprocess.run(["git", "show", freeze + ":" + str(Path(__file__).resolve().relative_to(REPORT.parent.parent))],
                                 cwd=REPORT, check=True, capture_output=True).stdout
    if not freeze or frozen_blob != Path(__file__).read_bytes():
        raise ValueError("Launcher must be committed byte-identically before use")
    dryrun = REPORT / "source031_queue_dryrun001.json"
    if args.mode == "enqueue":
        dry = json.loads(dryrun.read_text())
        if dry.get("status") != "validated_once_queue_not_started" or dry.get("qualification") is not None:
            raise ValueError("Actual remote read-only validation is required")
        executed = json.loads((REPORT / "source031_queue_dryrun001.execution.json").read_text())
        if executed.get("status") != "completed" or any(c["exit_code"] for c in executed["calls"]):
            raise ValueError("Read-only validation did not exit successfully")
    intent = dict(mode=args.mode, at_utc=datetime.now(timezone.utc).isoformat(),
                  coordinator_sha256=SCRIPT_SHA, peer_sha256=PEER_SHA, launcher_commit=freeze,
                  launcher_sha256=digest(__file__), qualification=None,
                  additional_training_signals=0, new_training_launches_by_transport=0)
    with prefix.with_suffix(".intent.json").open("x") as stream:
        stream.write(json.dumps(intent, indent=2) + "\n")
    calls = []
    started = time.monotonic()
    status = "failed_retained_no_retry"

    def execute(command, *, code=None):
        result = subprocess.run(command, input=code, text=True, capture_output=True, timeout=180)
        calls.append(dict(command=command, exit_code=result.returncode,
                          stdin_sha256=hashlib.sha256(code.encode()).hexdigest() if code else None))
        with prefix.with_suffix(".stdout").open("a") as stream:
            stream.write(result.stdout)
        with prefix.with_suffix(".stderr").open("a") as stream:
            stream.write(result.stderr)
        if result.returncode:
            raise RuntimeError("Transport returned " + str(result.returncode) + "; inspect evidence, do not retry")
        return result.stdout

    def remote(code):
        return execute([*SSH, shlex.join([PYTHON, "-B", "-"])], code=code)

    try:
        if args.mode == "transfer":
            incoming = RUN / "source031_queue_incoming001"
            remote(f'''from pathlib import Path
import json
root=Path({str(RUN)!r}); incoming=Path({str(incoming)!r})
for p in (incoming,Path({str(REMOTE_SCRIPT)!r}),root/'source031_queue/queue.intent.json',root/'source031_speed_boundary.json',root/'source031_queue_launch.intent.json'):
    if p.exists(): raise ValueError('Existing queue product: '+str(p))
incoming.mkdir(mode=0o700)
print(json.dumps(dict(status='exclusive_queue_incoming_reserved')))
''')
            execute(["scp", "-P", "2222", "-oBatchMode=yes", "-oConnectTimeout=10", str(SCRIPT), f"{HOST}:{incoming}/"])
            output = remote(f'''import hashlib,json,os
from pathlib import Path
incoming=Path({str(incoming)!r}); source=incoming/{SCRIPT.name!r}
with source.open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()!={SCRIPT_SHA!r}: raise ValueError('Uploaded coordinator SHA differs')
os.chmod(source,0o644); os.link(source,{str(REMOTE_SCRIPT)!r}); source.unlink(); incoming.rmdir()
print(json.dumps(dict(status='frozen_once_coordinator_transferred',file={str(REMOTE_SCRIPT)!r},sha256={SCRIPT_SHA!r},qualification=None)))
''')
        elif args.mode == "dryrun":
            output = remote(f'''import hashlib,subprocess,sys
from pathlib import Path
script=Path({str(REMOTE_SCRIPT)!r})
if script.is_symlink() or not script.is_file() or script.stat().st_mode & 0o7777 != 0o644:
    raise ValueError('Remote coordinator file/mode differs')
with script.open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()!={SCRIPT_SHA!r}: raise ValueError('Remote coordinator SHA differs')
{REMOTE_ENVIRONMENT}
result=subprocess.run([{PYTHON!r},'-B',{str(REMOTE_SCRIPT)!r},'--run-root',{str(RUN)!r}],env=env,capture_output=True)
sys.stdout.buffer.write(result.stdout); sys.stderr.buffer.write(result.stderr)
raise SystemExit(result.returncode)
''')
            if json.loads(output).get("status") != "validated_once_queue_not_started":
                raise ValueError("Remote validation did not return the expected read-only status")
        else:
            output = remote(f'''import hashlib,importlib.util,json,os,subprocess
from pathlib import Path
root=Path({str(RUN)!r}); script=Path({str(REMOTE_SCRIPT)!r})
{REMOTE_ENVIRONMENT}
if script.is_symlink() or not script.is_file() or script.stat().st_mode & 0o7777 != 0o644:
    raise ValueError('Remote coordinator file/mode differs')
with script.open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()!={SCRIPT_SHA!r}: raise ValueError('Coordinator changed')
for p in (root/'source031_queue/queue.intent.json',root/'source031_speed_boundary.json',root/'source031_queue_launch.intent.json',root/'source031_queue_launch.receipt.json',root/'source031_queue.launch.log',root/'source_revision_031.json'):
    if p.exists(): raise ValueError('Existing queue/launch product; inspect without retry: '+str(p))
job=json.loads((root/'job.json').read_text())
owners=((4141407,113423130,'chassis_remote_job.py'),(4141412,113423416,'run_task_campaign.py'),(4141576,113428247,'train_chassis.py'))
actual_parents={{}}
for pid,start,name in owners:
    proc=Path('/proc')/str(pid); stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
    argv=[b.decode() for b in (proc/'cmdline').read_bytes().split(b'\\0') if b]
    if int(stat[19])!=start or stat[0] in ('Z','T') or not any(Path(a).name==name for a in argv) or not any(str(root) in a for a in argv):
        raise ValueError('Original training owner identity/state differs')
    if name=='run_task_campaign.py' and (job['pid']!=pid or job['command']!=argv): raise ValueError('Native job differs')
    actual_parents[pid]=int(stat[1])
if actual_parents[4141412]!=4141407 or actual_parents[4141576]!=4141412: raise ValueError('Original owner parent chain differs')
utility=root/'source_serial_dispatch031_20261009/scripts/queue_task_campaign_continuation.py'
stage_file=root/'source031_deploy/stage.json'
if stage_file.is_symlink() or not stage_file.is_file(): raise ValueError('Actual stage file differs')
with stage_file.open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()!={STAGE_SHA!r}: raise ValueError('Actual stage SHA differs')
stage=json.loads(stage_file.read_text())
if utility.is_symlink() or not utility.is_file(): raise ValueError('Native publisher file differs')
with utility.open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()!=stage['files']['scripts/queue_task_campaign_continuation.py']: raise ValueError('Native publisher changed')
spec=importlib.util.spec_from_file_location('source031_transport_native_publisher',utility); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
publish=module.durable_json
source=root/'source_serial_dispatch029_20261009'
expected_owners=sorted([dict(pid=pid,role=module.ROLES[name],entry=str(source/'scripts'/name)) for pid,start,name in owners],key=lambda row:row['pid'])
if module.owned_processes(root)!=expected_owners: raise ValueError('Same-run training owners are not the exact original three')
command=[{PYTHON!r},'-B',str(script),'--run-root',str(root),'--apply']
publish(root/'source031_queue_launch.intent.json',dict(schema='source031-once-queue-launch-v1',coordinator_sha256={SCRIPT_SHA!r},launcher_commit={freeze!r},launcher_sha256={intent['launcher_sha256']!r},actual_remote_dryrun_sha256={digest(dryrun)!r},command=command,original_owners=owners,qualification=None),exclusive=True)
with (root/'source031_queue.launch.log').open('x') as log:
    child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    stat=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
    result=dict(status='once_coordinator_submitted_not_boundary_completed',pid=child.pid,starttime=int(stat[19]),command=command,log=str(root/'source031_queue.launch.log'),train_signals_by_transport=0,new_training_launches_by_transport=0,qualification=None)
    publish(root/'source031_queue_launch.receipt.json',result,exclusive=True)
print(json.dumps(result))
''')
        result = json.loads(output)
        prefix.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        status = "completed"
    finally:
        prefix.with_suffix(".execution.json").write_text(json.dumps(dict(status=status,
            wall_seconds=time.monotonic()-started, calls=calls, additional_training_signals=0,
            new_training_launches_by_transport=0, qualification=None), indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
