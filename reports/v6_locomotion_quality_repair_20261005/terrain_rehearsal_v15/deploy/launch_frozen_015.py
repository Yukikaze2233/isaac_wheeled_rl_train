"""Start exactly one native campaign owner after authenticated v15 migration."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

OUT = Path(__file__).resolve().parent
BASE = "/home/kaiser/robot-rl-sim60"
RUN = BASE + "/experiments/v6-task-campaign-20261001T110045Z-282358"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-p", "2222", "kaiser@192.168.64.234"]


def main():
    if (OUT / "terrain_rehearsal_launch_015.json").exists():
        raise FileExistsError("Launch receipt already exists; inspect actual state instead of repeating launch")
    revision = json.loads((OUT / "source_revision_015.json").read_text())
    native = json.loads((OUT.parent / "verification/native_population_015.json").read_text())
    tests = json.loads((OUT.parent / "verification/clean_archive_tests.json").read_text())
    assert native["status"] == tests["status"] == "passed"
    code = "RUN=" + repr(RUN) + "\nREVISION=" + repr(revision) + "\n" + r'''
import hashlib,json,pathlib,sys
r=pathlib.Path(RUN);src=pathlib.Path(REVISION['new_source_root'])
job=json.loads((r/'job.json').read_text())
assert not pathlib.Path('/proc/'+str(job['pid'])+'/cmdline').exists(),'Former owner still live'
assert hashlib.sha256((r/'job.json').read_bytes()).hexdigest()==REVISION['previous_job_sha256']
owners=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().split(b'\0');entry=next((a for a in args[1:] if a.endswith(b'.py')),None)
  if str(r/'train').encode() in b' '.join(args) and entry and any(entry.endswith(b'/'+n) for n in (b'run_task_campaign.py',b'train_chassis.py',b'chassis_remote_job.py')):owners.append(str(p))
 except (OSError,PermissionError):pass
assert not owners,owners
sys.path.insert(0,str(src/'scripts'))
from chassis_remote_job import validate_source_revision
validate_source_revision(r,REVISION['command'],job,r/'source_revision_015.json')
print(json.dumps(dict(status='single_launch_admitted_no_previous_owner',source_commit=REVISION['source_commit'])))
'''
    inner = "source " + shlex.quote(BASE + "/bin/sim60-runtime.sh") + " && export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 && " + shlex.join([BASE + "/env/bin/python", "-B", "-"])
    result = subprocess.run(SSH + [shlex.join(["bash", "-lc", inner])], input=code, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr[-3000:]
    admission = json.loads(result.stdout)
    source = revision["new_source_root"]
    wrapper = [BASE + "/env/bin/python", "-B", source + "/scripts/chassis_remote_job.py", "--run-root", RUN,
               "--resume-existing", "--source-revision-receipt", RUN + "/source_revision_015.json", "--", *revision["command"]]
    inner = "source " + shlex.quote(BASE + "/bin/sim60-runtime.sh") + " && export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 && " + shlex.join(wrapper) + " > " + shlex.quote(RUN + "/job.resume_015.log") + " 2>&1"
    session = "v6-task-campaign-20261001T110045Z-282358-r15"
    launch = shlex.join(["tmux", "-S", BASE + "/tmux.sock", "new-session", "-d", "-s", session, "-c", source, "bash", "-lc", inner])
    subprocess.run(SSH + [launch], check=True, timeout=20)
    receipt = dict(status="launched_waiting_for_native_full_state_restore_and_real_actor_PPO", admission=admission,
        launched_at_utc=datetime.now(timezone.utc).isoformat(), tmux=session, source_commit=revision["source_commit"],
        source_root=source, plan_sha256=revision["new_contract_sha256"], source_archive_sha256=revision["source_archive_sha256"],
        run_root=RUN, command=revision["command"], wrapper_command=wrapper, source_revision_receipt=RUN + "/source_revision_015.json",
        qualification=None, scope="sampling_only_existing_domain_clock_and_budget_not_skill_pass")
    with (OUT / "terrain_rehearsal_launch_015.json").open("x") as stream:
        json.dump(receipt, stream, indent=2); stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
