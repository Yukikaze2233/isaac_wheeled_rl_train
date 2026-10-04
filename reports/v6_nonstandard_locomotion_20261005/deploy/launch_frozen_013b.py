"""One owned, bootstrapped launch after the completed native contact probe."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

BASE = "/home/kaiser/robot-rl-sim60"
ROOT = BASE + "/experiments/v6-task-campaign-20261001T110045Z-282358"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-p", "2222", "kaiser@192.168.64.234"]


def remote(code):
    return json.loads(subprocess.check_output(SSH + ["python3 -c " + shlex.quote(code)], text=True))


state = remote(f"""
import json,pathlib
r=pathlib.Path({ROOT!r})
p=json.loads((r/'nonstandard_speed_probe_013b/report.json').read_text())
j=json.loads((r/'job.json').read_text())
s=json.loads((r/'source_revision_013b.json').read_text())
assert p['status']=='passed', p.get('error',p['status'])
assert p['optimizer_steps']==p['formal_updates_consumed']==0
assert p['counts']['release_requested_lane_frames']>0 and p['counts']['released_learning_lane_frames']>0
other=json.loads((r/'nonstandard_terrain_probe_013b/report.json').read_text())
assert other['status']=='passed' and other['optimizer_steps']==other['formal_updates_consumed']==0
assert other['counts']['extended_terrain_lanes']>0 and other['counts']['release_requested_lane_frames']>0
assert not pathlib.Path(f"/proc/{{j['pid']}}/cmdline").exists(), 'previous owner alive'
owned=[]
for f in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  c=f.read_bytes()
  if str(r/'train').encode() in c and any(x in c for x in (b'/run_task_campaign.py',b'/train_chassis.py')):
   owned.append(str(f))
 except (OSError,PermissionError): pass
assert not owned, owned
print(json.dumps({{'revision':s,'probe_status':p['status'],'counts':p['counts']}}))
""")
revision = state["revision"]
source = revision["new_source_root"]
command = revision["command"]
wrapper = [BASE + "/env/bin/python", "-B", source + "/scripts/chassis_remote_job.py",
           "--run-root", ROOT, "--resume-existing", "--source-revision-receipt",
           ROOT + "/source_revision_013b.json", "--", *command]
inner = ("source " + shlex.quote(BASE + "/bin/sim60-runtime.sh")
         + " && export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1"
         + " && " + shlex.join(wrapper) + " > " + shlex.quote(ROOT + "/job.resume_013.log") + " 2>&1")
session = "v6-task-campaign-20261001T110045Z-282358-r13"
launch = shlex.join(["tmux", "-S", BASE + "/tmux.sock", "new-session", "-d", "-s", session,
                     "-c", source, "bash", "-lc", inner])
subprocess.run(SSH + [launch], check=True)
receipt = dict(launched_at_utc=datetime.now(timezone.utc).isoformat(), tmux=session,
               source_commit=revision["source_commit"], new_source=source,
               run_root=ROOT, plan_sha256=revision["new_contract_sha256"],
               source_archive_sha256=revision["source_archive_sha256"],
               command=command, wrapper_command=wrapper,
               runtime_bootstrap=BASE + "/bin/sim60-runtime.sh",
               source_revision_receipt=ROOT + "/source_revision_013b.json",
               native_probe=ROOT + "/nonstandard_speed_probe_013b/report.json",
               native_probe_counts=state["counts"],
               previous_seal=ROOT + "/nonstandard_seal_013.json",
               status="launched_waiting_for_full_state_and_actual_actor_PPO_verification")
out=Path(__file__).parent/'nonstandard_launch_013b.json'
with out.open('x') as f:
    json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
