from pathlib import Path
import json,sys,datetime,hashlib
data=json.load(sys.stdin);root=Path("/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358")
for n in ("source_integrated_v26_20261008","source_integrated_v26_20261008.tar.gz","integrated_source_staging_028.json","boundary_observer_028","source_revision_028.json","boundary_continuation_028.receipt.json"):
 assert not (root/n).exists(),n
assert hashlib.sha256((root/"job.json").read_bytes()).hexdigest()==data["old_job_sha256"]
for pin in data["old_owners"].values():
 stat=(Path("/proc")/str(pin["pid"])/"stat").read_text().rpartition(")")[2].split();assert stat[0]!="Z" and int(stat[19])==pin["starttime"]
print(json.dumps(dict(status="production_alive_and_named_v26_staging_absent",read_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),owners=data["old_owners"],PPO_signals=0)))
