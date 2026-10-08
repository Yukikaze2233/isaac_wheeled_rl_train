from pathlib import Path
import sys,json,datetime
root=Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
directory=root/'boundary_observer_027'
sys.path.insert(0,str(directory))
from queue_task_campaign_boundary_continuation import BoundaryContinuationQueue
q=BoundaryContinuationQueue(json.loads((directory/'manifest.json').read_text()))
job=q.previous_job();owners=q.owners();q.authenticate_owners(job,owners)
q.staged();native=q.adapter.preflight(job)
campaign=json.loads((root/'train/campaign_state.json').read_text())
assert q.rec_evidence(campaign) is None
assert not any(p.exists() for p in (q.state_path,q.signal_intent,q.intent,q.queue_receipt,q.source_revision_receipt))
x=dict(status='native_staged_admission_passed_before_retiring_waiting_observer',native_preflight=native,manifest_sha256=q.config_sha256,read_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),PPO_updates=0,signals=0,qualification=None)
print(json.dumps(x))
