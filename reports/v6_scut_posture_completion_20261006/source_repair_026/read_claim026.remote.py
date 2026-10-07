from pathlib import Path
import json,hashlib,datetime
root=Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
plan=json.loads((root/'source_6f07ec298e01/contracts/v6_task_campaign_v21.json').read_text())
claim=Path(plan['budget_source_stage'])/(plan['campaign_id']+'.account.json')
data=claim.read_bytes();value=json.loads(data)
ledger=json.loads((root/'train/consumption.json').read_text())
print(json.dumps(dict(read_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim_file=str(claim),claim_sha256=hashlib.sha256(data).hexdigest(),claim=value,consumption=ledger,PPO_updates=0,signals=0,qualification=None)))
