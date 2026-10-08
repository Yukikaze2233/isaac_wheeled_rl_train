import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

directory = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_010/train')
files = {}
for name in ('model_final.pt', 'model_final.checkpoint.json', 'model_final.contract.json', 'agent_config.json', 'completion.json'):
    path = directory / name
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    files[name] = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(), size=len(content))
assert files['model_final.pt']['sha256'] == 'e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967'
seal = json.loads((directory / 'model_final.checkpoint.json').read_text())
assert (seal['learning_updates'], seal['task_actor_update']) == (41393, 14188)
print(json.dumps(dict(status='actual_source029_parent_pinned', at_utc=datetime.now(timezone.utc).isoformat(),
    files=files, consumed_updates=41393, recovery_actor=14188, remote_writes=0, PPO_updates=0, qualification=None)))
