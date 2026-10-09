"""Read the actual source031 CPU preflight log and stage receipt without writes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
stage = root / 'source031_deploy/stage.json'
log = root / 'source031_deploy/preflight.log'
expected = {
    stage: 'b7dae15f3878aa92039580387bc98e526078fc4f617064a35c18da8994f93828',
    log: 'f6bafbd166fab9f444d3574be71f3d382753626ae8db5be49d94904feb2f40b9',
}
for path, wanted in expected.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == wanted, str(path)
receipt = json.loads(stage.read_text())
assert receipt['source_commit'] == 'ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a'
assert len(receipt['files']) == 4343
assert receipt['dynamic_parent_bound'] is False
assert receipt['PPO_updates'] == 0
result = dict(
    status='static_remote_receipts_read_back',
    at_utc=datetime.now(timezone.utc).isoformat(),
    stage_sha256=expected[stage],
    preflight_sha256=hashlib.sha256((root/'source031_deploy/preflight.json').read_bytes()).hexdigest(),
    preflight_log_sha256=expected[log],
    preflight_log=log.read_text(),
    source_commit=receipt['source_commit'],
    file_count=len(receipt['files']),
    stage_only=not receipt['dynamic_parent_bound'],
    future_boundary_intent_exists=(root / 'source031_speed_boundary.json').exists(),
    dynamic_source_revision_exists=(root / 'source_revision_031.json').exists(),
    PPO_updates_by_reader=0,
    remote_writes=0,
    qualification=None,
)
print(json.dumps(result))
