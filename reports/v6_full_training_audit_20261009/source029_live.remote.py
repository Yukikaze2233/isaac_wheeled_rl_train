"""Compact read-only process/progress snapshot; no event loading or simulator."""
import datetime
import json
from pathlib import Path

root = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
roles = {'chassis_remote_job.py': 'wrapper', 'run_task_campaign.py': 'supervisor',
         'train_chassis.py': 'worker', 'evaluate_chassis.py': 'evaluator'}
result = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), owners=[],
              remote_writes=0, PPO_updates_by_reader=0)
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv = [x.decode() for x in path.read_bytes().split(b'\0') if x]
        entries = [x for x in argv if Path(x).name in roles]
        if not entries or not any(str(root) in x for x in argv):
            continue
        stat = (path.parent / 'stat').read_text().rsplit(')', 1)[1].split()
        result['owners'].append(dict(pid=int(path.parent.name), ppid=int(stat[1]),
            starttime=int(stat[19]), state=stat[0], role=roles[Path(entries[0]).name], argv=argv))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
for owner in result['owners']:
    if owner['role'] != 'worker':
        continue
    argv = owner['argv']
    directory = Path(argv[argv.index('--run-dir') + 1])
    result['worker_directory'] = str(directory)
    for name in ('progress.json', 'completion.json'):
        file = directory / name
        if file.exists():
            data = json.loads(file.read_text())
            result[name] = {k: data.get(k) for k in ('status', 'successful_updates',
                'attempted_updates', 'learning_updates', 'actor_updates_consumed',
                'task_actor_update', 'stage_actor_update', 'learning_lineage_updates',
                'critic_warmup_remaining')}
for relative in ('source029_deploy/launch.log', 'train.resume_024.log'):
    file = root / relative
    if file.exists():
        with file.open('rb') as stream:
            stream.seek(max(0, file.stat().st_size - 5000))
            result[relative] = stream.read().decode(errors='replace')[-3000:]
result['status'] = 'read_only_snapshot'
print(json.dumps(result))
