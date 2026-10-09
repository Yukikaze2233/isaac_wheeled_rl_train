"""Read frozen runtime routing and current scalar snapshots without changing training."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

root = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
directory = root / 'train/speed/attempt_003/train'
names = ('startup.json', 'integrated_admission.json', 'invocation.json')
paths = {name: (directory.parent / name if name == 'invocation.json' else directory / name) for name in names}
payloads = {name: paths[name].read_bytes() for name in names}
pins = {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}
pid = 4141576

def owner():
    path = Path('/proc') / str(pid)
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, ppid=int(stat[1]), starttime=int(stat[19]),
                argv=[v.decode() for v in (path / 'cmdline').read_bytes().split(b'\0') if v])

before = owner()
assert before['starttime'] == 113428247
assert any(str(directory) in value for value in before['argv'])
wanted_fields = {'integrated_runtime_candidate', 'integrated_sampling_candidate',
                 'locomotion_difficulty', 'ordinary_posture_curriculum',
                 'behavior_pool_fractions', 'clocks_at_source', 'component_runtime',
                 'row_ownership', 'environment_allocation'}
extracted = {}

def visit(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            target = path + '/' + key
            if key in wanted_fields:
                extracted[target] = child
            visit(child, target)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            visit(child, path + '/' + str(index))

for name, raw in payloads.items():
    visit(json.loads(raw), name)

run = 'v6-task-domains-d073fe4/speed/attempt_003/train'
tags = ['/curriculum/locomotion/grade', '/curriculum/locomotion/vx_max_m_s',
        '/curriculum/locomotion/yaw_max_rad_s', '/curriculum/locomotion/motion_height_min_m',
        '/curriculum/locomotion/motion_height_max_m',
        '/ordinary_posture/component/speed/relative_actor_updates',
        '/ordinary_posture/component/speed/birth_angle_cap_rad',
        '/ordinary_posture/component/speed/push_speed_cap_m_s']
for group in ('rotate_1rps', 'rotate_3rps', 'stand_305mm'):
    for term in ('stand_still_lin_vel', 'no_fork', 'no_fork_square',
                 'dense_yaw_broad', 'dense_yaw_precision', 'dense_height_broad',
                 'pitch_velocity', 'roll_velocity'):
        tags.append(f'LearningAudit/{group}/reward_density/{term}')
scalars = {}
for tag in tags:
    query = urllib.parse.urlencode(dict(run=run, tag=tag, format='json'))
    url = 'http://127.0.0.1:6006/data/plugin/scalars/scalars?' + query
    with urllib.request.urlopen(url, timeout=20) as response:
        points = json.load(response)
    scalars[tag] = dict(points=len(points), latest=points[-1] if points else None,
                        scope='latest TensorBoard snapshot; not a complete-window estimate')
assert owner() == before
assert pins == {name: hashlib.sha256(paths[name].read_bytes()).hexdigest() for name in names}
print(json.dumps(dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='readonly_frozen_routing_and_scalar_snapshots', metadata_sha256=pins,
    owner_before=before, owner_after=owner(), extracted=extracted, scalars=scalars,
    PPO_updates_by_reader=0, remote_writes=0, native_evaluations=0), allow_nan=False))
