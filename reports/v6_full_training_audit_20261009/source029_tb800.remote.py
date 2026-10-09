"""Read existing Kaiser TensorBoard scalars; no event scan or simulator."""
import datetime
import json
from urllib.parse import urlencode
from urllib.request import urlopen

BASE = 'http://127.0.0.1:6006'
RUN = 'v6-task-domains-d073fe4/speed/attempt_003/train'

def get(endpoint, **query):
    url = BASE + endpoint + ('?' + urlencode(query) if query else '')
    with urlopen(url, timeout=30) as response:
        return json.load(response)

tags = get('/data/plugin/scalars/tags')
if RUN not in tags:
    raise RuntimeError('Current speed run is not exposed by TensorBoard')
all_tags = sorted(tags[RUN])
timing = [tag for tag in all_tags if tag.startswith('Timing/')]
groups = ('stand_230mm', 'stand_305mm', 'stand_430mm', 'forward_05',
          'backward_05', 'rotate_1', 'rotate_1_reverse', 'start_stop_1')
metrics = ('height_error_m', 'planar_speed_m_s', 'command_vx_m_s',
           'command_yaw_rad_s', 'rollout_samples',
           'velocity/signed_forward_m_s', 'settled_stop/mean_speed_m_s',
           'settled_stop/samples')
selected = [tag for tag in all_tags if any(
    tag == f'LearningAudit/{group}/{metric}'
    for group in groups for metric in metrics)]
recovery = [tag for tag in all_tags if tag.startswith(
    ('RecoveryOutcomes/', 'RecoveryRollout/'))]
result = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              status='read_only_tensorboard_snapshot', run=RUN,
              remote_writes=0, PPO_updates_by_reader=0,
              scalar_tag_count=len(all_tags), timing={}, selected={},
              recovery={},
              available_group_tags=[tag for tag in all_tags
                  if tag.startswith('LearningAudit/stand_305mm/')],
              actor_window=dict(first=701, last=800, actor_TB_steps=[42103,42202], cumulative_counter_baseline_step=42102),
              interpretation='Eighth nonoverlapping 100 actual actor updates after 10 paid critic updates; sampled training, not deterministic qualification')
for tag in timing + selected + recovery:
    values = get('/data/plugin/scalars/scalars', run=RUN, tag=tag)
    field = 'timing' if tag in timing else 'recovery' if tag in recovery else 'selected'
    result[field][tag] = [row for row in values if 42102 <= row[1] <= 42202]
print(json.dumps(result))
