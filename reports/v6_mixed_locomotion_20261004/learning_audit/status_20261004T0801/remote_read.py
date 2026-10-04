"""Read-only frozen-worker audit; parse native TB records and cumulative sums."""
import datetime
import hashlib
import json
import math
import pathlib
import re
import statistics
import struct
import subprocess
import time

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c

ROOT = pathlib.Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
DIRECTORY = ROOT / 'train/speed/attempt_001/train'
PARENT_CONSUMED, INITIAL_ACTOR, CRITIC = 18028, 1684, 10
begin = time.monotonic()
history_bytes = (DIRECTORY / 'behavior_history.jsonl').read_bytes()
history = {x['successful_updates']: x for x in map(json.loads, history_bytes.splitlines())}
latest = max(history)
average_names = ('vx_mae_m_s', 'yaw_mae_rad_s', 'height_mae_m',
                 'planar_speed_mean_m_s', 'vx_actual_mean_m_s', 'vx_command_mean_m_s',
                 'vx_bias_m_s', 'height_actual_mean_m', 'height_command_mean_m')
count_names = ('frames', 'observed_frames', 'episodes', 'failures', 'timeouts', 'boundary_truncations')
windows = []
for a, b in [(10, 110), (110, 210), (latest - 300, latest - 200),
             (latest - 200, latest - 100), (latest - 100, latest)]:
    if a not in history or b not in history:
        continue
    groups = {}
    for name, new in history[b]['groups'].items():
        old = history[a]['groups'][name]
        n = new['frames'] - old['frames']
        group = {key: new[key] - old[key] for key in count_names}
        assert n > 0
        group.update({key: (new[key] * new['frames'] - old[key] * old['frames']) / n
                      for key in average_names})
        group['termination_reasons'] = {key: value - old['termination_reasons'].get(key, 0)
                                        for key, value in new['termination_reasons'].items()}
        groups[name] = group
    all_counts = {key: sum(g[key] for g in groups.values()) for key in count_names}
    assert all_counts['observed_frames'] == (b - a) * 4096 * 24
    all_counts['failures_per_ended_episode'] = all_counts['failures'] / all_counts['episodes']
    windows.append({'worker_updates': [a, b],
                    'actor_updates': [INITIAL_ACTOR + a - CRITIC, INITIAL_ACTOR + b - CRITIC],
                    'tb_steps_inclusive': [PARENT_CONSUMED + a, PARENT_CONSUMED + b - 1],
                    'groups': groups, 'all_groups': all_counts})

groups = ['stand_230mm', 'stand_250mm', 'stand_305mm', 'stand_430mm',
          'start_stop_05', 'start_stop_1', 'start_stop_3', 'velocity_curve_1']
metrics = ['stationary_10s/samples', 'stationary_10s/peak_mean_m',
           'stationary_10s/mean_speed_m_s', 'settled_stop/samples',
           'settled_stop/mean_speed_m_s', 'settled_stop/above_3cm_s_fraction']
tags = {f'LearningAudit/{group}/{metric}' for group in groups for metric in metrics}
tags.update(['Timing/iteration_wall_seconds', 'Timing/env_step_seconds',
             'Timing/optimizer_seconds', 'Timing/diagnostics_seconds', 'Timing/logger_seconds',
             'Loss/value', 'Loss/surrogate', 'Loss/learning_rate', 'Policy/mean_std',
             'Train/mean_reward', '/delay/strength', '/delay/enabled_fraction'])
pattern = re.compile(b'(?:' + b'|'.join(re.escape(x.encode()) for x in sorted(tags, key=len, reverse=True))
                     + b')(?![A-Za-z0-9_/])')
event = max(DIRECTORY.glob('events.out.tfevents.*'), key=lambda p: p.stat().st_mtime)
size = event.stat().st_size
series, chunks, unlocated = {}, [], 0
for offset, length in [(0, min(size, 192 * 1024**2)),
                       (max(0, size - 224 * 1024**2), min(size, 224 * 1024**2))]:
    with event.open('rb') as stream:
        stream.seek(offset)
        data = stream.read(length)
    seen = set()
    for match in pattern.finditer(data):
        found = False
        for pos in range(max(0, match.start() - 96), match.start() - 11):
            length, crc = struct.unpack_from('<QI', data, pos)
            end = pos + 12 + length + 4
            if not 0 < length < 16384 or end > len(data) or end - 4 < match.end():
                continue
            if masked_crc32c(data[pos:pos + 8]) != crc:
                continue
            payload = data[pos + 12:end - 4]
            if masked_crc32c(payload) != struct.unpack_from('<I', data, end - 4)[0]:
                continue
            found = True
            if pos in seen:
                break
            seen.add(pos)
            record = Event.FromString(payload)
            for value in record.summary.value:
                if value.tag in tags and value.HasField('simple_value'):
                    scalar = float(value.simple_value)
                    if not math.isfinite(scalar):
                        raise ValueError(f'Nonfinite {value.tag} at {record.step}')
                    series.setdefault(value.tag, {})[int(record.step)] = [scalar, record.wall_time]
            break
        unlocated += not found
    chunks.append({'offset': offset, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                   'crc_valid_selected_records': len(seen)})
    del data

def full_series(tag, a, b):
    source = series.get(tag, {})
    return [source[step][0] for step in range(a, b + 1)] if all(step in source for step in range(a, b + 1)) else None

for window in windows:
    a, b = window['tb_steps_inclusive']
    window['tensorboard'] = {}
    for tag in tags:
        if tag.startswith('LearningAudit/'):
            continue
        values = full_series(tag, a, b)
        window['tensorboard'][tag] = {'samples': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'min': min(values), 'max': max(values)} if values else {'incomplete': True}
    window['parking'] = {}
    window['stationary_10s'] = {}
    for group in groups:
        for phase, destination, fields in [
            ('settled_stop', 'parking', ['mean_speed_m_s', 'above_3cm_s_fraction']),
            ('stationary_10s', 'stationary_10s', ['peak_mean_m', 'mean_speed_m_s'])]:
            counts = full_series(f'LearningAudit/{group}/{phase}/samples', a, b)
            if counts is None:
                window[destination][group] = {'incomplete': True}
                continue
            n = sum(counts)
            result = {'complete_updates': len(counts), 'samples': n}
            for field in fields:
                values = full_series(f'LearningAudit/{group}/{phase}/{field}', a, b)
                result[field] = sum(x * c for x, c in zip(values, counts)) / n if n and values else None
            window[destination][group] = result

seals = []
for checkpoint in sorted((DIRECTORY / 'checkpoints').glob('update_*'))[-3:]:
    metadata = json.loads((checkpoint / 'model.checkpoint.json').read_text())
    checkpoint_hash = hashlib.file_digest((checkpoint / 'model.pt').open('rb'), 'sha256').hexdigest()
    seals.append({'directory': str(checkpoint), 'metadata': metadata,
                  'actual_sha256': checkpoint_hash, 'hash_matches': checkpoint_hash == metadata['checkpoint_sha256']})

progress = json.loads((ROOT / 'train/progress.json').read_text())
state = json.loads((ROOT / 'train/campaign_state.json').read_text())
worker_progress = json.loads((DIRECTORY / 'progress.json').read_text())
gpu_exe = '/usr/lib/wsl/lib/nvidia-smi'
gpu = subprocess.check_output([gpu_exe, '--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw',
                               '--format=csv,noheader'], text=True, timeout=8)
source = ROOT / 'source_122d8db63f37'
contracts = {name: hashlib.sha256((source / 'contracts' / name).read_bytes()).hexdigest() for name in
             ['v6_task_campaign_v10.json', 'v6_task_campaign_speed_worker_v10.json']}
result = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'method': 'Exact native cumulative mean*frames differences; TB weighted by actual phase/window samples; cumulative maxima excluded',
          'history_file': str(DIRECTORY / 'behavior_history.jsonl'),
          'history_sha256': hashlib.sha256(history_bytes).hexdigest(), 'history_records': len(history),
          'event_file': str(event), 'event_snapshot_bytes': size, 'event_chunks': chunks,
          'unlocated_tag_matches_at_chunk_edges': unlocated, 'selected_tags_found': len(series),
          'raw_series': series, 'windows': windows, 'progress': progress,
          'worker_progress': worker_progress,
          'state': {key: state.get(key) for key in ['status', 'consumed_updates', 'actor_updates_consumed',
                       'domain_consumed_updates', 'remaining_updates', 'contract_sha256', 'domains']},
          'checkpoint_seals': seals, 'contracts': contracts, 'gpu': gpu,
          'extract_seconds': time.monotonic() - begin}
print(json.dumps(result, allow_nan=False))
