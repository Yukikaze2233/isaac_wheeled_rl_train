import os,sys,json,math,hashlib,struct,time
from pathlib import Path
from datetime import datetime,timezone
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
BASELINE_STEP=0
def require(condition, message):
    if not condition:
        raise ValueError(message)

def canonical_tag(tag):
    if tag.startswith("Episode/") and tag[len("Episode/"):].lstrip("/").startswith("recovery/"):
        return "/" + tag[len("Episode/"):].lstrip("/")
    return tag

def aligned_tail_start(path, size, budget):
    """Find a real CRC-valid record within the bounded native suffix.

    An arbitrary byte offset is never treated as a TFRecord header. Only the
    suffix is certified when older records lie outside the requested byte cap.
    """
    if size <= budget:
        return 0
    requested = size - budget
    with path.open("rb") as stream:
        stream.seek(requested)
        chunk = stream.read(min(1024**2, budget))
    for position in range(max(0, len(chunk) - 15)):
        length, header_crc = struct.unpack_from("<QI", chunk, position)
        finish = position + length + 16
        if not 0 < length <= 16 * 1024**2 or finish > len(chunk):
            continue
        if masked_crc32c(chunk[position:position + 8]) != header_crc:
            continue
        payload = chunk[position + 12:finish - 4]
        if masked_crc32c(payload) != struct.unpack_from("<I", chunk, finish - 4)[0]:
            continue
        try:
            Event.FromString(payload)
        except Exception:
            continue
        return requested + position
    raise ValueError("No CRC-valid native TFRecord alignment within bounded suffix")

def parse_events(path, wanted, end_step, min_step=BASELINE_STEP, max_read_bytes=512 * 1024**2):
    """Stream a CRC-aligned native region, checking both CRCs per record.

    This reuses review_chassis_progress.events framing/CRC semantics without
    holding a whole event in RAM. Large files use a bounded suffix; an absent
    earlier baseline prevents certification. An incomplete live tail is never
    accepted. No reservoir, simulator or model is involved.
    """
    series, seen_records, duplicates, unsupported = {}, 0, 0, set()
    hasher = hashlib.sha256()
    offset = last_step = 0
    incomplete = 0
    stopped = "snapshot_eof"
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        start_offset = aligned_tail_start(path, size, max_read_bytes)
        stream.seek(start_offset)
        offset = start_offset
        end_limit = start_offset + max_read_bytes
        while offset < size:
            if offset + 12 > end_limit:
                stopped = "byte_limit"; break
            header = stream.read(min(12, size - offset))
            if len(header) < 12:
                incomplete = len(header); stopped = "incomplete_live_tail"; break
            length, header_crc = struct.unpack("<QI", header)
            require(masked_crc32c(header[:8]) == header_crc, "TFRecord header CRC mismatch at " + str(offset))
            require(0 < length <= 16 * 1024**2, "Unexpected TFRecord payload length")
            if offset + length + 16 > end_limit:
                stopped = "byte_limit"; break
            body = stream.read(min(length + 4, size - offset - 12))
            if len(body) != length + 4:
                incomplete = len(header) + len(body); stopped = "incomplete_live_tail"; break
            payload, data_crc = body[:-4], struct.unpack("<I", body[-4:])[0]
            require(masked_crc32c(payload) == data_crc, "TFRecord payload CRC mismatch at " + str(offset))
            event = Event.FromString(payload)
            hasher.update(header); hasher.update(body)
            offset += length + 16
            seen_records += 1
            if event.summary.value:
                last_step = max(last_step, int(event.step))
                # Train/*/time uses elapsed wall seconds as its step. Only an
                # admitted native update tag can establish the window boundary.
                if event.step > end_step and any(canonical_tag(value.tag) in wanted
                        for value in event.summary.value):
                    stopped = "first_record_after_target"; break
            if not min_step <= event.step <= end_step:
                continue
            for value in event.summary.value:
                tag = canonical_tag(value.tag)
                if tag not in wanted:
                    continue
                if not value.HasField("simple_value"):
                    unsupported.add(tag); continue
                number = float(value.simple_value)
                require(math.isfinite(number), "Nonfinite native scalar " + tag)
                item = [number, float(event.wall_time)]
                old = series.setdefault(tag, {}).get(int(event.step))
                require(old is None or old == item, "Conflicting duplicate native tag/step: " + tag)
                duplicates += int(old is not None)
                series[tag][int(event.step)] = item
    return series, dict(file=str(path.resolve()), file_bytes_at_open=size, read_complete_record_bytes=offset - start_offset,
        source_region_start_offset=start_offset, source_region_end_offset=offset,
        earlier_records_crc_verified=start_offset == 0,
        read_limit_bytes=max_read_bytes, crc_valid_records=seen_records, last_summary_step_observed=last_step,
        incomplete_live_tail_bytes=incomplete, stop_reason=stopped, duplicate_identical_records=duplicates,
        unsupported_selected_scalar_encodings=sorted(unsupported), complete_record_region_sha256=hasher.hexdigest(),
        coverage="bounded_CRC_valid_native_record_region_complete_selected_windows_not_reservoir")
WANTED=['Loss/surrogate', 'LearningAudit/stand_230mm/planar_speed_m_s', 'LearningAudit/stand_230mm/height_error_m', 'LearningAudit/stand_230mm/command_vx_m_s', 'LearningAudit/stand_230mm/command_yaw_rad_s', 'LearningAudit/stand_230mm/stationary_10s/endpoint_mean_m', 'LearningAudit/stand_230mm/stationary_10s/samples', 'LearningAudit/stand_230mm/stationary_10s/peak_max_m', 'LearningAudit/stand_305mm/planar_speed_m_s', 'LearningAudit/stand_305mm/height_error_m', 'LearningAudit/stand_305mm/command_vx_m_s', 'LearningAudit/stand_305mm/command_yaw_rad_s', 'LearningAudit/stand_305mm/stationary_10s/endpoint_mean_m', 'LearningAudit/stand_305mm/stationary_10s/samples', 'LearningAudit/stand_305mm/stationary_10s/peak_max_m', 'LearningAudit/stand_430mm/planar_speed_m_s', 'LearningAudit/stand_430mm/height_error_m', 'LearningAudit/stand_430mm/command_vx_m_s', 'LearningAudit/stand_430mm/command_yaw_rad_s', 'LearningAudit/stand_430mm/stationary_10s/endpoint_mean_m', 'LearningAudit/stand_430mm/stationary_10s/samples', 'LearningAudit/stand_430mm/stationary_10s/peak_max_m', 'LearningAudit/start_stop_1/settled_stop/mean_speed_m_s', 'LearningAudit/start_stop_1/settled_stop/samples', 'LearningAudit/start_stop_1/settled_stop/above_3cm_s_fraction', 'LearningAudit/release_stop_translate/settled_stop/mean_speed_m_s', 'LearningAudit/release_stop_translate/settled_stop/samples', 'LearningAudit/release_stop_translate/settled_stop/above_3cm_s_fraction', 'LearningAudit/release_stop_yaw/settled_stop/mean_speed_m_s', 'LearningAudit/release_stop_yaw/settled_stop/samples', 'LearningAudit/release_stop_yaw/settled_stop/above_3cm_s_fraction', 'Behavior/forward_3/vx_command_mean_m_s', 'Behavior/forward_3/vx_actual_mean_m_s', 'Behavior/forward_3/vx_mae_m_s', 'Behavior/forward_3/yaw_mae_rad_s', 'Behavior/forward_3/height_mae_m', 'Behavior/forward_3/tilt_max_deg', 'Behavior/forward_3/failures', 'Behavior/forward_3/episodes', 'Behavior/forward_3/boundary_truncations', 'Behavior/forward_3/planar_speed_mean_m_s', 'LearningAudit/forward_3/command_yaw_rad_s', 'LearningAudit/forward_3/command_vx_m_s', 'Behavior/rotate_3rps/vx_command_mean_m_s', 'Behavior/rotate_3rps/vx_actual_mean_m_s', 'Behavior/rotate_3rps/vx_mae_m_s', 'Behavior/rotate_3rps/yaw_mae_rad_s', 'Behavior/rotate_3rps/height_mae_m', 'Behavior/rotate_3rps/tilt_max_deg', 'Behavior/rotate_3rps/failures', 'Behavior/rotate_3rps/episodes', 'Behavior/rotate_3rps/boundary_truncations', 'Behavior/rotate_3rps/planar_speed_mean_m_s', 'LearningAudit/rotate_3rps/command_yaw_rad_s', 'LearningAudit/rotate_3rps/command_vx_m_s', 'RecoveryOutcomes/prefix_stand/ended', 'RecoveryOutcomes/prefix_stand/successes', 'RecoveryOutcomes/prefix_stand/failures', 'RecoveryOutcomes/prefix_stand/collection_timeouts', 'RecoveryOutcomes/prefix_release/ended', 'RecoveryOutcomes/prefix_release/successes', 'RecoveryOutcomes/prefix_release/failures', 'RecoveryOutcomes/prefix_release/collection_timeouts', 'RecoveryOutcomes/original_chain/ended', 'RecoveryOutcomes/original_chain/successes', 'RecoveryOutcomes/original_chain/failures', 'RecoveryOutcomes/original_chain/collection_timeouts', 'LearningAudit/release_stop_translate/release_posture/accelerating/pitch_abs_rad', 'LearningAudit/release_stop_translate/release_posture/accelerating/samples', 'LearningAudit/release_stop_translate/release_posture/accelerating/roll_abs_rad', 'LearningAudit/release_stop_translate/release_posture/cruising/pitch_abs_rad', 'LearningAudit/release_stop_translate/release_posture/cruising/samples', 'LearningAudit/release_stop_translate/release_posture/cruising/roll_abs_rad', 'LearningAudit/release_stop_translate/release_posture/decelerating/pitch_abs_rad', 'LearningAudit/release_stop_translate/release_posture/decelerating/samples', 'LearningAudit/release_stop_translate/release_posture/decelerating/roll_abs_rad', 'LearningAudit/release_stop_translate/release_posture/stationary_window/pitch_abs_rad', 'LearningAudit/release_stop_translate/release_posture/stationary_window/samples', 'LearningAudit/release_stop_translate/release_posture/stationary_window/roll_abs_rad', '/recovery/learning_rate', 'Timing/iteration_wall_seconds', 'Timing/env_step_seconds', 'Timing/optimizer_seconds', 'Policy/mean_std', 'RecoveryRollout/valid_fraction', 'Behavior/forward_05/vx_command_mean_m_s', 'Behavior/forward_05/vx_actual_mean_m_s', 'Behavior/forward_05/vx_mae_m_s', 'Behavior/forward_05/yaw_mae_rad_s', 'Behavior/forward_05/height_mae_m', 'Behavior/forward_05/planar_speed_mean_m_s', 'Behavior/backward_05/vx_command_mean_m_s', 'Behavior/backward_05/vx_actual_mean_m_s', 'Behavior/backward_05/vx_mae_m_s', 'Behavior/backward_05/yaw_mae_rad_s', 'Behavior/backward_05/height_mae_m', 'Behavior/backward_05/planar_speed_mean_m_s', 'Behavior/rotate_1/vx_command_mean_m_s', 'Behavior/rotate_1/vx_actual_mean_m_s', 'Behavior/rotate_1/vx_mae_m_s', 'Behavior/rotate_1/yaw_mae_rad_s', 'Behavior/rotate_1/height_mae_m', 'Behavior/rotate_1/planar_speed_mean_m_s', 'Behavior/rotate_1_reverse/vx_command_mean_m_s', 'Behavior/rotate_1_reverse/vx_actual_mean_m_s', 'Behavior/rotate_1_reverse/vx_mae_m_s', 'Behavior/rotate_1_reverse/yaw_mae_rad_s', 'Behavior/rotate_1_reverse/height_mae_m', 'Behavior/rotate_1_reverse/planar_speed_mean_m_s']

start=time.monotonic()
root=Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
run=root/'train/speed/attempt_004/train'; src=root/'source_serial_dispatch029_20261009'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
progress=read(run/'progress.json'); invocation=read(run.parent/'invocation.json'); startup=read(run/'startup.json'); resume=read(run/'resume_verification.json')
require(progress['pid']==40922 and progress['parent_updates']==44367 and progress['num_envs']==4096,'Wrong worker/origin')
require(progress['successful_updates']==progress['actor_updates_in_block'] and progress['task_actor_update']==2597+progress['successful_updates'],'Wrong learning clocks')
require(startup['contract_sha256']=='941fb9e1f7a8ced3aa83fca76e82b7cf3064b594602d1e2900e56e0df0cc6b64','Wrong speed contract')
require(str(src/'scripts/train_chassis.py') in invocation['command'],'Wrong frozen source')
require(all(resume[k] is True for k in ['actor_exact','critic_exact','optimizer_exact']),'Not full learning-state restore')
owner=(Path('/proc/40922')/'stat').read_text().rsplit(')',1)[1].split(); owner_ticks=owner[19]
metadata={str(p):sha(p) for p in [run/'startup.json',run.parent/'invocation.json',run/'resume_verification.json',src/'contracts/v6_task_campaign_speed_worker_v26.json']}
endpoint=progress['successful_updates']//100*100; begin=endpoint-99; lo=44367+begin-1; hi=44367+endpoint-1
require(endpoint>=100,'Window unavailable')
history={}
with (run/'behavior_history.jsonl').open() as f:
 for line in f:
  row=json.loads(line)
  if row['successful_updates'] in [begin-1,endpoint]:history[str(row['successful_updates'])]={'sha256':hashlib.sha256(line.encode()).hexdigest(),'groups':row['groups']}
require(len(history)==2,'History endpoint unavailable')
events=list(run.glob('events.out.tfevents.*.40922.*'));require(len(events)==1,'Ambiguous event')
series,audit=parse_events(events[0],set(WANTED),hi,min_step=lo-1,max_read_bytes=512*1024**2)
for tag in ['Loss/surrogate','LearningAudit/stand_230mm/planar_speed_m_s','RecoveryOutcomes/original_chain/ended']:
 require(all(s in series.get(tag,{}) for s in range(lo,hi+1)),'Incomplete100: '+tag)
require(all(sha(Path(p))==digest for p,digest in metadata.items()),'Metadata changed')
require((Path('/proc/40922')/'stat').read_text().rsplit(')',1)[1].split()[19]==owner_ticks,'Worker replaced')
result={'schema':'v6-wsl-resume-native100-window-v1','read_at_utc':datetime.now(timezone.utc).isoformat(),'updates':[begin,endpoint],'tb_steps':[lo,hi],'learning_clock_range':[2597+begin,2597+endpoint],'qualification':None,'scope':'Randomized sampled training; not deterministic deployment evaluation','event_region':audit,'raw_series':series,'history_endpoints':history,'progress_before':progress,'progress_after':read(run/'progress.json'),'immutable_metadata_sha256':metadata,'worker_start_ticks':owner_ticks,'reader_wall_seconds':time.monotonic()-start}
print(json.dumps(result,ensure_ascii=False,allow_nan=False))
