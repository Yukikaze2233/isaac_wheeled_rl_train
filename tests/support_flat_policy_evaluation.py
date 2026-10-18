"""Shared CPU fixtures for current runtime tests."""
from wheeled_tasks.chassis.flat_policy_evaluation import CENTER_REFERENCE, SCHEMA, FlatEpisodeLedger, command_kind, validate_protocol, verify_checkpoint_identity, runtime_status, require_native_execution
from wheeled_tasks.chassis.scut_flat_reward import WEIGHTS

def protocol(**changes):
    return dict(schema=SCHEMA, mode='source_distribution', actor_mode='mean', episodes_per_lane=1, environment_seed=17, actor_seed=31, policy_dt_s=0.02, horizon_seconds=20.0, center_reference=CENTER_REFERENCE, assessment='measurement_only', **changes)

def fixed(command=(1.0, 0.0, 0.3), **changes):
    p = protocol()
    p.update(mode='fixed_measurement', cases=[{'case_id': 'case', 'base_request': list(command)}], **changes)
    return p

def begin(obj, lane=0, episode_id=0, position=(0.0, 0.0, 0.0), yaw=0.0):
    obj.begin_episode(lane, episode_id, root_link_position_world=list(position), yaw_rad=yaw, reset_metadata={'ground_selected': True, 'source_reset_sample_id': episode_id})

def frame(tick=1, **changes):
    row = dict(episode_id=0, episode_tick=tick, reward=0.02, payments={name: 0.02 if name == 'track_lin_vel_xy' else 0.0 for name in WEIGHTS}, done=False, terminated=False, timeout=False, boundary=False, termination_reasons=[], learning_mask=True, bootstrap_mask=True, measurement_valid=True, command_override_active=False, effective_command=[1.0, 0.0, 0.3], base_request=[1.0, 0.0, 0.3], root_link_position_world=[tick * 0.02, 0.0, 0.0], root_com_velocity_body=[1.0, 0.0, 0.0], angular_velocity_body=[0.0, 0.0, 0.0], yaw_rad=0.0)
    row.update(changes)
    return row

def terminal(tick=1, **changes):
    row = frame(tick, done=True, terminated=True, termination_reasons=['source_attitude_or_contact'])
    row.update(changes)
    return row
