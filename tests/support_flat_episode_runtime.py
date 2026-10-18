"""Shared CPU fixtures for current runtime tests."""
from support_scut_reward_selection import config as reward_config
from wheeled_tasks.chassis.flat_episode_selection import VERSION, EXCLUDED, validate_flat_episode_selection, require_native_flat_admission, flat_episode_source_files

def component_config():
    cfg = reward_config()
    cfg.update(training_profile='v6_78d_scratch_v1', critic_dim=78, critic_observation_version='v6_scut78_v1', actor_observation_source='scut35_encoders_imu_commands', actor_frame_dim=35, manual_context35=True)
    del cfg['skill_specs']
    cfg['flat_episode_runtime'] = {'version': VERSION, 'branch_table': 'table.json', 'branch_table_sha256': 'a' * 64, 'length_definition': 'hip_axis_projected', 'training_knee_upper_deg': 105.0, 'minimum_clearance_m': 0.003, 'birth_rejection': {'mode': 'abort'}}
    return cfg
