"""V40 flat-world DirectRLEnv configuration; no legacy environment inheritance."""
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass as _configclass_module
if callable(_configclass_module):  # Isaac Lab <= 2.3
    configclass = _configclass_module
else:  # Isaac Lab >= 3.0: configclass 子包化，装饰器在子模块内
    from isaaclab.utils.configclass import configclass


@configclass
class V40EnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 20.0
    is_finite_horizon = False
    action_space = 6
    # DirectRLEnv 2.3.0 wraps observation_space as "policy" and state_space as "critic".
    # A dict here would produce an incorrect nested policy observation space.
    observation_space = 125
    state_space = 29
    sim = SimulationCfg(dt=0.005, render_interval=2)
    scene = InteractiveSceneCfg(
        # Independent USD clones keep per-env filtered-pair targets inspectable.
        # Re-enable replication/Fabric cloning only after target-server validation.
        num_envs=256, env_spacing=4.0, replicate_physics=False, clone_in_fabric=False,
    )
    # Filled from the validated contract BEFORE DirectRLEnv constructs the scene.
    robot_cfg: ArticulationCfg | None = None
    contact_sensor_cfg: ContactSensorCfg | None = None
    contract_path: str | None = None
    usd_cache_dir: str | None = None
    # Pre-converted USD (e.g. Sim 5.1 importer output); skips URDF re-conversion.
    usd_seed: str | None = None
    allow_research: bool = False
    stage: str = "stand"
    # Evaluation snapshots only; no added training observations/rewards or per-step readbacks by default.
    wheel_slip_diagnostics: bool = False
