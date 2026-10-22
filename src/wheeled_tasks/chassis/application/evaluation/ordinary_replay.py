"""Ordinary fixed-request replay overlay; never a training contract or skill grant.

The actor/asset/control identities stay in the exported bundle. Only the native
physics/PD clock changes to 1 ms, on the existing infinite plane. Source command
bookkeeping still advances in 5 ms units; policy/PC randomization clocks stay at
20 ms. No jump, recovery, world-frame translation or hardware qualification.
"""
from dataclasses import dataclass
import math


ROLE = "v7_ordinary_deployment_replay"


@dataclass(frozen=True)
class OrdinaryReplaySetup:
    command: tuple[float, float, float]  # body vx, yaw rate, root height
    seed: int
    stage_actor_update: int
    global_actor_update: int

    physics_dt = .001
    policy_dt = .02
    decimation = 20
    source_clock_stride = 5
    evaluation_surface = "plane_regression_v1"

    def __post_init__(self):
        if (type(self.command) is not tuple or len(self.command) != 3
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in self.command)
                or self.command[2] <= 0.):
            raise ValueError("Ordinary replay needs finite body vx/yaw and positive root height")
        if any(type(x) is not int or x < 0 for x in
               (self.seed, self.stage_actor_update, self.global_actor_update)):
            raise ValueError("Replay seed and original actor clocks must be nonnegative integers")
        if self.stage_actor_update > self.global_actor_update:
            raise ValueError("Replay stage clock exceeds original global actor clock")

    def validate_constructor(self, config, num_envs, seed, run_role):
        from wheeled_tasks.chassis.application.configuration.training_profile import validate_profile_selection
        from wheeled_tasks.chassis.application.configuration.flat_episode_selection import require_native_flat_admission
        validate_profile_selection(config)
        require_native_flat_admission(config)
        if (run_role != ROLE or type(num_envs) is not int or num_envs != 1
                or seed != self.seed or config.get("formal_v7_flat") is not True
                or config.get("formal_foundation_nominal_birth") is not True
                or config.get("formal_closure_monitor_only") is not True
                or config.get("record_diagnostics") is not True):
            raise ValueError("Ordinary replay requires one formal nominal-birth lane and its explicit role")
        for key in ("recovery_training", "step_jump_task", "jump_assist", "jump_landing",
                    "command_reference", "nonstandard_start", "v7_curriculum_runtime",
                    "recovery_takeover", "evaluation_exact_cases", "integrated_production"):
            if config.get(key):
                raise ValueError("Ordinary replay excludes task owner: " + key)
        # These paths carry their own physical-step delay/randomization rules.
        # This initial replay uses the current source's zero added transport.
        for key in ("signal_delay", "feedback_transport", "command_transport", "usb_transport"):
            if config.get(key, {}).get("enabled"):
                raise ValueError("Ordinary replay does not enable a transport model: " + key)
        if config.get("signal_perturbations", {}).get("max_delay_steps", 0) != 0:
            raise ValueError("Ordinary replay requires the existing zero policy-step delay")

    def advance_source_clock(self, flat, physical_tick):
        if physical_tick % self.source_clock_stride == 0:
            flat.advance_physics(.005)

    def apply_before_first_reset(self, env):
        self.validate_constructor(env.cfg, env.num_envs, env.cfg["seed"], ROLE)
        import torch
        flat = env.flat_episode
        if (flat is None or env.physical_tick != 0 or flat.commands.physics_steps != 0
                or bool((flat.episode_id != -1).any())):
            raise ValueError("Replay requests must be installed before the first native reset")
        vx, yaw, height = self.command
        flat.commands.set_fixed_requests(torch.arange(env.num_envs, device=env.device),
            torch.tensor([[vx, 0., yaw]], device=env.device, dtype=torch.float32),
            torch.tensor([height], device=env.device, dtype=torch.float32), evaluation_only=True)
        env.stage_actor_update = self.stage_actor_update
        env.global_actor_update = self.global_actor_update
        env.startup_report["ordinary_deployment_replay"] = self.receipt()

    def receipt(self):
        return {"scope": "ordinary_fixed_body_frame_command_on_plane_only", "role": ROLE,
            "command_vx_yaw_height": list(self.command), "environment_seed": self.seed,
            "physics_dt": self.physics_dt, "pc_control_dt": self.physics_dt,
            "policy_dt": self.policy_dt, "decimation": self.decimation,
            "source_command_dt": .005, "source_command_stride": self.source_clock_stride,
            "contact_history_window_s": .015, "solver_iterations": [12, 6],
            "training_contract_rewritten": False, "stage_actor_update": self.stage_actor_update,
            "global_actor_update": self.global_actor_update, "ppo_updates": 0,
            "domain_randomization": "unchanged source startup/reset/policy clocks; no new transport",
            "context": "built by build_scut35 from ordinary command input; no jump/recovery requests",
            "previous_action": "actual clipped control action reordered by build_scut35",
            "qualified": None, "hardware_deployment_ready": False}
