"""Round4 scratch lineage and successful-PPO-update curriculum clock."""
from copy import deepcopy
import json
from pathlib import Path


def validate_initialization_options(contract, args):
    if "round4" not in contract:
        return
    if any(getattr(args, name, None) for name in ("warm_start", "stage_transfer", "finetune")):
        raise ValueError("Round4 full requires random initialization from scratch; model transfer is forbidden")
    if not getattr(args, "ground_usd", None):
        raise ValueError("Round4 full requires --ground-usd with the pinned official Grid USD")
    if hasattr(args, "max_iterations"):
        profile = contract["round4"]
        if args.stage != "locomotion":
            raise ValueError("Round4 full training requires the locomotion command stage")
        if args.num_envs != profile["num_envs"]:
            raise ValueError("Round4 full training requires the contract's 1024 environments")
        if not getattr(args, "resume", None) and args.max_iterations != profile["planned_updates"]:
            raise ValueError("Round4 scratch launch requires the full 30000 planned updates")


def scratch_initialization(contract, seed):
    return {
        "mode": "scratch", "method": "random_initialization_from_scratch", "seed": seed,
        "architecture": {"actor_obs_dim": 125, "critic_obs_dim": 29, "action_dim": 6,
                         "actor_class": "MLPModel", "critic_class": "MLPModel",
                         "policy": deepcopy(contract["policy"])},
        "feature_flags": {
            "startup_material_randomization": True, "push_curriculum": True,
            "command_curriculum": True, "body_angular_rate_penalty": True,
            "standing_wheel_quiet": True, "terrain": False, "jump": False,
        },
        "profile_config": deepcopy(contract["round4"]),
    }


def checked_resume_state(checkpoint, provenance, expected_manifest):
    """Resume only an identical full contract whose checkpoint attests scratch origin.

    Use the completed-update clock in checkpoint infos, not RSL's saved loop index.
    This restores curriculum progress, not environment/RNG trajectory state.
    """
    from wheeled_algo.v40_export import _read_checkpoint

    source = provenance["run_manifest"]
    initialization = source.get("initialization")
    expected = expected_manifest["initialization"]
    if (source.get("training_profile") != "round4_full" or type(initialization) is not dict
            or initialization != expected or source.get("source_provenance") != initialization
            or initialization.get("mode") != "scratch"
            or initialization.get("method") != "random_initialization_from_scratch"):
        raise ValueError("Round4 resume requires the same scratch initialization metadata, seed and full profile")
    saved, digest = _read_checkpoint(Path(checkpoint))
    if digest != provenance["checkpoint_sha256"]:
        raise ValueError("Round4 resume checkpoint changed after validation")
    infos = saved.get("infos", {})
    if infos.get("initialization") != initialization:
        raise ValueError("Round4 resume checkpoint does not attest scratch initialization")
    progress = infos.get("training_curriculum")
    if (type(progress) is not dict or progress.get("timebase") != "successful_ppo_updates"
            or type(progress.get("completed_updates")) is not int or progress["completed_updates"] < 0
            or type(progress.get("state")) is not dict):
        raise ValueError("Round4 resume requires a completed-update curriculum snapshot")
    return deepcopy(progress)


class Round4Curriculum:
    """One invocation's successful-update offset, with environment-owned task state."""

    def __init__(self, env, manifest, *, completed_updates=0):
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError("Round4 curriculum requires a nonnegative completed update count")
        self.env, self.manifest = env, manifest
        self.initial_updates = completed_updates
        self.on_update(0)

    def on_update(self, completed_updates):
        absolute = self.initial_updates + completed_updates
        self.env.set_training_iteration(absolute)
        state = self.env.training_curriculum_state
        if type(state) is not dict:
            raise ValueError("Round4 env must expose a JSON curriculum state object")
        # Snapshot, never retain mutable references into the environment.
        state = json.loads(json.dumps(state, allow_nan=False))
        self.manifest["training_curriculum"] = {
            "timebase": "successful_ppo_updates", "completed_updates": absolute, "state": state,
        }
