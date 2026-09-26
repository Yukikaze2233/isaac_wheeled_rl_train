"""Explicit actor-only ABI migration; optimizer and critic are never relabeled."""
import torch


REFERENCE_SOURCE = "encoders_imu_command_reference36"
LEGACY_SOURCE = "scut35_encoders_imu_commands"


def is_reference_migration(source, target):
    return (source.get("actor_observation_source") == LEGACY_SOURCE
            and target.get("actor_observation_source") == REFERENCE_SOURCE
            and target.get("actor_migration") == "scut35_to_reference36_zero_new_columns")


def transfer_actor_state(state, source, target):
    if not is_reference_migration(source, target):
        if source["actor_dim"] != target["actor_dim"]:
            raise ValueError("Actor dimension change requires an explicit observation migration")
        if target.get("actor_observation_source") == REFERENCE_SOURCE:
            for key in ("actor_frame_dim", "history_length", "actor_observation_source", "actor_layout",
                        "action_dim", "policy_action_order", "v5_control", "policy_dt", "physics_dt",
                        "asset_manifest_sha256", "control_math_source"):
                if source.get(key) != target.get(key):
                    raise ValueError(f"Reference actor interface mismatch: {key}")
        return state, None
    if (source["actor_dim"], source["actor_frame_dim"], source["history_length"],
            target["actor_dim"], target["actor_frame_dim"], target["history_length"]) != (35, 35, 1, 36, 36, 1):
        raise ValueError("Only single-frame 35D to 36D command-reference migration is supported")
    for key in ("action_dim", "policy_action_order", "v5_control", "policy_dt", "physics_dt", "control_math_source",
                "asset_manifest_sha256", "task_modes", "phases"):
        if source.get(key) != target.get(key):
            raise ValueError(f"Reference migration changes the mechanical/control interface: {key}")
    original = state["mlp.0.weight"]
    if original.shape != (256, 35):
        raise ValueError("Reference migration requires the declared 256/128/64 actor")
    result = {key: value.clone() for key, value in state.items()}
    weight = original.new_zeros((256, 36))
    weight[:, :35] = original
    weight[:, 29:32] = 0.
    result["mlp.0.weight"] = weight
    return result, {"source_dim": 35, "target_dim": 36, "zero_initialized_columns": [29, 30, 31, 35],
                    "exploration_parameters": "preserved", "critic_optimizer": "fresh"}
