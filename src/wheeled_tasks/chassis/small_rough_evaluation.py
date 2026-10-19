"""Evaluate a flat/recovery actor on small rough without rewriting its parent."""
from copy import deepcopy


SURFACE = "small_rough_regression_v1"


class RoughBirthWindowMetrics:
    """Measure a finite window with static origins and explicit birth labels."""

    def __init__(self, env):
        import torch
        from .episode_metrics import EpisodeMetrics

        rough = env.flat_episode.commands._rough
        definition = rough["definition"]
        self.rows, self.columns = definition["rows"], definition["columns"]
        self.family_names = definition["keys"]
        self.column_family = rough["column_key_ids"].clone()
        self.start_xy = rough["start_xy"].clone()
        self.origins = env.origins.clone()
        self.source = torch.tensor([name == "source_rough" for name in env.scene_groups],
            dtype=torch.bool, device=env.device)
        self.birth_cells = ((self.origins[:, :2] - self.start_xy) / 9.).floor().long()
        inside = ((self.birth_cells[:, 0] >= 0) & (self.birth_cells[:, 0] < self.rows)
            & (self.birth_cells[:, 1] >= 0) & (self.birth_cells[:, 1] < self.columns))
        if not bool(self.source.any()) or bool((self.source & ~inside).any()):
            raise ValueError("Rough birth measurement requires actual rough origins inside the grid")
        pairs = self.birth_cells.cpu().tolist()
        families = self.column_family.cpu().tolist()
        groups = [f"row{row}/{self.family_names[families[col]]}" if active else "excluded"
            for (row, col), active in zip(pairs, self.source.cpu().tolist())]
        self.metrics = EpisodeMetrics(groups, env.device, env.policy_dt)
        self.current_counts = torch.zeros(self.rows * self.columns, dtype=torch.long, device=env.device)
        self.outside_count = torch.zeros((), dtype=torch.long, device=env.device)

    def observe(self, data):
        import torch

        world_xy = data["position"][:, :2] + self.origins[:, :2]
        finite = torch.isfinite(world_xy).all(-1)
        cells = ((torch.where(finite[:, None], world_xy, self.start_xy) - self.start_xy) / 9.).floor().long()
        inside = finite & (cells[:, 0] >= 0) & (cells[:, 0] < self.rows) & (cells[:, 1] >= 0) & (cells[:, 1] < self.columns)
        eligible = self.source & data["measurement_valid_mask"]
        index = cells[:, 0].clamp(0, self.rows - 1) * self.columns + cells[:, 1].clamp(0, self.columns - 1)
        self.current_counts.scatter_add_(0, index, (eligible & inside).long())
        self.outside_count += (eligible & ~inside).sum()
        same_birth_tile = inside & (cells == self.birth_cells).all(-1)
        self.metrics.observe({**data, "measurement_valid_mask": eligible & same_birth_tile}, active=self.source)

    def report(self):
        import torch

        values = self.metrics.report()
        values["groups"].pop("excluded", None)
        by_cell = self.current_counts.reshape(self.rows, self.columns)
        by_family = torch.zeros(self.rows, len(self.family_names), dtype=torch.long, device=by_cell.device)
        by_family.index_add_(1, self.column_family, by_cell)
        return {"scope": "birth-tile eligible control measurements with static origins; kinematic samples after leaving the birth tile are excluded",
            "episode_outcome_scope": "whole episodes grouped by birth tile, including outcomes after leaving it; not failure rates confined to the birth tile",
            "qualification": False, "ppo_credit": 0, "measurements": values,
            "family_names": list(self.family_names),
            "current_tile_eligible_frames": by_cell.cpu().tolist(),
            "current_row_family_eligible_frames": by_family.cpu().tolist(),
            "outside_grid_eligible_frames": int(self.outside_count),
            "current_tile_count_scope": "visited cells, not independent episodes, terrain birth coverage or optimized samples"}


def configuration(parent):
    """Change the measurement surface and owners, preserving physical inputs.

    This is an inference overlay, not a learning-state migration. Nominal
    births and the existing rough source sampler measure retained locomotion;
    they do not assess recovery on uneven ground or define training fractions.
    """
    from .training_profile import validate_profile_selection
    from .small_rough_course import KEY, selection
    mixed = selection(parent) is not None
    if (not mixed and (parent.get("reward_profile") != "scut0719_flat_v1"
            or parent.get("curriculum_stage") != "foundation"
            or parent.get("enabled_stages") != ["foundation"]
            or parent.get("formal_terrain", {}).get("type") != "plane")
            or parent.get("physx_solver_type") != 1
            or not validate_profile_selection(parent)):
        raise ValueError("Small rough regression requires an authenticated TGS retained-skill parent")
    result = deepcopy(parent)
    for key in ("recovery_training", "formal_recovery_episode_window", "formal_scripted_recovery_mix", "formal_flat_refresh",
                "formal_flat_rehearsal", "formal_dynamic_nonsteady_birth", "evaluation",
                "flat_special_mode_iteration_cap", KEY, "formal_stage_transition",
                "formal_stage_optimizer", "formal_critic_adaptation", "formal_rough_nonsteady_birth",
                "formal_rough_birth_anchor", "formal_flat_rehearsal_collider"):
        result.pop(key, None)
    result.update(
        reward_profile="scut0730_rough_v1", curriculum_stage="terrain", enabled_stages=["terrain"],
        stages=[dict(stage, name="terrain", terrain=["rough"]) for stage in parent["stages"]],
        scene_groups=[{"name": "source_rough", "terrain": ["rough"], "fraction": 1.}],
        formal_terrain={"type": "rough", "source": "scut0730"},
        formal_rough_initial_level=0, formal_foundation_nominal_birth=True,
        critic_height_reference="root_link_world_z",
        formal_stage_policy={"later_skills_enabled": False,
            "no_automatic_iteration_based_promotion": True,
            "scope": "inference only; original actor clocks retained, no PPO or optimizer migration"})
    result["scut_reward_mapping"]["root_height_reference"] = "root_link_z_minus_base_ground_scan"
    validate_profile_selection(result)
    return result


def retained_flat_configuration(parent, scenes, *, scripted):
    """Project an authenticated retained-skill actor onto nominal plane cases.

    Only inference ownership, scene allocation and flat payment/reference
    selection change. In particular the checkpoint clocks, mechanics, transport
    and recovery controller are not replaced by an older training contract.
    """
    from .small_rough_course import KEY, selection
    from .training_profile import validate_profile_selection
    from .flat_episode_selection import FLAT_SPECIAL_SCHEDULE
    mixed = selection(parent) is not None
    if (not validate_profile_selection(parent) or not mixed and (
            parent.get("reward_profile") != "scut0719_flat_v1"
            or parent.get("curriculum_stage") != "foundation"
            or parent.get("formal_terrain", {}).get("type") != "plane")):
        raise ValueError("Retained flat measurements require a flat/recovery or mixed parent")
    result = deepcopy(parent)
    for key in (KEY, "formal_scripted_recovery_mix", "formal_flat_refresh",
                "formal_flat_rehearsal", "formal_dynamic_nonsteady_birth", "evaluation",
                "formal_stage_transition", "formal_stage_optimizer", "formal_critic_adaptation", "formal_rough_initial_level",
                "formal_rough_nonsteady_birth", "formal_rough_birth_anchor", "formal_flat_rehearsal_collider"):
        result.pop(key, None)
    if not scripted:
        result.pop("recovery_training", None)
        result.pop("formal_recovery_episode_window", None)
    result.update(
        reward_profile="scut0719_flat_v1", curriculum_stage="foundation", enabled_stages=["foundation"],
        stages=[dict(stage, name="foundation", terrain=["flat"]) for stage in parent["stages"]],
        scene_groups=deepcopy(scenes),
        formal_terrain={"type": "plane", "source": "scut0719", "env_spacing_m": 4.},
        formal_foundation_nominal_birth=True, critic_height_reference="root_link_world_z",
        formal_stage_policy={"later_skills_enabled": False,
            "no_automatic_iteration_based_promotion": True,
            "flat_special_mode_schedule": FLAT_SPECIAL_SCHEDULE,
            "scope": "inference only; retained clocks and nominal plane births, no PPO"})
    result["scut_reward_mapping"]["root_height_reference"] = "root_link_z_minus_env_origin"
    # Recovery profiles and their physical controller remain selected above.
    # The typed evaluator owns its lane quotas and nominal ordinary births;
    # training rehearsal selectors would impose an unrelated fixed 25% quota.
    validate_profile_selection(result)
    return result


def validate_overlay(protocol, overlay):
    """Reuse the existing 480-lane mean-policy, one-episode rough protocol."""
    expected = dict(schema="flat-policy-evaluation-v1", mode="source_distribution",
        actor_mode="mean", episodes_per_lane=1, environment_seed=290929, actor_seed=290929,
        policy_dt_s=.02, horizon_seconds=20., center_reference="root_link_origin_world",
        assessment="measurement_only")
    if (protocol != expected or overlay.get("num_envs") != 480
            or overlay.get("lane_case_ids") != ["source_distribution"] * 480
            or overlay.get("record_diagnostics") is not True
            or overlay.get("evaluation_surface") != SURFACE):
        raise ValueError("Small rough regression requires the unchanged 480-lane mean source protocol")
    selected = overlay.get("contact_generation_distance")
    from .contact_generation_distance import SUPPORTED_VERSIONS
    if selected is not None and selected not in SUPPORTED_VERSIONS:
        raise ValueError("Unknown small rough contact-distance candidate")


def receipt(parent, effective):
    changed = sorted(key for key in parent.keys() | effective.keys()
                     if (key in parent) != (key in effective) or parent.get(key) != effective.get(key))
    return {"surface": SURFACE, "changed_configuration_fields": changed,
            "checkpoint_contract_rewritten": False, "optimizer_or_training_state_loaded": False,
            "physics_and_control": "inherited; optional named contact-distance intervention is recorded separately",
            "births": "nominal pose on actual terrain, initial row 0; full generated mesh remains accessible",
            "commands_and_rewards": "existing small rough source owners; no fixed flat or scripted recovery lanes",
            "actor_clock": "parent clocks held constant during inference",
            "qualification": "measurement only; not proof of rough recovery or a complete training course"}
