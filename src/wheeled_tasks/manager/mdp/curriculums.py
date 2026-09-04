"""Curriculum components (self-implemented).

- RewardWeightProgression: advance reward-weight stages once a tracked reward
  term's windowed mean beats a threshold for `min_episodes` — windowed-mean gating over (track_height driven stages).
- BaseVerticalAssistForceProgression: decay an upward assist force applied at
  the base in stages as the tracked reward improves. NOTE the key insight
  (article 4.3): assist forces alone are a PPO trap — they must be paired with
  reward-weight stages so "larger action" never maps to "less reward".
Both are env-agnostic: the env polls `step()` and applies returned effects.
"""
from dataclasses import dataclass, field


@dataclass
class Stage:
    reward_weights: dict = field(default_factory=dict)   # merge into env reward table
    reward_scales: dict = field(default_factory=dict)    # optional sigma rescale
    threshold: float = 0.4                               # tracked-term window mean to advance
    min_episodes: int = 500
    force_z: float = 0.0                                 # BaseVerticalAssist stage force


class RewardWeightProgression:
    """Windowed-mean gating over stages; restores defaults at the last stage."""

    def __init__(self, stages: list[Stage], window_size: int = 64, num_steps_per_env: int = 24):
        if not stages:
            raise ValueError("need at least one stage")
        self.stages = stages
        self.window: list[float] = []
        self.window_size = window_size
        self.num_steps_per_env = num_steps_per_env
        self.episodes_in_stage = 0
        self.stage_idx = 0

    def track(self, tracked_reward_mean: float, episodes_finished: int) -> None:
        self.window.append(tracked_reward_mean)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        self.episodes_in_stage += episodes_finished

    def step(self) -> tuple[int, dict]:
        """Returns (stage_idx, effects dict: reward_weights / force_z)."""
        window_mean = sum(self.window) / max(len(self.window), 1)
        last_stage = self.stage_idx == len(self.stages) - 1
        ready = (len(self.window) >= self.window_size
                 and self.episodes_in_stage >= self.stages[self.stage_idx].min_episodes
                 and window_mean >= self.stages[self.stage_idx].threshold)
        if ready:
            if last_stage:
                # final stage reached again: restore default reward (defaults-restore semantics)
                self.restore_defaults = True
            else:
                self.stage_idx += 1
                self.episodes_in_stage = 0
                self.window.clear()

        if getattr(self, "restore_defaults", False):
            effects = {"reward_weights": {}, "reward_scales": {}, "force_z": 0.0,
                       "stage_idx": self.stage_idx, "restore_defaults": True}
        else:
            stage = self.stages[self.stage_idx]
            effects = {"reward_weights": dict(stage.reward_weights),
                       "reward_scales": dict(stage.reward_scales),
                       "force_z": stage.force_z,
                       "stage_idx": self.stage_idx}
        return self.stage_idx, effects


class BaseVerticalAssistForceProgression:
    """Thin wrapper: stages carry force_z, progression logic is shared."""

    def __init__(self, stages: list[Stage], window_size: int = 64):
        self.progression = RewardWeightProgression(stages, window_size=window_size)

    def track(self, tracked_reward_mean: float, episodes_finished: int) -> None:
        self.progression.track(tracked_reward_mean, episodes_finished)

    def step(self) -> float:
        _, effects = self.progression.step()
        return effects["force_z"]
