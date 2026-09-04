"""Experiment framework: controlled-variable branches + registry.

An ExperimentSpec freezes one experimental variable against the baseline
(branch choice, history length, constraint limit, ...). The registry is the
single source of truth for `scripts/run_experiment.py` and
`scripts/compare_experiments.py`, mirroring how the registry enumerates
its Exp0xx series; on the training server each spec also carries the Isaac
Lab task id / runner_class string used to reproduce it at scale.
"""
from dataclasses import dataclass, field


@dataclass
class ExperimentSpec:
    name: str
    branch: str                       # ppo_hist | him | dreamwaq | np3o
    description: str
    overrides: dict = field(default_factory=dict)   # ExtTrainCfg overrides
    env_kwargs: dict = field(default_factory=dict)  # toy env kwargs (local runs)
    isaaclab_task: str = "WheeledBiped-Flat-v0"     # server-side task id
    isaaclab_note: str = ""                          # runner integration note

    @property
    def run_name(self) -> str:
        return f"{self.name}"


REGISTRY: dict[str, ExperimentSpec] = {}


def register(spec: ExperimentSpec) -> None:
    assert spec.name not in REGISTRY, f"duplicate experiment: {spec.name}"
    REGISTRY[spec.name] = spec


def get(name: str) -> ExperimentSpec:
    return REGISTRY[name]


def names() -> list[str]:
    return sorted(REGISTRY)


# --------------------------------------------------------------------------- #
# controlled-variable series (one variable per experiment)                          #
# --------------------------------------------------------------------------- #
register(ExperimentSpec(
    name="exp000_framestack5",
    branch="ppo_hist",
    description="Frame-stack baseline: plain PPO, actor eats a "
                "flattened 5-frame observation history, no estimator.",
    env_kwargs={"hist_len": 5},
    isaaclab_note="use_frame_stack=True, num_obs_hist=5",
))
register(ExperimentSpec(
    name="exp001_him_latent16",
    branch="him",
    description="HIM estimator branch: HIM one-step privileged estimator, "
                "3-frame history, 16D latent into the actor.",
    env_kwargs={"hist_len": 3},
    isaaclab_note="runner_class=HIMRunner (estimator + history stream)",
))
register(ExperimentSpec(
    name="exp002_dreamwaq_kl1",
    branch="dreamwaq",
    description="VAE-estimation branch: CENet-VAE implicit estimation, "
                "reconstruction + KL(weight 1.0) added to PPO.",
    env_kwargs={"hist_len": 3},
    isaaclab_note="runner_class=OnPolicyDreamWaqRunner (kl_weight=1.0)",
))
register(ExperimentSpec(
    name="exp003_np3o_limit05",
    branch="np3o",
    description="Constrained PPO: BarlowTwins representation + cost critic, "
                "Lagrangian tuned toward cost limit 0.5.",
    env_kwargs={"hist_len": 3},
    isaaclab_note="runner_class=OnConstraintPolicyRunner (num_costs=1)",
))
register(ExperimentSpec(
    name="exp004_np3o_limit15",
    branch="np3o",
    description="Controlled variable vs exp003: relax the cost limit to 1.5 — "
                "isolates the constraint's effect on the reward/constraint trade.",
    overrides={"seed_note": "identical to exp003 except cost_limit"},
    env_kwargs={"hist_len": 3},
    isaaclab_note="runner_class=OnConstraintPolicyRunner (looser limit)",
))
register(ExperimentSpec(
    name="exp006_gru_hist3",
    branch="gru",
    description="GRU-memory branch: GRU(64) memory over the 3-frame history, "
                "residual current-obs path into the actor.",
    env_kwargs={"hist_len": 3},
    isaaclab_note="runner_class=GRURunner (ActorCriticRecurrentGRU)",
))
register(ExperimentSpec(
    name="exp005_rough_curriculum",
    branch="ppo_hist",
    description="Server-side only: WheeledBiped-Rough-v0 with the reward-weight "
                "curriculum enabled (terrain stairs/slopes drive the FSM).",
    isaaclab_task="WheeledBiped-Rough-v0",
    isaaclab_note="enable_curriculum=True; requires Isaac Sim",
))


def branch_trainer(branch: str):
    """Resolve a branch name to its local trainer class (lazy import)."""
    from wheeled_algo.algorithms import DreamWaqTrainer, GRUTrainer, HIMTrainer, NP3OTrainer
    from .ppo_hist import PPOHistTrainer
    return {"ppo_hist": PPOHistTrainer, "him": HIMTrainer, "gru": GRUTrainer,
            "dreamwaq": DreamWaqTrainer, "np3o": NP3OTrainer}[branch]
