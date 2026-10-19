"""Read the actual RSL5 PPO object; no simulator/rollout/optimizer steps."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from wheeled_tasks.chassis.training_profile import (
    PROFILE, ALGORITHM, configure_runner, profile_identity, profile_source_files,
    restore_profile_clocks, restore_rng, validate_profile_selection, verify_learning_rate, verify_runner,
)


def contract():
    return dict(training_profile=PROFILE, actor_dim=35, actor_frame_dim=35,
        critic_dim=78, action_dim=6, history_length=1, physics_dt=.005, policy_dt=.02,
        position_iterations=12, velocity_iterations=6, num_steps_per_env=24,
        target_num_envs=4096, closure_gap_termination_m=.003,
        critic_observation_version="v6_scut78_v1", manual_context35=True,
        actor_observation_source="encoders_imu_command_reference36")


def config():
    return dict(seed=123, device="cpu", num_steps_per_env=48, save_interval=10,
        actor=dict(class_name="MLPModel", hidden_dims=[16], activation="relu",
            obs_normalization=True, distribution_cfg=dict(class_name="GaussianDistribution", init_std=.2)),
        critic=dict(class_name="MLPModel", hidden_dims=[16], activation="relu", obs_normalization=True),
        algorithm={**ALGORITHM, "value_loss_coef": 2.}, logger="tensorboard")


def test_historical_source_layout_keeps_its_original_checkpoint_digest():
    sources = dict.fromkeys(profile_source_files(contract()), "1" * 64)
    sources["src/wheeled_tasks/v40/core.py"] = sources.pop(
        "src/wheeled_tasks/chassis/control_core.py")
    sources["src/wheeled_tasks/agents/v40_ppo_cfg.py"] = sources.pop(
        "src/wheeled_tasks/agents/chassis_ppo_cfg.py")
    sources["src/wheeled_tasks/v40/contract.py"] = "2" * 64
    encoded = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    identity = profile_identity(contract(), sources)
    assert identity["training_source_sha256"] == hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="complete SHA-256"):
        profile_identity(contract(), {k: v for k, v in sources.items()
                                     if k != "src/wheeled_tasks/v40/contract.py"})
    with pytest.raises(ValueError, match="mixes retired and current"):
        profile_identity(contract(), sources | {
            "src/wheeled_tasks/chassis/control_core.py": "3" * 64})


class LayoutEnv:
    cfg = {}
    num_envs = 2
    num_actions = 6
    device = "cpu"
    def get_observations(self):
        return TensorDict(dict(policy=torch.zeros(2, 35), critic=torch.zeros(2, 78)), batch_size=[2])


@pytest.mark.parametrize("stage", ["foundation", "terrain", "mixed"])
def test_actual_ppo_value_and_full_configuration_after_final_overrides(stage):
    c = contract()
    cfg = configure_runner(config(), c, stage=stage)
    runner = OnPolicyRunner(LayoutEnv(), cfg, device="cpu")
    receipt = verify_runner(runner.alg, cfg, c)
    assert receipt["algorithm"]["value_loss_coef"] == 4.
    assert cfg["actor"]["hidden_dims"] == cfg["critic"]["hidden_dims"] == [256, 128, 64]
    assert runner.alg.actor.obs_normalization is runner.alg.critic.obs_normalization is False
    assert runner.alg.rnd is runner.alg.symmetry is None
    assert runner.alg.storage.num_transitions_per_env == 24
    assert receipt["initial_std"] == [1.] * 6


def test_legacy_configuration_stays_untouched_and_new_mismatch_is_rejected():
    legacy = config()
    assert configure_runner(legacy, {}, stage="terrain") == legacy
    c = contract()
    c["critic_dim"] = 81
    with pytest.raises(ValueError, match="critic_dim"):
        configure_runner(legacy, c, stage="terrain")


@pytest.mark.parametrize("change", [
    {"training_profile": None}, {"training_profile": "v6_78d_scratch_typo"},
    {"critic_dim": 81}, {"critic_observation_version": "wrong_order"},
    {"actor_observation_source": "old_81"}, {"manual_context35": False},
    {"history_length": True}, {"remediation": {"source": "old"}},
    {"continuation": {"source": "old"}}, {"task_campaign": {"domain": "flat"}},
    {"learning_rate_schedule": "fixed"}, {"entropy_coef": 0.},
    {"learning_rate": .001}, {"value_loss_coef": 2.}, {"initial_noise_std": .05},
])
def test_profile_layout_and_declared_ppo_cannot_silently_select_legacy_or_override(change):
    c = {**contract(), **change}
    with pytest.raises(ValueError):
        configure_runner(config(), c, stage="foundation")


def test_legacy_81_layout_is_allowed_without_new_profile():
    c = {"critic_dim": 81, "actor_dim": 35}
    assert validate_profile_selection(c) is False
    assert profile_identity(c, {}) == {}
    assert configure_runner(config(), c, stage="terrain") == config()


@pytest.mark.parametrize("key", ["wheel_material_priors", "pc_relative_priors"])
def test_privileged_randomization_owner_cannot_be_silently_ignored_by_legacy(key):
    legacy = {"critic_dim": 81, "actor_dim": 35, key: {"enabled": True}}
    with pytest.raises(ValueError, match="named 78D"):
        validate_profile_selection(legacy)
    assert validate_profile_selection({**legacy, key: {"enabled": False}}) is False
    selected = {**contract(), "friction_combine_mode": "multiply",
                key: {"enabled": True, "version": ("wheel_shape_buckets_v1"
                    if key == "wheel_material_priors" else "pc_relative_priors_v1")}}
    assert validate_profile_selection(selected) is True


@pytest.mark.parametrize("key", ["wheel_material_priors", "pc_relative_priors"])
@pytest.mark.parametrize("settings", [None, [], {"enabled": 1}, {"enabled": "false"}])
def test_randomization_owner_selection_is_typed_before_any_native_start(key, settings):
    with pytest.raises(ValueError, match="boolean selection"):
        validate_profile_selection({**contract(), key: settings})


@pytest.mark.parametrize("version", [None, "old", 1])
def test_pc_prior_version_is_rejected_before_native_construction(version):
    with pytest.raises(ValueError, match="before native"):
        validate_profile_selection({**contract(), "pc_relative_priors": {"enabled": True, "version": version}})


@pytest.mark.parametrize("extra", [
    {}, {"friction_combine_mode": "average"},
    {"friction_combine_mode": "multiply", "restitution_combine_mode": "average"},
    {"friction_combine_mode": "multiply", "contact_domain": {"enabled": True}},
])
def test_material_policy_conflicts_are_rejected_before_native_construction(extra):
    with pytest.raises(ValueError):
        validate_profile_selection({**contract(), **extra, "wheel_material_priors": {"enabled": True}})


def test_runner_receipt_uses_actual_observations_not_declared_dimensions():
    class WrongLayout(LayoutEnv):
        def get_observations(self):
            return TensorDict(dict(policy=torch.zeros(2, 35), critic=torch.zeros(2, 79)), batch_size=[2])
    c = contract()
    cfg = configure_runner(config(), c, stage="foundation")
    runner = OnPolicyRunner(WrongLayout(), cfg, device="cpu")
    with pytest.raises(RuntimeError, match="Instantiated PPO layout"):
        verify_runner(runner.alg, cfg, c)


def test_source_identity_is_order_independent_but_authenticates_every_source():
    c = contract()
    sources = {path: "a" * 64 for path in profile_source_files(c)}
    sources["src/wheeled_tasks/chassis/feedback_transport.py"] = "b" * 64
    identity = profile_identity(c, sources)
    assert profile_identity(c, dict(reversed(list(sources.items())))) == identity
    for path in sources:
        changed = {**sources, path: "c" * 64}
        assert profile_identity(c, changed)["training_source_sha256"] != identity["training_source_sha256"]
    with pytest.raises(ValueError, match="provenance"):
        profile_identity(c, {path: "a" * 64 for path in list(sources)[1:]})
    with pytest.raises(ValueError, match="provenance"):
        profile_identity(c, {**sources, "other": "invalid"})


def native_rate(rate, schedule="adaptive", group_rate=None):
    return SimpleNamespace(learning_rate=rate, schedule=schedule,
        optimizer=SimpleNamespace(param_groups=[{"lr": rate},
            {"lr": rate if group_rate is None else group_rate}]))


@pytest.mark.parametrize("rate", [1e-5, 1e-4 / 1.5, 1e-4 * 1.5, .008, 1e-2])
def test_adaptive_rate_can_increase_or_decrease_within_rsl_bounds(rate):
    assert verify_learning_rate(native_rate(rate), contract()) == rate


@pytest.mark.parametrize("rate", [1e-6, .010001, 0., -1., float("nan"), float("inf"), None, True])
def test_invalid_adaptive_rate_is_rejected(rate):
    with pytest.raises(RuntimeError):
        verify_learning_rate(native_rate(rate), contract())


def test_all_optimizer_groups_and_active_schedule_must_match():
    with pytest.raises(RuntimeError, match="differ"):
        verify_learning_rate(native_rate(.0002, group_rate=.0003), contract())
    with pytest.raises(RuntimeError, match="schedule"):
        verify_learning_rate(native_rate(.0002, "fixed"), contract())


def test_warmup_preserves_restored_rate_and_actor_resumes_adaptive():
    alg = native_rate(.0008, "fixed")
    assert verify_learning_rate(alg, contract(), actor_enabled=False, before_rate=.0008) == .0008
    alg.learning_rate = .0012
    for group in alg.optimizer.param_groups:
        group["lr"] = .0012
    with pytest.raises(RuntimeError, match="Critic warmup"):
        verify_learning_rate(alg, contract(), actor_enabled=False, before_rate=.0008)
    alg.schedule = "adaptive"
    assert verify_learning_rate(alg, contract()) == .0012


def test_old_fixed_contracts_keep_fixed_schedule_and_exact_rate():
    c = {"task_campaign": {"domain": "flat"}, "learning_rate": 1e-4}
    assert verify_learning_rate(native_rate(1e-4, "fixed"), c) == 1e-4
    for alg in (native_rate(2e-4, "fixed"), native_rate(1e-4, "adaptive")):
        with pytest.raises(RuntimeError, match="Fixed remediation"):
            verify_learning_rate(alg, c)


def saved_clocks():
    return dict(stage="foundation", stage_actor_update=10, global_actor_update_offset=1000,
                global_actor_update=1010, actor_updates_consumed=40)


def test_exact_restart_inherits_clocks_and_accepts_only_identical_explicit_values():
    infos = saved_clocks()
    c = contract()
    keys = ("stage_actor_update", "global_actor_update_offset", "actor_updates_consumed")
    expected = {key: value for key, value in infos.items() if key != "stage"}
    assert restore_profile_clocks(c, infos, stage="foundation", requested={key: None for key in keys}) == expected
    assert restore_profile_clocks(c, infos, stage="foundation", requested={key: infos[key] for key in keys}) == expected
    for key in keys:
        with pytest.raises(ValueError, match="Requested"):
            restore_profile_clocks(c, infos, stage="foundation", requested={key: 0})
    assert restore_profile_clocks({"critic_dim": 81}, {}, stage="legacy", requested={}) is None


@pytest.mark.parametrize("change", [
    {"stage": "terrain"}, {"stage_actor_update": None}, {"global_actor_update_offset": -1},
    {"global_actor_update": 10}, {"actor_updates_consumed": 9}, {"actor_updates_consumed": True},
])
def test_exact_restart_rejects_missing_inconsistent_or_different_stage_clocks(change):
    with pytest.raises(ValueError):
        restore_profile_clocks(contract(), {**saved_clocks(), **change}, stage="foundation", requested={})


@pytest.mark.parametrize("settings,required", [
    ({"nonstandard_start": {"enabled": True, "profiles_file": "birth.json"}},
        ["src/wheeled_algo/recovery_rollout.py", "birth.json"]),
    ({"step_jump_task": {"enabled": True}}, ["src/wheeled_tasks/chassis/step_sensors.py", "src/wheeled_tasks/chassis/step_trigger.py"]),
    ({"skill_specs": {"flat": {}}}, ["src/wheeled_tasks/chassis/release_stop.py", "src/wheeled_tasks/chassis/ordinary_large_stair.py"]),
    ({"diagnostic_logging": True}, ["src/wheeled_tasks/chassis/diagnostic_curriculum.py"]),
])
def test_profile_binds_selected_sample_control_and_diagnostic_dependencies(settings, required):
    sources = profile_source_files({**contract(), **settings})
    assert set(required) <= set(sources)
    assert {"src/wheeled_tasks/chassis/training_schedule.py", "src/wheeled_tasks/chassis/terrain_runout.py",
            "src/wheeled_tasks/chassis/rewards.py"} <= set(sources)


def test_disabled_birth_constructors_and_ordinary_height_dependency_are_still_bound():
    sources = profile_source_files({**contract(), "nonstandard_start": {"enabled": False},
                                    "ordinary_posture_curriculum": {"enabled": False}})
    assert len(sources) == len(set(sources))
    assert "src/wheeled_tasks/chassis/nonstandard_start.py" in sources
    assert "src/wheeled_tasks/chassis/ordinary_posture_curriculum.py" in sources
    assert "src/wheeled_algo/recovery_rollout.py" not in sources
    ordinary = profile_source_files({**contract(), "ordinary_posture_curriculum": {"enabled": True}})
    assert "src/wheeled_tasks/chassis/nonstandard_start.py" in ordinary


def test_fresh_seed_fork_overrides_parent_rng_without_touching_learning_state():
    torch.manual_seed(10)
    parent = torch.get_rng_state().clone()
    env = SimpleNamespace(generator=torch.Generator().manual_seed(20))
    infos = dict(rng_state=parent, env_rng_state=env.generator.get_state())
    restore_rng(torch, env, infos, seed=30, mode="fresh_fork")
    a = torch.rand(8)
    e = torch.rand(8, generator=env.generator)
    restore_rng(torch, env, infos, seed=30, mode="fresh_fork")
    assert torch.equal(torch.rand(8), a)
    assert torch.equal(torch.rand(8, generator=env.generator), e)
    restore_rng(torch, env, infos, seed=31, mode="fresh_fork")
    assert not torch.equal(torch.rand(8), a)
    restore_rng(torch, env, infos, seed=99, mode="inherit")
    assert torch.equal(torch.get_rng_state(), parent)
