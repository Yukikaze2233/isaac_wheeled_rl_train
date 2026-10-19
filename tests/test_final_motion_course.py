"""Final payments on retained command populations, without policy optimization."""
from copy import deepcopy
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.flat_command_sampling import SPECIAL_MODES
from wheeled_tasks.chassis.policy_transfer import formal_stage_load_kwargs
from wheeled_tasks.chassis.preflight import design_preflight
from wheeled_tasks.chassis.rough_commands import ScutRoughCommands
from wheeled_tasks.chassis.scut_reward_selection import validate_reward_selection
from wheeled_tasks.chassis.small_rough_course import (
    FINAL_MOTION_VERSION, allocate, install, selection, validate_resume,
)
from wheeled_tasks.chassis.task import choose_scene_groups
from wheeled_tasks.chassis.training_profile import (
    ALGORITHM, effective_profile_algorithm, profile_source_files, validate_profile,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/v7_formal_20261012"


def config(name="final_motion_review.json"):
    return json.loads((CONFIGS / name).read_bytes())


def commands(count, *, final=True):
    grid = torch.zeros(10, 14, 3)
    grid[..., 0] = ((torch.arange(10) + .5) * 9. - 45.)[:, None]
    grid[..., 1] = ((torch.arange(14) + .5) * 9. - 63.)[None, :]
    owner = ScutRoughCommands(count, "cpu", terrain_origins=grid,
                             env_origins=grid[0, torch.arange(count) % 14])
    if final:
        owner.configure_final_motion_modes()
    return owner


def test_review_binds_final_payments_without_changing_ppo_or_enabling_training():
    cfg, parent = config(), config("recovery_after10000_056.json")
    assert cfg["research_training_authorized"] is False
    assert cfg["wheel_material_priors"] == {"enabled": True, "version": "wheel_shape_buckets_v1"}
    assert cfg["friction_combine_mode"] == "multiply"
    assert cfg["contact_domain"]["enabled"] is False
    assert selection(cfg)["version"] == FINAL_MOTION_VERSION
    assert validate_profile(cfg)["position_iterations"] == 12
    assert validate_profile(cfg)["velocity_iterations"] == 6
    assert effective_profile_algorithm(cfg) == ALGORITHM
    selection_receipt = validate_reward_selection(cfg, stage_name="terrain")
    assert selection_receipt["nonzero_term_count"] == 31
    assert selection_receipt["tracking_commands"] == "actor_visible_targets_without_reward_deadzone"
    assert not selection_receipt["velocity_height_gate_enabled"]
    assert selection_receipt["nonzero_weights"]["track_lin_vel_xy_square"] == -1.
    assert validate_resume(parent, cfg)["optimizer"] == "fresh_Adam_actor_std_preserved_critic_fresh"
    parent_infos = {"stage": "foundation", "successful_updates_total": 14841,
                    "contract_sha256": cfg["formal_critic_adaptation"]["source_contract_sha256"]}
    assert formal_stage_load_kwargs(cfg, parent_infos, stage="terrain")["load_cfg"] == {
        "actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False}
    with pytest.raises(ValueError, match="authenticated foundation parent"):
        validate_resume(config("motion_retention_review.json"), cfg)
    assert validate_resume(cfg, cfg)["optimizer"] == "inherited"
    previous_materials = deepcopy(cfg)
    previous_materials["wheel_material_priors"]["version"] = "wheel_low_grip_01_v3"
    with pytest.raises(ValueError, match="baseline final-course entry"):
        validate_resume(previous_materials, cfg)
    files = profile_source_files(cfg)
    for name in ("final_rough_reward_runtime", "final_rough_commands", "final_rough_definition",
                 "rough_commands", "finite_energy_readback", "diagnostic_curriculum"):
        assert f"src/wheeled_tasks/chassis/{name}.py" in files
    assert "references/v7_final_terrain_b8ff79f3/source_manifest.json" in files


def test_actual_course_install_keeps_all_populations_and_20s_post_handoff():
    from wheeled_tasks.chassis.fixed_flat_rehearsal import rehearsal_summary
    cfg = config()
    groups = [name for name, _ in choose_scene_groups(cfg["scene_groups"], 4096)]
    spec = allocate(cfg, ROOT, 4096, groups)
    assert rehearsal_summary(spec)["reward"] == "unified31 chassis payments on every population"
    assert spec["population_lane_counts"] == {
        "fixed_flat": 512, "dynamic_flat": 1024, "temporal_flat": 1024,
        "recovery": 512, "source_rough": 1024}
    assert len(spec["lane_counts"]) == 17 and min(spec["lane_counts"].values()) >= 30
    owner = commands(4096, final=False)
    recovery = torch.tensor(["recovery_pose" in next(s for s in cfg["scene_groups"]
                          if s["name"] == group) for group in groups])
    receipt = install(spec, owner, recovery)
    assert receipt["population_lane_counts"] == spec["population_lane_counts"]
    assert owner.fixed_request_mask.sum() == 512
    assert owner._temporal_rehearsal.mask.sum() == 1024
    assert owner._rough["source_flat_rehearsal"]["mask"].sum() == 1536
    assert cfg["formal_recovery_episode_window"]["post_handoff_seconds"] == 20.
    control = load_chassis_control(ROOT / cfg["control_math_source"])
    expected = design_preflight(cfg, control, ROOT, stage="terrain", role="train",
        seed=20261020, num_envs=256, physics_probe=True)["runtime_expected"]
    assert expected["v7_final_rough_reward"]["tracking_commands"] == (
        "actor_visible_targets_without_reward_deadzone")


def test_final_mode_pool_uses_real_eligibility_and_continuous_signed_ranges():
    n = 16384
    owner = commands(n)
    owner.set_source_flat_rehearsal(torch.arange(n // 2), source_iteration_offset=14841)
    rng = torch.Generator().manual_seed(20261020)
    ids = torch.arange(n)
    owner.reset(ids, torch.zeros(n), torch.zeros(n, dtype=torch.bool), rng)
    assert owner.special_mode_id.eq(-1).all()
    owner.episode_physics_ticks.fill_(1000)
    owner._resample(ids, 14841, rng)
    for subset in (ids[:n // 2], ids[n // 2:]):
        eligible = subset[~owner.standing_first_draw[subset]]
        for index, probability in enumerate((.15, .15, .20, .10)):
            actual = owner.special_mode_id[eligible].eq(index).float().mean().item()
            assert abs(actual - probability) < .02
        zero = subset[owner.special_mode_id[subset] == 3]
        assert owner.generator_command3[zero].eq(0.).all()
        dash = subset[owner.special_mode_id[subset] == 2]
        vx = owner.generator_command3[dash, 0]
        assert vx.abs().min() >= 2. and vx.abs().max() <= 3.
        assert vx.unique().numel() > .9 * len(vx)
        assert .4 < (vx > 0).float().mean() < .6
        spin = subset[(owner.special_mode_id[subset] >= 0) & (owner.special_mode_id[subset] < 2)]
        assert owner.generator_command3[spin, 2].abs().max() <= 4.5 * math.pi
    assert owner.summary()["source_flat_training_rehearsal"]["special_mode_iteration_starts"] == [0] * 4
    assert [mode[2] for mode in commands(14, final=False).special_modes] == [.15, .15, .30]
    assert [mode[2] for mode in SPECIAL_MODES] == [.15, .15, .30]


def test_final_mode_restart_binds_constructor_and_preserves_random_stream():
    first, second = commands(28), commands(28)
    first_rng, second_rng = torch.Generator().manual_seed(13), torch.Generator().manual_seed(27)
    ids = torch.arange(28)
    first.reset(ids, torch.zeros(28), torch.zeros(28, dtype=torch.bool), first_rng)
    saved = deepcopy(first.state_dict(first_rng))
    second.load_state_dict(saved, second_rng)
    first.episode_physics_ticks.fill_(1000)
    second.episode_physics_ticks.fill_(1000)
    first._resample(ids, 14841, first_rng)
    second._resample(ids, 14841, second_rng)
    assert torch.equal(first.generator_command3, second.generator_command3)
    assert torch.equal(first_rng.get_state(), second_rng.get_state())
    with pytest.raises(ValueError, match="before the first reset"):
        first.configure_final_motion_modes()
    legacy = commands(28, final=False)
    before_rng = second_rng.get_state().clone()
    with pytest.raises(ValueError):
        legacy.load_state_dict(saved, second_rng)
    assert torch.equal(second_rng.get_state(), before_rng)


def test_update_probe_reads_actual_single_frame_policy_without_sampling_or_mutation():
    from tensordict import TensorDict
    from wheeled_tasks.chassis.diagnostic_curriculum import PolicyUpdateProbe
    from wheeled_tasks.chassis.flat_evaluation_runtime import _stock_actor
    actor = _stock_actor(None)
    observations = TensorDict({"policy": torch.zeros(24, 32, 35)}, batch_size=[24, 32])
    probe = PolicyUpdateProbe(actor, SimpleNamespace(observations=observations), [
        (torch.arange(16), {"group_name": "fixed_flat"}),
        (torch.arange(16, 32), {"group_name": "source_rough"})])
    weights = deepcopy(actor.state_dict())
    rng = torch.random.get_rng_state().clone()
    probe.capture()
    metrics = probe.metrics()
    assert metrics["DiagnosticUpdate/kl_rollout_probe"] == 0.
    assert metrics["DiagnosticUpdate/fixed_flat/action_mean_shift_l2"] == 0.
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, weights[name]) for name, value in actor.state_dict().items())


def test_train_entry_groups_probe_from_actual_population_without_legacy_skills():
    from wheeled_tasks.chassis.small_rough_course import ROLES
    tree = ast.parse((ROOT / "scripts/train_chassis.py").read_text())
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "c.get('diagnostic_logging') or final_motion")
    scope = dict(c={}, final_motion=True, env=SimpleNamespace(skills=None),
        runner=SimpleNamespace(alg=SimpleNamespace(actor=None, storage=None)),
        mixed_exposure=SimpleNamespace(roles=torch.arange(5).repeat_interleave(32)))
    exec(compile(ast.Module(body=[block], type_ignores=[]), "actual_train_probe_binding", "exec"), scope)
    assert set(scope["policy_probe"].groups) == set(ROLES)
    assert scope["policy_probe"].ids.unique().numel() == 16 * len(ROLES)
