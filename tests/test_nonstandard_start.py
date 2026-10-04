"""Legal reset geometry, passive hold ownership and finite capture boundaries."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.nonstandard_start import (
    NonstandardStart, PROFILE_FILE, PROFILE_SHA256, birth_probabilities,
    nonstandard_start_settings, runtime_expectation, support_height_for_footprint,
    validate_nonstandard_start,
)

ROOT = Path(__file__).resolve().parents[1]
BANK = json.loads((ROOT / PROFILE_FILE).read_text())
JOINTS = BANK["tree_joint_names"]


def rig(count=2, dt=.005, *, enabled=True, evaluation=False, level=None):
    settings = nonstandard_start_settings(100, enabled=enabled)
    if level is not None:
        probabilities = [0., 0., 0.]
        probabilities[level] = 1.
        settings["iteration_probability_schedule"] = [{"actor_update": 0, "probabilities": probabilities}]
    module = NonstandardStart(count, "cpu", dt, settings, ROOT, JOINTS, evaluation=evaluation)
    return module


def step(module, torque=None):
    torque = torch.ones(module.count, 6) if torque is None else torque
    result = module.apply_torque(torque)
    module.advance()
    return result


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("birth_geometry", ROOT / BANK["generator_file"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RecoveryProfileBuilder()


def test_frozen_bank_identity_and_independent_full_closed_chain_geometry(builder):
    assert hashlib.sha256((ROOT / PROFILE_FILE).read_bytes()).hexdigest() == PROFILE_SHA256
    assert BANK["asset_manifest_sha256"] == builder.manifest_sha256
    assert len(BANK["profiles"]) == 20
    for name, row in BANK["profiles"].items():
        q = row["full_joint_positions"]
        assert set(q) == set(JOINTS)
        check = builder.validate_q(q)
        assert check["max_closure_gap_m"] < 1e-8, name
        assert all(50 - .001 <= beta <= 95 + .001 for beta in check["passive_knee_deg"])
        assert row["control6_rad"] == [q[j] for j in BANK["control_joint_names"]]
        # Recompute every collision hull; don't trust the recorded gap scalar.
        bounds = builder.collision_bounds(q, np.eye(3))
        root = np.asarray(row["root_position_m"])
        for bound in bounds:
            assert (bound["points"] + root)[:, 2].min() > .0028, name
        lower = np.stack([bound["lower"] for bound in bounds]).min(0)
        upper = np.stack([bound["upper"] for bound in bounds]).max(0)
        np.testing.assert_allclose(row["collision_footprint_b_m"], [lower[0], upper[0], lower[1], upper[1]])
        assert row["root_quaternion_xyzw"] == [0., 0., 0., 1.]
        assert not row["dynamic_recovery_validated"]
    assert not BANK["hardware_deployment_ready"]


def test_common_rotation_keeps_relative_drive_pose_on_same_closure_branch(builder):
    for row in BANK["profiles"].values():
        parameters = row["generation_parameters"]
        base = builder.knee_pose(parameters["left_knee_deg"], parameters["right_knee_deg"])
        q = row["full_joint_positions"]
        for hip, auxiliary in (("L_joint1", "LL_joint1"), ("R_joint1", "RR_joint1")):
            assert q[auxiliary] - q[hip] == pytest.approx(base[auxiliary] - base[hip], abs=1e-10)
        np.testing.assert_allclose(builder.knees(q), builder.knees(base), atol=.001)


def test_curriculum_uses_absolute_domain_origin_and_smooth_reset_only_progression():
    settings = nonstandard_start_settings(3563)
    assert birth_probabilities(settings, 0) == [.85, .15, 0.]
    assert birth_probabilities(settings, 3563) == [.85, .15, 0.]
    assert birth_probabilities(settings, 3813) == pytest.approx([.8, .175, .025])
    assert birth_probabilities(settings, 4563) == pytest.approx([.7, .2, .1])
    assert birth_probabilities(settings, 5063) == [.65, .2, .15]
    assert birth_probabilities(settings, 9999) == [.65, .2, .15]
    module = rig(15000)
    module.reset(torch.arange(module.count), 1600, generator=torch.Generator().manual_seed(31))
    fractions = torch.bincount(module.level_ids, minlength=3) / module.count
    torch.testing.assert_close(fractions, torch.tensor([.65, .2, .15]), atol=.015, rtol=0.)
    previous = module.profile_ids.clone()
    module.begin_policy()
    for _ in range(301):
        step(module)
    assert torch.equal(module.profile_ids, previous)  # No continuous resampling.


@pytest.mark.parametrize("enabled,evaluation", [(False, False), (True, True)])
def test_disabled_or_evaluation_preserves_nominal_birth_and_rng(enabled, evaluation):
    module = rig(enabled=enabled, evaluation=evaluation, level=2)
    generator = torch.Generator().manual_seed(91)
    before = generator.get_state().clone()
    chosen = module.reset(torch.arange(2), 2000, generator=generator)
    assert chosen.tolist() == [-1, -1]
    assert torch.equal(before, generator.get_state())
    assert not module.holding.any() and not module.capturing.any()
    module.begin_policy()
    assert step(module).eq(1.).all()
    assert module.learning_mask.all()


def test_selected_reset_lanes_use_full_joint_order_and_conservative_support_plane():
    reversed_joints = list(reversed(JOINTS))
    settings = rig(level=1).settings
    module = NonstandardStart(3, "cpu", .005, settings, ROOT, reversed_joints)
    eligible = torch.tensor([True, False, True])
    module.reset(torch.arange(3), 100, eligible=eligible, generator=torch.Generator().manual_seed(3))
    root = torch.tensor([[1., 2., .4, 0., 0., 0., 1.]]).repeat(3, 1)
    joints, velocity = torch.full((3, len(JOINTS)), 7.), torch.ones(3, 6)
    untouched = (root[1].clone(), joints[1].clone(), velocity[1].clone())
    support = torch.tensor([.25, .5, -.1])
    module.initial_state(torch.arange(3), root, joints, velocity, support)
    for lane in (0, 2):
        row = BANK["profiles"][module.profile_names[module.profile_ids[lane]]]
        assert root[lane, 2] == pytest.approx(support[lane].item() + row["root_position_m"][2])
        torch.testing.assert_close(joints[lane], torch.tensor([row["full_joint_positions"][name] for name in reversed_joints]))
        assert velocity[lane].eq(0.).all()
    for actual, original in zip((root[1], joints[1], velocity[1]), untouched):
        assert torch.equal(actual, original)
    assert module.collision_footprints(torch.arange(3))[1].eq(0.).all()
    previous = module.profile_ids[2].clone()
    module.reset(torch.tensor([0]), 100, eligible=torch.zeros(3, dtype=torch.bool))
    assert module.profile_ids[0] == -1 and not module.holding[0]
    assert module.profile_ids[2] == previous and module.remaining_steps[2] == 40


@pytest.mark.parametrize("dt", [.001, .005, .01])
def test_exact_200ms_all_active_motor_zero_after_response_then_direct_rl(dt):
    module = rig(dt=dt, level=1)
    module.reset(torch.arange(2), 100)
    # Represents an already filtered PD/motor command, including both wheels.
    torque = torch.tensor([[40., -40., 4.5, 30., -30., -4.5]]).repeat(2, 1)
    passive_spring_effort = torch.tensor([[120., -110.], [115., -125.]])
    saved_passive = passive_spring_effort.clone()
    module.begin_policy()
    for _ in range(round(.2 / dt)):
        assert step(module, torque).eq(0.).all()
    assert not module.holding.any()
    assert not module.learning_mask.any()  # Last held interval remains excluded.
    assert module.bootstrap_mask(torch.zeros(2, dtype=torch.bool)).all()
    assert torch.equal(passive_spring_effort, saved_passive)
    module.begin_policy()
    torch.testing.assert_close(step(module, torque), torque)
    assert module.learning_mask.all()
    assert torch.equal(module.effective_action_history(torch.ones(2, 6)), torch.ones(2, 6))


def test_mixed_handoff_interval_and_autoreset_keep_distinct_learning_bootstrap_masks():
    module = rig(level=1)
    module.reset(torch.arange(2), 100)
    module.remaining_steps[:] = torch.tensor([1, 0])
    module.begin_policy()
    assert step(module)[0].eq(0.).all()
    assert step(module).eq(1.).all()
    assert module.learning_mask.tolist() == [False, True]
    actions = torch.full((2, 6), 3.)
    history = module.effective_action_history(actions)
    assert history[0].eq(0.).all() and history[1].eq(3.).all()
    assert actions.eq(3.).all()
    learning = module.learning_mask.clone()
    module.reset(torch.tensor([1]), 100)
    assert torch.equal(module.learning_mask, learning)  # Autoreset cannot relabel elapsed interval.
    assert module.bootstrap_mask(torch.zeros(2, dtype=torch.bool)).tolist() == [True, False]
    assert not module.bootstrap_mask(torch.ones(2, dtype=torch.bool)).any()


def test_capture_clamps_visible_and_target_copies_without_losing_height_or_future_target():
    module = rig(level=2)
    module.reset(torch.tensor([0]), 100)
    commands = torch.tensor([[5., -18., .43], [-5., 18., .23]])
    target = commands[:, :2].clone()
    saved_commands, saved_target = commands.clone(), target.clone()
    torch.testing.assert_close(module.cap_commands(commands), torch.tensor([[1., -1., .43], [-5., 18., .23]]))
    torch.testing.assert_close(module.cap_commands(target), torch.tensor([[1., -1.], [-5., 18.]]))
    assert torch.equal(commands, saved_commands) and torch.equal(target, saved_target)
    module.begin_policy()
    for _ in range(299):
        step(module)
    assert module.capturing.tolist() == [True, False] and not module.holding.any()
    # A resampled original target is not replaced by any remembered capped value.
    target[0] = torch.tensor([-4., 8.])
    torch.testing.assert_close(module.cap_commands(target)[0], torch.tensor([-1., 1.]))
    step(module)
    assert not module.capturing.any()
    torch.testing.assert_close(module.cap_commands(target), target)


def test_footprint_support_catches_internal_raised_strip_and_both_slope_signs():
    surfaces = torch.tensor([
        [[-2., 2., 0., 0., 0.], [-.1, .1, .3, 0., 0.], [0., 0., 99., 0., 0.]],
        [[0., 2., .1, .5, -.25], [3., 4., 10., 0., 0.], [0., 0., 0., 0., 0.]],
        [[0., 2., .1, -.5, .25], [3., 4., 10., 0., 0.], [0., 0., 0., 0., 0.]],
        [[3., 4., 10., 0., 0.], [0., 0., 99., 0., 0.], [0., 0., 0., 0., 0.]],
    ])
    valid = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    centers = torch.tensor([[0., 0.], [1., 0.], [1., 0.], [0., 0.]])
    bounds = torch.tensor([[-.5, .5, -.4, .4]]).repeat(4, 1)
    heights = support_height_for_footprint(surfaces, valid, centers, bounds)
    torch.testing.assert_close(heights, torch.tensor([.3, .95, -.05, 0.]))
    empty = support_height_for_footprint(surfaces[:, :0], valid[:, :0], centers, bounds)
    assert empty.eq(0.).all()


def test_native_ppo_existing_adapter_excludes_birth_hold_from_actor_and_critic():
    from test_recovery_rollout_mask import algorithm, observation
    from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask

    module = rig(level=1)
    module.reset(torch.tensor([0]), 100)
    alg = algorithm(steps=14, count=2, normalize_obs=True)
    adapter = attach_recovery_rollout_mask(alg, num_envs=2)
    for policy in range(14):
        module.begin_policy()
        for _ in range(4):
            step(module)
        with torch.no_grad():
            alg.act(observation(2, policy / 10))
        rewards = torch.tensor([1., 1.])
        rewards[~module.learning_mask] = torch.nan
        alg.process_env_step(observation(2, (policy + 1) / 10), rewards, torch.zeros(2), {
            "policy_learning_mask": module.learning_mask.clone(),
            "policy_bootstrap_mask": module.bootstrap_mask(torch.zeros(2, dtype=torch.bool)),
        })
    alg.compute_returns(observation(2, 1.4))
    assert adapter.learning[:, 0].tolist() == [False] * 10 + [True] * 4
    assert alg.storage.advantages[:10, 0].eq(0.).all()
    assert alg.storage.returns[:10, 0].eq(0.).all()
    assert torch.isfinite(alg.storage.advantages).all()
    losses = alg.update()
    assert adapter.metrics["valid_transitions"] == 18
    assert all(np.isfinite(value) for value in losses.values())
    assert all(torch.isfinite(parameter).all() for parameter in alg.actor.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in alg.critic.parameters())


@pytest.mark.parametrize("patch", [
    {"origin_actor_update": -1}, {"origin_actor_update": True}, {"enabled": 1},
    {"hold_seconds": .19}, {"capture_seconds": 2.}, {"clock": "frames"},
    {"profiles_sha256": "0" * 64}, {"iteration_probability_schedule": []},
])
def test_schema_rejects_unfrozen_or_invalid_semantics(patch):
    with pytest.raises(ValueError):
        validate_nonstandard_start(nonstandard_start_settings(0) | patch)


def test_runtime_settings_and_diagnostics_are_detached_snapshots():
    module = rig(level=1)
    module.reset(torch.tensor([0]), 100)
    expected = runtime_expectation(module.settings, .02)
    assert module.runtime_state(.02) == expected
    expected["settings"]["iteration_probability_schedule"][0]["probabilities"][1] = 0.
    assert module.settings["iteration_probability_schedule"][0]["probabilities"][1] == 1.
    diagnostics = module.diagnostics()
    diagnostics["nonstandard_start_holding"].zero_()
    diagnostics["nonstandard_start_profile_id"].fill_(-99)
    assert module.holding[0] and module.profile_ids[0] >= 0
    with pytest.raises(ValueError, match="whole physics steps"):
        rig(dt=.007)
    with pytest.raises(ValueError, match="unique"):
        module.reset(torch.tensor([0, 0]), 100)
