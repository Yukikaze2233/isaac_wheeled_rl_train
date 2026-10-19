"""Command/learning ownership tests; synthetic traces are not policy scores."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.flat_command_sampling import ScutFlatCommands
from wheeled_tasks.chassis.motion_transition_course import AXIS_VERSION, DEFAULT_SETTINGS, PHASES, VERSION
from wheeled_tasks.chassis.small_rough_course import MixedCourseExposure, allocate, install, selection, validate_resume
from wheeled_tasks.chassis.rough_commands import ScutRoughCommands
from wheeled_tasks.chassis.task import choose_scene_groups


ROOT = Path(__file__).resolve().parents[1]


def settings():
    result = deepcopy(DEFAULT_SETTINGS)
    result.update(phase_timeout_range_ticks=[[4, 4]] * 5, minimum_phase_ticks=[1] * 5,
                  settled_ticks=[2] * 6, minimum_final_hold_ticks=2)
    return result


def owner(count=5, ids=(2,), config=None):
    commands = ScutFlatCommands(count, "cpu")
    commands.set_motion_training_requests(torch.tensor(ids), config or settings())
    rng = torch.Generator().manual_seed(72)
    commands.reset(torch.arange(count), torch.zeros(count), torch.zeros(count, dtype=torch.bool), rng, 14841)
    return commands, rng


def step(commands, rng, *, measured=None, omega=None, done=None, advance=True):
    n = commands.num_envs
    velocity = torch.zeros(n, 3)
    velocity[:, 0] = commands.command3[:, 0] if measured is None else measured
    done = torch.zeros(n, dtype=torch.bool) if done is None else done
    commands.begin_policy()
    for _ in range(4):
        commands.advance_physics()
    old = commands.command3.clone()
    angular_velocity = torch.zeros(n, 3)
    if omega is not None:
        angular_velocity[:, 2] = omega
    commands.observe_motion_interval(velocity, angular_velocity, done, torch.ones(n, dtype=torch.bool))
    diagnostic = commands.diagnostics()
    torch.testing.assert_close(commands.command3, old)
    result = {"diagnostics": {"source_flat_episode": dict(diagnostic, episode_id=torch.zeros(n, dtype=torch.long)),
        "done": done.clone(), "timeout": done.clone(), "boundary": torch.zeros_like(done),
        "terminated": torch.zeros_like(done)}}
    if advance:
        commands.advance(.02, torch.zeros(n), 14841, rng)
    return result


def fixed_sequence(commands, lesson=1):
    course = commands._temporal_rehearsal
    course.lesson[course.ids] = lesson
    course.targets[course.ids] = torch.tensor([.5, -.25 if lesson == 3 else .25, -.5])
    return course


def test_candidate_retains_all_skill_owners_and_loads_only_as_review():
    p = ROOT / "configs/v7_formal_20261012"
    config = json.loads((p / "motion_retention_review.json").read_text())
    old = json.loads((p / "recovery_after10000_056.json").read_text())
    assert config["research_training_authorized"] is False
    assert selection(config)["version"] == VERSION
    assert validate_resume(old, config)["stage_transition"]
    groups = [name for name, _ in choose_scene_groups(config["scene_groups"], 4096)]
    spec = allocate(config, ROOT, 4096, groups)
    assert spec["population_lane_counts"] == dict(
        fixed_flat=512, dynamic_flat=1024, temporal_flat=1024, recovery=512, source_rough=1024)
    assert all(value >= 30 for value in spec["lane_counts"].values())
    assert {s["recovery_pose"] for s in config["scene_groups"] if "recovery_pose" in s} == {
        s["recovery_pose"] for s in old["scene_groups"] if "recovery_pose" in s}
    assert config["formal_tracking_reward_repair"] == "v7_exact_command_tracking_v1"


def test_small_physics_probe_preserves_training_review_gate_and_real_birth_population():
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.flat_episode_selection import require_native_flat_admission
    from wheeled_tasks.chassis.preflight import design_preflight
    config = json.loads((ROOT / "configs/v7_formal_20261012/motion_retention_review.json").read_text())
    control = load_chassis_control(ROOT / config["control_math_source"])
    with pytest.raises(ValueError, match="explicit source flat"):
        require_native_flat_admission(config)
    result = design_preflight(config, control, ROOT, stage="terrain", role="train",
                              seed=20261019, num_envs=256, physics_probe=True)
    assert result["physics_probe_no_optimization"]
    birth = result["runtime_expected"]["formal_dynamic_nonsteady_birth"]
    assert birth["source_lanes"] == 128 and birth["scripted_lanes"] == 32
    assert birth["flat_source_lanes"] == birth["rough_source_lanes"] == 64
    assert result["runtime_expected"]["rough_birth_anchor"] == {
        "version": "terrain-center-v1", "rough_lanes": 64,
        "x_offset_m": 0., "y_offset_m": 0., "maximum_xy_error_m": 0.}
    with pytest.raises(ValueError, match="explicit source flat"):
        require_native_flat_admission(config)
    assert config["research_training_authorized"] is False


def test_motion_candidate_removes_inactive_quiet_settings_without_reassigning_payments():
    from wheeled_tasks.chassis.small_rough_course import INACTIVE_QUIET_REWARD_KEYS
    directory = ROOT / "configs/v7_formal_20261012"
    candidate = json.loads((directory / "motion_retention_review.json").read_text())
    parent = json.loads((directory / "recovery_after10000_056.json").read_text())
    assert not INACTIVE_QUIET_REWARD_KEYS & candidate.keys()
    receipt = validate_resume(parent, candidate)
    assert receipt["removed_inactive_reward_fields"] == sorted(INACTIVE_QUIET_REWARD_KEYS)
    assert candidate["formal_tracking_reward_repair"] == "v7_exact_command_tracking_v1"
    modified = deepcopy(candidate)
    modified["quiet_velocity_weight"] = -8.
    with pytest.raises(ValueError, match="only be removed"):
        validate_resume(parent, modified)


def test_continuous_speed_height_signs_and_lesson_sampling():
    commands, rng = owner(4096, tuple(range(4096)), DEFAULT_SETTINGS)
    course = commands._temporal_rehearsal
    assert (commands.command3 == 0).all()
    assert course.targets[:, 0].unique().numel() > 4000
    assert course.height.unique().numel() > 4000
    assert course.targets.abs().min() >= .05 and course.targets.abs().max() <= 3.
    assert set(course.band.tolist()) == {0, 1, 2, 3}
    assert set(course.lesson.tolist()) == {0, 1, 2, 3}
    assert .45 < (course.targets[:, 0] > 0).float().mean() < .55
    direct = course.lesson == 3
    assert (course.targets[direct, 0] * course.targets[direct, 1] < 0).all()
    assert (course.targets[~direct, 0] * course.targets[~direct, 1] > 0).all()
    assert (course.targets[:, 0] * course.targets[:, 2] < 0).all()
    assert course.height.min() >= .20 and course.height.max() <= .42


def test_phase_progress_uses_actual_speed_and_preserves_paid_command():
    commands, rng = owner()
    course = fixed_sequence(commands)
    for _ in range(2):
        step(commands, rng)
    assert course.phase[2] == 1 and commands.command3[2, 0] == .5
    for i in range(4):
        info = step(commands, rng, measured=torch.zeros(5))
        paid = info["diagnostics"]["source_flat_episode"]["motion_course"]
        assert paid["phase"][2] == 1
        assert not paid["phase_passed"][2]
        assert paid["phase_timed_out"][2] == (i == 3)
        assert commands.learning_mask[2]
    assert course.phase[2] == 2
    assert not course.qualified_prefix[2]
    # Requested .5 m/s does not fabricate a measured .5 m/s entry.
    info = step(commands, rng)
    paid = info["diagnostics"]["source_flat_episode"]["motion_course"]
    assert paid["entry_vx"][2] == 0 and paid["entry_target"][2] == .5


def test_micro_command_cannot_qualify_while_stationary():
    commands, rng = owner()
    course = fixed_sequence(commands)
    course.targets[2, 0] = .05
    for _ in range(2):
        step(commands, rng)
    assert commands.command3[2, 0] == .05
    info = step(commands, rng, measured=torch.zeros(5))
    assert not info["diagnostics"]["source_flat_episode"]["motion_course"]["tracking"][2]


def test_stop_requires_continuous_holding_and_no_velocity_reset():
    commands, rng = owner()
    course = fixed_sequence(commands, lesson=2)
    for _ in range(4):
        step(commands, rng)
    assert course.phase[2] == 3 and commands.command3[2, 0] == 0
    step(commands, rng, measured=torch.full((5,), .2))
    assert course.phase[2] == 3
    step(commands, rng)
    assert course.phase[2] == 3
    info = step(commands, rng)
    assert course.phase[2] == 4 and commands.command3[2, 0] == -.5
    assert info["diagnostics"]["source_flat_episode"]["motion_course"]["phase_passed"][2]


def test_stationary_episode_keeps_zero_and_detects_drift_after_settling():
    commands, rng = owner()
    course = fixed_sequence(commands, lesson=0)
    for _ in range(3):
        step(commands, rng)
    step(commands, rng, measured=torch.full((5,), .2))
    for _ in range(3):
        info = step(commands, rng)
    paid = info["diagnostics"]["source_flat_episode"]["motion_course"]
    assert commands.command3[2].eq(0).all() and course.phase[2] == 0
    assert paid["hold_broken"][2] and not paid["qualified_sequence"][2]


def test_partial_reset_and_restart_preserve_other_lanes_and_rng():
    commands, rng = owner(5, (1, 2))
    fixed_sequence(commands)
    for _ in range(3):
        step(commands, rng)
    state = commands.state_dict(rng)
    restored, other_rng = owner(5, (1, 2))
    restored.load_state_dict(state, other_rng)
    for _ in range(5):
        a, b = step(commands, rng), step(restored, other_rng)
        torch.testing.assert_close(commands.command3, restored.command3)
        for key in a["diagnostics"]["source_flat_episode"]["motion_course"]:
            torch.testing.assert_close(a["diagnostics"]["source_flat_episode"]["motion_course"][key],
                                       b["diagnostics"]["source_flat_episode"]["motion_course"][key])
    before = commands._temporal_rehearsal.state_dict()
    for obj, generator in ((commands, rng), (restored, other_rng)):
        obj.reset(torch.tensor([1]), torch.zeros(1), torch.zeros(1, dtype=torch.bool), generator, 14841)
    for key, value in before.items():
        torch.testing.assert_close(value[2], commands._temporal_rehearsal.state_dict()[key][2])
    torch.testing.assert_close(commands._temporal_rehearsal.targets, restored._temporal_rehearsal.targets)
    assert torch.equal(rng.get_state(), other_rng.get_state())
    bad = deepcopy(commands.state_dict(rng))
    bad["motion_course"]["phase"][1] = 99
    before_rng = rng.get_state().clone()
    with pytest.raises(ValueError):
        commands.load_state_dict(bad, rng)
    assert torch.equal(before_rng, rng.get_state())


def test_effective_exposure_commits_only_final_loss_samples():
    spec = {"population_index": (0, 1, 2, 3, 4),
            "episode_policy_ticks": 12,  # Compact fixture, never a production horizon.
            "small_rough_mix": {"version": VERSION, "temporal": settings()}}
    exposure = MixedCourseExposure(spec, 12, "cpu")
    commands, rng = owner()
    fixed_sequence(commands, lesson=2)
    for tick in range(12):
        info = step(commands, rng, done=torch.full((5,), tick == 11))
        exposure.observe(tick, info)
    mask = torch.ones(12, 5, dtype=torch.bool)
    mask[:, 3] = False  # Script ownership never earns recovery actor exposure.
    prepared = exposure.prepare(mask)
    assert not any(exposure.counts.values())
    assert prepared["completed"] == 1
    masked = mask.clone()
    masked[2, 2] = False
    assert exposure.prepare(masked)["completed"] == 0
    counts = exposure.commit(prepared)
    assert counts["recovery"] == 0
    assert sum(counts[p] for p in PHASES) == counts["temporal_flat"] == 12
    assert counts["motion/reverse_after_verified_stop"] == 1
    assert counts["motion/entry/brake/from_target"] == 1
    assert exposure.complete_loss_sequences == 1
    resumed = MixedCourseExposure(spec, 12, "cpu")
    resumed.load_state_dict(exposure.state_dict())
    assert resumed.state_dict() == exposure.state_dict()
    wrong_horizon = dict(spec, episode_policy_ticks=999)
    with pytest.raises(ValueError, match="inconsistent"):
        MixedCourseExposure(wrong_horizon, 12, "cpu").load_state_dict(exposure.state_dict())


def test_rough_owner_installation_restart_and_heading_do_not_override_pure_axis():
    config = json.loads((ROOT / "configs/v7_formal_20261012/motion_retention_review.json").read_text())
    groups = [name for name, _ in choose_scene_groups(config["scene_groups"], 256)]
    spec = allocate(config, ROOT, 256, groups)
    tiles = torch.zeros(10, 14, 3)
    tiles[:, :, 0] = ((torch.arange(10) + .5) * 9. - 45.)[:, None]
    tiles[:, :, 1] = ((torch.arange(14) + .5) * 9. - 63.)[None, :]
    origins = tiles[0, torch.arange(256) % 14].clone()
    for lane, origin in spec["origins_by_env"].items():
        origins[lane] = torch.tensor(origin)
    def build():
        obj = ScutRoughCommands(256, "cpu", terrain_origins=tiles, env_origins=origins)
        readback = install(spec, obj, torch.tensor(spec["population_index"]) == 3)
        assert readback['motion_course']['settings'] == config['formal_small_rough_mix']['temporal']
        assert readback['motion_course'] == obj._temporal_rehearsal.summary()
        return obj
    commands, rng = build(), torch.Generator().manual_seed(7)
    commands.reset(torch.arange(256), torch.zeros(256), torch.zeros(256, dtype=torch.bool), rng)
    for _ in range(3):
        step(commands, rng, advance=False)
        commands.advance(.02, torch.ones(256), 0, rng, world_positions=origins)
    course = commands._temporal_rehearsal
    linear = course.ids[course.axis[course.ids] == 0]
    rotation = course.ids[course.axis[course.ids] == 1]
    assert commands.command3[linear, 2].eq(0).all()
    assert commands.command3[rotation, 0].eq(0).all()
    state = commands.state_dict(rng)
    other, other_rng = build(), torch.Generator().manual_seed(99)
    other.load_state_dict(state, other_rng)
    torch.testing.assert_close(other._temporal_rehearsal.targets, commands._temporal_rehearsal.targets)
    assert torch.equal(rng.get_state(), other_rng.get_state())


def axis_settings(fraction=.25):
    result = settings()
    result['axis_coverage'] = {
        'version': AXIS_VERSION, 'rotation_fraction_of_moving_lessons': fraction,
        'yaw_edges_rad_s': [0., 1., 2. * math.pi, 3.25 * math.pi, 4.5 * math.pi],
    }
    return result


def test_continuous_pure_spin_preserves_stationary_allocation_and_exact_zero_translation():
    commands, rng = owner(8192, tuple(range(8192)), axis_settings())
    course = commands._temporal_rehearsal
    moving, rotation = course.lesson != 0, course.axis == 1
    assert .23 < rotation[moving].float().mean() < .27
    assert .23 < (~moving).float().mean() < .27
    assert not rotation[~moving].any()
    assert course.targets[rotation, 0].unique().numel() > 1400
    assert course.targets[rotation].abs().min() > 0
    assert course.targets[rotation].abs().max() <= 4.5 * math.pi
    assert set(course.band[rotation].tolist()) == {0, 1, 2, 3}
    assert .45 < (course.targets[rotation, 0] > 0).float().mean() < .55
    for _ in range(2):
        step(commands, rng)
    assert commands.command3[rotation, 0].eq(0).all()
    torch.testing.assert_close(commands.command3[rotation, 2], course.targets[rotation, 0])
    assert commands.command3[~rotation, 2].eq(0).all()
    assert commands.command3[~moving].eq(0).all()


def test_axis_extension_does_not_shift_source_command_or_partial_reset_rng_stream():
    plain, first_rng = owner(64, tuple(range(32)), settings())
    axis, second_rng = owner(64, tuple(range(32)), axis_settings())
    assert torch.equal(first_rng.get_state(), second_rng.get_state())
    assert (axis._temporal_rehearsal.axis == 1).any()
    for _ in range(1000):
        step(plain, first_rng)
        step(axis, second_rng)
        assert torch.equal(plain.command3[32:], axis.command3[32:])
    assert (plain.command_counter[32:] > 1).all()
    assert torch.equal(first_rng.get_state(), second_rng.get_state())
    for commands, generator in ((plain, first_rng), (axis, second_rng)):
        commands.reset(torch.tensor([0]), torch.zeros(1), torch.zeros(1, dtype=torch.bool), generator, 14841)
    assert torch.equal(first_rng.get_state(), second_rng.get_state())
    assert torch.equal(plain.command3[32:], axis.command3[32:])


def test_tiny_spin_requires_measured_rotation_without_a_fictitious_center_drift_gate():
    commands, rng = owner(config=axis_settings(1.))
    course = fixed_sequence(commands)
    course.axis[2] = 1
    course.targets[2] = torch.tensor([1.e-4, 1., -1.])
    for _ in range(2):
        step(commands, rng)
    row = step(commands, rng)['diagnostics']['source_flat_episode']['motion_course']
    assert not row['tracking'][2] and not row['phase_passed'][2]
    assert commands.command3[2, 0] == 0 and commands.command3[2, 2] == 1.e-4
    for _ in range(2):
        row = step(commands, rng, omega=commands.command3[:, 2].clone())[
            'diagnostics']['source_flat_episode']['motion_course']
    assert row['phase_passed'][2] and course.phase[2] == 2
    row = step(commands, rng, measured=torch.full((5,), .2), omega=commands.command3[:, 2].clone())[
        'diagnostics']['source_flat_episode']['motion_course']
    assert row['tracking'][2] and row['planar_speed'][2] == .2
    assert commands.learning_mask[2]


def test_rotation_stop_then_reverse_needs_actual_angular_stop_without_reset():
    commands, rng = owner(config=axis_settings(1.))
    course = fixed_sequence(commands, lesson=2)
    course.axis[2] = 1
    course.targets[2] = torch.tensor([3., 2., -3.])
    for _ in range(4):
        step(commands, rng, omega=commands.command3[:, 2].clone())
    assert course.phase[2] == 3 and commands.command3[2].eq(0).all()
    row = step(commands, rng, omega=torch.full((5,), .4))[
        'diagnostics']['source_flat_episode']['motion_course']
    assert not row['tracking'][2] and course.phase[2] == 3
    assert row['entry_yaw'][2] == 3 and row['entry_target_yaw'][2] == 3
    step(commands, rng)
    assert course.phase[2] == 3
    step(commands, rng)
    assert course.phase[2] == 4
    assert commands.command3[2, 0] == 0 and commands.command3[2, 2] == -3
    assert course.entry_yaw[2] == 0 and course.entry_passed[2]


def test_axis_restart_is_complete_and_invalid_axis_is_rejected_atomically():
    commands, rng = owner(5, (1, 2), axis_settings(1.))
    fixed_sequence(commands)
    commands._temporal_rehearsal.axis[1:3] = 1
    for _ in range(3):
        step(commands, rng)
    state = commands.state_dict(rng)
    restored, other_rng = owner(5, (1, 2), axis_settings(1.))
    restored.load_state_dict(state, other_rng)
    for _ in range(5):
        first, second = step(commands, rng), step(restored, other_rng)
        torch.testing.assert_close(commands.command3, restored.command3)
        for key in first['diagnostics']['source_flat_episode']['motion_course']:
            torch.testing.assert_close(first['diagnostics']['source_flat_episode']['motion_course'][key],
                                       second['diagnostics']['source_flat_episode']['motion_course'][key])
    bad = deepcopy(commands.state_dict(rng))
    bad['motion_course']['axis'][1] = 2
    before = commands.state_dict(rng)
    with pytest.raises(ValueError, match='outside the declared domains'):
        commands.load_state_dict(bad, rng)
    assert torch.equal(before['generator_state'], rng.get_state())
    for key, value in before['motion_course'].items():
        assert torch.equal(value, commands._temporal_rehearsal.state_dict()[key])
    legacy, legacy_rng = owner(5, (1, 2))
    with pytest.raises(ValueError, match='identity'):
        legacy.load_state_dict(state, legacy_rng)


def test_rotation_exposure_has_separate_units_and_only_final_loss_frames_count():
    spec = {'population_index': (0, 1, 2, 3, 4), 'episode_policy_ticks': 12,
            'small_rough_mix': {'version': VERSION, 'temporal': axis_settings(1.)}}
    exposure = MixedCourseExposure(spec, 12, 'cpu')
    commands, rng = owner(config=axis_settings(1.))
    course = fixed_sequence(commands, lesson=2)
    course.axis[2] = 1
    course.targets[2] = torch.tensor([3., 2., -3.])
    for tick in range(12):
        row = step(commands, rng, omega=commands.command3[:, 2].clone(),
                   measured=torch.where(commands.command3[:, 2] != 0, .2, 0.),
                   done=torch.full((5,), tick == 11))
        exposure.observe(tick, row)
    mask = torch.ones(12, 5, dtype=torch.bool)
    mask[:, 3] = False
    prepared = exposure.prepare(mask)
    assert prepared['completed'] == 1 and not exposure.motion_counts
    excluded = mask.clone()
    excluded[:, 2] = False
    rejected = exposure.prepare(excluded)
    assert rejected['completed'] == 0
    assert all(value == 0 for key, value in rejected['motion_counts'].items()
               if key.startswith('motion/rotation'))
    storage = SimpleNamespace(values=torch.zeros(12, 5, 1), returns=torch.ones(12, 5, 1),
                              advantages=torch.ones(12, 5, 1))
    metrics = exposure.optimization_metrics(storage, mask)
    prefix = 'MixedCourse/command/rotation/'
    assert metrics[prefix + 'root_reference_planar_speed_mean_m_s'] == .2
    assert metrics[prefix + 'yaw_error_mean_rad_s'] == 0
    assert metrics[prefix + 'nonzero_request_transitions'] > 0
    excluded_metrics = exposure.optimization_metrics(storage, excluded)
    assert excluded_metrics[prefix + 'nonzero_request_transitions'] == 0
    counts = exposure.commit(prepared)
    assert counts['motion/rotation_frames'] == 12 and counts['motion/translation_frames'] == 0
    assert counts['motion/rotation/reverse_after_verified_stop'] == 1
    assert counts['motion/rotation/entry/brake/from_target'] == 1
    assert counts['motion/rotation/entry/brake/near_zero'] == 0
    assert all(value == 0 for key, value in counts.items() if key.startswith('motion/request/'))
