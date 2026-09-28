"""Runtime curricula follow actor updates and preserve in-flight request state."""
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.task import Phase, PhaseTracker
from wheeled_tasks.chassis.skill_commands import SkillCommands
from wheeled_tasks.chassis.training_schedule import (
    TrainingSchedule, actor_update_clock, jump_bin_probabilities, stage_actor_updates,
)


def env(count=4096, **settings):
    generator = torch.Generator().manual_seed(29)
    return SimpleNamespace(num_envs=count, device="cpu", global_actor_update=0, stage_actor_update=0,
        cfg={"new_asset_curriculum": {"enabled": True, **settings}},
        random=lambda n: torch.rand(n, generator=generator))


HEIGHT = {"actor_update_boundary": 200, "initial_fixed_fraction": .8,
          "final_fixed_fraction": .5, "fixed_heights_m": [.23, .25, .28, .305, .32, .35, .38, .4, .43]}
JUMP = {"bins_m": [.03, .06, .1, .15, .2, .25, .3],
        "initial_probabilities": [.3, .25, .2, .1, .05, .05, .05],
        "minimum_probability": .05, "confirmed_pass_transfer_fraction": .5}
PROTOCOL = {"stress_episode_fraction": 1., "interval_seconds": [.5, 4.],
            "repeat_probability": .1, "cancel_probability": .1, "new_probability": .8}


def test_actor_clock_does_not_depend_on_transition_count_or_parallelism():
    e = env()
    e.global_actor_update, e.stage_actor_update, e.training_transitions = 1220, 220, 8_000_000
    assert actor_update_clock(e) == 1220
    assert stage_actor_updates(e) == 220
    e.num_envs, e.training_transitions = 20000, 0
    assert (actor_update_clock(e), stage_actor_updates(e)) == (1220, 220)
    e.stage_actor_update = -1
    with pytest.raises(ValueError):
        stage_actor_updates(e)


def test_height_mixture_changes_at_exact_stage_update_and_retains_full_domain():
    e = env(50000, height_sampling_schedule=HEIGHT)
    schedule = TrainingSchedule(e)
    fixed = torch.tensor(HEIGHT["fixed_heights_m"])
    for update, expected in [(199, .8), (200, .5)]:
        e.stage_actor_update = update
        sample = schedule.sample_height(e.num_envs, [.23, .43])
        fraction = (sample[:, None] == fixed).any(-1).float().mean().item()
        assert fraction == pytest.approx(expected, abs=.01)
        assert sample.min() == pytest.approx(.23) and sample.max() == pytest.approx(.43)
        assert ((sample > .231) & (sample < .249)).any()


def test_jump_promotion_transfers_only_confirmed_mass_and_preserves_bin_floor():
    assert jump_bin_probabilities(JUMP) == JUMP["initial_probabilities"]
    p = jump_bin_probabilities(JUMP, [.03, .06])
    assert p[:2] == pytest.approx([.175, .15])
    assert sum(p) == pytest.approx(1.) and min(p) >= .05
    assert p[-1] > .05
    assert jump_bin_probabilities(JUMP, JUMP["bins_m"]) == JUMP["initial_probabilities"]
    with pytest.raises(ValueError):
        jump_bin_probabilities({**JUMP, "initial_probabilities": [1.] * 7})


def test_jump_sampler_updates_live_promotion_but_preserves_running_jump_domain():
    e = env(50000, jump_curriculum=JUMP)
    s = TrainingSchedule(e)
    ids = torch.arange(e.num_envs)
    s.sample_jump(ids, {"kind": "jump"})
    initial = (s.jump_bin == 0).float().mean().item()
    e.confirmed_jump_bins_m = [.03]
    s.sample_jump(ids, {"kind": "jump"})
    promoted = (s.jump_bin == 0).float().mean().item()
    assert initial == pytest.approx(.3, abs=.01)
    assert promoted == pytest.approx(.175, abs=.01)
    s.sample_jump(ids, {"kind": "running_jump", "semantics": {"jump_apex_delta_m": .1}})
    torch.testing.assert_close(s.jump_apex, torch.full_like(s.jump_apex, .1))
    assert (s.jump_bin == -1).all()
    s.sample_jump(ids, {"kind": "jump", "semantics": {"jump_apex_delta_m": .1},
                       "request_events": [{"time_s": .8, "type": "new"}]})
    torch.testing.assert_close(s.jump_apex, torch.full_like(s.jump_apex, .1))
    assert (s.jump_bin == -1).all()


def test_episode_degradation_is_unilateral_and_persists_until_reset():
    e = env(20000, unilateral_degradation_fraction=.05, edge_reset_fraction=.1)
    s = TrainingSchedule(e)
    velocity = torch.zeros(e.num_envs, 6)
    s.reset(torch.arange(e.num_envs), velocity)
    assert s.degraded.float().mean().item() == pytest.approx(.05, abs=.006)
    assert s.edge_reset.float().mean().item() == pytest.approx(.1, abs=.008)
    affected = (s.motor_scale < 1).sum(-1)
    assert torch.equal(affected == 3, s.degraded)
    assert not ((s.motor_scale[:, 0] < 1) & (s.motor_scale[:, 3] < 1)).any()
    assert velocity[:, 0].abs().max() <= .3 and velocity[:, 5].abs().max() <= .5
    assert velocity[:, 3:6].abs().max() <= .5
    assert (velocity[:, [1, 2]] == 0).all()
    previous = s.motor_scale.clone()
    s.command_duration(25)
    assert torch.equal(previous, s.motor_scale)
    s.reset(torch.tensor([0]), velocity[:1])
    assert torch.equal(previous[1:], s.motor_scale[1:])


def test_command_dwell_supports_short_and_protected_long_requests():
    e = env(20000, command_resampling_seconds=[.5, 4.],
            long_command_resampling_seconds=[4., 8.], long_command_fraction=.2)
    duration = TrainingSchedule(e).command_duration(e.num_envs)
    assert duration.min() >= .5 and duration.max() <= 8.
    assert (duration >= 4).float().mean().item() == pytest.approx(.2, abs=.01)


def test_research_height_boundary_has_twice_the_continuous_sampling_density():
    e = env(50000)
    settings = {"range_m": [.23, .43], "boundary_fraction_of_range": .1, "boundary_weight": 2.}
    values = TrainingSchedule(e).sample_continuous_height(e.num_envs, settings)
    outer = (values < .25) | (values > .41)
    assert outer.float().mean().item() == pytest.approx(1 / 3, abs=.01)
    assert values.min() >= .23 and values.max() <= .43


def test_busy_repeat_and_airborne_cancel_keep_start_and_landing_request():
    e = env(1, request_protocol=PROTOCOL)
    s = TrainingSchedule(e)
    s.reset(torch.tensor([0]))
    yes, no = torch.tensor([True]), torch.tensor([False])
    assert s.update_requests(torch.tensor([.8]), torch.tensor([Phase.GROUND]), yes, yes).item()
    start = s.request_start.clone()
    s.next_request[:] = 0.
    e.random = lambda n: torch.full((n,), .15)  # repeat
    s.update_requests(torch.tensor([1.]), torch.tensor([Phase.TAKEOFF]), yes, yes)
    assert s.request_event.item() == 2 and torch.equal(s.request_start, start)
    s.next_request[:] = 0.
    e.random = lambda n: torch.full((n,), .05)  # cancel
    assert s.update_requests(torch.tensor([1.2]), torch.tensor([Phase.FLIGHT]), yes, yes).item()
    assert s.request_event.item() == 3 and torch.equal(s.request_start, start)
    assert s.update_requests(torch.tensor([1.4]), torch.tensor([Phase.LANDING]), yes, yes).item()
    assert not s.update_requests(torch.tensor([2.]), torch.tensor([Phase.GROUND]), no, no).item()


def test_preflight_cancel_does_not_auto_rearm_from_sustained_initial_request():
    e = env(1, request_protocol=PROTOCOL)
    s = TrainingSchedule(e)
    s.reset(torch.tensor([0]))
    yes = torch.tensor([True])
    ground = torch.tensor([Phase.GROUND])
    s.update_requests(torch.tensor([.8]), ground, yes, yes)
    s.next_request[:] = 0.
    e.random = lambda n: torch.full((n,), .05)
    assert not s.update_requests(torch.tensor([1.]), ground, yes, yes).item()
    assert not s.update_requests(torch.tensor([1.1]), ground, yes, yes).item()


def test_exact_evaluation_bypasses_random_curriculum():
    e = env(16, jump_curriculum=JUMP, command_resampling_seconds=[1., 8.],
            unilateral_degradation_fraction=1., edge_reset_fraction=1.)
    e.cfg["evaluation_exact_cases"] = True
    s = TrainingSchedule(e)
    ids = torch.arange(16)
    velocity = torch.zeros(16, 6)
    s.reset(ids, velocity)
    assert not s.enabled and not s.degraded.any() and not velocity.any()
    assert s.command_duration(16) is None
    torch.testing.assert_close(s.sample_jump(ids, {"kind": "jump", "semantics": {"jump_apex_delta_m": .15}}),
                               torch.full((16,), .15))


def test_exact_subset_keeps_deterministic_request_semantics_without_stress_events():
    e = env(1, request_protocol=PROTOCOL)
    e.cfg["evaluation_exact_cases"] = True
    schedule = TrainingSchedule(e)
    schedule.reset(torch.tensor([0]))
    assert not schedule.enabled and schedule.requests_enabled
    assert not schedule.request_stress.any()
    yes = torch.tensor([True])
    schedule.update_requests(torch.tensor([.8]), torch.tensor([Phase.GROUND]), yes, yes)
    assert schedule.accepted_request_count.item() == 1


def test_same_time_redispatch_and_partial_reset_do_not_repeat_other_lane_events():
    e = env(2)
    e.scene_groups = ["protocol", "protocol"]
    e.cfg["skill_specs"] = {"protocol": {"request_events": [{"time_s": .8, "type": "new"},
                                                          {"time_s": .9, "type": "repeat"}]}}
    s = TrainingSchedule(e)
    ids = torch.arange(2)
    s.reset(ids)
    phase = torch.full((2,), Phase.GROUND)
    yes = torch.ones(2, dtype=torch.bool)
    s.update_requests(torch.full((2,), .8), phase, yes, yes)
    s.update_requests(torch.full((2,), .8), phase, yes, yes)
    assert s.accepted_request_count.tolist() == [1, 1]
    start = s.request_start[1].clone()
    cursor = s.explicit_cursor[1].clone()
    s.reset(torch.tensor([0]))
    assert s.accepted_request_count.tolist() == [0, 1]
    assert torch.equal(s.request_start[1], start) and torch.equal(s.explicit_cursor[1], cursor)
    s.update_requests(torch.tensor([0., .8]), phase, yes, yes)
    assert s.accepted_request_count.tolist() == [0, 1]


def test_skill_sampler_updates_task_semantics_and_keeps_active_jump_target_on_resample():
    e = env(64, jump_curriculum=JUMP, command_resampling_seconds=[1., 8.])
    e.cfg["skill_specs"] = {"jump": {"kind": "jump", "command": [0., 0., .305], "mode": 4,
                                  "semantics": {"jump_apex_delta_m": .03}}}
    e.scene_groups = ["jump"] * e.num_envs
    e.commands = torch.zeros(e.num_envs, 3)
    e.command_target = torch.zeros(e.num_envs, 2)
    for name in ("command_clock", "height_clock", "push_clock", "episode_limits", "episode_length_buf"):
        setattr(e, name, torch.zeros(e.num_envs))
    e.mode = torch.zeros(e.num_envs, dtype=torch.long)
    e.push_enabled = torch.zeros(e.num_envs, dtype=torch.bool)
    e.policy_dt = .02
    e.full_tasks = SimpleNamespace(cfg={"jump_apex_delta_m": .03, "jump_com_rise_m": .015})
    skills = SkillCommands(e)
    ids = torch.arange(e.num_envs)
    skills.sample(ids)
    torch.testing.assert_close(e.full_tasks.cfg["jump_apex_delta_m"], skills.schedule.jump_apex)
    torch.testing.assert_close(e.full_tasks.cfg["jump_com_rise_m"], (skills.schedule.jump_apex - .015).clamp_min(.005))
    assert e.command_clock.min() >= 1 and e.command_clock.max() <= 8
    previous = skills.schedule.jump_apex.clone()
    e.confirmed_jump_bins_m = [.03, .06]
    skills.sample(ids, reset_height=False)
    torch.testing.assert_close(skills.schedule.jump_apex, previous)
    torch.testing.assert_close(e.full_tasks.cfg["jump_apex_delta_m"], previous)


def test_exact_events_wait_for_real_flight_and_complete_two_distinct_jumps():
    events = [{"time_s": .8, "type": "new"}, {"time_s": .9, "type": "repeat"},
              {"time_s": 1., "type": "cancel", "after_phase": "flight"},
              {"time_s": 3., "type": "new", "after_phase": "ground_after_recovery"}]
    e = env(1)
    e.scene_groups = ["protocol"]
    e.cfg.update(evaluation_exact_cases=True, evaluation={"cases": [{"name": "protocol", "skill": {"request_events": events}}]})
    s = TrainingSchedule(e)
    s.reset(torch.tensor([0]))
    assert not s.enabled and s.requests_enabled
    yes = torch.tensor([True])
    def tick(t, phase):
        return s.update_requests(torch.tensor([t]), torch.tensor([phase]), yes, yes)
    assert not tick(.7, Phase.GROUND).item()
    assert tick(.8, Phase.GROUND).item()
    start = s.request_start.clone()
    tick(.9, Phase.TAKEOFF)
    tick(1., Phase.TAKEOFF)
    assert s.explicit_cursor.item() == 2  # cancel is gated by actual flight
    tick(1.2, Phase.FLIGHT)
    assert s.cancel_phase.item() == Phase.FLIGHT and s.request_active.item()
    assert torch.equal(s.request_start, start)
    tick(1.5, Phase.LANDING)
    tick(2., Phase.RECOVERY)
    tick(2.5, Phase.GROUND)
    assert s.completed_jump_count.item() == 1 and not s.request_active.item()
    assert s.pending_explicit_requests().item()
    tick(3., Phase.GROUND)
    assert s.accepted_request_count.item() == 1  # Physical recovery alone is not task acceptance.
    e.successful_jump_count = torch.tensor([1])
    tick(3.02, Phase.GROUND)
    tick(3.4, Phase.FLIGHT)
    tick(3.6, Phase.LANDING)
    tick(3.8, Phase.RECOVERY)
    tick(4., Phase.GROUND)
    assert s.completed_jump_count.item() == 2
    assert s.accepted_request_count.item() == 2 and s.repeat_request_count.item() == 1
    assert s.repeat_clock_reset_errors.item() == 0 and not s.pending_explicit_requests().item()


def test_command_sequence_waits_for_completed_jump_before_resuming_motion():
    e = env(1)
    e.cfg["skill_specs"] = {"sequence": {"kind": "jump", "command": [0., 0., .305], "mode": 4,
        "command_sequence": [{"time_s": 0., "command": [.4, 0., .305], "mode": 1},
                             {"time_s": 2., "command": [0., 0., .305], "mode": 1},
                             {"time_s": 3., "command": [0., 0., .305], "mode": 4},
                             {"time_s": 6., "command": [.4, 0., .305], "mode": 1,
                              "after_phase": "ground_after_recovery"}]}}
    e.scene_groups = ["sequence"]
    e.commands, e.command_target = torch.zeros(1, 3), torch.zeros(1, 2)
    e.mode = torch.zeros(1, dtype=torch.long)
    e.policy_dt, e.episode_length_buf = .02, torch.zeros(1)
    e.phase = SimpleNamespace(phase=torch.tensor([Phase.GROUND]))
    skills = SkillCommands(e)
    for t, expected_vx, mode in [(0., .4, 1), (2., 0., 1), (3., 0., 4), (6., 0., 4)]:
        e.episode_length_buf[:] = t / e.policy_dt
        skills.apply_command_sequence()
        assert e.command_target[0, 0] == pytest.approx(expected_vx) and e.mode.item() == mode
    assert skills.sequence_pending().item()
    skills.schedule.completed_jump_count[:] = 1
    skills.schedule.accepted_request_count[:] = 1
    skills.apply_command_sequence()
    assert e.mode.item() == 4  # Do not erase a jump before the full acceptance hold.
    e.successful_jump_count = torch.tensor([1])
    skills.apply_command_sequence()
    assert e.command_target[0, 0] == pytest.approx(.4) and e.mode.item() == 1
    assert not skills.sequence_pending().item()


def test_completed_request_releases_recovery_without_forcing_the_physical_phase():
    e = env(1)
    e.scene_groups = ["one"]
    e.cfg["skill_specs"] = {"one": {"request_events": [{"time_s": 0., "type": "new"}]}}
    s = TrainingSchedule(e)
    s.reset(torch.tensor([0]))
    phase = PhaseTracker(1, "cpu", .02)
    yes = torch.tensor([True])
    request = s.update_requests(torch.tensor([0.]), phase.phase, yes, yes)
    saw_flight = False
    for tick in range(40):
        contact = torch.full((1, 2), tick == 0 or tick >= 10, dtype=torch.bool)
        phase.update(contact, request, yes, recovery_takeoff=True)
        saw_flight |= phase.phase.item() == Phase.FLIGHT
        if saw_flight:
            assert phase.phase.item() != Phase.TAKEOFF
        request = s.update_requests(torch.tensor([(tick + 1) * .02]), phase.phase, yes, yes)
    assert saw_flight and phase.phase.item() == Phase.GROUND
    assert s.completed_jump_count.item() == 1 and not s.pending_explicit_requests().item()
