"""Real native CPU samplers and shared component state; no PhysX or PPO credit."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_integrated_lineage import fixture as original_lineage_fixture
from test_integrated_runtime import sources, lineage_fixture, first, terminal, candidate
from test_nonstandard_start_integration import execute, method
from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis import integrated_runtime as runtime
from wheeled_tasks.chassis.nonstandard_start import NonstandardStart, nonstandard_start_settings
from wheeled_tasks.chassis.ordinary_posture_curriculum import OrdinaryPostureCurriculum, posture_settings
from wheeled_tasks.chassis.recovery_curriculum import pose_probabilities
from wheeled_tasks.chassis.recovery_prefix_curriculum import _pose_settings, release_seconds
from wheeled_tasks.chassis.skill_commands import SkillCommands
from wheeled_tasks.chassis.step_assist import JumpAssist, JUMP_ASSIST_V1
from wheeled_tasks.chassis.task import Phase, terrain_surfaces
from wheeled_tasks.chassis.terrain_runout import validate_runout, route_runout_selection, append_route_runout, apply_route_boundary
from wheeled_tasks.chassis.training_schedule import stage_actor_updates

ROOT = Path(__file__).resolve().parents[1]
BANK = json.loads((ROOT / 'contracts/v6_nonstandard_start_profiles_v1.json').read_text())


def environment(config):
    names = list(config['skill_specs'])
    count = len(names)
    clock = config['integrated_runtime_candidate']['clocks_at_source']['domain_actor_updates']
    generator = torch.Generator().manual_seed(1729)
    env = NS(cfg=deepcopy(config), num_envs=count, device='cpu', scene_groups=names, integrated_runtime_enabled=True,
        task_actor_update=clock, global_actor_update=100000, stage_actor_update=clock,
        training_transitions=0, generator=generator, random=lambda n: torch.rand(n, generator=generator),
        policy_dt=.02, commands=torch.zeros(count, 3), command_target=torch.zeros(count, 2),
        mode=torch.zeros(count, dtype=torch.long), push_enabled=torch.zeros(count, dtype=torch.bool),
        phase=NS(phase=torch.zeros(count, dtype=torch.long)), full_tasks=None)
    for name in ('command_clock', 'height_clock', 'push_clock', 'episode_limits', 'episode_length_buf'):
        setattr(env, name, torch.zeros(count))
    return env


def later(f, sources, wanted):
    config = first(f, sources)
    for domain, delta in (('terrain', 5000), ('jump', 2500), ('stepjump', 500)):
        ref, record = terminal(f, config['integrated_runtime_candidate']['clocks_at_source'], delta)
        selection = lineage.select_learning_source(f.binding, domain, f.entry, parent=ref, parent_record=record)
        config = candidate(f, sources, domain, selection)
        if domain == wanted:
            return config
    raise AssertionError(wanted)


def bind_env_methods(env):
    """Execute the exact production helpers without importing the GPU application."""
    import ast
    for name in ('_has_integrated_runtime', '_component_configuration', '_curriculum_clock'):
        node = deepcopy(method(name))
        scope = {'__package__': 'wheeled_tasks.chassis', 'stage_actor_updates': stage_actor_updates}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<native helpers>', 'exec'), scope)
        setattr(env, name, scope[name].__get__(env))


def test_real_skill_sampler_scopes_speed_envelope_and_keeps_recovery_mature(lineage_fixture, sources):
    env = environment(first(lineage_fixture, sources))
    skills = SkillCommands(env)
    skills.sample(torch.arange(env.num_envs))
    probe = skills.verify_difficulty_samples()
    assert probe['sample_check'] == 'passed'
    assert probe['rows']['forward_5']['actor_update'] == 0
    assert probe['rows']['forward_5']['envelope']['vx_max_m_s'] == 2.
    assert runtime.component_actor_updates(env, 'recovery') == 15000
    prefix = env.cfg['recovery_training']['prefix_curriculum']
    assert pose_probabilities(_pose_settings(prefix), runtime.component_actor_updates(env, 'recovery')) == pytest.approx((.2,.3,.5))
    assert release_seconds(prefix, runtime.component_actor_updates(env, 'recovery')) == .2
    env.task_actor_update += 500
    assert runtime.component_actor_updates(env, 'recovery') == 15500
    assert skills.schedule.context('stand_305mm', height_sampling=True)[1] == 15500


def test_mixed_terrain_and_jump_rows_avoid_global_speed_envelope(lineage_fixture, sources):
    config = later(lineage_fixture, sources, 'jump')
    env = environment(config)
    skills = SkillCommands(env)
    skills.sample(torch.arange(env.num_envs))
    probe = skills.verify_difficulty_samples()
    owners = config['integrated_runtime_candidate']['row_ownership']
    assert set(probe['rows']) == {name for name, owner in owners.items() if owner['difficulty_component'] is not None}
    assert any(owner['component'] == 'terrain' and owner['difficulty_component'] is None for owner in owners.values())
    assert any(owner['component'] == 'jump' and owner['difficulty_component'] is None for owner in owners.values())
    assert skills.schedule.context('stand_305mm', height_sampling=True)[1] == 22500
    assert runtime.row_settings(env, 'forward_5', 'locomotion_difficulty')[1] == 7500
    assert skills.schedule.context('forward_5')[1] == 2500
    rows = [name for name, spec in config['skill_specs'].items() if spec['kind'] in ('jump', 'running_jump')
            and not spec.get('request_events') and not spec.get('command_sequence')]
    assert rows
    group = rows[0]
    ids = torch.tensor([env.scene_groups.index(group)])
    settings, clock = skills.schedule.context(group)
    assert clock == 0
    skills.schedule.sample_jump(ids, {**config['skill_specs'][group], 'group_name': group})
    assert any(skills.schedule.jump_apex[ids].item() == pytest.approx(value) for value in settings['jump_curriculum']['bins_m'])


def test_native_posture_helper_binds_mature_rehearsal_and_fresh_component(lineage_fixture, sources):
    env = environment(later(lineage_fixture, sources, 'terrain'))
    bind_env_methods(env)
    owners, settings = env._component_configuration('ordinary_posture_curriculum')
    posture = OrdinaryPostureCurriculum(env.num_envs, 'cpu', posture_settings(0), torch.ones(env.num_envs, dtype=torch.bool))
    posture.configure_components(owners, settings)
    env.ordinary_posture = posture
    clocks = env._curriculum_clock('ordinary_posture_curriculum')
    relative, angles, pushes = posture.amplitudes(clocks)
    anchor = env.scene_groups.index('stand_305mm')
    fresh = next(i for i, owner in enumerate(owners) if owner == 'terrain')
    assert angles[anchor] == pytest.approx(.15) and pushes[anchor] == pytest.approx(.25)
    assert angles[fresh] == pytest.approx(.03) and pushes[fresh] == pytest.approx(.05)
    posture.reset(torch.arange(env.num_envs), clocks, generator=env.generator)
    assert (posture.angles.abs() <= angles[:, None] + 1e-7).all()
    observed = posture.observed(clocks)
    assert 'relative_actor_updates' not in observed
    assert observed['component/recovery/relative_actor_updates'] == 20000 - settings['recovery']['origin_actor_update']
    assert observed['component/terrain/relative_actor_updates'] == 0


def test_shared_nonstandard_bank_samples_component_prior_and_preserves_hold():
    count = 20000
    settings = nonstandard_start_settings(0)
    birth = NonstandardStart(count, 'cpu', .005, settings, ROOT, BANK['tree_joint_names'])
    birth.configure_components(['speed'] * (count//2) + ['terrain'] * (count//2), {'speed': settings, 'terrain': settings})
    clock = {'speed': 5000, 'terrain': 0}
    birth.reset(torch.arange(count), clock, generator=torch.Generator().manual_seed(123))
    assert birth.level_ids[:count//2].eq(2).float().mean() == pytest.approx(.15, abs=.012)
    assert birth.level_ids[count//2:].eq(2).sum() == 0
    assert birth.remaining_steps[birth.selected].eq(40).all()
    birth.begin_policy()
    torque = birth.apply_torque(torch.ones(count, 6))
    assert torque[birth.selected].eq(0).all()
    birth.advance()
    assert birth.remaining_steps[birth.selected].eq(39).all()
    metrics = birth.metrics(clock)
    assert metrics['/nonstandard_start/component/speed/relative_actor_updates'] == 5000
    assert metrics['/nonstandard_start/component/terrain/relative_actor_updates'] == 0


def test_jump_assist_rehearsal_decay_and_new_component_are_vectorized():
    assist = JumpAssist(4, 'cpu', JUMP_ASSIST_V1)
    disabled = {**JUMP_ASSIST_V1, 'enabled': False}
    assist.configure_components(['jump', 'stepjump', 'recovery', None],
        {'jump': JUMP_ASSIST_V1, 'stepjump': JUMP_ASSIST_V1, 'recovery': disabled})
    clocks = {'jump': 500, 'stepjump': 0, 'recovery': 23000}
    device_clock = assist.actor_clock_tensor(clocks)
    assist.reset(torch.arange(4), clocks, torch.zeros(4))
    force = assist.force(torch.full((4,), 20.), torch.full((4,), .1), torch.ones(4), torch.ones(4),
        torch.zeros(4), .005, device_clock)
    assert force[0] == 0 and force[1] > 0 and force[2:].eq(0).all()
    assert assist.selected.tolist() == [False, True, False, False]
    assist.force(torch.full((4,), 20.), torch.full((4,), .1), torch.ones(4), torch.ones(4),
        torch.zeros(4), .005, assist.actor_clock_tensor({**clocks, 'stepjump': 300}))
    assert assist.last_force.eq(0).all()
    assert assist.impulse[1] > 0


def test_authenticated_mixed_runout_keeps_native_rows_and_boundary(lineage_fixture, sources):
    config = later(lineage_fixture, sources, 'jump')
    settings = validate_runout(config)
    assert settings == sources['terrain']['terrain_route_runout']
    selected = []
    for row in config['scene_groups']:
        kind = row['terrain'][0]
        if route_runout_selection(config, row['name'], kind):
            selected.append(row['name'])
            surfaces = terrain_surfaces(kind, config['terrain_limits'], .5, 0)
            extended = append_route_runout(surfaces, settings)
            assert extended[:-1] == surfaces and extended[-1].x1 == 6.2
    assert selected and all(config['integrated_runtime_candidate']['row_ownership'][name]['terrain_runout'] for name in selected)
    assert not route_runout_selection(config, 'recovery_stairs_handoff', 'stairs')
    x = torch.tensor([4., -4., 4.])
    limit = apply_route_boundary(x, torch.full((3,), 3.6), torch.tensor([True, True, False]), settings)
    torch.testing.assert_close(limit, torch.tensor([5.6, 3.6, 3.6]), rtol=0, atol=0)
    corrupted = deepcopy(config)
    corrupted['integrated_runtime_candidate']['source_bindings']['terrain']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='authenticated'):
        validate_runout(corrupted)
    corrupted['evaluation_exact_cases'] = True
    assert validate_runout(corrupted) is None


def test_component_request_reset_preserves_subset_buffer_order(lineage_fixture, sources):
    config = later(lineage_fixture, sources, 'stepjump')
    a, b = environment(config), environment(config)
    first_schedule, second_schedule = SkillCommands(a).schedule, SkillCommands(b).schedule
    assert len(first_schedule.context_batches) <= 5  # Component batches, not hundreds of per-row loops.
    ids = torch.arange(a.num_envs)[::2]
    shuffled = ids.flip(0)
    first_velocity = torch.zeros(len(ids), 6)
    second_velocity = torch.zeros_like(first_velocity)
    first_schedule.reset(ids, first_velocity)
    second_schedule.reset(shuffled, second_velocity)
    torch.testing.assert_close(first_schedule.motor_scale, second_schedule.motor_scale, rtol=0, atol=0)
    torch.testing.assert_close(first_schedule.request_stress, second_schedule.request_stress, rtol=0, atol=0)
    torch.testing.assert_close(first_velocity, second_velocity.flip(0), rtol=0, atol=0)
    for members, _, cfg in first_schedule.context_batches:
        chosen = members[torch.isin(members, ids)]
        if cfg['request_protocol']['stress_episode_fraction'] == 0:
            assert not first_schedule.request_stress[chosen].any()
        lo, hi = cfg['request_protocol']['interval_seconds']
        assert ((first_schedule.next_request[chosen] >= lo) & (first_schedule.next_request[chosen] <= hi)).all()
    eligible = torch.tensor([owner['request_component'] in ('jump', 'stepjump')
        for owner in config['integrated_runtime_candidate']['row_ownership'].values()])
    first_schedule.update_requests(torch.zeros(a.num_envs), torch.full((a.num_envs,), Phase.GROUND),
        eligible, eligible)
    assert first_schedule.accepted_request_count[eligible & ~first_schedule.has_explicit_events].eq(1).all()
    assert first_schedule.accepted_request_count[first_schedule.has_explicit_events].eq(0).all()
    assert first_schedule.accepted_request_count[~eligible].eq(0).all()
    previous = first_schedule.request_start.clone()
    first_schedule.update_requests(torch.ones(a.num_envs) * .1, torch.full((a.num_envs,), Phase.TAKEOFF),
        eligible, eligible)
    torch.testing.assert_close(first_schedule.request_start, previous, rtol=0, atol=0)
