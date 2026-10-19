"""Physical interval ownership and timeout inputs; CPU doubles are not recovery qualification."""
from copy import deepcopy
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict

from support_scut_flat_scripted_recovery import rig, tick, ready
from wheeled_tasks.chassis.flat_episode_selection import (
    RECOVERY_WINDOW_KEY, RECOVERY_WINDOW_SETTINGS, validate_recovery_episode_window,
)
from wheeled_tasks.chassis.recovery import RecoveryPhase
from wheeled_tasks.chassis.recovery_training import RecoveryTraining


ROOT = Path(__file__).resolve().parents[1]


def extended(env):
    env.cfg[RECOVERY_WINDOW_KEY] = deepcopy(RECOVERY_WINDOW_SETTINGS)
    env.recovery = RecoveryTraining(env, ROOT, env.cfg['recovery_training'])
    reset_excerpt = env.reset

    def reset(ids):
        # The shared AST fixture starts after this real reset counter clear.
        env.episode_length_buf[ids] = 0
        reset_excerpt(ids)
    env.reset = reset
    env.flat_episode.episode_id.fill_(-1)  # Match the newly constructed recovery producer.
    env.reset(torch.arange(env.num_envs))
    env.episode_length_buf.zero_()

    def observations(*, critic_only=False):
        critic = torch.zeros(env.num_envs, 78)
        critic[:, 0] = env.robot.data.root_link_pose_w.torch[:, 2]
        critic[:, 1] = env.commands[:, 0]
        critic[:, 2] = env.flat_episode.episode_id
        if critic_only:
            return critic
        return TensorDict({'policy': torch.zeros(env.num_envs, 35), 'critic': critic},
            batch_size=[env.num_envs])
    env.get_observations = observations
    return env.recovery


@pytest.mark.parametrize('field,value', [
    ('prelude_seconds', 20.), ('post_handoff_seconds', 19.98),
    ('post_handoff_seconds', True), ('start', 'first_rl_substep'),
    ('timeout_value', 'reset_observation'), ('unknown', 1),
])
def test_window_selection_is_explicit_and_does_not_weaken_old_contracts(rig, field, value):
    assert validate_recovery_episode_window(rig.cfg) is None
    c = deepcopy(rig.cfg)
    c[RECOVERY_WINDOW_KEY] = dict(RECOVERY_WINDOW_SETTINGS, **{field: value})
    with pytest.raises(ValueError, match='Recovery episode window'):
        validate_recovery_episode_window(c)


def test_script_time_does_not_consume_the_full_actor_window_or_change_ordinary_limits(rig):
    e = rig
    recovery = extended(e)
    for _ in range(10):
        _, _, done, _ = tick(e)
        assert not done.any()
    ready(e)
    old_id = int(e.flat_episode.episode_id[0])
    for interval in range(11, 1011):
        obs, _, done, info = tick(e)
        if interval == 999:
            assert not done.any()  # New window counts 1000 complete intervals.
        if interval == 1000:
            assert done.tolist() == [False, True]
            assert info['time_outs'].tolist() == [False, True]
        if interval < 1010:
            assert not done[0]
    data = info['diagnostics']
    assert done.tolist() == [True, False]
    assert not data['terminated'].any() and info['time_outs'][0]
    assert data['recovery_first_whole_actor_start_s'][0] == pytest.approx(.2)
    assert data['recovery_post_handoff_elapsed_s'][0] == pytest.approx(20.)
    assert data['recovery_post_handoff_eligible_s'][0] == pytest.approx(20.)
    assert data['recovery_post_handoff_full_window'][0]
    assert data['episode_ticks'][0] == 1010
    assert info['policy_learning_mask'][0] and not info['policy_bootstrap_mask'][0]
    assert info['policy_terminal_critic_observation'][0, 2] == old_id
    assert obs['critic'][0, 2] == old_id + 1
    assert recovery.window_first_actor_tick[0] == -1
    assert recovery.window_policy_ticks[0] == recovery.window_eligible_ticks[0] == 0


def test_no_handoff_at_eight_seconds_is_script_failure_not_a_timeout(rig):
    e = rig
    extended(e)
    for _ in range(399):
        _, _, done, _ = tick(e)
        assert not done[0]
    _, _, done, info = tick(e)
    assert done[0] and info['diagnostics']['reasons']['recovery_script'][0]
    assert not info['time_outs'][0] and not info['policy_learning_mask'][0]
    assert 'policy_terminal_critic_observation' not in info


def test_mixed_handoff_interval_cannot_start_the_window(rig):
    e = rig
    r = extended(e)
    r.begin_policy()  # Script owns the interval at its beginning.
    r.release_finished[0] = True
    r.fsm.phase[0] = RecoveryPhase.RL
    e.tick += 1
    r.advance_episode_window(r.learning_mask)
    assert r.window_first_actor_tick[0] == -1
    r.begin_policy()
    e.tick += 1
    r.advance_episode_window(r.learning_mask)
    assert r.window_first_actor_tick[0] == 1 and r.window_eligible_ticks[0] == 1
    with pytest.raises(RuntimeError, match='twice'):
        r.advance_episode_window(r.learning_mask)


def test_contact_failure_on_the_horizon_keeps_terminal_precedence(rig):
    e = rig
    r = extended(e)
    ready(e)
    r.window_policy_ticks[0] = r.window_eligible_ticks[0] = 999
    r.window_first_actor_tick[0] = 0
    e.scut_reward.terminal_counter[0] = 19
    e.contact_force[0, 0, 2] = 10.
    e.scut_reward.net_contact[0, 0, 2] = 10.
    e.scut_reward.finish_contacts()
    _, _, done, info = tick(e)
    assert done[0] and info['diagnostics']['terminated'][0]
    assert not info['time_outs'][0]
    assert 'policy_terminal_critic_observation' not in info


def test_nominal_inference_projection_keeps_window_only_for_scripted_cases():
    import json
    from wheeled_tasks.chassis.small_rough_evaluation import configuration, retained_flat_configuration
    c = json.loads((ROOT / 'configs/v7_formal_20261012/motion_retention_review.json').read_text())
    assert RECOVERY_WINDOW_KEY not in configuration(c)
    scenes = [{'name': 'ordinary', 'terrain': ['flat'], 'fraction': 1.}]
    flat = retained_flat_configuration(c, scenes, scripted=False)
    assert RECOVERY_WINDOW_KEY not in flat and 'formal_critic_adaptation' not in flat
    from wheeled_tasks.chassis.contact_generation_distance import selected
    assert selected(flat)


def test_motion_sampling_and_loss_exposure_share_the_new_exact_ordinary_horizon():
    import json
    from wheeled_tasks.chassis.small_rough_course import allocate, MixedCourseExposure
    c = json.loads((ROOT / 'configs/v7_formal_20261012/motion_retention_review.json').read_text())
    groups = [scene['name'] for scene in c['scene_groups']
        for _ in range(round(scene['fraction'] * 4096))]
    spec = allocate(c, ROOT, 4096, groups)
    assert spec['episode_policy_ticks'] == 1000
    exposure = MixedCourseExposure(spec, 24, 'cpu')
    assert exposure.episode_policy_ticks == 1000
    old = exposure.state_dict()
    old.pop('episode_policy_ticks')
    with pytest.raises(ValueError, match='inconsistent counts'):
        exposure.load_state_dict(old)


def test_terminal_critic_read_does_not_seed_actor_noise_or_history_caches():
    from support_flat_episode_env_wiring import method
    body = method('get_observations')
    scope = {'torch': torch, 'TensorDict': TensorDict, '__package__': 'wheeled_tasks.chassis'}
    exec(compile(ast.Module(body=[body], type_ignores=[]), '<actual-observation-reader>', 'exec'), scope)
    zero3, zero6 = torch.zeros(2, 3), torch.zeros(2, 6)
    gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    expected = torch.arange(156).reshape(2, 78).float()

    def forbidden(*a, **kw):
        raise AssertionError('A terminal critic read must not consume actor noise/history')
    e = NS(state=lambda: (zero3, zero3, gravity, torch.full((2,), .305), zero6, zero6, None),
        scut35=True, critic78=True, scut_reward=object(), references=None, skills=None,
        jump_requested=torch.zeros(2, dtype=torch.bool), mode=torch.zeros(2, dtype=torch.long),
        targets=torch.zeros(2, 4), commands=torch.tensor([[.5, 0., .305]]).repeat(2, 1),
        actions=zero6, v5=NS(nominal=torch.zeros(6)), cfg={'actor_frame_dim': 35, 'critic_dim': 78},
        _critic78_frame=lambda frame: expected,
        actor_sensor_noise=NS(apply=forbidden), history=NS(update=forbidden))
    actual = scope['get_observations'](e, critic_only=True)
    assert torch.equal(actual, expected)
    assert not hasattr(e, '_actor_sensor_noise_clean_frame')
