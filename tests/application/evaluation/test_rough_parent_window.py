"""Unoptimized parent inference keeps actual inputs and terrain measurements."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from tests.tasks.curricula.test_fixed_rehearsal_behavior import sample
from wheeled_tasks.chassis.application.evaluation.small_rough_evaluation import RoughBirthWindowMetrics, configuration
from wheeled_tasks.chassis.application.configuration.small_rough_course import MOTION_VERSIONS
from wheeled_tasks.chassis.application.configuration.training_profile import validate_profile_selection


ROOT = Path(__file__).resolve().parents[3]


def entry():
    spec = importlib.util.spec_from_file_location('rough_parent_window', ROOT / 'src/wheeled_tasks/chassis/application/training/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate():
    return json.loads((ROOT / 'configs/v7_formal_20261012/motion_retention_review.json').read_text())


def test_inference_surface_drops_training_critic_initialization_without_disabling_signals():
    c = candidate()
    measured = configuration(c)
    assert 'formal_critic_adaptation' not in measured
    assert 'formal_small_rough_mix' not in measured
    assert measured['policy_signal_delay'] == c['policy_signal_delay']
    assert measured['actor_sensor_noise'] == c['actor_sensor_noise']
    assert measured['research_training_authorized'] is False
    validate_profile_selection(measured)


def test_final_motion_inference_owns_its_surface_without_rehearsal_selector():
    from copy import deepcopy
    from wheeled_tasks.chassis.application.evaluation.small_rough_evaluation import retained_flat_configuration

    parent = json.loads((ROOT / 'configs/v7_formal_20261012/final_motion_review.json').read_bytes())
    original = deepcopy(parent)
    plane = retained_flat_configuration(parent,
        [dict(name='flat_measurement', terrain=['flat'], fraction=1.)], scripted=False)
    rough = configuration(parent)
    assert parent == original
    for effective in (plane, rough):
        assert 'formal_flat_rehearsal_collider' not in effective
        assert 'formal_small_rough_mix' not in effective
        assert 'physx_friction_offset_threshold_m' not in effective
        assert effective['actor_sensor_noise'] == parent['actor_sensor_noise']
        assert effective['policy_signal_delay'] == parent['policy_signal_delay']
        assert effective['formal_contact_generation_distance'] == parent['formal_contact_generation_distance']
        assert effective['asset_manifest_sha256'] == parent['asset_manifest_sha256']
        assert effective['v5_control'] == parent['v5_control']
        assert validate_profile_selection(effective)
    assert plane['formal_terrain']['type'] == 'plane'
    assert rough['formal_terrain']['type'] == 'rough'


def test_birth_tile_skills_are_not_credited_for_visiting_other_rows_or_families():
    rough = {'definition': {'rows': 10, 'columns': 14, 'keys': ('family_a', 'family_b')},
             'column_key_ids': torch.tensor([0, 0] + [1] * 12), 'start_xy': torch.zeros(2)}
    env = NS(origins=torch.tensor([[4.5, 4.5, 0.], [4.5, 13.5, 0.], [0., 300., 0.]]),
        scene_groups=['source_rough', 'source_rough', 'fixed_flat'], device='cpu', policy_dt=.02,
        flat_episode=NS(commands=NS(_rough=rough)))
    metrics = RoughBirthWindowMetrics(env)
    data = sample({'case_index': [-1] * 3, 'case_requests': []})
    data['measurement_valid_mask'] = torch.tensor([True, False, True])
    metrics.observe(data)
    data['episode_ticks'] += 1
    data['position'][0, 0] = 9.
    data['measurement_valid_mask'] = torch.tensor([True, True, True])
    metrics.observe(data)
    data['position'][0, 1] = 18.
    data['measurement_valid_mask'][1] = False
    metrics.observe(data)
    report = metrics.report()
    assert report['measurements']['groups']['row0/family_a']['frames'] == 2
    assert report['current_row_family_eligible_frames'][0] == [2, 0]
    assert report['current_row_family_eligible_frames'][1] == [1, 1]
    assert report['ppo_credit'] == 0 and report['qualification'] is False
    data['position'][0, 0] = 90.
    data['done'][0] = data['terminated'][0] = True
    metrics.observe(data)
    report = metrics.report()
    assert report['outside_grid_eligible_frames'] == 1
    assert report['measurements']['groups']['row0/family_a']['failures'] == 1
    assert 'including outcomes after leaving it' in report['episode_outcome_scope']


def test_shared_input_recorder_marks_mean_without_calling_the_sampling_path(tmp_path):
    module = entry()
    calls = []
    mean, std = torch.full((2, 6), .1), torch.full((2, 6), .2)
    alg = NS(transition=NS(distribution_params=(mean, std)))
    alg.act = lambda obs: calls.append('sample') or mean + .05
    env = NS(physical_tick=0, tick=0,
        flat_episode=NS(episode_id=torch.zeros(2, dtype=torch.long),
            commands=NS(episode_physics_ticks=torch.zeros(2, dtype=torch.long))))
    observations = {'policy': torch.zeros(2, 35), 'critic': torch.zeros(2, 78)}
    module.bind_sampled_policy_inputs(alg, env, tmp_path)
    alg.act(observations)
    stored = torch.load(tmp_path / 'policy_first_action.pt', weights_only=False)
    assert stored['actor_mode'] == 'sampled' and torch.equal(stored['raw_sampled_action'], mean + .05)
    env._record_formal_actor_input(observations, mean, mean, std, actor_mode='mean')
    stored = torch.load(tmp_path / 'policy_first_action.pt', weights_only=False)
    assert stored['actor_mode'] == 'mean' and 'raw_sampled_action' not in stored
    assert torch.equal(stored['raw_policy_action'], mean) and calls == ['sample']
    observations['policy'].fill_(42.)
    assert not env._formal_actor_input['policy_input35'].any()


def test_new_course_mean_probe_is_explicitly_unpaired_and_still_not_training(tmp_path):
    tree = ast.parse((ROOT / 'src/wheeled_tasks/chassis/application/training/train.py').read_text())
    guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
        and any(isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
            and isinstance(child.value.func, ast.Name) and child.value.func.id == 'bind_sampled_policy_inputs'
            for child in n.body))
    calls = []
    scope = dict(c=candidate(), args=NS(capture_numerical_failure=True, run_dir=tmp_path),
        MOTION_VERSIONS=MOTION_VERSIONS,
        report={'rng_restore': {'mode': 'fresh_fork', 'parent_rng_restored': False,
            'selection': {'phase': 'new_course_entry'}}},
        sampled_policy_validation=False, restored_policy_validation=True,
        runner=NS(alg=object()), env=object(), bind_sampled_policy_inputs=lambda *a: calls.append(True))
    exec(compile(ast.Module(body=[guard], type_ignores=[]), '<actual-mean-capture-admission>', 'exec'), scope)
    assert calls == [True]
    assert scope['report']['physical_capture_rng_scope'] == 'fresh_motion_course_not_paired_parent_replay'
    assert candidate()['research_training_authorized'] is False
