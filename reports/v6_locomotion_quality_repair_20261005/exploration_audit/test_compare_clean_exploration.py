"""Synthetic CPU reader tests; these fixtures are not native physics evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).with_name('compare_clean_exploration.py')
SPEC = importlib.util.spec_from_file_location('clean_exploration_comparison', PATH)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from locomotion_exploration_sampling import FirstCohortLedger

NAMES = tuple(case for case in reader.CASES for _ in range(4))


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def actions(arm):
    for tick in range(1, 1001):
        mu, sample = [[0.] * 6] * 12, [[.1] * 6] * 12
        yield {'policy_tick': tick, 'phase': ['stand'] * 12,
            'requested_command': [[0., 0., .305]] * 12, 'visible_command': [[0., 0., .305]] * 12,
            'rng_before_sha256': str(tick - 1), 'rng_after_sha256': str(tick),
            'draw_counter': tick, 'sample_element_count': tick * 72,
            'mu': mu, 'sample': sample, 'sent_policy_action': mu if arm == 'mean' else sample}


def frames(arm, *, ticks=range(1, 4001)):
    for tick in ticks:
        for lane, case in enumerate(NAMES):
            live = arm == 'mean' or lane != 0 or tick <= 8
            requested = [0.] * 6
            if lane == 0 and arm == 'mean' and tick > 8:
                requested[0] = 2.
            if lane == 0 and arm == 'stochastic' and tick <= 8:
                requested[1] = 2.
            yield {'lane': lane, 'case': case, 'physics_tick': tick, 'first_cohort': live,
                'phase': 'stand' if case == 'stand_305mm' else 'steady',
                'requested_motor_effort': requested, 'current_motor_bounds': [1.] * 6,
                'root_link_velocity_body_m_s': [.1, 0., 0.],
                'root_link_velocity_world_m_s': [.1, 0., 0.],
                'root_angular_velocity_body_rad_s': [0., 0., .03],
                'command': [0., 0., .305], 'height_m': .305,
                'pitch_deg': 2., 'roll_deg': 3., 'root_position_local_m': [.1, 0., .305],
                'phase_anchor_xy_m': [0., 0.]}


@pytest.fixture
def pair(tmp_path, monkeypatch):
    paths = {arm: tmp_path / arm for arm in ('mean', 'stochastic')}
    for arm, path in paths.items():
        path.mkdir()
        ledger = FirstCohortLedger(NAMES)
        failed = [arm == 'stochastic' and lane == 0 for lane in range(12)]
        ledger.observe(done=failed, terminated=failed, timeouts=[False] * 12,
                       success=[False] * 12, reasons={'boundary': [False] * 12}, physical_tick=8)
        ledger.observe(done=[True] * 12, terminated=failed, timeouts=[not x for x in failed],
                       success=[False] * 12, reasons={'boundary': [False] * 12}, physical_tick=4000)
        cohort = ledger.summary(physical_tick=4000)
        report = {'arm': arm, 'status': 'native_probe_completed', 'execution_status': 'completed',
            'policy_steps': 1000, 'physics_steps': 4000, 'formal_updates_consumed': 0,
            'actor_tensors_unchanged': True, 'source_identity_final_verified': True,
            'sealed_inputs_final_verified': True, 'source_revision': 'a' * 40,
            'source_authentication': {'synthetic': True},
            'input_authentication': {'files': {'checkpoint': {'sha256': 'b' * 64}, 'contract': {'sha256': 'c' * 64}}},
            'sampling_seed': 93001, 'native_policy_action_order': ['p'] * 6,
            'native_control_order': ['c'] * 6, 'draw_counter': 1000, 'sample_element_count': 72000,
            'first_cohort': cohort}
        write(path / 'first_cohort.json', {**cohort, 'lane_phase_metrics': []})
        write(path / 'birth.json', {'synthetic': True})
        write(path / 'clean_design_expectations.json', {'synthetic': True})
        design_sha = reader.sha(path / 'clean_design_expectations.json')
        write(path / 'clean_effective_contract.json', {'design_preflight': {
            'file': str(path / 'clean_design_expectations.json'), 'sha256': design_sha}, 'physics_dt': .005})
        raw_sha = reader.sha(path / 'clean_effective_contract.json')
        report['fixture'] = {'effective_contract_sha256': raw_sha, 'source_contract_sha256': 'c' * 64,
                             'design_sha256': design_sha}
        write(path / 'clean_fixture_receipt.json', report['fixture'])
        audit = {'contract_path': str(path / 'clean_effective_contract.json'), 'effective_contract_sha256': raw_sha,
                 'status': 'passed', 'effective_contract_sha_matches': True, 'runtime_config_differences': [],
                 'allowed_constructor_overrides': {'stage': 'speed', 'num_envs': 12, 'seed': 190619, 'terrain_level': 1.}}
        write(path / 'runtime_config_validation.json', audit)
        write(path / 'runtime.json', {'scene_groups': NAMES, 'birth': {'synthetic': True}, 'runtime_config_validation': audit})
        write(path / 'report.json', report)
        for name in ('actions.jsonl', 'physical_frames.jsonl'):
            (path / name).write_text('synthetic records supplied by CPU fixture\n')

    def records(path):
        arm = Path(path).parent.name
        return actions(arm) if Path(path).name == 'actions.jsonl' else frames(arm)
    monkeypatch.setattr(reader, 'records', records)
    return paths


def test_common_saturation_excludes_longer_survivor_tail(pair):
    result = reader.compare(pair['mean'], pair['stochastic'])
    assert result['common_exposure_physics_ticks_per_lane'] == [8] + [4000] * 11
    own = result['first_cohort_torque_saturation']['mean']['stand_305mm']
    shared = result['common_exposure_torque_saturation']
    assert own['count'] == 16000 and own['hit_counts'] == [3992, 0, 0, 0, 0, 0]
    assert shared['mean']['stand_305mm']['count'] == shared['stochastic']['stand_305mm']['count'] == 12008
    assert shared['mean']['stand_305mm']['hit_counts'] == [0] * 6
    assert shared['stochastic']['stand_305mm']['hit_counts'] == [0, 8, 0, 0, 0, 0]
    assert shared['stochastic']['stand_305mm']['saturation_fraction_control_order'][1] == pytest.approx(8 / 12008)
    assert result['qualification'] is None and result['formal_updates_consumed'] == 0
    bindings = result['authenticated_arm_local_bindings']
    assert bindings['mean']['raw_effective_contract_sha256'] != bindings['stochastic']['raw_effective_contract_sha256']
    assert bindings['mean']['canonical_effective_contract_sha256'] == bindings['stochastic']['canonical_effective_contract_sha256']


@pytest.mark.parametrize('change,match', [
    (lambda audit: audit.update(contract_path='/other/clean_effective_contract.json'), 'contract path'),
    (lambda audit: audit.update(effective_contract_sha256='0' * 64), 'contract SHA'),
    (lambda audit: audit['allowed_constructor_overrides'].update(terrain_level=.5), 'config validation differ'),
    (lambda audit: audit['runtime_config_differences'].append({'path': 'physics_dt'}), 'config validation differ'),
])
def test_runtime_output_normalization_preserves_sha_and_all_other_fields(pair, change, match):
    path = pair['stochastic']
    runtime = reader.read(path / 'runtime.json')
    change(runtime['runtime_config_validation'])
    write(path / 'runtime.json', runtime)
    write(path / 'runtime_config_validation.json', runtime['runtime_config_validation'])
    with pytest.raises(ValueError, match=match):
        reader.compare(pair['mean'], path)


@pytest.mark.parametrize('change,match', [
    (lambda row: row.update(case='forward_0p5'), 'lane/case/replica'),
    (lambda row: row['terminal'].update(physical_tick=12), 'first terminal'),
    (lambda row: row.update(partition='timeout'), 'partition'),
])
def test_native_first_terminal_metadata_cannot_be_relabelled(pair, change, match):
    path = pair['stochastic']
    cohort = reader.read(path / 'first_cohort.json')
    change(cohort['rows'][0])
    write(path / 'first_cohort.json', cohort)
    report = reader.read(path / 'report.json')
    report['first_cohort'] = {k: v for k, v in cohort.items() if k != 'lane_phase_metrics'}
    write(path / 'report.json', report)
    with pytest.raises(ValueError, match=match):
        reader.compare(pair['mean'], path)


def test_export_marks_empty_case_without_zero_filling_values(tmp_path, monkeypatch):
    import torch.utils.tensorboard
    scalars, texts = [], []

    class Writer:
        def __init__(self, path, **kwargs):
            Path(path).mkdir()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def add_scalar(self, name, value, step):
            scalars.append((name, value, step))
        def add_text(self, name, value, step):
            texts.append(value)

    monkeypatch.setattr(torch.utils.tensorboard, 'SummaryWriter', Writer)
    rows = list(frames('stochastic', ticks=(4, 8, 12)))
    # The native first-terminal frame at tick 8 remains eligible; tick 12 has
    # no live stand replicas in this synthetic event-export edge case.
    for row in rows:
        if row['physics_tick'] == 12 and row['case'] == 'stand_305mm':
            row['first_cohort'] = False
    monkeypatch.setattr(reader, 'records', lambda path: iter(deepcopy(rows)))
    exported = reader.export_events(tmp_path, 'stochastic')
    prefix = 'ExplorationProbe/clean/stochastic/stand_305mm/'
    assert (prefix + 'first_cohort_live_count', 0, 12) in scalars
    assert not any(name == prefix + 'planar_speed_m_s' and step == 12 for name, _, step in scalars)
    assert (prefix + 'first_cohort_live_count', 4, 8) in scalars
    assert exported['sampling_hz'] == 50 and exported['server_started'] is False
    assert any('Self-survivor instantaneous 50Hz sampled' in text for text in texts)
