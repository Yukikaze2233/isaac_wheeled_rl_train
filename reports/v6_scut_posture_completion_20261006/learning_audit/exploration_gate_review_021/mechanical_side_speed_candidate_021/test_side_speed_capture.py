"""CPU refusal/lifecycle and actual frozen FSM trajectory semantics, no App."""
import ast
from copy import deepcopy
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path('/tmp/scut-recovery-frozen-021-qv1ukmbr')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


m = load(HERE/'side_speed_capture.py', 'side_test_helper')
d = load(HERE/'run_side_speed_021.py', 'side_test_driver')
mechanical = load(HERE.parent/'mechanical_capture_candidate_021/mechanical_capture.py', 'side_test_frozen003')
native = load(ROOT/'src/wheeled_tasks/chassis/recovery.py', 'side_test_native_fsm')


def writer(path, value):
    with path.open('x') as stream: json.dump(value, stream, allow_nan=False)


def fake_override(tmp_path, config=None, step=None):
    fsm = types.SimpleNamespace(config=deepcopy(native.RecoveryFSM.DEFAULTS), DEFAULTS=native.RecoveryFSM.DEFAULTS)
    if config: fsm.config.update(config)
    env = types.SimpleNamespace(recovery=types.SimpleNamespace(fsm=fsm), step=step or (lambda actions: actions))
    override = m.SideOverride(env, mechanical.HookSet(), writer, tmp_path, lambda owner: None)
    return env, override


@pytest.mark.parametrize('value', [1.5, 0., -6., True, float('nan'), float('inf'), '6'])
def test_original_value6_required(tmp_path, value):
    with pytest.raises(ValueError, match='baseline'): fake_override(tmp_path, {m.KEY: value})


@pytest.mark.parametrize('key,value', [('dt', .001), ('max_script_seconds', 10.), ('side_angle_rad', 1.)])
def test_clocks_angle_global_limit_rejected(tmp_path, key, value):
    with pytest.raises(ValueError, match='clocks/angle'): fake_override(tmp_path, {key: value})


def test_instance_owned_dictionary_required(tmp_path):
    fsm = types.SimpleNamespace(config=native.RecoveryFSM.DEFAULTS, DEFAULTS=native.RecoveryFSM.DEFAULTS)
    with pytest.raises(ValueError, match='instance-owned'):
        m.SideOverride(types.SimpleNamespace(recovery=types.SimpleNamespace(fsm=fsm)), mechanical.HookSet(), writer, tmp_path, lambda _: None)


def test_only_one_value_and_no_rng_or_extra_step(tmp_path):
    calls = []
    env, override = fake_override(tmp_path, step=lambda value: calls.append(value) or value)
    prior, before = env.step, deepcopy(env.recovery.fsm.config)
    rng = torch.get_rng_state().clone()
    override.apply()
    assert env.recovery.fsm.config == before | {m.KEY: 1.5}
    assert env.step('native') == 'native' and calls == ['native']
    assert override.calls == 1 and torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match='duplicate'): override.apply()
    assert override.cleanup() == [] and env.step is prior and env.recovery.fsm.config == before
    assert override.cleanup() == []
    receipt = json.loads((tmp_path/'side_override_preclose.json').read_text())
    assert not receipt['complete'] and receipt['config_restored'] and receipt['override_hooks_restored']


def test_complete600_restored_before_parent_finish(tmp_path):
    env, override = fake_override(tmp_path)
    override.apply()
    for tick in range(600): assert env.step(tick) == tick
    assert override.finish() == []
    receipt = json.loads((tmp_path/'side_override_preclose.json').read_text())
    assert receipt['complete'] and receipt['policy_intervals_completed'] == 600 and receipt['config_restored']


def test_foreign_config_change_rejected_not_overwritten(tmp_path):
    env, override = fake_override(tmp_path); override.apply()
    env.recovery.fsm.config['orbit_speed_rad_s'] = 2.
    with pytest.raises(ValueError, match='Only admitted'): env.step(None)
    assert override.calls == 0
    assert override.cleanup() and env.recovery.fsm.config['orbit_speed_rad_s'] == 2.
    assert env.recovery.fsm.config[m.KEY] == 1.5
    assert json.loads((tmp_path/'side_override_preclose.json').read_text())['config_restored'] is False


def test_native_step_exception_restores_partial_lifecycle(tmp_path):
    def failure(_): raise RuntimeError('native failure')
    env, override = fake_override(tmp_path, step=failure); prior = env.step
    override.apply()
    with pytest.raises(RuntimeError, match='native failure'): env.step(None)
    assert override.cleanup() == [] and env.step is prior and env.recovery.fsm.config[m.KEY] == 6.
    assert override.calls == 0 and not override.complete


def test_foreign_hook_owner_rejected_without_clobber(tmp_path):
    env, override = fake_override(tmp_path); override.apply()
    foreign = lambda _: 99; env.step = foreign
    assert override.cleanup() and env.step is foreign


def test_measurement_extension_unique_module_no_frozen_bytes():
    module = load(HERE.parent/'mechanical_capture_candidate_021/mechanical_capture.py', 'side_extension_fixture')
    parent = module.MechanicalCapture
    assert m.expand_capture_module(module)['measurement_only']
    assert module.TRACE_STEPS == 2400 and mechanical.TRACE_STEPS == 260 and issubclass(module.MechanicalCapture, parent)
    with pytest.raises(ValueError, match='identity differs'): m.expand_capture_module(module)


def fsm(speed):
    pose = [0., .2, 0., 0., .2, 0.]
    fold = [1., 1.4, 0., 1., 1.4, 0.]
    return native.RecoveryFSM(2, 'cpu', {m.KEY: speed}, pose, fold, pose, pose, [1., 1., -1., -1.])


def update(owner):
    q = owner.script_targets.clone()
    return owner.update(q, torch.zeros_like(q), torch.tensor([[0., 1., 0.]]).expand(2, 3),
        torch.zeros(2, 3), torch.ones(2)*.1, torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))


def test_actual_frozen_fsm_all_tensors_preSIDE_bitwise_identical():
    a, b = fsm(6.), fsm(1.5)
    for owner in (a, b): owner.reset(torch.arange(2), torch.zeros(2, 6), True)
    rng = torch.get_rng_state().clone()
    for tick in range(400):
        update(a); update(b)
        for key, value in vars(a).items():
            if isinstance(value, torch.Tensor): assert torch.equal(value, getattr(b, key)), (tick, key)
        if int(a.phase[0]) == m.SIDE: break
    else: pytest.fail('Native fixture never reaches SIDE')
    assert torch.equal(rng, torch.get_rng_state())
    update(a); update(b)
    assert not torch.equal(a.script_targets, b.script_targets)


def start_SIDE(owner):
    owner.reset(torch.arange(2), torch.zeros(2, 6), True)
    owner.phase.fill_(m.SIDE); owner.route.fill_(4)
    owner.side_direction[:] = torch.tensor([[-1., -1., 1., 1.]])


def test_native_theta_over_v_hold_and_timeout_unmodified():
    owner = fsm(1.5); start_SIDE(owner)
    captured = {}
    for tick in range(1, 500):
        update(owner)
        if tick in (209, 210, 225, 226, 498, 499): captured[tick] = owner.script_targets.clone()
        if tick == 498: assert int(owner.phase[0]) == m.SIDE
    assert torch.equal(captured[210], captured[225])
    assert not torch.equal(captured[209], captured[210]) and not torch.equal(captured[225], captured[226])
    assert int(owner.phase[0]) == int(native.RecoveryPhase.FAILED) and int(owner.failure_code[0]) == 2
    assert owner.config['max_script_seconds'] == 8. and owner.config['side_angle_rad'] == math.pi/2


def test_native_coupled_reference_preserves_pair_offset():
    owner = fsm(1.5); start_SIDE(owner)
    owner.script_targets[:, (0, 1, 3, 4)] = torch.tensor([[1., 1.2, -2., -1.7]])
    owner.side_start[:] = owner.script_targets[:, (0, 1, 3, 4)]
    initial = owner.side_start[:, (1, 3)]-owner.side_start[:, (0, 2)]
    for _ in range(400):
        update(owner)
        refs = owner.script_targets[:, (0, 1, 3, 4)]
        assert torch.allclose(refs[:, (1, 3)]-refs[:, (0, 2)], initial, atol=1e-6, rtol=0.)


def trace(path, values):
    with gzip.open(path, 'wt') as stream:
        for value in values: stream.write(json.dumps(value)+'\n')


def trace_rows():
    return [dict(physical_step=step, state=dict(lane=lane, phase=5 if step == 3 else 1, full_q=[1., 0.]),
        applied_control=dict(script_targets_control6=[0.]*6) if step else None,
        contacts=dict(matrix=[0.], capacity_indicator=dict(sum_reported_normal_patch_counts=7,
            observed_count_at_capacity=False, max_contact_data_count=100)) if step else None)
        for step in range(4) for lane in m.LANES]


def test_preSIDE_bits_all_fields_and_declared_capacity_exclusion(tmp_path):
    original = trace_rows(); candidate = deepcopy(original)
    for row in candidate:
        if row['applied_control'] is not None: row['applied_control']['side_reference_audit'] = {}
        if row['contacts'] is not None: row['contacts']['capacity_indicator']['sum_reported_normal_patch_counts'] = 9
    a, b = tmp_path/'a.gz', tmp_path/'b.gz'; trace(a, original); trace(b, candidate)
    assert m.pre_side_review(a, b)['status'] == 'preSIDE_equal'
    candidate[-1]['state']['full_q'][1] = -0.
    trace(b, candidate)
    result = m.pre_side_review(a, b)
    assert result['status'] == 'preSIDE_rejected' and result['lanes'][1]['first_mismatch']['reason'].endswith('float bits differ')


@pytest.mark.parametrize('change', ['contact', 'missing', 'duplicate'])
def test_preSIDE_local_contact_missing_duplicate_rejected(tmp_path, change):
    original = trace_rows(); candidate = deepcopy(original)
    if change == 'contact': candidate[-1]['contacts']['matrix'][0] = .01
    if change == 'missing': candidate.pop()
    if change == 'duplicate': candidate.append(candidate[-1])
    a, b = tmp_path/'a.gz', tmp_path/'b.gz'; trace(a, original); trace(b, candidate)
    if change == 'duplicate':
        with pytest.raises(ValueError, match='Duplicate'): m.pre_side_review(a, b)
    else: assert m.pre_side_review(a, b)['status'] == 'preSIDE_rejected'


def coupled_rows():
    return [dict(physical_step=1, state=dict(lane=lane), applied_control=dict(
        script_targets_control6=[1.1, 1.3, 0., -1.9, -1.6, 0.], side_reference_audit=dict(
            native_config_side_speed_rad_s=1.5, phase=5, side_start_rad=[1., 1.2, -2., -1.7],
            side_direction=[1., 1., -1., -1.], axis_signs=[1., 1., -1., -1.]))) for lane in m.LANES]


@pytest.mark.parametrize('change', ['none', 'offset', 'one_side_rate', 'signs', 'direction', 'nonfinite', 'noSIDE'])
def test_coupling_review_refuses_bad_or_missing_evidence(tmp_path, change):
    values = coupled_rows(); audit = values[0]['applied_control']['side_reference_audit']
    if change == 'offset': values[0]['applied_control']['script_targets_control6'][1] += .05
    if change == 'one_side_rate':
        for axis in (3, 4): values[0]['applied_control']['script_targets_control6'][axis] += .05
    if change == 'signs': audit['axis_signs'][1] = -1.
    if change == 'direction': audit['side_direction'][1] = -1.
    if change == 'nonfinite': audit['side_start_rad'][0] = float('nan')
    if change == 'noSIDE': audit['phase'] = 1
    path = tmp_path/'trace.gz'; trace(path, values)
    if change in ('signs', 'direction', 'nonfinite'):
        with pytest.raises(ValueError, match='paired signs'): m.coupled_reference_review(path)
    else:
        assert m.coupled_reference_review(path)['status'] == ('paired_reference_preserved' if change == 'none' else 'paired_reference_rejected')


def test_factory_cold_does_not_import_env_or_isaac():
    tree = ast.parse((HERE/'run_side_speed_021.py').read_text())
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_side_observer')
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(factory))
    class Parent: pass
    class FakeCapture:
        @staticmethod
        def make_capture_observer(parent, *args): return parent
    cls = d.make_side_observer(Parent, FakeCapture(), None, None, None, ROOT, 'pending')
    assert issubclass(cls, Parent)


def test_native_module_authentication_rejects_unregistered_owner():
    owner = types.SimpleNamespace(count=4096)
    with pytest.raises(ValueError, match='constructed/registered'):
        d.authenticate_fsm(owner, ROOT, 'pending', lambda _: 'pending')


@pytest.mark.parametrize('failure', ['partial_parent_init', 'late_parent_finish', 'none'])
def test_composed_lifecycle_preclose_and_late_failure(tmp_path, monkeypatch, failure):
    calls = []
    env, _ = fake_override(tmp_path)
    prior = env.step
    class Parent:
        def __init__(self, e, output):
            calls.append('parent_init')
            if failure == 'partial_parent_init':
                self._before_close()
                raise RuntimeError('partial parent')
        def _before_close(self): calls.append('parent_close')
        def finish(self, output):
            assert env.recovery.fsm.config[m.KEY] == 6. and env.step is prior
            calls.append('parent_finish')
            if failure == 'late_parent_finish': raise RuntimeError('late parent')
            return {}
    fake_capture = types.SimpleNamespace(make_capture_observer=lambda parent, *args: parent,
        sha=lambda path: 'CPU-only')
    monkeypatch.setattr(d, 'authenticate_fsm', lambda *args: None)
    cls = d.make_side_observer(Parent, fake_capture, mechanical, m, writer, ROOT, 'CPU-only')
    if failure == 'partial_parent_init':
        with pytest.raises(RuntimeError, match='partial parent'): cls(env, tmp_path)
        assert env.recovery.fsm.config[m.KEY] == 6. and env.step is prior and calls[-1] == 'parent_close'
        return
    observer = cls(env, tmp_path)
    for tick in range(600): env.step(tick)
    if failure == 'late_parent_finish':
        with pytest.raises(RuntimeError, match='late parent'): observer.finish(tmp_path)
        assert (tmp_path/'side_observer_failure.json').is_file() and calls[-1] == 'parent_close'
    else: assert observer.finish(tmp_path)['isolated_SIDE_override_complete']
    assert env.recovery.fsm.config[m.KEY] == 6. and env.step is prior


def test_source_formula_is_the_original_native_ast():
    text = (ROOT/'src/wheeled_tasks/chassis/recovery.py').read_text()
    assert 'side_duration = self.config["side_angle_rad"] / self.config["side_speed_rad_s"]' in text
    assert 'elapsed - side_duration - .08' in text and 'elapsed >= 2 * side_duration + .4' in text
    assert 'self.age_ticks >= self._ticks(self.config["max_script_seconds"])' in text
