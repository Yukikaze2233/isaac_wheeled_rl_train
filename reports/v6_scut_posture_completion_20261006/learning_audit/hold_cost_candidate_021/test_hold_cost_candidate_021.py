"""Isolated CPU semantic boundaries against frozen020; no App, GPU or PPO."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch
torch.set_num_threads(1)

HERE = Path(__file__).parent
BINDINGS = json.loads((HERE / 'source_bindings.json').read_text())
BASE = Path(BINDINGS['base_source_root'])
sys.path.insert(0, str(BASE / 'src'))
from wheeled_tasks.chassis import rewards as original
from wheeled_tasks.chassis.recovery import RecoveryFSM, RecoveryPhase
from wheeled_tasks.chassis.recovery_training import RecoveryTraining
from wheeled_tasks.chassis.mechanism_observer import ClosedChainHeightModel

REWARD_REL = 'src/wheeled_tasks/chassis/rewards.py'
ENV_REL = 'src/wheeled_tasks/chassis/env.py'
spec = importlib.util.spec_from_file_location('wheeled_tasks.chassis.hold_candidate_rewards', HERE / 'candidate' / REWARD_REL)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
fsm_spec = importlib.util.spec_from_file_location('wheeled_tasks.chassis.hold_candidate_fsm', HERE / 'candidate/src/wheeled_tasks/chassis/recovery.py')
candidate_fsm = importlib.util.module_from_spec(fsm_spec)
fsm_spec.loader.exec_module(candidate_fsm)


def settings(enabled=True):
    result = deepcopy(candidate.RECOVERY_HOLD_COST_V1)
    result['enabled'] = enabled
    return result


def native_terms(dq, omega):
    n = len(dq)
    commands = torch.zeros(n, 3); commands[:, 2] = .305
    zeros = torch.zeros(n, 6)
    terms = original.reward_terms(torch.zeros(n, 3), omega,
        torch.tensor([[0., 0., -1.]] * n), commands[:, 2], commands,
        dq, zeros, zeros, zeros, zeros, zeros,
        torch.zeros(n, 2, 3), torch.ones(n), torch.ones(n), torch.zeros(n, dtype=torch.bool))
    broad, fine = original.dense_error_terms(omega[:, 2], original.DENSE_TRACKING_V1['yaw'])
    terms.update(dense_yaw_broad=broad, dense_yaw_precision=fine)
    terms.update(original.episode_event_terms(torch.arange(n) == 1, torch.arange(n) == 2, {}))
    return terms


def payment(terms):
    density = [value for name, value in terms.items() if not name.endswith('_event')]
    return torch.stack(density).sum(0) * .02 + terms['success_event'] + terms['failure_event']


def apply(terms, dq, omega, enabled=None, owned=None, held=None, cfg=None, **kwargs):
    ones = torch.ones(len(dq), dtype=torch.bool)
    return candidate.apply_recovery_hold_costs(terms, dq, omega,
        ones if enabled is None else enabled, ones if owned is None else owned,
        ones if held is None else held, settings() if cfg is None else cfg, **kwargs)


def test_source020_identity_and_semantic_sources_unchanged():
    for rel, digest in BINDINGS['original_source_sha256'].items():
        assert hashlib.sha256((BASE / rel).read_bytes()).hexdigest() == digest
    for rel in ('recovery_training.py', 'v5_control.py'):
        assert not (HERE / 'candidate/src/wheeled_tasks/chassis' / rel).exists()
    assert BINDINGS['production_changed_files'] == []
    assert BINDINGS['PPO_updates'] == BINDINGS['GPU_native_runs'] == 0


def gate_rig():
    q = torch.zeros(6)
    fsm = candidate_fsm.RecoveryFSM(16, 'cpu', {}, q, q, q, q, torch.ones(4))
    geometry = ClosedChainHeightModel(json.loads((BASE / 'contracts/v6_height_lookup_v1.json').read_text()), 'cpu')
    return fsm, geometry


def test_startup_binding_uses_true_native_gate_config_and_actual_two_side_geometry():
    fsm, geometry = gate_rig()
    expected = {'status':'authenticated_native_fsm_and_geometry',
        'source':'RecoveryFSM.constants+native_fsm_config+ClosedChainHeightModel.wheel_radius_m',
        'wheel_radius_m':.06,'wheel_rim_gate_m_s':.25,'leg_gate_rad_s':2.,'gyro_gate_rad_s':.75}
    assert candidate.validate_recovery_hold_gates(settings(), fsm, geometry) == expected
    # Float32 radius is not exactly Pythonfloat .06, but same-device/dtype
    # comparison against the authored scalar must not falsely reject it.
    assert geometry.wheel_radius_m.dtype == torch.float32
    assert float(geometry.wheel_radius_m[0]) != .06
    assert candidate.validate_recovery_hold_gates(settings(False), None, None) == {'status':'disabled'}


@pytest.mark.parametrize('field', ['gyro', 'wheel_radius', 'wheel_gate', 'leg_gate', 'asset_left', 'asset_right', 'asset_shape'])
def test_modified_native_fsm_gate_or_actual_geometry_refuses_startup(field):
    fsm, geometry = gate_rig()
    if field == 'gyro': fsm.config['handover_gyro_max'] = .751
    elif field == 'wheel_radius': fsm.WHEEL_RADIUS_M = .061
    elif field == 'wheel_gate': fsm.WHEEL_RIM_MAX_M_S = .251
    elif field == 'leg_gate': fsm.LEG_SPEED_MAX_RAD_S = 2.1
    elif field == 'asset_left': geometry.wheel_radius_m[0] = .061
    elif field == 'asset_right': geometry.wheel_radius_m[1] = .061
    else: geometry.wheel_radius_m = geometry.wheel_radius_m[:1]
    with pytest.raises(ValueError): candidate.validate_recovery_hold_gates(settings(), fsm, geometry)


def test_class_constant_extraction_preserves_native_fsm_gate_and_clock_outputs_bitwise():
    n = 64; q = torch.zeros(n, 6); zero = q[0]; signs = torch.ones(4)
    old = RecoveryFSM(n, 'cpu', {}, zero, zero, zero, zero, signs)
    new = candidate_fsm.RecoveryFSM(n, 'cpu', {}, zero, zero, zero, zero, signs)
    for fsm in (old, new):
        fsm.reset(torch.arange(n), q, torch.ones(n, dtype=torch.bool))
        fsm.phase[:] = RecoveryPhase.RL
    g = torch.tensor([[0.,0.,-1.]] * n); height = torch.full((n,), .305)
    all_true = torch.ones(n, dtype=torch.bool)
    random = torch.Generator().manual_seed(216)
    for tick in range(240):
        dq = torch.randn(n, 6, generator=random) * 3
        gyro = torch.randn(n, 3, generator=random) * .4
        # Genuine exact limits on both sides of strict < boundaries.
        dq[:8] = 0; gyro[:8] = 0
        dq[0, 2] = .25/.06; dq[1, 5] = -.25/.06
        dq[2, 0] = 2.; dq[3, 4] = -2.
        gyro[4, 0] = .75; gyro[5, 2] = -.75
        for fsm in (old, new): fsm.update(q, dq, g, gyro, height, all_true, all_true)
        for key, value in old.diagnostics().items():
            assert torch.equal(value, new.diagnostics()[key]), (tick, key)
        assert torch.equal(old.motion_released, new.motion_released)
        assert torch.equal(old.stable_ticks, new.stable_ticks)


@pytest.mark.parametrize('enabled', [True, False])
def test_complete_explicit_schema_accepts_training_and_disabled_eval(enabled):
    value = settings(enabled)
    assert candidate.validate_recovery_hold_cost(value) is value


@pytest.mark.parametrize('edit', [
    lambda v: v.update(enabled=1), lambda v: v.update(enabled='false'),
    lambda v: v.update(extra=True), lambda v: v.pop('gyro'),
    lambda v: v.update(wheel_radius_m=.061), lambda v: v.update(leg_gate_rad_s=2),
    lambda v: v.update(version='other'), lambda v: v['leg'].update(extra=1),
    lambda v: v['leg'].update(broad_scale=.5), lambda v: v['wheel'].update(fine_scale=1.1),
    lambda v: v['gyro'].update(fine_weight=0), lambda v: v['gyro'].update(fine_weight=True),
    lambda v: v['gyro'].update(broad_weight=float('nan')),
    lambda v: v['wheel'].update(fine_scale=float('inf')),
])
def test_strict_schema_rejects_drift_or_invalid_values(edit):
    value = settings(); edit(value)
    with pytest.raises(ValueError): candidate.validate_recovery_hold_cost(value)


def test_disabled_feature_returns_exact_original_dict_tensor_objects_and_payment():
    dq = torch.randn(17, 6); omega = torch.randn(17, 3)
    terms = native_terms(dq, omega); before = dict(terms); old_payment = payment(terms)
    diagnostics = {'stale': 1}
    assert apply(terms, dq, omega, cfg=settings(False), diagnostics=diagnostics) is terms
    assert diagnostics == {} and list(terms) == list(before)
    assert all(terms[k] is before[k] for k in terms)
    assert torch.equal(payment(terms), old_payment)


@pytest.mark.parametrize('mask_name', ['enabled', 'owned', 'held'])
def test_nonhold_ordinary_excluded_or_released_values_and_reduction_bitwise(mask_name):
    generator = torch.Generator().manual_seed(211)
    dq = torch.randn(4096, 6, generator=generator); omega = torch.randn(4096, 3, generator=generator)
    terms = native_terms(dq, omega); before = dict(terms); old_payment = payment(terms)
    apply(terms, dq, omega, **{mask_name: torch.zeros(4096, dtype=torch.bool)})
    assert list(terms) == list(before)
    for key in before: assert torch.equal(terms[key], before[key]), key
    assert torch.equal(payment(terms), old_payment)


def test_mixed_eligibility_changes_only_five_existing_motion_values_preserves_events():
    dq = torch.full((8, 6), 3.); omega = torch.full((8, 3), .8)
    terms = native_terms(dq, omega); before = dict(terms)
    enabled = torch.tensor([1, 1, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
    owned = torch.tensor([1, 0, 1, 0, 1, 1, 1, 1], dtype=torch.bool)
    held = torch.tensor([1, 1, 0, 0, 1, 1, 0, 1], dtype=torch.bool)
    mask = enabled & owned & held
    apply(terms, dq, omega, enabled=enabled, owned=owned, held=held)
    changed = {'leg_velocity', 'wheel_velocity', 'horizontal_angular_velocity', 'dense_yaw_broad', 'dense_yaw_precision'}
    assert list(terms) == list(before)
    for key in before:
        assert torch.equal(terms[key][~mask], before[key][~mask]), key
        if key not in changed: assert terms[key] is before[key], key
    assert terms['failure_event'][2] == -1. and terms['success_event'][1] == 5.
    assert terms['track_lin_vel'][mask].eq(1).all()
    assert terms['track_height'][mask].eq(1).all()


def test_gate_relative_scales_max_axis_sign_symmetry_and_no_cancellation():
    dq = torch.tensor([[2., 0., .25 / .06, 0., 0., 0.],
                       [2., -2., .25 / .06, -2., 2., -.25 / .06],
                       [-2., 2., -.25 / .06, 2., -2., .25 / .06]])
    omega = torch.tensor([[.75, 0., 0.], [0., .75, 0.], [0., 0., -.75]])
    terms = native_terms(dq, omega); diagnostics = {}
    apply(terms, dq, omega, diagnostics=diagnostics)
    for axis in ('leg', 'wheel', 'gyro'):
        torch.testing.assert_close(diagnostics[axis + '/gate_relative_error'], torch.ones(3), atol=1e-7, rtol=1e-7)
        expected = original.dense_error_terms(diagnostics[axis + '/gate_relative_error'], settings()[axis])
        assert torch.equal(diagnostics[axis + '/broad_density'], expected[0])
        assert torch.equal(diagnostics[axis + '/fine_density'], expected[1])
    assert terms['dense_yaw_broad'].eq(0).all() and terms['dense_yaw_precision'].eq(0).all()


def test_zero_motion_zero_cost_broad_tail_still_distinguishes_large_motion():
    magnitude = torch.tensor([0., .1, .5, 1., 2., 10., 100.])
    dq = magnitude[:, None].expand(-1, 6); omega = magnitude[:, None].expand(-1, 3)
    terms = native_terms(dq, omega); apply(terms, dq, omega)
    for name in ('leg_velocity', 'wheel_velocity', 'horizontal_angular_velocity'):
        assert terms[name][0] == 0 and torch.isfinite(terms[name]).all()
        assert (torch.diff(terms[name]) < 0).all()


def test_full_gyro_quiet_cost_never_weakens_existing_pure_yaw_density():
    yaw = torch.linspace(0., 6., 1001)
    dq = torch.zeros(len(yaw), 6); omega = torch.zeros(len(yaw), 3); omega[:, 2] = yaw
    terms = native_terms(dq, omega)
    old = terms['horizontal_angular_velocity'] + terms['dense_yaw_broad'] + terms['dense_yaw_precision']
    apply(terms, dq, omega)
    new = terms['horizontal_angular_velocity'] + terms['dense_yaw_broad'] + terms['dense_yaw_precision']
    assert torch.all(new <= old + 1e-7)


def test_mask_before_norm_excludes_nan_lane_without_hiding_active_nan():
    dq = torch.ones(3, 6); omega = torch.ones(3, 3)
    dq[1:] = float('nan'); omega[1:] = float('nan')
    terms = {k: torch.zeros(3) for k in ('leg_velocity', 'wheel_velocity', 'horizontal_angular_velocity', 'dense_yaw_broad', 'dense_yaw_precision')}
    diagnostics = {}
    apply(terms, dq, omega, owned=torch.tensor([1, 0, 1], dtype=torch.bool), diagnostics=diagnostics)
    for axis in ('leg', 'wheel', 'gyro'):
        assert diagnostics[axis + '/gate_relative_error'][1] == 0
        assert torch.isnan(diagnostics[axis + '/gate_relative_error'][2])
    assert terms['leg_velocity'][1] == 0 and torch.isnan(terms['leg_velocity'][2])


def ownership_rig():
    n = 6; q = torch.zeros(n, 6)
    r = RecoveryTraining.__new__(RecoveryTraining)
    r.settings = {'complete_started_blend': True}
    r.enabled = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    r.release_finished = torch.ones(n, dtype=torch.bool)
    r.premature_release = torch.zeros(n, dtype=torch.bool)
    r.elapsed = torch.zeros(n); r.wheel_probe_torque = torch.zeros(n, 2)
    r.learning_mask = torch.zeros(n, dtype=torch.bool); r.policy_motion_hold = torch.zeros(n, dtype=torch.bool)
    r.fsm = RecoveryFSM(n, 'cpu', {'complete_started_blend': True}, q[0], q[0], q[0], q[0], torch.ones(4))
    r.fsm.reset(torch.arange(n), q, r.enabled)
    r.fsm.phase[:5] = torch.tensor([RecoveryPhase.RL, RecoveryPhase.BLEND, RecoveryPhase.RL, RecoveryPhase.RL, RecoveryPhase.RL])
    r.fsm.blend[:] = 1.
    r.env = NS(dt=.005, v5=NS(LEGS=(0,1,3,4), WHEELS=(2,5),
        settings={'leg_kp':160., 'leg_kd':2.5}, wheel_prior={'kd':.6},
        current_motor_bounds=torch.full((6,),20.), requested_motor_effort=q.clone()))
    r.project = lambda x: x
    return r, q


def test_actual_native_interval_mask_keeps_terminal_actor_and_excludes_blend_or_release_tick():
    r, q = ownership_rig(); r.begin_policy()
    # Lane0: starts actor-owned then fails. Lane1: starts BLEND then RL.
    # Lane2: one release-disabled tick then pureRL. Lane3: real motion releases
    # within this interval. Lane4: never held. Ordinary lane5 must stay old.
    r.fsm.motion_released[4] = True
    r.policy_motion_hold[4] = False
    r.fsm.phase[0] = RecoveryPhase.FAILED
    r.fsm.phase[1] = RecoveryPhase.RL
    r.release_finished[2] = False
    r.fsm.motion_released[3] = True
    r.torque(q, q, q)
    r.release_finished[2] = True; r.torque(q, q, q)
    assert r.learning_mask.tolist() == [True, False, False, True, True, True]
    dq = torch.ones(6, 6); omega = torch.ones(6, 3)
    terms = native_terms(dq, omega); before = dict(terms); diagnostics = {}
    apply(terms, dq, omega, enabled=r.enabled, owned=r.learning_mask, held=r.policy_motion_hold, diagnostics=diagnostics)
    assert diagnostics['mask'].tolist() == [True, False, False, True, False, False]
    assert terms['failure_event'][2] == before['failure_event'][2]
    for key in before: assert torch.equal(terms[key][[1,2,4,5]], before[key][[1,2,4,5]])
    # Next interval resumes ordinary commanded tracking for lane3.
    r.begin_policy()
    assert not r.policy_motion_hold[3]
    later = native_terms(dq, omega); later_before = dict(later)
    apply(later, dq, omega, enabled=r.enabled, owned=r.learning_mask, held=r.policy_motion_hold)
    for key in later: assert torch.equal(later[key][3], later_before[key][3])


@pytest.mark.parametrize('bad', ['dq', 'omega', 'mask_dtype', 'catalog'])
def test_invalid_tensor_contract_fails_loudly(bad):
    dq = torch.ones(3, 6); omega = torch.ones(3, 3); terms = native_terms(dq, omega)
    kwargs = {}
    if bad == 'dq': dq = dq[:, :5]
    elif bad == 'omega': omega = omega[:, :2]
    elif bad == 'mask_dtype': kwargs['owned'] = torch.ones(3)
    else: terms.pop('dense_yaw_broad')
    with pytest.raises(ValueError): apply(terms, dq, omega, **kwargs)


def test_env_hook_lifecycle_logs_pre_reset_and_no_learning_or_release_mutation():
    tree = ast.parse((HERE / 'candidate' / ENV_REL).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'apply_recovery_hold_costs']
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args][-4:] == ['self.recovery.enabled', 'self.recovery.learning_mask', 'self.recovery.policy_motion_hold', 'self.recovery_hold_cost_settings']
    assert {k.arg for k in calls[0].keywords} == {'diagnostics'}
    methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    step = ast.unparse(methods['step'])
    assert step.index('self._recovery_metrics(') < step.index('self.reset(')
    metrics = ast.unparse(methods['_recovery_metrics'])
    assert "'/HoldCost/" in metrics and 'recovery_hold_cost_diagnostics' in metrics
    assert 'values[\'mask\']' in metrics
    helper = ast.unparse(ast.parse((HERE / 'candidate' / REWARD_REL).read_text()))
    assert 'recovery_takeover_' not in ast.unparse(next(n for n in ast.parse(helper).body if isinstance(n, ast.FunctionDef) and n.name == 'apply_recovery_hold_costs'))
    assert 'learning_mask.copy_' not in metrics and 'motion_released' not in metrics
