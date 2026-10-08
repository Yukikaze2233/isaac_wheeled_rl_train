"""Only CPU protocol/preparation checks. No Isaac scene is imported or run."""
import importlib.util
from pathlib import Path
from copy import deepcopy
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mean_replay_candidate', ROOT / 'scripts/prepare_rec_speed_mean_replay.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
GOLDEN = ROOT.parent / 'serial-dispatch-source028-20261009'
if GOLDEN.is_dir():
    r.frozen_runtime(GOLDEN)


def diagnostic(done, **changes):
    from wheeled_tasks.chassis.evaluation import RequestProtocolMetrics
    result = {'done': torch.tensor(done), 'recovery_enabled': torch.tensor([True] * len(done))}
    for key in RequestProtocolMetrics.RECOVERY_COUNTERS:
        result.setdefault('recovery_' + key, torch.zeros(len(done)))
    result['recovery_enabled'] = torch.ones(len(done))
    for key, value in changes.items():
        result[key] = torch.tensor(value)
    return result


def owner(pure=False, released=False, failed=False):
    return dict(pure_rl=pure, motion_released=released,
        ownership='failed_disarmed' if failed else 'rl_hold' if pure else 'script')


def test_literal_release_and_slew_are_separate():
    from wheeled_tasks.chassis.playback_command import advance_playback_command
    assert r.requested_command('brake_vx_positive', 4.98) == [.5, 0., .305]
    assert r.requested_command('brake_vx_positive', 5.) == [0., 0., .305]
    applied = advance_playback_command(torch.tensor([[.5, 0., .305]]),
        torch.tensor([r.requested_command('brake_vx_positive', 5.)]), .02,
        dict(vx_m_s2=1.5, yaw_rad_s2=4.))
    assert float(applied[0, 0]) > 0.
    for name in r.STANDS:
        assert r.requested_command(name, 12.)[:2] == [0., 0.]
    for name in r.FALLEN:
        assert r.requested_command(name, 12.) == [0., 0., .305]
    with pytest.raises(ValueError):
        r.requested_command('unknown', 1.)


def test_final_interval_mask_not_instantaneous_pure_credit():
    audit = r.ReplayEvents([r.FALLEN[0]])
    audit.observe(.2, diagnostic([False]), torch.tensor([False]), [owner(True)], [[0., 0.]], [[0., 0.]])
    audit.observe(.22, diagnostic([False]), torch.tensor([True]), [owner(True)], [[0., 0.]], [[0., 0.]])
    row = audit.report()['replicas'][0]
    assert row['first_pure_rl_s'] == pytest.approx(.22)
    assert row['policy_owned_seconds'] == pytest.approx(.02)
    assert row['first_motion_release_s'] is None
    audit.observe(.24, diagnostic([True]), torch.tensor([False]), [owner(False, True, True)], [[0., 0.]], [[0., 0.]])
    assert audit.report()['replicas'][0]['first_motion_release_s'] is None


def test_terminal_frame_once_and_exact_native_ended_denominator():
    # Four replicas of the *same true fallen profile*, not four different L2 lesson labels.
    groups = [r.FALLEN[0]] * 4
    audit = r.ReplayEvents(groups)
    audit.observe(1., diagnostic([True, False, True, False]), torch.tensor([True] * 4),
        [owner(True, True)] * 4, [[0., 0.]] * 4, [[0., 0.]] * 4)
    audit.observe(1.02, diagnostic([True] * 4), torch.tensor([True] * 4),
        [owner(True, True)] * 4, [[0., 0.]] * 4, [[0., 0.]] * 4)
    report = audit.report()
    assert report['groups'][r.FALLEN[0]]['native_recovery_ended_count'] == 4
    assert [x['policy_owned_seconds'] for x in report['replicas']] == pytest.approx([.02, .04, .02, .04])
    assert not any(x['right_censored'] for x in report['replicas'])
    assert report['qualification'] is None


def test_quiet_ten_seconds_after_applied_zero_not_pre_drive_stand():
    audit = r.ReplayEvents(['brake_vx_positive'])
    d = diagnostic([False])
    for tick in range(251):
        time = tick * .02
        req = r.requested_command('brake_vx_positive', time)
        applied = [[.5, 0.]] if time >= 1. else [[0., 0.]]
        audit.observe(time, d, torch.tensor([True]), [owner(True)], [req], applied)
    assert audit.rows[0]['quiet_seconds'] == 0.
    assert audit.rows[0]['first_zero_request_s'] == 5.
    assert audit.rows[0]['first_applied_zero_s'] is None
    for tick in range(500):
        audit.observe(5.34 + tick * .02, d, torch.tensor([True]), [owner(True)], [[0., 0.]], [[0., 0.]])
    assert audit.rows[0]['quiet_seconds'] == pytest.approx(10.)
    assert audit.report()['replicas'][0]['right_censored'] is True
    audit.observe(15.34, d, torch.tensor([True]), [owner(True)], [[0., 0.]], [[.01, 0.]])
    assert audit.rows[0]['quiet_seconds'] == 0.


def test_shared_abi_rejects_interface_change():
    c = {k: None for k in r.ABI_KEYS}
    c.update(physics_dt=.005, policy_dt=.02, actor_dim=35, critic_dim=81, action_dim=6, history_length=1)
    assert r.shared_abi([c, deepcopy(c)])['actor_dim'] == 35
    different = deepcopy(c)
    different['policy_action_order'] = ['changed']
    with pytest.raises(ValueError, match='policy_action_order'):
        r.shared_abi([c, different])


def test_actual_frozen_projection(tmp_path):
    golden = ROOT.parent / 'serial-dispatch-source028-20261009'
    if not golden.is_dir():
        pytest.skip('Exact local e5 runtime checkout is not available')
    root = r.frozen_runtime(golden)
    report = r.prepare_projection(root, tmp_path / 'projection')
    projected = r.read(tmp_path / 'projection/interactive_unscored_contract.json')
    design = r.read(tmp_path / 'projection/interactive_unscored_expectations.json')
    assert len(report['case_names']) == 11
    assert projected['evaluation']['episodes_per_case'] == 4
    assert report['control_protocol']['side_speed_rad_s'] == 1.5
    assert report['control_protocol']['complete_started_blend'] is True
    assert projected['auto_reset'] is False
    assert projected['evaluation_exact_cases'] is True
    assert design['qualification'] is None and design['baseline_score'] is None
    assert list(design['stages']['recovery']['roles']) == ['baseline']
    assert report['source_full_role']['role'] == 'full'


def test_common_controller_native_cpu_identity_and_protocol():
    if not GOLDEN.is_dir():
        pytest.skip('Exact local e5 runtime checkout is not available')
    from wheeled_tasks.chassis.recovery import RecoveryFSM
    source = r.read(GOLDEN / r.REC)
    settings = {key: value for key, value in source['recovery_training'].items() if key in RecoveryFSM.DEFAULTS}
    native = RecoveryFSM(1, 'cpu', settings, *([torch.zeros(6)] * 4), torch.ones(4))
    receipt = r.verify_common_controller(GOLDEN, native)
    assert receipt['side_speed_rad_s'] == 1.5
    assert receipt['complete_started_blend'] is True
    native.config['side_speed_rad_s'] = 6.
    with pytest.raises(ValueError, match='speed or gyro gate'):
        r.verify_common_controller(GOLDEN, native)


def test_real_pair_requires_exact_paid_parent_and_child_lineage():
    parent = dict(checkpoint_sha256=r.PARENT_SHA, original_contract_sha256=r.REC_SHA,
                  checkpoint_infos=dict(task_actor_update=14188))
    child = dict(original_contract_sha256=r.SPEED_SHA,
        checkpoint_infos=dict(task_actor_update=500, successful_updates_total=41903, learning_lineage_updates=28156,
            actor_updates_consumed=41723, critic_warmup_remaining=0),
        integrated_lineage=dict(lineage_id='source028-joint-serial-v1',
            plan_sha256='db693283b744400ed5b6b586055f1514b56aea9106580c40a3bc3ee882d36d92',
            origin_checkpoint_sha256=r.PARENT_SHA, domain='speed', joint_actor_updates=500, domain_actor_updates=500,
            domain_complete=False))
    r.validate_pair([parent, child])
    wrong = deepcopy(child)
    wrong['integrated_lineage']['origin_checkpoint_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='learning origin'):
        r.validate_pair([parent, wrong])
    wrong = deepcopy(child)
    wrong['checkpoint_infos']['successful_updates_total'] -= 10
    with pytest.raises(ValueError, match='10 critic'):
        r.validate_pair([parent, wrong])
    old = deepcopy(parent)
    old['checkpoint_sha256'] = 'old REC41374'
    with pytest.raises(ValueError, match='exact REC41393'):
        r.validate_pair([old, child])


def test_interface_smoke_is_explicit_and_cannot_be_real_pair():
    old = dict(checkpoint_sha256='old', original_contract_sha256=r.REC_SHA,
               checkpoint_infos=dict(task_actor_update=14169))
    r.validate_pair([old, deepcopy(old)], interface_smoke=True)
    with pytest.raises(ValueError, match='exact REC41393'):
        r.validate_pair([old, deepcopy(old)])
