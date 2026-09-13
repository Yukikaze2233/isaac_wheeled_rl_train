"""CPU regression for final-policy gating, reset-phase statistics and report confinement."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'src')]
import common
import evaluate_finished as evaluation
from watch_evaluation import validate_report_manifest


@pytest.fixture
def run(tmp_path, monkeypatch):
    root, audit = tmp_path / 'train', tmp_path / 'audit'
    root.mkdir()
    audit.mkdir()
    now = datetime.now(timezone.utc)
    (audit / 'train.started.json').write_text(json.dumps({'started_at': (now - timedelta(seconds=30)).isoformat()}))
    plan = {'run_dir': str(root), 'audit_dir': str(audit), 'updates': 10000,
            'contract_sha256': 'contract', 'identity': {'seed': 43, 'stage': 'locomotion'}}
    # Isolate gate ordering; the real receipt validator is tested by v40_job tests.
    import wheeled_algo.v40_job as job
    monkeypatch.setattr(job, 'validate_completion', lambda value: value)
    completion = {'status': 'completed', 'export_status': 'verified', 'completed_updates': 10000,
                  'requested_iterations': 10000, 'finished_at': now.isoformat(), 'artifacts': []}
    return root, audit, plan, completion


def test_pending_does_not_accept_periodic_checkpoints(run):
    root, _, plan, _ = run
    (root / 'model_200.pt').write_bytes(b'periodic only')
    assert evaluation.verified_final(plan) is None


@pytest.mark.parametrize('status,export', [('export_failed', 'export_failed'), ('stopped', 'verified')])
def test_partial_or_failed_export_is_blocked(run, status, export):
    root, _, plan, completion = run
    completion.update(status=status, export_status=export)
    (root / 'completion.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError, match='fully verified final export'):
        evaluation.verified_final(plan)


def test_stale_completion_is_rejected(run):
    root, _, plan, completion = run
    completion['finished_at'] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    (root / 'completion.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError, match='stale'):
        evaluation.verified_final(plan)


def test_wrong_final_manifest_rejected_before_policy_execution(run):
    root, _, plan, completion = run
    (root / 'completion.json').write_text(json.dumps(completion))
    (root / 'run_manifest.json').write_text(json.dumps({'seed': 42, 'stage': 'locomotion'}))
    with pytest.raises(ValueError, match='wrong final run metadata'):
        evaluation.verified_final(plan)


def test_csv_reset_phase_and_xy_displacement_do_not_cross_reset(tmp_path):
    # Two full synthetic 3s episodes; world positions jump by 100m at reset.
    rows = []
    for episode in range(2):
        for tick in range(1, 301):
            row = {'step': episode * 300 + tick, 'policy_tick': episode * 300 + tick,
                   'sample_kind': 'pre_reset', 'episode_step': tick, 'episode_time_s': tick / 100,
                   'x': episode * 100 + tick * .001, 'y': 0, 'z': .31 if tick <= 200 else .301,
                   'vx': .01, 'vy': 0, 'non_wheel_net_force_max_n': 16 if tick == 250 else 0,
                   'terminated': False, 'timeout': tick == 300, 'action_cmd_vx': 0, 'action_cmd_wz': 0,
                   'reward_cmd_vx': 0, 'reward_cmd_wz': 0, 'action_cmd_height': .3, 'reward_cmd_height': .3}
            row.update({'diagnostic_' + key: key == 'non_wheel_contact' and tick == 250 for key in evaluation.FLAGS})
            row['termination_non_wheel_contact'] = False
            rows.append(row)
    path = tmp_path / 'telemetry.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'status': 'completed', 'stop_reason': 'step_budget', 'policy_steps': 600,
               'diagnostic_flags_version': 1, 'termination_resets': 0, 'timeout_resets': 2,
               'termination_flags': {'non_wheel_contact': 0},
               'diagnostic_frames': {k: 2 if k == 'non_wheel_contact' else 0 for k in evaluation.FLAGS}}
    result = evaluation.summarize(path, summary, .3, 600)
    assert result['initialization']['frames'] == 400
    assert result['steady']['frames'] == 200
    assert result['steady']['height_mae_m'] == pytest.approx(.001)
    assert result['initialization']['height_p95_m'] == pytest.approx(.01)
    assert result['steady']['non_wheel_net_contact_candidate_frames'] == 2
    assert [e['sampled_xy_displacement_m'] for e in result['episodes']] == pytest.approx([.299, .299])
    assert result['research_criteria_pass'] is None  # Short engineering run cannot claim standing pass.
    summary['diagnostic_frames']['non_wheel_contact'] = 0
    with pytest.raises(ValueError, match='counts differ'):
        evaluation.summarize(path, summary, .3, 600)


def test_report_allowlist_rejects_metadata_and_checkpoint_injection():
    files = [{'path': name, 'size': 1, 'sha256': 'a' * 64}
             for name in ('evaluation_result.json', 'evaluation_report.md')]
    manifest = {'schema_version': 1, 'evaluation_id': 'eval01', 'training_plan_sha256': 'plan', 'files': files}
    validate_report_manifest(manifest, 'plan', 'eval01')
    with pytest.raises(ValueError, match='identity'):
        validate_report_manifest(manifest, 'wrong', 'eval01')
    for name in ('../secret', 'model_final.pt', 'h030/usd_cache/robot.usd'):
        with pytest.raises(ValueError, match='artifact'):
            validate_report_manifest({**manifest, 'files': files + [{'path': name, 'size': 1, 'sha256': 'a' * 64}]}, 'plan', 'eval01')


def test_waiter_expiration_publishes_reason_without_launching_sim(tmp_path, monkeypatch):
    repo = HERE.parents[1]
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'repo': str(repo), 'profile': 'train', 'updates': 10000,
                                'run_dir': str(tmp_path / 'train'), 'git_commit': 'a' * 40}))
    output = tmp_path / 'evaluation'
    monkeypatch.setattr(sys, 'argv', ['evaluate_finished.py', '--repo', str(repo), '--training-plan', str(plan),
                                    '--output', str(output), '--wait-seconds', '1', '--poll-seconds', '1'])
    clock = iter([0., 0., .5, 2.])
    monkeypatch.setattr(evaluation.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(evaluation.time, 'sleep', lambda _: None)
    monkeypatch.setattr(evaluation, 'training_alive', lambda _: False)
    monkeypatch.setattr(evaluation, 'verified_final', lambda _: None)
    monkeypatch.setattr(evaluation.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch before completion'))
    assert evaluation.main() == 2
    result = json.loads((output / 'evaluation_result.json').read_text())
    assert result['state'] == 'expired_waiting_completion'
    assert result['cases'] == [] and result['research_criteria_all_heights_pass'] is None
    assert (output / 'report_manifest.json').exists()


def test_offline_pull_watcher_has_retryable_state(tmp_path, monkeypatch):
    import watch_evaluation as watcher
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    output = tmp_path / 'pull'
    monkeypatch.setattr(sys, 'argv', ['watch_evaluation.py', '--host', 'kaiser@host',
        '--control-path', '/tmp/missing-master', '--remote-output', '/remote/eval',
        '--training-plan', str(plan), '--destination', str(output), '--once'])

    def unavailable(*args):
        raise common.Unavailable('existing master disconnected')

    monkeypatch.setattr(watcher, 'pull', unavailable)
    assert watcher.main() == 3
    state = json.loads((output / 'watch-state.json').read_text())
    assert state['state'] == 'transport_unavailable_retryable'
    assert 'no password/key changes' in state['action']
