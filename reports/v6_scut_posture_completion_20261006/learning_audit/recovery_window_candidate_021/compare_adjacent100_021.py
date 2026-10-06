#!/usr/bin/env python3
"""Compare authenticated adjacent native actor windows without more training."""
import argparse
import hashlib
import json
from pathlib import Path

COMMIT = 'd0dce6181df83b03fe166abcbadef3c81de2e57e'
READER = '5faa75e24b0dd484bdacb0273756642caab8fcc6d6e66f90c034a0bda6ac19a6'
METRICS = ('planar_speed_mean_m_s', 'vx_mae_m_s', 'yaw_mae_rad_s',
           'height_mae_m', 'vx_actual_mean_m_s', 'vx_command_mean_m_s')
COUNTS = ('frames', 'episodes', 'successes', 'failures', 'timeouts',
          'boundary_truncations', 'blocked_truncations')


def load(path):
    raw = path.read_bytes()
    r = json.loads(raw)
    assert r['status'] == 'requested_actor_windows_complete' and r['complete']
    assert r['source_commit'] == COMMIT and r['reader_sha256'] == READER
    assert r['qualification'] is None and r['PPO_updates_by_reader'] == 0
    w = r['windows']['100']
    assert w['complete'] and w['window']['actor_updates'] == 100
    assert w['window']['critic_warmup_excluded']
    assert all(x['complete'] and x['records'] == 100 for x in w['timing'].values())
    return r, w, hashlib.sha256(raw).hexdigest()


def delta(a, b):
    return dict(first100=a, second100=b, change=None if a is None or b is None else b-a)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--second', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    r1, a, h1 = load(args.first)
    r2, b, h2 = load(args.second)
    assert r1['identity']['event'] == r2['identity']['event']
    assert r1['identity']['source_files_sha256'] == r2['identity']['source_files_sha256']
    assert a['window']['task_actor_updates_inclusive'] == [3086, 3185]
    assert b['window']['task_actor_updates_inclusive'] == [3186, 3285]
    assert a['window']['event_steps_inclusive'] == [30016, 30115]
    assert b['window']['event_steps_inclusive'] == [30116, 30215]
    assert b['window']['baseline_step'] == a['window']['end_step']
    groups = {}
    assert a['behavior']['base_groups'] == b['behavior']['base_groups']
    for name in a['behavior']['base_groups']:
        x, y = (w['behavior']['groups'][name] for w in (a, b))
        groups[name] = dict(metrics={k: delta(x[k], y[k]) for k in METRICS},
                            counts={k: delta(x[k], y[k]) for k in COUNTS})
    stop = {}
    for name in a['optional_conditional_stop_phases']['groups']:
        x, y = (w['optional_conditional_stop_phases']['groups'][name]['settled_stop'] for w in (a, b))
        assert x['complete'] and y['complete']
        stop[name] = {k: delta(x[k], y[k]) for k in
                      ('samples', 'mean_speed_m_s', 'above_3cm_s_fraction', 'peak_speed_m_s')}
    recovery = {}
    for k in COUNTS:
        recovery[k] = delta(a['behavior']['pools']['scripted_recovery'][k],
                            b['behavior']['pools']['scripted_recovery'][k])
    teaching = {}
    for section in ('hold_cost', 'hold_gates'):
        teaching[section] = {}
        for tag, x in a['teaching'][section].items():
            y = b['teaching'][section][tag]
            assert x['complete'] and y['complete'] and x['records'] == y['records'] == 100
            teaching[section][tag] = delta(x['mean'], y['mean'])
    report = dict(schema='source021-adjacent100-descriptive-comparison-v1',
        status='authenticated_disjoint_windows', source_commit=COMMIT,
        input_sha256={str(args.first): h1, str(args.second): h2},
        inspector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        windows=[a['window'], b['window']], base_groups=groups, conditional_stop=stop,
        exact_mixed_recovery_counts=recovery, teaching=teaching,
        timing={tag: delta(x['mean'], b['timing'][tag]['mean']) for tag, x in a['timing'].items()},
        qualification=None, PPO_updates_by_comparison=0,
        limits=['Same-run exploratory learning windows, not same-seal causal A/B or deployment validation.',
                '54 recovery groups mix dynamic lessons; ended-episode success is not the final original-chain rate.',
                'Support/body-clear are FK/IMU observer decisions, true contacts are scored separately.',
                'Instantaneous marginal gates do not imply the continuous one-second joint gate.',
                'Conditional stopping counts and means exclude movement frames; not whole-path drift.',
                'Wall time is an actual shared-host observation, not source-only performance attribution.'])
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    assert hashlib.sha256(args.first.read_bytes()).hexdigest() == h1
    assert hashlib.sha256(args.second.read_bytes()).hexdigest() == h2
    print(json.dumps(dict(status=report['status'], output=str(args.output),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
