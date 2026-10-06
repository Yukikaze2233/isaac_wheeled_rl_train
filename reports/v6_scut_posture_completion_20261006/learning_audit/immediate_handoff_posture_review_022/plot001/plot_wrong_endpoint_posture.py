#!/usr/bin/env python3
"""Offline first-birth gravity/gyro plot; no simulation, SSH, GPU, model or PPO."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
REVIEW_PATH = HERE.parent / 'posture_review_read001.json'
REVIEW_SHA = '8bee1e8f1623865e6a1183be7bcb1483b8034b72eb4296a07048e8eb096800fb'
LANES = (1708, 2216)
COLORS = {'source022_baseline': '#126A8A', 'immediate_v1': '#C94C35'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert sha(REVIEW_PATH) == REVIEW_SHA
review = json.loads(REVIEW_PATH.read_text())
assert review['qualification'] is None
INPUTS = {str(REVIEW_PATH): REVIEW_SHA}
TRACES = {}
FACTS = {}
for arm in COLORS:
    authority = review['arms'][arm]
    root = Path(authority['root'])
    for name in ('identity.json', 'mechanical_5ms.jsonl.gz'):
        path = root / name
        expected = authority['input_sha256'][name]
        assert sha(path) == expected
        INPUTS[str(path)] = expected
    identity = json.loads((root / 'identity.json').read_text())
    assert identity['checkpoint_sha256'] == '286b9e26cdd7fe8726f9deb1ce5f24e005d5d918d3abc2b122fdb85b6b671c31'
    assert identity['source_commit_label'] == '723732adae37f12f8488f9537717664415a98d10'
    assert identity['seed'] == 290927 and authority['inference_arm'] == 'mean'
    series = {lane: [] for lane in LANES}
    with gzip.open(root / 'mechanical_5ms.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            state = row['state']
            lane = state['lane']
            if lane not in LANES or state['birth_id'] != 1:
                continue
            gravity = np.asarray(state['gravity_body'], dtype=float)
            gyro = np.asarray(state['gyro_body'], dtype=float)
            assert np.isfinite(gravity).all() and np.isfinite(gyro).all()
            assert np.linalg.norm(gravity) > 0
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2] / np.linalg.norm(gravity), -1, 1))))
            series[lane].append(dict(physical_step=row['physical_step'], simulated_s=row['simulated_s'], tilt_deg=tilt,
                gyro_norm_rad_s=float(np.linalg.norm(gyro)), phase=state['phase'], pure_rl=state['pure_rl'],
                release_finished=state['release_finished']))
    for lane in LANES:
        rows = series[lane]
        assert [row['physical_step'] for row in rows] == list(range(rows[-1]['physical_step'] + 1))
        assert all(row['simulated_s'] == row['physical_step'] * 0.005 for row in rows)
        actual_handoff = next(row for row in rows if row['release_finished'] and row['pure_rl'])
        audit = authority['lanes'][str(lane)]
        assert actual_handoff['physical_step'] == audit['first_real_actor_handoff']['physical_step']
        assert rows[-1]['physical_step'] == audit['last_observed_physical_step']
        assert abs(rows[-1]['tilt_deg'] - audit['last_observed_truth']['tilt_deg']) < 1e-10
        assert abs(rows[-1]['gyro_norm_rad_s'] - audit['last_observed_truth']['gyro_norm_rad_s']) < 1e-10
        TRACES[(arm, lane)] = rows
        FACTS.setdefault(arm, {})[str(lane)] = dict(handoff=actual_handoff,
            last_first_birth_state=rows[-1], outcome=audit['outcome'], first_birth_records=len(rows))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 11, 'axes.titleweight': 'bold', 'text.color': '#243543',
    'axes.labelcolor': '#344553', 'axes.edgecolor': '#B5C1CA', 'xtick.color': '#526575',
    'ytick.color': '#526575', 'axes.facecolor': '#FCFDFE', 'figure.facecolor': 'white', 'svg.fonttype': 'none'})
fig, axes = plt.subplots(2, 2, figsize=(16, 10.8))
fig.subplots_adjust(left=0.075, right=0.965, top=0.83, bottom=0.205, wspace=0.21, hspace=0.40)
fig.suptitle('Native001 selected the initial PREPARE endpoint too early', x=0.075, y=0.971, ha='left', fontsize=21, fontweight='bold')
fig.text(0.075, 0.93, 'Negative diagnostic result  |  baseline vs native001  |  historical REC3076 mean policy, source022, seed 290927', fontsize=12.5)
fig.text(0.075, 0.895, 'Raw 5 ms postphysics truth, birth 1 only. Total tilt comes from normalized body gravity; gyro is the body-vector norm.', fontsize=11)
fig.text(0.075, 0.868, 'Insets enlarge 0–0.33 s. Handoff requires release_finished AND pure_rl; the initial phase-9 passive placeholder is excluded.', fontsize=11, color='#526575')

titles = {1708: 'Lane 1708: step200_wheels_split', 2216: 'Lane 2216: double_step_first_riser_split'}
for row_index, lane in enumerate(LANES):
    for column, metric in enumerate(('tilt_deg', 'gyro_norm_rad_s')):
        axis = axes[row_index, column]
        axis.set_title(titles[lane] + ('  |  total tilt' if column == 0 else '  |  gyro magnitude'), loc='left', pad=12)
        axis.set_ylabel('Total body inclination (°)' if column == 0 else 'Body gyro norm (rad/s)')
        axis.set_xlabel('Simulation time since first birth (s)')
        axis.set_xlim(-0.03, 12.12)
        axis.set_ylim(0, 185 if column == 0 else 25)
        inset = axis.inset_axes([0.515, 0.455, 0.46, 0.46])
        inset.set_title('Early endpoint: 0–0.33 s', fontsize=10, fontweight='normal', pad=4)
        inset.set_xlim(0, 0.33)
        inset.set_ylim(0, 65 if column == 0 else 15)
        inset.tick_params(labelsize=9)
        inset.set_xticks([0, 0.1, 0.2, 0.3])
        inset.axvspan(0, 0.035, color='#8C98A2', alpha=0.20, lw=0)
        for arm in COLORS:
            rows = TRACES[(arm, lane)]
            x = np.asarray([point['simulated_s'] for point in rows])
            y = np.asarray([point[metric] for point in rows])
            color = COLORS[arm]
            style = '-' if arm == 'source022_baseline' else '--'
            axis.plot(x, y, color=color, ls=style, lw=1.45, zorder=3)
            mask = x <= 0.33
            inset.plot(x[mask], y[mask], color=color, ls=style, lw=1.8, zorder=3)
            handoff = FACTS[arm][str(lane)]['handoff']
            axis.axvline(handoff['simulated_s'], color=color, ls=':', lw=1.2, zorder=2)
            axis.scatter(handoff['simulated_s'], handoff[metric], color=color, marker='v', s=38, zorder=5)
            if arm == 'immediate_v1':
                inset.axvline(handoff['simulated_s'], color=color, ls=':', lw=1.2)
                inset.scatter(handoff['simulated_s'], handoff[metric], color=color, marker='v', s=27, zorder=5)
                inset.scatter(x[-1], y[-1], color=color, marker='x', s=50, zorder=5)
                axis.scatter(x[-1], y[-1], color=color, marker='x', s=50, zorder=5)
        handoff = FACTS['source022_baseline'][str(lane)]['handoff']
        axis.text(handoff['simulated_s'] + 0.13, (171 if column == 0 else 23),
                  f"Baseline real RL\n{handoff['simulated_s']:.3f} s", color=COLORS['source022_baseline'], fontsize=10, va='top')
        inset.text(0.035, 0.97, '001 RL: 0.265 s\nterminal: 0.300 s', transform=inset.transAxes,
                   fontsize=9, va='top', color=COLORS['immediate_v1'])
        for target in (axis, inset):
            target.grid(color='#DAE2E8', lw=0.6, alpha=0.6)
            target.spines[['top', 'right']].set_visible(False)
        if column == 1:
            axis.axhline(0.75, color='#9AA6AE', ls=(0, (4, 4)), lw=0.8)

legend = [Line2D([0], [0], color=COLORS['source022_baseline'], lw=1.8, label='Baseline: full script + blend'),
          Line2D([0], [0], color=COLORS['immediate_v1'], ls='--', lw=1.8, label='001: initial PREPARE handoff; first birth ends'),
          Line2D([0], [0], color='#526575', marker='v', ls=':', label='True script-to-RL handoff'),
          Line2D([0], [0], color=COLORS['immediate_v1'], marker='x', ls='none', label='First-birth terminal')]
fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(0.072, 0.135), ncol=2, frameon=False, fontsize=11, columnspacing=2)
fig.text(0.075, 0.102, '001 hands over at 0.265 s before FOLD / SIDE / ORBIT / THRUST / CAPTURE, then terminates at 0.300 s.', fontsize=10.5)
fig.text(0.075, 0.077, 'Baseline enters real RL at 4.885 / 4.920 s; both births are unfinished at the 12 s observation limit, with no motion release.', fontsize=10.5)
fig.text(0.075, 0.052, 'Gray inset band: passive phase-9 placeholder (0–0.035 s). Curves stop at first-birth termination; later resets are not joined.', fontsize=10.5)
fig.text(0.075, 0.027, 'Descriptive comparison: early global001 interventions can change resets/RNG. Completed-action immediate handoff was not tested; no qualification.', fontsize=10.5, color='#526575')

outputs = [HERE / 'wrong_initial_prepare_endpoint_posture.png', HERE / 'wrong_initial_prepare_endpoint_posture.svg']
assert not any(path.exists() for path in outputs)
fig.savefig(outputs[0], dpi=175)
fig.savefig(outputs[1])
plt.close(fig)
assert all(sha(path) == value for path, value in INPUTS.items())
receipt = dict(schema='source022-native001-wrong-endpoint-posture-plot-v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
    status='generated_pending_visual_review', inputs_sha256=INPUTS, input_after_sha256={path: sha(path) for path in INPUTS},
    inputs_unchanged=True, script_sha256=sha(__file__), outputs_sha256={str(path): sha(path) for path in outputs},
    formulas=dict(tilt='degrees(acos(clip(-gravity_body_z/norm(gravity_body),-1,1)))', gyro='norm(gyro_body) rad/s'),
    sample_scope='Raw5ms postphysics state, selected lane1708/2216, birth_id1 only; no smoothing or reward inversion',
    handoff_detector='state.release_finished AND state.pure_rl; exclude initial passive phase9',
    same_historical_model_sha256='286b9e26cdd7fe8726f9deb1ce5f24e005d5d918d3abc2b122fdb85b6b671c31',
    facts=FACTS, endpoint_error=review['interpretation']['confirmed_endpoint_semantics_error'],
    limitations=review['interpretation'], SSH_actions=0, GPU_actions=0, PPO_updates=0, qualification=None)
with (HERE / 'plot_receipt.json').open('x') as output:
    json.dump(receipt, output, indent=2, allow_nan=False)
    output.write('\n')
print(json.dumps(dict(status=receipt['status'], png=str(outputs[0]), facts=FACTS)))
