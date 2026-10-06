#!/usr/bin/env python3
"""Offline static scientific plot from the two preserved TensorBoard HTTP JSONs."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
INPUTS = {
    BASE / 'read001/curves.json': '17beece01856be02cba92d1f194dac772e2fdfccace2649a9979441486e0c6a7',
    BASE / 'read002/curves.json': 'b630b5e68508eb59e3c9c647a6531c12ab45b87c1b8acde5be77c76c268bcd82',
}
COLORS = ('#087E8B', '#D47823', '#74539B')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert all(digest(path) == sha for path, sha in INPUTS.items()), 'Preserved input SHA differs'
CURVES = {}
DOCUMENTS = []
for path in INPUTS:
    document = json.loads(path.read_text())
    DOCUMENTS.append(document)
    assert document['qualification'] is None and document['PPO_updates'] == 0
    assert document['run'] == 'v6-task-domains-d073fe4/recovery/attempt_006/train'
    for tag, record in document['curves'].items():
        assert tag not in CURVES and record['status'] == 'read'
        points = np.asarray(record['points'], dtype=float)
        assert points.shape == (record['records'], 3) and np.isfinite(points).all()
        assert np.all(points[1:, 1] > points[:-1, 1])
        assert np.equal(points[:, 1], np.floor(points[:, 1])).all()
        CURVES[tag] = points
assert len(CURVES) == 15

def trailing_sampled_mean(x, y):
    """Equal-weight sampled observations within current logged update minus49..current."""
    left = np.searchsorted(x, x - 49, side='left')
    sums = np.concatenate(([0.0], np.cumsum(y)))
    right = np.arange(len(x)) + 1
    return (sums[right] - sums[left]) / (right - left)

def draw(axis, tags, labels, scale=1, smooth=True, styles=None):
    for index, (tag, label) in enumerate(zip(tags, labels)):
        points = CURVES[tag]
        x, y = points[:, 1], points[:, 2] * scale
        color = COLORS[index]
        style = styles[index] if styles else '-'
        if smooth:
            axis.plot(x, y, color=color, lw=0.75, alpha=0.24, zorder=2)
            axis.plot(x, trailing_sampled_mean(x, y), color=color, lw=2.1, label=label, ls=style, zorder=3)
        else:
            axis.plot(x, y, color=color, lw=1.7, label=label, zorder=3)
    axis.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.96, loc='upper right', fontsize=10.5)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
    'axes.titleweight': 'bold', 'axes.labelsize': 11, 'axes.edgecolor': '#BAC4CC',
    'axes.labelcolor': '#324151', 'xtick.color': '#566676', 'ytick.color': '#566676',
    'text.color': '#223140', 'figure.facecolor': 'white', 'axes.facecolor': '#FCFDFE',
    'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})
fig, axes = plt.subplots(3, 2, figsize=(16, 12.1), sharex=True)
fig.subplots_adjust(left=0.075, right=0.965, top=0.855, bottom=0.13, wspace=0.2, hspace=0.36)
fig.suptitle('Source022  |  Recovery training curves', x=0.075, y=0.972, ha='left', fontsize=21, fontweight='bold')
fig.text(0.075, 0.934, 'Existing Kaiser TensorBoard scalar API  ·  reservoir sampled  ·  training observations, not fixed evaluation', fontsize=12.5)
fig.text(0.075, 0.904, 'Thin lines: raw sampled values. Bold lines: equal-weight sampled-point mean over a trailing 50 logged-update span.', fontsize=11, color='#566676')
fig.text(0.075, 0.882, 'Native logged update is preserved. Panel D shows raw cumulative Behavior means without additional smoothing.', fontsize=11, color='#566676')

lesson_tags = ['/recovery/lesson/' + level for level in ('prefix_stand', 'prefix_release', 'original_chain')]
labels = ['L0  prefix_stand', 'L1  prefix_release', 'L2  original_chain']
axis = axes[0, 0]
draw(axis, [tag + '/probability' for tag in lesson_tags], labels, 100)
axis.set_title('A  Lesson sampling probabilities', loc='left', pad=11)
axis.set_ylabel('Probability (%)')
axis.set_ylim(0, 65)

axis = axes[0, 1]
draw(axis, [tag + '/success_fraction' for tag in lesson_tags], labels, 100, styles=('-', '--', ':'))
axis.set_title('B  Lesson success in current populations', loc='left', pad=11)
axis.set_ylabel('Logged fraction (%)')
axis.set_ylim(-0.6, 4)
axis.set_yticks([0, 1, 2, 3, 4])
assert all(np.equal(CURVES[tag + '/success_fraction'][:, 2], 0).all() for tag in lesson_tags)
axis.text(0.03, 0.70, 'All three curves overlap at 0\n1,000 sampled points per lesson\nCurrent-population fractions, not ended-episode rates',
          transform=axis.transAxes, fontsize=11, va='top', linespacing=1.6)

axis = axes[1, 0]
draw(axis, ['/recovery/hold_gate/' + name + '_pass_fraction' for name in ('gyro', 'wheel_speed', 'leg_speed')],
     ['Gyro gate', 'Wheel-speed gate', 'Leg-speed gate'], 100)
axis.set_title('C  Individual hold gates during RL hold', loc='left', pad=11)
axis.set_ylabel('Marginal pass fraction (%)')
axis.set_ylim(0, 33)
axis.text(0.02, 0.05, 'Separate gates; these curves do not give joint 1 s stability', transform=axis.transAxes, fontsize=10, color='#566676')

axis = axes[1, 1]
draw(axis, ['Behavior/stand_' + height + 'mm/planar_speed_mean_m_s' for height in ('230', '305', '430')],
     ['230 mm', '305 mm', '430 mm'], 100, smooth=False)
axis.set_title('D  Standing planar speed: cumulative means', loc='left', pad=11)
axis.set_ylabel('Planar speed (cm/s)')
axis.set_ylim(0, 16)
axis.text(0.02, 0.05, '104 logged points per height; not the latest 100-update mean', transform=axis.transAxes, fontsize=10, color='#566676')

axis = axes[2, 0]
draw(axis, ['LearningAudit/' + name + '/velocity/signed_forward_m_s' for name in ('forward_05', 'backward_05')],
     ['Forward: actual signed vx', 'Backward: actual signed vx'])
axis.set_title('E  Signed forward velocity at ±0.5 m/s commands', loc='left', pad=11)
axis.set_ylabel('Signed vx (m/s)')
axis.set_ylim(-0.58, 0.58)
for goal in (-0.5, 0.5):
    axis.axhline(goal, color='#87949F', linestyle=(0, (5, 4)), linewidth=1)
axis.axhline(0, color='#CED6DD', linewidth=0.8)
axis.text(0.02, 0.075, 'Dashed references: commanded −0.5 and +0.5 m/s', transform=axis.transAxes, fontsize=10, color='#566676')
axis.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.96, loc='center right', fontsize=10.5)

axis = axes[2, 1]
stop_tag = 'LearningAudit/start_stop_1/settled_stop/mean_speed_m_s'
draw(axis, [stop_tag], ['start_stop_1: conditional phase mean'], 100)
axis.set_title('F  Settled-stop speed: one command-transition group', loc='left', pad=11)
axis.set_ylabel('Planar speed (cm/s)')
axis.set_ylim(-0.3, 26)
axis.axhline(3, color='#87949F', linestyle=(0, (5, 4)), linewidth=1)
axis.text(0.985, 3.6, '3 cm/s reference', transform=axis.get_yaxis_transform(), ha='right', fontsize=10, color='#566676')

for axis in axes.flat:
    axis.set_xlim(30790, 31862)
    axis.axvspan(30802, 30811, color='#8B98A5', alpha=0.16, linewidth=0, zorder=1)
    axis.grid(axis='y', color='#D9E0E6', linewidth=0.7, alpha=0.7)
    axis.grid(axis='x', color='#E9EDF1', linewidth=0.5, alpha=0.6)
    axis.spines[['top', 'right']].set_visible(False)
    axis.xaxis.set_major_locator(MultipleLocator(200))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axis.xaxis.set_major_formatter(formatter)
for axis in axes[2]:
    axis.set_xlabel('Native logged update')

fig.text(0.075, 0.079, 'Scope: source022 / recovery attempt_006. Grey start band: logged steps 30802–30811 (10 critic-only updates).', fontsize=10.5)
fig.text(0.075, 0.055, 'HTTP snapshots: 2026-10-06 20:57–20:58 UTC+8. Reservoir samples are not complete-window or deterministic evaluation.', fontsize=10.5)
fig.text(0.075, 0.031, 'Stop phase sample counts were not queried: an initial zero may encode no samples. Use the independent 100-update report for weighted comparisons.', fontsize=10.5, color='#566676')

outputs = [HERE / 'source022_training_curves.png', HERE / 'source022_training_curves.svg']
assert not any(path.exists() for path in outputs), 'Preserve previous plot outputs'
fig.savefig(outputs[0], dpi=170)
fig.savefig(outputs[1])
plt.close(fig)
assert all(digest(path) == sha for path, sha in INPUTS.items()), 'Input bytes changed during plotting'
summary = {}
for tag, points in CURVES.items():
    summary[tag] = dict(records=len(points), native_logged_update_range=[int(points[0, 1]), int(points[-1, 1])],
        first_raw_value=float(points[0, 2]), last_raw_value=float(points[-1, 2]),
        min_raw_value=float(points[:, 2].min()), max_raw_value=float(points[:, 2].max()),
        final_trailing50_sampled_point_mean=float(trailing_sampled_mean(points[:, 1], points[:, 2])[-1]))
receipt = dict(schema='source022-TB-HTTP-offline-six-panel-plot-v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
    status='generated_pending_visual_review', run=DOCUMENTS[0]['run'], source='preserved existing Kaiser TensorBoard HTTP JSON only',
    inputs_sha256={str(path): sha for path, sha in INPUTS.items()}, inputs_after_sha256={str(path): digest(path) for path in INPUTS},
    inputs_unchanged=True, script_sha256=digest(__file__), outputs_sha256={str(path): digest(path) for path in outputs},
    sampling='TensorBoard scalar API reservoir sampled; each tag preserves its own native x coordinates',
    smoothing='Equal-weight sampled observations in inclusive native logged update x−49..x; no interpolation, no frame/time weighting, no full-window claim',
    panel_D='Raw cumulative Behavior mean; no extra smoothing, no conversion to current 100-update averages',
    marginal_gates='RL-hold conditional individual gate pass fractions, not continuous joint dwell probability',
    success='Logged current-population fractions, not ended-episode success rate or L2 qualification',
    stop_scope='Only start_stop_1 conditional stop mean; no phase sample counts in queried JSON, not six-group weighted stop result',
    raw_curve_summary=summary, SSH_actions=0, GPU_actions=0, PPO_updates=0, local_TensorBoard_started=False, qualification=None)
with (HERE / 'plot_receipt.json').open('x') as output:
    json.dump(receipt, output, indent=2, allow_nan=False)
    output.write('\n')
print(json.dumps(dict(status=receipt['status'], png=str(outputs[0]), svg=str(outputs[1]), input_files_unchanged=True)))
