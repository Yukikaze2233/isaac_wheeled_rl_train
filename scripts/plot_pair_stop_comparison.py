"""Export knee-stop sensitivity from TensorBoard events and audited nominal FK."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for row, side in enumerate(('left', 'right')):
        events = EventAccumulator(str(args.replay/side/'events'), size_guidance={'scalars': 0}).Reload()
        table = json.loads((args.coverage/f'{side}.json').read_text())['table']
        base = f'stop_comparison/{side}/historical_110/closed_loop/segment79'
        def series(tag):
            values = events.Scalars(tag)
            return np.array([v.step*.001 for v in values]), np.array([v.value for v in values])
        t, hip = series(base+'/axis0/q_real_rad')
        ta, aux = series(base+'/axis1/q_real_rad')
        if not np.array_equal(t, ta):
            raise ValueError('Mismatched held-out sample grids')
        delta = aux-hip
        grid = np.array(table['delta_grid_rad'])
        if delta.min() < grid.min()-1e-6 or delta.max() > grid.max()+1e-6:
            raise ValueError('Measured points exceed validated FK interpolation domain')
        beta = np.interp(delta, grid, table['knee_inner_deg'])
        axes[row, 0].plot(t, beta, color='#161616', lw=1.3, label='bag: nominal-coordinate FK')
        for label, title, color in [('historical_110','historical stop 110 deg','#6586a6'),
                                     ('nominal_105','nominal stop 105 deg','#c96838'),
                                     ('calibration_plateau','effective stop from initial dwell','#318a7c')]:
            prefix = f'stop_comparison/{side}/{label}/closed_loop/segment79'
            tb, b = series(prefix+'/beta_sim_deg')
            tf, force = series(prefix+'/limit_constraint_nm')
            if not np.array_equal(t, tb) or not np.array_equal(t, tf):
                raise ValueError('Comparison requires identical time grids, no time warping')
            axes[row, 0].plot(tb, b, lw=1., color=color, label=title)
            axes[row, 1].plot(tf, force, lw=1., color=color, label=title)
        axes[row, 0].set_ylabel(f'{side.title()} knee inner angle (deg)')
        axes[row, 1].set_ylabel(f'{side.title()} simulated stop reaction (Nm)')
        for ax in axes[row]:
            ax.grid(alpha=.2)
        axes[row, 0].legend(fontsize=8, loc='best')
    for ax in axes[-1]: ax.set_xlabel('Time within held-out segment 79 (s)')
    axes[0, 0].set_title('Nominal-coordinate knee angle')
    axes[0, 1].set_title('Simulation constraint force; no measured contact-force channel')
    fig.suptitle('Changing the stop can improve motor-angle RMSE by adding simulated contact\n'
                 'Fixed PD 160/2.5 and fixed plant candidate; bag FK stays below the initial-dwell plateau', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
