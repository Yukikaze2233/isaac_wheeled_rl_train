#!/usr/bin/env python3
"""Export a measured/CAD/fitted comparison from identification TensorBoard data."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--side', choices=['left', 'right'], required=True)
    parser.add_argument('--segment', type=int, default=79)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    readers = [EventAccumulator(str(path/'events'), size_guidance={'scalars': 0}).Reload()
               for path in (args.baseline, args.candidate)]
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for axis in range(2):
        prefix = f'replay/{args.side}/segment_{args.segment}/axis_{axis}'

        def series(reader, name):
            events = reader.Scalars(prefix+'/'+name)
            return np.array([e.step*.001 for e in events]), np.array([e.value for e in events])

        t, real_q = series(readers[0], 'q_real_rad')
        _, ref = series(readers[0], 'q_ref_rad')
        _, real_dq = series(readers[0], 'dq_real_rad_s')
        axes[0, axis].plot(t, ref, color='#888888', lw=.8, ls='--', label='held target')
        axes[0, axis].plot(t, real_q, color='#171717', lw=1.2, label='measured')
        axes[1, axis].plot(t, real_dq, color='#171717', lw=.7, label='measured')
        for reader, label, color in zip(readers, ['CAD baseline', 'conditional fit'], ['#c66b25', '#177c9b']):
            ts, q = series(reader, 'q_sim_rad')
            tv, dq = series(reader, 'dq_sim_rad_s')
            if len(t) != len(ts) or not np.array_equal(t, ts):
                raise ValueError('Comparisons require identical sample grids; no hidden time warping.')
            axes[0, axis].plot(ts, q, color=color, lw=1., label=label)
            axes[1, axis].plot(tv, dq, color=color, lw=.8, label=label)
            axes[2, axis].plot(ts, q-real_q, color=color, lw=1., label=label)
        axes[0, axis].set_title(['Hip drive', 'Auxiliary knee drive'][axis])
        axes[0, axis].set_ylabel('V6 drive angle (rad)')
        axes[1, axis].set_ylabel('Drive speed (rad/s)')
        axes[2, axis].set_ylabel('Simulation - measured (rad)')
        axes[2, axis].set_xlabel('Time within held-out segment (s)')
        for row in range(3):
            axes[row, axis].grid(alpha=.2)
        axes[0, axis].legend(fontsize=8, loc='best')
    fig.suptitle(f'{args.side.title()} installed leg | held-out segment {args.segment}\n'
                 '160/2.5 PC PD at 1 kHz; 50 Hz target | conditional fit, not a training release', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
