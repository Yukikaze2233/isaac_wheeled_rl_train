#!/usr/bin/env python3
"""Compare finished experiment runs (metrics.jsonl written by run_experiment).

    python3 scripts/compare_experiments.py runs/*            # table to stdout
    python3 scripts/compare_experiments.py runs/* --csv out.csv
"""
import argparse
import csv
import json
import os
import sys


def load_metrics(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--tail", type=int, default=10, help="average the last N rows per run")
    args = p.parse_args()

    table, csv_rows = [], []
    for run_dir in args.run_dirs:
        rows = load_metrics(run_dir)
        if not rows:
            continue
        name = os.path.basename(os.path.normpath(run_dir))
        last_it = rows[-1]["iteration"]
        tail = rows[-args.tail:]
        mean_ep = sum(r["ep_reward"] for r in tail) / len(tail)
        mean_extra = sum(r["extra"] for r in tail) / len(tail)
        mean_kl = sum(r["kl"] for r in tail) / len(tail)
        table.append((name, last_it + 1, mean_ep, mean_extra, mean_kl))
        csv_rows.append({"experiment": name, "iterations": last_it + 1,
                         "tail_mean_ep_reward": round(mean_ep, 4),
                         "tail_mean_extra_loss": round(mean_extra, 6),
                         "tail_mean_kl": round(mean_kl, 6)})

    if not table:
        print("no metrics.jsonl found in the given dirs")
        return

    width = max(len(r[0]) for r in table) + 2
    print(f"{'experiment':<{width}}{'iters':>7}{'ep_reward':>12}{'extra':>10}{'kl':>9}")
    for name, it, ep, extra, kl in sorted(table, key=lambda r: -r[2]):
        print(f"{name:<{width}}{it:>7}{ep:>12.2f}{extra:>10.4f}{kl:>9.4f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"csv written: {args.csv}")


if __name__ == "__main__":
    main()
