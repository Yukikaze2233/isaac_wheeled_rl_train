"""Experiment framework smoke test: registry, iterative resume, comparison.

Runs two registered branches (frame-stack baseline + HIM) for a few iterations
each via the same code path as scripts/run_experiment.py, verifies checkpoint
resume continues the iteration counter, and that metrics.jsonl rows exist for
compare_experiments.py to consume.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep +
           os.path.join(ROOT, "tests") + os.pathsep +
           os.environ.get("PYTHONPATH", ""))


def run_cli(*args):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_experiment.py"), *args],
                         capture_output=True, text=True, env=ENV, timeout=280)
    assert out.returncode == 0, f"run_experiment failed:\n{out.stderr[-800:]}"
    return out.stdout


def main():
    # 1) registry lists the controlled-variable series
    out = run_cli("--list")
    for name in ("exp000_framestack5", "exp001_him_latent16", "exp003_np3o_limit05"):
        assert name in out, f"{name} missing from registry"

    # 2) run the baseline briefly, then resume it — iteration counter continues
    log_dir = "/tmp/exp_test/exp000_framestack5"
    run_cli("--exp", "exp000_framestack5", "--iterations", "8", "--log-dir", log_dir)
    out = run_cli("--exp", "exp000_framestack5", "--iterations", "8", "--resume",
                  "--log-dir", log_dir)
    assert "resumed from iteration 8" in out, f"resume failed:\n{out[-400:]}"

    rows = [json.loads(l) for l in open(os.path.join(log_dir, "metrics.jsonl")) if l.strip()]
    assert len(rows) == 16 and rows[-1]["iteration"] == 15, f"metrics broken: {len(rows)} rows"
    assert all(k in rows[0] for k in ("iteration", "ep_reward", "extra", "kl", "lr"))

    # 3) a second branch runs through the same path
    run_cli("--exp", "exp001_him_latent16", "--iterations", "8",
            "--log-dir", "/tmp/exp_test/exp001_him_latent16")

    # 4) comparison table reads both runs
    cmp_out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "compare_experiments.py"),
                              "/tmp/exp_test/exp000_framestack5", "/tmp/exp_test/exp001_him_latent16"],
                             capture_output=True, text=True, env=ENV)
    assert cmp_out.returncode == 0
    assert "exp000_framestack5" in cmp_out.stdout and "exp001_him_latent16" in cmp_out.stdout
    print(cmp_out.stdout.strip())
    print("EXPERIMENT FRAMEWORK TESTS PASSED")


if __name__ == "__main__":
    main()
