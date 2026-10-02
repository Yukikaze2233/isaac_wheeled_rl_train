#!/usr/bin/env python3
"""Paired delay implementation equivalence/timing; consumes no PPO updates."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from wheeled_tasks.chassis.signal_delay import PhysicalSignalDelay

    path = ROOT / "contracts/v6_task_campaign_flat_worker_v6.json"
    config = json.loads(path.read_text())
    legacy_path = args.reference_source / "src/wheeled_tasks/chassis/signal_delay.py"
    module = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("legacy_delay", legacy_path))
    spec.loader.exec_module(module)
    count, device = config["target_num_envs"], "cuda:0"
    torch.set_num_threads(4)

    def make(cls):
        return cls(count, device, config["signal_delay"], config["physics_dt"],
                   torch.Generator(device=device).manual_seed(2102))

    reference, optimized = make(module.PhysicalSignalDelay), make(PhysicalSignalDelay)
    sensors = torch.arange(count * 18, device=device).reshape(count, 18).float() * .0001
    target = sensors[:, :6]
    for delay in (reference, optimized):delay.reset(torch.arange(count, device=device), sensors)
    # Includes multiple wraparounds, all ages, clean samples and partial/full resets.
    with torch.inference_mode():
        for tick in range(113):
            if tick in (3, 29, 60, 80):
                ids = torch.arange(count, device=device) if tick == 80 else torch.arange(tick, count, 31, device=device)
                for delay in (reference, optimized):delay.reset(ids, sensors)
            if tick % 4 == 0:
                for delay in (reference, optimized):delay.begin_policy((1400, 2000, 2500)[tick // 4 % 3])
            left = reference.targets(target[:, :4] + tick, target[:, 4:] - tick)
            right = optimized.targets(target[:, :4] + tick, target[:, 4:] - tick)
            # Leg target has four features, wheel target has two.
            for a, b in zip(left, right, strict=True):torch.testing.assert_close(a, b, rtol=0., atol=0.)
            for delay in (reference, optimized):delay.observe(sensors + tick)
            torch.testing.assert_close(reference.feedback, optimized.feedback, rtol=0., atol=0.)
            torch.testing.assert_close(reference.lags, optimized.lags, rtol=0., atol=0.)
            torch.testing.assert_close(reference.generator.get_state(), optimized.generator.get_state(), rtol=0., atol=0.)
        for key, value in reference.metrics().items():
            torch.testing.assert_close(torch.as_tensor(value), torch.as_tensor(optimized.metrics()[key]), rtol=0., atol=0.)

        def block(delay):
            for tick in range(96):
                if tick % 4 == 0:delay.begin_policy(2500)
                delay.targets(target[:, :4], target[:, 4:])
                delay.observe(sensors)
                if tick % 4 == 3:delay.metrics()

        samples = {"reference": [], "optimized": []}
        for delay in (reference, optimized):block(delay)
        for repetition in range(6):
            order = (("reference", reference), ("optimized", optimized))
            for name, delay in (order if repetition % 2 == 0 else order[::-1]):
                torch.cuda.synchronize()
                started = time.perf_counter()
                block(delay)
                torch.cuda.synchronize()
                samples[name].append(time.perf_counter() - started)
        operators = {}
        for name, delay in (("reference", reference), ("optimized", optimized)):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                block(delay)
                torch.cuda.synchronize()
            operators[name] = {event.key: event.count for event in profile.key_averages()
                               if event.key in ("aten::item", "aten::_local_scalar_dense", "aten::roll", "aten::is_nonzero")}
        assert not operators["optimized"], operators
    result = dict(status="paired_delay_equivalence_and_microbenchmark_passed", created_at=datetime.now(timezone.utc).isoformat(),
        scope="delay_layer_only_not_full_PPO_or_skill_acceptance", formal_updates_consumed=0,
        num_envs=count, physics_steps_per_block=96, contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reference_source=str(args.reference_source), reference_module_sha256=hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
        samples_seconds=samples, median_seconds={k: statistics.median(v) for k, v in samples.items()}, operators=operators,
        exact_feedback_targets_lags_rng_and_metrics=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
