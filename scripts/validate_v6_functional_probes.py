#!/usr/bin/env python3
"""Run CPU instances of transport, signal and curriculum modules before launch.

These probes validate executable timing/state semantics, not PhysX, hardware
identification or a learned policy. Every assertion uses observed module output.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.command_transport import UsbCommandTransport
from wheeled_tasks.chassis.robustness import V5SignalPerturbations
from wheeled_tasks.chassis.task import Phase
from wheeled_tasks.chassis.training_schedule import TrainingSchedule, jump_bin_probabilities


class ProbeFailure(RuntimeError):
    pass


def require(condition, message):
    if not bool(condition):
        raise ProbeFailure(message)


def close(actual, expected, message, atol=1e-7):
    require(torch.allclose(actual, expected, rtol=0., atol=atol), message)


def source_hashes():
    paths = ["scripts/validate_v6_functional_probes.py",
             "src/wheeled_tasks/chassis/command_transport.py",
             "src/wheeled_tasks/chassis/robustness.py",
             "src/wheeled_tasks/chassis/training_schedule.py",
             "src/wheeled_tasks/chassis/task.py", "src/wheeled_tasks/v40/core.py",
             "src/wheeled_tasks/v40/contract.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def transport_probe(seed):
    config = {"can_period_ms": 1., "delay_ms": [0., 1.2], "enabled_fraction": 1.,
              "drop_probability_max": 0., "burst_probability": 0., "burst_packets": [1, 2]}
    names = ["zero_ms", "point08_ms", "1p2_ms", "disabled", "all_drop", "fixed_burst"]
    profiles = [{"delay_ms": 0.}, {"delay_ms": .08}, {"delay_ms": 1.2}, None,
                {"delay_ms": .08, "drop_probability": 1.},
                {"delay_ms": .08, "burst_every_packets": 4, "burst_packets": 2}]
    link = UsbCommandTransport(len(profiles), "cpu", .001, config, seed, profiles=profiles)
    require(link.substeps == 1, "1 ms physics must execute exactly one CAN tick")
    output_rows, age_rows = [], []
    held = [0.] * len(profiles)
    held_generated = [0] * len(profiles)
    queue = [{} for _ in profiles]
    expected_lost = [0] * len(profiles)
    for tick in range(12):
        command = torch.full((len(profiles), 6), float(tick + 1))
        output = link.apply_torque(command)
        expected_output, expected_age = [], []
        for row, profile in enumerate(profiles):
            if profile is None:
                expected_output.append(float(tick + 1))
                expected_age.append(0.)
                continue
            if tick in queue[row]:
                held[row], held_generated[row] = queue[row].pop(tick)
            expected_output.append(held[row])
            expected_age.append((tick - held_generated[row]) * .001)
            # Independent packet timeline; source batch is consumed no earlier
            # than the following CAN tick, even for a configured 0 ms delay.
            delay_ticks = 2 if profile["delay_ms"] > 1. else 1
            period = profile.get("burst_every_packets", 0)
            dropped = profile.get("drop_probability", 0.) == 1.
            dropped |= bool(period and tick >= period and tick % period < profile["burst_packets"])
            if dropped:
                expected_lost[row] += 1
            else:
                queue[row][tick + delay_ticks] = (float(tick + 1), tick)
        close(output, torch.tensor(expected_output)[:, None].expand(-1, 6), f"CAN output mismatch at tick {tick}")
        close(link.age, torch.tensor(expected_age), f"CAN command age mismatch at tick {tick}")
        output_rows.append(output[:, 0].tolist())
        age_rows.append((link.age * 1000).tolist())
    close(link.lost, torch.tensor(expected_lost, dtype=torch.float32), "Packet drop counters disagree with applied commands")
    require(link.tick == 12, "Transport advanced an unexpected number of ticks")
    return {"passed": True, "physics_dt_s": .001, "can_period_s": .001, "ticks": link.tick,
            "zero_delay_semantics": "enabled stream consumes a newly sent batch no earlier than next CAN tick; disabled bypass is immediate",
            "cases": {name: {"profile": profiles[i], "output_by_tick": [r[i] for r in output_rows],
                      "command_age_ms_by_tick": [r[i] for r in age_rows], "lost_packets": int(link.lost[i])}
                      for i, name in enumerate(names)}}


def phase_transport_probe(seed):
    """Check interval torque, end cache and packet faults with explicit events."""
    config = {"can_period_ms": 1., "delay_ms": [0., 1.2], "enabled_fraction": 1.,
              "drop_probability_max": 0., "burst_probability": 0., "burst_packets": [1, 2],
              "delivery_model": "phase_aware_hold_v1", "can_phase_ms": [0., 1.],
              "jitter_ms": .02, "jitter_correlation": .9}
    names = ["before_phase", "after_phase_next_tick", "zero_delay_zero_phase",
             "disabled", "iid_all_drop_after_warmup", "fixed_burst"]
    profiles = [{"delay_ms": .08, "can_phase_ms": .25},
                {"delay_ms": .8, "can_phase_ms": .25},
                {"delay_ms": 0., "can_phase_ms": 0.}, None,
                {"delay_ms": .08, "can_phase_ms": .25},
                {"delay_ms": .08, "can_phase_ms": .25,
                 "burst_every_packets": 4, "burst_packets": 2}]
    link = UsbCommandTransport(len(profiles), "cpu", .001, config, seed, profiles=profiles)
    require(link.substeps == 1, "Phase-aware physics must retain exactly one CAN event per tick")
    output_rows, held_rows = [], []
    held = [0.] * len(profiles)
    queue = [[] for _ in profiles]
    expected_lost = [0] * len(profiles)
    for tick in range(12):
        if tick == 1:
            # Fault injection after one delivered packet proves that the IID
            # drop branch holds a nonzero cache, rather than clearing to zero.
            link.loss_probability[4] = 1.
        target = float(tick + 1)
        output = link.apply_torque(torch.full((len(profiles), 6), target))
        expected_output, expected_held = [], []
        for row, profile in enumerate(profiles):
            if profile is None:
                expected_output.append(target)
                expected_held.append(0.)
                continue
            old = held[row]
            period = profile.get("burst_every_packets", 0)
            dropped = row == 4 and tick >= 1
            dropped |= bool(period and tick >= period and tick % period < profile["burst_packets"])
            if dropped:
                expected_lost[row] += 1
            else:
                queue[row].append((tick + profile["delay_ms"], target))
            event_ms = tick + profile["can_phase_ms"]
            arrived = [packet for packet in queue[row] if packet[0] <= event_ms]
            queue[row] = [packet for packet in queue[row] if packet[0] > event_ms]
            if arrived:
                held[row] = arrived[-1][1]
            fraction = profile["can_phase_ms"]
            expected_output.append(old * fraction + held[row] * (1 - fraction))
            expected_held.append(held[row])
        close(output, torch.tensor(expected_output)[:, None].expand(-1, 6),
              f"Phase-aware interval torque mismatch at tick {tick}")
        close(link.held, torch.tensor(expected_held)[:, None].expand(-1, 6),
              f"Phase-aware end cache mismatch at tick {tick}")
        output_rows.append(output[:, 0].tolist())
        held_rows.append(link.held[:, 0].tolist())
    close(link.lost, torch.tensor(expected_lost, dtype=torch.float32),
          "Phase-aware packet loss counters disagree with executed holds")
    jitter = phase_jitter_probe(seed, config)
    return {"passed": True, "delivery_model": config["delivery_model"],
            "physics_dt_s": .001, "can_period_s": .001, "ticks": link.tick,
            "scope": "interval impulse and end-cache semantics; no intra-tick physical feedback",
            "iid_fault_injection": "p=0 at tick0; p=1 at ticks1..11 after a real delivery",
            "cases": {name: {"profile": profiles[i],
                "interval_torque_by_tick": [row[i] for row in output_rows],
                "end_held_by_tick": [row[i] for row in held_rows],
                "lost_packets": int(link.lost[i])} for i, name in enumerate(names)},
            "jitter_boundary": jitter}


def phase_jitter_probe(seed, config, count=64):
    """Use seeded jitter around a CAN boundary and an independent scalar queue."""
    profiles = [{"delay_ms": .1, "can_phase_ms": .1,
                 "jitter_ms": .02, "jitter_correlation": 0.}] * count
    links = [UsbCommandTransport(count, "cpu", .001, config, seed, profiles=profiles)
             for _ in range(2)]
    held, queue = [0.] * count, [[] for _ in range(count)]
    early_count = late_count = 0
    delay_min, delay_max = float("inf"), 0.
    first_delays, first_outputs = None, None
    for tick in range(12):
        target = torch.full((count, 6), float(tick + 1))
        first, second = [link.apply_torque(target) for link in links]
        require(torch.equal(first, second) and torch.equal(links[0].delay, links[1].delay),
                "Phase-aware fixed-seed jitter replay diverged")
        delays = (links[0].delay * 1000).tolist()
        expected = []
        for row, delay in enumerate(delays):
            early_count += int(delay <= .1)
            late_count += int(delay > .1)
            old = held[row]
            queue[row].append((tick + delay, float(tick + 1)))
            arrived = [packet for packet in queue[row] if packet[0] <= tick + .1 + 1e-7]
            queue[row] = [packet for packet in queue[row] if packet[0] > tick + .1 + 1e-7]
            if arrived:
                held[row] = arrived[-1][1]
            expected.append(.1 * old + .9 * held[row])
        close(first, torch.tensor(expected)[:, None].expand(-1, 6),
              f"Phase-aware jitter event output mismatch at tick {tick}", atol=1e-6)
        delay_min, delay_max = min(delay_min, min(delays)), max(delay_max, max(delays))
        if tick == 0:
            first_delays, first_outputs = delays, first[:, 0].tolist()
            require((first[:, 0] == 0).any() and (first[:, 0] > 0).any(),
                    "Sub-ms jitter did not cross CAN boundary in actual output")
    require(early_count > 0 and late_count > 0, "Jitter only exercised one side of the CAN event")
    require(delay_min >= .08 - 1e-7 and delay_max <= .12 + 1e-7,
            "Explicit jitter innovation exceeded its configured range")
    return {"passed": True, "count": count, "ticks": 12, "phase_ms": .1,
            "base_delay_ms": .1, "jitter_half_width_ms": .02, "jitter_correlation": 0.,
            "early_packet_count": early_count, "late_packet_count": late_count,
            "delay_observed_range_ms": [delay_min, delay_max],
            "first_tick_delay_ms": first_delays, "first_tick_interval_torque": first_outputs,
            "fixed_seed_replay_equal": True, "scalar_event_queue_matches": True}


def signal_probe(seed, count=2048):
    config = {"frame_dim": 35, "max_delay_steps": 2, "noise_scale": 1., "enabled_fraction": .5}
    replicas = [V5SignalPerturbations(count, "cpu", config, torch.Generator().manual_seed(seed)) for _ in range(2)]
    ids = torch.arange(count)
    for model in replicas:
        model.reset(ids)
    a, b = replicas
    require(torch.equal(a.enabled, b.enabled) and torch.equal(a.obs_lag, b.obs_lag)
            and torch.equal(a.act_lag, b.act_lag), "Signal fixed-seed reset is not reproducible")
    enabled, obs_lag, act_lag = a.enabled.clone(), a.obs_lag.clone(), a.act_lag.clone()
    require(enabled.any() and (~enabled).any(), "Signal probe did not exercise enabled and disabled lanes")
    require(((obs_lag >= 0) & (obs_lag <= 2) & (act_lag >= 0) & (act_lag <= 2)).all(), "Signal lag outside contract")
    require((obs_lag[~enabled] == 0).all() and (act_lag[~enabled] == 0).all(), "Disabled signal lane retains lag")
    nonsensor = [i for i in range(35) if i not in a.sensor_indices]
    maximum_noise = 0.
    for tick in range(6):
        clean = torch.full((count, 35), float(tick))
        actions = torch.full((count, 6), float(tick + 1))
        observed = [p.observation(clean, tick) for p in replicas]
        applied = [p.action(actions, tick) for p in replicas]
        require(torch.equal(observed[0], observed[1]) and torch.equal(applied[0], applied[1]), "Signal replay diverged")
        require(torch.equal(a.observation(clean, tick), observed[0]), "Repeated observation read advanced noise or lag")
        close(applied[0][:, 0], (tick + 1 - act_lag).clamp_min(1).float(), "Action lag differs from selected episode lag")
        close(observed[0][:, nonsensor], clean[:, nonsensor], "Signal perturbation altered command/action/mode slots")
        close(observed[0][~enabled], clean[~enabled], "Disabled signal lane changed observation")
        require(torch.isfinite(observed[0]).all(), "Signal noise created nonfinite values")
        newest = a.observations.buffer[:, -1]
        maximum_noise = max(maximum_noise, float((newest - clean).abs().max()))
        require(torch.equal(a.enabled, enabled) and torch.equal(a.obs_lag, obs_lag)
                and torch.equal(a.act_lag, act_lag), "Signal episode parameters changed between resets")
    # A separate zero-noise instance proves observation delay, independently of
    # Gaussian sample magnitude and the action history implementation.
    noiseless = V5SignalPerturbations(count, "cpu", {**config, "noise_scale": 0.}, torch.Generator().manual_seed(seed))
    noiseless.reset(ids)
    for tick in range(5):
        clean = torch.full((count, 35), float(tick))
        value = noiseless.observation(clean, tick)
        close(value[:, 4], (tick - noiseless.obs_lag).clamp_min(0).float(), "Observation lag differs from selected episode lag")
    require(maximum_noise > 0, "Enabled signal noise was never exercised")
    return {"passed": True, "count": count, "ticks": 6, "enabled_fraction": float(enabled.float().mean()),
            "obs_lag_range_steps": [int(obs_lag.min()), int(obs_lag.max())],
            "action_lag_range_steps": [int(act_lag.min()), int(act_lag.max())],
            "configured_noise_std_max": float(a.noise.max()), "observed_noise_abs_max": maximum_noise,
            "episode_parameters_constant": True, "fixed_seed_replay_equal": True,
            "noise_model": "Gaussian; observed maximum is a sample statistic, not a hard bound"}


def schedule_probe(seed, count=8192):
    height = {"actor_update_boundary": 200, "initial_fixed_fraction": .8, "final_fixed_fraction": .5,
              "fixed_heights_m": [.23, .25, .28, .305, .32, .35, .38, .4, .43]}
    jump = {"bins_m": [.03, .06, .1, .15, .2, .25, .3],
            "initial_probabilities": [.3, .25, .2, .1, .05, .05, .05],
            "minimum_probability": .05, "confirmed_pass_transfer_fraction": .5}
    protocol = {"stress_episode_fraction": .5, "interval_seconds": [.5, 4.],
                "repeat_probability": .1, "cancel_probability": .1, "new_probability": .8}
    settings = {"enabled": True, "height_sampling_schedule": height, "jump_curriculum": jump,
                "unilateral_degradation_fraction": .05, "edge_reset_fraction": .1,
                "request_protocol": protocol, "command_resampling_seconds": [.5, 4.],
                "long_command_resampling_seconds": [4., 8.], "long_command_fraction": .2}
    def make_env():
        generator = torch.Generator().manual_seed(seed)
        return SimpleNamespace(num_envs=count, device="cpu", cfg={"new_asset_curriculum": deepcopy(settings)},
            global_actor_update=1000, stage_actor_update=199, random=lambda n: torch.rand(n, generator=generator))
    envs = [make_env(), make_env()]
    models = [TrainingSchedule(e) for e in envs]
    ids = torch.arange(count)
    velocities = [torch.zeros(count, 6) for _ in models]
    for model, velocity in zip(models, velocities):
        model.reset(ids, velocity)
    a, b = models
    require(torch.equal(a.motor_scale, b.motor_scale) and torch.equal(a.edge_reset, b.edge_reset)
            and torch.equal(velocities[0], velocities[1]), "Curriculum fixed-seed reset differs")
    require(torch.equal((a.motor_scale < 1).sum(-1) == 3, a.degraded), "Degradation must affect exactly one side")
    require(not ((a.motor_scale[:, 0] < 1) & (a.motor_scale[:, 3] < 1)).any(), "Both sides degraded in a unilateral sample")
    velocity = velocities[0]
    require(velocity[:, 0].abs().max() <= .3 and velocity[:, 3:6].abs().max() <= .5
            and (velocity[:, [1, 2]] == 0).all(), "Edge reset velocity exceeds bounded prior")
    require(velocity[:, 3:5].abs().sum() > 0, "Roll/pitch recovery excitation absent")
    before = a.motor_scale.clone()
    fractions = []
    for update, expected_fraction in ((199, .8), (200, .5)):
        envs[0].stage_actor_update = update
        values = a.sample_height(count, [.23, .43])
        require((values >= .23).all() and (values <= .43).all(), "Sampled height outside full domain")
        observed_fraction = float((values[:, None] == torch.tensor(height["fixed_heights_m"])).any(-1).float().mean())
        require(abs(observed_fraction - expected_fraction) < .04, "Height mixture did not follow actor-update boundary")
        fractions.append(observed_fraction)
    duration = a.command_duration(count)
    require(duration.min() >= .5 and duration.max() <= 8., "Command dwell outside its domain")
    long_fraction = float((duration >= 4.).float().mean())
    require(abs(long_fraction - .2) < .04, "Protected long-command fraction missing")
    a.sample_jump(ids, {"kind": "jump"})
    require((a.jump_bin >= 0).all() and (a.jump_bin < 7).all(), "Jump sample outside bin catalogue")
    initial_fraction = float((a.jump_bin == 0).float().mean())
    envs[0].confirmed_jump_bins_m = [.03]
    a.sample_jump(ids, {"kind": "jump"})
    promoted_fraction = float((a.jump_bin == 0).float().mean())
    require(abs(initial_fraction - .3) < .04 and abs(promoted_fraction - .175) < .04,
            "Confirmed jump promotion did not change runtime sampling")
    require(torch.equal(before, a.motor_scale), "Command resampling changed episode motor degradation")
    require(min(jump_bin_probabilities(jump, [.03])) >= .05, "Promotion removed a protected jump bin")
    # Exercise a controlled request trace through the actual stateful module.
    a.request_stress[:] = True
    a.next_request[:] = torch.inf
    eligible = torch.ones(count, dtype=torch.bool)
    active = a.update_requests(torch.full((count,), .8), torch.full((count,), Phase.GROUND), eligible, eligible)
    require(active.all(), "Initial jump request was not accepted")
    starts = a.request_start.clone()
    envs[0].random = lambda n: torch.full((n,), .15)
    a.next_request[:] = 0.
    a.update_requests(torch.ones(count), torch.full((count,), Phase.TAKEOFF), eligible, eligible)
    require((a.request_event == 2).all() and torch.equal(starts, a.request_start), "Busy repeat reset reference phase")
    envs[0].random = lambda n: torch.full((n,), .05)
    a.next_request[:] = 0.
    active = a.update_requests(torch.full((count,), 1.2), torch.full((count,), Phase.FLIGHT), eligible, eligible)
    require(active.all() and torch.equal(starts, a.request_start), "Airborne cancel discarded landing request")
    return {"passed": True, "count": count, "stage_updates_checked": [199, 200],
            "height_fixed_fraction_observed": fractions, "jump_first_bin_fraction_before_after": [initial_fraction, promoted_fraction],
            "unilateral_degradation_fraction": float(a.degraded.float().mean()),
            "edge_reset_fraction": float(a.edge_reset.float().mean()), "long_command_fraction": long_fraction,
            "velocity_abs_max": velocity.abs().amax(0).tolist(), "episode_motor_scale_constant": True,
            "busy_repeat_preserves_start": True, "airborne_cancel_preserves_landing": True,
            "reset_scope": "bounded velocity perturbations including roll/pitch rates; no pose teleport or validated trajectory-state bank"}


def run_probes(seed=617):
    before = source_hashes()
    result = {"schema_version": 1, "scope": "CPU executable module probes; no GPU or PPO updates",
              "seed": seed, "torch_version": torch.__version__, "source_sha256": before,
              "transport": transport_probe(seed), "transport_phase_aware": phase_transport_probe(seed),
              "signal": signal_probe(seed),
              "training_schedule": schedule_probe(seed)}
    require(before == source_hashes(), "Probe sources changed while the receipt was being generated")
    result["passed"] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--contract", type=Path, help="Optionally bind this receipt to a frozen runtime contract")
    args = parser.parse_args(argv)
    try:
        report = run_probes(args.seed)
        if args.contract:
            report["contract_sha256"] = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    except Exception as error:
        report = {"schema_version": 1, "passed": False, "seed": args.seed,
                  "source_sha256": source_hashes(), "error": f"{type(error).__name__}: {error}"}
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
