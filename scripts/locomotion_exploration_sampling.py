"""Native actor sampling and first-cohort accounting for an independent probe.

This module does not import Isaac, construct PPO, change saved std, or reset an
environment. The caller authenticates the PT/contract/seal and owns the scene.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import math

import torch


CASES = ("stand_305mm", "forward_0p5", "backward_0p5")
PARTITIONS = ("failure", "success", "boundary_truncation", "timeout", "right_censored_unfinished")
PRIMARY_WINDOWS = {"stand_305mm": ("stand",), "forward_0p5": ("steady", "stop"),
                   "backward_0p5": ("steady", "stop")}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rng_sha(state):
    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()


def load_native_actor(saved, agent_config, source, device):
    """Strictly load the saved native 35D -> 6D MLP, including its distribution.

    PT provenance and the complete checkpoint's critic/optimizer authentication
    belong to the caller. No optimizer or ONNX wrapper is constructed here.
    """
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    _require(tuple(source.get(k) for k in ("actor_dim", "critic_dim", "action_dim")) == (35, 81, 6),
             "Native exploration probe requires 35/81/6 ABI")
    _require(agent_config.get("obs_groups") == {"actor": ["policy"], "critic": ["critic"]},
             "Native actor observation groups changed")
    args = deepcopy(agent_config.get("actor", {}))
    _require(args.pop("class_name", None) == "MLPModel", "Native MLPModel required")
    _require(args.get("hidden_dims") == [256, 128, 64] and args.get("activation") == "elu"
             and args.get("obs_normalization") is False, "Native actor architecture changed")
    distribution = args.get("distribution_cfg", {})
    _require(isinstance(distribution, Mapping)
             and distribution.get("class_name") == "GaussianDistribution"
             and distribution.get("std_type") == "scalar", "Native scalar Gaussian distribution required")
    state = saved.get("actor_state_dict") if isinstance(saved, Mapping) else None
    _require(isinstance(state, Mapping) and bool(state), "Saved actor state missing")
    std = state.get("distribution.std_param")
    _require(isinstance(std, torch.Tensor) and tuple(std.shape) == (6,), "Saved six-axis std missing")
    _require(all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                 for value in state.values()), "Saved actor state is nonfinite or not tensor-valued")
    observations = TensorDict({"policy": torch.zeros(1, 35), "critic": torch.zeros(1, 81)}, batch_size=[1])
    actor = MLPModel(observations, agent_config["obs_groups"], "actor", 6, **args)
    actor.load_state_dict(state, strict=True)
    loaded = actor.state_dict()
    _require(all(value.dtype == loaded[name].dtype and torch.equal(value.cpu(), loaded[name].cpu())
                 for name, value in state.items()),
             "Strict actor load changed saved tensors")
    return actor.eval().to(device)


class NativeActionSampler:
    """Call the original distribution sampler using a private device RNG stream.

    The mean arm also calls draw; its caller sends ``mu`` instead of ``sample``.
    Use one sampler per process, without concurrent global-RNG sampling threads.
    """

    def __init__(self, actor, seed, device):
        _require(type(seed) is int and seed >= 0, "Nonnegative integer sampling seed required")
        self.device = torch.device(device)
        _require(self.device.type in ("cpu", "cuda"), "CPU or explicit CUDA device required")
        _require(self.device.type != "cuda" or self.device.index is not None, "CUDA device index required")
        _require(getattr(actor, "distribution", None) is not None, "Native actor distribution missing")
        _require(all(p.device == self.device for p in actor.parameters()), "Actor/sampler device mismatch")
        self.actor = actor
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.draw_counter = 0
        self.sample_element_count = 0

    def draw(self, observations):
        """Return native mean/sample/std tensors and private RNG provenance."""
        with torch.inference_mode():
            mu = self.actor(observations, stochastic_output=False)
            _require(mu.ndim == 2 and mu.shape[-1] == 6 and mu.device == self.device
                     and bool(torch.isfinite(mu).all()), "Native actor mean invalid")
            # Deterministic MLPModel.forward does not refresh its distribution.
            self.actor.distribution.update(mu)
            std = self.actor.output_std
            _require(std.shape == mu.shape and bool(torch.isfinite(std).all()) and bool((std > 0).all()),
                     "Native effective std invalid")
            before = self.generator.get_state()
            devices = [self.device.index] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                if self.device.type == "cuda":
                    torch.cuda.set_rng_state(before, self.device)
                else:
                    torch.random.set_rng_state(before)
                sample = self.actor.distribution.sample()
                after = (torch.cuda.get_rng_state(self.device) if self.device.type == "cuda"
                         else torch.random.get_rng_state())
            _require(sample.shape == mu.shape and bool(torch.isfinite(sample).all()), "Native sample invalid")
            self.generator.set_state(after)
            self.draw_counter += 1
            self.sample_element_count += mu.numel()
            return {"mu": mu.clone(), "sample": sample.clone(), "effective_std": std.clone(),
                    "potential_sample_minus_mean": (sample - mu).clone(),
                    "rng_before_sha256": _rng_sha(before), "rng_after_sha256": _rng_sha(after),
                    "draw_counter": self.draw_counter, "sample_element_count": self.sample_element_count}


def _bool_vector(value, count, name):
    if isinstance(value, torch.Tensor):
        _require(value.dtype == torch.bool and tuple(value.shape) == (count,), f"Invalid {name} mask")
        value = value.tolist()
    _require(isinstance(value, Sequence) and len(value) == count
             and all(type(x) is bool for x in value), f"Invalid {name} mask")
    return tuple(value)


class FirstCohortLedger:
    """Preserve the first native outcome of 3 cases x N4, regardless of later steps.

    After first native done, the caller may continue simulation and sampling;
    that lane never reopens and must not contribute more first-cohort frames.
    """

    def __init__(self, case_names, *, physics_dt=.005, replicas=4):
        names = tuple(case_names)
        _require(type(replicas) is int and replicas == 4, "Exactly N4 per case required")
        _require(set(names) == set(CASES) and all(n == replicas for n in Counter(names).values()),
                 "Three fixed locomotion cases with N4 each required")
        _require(math.isfinite(physics_dt) and physics_dt == .005, "200Hz physical clock required")
        self.physics_dt = physics_dt
        self.replicas = replicas
        self.last_physics_tick = -1
        seen = Counter()
        self.rows = []
        for lane, name in enumerate(names):
            self.rows.append({"lane": lane, "case": name, "replica": seen[name], "birth_id": 1,
                              "partition": None, "terminal": None, "windows": {}})
            seen[name] += 1

    @property
    def live(self):
        return tuple(row["partition"] is None for row in self.rows)

    def observe(self, *, done, terminated, timeouts, success, reasons, physical_tick):
        """Close and return only newly native-terminal lane indices."""
        _require(type(physical_tick) is int and physical_tick >= 0
                 and physical_tick > self.last_physics_tick, "Native physical clock must advance")
        count = len(self.rows)
        done, terminated, timeouts, success = (
            _bool_vector(v, count, name) for v, name in
            ((done, "done"), (terminated, "terminated"), (timeouts, "timeouts"), (success, "success")))
        _require(isinstance(reasons, Mapping) and all(isinstance(k, str) for k in reasons), "Native reasons required")
        masks = {name: _bool_vector(value, count, name) for name, value in reasons.items()}
        _require(all(d or not (t or h) for d, t, h in zip(done, terminated, timeouts)),
                 "Native termination/timeout missing done")
        closed = []
        for lane, row in enumerate(self.rows):
            if row["partition"] is not None or not done[lane]:
                continue
            flags = {name: mask[lane] for name, mask in masks.items()}
            truncated = flags.get("boundary", False) or flags.get("blocked", False)
            partition = ("failure" if terminated[lane] else "success" if success[lane] else
                         "boundary_truncation" if truncated else "timeout" if timeouts[lane] else None)
            _require(partition is not None, "Native done has no native outcome")
            row["partition"] = partition
            row["terminal"] = {"physical_tick": physical_tick, "time_s": physical_tick * self.physics_dt,
                               "done": done[lane], "terminated": terminated[lane], "timeout": timeouts[lane],
                               "success": success[lane], "reasons": flags}
            closed.append(lane)
        self.last_physics_tick = physical_tick
        return tuple(closed)

    def record_window(self, lane, name, value, *, complete, observed_seconds):
        """Record a final window statistic from eligible first-cohort frames only.

        The caller verifies frame/tick coverage before setting complete=True.
        Partial values stay in the row but are excluded from completed means.
        """
        _require(type(lane) is int and 0 <= lane < len(self.rows), "Invalid window lane")
        _require(isinstance(name, str) and bool(name) and type(complete) is bool, "Invalid window identity")
        _require(math.isfinite(observed_seconds) and observed_seconds >= 0., "Invalid observed window duration")
        _require(value is None or (type(value) in (int, float) and math.isfinite(value)), "Invalid window statistic")
        _require(not complete or (value is not None and observed_seconds > 0.), "Complete window lacks observations")
        windows = self.rows[lane]["windows"]
        _require(name not in windows, "First-cohort window already recorded")
        windows[name] = {"complete": complete, "value": value, "observed_seconds": observed_seconds}

    def summary(self, *, physical_tick):
        _require(type(physical_tick) is int and physical_tick >= 0
                 and physical_tick >= self.last_physics_tick, "Invalid actual summary clock")
        rows = deepcopy(self.rows)
        for row in rows:
            if row["partition"] is None:
                row["partition"] = "right_censored_unfinished"
                row["censor_time_s"] = physical_tick * self.physics_dt
                row["exposure_physics_ticks"] = physical_tick
            else:
                row["exposure_physics_ticks"] = row["terminal"]["physical_tick"]
            row["exposure_seconds"] = row["exposure_physics_ticks"] * self.physics_dt
        cases = {}
        for name in CASES:
            members = [row for row in rows if row["case"] == name]
            partitions = {part: sum(row["partition"] == part for row in members) for part in PARTITIONS}
            names = set(PRIMARY_WINDOWS[name]).union(*(row["windows"] for row in members))
            windows = {}
            for window in sorted(names):
                recorded = [row["windows"][window] for row in members if window in row["windows"]]
                completed = [record for record in recorded if record["complete"]]
                values = [record["value"] for record in completed]
                windows[window] = {"complete_count": len(completed), "denominator": self.replicas,
                                   "complete_fraction": len(completed) / self.replicas,
                                   "recorded_count": len(recorded), "mean": sum(values) / len(values) if values else None,
                                   "completed_observed_seconds": sum(record["observed_seconds"] for record in completed)}
            cases[name] = {"fixed_birth_denominator": self.replicas, "partitions": partitions,
                           "windows": windows, "exposure_seconds": sum(row["exposure_seconds"] for row in members)}
        partitions = {part: sum(row["partition"] == part for row in rows) for part in PARTITIONS}
        return {"fixed_birth_denominator": len(rows), "actual_physics_tick": physical_tick,
                "actual_simulated_seconds": physical_tick * self.physics_dt, "partitions": partitions,
                "cases": cases, "rows": rows, "qualification": None,
                "post_first_terminal_frames_included": False}
