"""Observe actual final payments without recomputing or changing any reward.

Mask-owned returns describe physical payments, not PPO targets, advantages or
optimized populations. Discounting uses the actual algorithm gamma and original
episode clock, including intervals excluded from learning. State is local to a
worker; a checkpoint records evidence and never restores a simulator episode.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch


VERSION = "reward_episode_measurement_v1"
TRANSITION_SCHEMA = "reward_episode_transition_v1"
PAYMENT_UNITS = "already_dt_scaled_payment_per_policy_step"
OUTCOMES = ("failure", "reported_success", "timeout", "boundary_truncation", "unclassified_done")
STREAMS = ("physical", "mask_owned", "mask_excluded")


class EpisodePaymentJournal:
    """Per-term full-episode accounting from the environment's returned reward.

    Feed exactly one returned transition before RSL timeout augmentation. The
    pre-reset identity owns the terminal payment. A partial first episode is
    counted separately; missing prefix payments are never invented. Unscheduled
    resets, duplicate policy intervals and inconsistent units/parts are errors.
    No skill pass, curriculum credit, threshold or reward is produced here.
    """

    def __init__(self, scene_groups, device, *, gamma, source_identity):
        if (not isinstance(scene_groups, (list, tuple)) or not scene_groups
                or any(not isinstance(g, str) or not g for g in scene_groups)):
            raise ValueError("Explicit nonempty scene groups are required")
        if type(gamma) not in (float, int) or not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("The actual algorithm gamma must be finite in (0,1]")
        if not isinstance(source_identity, dict) or not source_identity:
            raise ValueError("Explicit runtime/source identity is required")
        self.groups = list(dict.fromkeys(scene_groups))
        self.scene_groups, self.device = tuple(scene_groups), torch.device(device)
        self.gamma, self.source_identity = float(gamma), deepcopy(source_identity)
        self.count, self.last_interval = len(scene_groups), None
        self.group_ids = torch.tensor([self.groups.index(g) for g in scene_groups], device=self.device)
        self.expected_ids = torch.full((self.count,), -1, dtype=torch.long, device=self.device)
        self.expected_ticks = torch.ones(self.count, dtype=torch.long, device=self.device)
        self.started = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self.from_birth = torch.zeros_like(self.started)
        self.closed = torch.zeros_like(self.started)
        self.term_names = None
        self.invalid_reason = None
        self.observed_transitions = 0
        self.max_reconciliation_error = torch.zeros((), dtype=torch.float64, device=self.device)
        self.live_returns = torch.zeros(self.count, 3, 2, dtype=torch.float64, device=self.device)
        shape = (len(self.groups), len(OUTCOMES))
        self.complete_count = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.partial_count = torch.zeros_like(self.complete_count)
        self.return_sums = torch.zeros(*shape, 3, 2, dtype=torch.float64, device=self.device)
        self.return_min = torch.full(shape, torch.inf, dtype=torch.float64, device=self.device)
        self.return_max = torch.full(shape, -torch.inf, dtype=torch.float64, device=self.device)
        self.last_completed = None

    def _tensor(self, name, value, *, boolean=False, integer=False):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.count,)
                or value.device != self.device):
            raise ValueError(f"{name} requires [{self.count}] on the journal device")
        if boolean and value.dtype != torch.bool:
            raise ValueError(f"{name} requires bool")
        if integer and value.dtype != torch.long:
            raise ValueError(f"{name} requires int64")
        if not boolean and not integer and value.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} requires floating point payments")
        return value.detach()

    def _require_valid(self):
        if self.invalid_reason is not None:
            raise RuntimeError(f"Episode payment journal is invalid: {self.invalid_reason}")

    @torch.no_grad()
    def observe_transition(self, reward, done, *, payments, payload):
        self._require_valid()
        if (not isinstance(payload, dict) or payload.get("schema") != TRANSITION_SCHEMA
                or payload.get("payment_units") != PAYMENT_UNITS or payload.get("policy_dt_s") != .02):
            raise ValueError("Explicit final-payment units and 20ms transition schema required")
        tick = payload.get("policy_interval_id")
        if (type(tick) is not int or tick <= 0
                or self.last_interval is not None and tick != self.last_interval + 1):
            raise ValueError("Policy intervals must advance exactly once")
        reward = self._tensor("reward", reward)
        done = self._tensor("done", done, boolean=True)
        flags = {n: self._tensor(n, payload[n], boolean=True) for n in
                 ("done", "terminated", "timeouts", "success", "boundary", "policy_learning_mask", "reset_mask")}
        ids = self._tensor("episode_ids", payload["episode_ids"], integer=True)
        returned = self._tensor("returned_episode_ids", payload["returned_episode_ids"], integer=True)
        ticks = self._tensor("episode_ticks", payload["episode_ticks"], integer=True)
        if (not isinstance(payments, dict) or not payments
                or any(not isinstance(n, str) or not n for n in payments)):
            raise ValueError("A complete named final payment mapping is required")
        names = tuple(sorted(payments))
        if self.term_names is not None and names != self.term_names:
            raise ValueError("Final payment names changed within the worker")
        parts = [self._tensor(n, payments[n]) for n in names]
        if any(p.dtype != reward.dtype for p in parts):
            raise ValueError("Final parts and returned reward must share dtype")
        matrix = torch.stack(parts, -1).to(torch.float64)
        actual = reward.to(torch.float64)
        total = matrix.sum(-1)
        magnitude = matrix.abs().sum(-1) + actual.abs()
        residual = (total - actual).abs()
        # Floating error bound, not a behavioural acceptance threshold. Covers
        # both the native accumulation tree and summing already rounded parts.
        u = torch.finfo(reward.dtype).eps / 2
        operations = 8 + 2 * len(names)
        bound = operations * u / (1 - operations * u) * magnitude
        valid = (torch.isfinite(matrix).all(-1) & torch.isfinite(actual)
            & torch.isfinite(total) & torch.isfinite(magnitude)
            & torch.isfinite(residual) & torch.isfinite(bound) & (residual <= bound) & (ids >= 0) & (ticks >= 1)
            & (flags["done"] == done) & (flags["reset_mask"] == (returned != ids))
            & (~flags["reset_mask"] | (done & (returned == ids + 1)))
            & (flags["reset_mask"] | (returned == ids))
            & (~flags["terminated"] | done)
            & (~self.started | ((ids == self.expected_ids) & (ticks == self.expected_ticks) & ~self.closed)))
        if not bool(valid.all()):
            raise ValueError("Final reward parts, flags or pre/reset episode identity do not reconcile")
        # A failed allocation/reduction after mutation must not leave usable
        # partial evidence. The worker cannot publish or resume this journal.
        try:
            if self.term_names is None:
                self.term_names = names
                self.live_terms = torch.zeros(self.count, 3, 2, len(names), dtype=torch.float64, device=self.device)
                self.term_sums = torch.zeros(len(self.groups), len(OUTCOMES), 3, 2, len(names),
                                             dtype=torch.float64, device=self.device)
            self.from_birth |= ~self.started & (ticks == 1)
            discount = torch.pow(actual.new_tensor(self.gamma), ticks - 1)
            owned = flags["policy_learning_mask"].to(torch.float64)
            stream_masks = torch.stack((torch.ones_like(owned), owned, 1 - owned), -1)
            factors = stream_masks[:, :, None] * torch.stack((torch.ones_like(discount), discount), -1)[:, None, :]
            self.live_returns += actual[:, None, None] * factors
            self.live_terms += matrix[:, None, None, :] * factors[:, :, :, None]
            self.max_reconciliation_error = torch.maximum(self.max_reconciliation_error, residual.max())
            outcome = torch.full_like(ids, OUTCOMES.index("unclassified_done"))
            for name, mask in (("timeout", flags["timeouts"]),
                               ("boundary_truncation", flags["boundary"]),
                               ("reported_success", flags["success"]), ("failure", flags["terminated"])):
                outcome = torch.where(mask, OUTCOMES.index(name), outcome)
            full = done & self.from_birth
            partial = done & ~self.from_birth
            indices = self.group_ids * len(OUTCOMES) + outcome
            self.complete_count.view(-1).scatter_add_(0, indices, full.long())
            self.partial_count.view(-1).scatter_add_(0, indices, partial.long())
            self.return_sums.view(-1, 3, 2).index_add_(0, indices, self.live_returns * full[:, None, None])
            self.term_sums.view(-1, 3, 2, len(names)).index_add_(0, indices, self.live_terms * full[:, None, None, None])
            self.return_min.view(-1).scatter_reduce_(0, indices, self.live_returns[:, 0, 0].masked_fill(~full, torch.inf),
                                                    reduce="amin", include_self=True)
            self.return_max.view(-1).scatter_reduce_(0, indices, self.live_returns[:, 0, 0].masked_fill(~full, -torch.inf),
                                                    reduce="amax", include_self=True)
            self.last_completed = {"completed_mask": done.clone(), "full_from_birth": self.from_birth.clone(),
                "episode_ids": ids.clone(), "episode_ticks": ticks.clone(), "group_ids": self.group_ids.clone(),
                "outcome_indices": outcome.clone(), "returns": self.live_returns.clone(), "terms": self.live_terms.clone()}
            reset = flags["reset_mask"]
            self.live_returns[reset] = 0
            self.live_terms[reset] = 0
            self.from_birth = torch.where(reset, True, self.from_birth)
            self.closed = done & ~reset
            self.expected_ids.copy_(returned)
            self.expected_ticks.copy_(torch.where(reset, 1, ticks + 1))
            self.started.fill_(True)
            aggregates = (self.live_returns, self.live_terms, self.return_sums, self.term_sums)
            if not bool(torch.stack([torch.isfinite(v).all() for v in aggregates]).all()):
                raise ValueError("Cumulative episode payment accounting overflowed")
            self.last_interval = tick
            self.observed_transitions += 1
        except BaseException as exc:
            self.invalid_reason = f"mutation_failed:{type(exc).__name__}"
            raise

    def metrics(self):
        """RSL logger scalars, updated from completed full episodes only."""
        self._require_valid()
        result = {"/reward_episode/policy_intervals": self.observed_transitions,
                  "/reward_episode/max_step_sum_residual": self.max_reconciliation_error}
        # Keep the training tag population independent of scene count. Named
        # per-group totals and individual last-terminal rows stay in checkpoints.
        counts, partial = self.complete_count.sum(0), self.partial_count.sum(0)
        returns = self.return_sums.sum(0)
        terms = self.term_sums.sum(0) if self.term_names is not None else None
        reductions = (returns,) if terms is None else (returns, terms)
        if not bool(torch.stack([torch.isfinite(value).all() for value in reductions]).all()):
            self.invalid_reason = "publication_failed:cross_scene_overflow"
            raise ValueError("Cross-scene episode payment publication overflowed")
        for j, outcome in enumerate(OUTCOMES):
            prefix = f"/reward_episode/{outcome}/"
            count = counts[j]
            result[prefix + "full_count"] = count
            result[prefix + "partial_count"] = partial[j]
            denominator = count.clamp_min(1)
            for k, stream in enumerate(STREAMS):
                result[prefix + stream + "_return_mean"] = returns[j, k, 0] / denominator
                result[prefix + stream + "_discounted_return_mean"] = returns[j, k, 1] / denominator
            if self.term_names is not None:
                for k, name in enumerate(self.term_names):
                    result[prefix + "term/" + name + "_mean"] = terms[j, 0, 0, k] / denominator
        return result

    def checkpoint_state(self):
        self._require_valid()
        state = {"schema": VERSION, "gamma": self.gamma, "scene_groups": list(self.scene_groups),
            "source_identity": deepcopy(self.source_identity), "term_names": self.term_names,
            "last_interval": self.last_interval, "observed_transitions": self.observed_transitions,
            "exact_resume_supported": False, "scope": "physical_payment_measurements_current_worker_not_PPO_targets"}
        for name in ("expected_ids", "expected_ticks", "started", "from_birth", "closed", "live_returns",
                     "complete_count", "partial_count", "return_sums", "return_min", "return_max",
                     "max_reconciliation_error", "live_terms", "term_sums"):
            if hasattr(self, name):
                state[name] = getattr(self, name).clone()
        state["last_completed"] = ({k: v.clone() for k, v in self.last_completed.items()}
                                   if self.last_completed is not None else None)
        return state

    def summary(self):
        self._require_valid()
        return {"schema": VERSION, "gamma": self.gamma, "observed_policy_intervals": self.observed_transitions,
            "source_identity": deepcopy(self.source_identity), "term_names": list(self.term_names or ()),
            "full_episode_count": int(self.complete_count.sum()), "partial_episode_count": int(self.partial_count.sum()),
            "max_step_sum_residual": float(self.max_reconciliation_error),
            "scope": "physical_payment_measurements_current_worker_not_PPO_targets",
            "actor_discount_clock": "original_episode_ticks_including_excluded_intervals",
            "exact_resume_supported": False, "physical_or_skill_qualification": False}
