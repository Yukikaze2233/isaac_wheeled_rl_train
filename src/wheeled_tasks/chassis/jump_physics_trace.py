"""Bounded, opt-in physics samples for jump diagnosis; no skill classification.

The trace retains the four 5 ms samples of the last 20 ms policy interval.
Composer inputs describe dispatch, not measured solver forces. Net contact force
can cancel; terrain pair counts cover only the caller's named filter paths.
Neither sampled contacts nor a zero wrench establish unassisted whole-body flight.
"""
from __future__ import annotations

from copy import deepcopy
from functools import wraps
import hashlib
import json
from pathlib import Path

import torch

from .jump_actuation_evidence import SCHEMA as ACTUATION_SCHEMA


SCHEMA = "jump-physics-trace-v1"
COMPOSERS = ("instantaneous", "permanent")
WRENCH_FIELDS = ("global_force_w", "global_torque_w", "global_force_at_com_w",
                 "local_force_b", "local_torque_b")
REQUEST_FIELDS = ("episode_ids", "request_ids", "source_request_ids", "request_source",
                  "request_kind_ids", "route_stage_ids", "accepted_tick")


def _mutation(method):
    """An incomplete copy cannot later be reported as a complete measurement."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        self._require_valid()
        try:
            return method(self, *args, **kwargs)
        except Exception as error:
            self._fault = f"{method.__name__}: {type(error).__name__}: {error}"
            raise
    return guarded


class JumpPhysicsTrace:
    def __init__(self, count, device, *, body_names, surface_counts, physics_dt,
                 policy_dt, source_identity):
        if (type(count) is not int or count <= 0 or physics_dt != .005 or policy_dt != .02
                or isinstance(physics_dt, bool) or isinstance(policy_dt, bool)):
            raise ValueError("Trace requires positive count and explicit 200/50 Hz timing")
        if (not isinstance(body_names, (list, tuple)) or not body_names
                or any(not isinstance(name, str) or not name for name in body_names)
                or len(set(body_names)) != len(body_names)):
            raise ValueError("Distinct native body names in contact sensor order are required")
        if (not isinstance(surface_counts, (list, tuple)) or len(surface_counts) != count
                or any(type(n) is not int or n <= 0 for n in surface_counts)):
            raise ValueError("One positive terrain filter count per environment is required")
        if not isinstance(source_identity, dict) or not source_identity:
            raise ValueError("Configuration, contact view and dispatch source identity required")
        self.count, self.device = count, torch.device(device)
        self.body_count, self.surface_count = len(body_names), max(surface_counts)
        self._identity = {
            "schema": SCHEMA, "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "configuration_and_callsite": deepcopy(source_identity),
            "body_names": list(body_names), "surface_counts": list(surface_counts),
            "physics_dt_s": .005, "policy_dt_s": .02,
            "sample_order": "composer_inputs_before_dispatch_then_contacts_and_kinematics_after_step",
            "wrench_frames": "world_origin_torque_and_world_at_COM_force_separate_from_local_link_wrench",
            "contacts": "all_partner_net_vectors_and_named_terrain_pairs_not_contact_absence_proof",
            "contact_indices": "shared_view_patch_offsets_not_lane_local; zero_count_may_have_negative_start",
            "velocity_writes": "known_legacy_push_dispatch_and_native_readback_not_impulse_measurement",
            "coverage": "known_callsites_only; reset_state_writes_and_substep_script_ownership_not_recorded",
            "retention": "last_policy_interval_and_pending_boundary_writes_only",
            "sample_valid": "finite_numeric_samples_and_patch_index_bounds_only_not_complete_contacts_or_valid_dynamics",
            "velocity_write_validity": "separate_per_event_finite_flags; not_implied_by_sample_valid",
            "qualification": "measurement_only_no_active_takeoff_skill_pass_reward_or_PPO_credit",
        }
        encoded = json.dumps(self._identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.source_identity_sha256 = hashlib.sha256(encoded).hexdigest()
        self._fault = None
        self._last_tick = None
        self._open = self._prepared = False
        self._slot = 0
        self._actuation_source = None
        self.episode_ids = torch.full((count,), -1, dtype=torch.long, device=self.device)
        counts = torch.tensor(surface_counts, device=self.device)
        self._surface_mask = torch.arange(self.surface_count, device=self.device)[None, None, :] < counts[:, None, None]
        self._samples = {
            "body_net_force_n": self._buffer(self.body_count, 3),
            "surface_force_n": self._buffer(self.body_count, self.surface_count, 3),
            "surface_contact_counts": self._buffer(self.body_count, self.surface_count, dtype=torch.long),
            "surface_contact_starts": self._buffer(self.body_count, self.surface_count, dtype=torch.long),
            "contact_data_capacity": self._buffer(dtype=torch.long),
            "com_position_world_m": self._buffer(3), "com_velocity_world_m_s": self._buffer(3),
            "base_link_pose_world": self._buffer(7), "wheel_centers_world_m": self._buffer(2, 3),
        }
        self._composers = {name: {"active": torch.zeros(4, dtype=torch.bool, device=self.device),
            **{field: self._buffer(self.body_count, 3) for field in WRENCH_FIELDS}} for name in COMPOSERS}
        self._sample_valid = self._buffer(dtype=torch.bool)
        self._ticks = torch.zeros(4, dtype=torch.long, device=self.device)
        self._pending_writes, self._interval_writes = [], []
        self._last_transition = None
        self._closed_intervals = 0
        self._worker_invalid = torch.zeros((), dtype=torch.long, device=self.device)
        self._discarded_write_rows = 0

    def _buffer(self, *tail, dtype=torch.float32):
        return torch.zeros((4, self.count, *tail), dtype=dtype, device=self.device)

    def _require_valid(self):
        if self._fault is not None:
            raise RuntimeError(f"Jump physics trace is invalid: {self._fault}")

    def _tensor(self, name, value, shape, dtype):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != dtype or value.device != self.device):
            raise ValueError(f"{name} has wrong shape/dtype/device")

    def _current_tick(self, tick):
        if type(tick) is not int or tick < 0 or (self._last_tick is not None and tick != self._last_tick):
            raise ValueError("Expected the current completed physical tick")

    def _ids(self, ids):
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1:
            raise ValueError("Expected a vector of environment ids")
        self._tensor("ids", ids, (ids.numel(),), torch.long)
        if bool(((ids < 0) | (ids >= self.count)).any()) or len(ids.unique()) != len(ids):
            raise ValueError("Environment ids must be distinct and in range")

    @_mutation
    def reset(self, ids, *, physical_tick):
        self._current_tick(physical_tick)
        self._ids(ids)
        if self._open:
            raise ValueError("Close the pre-reset policy trace before resetting")
        # A state reset supersedes a pending boundary write for those old lanes.
        remaining = []
        for event in self._pending_writes:
            keep = ~torch.isin(event["ids"], ids)
            self._discarded_write_rows += int((~keep).sum().item())
            if bool(keep.any()):
                remaining.append({k: v[keep].clone() if isinstance(v, torch.Tensor) else v
                                  for k, v in event.items()})
        self._pending_writes = remaining
        self.episode_ids[ids] += 1
        self._last_tick = physical_tick

    @_mutation
    def record_velocity_write(self, ids, *, physical_tick, source, before_world,
                              commanded_world, readback_world):
        self._current_tick(physical_tick)
        self._ids(ids)
        if self._open or source != "legacy_push" or bool((self.episode_ids[ids] < 0).any()):
            raise ValueError("Known post-reset boundary velocity writes only")
        values = {"before_world": before_world, "commanded_world": commanded_world, "readback_world": readback_world}
        for name, value in values.items():
            self._tensor(name, value, (ids.numel(), 6), torch.float32)
        self._pending_writes.append({"ids": ids.clone(), "episode_ids": self.episode_ids[ids].clone(),
            "physical_tick": physical_tick, "first_following_physics_tick": physical_tick + 1,
            "source": source, **{name: value.detach().clone() for name, value in values.items()},
            "finite": torch.stack([torch.isfinite(v).all(-1) for v in values.values()]).all(0)})

    @_mutation
    def begin_policy(self):
        if self._open or self._last_tick is None or bool((self.episode_ids < 0).any()):
            raise ValueError("All lanes must be reset and the previous policy interval closed")
        self._open, self._prepared, self._slot = True, False, 0
        self._interval_writes, self._pending_writes = self._pending_writes, []
        self._last_transition = None

    @_mutation
    def prepare_physics(self, *, physical_tick, composer_inputs):
        self._current_tick(physical_tick)
        if not self._open or self._prepared or self._slot >= 4:
            raise ValueError("Each policy interval requires exactly four prepare/finish pairs")
        if not isinstance(composer_inputs, dict) or set(composer_inputs) != set(COMPOSERS):
            raise ValueError("Both instantaneous and permanent composer inputs are required")
        for name in COMPOSERS:
            row = composer_inputs[name]
            if (not isinstance(row, dict) or set(row) != {"active", *WRENCH_FIELDS}
                    or type(row["active"]) is not bool):
                raise ValueError("Raw composer fields and a global active boolean are required")
            for field in WRENCH_FIELDS:
                self._tensor(field, row[field], (self.count, self.body_count, 3), torch.float32)
        valid = torch.ones(self.count, dtype=torch.bool, device=self.device)
        for name, row in composer_inputs.items():
            self._composers[name]["active"][self._slot] = row["active"]
            for field in WRENCH_FIELDS:
                self._composers[name][field][self._slot].copy_(row[field].detach())
                valid &= torch.isfinite(row[field]).flatten(1).all(-1)
        self._sample_valid[self._slot].copy_(valid)
        self._prepared = True

    @_mutation
    def finish_physics(self, *, physical_tick, **samples):
        if (not self._open or not self._prepared or type(physical_tick) is not int
                or physical_tick != self._last_tick + 1 or set(samples) != set(self._samples)):
            raise ValueError("Finish needs the next physical tick and every raw post-step field")
        for name, buffer in self._samples.items():
            self._tensor(name, samples[name], buffer.shape[1:], buffer.dtype)
        valid = self._sample_valid[self._slot].clone()
        for name, buffer in self._samples.items():
            value = samples[name]
            buffer[self._slot].copy_(value.detach())
            if value.is_floating_point():
                valid &= torch.isfinite(value).flatten(1).all(-1)
        counts, starts = samples["surface_contact_counts"], samples["surface_contact_starts"]
        capacity = samples["contact_data_capacity"][:, None, None]
        # Subtraction avoids signed integer overflow in start + count.
        index_valid = ((counts >= 0) & (counts <= capacity) & ((counts == 0)
            | ((starts >= 0) & (starts <= capacity - counts))))
        index_valid = torch.where(self._surface_mask, index_valid, (counts == 0) & (starts == 0))
        valid &= index_valid.flatten(1).all(-1) & (samples["contact_data_capacity"] > 0)
        padding_zero = torch.where(self._surface_mask[..., None],
            torch.ones_like(samples["surface_force_n"], dtype=torch.bool), samples["surface_force_n"] == 0)
        valid &= padding_zero.flatten(1).all(-1)
        self._sample_valid[self._slot].copy_(valid)
        self._ticks[self._slot] = physical_tick
        self._last_tick, self._slot, self._prepared = physical_tick, self._slot + 1, False

    @_mutation
    def policy_transition(self, done, terminated, timeouts, *, jump_transition):
        if not self._open or self._prepared or self._slot != 4:
            raise ValueError("Only a complete four-sample policy interval can close")
        for name, value in (("done", done), ("terminated", terminated), ("timeouts", timeouts)):
            self._tensor(name, value, (self.count,), torch.bool)
        jump = jump_transition
        if (not isinstance(jump, dict) or jump.get("schema") != ACTUATION_SCHEMA
                or jump.get("physical_tick") != self._last_tick or jump.get("physics_dt_s") != .005):
            raise ValueError("Expected the matching actual actuation measurement transition")
        source = jump.get("source_identity_sha256")
        if (not isinstance(source, str) or len(source) != 64 or any(c not in "0123456789abcdef" for c in source)
                or self._actuation_source not in (None, source)):
            raise ValueError("Actuation source identity is missing or changed within the worker")
        for name in REQUEST_FIELDS + ("policy_physics_samples",):
            self._tensor(name, jump.get(name), (self.count,), torch.long)
        for name in ("policy_learning_mask", "done", "terminated", "timeouts"):
            self._tensor(name, jump.get(name), (self.count,), torch.bool)
        if (not torch.equal(jump["episode_ids"], self.episode_ids)
                or not bool((jump["policy_physics_samples"] == 4).all())
                or any(not torch.equal(jump[k], v) for k, v in (("done", done), ("terminated", terminated), ("timeouts", timeouts)))
                or bool(((terminated | timeouts) & ~done).any())):
            raise ValueError("Actuation trace episode, samples or termination boundary does not match")
        if type(jump.get("policy_interval_id")) is not int or jump["policy_interval_id"] < 0:
            raise ValueError("Actual environment policy interval id is required")
        if jump.get("policy_learning_mask_source") not in ("explicit_environment_mask", "PPO_default_all_unmasked"):
            raise ValueError("Final PPO ownership mask source is required")
        result = {"schema": SCHEMA, "scope": "measurement_only", "source_identity_sha256": self.source_identity_sha256,
            "actuation_source_identity_sha256": source, "physical_tick": self._last_tick,
            "physics_dt_s": .005, "policy_dt_s": .02, "policy_interval_id": jump["policy_interval_id"],
            **{name: jump[name].clone() for name in REQUEST_FIELDS},
            "policy_learning_mask": jump["policy_learning_mask"].clone(),
            "policy_learning_mask_source": jump["policy_learning_mask_source"],
            "done": done.clone(), "terminated": terminated.clone(), "timeouts": timeouts.clone(),
            "sample_ticks": self._ticks.clone(), "sample_valid": self._sample_valid.clone(),
            "samples": {name: value.clone() for name, value in self._samples.items()},
            "composer_inputs": deepcopy(self._composers), "boundary_velocity_writes": deepcopy(self._interval_writes)}
        self._last_transition = deepcopy(result)
        self._actuation_source = source
        self._closed_intervals += 1
        self._worker_invalid += (~self._sample_valid).sum()
        self._open = False
        return result

    def metrics(self):
        self._require_valid()
        if self._open or self._last_transition is None:
            raise ValueError("Metrics require a closed policy trace")
        prefix = "/jump_physics_trace/measurement_only/"
        return {prefix + "physics_samples": torch.tensor(4 * self.count, device=self.device),
            prefix + "invalid_samples": (~self._sample_valid).sum(),
            prefix + "boundary_velocity_write_rows": torch.tensor(
                sum(len(event["ids"]) for event in self._interval_writes), device=self.device),
            prefix + "nonfinite_boundary_velocity_write_rows": sum(
                ((~event["finite"]).sum() for event in self._interval_writes),
                torch.zeros((), dtype=torch.long, device=self.device))}

    def summary(self):
        self._require_valid()
        return {"schema": "jump-physics-trace-summary-v1", "source_identity_sha256": self.source_identity_sha256,
            "source_identity": deepcopy(self._identity), "scope": "current_worker_measurement_only_no_exact_resume",
            "physical_tick": self._last_tick, "open_policy_interval": self._open,
            "completed_policy_intervals": self._closed_intervals,
            "invalid_completed_samples": int(self._worker_invalid.item()),
            "pending_boundary_write_rows": sum(len(event["ids"]) for event in self._pending_writes),
            "pending_write_rows_superseded_by_reset": self._discarded_write_rows}

    def checkpoint_state(self):
        self._require_valid()
        if self._open:
            raise ValueError("Cannot checkpoint a partial physics trace")
        return {"schema": "jump-physics-trace-checkpoint-v1", "summary": self.summary(),
            "current_episode_ids": self.episode_ids.clone(), "last_pre_reset_transition": deepcopy(self._last_transition),
            "pending_boundary_velocity_writes": deepcopy(self._pending_writes),
            "native_physical_exact_resume_qualified": False}
