"""CPU-testable source flat reset/command owner for production environment wiring.

Source requests come from pinned flat baseline. Mapping, collision placement and
bounded rejection are declared local adaptations. This is not native admission.

Source reset IK increases both Y-axis active angles as raw leg_angle increases;
the actual USD wheel center moves toward -X. The mapper uses base-forward as
positive, so ONLY the reset adapter negates this source angle. Source dimensions
and their small zero-angle offsets are not copied into this chassis geometry.
"""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from wheeled_tasks.chassis.tasks.commands.restart_state import _restart_tensor
from wheeled_tasks.chassis.tasks.commands.flat_command_sampling import ScutFlatCommands
from wheeled_tasks.chassis.tasks.recovery.nonsteady_initial_state import (
    CONTROL_JOINT_NAMES, FlatBirthSampler, FlatBirthRequests, RESET_VERSION,
)


EPISODE_STATE_VERSION = "own_flat_episode_restart_v1"
FINAL_EPISODE_STATE_VERSION = "v7-final-rough-episode-restart-v1"
FINAL_BIRTH_ADAPTER_VERSION = "v7-final-rough-accepted-births-v1"
MECHANICAL_RESTART_VERSION = "own_flat_mechanical_restart_identity_v1"
_EPISODE_TENSORS = {
    **{name: (torch.long, None) for name in ("episode_id", "birth_proposal_count", "birth_rejection_count")},
    **{name: (torch.float32, None) for name in ("placement_raise_m", "minimum_shape_clearance_m")},
    **{name: (torch.float32, 2) for name in ("birth_source_angle_rad", "birth_own_angle_rad")},
}
_REQUEST_FIELDS = {"env_ids": (torch.long, None), "ground_birth": (torch.bool, None),
    "leg_mode": (torch.int8, 2), **{name: (torch.float32, width) for name, width in (
    ("virtual_leg_length_m", 2), ("virtual_leg_angle_rad", 2), ("virtual_foot_height_m", 2),
    ("wheel_angle_rad", 2), ("control_velocity_rad_s", 6), ("source_root_position_m", 3),
    ("source_root_rpy_rad", 3), ("source_root_velocity6", 6))}}
_FINAL_REQUEST_FIELDS = {"birth_kind": (torch.int8, None), "air_mode_index": (torch.long, None)}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mechanical_identity(value):
    fields = {"asset_manifest_sha256", "model_spec_sha256", "branch_table_sha256", "adapter_configuration_sha256"}
    if (not isinstance(value, dict) or set(value) != fields | {"schema"}
            or value["schema"] != MECHANICAL_RESTART_VERSION
            or any(not isinstance(value[key], str) or len(value[key]) != 64
                or any(c not in "0123456789abcdef" for c in value[key]) for key in fields)):
        raise ValueError("Restart requires a construction-bound mechanical identity")
    return deepcopy(value)


@dataclass(frozen=True)
class FlatResetState:
    env_ids: torch.Tensor
    requests: FlatBirthRequests
    root_pose: torch.Tensor
    root_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    proposal_counts: torch.Tensor
    placement_raise_m: torch.Tensor
    minimum_shape_clearance_m: torch.Tensor


class FlatEpisodeRuntime:
    """One owner; externally written reset states are committed exactly once.

    ``abort`` never changes a sampled request. ``redraw`` resamples only refused
    lanes using the same source request distribution and caller RNG; it changes
    the accepted distribution and reports refusals. Neither clips requests nor
    substitutes nominal poses. Contract selection remains subject to review.
    """
    def __init__(self, num_envs, device, mapper, placement, tree_joint_names, *, birth_rejection,
                 restart_identity=None, commands=None):
        self.commands = ScutFlatCommands(num_envs, device) if commands is None else commands
        if (not isinstance(self.commands, ScutFlatCommands) or self.commands.num_envs != num_envs
                or self.commands.device != torch.device(device)):
            raise ValueError("Flat command owner must match the fixed episode lanes/device")
        self.sampler = FlatBirthSampler(num_envs, device)
        self.mapper, self.placement = mapper, placement
        self.device, self.num_envs = self.commands.device, num_envs
        self.tree_joint_names = tuple(tree_joint_names)
        if len(self.tree_joint_names) != 18 or len(set(self.tree_joint_names)) != 18:
            raise ValueError("Flat episode adapter requires the full named 18-axis tree")
        self.rejection = dict(birth_rejection)
        if self.rejection == {"mode": "abort"}:
            self.max_attempts = 1
        elif (set(self.rejection) == {"mode", "max_attempts"} and self.rejection["mode"] == "redraw"
                and type(self.rejection["max_attempts"]) is int and self.rejection["max_attempts"] > 0):
            self.max_attempts = self.rejection["max_attempts"]
        else:
            raise ValueError("Flat birth rejection needs an explicit bounded adapter")
        self.episode_id = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.birth_proposal_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.birth_rejection_count = torch.zeros_like(self.birth_proposal_count)
        self.placement_raise_m = torch.zeros(num_envs, device=self.device)
        self.minimum_shape_clearance_m = torch.zeros_like(self.placement_raise_m)
        self.birth_source_angle_rad = torch.zeros(num_envs, 2, device=self.device)
        self.birth_own_angle_rad = torch.zeros_like(self.birth_source_angle_rad)
        # Reset-request telemetry only: current worker, never resumed or used
        # for ownership/PPO. ground=False includes nominal and source requests.
        # A restored pending request can commit here without a local proposal.
        self._birth_coverage = {
            name: torch.zeros((), dtype=torch.long, device=self.device) for name in (
                "proposal_count", "rejected_proposal_count", "unclassified_rejected_proposal_count",
                "accepted_reset_count", "accepted_ground_false_reset_count", "accepted_ground_true_reset_count")}
        self._birth_coverage.update(
            length_min=torch.full((2,), torch.inf, device=self.device),
            length_max=torch.full((2,), -torch.inf, device=self.device),
            velocity_abs_max=torch.zeros(6, device=self.device), rejection_reasons={})
        self._pending = None
        # Factories freeze verified bytes at construction. A caller cannot
        # supply a new identity only when loading a checkpoint.
        self._restart_mechanical_identity = (None if restart_identity is None
                                             else _mechanical_identity(restart_identity))

    @property
    def pending_reset(self):
        """The exact pending proposal object, including after owner restoration.

        Returning it does not prove whether native reset writes have happened.
        The outer restart coordinator must own that transaction boundary.
        """
        return self._pending

    @property
    def final_source_birth_enabled(self):
        return hasattr(self, "_final_birth")

    def enable_final_source_births(self, *, birth_definition_sha256):
        """Select explicit final requests without choosing lanes or a new course.

        Retained source-flat lanes still use the existing flat distribution.
        The caller's bounded rejection policy remains explicit; this adapter
        does not select the pending attainable-length conditioning decision.
        """
        from wheeled_tasks.chassis.tasks.recovery.final_rough_birth import FinalRoughBirthSampler
        from wheeled_tasks.chassis.tasks.commands.final_rough_commands import FinalRoughCommands
        if (self.final_source_birth_enabled or self._pending is not None
                or bool((self.episode_id >= 0).any()) or bool(self.birth_proposal_count.any())
                or not isinstance(self.commands, FinalRoughCommands)):
            raise ValueError("Final source births require a fresh final command/reset owner")
        sampler = FinalRoughBirthSampler(self.num_envs, self.device)
        if sampler.definition_sha256 != birth_definition_sha256:
            raise ValueError("Final source births require the authenticated effective definition")
        state = {"sampler": sampler,
            "birth_kind": torch.full((self.num_envs,), -1, dtype=torch.int8, device=self.device),
            "air_mode_index": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)}
        self.commands.enable_air_birth_command_limits(birth_definition_sha256=birth_definition_sha256)
        self._final_birth = state

    def _request_fields(self):
        return {**_REQUEST_FIELDS, **_FINAL_REQUEST_FIELDS} if self.final_source_birth_enabled else _REQUEST_FIELDS

    def _requests_from_fields(self, values):
        if self.final_source_birth_enabled:
            from wheeled_tasks.chassis.tasks.recovery.final_rough_birth import FinalRoughBirthRequests
            return FinalRoughBirthRequests(**values)
        return FlatBirthRequests(**values)

    @staticmethod
    def _validate_final_labels(values):
        kind, mode, ground = values["birth_kind"], values["air_mode_index"], values["ground_birth"]
        if (kind.dtype != torch.int8 or mode.dtype != torch.long or ground.dtype != torch.bool
                or kind.shape != mode.shape or kind.shape != ground.shape
                or bool(((kind < -2) | (kind > 2)).any())
                or not torch.equal(mode, torch.where(kind == 2, 0, -1))
                or bool(((kind >= 0) & (ground != (kind == 1))).any())
                or bool(((kind == -2) & ground).any())):
            raise ValueError("Final birth kind, air mode and ground request disagree")

    def _sample_birth_requests(self, ids, generator, iteration):
        if not self.final_source_birth_enabled:
            return self.sampler.sample(ids, generator)
        if bool(self.commands.fixed_request_mask[ids].any()):
            raise ValueError("Fixed rehearsal lanes require their explicit nominal reset subset")
        retained = self.commands._final.get("source_flat_mask", torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device))[ids]
        fields = None
        kinds = torch.full((len(ids),), -1, dtype=torch.int8, device=self.device)
        modes = torch.full((len(ids),), -1, dtype=torch.long, device=self.device)
        # Order is versioned locally: final requests first, retained flat next.
        for selected, final in ((~retained, True), (retained, False)):
            rows = selected.nonzero(as_tuple=False).flatten()
            if not len(rows):
                continue
            selected_ids = ids[rows]
            if final:
                batch = self._final_birth["sampler"].sample(selected_ids, generator,
                    iteration=iteration, air_allowed=self.commands.air_birth_allowed(selected_ids),
                    ground_allowed=self.commands.ground_birth_allowed(selected_ids),
                    axis_allowed=self.commands.axis_reset_mask(selected_ids))
                requests = batch.requests
                kinds[rows], modes[rows] = batch.birth_kind, batch.air_mode_index
            else:
                requests = self.sampler.sample(selected_ids, generator)
            if fields is None:
                fields = {name: value.new_empty((len(ids), *value.shape[1:]))
                          for name, value in vars(requests).items()}
            for name, value in vars(requests).items():
                fields[name][rows] = value
        return self._requests_from_fields({**fields, "birth_kind": kinds, "air_mode_index": modes})

    def _restart_identity(self):
        if self._restart_mechanical_identity is None:
            raise ValueError("Flat restart identity was not bound at construction")
        if set(vars(self.sampler)) != {"num_envs", "device"}:
            raise ValueError("Birth sampler gained mutable state; version its restart schema")
        if self.sampler.num_envs != self.num_envs or self.sampler.device != self.device:
            raise ValueError("Birth sampler restart identity mismatch")
        expected = set(_EPISODE_TENSORS) | {"commands", "sampler", "mapper", "placement", "device", "num_envs",
            "tree_joint_names", "rejection", "max_attempts", "_pending", "_restart_mechanical_identity",
            "_birth_coverage"}
        if self.final_source_birth_enabled:
            expected.add("_final_birth")
        if set(vars(self)) != expected:
            raise ValueError("Flat owner restart inventory changed; version the complete state schema")
        for name, (dtype, width) in _EPISODE_TENSORS.items():
            tensor = getattr(self, name)
            shape = (self.num_envs,) if width is None else (self.num_envs, width)
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != self.device
                    or tensor.requires_grad or not tensor.is_contiguous()):
                raise ValueError(f"Destination flat buffer {name} changed shape/dtype/device")
        identity = {"mechanical": deepcopy(self._restart_mechanical_identity), "num_envs": self.num_envs,
            "device": str(self.device), "tree_joint_names": list(self.tree_joint_names),
            "birth_rejection": deepcopy(self.rejection), "max_attempts": self.max_attempts,
            "mapper_dtype": str(self.mapper.dtype), "placement_dtype": str(self.placement.dtype),
            "sampler_version": RESET_VERSION, "source_angle_to_own": "negate_raw_radians"}
        if self.final_source_birth_enabled:
            state = self._final_birth
            if set(state) != {"sampler", "birth_kind", "air_mode_index"}:
                raise ValueError("Final birth owner inventory changed")
            sampler = state["sampler"]
            if (sampler.num_envs != self.num_envs or sampler.device != self.device
                    or set(vars(sampler)) != {"num_envs", "device", "_definition_canonical"}
                    or sampler.definition_sha256 != self.commands._final["air_birth_limits"]["definition_sha256"]):
                raise ValueError("Final sampler and command owner identities disagree")
            for name, (dtype, _) in _FINAL_REQUEST_FIELDS.items():
                value = state[name]
                if (value.shape != (self.num_envs,) or value.dtype != dtype or value.device != self.device
                        or value.requires_grad or not value.is_contiguous()):
                    raise ValueError("Final birth destination buffer changed: " + name)
            identity["final_birth"] = {"version": FINAL_BIRTH_ADAPTER_VERSION,
                "sampler_version": sampler.VERSION, "definition_sha256": sampler.definition_sha256,
                "group_rng_order": "final_then_retained_flat", "root_velocity_frame": "com_world"}
        return identity

    @staticmethod
    def _encode_pending(pending):
        if pending is None:
            return None
        return {name: ({key: value.detach().cpu().clone() for key, value in vars(pending.requests).items()}
                        if name == "requests" else getattr(pending, name).detach().cpu().clone())
                for name in FlatResetState.__dataclass_fields__}

    def _prepare_pending(self, value):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != set(FlatResetState.__dataclass_fields__):
            raise ValueError("Pending reset restart fields mismatch")
        ids = value["env_ids"]
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("Pending reset needs nonempty unique lane ids")
        ids = _restart_tensor(ids, ids.shape, torch.long, "pending env_ids")
        if ids.unique().numel() != ids.numel() or bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("Pending reset lane ids are invalid")
        n = ids.numel()
        raw = value["requests"]
        request_fields = self._request_fields()
        if not isinstance(raw, dict) or set(raw) != set(request_fields):
            raise ValueError("Pending source birth request fields mismatch")
        requests = {name: _restart_tensor(raw[name], (n,) if width is None else (n, width), dtype,
                    "pending request " + name) for name, (dtype, width) in request_fields.items()}
        if (not torch.equal(ids, requests["env_ids"]) or bool((requests["virtual_leg_length_m"] <= 0).any())
                or bool(((requests["leg_mode"] < -1) | (requests["leg_mode"] > 1)).any())
                or not torch.equal(requests["leg_mode"] == -1, ~requests["ground_birth"][:, None].expand(n, 2))):
            raise ValueError("Pending source birth request domain is invalid")
        if self.final_source_birth_enabled:
            self._validate_final_labels(requests)
            self.commands._air_reset_modes(ids.to(self.device), requests["ground_birth"].to(self.device),
                requests["air_mode_index"].to(self.device), self._final_birth["sampler"].definition_sha256)
        fields = {"env_ids": ids}
        for name, width, dtype in (("root_pose", 7, self.placement.dtype), ("root_velocity", 6, self.placement.dtype),
                ("joint_position", 18, self.mapper.dtype), ("joint_velocity", 18, self.mapper.dtype),
                ("proposal_counts", None, torch.long), ("placement_raise_m", None, self.placement.dtype),
                ("minimum_shape_clearance_m", None, self.placement.dtype)):
            fields[name] = _restart_tensor(value[name], (n,) if width is None else (n, width), dtype, "pending " + name)
        if (bool((fields["proposal_counts"] < 1).any()) or bool((fields["proposal_counts"] > self.max_attempts).any())
                or bool((fields["placement_raise_m"] < 0).any())
                or bool((fields["minimum_shape_clearance_m"] < 0).any())
                or not torch.allclose(fields["root_pose"][:, 3:].norm(dim=1), torch.ones(n, dtype=self.placement.dtype),
                                      atol=1e-6, rtol=0.)):
            raise ValueError("Pending reset counts/placement/quaternion are invalid")
        fields = {name: tensor.to(self.device).clone() for name, tensor in fields.items()}
        fields["requests"] = self._requests_from_fields({name: tensor.to(self.device).clone()
                                                       for name, tensor in requests.items()})
        return FlatResetState(**fields)

    def state_dict(self, generator):
        """Capture owned state and shared env RNG; no PhysX/PPO restart claim."""
        result = {"schema": EPISODE_STATE_VERSION, "identity": self._restart_identity(),
            "commands": self.commands.state_dict(generator),
            "tensors": {name: getattr(self, name).detach().cpu().clone() for name in _EPISODE_TENSORS},
            "pending_reset": self._encode_pending(self._pending)}
        if self.final_source_birth_enabled:
            result["schema"] = FINAL_EPISODE_STATE_VERSION
            result["final_birth"] = {name: self._final_birth[name].detach().cpu().clone()
                                     for name in _FINAL_REQUEST_FIELDS}
        self._prepare_restart(result, generator)
        return result

    def _prepare_restart(self, state, generator):
        final = self.final_source_birth_enabled
        fields = {"schema", "identity", "commands", "tensors", "pending_reset"} | ({"final_birth"} if final else set())
        if (not isinstance(state, dict) or set(state) != fields
                or state["schema"] != (FINAL_EPISODE_STATE_VERSION if final else EPISODE_STATE_VERSION)
                or _canonical(state["identity"]) != _canonical(self._restart_identity())):
            raise ValueError("Flat episode restart schema/config/mechanical identity mismatch")
        raw = state["tensors"]
        if not isinstance(raw, dict) or set(raw) != set(_EPISODE_TENSORS):
            raise ValueError("Flat episode restart tensor inventory mismatch")
        values = {name: _restart_tensor(raw[name], (self.num_envs,) if width is None else (self.num_envs, width), dtype,
            name) for name, (dtype, width) in _EPISODE_TENSORS.items()}
        if (bool((values["episode_id"] < -1).any())
                or bool((values["birth_proposal_count"] < 0).any()) or bool((values["birth_rejection_count"] < 0).any())
                or bool((values["birth_rejection_count"] > values["birth_proposal_count"]).any())
                or bool((values["placement_raise_m"] < 0).any()) or bool((values["minimum_shape_clearance_m"] < 0).any())
                or not torch.equal(values["birth_own_angle_rad"], -values["birth_source_angle_rad"])):
            raise ValueError("Flat episode restart counters/placement/angle domains are invalid")
        commands = self.commands._prepare_restart(state["commands"], generator)
        if hasattr(self.commands, "episode_node_indices"):
            # Both owners must describe the same actual reset, including an
            # unassigned lane. Validate before either state or RNG is committed.
            if not torch.equal(values["episode_id"], commands[1]["episode_ids"].detach().cpu()):
                raise ValueError("Flat and curriculum command episode identities disagree")
        pending = self._prepare_pending(state["pending_reset"])
        if pending is not None and bool((pending.proposal_counts.detach().cpu()
                > values["birth_proposal_count"][pending.env_ids.detach().cpu()]).any()):
            raise ValueError("Pending proposal counts exceed cumulative birth exposure")
        prepared = {name: tensor.to(self.device).clone() for name, tensor in values.items()}
        if not final:
            return prepared, commands, pending
        raw_final = state["final_birth"]
        if not isinstance(raw_final, dict) or set(raw_final) != set(_FINAL_REQUEST_FIELDS):
            raise ValueError("Final episode restart needs its complete committed birth context")
        context = {name: _restart_tensor(raw_final[name], (self.num_envs,), dtype, name)
                   for name, (dtype, _) in _FINAL_REQUEST_FIELDS.items()}
        self._validate_final_labels({**context, "ground_birth": commands[0][0]["ground_birth"].cpu()})
        if (not torch.equal(context["air_mode_index"], commands[2]["mode_index"].cpu())
                or bool(((values["episode_id"] < 0) & (context["birth_kind"] != -1)).any())):
            raise ValueError("Episode and command restart birth contexts disagree")
        return prepared, commands, pending, {name: value.to(self.device).clone()
                                            for name, value in context.items()}

    def validate_state_dict(self, state, generator):
        """Side-effect-free full validation for an outer restart coordinator."""
        self._prepare_restart(state, generator)

    def load_state_dict(self, state, generator):
        """All owner/RNG validation completes before any visible write."""
        tensors, commands, pending, *context = self._prepare_restart(state, generator)
        self.commands._commit_restart(commands, generator)
        for name, value in tensors.items():
            getattr(self, name).copy_(value)
        self._pending = pending
        if context:
            for name, value in context[0].items():
                self._final_birth[name].copy_(value)

    @property
    def holding(self):
        return self.commands.holding

    @property
    def learning_mask(self):
        return self.commands.learning_mask

    def prepare_reset(self, ids, generator, *, origins, reset_xy_world, ground_z_world,
                      nominal_q=None, nominal_mask=None, iteration=None):
        ids = self.sampler._ids(ids)
        if self._pending is not None:
            raise RuntimeError("A flat reset proposal is still awaiting native write/commit")
        n = ids.numel()
        if n == 0:
            raise ValueError("Flat reset preparation requires a nonempty lane set")
        if self.final_source_birth_enabled:
            iteration = self.commands._iteration(iteration)
        elif iteration is not None:
            raise ValueError("A final birth clock requires its explicit final source owner")
        if nominal_mask is not None:
            return self._prepare_mixed_reset(ids, generator, origins=origins,
                reset_xy_world=reset_xy_world, ground_z_world=ground_z_world,
                nominal_q=nominal_q, nominal_mask=nominal_mask, iteration=iteration)
        if nominal_q is not None:
            # Initial formal foundation is upright only. The archived folded
            # births remain a later, separately admitted recovery capability.
            q = nominal_q.to(device=self.device, dtype=self.mapper.dtype)
            metrics = self.mapper.metrics(q)
            zero3 = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
            zero6 = torch.zeros(n, 6, dtype=torch.float32, device=self.device)
            length = metrics["length_m"].float()
            angle = -metrics["angle_rad"].float()
            wheel_indices = [self.tree_joint_names.index(name) for name in ("L_joint3", "R_joint3")]
            requests = FlatBirthRequests(ids.clone(), torch.zeros(n, dtype=torch.bool, device=self.device),
                torch.full((n, 2), -1, dtype=torch.int8, device=self.device),
                length, angle, length * angle.cos(), q[:, wheel_indices].float(), zero6,
                zero3, zero3.clone(), zero6.clone())
            if self.final_source_birth_enabled:
                requests = self._requests_from_fields({**vars(requests),
                    "birth_kind": torch.full((n,), -2, dtype=torch.int8, device=self.device),
                    "air_mode_index": torch.full((n,), -1, dtype=torch.long, device=self.device)})
            placed = self.placement.propose(q, zero3.to(q), zero3.to(q), zero6.to(q),
                origins.to(q), reset_xy_world.to(q), ground_z_world.to(q))
            if not bool(placed.valid.all()) or not bool((self.mapper.kinematics.residual(q).abs() < 1e-6).all()):
                raise ValueError("Nominal foundation birth violates collision placement or closure")
            counts = torch.ones(n, dtype=torch.long, device=self.device)
            self.birth_proposal_count[ids] += 1
            self._birth_coverage["proposal_count"] += n
            self._pending = FlatResetState(ids.clone(), requests, placed.root_pose, placed.root_velocity,
                q, torch.zeros_like(q), counts, placed.placement_raise_m, placed.minimum_shape_clearance_m)
            return self._pending
        counts = torch.zeros(n, dtype=torch.long, device=self.device)
        remaining = torch.arange(n, device=self.device)
        saved_requests = saved_q = saved_dq = saved_place = None
        for _ in range(self.max_attempts):
            requests = self._sample_birth_requests(ids[remaining], generator, iteration)
            dtype = self.mapper.dtype
            mapped = self.mapper.map_requests(requests.virtual_leg_length_m.to(dtype=dtype), -requests.virtual_leg_angle_rad.to(dtype=dtype),
                requests.wheel_angle_rad.to(dtype=dtype), requests.control_velocity_rad_s.to(dtype=dtype))
            # Source draws are deliberately float32; the mechanical/collision
            # solver has its own explicit precision. Never rely on defaults.
            dtype = self.placement.dtype
            placed = self.placement.propose(mapped.q.to(dtype=dtype), requests.source_root_position_m.to(dtype=dtype),
                requests.source_root_rpy_rad.to(dtype=dtype), requests.source_root_velocity6.to(dtype=dtype),
                origins[remaining].to(dtype=dtype), reset_xy_world[remaining].to(dtype=dtype), ground_z_world[remaining].to(dtype=dtype),
                **({"root_velocity_frame": "com_world"} if self.final_source_birth_enabled else {}))
            valid = mapped.valid & placed.valid
            self._birth_coverage["proposal_count"] += len(valid)
            self._record_birth_rejections(mapped, placed, ~valid)
            counts[remaining] += 1
            self.birth_proposal_count[ids[remaining]] += 1
            self.birth_rejection_count[ids[remaining[~valid]]] += 1
            if saved_requests is None:
                saved_requests = {key: value.new_empty((n, *value.shape[1:]))
                                  for key, value in vars(requests).items()}
                saved_q, saved_dq = mapped.q.new_empty(n, 18), mapped.dq.new_empty(n, 18)
                saved_place = {key: getattr(placed, key).new_empty((n, *getattr(placed, key).shape[1:]))
                               for key in ("root_pose", "root_velocity", "placement_raise_m", "minimum_shape_clearance_m")}
            accepted = remaining[valid]
            for key, value in vars(requests).items():
                saved_requests[key][accepted] = value[valid]
            saved_q[accepted], saved_dq[accepted] = mapped.q[valid], mapped.dq[valid]
            for key in saved_place:
                saved_place[key][accepted] = getattr(placed, key)[valid]
            remaining = remaining[~valid]
            if remaining.numel() == 0:
                break
        if remaining.numel():
            # Abort atomically: no episode identity/command state is committed.
            # RNG and refusal counters intentionally preserve consumed attempts.
            raise RuntimeError(f"Flat mechanical birth rejected lanes {ids[remaining].tolist()} after {self.max_attempts} attempts")
        requests = self._requests_from_fields(saved_requests)
        self._pending = FlatResetState(ids.clone(), requests, saved_place["root_pose"],
            saved_place["root_velocity"], saved_q, saved_dq, counts,
            saved_place["placement_raise_m"], saved_place["minimum_shape_clearance_m"])
        return self._pending

    def _prepare_mixed_reset(self, ids, generator, *, origins, reset_xy_world,
                             ground_z_world, nominal_q, nominal_mask, iteration=None):
        """Prepare two birth populations as one pending native-write transaction.

        The mask and q18 rows correspond to the supplied ids, not all native
        lanes. Only the dynamic subset samples requests. Subset proposals never
        commit commands or episodes; refusals/RNG consumption are not refunded.
        """
        n = len(ids)
        if (not isinstance(nominal_mask, torch.Tensor) or nominal_mask.dtype != torch.bool
                or nominal_mask.shape != (n,) or nominal_mask.device != self.device):
            raise ValueError("Mixed birth nominal_mask requires bool[K] on the episode device")
        if not isinstance(nominal_q, torch.Tensor) or nominal_q.shape != (n, 18):
            raise ValueError("Mixed birth nominal_q requires complete q18 rows for the reset ids")
        if bool(nominal_mask.all()):
            return self.prepare_reset(ids, generator, origins=origins,
                reset_xy_world=reset_xy_world, ground_z_world=ground_z_world, nominal_q=nominal_q,
                iteration=iteration)
        if not bool(nominal_mask.any()):
            return self.prepare_reset(ids, generator, origins=origins,
                reset_xy_world=reset_xy_world, ground_z_world=ground_z_world, iteration=iteration)
        q = nominal_q.to(device=self.device, dtype=self.mapper.dtype)
        subsets = []
        try:
            for selected, fixed in ((nominal_mask, True), (~nominal_mask, False)):
                rows = selected.nonzero(as_tuple=False).flatten()
                state = self.prepare_reset(ids[rows], generator,
                    origins=origins[rows], reset_xy_world=reset_xy_world[rows],
                    ground_z_world=ground_z_world[rows], nominal_q=q[rows] if fixed else None,
                    iteration=iteration)
                subsets.append((rows, state))
                self._pending = None
            template = subsets[0][1]
            requests = {name: value.new_empty((n, *value.shape[1:]))
                        for name, value in vars(template.requests).items()}
            fields = {name: getattr(template, name).new_empty((n, *getattr(template, name).shape[1:]))
                      for name in FlatResetState.__dataclass_fields__ if name not in ("env_ids", "requests")}
            for rows, state in subsets:
                for name in requests:
                    requests[name][rows] = getattr(state.requests, name)
                for name in fields:
                    fields[name][rows] = getattr(state, name)
            self._pending = FlatResetState(env_ids=ids.clone(), requests=self._requests_from_fields(requests), **fields)
            return self._pending
        except Exception:
            # No subset may remain independently writable after a batch abort.
            self._pending = None
            raise

    def commit_reset(self, state, heading, generator, iteration):
        if state is not self._pending or self._pending is None:
            raise RuntimeError("Flat reset commit must refer to its exact pending state")
        ids = state.env_ids
        kwargs = {}
        if self.final_source_birth_enabled:
            self._validate_final_labels(vars(state.requests))
            kwargs.update(air_mode_index=state.requests.air_mode_index,
                          birth_definition_sha256=self._final_birth["sampler"].definition_sha256)
        if hasattr(self.commands, "episode_node_indices"):
            kwargs["next_episode_ids"] = self.episode_id[ids] + 1
        self.commands.reset(ids, heading, state.requests.ground_birth, generator, iteration=iteration, **kwargs)
        self.episode_id[ids] += 1
        if self.final_source_birth_enabled:
            for name in _FINAL_REQUEST_FIELDS:
                self._final_birth[name][ids] = getattr(state.requests, name)
        self.placement_raise_m[ids] = state.placement_raise_m.to(torch.float32)
        self.minimum_shape_clearance_m[ids] = state.minimum_shape_clearance_m.to(torch.float32)
        self.birth_source_angle_rad[ids] = state.requests.virtual_leg_angle_rad
        self.birth_own_angle_rad[ids] = -state.requests.virtual_leg_angle_rad
        coverage = self._birth_coverage
        ground_true = state.requests.ground_birth.long().sum()
        coverage["accepted_reset_count"] += len(ids)
        coverage["accepted_ground_true_reset_count"] += ground_true
        coverage["accepted_ground_false_reset_count"] += len(ids) - ground_true
        lengths = state.requests.virtual_leg_length_m.detach()
        coverage["length_min"].copy_(torch.minimum(coverage["length_min"], lengths.amin(0)))
        coverage["length_max"].copy_(torch.maximum(coverage["length_max"], lengths.amax(0)))
        coverage["velocity_abs_max"].copy_(torch.maximum(coverage["velocity_abs_max"],
            state.requests.control_velocity_rad_s.detach().abs().amax(0)))
        self._pending = None

    def _record_birth_rejections(self, mapped, placed, rejected):
        """Nonexclusive diagnostic flags; placement failures can be consequences."""
        coverage = self._birth_coverage
        coverage["rejected_proposal_count"] += rejected.long().sum()
        classified = torch.zeros_like(rejected)
        for owner, proposal in (("mapping", mapped), ("placement", placed)):
            reasons = getattr(proposal, "reasons", {})
            if not isinstance(reasons, dict):
                continue
            for name, reason in reasons.items():
                if (not isinstance(reason, torch.Tensor) or reason.dtype != torch.bool
                        or reason.shape != rejected.shape or reason.device != rejected.device):
                    continue
                hits = rejected & ~proposal.valid & reason
                key = f"{owner}/{name}"
                if key not in coverage["rejection_reasons"]:
                    coverage["rejection_reasons"][key] = torch.zeros((), dtype=torch.long, device=self.device)
                coverage["rejection_reasons"][key] += hits.long().sum()
                classified |= hits
        coverage["unclassified_rejected_proposal_count"] += (rejected & ~classified).long().sum()

    def begin_policy(self):
        return self.commands.begin_policy()

    def commit_scripted_reset(self, ids, heading, generator, iteration):
        """Commit commands after the recovery owner wrote authenticated poses.

        No source birth was proposed for these lanes. Zero placement fields are
        unavailable placeholders, identified by the caller's recovery-enabled
        mask; proposals/rejections and their denominators remain source-only.
        """
        if self._pending is not None:
            raise RuntimeError("Commit the pending source birth before scripted reset commands")
        ids = self.sampler._ids(ids)
        self.commands.reset(ids, heading, torch.zeros_like(ids, dtype=torch.bool), generator, iteration=iteration)
        self.episode_id[ids] += 1
        if self.final_source_birth_enabled:
            for name in _FINAL_REQUEST_FIELDS:
                self._final_birth[name][ids] = -1
        for name in ("placement_raise_m", "minimum_shape_clearance_m", "birth_source_angle_rad", "birth_own_angle_rad"):
            getattr(self, name)[ids] = 0.

    def gate_active_effort(self, effort):
        if effort.shape != (self.num_envs, 6):
            raise ValueError("Flat hold gates exactly six active motor efforts")
        return torch.where(self.holding[:, None], 0., effort)

    def advance_physics(self, dt):
        self.commands.advance_physics(dt)

    def advance(self, dt, heading, iteration, generator, *, world_positions=None):
        if world_positions is None:
            self.commands.advance(dt, heading, iteration, generator)
        else:
            self.commands.advance(dt, heading, iteration, generator, world_positions=world_positions)

    def publish(self, commands, mode, targets, jump_requested):
        # No command projection, slew, new height draw, or old command owner.
        commands[:, 0] = self.commands.command3[:, 0]
        commands[:, 1] = self.commands.command3[:, 2]
        commands[:, 2] = self.commands.height
        mode.fill_(1)
        targets.zero_()
        jump_requested.zero_()

    def bootstrap_mask(self, terminated):
        return self.commands.bootstrap_mask(terminated)

    def diagnostics(self):
        result = self.commands.diagnostics()
        result.update(episode_id=self.episode_id.clone(), birth_proposal_count=self.birth_proposal_count.clone(),
            birth_rejection_count=self.birth_rejection_count.clone(), placement_raise_m=self.placement_raise_m.clone(),
            minimum_shape_clearance_m=self.minimum_shape_clearance_m.clone(),
            birth_source_angle_rad=self.birth_source_angle_rad.clone(), birth_own_angle_rad=self.birth_own_angle_rad.clone())
        if self.final_source_birth_enabled:
            result["final_birth_kind"] = self._final_birth["birth_kind"].clone()
            result["final_birth_air_mode_index"] = self._final_birth["air_mode_index"].clone()
        return result

    def metrics(self, *, source_birth_mask=None):
        result = self.commands.metrics()
        result.update({"/source_birth/proposals": self.birth_proposal_count.sum(),
            "/source_birth/rejections": self.birth_rejection_count.sum(),
            "/source_birth/rejection_fraction": self.birth_rejection_count.sum().float() / self.birth_proposal_count.sum().clamp_min(1),
            "/source_birth/placement_raise_mean_m": (self.placement_raise_m.mean() if source_birth_mask is None else
                self.placement_raise_m[source_birth_mask].sum() / source_birth_mask.sum().clamp_min(1))})
        coverage = self._birth_coverage
        prefix = "/source_birth/current_worker/"
        for name in ("proposal_count", "rejected_proposal_count", "unclassified_rejected_proposal_count",
                     "accepted_reset_count", "accepted_ground_false_reset_count", "accepted_ground_true_reset_count"):
            result[prefix + name] = coverage[name].clone()
        available = coverage["accepted_reset_count"] > 0
        result[prefix + "accepted_requested_summary_available"] = available
        for index, side in enumerate(("left", "right")):
            for bound in ("min", "max"):
                value = coverage["length_" + bound][index]
                result[prefix + f"accepted_requested_length_{side}_{bound}_m"] = torch.where(available, value, 0.)
        for index, name in enumerate(CONTROL_JOINT_NAMES):
            result[prefix + f"accepted_requested_velocity_abs_max_rad_s/{name}"] = coverage["velocity_abs_max"][index].clone()
        result.update({prefix + "rejected_reason_nonexclusive/" + name: value.clone()
                       for name, value in coverage["rejection_reasons"].items()})
        if self.final_source_birth_enabled:
            kinds = self._final_birth["birth_kind"]
            for code, name in ((-2, "nominal"), (-1, "outside_final"),
                               (0, "ordinary"), (1, "ground"), (2, "air")):
                result["/source_birth/final_current_episode/" + name + "_fraction"] = (
                    (kinds == code).float().mean())
        return result

    def final_birth_summary(self):
        if not self.final_source_birth_enabled:
            return None
        sampler = self._final_birth["sampler"]
        return {"version": FINAL_BIRTH_ADAPTER_VERSION,
            "definition_sha256": sampler.definition_sha256,
            "sampler_version": sampler.VERSION,
            "class_codes": {"nominal": -2, "retained_or_scripted": -1,
                            "ordinary": 0, "ground": 1, "air": 2},
            "class_scope": "Explicit accepted reset requests, not inferred pose or velocity",
            "group_rng_order": "final_then_retained_flat",
            "root_velocity_frame": "com_world",
            "birth_rejection": deepcopy(self.rejection),
            "length_conditioning": "No new conditioning; existing bounded whole-request rejection only",
            "pending_context": "Complete request class/mode fields travel with the native-write transaction",
            "restart_schema": FINAL_EPISODE_STATE_VERSION,
            "air_command_limits": "Installed from the same accepted reset mode and definition",
            "ppo_exposure_producer_bound": False,
            "stage_or_native_qualified": False}

    def final_birth_exposure_binding(self, source_dynamic_mask):
        """Bind labels to the same reset producer and actual native lane roles."""
        if not self.final_source_birth_enabled:
            raise ValueError("Final exposure requires its explicit accepted-birth owner")
        if (not isinstance(source_dynamic_mask, torch.Tensor)
                or source_dynamic_mask.shape != (self.num_envs,)
                or source_dynamic_mask.dtype != torch.bool or source_dynamic_mask.device != self.device
                or bool((source_dynamic_mask & self.commands.fixed_request_mask).any())):
            raise ValueError("Final exposure requires actual source lanes excluding fixed requests")
        retained = self.commands._final.get("source_flat_mask", torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device))
        return {"final_birth_definition_sha256": self._final_birth["sampler"].definition_sha256,
                "final_birth_lane_mask": (source_dynamic_mask & ~retained).clone()}


def build_flat_episode_runtime(config, manifest, spec, root, num_envs, device, *, curriculum=None,
                               rough_terrain=None, environment_origins=None, evaluation_setup=None,
                               replay_setup=None):
    """Build CPU components from sealed bytes; never an admission function."""
    from wheeled_tasks.chassis.application.configuration.flat_episode_selection import validate_flat_episode_selection
    settings = validate_flat_episode_selection(config)
    if settings is None:
        return None
    plane_regression = False
    if replay_setup is not None:
        from wheeled_tasks.chassis.application.evaluation.ordinary_replay import OrdinaryReplaySetup, ROLE
        if (type(replay_setup) is not OrdinaryReplaySetup or evaluation_setup is not None
                or rough_terrain is not None or environment_origins is not None or curriculum is not None):
            raise ValueError("Ordinary replay requires its sole fixed-plane command owner")
        replay_setup.validate_constructor(config, num_envs, replay_setup.seed, ROLE)
        plane_regression = True
    if evaluation_setup is not None:
        from wheeled_tasks.chassis.application.evaluation.flat_evaluation_setup import FlatEvaluationSetup
        if (type(evaluation_setup) is not FlatEvaluationSetup
                or evaluation_setup.evaluation_surface != "plane_regression_v1"
                or evaluation_setup.run_role != "source_rough_plane_regression"):
            raise ValueError("Plane command adaptation requires its typed rough regression setup")
        evaluation_setup.validate_constructor(config, num_envs,
            evaluation_setup.protocol["environment_seed"], evaluation_setup.run_role)
        if rough_terrain is not None or environment_origins is not None or curriculum is not None:
            raise ValueError("Plane regression cannot bind a rough terrain or curriculum command owner")
        plane_regression = True
    root = Path(root).resolve()
    path = (root / settings["branch_table"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Flat branch table must lie inside the bound workspace")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != settings["branch_table_sha256"]:
        raise ValueError("Flat branch table bytes changed")
    from wheeled_tasks.chassis.control.leg_pose_mapping import LegPoseMapper
    from wheeled_tasks.chassis.simulation.runtime.flat_birth_placement import PhysicalFlatPlacement
    table = json.loads(content)
    manifest_path = root / config["asset_directory"] / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != table.get("asset_manifest_sha256"):
        raise ValueError("Branch table asset manifest byte identity differs from the supplied asset")
    mapper = LegPoseMapper(spec, manifest, table, device=device, dtype=torch.float64)
    # The mapping constructor must independently enforce these declared limits.
    if mapper.length_definition != settings["length_definition"]:
        raise ValueError("Branch table length definition differs from the contract")
    if table["training_knee_domain_deg"][1] != settings["training_knee_upper_deg"]:
        raise ValueError("Branch table training knee limit differs from the contract")
    placement = PhysicalFlatPlacement(spec, manifest, root / config["asset_directory"],
        clearance_m=settings["minimum_clearance_m"], device=device, dtype=torch.float64)
    if placement.manifest_sha256 != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("Asset manifest bytes changed during flat owner construction")
    # Freeze actual constructor-validated source bytes and numerical settings.
    # This is component identity, not proof of native reset/solver equivalence.
    adapter = {"settings": settings, "placement": placement.summary(),
        "mapper": {"length_definition": mapper.length_definition, "dtype": str(mapper.dtype),
                   "length_tolerance_m": mapper.length_tolerance_m, "angle_tolerance_rad": mapper.angle_tolerance_rad},
        "kinematics": {name: getattr(mapper.kinematics, name) for name in
            ("position_tolerance_m", "velocity_tolerance_m_s", "limit_tolerance", "rank_rtol")}}
    identity = {"schema": MECHANICAL_RESTART_VERSION,
        "branch_table_sha256": hashlib.sha256(content).hexdigest(),
        "asset_manifest_sha256": placement.manifest_sha256,
        "model_spec_sha256": placement.model_spec_sha256,
        "adapter_configuration_sha256": hashlib.sha256(_canonical(adapter).encode()).hexdigest()}
    commands = None
    if plane_regression:
        # The typed overlay installs all fixed requests before the first reset.
        # Mechanical birth, rough rewards and the original contract stay intact.
        commands = ScutFlatCommands(num_envs, device)
    elif config.get("formal_terrain", {}).get("type") == "rough":
        if (config.get("reward_profile") not in ("scut0730_rough_v1", "v7_final_rough_v1") or rough_terrain is None
                or environment_origins is None or "v7_curriculum_runtime" in config):
            raise ValueError("0730 rough commands require bound shared terrain and the rough reward profile")
        if config["formal_terrain"].get("source") == "v7_final_b8ff79f3":
            from wheeled_tasks.chassis.tasks.commands.final_rough_commands import FinalRoughCommands
            command_class = FinalRoughCommands
        else:
            from wheeled_tasks.chassis.tasks.commands.rough_commands import ScutRoughCommands
            command_class = ScutRoughCommands
        commands = command_class(num_envs, device,
            terrain_origins=rough_terrain.importer.terrain_origins, env_origins=environment_origins)
    elif "v7_curriculum_runtime" in config:
        from wheeled_tasks.chassis.application.configuration.curriculum_selection import load_curriculum_definition
        from wheeled_tasks.chassis.tasks.curricula.curriculum_sampling import CurriculumFlatCommands
        definition = load_curriculum_definition(config, root)
        if curriculum is None or curriculum.definition != definition or not curriculum.external_budget:
            raise ValueError("Course command owner requires the same explicit external-budget controller")
        commands = CurriculumFlatCommands(num_envs, device, definition=definition,
            capabilities=config["v7_curriculum_runtime"]["capabilities"], sampling_plan=curriculum.sampling_plan)
    owner = FlatEpisodeRuntime(num_envs, device, mapper, placement, manifest["tree_joint_names"],
        birth_rejection=settings["birth_rejection"], restart_identity=identity, commands=commands)
    if "final_rough_birth_adapter" in config:
        selected = config["final_rough_birth_adapter"]
        if (not isinstance(selected, dict) or set(selected) != {"version", "birth_definition_sha256"}
                or selected["version"] != FINAL_BIRTH_ADAPTER_VERSION or plane_regression
                or config.get("formal_terrain", {}).get("source") != "v7_final_b8ff79f3"
                or config.get("reward_profile") != "v7_final_rough_v1"):
            raise ValueError("Final birth adapter requires its explicit final-terrain component selection")
        owner.enable_final_source_births(birth_definition_sha256=selected["birth_definition_sha256"])
    return owner
