"""Measurements restricted to V7 dynamic flat births.

Use the paid, pre-reset ``diagnostics.source_flat_episode`` snapshot, not the
returned newborn's labels. ``ground_request_false`` includes ordinary and
nonstandard poses; neither bucket certifies floor self-righting or a power
loss while moving. The motor-off measurement is the source owner's whole
interval gate, not a force measurement (passive springs can remain active).

The caller supplies the real native stream identity, env.tick, physical_tick,
and the actual dynamic-source lane mask after lane reordering. Actor credit
uses RecoveryRolloutMask.loss_population_mask() once per successful update,
after optimizer postchecks. Global learning/update clocks and epochs are not
inputs. This observer neither chooses births nor changes PPO masks or rewards.

An explicitly bound final reset producer selects a separate versioned group
layout: retained flat, ordinary, ground and air. Its class comes from the paid
accepted request, never the returned newborn or a pose/height heuristic.
"""
from copy import deepcopy
import hashlib
import json

import torch


GROUPS = ("ground_request_false", "ground_request_true")
FINAL_GROUPS = ("retained_flat", "ordinary", "ground", "air")
FIELDS = (
    "policy_intervals", "physics_ticks", "learning_eligible_intervals",
    "masked_intervals", "motor_off_owner_intervals",
    "post_motor_off_learning_eligible_intervals", "observed_episodes",
    "birth_observed_episodes", "closed_episodes", "terminated_episodes",
    "timeout_episodes", "complete_observed_episodes",
    "complete_motor_off_window_episodes", "post_motor_off_learning_eligible_episodes",
    "complete_episodes_with_post_motor_off_learning_eligible",
    "actor_direct_loss_intervals", "post_motor_off_actor_direct_loss_intervals",
    "complete_episodes_with_actor_loss",
    "complete_motor_off_episodes_with_post_motor_off_actor_loss",
)
STATE_VERSION = "v7-source-birth-exposure-candidate-v1"
FINAL_STATE_VERSION = "v7-source-birth-exposure-final-v2"
FLAT_HISTORY_VERSION = "v7-inherited-flat-exposure-v1"
PHYSICS_TICKS_PER_POLICY = 4
MOTOR_OFF_PHYSICS_TICKS = 40


class SourceBirthExposure:
    """Count actual source opportunities separately from optimized samples.

    An episode is complete *observationally* only if every interval from its
    tick 4 birth through its pre-reset terminal interval was observed. This is
    not a success criterion. Partial attachment still measures intervals.
    Checkpoints restore counters and duplicate detection, not native physics.
    A new native worker must use a new stream_id; its partial episodes cannot
    inherit a previous worker's milestones.
    """

    def __init__(self, *, num_envs, steps_per_rollout, source_dynamic_mask,
                 stream_id, device="cpu", defer_tensor_validation=False,
                 final_birth_definition_sha256=None, final_birth_lane_mask=None):
        if any(type(value) is not int or value < 1
               for value in (num_envs, steps_per_rollout)):
            raise ValueError("Birth exposure requires positive rollout dimensions")
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError("Birth exposure requires the actual native stream identity")
        if type(defer_tensor_validation) is not bool:
            raise ValueError("Birth exposure tensor validation mode must be boolean")
        self.defer_tensor_validation = defer_tensor_validation
        self.fault = None
        self.num_envs, self.steps_per_rollout = num_envs, steps_per_rollout
        self.groups = FINAL_GROUPS if final_birth_definition_sha256 is not None else GROUPS
        self.counts = torch.zeros((len(self.groups), len(FIELDS)), dtype=torch.long, device=device)
        self.device = self.counts.device
        self.source_dynamic_mask = self._vector(source_dynamic_mask, "source_dynamic_mask", torch.bool)
        self.final_birth_definition_sha256 = final_birth_definition_sha256
        self.final_birth_lane_mask = None
        if final_birth_definition_sha256 is not None:
            if (not isinstance(final_birth_definition_sha256, str)
                    or len(final_birth_definition_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in final_birth_definition_sha256)):
                raise ValueError("Final birth exposure requires its producer definition fingerprint")
            self.final_birth_lane_mask = self._vector(final_birth_lane_mask, "final_birth_lane_mask", torch.bool)
            if bool((self.final_birth_lane_mask & ~self.source_dynamic_mask).any()):
                raise ValueError("Final birth lanes must belong to the actual dynamic source population")
        elif final_birth_lane_mask is not None:
            raise ValueError("Final birth exposure lanes require their producer definition fingerprint")
        self.stream_id, self.seen_stream_ids = stream_id, [stream_id]
        self.last_policy_interval_id = self.last_physical_tick = None
        self.completed_actor_updates = self.critic_only_updates = 0
        self.failed_updates = self.discarded_rollouts = 0
        self._flat_history = None
        self._history_attachment_open = True
        self.live = self._empty_live()
        self.collecting = False
        self.rows, self.population = [], None

    def _empty_live(self):
        integers = {name: torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                    for name in ("ticks", "motor_off_ticks")}
        integers["expected_ids"] = torch.full((self.num_envs,), -1, dtype=torch.long,
                                               device=self.device)
        booleans = {name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    for name in ("seen", "ground", "complete", "post_eligible",
                                 "actor_seen", "post_actor_seen")}
        if self.final_birth_definition_sha256 is not None:
            integers["kind"] = torch.full((self.num_envs,), -2, dtype=torch.long, device=self.device)
        return {**integers, **booleans}

    def _read_final_kind(self, owner, name):
        kind = self._vector(owner["final_birth_kind"], name + " birth_kind", torch.int8)
        mode = self._vector(owner["final_birth_air_mode_index"], name + " air_mode", torch.long)
        invalid = (((kind < -2) | (kind > 2)).any()
            | (mode != torch.where(kind == 2, 0, -1)).any()
            | (self.source_dynamic_mask & ((kind < -1)
                | ((kind >= 0) != self.final_birth_lane_mask))).any())
        return kind, invalid

    def _vector(self, value, name, dtype):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                or value.dtype != dtype or value.device != self.device):
            raise ValueError(f"Birth exposure {name} must be {dtype}[N] on the producer device")
        return value.detach().clone()

    def begin_rollout(self, *, actor_enabled):
        if self.collecting or self.fault is not None or type(actor_enabled) is not bool:
            raise ValueError("Birth exposure cannot begin a pending rollout")
        self.collecting, self.actor_enabled = True, actor_enabled
        self._history_attachment_open = False
        self.rows, self.population = [], None
        # Sampling runs in inference mode; binding and rollback run outside it.
        # Keep the mutable latch and rollback snapshots ordinary tensors.
        with torch.inference_mode(False):
            self.initial_actor_seen = self.live["actor_seen"].clone()
            self.initial_post_actor_seen = self.live["post_actor_seen"].clone()
            self.invalid = torch.zeros((), dtype=torch.bool, device=self.device)
            self.validated = not self.defer_tensor_validation
            if self.defer_tensor_validation:
                self.backup_counts = self.counts.clone()
                self.backup_live = {name: value.clone() for name, value in self.live.items()}
                self.backup_clock = self.last_policy_interval_id, self.last_physical_tick

    def _validate_pending(self):
        if self.collecting and not self.validated:
            if bool(self.invalid):
                self.counts, self.live = self.backup_counts, self.backup_live
                self.last_policy_interval_id, self.last_physical_tick = self.backup_clock
                self.fault = "invalid_source_transition_or_loss_population"
                self.collecting, self.rows, self.population = False, [], None
                raise ValueError("Birth exposure source clock, gate, reset or loss population differs")
            self.validated = True

    def observe_transition(self, *, step_index, policy_interval_id, physical_tick, info):
        """Consume existing diagnostic snapshots before native storage advances.

        record_diagnostics must be enabled. Tick IDs are the real worker's
        policy/5ms clocks, not inherited learning-update counters. At tick 40
        the source interval learning_mask remains false although holding has
        cleared; the first fully post-off interval ends at tick 44.
        """
        if (not self.collecting or self.population is not None
                or type(step_index) is not int or step_index != len(self.rows)
                or step_index >= self.steps_per_rollout):
            raise ValueError("Birth exposure requires the next actual rollout row")
        if (type(policy_interval_id) is not int or policy_interval_id < 1
                or type(physical_tick) is not int
                or physical_tick != PHYSICS_TICKS_PER_POLICY * policy_interval_id
                or self.last_policy_interval_id is not None
                and (policy_interval_id != self.last_policy_interval_id + 1
                     or physical_tick != self.last_physical_tick + PHYSICS_TICKS_PER_POLICY)):
            raise ValueError("Birth exposure requires consecutive actual physical intervals")
        try:
            diagnostics = info["diagnostics"]
            owner = diagnostics["source_flat_episode"]
            returned = info["returned_episode_state"]
            returned_owner = returned["source_flat_episode"]
            row = {"ids": self._vector(owner["episode_id"], "episode_id", torch.long),
                   "ticks": self._vector(owner["episode_physics_ticks"],
                                         "episode_physics_ticks", torch.long),
                   "ground": self._vector(owner["ground"], "ground", torch.bool),
                   "owner_learning": self._vector(owner["learning_mask"], "owner learning_mask", torch.bool),
                   "holding": self._vector(owner["holding"], "holding", torch.bool),
                   "learning": self._vector(info["policy_learning_mask"], "policy_learning_mask", torch.bool),
                   "done": self._vector(diagnostics["done"], "done", torch.bool),
                   "terminated": self._vector(diagnostics["terminated"], "terminated", torch.bool),
                   "timeout": self._vector(diagnostics["timeout"], "timeout", torch.bool),
                   "returned_ids": self._vector(returned["episode_id"], "returned episode_id", torch.long),
                   "returned_ticks": self._vector(returned_owner["episode_physics_ticks"], "returned ticks", torch.long),
                   "reset": self._vector(returned["reset_mask"], "reset_mask", torch.bool)}
        except (KeyError, TypeError) as exc:
            raise ValueError("Birth exposure requires paid and returned source-flat diagnostics") from exc
        semantic_invalid = torch.zeros((), dtype=torch.bool, device=self.device)
        for field in ("fixed_training_rehearsal", "fixed_evaluation_request", "scripted_recovery_birth"):
            if field in owner:
                excluded = self._vector(owner[field], field, torch.bool)
                semantic_invalid |= (excluded & self.source_dynamic_mask).any()
        if "policy_learning_mask" in diagnostics:
            semantic_invalid |= (row["learning"] != self._vector(
                diagnostics["policy_learning_mask"], "diagnostic learning_mask", torch.bool)).any()
        if "episode_id" in diagnostics:
            semantic_invalid |= (row["ids"] != self._vector(
                diagnostics["episode_id"], "diagnostic episode_id", torch.long)).any()
        semantic_invalid |= (row["returned_ids"] != self._vector(
            returned_owner["episode_id"], "returned owner episode_id", torch.long)).any()
        known = self.live["expected_ids"] >= 0
        seen = self.live["seen"]
        if self.final_birth_definition_sha256 is not None:
            try:
                kind, kind_invalid = self._read_final_kind(owner, "paid")
                returned_kind, returned_invalid = self._read_final_kind(returned_owner, "returned")
                returned_ground = self._vector(returned_owner["ground"], "returned ground", torch.bool)
            except (KeyError, TypeError) as exc:
                raise ValueError("Final exposure requires the paid and returned accepted birth context") from exc
            semantic_invalid |= kind_invalid | returned_invalid
            semantic_invalid |= ((kind >= 0) & (row["ground"] != (kind == 1))).any()
            semantic_invalid |= ((returned_kind >= 0) & (returned_ground != (returned_kind == 1))).any()
            semantic_invalid |= (seen & (kind != self.live["kind"])).any()
            semantic_invalid |= (~row["done"] & (returned_kind != kind)).any()
            row["kind"] = kind
        motor_off = row["ground"] & (row["ticks"] <= MOTOR_OFF_PHYSICS_TICKS)
        invalid = ((row["ids"] < 0) | (row["ticks"] < PHYSICS_TICKS_PER_POLICY)
                   | (row["ticks"] % PHYSICS_TICKS_PER_POLICY != 0)
                   | (known & (row["ids"] != self.live["expected_ids"]))
                   | (seen & (row["ticks"] != self.live["ticks"] + PHYSICS_TICKS_PER_POLICY))
                   | (known & ~seen & (row["ticks"] != PHYSICS_TICKS_PER_POLICY))
                   | (seen & (row["ground"] != self.live["ground"]))
                   | (row["holding"] != (
                       row["ground"] & (row["ticks"] < MOTOR_OFF_PHYSICS_TICKS)))
                   | (row["owner_learning"] != ~motor_off)
                   | (row["learning"] & ~row["owner_learning"])
                   | (row["reset"] != row["done"])
                   | (row["returned_ids"] != row["ids"] + row["done"].long())
                   | (row["returned_ticks"] != torch.where(row["done"], 0, row["ticks"]))
                   | ((row["terminated"] | row["timeout"]) & ~row["done"]))
        semantic_invalid |= invalid.any()
        if not self.defer_tensor_validation and bool(semantic_invalid):
            raise ValueError("Birth exposure source clock, gate or reset identities differ")
        self.invalid |= semantic_invalid

        # Deferred values remain unpublished until the latch passes at bind or cleanup.
        new = ~seen
        complete = torch.where(new, row["ticks"] == PHYSICS_TICKS_PER_POLICY, self.live["complete"])
        off_ticks = torch.where(new, 0, self.live["motor_off_ticks"]) + motor_off.long() * PHYSICS_TICKS_PER_POLICY
        post_eligible = row["ground"] & ~motor_off & row["learning"]
        previous_post = self.live["post_eligible"] & ~new
        has_post = previous_post | post_eligible
        off_complete = complete & row["ground"] & (off_ticks == MOTOR_OFF_PHYSICS_TICKS)
        row.update(new=new, complete=complete, off_complete=off_complete,
                   post_off=row["ground"] & ~motor_off)
        self._add(row, {
            "policy_intervals": torch.ones_like(new),
            "physics_ticks": torch.full_like(row["ticks"], PHYSICS_TICKS_PER_POLICY),
            "learning_eligible_intervals": row["learning"], "masked_intervals": ~row["learning"],
            "motor_off_owner_intervals": motor_off,
            "post_motor_off_learning_eligible_intervals": post_eligible,
            "observed_episodes": new,
            "birth_observed_episodes": new & (row["ticks"] == PHYSICS_TICKS_PER_POLICY),
            "closed_episodes": row["done"], "terminated_episodes": row["done"] & row["terminated"],
            "timeout_episodes": row["done"] & row["timeout"],
            "complete_observed_episodes": row["done"] & complete,
            "complete_motor_off_window_episodes": off_complete & motor_off,
            "post_motor_off_learning_eligible_episodes": post_eligible & ~previous_post,
            "complete_episodes_with_post_motor_off_learning_eligible": row["done"] & complete & has_post,
        })
        self.live.update(expected_ids=row["returned_ids"].clone(),
                         ticks=row["returned_ticks"].clone(), seen=~row["done"],
                         ground=row["ground"].clone(), complete=complete,
                         motor_off_ticks=off_ticks, post_eligible=has_post)
        if self.final_birth_definition_sha256 is not None:
            self.live["kind"] = row["kind"].long()
        self.last_policy_interval_id, self.last_physical_tick = policy_interval_id, physical_tick
        self.rows.append(row)

    def _add(self, row, values):
        selected = self.source_dynamic_mask
        columns = torch.zeros((self.num_envs, len(FIELDS)), dtype=torch.long, device=self.device)
        for name, value in values.items():
            columns[:, FIELDS.index(name)] = value.long()
        groups = (row["ground"].long() if self.final_birth_definition_sha256 is None
                  else (row["kind"].long() + 1).clamp(0, len(self.groups) - 1))
        self.counts.index_add_(0, groups[selected], columns[selected])

    def bind_loss_population(self, *, final_learning_mask, optimized_population_mask):
        """Bind the unique original population, including native minibatch tail omission."""
        if not self.collecting or self.population is not None or len(self.rows) != self.steps_per_rollout:
            raise ValueError("Birth loss exposure requires one complete unbound rollout")
        shape = (self.steps_per_rollout, self.num_envs)
        for mask in (final_learning_mask, optimized_population_mask):
            if (not isinstance(mask, torch.Tensor) or mask.shape != shape
                    or mask.dtype != torch.bool or mask.device != self.device):
                raise ValueError("Birth loss masks must match actual rollout storage")
        observed = torch.stack([row["learning"] for row in self.rows])
        mismatch = ((final_learning_mask != observed).any()
                    | (optimized_population_mask & ~final_learning_mask).any())
        if self.defer_tensor_validation:
            self.invalid |= mismatch
            self._validate_pending()
        elif bool(mismatch):
            raise ValueError("Birth loss population differs from the observed PPO masks")
        self.population = optimized_population_mask.detach().clone()

    def _close_rollout(self, *, credit_actor):
        self._validate_pending()
        actor_seen, post_actor_seen = self.initial_actor_seen.clone(), self.initial_post_actor_seen.clone()
        for index, row in enumerate(self.rows):
            population = (self.population[index] if credit_actor else torch.zeros_like(row["learning"]))
            actor_seen = (actor_seen & ~row["new"]) | population
            post_actor_seen = (post_actor_seen & ~row["new"]) | (population & row["post_off"])
            self._add(row, {
                "actor_direct_loss_intervals": population,
                "post_motor_off_actor_direct_loss_intervals": population & row["post_off"],
                "complete_episodes_with_actor_loss": row["done"] & row["complete"] & actor_seen,
                "complete_motor_off_episodes_with_post_motor_off_actor_loss": (
                    row["done"] & row["complete"] & row["off_complete"] & post_actor_seen),
            })
            actor_seen &= ~row["done"]
            post_actor_seen &= ~row["done"]
        self.live.update(actor_seen=actor_seen, post_actor_seen=post_actor_seen)
        self.collecting, self.rows, self.population = False, [], None

    def finish_update(self, *, actor_optimized):
        """Call only after native optimizer work and all update postchecks succeeded."""
        if (not self.collecting or self.population is None or type(actor_optimized) is not bool
                or actor_optimized != self.actor_enabled):
            raise ValueError("Birth actor credit requires the actual successful optimizer phase")
        self._close_rollout(credit_actor=actor_optimized)
        if actor_optimized:
            self.completed_actor_updates += 1
        else:
            self.critic_only_updates += 1

    def fail_update(self):
        """Keep physical exposure; the failed current update earns no actor samples."""
        if self.collecting:
            self._close_rollout(credit_actor=False)
            self.failed_updates += 1

    def discard_rollout(self):
        """Retain a partial or unpaid collection without attributing actor loss."""
        if self.collecting:
            self._close_rollout(credit_actor=False)
            self.discarded_rollouts += 1

    def metrics(self):
        """Cumulative counts for the existing logger; reads never advance any clock."""
        visible = (self.backup_counts if self.collecting and not self.validated else self.counts)
        rows = visible.detach().cpu().tolist()
        result = {f"/SourceBirthExposure/{name}": sum(row[i] for row in rows)
                  for i, name in enumerate(FIELDS)}
        for group, row in zip(self.groups, rows):
            result.update({f"/SourceBirthExposure/{group}/{name}": value
                           for name, value in zip(FIELDS, row)})
        return result

    def checkpoint_state(self):
        if self.collecting or self.fault is not None:
            raise ValueError("Birth exposure cannot snapshot pending optimizer attribution")
        result = {"version": STATE_VERSION, "num_envs": self.num_envs,
                "steps_per_rollout": self.steps_per_rollout, "groups": list(self.groups),
                "fields": list(FIELDS),
                "source_dynamic_mask": self.source_dynamic_mask.cpu().tolist(),
                "counts": self.counts.cpu().tolist(), "stream_id": self.stream_id,
                "seen_stream_ids": list(self.seen_stream_ids),
                "last_policy_interval_id": self.last_policy_interval_id,
                "last_physical_tick": self.last_physical_tick,
                "live": {name: value.cpu().tolist() for name, value in self.live.items()},
                **{name: getattr(self, name) for name in (
                    "completed_actor_updates", "critic_only_updates", "failed_updates", "discarded_rollouts")},
                "skill_qualification": "not_assessed", "native_exact_resume_qualified": False}
        if self.final_birth_definition_sha256 is not None:
            result.update(version=FINAL_STATE_VERSION,
                final_birth_definition_sha256=self.final_birth_definition_sha256,
                final_birth_lane_mask=self.final_birth_lane_mask.cpu().tolist())
        return result

    def inherit_flat_history(self, state, *, parent_contract_json, parent_checkpoint_sha256):
        """Archive validated flat measurements without inventing final classes.

        The existing registered checkpoint loader authenticates model/contract
        provenance and permits the stage transition. This observer retains the
        old measurement journal only; new class counters and native cursors
        remain fresh. Learning, optimizer and paid clocks are not touched.
        """
        if (self.last_policy_interval_id is not None or bool(self.counts.any())
                or self.completed_actor_updates or self.critic_only_updates
                or self.failed_updates or self.discarded_rollouts):
            raise ValueError("Flat history inheritance requires a fresh final observer")
        if not isinstance(parent_contract_json, str):
            raise ValueError("Flat exposure history requires the exact parent contract text")
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        record = {"version": FLAT_HISTORY_VERSION,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "parent_contract_json": parent_contract_json,
            "parent_contract_sha256": hashlib.sha256(parent_contract_json.encode()).hexdigest(),
            "source_state_sha256": hashlib.sha256(encoded).hexdigest(), "state": state}
        self.restore_flat_history(record)

    def restore_flat_history(self, record):
        """Attach the separately sealed history before the first new rollout."""
        if (self.final_birth_definition_sha256 is None or not self._history_attachment_open
                or self.collecting or self.fault is not None or self._flat_history is not None):
            raise ValueError("Flat exposure history requires an idle final observer before collection")
        if record is None:
            return
        expected = {"version", "parent_checkpoint_sha256", "parent_contract_json",
                    "parent_contract_sha256", "source_state_sha256", "state"}
        if (not isinstance(record, dict) or set(record) != expected
                or record["version"] != FLAT_HISTORY_VERSION):
            raise ValueError("Flat exposure history fields differ")
        for field in ("parent_checkpoint_sha256", "parent_contract_sha256", "source_state_sha256"):
            value = record[field]
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("Flat exposure history requires complete source fingerprints")
        raw_contract = record["parent_contract_json"]
        state = record["state"]
        if (not isinstance(raw_contract, str)
                or hashlib.sha256(raw_contract.encode()).hexdigest() != record["parent_contract_sha256"]
                or hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"),
                    allow_nan=False).encode()).hexdigest() != record["source_state_sha256"]):
            raise ValueError("Flat exposure history bytes differ from the sealed sources")
        contract = json.loads(raw_contract)
        from .fixed_flat_rehearsal import validate_dynamic_nonsteady_birth
        if (not isinstance(contract, dict) or validate_dynamic_nonsteady_birth(contract) is not True
                or contract.get("formal_terrain") != {"type": "plane", "source": "scut0719", "env_spacing_m": 4.}
                or contract.get("reward_profile") != "scut0719_flat_v1"
                or (contract.get("actor_dim"), contract.get("critic_dim"), contract.get("action_dim")) != (35, 78, 6)
                or any(type(contract.get(key)) is not int for key in (
                    "actor_dim", "critic_dim", "action_dim", "target_num_envs", "num_steps_per_env"))
                or contract.get("physics_dt") != .005 or contract.get("policy_dt") != .02
                or contract.get("target_num_envs") != self.num_envs
                or contract.get("num_steps_per_env") != self.steps_per_rollout
                or not isinstance(state, dict) or state.get("version") != STATE_VERSION
                or not {"source_dynamic_mask", "stream_id"} <= state.keys()):
            raise ValueError("Inherited measurements must be the explicit same-layout flat source journal")
        mask = torch.as_tensor(state["source_dynamic_mask"], device=self.device)
        legacy = SourceBirthExposure(num_envs=self.num_envs, steps_per_rollout=self.steps_per_rollout,
            source_dynamic_mask=mask, stream_id=state["stream_id"], device=self.device)
        legacy.restore_state(state)
        if self.stream_id in legacy.seen_stream_ids:
            raise ValueError("Flat history inheritance requires a new native stream")
        self._flat_history = deepcopy(record)

    def checkpoint_flat_history(self):
        """The checkpoint owns this immutable archive separately from v2 state."""
        if self.collecting or self.fault is not None:
            raise ValueError("Flat history cannot be sealed during pending attribution")
        return deepcopy(self._flat_history)

    def flat_history_metrics(self):
        """Publish fixed inherited measurements once, with a separate prefix."""
        if self._flat_history is None:
            return {}
        rows = self._flat_history["state"]["counts"]
        result = {f"/InheritedFlatExposure/{name}": sum(row[i] for row in rows)
                  for i, name in enumerate(FIELDS)}
        for group, row in zip(GROUPS, rows):
            result.update({f"/InheritedFlatExposure/{group}/{name}": value
                           for name, value in zip(FIELDS, row)})
        return result

    def restore_state(self, state, *, allow_population_change=False):
        """Restore into a fresh observer, preserving counters without replay credit.

        The same stream restores exact observation cursors. A different real
        native stream starts new cursors and cannot claim cross-worker complete
        episodes. An explicitly admitted population change also requires a new
        stream; historical counts retain their original population. Reusing an
        earlier stream identity is rejected.
        """
        if type(allow_population_change) is not bool:
            raise ValueError("Birth exposure population change permission must be boolean")
        if (self.collecting or self.fault is not None or self._flat_history is not None
                or self.last_policy_interval_id is not None
                or bool(self.counts.any()) or self.failed_updates or self.discarded_rollouts
                or self.completed_actor_updates or self.critic_only_updates):
            raise ValueError("Birth exposure restore requires a fresh observer")
        expected = self.checkpoint_state()
        state = deepcopy(state)
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("Birth exposure snapshot fields differ")
        for name in ("version", "num_envs", "steps_per_rollout", "groups", "fields",
                     "skill_qualification", "native_exact_resume_qualified"):
            if state[name] != expected[name]:
                raise ValueError("Birth exposure snapshot dimensions or source lanes differ")
        previous_mask = self._vector(torch.as_tensor(state["source_dynamic_mask"], device=self.device),
                                     "snapshot source_dynamic_mask", torch.bool)
        if (not torch.equal(previous_mask, self.source_dynamic_mask)
                and (not allow_population_change or self.stream_id == state["stream_id"])):
            raise ValueError("Birth exposure snapshot source lanes require an admitted new stream")
        if self.final_birth_definition_sha256 is not None:
            if state["final_birth_definition_sha256"] != self.final_birth_definition_sha256:
                raise ValueError("Final birth exposure source definition differs")
            previous_final_mask = self._vector(torch.as_tensor(state["final_birth_lane_mask"], device=self.device),
                                               "snapshot final_birth_lane_mask", torch.bool)
            if (bool((previous_final_mask & ~previous_mask).any())
                    or not torch.equal(previous_final_mask, self.final_birth_lane_mask)
                    and (not allow_population_change or self.stream_id == state["stream_id"])):
                raise ValueError("Final birth exposure lanes require an admitted new stream")
        streams = state["seen_stream_ids"]
        if (not isinstance(streams, list) or not streams
                or any(not isinstance(value, str) or not value.strip() for value in streams)
                or len(set(streams)) != len(streams)
                or state["stream_id"] != streams[-1]
                or self.stream_id in streams and self.stream_id != state["stream_id"]):
            raise ValueError("Birth exposure stream identity was reused")
        counts = torch.as_tensor(state["counts"], device=self.device)
        if counts.shape != self.counts.shape or counts.dtype != torch.long or bool((counts < 0).any()):
            raise ValueError("Birth exposure snapshot counts must be nonnegative integers")
        counters = ("completed_actor_updates", "critic_only_updates", "failed_updates", "discarded_rollouts")
        if any(type(state[name]) is not int or state[name] < 0 for name in counters):
            raise ValueError("Birth exposure snapshot update counts are invalid")
        policy, physical = state["last_policy_interval_id"], state["last_physical_tick"]
        if ((policy is None) != (physical is None)
                or policy is not None and (type(policy) is not int or policy < 1
                    or type(physical) is not int or physical != PHYSICS_TICKS_PER_POLICY * policy)):
            raise ValueError("Birth exposure snapshot physical clock differs")
        if not isinstance(state["live"], dict) or set(state["live"]) != set(self.live):
            raise ValueError("Birth exposure snapshot episode cursors differ")
        live = {name: self._vector(torch.as_tensor(state["live"][name], device=self.device),
                                   name, template.dtype)
                for name, template in self.live.items()}
        if bool(((live["expected_ids"] < -1) | (live["ticks"] < 0)
                 | (live["ticks"] % PHYSICS_TICKS_PER_POLICY != 0)
                 | (live["motor_off_ticks"] < 0) | (live["motor_off_ticks"] > MOTOR_OFF_PHYSICS_TICKS)
                 | (live["motor_off_ticks"] % PHYSICS_TICKS_PER_POLICY != 0)
                 | (live["seen"] & ((live["expected_ids"] < 0) | (live["ticks"] == 0)))
                 | (~live["seen"] & (live["ticks"] != 0))).any()):
            raise ValueError("Birth exposure snapshot episode clocks are invalid")
        if self.final_birth_definition_sha256 is not None:
            kind = live["kind"]
            if bool((((kind < -2) | (kind > 2))
                     | (live["seen"] & previous_mask & ((kind < -1)
                        | ((kind >= 0) != previous_final_mask)))
                     | (live["seen"] & (kind >= 0) & (live["ground"] != (kind == 1)))).any()):
                raise ValueError("Final birth exposure episode cursors changed class")
        self.counts = counts.clone()
        self.seen_stream_ids = list(streams)
        if self.stream_id == state["stream_id"]:
            self.live = live
            self.last_policy_interval_id, self.last_physical_tick = policy, physical
        else:
            self.seen_stream_ids.append(self.stream_id)
        for name in counters:
            setattr(self, name, state[name])
