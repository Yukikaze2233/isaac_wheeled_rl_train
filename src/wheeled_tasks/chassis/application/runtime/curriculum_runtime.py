"""Stage physical rollout evidence, then credit only verified optimizer samples.

BudgetOwner is the only paid-consumption authority. This adapter mirrors its
tokens, never reserves/refunds work. Native admission and the versioned course
selector belong to the caller; a definition hash is not human authorization.
"""
from copy import deepcopy

import torch

from wheeled_tasks.chassis.tasks.curricula.autonomous_curriculum import AutonomousCurriculum, _digest, validate_definition
from wheeled_tasks.chassis.tasks.curricula.autonomous_exposure import ExposureAccumulator
from wheeled_tasks.chassis.application.runtime.training_budget import BudgetOwner, RECEIPT_SCHEMA


SCHEMA = "v7-curriculum-runtime-state-v1"
BINDING_KEYS = ("campaign_id", "ledger_id", "campaign_policy_sha256", "attempt_id")
CATEGORIES = ("foundation", "current", "previous")


class CurriculumRuntime:
    def __init__(self, definition, *, definition_sha256, budget_owner, training_seeds,
                 num_envs, steps_per_rollout, device="cpu", node_allowlist=None,
                 allow_provisional=False, curriculum=None):
        if type(budget_owner) is not BudgetOwner:
            raise TypeError("Curriculum requires the actual shared BudgetOwner")
        if type(allow_provisional) is not bool:
            raise ValueError("allow_provisional must be an explicit CPU fixture boolean")
        self.device = torch.device(device)
        if allow_provisional and self.device.type != "cpu":
            raise ValueError("Provisional curricula are CPU fixtures only")
        self.definition = deepcopy(validate_definition(definition))
        if (not isinstance(definition_sha256, str) or len(definition_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in definition_sha256)):
            raise ValueError("Explicit raw-byte curriculum SHA-256 is required")
        # The strict caller selector authenticates raw JSON bytes. Their digest
        # differs from the canonical structural digest used by the reducers.
        self.canonical_definition_sha256 = _digest(self.definition)
        for value in (num_envs, steps_per_rollout):
            if type(value) is not int or value < 1:
                raise ValueError("Fixed rollout dimensions must be positive integers")
        if self.definition["clock"]["reference_transitions_per_update"] != num_envs * steps_per_rollout:
            raise ValueError("Definition reference batch differs from actual rollout dimensions")
        self.definition_sha256 = definition_sha256
        self.num_envs, self.steps_per_rollout = num_envs, steps_per_rollout
        self.owner, self.allow_provisional = budget_owner, allow_provisional
        self.training_seeds, self.node_allowlist = list(training_seeds), node_allowlist
        empty_course = self._new_course()
        if curriculum is not None and (type(curriculum) is not AutonomousCurriculum
                or not curriculum.external_budget
                or curriculum.definition_sha256 != self.canonical_definition_sha256
                or curriculum.state_dict() != empty_course.state_dict()):
            raise ValueError("Shared curriculum must have identical definition/seeds and pristine external state")
        self.course = empty_course if curriculum is None else curriculum
        self.exposure = self._new_exposure()
        self.device = self.exposure.device  # Resolve aliases such as cpu:0/cuda.
        self.event_keys = tuple(self.exposure.event_keys)
        summary = budget_owner.summary()
        self.owner_binding = {key: summary[key] for key in BINDING_KEYS}
        self.owner_binding["review_sha256"] = budget_owner.review_sha256
        self.live_episode_ids = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.live_episode_categories = torch.full_like(self.live_episode_ids, -1)
        self.event_high_water = torch.full((len(self.event_keys), num_envs), -1,
                                          dtype=torch.long, device=self.device)
        self.journal = {}
        self.unpaid_rollouts = 0
        self.evidence_fault = None
        self._clear_pending()

    def _new_course(self):
        return AutonomousCurriculum(self.definition, training_seeds=self.training_seeds,
            node_allowlist=self.node_allowlist, allow_provisional=self.allow_provisional,
            external_budget=True)

    def _new_exposure(self):
        return ExposureAccumulator(self.definition, self.num_envs, self.device,
                                   allow_provisional=self.allow_provisional)

    def _clear_pending(self, *, cleared_invalid=None):
        # Completion prepares this scalar BEFORE any visible course commit.
        # No tensor allocation/device operation is allowed after that commit.
        if cleared_invalid is None:
            cleared_invalid = torch.zeros((), dtype=torch.long, device=self.device)
        self.collecting = False
        self.actor_enabled = None
        self.node_id = None
        self.rows = []
        self.pending_token = None
        self.population = None
        self.pending_episode_nodes = None
        self.pending_episode_started = None
        self.invalid = cleared_invalid

    def _idle(self):
        if self.evidence_fault is not None:
            raise ValueError("Faulted physical evidence cannot continue or checkpoint")
        if self.collecting or self.pending_token is not None:
            raise ValueError("Unsettled rollout/update cannot checkpoint or change curriculum")

    @property
    def has_pending(self):
        return self.pending_token is not None

    def _vector(self, value, name, dtype):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                or value.dtype != dtype or value.device != self.device):
            raise ValueError(f"{name} requires [{self.num_envs}] {dtype} on {self.device}")

    def _matrix(self, value, name):
        if (not isinstance(value, torch.Tensor)
                or value.shape != (self.steps_per_rollout, self.num_envs)
                or value.dtype != torch.bool or value.device != self.device):
            raise ValueError(f"{name} requires fixed [steps, environments] boolean tensor")

    def begin_rollout(self, *, actor_enabled):
        self._idle()
        if type(actor_enabled) is not bool:
            raise ValueError("actor_enabled must be bool")
        self.collecting, self.actor_enabled = True, actor_enabled
        self.node_id = self.course.node["id"]
        self.pending_episode_nodes = self.exposure.episode_nodes.clone()
        self.pending_episode_started = self.exposure.episode_started.clone()

    def observe_transition(self, *, step_index, node_indices, episode_ids,
                           returned_episode_ids, learning_mask, done, episode_complete,
                           key_phase_events=None, key_phase_occurrence_ids=None,
                           category_indices=None):
        """Copy the paid pre-reset row and its separately returned birth IDs.

        Event IDs are per-lane/per-episode monotonically increasing occurrence
        IDs, with -1 on non-pulses. Observers, not this adapter, establish the
        physical meaning. Occupancy or repeated pulses cannot earn extra credit.
        """
        if not self.collecting or self.pending_token is not None:
            raise ValueError("Observation requires an unbound collecting rollout")
        if type(step_index) is not int or step_index != len(self.rows) or step_index >= self.steps_per_rollout:
            raise ValueError("Rollout storage step was repeated, skipped or exceeded")
        for name, value in (("node_indices", node_indices), ("episode_ids", episode_ids),
                            ("returned_episode_ids", returned_episode_ids)):
            self._vector(value, name, torch.long)
        for name, value in (("learning_mask", learning_mask), ("done", done),
                            ("episode_complete", episode_complete)):
            self._vector(value, name, torch.bool)
        events, occurrences = key_phase_events, key_phase_occurrence_ids
        if (not isinstance(events, dict) or not isinstance(occurrences, dict)
                or set(events) != set(self.event_keys) or set(occurrences) != set(self.event_keys)):
            raise ValueError("Every declared event needs an explicit pulse and occurrence-ID vector")
        for key in self.event_keys:
            self._vector(events[key], key, torch.bool)
            self._vector(occurrences[key], key + " occurrence IDs", torch.long)
        if category_indices is None:
            if not self.allow_provisional:
                raise ValueError("Explicit episode sampling categories are required outside CPU fixtures")
            categories = torch.full_like(node_indices, -1)
        else:
            self._vector(category_indices, "category_indices", torch.long)
            categories = category_indices
        # Tensor-only validation here: defer host synchronization to bind/finish.
        bad = ((episode_ids < 0) | (returned_episode_ids < 0)
               | ((self.live_episode_ids >= 0) & (episode_ids != self.live_episode_ids))
               | ((~done) & (returned_episode_ids != episode_ids))
               | (done & (returned_episode_ids <= episode_ids))
               | (episode_complete & ~done)
               | (self.pending_episode_started & (node_indices != self.pending_episode_nodes))
               | (node_indices < -1) | (node_indices >= len(self.exposure.node_ids)))
        if category_indices is not None:
            frontier = self.exposure.node_indices[self.node_id]
            bad |= ((categories < 0) | (categories > 2)
                    | ((self.live_episode_categories >= 0) & (categories != self.live_episode_categories))
                    | ((categories == 0) & (node_indices != -1))
                    | ((categories == 1) & (node_indices != frontier))
                    | ((categories == 2) & ((node_indices < 0) | (node_indices == frontier))))
        self.pending_episode_nodes = torch.where(self.pending_episode_started,
                                                self.pending_episode_nodes, node_indices)
        self.pending_episode_nodes = torch.where(done, -1, self.pending_episode_nodes)
        self.pending_episode_started = ~done
        for index, key in enumerate(self.event_keys):
            pulse, event_id = events[key], occurrences[key]
            bad |= (pulse & ((event_id < 0) | (event_id <= self.event_high_water[index])))
            bad |= ~pulse & (event_id != -1)
            self.event_high_water[index] = torch.where(pulse, event_id, self.event_high_water[index])
            self.event_high_water[index] = torch.where(done, -1, self.event_high_water[index])
        self.invalid += bad.sum()
        self.live_episode_ids = returned_episode_ids.clone()
        self.live_episode_categories = torch.where(done, -1, categories).clone()
        self.rows.append({"nodes": node_indices.clone(), "learning": learning_mask.clone(),
                          "done": done.clone(), "complete": episode_complete.clone(),
                          "categories": categories.clone(),
                          "events": {key: value.clone() for key, value in events.items()}})

    def _verified(self, token, *, previous=False):
        result = self.owner.verify_update(token, allow_previous_lease=previous)
        if any(result["summary"][key] != self.owner_binding[key] for key in BINDING_KEYS):
            raise ValueError("Shared budget ownership changed")
        if result["review_sha256"] != self.owner_binding["review_sha256"]:
            raise ValueError("Shared budget review changed")
        return result

    @staticmethod
    def _update_id(token):
        return f"{token['ledger_id']}:{token['attempt_id']}:{token['ordinal']}"

    def bind_paid_update(self, token, *, final_learning_mask, optimized_population_mask):
        if not self.collecting or self.pending_token is not None:
            raise ValueError("Paid binding requires exactly one collecting rollout")
        receipt = self._verified(token)
        if receipt["row"]["status"] != "reserved":
            raise ValueError("Bind only an actual durable pre-optimizer reservation")
        if token["phase"] != ("actor" if self.actor_enabled else "critic"):
            raise ValueError("Paid phase differs from the collected rollout")
        uid = self._update_id(token)
        if uid in self.journal:
            raise ValueError("Cannot bind a consumed update to new samples")
        # Mirror the already durable reservation even if evidence validation
        # below fails. The caller must mark the owner failure, then abort here.
        self.course.observe_external_reservation(uid, token, receipt["row"])
        self.pending_token = deepcopy(token)
        if len(self.rows) != self.steps_per_rollout:
            raise ValueError("Optimizer requires the complete collected rollout")
        self._matrix(final_learning_mask, "final ownership mask")
        self._matrix(optimized_population_mask, "actual optimized population")
        observed = torch.stack([row["learning"] for row in self.rows])
        if (not torch.equal(observed, final_learning_mask)
                or bool((optimized_population_mask & ~final_learning_mask).any())):
            raise ValueError("Actual PPO population differs from final ownership evidence")
        if int(self.invalid.item()):
            raise ValueError("Invalid episode/reset identity or repeated physical event pulse")
        self.population = optimized_population_mask.clone()

    def _replay(self, *, credit):
        accumulator = self._new_exposure()
        accumulator.load_state_dict(self.exposure.state_dict())
        invalid_rows = int(self.invalid.item())
        # Invalid physical history must not make failure cleanup repeat the
        # original observer error. Drop its history only for zero credit, and
        # permanently block continuation/checkpoint when this result commits.
        if invalid_rows and not credit:
            accumulator.reset()
        accumulator.begin_update(actor_enabled=credit)
        if not invalid_rows or credit:
            for index, row in enumerate(self.rows):
                mask = self.population[index] if credit else torch.zeros_like(row["learning"])
                accumulator.observe(row["nodes"], mask, row["done"], row["complete"],
                                    key_phase_events=row["events"])
        result = accumulator.flush(self.node_id, actor_optimized=credit)
        raw = torch.stack((sum((r["learning"].sum() for r in self.rows),
                               torch.zeros((), dtype=torch.long, device=self.device)),
                           sum((r["done"].sum() for r in self.rows),
                               torch.zeros((), dtype=torch.long, device=self.device))))
        raw_values = raw.cpu().tolist()
        result["observed_transitions"] = len(self.rows) * self.num_envs
        result["ownership_eligible_transitions"] = raw_values[0]
        result["physical_episode_ends"] = raw_values[1]
        # Reset sampling quotas and actual optimizer population are distinct.
        zero = torch.zeros((), dtype=torch.long, device=self.device)
        grouped = torch.stack([
            sum(((row["learning"] & (row["categories"] == category)).sum()
                 for row in self.rows), zero.clone())
            for category in range(3)] + [
            sum(((self.population[index] & (row["categories"] == category)).sum()
                 for index, row in enumerate(self.rows)), zero.clone()) if credit else zero.clone()
            for category in range(3)] + [self.population.sum() if credit else zero.clone()])
        category_values = grouped.cpu().tolist()
        result["ownership_eligible_by_category"] = dict(zip(CATEGORIES, category_values[:3]))
        result["direct_loss_transitions_by_category"] = dict(zip(CATEGORIES, category_values[3:6]))
        result["direct_loss_total"] = category_values[6]
        result["unclassified_fixture_direct_loss_transitions"] = category_values[6] - sum(category_values[3:6])
        result["evidence_fault"] = (f"Invalid physical episode/node/event evidence in {invalid_rows} lane-rows"
                                    if invalid_rows else None)
        return accumulator, result

    def _finish(self, token, *, failed):
        already_recorded = any(entry["token"] == token for entry in self.journal.values())
        live = self._verified(token, previous=already_recorded)
        uid = self._update_id(token)
        expected = {"failed", "postcheck_failed"} if failed else {"completed"}
        if live["row"]["status"] not in expected:
            raise ValueError("Shared ledger has not verified the requested completion/failure")
        if uid in self.journal:
            previous = self.journal[uid]
            if previous["token"] != token or previous["failed"] != failed:
                raise ValueError("Conflicting runtime completion replay")
            return deepcopy(previous)
        if failed and self.collecting and self.pending_token is None:
            # A failure between durable reserve and binding still consumes its
            # actual row. Never fabricate a reserved status for that failure.
            self.course.observe_external_reservation(uid, token, live["row"])
            self.pending_token = deepcopy(token)
        if token != self.pending_token or not self.collecting:
            raise ValueError("Update is not bound to this pending rollout")
        if not failed and (self.population is None or not live["row"]["optimizer_completed"]):
            raise ValueError("Successful credit requires actual optimizer completion and its population")
        credit = not failed and self.actor_enabled
        accumulator, result = self._replay(credit=credit)
        staged_course = self._new_course()
        staged_course.load_state_dict(self.course.state_dict())
        staged_course.finish_update(uid, actor_optimized=credit, exposure=result["exposure"])
        entry = {"token": deepcopy(token), "failed": failed, "actor_credit": credit,
                 "node_id": self.node_id, **result}
        prepared_journal = deepcopy(self.journal)
        prepared_journal[uid] = entry
        returned_entry = deepcopy(entry)
        cleared_invalid = torch.zeros((), dtype=torch.long, device=self.device)
        self.course.load_state_dict(staged_course.state_dict())
        self.journal = prepared_journal
        self.exposure = accumulator
        self.evidence_fault = result["evidence_fault"]
        self._clear_pending(cleared_invalid=cleared_invalid)
        return returned_entry

    def finish_paid_update(self, token):
        """Call only after the complete train update and all postchecks returned."""
        return self._finish(token, failed=False)

    def fail_paid_update(self, token):
        """The owner must record failure first. No actor exposure is awarded."""
        return self._finish(token, failed=True)

    def abort_unpaid_rollout(self, reason):
        if not self.collecting or self.pending_token is not None:
            raise ValueError("Only an unreserved rollout can be abandoned without a paid token")
        if not isinstance(reason, str) or not reason:
            raise ValueError("Unpaid rollout abandonment needs an explicit reason")
        accumulator, result = self._replay(credit=False)
        cleared_invalid = torch.zeros((), dtype=torch.long, device=self.device)
        self.exposure = accumulator
        self.evidence_fault = result["evidence_fault"]
        self.unpaid_rollouts += 1
        self._clear_pending(cleared_invalid=cleared_invalid)
        return {"reason": reason, "paid_update": False, **result}

    def reset(self, ids):
        """Only external resets call this; normal done rows already clear history."""
        self._idle()
        if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 1
                or ids.device != self.device or bool(((ids < 0) | (ids >= self.num_envs)).any())
                or ids.unique().numel() != ids.numel()):
            raise ValueError("Reset IDs must be unique valid lanes on the runtime device")
        self.exposure.reset(ids)
        self.live_episode_ids[ids] = -1
        self.live_episode_categories[ids] = -1
        self.event_high_water[:, ids] = -1

    def _budget_receipt(self, receipt):
        if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
            raise ValueError("Curriculum checkpoint requires its actual budget checkpoint receipt")
        for key, value in self.owner_binding.items():
            if receipt.get(key) != value:
                raise ValueError(f"Checkpoint budget ownership mismatch: {key}")
        current = self.owner.summary()
        for key in ("paid_updates_attempt", "paid_updates_global"):
            if type(receipt.get(key)) is not int or not 0 <= receipt[key] <= current[key]:
                raise ValueError("Checkpoint claims impossible paid counters")
        if not isinstance(receipt.get("checkpoint_id"), str) or not receipt["checkpoint_id"]:
            raise ValueError("Missing prepared/registered checkpoint identity")

    def checkpoint_state(self, *, budget_receipt, env_episode_ids=None):
        self._idle()
        self._budget_receipt(budget_receipt)
        if budget_receipt["paid_updates_attempt"] != self.owner.summary()["paid_updates_attempt"]:
            raise ValueError("Checkpoint preparation is stale for this attempt")
        live = self.live_episode_ids.clone()
        if env_episode_ids is not None:
            self._vector(env_episode_ids, "environment episode IDs", torch.long)
            if bool(((env_episode_ids < 0) | ((live >= 0) & (live != env_episode_ids))).any()):
                raise ValueError("Environment episode IDs differ from returned rollout state")
            live = env_episode_ids.clone()
        if bool((live < 0).any()):
            raise ValueError("Initial/reset checkpoint needs explicit environment episode IDs")
        return {"schema": SCHEMA, "definition_sha256": self.definition_sha256,
                "canonical_definition_sha256": self.canonical_definition_sha256,
                "num_envs": self.num_envs, "steps_per_rollout": self.steps_per_rollout,
                "owner_binding": deepcopy(self.owner_binding), "budget_receipt": deepcopy(budget_receipt),
                "course": self.course.state_dict(), "exposure": self.exposure.state_dict(),
                "live_episode_ids": live, "live_episode_categories": self.live_episode_categories.clone(),
                "event_high_water": self.event_high_water.clone(),
                "journal": deepcopy(self.journal), "unpaid_rollouts": self.unpaid_rollouts,
                "evidence_fault": self.evidence_fault}

    def load_checkpoint_state(self, state, *, budget_receipt, env_episode_ids):
        """Verify the registered budget snapshot before atomically restoring state."""
        self._idle()
        self._budget_receipt(budget_receipt)
        self.owner.verify_checkpoint_receipt(budget_receipt)
        expected_keys = {"schema", "definition_sha256", "canonical_definition_sha256", "num_envs", "steps_per_rollout", "owner_binding",
                         "budget_receipt", "course", "exposure", "live_episode_ids", "live_episode_categories", "event_high_water",
                         "journal", "unpaid_rollouts", "evidence_fault"}
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError("Unknown or incomplete curriculum checkpoint")
        if state["evidence_fault"] is not None:
            raise ValueError("Faulted physical evidence cannot restore")
        for key, expected in (("schema", SCHEMA), ("definition_sha256", self.definition_sha256),
                              ("canonical_definition_sha256", self.canonical_definition_sha256),
                              ("num_envs", self.num_envs), ("steps_per_rollout", self.steps_per_rollout),
                              ("owner_binding", self.owner_binding), ("budget_receipt", budget_receipt)):
            if state[key] != expected:
                raise ValueError(f"Curriculum checkpoint identity mismatch: {key}")
        self._vector(env_episode_ids, "restored environment episode IDs", torch.long)
        live = state["live_episode_ids"]
        if not isinstance(live, torch.Tensor) or live.dtype != torch.long or live.shape != (self.num_envs,):
            raise ValueError("Invalid restored episode-ID state")
        live = live.to(self.device).clone()
        if bool((live < 0).any()) or not torch.equal(live, env_episode_ids):
            raise ValueError("Environment episode identity differs from the checkpoint")
        high_water = state["event_high_water"]
        if (not isinstance(high_water, torch.Tensor) or high_water.dtype != torch.long
                or high_water.shape != self.event_high_water.shape or bool((high_water < -1).any())):
            raise ValueError("Invalid restored event occurrence history")
        categories = state["live_episode_categories"]
        if (not isinstance(categories, torch.Tensor) or categories.dtype != torch.long
                or categories.shape != (self.num_envs,) or bool(((categories < -1) | (categories > 2)).any())):
            raise ValueError("Invalid restored sampling category history")
        categories = categories.to(self.device).clone()
        if type(state["unpaid_rollouts"]) is not int or state["unpaid_rollouts"] < 0:
            raise ValueError("Invalid unpaid rollout diagnostic")
        course, accumulator = self._new_course(), self._new_exposure()
        course.load_state_dict(state["course"])
        accumulator.load_state_dict(state["exposure"])
        if (accumulator.collecting or accumulator.actor_enabled or bool(accumulator.counts.any())
                or bool(accumulator.invalid_node_count.any())):
            raise ValueError("Cannot restore partial or invalid exposure reductions")
        if bool(((accumulator.episode_nodes < -1)
                 | (accumulator.episode_nodes >= len(accumulator.node_ids))
                 | (~accumulator.episode_started & (accumulator.episode_nodes != -1))
                 | (accumulator.episode_has_rl & (~accumulator.episode_started
                                                | (accumulator.episode_nodes < 0)))).any()):
            raise ValueError("Invalid cross-rollout episode ownership history")
        frontier = accumulator.node_indices[course.node["id"]]
        if bool(((~accumulator.episode_started & (categories != -1))
                 | (accumulator.episode_started & (categories == -1) & (not self.allow_provisional))
                 | ((categories == 0) & (accumulator.episode_nodes != -1))
                 | ((categories == 1) & (accumulator.episode_nodes != frontier))
                 | ((categories == 2) & ((accumulator.episode_nodes < 0)
                                         | (accumulator.episode_nodes == frontier)))).any()):
            raise ValueError("Sampling categories differ from episode-latched node history")
        reservations = course.state_dict()["reservations"]
        journal = state["journal"]
        if not isinstance(journal, dict) or set(journal) != set(reservations):
            raise ValueError("Exposure journal and mirrored reservations differ")
        for uid, entry in journal.items():
            token = entry["token"]
            verified = self._verified(token, previous=True)
            if (uid != self._update_id(token) or token["ordinal"] > budget_receipt["paid_updates_attempt"]
                    or token["sequence"] > budget_receipt["paid_updates_global"]
                    or reservations[uid].get("external_budget_token") != token):
                raise ValueError("Mirrored reservation does not belong to this checkpoint")
            if type(entry["failed"]) is not bool or type(entry["actor_credit"]) is not bool:
                raise ValueError("Invalid completion flags")
            if entry.get("evidence_fault") is not None:
                raise ValueError("Faulted physical evidence journal cannot restore")
            status = verified["row"]["status"]
            expected_status = {"failed", "postcheck_failed"} if entry["failed"] else {"completed"}
            if (status not in expected_status or entry["actor_credit"] != (not entry["failed"] and token["phase"] == "actor")
                    or reservations[uid]["completion"] is None
                    or reservations[uid]["completion"]["actor_optimized"] != entry["actor_credit"]
                    or reservations[uid]["completion"]["exposure"] != entry["exposure"]):
                raise ValueError("Course credit differs from actual paid completion")
        prepared_high_water = high_water.to(self.device).clone()
        prepared_journal = deepcopy(journal)
        self.course.load_state_dict(course.state_dict())
        self.exposure = accumulator
        self.live_episode_ids = live
        self.live_episode_categories = categories
        self.event_high_water = prepared_high_water
        self.journal, self.unpaid_rollouts = prepared_journal, state["unpaid_rollouts"]

    def sampling_plan(self):
        return self.course.sampling_plan()

    def request_evaluation(self, *args, **kwargs):
        self._idle()
        return self.course.request_evaluation(*args, **kwargs)

    def record_evaluation(self, *args, **kwargs):
        self._idle()
        return self.course.record_evaluation(*args, **kwargs)
