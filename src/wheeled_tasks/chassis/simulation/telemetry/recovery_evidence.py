"""Measure the legacy recovery funnel and actual PPO loss population.

Physical events are not V7 qualification or curriculum credit. Only a completed
actor optimizer plus its postchecks earns direct-loss exposure. The paid owner
remains the sole authority for consumption, including failed updates.
"""
from copy import deepcopy
import hashlib
import json

import torch


EVENTS = ("script_ready", "rl_entered", "legacy_physical_hold_qualified",
          "legacy_followup_complete")
FAILURES = ("recovery_script", "recovery_actor_failure", "recovery_premature_release")
LESSONS = ("unclassified", "prefix_stand", "prefix_release", "original_chain")
FIELDS = ("policy_intervals", "learning_eligible_intervals", "masked_intervals",
          "observed_episodes", "closed_episodes", "terminated_episodes", "timeout_episodes",
          *FAILURES, *EVENTS, "actor_direct_loss_intervals",
          *("actor_direct_loss_" + name for name in EVENTS),
          "episodes_with_post_handoff_actor_loss",
          "source_contact_or_attitude_measured_intervals",
          "post_handoff_source_contact_or_attitude_termination_episodes")
BREAKDOWN_FIELDS = ("learning_eligible_intervals", "masked_intervals", "closed_episodes",
                    *FAILURES, "script_ready", "rl_entered", "legacy_followup_complete",
                    "actor_direct_loss_intervals", "episodes_with_post_handoff_actor_loss",
                    "source_contact_or_attitude_measured_intervals",
                    "post_handoff_source_contact_or_attitude_termination_episodes")
SCHEMA = "v7-recovery-evidence-state-v2"
TRANSITION_SCHEMA = "v6-recovery-physical-transition-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class RecoveryEvidenceJournal:
    """Separate controller observations from successful actor loss samples.

    Identity/value validation is accumulated on device and checked once before
    optimizer work. Episodes may span rollouts; milestones are accepted once per
    real episode. Snapshot restore is deliberately unsupported: these counters
    do not certify an exact native physical restart.
    """
    def __init__(self, *, num_envs, steps_per_rollout, source_identity, device="cpu"):
        if any(type(value) is not int or value < 1 for value in (num_envs, steps_per_rollout)):
            raise ValueError("Recovery evidence requires positive rollout dimensions")
        names = source_identity.get("profile_names")
        if (source_identity.get("schema") != "v6-recovery-physical-event-source-v1"
                or not isinstance(names, list) or not names or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)):
            raise ValueError("Recovery evidence requires the real producer profile identity")
        self.source_identity = deepcopy(source_identity)
        self.source_identity_sha256 = _digest(source_identity)
        self.num_envs, self.steps_per_rollout = num_envs, steps_per_rollout
        self.device = torch.device(device)
        self.counts = torch.zeros((len(LESSONS), len(names), len(FIELDS)),
                                  dtype=torch.long, device=self.device)
        self.device = self.counts.device
        self.live_ids = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.last_credited_actor_episode_ids = torch.full_like(self.live_ids, -1)
        self.live_lessons = torch.full_like(self.live_ids, -1)
        self.live_profiles = torch.full_like(self.live_ids, -1)
        self.seen_events = torch.zeros((len(EVENTS), num_envs), dtype=torch.bool, device=self.device)
        self.closed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.new_episode = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.lane_enabled = None
        self.last_interval_id = None
        self.completed_actor_updates = self.critic_only_updates = self.failed_updates = 0
        self.discarded_rollouts = 0
        self.fault = None
        self.collecting = False
        self.rows = []
        self.population = None

    def begin_rollout(self, *, actor_enabled):
        if self.fault is not None or self.collecting or type(actor_enabled) is not bool:
            raise ValueError("Recovery evidence cannot begin a pending/faulted rollout")
        self.collecting, self.actor_enabled = True, actor_enabled
        self.rows, self.population = [], None
        self.invalid = torch.zeros((), dtype=torch.bool, device=self.device)
        self.pending_ids = self.live_ids.clone()
        self.pending_lessons = self.live_lessons.clone()
        self.pending_profiles = self.live_profiles.clone()
        self.pending_events = self.seen_events.clone()
        self.pending_closed = self.closed.clone()
        self.pending_new_episode = self.new_episode.clone()
        self.pending_interval_id = self.last_interval_id
        self.physical_committed = False

    def _tensor(self, name, value, dtype):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                or value.dtype != dtype or value.device != self.device):
            raise ValueError(f"Recovery {name} must be {dtype}[N] on the producer device")
        return value.detach().clone()

    def observe_transition(self, *, step_index, payload, source_contact_or_attitude=None):
        if (not self.collecting or self.population is not None or type(step_index) is not int
                or step_index != len(self.rows) or step_index >= self.steps_per_rollout):
            raise ValueError("Recovery evidence requires the next actual rollout row")
        expected = {"schema", "source_identity_sha256", "enabled", "episode_ids",
                    "episode_lesson_ids", "episode_profile_ids", "events", "occurrence_ids",
                    "policy_learning_mask", "phase", "done", "terminated", "timeouts",
                    "failure_reasons", "returned_episode_ids", "reset_mask", "policy_interval_id"}
        if (not isinstance(payload, dict) or set(payload) != expected
                or payload["schema"] != TRANSITION_SCHEMA
                or payload["source_identity_sha256"] != self.source_identity_sha256
                or set(payload["events"]) != set(EVENTS)
                or set(payload["occurrence_ids"]) != set(EVENTS)
                or set(payload["failure_reasons"]) != set(FAILURES)):
            raise ValueError("Recovery transition differs from the frozen physical producer")
        interval_id = payload["policy_interval_id"]
        if (type(interval_id) is not int or interval_id < 1
                or self.pending_interval_id is not None and interval_id != self.pending_interval_id + 1):
            self.fault = "duplicate_or_missing_physical_interval"
            raise ValueError("Recovery evidence requires the next physical policy interval")
        row = {key: self._tensor(key, payload[key], torch.long) for key in (
            "episode_ids", "episode_lesson_ids", "episode_profile_ids", "returned_episode_ids", "phase")}
        row.update({key: self._tensor(key, payload[key], torch.bool) for key in (
            "enabled", "policy_learning_mask", "done", "terminated", "timeouts", "reset_mask")})
        row["events"] = torch.stack([self._tensor(name, payload["events"][name], torch.bool)
                                      for name in EVENTS])
        occurrence = torch.stack([self._tensor(name, payload["occurrence_ids"][name], torch.long)
                                  for name in EVENTS])
        row["failures"] = torch.stack([self._tensor(name, payload["failure_reasons"][name], torch.bool)
                                        for name in FAILURES])
        # This optional witness is already captured before native autoreset.
        # Missing diagnostics are unmeasured, rather than zero source failures.
        row["source_contact_or_attitude_measured"] = torch.full_like(
            row["enabled"], source_contact_or_attitude is not None)
        row["source_contact_or_attitude"] = (torch.zeros_like(row["enabled"])
            if source_contact_or_attitude is None else self._tensor(
                "source_contact_or_attitude", source_contact_or_attitude, torch.bool))
        ids, lessons, profiles, enabled = (row[key] for key in (
            "episode_ids", "episode_lesson_ids", "episode_profile_ids", "enabled"))
        known = self.pending_ids >= 0
        if self.lane_enabled is None:
            self.lane_enabled = enabled.clone()
        self.invalid |= (enabled != self.lane_enabled).any()
        self.invalid |= (ids < 0).any() | (known & (ids != self.pending_ids)).any()
        self.invalid |= (enabled & ((lessons < -1) | (lessons > 2)
                                    | (profiles < 0) | (profiles >= self.counts.shape[1]))).any()
        self.invalid |= (~enabled & ((lessons != -1) | (profiles != -1))).any()
        # Labels belong to the physical episode; a global curriculum clock may
        # not silently re-label a still-running trajectory.
        same_episode = known & (ids == self.pending_ids) & (self.pending_profiles >= 0)
        self.invalid |= (same_episode & ((lessons != self.pending_lessons)
                                         | (profiles != self.pending_profiles))).any()
        new = self.pending_new_episode | ~known
        row["new_episodes"] = new & enabled
        self.pending_events[:, new] = False
        self.pending_closed[new] = False
        row["new_done"] = row["done"] & enabled & ~self.pending_closed
        self.invalid |= ((row["terminated"] | row["timeouts"]) & ~row["done"]).any()
        self.invalid |= (row["reset_mask"] & ~row["done"]).any()
        self.invalid |= (row["returned_episode_ids"] != ids + row["reset_mask"].long()).any()
        self.invalid |= (row["events"] & (~enabled.unsqueeze(0) | self.pending_events)).any()
        self.invalid |= (torch.where(row["events"], occurrence < 0, occurrence != -1)).any()
        self.invalid |= (row["failures"] & ~enabled.unsqueeze(0)).any()
        self.invalid |= (row["failures"].long().sum(0) > 1).any()
        self.pending_events |= row["events"]
        # Keep the old episode's handoff even when this transition resets it.
        row["ever_rl_entered"] = self.pending_events[EVENTS.index("rl_entered")].clone()
        self.pending_closed |= row["done"] & enabled
        self.pending_ids = row["returned_episode_ids"].clone()
        self.pending_lessons, self.pending_profiles = lessons.clone(), profiles.clone()
        reset = row["reset_mask"]
        # A reset opens an as-yet unobserved episode; its labels must be read
        # from the next actual physical transition, not guessed here.
        self.pending_lessons[reset] = self.pending_profiles[reset] = -1
        self.pending_events[:, reset] = False
        self.pending_closed[reset] = False
        # Separate first observation from the next-ID continuation check.
        self.pending_new_episode = reset.clone()
        self.pending_interval_id = interval_id
        self.rows.append(row)

    def _validated_rows(self, *, require_full):
        if not self.collecting or (require_full and len(self.rows) != self.steps_per_rollout):
            raise ValueError("Recovery evidence requires a complete actual rollout")
        if bool(self.invalid):
            self.fault = "invalid_episode_event_identity"
            raise ValueError("Recovery physical episode/event identity is inconsistent")
        return self.rows

    def _count(self, values, rows):
        additions = torch.zeros_like(self.counts)
        width = self.counts.shape[1]
        for row, fields in zip(rows, values):
            selected = row["enabled"]
            index = (row["episode_lesson_ids"][selected] + 1) * width + row["episode_profile_ids"][selected]
            additions.view(-1, len(FIELDS)).index_add_(0, index, fields[:, selected].T.long())
        return additions

    def _commit_physical(self, *, require_full=True):
        rows = self._validated_rows(require_full=require_full)
        if self.physical_committed:
            return
        values = []
        for row in rows:
            learning = row["policy_learning_mask"]
            columns = [torch.ones_like(learning), learning, ~learning, row["new_episodes"],
                       row["new_done"], row["new_done"] & row["terminated"],
                       row["new_done"] & row["timeouts"]]
            columns += list(row["failures"] & row["new_done"].unsqueeze(0))
            columns += list(row["events"])
            columns += [torch.zeros_like(learning) for _ in range(1 + len(EVENTS))]
            columns += [torch.zeros_like(learning), row["source_contact_or_attitude_measured"],
                row["new_done"] & row["terminated"] & row["ever_rl_entered"]
                & row["source_contact_or_attitude"]]
            values.append(torch.stack(columns))
        additions = self._count(values, rows)
        next_counts = self.counts + additions
        next_ids = self.pending_ids.clone()
        next_lessons, next_profiles = self.pending_lessons.clone(), self.pending_profiles.clone()
        next_events, next_closed = self.pending_events.clone(), self.pending_closed.clone()
        next_new_episode = self.pending_new_episode.clone()
        # Prepare every tensor before any visible commit. Cleanup after an
        # allocation failure must never count the same physical rollout twice.
        self.counts, self.live_ids = next_counts, next_ids
        self.live_lessons, self.live_profiles = next_lessons, next_profiles
        self.seen_events, self.closed = next_events, next_closed
        self.new_episode, self.last_interval_id = next_new_episode, self.pending_interval_id
        self.physical_committed = True

    def bind_loss_population(self, *, final_learning_mask, optimized_population_mask):
        if self.population is not None:
            raise ValueError("Recovery loss population already bound")
        rows = self._validated_rows(require_full=True)
        shape = (self.steps_per_rollout, self.num_envs)
        for value in (final_learning_mask, optimized_population_mask):
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.dtype != torch.bool or value.device != self.device):
                raise ValueError("Recovery loss masks must match actual PPO storage")
        observed = torch.stack([row["policy_learning_mask"] for row in rows])
        if (not torch.equal(final_learning_mask, observed)
                or bool((optimized_population_mask & ~final_learning_mask).any())):
            self.fault = "loss_population_mismatch"
            raise ValueError("Recovery final PPO masks differ from physical intervals")
        self.population = optimized_population_mask.detach().clone()
        self._commit_physical()

    def finish_update(self, *, actor_optimized):
        if type(actor_optimized) is not bool or actor_optimized != self.actor_enabled or self.population is None:
            raise ValueError("Recovery credit requires the actual optimizer phase and bound loss population")
        self._validated_rows(require_full=True)
        if actor_optimized:
            values = []
            next_credited_ids = self.last_credited_actor_episode_ids.clone()
            for row, population in zip(self.rows, self.population):
                # Credit the first actually optimized post-handoff sample, even
                # in an open episode. Native episode IDs only advance on reset.
                first = (row["enabled"] & row["ever_rl_entered"] & population
                         & (row["episode_ids"] != next_credited_ids))
                next_credited_ids[first] = row["episode_ids"][first]
                columns = [torch.zeros_like(population)
                           for _ in range(FIELDS.index("actor_direct_loss_intervals"))]
                columns += [population] + list(row["events"] & population.unsqueeze(0))
                columns += [first, torch.zeros_like(population), torch.zeros_like(population)]
                values.append(torch.stack(columns))
            additions = self._count(values, self.rows)
            next_counts = self.counts + additions
            self.counts, self.last_credited_actor_episode_ids = next_counts, next_credited_ids
            self.completed_actor_updates += 1
        else:
            self.critic_only_updates += 1
        self._close_rollout()

    def fail_update(self):
        if not self.collecting:
            return
        self._commit_physical()
        self.failed_updates += 1
        self._close_rollout()

    def discard_rollout(self):
        """An unpaid/partial collection retains measurements, never actor credit."""
        if not self.collecting:
            return
        self._commit_physical(require_full=False)
        self.discarded_rollouts += 1
        self._close_rollout()

    def _close_rollout(self):
        self.collecting, self.rows, self.population = False, [], None

    def checkpoint_state(self):
        if self.fault is not None or self.collecting:
            raise ValueError("Recovery evidence cannot seal pending/faulted samples")
        return {"schema": SCHEMA, "source_identity": deepcopy(self.source_identity),
                "source_identity_sha256": self.source_identity_sha256,
                "lesson_names": list(LESSONS), "field_names": list(FIELDS),
                "counts": self.counts.detach().cpu().tolist(),
                "completed_actor_updates": self.completed_actor_updates,
                "critic_only_updates": self.critic_only_updates,
                "failed_updates": self.failed_updates, "discarded_rollouts": self.discarded_rollouts,
                "last_physical_policy_interval_id": self.last_interval_id,
                "counting_scope": "current_worker_only_not_resumed",
                "parent_evidence_restored": False,
                "post_handoff_actor_episode_scope": "first successful actor loss sample per native episode; open episodes included",
                "source_contact_or_attitude_scope": (
                    "nonexclusive source reason, including concurrent mechanical reasons; "
                    "only supplied intervals measured; zero measured intervals means unavailable"),
                "scope": "legacy_physical_funnel_and_direct_actor_loss_measurement",
                "skill_qualification": "not_assessed", "native_exact_resume_qualified": False}

    def metrics(self):
        """Read existing worker counts by lesson and actual initial profile.

        These are measurements, not skill passes or resumed exposure credit.
        Aggregate on device and transfer once; retain the existing total tags.
        """
        rows = torch.cat((self.counts.sum(1), self.counts.sum(0))).detach().cpu().tolist()
        lessons = rows[:len(LESSONS)]
        totals = [sum(row[column] for row in lessons) for column in range(len(FIELDS))]
        result = {f"/RecoveryEvidence/{name}": value for name, value in zip(FIELDS, totals)}
        columns = [(name, FIELDS.index(name)) for name in BREAKDOWN_FIELDS]
        for axis, names, values in (("lesson", LESSONS, lessons),
                ("profile", self.source_identity["profile_names"], rows[len(LESSONS):])):
            for label, row in zip(names, values):
                result.update({f"/RecoveryEvidence/{axis}/{label}/{name}": row[column]
                               for name, column in columns})
        return result
