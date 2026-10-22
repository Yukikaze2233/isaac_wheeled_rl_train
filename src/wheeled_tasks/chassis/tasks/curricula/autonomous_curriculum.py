"""Versioned, deterministic capability progression; no simulator/PPO side effects.

The caller reduces *current-node, PPO-owned* samples after the recovery mask,
persists reservations before optimization, and checkpoints this state together
with learning state/RNG. Sampling plans apply at episode reset. Native observers,
not reward values, provide evaluation metrics. Nothing here starts/stops PPO.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "v6-autonomous-curriculum-v1"
V7_SCHEMA = "v7-autonomous-curriculum-v1"
STATE_SCHEMA = "v6-autonomous-curriculum-state-v1"


def measurement_only(definition):
    return (definition.get("schema") == V7_SCHEMA
            and definition.get("evaluation") == {"mode": "measurement_only"})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _count(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def load_definition(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[5] / "configs/v6_autonomous_curriculum_v1.json"
    definition = json.loads(Path(path).read_text())
    validate_definition(definition)
    return definition


def validate_definition(definition):
    """Reject ambiguous counters, impossible structural budgets and dangling cases."""
    if definition.get("schema") not in (SCHEMA, V7_SCHEMA):
        raise ValueError("Unsupported autonomous curriculum schema")
    measured = measurement_only(definition)
    if definition.get("schema") == V7_SCHEMA:
        if not measured:
            raise ValueError("V7 physical promotion awaits measured thresholds and a separately reviewed scoring schema")
        if definition.get("budget") != {"authority": "external_common20000", "total_paid_update_cap": 20000}:
            raise ValueError("V7 uses only the external common20000 budget; no separate branch/GPU/phase cap")
    if definition.get("identity") != dict(actor_dim=35, critic_dim=78, action_dim=6,
                                         history_length=1, policy_dt_s=.02, physics_dt_s=.005):
        raise ValueError("This curriculum requires the independent 35/78/6 contract")
    clock, evaluation = definition["clock"], definition["evaluation"]
    for key in ("reference_transitions_per_update", "minimum_current_node_rl_transitions_per_qualifying_update",
                "evaluation_interval_qualifying_updates"):
        _count(clock[key], key, minimum=1)
    if clock["minimum_current_node_rl_transitions_per_qualifying_update"] > clock["reference_transitions_per_update"]:
        raise ValueError("Qualifying sample threshold exceeds a full rollout")
    for key, required in (() if measured else (("seeds_per_round", 2), ("episodes_per_seed_per_case", 16),
                          ("pass_required_per_case", 31), ("promotion_consecutive_passes", 2),
                          ("regression_consecutive_failures", 2), ("recovery_consecutive_passes", 2))):
        if type(evaluation[key]) is not int or evaluation[key] != required:
            raise ValueError(f"v1 fixes {key}={required}")
    cases = definition["cases"]
    if not cases or not definition["foundation_case_ids"]:
        raise ValueError("Cases and foundation protection anchors are required")
    for cid, spec in cases.items():
        _identifier(cid, "case ID")
        if not math.isfinite(spec["minimum_observed_duration_s"]) or spec["minimum_observed_duration_s"] <= 0:
            raise ValueError("A case must have a positive complete physical observation window")
        if measured and spec["thresholds"] != {}:
            raise ValueError("Measurement-only V7 must not import unreviewed hard case thresholds")
        if not measured and not spec["thresholds"]:
            raise ValueError("Case thresholds cannot be empty")
        for metric, bounds in spec["thresholds"].items():
            _identifier(metric, "metric")
            if not bounds or set(bounds) - {"min", "max"}:
                raise ValueError("Only explicit min/max metric bounds are supported")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds.values()):
                raise ValueError("Metric bounds must be finite numbers")
            if bounds.get("min", -math.inf) > bounds.get("max", math.inf):
                raise ValueError("Inverted metric bounds")
    node_ids, phase_ids = set(), set()
    total_cap = 0
    for phase in definition["phases"]:
        pid = _identifier(phase["id"], "phase ID")
        if pid in phase_ids or not phase["nodes"]:
            raise ValueError("Phase IDs must be unique and contain nodes")
        phase_ids.add(pid)
        if measured:
            if phase.get("paid_update_cap") is not None:
                raise ValueError("Measurement-only phases cannot own spending caps")
        else:
            total_cap += _count(phase["paid_update_cap"], "phase budget", minimum=1)
        minimum_updates = 0
        for node in phase["nodes"]:
            nid = _identifier(node["id"], "node ID")
            if nid in node_ids or not node["required_cases"]:
                raise ValueError("Node IDs must be unique and contain required cases")
            node_ids.add(nid)
            if len(node["required_cases"]) != len(set(node["required_cases"])) or set(node["required_cases"]) - cases.keys():
                raise ValueError("Unknown or duplicate required case")
            exposure = node["minimum_exposure"]
            for key in ("qualifying_actor_updates", "valid_rl_transitions", "completed_rl_episodes"):
                _count(exposure[key], key, minimum=1)
            if not exposure["key_phase_counts"]:
                raise ValueError("Every node needs physical key-phase exposure")
            for key, value in exposure["key_phase_counts"].items():
                _identifier(key, "key phase")
                _count(value, key, minimum=1)
            fractions = node["recovery_lesson_fractions"]
            if fractions:
                _validate_fractions(fractions)
            interval = clock["evaluation_interval_qualifying_updates"]
            first = max(exposure["qualifying_actor_updates"], interval,
                        math.ceil(exposure["valid_rl_transitions"] / clock["reference_transitions_per_update"]))
            minimum_updates += math.ceil(first / interval) * interval + interval
        if not measured and minimum_updates > phase["paid_update_cap"]:
            raise ValueError(f"{pid}: structural minimum exceeds paid budget")
    if not measured and total_cap != definition["budget"]["total_paid_update_cap"]:
        raise ValueError("Phase budget sum must match the campaign ceiling")
    if set(definition["foundation_case_ids"]) - cases.keys():
        raise ValueError("Unknown foundation anchor")
    for key in ("regular", "regression", "recovery", "recovery_regression", "first_node"):
        fractions = definition["sampling"][key]
        if set(fractions) != {"foundation", "current", "previous"}:
            raise ValueError("Sampling categories must be explicit")
        _validate_fractions(fractions)
    _validate_fractions(definition["sampling"]["foundation_internal_kind_fractions"])
    if set(definition["sampling"]["recovery_lesson_node_ids"].values()) - node_ids:
        raise ValueError("Unknown recovery rehearsal node")
    if set(definition.get("prototype", {}).get("node_allowlist", [])) - node_ids:
        raise ValueError("Unknown prototype node")
    return definition


def _validate_fractions(fractions):
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in fractions.values()):
        raise ValueError("Sampling fractions must be finite and nonnegative")
    if not math.isclose(sum(fractions.values()), 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError("Sampling fractions must sum to one")


def episode_passes(case, row):
    """Missing/NaN metrics, truncation and incomplete trajectories never pass."""
    if not case.get("thresholds"):
        return False
    if row.get("complete") is not True or row.get("truncated") is not False:
        return False
    duration = row.get("observed_duration_s")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < case["minimum_observed_duration_s"]:
        return False
    for key, bounds in case["thresholds"].items():
        value = row.get("metrics", {}).get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            return False
        if value < bounds.get("min", -math.inf) or value > bounds.get("max", math.inf):
            return False
    return True


class AutonomousCurriculum:
    """Atomic in-memory receipts; caller supplies durable checkpoint transactions.

    reserve_update() is written to the caller's paid ledger before optimizer
    work. finish_update() credits masked exposure only after successful actor
    optimization; failed/critic reservations remain consumed. Repeating an ID
    is harmless only with identical contents. An outstanding reservation after
    a crash is never refunded or credited without a completion receipt.
    """

    def __init__(self, definition=None, *, training_seeds, paid_update_cap=None, node_allowlist=None,
                 allow_provisional=False, external_budget=False):
        self.definition = deepcopy(load_definition() if definition is None else validate_definition(definition))
        if type(external_budget) is not bool:
            raise ValueError("External budget mode must be explicit")
        if external_budget and paid_update_cap is not None:
            raise ValueError("External BudgetOwner is the only spending-cap authority")
        if measurement_only(self.definition) and not external_budget:
            raise ValueError("V7 measurement runtime requires the real external budget owner")
        if (not measurement_only(self.definition) and self.definition.get("approval", {}).get("status") != "user_approved"
                and not allow_provisional):
            raise ValueError("Curriculum definition is provisional, not user approved; explicit fixture opt-in required")
        self.allow_provisional = bool(allow_provisional)
        self.external_budget = external_budget
        self.definition_sha256 = _digest(self.definition)
        self.nodes = [(p["id"], n) for p in self.definition["phases"] for n in p["nodes"]]
        self.by_node = {node["id"]: (pid, node) for pid, node in self.nodes}
        self.phase_caps = {p["id"]: p.get("paid_update_cap") for p in self.definition["phases"]}
        seeds = sorted(set(_count(v, "training seed") for v in training_seeds))
        if not seeds:
            raise ValueError("Training seeds are required to reject evaluation contamination")
        cap = self.definition["budget"]["total_paid_update_cap"] if paid_update_cap is None else paid_update_cap
        _count(cap, "paid update cap", minimum=1)
        if cap > self.definition["budget"]["total_paid_update_cap"]:
            raise ValueError("Cannot extend the definition's consumption budget")
        allow = [node["id"] for _, node in self.nodes] if node_allowlist is None else list(node_allowlist)
        if not allow or len(allow) != len(set(allow)) or set(allow) - self.by_node.keys() or self.nodes[0][1]["id"] not in allow:
            raise ValueError("Node allowlist must contain the initial node and only known unique nodes")
        self._state = dict(schema=STATE_SCHEMA, definition_sha256=self.definition_sha256,
            training_seeds=seeds, paid_update_cap=cap, node_allowlist=allow,
            node_index=0, completed=False, promotion_blocked_by_scope=False,
            frozen=False, pass_streak=0, regression_streak=0, recovery_streak=0,
            passed_nodes=[], protected_cases=[], paid_updates=0,
            phase_paid_updates={pid: 0 for pid in self.phase_caps},
            exposure={node["id"]: dict(qualifying_actor_updates=0, valid_rl_transitions=0,
                       completed_rl_episodes=0, key_phase_counts={}) for _, node in self.nodes},
            last_eval_qualifying_update={node["id"]: 0 for _, node in self.nodes},
            reservations={}, evaluations={}, used_eval_seeds=[], used_checkpoints=[])
        if external_budget:
            self._state["external_budget"] = True

    @property
    def node(self):
        return deepcopy(self.nodes[self._state["node_index"]][1])

    @property
    def phase_id(self):
        return self.nodes[self._state["node_index"]][0]

    def budget_status(self):
        state = self._state
        if self.external_budget:
            return dict(authority="external_common20000", paid_updates=state["paid_updates"],
                paid_update_cap=None, phase_paid_updates=state["phase_paid_updates"][self.phase_id],
                phase_paid_update_cap=None, exhausted=None,
                local_counts_are_diagnostic_mirrors=True)
        return dict(paid_updates=state["paid_updates"], paid_update_cap=state["paid_update_cap"],
                    phase_paid_updates=state["phase_paid_updates"][self.phase_id],
                    phase_paid_update_cap=self.phase_caps[self.phase_id],
                    exhausted=(state["paid_updates"] >= state["paid_update_cap"] or
                               state["phase_paid_updates"][self.phase_id] >= self.phase_caps[self.phase_id]))

    def reserve_update(self, update_id):
        if self.external_budget:
            raise ValueError("Only BudgetOwner may reserve paid work in external budget mode")
        update_id = _identifier(update_id, "update ID")
        state = self._state
        if update_id in state["reservations"]:
            return deepcopy(state["reservations"][update_id])
        if self.budget_status()["exhausted"]:
            raise ValueError("Paid consumption ceiling reached; no automatic extension")
        reservation = dict(node_id=self.node["id"], phase_id=self.phase_id, completion=None)
        state["reservations"][update_id] = reservation
        state["paid_updates"] += 1
        state["phase_paid_updates"][self.phase_id] += 1
        return deepcopy(reservation)

    def observe_external_reservation(self, update_id, token, verified_row):
        """Mirror a reservation already authenticated by the runtime's owner.

        This method never allocates budget, enforces a phase cap or refunds work.
        Live owner verification is also required before completion and restore.
        """
        if not self.external_budget:
            raise ValueError("External reservation requires external budget mode")
        update_id = _identifier(update_id, "update ID")
        keys = {"ledger_id", "sequence", "attempt_id", "ordinal", "lease_id", "phase"}
        if (not isinstance(token, dict) or set(token) != keys or not isinstance(verified_row, dict)
                or any(verified_row.get(k) != token[k] for k in keys - {"ledger_id"})
                or token["phase"] not in ("actor", "critic")):
            raise ValueError("External reservation token/row mismatch")
        for key in ("sequence", "ordinal"):
            _count(token[key], key, minimum=1)
        for key in ("ledger_id", "attempt_id", "lease_id"):
            _identifier(token[key], key)
        existing = self._state["reservations"].get(update_id)
        if existing is not None:
            if existing.get("external_budget_token") != token:
                raise ValueError("External reservation ID was rebound")
            return deepcopy(existing)
        if any(r.get("external_budget_token") == token for r in self._state["reservations"].values()):
            raise ValueError("A real budget token cannot be mirrored under another update ID")
        status = verified_row.get("status")
        if (status not in ("reserved", "failed", "postcheck_failed")
                or verified_row.get("optimizer_completed") is not (status == "postcheck_failed")):
            raise ValueError("Observe a real pre-optimizer reservation or consumed failure")
        reservation = dict(node_id=self.node["id"], phase_id=self.phase_id, completion=None,
                           external_budget_token=deepcopy(token))
        self._state["reservations"][update_id] = reservation
        self._state["paid_updates"] += 1
        self._state["phase_paid_updates"][self.phase_id] += 1
        return deepcopy(reservation)

    def finish_update(self, update_id, *, actor_optimized, exposure=None):
        if type(actor_optimized) is not bool:
            raise ValueError("actor_optimized must be bool, true only after successful optimization")
        if update_id not in self._state["reservations"]:
            raise ValueError("Reserve and persist consumption before optimization")
        exposure = deepcopy(exposure or dict(valid_rl_transitions=0, completed_rl_episodes=0, key_phase_counts={}))
        if set(exposure) != {"valid_rl_transitions", "completed_rl_episodes", "key_phase_counts"}:
            raise ValueError("Exposure requires exactly masked transitions, completed RL episodes and physical key phases")
        valid = _count(exposure["valid_rl_transitions"], "valid RL transitions")
        episodes = _count(exposure["completed_rl_episodes"], "completed RL episodes")
        if valid > self.definition["clock"]["reference_transitions_per_update"] or episodes > valid:
            raise ValueError("Exposure exceeds a rollout or completed episodes lack RL-owned samples")
        reservation = self._state["reservations"][update_id]
        if self.external_budget and actor_optimized and reservation["external_budget_token"]["phase"] != "actor":
            raise ValueError("Critic consumption cannot acquire actor exposure")
        expected_keys = self.by_node[reservation["node_id"]][1]["minimum_exposure"]["key_phase_counts"]
        if set(exposure["key_phase_counts"]) - expected_keys.keys():
            raise ValueError("Unknown key phase for the reserved node")
        for key, value in exposure["key_phase_counts"].items():
            _count(value, key)
            if value > valid:
                raise ValueError("Key-phase events must have RL-owned evidence")
        payload = dict(actor_optimized=actor_optimized, exposure=exposure)
        digest = _digest(payload)
        if reservation["completion"] is not None:
            if reservation["completion"]["sha256"] != digest:
                raise ValueError("Conflicting completion for an already consumed update")
            return deepcopy(reservation["completion"])
        if not actor_optimized and (valid or episodes or any(exposure["key_phase_counts"].values())):
            raise ValueError("Failed/critic-only work cannot claim actor exposure")
        if actor_optimized:
            counts = self._state["exposure"][reservation["node_id"]]
            counts["valid_rl_transitions"] += valid
            counts["completed_rl_episodes"] += episodes
            qualifies = valid >= self.definition["clock"]["minimum_current_node_rl_transitions_per_qualifying_update"]
            counts["qualifying_actor_updates"] += int(qualifies)
            for key, value in exposure["key_phase_counts"].items():
                counts["key_phase_counts"][key] = counts["key_phase_counts"].get(key, 0) + value
        receipt = dict(sha256=digest, actor_optimized=actor_optimized, exposure=exposure)
        reservation["completion"] = receipt
        return deepcopy(receipt)

    def record_update(self, update_id, *, actor_optimized, exposure=None):
        """Convenience for a caller that already owns a durable paid reservation."""
        self.reserve_update(update_id)
        return self.finish_update(update_id, actor_optimized=actor_optimized, exposure=exposure)

    def exposure_status(self, node_id=None):
        node_id = self.node["id"] if node_id is None else node_id
        minimum = self.by_node[node_id][1]["minimum_exposure"]
        counts = self._state["exposure"][node_id]
        deficits = {key: max(0, minimum[key] - counts[key]) for key in
                    ("qualifying_actor_updates", "valid_rl_transitions", "completed_rl_episodes")}
        deficits["key_phase_counts"] = {key: max(0, value - counts["key_phase_counts"].get(key, 0))
                                        for key, value in minimum["key_phase_counts"].items()}
        met = not any(deficits[k] for k in deficits if k != "key_phase_counts") and not any(deficits["key_phase_counts"].values())
        return dict(counts=deepcopy(counts), minimum=deepcopy(minimum), deficits=deficits,
                    met=met and not measurement_only(self.definition),
                    **({"measurement_counts_only": True, "physical_minimum_review": "pending"}
                       if measurement_only(self.definition) else {}),
                    full_batch_equivalent_updates=counts["valid_rl_transitions"] / self.definition["clock"]["reference_transitions_per_update"])

    def evaluation_due(self):
        if measurement_only(self.definition):
            return False
        if self._state["completed"] or self._state["promotion_blocked_by_scope"]:
            return False
        nid = self.node["id"]
        if any(r["request"]["node_id"] == nid and r["receipt"] is None for r in self._state["evaluations"].values()):
            return False
        elapsed = self._state["exposure"][nid]["qualifying_actor_updates"] - self._state["last_eval_qualifying_update"][nid]
        return elapsed >= self.definition["clock"]["evaluation_interval_qualifying_updates"]

    def required_case_ids(self):
        return sorted(set(self.node["required_cases"]) | set(self._state["protected_cases"]))

    def request_evaluation(self, eval_id, *, checkpoint_id, seeds):
        """Freeze scope/exposure for a side evaluation; never advances clocks."""
        if measurement_only(self.definition):
            raise ValueError("V7 measurements do not grant physical promotion before threshold review")
        eval_id, checkpoint_id = _identifier(eval_id, "evaluation ID"), _identifier(checkpoint_id, "checkpoint ID")
        seeds = sorted(_count(seed, "evaluation seed") for seed in seeds)
        identity = dict(checkpoint_id=checkpoint_id, seeds=seeds)
        state = self._state
        if eval_id in state["evaluations"]:
            existing = state["evaluations"][eval_id]["request"]
            if {key: existing[key] for key in identity} != identity:
                raise ValueError("Evaluation ID already binds another checkpoint/seed set")
            return deepcopy(existing)
        if not self.evaluation_due():
            raise ValueError("Evaluation not due, another evaluation pending, or allowed scope completed")
        if len(seeds) != 2 or len(set(seeds)) != 2 or set(seeds) & set(state["training_seeds"] + state["used_eval_seeds"]):
            raise ValueError("Each round needs two fresh, independent non-training seeds")
        if checkpoint_id in state["used_checkpoints"]:
            raise ValueError("Consecutive passes need distinct checkpoint snapshots")
        request = dict(eval_id=eval_id, definition_sha256=self.definition_sha256,
            node_id=self.node["id"], case_ids=self.required_case_ids(),
            current_cases=list(self.node["required_cases"]), protected_cases=list(state["protected_cases"]),
            exposure=self.exposure_status(), **identity)
        state["evaluations"][eval_id] = dict(request=request, receipt=None)
        state["used_eval_seeds"].extend(seeds)
        state["used_checkpoints"].append(checkpoint_id)
        state["last_eval_qualifying_update"][self.node["id"]] = request["exposure"]["counts"]["qualifying_actor_updates"]
        return deepcopy(request)

    def record_evaluation(self, eval_id, results):
        """results: {case_id: [{seed, episode_id, complete, truncated,
        observed_duration_s, metrics}, ...]}. At most 16 unique rows per seed;
        omitted rows/cases are failures, never removed from the denominator.
        """
        if measurement_only(self.definition):
            raise ValueError("V7 measurement-only definition has no approved pass/fail scoring")
        entry = self._state["evaluations"].get(eval_id)
        if entry is None:
            raise ValueError("Unknown evaluation request")
        digest = _digest(results)
        if entry["receipt"] is not None:
            if entry["receipt"]["sha256"] != digest:
                raise ValueError("Conflicting result for an already processed evaluation")
            return deepcopy(entry["receipt"])
        request = entry["request"]
        if set(results) - set(request["case_ids"]):
            raise ValueError("Evaluation cannot import future/unrequested case credit")
        scores = {}
        for cid in request["case_ids"]:
            seen, per_seed, passed = set(), {seed: 0 for seed in request["seeds"]}, 0
            for row in results.get(cid, []):
                seed = _count(row["seed"], "evaluation seed")
                episode_id = _identifier(row["episode_id"], "episode ID")
                key = (seed, episode_id)
                if seed not in per_seed or key in seen:
                    raise ValueError("Foreign seed or duplicate episode in evaluation case")
                seen.add(key)
                per_seed[seed] += 1
                if per_seed[seed] > 16:
                    raise ValueError("More than 16 episodes for one case/seed")
                passed += int(episode_passes(self.definition["cases"][cid], row))
            scores[cid] = dict(passed=passed, total=32, observed=len(seen), passed_gate=passed >= 31)
        state = self._state
        current_pass = all(scores[cid]["passed_gate"] for cid in request["current_cases"])
        protection_pass = all(scores[cid]["passed_gate"] for cid in request["protected_cases"])
        action = "continue"
        if request["node_id"] != self.node["id"] or state["completed"]:
            action = "stale_no_progression"
        else:
            if not protection_pass:
                state["pass_streak"] = state["recovery_streak"] = 0
                state["regression_streak"] += 1
                if state["regression_streak"] >= 2:
                    state["frozen"] = True
                    action = "freeze_difficulty_continue_ppo"
            elif state["frozen"]:
                state["regression_streak"] = 0
                state["recovery_streak"] += 1
                if state["recovery_streak"] >= 2:
                    state["frozen"] = False
                    state["recovery_streak"] = 0
                    action = "unfreeze_continue_ppo"
                # Recovery rounds do not also count as promotion rounds.
                state["pass_streak"] = 0
            else:
                state["regression_streak"] = 0
                eligible = current_pass and request["exposure"]["met"]
                state["pass_streak"] = state["pass_streak"] + 1 if eligible else 0
                if state["pass_streak"] >= 2:
                    action = self._promote()
        receipt = dict(sha256=digest, node_id=request["node_id"], scores=scores,
            current_pass=current_pass, protection_pass=protection_pass,
            minimum_exposure_met=request["exposure"]["met"], action=action,
            continue_ppo=True, budget_status=self.budget_status())
        entry["receipt"] = receipt
        return deepcopy(receipt)

    def _promote(self):
        if measurement_only(self.definition):
            raise ValueError("Measurements alone cannot promote V7 difficulty")
        state, node = self._state, self.node
        if node["id"] not in state["passed_nodes"]:
            state["passed_nodes"].append(node["id"])
        state["protected_cases"] = sorted(set(state["protected_cases"]) | set(node["required_cases"]))
        state["pass_streak"] = state["regression_streak"] = state["recovery_streak"] = 0
        next_index = state["node_index"] + 1
        if next_index == len(self.nodes):
            state["completed"] = True
            return "curriculum_complete"
        if self.nodes[next_index][1]["id"] not in state["node_allowlist"]:
            state["promotion_blocked_by_scope"] = True
            return "allowed_scope_complete_no_higher_skill_credit"
        state["node_index"] = next_index
        return "advance_at_next_reset_continue_ppo"

    def sampling_plan(self):
        state, node = self._state, self.node
        foundation = [nid for nid in state["passed_nodes"] if self.by_node[nid][0] == self.nodes[0][0]]
        previous = [nid for nid in state["passed_nodes"] if self.by_node[nid][0] not in (self.nodes[0][0], self.phase_id)]
        recovery = bool(node["recovery_lesson_fractions"])
        key = ("recovery_regression" if recovery else "regression") if state["frozen"] else ("recovery" if recovery else "regular")
        fractions = deepcopy(self.definition["sampling"][key])
        if not foundation:
            fractions = deepcopy(self.definition["sampling"]["first_node"])
        elif not previous:
            fractions["foundation"] += fractions["previous"]
            fractions["previous"] = 0.
        lessons = node["recovery_lesson_fractions"]
        foundation_cases = sorted({cid for nid in foundation for cid in self.by_node[nid][1]["required_cases"]})
        kind_weights = self.definition["sampling"]["foundation_internal_kind_fractions"]
        learned_kinds = {self.definition["cases"][cid]["kind"] for cid in foundation_cases}
        kind_total = sum(kind_weights[kind] for kind in learned_kinds)
        case_weights = {cid: kind_weights[self.definition["cases"][cid]["kind"]] / kind_total /
                        sum(self.definition["cases"][other]["kind"] == self.definition["cases"][cid]["kind"]
                            for other in foundation_cases) for cid in foundation_cases}
        previous_phases = sorted({self.by_node[nid][0] for nid in previous})
        previous_weights = {nid: 1. / len(previous_phases) /
                            sum(self.by_node[other][0] == self.by_node[nid][0] for other in previous)
                            for nid in previous}
        return dict(node_id=node["id"], difficulty=deepcopy(node["difficulty"]),
                    fractions=fractions, foundation_nodes=foundation, previous_nodes=previous,
                    foundation_case_fractions=case_weights, previous_node_fractions=previous_weights,
                    current_case_fractions={cid: 1. / len(node["required_cases"]) for cid in node["required_cases"]},
                    recovery_lesson_fractions=deepcopy(lessons),
                    recovery_lesson_node_ids={key: self.definition["sampling"]["recovery_lesson_node_ids"][key]
                                              for key in lessons},
                    recovery_total_fractions={k: fractions["current"] * v for k, v in lessons.items()},
                    latch_until_episode_reset=True, frozen=state["frozen"])

    def metrics(self):
        exposure, budget = self.exposure_status(), self.budget_status()
        return {"AutonomousCurriculum/node_index": self._state["node_index"],
                "AutonomousCurriculum/qualifying_actor_updates": exposure["counts"]["qualifying_actor_updates"],
                "AutonomousCurriculum/valid_rl_transitions": exposure["counts"]["valid_rl_transitions"],
                "AutonomousCurriculum/full_batch_equivalent_updates": exposure["full_batch_equivalent_updates"],
                "AutonomousCurriculum/completed_rl_episodes": exposure["counts"]["completed_rl_episodes"],
                "AutonomousCurriculum/minimum_exposure_met": float(exposure["met"]),
                "AutonomousCurriculum/frozen": float(self._state["frozen"]),
                "AutonomousCurriculum/paid_updates": budget["paid_updates"]}

    def state_dict(self):
        return deepcopy(self._state)

    def load_state_dict(self, state):
        """Strict identity and ledger checks; never silently reset bad progress."""
        candidate = deepcopy(state)
        for key in ("schema", "definition_sha256", "training_seeds", "paid_update_cap", "node_allowlist"):
            if candidate.get(key) != self._state[key]:
                raise ValueError(f"Curriculum resume identity mismatch: {key}")
        if candidate.get("external_budget", False) != self.external_budget:
            raise ValueError("Curriculum resume budget authority mismatch")
        if measurement_only(self.definition) and (candidate["node_index"] != 0 or
                any(candidate[k] for k in ("completed", "promotion_blocked_by_scope", "frozen", "pass_streak",
                    "regression_streak", "recovery_streak", "passed_nodes", "protected_cases", "evaluations",
                    "used_eval_seeds", "used_checkpoints"))):
            raise ValueError("Measurement-only checkpoint cannot contain capability credit")
        _digest(candidate)  # Reject NaN/non-JSON states before mutation.
        index = _count(candidate["node_index"], "node index")
        if index >= len(self.nodes):
            raise ValueError("Unknown restored node")
        if candidate["paid_updates"] != len(candidate["reservations"]):
            raise ValueError("Paid ledger count disagrees with durable reservations")
        rebuilt = AutonomousCurriculum(self.definition, training_seeds=candidate["training_seeds"],
            paid_update_cap=None if self.external_budget else candidate["paid_update_cap"], node_allowlist=candidate["node_allowlist"],
            allow_provisional=self.allow_provisional, external_budget=self.external_budget)
        phase_counts = {pid: 0 for pid in self.phase_caps}
        exposure = deepcopy(rebuilt._state["exposure"])
        external_tokens = set()
        for reservation in candidate["reservations"].values():
            if self.external_budget:
                token = reservation.get("external_budget_token")
                if (not isinstance(token, dict) or set(token) !=
                        {"ledger_id", "sequence", "attempt_id", "ordinal", "lease_id", "phase"}
                        or token["phase"] not in ("actor", "critic")):
                    raise ValueError("Restored mirror lacks a complete external budget token")
                for key in ("sequence", "ordinal"):
                    _count(token[key], key, minimum=1)
                for key in ("ledger_id", "attempt_id", "lease_id"):
                    _identifier(token[key], key)
                token_digest = _digest(token)
                if token_digest in external_tokens:
                    raise ValueError("Restored mirror repeats a real budget token")
                external_tokens.add(token_digest)
            nid = reservation["node_id"]
            if nid not in self.by_node or reservation["phase_id"] != self.by_node[nid][0]:
                raise ValueError("Reservation node/phase mismatch")
            phase_counts[reservation["phase_id"]] += 1
            completion = reservation["completion"]
            if completion is None:
                continue
            payload = {key: completion[key] for key in ("actor_optimized", "exposure")}
            if completion["sha256"] != _digest(payload):
                raise ValueError("Corrupt update completion digest")
            # Reuse completion validation without invoking progression or RNG.
            rebuilt._state["reservations"] = {"validation": {**reservation, "completion": None}}
            rebuilt._state["exposure"] = exposure
            rebuilt.finish_update("validation", **payload)
        if candidate["phase_paid_updates"] != phase_counts or candidate["exposure"] != exposure:
            raise ValueError("Restored exposure does not reconcile to completed paid work")
        if set(candidate["protected_cases"]) - self.definition["cases"].keys():
            raise ValueError("Unknown restored protection case")
        passed = candidate["passed_nodes"]
        if passed != [node["id"] for _, node in self.nodes[:len(passed)]]:
            raise ValueError("Restored passed nodes must be an ordered curriculum prefix")
        expected_protected = sorted({cid for nid in passed for cid in self.by_node[nid][1]["required_cases"]})
        if candidate["protected_cases"] != expected_protected:
            raise ValueError("Restored protection does not match passed capability nodes")
        for key in ("completed", "promotion_blocked_by_scope", "frozen"):
            if type(candidate[key]) is not bool:
                raise ValueError(f"Invalid restored boolean: {key}")
        expected_index = len(passed) - int(candidate["completed"] or candidate["promotion_blocked_by_scope"])
        if index != expected_index or (candidate["completed"] and len(passed) != len(self.nodes)):
            raise ValueError("Restored current node disagrees with capability progress")
        for key in ("pass_streak", "regression_streak", "recovery_streak"):
            _count(candidate[key], key)
        requested_seeds, requested_checkpoints = [], []
        last_eval = {nid: 0 for nid in self.by_node}
        for eval_id, evaluation in candidate["evaluations"].items():
            request = evaluation["request"]
            if (request["eval_id"] != eval_id or request["definition_sha256"] != self.definition_sha256
                    or request["node_id"] not in self.by_node):
                raise ValueError("Restored evaluation identity mismatch")
            if request["current_cases"] != self.by_node[request["node_id"]][1]["required_cases"]:
                raise ValueError("Restored evaluation current-case scope mismatch")
            if request["case_ids"] != sorted(set(request["current_cases"]) | set(request["protected_cases"])):
                raise ValueError("Restored evaluation protection scope mismatch")
            requested_seeds.extend(request["seeds"])
            requested_checkpoints.append(request["checkpoint_id"])
            last_eval[request["node_id"]] = max(last_eval[request["node_id"]], request["exposure"]["counts"]["qualifying_actor_updates"])
        if (candidate["used_eval_seeds"] != requested_seeds or candidate["used_checkpoints"] != requested_checkpoints
                or candidate["last_eval_qualifying_update"] != last_eval
                or len(requested_checkpoints) != len(set(requested_checkpoints))):
            raise ValueError("Restored evaluation ledger does not reconcile")
        if len(candidate["used_eval_seeds"]) != len(set(candidate["used_eval_seeds"])) or set(candidate["used_eval_seeds"]) & set(candidate["training_seeds"]):
            raise ValueError("Restored evaluation seeds are not independent")
        self._state = candidate


def theoretical_minimum_consumption(definition=None):
    """Optimistic floor only: every update qualifies, all physical gates pass.

    Script masking/rehearsal, missing phase exposure or failed gates can only
    increase consumption. Ceiling exhaustion never grants capability credit.
    """
    definition = load_definition() if definition is None else validate_definition(definition)
    if measurement_only(definition):
        raise ValueError("No physical completion budget can be inferred before measured gate review")
    clock = definition["clock"]
    interval, batch = clock["evaluation_interval_qualifying_updates"], clock["reference_transitions_per_update"]
    rows = []
    for phase in definition["phases"]:
        nodes = []
        for node in phase["nodes"]:
            minimum = node["minimum_exposure"]
            first = max(interval, minimum["qualifying_actor_updates"], math.ceil(minimum["valid_rl_transitions"] / batch))
            first = math.ceil(first / interval) * interval
            nodes.append(dict(node_id=node["id"], minimum_paid_updates=first + interval,
                              required_minimum_exposure=deepcopy(minimum)))
        floor = sum(node["minimum_paid_updates"] for node in nodes)
        rows.append(dict(phase_id=phase["id"], minimum_paid_updates=floor,
                         paid_update_cap=phase["paid_update_cap"], nodes=nodes))
    return rows
