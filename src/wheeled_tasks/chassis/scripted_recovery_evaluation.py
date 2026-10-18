"""Explicit flat recovery measurements on the existing source evaluator.

No training allocation, physical intervention, new gate or PPO credit is defined
here. Handoff windows begin at the actual continuous script's producer event.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

SCHEMA = "scut-scripted-recovery-evaluation-v1"
ROLE = "source_scripted_recovery_evaluation"
EVENTS = ("script_ready", "rl_entered", "legacy_physical_hold_qualified", "legacy_followup_complete")
CHAIN_BY_PROFILE = {
    "upright": "upright_script_handoff", "crouched": "crouched_script_handoff",
    "upright_release": "upright_zero_torque", "crouched_release": "crouched_zero_torque",
    "pitch_forward_45": "tilted_script_handoff", "pitch_backward_45": "tilted_script_handoff",
    **{name: "fallen_full_chain" for name in (
        "left_side_90", "right_side_90", "front_down_90", "back_down_90",
        "front_down_left_forward_right_back_90", "front_down_left_back_right_forward_90",
        "back_down_left_forward_right_back_90", "back_down_left_back_right_forward_90",
        "prone_belly_legs_above", "prone_back_legs_above")},
}


def selected(protocol):
    return protocol.get("schema") == SCHEMA


def validate_protocol(protocol):
    from .flat_policy_evaluation import _validate_flat_protocol
    if (protocol.get("role") != ROLE or protocol.get("handoff_source") != "continuous_native_script"
            or protocol.get("mode") != "fixed_measurement"):
        raise ValueError("Recovery measurement requires its explicit role and continuous native script")
    plain = deepcopy(dict(protocol))
    plain.pop("role", None)
    plain.pop("handoff_source", None)
    plain["schema"] = "flat-policy-evaluation-v1"
    cases = plain.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Recovery protocol requires explicit cases")
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "base_request", "profile", "chain_type"}:
            raise ValueError("Recovery case requires case_id, base_request, profile and chain_type")
        profile, chain = case.pop("profile"), case.pop("chain_type")
        if profile is not None and not isinstance(profile, str):
            raise ValueError("Recovery profile must be an explicit existing profile name or ordinary null")
        expected = "ordinary_reference" if profile is None else CHAIN_BY_PROFILE.get(profile)
        if expected is None or chain != expected:
            raise ValueError("Recovery chain type must describe the existing flat profile")
    _validate_flat_protocol(plain)
    if not any(c["profile"] is None for c in protocol["cases"]) or not any(c["profile"] is not None for c in protocol["cases"]):
        raise ValueError("Recovery evaluation requires explicitly allocated recovery and ordinary reference lanes")
    return deepcopy(dict(protocol))


def scene_groups(protocol, lanes):
    """Caller quotas become evaluation-only scene selection, never training mix."""
    checked = validate_protocol(protocol)
    counts = Counter(lanes)
    if set(counts) != {c["case_id"] for c in checked["cases"]} or not 1 <= len(lanes) <= 4096:
        raise ValueError("Recovery lanes must explicitly cover every case within 4096 environments")
    groups = []
    for case in checked["cases"]:
        row = dict(name=case["case_id"], terrain=["flat"], fraction=counts[case["case_id"]] / len(lanes))
        if case["profile"] is not None:
            row["recovery_pose"] = case["profile"]
        groups.append(row)
    # Physical clone indices start in group order. Native view permutations are
    # read back later and applied to both commands and the measurement ledger.
    if list(lanes) != [g["name"] for g in groups for _ in range(counts[g["name"]])]:
        raise ValueError("Recovery lane list must follow explicit protocol case order")
    from .task import choose_scene_groups
    if [name for name, _ in choose_scene_groups(groups, len(lanes))] != list(lanes):
        raise ValueError("Physical scene allocation differs from declared recovery lane counts")
    return groups


def validate_binding(protocol, config, lanes, root):
    from .flat_episode_selection import validate_scut_flat_recovery
    checked = validate_protocol(protocol)
    root = Path(root)
    if validate_scut_flat_recovery(config, root) is None:
        raise ValueError("Recovery evaluation requires an authenticated scripted recovery checkpoint contract")
    result = deepcopy(config)
    result["scene_groups"] = scene_groups(checked, lanes)
    if "formal_small_rough_mix" in config:
        from .small_rough_evaluation import retained_flat_configuration
        result = retained_flat_configuration(config, result["scene_groups"], scripted=True)
    validate_scut_flat_recovery(result, root)  # Includes profile/lookup/model/manifest hashes and flat-only geometry.
    table = json.loads((root / config["recovery_training"]["profiles_file"]).read_bytes())
    for case in checked["cases"]:
        if case["profile"] is not None:
            profile = table["profiles"][case["profile"]]
            if case["chain_type"].endswith("zero_torque") and profile.get("initial_release_s", 0.) <= 0:
                raise ValueError("Zero-torque case requires the bound profile's real release interval")
    return result


def actual_lane_cases(setup, env):
    order = getattr(env, "clone_indices", None)
    if not isinstance(order, list) or any(type(i) is not int for i in order) or sorted(order) != list(range(env.num_envs)):
        raise ValueError("Recovery evaluator requires the actual native clone permutation")
    lanes = tuple(setup.lane_case_ids[i] for i in order)
    recovery = getattr(env, "recovery", None)
    cases = {c["case_id"]: c for c in setup.protocol["cases"]}
    expected = [cases[name]["profile"] for name in lanes]
    if (recovery is None or not recovery.source_flat_handoff
            or recovery.pose_curriculum is not None or list(env.scene_groups) != list(lanes)
            or list(recovery.profile_names) != expected
            or recovery.enabled.detach().cpu().tolist() != [p is not None for p in expected]):
        raise ValueError("Actual recovery lane/profile ownership differs from the evaluation protocol")
    return lanes


class RecoveryMeasurements:
    """Observe paid pre-reset events; never turn a milestone into env.done."""

    ACTION_HISTORY_PAYMENTS = ("action_rate", "action_smoothness_leg", "action_smoothness_wheel")

    def __init__(self, protocol, lane_cases, source_identity):
        from .recovery_evidence import _digest
        self.protocol = validate_protocol(protocol)
        self.lanes = tuple(lane_cases)
        self.cases = {c["case_id"]: c for c in protocol["cases"]}
        self.source_identity = deepcopy(source_identity)
        self.source_sha = _digest(source_identity)
        self.profile_ids = {name: i for i, name in enumerate(source_identity["profile_names"])}
        self.active, self.rows = {}, []
        self.last_interval_id = 0

    @staticmethod
    def terminal_measurement(row, source_identity):
        """Describe the actual takeover without imposing legacy blend gates."""
        settings = source_identity.get("effective_recovery_settings", {})
        direct = settings.get("direct_takeover", {})
        direct_version = direct.get("version")
        if direct_version:
            if (source_identity.get("takeover_version") != direct_version
                    or settings.get("blend_seconds") != 0.):
                raise ValueError("Direct recovery measurement has inconsistent producer identity")
            mode = "direct"
        else:
            mode = "blended" if settings.get("blend_seconds", 0.) > 0. else "unspecified"
        terminal = row["last_diagnostics"]
        entered = row["ever_completed"]["rl_entered"]
        return {
            "takeover_mode": mode,
            "rl_entered_observed": entered,
            "handover_count": terminal["handover_count"],
            "single_handover_observed": bool(entered and terminal["handover_count"] == 1),
            "blend_observed": bool(terminal["ever_blended"]),
            "legacy_blend_and_hold_classifier_applicable": mode == "blended",
            "endpoint_continuous_followup_s": terminal["followup_s"],
            "endpoint_producer_followup_complete": bool(terminal["true_success"]),
            "legacy_pre_release_hold_peak_s": terminal["rl_stable_s"],
            "terminal_after_handoff": bool(row["last_terminated"] and entered),
            "qualification": False,
        }

    def prepare(self, extras, active_lanes, *, policy_interval_id):
        from .flat_evaluation_runtime import PreResetLedgerAdapter
        from .evaluation import RequestProtocolMetrics
        payload, data = extras.get("recovery_transition", {}), extras["diagnostics"]
        if (payload.get("schema") != "v6-recovery-physical-transition-v1"
                or payload.get("source_identity_sha256") != self.source_sha
                or set(payload.get("events", {})) != set(EVENTS)
                or set(payload.get("occurrence_ids", {})) != set(EVENTS)):
            raise ValueError("Recovery evaluation requires the authenticated physical event producer")
        import torch
        interval_id = payload.get("policy_interval_id")
        if (type(interval_id) is not int or interval_id != self.last_interval_id + 1
                or interval_id != policy_interval_id):
            raise ValueError("Recovery event interval differs from the next paid source interval")
        returned = extras["returned_episode_state"]
        for source, target, field in (
                (payload, data, "done"), (payload, data, "terminated"),
                (payload, data, "policy_learning_mask"),
                (payload, extras, "policy_learning_mask")):
            if not torch.equal(source[field], target[field]):
                raise ValueError("Recovery transition differs from paid source " + field)
        for source_name, target_name in (("returned_episode_ids", "episode_id"), ("reset_mask", "reset_mask")):
            if not torch.equal(payload[source_name], returned[target_name]):
                raise ValueError("Recovery returned episode/reset identity differs")
        if not torch.equal(payload["timeouts"], data["timeout"]):
            raise ValueError("Recovery timeout differs from paid source timeout")
        n = len(self.lanes)
        def values(value, name, kind="float", tail=()):
            return PreResetLedgerAdapter._values(value, n, tail, name, kind=kind)
        ids = values(payload["episode_ids"], "recovery episode", "int")
        profiles = values(payload["episode_profile_ids"], "profile", "int")
        enabled = values(payload["enabled"], "enabled", "bool")
        ticks = values(data["episode_tick"], "episode_tick", "int")
        flat_ids = values(data["episode_id"], "flat episode", "int")
        masks = values(payload["policy_learning_mask"], "mask", "bool")
        done = values(payload["done"], "done", "bool")
        terminated = values(payload["terminated"], "terminated", "bool")
        events = {k: values(v, k, "bool") for k, v in payload["events"].items()}
        occurrences = {k: values(v, k, "int") for k, v in payload["occurrence_ids"].items()}
        diagnostics = {}
        for key in RequestProtocolMetrics.RECOVERY_COUNTERS:
            tensor = data["recovery_" + key]
            kind = "bool" if tensor.dtype == torch.bool else "int" if tensor.dtype == torch.int64 else "float"
            diagnostics[key] = values(tensor, key, kind)
        held = values(data["recovery_policy_motion_hold"], "recovery interval hold", "bool")
        override = values(data["recovery_policy_command_override_active"], "recovery command latch", "bool")
        source_override = values(data["source_flat_episode"]["paid_override_active"], "source command latch", "bool")
        paid_override = values(data["command_override_active"], "paid command latch", "bool")
        phase = values(payload["phase"], "phase", "int")
        returned_ids = values(payload["returned_episode_ids"], "returned episode", "int")
        reset = values(payload["reset_mask"], "reset", "bool")
        timeouts = values(payload["timeouts"], "timeouts", "bool")
        valid = values(data["measurement_valid_mask"], "valid", "bool")
        command = values(data["effective_command"], "command", tail=(3,))
        height = values(data["height"], "height")
        velocity = values(data["root_COM_velocity_body"], "velocity", tail=(3,))
        omega = values(data["angular_velocity_body"], "omega", tail=(3,))
        actions = {name: values(data[name], name, tail=(6,)) for name in (
            "raw_actor_actions", "raw_previous_actor_actions", "raw_before_previous_actor_actions")}
        action_payments = {name: values(data["payments"][name], name)
                           for name in self.ACTION_HISTORY_PAYMENTS}
        paid_reward = values(data["reward"], "reward")
        phase_vectors = {name: values(data[name], name, tail=(width,)) for name, width in (
            ("gravity", 3), ("motor_position", 6), ("motor_velocity", 6), ("motor_effort", 6),
            ("recovery_continuous_q", 6), ("recovery_leg_target_error_rad", 4),
            ("recovery_wheel_velocity_target", 2))}
        phase_scalars = {}
        for name in ("recovery_observed_height", "recovery_observed_support",
                     "recovery_reroute_count", "recovery_direct_capture_ready",
                     "recovery_sensor_time_valid", "recovery_valid_sensor_history_s"):
            if name in data:
                tensor = data[name]
                kind = "bool" if tensor.dtype == torch.bool else "int" if tensor.dtype == torch.int64 else "float"
                phase_scalars[name] = values(tensor, name, kind)
        pending = []
        for lane in active_lanes:
            case = self.cases[self.lanes[lane]]
            if paid_override[lane] != (source_override[lane] or override[lane]) or (override[lane] and not enabled[lane]):
                raise ValueError("Paid command coverage differs from its source/recovery interval latches")
            if enabled[lane] != (case["profile"] is not None):
                raise ValueError("Recovery enabled mask differs from explicit lane profile")
            if (done[lane] != (terminated[lane] or timeouts[lane]) or reset[lane] != done[lane]
                    or returned_ids[lane] != ids[lane] + int(reset[lane])
                    or phase[lane] != diagnostics["phase"][lane]):
                raise ValueError("Recovery done/reset/phase does not describe the paid interval")
            for name in EVENTS:
                if ((events[name][lane] and (not enabled[lane] or occurrences[name][lane] < 0))
                        or (not events[name][lane] and occurrences[name][lane] != -1)):
                    raise ValueError("Recovery milestone occurrence does not match its physical pulse")
            if not enabled[lane]:
                continue
            if ids[lane] != flat_ids[lane] or profiles[lane] != self.profile_ids[case["profile"]]:
                raise ValueError("Recovery paid episode/profile identity differs from the source ledger")
            previous = self.active.get(lane)
            if previous is None:
                if ticks[lane] != 1:
                    raise ValueError("Recovery measurements must begin at the physical episode start")
                previous = dict(lane=lane, case_id=self.lanes[lane], profile=case["profile"], chain_type=case["chain_type"],
                    episode_id=ids[lane], policy_ticks=0, ever_completed={e: False for e in EVENTS},
                    first_event_reference_tick={e: None for e in EVENTS}, eligible_rl_frames=0, masked_frames=0,
                    previous_actor_owned_intervals=[], first_actor_action_history=[],
                    controller_phase_history=[],
                    followup_observation=dict(observed_frames=0, predicate_frames=0,
                        longest_producer_followup_s=0., predicate_losses=0,
                        command_changes=0, losses_with_same_interval_command_change=0,
                        previous_command=None, previous_predicate=False),
                    post_handoff_windows={key: dict(frames=0, valid_frames=0, height_abs_m=0., vx_abs_m_s=0., yaw_abs_rad_s=0.)
                        for key in ("rl_hold", "rl_followup")}, followup_predicate_lost_after_completion=False)
            # Completed phase records are immutable. Copy only the list and
            # current segment, avoiding a growing history copy on every step.
            phase_history = previous["controller_phase_history"]
            row = deepcopy({key: value for key, value in previous.items() if key != "controller_phase_history"})
            row["controller_phase_history"] = list(phase_history)
            if ids[lane] != row["episode_id"] or ticks[lane] != row["policy_ticks"] + 1:
                raise ValueError("Recovery measurement episode/tick is discontinuous")
            terminal = {k: v[lane] for k, v in diagnostics.items()}
            # Validate real terminal fields using the existing evaluator's schema,
            # but do not promote its historical thresholds to V7 acceptance.
            from .evaluation import _recovery_row_checks
            if not _recovery_row_checks(terminal)["evidence"]:
                raise ValueError("Recovery diagnostics are incomplete or nonfinite")
            phase_key = [phase[lane], terminal["route"], terminal["pure_rl"],
                         terminal["motion_released"], masks[lane]]
            witness = dict(policy_interval_id=interval_id, episode_policy_tick=ticks[lane],
                controller=terminal, done=done[lane], terminated=terminated[lane],
                measurement_valid=valid[lane], policy_learning_mask=masks[lane],
                effective_command=command[lane], height=height[lane],
                root_COM_velocity_body=velocity[lane], angular_velocity_body=omega[lane],
                **{name: readings[lane] for name, readings in {**phase_vectors, **phase_scalars}.items()})
            error = max(abs(v) for v in phase_vectors["recovery_leg_target_error_rad"][lane])
            if not phase_history or phase_history[-1]["key"] != phase_key:
                segment = dict(key=phase_key, observed_policy_frames=1, first=witness, last=witness,
                    max_leg_target_error_rad=error)
                row["controller_phase_history"].append(segment)
            else:
                segment = dict(phase_history[-1], last=witness,
                    observed_policy_frames=phase_history[-1]["observed_policy_frames"] + 1,
                    max_leg_target_error_rad=max(phase_history[-1]["max_leg_target_error_rad"], error))
                row["controller_phase_history"][-1] = segment
            for name in EVENTS:
                if events[name][lane]:
                    if row["ever_completed"][name] or occurrences[name][lane] < 0:
                        raise ValueError("Recovery milestone pulse duplicated or missing its physical occurrence")
                    row["ever_completed"][name] = True
                    row["first_event_reference_tick"][name] = occurrences[name][lane]
            if row["ever_completed"]["legacy_followup_complete"] and not terminal["true_success"]:
                row["followup_predicate_lost_after_completion"] = True
            row["policy_ticks"] = ticks[lane]
            row["eligible_rl_frames"] += masks[lane]
            row["masked_frames"] += not masks[lane]
            history = row["previous_actor_owned_intervals"]
            if (row["ever_completed"]["rl_entered"] and masks[lane]
                    and len(row["first_actor_action_history"]) < 3):
                # Keep the paid interval's real ownership. A mid-interval
                # handoff can end in RL while still having a masked prefix.
                previous_actor = len(history) >= 1 and history[-1]
                previous_two_actor = len(history) == 2 and all(history)
                row["first_actor_action_history"].append({
                    "policy_interval_id": interval_id,
                    "episode_policy_tick": ticks[lane],
                    "eligible_actor_interval_index": len(row["first_actor_action_history"]) + 1,
                    "prior_interval_actor_owned": previous_actor,
                    "prior_two_intervals_actor_owned": previous_two_actor,
                    "done": done[lane], "measurement_valid": valid[lane],
                    "effective_command": command[lane],
                    "observed_height_m": height[lane],
                    "body_COM_velocity_m_s": velocity[lane],
                    "body_angular_velocity_rad_s": omega[lane],
                    "paid_reward": paid_reward[lane],
                    **{name: readings[lane] for name, readings in actions.items()},
                    "paid_action_history_terms": {
                        name: readings[lane] for name, readings in action_payments.items()},
                    "terms_crossing_control_ownership": [
                        name for name in self.ACTION_HISTORY_PAYMENTS
                        if not (previous_actor if name == "action_rate" else previous_two_actor)],
                })
            row["previous_actor_owned_intervals"] = (history + [masks[lane]])[-2:]
            if row["ever_completed"]["rl_entered"] and terminal["pure_rl"] and masks[lane]:
                followup = row["followup_observation"]
                if valid[lane]:
                    prior_command = followup["previous_command"]
                    command_changed = prior_command is not None and any(
                        abs(a - b) > 1e-7 for a, b in zip(prior_command, command[lane]))
                    predicate = terminal["followup_s"] > 0.
                    lost = followup["previous_predicate"] and not predicate
                    followup["observed_frames"] += 1
                    followup["predicate_frames"] += predicate
                    followup["longest_producer_followup_s"] = max(
                        followup["longest_producer_followup_s"], terminal["followup_s"])
                    followup["predicate_losses"] += lost
                    followup["command_changes"] += command_changed
                    followup["losses_with_same_interval_command_change"] += lost and command_changed
                    followup["previous_command"] = command[lane]
                    followup["previous_predicate"] = predicate
                else:
                    # An unmeasured interval cannot establish an adjacent loss
                    # or command-change association across that gap.
                    followup["previous_command"] = None
                    followup["previous_predicate"] = False
                window = row["post_handoff_windows"]["rl_followup" if terminal["motion_released"] and not held[lane] else "rl_hold"]
                window["frames"] += 1
                if valid[lane]:
                    window["valid_frames"] += 1
                    window["height_abs_m"] += abs(height[lane] - command[lane][2])
                    window["vx_abs_m_s"] += abs(velocity[lane][0] - command[lane][0])
                    window["yaw_abs_rad_s"] += abs(omega[lane][2] - command[lane][1])
            else:
                row["followup_observation"]["previous_command"] = None
                row["followup_observation"]["previous_predicate"] = False
            row.update(last_diagnostics=terminal, last_done=done[lane], last_terminated=terminated[lane])
            pending.append((lane, row))
        return interval_id, pending

    def accept(self, pending):
        interval_id, rows = pending
        if interval_id != self.last_interval_id + 1:
            raise ValueError("Recovery batch was already accepted or skipped")
        self.last_interval_id = interval_id
        for lane, row in rows:
            self.active[lane] = row

    def close_records(self, records):
        from .evaluation import recovery_episode_diagnostics
        for record in records:
            lane = record["lane"]
            row = self.active.get(lane)
            if row is None:
                continue
            if row["episode_id"] != record["episode_id"]:
                raise ValueError("Closed recovery record belongs to another physical episode")
            row = deepcopy(row)
            row.pop("previous_actor_owned_intervals")
            row["outcome"] = record["outcome"]
            row["legacy_terminal_diagnosis"] = recovery_episode_diagnostics(
                {"recovery_episodes": [row["last_diagnostics"]]}, 1)[0]
            row["terminal_measurement"] = self.terminal_measurement(row, self.source_identity)
            followup = row["followup_observation"]
            followup.pop("previous_command")
            followup.pop("previous_predicate")
            followup["predicate_fraction"] = (followup["predicate_frames"] / followup["observed_frames"]
                if followup["observed_frames"] else None)
            row["terminal_after_handoff"] = bool(row["last_terminated"] and row["ever_completed"]["rl_entered"])
            for window in row["post_handoff_windows"].values():
                count = window["valid_frames"]
                window["seconds"] = window["frames"] * self.protocol["policy_dt_s"]
                for name in ("height_abs_m", "vx_abs_m_s", "yaw_abs_rad_s"):
                    window[name.replace("abs", "mae")] = window.pop(name) / count if count else None
            record["scripted_recovery"] = row
            self.rows.append(row)
            self.active.pop(lane)

    def report(self):
        cases = {}
        for name, case in self.cases.items():
            rows = [r for r in self.rows if r["case_id"] == name]
            cases[name] = dict(profile=case["profile"], chain_type=case["chain_type"],
                recorded_episodes=len(rows), closed_episodes=sum(r["outcome"] != "censored" for r in rows),
                censored_episodes=sum(r["outcome"] == "censored" for r in rows),
                ever_completed={e: sum(r["ever_completed"][e] for r in rows) for e in EVENTS},
                terminal_after_handoff=sum(r["terminal_after_handoff"] for r in rows),
                endpoint_producer_followup_complete=sum(
                    r["terminal_measurement"]["endpoint_producer_followup_complete"] for r in rows),
                followup_predicate_lost_after_completion=sum(r["followup_predicate_lost_after_completion"] for r in rows))
        return dict(schema=SCHEMA, cases=cases, records=deepcopy(self.rows), open_episodes=len(self.active),
            controller_phase_history_scope=("First and last pre-reset observations of each contiguous sampled "
                "phase/route/pure-RL/motion-release/whole-interval-mask segment, plus maximum leg target error. "
                "Samples are paid 20ms endpoints, not a complete 1ms state-machine transition trace. Motor "
                "position/velocity/effort and continuous feedback use the existing CONTROL order; leg error "
                "uses its four leg axes. Script target error is relative to continuous feedback, not native "
                "position. Observed support and capture-ready are controller estimates, not raw contact points. "
                "Optional absent sensor fields mean unmeasured. No controller state or threshold is changed."),
            action_history_scope=("First three whole actor-eligible policy intervals after the actual rl_entered event, "
                "including an actor-owned terminal interval. Raw policy-order outputs and already-paid 20ms terms "
                "are copied from pre-reset diagnostics. Ownership comes from the existing whole-interval learning "
                "mask. Crossing-ownership terms are measured, not removed, recomputed or declared erroneous. "
                "No counterfactual return or physical intervention is implied."),
            producer_identity=deepcopy(self.source_identity), assessment="measurement_only", qualification=False,
            followup_observation_scope=("Only valid whole actor-owned pure-RL endpoints are counted. "
                "The producer's existing followup predicate includes contacts, posture, height and tracking; "
                "its continuous timer can include the initial partially masked handoff interval. "
                "Command-change coincidence is not evidence of causality. No new physical pass threshold "
                "is introduced. Direct takeover has neither a required blend nor the legacy pre-release "
                "one-second hold; its archived legacy_terminal_diagnosis is not applicable. "
                "These per-interval summaries cannot be reconstructed from old first/last phase witnesses."),
            ppo_updates=0, actor_direct_loss_credit=0, handoff_source="continuous_native_script",
            scope="Ordinary reference cases remain in episode_ledger/case_tracking_measurements and are not counted as recovery episodes. command_override_frames includes source reset overrides and actual recovery hold intervals. Main tracking spans the whole chain; post_handoff_windows isolate the real handoff. Loss of the followup predicate can reflect changed commands, not necessarily physical instability. Milestones are ever-observed producer events, separate from endpoint state and later loss. Legacy physical predicates/diagnoses are measurements, not approved V7 pass thresholds. Windows include only whole eligible pure-RL policy intervals; no dynamic snapshot was imported.")
