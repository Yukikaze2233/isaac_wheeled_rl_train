"""Passive selected-lane REC journal; attach to an existing diagnostic only.

The caller supplies a fresh source023 mechanical_trace.Hooks instance. No
simulator access, producer calls, RNG, driver or process lifecycle lives here.
Selected device snapshots are serialized only by after_policy (50 Hz).
"""
from collections import deque

import torch


GATES = ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed")
FSM_FIELDS = ("enabled", "phase", "route", "age_ticks", "phase_ticks", "stable_ticks",
              "stable_peak_ticks", "stable_break_count", "stable_pass", "holding_rl",
              "motion_released", "failure_code")
OUTCOME_FIELDS = ("true_stable_seconds", "true_stable_peak", "motion_release_observed",
                  "premature_release", "followup_seconds", "true_success", "goal_platform")
HOLD_FIELDS = ("mask",) + tuple(axis + "/" + field for axis in ("leg", "wheel", "gyro")
                               for field in ("gate_relative_error", "broad_density", "fine_density"))


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, deque)):
        return [_plain(item) for item in value]
    return value


class HoldingTrace:
    """One bounded journal, with no all-lane device-to-host reads at 5 ms."""

    def __init__(self, env, hooks, lanes, *, max_policy_records=64, max_pending_events=24):
        if (env.dt, env.policy_dt, env.decimation) != (.005, .02, 4):
            raise ValueError("Holding trace requires the native 5/20 ms clocks")
        lanes = tuple(lanes)
        if (not lanes or len(lanes) > 8 or len(set(lanes)) != len(lanes)
                or any(type(lane) is not int or not 0 <= lane < env.num_envs for lane in lanes)):
            raise ValueError("Select one to eight distinct native lanes")
        if (type(max_policy_records) is not int or max_policy_records < 1
                or type(max_pending_events) is not int or max_pending_events < 8):
            raise ValueError("Invalid bounded holding journal capacity")
        if hooks.items:
            raise ValueError("Holding trace needs its own fresh Hooks owner")
        self.env, self.recovery, self.hooks = env, env.recovery, hooks
        self.fsm = self.recovery.fsm
        self.lanes = lanes
        self.indices = torch.tensor(lanes, dtype=torch.long, device=self.fsm.device)
        self.birth = torch.zeros(len(lanes), dtype=torch.long, device=self.fsm.device)
        self.records = deque(maxlen=max_policy_records)
        self.between = deque(maxlen=max_pending_events)
        self.capacity = max_pending_events
        self.interval = None
        self.closed = False
        self.sequence = self.policy_count = self.dropped_events = self.dropped_policies = 0
        self.reset_depth = 0
        self.calls = dict(begin_policy=0, fsm_update=0, fsm_reset=0, episode_reset=0, outcome=0)
        # Validate existing fields once, before binding any original function.
        self._fsm_state()
        self._outcome_state()
        for name in ("learning_mask", "policy_motion_hold"):
            if getattr(self.recovery, name).shape != (env.num_envs,):
                raise ValueError("Native REC ownership population differs")
        try:
            hooks.bind(self.recovery, "begin_policy", self._begin_hook)
            hooks.bind(self.fsm, "update", self._update_hook)
            hooks.bind(self.fsm, "reset", self._fsm_reset_hook)
            hooks.bind(self.recovery, "reset", self._episode_reset_hook)
            hooks.bind(self.recovery, "outcome", self._outcome_hook)
        except BaseException:
            hooks.restore()
            raise

    def _sample(self, value):
        # Malformed original inputs are left for the original validator. Never
        # replace its exception with an indexing exception from instrumentation.
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.env.num_envs:
            return value.detach().index_select(0, self.indices).clone()
        return None

    def _fsm_state(self):
        return {**{name: self._sample(getattr(self.fsm, name)) for name in FSM_FIELDS},
                "gates": {name: self._sample(self.fsm.stable_pass_masks[name]) for name in GATES}}

    def _outcome_state(self):
        return {name: self._sample(getattr(self.recovery, name)) for name in OUTCOME_FIELDS}

    def _event(self, kind, **fields):
        self.sequence += 1
        return dict(sequence=self.sequence, kind=kind, native_env_tick=int(self.env.tick),
                    birth_serial_since_attach=self.birth.clone(), **fields)

    def _append(self, event):
        target = self.between if self.interval is None else self.interval["events"]
        if len(target) == target.maxlen:
            self.dropped_events += 1
        target.append(event)

    @staticmethod
    def _inputs(names, args, kwargs):
        return {name: args[index] if index < len(args) else kwargs.get(name)
                for index, name in enumerate(names)}

    def _begin_hook(self, original):
        def wrapped(*args, **kwargs):
            self.calls["begin_policy"] += 1
            before = self._sample(self.recovery.learning_mask)
            result = original(*args, **kwargs)
            if self.interval is not None:
                self.dropped_policies += 1
            self.policy_count += 1
            self.interval = dict(policy_tick=int(self.env.tick) + 1,
                birth_at_begin=self.birth.clone(), reference_calls=0, outcome_calls=0,
                events=deque(maxlen=self.capacity), between_policy_events=list(self.between),
                dropped_events_at_begin=self.dropped_events,
                begin=self._event("begin_policy_20ms", learning_mask_before=before,
                    learning_mask_after=self._sample(self.recovery.learning_mask),
                    recovery_enabled=self._sample(self.recovery.enabled),
                    release_finished=self._sample(self.recovery.release_finished),
                    policy_motion_hold=self._sample(self.recovery.policy_motion_hold),
                    fsm=self._fsm_state()))
            self.between.clear()
            return result
        return wrapped

    def _update_hook(self, original):
        def wrapped(*args, **kwargs):
            self.calls["fsm_update"] += 1
            if self.interval is not None:
                self.interval["reference_calls"] += 1
                ordinal = self.interval["reference_calls"]
                reference = (self.interval["policy_tick"] - 1) * self.env.decimation + ordinal
            else:
                ordinal = reference = None
            inputs = self._inputs(("q_control", "dq_control", "gravity", "gyro",
                                   "estimated_height", "support", "body_clear"), args, kwargs)
            sampled_inputs = {name: self._sample(value) for name, value in inputs.items()}
            before = self._fsm_state()
            result = original(*args, **kwargs)
            self._append(self._event("fsm_reference_5ms", reference_index=reference,
                ordinal_within_policy=ordinal,
                nominal_observation_physical_endpoint=None if reference is None else reference - 1,
                inputs=sampled_inputs,
                before=before, after=self._fsm_state(),
                actor_interval_mask_so_far=self._sample(self.recovery.learning_mask)))
            return result
        return wrapped

    def _fsm_reset_hook(self, original):
        def wrapped(*args, **kwargs):
            self.calls["fsm_reset"] += 1
            ids = args[0] if args else kwargs.get("ids")
            before = self._fsm_state()
            result = original(*args, **kwargs)
            touched = torch.isin(self.indices, torch.as_tensor(ids, device=self.indices.device))
            self._append(self._event("fsm_reset", episode_reset_nested=bool(self.reset_depth),
                                     selected_reset_mask=touched.clone(),
                                     before=before, after=self._fsm_state()))
            return result
        return wrapped

    def _episode_reset_hook(self, original):
        def wrapped(*args, **kwargs):
            self.calls["episode_reset"] += 1
            ids = args[0] if args else kwargs.get("ids")
            before = self._fsm_state()
            birth_before = self.birth.clone()
            self.reset_depth += 1
            try:
                result = original(*args, **kwargs)
            finally:
                self.reset_depth -= 1
            # Only an actual RecoveryTraining.reset advances episode identity.
            # A release-time FSM.reset inside advance_reference does not.
            touched = torch.isin(self.indices, torch.as_tensor(ids, device=self.indices.device))
            self.birth += touched.long()
            self._append(self._event("episode_reset", selected_reset_mask=touched.clone(),
                birth_before=birth_before, before=before, after=self._fsm_state()))
            return result
        return wrapped

    def _outcome_hook(self, original):
        def wrapped(*args, **kwargs):
            self.calls["outcome"] += 1
            if self.interval is not None:
                self.interval["outcome_calls"] += 1
            inputs = self._inputs(("contact", "undesired", "gravity", "omega", "height", "velocity"), args, kwargs)
            inputs["goal_platform"] = kwargs.get("goal_platform")
            sampled_inputs = {name: self._sample(value) for name, value in inputs.items()}
            before = self._outcome_state()
            result = original(*args, **kwargs)
            self._append(self._event("truth_outcome_20ms", physical_endpoint=int(self.env.tick) * self.env.decimation,
                inputs=sampled_inputs,
                before=before, after=self._outcome_state(), original_success=self._sample(result)))
            return result
        return wrapped

    def after_policy(self, tick, extras):
        """Consume the existing pre-reset whole-interval mask; never recompute it."""
        interval = self.interval
        if interval is None or tick != interval["policy_tick"] or tick != int(self.env.tick):
            raise ValueError("Holding trace policy endpoint is not aligned")
        mask = extras["policy_learning_mask"]
        if mask.shape != (self.env.num_envs,) or mask.dtype != torch.bool:
            raise ValueError("Whole-policy actor mask shape/dtype differs")
        diagnostics = getattr(self.env, "recovery_hold_cost_diagnostics", {})
        row = {**interval, "actor_valid_whole_policy": self._sample(mask),
            "hold_cost_producer_diagnostics_20ms": {key: self._sample(value)
                for key, value in diagnostics.items() if key in HOLD_FIELDS},
            "complete_reference_and_outcome_counts": interval["reference_calls"] == 4 and interval["outcome_calls"] == 1,
            "events_dropped_in_interval": self.dropped_events - interval["dropped_events_at_begin"]}
        if len(self.records) == self.records.maxlen:
            self.dropped_policies += 1
        self.records.append(_plain(row))
        self.interval = None

    def close(self):
        if not self.closed:
            self.hooks.restore()
            self.closed = True

    def summary(self):
        return dict(schema="source023-passive-selected-rec-hold-v1", lanes=list(self.lanes),
            records=list(self.records), calls=dict(self.calls), dropped_events=self.dropped_events,
            dropped_policies=self.dropped_policies, pending_interval=self.interval is not None,
            hooks_restored=self.closed, qualification=None, observer_PPO_updates=0,
            native_execution_verified=False, additional_simulator_queries=0,
            timing_scope={"fsm_reference": "5ms pre-step inputs; nominal endpoint k-1, USB feedback age unobserved",
                          "native_gate_masks": "outputs of that exact FSM.update; estimated support/body_clear",
                          "actor_interval_mask_so_far": "existing mask at reference callback, before this substep torque bookkeeping",
                          "true_outcome": "20ms / 50Hz physical endpoint 4*n only",
                          "actor_valid_whole_policy": "original extras mask, birth_at_begin survives autoreset",
                          "hold_cost_diagnostics": "existing producer state at 20ms; absent means unobserved"},
            unsupported=["true contact at every 5ms", "feedback timestamps/age", "unobserved resets before attachment",
                         "full production execution or function qualification", "GPU timing overhead"])
