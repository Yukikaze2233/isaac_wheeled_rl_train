"""Isolated source022 reference-end handoff; no app/PPO/source file mutation.

The authenticated native update body executes once per original observer call.
Only PREPARE readiness and its destination differ in the immediate arm. The
endpoint is the native generated reference's remaining < .02 rad, not measured
joint convergence. Native post-RL dwell/command-release/outcome remain intact.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import types


SOURCE_SHA256 = "04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6"
READY = "ready = preparing & stable & (remaining < .02) & (measured_error < .12)"
ENTER = "self._enter(preparing & (self.ready_ticks >= self._ticks(.1)), RecoveryPhase.BLEND)"
IMMEDIATE_READY = "ready = preparing & (remaining < .02)"
IMMEDIATE_ENTER = "self._enter(ready, RecoveryPhase.RL)"
TELEMETRY = "self._diagnostic_prepare_endpoint(remaining, measured_error, preparing, ready, stable, valid)"


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def node_text(text):
    return ast.parse(text).body[0]


def ast_identity(node):
    return ast.dump(node, include_attributes=False)


def compile_update(native, arm):
    """Return an explicit instance binding; verify the whole AST change scope."""
    require(arm in ("baseline", "immediate"), "Unknown diagnostic arm")
    require(isinstance(native, types.MethodType), "Native update must be a bound method")
    fsm, original = native.__self__, native.__func__
    require(original is type(fsm).update, "Native update class owner differs")
    source = Path(inspect.getsourcefile(original)).resolve()
    require(hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256,
            "Actual source022 RecoveryFSM bytes differ")
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef),
            "Native update source must be exactly one function")
    original_tree = deepcopy(tree)
    body = tree.body[0].body
    ready_hits = [i for i, node in enumerate(body) if ast_identity(node) == ast_identity(node_text(READY))]
    enter_hits = [i for i, node in enumerate(body) if ast_identity(node) == ast_identity(node_text(ENTER))]
    require(len(ready_hits) == len(enter_hits) == 1, "Frozen endpoint/handoff statements differ")
    if arm == "immediate":
        body[ready_hits[0]] = ast.copy_location(node_text(IMMEDIATE_READY), body[ready_hits[0]])
        body[enter_hits[0]] = ast.copy_location(node_text(IMMEDIATE_ENTER), body[enter_hits[0]])
    expected_tree = deepcopy(original_tree)
    if arm == "immediate":
        expected_tree.body[0].body[ready_hits[0]] = node_text(IMMEDIATE_READY)
        expected_tree.body[0].body[enter_hits[0]] = node_text(IMMEDIATE_ENTER)
    require(ast_identity(tree) == ast_identity(expected_tree), "Non-handoff control AST changed")
    control_ast_sha = hashlib.sha256(ast_identity(tree).encode()).hexdigest()
    # Observation of existing locals, before readiness counters/enter, adds no
    # target, FSM, RNG, reward, contact or actuator write.
    body.insert(ready_hits[0] + 1, ast.copy_location(node_text(TELEMETRY), body[ready_hits[0]]))
    ast.fix_missing_locations(tree)
    namespace = {}
    filename = str(Path(__file__).resolve()) + ":generated_" + arm
    exec(compile(tree, filename, "exec"), original.__globals__, namespace)
    replacement = namespace[original.__name__]
    replacement.__qualname__ = original.__qualname__
    replacement.__module__ = original.__module__
    return types.MethodType(replacement, fsm), dict(
        arm=arm, source_path=str(source), source_sha256=SOURCE_SHA256,
        control_ast_sha256=control_ast_sha, native_body_copies_executed_per_call=1,
        modified_control_statements=([IMMEDIATE_READY, IMMEDIATE_ENTER] if arm == "immediate" else []),
        telemetry_statement=TELEMETRY, single_semantic_variable="PREPARE reference-end actor handoff rule")


class ImmediateHandoffHook:
    """Attach after original55 native observer authentication and before stepping.

    hooks is a private existing HookSet. Restoration must precede inherited
    observer finish/cleanup. No native app, simulator, optimizer or runner is
    created here. The root owns actual paired native execution.
    """

    def __init__(self, env, observer, hooks, output, writer, plain, *, arm="immediate",
                 cohort=lambda: (), expected_calls=2400):
        self.env, self.observer, self.fsm = env, observer, env.recovery.fsm
        self.hooks, self.output, self.writer, self.plain = hooks, Path(output), writer, plain
        self.arm, self.cohort, self.expected_calls = arm, cohort, expected_calls
        self.original = observer.original_update
        require(isinstance(self.original, types.MethodType)
                and self.original.__self__ is self.fsm and self.original.__func__ is type(self.fsm).update,
                "Attach to the authenticated original native observer update binding")
        require(not hasattr(self.fsm, "_diagnostic_prepare_endpoint"), "Endpoint hook already present")
        self.compiled, self.code_receipt = compile_update(self.original, arm)
        self.config = deepcopy(self.fsm.config)
        require(self.config["side_speed_rad_s"] == 1.5 and self.config["dt"] == .005
                and self.config["max_script_seconds"] == 8. and self.config["blend_seconds"] == .2
                and self.config["stable_seconds"] == 1., "Frozen native diagnostic config differs")
        self.calls_attempted = self.calls_completed = self.endpoint_calls = 0
        self.intervened_lane_transitions = 0
        self.first_global_intervention = None
        self.first_interventions = {}
        self.first_reference_tick = self.last_reference_tick = None
        self.applied = self.closed = self.complete = self.in_update = False
        self.events = None
        self.pre_phase = self.pre_reference = None
        import torch
        self.first_endpoint = torch.full_like(self.fsm.phase, -1)

    def assert_owner(self):
        require(self.applied and not self.closed and self.fsm.config == self.config,
                "Immediate handoff hook/config ownership differs")

    def endpoint(self, remaining, measured_error, preparing, ready, stable, valid):
        self.assert_owner()
        require(self.in_update, "Endpoint callback outside the one native body call")
        self.endpoint_calls += 1
        endpoint = preparing & (remaining < .02)
        first = endpoint & (self.first_endpoint < 0)
        self.first_endpoint[first] = self.calls_attempted
        intervention = ready if self.arm == "immediate" else ready & False
        if bool(intervention.any()):
            self.intervened_lane_transitions += int(intervention.sum())
            if self.first_global_intervention is None:
                self.first_global_intervention = self.calls_attempted
            for lane, birth in self.cohort():
                if birth == 1 and bool(intervention[lane]) and lane not in self.first_interventions:
                    self.first_interventions[lane] = self.calls_attempted
        selected = []
        for lane, birth in self.cohort():
            if bool(endpoint[lane]):
                selected.append(dict(lane=lane, birth_id=birth, reference_endpoint=True,
                    immediate_requested=bool(intervention[lane]), native_remaining_rad=float(remaining[lane]),
                    native_measured_pair_error_rad=float(measured_error[lane]),
                    phase_before_update=int(self.pre_phase[lane]), phase_before_handoff=int(self.fsm.phase[lane]),
                    previous_script_target_control6=self.plain(self.pre_reference[lane]),
                    generated_endpoint_script_target_control6=self.plain(self.fsm.script_targets[lane]),
                    existing_native_eight_gates={k:bool(v[lane]) for k,v in self.fsm.stable_pass_masks.items()},
                    original_stable=bool(stable[lane]), original_valid=bool(valid[lane])))
        if bool(first.any()) or selected or bool(intervention.any()):
            self.events.write(json.dumps(dict(update_call=self.calls_attempted,
                native_reference_tick=self.last_reference_tick, arm=self.arm,
                input_physical_step=self.calls_attempted-1,
                newly_observed_endpoint_lanes=self.plain(first.nonzero().flatten()),
                immediate_intervention_lanes=self.plain(intervention.nonzero().flatten()), selected=selected,
                endpoint_scope="Generated native PREPARE reference completed within .02rad; measured q may lag",
                torque_scope="The upcoming physics substep uses native torque provider after this reference boundary",
                no_mechanical_convergence_or_functional_success_claim=True), allow_nan=False) + "\n")

    def apply(self):
        require(not self.applied and not self.closed and self.fsm.config == self.config,
                "Duplicate/closed/foreign-config hook apply")
        self.events = gzip.open(self.output / "immediate_handoff_endpoint_events.jsonl.gz", "xt")
        self.applied = True
        def counted(*args, **kwargs):
            self.assert_owner()
            require(not self.in_update, "Recursive native update")
            self.calls_attempted += 1
            self.last_reference_tick = int(self.env.recovery.reference_tick)
            if self.first_reference_tick is None:
                self.first_reference_tick = self.last_reference_tick
            self.pre_phase = self.fsm.phase.clone()
            self.pre_reference = self.fsm.script_targets.clone()
            self.in_update = True
            try:
                result = self.compiled(*args, **kwargs)
                require(result is self.fsm.script_targets, "Original update return identity changed")
                self.calls_completed += 1
                self.assert_owner()
                return result
            finally:
                self.in_update = False
                self.pre_phase = self.pre_reference = None
        try:
            # NativeObserver.update calls its dynamically owned original_update
            # once. All outer read-only observation callbacks remain installed.
            # HookSet requires a preexisting attribute for lossless restoration.
            self.fsm._diagnostic_prepare_endpoint = None
            self.hooks.install(self.fsm, "_diagnostic_prepare_endpoint", self.endpoint)
            self.hooks.install(self.observer, "original_update", counted)
            self.writer(self.output / "immediate_handoff_hook_applied.json", self.receipt("applied_before_first_policy"))
        except BaseException:
            self.cleanup()
            raise

    def receipt(self, status):
        return dict(schema="source022-isolated-reference-end-handoff-v1", status=status,
            arm=self.arm, code=self.code_receipt, original_source_files_modified=False,
            explicit_runtime_control_override=self.arm == "immediate",
            endpoint_definition="Native PREPARE generated remaining<.02rad; measured_error and eight gates are recorded, not handoff predicates",
            control_handoff="Immediate arm enters pureRL/blend=1 at the generated-reference boundary before the upcoming physical torque call",
            endpoint_last_script_target_has_not_yet_been_applied_for_a_full_physics_substep=True,
            post_RL_semantics="Original continuous1s stable gate, physical truth outcome, and followup vx/yaw release remain; these do not block actor torque ownership",
            config_unchanged=self.fsm.config == self.config, config=self.config,
            native_body_calls_attempted=self.calls_attempted, native_body_calls_completed=self.calls_completed,
            endpoint_telemetry_calls=self.endpoint_calls,
            first_global_intervention_update_call=self.first_global_intervention,
            selected_first_birth_first_intervention_update_call=self.first_interventions,
            intervened_lane_transitions=self.intervened_lane_transitions,
            native_reference_tick_first=self.first_reference_tick, native_reference_tick_last=self.last_reference_tick,
            complete=self.complete, hooks_restored=self.closed and self.hooks.closed,
            no_pending_update=not self.in_update, no_state_teleport=True, no_second_FSM_or_controller_call=True,
            PPO_updates=0, formal_updates_consumed=0, qualification=None)

    def finish(self):
        self.assert_owner()
        require(not self.in_update and self.calls_attempted == self.calls_completed == self.expected_calls,
                "Native reference call completion differs")
        require(self.endpoint_calls == self.calls_completed,
                "Original body endpoint telemetry was not executed once per update")
        require(self.last_reference_tick - self.first_reference_tick == self.expected_calls - 1,
                "Original native reference cadence differs")
        self.complete = True
        return self.cleanup()

    def cleanup(self):
        if self.closed:
            return []
        errors = self.hooks.restore()
        if self.in_update:
            errors.append("Cleanup during pending native update")
        if self.fsm.config != self.config:
            errors.append("Foreign FSM config mutation")
        if "_diagnostic_prepare_endpoint" in vars(self.fsm) and self.fsm._diagnostic_prepare_endpoint is None:
            delattr(self.fsm, "_diagnostic_prepare_endpoint")
        else:
            errors.append("Endpoint telemetry restoration ownership changed")
        self.closed = True
        if self.events is not None:
            self.events.close()
        self.writer(self.output / "immediate_handoff_hook_preclose.json", self.receipt(
            "complete_restored_preclose" if self.complete and not errors else "incomplete_preserved") | {"errors": errors})
        return errors
