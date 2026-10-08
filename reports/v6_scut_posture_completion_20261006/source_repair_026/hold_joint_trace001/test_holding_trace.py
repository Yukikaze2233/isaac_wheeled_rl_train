"""CPU-only semantics on frozen source023 functions, without Isaac imports."""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COMMIT = "6f07ec298e01d2686216787cfba34b394151f1aa"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace = load(HERE / "holding_trace.py", "cpu_holding_trace")
mechanical = load(HERE.parents[1] / "source_repair_023/latest_seal_mechanical_trace001/mechanical_trace.py", "cpu_existing_mechanical_hooks")


def frozen(path):
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


RECOVERY_BYTES = frozen("src/wheeled_tasks/chassis/recovery.py")
assert hashlib.sha256(RECOVERY_BYTES).hexdigest() == "04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6"
native = {}
exec(compile(RECOVERY_BYTES, "source023/recovery.py", "exec"), native)
training_tree = ast.parse(frozen("src/wheeled_tasks/chassis/recovery_training.py"))
training_class = next(node for node in training_tree.body if isinstance(node, ast.ClassDef) and node.name == "RecoveryTraining")
required = ("begin_policy", "outcome", "pure_rl", "prelude", "failed")
selected = [node for node in training_class.body if isinstance(node, ast.FunctionDef) and node.name in required]
training_ast = ast.Module(body=[ast.ClassDef(name="FrozenTraining", bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
ast.fix_missing_locations(training_ast)
training_space = dict(torch=torch, math=math)
exec(compile(training_ast, "source023/recovery_training.py:selected_native_methods", "exec"), training_space)


class Recovery(training_space["FrozenTraining"]):
    def __init__(self, env):
        self.env = env
        n = env.num_envs
        self.fsm = native["RecoveryFSM"](n, "cpu", {}, *([torch.zeros(6)] * 4), torch.ones(4))
        self.enabled = torch.ones(n, dtype=torch.bool)
        self.release_finished = torch.ones(n, dtype=torch.bool)
        self.elapsed = torch.zeros(n)
        self.settings = {}
        self.required_platform_height = torch.zeros(n)
        self.followup_command = torch.zeros(n, 3)
        self.followup_command[:, 2] = .305
        for name in ("learning_mask", "policy_motion_hold", "motion_release_observed", "premature_release", "true_success", "goal_platform"):
            setattr(self, name, torch.zeros(n, dtype=torch.bool))
        for name in ("true_stable_seconds", "true_stable_peak", "followup_seconds"):
            setattr(self, name, torch.zeros(n))
        self.original_calls = dict(begin_policy=0, update=0, reset=0, outcome=0, episode_reset=0)
        self.last_returns = {}
        for owner, name, key in ((self, "begin_policy", "begin_policy"), (self.fsm, "update", "update"),
                                 (self.fsm, "reset", "reset"), (self, "outcome", "outcome")):
            original = getattr(owner, name)

            def counted(*args, _original=original, _key=key, **kwargs):
                self.original_calls[_key] += 1
                result = _original(*args, **kwargs)
                self.last_returns[_key] = result
                return result
            setattr(owner, name, counted)
        # Episode lifecycle fixture only. The real FSM reset, begin_policy and
        # outcome bodies above are frozen native functions, not reimplementations.
        self.fsm.reset(torch.arange(n), torch.zeros(n, 6), True)
        self.fsm.phase.fill_(int(native["RecoveryPhase"].RL))

    def reset(self, ids):
        self.original_calls["episode_reset"] += 1
        self.fsm.reset(ids, torch.zeros(self.env.num_envs, 6), self.enabled)
        for name in trace.OUTCOME_FIELDS:
            getattr(self, name)[ids] = 0
        return self.env.reset_token


def env(count=4):
    e = SimpleNamespace(num_envs=count, tick=0, dt=.005, policy_dt=.02, decimation=4, reset_token=object())
    e.recovery = Recovery(e)
    e.recovery_hold_cost_diagnostics = {}
    return e


def inputs(e):
    n = e.num_envs
    q = torch.zeros(n, 6)
    gravity = torch.zeros(n, 3)
    gravity[:, 2] = -1.
    return (q, torch.zeros_like(q), gravity, torch.zeros(n, 3), torch.full((n,), .305),
            torch.ones(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool))


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), value.contiguous().numpy().tobytes())
    if isinstance(value, dict):
        return {key: tensor_bytes(item) for key, item in value.items() if isinstance(item, (torch.Tensor, dict))}
    return None


def cycle(e, observer=None, *, broken_substep=None, reset_ids=None, internal_reset=False):
    r = e.recovery
    assert r.begin_policy() is r.last_returns["begin_policy"]
    values = inputs(e)
    before = [tensor_bytes(value) for value in values]
    for index in range(1, 5):
        support = values[5].clone()
        if index == broken_substep:
            support[1] = False
        result = r.fsm.update(*values[:5], support, values[6])
        assert result is r.last_returns["update"] is r.fsm.script_targets
        if index == 1 and internal_reset:
            assert r.fsm.reset(torch.tensor([3]), values[0], True) is r.last_returns["reset"]
    assert before == [tensor_bytes(value) for value in values]
    e.tick += 1
    contact = torch.ones(e.num_envs, 2, dtype=torch.bool)
    # Different truth from estimated support is intentional; do not equate it.
    contact[1, 0] = False
    undesired = torch.zeros(e.num_envs, dtype=torch.bool)
    omega = torch.zeros(e.num_envs, 3)
    velocity = torch.zeros_like(omega)
    result = r.outcome(contact, undesired, values[2], omega, values[4], velocity)
    assert result is r.last_returns["outcome"] is r.true_success
    extras = {"policy_learning_mask": r.learning_mask.clone()}
    e.recovery_hold_cost_diagnostics = {"mask": r.enabled & r.learning_mask & r.policy_motion_hold,
                                      "wheel/broad_density": torch.full((e.num_envs,), -.1)}
    extra_bytes = tensor_bytes(extras)
    if reset_ids is not None:
        assert r.reset(torch.tensor(reset_ids)) is e.reset_token
    if observer:
        observer.after_policy(e.tick, extras)
    assert tensor_bytes(extras) == extra_bytes
    return extras


def test_frozen_functions_unchanged_return_tensor_bytes_rng_and_exact_calls():
    states = []
    for instrumented in (False, True):
        torch.manual_seed(818)
        e = env()
        original_methods = (e.recovery.begin_policy, e.recovery.fsm.update, e.recovery.reset)
        rng = torch.get_rng_state().clone()
        observer = trace.HoldingTrace(e, mechanical.Hooks(), [1, 3]) if instrumented else None
        assert torch.equal(rng, torch.get_rng_state())
        for _ in range(3):
            cycle(e, observer, broken_substep=3)
        if observer:
            observer.close()
            assert (e.recovery.begin_policy, e.recovery.fsm.update, e.recovery.reset) == original_methods
            assert observer.calls == dict(begin_policy=3, fsm_update=12, fsm_reset=0, episode_reset=0, outcome=3)
        states.append((tensor_bytes(vars(e.recovery.fsm)), tensor_bytes(vars(e.recovery)),
                       e.recovery.original_calls, torch.get_rng_state().clone()))
    assert states[0][:3] == states[1][:3]
    assert torch.equal(states[0][3], states[1][3])
    assert not torch.cuda.is_initialized()


def test_joint_native_masks_before_after_break_and_distinct_truth_clock():
    e = env()
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1, 3])
    cycle(e, observer, broken_substep=3)
    row = observer.records[-1]
    ref = [item for item in row["events"] if item["kind"] == "fsm_reference_5ms"]
    assert [item["reference_index"] for item in ref] == [1, 2, 3, 4]
    assert [item["nominal_observation_physical_endpoint"] for item in ref] == [0, 1, 2, 3]
    assert ref[2]["before"]["stable_ticks"] == [2, 2]
    assert ref[2]["after"]["stable_ticks"] == [0, 3]
    assert ref[2]["after"]["stable_break_count"] == [1, 0]
    assert ref[2]["after"]["gates"]["support"] == [False, True]
    assert set(ref[2]["after"]["gates"]) == set(trace.GATES)
    outcome = next(item for item in row["events"] if item["kind"] == "truth_outcome_20ms")
    assert outcome["physical_endpoint"] == 4
    assert outcome["inputs"]["contact"] == [[False, True], [True, True]]
    assert "contact" not in ref[3]["inputs"]
    assert row["begin"]["policy_motion_hold"] == [True, True]
    assert row["actor_valid_whole_policy"] == [True, True]
    assert row["complete_reference_and_outcome_counts"]
    observer.close()


def test_reference_wraps_policy_clock_and_autoreset_preserves_old_actor_birth():
    e = env()
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1, 3])
    cycle(e, observer, reset_ids=[1])
    row = observer.records[-1]
    assert row["birth_at_begin"] == [0, 0]
    assert row["actor_valid_whole_policy"] == [True, True]
    reset = next(item for item in row["events"] if item["kind"] == "episode_reset")
    assert reset["birth_before"] == [0, 0]
    assert reset["birth_serial_since_attach"] == [1, 0]
    nested = next(item for item in row["events"] if item["kind"] == "fsm_reset")
    assert nested["episode_reset_nested"]
    cycle(e, observer)
    row = observer.records[-1]
    assert row["birth_at_begin"] == [1, 0]
    refs = [item for item in row["events"] if item["kind"] == "fsm_reference_5ms"]
    assert refs[0]["reference_index"] == 5
    assert refs[0]["nominal_observation_physical_endpoint"] == 4
    observer.close()


def test_release_time_fsm_reset_is_not_episode_birth_or_false_contact():
    e = env()
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1, 3])
    cycle(e, observer, internal_reset=True)
    row = observer.records[-1]
    reset = next(item for item in row["events"] if item["kind"] == "fsm_reset")
    assert not reset["episode_reset_nested"]
    assert reset["birth_serial_since_attach"] == [0, 0]
    assert observer.birth.tolist() == [0, 0]
    assert observer.calls["episode_reset"] == 0
    observer.close()


def test_first_release_is_native_200tick_event_and_begin_hold_remains_true():
    e = env()
    e.recovery.fsm.stable_ticks.fill_(199)
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1])
    cycle(e, observer)
    row = observer.records[-1]
    ref = next(item for item in row["events"] if item["kind"] == "fsm_reference_5ms")
    assert ref["before"]["motion_released"] == [False]
    assert ref["after"]["motion_released"] == [True]
    assert ref["after"]["stable_ticks"] == [200]
    assert row["begin"]["policy_motion_hold"] == [True]
    assert row["hold_cost_producer_diagnostics_20ms"]["mask"] == [True]
    observer.close()


def test_blend_to_rl_records_native_phase_before_dwell_and_excluded_whole_interval():
    e = env()
    e.recovery.fsm.phase.fill_(int(native["RecoveryPhase"].BLEND))
    e.recovery.fsm.phase_ticks.fill_(39)
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1])
    cycle(e, observer)
    row = observer.records[-1]
    references = [item for item in row["events"] if item["kind"] == "fsm_reference_5ms"]
    assert references[0]["before"]["phase"] == [8]
    assert references[0]["after"]["phase"] == [9]
    assert references[0]["after"]["holding_rl"] == [False]
    assert references[0]["after"]["stable_ticks"] == [0]
    assert references[1]["after"]["holding_rl"] == [True]
    assert references[1]["after"]["stable_ticks"] == [1]
    assert row["begin"]["learning_mask_after"] == [False]
    assert row["actor_valid_whole_policy"] == [False]
    assert row["hold_cost_producer_diagnostics_20ms"]["mask"] == [False]
    observer.close()


def test_selected_snapshots_do_not_keep_or_sync_all4096_rows(monkeypatch):
    e = env(4096)
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1, 2216])
    original_cpu = torch.Tensor.cpu
    sizes = []

    def checked_cpu(value, *args, **kwargs):
        sizes.append(tuple(value.shape))
        assert value.shape[0] == 2
        return original_cpu(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", checked_cpu)
    e.recovery.begin_policy()
    values = inputs(e)
    for _ in range(4):
        e.recovery.fsm.update(*values)
    assert sizes == []  # no per-5ms host sync, including native gates
    e.tick += 1
    e.recovery.outcome(torch.ones(4096, 2, dtype=torch.bool), torch.zeros(4096, dtype=torch.bool),
                       values[2], values[3], values[4], values[3])
    observer.after_policy(1, {"policy_learning_mask": e.recovery.learning_mask.clone()})
    assert sizes and all(shape[0] == 2 for shape in sizes)
    observer.close()


def test_capacity_marks_lost_history_and_default_has_no_added_functions():
    e = env()
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1], max_policy_records=2, max_pending_events=8)
    for _ in range(5):
        cycle(e, observer)
    assert len(observer.records) == 2 and observer.dropped_policies == 3
    e.recovery.begin_policy()
    values = inputs(e)
    for _ in range(10):
        e.recovery.fsm.update(*values)
    assert len(observer.interval["events"]) == 8 and observer.dropped_events == 2
    observer.close()
    observer.close()
    assert observer.summary()["pending_interval"]
    assert observer.summary()["qualification"] is None


def test_original_exception_propagates_unchanged_and_cleanup_restores(monkeypatch):
    e = env()
    exception = RuntimeError("original-only")
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise exception
    monkeypatch.setattr(e.recovery.fsm, "update", broken)
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1])
    try:
        with pytest.raises(RuntimeError) as caught:
            e.recovery.fsm.update(*inputs(e))
        assert caught.value is exception and calls == [1]
    finally:
        observer.close()
    assert e.recovery.fsm.update is broken
    assert observer.summary()["hooks_restored"]


def test_invalid_original_shapes_raise_native_exception_not_observer_indexing():
    e = env()
    values = inputs(e)
    with pytest.raises(ValueError) as native_error:
        e.recovery.fsm.update(values[0][:1], *values[1:])
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [3])
    calls_before = e.recovery.original_calls["update"]
    with pytest.raises(ValueError) as observed_error:
        e.recovery.fsm.update(values[0][:1], *values[1:])
    assert str(observed_error.value) == str(native_error.value)
    assert e.recovery.original_calls["update"] == calls_before + 1
    observer.close()


@pytest.mark.parametrize("gate", trace.GATES)
def test_each_native_gate_break_is_recorded_without_own_gate_solver(gate):
    e = env()
    e.recovery.fsm.stable_ticks.fill_(1)
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1])
    e.recovery.begin_policy()
    values = list(inputs(e))
    if gate == "valid":
        values[0][1, 0] = math.nan
    elif gate == "support":
        values[5][1] = False
    elif gate == "body_clear":
        values[6][1] = False
    elif gate == "tilt":
        values[2][1] = torch.tensor([math.sin(math.radians(9)), 0., -math.cos(math.radians(9))])
    elif gate == "gyro":
        values[3][1, 1] = .75  # native threshold is strict
    elif gate == "height":
        values[4][1] = .4
    elif gate == "wheel_speed":
        values[1][1, 2] = 4.2
    else:
        values[1][1, 0] = 2.  # native threshold is strict
    original_bytes = [tensor_bytes(value) for value in values]
    assert e.recovery.fsm.update(*values) is e.recovery.fsm.script_targets
    item = observer.interval["events"][-1]
    assert not bool(item["after"]["gates"][gate][0])
    assert item["before"]["stable_ticks"].tolist() == [1]
    assert item["after"]["stable_ticks"].tolist() == [0]
    assert item["after"]["stable_break_count"].tolist() == [1]
    assert original_bytes == [tensor_bytes(value) for value in values]
    observer.close()


@pytest.mark.parametrize("mutation", ["clock", "lanes", "capacity", "shared_hooks"])
def test_rejected_attachment_never_binds_original_functions(mutation):
    e = env()
    original = e.recovery.fsm.update
    hooks = mechanical.Hooks()
    lanes, kwargs = [1], {}
    if mutation == "clock":
        e.dt = .001
    elif mutation == "lanes":
        lanes = [1, 1]
    elif mutation == "capacity":
        kwargs["max_pending_events"] = 1
    else:
        hooks.bind(e, "reset_token", lambda original: original)
    with pytest.raises(ValueError):
        trace.HoldingTrace(e, hooks, lanes, **kwargs)
    assert e.recovery.fsm.update is original
    hooks.restore()


def test_endpoint_alignment_and_missing_wholemask_do_not_manufacture_credit():
    e = env()
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1])
    e.recovery.begin_policy()
    with pytest.raises(ValueError, match="not aligned"):
        observer.after_policy(1, {"policy_learning_mask": torch.ones(4, dtype=torch.bool)})
    e.tick = 1
    with pytest.raises(KeyError):
        observer.after_policy(1, {})
    with pytest.raises(ValueError, match="shape/dtype"):
        observer.after_policy(1, {"policy_learning_mask": torch.ones(4)})
    assert not observer.records
    observer.close()


def test_partial_attachment_cleanup_and_hook_collision_are_explicit():
    e = env()
    hooks = mechanical.Hooks()
    bind = hooks.bind
    count = 0
    original = e.recovery.begin_policy

    def fail_second(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("fixture bind failure")
        bind(*args)
    hooks.bind = fail_second
    with pytest.raises(RuntimeError, match="bind failure"):
        trace.HoldingTrace(e, hooks, [1])
    assert e.recovery.begin_policy is original and hooks.items == []
    observer = trace.HoldingTrace(e, mechanical.Hooks(), [1])
    alien = lambda: None
    e.recovery.begin_policy = alien
    with pytest.raises(ValueError, match="another owner"):
        observer.close()
    assert e.recovery.begin_policy is alien
    assert not observer.closed


def test_observer_ast_has_no_simulator_rng_producer_or_process_calls():
    tree = ast.parse((HERE / "holding_trace.py").read_text())
    forbidden = {"step", "reset_joint", "motor_efforts", "set_joint_effort_target_index", "advance_reference",
                 "torque", "outcome", "begin_policy", "reset", "update", "diagnostics", "rand", "randn",
                 "manual_seed", "get_rng_state", "get_state", "subprocess", "SimulationApp", "Popen"}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)} & forbidden
    assert not {node.func.id for node in calls if isinstance(node.func, ast.Name)} & forbidden
    assert not any(isinstance(node, ast.Import) and any(name.name != "torch" for name in node.names)
                   for node in tree.body)
