"""V7 CPU state oracle extracted from the frozen source, without native imports.

The terrain command sampler only records entry events here. The source's .15
terrain sampling, RNG consumption, command override and chassis contact/scan
mapping are separate unimplemented adapters, not certified by these checks.
"""
import ast
import copy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.source_airborne_state import (
    ENTER, EXIT, POLICY_DT_S, AirbornePhysicalObservation, SourceAirborneState,
)
from test_v7_final_rough_reward import assert_source_oracle, frame as reward_frame


FROZEN = Path(__file__).resolve().parents[1] / "references/v7_final_rough_b8ff79f3"
FIELDS = {
    "airborne": "height_reward_airborne_state",
    "enter_time_s": "airborne_enter_time",
    "wheel_contact_elapsed_s": "airborne_wheel_contact_exit_time",
    "base_contact_elapsed_s": "airborne_base_contact_exit_time",
    "current_duration_s": "airborne_current_duration",
    "maximum_duration_s": "airborne_max_duration",
    "entry_command3": "airborne_entry_command",
    "pending_force_enter": "airborne_force_enter_request",
    "last_update_step": "airborne_state_last_update_step",
}


@lru_cache
def frozen_configuration():
    tree = ast.parse((FROZEN / "env_cfg.py").read_text())
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "WheelbipeV14FlatEnvCfg")
    assignment = next(node for node in base.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "airborne_state_machine_cfg"
                for target in node.targets))
    cfg = ast.literal_eval(assignment.value)
    # Flat-v1 activates the inherited machine; target/trajectory stay disabled.
    flat = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "WheelbipeV14FlatEnvCfg_v1")
    post = next(node for node in flat.body if isinstance(node, ast.FunctionDef)
                and node.name == "__post_init__")
    enabled = next(node for node in post.body if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "self.airborne_state_machine_cfg['enabled']")
    cfg["enabled"] = ast.literal_eval(enabled.value)
    assert cfg["enabled"] is True
    assert cfg["target_height"]["enabled"] is False
    assert cfg["landing_trajectory"]["enabled"] is False
    return cfg


@lru_cache
def frozen_state_class():
    tree = ast.parse((FROZEN / "airborne.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "AirborneStateMachine")
    names = {"_cfg", "_as_name_tuple", "_get_allowed_terrain_mask",
        "_height_reward_override_enabled", "_get_body_relative_ground_height",
        "_ensure_entry_command_buffers", "_clear_entry_command", "_clear_command_override",
        "_ensure_landing_trajectory_buffers", "_clear_landing_trajectory",
        "_landing_trajectory_cfg", "_update_landing_trajectory", "_zero_exit_timers",
        "_get_base_contact_indices", "_get_current_contact_forces",
        "on_observation_step", "on_reset", "get_height_reward_reference_height",
        "get_height_reward_target_height"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in names]
    assert {node.name for node in methods} == names
    extracted = ast.ClassDef(name="FrozenAirborneState", bases=[], keywords=[],
                             body=copy.deepcopy(methods), decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
    scope = {"torch": torch}
    exec(compile(module, str(FROZEN / "airborne.py"), "exec"), scope)
    return scope["FrozenAirborneState"]


def observation(n=1, *, body=.34, heights=(.18, .18), wheel_force=(0., 0.),
                body_force=(0., 0., 0.), allowed=True, stair=False):
    def rows(values):
        value = torch.tensor(values, dtype=torch.float32)
        return value.expand(n, *value.shape).clone()

    wheel = torch.zeros(n, 2, 3)
    wheel[:, :, 0] = rows(wheel_force)
    bodies = torch.zeros(n, len(body_force), 3)
    bodies[:, :, 1] = rows(body_force)
    return AirbornePhysicalObservation(
        body_relative_ground_height_m=torch.full((n,), body, dtype=torch.float32),
        wheel_relative_ground_heights_m=rows(heights),
        wheel_contact_forces_world_n=wheel, exit_body_contact_forces_world_n=bodies,
        commands3=rows([1.5, -.2, .3]),
        terrain_allowed=torch.full((n,), allowed, dtype=torch.bool),
        stair_active=torch.full((n,), stair, dtype=torch.bool))


class StateOracle:
    def __init__(self, n=1):
        self.actual = SourceAirborneState(n, "cpu")
        self.machine = frozen_state_class()()
        initial = {}
        for key, native in FIELDS.items():
            dtype = (torch.bool if key in ("airborne", "pending_force_enter")
                     else torch.long if key == "last_update_step" else torch.float32)
            shape = ((n, 2) if key == "wheel_contact_elapsed_s"
                     else (n, 3) if key == "entry_command3" else (n,))
            initial[native] = torch.full(shape, -1 if key == "last_update_step" else 0, dtype=dtype)
        # Independent native initial buffers: matching a source mutation cannot
        # mutate actual, and candidate initial values cannot define the oracle.
        self.env = NS(num_envs=n, device="cpu", step_dt=.02, common_step_counter=0,
            cfg=NS(airborne_state_machine_cfg=copy.deepcopy(frozen_configuration())),
            **initial)
        # Named allowance exercises the source's general entry gate. The final
        # pinned config has empty names and is unrestricted; no terrain mapper
        # implementation is being supplied or tested here.
        self.env.cfg.airborne_state_machine_cfg["allowed_terrain_names"] = ("fixture_allowed",)
        self.sampled_events = []
        self.machine._sample_command_override_on_enter = (
            lambda env, cfg, event: self.sampled_events.append(event.clone()))

    def snapshot(self):
        return {key: getattr(self.env, native).clone() for key, native in FIELDS.items()}

    def assert_equal(self, actual=None):
        actual = self.actual.snapshot() if actual is None else actual
        expected = self.snapshot()
        assert set(actual) == set(expected)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0., atol=0., equal_nan=True)

    def request(self, mask):
        mask = torch.as_tensor(mask, dtype=torch.bool)
        self.actual.request_force_enter(mask)
        self.env.airborne_force_enter_request |= mask
        self.assert_equal()

    def reset(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long)
        self.actual.reset(ids)
        self.machine.on_reset(self.env, ids)
        self.assert_equal()

    def observe(self, value, tick, *, via_reward=False):
        self.env.common_step_counter = tick
        self.env.ground_z_est = torch.zeros(self.env.num_envs)
        self.env.robot = NS(data=NS(root_pos_w=torch.zeros(self.env.num_envs, 3)))
        self.env.robot.data.root_pos_w[:, 2] = value.body_relative_ground_height_m
        self.env.command = value.commands3.clone()
        self.env.wheel_forward_stair_state = value.stair_active.clone()
        self.env.get_terrain_name_mask = lambda names: (
            value.terrain_allowed.clone() if names else torch.ones(self.env.num_envs, dtype=torch.bool))
        self.env._get_wheel_relative_ground_heights_raw = (
            lambda: value.wheel_relative_ground_heights_m.clone())
        self.env.contact_sensor = NS(data=NS(net_forces_w=torch.cat((
            value.wheel_contact_forces_world_n, value.exit_body_contact_forces_world_n), dim=1)))
        self.env._desired_contact_link_idx = [0, 1]
        self.env._undesired_contact_link_idx = list(range(2, self.env.contact_sensor.data.net_forces_w.shape[1]))
        before = self.env.height_reward_airborne_state.clone()
        updated = self.env.airborne_state_last_update_step != tick
        count = len(self.sampled_events)
        rng = torch.random.get_rng_state().clone()
        if via_reward:
            reference = torch.arange(self.env.num_envs, dtype=torch.float32) + .23
            returned = self.machine.get_height_reward_reference_height(
                self.env, reference, torch.zeros(self.env.num_envs, 2))
            assert torch.equal(returned, reference)
        else:
            self.machine.on_observation_step(self.env)
        expected_entry = (self.sampled_events[-1] if len(self.sampled_events) > count
                          else torch.zeros_like(updated))
        result = self.actual.observe(value, policy_tick=tick)
        assert torch.equal(result.updated, updated)
        assert torch.equal(result.entered, expected_entry)
        assert torch.equal(result.exited, updated & before & ~self.env.height_reward_airborne_state)
        torch.testing.assert_close(self.env.command, value.commands3, rtol=0., atol=0., equal_nan=True)
        assert torch.equal(torch.random.get_rng_state(), rng)
        self.assert_equal(result.state)
        return result


def test_native_constants_and_float32_owner_follow_frozen_configuration():
    cfg = frozen_configuration()
    assert dict(ENTER) == cfg["enter"]
    assert dict(EXIT) == cfg["exit"]
    assert POLICY_DT_S == .02
    oracle = StateOracle(2)
    for key, value in oracle.actual.snapshot().items():
        expected_dtype = (torch.bool if key in ("airborne", "pending_force_enter")
                          else torch.long if key == "last_update_step" else torch.float32)
        assert value.dtype == expected_dtype
        assert torch.equal(value, torch.full_like(value, -1 if key == "last_update_step" else 0))
    oracle.assert_equal()


def test_cpu_device_alias_uses_the_allocated_tensor_device():
    owner = SourceAirborneState(1, "cpu:0")
    assert owner.device == owner.active.device == torch.device("cpu")
    owner.request_force_enter(torch.tensor([True]))
    result = owner.observe(observation(body=.1), policy_tick=0)
    assert result.entered.item()


def test_disabled_target_getter_does_not_advance_state_or_consume_requests():
    oracle = StateOracle(2)
    oracle.request([True, False])
    before = oracle.snapshot()
    oracle.env.common_step_counter = 12
    target = torch.tensor([.2, .3])
    returned = oracle.machine.get_height_reward_target_height(oracle.env, target)
    assert returned is target
    for key, value in before.items():
        assert torch.equal(oracle.snapshot()[key], value)
    oracle.assert_equal()
    result = oracle.observe(observation(2, body=.1), 12, via_reward=True)
    assert torch.equal(result.entered, torch.tensor([True, False]))


@pytest.mark.parametrize("kwargs,entered", [
    ({}, True), ({"body": .3}, False), ({"body": .300001}, True),
    ({"heights": (.14, .18)}, False), ({"heights": (.140001, .18)}, True),
    ({"allowed": False}, False), ({"stair": True}, False), ({"body": float("nan")}, False),
    ({"body": .7, "heights": (.145, .145), "wheel_force": (30., 30.),
      "body_force": (0., 0., 6.)}, True),
])
def test_natural_entry_strict_height_and_allowance_gates(kwargs, entered):
    oracle = StateOracle()
    result = oracle.observe(observation(**kwargs), 1)
    assert result.entered.item() is entered
    assert result.state["airborne"].item() is entered
    assert torch.count_nonzero(result.state["wheel_contact_elapsed_s"]) == 0
    assert torch.count_nonzero(result.state["base_contact_elapsed_s"]) == 0
    if entered:
        assert result.state["current_duration_s"].item() == torch.tensor(.02).item()
        assert torch.equal(result.state["entry_command3"], observation().commands3)


@pytest.mark.parametrize("allowed,stair,entered", [(True, False, True),
    (False, False, False), (True, True, False)])
def test_forced_entry_bypasses_height_but_consumes_denied_requests(allowed, stair, entered):
    oracle = StateOracle()
    oracle.request([True])
    value = observation(body=.1, heights=(.06, .06), allowed=allowed, stair=stair)
    result = oracle.observe(value, 0)
    assert result.entered.item() is entered
    assert not result.state["pending_force_enter"].any()
    oracle.observe(value, 1)
    assert not oracle.actual.force_enter_request.any()


def test_vector_norm_strict_contact_thresholds_and_low_force_timer_history():
    oracle = StateOracle()
    oracle.request([True])
    oracle.observe(observation(body=.1, heights=(.06, .06)), 1)
    value = observation(body=.1, heights=(.1, .1), wheel_force=(21., 21.), body_force=(0., 0., 6.))
    first = oracle.observe(value, 2)
    assert torch.equal(first.state["wheel_contact_elapsed_s"], torch.full((1, 2), .02))
    assert torch.equal(first.state["base_contact_elapsed_s"], torch.tensor([.02]))
    # Norm exactly20/5 is not stable. Low wheel height freezes, as does lost base force.
    wheel = torch.tensor([[[12., 16., 0.], [0., -20., 0.]]])
    bodies = torch.tensor([[[0., 0., 0.], [3., 4., 0.], [0., 0., 0.]]])
    frozen = oracle.observe(replace(value, wheel_contact_forces_world_n=wheel,
                                   exit_body_contact_forces_world_n=bodies), 3)
    assert torch.equal(frozen.state["wheel_contact_elapsed_s"], first.state["wheel_contact_elapsed_s"])
    assert torch.equal(frozen.state["base_contact_elapsed_s"], first.state["base_contact_elapsed_s"])
    cleared = oracle.observe(replace(value, wheel_relative_ground_heights_m=torch.tensor([[.15, .1]]),
                                     wheel_contact_forces_world_n=torch.tensor([[[30., 0., 0.], [0., 0., 0.]]]),
                                     exit_body_contact_forces_world_n=torch.zeros(1, 3, 3)), 4)
    assert torch.equal(cleared.state["wheel_contact_elapsed_s"], torch.tensor([[0., .02]]))
    assert torch.equal(cleared.state["base_contact_elapsed_s"], torch.tensor([.02]))
    # A diagonal vector's norm exceeds20 despite every coordinate being below20.
    resumed = oracle.observe(replace(value, wheel_contact_forces_world_n=torch.tensor([
        [[15., 15., 0.], [0., 0., 0.]]]),
        exit_body_contact_forces_world_n=torch.zeros(1, 3, 3)), 5)
    assert torch.equal(resumed.state["wheel_contact_elapsed_s"], torch.tensor([[.02, .02]]))


@pytest.mark.parametrize("kind", ["wheel", "body", "duration", "stair"])
def test_exit_thresholds_preserve_contact_timers_and_episode_peak(kind):
    oracle = StateOracle()
    oracle.request([True])
    oracle.observe(observation(body=.1, heights=(.06, .06)), 0)
    value = observation(body=.1, heights=(.1, .1))
    if kind == "wheel":
        value = replace(value, wheel_contact_forces_world_n=torch.tensor([[[21., 0., 0.], [0., 0., 0.]]]))
    elif kind == "body":
        value = replace(value, exit_body_contact_forces_world_n=torch.tensor([
            [[0., 0., 0.], [0., 0., 0.], [0., 0., 6.]]]))
    elif kind == "stair":
        value = replace(value, stair_active=torch.tensor([True]))
    last_active = oracle.actual.snapshot()
    for tick in range(1, 60):
        result = oracle.observe(value, tick)
        if result.exited.item():
            break
        last_active = result.state
    else:
        pytest.fail("Frozen-source exit never occurred")
    assert not result.state["airborne"].any()
    assert result.state["current_duration_s"].item() == 0.
    assert torch.equal(result.state["maximum_duration_s"], last_active["maximum_duration_s"])
    if kind == "wheel":
        assert result.state["wheel_contact_elapsed_s"][0, 0] >= .3
        assert result.state["wheel_contact_elapsed_s"][0, 1] == 0.
        assert last_active["wheel_contact_elapsed_s"][0, 0] < .3
    elif kind == "body":
        assert result.state["base_contact_elapsed_s"].item() >= .25
        assert last_active["base_contact_elapsed_s"].item() < .25
    elif kind == "duration":
        assert (last_active["current_duration_s"] + .02).item() >= 1.
        assert last_active["current_duration_s"].item() < 1.
    grounded = oracle.observe(observation(body=.1, heights=(.06, .06)), tick + 1)
    for key in ("wheel_contact_elapsed_s", "base_contact_elapsed_s", "maximum_duration_s"):
        assert torch.equal(grounded.state[key], result.state[key])
    entered = oracle.observe(observation(), tick + 2)
    assert entered.entered.item()
    assert not entered.state["wheel_contact_elapsed_s"].any()
    assert not entered.state["base_contact_elapsed_s"].any()
    assert torch.equal(entered.state["maximum_duration_s"], result.state["maximum_duration_s"])


def test_reward_then_observation_uses_once_per_lane_clock_with_height_override_disabled():
    oracle = StateOracle(2)
    value = observation(2)
    first = oracle.observe(value, 7, via_reward=True)
    again = oracle.observe(replace(value, stair_active=torch.ones(2, dtype=torch.bool)), 7)
    assert not again.updated.any()
    assert not again.entered.any() and not again.exited.any()
    for key in FIELDS:
        assert torch.equal(first.state[key], again.state[key])
    # A late request after the same tick's update is retained until a new update.
    oracle.request([False, True])
    oracle.observe(value, 7)
    assert oracle.actual.force_enter_request[1]
    oracle.observe(value, 8)
    assert not oracle.actual.force_enter_request.any()


def test_partial_reset_can_reenter_same_global_tick_and_retains_source_entry_cache():
    oracle = StateOracle(2)
    first = oracle.observe(observation(2), 4)
    oracle.request([True, True])
    oracle.reset([0])
    assert torch.equal(oracle.actual.entry_command3, first.state["entry_command3"])
    assert torch.equal(oracle.actual.last_update_step, torch.tensor([-1, 4]))
    assert torch.equal(oracle.actual.force_enter_request, torch.tensor([False, True]))
    # Grounded reset lane clears its stale entry command; the other lane is untouched.
    grounded = oracle.observe(observation(2, body=.1, heights=(.06, .06)), 4)
    assert torch.equal(grounded.updated, torch.tensor([True, False]))
    assert torch.equal(grounded.state["entry_command3"][0], torch.zeros(3))
    assert torch.equal(grounded.state["entry_command3"][1], first.state["entry_command3"][1])
    oracle.reset([0])
    value = observation(2)
    value.commands3[0] = torch.tensor([-1.5, .2, -.3])
    reentered = oracle.observe(value, 4)
    assert torch.equal(reentered.entered, torch.tensor([True, False]))
    assert torch.equal(reentered.state["entry_command3"][0], value.commands3[0])
    assert reentered.state["current_duration_s"][0].item() == torch.tensor(.02).item()
    assert reentered.state["current_duration_s"][1].item() == first.state["current_duration_s"][1].item()


def test_counter_gaps_do_not_backfill_contact_or_flight_duration():
    oracle = StateOracle()
    oracle.request([True])
    value = observation(body=.1, heights=(.1, .1), wheel_force=(21., 0.), body_force=(0., 6., 0.))
    oracle.observe(value, 1)
    jumped = oracle.observe(value, 10000)
    assert torch.equal(jumped.state["wheel_contact_elapsed_s"], torch.tensor([[.02, 0.]]))
    assert torch.equal(jumped.state["base_contact_elapsed_s"], torch.tensor([.02]))
    assert torch.equal(jumped.state["current_duration_s"], torch.tensor([.04]))


def test_terrain_gate_blocks_entry_but_does_not_exit_existing_airborne_state():
    oracle = StateOracle()
    oracle.observe(observation(), 0)
    oracle.request([True])
    result = oracle.observe(observation(body=.1, heights=(.18, .18), allowed=False), 1)
    assert result.state["airborne"].item()
    assert not result.entered.any() and not result.exited.any()
    assert not result.state["pending_force_enter"].any()


@pytest.mark.parametrize("field", ["height_nan", "force_nan", "force_inf", "command_nan"])
def test_nonfinite_physical_comparisons_match_source_without_repairs(field):
    oracle = StateOracle()
    oracle.request([True])
    oracle.observe(observation(body=.1, heights=(.1, .1), wheel_force=(30., 0.)), 0)
    oracle.observe(observation(body=.1, heights=(.1, .1), wheel_force=(30., 0.)), 1)
    value = observation(body=.1, heights=(.1, .1), wheel_force=(30., 0.))
    if field == "height_nan":
        value.wheel_relative_ground_heights_m[0, 0] = float("nan")
    elif field == "force_nan":
        value.wheel_contact_forces_world_n[0, 0, 0] = float("nan")
    elif field == "force_inf":
        value.exit_body_contact_forces_world_n[0, 2, 0] = float("inf")
    else:
        oracle.reset([0])
        oracle.request([True])
        value.commands3[0, 0] = float("nan")
    oracle.observe(value, 2)


def test_updated_snapshots_feed_final_31_payment_and_do_not_alias_future_state():
    oracle = StateOracle()
    base_frame = reward_frame()
    entry = oracle.observe(observation(heights=(.145, .145), wheel_force=(30., 30.)), 1)

    def pay(state):
        return assert_source_oracle(replace(base_frame, airborne=state["airborne"],
            wheel_contact_elapsed_s=state["wheel_contact_elapsed_s"],
            base_contact_elapsed_s=state["base_contact_elapsed_s"]))

    first = pay(entry.state)
    assert first.payments["airborne_air_wheel_zero_torque_exp"].item() > 0.
    assert first.payments["airborne_precontact_wheel_directional_speed_shortfall"].item() == 0.
    contact = oracle.observe(observation(body=.2, heights=(.1, .1), wheel_force=(30., 0.)), 2)
    second = pay(contact.state)
    assert second.payments["airborne_air_wheel_zero_torque_exp"].item() == 0.
    assert second.payments["airborne_precontact_wheel_directional_speed_shortfall"].item() < 0.
    frozen = oracle.observe(observation(body=.2, heights=(.1, .1)), 3)
    third = pay(frozen.state)
    assert torch.equal(third.payments["airborne_precontact_wheel_directional_speed_shortfall"],
                       second.payments["airborne_precontact_wheel_directional_speed_shortfall"])
    cleared = oracle.observe(observation(body=.2, heights=(.15, .1)), 4)
    assert pay(cleared.state).payments["airborne_air_wheel_zero_torque_exp"].item() > 0.
    exited = oracle.observe(observation(body=.2, heights=(.1, .1), stair=True), 5)
    assert not exited.state["airborne"].any()
    after = pay(exited.state)
    assert after.payments["airborne_air_wheel_zero_torque_exp"].item() == 0.
    assert first.total.numel() == after.total.numel() == 1
    # Subsequent reset/update must not alter the entry transition already paid.
    oracle.reset([0])
    assert entry.state["airborne"].item()
    assert not entry.state["wheel_contact_elapsed_s"].any()
    assert torch.equal(pay(entry.state).total, first.total)
