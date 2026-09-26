"""Behavioral contracts for35D continuation, contact domains and USB command age."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.command_transport import UsbCommandTransport
from wheeled_tasks.chassis.contact_domain import ContactDomain
from wheeled_tasks.chassis.evaluation import capability_gate
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.policy_transfer import transfer_actor_state
from wheeled_tasks.chassis.references import CommandReference
from wheeled_tasks.chassis.scut_observation import build_manual35, build_scut35
from wheeled_tasks.chassis.task import Surface

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def configs():
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v6_scut35_continuation_v1.json"), load)
    return plan, [stage_contract(load(plan["base_contract"]), plan, r, 16384) for r in plan["stages"]]


def test_uniform_manifest_budget_and_control_contract(configs):
    plan, stages = configs
    assert [s["total_updates"] for s in stages] == [500, 750, 750, 500, 1000, 2500, 1000]
    assert sum(s["total_updates"] * s["target_num_envs"] * 24 for s in stages) == 1622016000
    for stage in stages:
        assert (stage["actor_dim"], stage["critic_dim"], stage["history_length"]) == (35, 81, 1)
        assert stage["physics_dt"] == .001 and stage["policy_dt"] == .02
        assert not stage["usb_transport"]["enabled"] and not stage["step_assist"]["enabled"]
        assert stage["evaluation"]["cases"] == stages[0]["evaluation"]["cases"]
        assert stage["evaluation"]["protocol_id"] == "v6-scut35-1khz-fixed-v1"
        assert stage["evaluation"]["cumulative_retention"]
        assert stage["command_transport"]["delay_ms"] == [.1, 5.]
    assert "stand__mu_010" in stages[0]["evaluation"]["retention_case_names"]


def transport(**kwargs):
    cfg = {"delay_ms": [.1, 5.], "can_period_ms": 1., "enabled_fraction": 1.,
           "drop_probability_max": 0., "burst_probability": 0., "burst_packets": [2, 3]}
    cfg.update(kwargs)
    return UsbCommandTransport(2, "cpu", .001, cfg, 617,
        [{"delay_ms": 2.}, {"delay_ms": 2.}])


def test_usb_delays_actual_torque_and_holds_on_packet_loss():
    link = transport()
    one, two = torch.ones(2, 6), torch.full((2, 6), 2.)
    assert not link.apply_torque(one).any()
    assert not link.apply_torque(two).any()
    torch.testing.assert_close(link.apply_torque(two), one)
    link.loss_probability[:] = 1.
    for _ in range(10):
        held = link.apply_torque(torch.full((2, 6), 99.))
    torch.testing.assert_close(held, two)
    assert link.age.min() > .005
    assert link.lost.min() == 10


def test_reset_discards_pending_torque_and_preserves_other_rows():
    link = transport()
    command = torch.full((2, 6), 8.)
    link.apply_torque(command)
    link.reset(torch.tensor([0]))
    link.loss_probability[0] = 1.
    link.apply_torque(command)
    output = link.apply_torque(command)
    assert not output[0].any()
    torch.testing.assert_close(output[1], command[1])


def test_fixed_burst_has_exact_packet_count_and_common_nominal_lane():
    link = transport()
    link.fixed_burst_every[:] = 10
    link.fixed_burst_length[:] = 3
    for tick in range(14):
        link.apply_torque(torch.full((2, 6), float(tick)))
    assert link.lost.tolist() == [3, 3]
    link.domain_draw = torch.tensor([.1, .9])
    link.profiles = None
    link.fraction = .5
    for _ in range(3):
        link.reset(link.rows)
        assert link.enabled.tolist() == [True, False]


def test_ordered_arrivals_do_not_reapply_old_commands():
    link = transport()
    outputs = []
    for tick, delay in enumerate((.005, .001, .003, .001, .005, .001, .001, .001)):
        link.delay[:] = delay
        outputs.append(float(link.apply_torque(torch.full((2, 6), float(tick + 1)))[0, 0]))
    assert outputs == sorted(outputs)
    assert max(outputs) > 0


def test_nominal_and_low_grip_material_mapping():
    settings = {"enabled_fraction": .5, "friction_range": [.1, 1.2], "low_grip_fraction": .5}
    domain = ContactDomain(2, settings, 617, [None, {"friction": .1}])
    surfaces = [Surface(-4, 0, friction=.5), Surface(0, 4, friction=.7)]
    assert domain.surfaces(surfaces, 0) == surfaces
    assert domain.surfaces(surfaces, 1)[0].friction == pytest.approx(.1)
    assert domain.surfaces(surfaces, 1)[1].friction == pytest.approx(.14)
    sampled = ContactDomain(4096, settings, 617)
    assert (sampled.mu == .5).sum() > 1500
    assert (sampled.mu < .4).sum() > 500


def test_manual35_preserves_legacy_normal_and_jump_and_codec(configs):
    _, stages = configs
    ref = CommandReference(3, "cpu", stages[0]["command_reference"])
    request = torch.tensor([False, True, False])
    apex = torch.full((3,), .06)
    ref.update(torch.full((3,), .305), torch.full((3,), 1.5), torch.tensor([0., 0., 1.]),
               torch.tensor([0., 0., .3]), torch.full((3,), 7.), request, torch.full((3,), .2), apex)
    commands = torch.tensor([[.5, 0., .305]]).repeat(3, 1)
    commands[2, 2] = ref.height[2]
    z6, z3 = torch.zeros(3, 6), torch.zeros(3, 3)
    gravity = torch.tensor([[0., 0., -1.]]).repeat(3, 1)
    args = (z3, gravity, commands, z6, z6, z6, torch.zeros(6), request, apex)
    frame = build_manual35(*args, ref)
    legacy = build_scut35(*args, ref.elapsed)
    torch.testing.assert_close(frame[:2], legacy[:2])
    assert frame.shape == (3, 35)
    assert frame[:, 28:33].sum(-1).tolist() == [1, 1, 1]
    assert frame[2, 29] == 1 and frame[2, 30] == 0
    spec = importlib.util.spec_from_file_location("io_reference", ROOT / "docs/examples/v5_policy_io.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoded = module.build_observation(np.zeros(6), np.zeros(6), np.zeros(3), np.array([0., 0., -1.]),
        commands[2].numpy(), np.zeros(6), np.zeros(6), control_mode="stair",
        context_height_m=float(commands[2, 2]), context_elapsed_s=.3)
    np.testing.assert_allclose(decoded[0], frame[2].numpy(), atol=1e-6)


def test_clock_transfer_keeps_every_actor_parameter(configs):
    _, stages = configs
    target = stages[0]
    source = {**target, "physics_dt": .005}
    weights = {"mlp.0.weight": torch.randn(256, 35), "distribution.std_param": torch.randn(6)}
    transferred, receipt = transfer_actor_state(weights, source, target)
    assert transferred is weights and receipt["physics_dt_new"] == .001
    bad = {**source, "physics_dt": .01}
    with pytest.raises(ValueError):
        transfer_actor_state(weights, bad, target)


def test_newly_mastered_jump_remains_protected_after_stage_change():
    baseline = {"cases": {n: {"passed": n == "stand", "checks": {"mechanics": True}} for n in ("stand", "jump", "step")},
                "rank_lower_is_better": [1, 0, 1]}
    settings = {"retention_case_names": ["stand", "jump", "step"], "promotion_case_names": ["step"],
                "protected_case_names": ["stand", "jump"], "initial_passed_case_names": ["stand"]}
    candidate = deepcopy(baseline)
    candidate["cases"]["step"]["passed"] = True
    gate = capability_gate(candidate, baseline, settings)
    assert not gate["passed"] and gate["lost_parent_passes"] == ["jump"]
