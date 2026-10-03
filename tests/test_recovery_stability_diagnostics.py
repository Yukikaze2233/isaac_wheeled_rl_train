"""Sensor dwell diagnostics and opt-in blend timing; no physical acceptance."""
import ast
import inspect
import math
import textwrap

import pytest
import torch

from test_recovery_fsm import NOMINAL, make, update, until_phase
from wheeled_tasks.chassis.recovery import RecoveryFSM, RecoveryPhase as P


GATES = ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed")


@pytest.mark.parametrize("gate", GATES)
def test_each_failed_gate_is_visible_and_breaks_only_its_lane_dwell(gate):
    fsm = make(2)
    fsm.phase[:] = P.RL
    for _ in range(7):
        update(fsm)
    kwargs = {}
    if gate == "valid":
        q = NOMINAL.repeat(2, 1)
        q[0, 0] = float("nan")
        kwargs["q"] = q
    elif gate in ("support", "body_clear"):
        kwargs[gate] = torch.tensor([False, True])
    elif gate == "tilt":
        kwargs["pitch"] = torch.tensor([math.radians(9.), 0.])
    elif gate == "gyro":
        kwargs["gyro"] = torch.tensor([[.8, 0., 0.], [0., 0., 0.]])
    elif gate == "height":
        kwargs["height"] = torch.tensor([.37, .305])
    else:
        dq = torch.zeros(2, 6)
        dq[0, 2 if gate == "wheel_speed" else 0] = 5. if gate == "wheel_speed" else 2.
        kwargs["dq"] = dq
    update(fsm, **kwargs)
    diag = fsm.diagnostics()
    for name in GATES:
        assert diag[f"recovery_stable_{name}_pass"].tolist() == [name != gate, True]
    assert diag["recovery_stable_pass"].tolist() == [False, True]
    assert fsm.stable_ticks.tolist() == [0, 8]
    assert fsm.stable_peak_ticks.tolist() == [7, 8]
    assert diag["recovery_stable_break_count"].tolist() == [1, 0]
    for _ in range(3):
        update(fsm, **kwargs)
    assert fsm.stable_break_count.tolist() == [1, 0]


def test_masks_preserve_legacy_predicate_at_boundaries():
    n = 18
    fsm = make(n)
    fsm.phase[:] = P.RL
    pitch = torch.linspace(0., .2, n)
    gravity = torch.stack((pitch.sin(), torch.zeros(n), -pitch.cos()), -1)
    gyro = torch.zeros(n, 3)
    gyro[:, 0] = torch.linspace(0., .9, n)
    gyro[0, 0] = fsm.config["handover_gyro_max"]
    height = torch.full((n,), .305)
    height[1] = fsm.config["handover_height_min"]
    height[2] = fsm.config["handover_height_max"]
    dq = torch.zeros(n, 6)
    dq[3, 0] = 2.
    dq[4, 2] = .25 / .06
    support = torch.ones(n, dtype=torch.bool)
    support[5] = False
    clear = torch.ones_like(support)
    clear[6] = False
    q = NOMINAL.repeat(n, 1)
    q[7, 0] = float("nan")
    valid = (torch.isfinite(q).all(-1) & torch.isfinite(dq).all(-1)
             & torch.isfinite(gravity).all(-1) & torch.isfinite(gyro).all(-1)
             & torch.isfinite(height) & (gravity.norm(dim=-1) > .5))
    normalized = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    tilt = torch.acos((-normalized[:, 2]).clamp(-1., 1.))
    expected = (valid & support & clear & (tilt < fsm.config["handover_tilt_rad"])
                & (gyro.norm(dim=-1) < fsm.config["handover_gyro_max"])
                & (height > fsm.config["handover_height_min"])
                & (height < fsm.config["handover_height_max"])
                & (dq[:, fsm.WHEEL_IDS].abs().amax(-1) * .06 < .25)
                & (dq[:, fsm.LEG_IDS].abs().amax(-1) < 2.))
    update(fsm, q=q, dq=dq, gravity=gravity, gyro=gyro, height=height,
           support=support, body_clear=clear)
    assert torch.equal(fsm.diagnostics()["recovery_stable_pass"], expected)


def test_predicate_pass_does_not_imply_rl_owns_the_clock():
    fsm = make(4, active=torch.tensor([True, True, True, False]))
    fsm.phase[:] = torch.tensor([P.PREPARE, P.BLEND, P.RL, P.RL])
    update(fsm)
    diag = fsm.diagnostics()
    assert diag["recovery_stable_pass"].all()
    assert diag["recovery_ready"].tolist() == [True, False, False, False]
    assert diag["recovery_holding_rl"].tolist() == [False, False, True, False]
    assert fsm.stable_ticks.tolist() == [0, 0, 1, 0]


def test_peak_and_break_counts_survive_release_and_reset_only_selected_lanes():
    fsm = make(3)
    fsm.phase[:] = P.RL
    for _ in range(12):
        update(fsm)
    update(fsm, support=False)
    for _ in range(200):
        update(fsm)
    assert fsm.motion_released.all()
    update(fsm)
    before = fsm.diagnostics()
    assert fsm.stable_ticks.eq(0).all()
    assert fsm.stable_peak_ticks.tolist() == [200, 200, 200]
    assert fsm.stable_break_count.tolist() == [1, 1, 1]
    fsm.reset([1], NOMINAL.repeat(3, 1), True)
    after = fsm.diagnostics()
    for key in before:
        torch.testing.assert_close(after[key][[0, 2]], before[key][[0, 2]], rtol=0., atol=0.)
    for key in ("recovery_stable_peak_seconds", "recovery_stable_break_count",
                "recovery_stable_pass", "recovery_ready", "recovery_holding_rl",
                *(f"recovery_stable_{name}_pass" for name in GATES)):
        assert after[key][1] == 0
    assert before["recovery_stable_support_pass"][1]  # Snapshot must not alias live masks.


def test_diagnostics_have_fixed_lane_shape_and_do_not_change_rng():
    fsm = make(3)
    state = torch.random.get_rng_state().clone()
    for _ in range(5):
        update(fsm)
        diag = fsm.diagnostics()
        assert all(value.shape == (3,) and value.device.type == "cpu" for value in diag.values())
    assert torch.equal(torch.random.get_rng_state(), state)
    # Device reductions are allowed; Python extraction/transfer is not. This
    # guards additions to update/diagnostics without requiring a CUDA machine.
    for method in (RecoveryFSM.update, RecoveryFSM.diagnostics):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr in {"item", "cpu", "numpy", "tolist"}
                       for node in ast.walk(tree))


def test_loss_after_successful_release_is_not_counted_as_a_broken_dwell():
    fsm = make()
    fsm.phase[:] = P.RL
    for _ in range(200):
        update(fsm)
    assert fsm.motion_released.item()
    update(fsm, support=False)
    assert fsm.stable_ticks.item() == 0
    assert fsm.stable_peak_ticks.item() == 200
    assert fsm.stable_break_count.item() == 0


@pytest.mark.parametrize("value", [None, 0, 1, "true", .5])
def test_blend_completion_option_requires_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        make(config={"complete_started_blend": value})


def test_default_and_explicit_false_keep_legacy_blend_timeout():
    default = make(config={"max_script_seconds": .25})
    explicit = make(config={"max_script_seconds": .25, "complete_started_blend": False})
    for _ in range(50):
        update(default)
        update(explicit)
        for key, value in default.diagnostics().items():
            assert torch.equal(value, explicit.diagnostics()[key]), key
        assert torch.equal(default.script_targets, explicit.script_targets)
    assert default.failure_code.item() == 6 and default.age_ticks.item() == 50


def test_opt_in_allows_started_blend_to_finish_full_200ms():
    fsm = make(config={"max_script_seconds": .25, "complete_started_blend": True})
    until_phase(fsm, P.BLEND)
    start = fsm.age_ticks.item()
    for tick in range(1, 40):
        update(fsm)
        assert fsm.blending.item() and not fsm.failed.item()
        assert fsm.blend.item() == pytest.approx(tick / 40, abs=1e-6)
    update(fsm)
    assert fsm.pure_rl.item() and not fsm.motion_released.item()
    assert fsm.age_ticks.item() == start + 40 > fsm._ticks(.25)
    for _ in range(200):
        update(fsm)
    assert fsm.motion_released.item() and fsm.failure_code.item() == 0


def test_opt_in_does_not_extend_unfinished_script_or_relax_takeover_faults():
    blocked = make(config={"max_script_seconds": .25, "complete_started_blend": True})
    for _ in range(50):
        update(blocked, support=False)
    assert blocked.failed.item() and blocked.failure_code.item() == 6
    for kind, expected in (("tilt", 5), ("invalid", 1)):
        fsm = make(config={"max_script_seconds": .25, "complete_started_blend": True})
        until_phase(fsm, P.BLEND)
        if kind == "tilt":
            update(fsm, pitch=math.radians(60.))
        else:
            q = NOMINAL[None].clone()
            q[0, 0] = float("nan")
            update(fsm, q=q)
        assert fsm.failed.item() and fsm.failure_code.item() == expected


def test_opt_in_still_requires_support_to_enter_capture():
    fsm = make(config={"complete_started_blend": True})
    fsm.phase[:] = P.ORBIT
    update(fsm, pitch=math.radians(20.), support=False)
    assert fsm.phase.item() == P.ORBIT
    update(fsm, pitch=math.radians(20.), support=True)
    assert fsm.phase.item() == P.CAPTURE
