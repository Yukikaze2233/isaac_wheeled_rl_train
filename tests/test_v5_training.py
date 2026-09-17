"""V5 control-coordinate and observation/force contract checks without Isaac."""
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.v5_control import V5Control


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controller():
    read = lambda p: json.loads((ROOT / p).read_text())
    directory = "model/纯底盘_v5/urdf/"
    return V5Control(read(directory + "manifest.json"), read(directory + "model_spec.json"),
                     read(directory + "fit_10mpa.json"), read("contracts/own_v40_v2.json"),
                     read("contracts/v5_foundation_v1.json")["v5_control"], "cpu")


def test_true_active_cranks_not_passive_knees(controller):
    assert controller.ACTIVE[1] == "LL_joint1"
    assert controller.ACTIVE[4] == "RR_joint1"
    q = controller.nominal.repeat(2, 1)
    q[:, 1] += 4 * torch.pi
    legs, wheels, actions = controller.decode(torch.zeros_like(q), q)
    torch.testing.assert_close(legs, q[:, [0, 1, 3, 4]], atol=2e-6, rtol=0)
    assert torch.count_nonzero(wheels) == 0
    tau = controller.motor_efforts(q, torch.zeros_like(q), legs, wheels)
    assert tau.abs().max() < 1e-4


def test_gas_force_is_passive_positive_extension(controller):
    compression = torch.tensor([[0., .04], [.056, .072]])
    position = controller.s0 - compression
    forces = controller.spring_efforts(position)
    assert forces.shape == (2, 2)
    assert forces[0, 0] == pytest.approx(280.)
    assert forces[0, 1] == pytest.approx(317.4302, abs=1e-3)
    assert forces[1, 1] == pytest.approx(418.7601, abs=1e-3)
    s, ds = controller.spring_state(position, torch.ones_like(position) * .2)
    torch.testing.assert_close(s, compression)
    torch.testing.assert_close(ds, torch.full_like(ds, -.2))


def test_frame_and_control_limits(controller):
    q = controller.nominal[None]
    actions = torch.ones_like(q) * 100
    legs, wheels, clipped = controller.decode(actions, q)
    tau = controller.motor_efforts(q, torch.zeros_like(q), legs, wheels)
    assert clipped.max() == 3
    assert tau[:, [0, 1, 3, 4]].abs().max() <= 40
    assert tau[:, [2, 5]].abs().max() <= 3.838
    frame = controller.proprioception(torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]),
                                     torch.tensor([[0., 0., .32]]), q, torch.zeros_like(q), clipped)
    assert frame.shape == (1, 25) and torch.isfinite(frame).all()
    contract = json.loads((ROOT / "contracts/v5_foundation_v1.json").read_text())
    assert contract["actor_frame_dim"] == 25 + 4 + 5 + 4 + 2 + 5 + 1
    assert contract["actor_dim"] == 230 and contract["critic_dim"] == 46 + 3 + 1 + 2 + 36 + 4
    assert contract["enabled_stages"] == ["foundation"]
