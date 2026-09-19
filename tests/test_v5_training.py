"""V5 control-coordinate and observation/force contract checks without Isaac."""
import json
import importlib.util
from pathlib import Path
import tarfile

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


def test_mechanically_coupled_margins_are_not_double_counted(controller):
    knees = controller.knee_bounds.mean(-1)[None]
    springs = controller.s0 - controller.stroke * .95
    risk = controller.working_margin_risk(knees, springs[None], .08)
    torch.testing.assert_close(risk, torch.full_like(risk, .5), atol=1e-6, rtol=0)
    stops = controller.knee_bounds[:, 0][None]
    risk = controller.working_margin_risk(stops, (controller.s0 - controller.stroke)[None], .08)
    torch.testing.assert_close(risk, torch.ones_like(risk), atol=1e-6, rtol=0)


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


def test_parallel_scene_coverage_and_torque_units():
    from wheeled_tasks.chassis.task import choose_scene_groups
    from wheeled_tasks.chassis.torque_monitor import TorqueMonitor
    contract = json.loads((ROOT / "contracts/v5_mixed_v1.json").read_text())
    scenes = choose_scene_groups(contract["scene_groups"], 1024)
    assert len(scenes) == 1024
    assert {g for g, _ in scenes} == {"stand", "translate", "rotate", "step_up", "step_down"}
    assert {t for _, t in scenes} >= {"flat", "slope", "platform", "step_up", "stairs", "jump", "step_down"}
    assert sum(t == "flat" for _, t in scenes) >= 409
    monitor = TorqueMonitor(["stand", "rotate"], V5Control.ACTIVE, "cpu", 3.8)
    torque = torch.tensor([[10., -20., 1., 10., -20., 1.], [40., 0., 2., 0., 0., 2.]])
    bounds = torch.tensor([[40., 40., 3.8, 40., 40., 3.8], [40., 40., 2., 40., 40., 2.]])
    monitor.observe(torque, torch.ones_like(torque) * 2, torch.ones(2, 2) * 350,
                    torch.ones(2, 2) * .1, torch.ones(2, 2) * .05, torque, bounds)
    report = monitor.report()
    assert report["groups"]["stand"]["rms_motor_torque_nm"] == [10., 20., 1., 10., 20., 1.]
    assert report["groups"]["rotate"]["saturation_fraction_95pct"] == [1., 0., 1., 0., 0., 1.]
    assert report["groups"]["stand"]["peak_gas_force_n"] == [350., 350.]
    assert report["groups"]["stand"]["mean_negative_mechanical_power_w"][1] == 40.


def test_torque_statistics_do_not_lose_small_increments_late_in_training():
    from wheeled_tasks.chassis.torque_monitor import TorqueMonitor
    count = 204
    monitor = TorqueMonitor(["rotate"] * count, V5Control.ACTIVE, "cpu", 3.8)
    initial = 2**29
    monitor.samples.fill_(initial)
    monitor.square.fill_(initial)
    monitor.positive_power.fill_(initial)
    monitor.negative_power.fill_(initial)
    torque = torch.tensor([3., -3., 3., -3., 3., -3.]).repeat(count, 1)
    monitor.observe(torque, torch.full_like(torque, 2.), torch.full((count, 2), 350.),
                    torch.zeros(count, 2), torch.full((count, 2), .05), torque, torch.full_like(torque, 40.))
    assert int(monitor.samples[0]) - initial == count
    assert float(monitor.square[0, 0]) - initial == count * 9
    assert float(monitor.positive_power[0, 0]) - initial == count * 6
    assert float(monitor.negative_power[0, 1]) - initial == count * 6


def test_remote_finalization_keeps_checkpoint_and_training_status(tmp_path):
    spec = importlib.util.spec_from_file_location("chassis_job", ROOT / "scripts/chassis_remote_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train = tmp_path / "train"
    train.mkdir()
    (train / "completion.json").write_text(json.dumps({"status": "stopped", "successful_updates": 17,
                                                      "export": {"verified": True}}))
    (train / "model_final.pt").write_bytes(b"test checkpoint bytes")
    (train / "model_0.pt").write_bytes(b"test periodic checkpoint")
    (train / "events.out.tfevents.test").write_bytes(b"test event bytes")
    result = module.finalize(tmp_path, 0)
    assert result["training_status"] == "stopped" and result["export_verified"]
    assert result["successful_updates"] == 17
    with tarfile.open(tmp_path / "delivery.tar.gz") as archive:
        assert set(archive.getnames()) == set(result["files"])
        for name in result["files"]:
            import hashlib
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == result["files"][name]["sha256"]
