"""Telemetry optimization must retain every physics sample and group statistic."""
import pytest
import torch

from wheeled_tasks.chassis.torque_monitor import TorqueMonitor
from wheeled_tasks.chassis.v5_control import V5Control


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_deferred_group_reduction_matches_raw_physics_samples(device):
    groups = ["rotate", "stand", "rotate", "rotate", "stand"]
    generator = torch.Generator().manual_seed(42)
    torque = torch.randn(7, 5, 6, generator=generator).to(device)
    velocity = torch.randn(7, 5, 6, generator=generator).to(device)
    force = torch.randn(7, 5, 2, generator=generator).to(device)
    speed = torch.randn(7, 5, 2, generator=generator).to(device)
    compression = torch.rand(7, 5, 2, generator=generator).to(device) * .08
    bounds = torch.full((5, 6), 1.2, device=device)
    monitor = TorqueMonitor(groups, V5Control.ACTIVE, device, 3.8)
    for step in range(7):
        monitor.observe(torque[step], velocity[step], force[step], speed[step], compression[step],
                        torque[step], bounds)
        if step == 2:
            assert monitor.report()["groups"]["rotate"]["physics_samples"] == 9
    report = monitor.report()
    for name in set(groups):
        ids = [i for i, group in enumerate(groups) if group == name]
        group = report["groups"][name]
        assert group["physics_samples"] == 7 * len(ids)
        raw_torque = torque[:, ids].double().flatten(0, 1)
        raw_velocity = velocity[:, ids].double().flatten(0, 1)
        assert group["rms_motor_torque_nm"] == pytest.approx(raw_torque.square().mean(0).sqrt().tolist())
        assert group["peak_motor_torque_nm"] == pytest.approx(raw_torque.abs().amax(0).tolist())
        assert group["saturation_fraction_95pct"] == pytest.approx((raw_torque.abs() >= .95 * 1.2).double().mean(0).tolist())
        power = raw_torque * raw_velocity
        assert group["mean_positive_mechanical_power_w"] == pytest.approx(power.clamp_min(0).mean(0).tolist())
        assert group["mean_negative_mechanical_power_w"] == pytest.approx((-power).clamp_min(0).mean(0).tolist())
        assert group["gas_compression_min_m"] == pytest.approx(compression[:, ids].flatten(0, 1).amin(0).tolist())
        assert group["gas_compression_max_m"] == pytest.approx(compression[:, ids].flatten(0, 1).amax(0).tolist())
