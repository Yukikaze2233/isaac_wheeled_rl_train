"""Coaxial root inputs must co-rotate when the physical knee angle is held."""
import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("motor_coupling_audit", ROOT / "scripts/audit_v5_motor_coupling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MotorCouplingAudit().run()


def test_fixed_inner_angle_requires_equal_output_axis_increments_across_domain(audit):
    assert audit["status"] == "passed"
    assert sum(row["poses_tested"] for row in audit["results"]) == 70
    for row in audit["results"]:
        assert row["max_closure_gap_m"] < 1e-6
        assert row["max_aux_minus_hip_increment_error_rad"] < math.radians(.001)


@pytest.mark.parametrize("side", ["L", "R"])
def test_common_motion_preserves_knee_but_hip_only_motion_changes_it(audit, side):
    examples = {row["case"]: row for row in audit["examples"] if row["side"] == side}
    initial = examples["nominal"]["inner_angle_deg"]
    assert examples["both_plus_10deg"]["inner_angle_deg"] == pytest.approx(initial, abs=1e-3)
    assert abs(examples["hip_only_plus_10deg"]["inner_angle_deg"] - initial) > 1.


@pytest.mark.parametrize("side,auxiliary", [("L", "LL_joint1"), ("R", "RR_joint1")])
def test_held_thigh_knee_motion_rotates_only_auxiliary_motor_output(audit, side, auxiliary):
    examples = {row["case"]: row for row in audit["examples"] if row["side"] == side}
    nominal = examples["nominal"]
    moved = examples["hip_held_inner_plus_10deg"]
    assert moved["hip_rad"] == nominal["hip_rad"]
    assert moved["inner_angle_deg"] == pytest.approx(nominal["inner_angle_deg"] + 10., abs=1e-8)
    assert abs(moved["active_increments_rad"][auxiliary]) > .1
    for name, change in moved["active_increments_rad"].items():
        if name != auxiliary:
            assert change == pytest.approx(0., abs=1e-8)
    assert examples["auxiliary_only_plus_10deg"]["hip_rad"] == nominal["hip_rad"]
    assert abs(examples["auxiliary_only_plus_10deg"]["inner_angle_deg"] - nominal["inner_angle_deg"]) > 1.
