"""CPU checks of static candidates; these do not certify dynamic recovery."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recovery_static_targets", ROOT / "scripts/recovery_static_targets.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def solver():
    return MODULE.RecoveryStaticTargetBuilder()


@pytest.fixture(scope="module")
def targets(solver):
    return solver.build_targets()


def test_targets_are_closed_balanced_supported_and_inside_105_degree_limit(solver, targets):
    assert targets["asset_manifest_sha256"] == MODULE._PROFILES.DEFAULT_MANIFEST_SHA256
    assert targets["passive_knee_domain_deg"] == [40., 105.]
    assert targets["axis_signs"] == [1., 1., -1., -1.]
    assert not targets["gas_force_model_identified"]
    for key, row in targets["targets"].items():
        q = row["full_joint_positions"]
        assert set(q) == set(solver.names)
        assert row["height_m"] == pytest.approx(float(key), abs=1e-8)
        assert solver.validate_q(q)["max_closure_gap_m"] < 1e-8
        assert all(40. < beta < 105. for beta in solver.knees(q))
        rotation = Rotation.from_quat(row["root_quaternion_xyzw"]).as_matrix()
        center, wheels = solver.mass_and_wheel_centers(q, rotation)
        assert center[0] == pytest.approx(wheels[:, 0].mean(), abs=1e-8)
        bounds = solver.collision_bounds(q, rotation)
        assert min(part["lower"][2] + row["height_m"] for part in bounds) >= -1e-8
        for side in ("L", "R"):
            bottom = min(part["lower"][2] for part in bounds if part["body"] == side + "_link3")
            assert bottom + row["height_m"] == pytest.approx(0., abs=1e-7)
        assert not row["dynamic_validation_passed"]
        assert not row["pd_target_is_reset_pose"]


def test_pd_reference_produces_required_static_effort_without_moving_geometry(solver, targets):
    for row in targets["targets"].values():
        reference = row["prepare_reference"]
        geometry = np.asarray(reference["geometry_control6_rad"])
        command = np.asarray(reference["pd_control6_rad"])
        torque = np.asarray(reference["hold_torque_nm"])
        np.testing.assert_allclose(geometry, [row["full_joint_positions"][name] for name in solver.controls])
        np.testing.assert_allclose(reference["kp"] * (command - geometry), torque, atol=1e-8)
        assert np.max(np.abs(torque)) < 5.1
        assert command[[2, 5]].tolist() == geometry[[2, 5]].tolist()
        assert not np.allclose(command, geometry)
        assert row["static_load"]["mapping_residual"] < 1e-7
        np.testing.assert_allclose(sum(row["static_load"]["wheel_normals_n"]), solver.mass * 9.81)


def test_static_effort_matches_independent_constrained_potential_difference(solver, targets):
    # Solve complete nearby closed-chain states, rather than differentiating
    # independent tree coordinates as the implementation does.
    row = targets["targets"]["0.34"]
    q = row["full_joint_positions"]
    rotation = Rotation.from_quat(row["root_quaternion_xyzw"]).as_matrix()
    normals = np.asarray(row["static_load"]["wheel_normals_n"])
    spring_primitive = np.polynomial.polynomial.polyint(solver.fit["monomial_coefficients_n"])
    stroke = solver.fit["stroke_m"]

    def potential(configuration):
        frames = solver.fk(solver.spec, configuration)
        gravity = sum(body["mass"] * 9.81 *
                      (rotation @ (frames[body["name"]][:3, :3] @ body["com"]
                                   + frames[body["name"]][:3, 3]))[2]
                      for body in solver.spec["bodies"])
        wheel_z = np.array([(rotation @ solver.wheel(side, configuration)[0])[2] for side in ("L", "R")])
        compression = np.array([solver.spec["spring_binding"][name]["compression_at_q_zero_m"] - configuration[name]
                                for name in solver.manifest["spring_joint_names"]])
        gas = stroke * np.polynomial.polynomial.polyval(compression / stroke, spring_primitive).sum()
        return gravity - normals @ wheel_z + gas

    direction = np.array([.4, -.3, 0., -.2, .1, 0.])
    step = 1e-5
    states = [solver.solve({name: q[name] + sign * step * direction[i]
                            for i, name in enumerate(solver.controls)}, q)
              for sign in (-1., 1.)]
    measured = (potential(states[1]) - potential(states[0])) / (2 * step)
    expected = np.asarray(row["prepare_reference"]["hold_torque_nm"]) @ direction
    assert measured == pytest.approx(expected, abs=2e-5)


def test_invalid_requests_and_unidentified_gas_extrapolation_are_rejected(solver, targets, monkeypatch):
    for height, gain in ((float("nan"), 160.), (-.1, 160.), (.305, 0.), (.305, float("inf"))):
        with pytest.raises(ValueError):
            solver.target(height, gain)
    row = targets["targets"]["0.305"]
    q = row["full_joint_positions"]
    rotation = Rotation.from_quat(row["root_quaternion_xyzw"]).as_matrix()
    center, wheels = solver.mass_and_wheel_centers(q, rotation)
    # The geometry remains legal, but this deliberately narrower fit does not
    # cover its compression: static load generation must refuse extrapolation.
    monkeypatch.setitem(solver.fit, "normalized_fit_domain", [0., .1])
    with pytest.raises(ValueError, match="gas-curve extrapolation"):
        solver.static_load({"q": q, "rotation": rotation, "com": center, "wheels": wheels})
