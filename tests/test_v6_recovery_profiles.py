"""Geometry proof checks for reset candidates, not dynamic recovery success."""
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v6_recovery_profile_builder", ROOT / "scripts/build_v6_recovery_profiles.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def builder():
    return MODULE.RecoveryProfileBuilder()


@pytest.fixture(scope="module")
def frozen():
    return json.loads((ROOT / "contracts/v6_recovery_profiles_v1.json").read_text())


def test_identity_and_complete_flat_matrix(builder, frozen):
    assert frozen["asset_manifest_sha256"] == MODULE.DEFAULT_MANIFEST_SHA256
    assert frozen["generator_sha256"] == MODULE.digest(ROOT / "scripts/build_v6_recovery_profiles.py")
    assert set(MODULE.FLAT_POSES) <= frozen["profiles"].keys()
    assert len(MODULE.FLAT_POSES) == 16
    assert frozen["axis_signs"] == [1., 1., -1., -1.]
    assert frozen["tree_joint_names"] == builder.manifest["tree_joint_names"]
    assert not frozen["hardware_deployment_ready"]


def test_all_reset_states_have_closed_legal_full_configuration(builder, frozen):
    for name, row in frozen["profiles"].items():
        q = row["full_joint_positions"]
        assert set(q) == set(frozen["tree_joint_names"]), name
        validation = builder.validate_q(q)
        assert validation["max_closure_gap_m"] < 1e-8, name
        assert row["geometry_valid"] and row["enabled"]
        assert not row["dynamic_recovery_validated"]
        assert row["state_writes"] == "once_at_episode_reset_only"
        assert row["control6_rad"] == [q[j] for j in frozen["control_joint_names"]]
        np.testing.assert_allclose(np.linalg.norm(row["root_quaternion_xyzw"]), 1.)
        assert set(row["full_joint_velocities"].values()) == {0.}


def test_flat_and_step_hulls_do_not_penetrate_frozen_terrain(builder, frozen):
    # Test every convex vertex and riser crossing chord independently from the
    # profile's recorded minimum-clearance scalar.
    for name, profile in frozen["profiles"].items():
        rotation = Rotation.from_quat(profile["root_quaternion_xyzw"]).as_matrix()
        root = np.array(profile["root_position_m"])
        for row in builder.collision_bounds(profile["full_joint_positions"], rotation):
            points = row["points"] + root
            # A circumscribed wheel prism extends slightly beyond the exact
            # analytic floor contact; the 3 mm reset gap contains that bound.
            assert points[:, 2].min() > 0.0028, name
            for surface in profile["terrain_platforms"]:
                clipped = builder.halfspace_min_z(points, surface["start_x_m"])
                if clipped is not None:
                    assert clipped >= surface["height_m"] + .0029, name


def test_prone_shell_support_and_split_semantics(frozen):
    for name in ("prone_belly_legs_above", "prone_back_legs_above"):
        checks = frozen["profiles"][name]["geometry_validation"]
        assert checks["lowest_support_body"] == "base_link"
        assert min(checks["wheel_bottom_above_shell_top_m"]) > 0
    split = frozen["profiles"]["step200_wheels_split"]["geometry_validation"]
    assert split["wheel_centers_world_m"][0][0] > .06
    assert split["wheel_centers_world_m"][1][0] < -.06
    assert max(split["wheel_bottom_clearance_over_center_surface_m"]) < .02
    shell = frozen["profiles"]["step200_shell_high_wheels_low"]["geometry_validation"]
    assert shell["lowest_support_body"] == "base_link"
    assert max(wheel[0] for wheel in shell["wheel_centers_world_m"]) < -.06
    assert max(shell["wheel_bottom_clearance_over_center_surface_m"]) < .02


def test_targets_are_v6_canonical_and_not_old_v5_signs(builder, frozen):
    assert frozen["targets"]["nominal"]["control6_rad"][0] < 0
    assert frozen["targets"]["support"] == frozen["targets"]["nominal"]
    for name, beta in (("fold", 50.), ("thrust", 95.)):
        q = frozen["targets"][name]["full_joint_positions"]
        np.testing.assert_allclose(builder.knees(q), [beta, beta], atol=.001)
        assert np.max(np.abs(frozen["targets"][name]["control6_rad"])) < np.pi
    for profile in frozen["profiles"].values():
        assert "paired_equivalent_reference_turns" in profile["common_winding_policy"]
        assert "handoff_requires_canonical_active_angles" not in profile


def test_prepare_reference_is_separate_from_the_unchanged_reset_geometry(builder, frozen):
    # These identities preserve all 26 pre-compensation reset states and the
    # old nominal/fold/thrust/support geometry during the prepare-reference A/B.
    identities = {"profiles": "bce09b71f82b86a1eed2a745352a6d94617f886879341d801db24e37be67a237",
                  "targets": "7d071e2e8b5e2b30ab0d3590d8f1c990ba87c38b7cdd264bab9dfc67eed66241"}
    for key, expected in identities.items():
        content = json.loads(json.dumps(frozen[key]))
        if key == "profiles":
            # Restore only retired descriptive metadata when comparing the
            # pre-compensation identity. Its literal-canonical-angle claim was
            # incorrect; all physical reset coordinates must still be identical.
            for row in content.values():
                row.update(goal_target="support",
                    canonical_handoff_control6_rad=frozen["targets"]["nominal"]["control6_rad"],
                    handoff_requires_canonical_active_angles=True,
                    common_winding_policy="preserve_continuous_angles_until_script_returns_to_canonical_targets; never_wrap_PD_error")
        payload = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(payload).hexdigest() == expected
    reference = frozen["prepare_reference"]
    assert reference["height_m"] == pytest.approx(.305, abs=1e-8)
    assert reference["kp"] == 160.
    assert not reference["pd_target_is_reset_pose"]
    assert not reference["dynamic_validation_passed"]
    q = reference["full_joint_positions"]
    assert builder.validate_q(q)["max_closure_gap_m"] < 1e-8
    np.testing.assert_allclose(reference["geometry_control6_rad"], [q[name] for name in builder.controls])
    np.testing.assert_allclose(160. * (np.array(reference["pd_control6_rad"])
                                     - reference["geometry_control6_rad"]),
                               reference["hold_torque_nm"], atol=1e-8)
    assert reference["provenance"]["static_helper_sha256"] == MODULE.digest(ROOT / "scripts/recovery_static_targets.py")
    assert reference["provenance"]["spring_fit_sha256"] == MODULE.digest(builder.bundle / "fit_10mpa.json")


def test_paired_common_rotation_preserves_knee_and_continuous_coordinates(builder):
    q = frozen_pose = dict(builder.nominal)
    moved = builder.common_rotate(q, .8, -.8)
    np.testing.assert_allclose(builder.knees(moved), builder.knees(frozen_pose), atol=.001)
    for hip, auxiliary in (("L_joint1", "LL_joint1"), ("R_joint1", "RR_joint1")):
        assert moved[auxiliary] - moved[hip] == pytest.approx(q[auxiliary] - q[hip], abs=1e-12)


def test_riser_clipping_considers_edges_not_only_vertices():
    points = np.array([[-1., 0., -2.], [1., 0., 2.], [1., 0., 3.]])
    assert MODULE.RecoveryProfileBuilder.halfspace_min_z(points, 0.) == pytest.approx(0.)
    assert MODULE.RecoveryProfileBuilder.halfspace_min_z(points, 2.) is None


def test_double_step_is_additional_250mm_and_goal_keeps_upper_support(frozen):
    for name, row in frozen["profiles"].items():
        if not name.startswith("double_step_"):
            continue
        assert row["stair_heights_m"] == [.2, .45]
        assert row["stair_tread_lengths_m"][0] == .8
        assert row["required_platform_height_m"] in (.2, .45)
        assert row["goal_support_height_m"] == row["required_platform_height_m"]
        assert row["min_ending_x"] == (.22 if row["required_platform_height_m"] == .2 else 1.02)
        assert "not_recorded_failure_trajectory" in row["source"]
